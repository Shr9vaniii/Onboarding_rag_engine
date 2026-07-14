import json
import re
import time
from pathlib import Path
import ast
import requests
from bs4 import BeautifulSoup, Tag

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://fastapi.tiangolo.com"
OUTPUT_DIR = Path("../enterprise_data/wikis_and_docs_scraped_v2")
DELAY      = 0.4   # seconds between requests — be polite
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# All pages to scrape — same list as your original scraper
ALL_SECTIONS = [


    # Tutorial — how-to guides
    "/tutorial/first-steps/",
    "/tutorial/path-params/",
    "/tutorial/query-params/",
    "/tutorial/body/",
    "/tutorial/body-multiple-params/",
    "/tutorial/query-params-str-validations/",
    "/tutorial/path-params-numeric-validations/",
    "/tutorial/body-fields/",
    "/tutorial/body-nested-models/",
    "/tutorial/response-model/",
    "/tutorial/extra-models/",
    "/tutorial/response-status-code/",
    "/tutorial/request-forms/",
    "/tutorial/request-files/",
    "/tutorial/request-forms-and-files/",
    "/tutorial/handling-errors/",
    "/tutorial/path-operation-configuration/",
    "/tutorial/encoder/",
    "/tutorial/body-updates/",
    "/tutorial/dependencies/",
    "/tutorial/dependencies/classes-as-dependencies/",
    "/tutorial/dependencies/sub-dependencies/",
    "/tutorial/dependencies/dependencies-in-path-operation-decorators/",
    "/tutorial/dependencies/global-dependencies/",
    "/tutorial/dependencies/dependencies-with-yield/",
    "/tutorial/security/",
    "/tutorial/security/oauth2-jwt/",
    "/tutorial/security/http-basic-auth/",
    "/tutorial/middleware/",
    "/tutorial/cors/",
    "/tutorial/sql-databases/",
    "/tutorial/bigger-applications/",
    "/tutorial/background-tasks/",
    "/tutorial/metadata/",
    "/tutorial/static-files/",
    "/tutorial/testing/",
    "/tutorial/debugging/",

    # Advanced
    "/advanced/path-operation-advanced-configuration/",
    "/advanced/additional-status-codes/",
    "/advanced/response-directly/",
    "/advanced/custom-response/",
    "/advanced/websockets/",
    "/advanced/events/",
    "/advanced/middleware/",
    "/advanced/sql-databases-peewee/",
    "/advanced/async-sql-databases/",
    "/advanced/nosql-databases/",
    "/advanced/sub-applications/",
    "/advanced/behind-a-proxy/",
    "/advanced/templates/",
    "/advanced/graphql/",
    "/advanced/testing-websockets/",
    "/advanced/testing-events/",
    "/advanced/testing-dependencies/",
    "/advanced/async-tests/",
    "/advanced/settings/",
    "/advanced/openapi-callbacks/",
    "/advanced/openapi-webhooks/",
    "/advanced/generate-clients/",

    # Deployment
    "/deployment/concepts/",
    "/deployment/docker/",
    "/deployment/server-workers/",
    "/deployment/https/",
    "/deployment/manually/",

    # How-to
    "/how-to/general/",
    "/how-to/graphql/",
    "/how-to/custom-request-and-route/",
    "/how-to/conditional-openapi/",
    "/how-to/extending-openapi/",
    "/how-to/custom-docs-ui-assets/",
    "/how-to/configure-swagger-ui/",
    "/how-to/separate-openapi-schemas/",
    "/how-to/custom-openapi-ui-oauth2/",


        "/tutorial/query-param-models/",
        "/tutorial/schema-extra-example/",
        "/tutorial/extra-data-types/",
        "/tutorial/cookie-params/",
        "/tutorial/header-params/",
        "/tutorial/cookie-param-models/",
        "/tutorial/header-param-models/",
        "/tutorial/request-form-models/",
        "/tutorial/security/first-steps/",
        "/tutorial/security/get-current-user/",
        "/tutorial/security/simple-oauth2/",
        "/tutorial/stream-json-lines/",
        "/tutorial/server-sent-events/",
        "/tutorial/frontend/",
        "/advanced/stream-data/",
        "/advanced/path-operation-advanced-configuration/",
        "/advanced/websockets/",
        "/advanced/additional-responses/",
        "/advanced/response-cookies/",
        "/advanced/response-headers/",
        "/advanced/response-change-status-code/",
        "/advanced/advanced-dependencies/",
        "/advanced/security/",
        "/advanced/security/oauth2-scopes/",
        "/advanced/security/http-basic-auth/",
        "/advanced/using-request-directly/",
        "/advanced/dataclasses/",
        "/advanced/wsgi/",
        "/advanced/advanced-python-types/",
        "/advanced/json-base64-bytes/",
        "/advanced/strict-content-type/",
        "/deployment/",
        "/deployment/versions/",
        "/deployment/fastapicloud/",
        "/deployment/cloud/",
        "/how-to/migrate-from-pydantic-v1-to-pydantic-v2/",
        "/how-to/general/",
        "/how-to/authentication-error-status-code/"
]

# Python version preference order — newest wins when multiple tabs exist
VERSION_PRIORITY = ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8", "3.6", "Annotated"]


# ── HTML extraction helpers ───────────────────────────────────────────────────

def get_content_container(soup: BeautifulSoup) -> Tag | None:
    """
    Returns the main content article for Zensical/MkDocs Material sites.
    Confirmed selector from page inspection: article.md-content__inner
    Falls back gracefully in case the class changes.
    """
    selectors = [
        ("article", "md-content__inner"),
        ("div",     "md-content"),
        ("article", None),
        ("main",    "md-main"),
    ]
    for tag_name, class_name in selectors:
        if class_name:
            found = soup.find(tag_name, class_=class_name)
        else:
            found = soup.find(tag_name)
        if found:
            return found
    return None


def extract_internal_links(current_url:str,content_tag: Tag, current_path: str = "") -> list[dict]:
    links = []
    seen = set()

    for a in content_tag.find_all("a", href=True):
        href = a["href"]
        anchor = a.get_text(strip=True)

        # Skip: anchors, empty, icons (¶), sponsor/social
        if (
            not href
            or href.startswith("#")
            or anchor in ("¶", "", "↩")
            or "github.com" in href
            or "twitter" in href
            or "linkedin" in href
        ):
            continue

        # Resolve relative paths to absolute doc paths
        if href.startswith("../") or href.startswith("./"):
            from urllib.parse import urljoin
            resolved = urljoin(current_path, href)
            url=urljoin(current_url,href)
            # Strip the domain if present
            resolved = resolved.replace("https://fastapi.tiangolo.com", "")
            normalised = resolved
        elif href.startswith("/"):
            normalised = href
        elif "fastapi.tiangolo.com" in href:
            normalised = href.split("fastapi.tiangolo.com")[1]
        else:
            continue   # truly external

        key = (normalised, anchor)
        if key not in seen:
            seen.add(key)
            links.append({"target_url":url,"target_slug": normalised, "anchor_text": anchor})

    return links


def pick_best_version(code_blocks: list[dict]) -> dict:
    """
    When a tabbed code block has multiple Python version variants,
    return only the newest/most-modern one.

    
    """
    if not code_blocks:
        return
    versioned = [b for b in code_blocks if b.get("version")!="unknown"]

    if not versioned:
        return code_blocks
    else:
        for preferred in VERSION_PRIORITY:
            for block in code_blocks:
                if preferred in versioned:
                    return [block]
    return versioned[-1]


def extract_mentions(code_blocks: list) -> dict:
    """
    Extracts FastAPI class/function identifiers from prose text.
    These are backtick-wrapped PascalCase/camelCase names — used
    as the cross-linking key between doc chunks and code_contracts.
    """
    mentions={
            "imports":set(),
            "functions":set(),
            "decorators":set(),
            "calls":set()
    }

    for c in code_blocks:
        try:
            tree=ast.parse(c)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # imports
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    mentions["imports"].add(alias.name)

            # function defs
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mentions["functions"].add(node.name if isinstance(node,ast.FunctionDef) else "async "+node.name)

                for dec in node.decorator_list:
                    if isinstance(dec, ast.Call):
                        dec = dec.func

                    if isinstance(dec, ast.Attribute):
                        mentions["decorators"].add(f"{ast.unparse(dec.value)}.{dec.attr}")
                    elif isinstance(dec, ast.Name):
                        mentions["decorators"].add(dec.id)

            # function/class calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    mentions["calls"].add(func.id)
                elif isinstance(func, ast.Attribute):
                    mentions["calls"].add(f"{ast.unparse(func.value)}.{func.attr}")


    return mentions

   


# ── Section walker ────────────────────────────────────────────────────────────

def walk_content(content_tag: Tag) -> list[dict]:
    sections = []
    current = _new_section(["(intro)"])

    def flush():
        has_prose = any(l.strip() for l in current["prose"])
        has_code  = bool(current["code_blocks"])
        has_admon = bool(current["admonitions"])
        if has_prose or has_code or has_admon:
            current["prose"] = [
                l for l in current["prose"]
                if l.strip() and not l.strip().startswith("¶")
            ]
            sections.append(dict(current))

    def process_element(elem: Tag):
        """Recursively processes any element — handles arbitrary nesting."""
        if not isinstance(elem, Tag):
            return

        tag = elem.name
        classes = elem.get("class", [])

        # ── Headings ─────────────────────────────────────────────────────────
        if tag in ("h1", "h2", "h3", "h4"):
            flush()
            new_heading = _extract_prose_line(elem)
            level = int(tag[1])
            if sections:
                parent_stack = sections[-1]["header_path"]
                new_stack = parent_stack[:level - 1] + [new_heading]
            else:
                new_stack = [new_heading]
            
            current.update(_new_section(new_stack))
            
            current["prose"]       = []
            current["admonitions"] = []
            current["code_blocks"] = []
            return

        # ── Admonitions: div.admonition ───────────────────────────────────────
        if tag == "table":
            current["tables"].append(
                _parse_table(elem,current)
            )
            return
        if tag == "div" and "admonition" in classes:
            admon_type = next(
                (c for c in classes
                 if c in {"warning","note","tip","danger","info",
                           "check","question","success","failure","bug"}),
                "note"
            )
            title_tag  = elem.find("p", class_="admonition-title")
            title_text = title_tag.get_text(strip=True) if title_tag else admon_type.title()
            body_parts = [
                _extract_prose_line(p)
                for p in elem.find_all("p")
                if p != title_tag
            ]
            body = " ".join(body_parts).strip()
            current["admonitions"].append({
                "type": admon_type, "title": title_text, "body": body
            })
            if body:
                current["prose"].append(f"[{title_text}] {body}")
            return

        # ── Tabbed code: div.tabbed-set ───────────────────────────────────────
        # In process_element, replace the tabbed-set handler with this:
        if tag == "div" and "tabbed-set" in classes:
            # Extract code blocks first
            tab_blocks = _extract_tabbed_code(elem)
            if tab_blocks:
                best = pick_best_version(tab_blocks)
                current["code_blocks"].append(best)
            
            # FIXED: find ALL admonitions anywhere in the tabbed subtree
            # regardless of nesting depth (details/input/tabbed-content/tabbed-block)
            for admon in elem.find_all("div", class_="admonition"):
                admon_type = next(
                    (c for c in admon.get("class", [])
                    if c in {"warning","note","tip","danger","info",
                            "check","question","success","failure","bug"}),
                    "note"
                )
                title_tag  = admon.find("p", class_="admonition-title")
                title_text = title_tag.get_text(strip=True) if title_tag else admon_type.title()
                body_parts = [
                    _extract_prose_line(p)
                    for p in admon.find_all("p")
                    if p != title_tag
                ]
                body = " ".join(body_parts).strip()
                current["admonitions"].append({
                    "type": admon_type, "title": title_text, "body": body
                })
                if body:
                    current["prose"].append(f"[{title_text}] {body}")
            return

        # ── Details/summary (collapsed sections) ─────────────────────────────
        if tag == "details":
            summary = elem.find("summary")
            if summary:
                current["prose"].append(
                    f"[Collapsed] {summary.get_text(strip=True)}"
                )
            # Recurse into details content
            for child in elem.children:
                if isinstance(child, Tag) and child.name != "summary":
                    process_element(child)
            return

        # ── Plain pre > code (non-tabbed) ────────────────────────────────────
        if tag == "pre":
            # code.md-code--content wasn't found → just find("code") works
            code_tag = elem.find("code")
            if code_tag:
                current["code_blocks"].append({
                    "version": "unknown",
                    "code": code_tag.get_text()
                })
            return

        # ── Prose elements ────────────────────────────────────────────────────
        if tag == "p":
            line = _extract_prose_line(elem)
            if line:
                current["prose"].append(line)
            return

        if tag in ("ul", "ol"):
            for li in elem.find_all("li", recursive=False):
                line = _extract_prose_line(li)
                if line:
                    current["prose"].append("• " + line)
            return

        if tag == "dl":
            for child in elem.children:
                if isinstance(child, Tag) and child.name in ("dt", "dd"):
                    line = _extract_prose_line(child)
                    if line:
                        current["prose"].append(line)
            return

        # ── Everything else — recurse into children ───────────────────────────
        for child in elem.children:
            process_element(child)

    # Process all direct children of the article
    for child in content_tag.children:
        process_element(child)

    flush()
    return sections


def _new_section(header_path: list[str]) -> dict:
    return {
        "header_path":    header_path,

        "prose":          [],
        "tables":         [],
        "admonitions":    [],
        "code_blocks":    [],
        "mentions_classes": [],
    }

def _parse_table(table:Tag,current_section:dict):
    if table.select_one(".doc-param-details"):
        return _parse_doc_table(table,current_section)
    return _parse_markdown_table(table)

def _parse_markdown_table(table: Tag):

    result = {
        "kind": "table",
        "headers": [],
        "rows": []
    }

    header = table.find("thead")

    if header:

        tr = header.find("tr")

        if tr:
            result["headers"] = [
                _extract_prose_line(th)
                for th in tr.find_all(["th", "td"], recursive=False)
            ]

    body = table.find("tbody")

    rows = body.find_all("tr", recursive=False) if body else table.find_all("tr", recursive=False)

    for tr in rows:

        row = [
            _extract_prose_line(td)
            for td in tr.find_all(["td", "th"], recursive=False)
        ]

        if row:
            result["rows"].append(row)

    return result

def _parse_doc_table(table: Tag, current_section: dict) -> dict:

    category = current_section["header_path"][-1].lower()

    result = {
        "kind": category,
        "items": []
    }

    for row in table.select("tbody > tr.doc-section-item"):

        cols = row.find_all("td", recursive=False)

        if len(cols) < 2:
            continue

        left = cols[0]
        right = cols[1]

        item = {}

        # -----------------------
        # name
        # -----------------------

        code = left.find("code")

        if code:
            item["name"] = code.get_text(strip=True)
        else:
            item["name"] = _extract_prose_line(left)

        # -----------------------
        # description
        # -----------------------

        desc = right.select_one(".doc-md-description")

        if desc:
            item["description"] = _extract_prose_line(desc)
        else:
            item["description"] = _extract_prose_line(right)

        # -----------------------
        # annotation (type)
        # -----------------------

        annotation = right.select_one(".doc-param-annotation")

        if annotation:
            item["annotation"] = _extract_prose_line(annotation)

        # -----------------------
        # default value
        # -----------------------

        default = right.select_one(".doc-param-default")

        if default:
            item["default"] = _extract_prose_line(default)

        result["items"].append(item)

    return result


def _has_class(tag: Tag, class_name: str) -> bool:
    classes = tag.get("class", [])
    return class_name in classes


def _get_admonition_type(admon_tag: Tag) -> str:
    """Extract type from class list: 'admonition warning' → 'warning'"""
    classes = admon_tag.get("class", [])
    known = {"warning", "note", "tip", "danger", "info", "check",
             "question", "success", "failure", "bug", "example", "quote"}
    for cls in classes:
        if cls in known:
            return cls
    return "note"   # default


def _extract_tabbed_code(tabset_tag: Tag) -> list[dict]:
    labels_div = tabset_tag.find("div", class_="tabbed-labels")
    if not labels_div:
        return []

    # FIXED: use <label> not <a>
    version_labels = [
        label.get_text(strip=True)
        for label in labels_div.find_all("label")
        if label.get_text(strip=True)
    ]

    tabbed_content = tabset_tag.find("div", class_="tabbed-content")
    if not tabbed_content:
        return []

    code_blocks = []
    blocks = tabbed_content.find_all("div", class_="tabbed-block")

    for i, block in enumerate(blocks):
        pre = block.find("pre")
        if not pre:
            continue
        code_tag = pre.find("code")
        if not code_tag:
            continue
        version = version_labels[i] if i < len(version_labels) else "unknown"
        code_blocks.append({
            "version": version,
            "code": code_tag.get_text()
        })

    return code_blocks


def _extract_prose_line(tag: Tag) -> str:
    """
    Extracts clean prose text from a single element.
    Wraps inline <code> in backticks BEFORE extracting text so
    identifiers don't concatenate with surrounding words:
      "use aResponselike" → "use a `Response` like"
    """
    # Work on a copy — don't mutate the live tree
    copy = BeautifulSoup(str(tag), "html.parser")

    # Wrap inline code with spaces + backticks
    for code in copy.find_all("code"):
        text = code.get_text()
        if text.strip():
            code.replace_with(f" `{text.strip()}` ")

    line = copy.get_text(separator=" ").strip()

    # Normalise multiple spaces that appear after replacement
    line = re.sub(r"  +", " ", line)
    # Strip paragraph anchor symbols
    line = line.replace("¶", "").strip()

    return line if len(line) > 2 else ""


# ── Page scraper ──────────────────────────────────────────────────────────────

def scrape_page(path: str) -> dict | None:
    url = BASE_URL + path.rstrip("/") + "/"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"  SKIP {path} — HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  ERROR {path} — {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    content = get_content_container(soup)
    if not content:
        print(f"  SKIP {path} — no content container found")
        return None

    # Extract links BEFORE get_text() destroys the <a> tags
    related_pages = extract_internal_links(url,content)

    h1 = content.find("h1") or soup.find("h1")
    title = h1.get_text(strip=True).replace("¶", "").strip() if h1 else path

    section = path.strip("/").split("/")[0]   # tutorial / advanced / reference / etc

    # Walk and extract sections with full structure
    sections = walk_content(content)

    # Post-process: attach mentions_classes to each section
    for sec in sections:
        full_text = " ".join(sec["prose"])
        code_blocks=[]
        for c in sec["code_blocks"]:
            try:
                if is_python_module(c["code"]):
                    code_blocks.append(c["code"])
            except SyntaxError:
                pass
        sec["mentions_classes"] = extract_mentions(code_blocks)

    total_prose_lines = sum(len(s["prose"]) for s in sections)
    if total_prose_lines < 3:
        print(f"  SKIP {path} — too little content ({total_prose_lines} lines)")
        return None
    
    

    data= {
        "url":          url,
        "path":         path,
        "section":      section,
        "title":        title,
        "related_pages": related_pages,
        "sections":     sections,
    }

    for section in data["sections"]:
        for key, value in section["mentions_classes"].items():
            if isinstance(value, set):
                section["mentions_classes"][key] = sorted(value)

    return data


# ── Chunk builder — for embedding ─────────────────────────────────────────────

'''def build_chunks(page_data: dict) -> list[dict]:
    """
    Converts a scraped page dict into embeddable chunks.
    One prose chunk + one code chunk per section.
    Code chunks only use the best Python version variant.
    """
    chunks = []
    url     = page_data["url"]
    title   = page_data["title"]
    section = page_data["section"]
    related = ",".join(l["target_path"] for l in page_data.get("related_pages", [])[:8])

    for sec in page_data["sections"]:
        header_path = " > ".join(sec["header_path"])

        # ── Prose chunk ───────────────────────────────────────────────────────
        prose_lines = sec["prose"]
        if prose_lines:
            prose_text = "\n".join(prose_lines)
            chunks.append({
                "content": f"Page: {title} | {header_path}\n\n{prose_text}",
                "metadata": {
                    "type":             "wikis_arch_docs",
                    "subtype":          "prose",
                    "source":           url,
                    "title":            title,
                    "section":          section,
                    "header_path":      header_path,
                    "mentions_classes": ",".join(sec["mentions_classes"]),
                    "related_pages":    related,
                }
            })

        # ── Code chunk (best version only) ────────────────────────────────────
        if sec["code_blocks"]:
            best = pick_best_version(sec["code_blocks"])
            chunks.append({
                "content": (
                    f"Code example: {title} | {header_path}\n"
                    f"Version: {best['version']}\n\n"
                    f"```python\n{best['code'].strip()}\n```"
                ),
                "metadata": {
                    "type":          "wikis_arch_docs",
                    "subtype":       "code_example",
                    "source":        url,
                    "title":         title,
                    "section":       section,
                    "header_path":   header_path,
                    "python_version": best["version"],
                    "related_pages": related,
                }
            })

    return chunks'''


# ── Main runner ───────────────────────────────────────────────────────────────

'''def scrape_all(embed: bool = False, collection=None):
    """
    Scrapes all pages in ALL_SECTIONS.
    If embed=True and collection is provided, embeds chunks into ChromaDB
    immediately after each page is scraped.
    Otherwise just writes JSON files to OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scraped  = 0
    skipped  = 0
    failed   = 0
    all_chunks = []

    print(f"\nScraping {len(ALL_SECTIONS)} pages from {BASE_URL}\n")

    for path in ALL_SECTIONS:
        safe_name = path.strip("/").replace("/", "-") + ".json"
        out_path  = OUTPUT_DIR / safe_name

        # Skip already-scraped files
        if out_path.exists():
            print(f"  SKIP (cached) {path}")
            skipped += 1
            continue

        page = scrape_page(path)

        if page is None:
            failed += 1
            time.sleep(DELAY)
            continue

        # Write raw scraped JSON
        out_path.write_text(
            json.dumps(page, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        chunks = build_chunks(page)
        all_chunks.extend(chunks)

        code_count  = sum(1 for c in chunks if c["metadata"]["subtype"] == "code_example")
        prose_count = sum(1 for c in chunks if c["metadata"]["subtype"] == "prose")

        print(
            f"  ✓ [{page['section']:12s}] {page['title'][:50]:<50s}"
            f"  sections={len(page['sections'])}"
            f"  prose={prose_count}"
            f"  code={code_count}"
            f"  links={len(page['related_pages'])}"
        )
        scraped += 1
        time.sleep(DELAY)

    print(f"\n{'─'*60}")
    print(f"  Scraped:  {scraped}")
    print(f"  Skipped:  {skipped} (already cached)")
    print(f"  Failed:   {failed}")
    print(f"  Chunks:   {len(all_chunks)} total")
    print(f"{'─'*60}\n")

    if embed and collection and all_chunks:
        _embed_chunks(all_chunks, collection)

    return all_chunks


def _embed_chunks(chunks: list[dict], collection):
    """Embeds chunks into your existing ChromaDB collection."""
    import uuid
    documents = [c["content"]  for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [str(uuid.uuid4()) for _ in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
        print(f"  Embedded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

    print(f"  Done — {len(chunks)} chunks embedded.")

'''
# ── Debug helper ──────────────────────────────────────────────────────────────

'''def inspect_page(path: str):
    """
    Scrape a single page and print its full structure.
    Use this to verify extraction quality before running the full batch.

      from scrape_docs_v2 import inspect_page
      inspect_page("/tutorial/background-tasks/")
    """
    page = scrape_page(path)
    if not page:
        print("Failed to scrape page.")
        return

    print(f"\nTitle:    {page['title']}")
    print(f"URL:      {page['url']}")
    print(f"Sections: {len(page['sections'])}")
    print(f"Links:    {len(page['related_pages'])}")

    for i, sec in enumerate(page["sections"]):
        print(f"\n  [{i}] {' > '.join(sec['header_path'])}")
        print(f"       prose lines : {len(sec['prose'])}")
        print(f"       admonitions : {len(sec['admonitions'])}")
        print(f"       code blocks : {len(sec['code_blocks'])}")
        print(f"       mentions    : {sec['mentions_classes']}")

        if sec["prose"]:
            print(f"       first line  : {sec['prose'][0][:80]}")
        if sec["admonitions"]:
            a = sec["admonitions"][0]
            print(f"       admonition  : [{a['type']}] {a['title']} — {a['body'][:60]}")
        if sec["code_blocks"]:
            best = pick_best_version(sec["code_blocks"])
            print(f"       best code   : ({best['version']}) {best['code'][:60].strip()}...")

    print(f"\n  Related pages ({len(page['related_pages'])}):")
    for link in page["related_pages"][:5]:
        print(f"    {link['anchor_text'][:30]:30s} → {link['target_path']}")
    if len(page["related_pages"]) > 5:
        print(f"    ... and {len(page['related_pages']) - 5} more")

    chunks = build_chunks(page)
    print(f"\n  Chunks produced: {len(chunks)}")
    for c in chunks[:3]:
        print(f"    [{c['metadata']['subtype']:12s}] {c['content'][:80]}")'''


def scrape_all():
    results=[]
   
    for path in ALL_SECTIONS:
        data=scrape_page(path)
        if data:
            safe_name = path.strip("/").replace("/", "-") + ".json"
            out_path = OUTPUT_DIR / safe_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ {data['title'][:55]}")
            results.append(data)
        time.sleep(0.4)

    print(f"\nDone. Scraped {len(results)}/{len(ALL_SECTIONS)} pages.")
    return results


def is_python_module(code: str) -> bool:
    code = code.lstrip()

    return (
        code.startswith(("from ", "import ", "def ", "async def ", "class ", "@"))
        or "\nfrom " in code
        or "\nimport " in code
    )




'''def _extract_reference_description(contents_div: Tag) -> str:
    """
    Description = plain <p> tags (no class) directly inside doc-contents.
    Stops at div.doc-children or details.mkdocstrings.
    """
    if not contents_div:
        return ""
    parts = []
    for child in contents_div.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", [])
        # Stop at children block or source code
        if "doc-children" in classes or "mkdocstrings-source" in classes:
            break
        # Skip bases line, headings, code blocks
        if "doc-class-bases" in classes:
            continue
        if child.name in ("h3", "h4", "h5", "div", "details"):
            continue
        # Plain <p> = description
        if child.name == "p" and not classes:
            line = _extract_prose_line(child)
            if line:
                parts.append(line)
    return " ".join(parts).strip()


def _extract_reference_params(contents_div: Tag) -> list[dict]:
    """
    Parameters are tr.doc-section-item elements directly inside doc-contents.
    No <table> wrapper — bare <tr> tags.

    Each row structure:
      <tr class="doc-section-item">
        <td> ← parameter name (first td, has <code>)
        <td class="doc-param-details">
          <div class="doc-md-description"> ← description
          <span class="doc-param-annotation"> ← "TYPE: Callable[P,Any]"
          <span class="doc-param-default">    ← "DEFAULT: None" (if present)
    """
    if not contents_div:
        return []

    params = []
    for row in contents_div.find_all("tr", class_="doc-section-item"):
        cols = row.find_all("td", recursive=False)
        if len(cols) < 1:
            continue

        param = {}

        # Name — first td, look for <code> inside
        name_td = cols[0]
        name_code = name_td.find("code")
        param["name"] = name_code.get_text(strip=True) if name_code \
                        else _extract_prose_line(name_td)

        if len(cols) < 2:
            params.append(param)
            continue

        details_td = cols[1]

        # Description
        desc = details_td.find("div", class_="doc-md-description")
        param["description"] = _extract_prose_line(desc) if desc \
                               else _extract_prose_line(details_td)

        # Type annotation — "TYPE: Callable[P,Any]" → strip "TYPE:"
        annotation = details_td.find("span", class_="doc-param-annotation")
        if annotation:
            raw = annotation.get_text(strip=True)
            param["type"] = re.sub(r"^TYPE:\s*", "", raw).strip()

        # Default value — "DEFAULT: None" → strip "DEFAULT:"
        default = details_td.find("span", class_="doc-param-default")
        if default:
            raw = default.get_text(strip=True)
            param["default"] = re.sub(r"^DEFAULT:\s*", "", raw).strip()

        params.append(param)

    return params


def _extract_children(doc_children_div: Tag, page_url: str, page_title: str) -> list[dict]:
    """
    Recursively extracts nested doc-objects (methods, attributes)
    from div.doc-children.
    """
    if not doc_children_div:
        return []

    chunks = []
    for obj in doc_children_div.find_all("div", class_="doc-object", recursive=False):
        chunk = _doc_object_to_chunk(obj, page_url, page_title)
        if chunk:
            chunks.append(chunk)
    return chunks


def _doc_object_to_chunk(doc_obj: Tag, page_url: str, page_title: str) -> dict | None:
    """
    Converts one div.doc-object into one embeddable chunk.
    Handles classes, methods, functions, attributes.
    Recurses into div.doc-children for nested objects.
    """
    classes = doc_obj.get("class", [])

    if "doc-class" in classes:     kind = "class"
    elif "doc-function" in classes: kind = "function"
    elif "doc-attribute" in classes:kind = "attribute"
    elif "doc-method" in classes:   kind = "method"
    else:                           kind = "object"

    # Heading — span.doc-object-name has the clean name
    name_span = doc_obj.find("span", class_="doc-object-name")
    short_name = name_span.get_text(strip=True) if name_span else ""

    # Full dotted heading e.g. "fastapi.BackgroundTasks.add_task"
    heading_tag = doc_obj.find(class_="doc-heading")
    full_heading = ""
    if heading_tag:
        full_heading = heading_tag.get_text(strip=True).replace("¶", "").strip()
        # Remove trailing labels like "instance-attribute" "classmethod"
        for label in ("instance-attribute", "classmethod", "staticmethod",
                       "abstractmethod", "property", "cached-property"):
            full_heading = full_heading.replace(label, "").strip()

    if not short_name and not full_heading:
        return None

    # Signature
    sig_div = doc_obj.find("div", class_="doc-signature")
    signature = ""
    if sig_div:
        pre = sig_div.find("pre")
        raw = pre.get_text(strip=True) if pre else sig_div.get_text(strip=True)
        signature = re.sub(r"¶", "", raw).strip()

    # Contents div — description + params
    contents_div = doc_obj.find("div", class_="doc-contents", recursive=False)

    description = _extract_reference_description(contents_div)
    params      = _extract_reference_params(contents_div)

    # Skip source code block entirely — details.mkdocstrings
    # already excluded by _extract_reference_description

    # ── Build chunk text ──────────────────────────────────────────────────────
    lines = [f"{kind.title()}: {full_heading or short_name}"]

    if signature:
        lines.append(f"Signature: {signature}")

    if description:
        lines.append(f"\nDescription: {description}")

    if params:
        lines.append("\nParameters:")
        for p in params:
            parts = [f"  {p.get('name', '')}"]
            if p.get("type"):
                parts.append(f"type: {p['type']}")
            if p.get("default"):
                parts.append(f"default: {p['default']}")
            if p.get("description"):
                parts.append(p["description"])
            lines.append(" — ".join(parts))

    content = "\n".join(lines).strip()
    if len(content) < 15:
        return None

    chunk = {
        "content": content,
        "metadata": {
            "type":        "wikis_arch_docs",
            "subtype":     "reference",
            "source":      page_url,
            "title":       page_title,
            "section":     "reference",
            "class_name":  short_name or short_name,
            "kind":        kind,
            "header_path": full_heading,
        }
    }

    # ── Recurse into children ─────────────────────────────────────────────────
    # Children are in div.doc-children inside doc-contents
    children_div = contents_div.find("div", class_="doc-children") \
                   if contents_div else None
    if children_div:
        child_chunks = _extract_children(children_div, page_url, page_title)
        # Return class chunk + all its method/attribute chunks as flat list
        return [chunk] + child_chunks

    return chunk


def scrape_reference_page(path: str) -> list[dict]:
    """
    Scrapes one reference page → chunks directly, no intermediate JSON.
    One chunk per doc-object (class + its methods/attributes recursively).
    """
    url = BASE_URL + path.rstrip("/") + "/"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"  SKIP {path} — HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"  ERROR {path} — {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    content = get_content_container(soup)
    if not content:
        return []

    h1 = content.find("h1")
    page_title = h1.get_text(strip=True).replace("¶", "").strip() if h1 else path

    # Only top-level doc-objects — children are handled recursively inside
    top_level_objects = content.find_all(
        "div",
        class_="doc-object",
        recursive=True
    )
    # Filter to only direct children of the article content
    # (not nested inside other doc-objects)
    top_level_objects = [
        obj for obj in top_level_objects
        if obj.parent == content or
           (obj.parent and obj.parent.parent == content)
    ]

    chunks = []
    for obj in top_level_objects:
        result = _doc_object_to_chunk(obj, url, page_title)
        if result is None:
            continue
        if isinstance(result, list):
            chunks.extend(result)
        else:
            chunks.append(result)

    print(
        f"  ✓ [reference    ] {page_title[:50]:<50}"
        f"  objects={len(top_level_objects)}"
        f"  chunks={len(chunks)}"
    )
    return chunks


def scrape_all_reference() -> list[dict]:
    all_chunks = []
    print(f"\nScraping {len(REFERENCE_PAGES)} reference pages...\n")
    for path in REFERENCE_PAGES:
        chunks = scrape_reference_page(path)
        all_chunks.extend(chunks)
        time.sleep(DELAY)
    print(f"\n{'─'*60}")
    print(f"  Total reference chunks: {len(all_chunks)}")
    print(f"{'─'*60}\n")
    return all_chunks


def inspect_reference(path: str = "/reference/background/"):
    chunks = scrape_reference_page(path)
    print(f"\n{len(chunks)} chunks:\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] {c['metadata']['kind'].upper()}: {c['metadata']['class_name']}")
        print(f"    {c['content'][:300]}")
        print()

'''


if __name__ == "__main__":
    scrape_all()