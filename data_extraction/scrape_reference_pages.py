"""
FastAPI Reference Page Scraper
==============================
Scrapes all /reference/* pages from fastapi.tiangolo.com and produces
structured JSON files matching the agreed schema.

Output per page:
{
    "url": "...",
    "path": "...",
    "title": "...",
    "module": "fastapi.background",
    "section": "reference",
    "import_from": "from fastapi import BackgroundTasks",
    "related_pages": [{"target_path": "...", "anchor_text": "..."}],
    "objects": [OBJECT, ...]
}

Each OBJECT:
{
    "kind": "class|function|method|attribute|property",
    "name": "BackgroundTasks",
    "qualified_name": "fastapi.BackgroundTasks",
    "parent": None,
    "module": "fastapi.background",
    "anchor": "#fastapi.BackgroundTasks",
    "signature": "BackgroundTasks(tasks=None)",
    "bases": ["StarletteBackgroundTasks"],
    "labels": ["instance-attribute"],
    "description": ["plain string paragraphs..."],
    "parameters": [
        {
            "name": "tasks",
            "type": "list[BackgroundTask]",
            "default": "None",
            "description": "The list of background tasks",
            "required": False
        }
    ],
    "returns": None,
    "raises": [],
    "examples": [{"label": "Python 3.10+", "code": "..."}],
    "admonitions": [{"type": "warning", "title": "Warning", "body": "..."}],
    "children": ["fastapi.BackgroundTasks.add_task"],
    "deprecated": False,
    "inherited_from": None
}

Usage:
    python scrape_reference.py
    # Safe to re-run — already scraped files are skipped
    # Inspect one page first:
    #   from scrape_reference import inspect_page
    #   inspect_page("/reference/background/")
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://fastapi.tiangolo.com"
OUTPUT_DIR = Path("../enterprise_data/reference_scraped")
DELAY      = 0.5

REFERENCE_PAGES = [
    "/reference/fastapi/",
    "/reference/apirouter/",
    "/reference/parameters/",
    "/reference/exceptions/",
    "/reference/response/",
    "/reference/responses/",
    "/reference/middleware/",
    "/reference/background/",
    "/reference/uploadfile/",
    "/reference/testclient/",
    "/reference/httpconnection/",
    "/reference/request/",
    "/reference/websockets/",
    "/reference/dependencies/",
    "/reference/security/",
    "/reference/staticfiles/",
    "/reference/templating/",
    "/reference/openapi/docs/",
    "/reference/openapi/models/",
    "/reference/encoders/",
    "/reference/status/",
]

VERSION_PRIORITY = ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8", "Annotated"]

# Known label classes from Zensical/MkDocs Material
LABEL_CLASSES = {
    "instance-attribute", "classmethod", "staticmethod",
    "abstractmethod", "property", "cached-property",
    "dataclass-field", "module-attribute"
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_content_container(soup: BeautifulSoup) -> Tag | None:
    selectors = [
        ("article", "md-content__inner"),
        ("div",     "md-content"),
        ("article", None),
    ]
    for tag_name, class_name in selectors:
        found = soup.find(tag_name, class_=class_name) if class_name \
                else soup.find(tag_name)
        if found:
            return found
    return None


def clean_text(text: str) -> str:
    """Remove ¶ anchors, normalise whitespace."""
    text = text.replace("¶", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_prose_line(tag: Tag) -> str:
    """Extract prose with inline code backtick-wrapped."""
    copy = BeautifulSoup(str(tag), "html.parser")
    for code in copy.find_all("code"):
        t = code.get_text().strip()
        if t:
            code.replace_with(f" `{t}` ")
    line = copy.get_text(separator=" ").strip()
    line = re.sub(r"  +", " ", line)
    return clean_text(line)


def pick_best_version(examples: list[dict]) -> list[dict]:
    """
    From a list of tabbed code examples, keep only the best version.
    Returns a list so callers can extend() directly.
    """
    if not examples:
        return []
    for preferred in VERSION_PRIORITY:
        for ex in examples:
            if preferred in ex.get("label", ""):
                return [ex]
    return [examples[-1]]


def extract_internal_links(content_tag: Tag, current_path: str) -> list[dict]:
    """Extract cross-page doc links — skip anchor-only and external."""
    from urllib.parse import urljoin

    links = []
    seen  = set()

    for a in content_tag.find_all("a", href=True):
        href        = a["href"]
        anchor_text = a.get_text(strip=True)

        if not href or href.startswith("#") or not anchor_text or anchor_text == "¶":
            continue
        if any(d in href for d in [
            "github.com", "twitter", "linkedin",
            "starlette.dev", "pydantic.dev",
            "python.org", "pypi.org"
        ]):
            continue

        original   = None
        normalised = None

        if href.startswith("../") or href.startswith("./"):
            original   = urljoin(BASE_URL + current_path, href)
            normalised = original.replace(BASE_URL, "")

        elif href.startswith("/"):
            normalised = href
            original   = BASE_URL.rstrip("/") + href

        elif "fastapi.tiangolo.com" in href:
            # Already a full URL
            original   = href
            normalised = href.split("fastapi.tiangolo.com")[1]

        else:
            continue

        # Strip fragment from slug but keep it in original
        slug = normalised.split("#")[0] if normalised else None

        key = (slug, anchor_text)
        if key not in seen:
            seen.add(key)
            links.append({
                "slug":        slug,           # clean path, no fragment
                "original":    original,       # full clickable URL with fragment
                "anchor_text": anchor_text,
            })

    return links


# ── Parameter extractor ───────────────────────────────────────────────────────

def extract_parameters(contents_div: Tag) -> list[dict]:
    """
    Extracts parameters from tr.doc-section-item rows.
    Structure (confirmed from page inspection):
      tr.doc-section-item
        td[0]                      → parameter name (has <code>)
        td[1].doc-param-details
          div.doc-md-description   → description text
          span.doc-param-annotation → "TYPE: ..." 
          span.doc-param-default    → "DEFAULT: ..."
    """
    params = []
    if not contents_div:
        return params

    for row in contents_div.find_all("tr", class_="doc-section-item"):
        cols = row.find_all("td", recursive=False)
        if not cols:
            continue

        param = {"name": "", "type": "", "default": "",
                 "description": "", "required": True}

        # Name
        name_code = cols[0].find("code")
        param["name"] = name_code.get_text(strip=True) if name_code \
                        else clean_text(cols[0].get_text())

        if len(cols) < 2:
            params.append(param)
            continue

        details = cols[1]

        # Description
        desc_div = details.find("div", class_="doc-md-description")
        if desc_div:
            param["description"] = extract_prose_line(desc_div)

        # Type — "TYPE: X" → strip prefix
        ann = details.find("span", class_="doc-param-annotation")
        if ann:
            raw = ann.get_text(strip=True)
            param["type"] = re.sub(r"^TYPE:\s*", "", raw).strip()

        # Default — "DEFAULT: X" → strip prefix
        default = details.find("span", class_="doc-param-default")
        if default:
            raw = default.get_text(strip=True)
            param["default"] = re.sub(r"^DEFAULT:\s*", "", raw).strip()
            param["required"] = False   # has a default → not required

        params.append(param)

    return params


# ── Admonition extractor ──────────────────────────────────────────────────────

def extract_admonitions(tag: Tag) -> list[dict]:
    """Extract all admonition boxes from a tag."""
    admons = []
    known = {"warning", "note", "tip", "danger", "info", "check",
             "question", "success", "failure", "bug", "example"}

    for div in tag.find_all("div", class_="admonition"):
        classes   = div.get("class", [])
        admon_type = next((c for c in classes if c in known), "note")
        title_tag  = div.find("p", class_="admonition-title")
        title      = title_tag.get_text(strip=True) if title_tag else admon_type.title()

        body_parts = []
        for p in div.find_all("p"):
            if p == title_tag:
                continue
            body_parts.append(extract_prose_line(p))

        admons.append({
            "type":  admon_type,
            "title": title,
            "body":  " ".join(body_parts).strip()
        })

    return admons


# ── Example extractor ─────────────────────────────────────────────────────────

def extract_examples(tag: Tag) -> list[dict]:
    """
    Extracts code examples — both tabbed (div.tabbed-set) and plain (pre>code).
    Returns only the best version for tabbed examples.
    """
    examples = []

    # Tabbed examples
    for tabset in tag.find_all("div", class_="tabbed-set"):
        labels_div = tabset.find("div", class_="tabbed-labels")
        version_labels = []
        if labels_div:
            version_labels = [
                label.get_text(strip=True)
                for label in labels_div.find_all("label")
                if label.get_text(strip=True)
            ]

        tabbed_content = tabset.find("div", class_="tabbed-content")
        if not tabbed_content:
            continue

        tab_examples = []
        for i, block in enumerate(tabbed_content.find_all("div", class_="tabbed-block")):
            pre = block.find("pre")
            if not pre:
                continue
            code = pre.find("code")
            if not code:
                continue
            label = version_labels[i] if i < len(version_labels) else "unknown"
            tab_examples.append({"label": label, "code": code.get_text()})

        examples.extend(pick_best_version(tab_examples))

    # Plain pre > code (non-tabbed)
    for pre in tag.find_all("pre"):
        # Skip if inside a tabbed-set (already handled above)
        if pre.find_parent("div", class_="tabbed-set"):
            continue
        # Skip source code details blocks
        if pre.find_parent("details", class_="mkdocstrings-source"):
            continue
        code = pre.find("code")
        if code:
            examples.append({"label": "python", "code": code.get_text()})

    return examples

def extract_class_level_examples(contents_div: Tag) -> list[dict]:
    """
    For reference page objects (FastAPI class etc.), only the FIRST
    code example belongs to the class itself.
    All subsequent examples belong to individual parameters — they
    appear as siblings in the flat contents_div structure.

    Strategy: collect <pre> tags that appear before the first
    parameter description block. Stop as soon as we hit parameter content.
    """
    examples = []
    PARAM_SIGNALS = {
        "doc-section",      # parameter sections if present
    }
    PARAM_TEXT_SIGNALS = [
        "Parameters:",
        "param ",
        "→",               # your parameter arrow marker
    ]

    for element in contents_div.children:
        if not isinstance(element, Tag):
            continue

        classes = set(element.get("class", []))

        # Stop at children div
        if "doc-children" in classes:
            break

        # Stop at parameter sections
        if classes & PARAM_SIGNALS:
            break

        # Stop if element text looks like parameter descriptions
        text = element.get_text(strip=True)
        if any(signal in text for signal in PARAM_TEXT_SIGNALS):
            break

        # Tabbed examples
        if "tabbed-set" in classes:
            tab_examples = _extract_tabbed(element)
            examples.extend(pick_best_version(tab_examples))
            continue

        # Plain pre > code
        if element.name == "pre":
            code = element.find("code")
            if code:
                examples.append({"label": "python", "code": code.get_text()})
            continue

        # Pre nested inside a div (e.g. highlight wrapper)
        nested_pre = element.find("pre", recursive=False)
        if nested_pre:
            code = nested_pre.find("code")
            if code:
                examples.append({"label": "python", "code": code.get_text()})

    return examples


def _extract_tabbed(tabset: Tag) -> list[dict]:
    """Shared helper — extract tab labels + code from a tabbed-set."""
    labels_div = tabset.find("div", class_="tabbed-labels")
    version_labels = []
    if labels_div:
        version_labels = [
            l.get_text(strip=True)
            for l in labels_div.find_all("label")
            if l.get_text(strip=True)
        ]
    tab_examples = []
    tabbed_content = tabset.find("div", class_="tabbed-content")
    if tabbed_content:
        for i, block in enumerate(
            tabbed_content.find_all("div", class_="tabbed-block")
        ):
            pre = block.find("pre")
            if not pre:
                continue
            code = pre.find("code")
            if not code:
                continue
            label = version_labels[i] if i < len(version_labels) else "unknown"
            tab_examples.append({"label": label, "code": code.get_text()})
    return tab_examples


# ── Description extractor ─────────────────────────────────────────────────────

def extract_description(contents_div: Tag) -> list[str]:
    """
    Extracts plain description paragraphs from doc-contents.
    Stops at div.doc-children or details.mkdocstrings-source.
    Skips: bases line, headings, code blocks, parameter tables.
    """
    if not contents_div:
        return []

    paragraphs = []
    for child in contents_div.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", [])

        # Stop at children block or source details
        if "doc-children" in classes or "mkdocstrings-source" in classes:
            break

        # Skip non-description elements
        if "doc-class-bases" in classes:
            continue
        if child.name in ("h3", "h4", "h5", "details", "table"):
            continue
        if child.name == "div" and (
            "highlight" in classes or
            "tabbed-set" in classes or
            "admonition" in classes or
            "doc-" in " ".join(classes)
        ):
            continue

        # Plain <p> tags = description
        if child.name == "p" and not classes:
            line = extract_prose_line(child)
            if line:
                paragraphs.append(line)

    return paragraphs


# ── Signature extractor ───────────────────────────────────────────────────────

def extract_signature(doc_obj: Tag) -> str:
    sig_div = doc_obj.find("div", class_="doc-signature")
    if not sig_div:
        return ""
    pre = sig_div.find("pre")
    raw = pre.get_text() if pre else sig_div.get_text()
    return clean_text(raw)


def extract_bases(doc_obj: Tag) -> list[str]:
    bases_p = doc_obj.find("p", class_="doc-class-bases")
    if not bases_p:
        return []
    # Get text from <a> or <code> tags — these have the actual base class names
    bases = []
    for tag in bases_p.find_all(["a", "code"]):
        text = tag.get_text(strip=True)
        if text and text not in ("Bases:", "Bases"):
            bases.append(text)
    return bases if bases else []


# ── Main object extractor ─────────────────────────────────────────────────────

def extract_object(
    doc_obj: Tag,
    page_module: str,
    parent_name: str | None = None
) -> list[dict]:
    """
    Extracts one div.doc-object into one or more OBJECT dicts.
    Returns a flat list: [the object itself] + [all its children].
    Children are extracted recursively and flattened.
    """
    classes = doc_obj.get("class", [])

    # Determine kind
    if "doc-class"    in classes: kind = "class"
    elif "doc-function" in classes:
        kind = "method" if parent_name else "function"
    elif "doc-attribute" in classes: kind = "attribute"
    elif "doc-property"  in classes: kind = "property"
    else: kind = "object"

    # Short name from span.doc-object-name
    name_span = doc_obj.find("span", class_="doc-object-name")
    short_name = name_span.get_text(strip=True) if name_span else ""

    # Full heading e.g. "fastapi.BackgroundTasks"
    heading_tag = doc_obj.find(class_="doc-heading")
    full_heading = ""
    if heading_tag:
        raw = heading_tag.get_text(strip=True).replace("¶", "")
        # Remove trailing label words
        for label in LABEL_CLASSES:
            raw = raw.replace(label, "")
        full_heading = raw.strip()

    if not short_name and not full_heading:
        return []

    # Anchor from heading id
    anchor = ""
    if heading_tag and isinstance(heading_tag, Tag):
        anchor = "#" + heading_tag.get("id", "").strip()
        if anchor == "#":
            # Try h2/h3 inside the object
            for h in doc_obj.find_all(["h2", "h3", "h4"]):
                if h.get("id"):
                    anchor = "#" + h["id"]
                    break

    # Labels (instance-attribute, classmethod, etc.)
    labels = []
    heading_tag_for_labels = doc_obj.find(class_="doc-heading", recursive=False)
    if not heading_tag_for_labels:
        # Try one level deep — heading is usually direct child
        for child in doc_obj.children:
            if isinstance(child, Tag) and "doc-heading" in child.get("class", []):
                heading_tag_for_labels = child
                break

    if heading_tag_for_labels:
        for label_tag in heading_tag_for_labels.find_all("small", class_="doc-label"):
            text = label_tag.get_text(strip=True)
            if text:
                labels.append(text)

    # Bases (for classes)
    bases = []
    bases = extract_bases(doc_obj)

    # Deprecated?
    deprecated = any(
        "deprecated" in lbl.lower() for lbl in labels
    )

    # Inherited from?
    inherited_from = None
    if bases and kind == "method":
        # If method is on a class that inherits, note it
        pass   # can be enhanced later

    # Contents div
    contents_div = doc_obj.find("div", class_="doc-contents", recursive=False)
    children_div_tag = None
    if contents_div:
        children_div_tag = contents_div.find("div", class_="doc-children")
        if children_div_tag:
            children_div_tag = children_div_tag.extract()

    # Description
    description = extract_description(contents_div)

    # Parameters
    parameters = extract_parameters(contents_div)

    # Examples — from contents but NOT from doc-children
    examples = extract_examples(contents_div) if contents_div else []

# AFTER — use scoped extractor for reference page objects
    if kind in ("class", "function", "method"):
        examples = extract_class_level_examples(contents_div) if contents_div else []
    else:
        # attributes and properties have no parameter children — full extract fine
        examples = extract_examples(contents_div) if contents_div else []

    print(f"DEBUG: examples count before reattach = {len(examples)}")

# Admonitions
    admonitions = extract_admonitions(contents_div) if contents_div else []

    # Returns
    returns = None
    if contents_div:
        for elem in contents_div.find_all(["p", "h4", "h5"]):
            if elem.get_text(strip=True).lower().startswith("return"):
                sibling = elem.find_next_sibling()
                if sibling:
                    returns = {
                        "type": "",
                        "description": extract_prose_line(sibling)
                    }
                break

    # Raises
    raises = []
    if contents_div:
        for elem in contents_div.find_all(["p", "h4", "h5"]):
            if "Raise" in elem.get_text(strip=True):
                for sib in elem.find_next_siblings():
                    if sib.name in ("h3", "h4"):
                        break
                    text = extract_prose_line(sib)
                    if text:
                        raises.append({"type": "", "description": text})
                break

    # ── Re-attach children before recursion ──────────────────────────────────────
    if contents_div and children_div_tag:
        contents_div.append(children_div_tag)

    # Build the object dict
    if parent_name:
        qualified_name = f"{parent_name}.{short_name}"
    else:
        qualified_name = full_heading or short_name

    obj = {
        "kind":           kind,
        "name":           short_name or qualified_name.split(".")[-1],
        "qualified_name": qualified_name,
        "parent":         parent_name,
        "module":         page_module,
        "anchor":         anchor,
        "signature":      extract_signature(doc_obj),
        "bases":          bases,
        "labels":         labels,
        "description":    description,
        "parameters":     parameters,
        "returns":        returns,
        "raises":         raises,
        "examples":       examples,
        "admonitions":    admonitions,
        "children":       [],   # populated below
        "deprecated":     deprecated,
        "inherited_from": inherited_from,
    }

    # ── Recurse into children ─────────────────────────────────────────────────
    result = [obj]
    children_div = contents_div.find("div", class_="doc-children") \
                   if contents_div else None

    if children_div:
        for child_obj in children_div.find_all("div", class_="doc-object", recursive=False):
            child_results = extract_object(child_obj, page_module, short_name)
            for child in child_results:
                obj["children"].append(child["qualified_name"])  # ← append child's own name
                result.append(child)

    return result


# ── Page scraper ──────────────────────────────────────────────────────────────

def scrape_reference_page(path: str) -> dict | None:
    """
    Scrapes one reference page → structured JSON.
    Returns None if page can't be fetched or has no content.
    """
    url = BASE_URL + path.rstrip("/") + "/"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"  SKIP {path} — HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  ERROR {path} — {e}")
        return None

    soup    = BeautifulSoup(response.text, "html.parser")
    content = get_content_container(soup)
    if not content:
        print(f"  SKIP {path} — no content container")
        return None

    # Title
    h1 = content.find("h1")
    title = clean_text(h1.get_text()) if h1 else path.strip("/").split("/")[-1]

    # Module — derive from page description or path
    # e.g. /reference/background/ → fastapi.background
    # The page usually has "from fastapi import X" in a code block near the top
    module = _infer_module(content, path)

    # Import statement — first code block at page level
    import_from = _extract_import_statement(content)

    # Cross-page links
    related_pages = extract_internal_links(content, path)

    # All top-level doc-objects
    # Top-level = direct children of content, or one div deep
    top_objects = _find_top_level_objects(content)

    all_objects = []
    for obj_tag in top_objects:
        extracted = extract_object(obj_tag, module)
        all_objects.extend(extracted)

    page = {
        "url":          url,
        "path":         path,
        "title":        title,
        "module":       module,
        "section":      "reference",
        "import_from":  import_from,
        "related_pages": related_pages,
        "objects":      all_objects,
    }

    print(
        f"  ✓ {path:<40}"
        f"  objects={len(all_objects):<4}"
        f"  import='{import_from}'"
    )
    return page


def _find_top_level_objects(content: Tag) -> list[Tag]:
    """
    Finds top-level div.doc-object elements — not nested inside other doc-objects.
    These are the root classes/functions on the page.
    """
    all_objects  = content.find_all("div", class_="doc-object")
    child_objects = set()

    for obj in all_objects:
        children_div = obj.find("div", class_="doc-children")
        if children_div:
            for child in children_div.find_all("div", class_="doc-object"):
                child_objects.add(id(child))

    return [obj for obj in all_objects if id(obj) not in child_objects]


def _infer_module(content: Tag, path: str) -> str:
    """
    Infers the Python module from the page content.
    Looks for the qualified name in the first doc-heading.
    Falls back to path-based inference.
    """
    first_heading = content.find(class_="doc-heading")
    if first_heading:
        text = clean_text(first_heading.get_text())
        # "fastapi.BackgroundTasks" → "fastapi.background" isn't directly here
        # but the qualified name prefix gives us the package
        parts = text.split(".")
        if len(parts) >= 2 and parts[0] == "fastapi":
            # Can't reliably get the submodule from heading alone
            # Fall through to path inference
            pass

    # Path inference: /reference/background/ → fastapi.background
    segment = path.strip("/").replace("reference/", "").replace("/", ".")
    if segment:
        return f"fastapi.{segment}".rstrip(".")
    return "fastapi"


def _extract_import_statement(content: Tag) -> str:
    """
    Finds the 'from fastapi import X' statement shown near the top of
    each reference page. This is always in a plain pre>code block
    before the first doc-object.
    """
    first_obj = content.find("div", class_="doc-object")
    if not first_obj:
        return ""

    # Look for pre>code blocks before the first doc-object
    for pre in content.find_all("pre"):
        if first_obj and pre.find_parent("div", class_="doc-object"):
            break
        code = pre.find("code")
        if code:
            text = code.get_text().strip()
            if text.startswith("from fastapi") or text.startswith("import fastapi"):
                return text.split("\n")[0].strip()

    return ""


# ── Batch runner ──────────────────────────────────────────────────────────────

def scrape_all() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"\nScraping {len(REFERENCE_PAGES)} reference pages...\n")

    for path in REFERENCE_PAGES:
        safe_name = path.strip("/").replace("/", "-") + ".json"
        out_path  = OUTPUT_DIR / safe_name

        if out_path.exists():
            print(f"  SKIP (cached) {path}")
            continue

        page = scrape_reference_page(path)

        if page is None:
            continue

        out_path.write_text(
            json.dumps(page, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        results.append(page)
        time.sleep(DELAY)

    total_objects = sum(len(p["objects"]) for p in results)
    print(f"\n{'─'*60}")
    print(f"  Pages scraped:   {len(results)}/{len(REFERENCE_PAGES)}")
    print(f"  Total objects:   {total_objects}")
    print(f"  Output:          {OUTPUT_DIR}/")
    print(f"{'─'*60}\n")

    return results


# ── Debug helpers ─────────────────────────────────────────────────────────────

def inspect_page(path: str = "/reference/background/"):
    """
    Scrape one page and print a detailed summary without saving.
    Use to verify extraction quality before full run.

    Usage:
        from scrape_reference import inspect_page
        inspect_page("/reference/background/")
    """
    page = scrape_reference_page(path)
    if not page:
        print("Failed to scrape.")
        return

    print(f"\n{'='*60}")
    print(f"Title:        {page['title']}")
    print(f"Module:       {page['module']}")
    print(f"Import:       {page['import_from']}")
    print(f"Objects:      {len(page['objects'])}")
    print(f"Related:      {len(page['related_pages'])}")
    print(f"{'='*60}")

    for obj in page["objects"]:
        indent = "    " if obj["parent"] else ""
        print(f"\n{indent}[{obj['kind'].upper()}] {obj['qualified_name']}")
        if obj["signature"]:
            print(f"{indent}  Signature:   {obj['signature'][:80]}")
        if obj["bases"]:
            print(f"{indent}  Bases:       {obj['bases']}")
        if obj["labels"]:
            print(f"{indent}  Labels:      {obj['labels']}")
        if obj["description"]:
            print(f"{indent}  Description: {obj['description'][0][:80]}")
        if obj["parameters"]:
            print(f"{indent}  Parameters:  {len(obj['parameters'])}")
            for p in obj["parameters"][:3]:
                req = "" if p["required"] else f" = {p['default']}"
                print(f"{indent}    {p['name']}: {p['type']}{req}")
                if p["description"]:
                    print(f"{indent}      → {p['description'][:60]}")
        if obj["examples"]:
            print(f"{indent}  Examples:    {len(obj['examples'])}")
            print(f"{indent}    [{obj['examples'][0]['label']}] {obj['examples'][0]['code'][:60].strip()}...")
        if obj["admonitions"]:
            a = obj["admonitions"][0]
            print(f"{indent}  Admonition:  [{a['type']}] {a['title']} — {a['body'][:50]}")
        if obj["children"]:
            print(f"{indent}  Children:    {obj['children']}")
        if obj["deprecated"]:
            print(f"{indent}  ⚠️  DEPRECATED")


def inspect_raw_objects(path: str = "/reference/background/"):
    """
    Print the count and class structure of all div.doc-object on a page.
    Useful for debugging when inspect_page() shows 0 objects.
    """
    url = BASE_URL + path.rstrip("/") + "/"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    content = get_content_container(soup)
    if not content:
        print("No content container found")
        return

    all_objs = content.find_all("div", class_="doc-object")
    top_objs = _find_top_level_objects(content)

    print(f"All doc-objects:     {len(all_objs)}")
    print(f"Top-level objects:   {len(top_objs)}")
    print()
    for obj in top_objs:
        classes  = obj.get("class", [])
        heading  = obj.find(class_="doc-heading")
        name     = clean_text(heading.get_text()) if heading else "?"
        children = obj.find("div", class_="doc-children")
        n_children = len(children.find_all("div", class_="doc-object", recursive=False)) \
                     if children else 0
        print(f"  {name:<40} classes={classes}  children={n_children}")

def debug_fastapi_examples(path="/reference/fastapi/"):
    import requests
    from bs4 import BeautifulSoup
    
    url = "https://fastapi.tiangolo.com" + path
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    content = get_content_container(soup)
    
    first_obj = content.find("div", class_="doc-object")
    contents = first_obj.find("div", class_="doc-contents", recursive=False)
    
    # Remove children first
    children = contents.find("div", class_="doc-children")
    if children:
        children.extract()
    
    # Count pre>code blocks remaining
    pre_tags = contents.find_all("pre")
    tabsets  = contents.find_all("div", class_="tabbed-set")
    
    print(f"pre tags after children removed: {len(pre_tags)}")
    print(f"tabbed-sets after children removed: {len(tabsets)}")
    
    # Print first 3
    for i, pre in enumerate(pre_tags[:3]):
        code = pre.find("code")
        if code:
            print(f"\n[{i}] {code.get_text()[:80].strip()}")




if __name__ == "__main__":
    # Inspect one page first — verify before full run
    #inspect_page("/reference/fastapi/")

    # Then full run:
    scrape_all()