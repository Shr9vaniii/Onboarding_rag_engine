"""
Enrichment + Embedding Pipeline
================================
Converts all structured JSON sources into natural language chunks
and embeds them into ChromaDB using Qwen3-Embedding-0.6B.

Sources handled:
  1. code_contracts/    AST-parsed .py files (fastapi/, docs_src/, tests/)
  2. reference_scraped/ Reference page objects (classes, methods, attributes)
  3. issues/            Summarised GitHub issues
  4. discussions/       Summarised GitHub discussions
  5. wikis/             Scraped doc pages (tutorial, advanced, how-to, deployment)

Colab usage:
  1. Mount Drive first:
       from google.colab import drive
       drive.mount('/content/drive')
  2. Set BASE_DIR to your Drive path (see CONFIG below)
  3. Run embed_all() — resumes from checkpoint automatically

Checkpoint:
  Saved after every file. If Colab disconnects, re-run and it continues
  from where it left off. Checkpoint lives in BASE_DIR/embedding_checkpoint.json

Memory management:
  - Embeds in small batches (EMBED_BATCH_SIZE)
  - Clears CUDA cache between batches
  - Prints VRAM usage per batch
"""

import gc
import json
import re
import uuid
from pathlib import Path
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Change this to your Google Drive path when running on Colab:
# BASE_DIR = Path("/content/drive/MyDrive/enterprise_data")
BASE_DIR=Path("drive/MyDrive/Data")
CODE_CONTRACTS_DIR  = BASE_DIR / "code_contracts_v3"
REFERENCE_DIR       =BASE_DIR / "reference_scraped"
ISSUES_DIR          = BASE_DIR /  "issues"
DISCUSSIONS_DIR     =BASE_DIR /  "discussion"
WIKIS_DIR           = BASE_DIR / "wikis_and_docs_scraped_v2"
DB_PATH             = BASE_DIR / "chroma_db_v3"
CHECKPOINT_PATH     = BASE_DIR / "embedding_checkpoint.json"

EMBED_BATCH_SIZE    = 32     # safe for T4 with Qwen3-0.6B
COLLECTION_NAME     = "engineering_knowledge"

# Qwen3-Embedding-0.6B via SentenceTransformers
MODEL_NAME          = "Qwen/Qwen3-Embedding-0.6B"

# Instruction prefixes per chunk type — Qwen3 understands these
PREFIXES = {
    "code_contracts":  "Represent this for searching code signatures and API definitions: ",
    "wiki_arch_docs":  "Represent this for searching technical documentation: ",
    "bug_history":     "Represent this for searching bug reports and issue discussions: ",
    "community_qa":    "Represent this for searching Q&A and community discussions: ",
    "reference":       "Represent this for searching API reference documentation: ",
}


# ── CHECKPOINT ────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {"embedded": [], "failed": []}

def save_checkpoint(cp: dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(cp, indent=2), encoding="utf-8")


# ── MEMORY ────────────────────────────────────────────────────────────────────

def clear_memory():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def print_vram():
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"    VRAM: {used:.1f}/{total:.1f} GB")
    except Exception:
        pass


# ── MODEL LOADER ──────────────────────────────────────────────────────────────

def load_embedding_model():
    """Loads Qwen3-Embedding-0.6B via SentenceTransformers."""
    from sentence_transformers import SentenceTransformer
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("Model loaded.\n")
    print_vram()
    return model


def load_collection():
    """Creates or loads ChromaDB collection."""
    import chromadb
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB collection: '{COLLECTION_NAME}' ({collection.count()} existing chunks)")
    return collection


# ── EMBED + ADD ───────────────────────────────────────────────────────────────

def embed_and_add(
    model,
    collection,
    chunks: list[dict],
    chunk_type: str
):
    """
    Embeds a list of chunks and adds them to ChromaDB.
    chunks: [{"content": str, "metadata": dict}]
    chunk_type: key into PREFIXES dict
    """
    if not chunks:
        return

    prefix = PREFIXES.get(chunk_type, "")

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        texts = [prefix + c["content"] for c in batch]
        metas = [c["metadata"] for c in batch]
        ids   = [str(uuid.uuid4()) for _ in batch]

        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            collection.add(
                embeddings=embeddings,
                documents=[c["content"] for c in batch],
                metadatas=metas,
                ids=ids,
            )
        except Exception as e:
            print(f"    Batch error: {e}")
        finally:
            clear_memory()


# ── ENRICHERS ─────────────────────────────────────────────────────────────────
# Each enricher takes a loaded JSON dict and returns list[{"content", "metadata"}]

# ── 1. CODE CONTRACTS ─────────────────────────────────────────────────────────

def _classify_contract(file_path: str) -> str:
    """Route file to enrichment strategy based on path prefix."""
    p = file_path.replace("\\", "/").lower()
    if p.startswith("tests/"):   return "test"
    if p.startswith("docs_src/"): return "example"
    if p.startswith("fastapi/"):  return "internal"
    if p.startswith("scripts/"):  return "skip"
    return "skip"


def _topic_from_path(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        return parts[-2].replace("_", " ")
    return parts[-1].replace(".py", "").replace("_", " ")


def enrich_code_contract(data: dict) -> list[dict]:
    file_path = data.get("file_path", "")
    elements  = data.get("elements", {})
    strategy  = _classify_contract(file_path)

    if strategy == "skip":
        return []

    chunks = []
    topic  = _topic_from_path(file_path)

    imports  = elements.get("imports", [])
    classes  = elements.get("classes", [])
    functions = elements.get("functions", [])

    # Build method → class lookup to know if a function is a method
    method_to_class = {}
    for cls in classes:
        for method in cls.get("methods", []):
            method_to_class[method] = cls["name"]

    # Key imports (FastAPI-related)
    key_imports = [
        f"`{sym}` from `{imp['module']}`"
        for imp in imports
        for sym in imp.get("symbols", [])
        if imp.get("module", "").startswith(("fastapi", "pydantic", "starlette"))
    ]

    # ── Pydantic models ───────────────────────────────────────────────────────
    for cls in classes:
        attrs    = cls.get("attributes", [])
        methods  = cls.get("methods", [])
        is_model = "BaseModel" in cls.get("bases", [])

        lines = [
            f"{'Schema' if is_model else 'Class'}: {cls['name']}",
            f"File: {file_path}",
            f"Topic: {topic}",
        ]

        if is_model:
            lines.append(f"Used for: request body or response schema in {topic}")
        
        if cls.get("bases"):
            lines.append(f"Inherits: {', '.join(cls['bases'])}")

        # Fields / Attributes
        if attrs:
            lines.append("\nFields:" if is_model else "\nAttributes:")
            for a in attrs:
                field_line = f"  {a['name']}: {a.get('type', 'Any')}"
                if a.get("default"):
                    field_line += f" = {a['default']} (optional)"
                else:
                    field_line += " (required)"
                lines.append(field_line)

            required = [a["name"] for a in attrs if not a.get("default")]
            optional = [a["name"] for a in attrs if a.get("default")]
            if required:
                lines.append(f"Required: {', '.join(required)}")
            if optional:
                lines.append(f"Optional: {', '.join(optional)}")

        # Methods list (just names — bodies are separate function chunks)
        if methods:
            lines.append(f"Methods: {', '.join(methods)}")

        if cls.get("docstring"):
            lines.append(f"\nDescription: {cls['docstring'][:200]}")
        else:
            field_names = ", ".join(a["name"] for a in attrs[:5])
            if is_model:
                lines.append(
                    f"\nDescription: Data schema for {topic} "
                    f"with fields: {field_names}"
                )
            else:
                lines.append(f"\nDescription: Class {cls['name']} in {topic}")

        chunks.append({
            "content": "\n".join(lines),
            "metadata": {
                "type":    "code_contracts",
                "subtype": strategy,
                "source":  file_path,
                "topic":   topic,
                "name":    cls["name"],
                "kind":    "class",
            }
        })

    # ── Functions / routes ────────────────────────────────────────────────────
    for fn in functions:
        fn_name = fn["name"]
        is_method = fn_name in method_to_class

        lines = [f"Function: {fn_name}", f"File: {file_path}", f"Topic: {topic}"]
        calls       = fn.get("calls", [])

        # Signature
        
        args = fn.get("args", [])
        arg_strs = []
        for a in args:
            if a["name"] in ("self", "cls"):
                continue
            s = f"{a['name']}: {a['type']}" if a.get("type") else a["name"]
            if a.get("default"):
                s += f" = {a['default']}"
            arg_strs.append(s)

        is_async = fn.get("async", False)
        prefix_kw = "async def" if is_async else "def"
        return_type = fn.get("return_type", "")
        sig = f"{prefix_kw} {fn_name}({', '.join(arg_strs)})"
        if return_type:
            sig += f" -> {return_type}"
        lines.append(f"Signature: {sig}")

        if return_type:
            lines.append(f"Returns: {return_type}")


        if args:
            lines.append("\nArguments:")
            for a in args:
                req  = "required" if not a.get("default") \
                       else f"optional, default={a['default']}"
                line = f"  {a['name']}: {a.get('type', 'Any')} — {req}"
                if a.get("doc"):
                    line += f" — {a['doc']}"
                lines.append(line)

        # HTTP route info
        for dec in fn.get("decorators", []):
            if dec.get("http_method"):
                route_line = f"HTTP: {dec['http_method']} {dec.get('route', '')}"
                if dec.get("response_model"):
                    route_line += f" → {dec['response_model']}"
                lines.append(route_line)

        # Docstring
        if fn.get("docstring"):
            lines.append(f"Description: {fn['docstring'][:200]}")

        # Dependencies
        if fn.get("dependencies"):
            lines.append(f"Dependencies: {', '.join(fn['dependencies'])}")

        # Raises
        if fn.get("raises"):
            lines.append(f"Raises: {', '.join(fn['raises'])}")

        fa_calls = [c for c in calls if any(
            kw in c for kw in ("Depends", "HTTPException", "Response",
                               "BackgroundTask", "Security", "Query",
                               "Path", "Body", "Header", "Cookie")
        )]
        if fa_calls:
            lines.append(f"Calls: {', '.join(fa_calls)}")


        # Source code — only for docs_src and tests
        if fn.get("source_code"):
            lines.append(f"\nCode:\n```python\n{fn['source_code']}\n```")

        # Natural language description
        decorators = fn.get("decorators", [])
        has_route  = any(d.get("http_method") for d in decorators)

        if strategy == "example" and has_route:
            methods = [d["http_method"] for d in decorators if d.get("http_method")]
            route   = next((d.get("route", "") for d in decorators if d.get("route")), "")
            desc = f"Example showing how to implement a {'/'.join(methods)} endpoint at {route} for {topic}"
        elif strategy == "test":
            desc = f"Test function demonstrating how to test {topic} in FastAPI"
        elif strategy == "internal":
            desc = f"Internal FastAPI function {fn_name} in {file_path}"
        else:
            desc = f"Function {fn_name} for {topic}"

        if fn.get("docstring"):
            desc = fn["docstring"][:150]

        lines.append(f"\nDescription: {desc}")

        chunks.append({
            "content": "\n".join(lines),
            "metadata": {
                "type":     "code_contracts",
                "subtype":  strategy,
                "source":   file_path,
                "topic":    topic,
                "name":     fn_name,
                "kind":     "function",
                "has_route": str(has_route),
                "is_async":  str(is_async),
            }
        })

    return chunks


# ── 2. REFERENCE PAGES ────────────────────────────────────────────────────────

def enrich_reference_object(obj: dict, page_url: str, import_from: str) -> dict | None:
    """One reference object → one chunk."""
    kind           = obj.get("kind", "object")
    qualified_name = obj.get("qualified_name", "")
    signature      = obj.get("signature", "")
    description    = obj.get("description", [])
    parameters     = obj.get("parameters", [])
    examples       = obj.get("examples", [])
    admonitions    = obj.get("admonitions", [])
    bases          = obj.get("bases", [])
    labels         = obj.get("labels", [])
    parent         = obj.get("parent")
    related_pages= obj.get("related_pages", [])

    lines = [f"{kind.title()}: {qualified_name}"]

    if import_from:
        lines.append(f"Import: {import_from}")

    if bases:
        lines.append(f"Inherits from: {', '.join(bases)}")

    if labels:
        lines.append(f"Labels: {', '.join(labels)}")

    if signature:
        lines.append(f"Signature: {signature}")

    if description:
        lines.append(f"\nDescription: {' '.join(description)}")

    if parameters:
        lines.append("\nParameters:")
        for p in parameters:
            req  = "" if p.get("required", True) else f" = {p.get('default', '')}"
            line = f"  {p['name']}: {p.get('type', 'Any')}{req}"
            if p.get("description"):
                line += f" — {p['description'][:120]}"
            lines.append(line)

    if admonitions:
        for admon in admonitions:
            lines.append(f"\n[{admon['title']}] {admon['body']}")

    related_str = ",".join(                         # ← added
        p.get("slug", p.get("target_path", ""))
        for p in related_pages[:5]
    )

    # Best example — first one only to keep chunk size reasonable
    if examples:
        ex = examples[0]
        lines.append(f"\nExample:\n```python\n{ex['code'][:500].strip()}\n```")

    content = "\n".join(lines).strip()
    if len(content) < 20:
        return None

    return {
        "content": content,
        "metadata": {
            "type":           "wikis_arch_docs",
            "subtype":        "reference",
            "source":         page_url,
            "qualified_name": qualified_name,
            "kind":           kind,
            "parent":         parent or "",
            "section":        "reference",
            "related_pages":   related_str,
        }
    }


def enrich_reference_page(data: dict) -> list[dict]:
    page_url    = data.get("url", "")
    import_from = data.get("import_from", "")
    objects     = data.get("objects", [])

    chunks = []
    for obj in objects:
        chunk = enrich_reference_object(obj, page_url, import_from)
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 3. ISSUES (summarised) ────────────────────────────────────────────────────

def enrich_issue(data: dict) -> list[dict]:
    chunks = []
    url    = data.get("url", "")

    summary    = data.get("summary", "")
    problem    = data.get("problem", "")
    root_cause = data.get("root_cause", "")
    solution   = data.get("solution", "")
    workaround = data.get("workaround", "")
    onboarding = data.get("onboarding_note", "")
    lessons    = data.get("lessons_learned", [])
    components = data.get("affected_components", [])
    keywords   = data.get("keywords", [])
    status     = data.get("status", "")
    versions   = data.get("versions", {})
    confidence = data.get("confidence", "")

    # Skip low-confidence summaries
    if confidence == "low" and not solution and not workaround:
        return []

    version_str = ""
    affected = versions.get("affected", [])
    fixed    = versions.get("fixed", [])
    if affected:
        version_str = f"Affected: {', '.join(affected)}"
    if fixed:
        version_str += f" | Fixed: {', '.join(fixed)}"

    # ── Chunk 1: Problem + root cause ─────────────────────────────────────────
    lines = [f"Bug Report: {summary}"]
    if problem:
        lines.append(f"\nProblem: {problem}")
    if root_cause:
        lines.append(f"Root cause: {root_cause}")
    if components:
        lines.append(f"Affected: {', '.join(components)}")
    if version_str:
        lines.append(f"Versions: {version_str}")
    if status:
        lines.append(f"Status: {status}")

    chunks.append({
        "content": "\n".join(lines),
        "metadata": {
            "type":       "bug_history",
            "subtype":    "problem",
            "source":     url,
            "status":     status,
            "confidence": confidence,
            "components": ",".join(components),
            "keywords":   ",".join(keywords),  
        
        }
    })

    # ── Chunk 2: Solution + workaround ────────────────────────────────────────
    if solution or workaround:
        lines = [f"Fix for: {summary}"]
        if solution:
            lines.append(f"\nSolution: {solution}")
        if workaround:
            lines.append(f"Workaround: {workaround}")
        if onboarding:
            lines.append(f"\nOnboarding note: {onboarding}")
        if lessons:
            lines.append("\nLessons learned:")
            for lesson in lessons:
                lines.append(f"  - {lesson}")

        chunks.append({
            "content": "\n".join(lines),
            "metadata": {
                "type":       "bug_history",
                "subtype":    "solution",
                "source":     url,
                "status":     status,
                "confidence": confidence,
                "components": ",".join(components),
                "keywords":   ",".join(keywords),  
            }
            
        })

    return chunks


# ── 4. DISCUSSIONS (summarised) ───────────────────────────────────────────────

def enrich_discussion(data: dict) -> list[dict]:
    chunks = []
    url    = data.get("url", "")

    question   = data.get("question", "")
    answer     = data.get("answer", "")
    context    = data.get("context", "")
    insights   = data.get("community_insights", [])
    code_ex    = data.get("code_examples", [])
    onboarding = data.get("onboarding_note", "")
    keywords   = data.get("keywords", [])
    status     = data.get("status", "")
    confidence = data.get("confidence", "")

    if confidence == "low" and not answer:
        return []

    # ── Chunk 1: Q + A ────────────────────────────────────────────────────────
    lines = []
    if question:
        lines.append(f"Question: {question}")
    if context:
        lines.append(f"Context: {context}")
    if answer:
        lines.append(f"\nAnswer: {answer}")
    if insights:
        lines.append("\nCommunity insights:")
        for insight in insights[:3]:
            lines.append(f"  - {insight}")
    if onboarding:
        lines.append(f"\nTakeaway: {onboarding}")
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")

    if lines:
        chunks.append({
            "content": "\n".join(lines),
            "metadata": {
                "type":       "community_qa",
                "subtype":    "qa",
                "source":     url,
                "status":     status,
                "confidence": confidence,
                "is_answered": str(data.get("is_answered", False)),
                "keywords":   ",".join(keywords)
            }
        })

    # ── Chunk 2: Code examples ────────────────────────────────────────────────
    # Clean and embed meaningful code examples
    clean_codes = []
    for ex in code_ex:
        if not isinstance(ex, str):
            continue
        # Strip markdown fences
        clean = re.sub(r"```\w*\n?", "", ex).strip()
        if len(clean) > 50:
            clean_codes.append(clean)

    if clean_codes and question:
        lines = [
            f"Code example for: {question}",
            f"\n```python\n{clean_codes[0][:800]}\n```"
        ]
        if onboarding:
            lines.append(f"\nNote: {onboarding}")

        chunks.append({
            "content": "\n".join(lines),
            "metadata": {
                "type":       "community_qa",
                "subtype":    "code_example",
                "source":     url,
                "status":     status,
                "confidence": confidence,
                "keywords":   ",".join(keywords)
            }
        })

    return chunks


# ── 5. WIKI / ARCH DOCS ───────────────────────────────────────────────────────

def _pick_best_code(code_blocks: list[dict]) -> dict | None:
    """Pick newest Python version from code blocks."""
    if not code_blocks:
        return None
    priority = ["3.13", "3.12", "3.11", "3.10", "3.9", "3.8", "Annotated"]
    for pref in priority:
        for block in code_blocks:
            if pref in block.get("version", ""):
                return block
    return code_blocks[-1]


def _flatten_mentions(mentions: dict | list) -> str:
    """Flatten mentions_classes dict or list to comma-separated string."""
    if isinstance(mentions, list):
        return ",".join(mentions)
    if isinstance(mentions, dict):
        all_items = []
        for v in mentions.values():
            if isinstance(v, list):
                all_items.extend(v)
        return ",".join(sorted(set(all_items)))
    return ""


def enrich_wiki_section(section: dict, page_data: dict) -> list[dict]:
    """One wiki section → up to 2 chunks (prose + code)."""
    chunks = []

    header_path  = " > ".join(section.get("header_path", []))
    prose        = section.get("prose", [])
    code_blocks  = section.get("code_blocks", [])
    admonitions  = section.get("admonitions", [])
    mentions     = section.get("mentions_classes", {})
    tables       = section.get("tables", [])
    related_pages= section.get("related_pages", [])
    page_url     = page_data.get("url", "")
    page_title   = page_data.get("title", "")
    section_name = page_data.get("section", "")
    related      = ",".join(
        p.get("target_url", p.get("target_path", ""))
        for p in page_data.get("related_pages", [])[:7]
    )
    related_lines=[]
    for r in related_pages:
        anchor=r.get("anchor_text","")
        if anchor and len(anchor)>3:
            related_lines.append(f" -{anchor}")

    
    mentions_str = _flatten_mentions(mentions)

    # ── Prose chunk ───────────────────────────────────────────────────────────
    prose_parts = []

    if prose:
        prose_parts.append("\n".join(prose))

    '''for admon in admonitions:
        prose_parts.append(f"[{admon['title']}] {admon['body']}")'''

    # Tables → plain text
    for table in tables:
        if table.get("kind") == "table":
            headers = " | ".join(table.get("headers", []))
            rows    = "\n".join(" | ".join(row) for row in table.get("rows", []))
            prose_parts.append(f"{headers}\n{rows}".strip())
        elif table.get("items"):
            for item in table["items"]:
                parts = [f"`{item.get('name', '')}`"]
                if item.get("annotation"):
                    parts.append(f"type: {item['annotation']}")
                if item.get("default"):
                    parts.append(f"default: {item['default']}")
                if item.get("description"):
                    parts.append(item["description"])
                prose_parts.append(" — ".join(parts))
    lines=""

    if related_lines:
        lines = f"\n\nRelated Documentation: {', '.join(related_lines)}"

    if prose_parts:
        content = (
            f"Page: {page_title} | {header_path}\n\n"
            + "\n\n".join(prose_parts) + lines
        )
        chunks.append({
            "content": content,
            "metadata": {
                "type":             "wikis_arch_docs",
                "subtype":          "prose",
                "source":           page_url,
                "title":            page_title,
                "section":          section_name,
                "header_path":      header_path,
                "mentions_classes": mentions_str,
                "related_pages":    related,
                "has_code":         str(bool(code_blocks)),
                "has_admonition":   str(bool(admonitions)),
            }
        })

    # ── Code chunk ────────────────────────────────────────────────────────────
    best_code = _pick_best_code(code_blocks)
    if best_code:
        code_content = (
            f"Code example: {page_title} | {header_path}\n"
            f"Version: {best_code.get('version', 'python')}\n\n"
            f"```python\n{best_code['code'].strip()}\n```"
        )
        chunks.append({
            "content": code_content,
            "metadata": {
                "type":          "wikis_arch_docs",
                "subtype":       "code_example",
                "source":        page_url,
                "title":         page_title,
                "section":       section_name,
                "header_path":   header_path,
                "python_version": best_code.get("version", "unknown"),
                "related_pages": related,
            }
        })

    return chunks


def enrich_wiki_page(data: dict) -> list[dict]:
    chunks = []
    for section in data.get("sections", []):
        chunks.extend(enrich_wiki_section(section, data))
    return chunks


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def process_source(
    source_dir: Path,
    enricher,
    chunk_type: str,
    model,
    collection,
    checkpoint: dict,
    glob_pattern: str = "*.json",
    label: str = ""
) -> int:
    """
    Generic processor: reads JSON files from source_dir, enriches,
    embeds, checkpoints. Returns count of files processed.
    """
    if not source_dir.exists():
        print(f"  SKIP {label} — directory not found: {source_dir}")
        return 0

    files = sorted(source_dir.glob(glob_pattern))
    embedded_set = set(checkpoint.get("embedded", []))
    pending = [f for f in files if f.name not in embedded_set]

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  Total: {len(files)} | Already done: {len(files)-len(pending)} | Pending: {len(pending)}")
    print(f"{'─'*55}")

    processed = 0
    for i, file_path in enumerate(pending, 1):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            chunks = enricher(data)

            if chunks:
                embed_and_add(model, collection, chunks, chunk_type)
                print(f"  [{i:>5}/{len(pending)}] {file_path.name:<40} → {len(chunks)} chunks")
            else:
                print(f"  [{i:>5}/{len(pending)}] {file_path.name:<40} → skipped (no chunks)")

            checkpoint["embedded"].append(file_path.name)
            save_checkpoint(checkpoint)
            processed += 1

        except Exception as e:
            print(f"  [{i:>5}/{len(pending)}] {file_path.name:<40} ERROR: {e}")
            checkpoint["failed"].append(file_path.name)
            save_checkpoint(checkpoint)

    return processed


def embed_all():
    """Main entry point — embeds all sources into ChromaDB."""

    print("\n" + "="*55)
    print("  ENRICHMENT + EMBEDDING PIPELINE")
    print("="*55)

    # Load model + collection
    model      = load_embedding_model()
    collection = load_collection()
    checkpoint = load_checkpoint()

    total = 0

    # ── 1. Code contracts ─────────────────────────────────────────────────────
    total += process_source(
        source_dir   = CODE_CONTRACTS_DIR,
        enricher     = enrich_code_contract,
        chunk_type   = "code_contracts",
        model        = model,
        collection   = collection,
        checkpoint   = checkpoint,
        label        = "Code Contracts (fastapi/ + docs_src/ + tests/)",
    )

    # ── 2. Reference pages ────────────────────────────────────────────────────
    total += process_source(
        source_dir   = REFERENCE_DIR,
        enricher     = enrich_reference_page,
        chunk_type   = "reference",
        model        = model,
        collection   = collection,
        checkpoint   = checkpoint,
        label        = "Reference Pages",
    )

    # ── 3. Issues ─────────────────────────────────────────────────────────────
    total += process_source(
        source_dir   = ISSUES_DIR,
        enricher     = enrich_issue,
        chunk_type   = "bug_history",
        model        = model,
        collection   = collection,
        checkpoint   = checkpoint,
        label        = "Summarised Issues (bug_history)",
    )

    # ── 4. Discussions ────────────────────────────────────────────────────────
    total += process_source(
        source_dir   = DISCUSSIONS_DIR,
        enricher     = enrich_discussion,
        chunk_type   = "community_qa",
        model        = model,
        collection   = collection,
        checkpoint   = checkpoint,
        label        = "Summarised Discussions (community_qa)",
    )

    # ── 5. Wikis ──────────────────────────────────────────────────────────────
    total += process_source(
        source_dir   = WIKIS_DIR,
        enricher     = enrich_wiki_page,
        chunk_type   = "wiki_arch_docs",
        model        = model,
        collection   = collection,
        checkpoint   = checkpoint,
        label        = "Wiki / Arch Docs",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Files processed:  {total}")
    print(f"  Total chunks:     {collection.count()}")
    print(f"  DB path:          {DB_PATH}")
    print(f"{'='*55}\n")

    # Cleanup
    del model
    clear_memory()
    print("Memory freed.")


# ── COLAB ENTRY POINT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For Colab — mount Drive first:
    #   from google.colab import drive
    #   drive.mount('/content/drive')
    #   Then set BASE_DIR at the top of this file to your Drive path
    #embed_all()
    '''data=json.loads(open(CODE_CONTRACTS_DIR/"docs_src-additional_responses-tutorial001_py310.json", "r", encoding="utf-8").read())
    chunks=enrich_code_contract(data)
    for c in chunks:
        print(c["content"])
        print("---------")'''