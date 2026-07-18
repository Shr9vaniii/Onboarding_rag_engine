"""
Prompt and context formatting — mirrors Fine-tuning/build_jsonl.py for train/inference parity.
"""

from __future__ import annotations

TYPE_SOURCE_LABELS = {
    "code_contracts": "code contract",
    "wikis_arch_docs": "documentation",
    "bug_history": "bug report",
    "community_qa": "community discussion",
    "reference": "API reference",
}

METADATA_FIELDS = {
    "code_contracts": ["source", "topic", "name", "kind", "subtype"],
    "wikis_arch_docs": [
        "source", "title", "header_path", "section",
        "mentions_classes", "related_pages", "subtype",
    ],
    "reference": ["source", "qualified_name", "kind", "parent", "section", "related_pages"],
    "bug_history": ["source", "subtype"],
    "community_qa": ["source", "subtype"],
}

METADATA_LABELS = {
    "source": "Source",
    "topic": "Topic",
    "name": "Name",
    "kind": "Kind",
    "subtype": "Subtype",
    "title": "Page",
    "header_path": "Section",
    "section": "Doc section",
    "mentions_classes": "Mentioned APIs",
    "related_pages": "Related pages",
    "qualified_name": "API",
    "parent": "Parent",
}

SYSTEM_PROMPT = """You are an expert onboarding assistant for a backend engineering team using FastAPI.

Your role is to help new engineers understand the codebase, APIs, bugs, and architectural decisions.

When answering:
- Base your answer STRICTLY on the provided context
- If the context is insufficient, say so explicitly
- Use this exact structure:

**Direct Answer:**
[One clear sentence answering the question directly]

**Details:**
[Technical explanation with specifics from the context]

**Source:**
[source type] — [exact Source path or URL from the metadata block when available]

If related pages are listed in the metadata and relevant, you may mention them in **Details** or **Recommendation**.
NOTE: Do not invent file paths, URLs, or related pages that are not in the context.

If you cannot answer from the context:
**Direct Answer:**
I don't have enough information in the provided context to answer this accurately.

**What I found:**
[What the context does contain, if anything relevant]

**Recommendation:**
[Where the engineer should look instead]"""


def resolve_chunk_type(metadata: dict, fallback: str = "wikis_arch_docs") -> str:
    if metadata.get("subtype") == "reference":
        return "reference"
    return metadata.get("type", fallback)


def format_metadata_header(metadata: dict, chunk_type: str) -> str:
    fields = METADATA_FIELDS.get(chunk_type, ["source", "subtype"])
    lines = []
    for key in fields:
        value = metadata.get(key, "")
        if not value or not str(value).strip():
            continue
        label = METADATA_LABELS.get(key, key.replace("_", " ").title())
        lines.append(f"{label}: {value}")
    if not lines:
        return ""
    return "--- Metadata ---\n" + "\n".join(lines)


def build_training_context(content: str, metadata: dict, chunk_type: str) -> str:
    header = format_metadata_header(metadata, chunk_type)
    if header:
        return f"{header}\n\n--- Content ---\n{content}"
    return content


def format_user_message(question: str, content: str, metadata: dict, chunk_type: str) -> str:
    display_type = TYPE_SOURCE_LABELS.get(chunk_type, chunk_type)
    full_context = build_training_context(content, metadata, chunk_type)
    return f"""Context ({display_type}):
{full_context}

Question: {question}"""


def format_multi_chunk_user_message(
    question: str,
    hits: list[dict],
    *,
    max_chars_per_chunk: int = 1200,
) -> str:
    """Format top-k retrieved chunks for the LLM (training used 1 chunk; inference may use several)."""
    if not hits:
        return f"Question: {question}"

    if len(hits) == 1:
        h = hits[0]
        return format_user_message(
            question,
            h["content"],
            h["metadata"],
            h.get("chunk_type") or resolve_chunk_type(h["metadata"]),
        )

    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        chunk_type = hit.get("chunk_type") or resolve_chunk_type(hit["metadata"])
        display_type = TYPE_SOURCE_LABELS.get(chunk_type, chunk_type)
        ctx = build_training_context(hit["content"], hit["metadata"], chunk_type)
        if len(ctx) > max_chars_per_chunk:
            ctx = ctx[:max_chars_per_chunk].rstrip() + "\n...[truncated]"
        blocks.append(f"Context {i} ({display_type}):\n{ctx}")

    n = len(blocks)
    intro = (
        f"The following {n} context blocks were retrieved. "
        "Answer using only information from these blocks.\n\n"
    )
    return f"{intro}{chr(10).join(blocks)}\n\nQuestion: {question}"""
