import re
from enum import Enum
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

DB_PATH = Path("./enterprise_data/chroma_db_v2")


print("Initializing Qwen3-Embedding-0.6B on CUDA...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_or_create_collection(
    name="engineering_knowledge",
    embedding_function=embedding_function
)


class Intent(Enum):
    CODE_LOOKUP      = "code_contracts"      # what does X do, signature, args
    EXAMPLE_REQUEST  = "code_examples"       # show me, how to write
    BUG_SEARCH       = "bug_history"         # error, not working, broken
    DECISION_LOOKUP  = "decision_history"    # why, reasoning, design
    CONCEPT_LOOKUP   = "wikis_arch_docs"     # how does, what is, explain
    AMBIGUOUS        = None                  # search multiple

@dataclass
class ClassifiedQuery:
    original: str
    rewritten: str
    intent: Intent
    confidence: str          # "high" | "medium" | "low"
    search_types: list[str]  # actual ChromaDB filter values


# ── Signal patterns per intent ────────────────────────────────────────────────

INTENT_PATTERNS = [

    # CODE_LOOKUP — user wants a specific function/class signature
    (Intent.CODE_LOOKUP, "high", [
        r"\b(what arguments?|what params?|signature of|return type of)\b",
        r"\b(takes|accepts)\b.{0,30}\b(argument|param|input)\b",
        r"\bhow (is|was) .{0,30} (defined|declared|typed)\b",
        r"\bdef .+\(|class .+:\b",
    ]),

    # EXAMPLE_REQUEST — user wants runnable code
    (Intent.EXAMPLE_REQUEST, "high", [
        r"\b(show me|give me|example of|sample|snippet|boilerplate|template)\b",
        r"\bhow (do I|to|can I) (write|create|define|build|implement)\b",
        r"\bwhat does.{0,30}look like\b",
    ]),

    # BUG_SEARCH — user has a problem right now
    (Intent.BUG_SEARCH, "high", [
        r"\b(error|exception|traceback|stacktrace)\b",
        r"\b(not working|doesn'?t work|broken|fails?|crashing)\b",
        r"\b(getting|throws?|raises?)\b.{0,30}\b(error|exception)\b",
        r"\bwhy (is|does|isn'?t|doesn'?t)\b",
    ]),

    # DECISION_LOOKUP — user wants reasoning/history
    (Intent.DECISION_LOOKUP, "high", [
        r"\bwhy (was|were|did|is|are)\b.{0,40}\b(added|introduced|designed|built|changed|removed|deprecated)\b",
        r"\b(reasoning|rationale|motivation|intention) (behind|for|of)\b",
        r"\bwhat (was|is) the (reason|purpose|goal|decision)\b",
        r"\bhow (was|were) .{0,30} (implemented|designed|architected)\b",
        r"\bwho decided\b",
    ]),

    # CONCEPT_LOOKUP — user wants explanation
    (Intent.CONCEPT_LOOKUP, "high", [
        r"\b(what is|what are|explain|describe|tell me about)\b",
        r"\bhow does .{0,30} work\b",
        r"\bwhat (is the purpose|does .{0,30} do)\b",
        r"\b(deploy|setup|configure|install|run)\b",
    ]),
]

# Medium confidence — overlapping signals
MEDIUM_PATTERNS = [
    (Intent.BUG_SEARCH,     r"\b(issue|problem|trouble|debug)\b"),
    (Intent.CODE_LOOKUP,    r"\b(function|method|class|import|module)\b"),
    (Intent.CONCEPT_LOOKUP, r"\bhow (to|do)\b"),
    (Intent.EXAMPLE_REQUEST,r"\b(use|using|usage)\b"),
    (Intent.DECISION_LOOKUP,r"\bwhy\b"),
]


def classify_intent(query: str) -> ClassifiedQuery:
    q = query.lower().strip()

    # ── Pass 1: High confidence patterns ─────────────────────────────────────
    for intent, confidence, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q):
                return ClassifiedQuery(
                    original=query,
                    rewritten=_rewrite(query, intent),
                    intent=intent,
                    confidence="high",
                    search_types=_resolve_types(intent, "high")
                )

    # ── Pass 2: Medium confidence ─────────────────────────────────────────────
    for intent, pattern in MEDIUM_PATTERNS:
        if re.search(pattern, q):
            return ClassifiedQuery(
                original=query,
                rewritten=_rewrite(query, intent),
                intent=intent,
                confidence="medium",
                search_types=_resolve_types(intent, "medium")
            )

    # ── Pass 3: Ambiguous — search broadly ───────────────────────────────────
    return ClassifiedQuery(
        original=query,
        rewritten=query,
        intent=Intent.AMBIGUOUS,
        confidence="low",
        search_types=["wikis_arch_docs", "code_contracts", "bug_history"]
        # deliberately exclude decision_history from ambiguous
        # it was overpowering everything — only include when explicitly needed
    )


def _resolve_types(intent: Intent, confidence: str) -> list[str]:
    """Map intent + confidence to ChromaDB filter types."""

    mapping = {
        Intent.CODE_LOOKUP:     {
            "high":   ["code_contracts"],
            "medium": ["code_contracts", "wikis_arch_docs"],
        },
        Intent.EXAMPLE_REQUEST: {
            "high":   ["code_examples"],
            "medium": ["code_examples", "code_contracts"],
        },
        Intent.BUG_SEARCH:      {
            "high":   ["bug_history"],
            "medium": ["bug_history", "wikis_arch_docs"],
        },
        Intent.DECISION_LOOKUP: {
            "high":   ["decision_history"],
            "medium": ["decision_history", "wikis_arch_docs"],
        },
        Intent.CONCEPT_LOOKUP:  {
            "high":   ["wikis_arch_docs"],
            "medium": ["wikis_arch_docs", "code_examples"],
        },
    }
    return mapping.get(intent, {}).get(confidence, ["wikis_arch_docs"])




def _rewrite(query: str, intent: Intent) -> str:
    q = query.lower()
    prefixes = {
        Intent.CODE_LOOKUP:     "Represent this question for searching code signatures and API definitions: ",
        Intent.EXAMPLE_REQUEST: "Represent this question for searching code examples and tutorials: ",
        Intent.BUG_SEARCH:      "Represent this question for searching bug reports and issue discussions: ",
        Intent.DECISION_LOOKUP: "Represent this question for searching architectural decisions and PR reasoning: ",
        Intent.CONCEPT_LOOKUP:  "Represent this question for searching technical documentation: ",
        Intent.AMBIGUOUS:       "Represent this question for searching technical documentation: ",
    }
    
    prefix=prefixes.get(intent,"")
    rewritten=prefix+query
    return rewritten
    



def retrieve(
    collection,
    query: str,
    n_results: int = 5,
    score_threshold: float = 0.25
) -> list[dict]:

    classified = classify_intent(query)

    print(f"\nQuery:      {classified.original}")
    print(f"Intent:     {classified.intent.name} ({classified.confidence})")
    print(f"Rewritten:  {classified.rewritten}")
    print(f"Searching:  {classified.search_types}")

    # Build ChromaDB where clause
    if len(classified.search_types) == 1:
        where = {"type": classified.search_types[0]}
    else:
        where = {"type": {"$in": classified.search_types}}

    results = collection.query(
        query_texts=[classified.rewritten],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    top_score = 1 - results["distances"][0][0]

    # Fallback — if confidence too low, broaden search
    # but always exclude decision_history unless explicitly asked
    if top_score < score_threshold:
        print(f"  Low score ({top_score:.3f}) → broadening search")
        fallback_types = ["wikis_arch_docs", "code_contracts",
                          "code_examples", "bug_history"]
        if classified.intent == Intent.DECISION_LOOKUP:
            fallback_types.append("decision_history")

        results = collection.query(
            query_texts=[classified.rewritten],
            n_results=n_results,
            where={"type": {"$in": fallback_types}},
            include=["documents", "metadatas", "distances"]
        )

    # Format output
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "content": doc,
            "type": meta["type"],
            "source": meta.get("source", ""),
            "score": round(1 - dist, 3)
        })
        print(f"  [{meta['type']:20s}] score: {1-dist:.3f} | {meta.get('source','')[:60]}")

    return chunks


queries = [
    "what arguments does HTTPException take",
    "show me how to use BackgroundTasks",
    "CORS error when calling from browser",
    "why was root_path added",
    "how does dependency injection work",
    "async route handler signature",
    "dependency injection not working",
]

for q in queries:
    chunks = retrieve(collection, q, n_results=3)
    print()