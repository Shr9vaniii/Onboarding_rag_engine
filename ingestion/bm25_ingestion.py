from __future__ import annotations

import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BM25_INDEX_PATH = ROOT / "enterprise_data" / "bm25_index.pkl"

# Keep tokenization identical to retrieval/hybrid_retrieval.tokenize_query
# (minus stopword filtering — index stores raw tokens; queries filter stopwords).
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize_document(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25_index(collection, *, index_path: Path | None = None) -> dict:
    """Build BM25 index from all documents in ChromaDB and save to disk."""
    from rank_bm25 import BM25Okapi

    path = index_path or BM25_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    print("Building BM25 index...")
    total = collection.count()
    print(f"Total chunks in collection: {total}")
    if total == 0:
        raise ValueError("Collection is empty — check your ChromaDB path and collection name")

    all_data = collection.get(limit=total, include=["documents", "metadatas"])
    documents = all_data["documents"]
    ids = all_data["ids"]
    metadatas = all_data["metadatas"]

    tokenized = [tokenize_document(doc) for doc in documents]
    index = BM25Okapi(tokenized)

    payload = {
        "index": index,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "tokenizer": "regex_a-z0-9_",
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"BM25 index built and saved to {path} for {len(documents)} documents.")
    return payload


def load_bm25_index(index_path: Path | None = None) -> dict | None:
    path = index_path or BM25_INDEX_PATH
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


_bm25_payload: dict | None = None


def get_bm25(collection, *, force_rebuild: bool = False) -> dict:
    global _bm25_payload
    if force_rebuild:
        _bm25_payload = None
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()

    if _bm25_payload is None:
        _bm25_payload = load_bm25_index()
        # Rebuild if missing or built with the old whitespace tokenizer.
        if _bm25_payload is None or _bm25_payload.get("tokenizer") != "regex_a-z0-9_":
            _bm25_payload = build_bm25_index(collection)
    return _bm25_payload


if __name__ == "__main__":
    from retrieval.hybrid_retrieval import get_collection

    get_bm25(get_collection(), force_rebuild=True)
