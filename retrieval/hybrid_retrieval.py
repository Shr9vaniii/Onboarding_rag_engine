from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
import re
from pathlib import Path
import chromadb
from ingestion.bm25_ingestion import get_bm25


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "enterprise_data" / "chroma_db_v3"
COLLECTION_NAME = "engineering_knowledge"

_chroma_client = None
_collection = None
_reranker = None
model = None
model_name = "Qwen/Qwen3-Embedding-0.6B"

# Common English + question words that pollute BM25 scoring for short queries.
BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "should", "so",
    "that", "the", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "me", "my", "take", "takes",
    "use", "using", "get", "getting", "there", "have", "has", "about", "arguments",
    "argument", "args", "arg", "param", "params", "parameter", "parameters",
}


def tokenize_query(query: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords. Falls back to raw tokens if empty."""
    raw = re.findall(r"[a-z0-9_]+", query.lower())
    filtered = [t for t in raw if t not in BM25_STOPWORDS and len(t) > 1]
    return filtered or raw


def metadata_matches(meta: dict, where: dict) -> bool:
    """Evaluate a Chroma-style `where` filter against a metadata dict (supports $and/$or/$ne/$in)."""
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            if not all(metadata_matches(meta, c) for c in cond):
                return False
        elif key == "$or":
            if not any(metadata_matches(meta, c) for c in cond):
                return False
        elif isinstance(cond, dict):
            for op, val in cond.items():
                actual = meta.get(key)
                if op == "$ne" and actual == val:
                    return False
                if op == "$eq" and actual != val:
                    return False
                if op == "$in" and actual not in val:
                    return False
                if op == "$nin" and actual in val:
                    return False
        else:
            if meta.get(key) != cond:
                return False
    return True


SIGNATURE_QUERY_RE = re.compile(
    r"\b(arguments?|params?|parameters?|signature|return type|takes|accepts|fields?)\b",
    re.I,
)


def extract_entities(query: str) -> list[str]:
    """PascalCase identifiers likely to be classes/APIs (HTTPException, Depends, ...)."""
    return re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", query)


def chunk_entity_names(metadata: dict) -> list[str]:
    names: list[str] = []
    for key in ("name", "qualified_name", "parent"):
        val = metadata.get(key, "")
        if val and str(val).strip():
            names.append(str(val).strip())
    return names


def entity_matches_chunk(entity: str, metadata: dict) -> bool:
    ent_lower = entity.lower()
    for name in chunk_entity_names(metadata):
        nl = name.lower()
        if nl == ent_lower or nl.endswith("." + ent_lower):
            return True
        if nl.split(".")[-1] == ent_lower:
            return True
    return False


def entity_boost(candidate: dict, query: str) -> float:
    """
    Boost definition chunks (class/reference) when query names a specific API.
    Penalize usage examples for signature/argument questions.
    """
    meta = candidate.get("metadata") or {}
    entities = extract_entities(query)
    if not entities:
        return 0.0

    signature_q = bool(SIGNATURE_QUERY_RE.search(query))
    kind = meta.get("kind", "")
    subtype = meta.get("subtype", "")

    matched = any(entity_matches_chunk(ent, meta) for ent in entities)
    if not matched:
        if signature_q and (subtype == "example" or kind == "function"):
            return -0.8
        return 0.0

    boost = 3.0
    if kind == "class":
        boost += 2.0
    elif kind == "method" and signature_q:
        boost -= 1.0  # prefer class/constructor over methods for "what arguments"
    if subtype == "reference":
        boost += 1.5
    if kind == "attribute" and signature_q:
        boost += 0.3
    if signature_q and subtype == "example":
        boost -= 2.0
    return boost


def fetch_entity_definitions(collection, query: str, where=None) -> list[dict]:
    """Direct metadata lookup for class/API definitions when query names an entity."""
    if not SIGNATURE_QUERY_RE.search(query):
        return []
    entities = extract_entities(query)
    if not entities:
        return []

    hits: list[dict] = []
    seen: set[str] = set()

    for ent in entities:
        for lookup in (
            {"name": ent},
            {"qualified_name": ent},
            {"qualified_name": f"fastapi.{ent}"},
        ):
            filt: dict = lookup
            if where:
                filt = {"$and": [where, lookup]}
            try:
                res = collection.get(
                    where=filt,
                    limit=5,
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue
            for doc_id, doc, meta in zip(
                res.get("ids", []),
                res.get("documents", []),
                res.get("metadatas", []),
            ):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                hits.append({"id": doc_id, "content": doc, "metadata": meta})
    return hits


PREFIXES = {
    "code_contracts": "Represent this for searching code signatures and API definitions: ",
    "wikis_arch_docs": "Represent this for searching technical documentation: ",
    "bug_history": "Represent this for searching bug reports and issue discussions: ",
    "community_qa": "Represent this for searching Q&A and community discussions: ",
    "reference": "Represent this for searching API reference documentation: ",
}
 
def load_model():
    global model
    if model is None:
        model=SentenceTransformer(model_name)
    return model

def get_embedding_for_query(query:str)->list[float]:
    model=load_model()
    embedding=model.encode(query).tolist()
    return embedding

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading Reranker...")
        _reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Reranker loaded.")

    return _reranker


def get_collection(*, verbose: bool = False):
    """Return a process-wide Chroma collection singleton (absolute DB path)."""
    global _chroma_client, _collection
    if _collection is None:
        DB_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        if verbose:
            print(f"Chroma collection '{COLLECTION_NAME}' count={_collection.count()}")
    return _collection


def hybrid_retrieve(
        query:str,
        intent:str,
        n_dense:int =10,
        n_bm25: int=10,
        n_final:int=5,
        where=None,
        rerank:bool=True
)-> list[dict]:
    """Full Pipeline:
    1.Dense retrieval (Qwen3 embeddings)
    2.BM25 sparse retrieval
    3.RRF fusion
    4.Cross-encoder reranking
    """


    collection=get_collection()
    prefix=PREFIXES.get(intent,PREFIXES["wikis_arch_docs"])
    rewritten_query=prefix+query
    embedding=get_embedding_for_query(rewritten_query)


    #1.Dense retrieval

    dense_results=collection.query(query_embeddings=embedding,
        n_results=n_dense,
        where=where,
        include=["documents","metadatas","distances"]
    )

    dense_docs=dense_results["documents"][0]
    dense_metas=dense_results["metadatas"][0]
    dense_ids=dense_results["ids"][0]
    dense_dist=dense_results["distances"][0]

    dense_rr={
        doc_id: 1/(rank+1) for rank,doc_id in enumerate(dense_ids)
    }

    #2.BM25 sparse retrieval

    bm25_payload=get_bm25(collection)
    bm25_index=bm25_payload["index"]
    bm25_ids=bm25_payload["ids"]
    bm25_docs=bm25_payload["documents"]
    bm25_metas=bm25_payload["metadatas"]

    #tokenize the query — drop stopwords so rare/discriminative terms dominate
    query_tokens=tokenize_query(query)
    bm25_scores=bm25_index.get_scores(query_tokens)

    #rank ALL docs by BM25, then keep the top-N that also pass the `where` filter
    #(BM25 scores the whole collection, so honour intent filtering here too)
    ranked_idx=np.argsort(bm25_scores)[::-1]
    bm25_rr={}
    rank=0
    for i in ranked_idx:
        if bm25_scores[i]<=0:
            break
        if where is not None and not metadata_matches(bm25_metas[i], where):
            continue
        bm25_rr[bm25_ids[i]]=1/(rank+1)
        rank+=1
        if rank>=n_bm25:
            break

    #3.RRF fusion

    k=60 #RRF constant
    all_ids=set(dense_rr.keys()) | set(bm25_rr.keys())

    rrf_scores={
        doc_id:(
            0.75 * dense_rr.get(doc_id,0)+
            0.25* bm25_rr.get(doc_id,0)
        )
        for doc_id in all_ids
    }

    #build candidates

    id_to_doc={doc_id:doc for doc_id,doc in zip(dense_ids,dense_docs)}
    id_to_meta = {doc_id: meta for doc_id, meta in zip(dense_ids, dense_metas)}
    id_to_dist = {doc_id: dist for doc_id, dist in zip(dense_ids, dense_dist)}

    for i,doc_id in enumerate(bm25_ids):
        if doc_id not in id_to_doc:
            id_to_doc[doc_id]=bm25_docs[i]
            id_to_meta[doc_id]=bm25_metas[i]
            id_to_dist[doc_id]=1.0

    # Inject entity definition chunks (class/reference) for signature queries
    for hit in fetch_entity_definitions(collection, query, where):
        doc_id = hit["id"]
        id_to_doc[doc_id] = hit["content"]
        id_to_meta[doc_id] = hit["metadata"]
        id_to_dist.setdefault(doc_id, 0.0)
        rrf_scores[doc_id] = max(rrf_scores.get(doc_id, 0.0), 0.5)

    sorted_ids=sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    candidates=[{
        "id": doc_id,
        "content": id_to_doc.get(doc_id, ""),
        "metadata": id_to_meta.get(doc_id, {}),
        "rrf_score": rrf_scores[doc_id],
        "dense_score": 1 - id_to_dist.get(doc_id, 1.0),
    } for doc_id in sorted_ids if id_to_doc.get(doc_id)]

    #4. cross-encoder reranking

    if rerank and candidates:
        reranker=get_reranker()
        pairs=[(query,c["content"]) for c in candidates]

        scores=reranker.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        for candidate in candidates:
            candidate["entity_boost"] = entity_boost(candidate, query)
            candidate["final_score"] = candidate["rerank_score"] + candidate["entity_boost"]
        candidates = sorted(
            candidates,
            key=lambda x: x["final_score"],
            reverse=True,
        )
    else:
        for candidate in candidates:
            candidate["entity_boost"] = entity_boost(candidate, query)
            candidate["final_score"] = candidate["rrf_score"] + candidate["entity_boost"]
        candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    return candidates[:n_final]

def retrieve(query:str,intent:str="wikis_arch_docs",n:str=5,where=None):
    results=hybrid_retrieve(query,intent=intent,n_final=n,where=where)

    for i, r in enumerate(results, 1):
        rerank = r.get("rerank_score", 0)
        dense  = r.get("dense_score", 0)
        rrf    = r.get("rrf_score", 0)

        print(f"\n[{i}] Rerank: {rerank:.3f} | Dense: {dense:.3f} | RRF: {rrf:.3f}")
        print(f"     Type: {r['metadata'].get('type')} | "
              f"Subtype: {r['metadata'].get('subtype')}")
        print(f"     Source: {str(r['metadata'].get('source',''))[:70]}")
        print(f"     {r['content'][:250]}")
        print(f"     {'─'*50}")

if __name__=="__main__":
    
    retrieve("what arguments does HTTPException take",intent="reference",where={"subtype":"reference"})



    