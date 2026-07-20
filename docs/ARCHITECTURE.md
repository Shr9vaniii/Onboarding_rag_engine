# Architecture diagrams

## System context

```mermaid
flowchart LR
    Browser["Small chat UI"] --> API["FastAPI /ask"]
    API --> Session["SQLite session memory"]
    API --> Router["Rewrite then decompose"]
    Router --> Retrieval["Dense + BM25 + RRF + reranker"]
    Retrieval --> Cache["Redis Cloud answer cache"]
    Cache --> GPU["RunPod LoRA endpoint"]
    GPU --> API
```

## Ask sequence

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant S as SQLite
    participant G as Groq
    participant R as Hybrid retriever
    participant C as Redis
    participant L as LoRA server

    U->>A: POST /ask
    A->>S: load session / create
    A->>G: rewrite (if session)
    A->>G: decompose rewritten query
    alt multi-query
        loop each sub-query
            A->>R: retrieve
            A->>C: get/set answer
            A->>L: generate (on miss)
        end
        A-->>U: stitched partial-safe answer + citations
    else single-query
        A->>R: retrieve (optional dual)
        A->>C: get/set answer
        A->>L: generate (on miss)
        A->>S: add_turn
        A-->>U: answer + citations + timings
    end
```

## Cache keying

```
sha256(CACHE_VERSION | model_version | prompt_version | normalized_query | chunk_ids)
```

Stale answers invalidate when corpus chunk IDs change or versions bump.
