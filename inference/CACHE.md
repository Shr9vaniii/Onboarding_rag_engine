# Redis Cloud answer cache

Caches generated answers in Redis Cloud so repeat questions skip Colab generation.

## Setup

1. Create a free DB at [Redis Cloud](https://redis.io/cloud/)
2. Copy the public endpoint URL (use `rediss://` if TLS is required)
3. Add to project `.env`:

```
REDIS_URL=redis://default:YOUR_PASSWORD@YOUR_HOST:PORT
```

TLS example:

```
REDIS_URL=rediss://default:YOUR_PASSWORD@YOUR_HOST:PORT
```

4. Install client:

```bash
pip install redis
```

## Behavior

- **Key:** hash of `normalized_query + retrieved chunk ids`
- **TTL:** 7 days (`CACHE_TTL_SECONDS` to override)
- **Skipped:** abstention answers (`I don't have enough information...`)
- If `REDIS_URL` is missing or Redis is down, pipeline still runs (cache no-op)

## Usage

```powershell
# normal (cache on)
python -m inference.rag_engine "what arguments does HTTPException take?" --backend remote -v

# run again — should show Cache: HIT
python -m inference.rag_engine "what arguments does HTTPException take?" --backend remote -v

# bypass
python -m inference.rag_engine "what arguments does HTTPException take?" --backend remote -v --no-cache
```
