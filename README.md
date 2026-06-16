# Sales Research Agent

A multi-node LangGraph pipeline that produces a structured sales briefing for any prospect. Given a name and company, it searches internal knowledge docs and the web, analyses fit, writes a briefing, and puts it through a critic loop before delivering the final output.

## How it works

```
researcher → analyst → writer → critic → (score < 7 → writer) → save → END
```

**researcher** queries the RAG knowledge base first, then searches the web and scrapes relevant pages.  
**analyst** extracts pain points, fit score, and likely objections from the raw research.  
**writer** formats everything into a 60-second sales briefing.  
**critic** scores the briefing 1–10 and sends it back to the writer if the score is below 7.  
**save** logs completion.

## Project structure

```
sales_agent/
├── graph.py              # Entry point — runs the pipeline
├── tools.py              # All agent tools + dispatcher
├── rag_store.py          # pgvector RAG (read-only, query only)
├── state.py              # LangGraph AgentState
├── config.py             # All configuration, read from .env
├── .env.example          # Environment variable template
├── requirements.txt
├── docker-compose.yml    # PostgreSQL/pgvector + Ollama
└── pg_ingest/            # One-time knowledge ingestion pipeline
    ├── ingest_web.py     # CLI entry point
    ├── scraper.py
    ├── chunker.py
    ├── embedder.py
    └── pg_store.py
```

## Setup

### 1. Start infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL with the pgvector extension and Ollama for local embeddings.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values. Required variables:

| Variable | Description |
|---|---|
| `VLLM_URL` | Base URL of your vLLM server (e.g. `http://host/v1`) |
| `VLLM_MODEL` | Model name (default: `Qwen/Qwen2.5-32B-Instruct`) |
| `VLLM_API_KEY` | vLLM API key |
| `NGINX_USER` / `NGINX_PASSWORD` | Basic auth credentials for the vLLM proxy |
| `PG_PASSWORD` | PostgreSQL password (default: `postgres`) |
| `EMBED_PROVIDER` | `ollama` (default) or `openai` |
| `EMBED_MODEL` | Embedding model name (default: `nomic-embed-text`) |
| `EMBED_DIMS` | Embedding dimensions — must match the model (default: `768`) |

For OpenAI embeddings instead of Ollama, set:

```env
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
EMBED_DIMS=1536
OPENAI_API_KEY=sk-...
```

### 4. Ingest knowledge documents (optional)

The RAG store is read-only at runtime. Populate it once with the ingestion pipeline:

```bash
python -m pg_ingest.ingest_web                     # all sources
python -m pg_ingest.ingest_web --sources nar       # single source
python -m pg_ingest.ingest_web --build-index       # rebuild HNSW index after ingestion
```

To add your own sources, edit the `SOURCES` dict in `pg_ingest/ingest_web.py`.

### 5. Run

```bash
python graph.py
```

You will be prompted for a prospect name and company. The pipeline runs and prints the final briefing to stdout.

## Tools available to the agent

| Tool | Description |
|---|---|
| `query_knowledge_base` | Semantic search over ingested docs in pgvector. Called first. |
| `web_search` | DuckDuckGo search, returns top 5 results. |
| `scrape_page` | Fetches a URL and returns up to 3 000 chars of visible text. |

## Configuration reference

All values have defaults and can be overridden in `.env`.

| Variable | Default | Description |
|---|---|---|
| `PG_HOST` | `localhost` | PostgreSQL host |
| `PG_PORT` | `5432` | PostgreSQL port |
| `PG_DB` | `ragdb_sales` | Database name |
| `PG_USER` | `postgres` | Database user |
| `PG_TABLE` | `doc_chunks` | Knowledge table (populated by pg_ingest) |
| `RAG_K` | `5` | Number of RAG results returned per query |
| `RAG_FETCH_FACTOR` | `3` | Fetch `k × factor` rows before ranking, then trim to `k` |
| `CHUNK_SIZE` | `1000` | Characters per chunk during ingestion |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `MAX_PAGES` | `2000` | Max pages to scrape per source during ingestion |