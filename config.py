"""Central configuration for the sales agent project.

All modules import from here. Values are read from environment variables
(loaded from .env by the entry points).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass
class Config:
    # ── vLLM / LLM ────────────────────────────────────────────────────
    vllm_url: str      = os.getenv("VLLM_URL", "")
    vllm_model: str    = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
    vllm_api_key: str  = os.getenv("VLLM_API_KEY", "")
    nginx_user: str    = os.getenv("NGINX_USER", "")
    nginx_password: str = os.getenv("NGINX_PASSWORD", "")

    # ── PostgreSQL / pgvector ─────────────────────────────────────────
    pg_host: str     = os.getenv("PG_HOST", "localhost")
    pg_port: int     = int(os.getenv("PG_PORT", "5432"))
    pg_db: str       = os.getenv("PG_DB", "ragdb_sales")
    pg_user: str     = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "postgres")
    pg_table: str    = os.getenv("PG_TABLE", "doc_chunks")

    # ── Embeddings ────────────────────────────────────────────────────
    embed_provider: str   = os.getenv("EMBED_PROVIDER", "ollama")
    embed_model: str      = os.getenv("EMBED_MODEL", "nomic-embed-text")
    embed_dims: int       = int(os.getenv("EMBED_DIMS", "768"))
    embed_api_key: str    = os.getenv("EMBED_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    embed_base_url: str   = os.getenv("EMBED_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "100"))

    # ── RAG retrieval ─────────────────────────────────────────────────
    rag_k: int            = int(os.getenv("RAG_K", "5"))
    rag_fetch_factor: int = int(os.getenv("RAG_FETCH_FACTOR", "3"))

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int    = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # ── Scraping ──────────────────────────────────────────────────────
    max_pages: int          = int(os.getenv("MAX_PAGES", "2000"))
    rate_limit_delay: float = float(os.getenv("RATE_LIMIT_DELAY", "0.3"))
    request_timeout: int    = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries: int        = int(os.getenv("MAX_RETRIES", "3"))


config = Config()
