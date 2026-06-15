import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=False)


@dataclass
class Config:
    # PostgreSQL
    pg_host: str = os.getenv("PG_HOST", "localhost")
    pg_port: int = int(os.getenv("PG_PORT", "5432"))
    pg_db: str = os.getenv("PG_DB", "ragdb_sales")
    pg_user: str = os.getenv("PG_USER", "postgres")
    pg_password: str = os.getenv("PG_PASSWORD", "")
    pg_table: str = os.getenv("PG_TABLE", "doc_chunks")

    # Embeddings
    embed_provider: str = os.getenv("EMBED_PROVIDER", "ollama") 
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_dims: int = int(os.getenv("EMBED_DIMS", "1536"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "100"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Scraping
    max_pages: int = int(os.getenv("MAX_PAGES", "2000"))
    rate_limit_delay: float = float(os.getenv("RATE_LIMIT_DELAY", "0.3"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))


config = Config()
