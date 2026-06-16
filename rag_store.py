"""pgvector-backed RAG store — query only.

Reads from doc_chunks (populated by pg_ingest). No memory writes.

Usage:
    store = RAGStore()
    results = store.query("company pain points")
    store.close()
"""

import logging
import time

import psycopg2

from config import config

logger = logging.getLogger(__name__)


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class RAGStore:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_db,
            user=config.pg_user,
            password=config.pg_password,
        )
        self._ensure_vector_extension()
        self._embed_client = self._make_embed_client()

    # ── Setup ─────────────────────────────────────────────────────────

    def _ensure_vector_extension(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.conn.commit()

    def _make_embed_client(self):
        import openai
        if config.embed_provider == "openai":
            return openai.OpenAI(api_key=config.embed_api_key)
        return openai.OpenAI(
            api_key=config.embed_api_key or "ollama",
            base_url=f"{config.embed_base_url}/v1",
        )

    # ── Embedding ─────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        for attempt in range(3):
            try:
                resp = self._embed_client.embeddings.create(
                    model=config.embed_model,
                    input=[text],
                )
                return resp.data[0].embedding
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning("Embedding error (%s); retrying in %ds", exc, wait)
                time.sleep(wait)

    # ── Read ──────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        k: int = None,
        component: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        """Return top-k most relevant chunks from doc_chunks as dicts: {content, metadata, score}."""
        k = k or config.rag_k
        vec = _vec_literal(self._embed(query_text))

        filters: list[str] = []
        params: list = [vec]
        if component:
            filters.append("component = %s")
            params.append(component)
        if doc_type:
            filters.append("doc_type = %s")
            params.append(doc_type)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        fetch_k = k * config.rag_fetch_factor
        params += [vec, fetch_k]

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT source_url, title, component, doc_type, content,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {config.pg_table}
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, params)
                cols = [d[0] for d in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        except psycopg2.errors.UndefinedTable:
            logger.warning("Knowledge table '%s' not found — run pg_ingest first", config.pg_table)
            self.conn.rollback()
            return []

        return [
            {
                "content": r["content"],
                "metadata": {
                    "source":    r.get("source_url", ""),
                    "component": r.get("component", ""),
                    "doc_type":  r.get("doc_type", ""),
                    "title":     r.get("title", ""),
                },
                "score": round(float(r["similarity"]), 4),
            }
            for r in rows
        ]

    # ── Count ─────────────────────────────────────────────────────────

    @property
    def knowledge_count(self) -> int:
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {config.pg_table};")
                return cur.fetchone()[0]
        except Exception:
            self.conn.rollback()
            return 0

    def close(self) -> None:
        self.conn.close()
