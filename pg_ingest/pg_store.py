"""PGVector storage layer — no numpy dependency; vectors passed as SQL literals."""

import logging
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)


def _vec_literal(v: list[float]) -> str:
    """Encode a Python float list as a pgvector literal string, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class PGVectorStore:
    def __init__(self, config):
        self.config = config
        self.conn = psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_db,
            user=config.pg_user,
            password=config.pg_password,
        )
        self._setup()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.pg_table} (
                    id          SERIAL PRIMARY KEY,
                    source_url  TEXT NOT NULL,
                    title       TEXT,
                    component   TEXT,
                    doc_type    TEXT,
                    chunk_index INTEGER,
                    content     TEXT NOT NULL,
                    embedding   vector({self.config.embed_dims}),
                    content_hash TEXT UNIQUE,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        self.conn.commit()
        logger.info(f"Table '{self.config.pg_table}' ready (dims={self.config.embed_dims})")

    def create_index(self) -> None:
        """Build an HNSW index after bulk ingestion.

        HNSW does not require pre-existing rows (unlike IVFFlat) so it can be
        called at any time.  Run it once after your first full ingest.
        """
        idx = f"{self.config.pg_table}_hnsw_idx"
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {self.config.pg_table}
                USING hnsw (embedding vector_cosine_ops);
            """)
        self.conn.commit()
        logger.info(f"HNSW index '{idx}' ready")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks, embeddings: list[list[float]]) -> None:
        rows = [
            (
                c.source_url,
                c.title,
                c.component,
                c.doc_type,
                c.chunk_index,
                c.content,
                _vec_literal(emb),
                c.content_hash,
            )
            for c, emb in zip(chunks, embeddings)
        ]
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {self.config.pg_table}
                    (source_url, title, component, doc_type, chunk_index,
                     content, embedding, content_hash)
                VALUES %s
                ON CONFLICT (content_hash) DO NOTHING
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s::vector, %s)",
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.config.pg_table};")
            return cur.fetchone()[0]

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        component: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        vec = _vec_literal(embedding)
        filters: list[str] = []
        params: list = []

        if component:
            filters.append("component = %s")
            params.append(component)
        if doc_type:
            filters.append("doc_type = %s")
            params.append(doc_type)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT source_url, title, component, doc_type, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.config.pg_table}
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [vec] + params + [vec, k],
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()
