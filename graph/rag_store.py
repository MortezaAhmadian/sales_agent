"""pgvector-backed RAG store — knowledge (doc_chunks) + agent memory (agent_memory).

Knowledge table (doc_chunks) is populated by the pg_ingest pipeline and is read-only here.
Memory table (agent_memory) is written by the agent via save_deployment_memory.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field

import psycopg2

from config import (
    EMBED_API_KEY,
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
    EMBED_PROVIDER,
    MEMORY_TABLE,
    PG_DB,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_TABLE,
    PG_USER,
    RAG_FETCH_FACTOR,
    RAG_K,
)

logger = logging.getLogger(__name__)


@dataclass
class DocChunk:
    content:  str
    metadata: dict = field(default_factory=dict)


def _vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


class RAGStore:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASSWORD,
        )
        self._setup_memory_table()

        import openai
        self._embed_client = openai.OpenAI(
            api_key=EMBED_API_KEY,
            base_url=EMBED_BASE_URL if EMBED_PROVIDER != "openai" else None,
        )

    def _setup_memory_table(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
                    id           SERIAL PRIMARY KEY,
                    content      TEXT NOT NULL,
                    component    TEXT,
                    phase        TEXT,
                    tags         TEXT,
                    embedding    vector({EMBED_DIMS}),
                    content_hash TEXT UNIQUE,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        self.conn.commit()
        logger.info("Memory table '%s' ready", MEMORY_TABLE)

    # ── Counts ────────────────────────────────────────────────────────────────

    @property
    def knowledge_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {PG_TABLE};")
            return cur.fetchone()[0]

    @property
    def memory_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {MEMORY_TABLE};")
            return cur.fetchone()[0]

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        for attempt in range(3):
            try:
                resp = self._embed_client.embeddings.create(
                    model=EMBED_MODEL, input=[text]
                )
                return resp.data[0].embedding
            except Exception as exc:
                if attempt == 2:
                    raise
                logger.warning("Embedding error (%s); retrying in %ds", exc, 2 ** attempt)
                time.sleep(2 ** attempt)

    # ── Write (memory only) ───────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[DocChunk], collection: str = "knowledge") -> int:
        """Write chunks to agent_memory. Knowledge table is managed by pg_ingest."""
        if not chunks or collection != "memory":
            return 0
        inserted = 0
        with self.conn.cursor() as cur:
            for c in chunks:
                emb = self._embed(c.content)
                h   = hashlib.md5(c.content[:200].encode()).hexdigest()
                cur.execute(f"""
                    INSERT INTO {MEMORY_TABLE}
                        (content, component, phase, tags, embedding, content_hash)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                """, (
                    c.content,
                    c.metadata.get("component", ""),
                    c.metadata.get("phase", ""),
                    c.metadata.get("tags", ""),
                    _vec_literal(emb),
                    h,
                ))
                inserted += 1
        self.conn.commit()
        return inserted

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        k: int = RAG_K,
        component: str | None = None,
        doc_type: str | None = None,
        collection: str = "knowledge",
        include_memory: bool = True,
    ) -> list[dict]:
        """Return top-k most similar chunks as dicts: {content, metadata, score}."""
        vec = _vec_literal(self._embed(query_text))

        if collection != "knowledge":
            return self._query_memory(vec, k=k)

        # ── Knowledge search ──────────────────────────────────────────────────
        filters: list[str] = []
        params:  list      = [vec]
        if component:
            filters.append("component = %s")
            params.append(component)
        if doc_type:
            filters.append("doc_type = %s")
            params.append(doc_type)
        where    = f"WHERE {' AND '.join(filters)}" if filters else ""
        fetch_k  = k * RAG_FETCH_FACTOR
        params  += [vec, fetch_k]

        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT source_url, title, component, doc_type, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {PG_TABLE}
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params)
            cols = [d[0] for d in cur.description]
            knowledge_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        results = [
            {
                "content": r["content"],
                "metadata": {
                    "source_file": r.get("source_url", ""),
                    "component":   r.get("component", ""),
                    "doc_type":    r.get("doc_type", ""),
                    "title":       r.get("title", ""),
                },
                "score": round(float(r["similarity"]), 4),
            }
            for r in knowledge_rows
        ]

        # ── Blend in up to 2 memory chunks ───────────────────────────────────
        if include_memory and self.memory_count > 0:
            seen = {r["content"][:80] for r in results}
            for r in self._query_memory(vec, k=2):
                if r["content"][:80] not in seen:
                    results.append(r)

        # Deduplicate and return top-k by score
        seen_sigs: set[str] = set()
        output: list[dict]  = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            sig = r["content"][:80]
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                output.append(r)
            if len(output) >= k + 2:
                break
        return output

    def _query_memory(self, vec_literal: str, k: int) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT content, component, phase, tags,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {MEMORY_TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [vec_literal, vec_literal, k])
            rows = cur.fetchall()
        return [
            {
                "content": row[0],
                "metadata": {
                    "component":   row[1] or "",
                    "phase":       row[2] or "",
                    "tags":        row[3] or "",
                    "from_memory": True,
                },
                "score": round(float(row[4]), 4),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.conn.close()
