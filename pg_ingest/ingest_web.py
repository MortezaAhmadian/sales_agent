"""CLI entry point for documentation ingestion into PGVector.

Run from the src/pg_ingest directory (or any directory with .env loaded):

    python ingest_web.py                       # all sources
    python ingest_web.py --sources kubernetes  # single source
    python ingest_web.py --sources helm python --max-pages 200
    python ingest_web.py --build-index         # (re)build HNSW index only
"""

import argparse
import logging
import sys

from tqdm import tqdm

from .config import config
from .scraper import DocScraper
from .chunker import TextChunker
from .embedder import Embedder
from .pg_store import PGVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES: dict[str, dict] = {
    "kubernetes": {
        "seed_url": "https://kubernetes.io/docs/home/",
        "path_filter": "/docs/",
        "component": "kubernetes",
        "max_pages": 2000,
    },
    "helm": {
        "seed_url": "https://helm.sh/docs/",
        "path_filter": "/docs/",
        "component": "helm",
        "max_pages": 500,
    },
    "python": {
        "seed_url": "https://docs.python.org/3/",
        "path_filter": "/3/",
        "component": "python",
        "max_pages": 2000,
    },
}

# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------


def ingest_source(
    name: str,
    source: dict,
    scraper: DocScraper,
    chunker: TextChunker,
    embedder: Embedder,
    store: PGVectorStore,
    max_pages_override: int | None,
) -> None:
    max_pages = max_pages_override or source["max_pages"]
    logger.info(f"=== [{name}] Starting ingestion (max_pages={max_pages}) ===")

    pages = scraper.scrape_site(
        seed_url=source["seed_url"],
        path_filter=source["path_filter"],
        component=source["component"],
        max_pages=max_pages,
    )
    logger.info(f"[{name}] Scraped {len(pages)} pages")

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunker.chunk_page(page))
    logger.info(f"[{name}] Generated {len(all_chunks)} chunks")

    if not all_chunks:
        logger.warning(f"[{name}] No chunks produced — skipping")
        return

    bs = config.embed_batch_size
    with tqdm(total=len(all_chunks), desc=f"embed+store [{name}]", unit="chunk") as pbar:
        for i in range(0, len(all_chunks), bs):
            batch = all_chunks[i : i + bs]
            embeddings = embedder.embed_batch([c.content for c in batch])
            store.upsert_chunks(batch, embeddings)
            pbar.update(len(batch))

    logger.info(f"[{name}] Done — total rows in DB: {store.count()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documentation websites into PostgreSQL/pgvector"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCES) + ["all"],
        default=["all"],
        help="Which sources to ingest (default: all)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override max pages per source",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="(Re)build the HNSW index after ingestion",
    )
    args = parser.parse_args()

    sources_to_run = (
        list(SOURCES) if "all" in args.sources else args.sources
    )

    scraper = DocScraper(config)
    chunker = TextChunker(config)
    embedder = Embedder(config)
    store = PGVectorStore(config)

    try:
        for name in sources_to_run:
            ingest_source(
                name,
                SOURCES[name],
                scraper,
                chunker,
                embedder,
                store,
                args.max_pages,
            )
        if args.build_index:
            logger.info("Building HNSW index…")
            store.create_index()
        logger.info(f"Ingestion complete — total chunks in DB: {store.count()}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
