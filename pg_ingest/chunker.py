"""Split scraped pages into overlapping text chunks with metadata."""

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .scraper import ScrapedPage


@dataclass
class DocChunk:
    content: str
    source_url: str
    title: str
    component: str
    doc_type: str
    chunk_index: int
    content_hash: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    def __init__(self, config):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_page(self, page: ScrapedPage) -> list[DocChunk]:
        if not page.text.strip():
            return []

        splits = self._splitter.split_text(page.text)
        doc_type = _infer_doc_type(page.url)
        chunks = []

        for i, split in enumerate(splits):
            if not split.strip():
                continue
            # Hash on URL + first 200 chars so we can deduplicate on re-ingest
            content_hash = hashlib.md5(
                f"{page.url}::{split[:200]}".encode()
            ).hexdigest()
            chunks.append(
                DocChunk(
                    content=split,
                    source_url=page.url,
                    title=page.title,
                    component=page.component,
                    doc_type=doc_type,
                    chunk_index=i,
                    content_hash=content_hash,
                )
            )

        return chunks


def _infer_doc_type(url: str) -> str:
    p = url.lower()
    if any(x in p for x in ["/api/", "/reference/", "/api-reference/"]):
        return "api-reference"
    if any(x in p for x in ["/tutorial", "/getting-started", "/quickstart"]):
        return "tutorial"
    if "/concepts/" in p:
        return "concepts"
    if "/tasks/" in p:
        return "tasks"
    if "/setup/" in p or "/install" in p:
        return "setup"
    return "docs"
