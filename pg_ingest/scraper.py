"""Web scraper with sitemap-first discovery and BFS crawl fallback."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urldefrag, urlparse
from xml.etree import ElementTree

import requests
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml"]
_SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Ingestor/1.0; +research)"}


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    component: str
    metadata: dict = field(default_factory=dict)


class DocScraper:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self._visited: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_site(
        self,
        seed_url: str,
        path_filter: str,
        component: str,
        max_pages: Optional[int] = None,
    ) -> list[ScrapedPage]:
        limit = max_pages or self.config.max_pages
        parsed = urlparse(seed_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        logger.info(f"[{component}] Discovering URLs via sitemap…")
        urls = self._discover_sitemap(base_url, path_filter)

        if urls:
            logger.info(f"[{component}] {len(urls)} URLs found in sitemap")
            return self._fetch_url_list(urls[:limit], component, base_url)

        logger.info(f"[{component}] No sitemap — falling back to BFS crawl from {seed_url}")
        return self._crawl(seed_url, path_filter, component, limit)

    # ------------------------------------------------------------------
    # Sitemap discovery
    # ------------------------------------------------------------------

    def _discover_sitemap(self, base_url: str, path_filter: str) -> list[str]:
        for path in _SITEMAP_PATHS:
            html = self._fetch(urljoin(base_url, path))
            if not html:
                continue
            urls = self._parse_sitemap_xml(html, path_filter)
            if urls:
                return urls
        return []

    def _parse_sitemap_xml(self, xml_text: str, path_filter: str) -> list[str]:
        urls: list[str] = []
        try:
            root = ElementTree.fromstring(xml_text)
            tag = root.tag.lower()

            if "sitemapindex" in tag:
                # Recurse into each child sitemap
                for loc_el in root.iter(f"{{{_SITEMAP_NS}}}loc"):
                    child_xml = self._fetch(loc_el.text.strip())
                    if child_xml:
                        urls.extend(self._parse_sitemap_xml(child_xml, path_filter))
            else:
                # Regular sitemap
                for loc_el in root.iter(f"{{{_SITEMAP_NS}}}loc"):
                    url = self._normalise(loc_el.text.strip())
                    if path_filter in url:
                        urls.append(url)
        except ElementTree.ParseError as exc:
            logger.debug(f"Sitemap parse error: {exc}")
        return urls

    # ------------------------------------------------------------------
    # Fetching URL list (from sitemap)
    # ------------------------------------------------------------------

    def _fetch_url_list(
        self, urls: list[str], component: str, base_url: str
    ) -> list[ScrapedPage]:
        pages = []
        total = len(urls)
        for i, url in enumerate(urls):
            if url in self._visited:
                continue
            self._visited.add(url)
            logger.info(f"[{component}] [{i + 1}/{total}] {url}")

            html = self._fetch(url)
            if html:
                page = self._make_page(html, url, component, base_url)
                if page:
                    pages.append(page)

            time.sleep(self.config.rate_limit_delay)
        return pages

    # ------------------------------------------------------------------
    # BFS crawl fallback
    # ------------------------------------------------------------------

    def _crawl(
        self, seed_url: str, path_filter: str, component: str, limit: int
    ) -> list[ScrapedPage]:
        parsed = urlparse(seed_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        queue: deque[str] = deque([self._normalise(seed_url)])
        pages = []

        while queue and len(self._visited) < limit:
            url = queue.popleft()
            if url in self._visited or path_filter not in url:
                continue
            self._visited.add(url)
            logger.info(f"[{component}] [{len(self._visited)}] {url}")

            html = self._fetch(url)
            if not html:
                continue

            page = self._make_page(html, url, component, base_url)
            if page:
                pages.append(page)

            for link in self._extract_links(html, url):
                if self._same_domain(link, base_url) and link not in self._visited:
                    queue.append(link)

            time.sleep(self.config.rate_limit_delay)

        return pages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> Optional[str]:
        for attempt in range(self.config.max_retries):
            try:
                r = self.session.get(url, timeout=self.config.request_timeout)
                r.raise_for_status()
                return r.text
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code < 500:
                    return None  # 4xx — not retrying
            except Exception:
                pass
            time.sleep(2 ** attempt)
        logger.debug(f"Failed to fetch {url} after {self.config.max_retries} attempts")
        return None

    def _make_page(
        self, html: str, url: str, component: str, base_url: str
    ) -> Optional[ScrapedPage]:
        title, text = self._extract_text(html, url)
        if not text.strip():
            return None
        return ScrapedPage(
            url=url,
            title=title,
            text=text,
            component=component,
            metadata={"source_domain": base_url},
        )

    def _extract_text(self, html: str, url: str) -> tuple[str, str]:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        if not text:
            for sel in ["main", "article", '[role="main"]', ".content", "#content"]:
                el = soup.select_one(sel)
                if el:
                    text = el.get_text(separator="\n", strip=True)
                    break

        return title, text or ""

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = self._normalise(urljoin(base_url, a["href"]))
            if href.startswith("http"):
                links.append(href)
        return links

    @staticmethod
    def _normalise(url: str) -> str:
        url, _ = urldefrag(url)
        return url.rstrip("/")

    @staticmethod
    def _same_domain(url: str, base: str) -> bool:
        return urlparse(url).netloc == urlparse(base).netloc
