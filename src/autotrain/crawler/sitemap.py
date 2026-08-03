"""
sitemap.xml discovery.

A sitemap is the cheapest possible way to learn a site's URL inventory: one
request can replace thousands of link-graph traversals, and it carries
lastmod timestamps that drive incremental re-crawls.

Handles:
  - sitemap index files (nested <sitemapindex> -> <sitemap> -> <loc>)
  - gzipped sitemaps (.xml.gz, or gzip magic bytes regardless of extension)
  - the conventional locations when robots.txt names none
  - recursion depth limits so a self-referential index cannot loop forever
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Iterable

import httpx
from lxml import etree

from autotrain.utils.logging import get_logger

logger = get_logger("crawler.sitemap")

# Conventional paths tried when robots.txt declares no Sitemap directive.
COMMON_SITEMAP_PATHS: tuple[str, ...] = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/sitemap.xml.gz",
    "/sitemaps/sitemap.xml",
    "/wp-sitemap.xml",
)

_MAX_DEPTH = 4
_MAX_SITEMAPS = 200
_MAX_BYTES = 50_000_000


@dataclass
class SitemapEntry:
    loc: str
    lastmod: str | None = None
    changefreq: str | None = None
    priority: float | None = None
    source: str = ""


@dataclass
class SitemapResult:
    entries: list[SitemapEntry] = field(default_factory=list)
    sitemaps_fetched: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [e.loc for e in self.entries]


def _decompress(content: bytes, url: str) -> bytes:
    """Gunzip when the payload is gzipped, by magic bytes not by extension."""
    if content[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(content)
        except OSError as exc:
            logger.debug(f"gzip decompress failed for {url}: {exc}")
    return content


def _localname(tag: object) -> str:
    """Strip the XML namespace: '{http://...}url' -> 'url'."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def parse_sitemap_bytes(content: bytes, source_url: str) -> tuple[list[SitemapEntry], list[str]]:
    """
    Parse one sitemap document.

    Returns (page_entries, nested_sitemap_urls). A document is either a
    urlset (pages) or a sitemapindex (more sitemaps); some real-world files
    mix both, so we collect whatever is present.
    """
    content = _decompress(content, source_url)
    entries: list[SitemapEntry] = []
    nested: list[str] = []

    try:
        # recover=True: real sitemaps are frequently malformed at the tail.
        parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=False)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError as exc:
        logger.debug(f"sitemap parse error at {source_url}: {exc}")
        return entries, nested

    if root is None:
        return entries, nested

    for element in root.iter():
        name = _localname(element.tag)

        if name == "sitemap":
            loc = _child_text(element, "loc")
            if loc:
                nested.append(loc)

        elif name == "url":
            loc = _child_text(element, "loc")
            if not loc:
                continue
            priority_raw = _child_text(element, "priority")
            try:
                priority = float(priority_raw) if priority_raw else None
            except ValueError:
                priority = None
            entries.append(
                SitemapEntry(
                    loc=loc,
                    lastmod=_child_text(element, "lastmod"),
                    changefreq=_child_text(element, "changefreq"),
                    priority=priority,
                    source=source_url,
                )
            )

    return entries, nested


def _child_text(element: etree._Element, name: str) -> str | None:
    for child in element:
        if _localname(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


async def discover_sitemaps(
    client: httpx.AsyncClient,
    base_url: str,
    declared: Iterable[str] = (),
    *,
    timeout: float = 30.0,
) -> SitemapResult:
    """
    Walk every sitemap reachable from `declared` plus the conventional paths.

    Breadth-first with a visited set, so a sitemap index that references
    itself terminates instead of recursing forever.
    """
    result = SitemapResult()

    from urllib.parse import urljoin

    queue: list[tuple[str, int]] = [(url, 0) for url in declared]
    if not queue:
        queue = [(urljoin(base_url, path), 0) for path in COMMON_SITEMAP_PATHS]

    seen: set[str] = set()

    while queue and len(result.sitemaps_fetched) < _MAX_SITEMAPS:
        url, depth = queue.pop(0)
        if url in seen or depth > _MAX_DEPTH:
            continue
        seen.add(url)

        try:
            response = await client.get(url, timeout=timeout)
        except httpx.HTTPError as exc:
            result.errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            continue

        if response.status_code != 200:
            # Missing conventional paths are expected, not worth reporting.
            if declared:
                result.errors.append({"url": url, "error": f"HTTP {response.status_code}"})
            continue

        content = response.content
        if len(content) > _MAX_BYTES:
            result.errors.append({"url": url, "error": "sitemap too large"})
            continue

        entries, nested = parse_sitemap_bytes(content, url)
        if entries or nested:
            result.sitemaps_fetched.append(url)
            result.entries.extend(entries)
            for child in nested:
                if child not in seen:
                    queue.append((child, depth + 1))
            logger.debug(
                f"sitemap {url}: {len(entries)} urls, {len(nested)} nested"
            )

    if result.entries:
        logger.info(
            f"sitemap discovery: {len(result.entries):,} URLs from "
            f"{len(result.sitemaps_fetched)} sitemap file(s)"
        )
    else:
        logger.info("no sitemap found — relying on link discovery")

    return result
