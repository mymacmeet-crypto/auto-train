"""
Phases 1 and 2 — website discovery and HTML collection.

Discovery and collection are one pass, not two, because you must fetch a
page to read its links. Splitting them would mean fetching the whole site
twice. The phase therefore emits both the discovery artifacts (urls.json,
sitemap.json, crawl_report.json) and the raw HTML store.

Concurrency model: a fixed pool of N workers drains a shared frontier.
Workers push newly discovered in-scope URLs back onto the frontier, so the
crawl expands breadth-first without recursion and without unbounded task
creation on a 10,000-page site.

Resume: raw_html/index.jsonl is replayed at startup. Any URL already
recorded with a saved body is neither re-fetched nor re-numbered, so an
interrupted crawl continues exactly where it stopped, and a re-run against
an unchanged site is nearly free.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from lxml import html as lxml_html

from autotrain.crawler.fetcher import Fetcher, FetchResult, HostRateLimiter
from autotrain.crawler.robots import fetch_robots
from autotrain.crawler.sitemap import discover_sitemaps
from autotrain.crawler.urlnorm import (
    PatternFilter,
    looks_like_trap,
    normalize_url,
    same_site,
)
from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import sha256_text, write_json

logger = get_logger("crawler")


@dataclass
class PageRecord:
    """One crawled page, as persisted to raw_html/index.jsonl."""

    index: int
    requested_url: str
    final_url: str
    status_code: int
    ok: bool
    content_type: str
    encoding: str
    headers: dict[str, str]
    redirect_chain: list[dict[str, Any]]
    content_length: int
    elapsed_ms: int
    fetched_at: float
    depth: int
    discovered_from: str | None
    content_hash: str
    html_path: str
    links_found: int = 0
    error: str | None = None
    duplicate_of: int | None = None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CrawlStats:
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    discovered: int = 0
    fetched: int = 0
    ok: int = 0
    failed: int = 0
    skipped_offsite: int = 0
    skipped_robots: int = 0
    skipped_pattern: int = 0
    skipped_trap: int = 0
    skipped_content_type: int = 0
    skipped_duplicate_content: int = 0
    from_cache: int = 0
    bytes_downloaded: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    trap_reasons: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)


class Crawler:
    """Breadth-first, scope-limited, resumable site crawler."""

    def __init__(self, config: AutotrainConfig, workspace: Workspace) -> None:
        self.cfg = config
        self.ws = workspace
        self.ccfg = config.crawler

        seed = config.domain if "://" in config.domain else f"https://{config.domain}"
        normalized_seed = normalize_url(seed)
        if not normalized_seed:
            raise ValueError(f"cannot parse domain: {config.domain!r}")
        self.seed_url = normalized_seed
        self.root_host = urlsplit(self.seed_url).hostname or ""

        self.filter = PatternFilter(self.ccfg.include_patterns, self.ccfg.exclude_patterns)

        # Frontier holds (url, depth, parent). seen guards against re-queueing.
        self._frontier: deque[tuple[str, int, str | None]] = deque()
        self._seen: set[str] = set()
        self._records: list[PageRecord] = []
        self._by_url: dict[str, PageRecord] = {}
        self._content_hashes: dict[str, int] = {}
        self._next_index = 1
        self._lock = asyncio.Lock()
        self.stats = CrawlStats()
        self._sitemap_payload: dict[str, Any] = {}
        self._robots_info: dict[str, Any] = {}

    # -- resume ------------------------------------------------------------

    def _load_cache(self) -> None:
        """Replay index.jsonl so a resumed run keeps its page numbering."""
        path = self.ws.fetch_index
        if not path.exists():
            return

        ttl = self.cfg.runtime.cache_ttl_hours
        cutoff = time.time() - ttl * 3600 if ttl > 0 else None
        loaded = 0

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if cutoff and data.get("fetched_at", 0) < cutoff:
                    continue  # stale: allow a re-fetch

                try:
                    record = PageRecord(**data)
                except TypeError:
                    continue

                # A record is only usable if its HTML is still on disk.
                if record.ok and not (self.ws.root / record.html_path).exists():
                    continue

                self._records.append(record)
                self._by_url[record.requested_url] = record
                self._seen.add(record.requested_url)
                if record.final_url:
                    self._seen.add(record.final_url)
                if record.content_hash:
                    self._content_hashes.setdefault(record.content_hash, record.index)
                self._next_index = max(self._next_index, record.index + 1)
                loaded += 1

        if loaded:
            self.stats.from_cache = loaded
            logger.info(f"resuming: {loaded:,} pages already in the cache")

    # -- frontier ----------------------------------------------------------

    def _enqueue(self, url: str, depth: int, parent: str | None) -> bool:
        """Normalise, scope-check, and queue a URL. Returns True if queued."""
        normalized = normalize_url(url, base=parent)
        if not normalized:
            return False
        if normalized in self._seen:
            return False

        # The seed is exempt from the pattern and trap filters. The user named
        # it explicitly, and an include_patterns regex that happens not to
        # match the entry point (a trailing slash is enough) would otherwise
        # silently produce an empty crawl. robots.txt still applies.
        is_seed = normalized == self.seed_url

        if not same_site(normalized, self.root_host, include_subdomains=self.ccfg.include_subdomains):
            self.stats.skipped_offsite += 1
            return False

        if is_seed:
            if self._robots and not self._robots.allows(normalized, self.ccfg.user_agent):
                self.stats.skipped_robots += 1
                logger.error(f"robots.txt disallows the seed URL {normalized}")
                return False
            self._seen.add(normalized)
            self._frontier.append((normalized, depth, parent))
            self.stats.discovered += 1
            return True

        allowed, reason = self.filter.allows(normalized)
        if not allowed:
            self.stats.skipped_pattern += 1
            logger.debug(f"pattern-skip {normalized} ({reason})")
            return False

        trap = looks_like_trap(
            normalized,
            max_repeated_segments=self.ccfg.max_repeated_path_segments,
            max_path_depth=self.ccfg.max_path_depth,
            max_query_params=self.ccfg.max_query_params,
            max_length=self.ccfg.max_url_length,
        )
        if trap:
            self.stats.skipped_trap += 1
            key = trap.split(":", 1)[0]
            self.stats.trap_reasons[key] = self.stats.trap_reasons.get(key, 0) + 1
            logger.debug(f"trap-skip {normalized} ({trap})")
            return False

        if self._robots and not self._robots.allows(normalized, self.ccfg.user_agent):
            self.stats.skipped_robots += 1
            logger.debug(f"robots-skip {normalized}")
            return False

        if depth > self.ccfg.max_depth:
            return False

        self._seen.add(normalized)
        self._frontier.append((normalized, depth, parent))
        self.stats.discovered += 1
        return True

    # -- link extraction ---------------------------------------------------

    def _extract_links(self, html_text: str, base_url: str) -> list[str]:
        """
        Pull every navigable href out of a page.

        Uses lxml rather than BeautifulSoup here purely for speed: at 10k
        pages the parser is the hot loop, and lxml is roughly an order of
        magnitude faster on this workload.
        """
        try:
            doc = lxml_html.fromstring(html_text)
        except (lxml_html.etree.ParserError, ValueError):
            return []

        # <base href> changes relative resolution for the whole document.
        effective_base = base_url
        for base_el in doc.iter("base"):
            href = base_el.get("href")
            if href:
                effective_base = normalize_url(href, base=base_url) or base_url
            break

        # Honour rel=nofollow only when robots are being respected at all.
        links: list[str] = []
        for element in doc.iter("a"):
            href = element.get("href")
            if not href:
                continue
            rel = (element.get("rel") or "").lower()
            if self.ccfg.respect_robots and "nofollow" in rel:
                continue
            links.append(href)

        # Canonical link often points at the preferred URL for this content.
        for link_el in doc.iter("link"):
            if (link_el.get("rel") or "").lower() == "canonical":
                href = link_el.get("href")
                if href:
                    links.append(href)

        resolved = []
        for href in links:
            normalized = normalize_url(href, base=effective_base)
            if normalized:
                resolved.append(normalized)
        return resolved

    # -- worker ------------------------------------------------------------

    async def _process(self, fetcher: Fetcher, url: str, depth: int, parent: str | None) -> None:
        cached = self._by_url.get(url)
        if cached is not None:
            # Already fetched in an earlier run; still mine it for links so
            # newly-added pages are discovered on an incremental re-crawl.
            if cached.ok:
                html_file = self.ws.root / cached.html_path
                try:
                    html_text = html_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    html_text = ""
                if html_text:
                    for link in self._extract_links(html_text, cached.final_url or url):
                        async with self._lock:
                            self._enqueue(link, depth + 1, cached.final_url or url)
            return

        result: FetchResult = await fetcher.fetch(url)
        self.stats.fetched += 1

        code = str(result.status_code or "error")
        self.stats.status_counts[code] = self.stats.status_counts.get(code, 0) + 1

        if result.error and not result.ok:
            if result.error.startswith("content_type_skipped"):
                self.stats.skipped_content_type += 1
            else:
                self.stats.failed += 1
                key = result.error.split(":", 1)[0]
                self.stats.errors[key] = self.stats.errors.get(key, 0) + 1

        async with self._lock:
            index = self._next_index
            self._next_index += 1

        content_hash = sha256_text(result.html) if result.html else ""
        duplicate_of: int | None = None
        html_path = ""

        if result.ok and content_hash:
            existing = self._content_hashes.get(content_hash)
            if existing is not None:
                # Byte-identical page under a different URL. Record the
                # relationship but do not store a second copy.
                duplicate_of = existing
                self.stats.skipped_duplicate_content += 1
            else:
                self._content_hashes[content_hash] = index
                target = self.ws.page_html(index)
                try:
                    target.write_text(result.html, encoding="utf-8")
                    html_path = str(target.relative_to(self.ws.root)).replace("\\", "/")
                    self.stats.bytes_downloaded += result.content_length
                    self.stats.ok += 1
                except OSError as exc:
                    logger.warning(f"could not write {target}: {exc}")
                    result.ok = False
                    result.error = f"write_failed: {exc}"

        links: list[str] = []
        if result.ok and result.html:
            links = self._extract_links(result.html, result.final_url or url)
            async with self._lock:
                for link in links:
                    self._enqueue(link, depth + 1, result.final_url or url)

        record = PageRecord(
            index=index,
            requested_url=url,
            final_url=result.final_url or url,
            status_code=result.status_code,
            ok=bool(result.ok and duplicate_of is None),
            content_type=result.content_type,
            encoding=result.encoding,
            headers=result.headers,
            redirect_chain=result.redirect_chain,
            content_length=result.content_length,
            elapsed_ms=result.elapsed_ms,
            fetched_at=result.fetched_at,
            depth=depth,
            discovered_from=parent,
            content_hash=content_hash,
            html_path=html_path,
            links_found=len(links),
            error=result.error,
            duplicate_of=duplicate_of,
        )

        async with self._lock:
            self._records.append(record)
            self._by_url[url] = record
            if result.final_url:
                self._seen.add(result.final_url)
            # Append-as-we-go: a kill -9 loses at most the in-flight page.
            with self.ws.fetch_index.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")

    # -- driver ------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        self.ws.raw_html.mkdir(parents=True, exist_ok=True)
        self._robots = None  # set below; _enqueue reads it

        limits = httpx.Limits(
            max_connections=self.ccfg.concurrency * 2,
            max_keepalive_connections=self.ccfg.concurrency,
        )
        headers = {
            "User-Agent": self.ccfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with httpx.AsyncClient(
            headers=headers,
            limits=limits,
            follow_redirects=False,
            http2=True,
            verify=True,
        ) as client:
            # ---- robots.txt --------------------------------------------
            self._robots = await fetch_robots(
                client,
                self.seed_url,
                self.ccfg.user_agent,
                enabled=self.ccfg.respect_robots,
            )
            self._robots_info = {
                "url": self._robots.origin,
                "enabled": self._robots.enabled,
                "fetched": self._robots.fetched,
                "status_code": self._robots.status_code,
                "crawl_delay": self._robots.crawl_delay,
                "sitemaps_declared": self._robots.sitemaps,
            }

            delay = self.ccfg.delay_seconds
            if self._robots.crawl_delay:
                # A site asking for slower crawling gets it.
                delay = max(delay, self._robots.crawl_delay)
            limiter = HostRateLimiter(delay)
            if delay:
                logger.info(f"per-host delay: {delay:.2f}s")

            fetcher = Fetcher(
                client,
                timeout=self.ccfg.request_timeout,
                retries=self.ccfg.retries,
                backoff=self.ccfg.retry_backoff,
                max_bytes=self.ccfg.max_page_bytes,
                allowed_content_types=tuple(self.ccfg.allowed_content_types),
                rate_limiter=limiter,
            )

            self._load_cache()

            # ---- sitemap ------------------------------------------------
            sitemap_urls: list[str] = []
            if self.ccfg.follow_sitemap:
                sitemap = await discover_sitemaps(
                    client,
                    self.seed_url,
                    self._robots.sitemaps,
                    timeout=self.ccfg.request_timeout,
                )
                sitemap_urls = sitemap.urls
                self._sitemap_payload = {
                    "sitemaps_fetched": sitemap.sitemaps_fetched,
                    "declared_in_robots": self._robots.sitemaps,
                    "url_count": len(sitemap.entries),
                    "errors": sitemap.errors,
                    "entries": [
                        {
                            "loc": e.loc,
                            "lastmod": e.lastmod,
                            "changefreq": e.changefreq,
                            "priority": e.priority,
                        }
                        for e in sitemap.entries
                    ],
                }
            else:
                self._sitemap_payload = {"disabled": True, "url_count": 0, "entries": []}

            # ---- seed the frontier --------------------------------------
            self._enqueue(self.seed_url, 0, None)
            queued_from_sitemap = sum(1 for u in sitemap_urls if self._enqueue(u, 1, self.seed_url))
            if queued_from_sitemap:
                logger.info(f"queued {queued_from_sitemap:,} URLs from sitemap")

            # ---- drain --------------------------------------------------
            await self._drain(fetcher)

        self.stats.finished_at = time.time()
        return self._write_reports()

    async def _drain(self, fetcher: Fetcher) -> None:
        """Run `concurrency` workers until the frontier empties or budget hits."""
        budget = self.ccfg.max_pages
        progress = make_progress()
        completed = 0

        with progress:
            task = progress.add_task("crawling", total=min(budget, max(len(self._frontier), 1)))

            async def worker() -> None:
                nonlocal completed
                while True:
                    async with self._lock:
                        if not self._frontier:
                            return
                        if len(self._records) >= budget:
                            return
                        url, depth, parent = self._frontier.popleft()

                    try:
                        await self._process(fetcher, url, depth, parent)
                    except Exception as exc:  # one page must never kill the crawl
                        logger.warning(f"unhandled error on {url}: {type(exc).__name__}: {exc}")
                        self.stats.failed += 1

                    completed += 1
                    total = min(budget, len(self._records) + len(self._frontier))
                    progress.update(task, completed=completed, total=max(total, completed))

            workers = [asyncio.create_task(worker()) for _ in range(self.ccfg.concurrency)]

            # Workers exit when the frontier is momentarily empty; restart
            # them while new links are still arriving.
            while True:
                await asyncio.gather(*workers, return_exceptions=True)
                async with self._lock:
                    remaining = len(self._frontier)
                    at_budget = len(self._records) >= budget
                if remaining == 0 or at_budget:
                    break
                workers = [asyncio.create_task(worker()) for _ in range(self.ccfg.concurrency)]

        if len(self._records) >= budget:
            logger.warning(
                f"stopped at max_pages={budget:,}; {len(self._frontier):,} URLs left unvisited"
            )

    # -- artifacts ---------------------------------------------------------

    def _write_reports(self) -> dict[str, Any]:
        stats = self.stats
        duration = round((stats.finished_at or time.time()) - stats.started_at, 2)

        url_inventory = [
            {
                "index": r.index,
                "url": r.requested_url,
                "final_url": r.final_url,
                "status": r.status_code,
                "ok": r.ok,
                "depth": r.depth,
                "content_type": r.content_type,
                "bytes": r.content_length,
                "discovered_from": r.discovered_from,
                "redirects": len(r.redirect_chain),
                "duplicate_of": r.duplicate_of,
                "html_path": r.html_path or None,
                "error": r.error,
            }
            for r in sorted(self._records, key=lambda x: x.index)
        ]

        # Corpus totals, not just what this invocation added. On a resumed run
        # `stats.ok` is 0 because everything came from the cache, and reporting
        # that as "0 pages crawled" would be actively misleading.
        total_ok = sum(1 for r in self._records if r.ok)
        total_failed = sum(1 for r in self._records if not r.ok and r.duplicate_of is None)

        write_json(
            {
                "domain": self.cfg.domain,
                "seed_url": self.seed_url,
                "root_host": self.root_host,
                "generated_at": time.time(),
                "total": len(url_inventory),
                "ok": total_ok,
                "urls": url_inventory,
                "unvisited": [u for u, _, _ in list(self._frontier)[:5000]],
            },
            self.ws.urls_json,
        )

        write_json(self._sitemap_payload, self.ws.sitemap_json)

        report = {
            "domain": self.cfg.domain,
            "seed_url": self.seed_url,
            "started_at": stats.started_at,
            "finished_at": stats.finished_at,
            "duration_seconds": duration,
            "pages": {
                # Corpus-wide totals across every run that fed this workspace.
                "discovered": max(stats.discovered, len(self._records)),
                "stored_ok": total_ok,
                "failed": total_failed,
                "left_unvisited": len(self._frontier),
                # This invocation only.
                "fetched_this_run": stats.fetched,
                "newly_stored_this_run": stats.ok,
                "from_cache": stats.from_cache,
                # Kept for backward compatibility with older reports.
                "fetched": stats.fetched,
            },
            "skipped": {
                "offsite": stats.skipped_offsite,
                "robots": stats.skipped_robots,
                "pattern": stats.skipped_pattern,
                "trap": stats.skipped_trap,
                "content_type": stats.skipped_content_type,
                "duplicate_content": stats.skipped_duplicate_content,
            },
            "trap_reasons": stats.trap_reasons,
            "status_counts": stats.status_counts,
            "errors": stats.errors,
            "bytes_downloaded": stats.bytes_downloaded,
            "pages_per_second": round(stats.fetched / duration, 2) if duration > 0 else 0.0,
            "robots": self._robots_info,
            "sitemap": {
                "url_count": self._sitemap_payload.get("url_count", 0),
                "files": self._sitemap_payload.get("sitemaps_fetched", []),
            },
            "config": {
                "max_pages": self.ccfg.max_pages,
                "max_depth": self.ccfg.max_depth,
                "concurrency": self.ccfg.concurrency,
                "respect_robots": self.ccfg.respect_robots,
                "include_subdomains": self.ccfg.include_subdomains,
            },
        }
        write_json(report, self.ws.crawl_report)

        logger.info(
            f"crawl complete: {total_ok:,} pages in the corpus "
            f"({stats.ok:,} new this run, {stats.from_cache:,} from cache), "
            f"{total_failed:,} failed, {stats.skipped_duplicate_content:,} duplicate bodies, "
            f"{duration:.1f}s"
        )

        # An empty crawl is always a configuration problem, and the reason is
        # always in the skip counters. Say which one rather than letting the
        # extract phase fail with "no pages found" a moment later.
        if total_ok == 0:
            reasons: list[str] = []
            if stats.skipped_pattern:
                reasons.append(
                    f"{stats.skipped_pattern:,} URL(s) rejected by include/exclude patterns — "
                    f"check crawler.include_patterns matches the normalised URL form "
                    f"(no trailing slash, e.g. {self.seed_url})"
                )
            if stats.skipped_robots:
                reasons.append(
                    f"{stats.skipped_robots:,} URL(s) disallowed by robots.txt — "
                    f"set crawler.respect_robots=false only if you are authorised to"
                )
            if stats.skipped_offsite:
                reasons.append(
                    f"{stats.skipped_offsite:,} URL(s) were off-site — "
                    f"set crawler.include_subdomains=true if the content lives on a subdomain"
                )
            if stats.skipped_content_type:
                reasons.append(
                    f"{stats.skipped_content_type:,} response(s) were not HTML"
                )
            if stats.failed:
                reasons.append(f"{stats.failed:,} request(s) failed: {stats.errors}")
            if not reasons:
                reasons.append("the seed URL returned no usable HTML")

            logger.error("crawl stored zero pages. Likely cause(s):")
            for reason in reasons:
                logger.error(f"  - {reason}")

        return report


async def run_crawl(config: AutotrainConfig, workspace: Workspace) -> dict[str, Any]:
    return await Crawler(config, workspace).run()


def crawl_sync(config: AutotrainConfig, workspace: Workspace) -> dict[str, Any]:
    """Synchronous entry point used by the orchestrator."""
    return asyncio.run(run_crawl(config, workspace))
