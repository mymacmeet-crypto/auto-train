"""Phases 1-2: website discovery and HTML collection."""

from autotrain.crawler.crawler import Crawler, PageRecord, crawl_sync, run_crawl
from autotrain.crawler.fetcher import Fetcher, FetchResult, HostRateLimiter
from autotrain.crawler.robots import RobotsPolicy, fetch_robots
from autotrain.crawler.sitemap import SitemapEntry, discover_sitemaps, parse_sitemap_bytes
from autotrain.crawler.urlnorm import (
    PatternFilter,
    looks_like_trap,
    normalize_url,
    same_site,
)

__all__ = [
    "Crawler",
    "PageRecord",
    "crawl_sync",
    "run_crawl",
    "Fetcher",
    "FetchResult",
    "HostRateLimiter",
    "RobotsPolicy",
    "fetch_robots",
    "SitemapEntry",
    "discover_sitemaps",
    "parse_sitemap_bytes",
    "PatternFilter",
    "looks_like_trap",
    "normalize_url",
    "same_site",
]
