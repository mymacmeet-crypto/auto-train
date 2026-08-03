"""
robots.txt fetching and evaluation.

Uses Python's stdlib RobotFileParser for the matching rules, wrapped so that:
  - fetch failures are explicit states rather than silent "allow everything"
  - Crawl-delay and Sitemap directives are surfaced to the crawler
  - the whole thing is a no-op (allow all) when respect_robots is false

A 4xx on robots.txt means "no restrictions" per the standard. A 5xx or a
timeout is ambiguous; we treat it as allow-all but log a warning, matching
what mainstream crawlers do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from autotrain.utils.logging import get_logger

logger = get_logger("crawler.robots")


@dataclass
class RobotsPolicy:
    """Evaluated robots.txt for one origin."""

    origin: str
    enabled: bool = True
    fetched: bool = False
    status_code: int | None = None
    crawl_delay: float | None = None
    sitemaps: list[str] = field(default_factory=list)
    _parser: RobotFileParser | None = None
    raw: str = ""

    def allows(self, url: str, user_agent: str) -> bool:
        if not self.enabled or self._parser is None:
            return True
        try:
            return self._parser.can_fetch(user_agent, url)
        except Exception:
            # A malformed robots.txt should not stop the crawl.
            return True


def robots_url_for(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


async def fetch_robots(
    client: httpx.AsyncClient,
    origin_url: str,
    user_agent: str,
    *,
    enabled: bool = True,
    timeout: float = 15.0,
) -> RobotsPolicy:
    """Fetch and parse robots.txt for the origin of `origin_url`."""
    policy = RobotsPolicy(origin=robots_url_for(origin_url), enabled=enabled)

    if not enabled:
        logger.info("robots.txt disabled by configuration — crawling everything in scope")
        return policy

    try:
        response = await client.get(policy.origin, timeout=timeout)
        policy.status_code = response.status_code
    except httpx.HTTPError as exc:
        logger.warning(f"robots.txt unreachable ({exc.__class__.__name__}) — proceeding as allow-all")
        return policy

    if response.status_code >= 500:
        logger.warning(
            f"robots.txt returned {response.status_code} — proceeding as allow-all"
        )
        return policy

    if response.status_code >= 400:
        logger.info(f"no robots.txt ({response.status_code}) — no crawl restrictions")
        policy.fetched = True
        return policy

    text = response.text
    policy.raw = text
    policy.fetched = True

    parser = RobotFileParser()
    parser.set_url(policy.origin)
    try:
        parser.parse(text.splitlines())
    except Exception as exc:
        logger.warning(f"robots.txt failed to parse ({exc}) — proceeding as allow-all")
        return policy

    policy._parser = parser

    # RobotFileParser exposes crawl_delay only for a matching agent group.
    for agent in (user_agent, "*"):
        try:
            delay = parser.crawl_delay(agent)
        except Exception:
            delay = None
        if delay is not None:
            policy.crawl_delay = float(delay)
            break

    # Sitemap directives are global, not per-agent; stdlib exposes them
    # through site_maps() on 3.8+, but parse them manually as a fallback.
    try:
        found = parser.site_maps() or []
    except Exception:
        found = []
    if not found:
        found = [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.lower().startswith("sitemap:")
        ]
    policy.sitemaps = [s for s in found if s]

    logger.info(
        f"robots.txt loaded — {len(policy.sitemaps)} sitemap directive(s), "
        f"crawl-delay {policy.crawl_delay if policy.crawl_delay is not None else 'unset'}"
    )
    return policy
