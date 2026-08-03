"""
HTTP fetching with retries, per-host rate limiting, and full response capture.

Phase 2 requires that we record the raw HTML *plus* the final URL, status,
headers, and redirect chain. httpx gives us all of that if we disable
automatic redirect following and walk the chain ourselves — which we do,
because a redirect chain is itself information (it tells the deduplicator
that /old-page and /new-page are the same document).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from autotrain.utils.logging import get_logger

logger = get_logger("crawler.fetcher")

MAX_REDIRECTS = 10


@dataclass
class FetchResult:
    """Everything Phase 2 must persist about one page fetch."""

    requested_url: str
    final_url: str = ""
    status_code: int = 0
    ok: bool = False
    content_type: str = ""
    encoding: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    redirect_chain: list[dict[str, Any]] = field(default_factory=list)
    html: str = ""
    content_length: int = 0
    elapsed_ms: int = 0
    fetched_at: float = 0.0
    error: str | None = None
    attempts: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """The JSON record written to raw_html/index.jsonl (excludes the body)."""
        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "ok": self.ok,
            "content_type": self.content_type,
            "encoding": self.encoding,
            "headers": self.headers,
            "redirect_chain": self.redirect_chain,
            "content_length": self.content_length,
            "elapsed_ms": self.elapsed_ms,
            "fetched_at": self.fetched_at,
            "error": self.error,
            "attempts": self.attempts,
        }


class HostRateLimiter:
    """
    Enforce a minimum interval between request starts for each host.

    Per-host rather than global: crawling a site that serves images from a
    CDN subdomain should not throttle the CDN because the origin is slow.
    """

    def __init__(self, delay_seconds: float = 0.0) -> None:
        self._delay = max(0.0, delay_seconds)
        self._next_allowed: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def set_delay(self, delay_seconds: float) -> None:
        self._delay = max(0.0, delay_seconds)

    async def acquire(self, host: str) -> None:
        if self._delay <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            earliest = self._next_allowed.get(host, 0.0)
            if now < earliest:
                await asyncio.sleep(earliest - now)
                now = time.monotonic()
            self._next_allowed[host] = now + self._delay


# Status codes worth retrying: transient server and rate-limit conditions.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524})


class Fetcher:
    """Async HTTP client wrapper that produces FetchResult objects."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 1.5,
        max_bytes: int = 5_000_000,
        allowed_content_types: tuple[str, ...] = ("text/html", "application/xhtml+xml"),
        rate_limiter: HostRateLimiter | None = None,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._retries = max(0, retries)
        self._backoff = backoff
        self._max_bytes = max_bytes
        self._allowed = tuple(t.lower() for t in allowed_content_types)
        self._limiter = rate_limiter or HostRateLimiter(0.0)

    async def fetch(self, url: str) -> FetchResult:
        result = FetchResult(requested_url=url, fetched_at=time.time())
        started = time.monotonic()

        for attempt in range(self._retries + 1):
            result.attempts = attempt + 1
            try:
                await self._follow(url, result)
            except httpx.TimeoutException:
                result.error = "timeout"
            except httpx.TooManyRedirects:
                result.error = "too_many_redirects"
                break
            except httpx.HTTPError as exc:
                result.error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # defensive: one bad page must not kill the crawl
                result.error = f"{type(exc).__name__}: {exc}"
                break
            else:
                result.error = None
                break

            transient = result.error == "timeout" or result.status_code in RETRYABLE_STATUS
            if attempt < self._retries and transient:
                delay = self._backoff ** attempt
                # Honour Retry-After when the server sent one.
                retry_after = result.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, min(float(retry_after), 60.0))
                    except ValueError:
                        pass
                logger.debug(f"retry {attempt + 1}/{self._retries} for {url} in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            break

        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    async def _follow(self, url: str, result: FetchResult) -> None:
        """Walk redirects manually so the whole chain is recorded."""
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            host = httpx.URL(current).host or ""
            await self._limiter.acquire(host)

            response = await self._client.get(
                current, timeout=self._timeout, follow_redirects=False
            )
            result.status_code = response.status_code
            result.headers = {k.lower(): v for k, v in response.headers.items()}
            result.final_url = str(response.url)

            if response.is_redirect:
                location = response.headers.get("location", "")
                if not location:
                    result.error = "redirect_without_location"
                    return
                nxt = str(response.url.join(location))
                result.redirect_chain.append(
                    {"from": current, "to": nxt, "status": response.status_code}
                )
                if hop >= MAX_REDIRECTS:
                    raise httpx.TooManyRedirects("redirect limit reached", request=response.request)
                current = nxt
                continue

            # Terminal response.
            content_type = result.headers.get("content-type", "")
            result.content_type = content_type.split(";")[0].strip().lower()

            if response.status_code != 200:
                result.ok = False
                return

            if self._allowed and result.content_type not in self._allowed:
                result.ok = False
                result.error = f"content_type_skipped:{result.content_type or 'unknown'}"
                return

            content = response.content
            result.content_length = len(content)
            if result.content_length > self._max_bytes:
                result.ok = False
                result.error = f"too_large:{result.content_length}"
                return

            result.encoding = response.encoding or "utf-8"
            try:
                result.html = response.text
            except (UnicodeDecodeError, LookupError):
                result.html = content.decode("utf-8", errors="replace")
                result.encoding = "utf-8(replaced)"

            result.ok = bool(result.html.strip())
            if not result.ok:
                result.error = "empty_body"
            return

        result.error = "too_many_redirects"
