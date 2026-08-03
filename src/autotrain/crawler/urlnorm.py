"""
URL normalisation, scoping, and trap detection.

Normalisation exists to answer one question correctly: "have I already seen
this page?" Two URLs that fetch identical bytes must produce identical keys,
or a 10,000-page crawl becomes a 40,000-page crawl of the same content under
different query-parameter orderings.
"""

from __future__ import annotations

import re
from collections import Counter
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

# Query parameters that never change the rendered content. Stripping them
# collapses the tracking-parameter explosion that social sharing creates.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_social", "utm_brand",
        "gclid", "gclsrc", "dclid", "fbclid", "msclkid", "twclid", "igshid",
        "mc_cid", "mc_eid", "_ga", "_gl", "yclid", "wickedid", "s_kwcid",
        "ref", "referrer", "source", "campaign",
        "share", "shared", "amp", "output", "print",
        "sessionid", "session_id", "phpsessid", "jsessionid", "sid",
    }
)

DEFAULT_PORTS = {"http": 80, "https": 443}

# Index filenames that are equivalent to the bare directory.
_INDEX_FILES = re.compile(r"/(?:index|default|home)\.(?:html?|php|aspx?|jsp)$", re.I)


def normalize_url(url: str, *, base: str | None = None, keep_query: bool = True) -> str | None:
    """
    Canonicalise a URL. Returns None if it is not a fetchable http(s) URL.

    Applied transformations:
      - resolve against `base` when relative
      - lowercase scheme and host (path case is preserved: many servers care)
      - drop the fragment entirely (never a distinct document)
      - drop default ports
      - drop tracking parameters, sort the rest for a stable key
      - collapse /./ and /../, deduplicate slashes
      - normalise index.html -> directory form
      - strip a trailing slash except on the site root
    """
    if not url:
        return None

    url = url.strip()
    # Reject non-navigational schemes early.
    lowered = url.lower()
    for bad in ("javascript:", "mailto:", "tel:", "data:", "blob:", "about:", "file:", "ftp:"):
        if lowered.startswith(bad):
            return None

    if base:
        url = urljoin(base, url)

    try:
        parts = urlsplit(url)
    except ValueError:
        return None

    if parts.scheme not in ("http", "https"):
        return None
    if not parts.hostname:
        return None

    host = parts.hostname.lower().rstrip(".")
    # IDN safety: keep the encoded form so the key is byte-stable.
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        pass

    netloc = host
    if parts.port and parts.port != DEFAULT_PORTS.get(parts.scheme):
        netloc = f"{host}:{parts.port}"

    path = _normalize_path(parts.path)

    query = ""
    if keep_query and parts.query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        # Sorting makes ?a=1&b=2 and ?b=2&a=1 the same key.
        pairs.sort()
        query = urlencode(pairs, doseq=True)

    return urlunsplit((parts.scheme, netloc, path, query, ""))


def _normalize_path(path: str) -> str:
    if not path:
        return "/"

    # Resolve . and .. without touching the network.
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)

    out = "/" + "/".join(segments)

    # index.html and its friends are the directory itself.
    out = _INDEX_FILES.sub("/", out)

    # Canonical form has no trailing slash (except root) so /docs and /docs/
    # do not both get crawled.
    if len(out) > 1:
        out = out.rstrip("/")
    return out or "/"


def registrable_host(host: str) -> str:
    """
    Approximate the registrable domain without a public-suffix dependency.

    Handles the common two-level suffixes (co.uk, com.au, ...) explicitly;
    everything else falls back to the last two labels. Only used for
    subdomain scoping, where an occasional miss is harmless.
    """
    host = host.lower().strip(".")
    labels = host.split(".")
    if len(labels) <= 2:
        return host

    two_level = {
        "co", "com", "net", "org", "gov", "edu", "ac", "mil",
    }
    if labels[-2] in two_level and len(labels[-1]) <= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def same_site(url: str, root_host: str, *, include_subdomains: bool) -> bool:
    """Is `url` inside the crawl scope defined by `root_host`?"""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    host = host.lower().rstrip(".")
    root_host = root_host.lower().rstrip(".")

    if host == root_host:
        return True
    if not include_subdomains:
        # www is conventionally the same site as the apex. removeprefix, not
        # lstrip: lstrip strips a *character set* and would turn
        # "web.example.com" into "eb.example.com".
        return host.removeprefix("www.") == root_host.removeprefix("www.")
    return registrable_host(host) == registrable_host(root_host)


def looks_like_trap(
    url: str,
    *,
    max_repeated_segments: int = 3,
    max_path_depth: int = 24,
    max_query_params: int = 8,
    max_length: int = 2048,
) -> str | None:
    """
    Detect crawler traps. Returns a reason string, or None when the URL is fine.

    Three signatures catch nearly all of them:
      - a path segment repeating many times: /a/b/a/b/a/b/...
      - absurd path depth from relative-link mistakes
      - a combinatorial explosion of query parameters (faceted search)
    """
    if len(url) > max_length:
        return "url_too_long"

    try:
        parts = urlsplit(url)
    except ValueError:
        return "unparseable"

    segments = [s for s in parts.path.split("/") if s]
    if len(segments) > max_path_depth:
        return "path_too_deep"

    if segments:
        counts = Counter(segments)
        most_common, count = counts.most_common(1)[0]
        # A legitimate path essentially never repeats one segment this many
        # times, regardless of how short that segment is.
        if count > max_repeated_segments:
            return f"repeated_segment:{most_common}"

        # Detect an alternating cycle like /a/b/a/b/ that no single segment
        # count would flag at low depth.
        if len(segments) >= 6:
            for period in (2, 3):
                if len(segments) >= period * 3:
                    window = segments[-period:]
                    repeats = 1
                    idx = len(segments) - period * 2
                    while idx >= 0 and segments[idx : idx + period] == window:
                        repeats += 1
                        idx -= period
                    if repeats >= 3:
                        return f"cyclic_path:{'/'.join(window)}"

    if parts.query:
        params = parse_qsl(parts.query, keep_blank_values=True)
        if len(params) > max_query_params:
            return "too_many_query_params"

    return None


class PatternFilter:
    """Compiled include/exclude regex filter over URLs."""

    def __init__(self, include: list[str], exclude: list[str]) -> None:
        self._include = [re.compile(p, re.I) for p in include]
        self._exclude = [re.compile(p, re.I) for p in exclude]

    def allows(self, url: str) -> tuple[bool, str]:
        for pattern in self._exclude:
            if pattern.search(url):
                return False, f"excluded:{pattern.pattern[:40]}"
        if self._include:
            for pattern in self._include:
                if pattern.search(url):
                    return True, ""
            return False, "not_in_include_patterns"
        return True, ""
