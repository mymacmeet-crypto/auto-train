"""URL normalisation, scoping, and trap detection."""

from __future__ import annotations

import pytest

from autotrain.crawler.urlnorm import (
    PatternFilter,
    looks_like_trap,
    normalize_url,
    registrable_host,
    same_site,
)


class TestNormalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://Example.COM/path", "https://example.com/path"),
            ("HTTPS://example.com:443/a", "https://example.com/a"),
            ("http://example.com:80/a", "http://example.com/a"),
            ("https://example.com:8443/a", "https://example.com:8443/a"),
            ("https://example.com/a#fragment", "https://example.com/a"),
            ("https://example.com/docs/", "https://example.com/docs"),
            ("https://example.com/", "https://example.com/"),
            ("https://example.com", "https://example.com/"),
            ("https://example.com//a///b", "https://example.com/a/b"),
            ("https://example.com/a/./b/../c", "https://example.com/a/c"),
            ("https://example.com/index.html", "https://example.com/"),
            ("https://example.com/docs/index.html", "https://example.com/docs"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert normalize_url(raw) == expected

    def test_trailing_slash_and_bare_are_identical(self) -> None:
        assert normalize_url("https://e.com/docs") == normalize_url("https://e.com/docs/")

    def test_query_params_sorted(self) -> None:
        a = normalize_url("https://e.com/s?b=2&a=1")
        b = normalize_url("https://e.com/s?a=1&b=2")
        assert a == b == "https://e.com/s?a=1&b=2"

    def test_tracking_params_stripped(self) -> None:
        result = normalize_url("https://e.com/p?utm_source=x&fbclid=y&real=1")
        assert result == "https://e.com/p?real=1"

    def test_all_params_tracking_leaves_no_query(self) -> None:
        assert normalize_url("https://e.com/p?utm_source=x") == "https://e.com/p"

    @pytest.mark.parametrize(
        "raw",
        ["javascript:void(0)", "mailto:a@b.com", "tel:+123", "data:text/html,x",
         "ftp://e.com/f", "about:blank", ""],
    )
    def test_non_navigational_rejected(self, raw: str) -> None:
        assert normalize_url(raw) is None

    def test_relative_resolution(self) -> None:
        assert normalize_url("../c", base="https://e.com/a/b/") == "https://e.com/a/c"
        assert normalize_url("/abs", base="https://e.com/a/b") == "https://e.com/abs"

    def test_path_case_preserved(self) -> None:
        # Many servers are case-sensitive on paths; only host is lowercased.
        assert normalize_url("https://E.com/MyPage") == "https://e.com/MyPage"


class TestScope:
    def test_exact_host(self) -> None:
        assert same_site("https://e.com/a", "e.com", include_subdomains=False)

    def test_www_equals_apex(self) -> None:
        assert same_site("https://www.e.com/a", "e.com", include_subdomains=False)
        assert same_site("https://e.com/a", "www.e.com", include_subdomains=False)

    def test_prefix_not_stripped_by_character_class(self) -> None:
        # Regression: lstrip("www.") would mangle "web.example.com".
        assert not same_site("https://web.e.com/a", "e.com", include_subdomains=False)

    def test_subdomains_off(self) -> None:
        assert not same_site("https://docs.e.com/a", "e.com", include_subdomains=False)

    def test_subdomains_on(self) -> None:
        assert same_site("https://docs.e.com/a", "e.com", include_subdomains=True)

    def test_foreign_domain_always_rejected(self) -> None:
        assert not same_site("https://evil.com/a", "e.com", include_subdomains=True)

    def test_lookalike_domain_rejected(self) -> None:
        assert not same_site("https://e.com.evil.com/a", "e.com", include_subdomains=True)

    def test_registrable_two_level_suffix(self) -> None:
        assert registrable_host("docs.example.co.uk") == "example.co.uk"
        assert registrable_host("example.com") == "example.com"


class TestTraps:
    def test_clean_url_passes(self) -> None:
        assert looks_like_trap("https://e.com/docs/guide/intro") is None

    def test_repeated_segment(self) -> None:
        reason = looks_like_trap("https://e.com/a/a/a/a/a/b")
        assert reason is not None and "repeated_segment" in reason

    def test_alternating_cycle_detected(self) -> None:
        assert looks_like_trap("https://e.com/" + "a/b/" * 8) is not None

    def test_cycle_below_repeat_threshold_still_caught(self) -> None:
        # Each of a, b, c appears exactly 3 times, so the repeated-segment
        # rule does not fire; only the cycle detector catches this.
        reason = looks_like_trap("https://e.com/" + "a/b/c/" * 3)
        assert reason is not None and "cyclic" in reason

    def test_too_deep(self) -> None:
        assert looks_like_trap("https://e.com/" + "x/" * 30) == "path_too_deep"

    def test_too_many_params(self) -> None:
        url = "https://e.com/s?" + "&".join(f"f{i}=1" for i in range(12))
        assert looks_like_trap(url) == "too_many_query_params"

    def test_too_long(self) -> None:
        assert looks_like_trap("https://e.com/" + "x" * 3000) == "url_too_long"


class TestPatternFilter:
    def test_default_allows(self) -> None:
        assert PatternFilter([], []).allows("https://e.com/a")[0]

    def test_exclude_blocks(self) -> None:
        allowed, reason = PatternFilter([], [r"\.pdf$"]).allows("https://e.com/f.pdf")
        assert not allowed and "excluded" in reason

    def test_include_restricts_to_subtree(self) -> None:
        f = PatternFilter([r"/docs/"], [])
        assert f.allows("https://e.com/docs/a")[0]
        assert not f.allows("https://e.com/blog/a")[0]

    def test_exclude_wins_over_include(self) -> None:
        f = PatternFilter([r"/docs/"], [r"/docs/private"])
        assert not f.allows("https://e.com/docs/private/x")[0]
