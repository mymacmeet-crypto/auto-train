"""
Regressions around crawl scoping.

Both cases here were real bugs found by running the pipeline against a live
documentation site, not hypotheticals.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from autotrain.crawler.crawler import crawl_sync
from autotrain.utils.config import load_config
from autotrain.utils.paths import Workspace

BODY = (
    "<html><body><main><h1>{title}</h1>"
    "<p>{filler}</p>"
    '<a href="/docs/child">Child</a>'
    "</main></body></html>"
)


def _mock_infrastructure(host: str = "https://site.test") -> None:
    respx.get(f"{host}/robots.txt").mock(return_value=httpx.Response(404))
    for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
                 "/sitemap.xml.gz", "/sitemaps/sitemap.xml", "/wp-sitemap.xml"):
        respx.get(f"{host}{path}").mock(return_value=httpx.Response(404))


def _page(title: str) -> httpx.Response:
    return httpx.Response(
        200,
        html=BODY.format(title=title, filler="Documentation body text. " * 30),
        headers={"content-type": "text/html; charset=utf-8"},
    )


class TestSeedExemption:
    @respx.mock
    def test_seed_survives_a_pattern_that_does_not_match_it(self, tmp_path) -> None:
        """
        A trailing slash in include_patterns must not empty the crawl.

        The seed https://site.test/docs/ normalises to .../docs, so a pattern
        anchored with a trailing slash would reject the entry point and the
        run would end with zero pages.
        """
        _mock_infrastructure()
        respx.get("https://site.test/docs").mock(return_value=_page("Docs Home"))
        respx.get("https://site.test/docs/child").mock(return_value=_page("Child"))

        cfg = load_config(
            domain="https://site.test/docs/",
            overrides=[
                f"runtime.workspace={tmp_path.as_posix()}",
                "crawler.concurrency=1",
                r"crawler.include_patterns=^https://site\.test/docs/",
            ],
        )
        ws = Workspace(tmp_path / "site").ensure()
        report = crawl_sync(cfg, ws)

        assert report["pages"]["stored_ok"] >= 1, "seed URL was filtered out by its own scope pattern"

    @respx.mock
    def test_pattern_still_scopes_discovered_links(self, tmp_path) -> None:
        """The seed exemption must not leak into general link filtering."""
        _mock_infrastructure()
        home = httpx.Response(
            200,
            html=(
                "<html><body><main><p>" + "Body text here. " * 30 + "</p>"
                '<a href="/docs/child">In scope</a>'
                '<a href="/blog/post">Out of scope</a>'
                "</main></body></html>"
            ),
            headers={"content-type": "text/html"},
        )
        respx.get("https://site.test/").mock(return_value=home)
        respx.get("https://site.test/docs/child").mock(return_value=_page("Child"))
        blog = respx.get("https://site.test/blog/post").mock(return_value=_page("Blog"))

        cfg = load_config(
            domain="https://site.test",
            overrides=[
                f"runtime.workspace={tmp_path.as_posix()}",
                "crawler.concurrency=1",
                r"crawler.include_patterns=^https://site\.test/docs",
            ],
        )
        ws = Workspace(tmp_path / "site").ensure()
        report = crawl_sync(cfg, ws)

        assert not blog.called, "out-of-scope link was fetched"
        assert report["skipped"]["pattern"] >= 1

    @respx.mock
    def test_robots_still_applies_to_the_seed(self, tmp_path) -> None:
        """Exempt from patterns, not from robots.txt."""
        respx.get("https://site.test/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
        )
        for path in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml",
                     "/sitemap.xml.gz", "/sitemaps/sitemap.xml", "/wp-sitemap.xml"):
            respx.get(f"https://site.test{path}").mock(return_value=httpx.Response(404))
        root = respx.get("https://site.test/").mock(return_value=_page("Home"))

        cfg = load_config(
            domain="https://site.test",
            overrides=[f"runtime.workspace={tmp_path.as_posix()}", "crawler.concurrency=1"],
        )
        ws = Workspace(tmp_path / "site").ensure()
        report = crawl_sync(cfg, ws)

        assert not root.called
        assert report["pages"]["stored_ok"] == 0


class TestEmptyCrawlReporting:
    @respx.mock
    def test_corpus_totals_survive_a_cached_rerun(self, tmp_path) -> None:
        """
        A fully-cached second run must not report an empty corpus.

        stats.ok counts only newly stored pages, so a resumed run would
        otherwise say "0 pages stored" for a workspace holding hundreds.
        """
        _mock_infrastructure()
        respx.get("https://site.test/").mock(return_value=_page("Home"))
        respx.get("https://site.test/docs/child").mock(return_value=_page("Child"))

        overrides = [f"runtime.workspace={tmp_path.as_posix()}", "crawler.concurrency=1"]
        cfg = load_config(domain="https://site.test", overrides=overrides)
        ws = Workspace(tmp_path / "site").ensure()

        first = crawl_sync(cfg, ws)
        assert first["pages"]["stored_ok"] >= 1
        assert first["pages"]["from_cache"] == 0

        second = crawl_sync(cfg, ws)
        assert second["pages"]["stored_ok"] == first["pages"]["stored_ok"]
        assert second["pages"]["from_cache"] >= 1
        assert second["pages"]["newly_stored_this_run"] == 0
