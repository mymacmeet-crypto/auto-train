"""Extraction (Phase 3) and corpus cleaning (Phase 4)."""

from __future__ import annotations

import pytest

from autotrain.cleaner.clean import (
    NearDuplicateIndex,
    find_boilerplate_lines,
    remove_lines,
    strip_nav_runs,
)
from autotrain.extractor.extract import extract_page
from autotrain.extractor.markdown import html_to_markdown
from autotrain.utils.config import load_config
from autotrain.utils.text import hamming, simhash


class TestMarkdownConversion:
    def test_headings_keep_levels(self) -> None:
        md = html_to_markdown("<h1>One</h1><h2>Two</h2><h3>Three</h3>")
        assert "# One" in md and "## Two" in md and "### Three" in md

    def test_code_block_preserves_indentation(self) -> None:
        html = '<pre><code class="language-python">def f():\n    return 1\n</code></pre>'
        md = html_to_markdown(html)
        assert "```python" in md
        assert "    return 1" in md, "indentation inside code must survive verbatim"

    def test_code_language_from_various_class_names(self) -> None:
        for cls in ("language-js", "lang-js", "highlight-js", "js"):
            md = html_to_markdown(f'<pre><code class="{cls}">x=1</code></pre>')
            assert "```js" in md, cls

    def test_nested_backticks_get_longer_fence(self) -> None:
        md = html_to_markdown("<pre><code>a ``` b</code></pre>")
        assert "````" in md

    def test_table_becomes_pipe_table(self) -> None:
        html = "<table><tr><th>Name</th><th>Type</th></tr><tr><td>id</td><td>int</td></tr></table>"
        md = html_to_markdown(html)
        assert "| Name | Type |" in md
        assert "| --- | --- |" in md
        assert "| id | int |" in md

    def test_ragged_table_is_padded(self) -> None:
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td></tr></table>"
        md = html_to_markdown(html)
        assert md.count("|") >= 6  # still a well-formed 2-column table

    def test_nested_lists_indent(self) -> None:
        html = "<ul><li>a<ul><li>b</li></ul></li></ul>"
        md = html_to_markdown(html)
        assert "- a" in md and "  - b" in md

    def test_ordered_list_respects_start(self) -> None:
        md = html_to_markdown('<ol start="3"><li>x</li><li>y</li></ol>')
        assert "3. x" in md and "4. y" in md

    def test_definition_list(self) -> None:
        md = html_to_markdown("<dl><dt>timeout</dt><dd>seconds to wait</dd></dl>")
        assert "**timeout**" in md and ": seconds to wait" in md

    def test_scripts_and_styles_dropped(self) -> None:
        md = html_to_markdown("<p>keep</p><script>evil()</script><style>.a{}</style>")
        assert "keep" in md and "evil" not in md and ".a{" not in md

    def test_inline_code_and_emphasis(self) -> None:
        md = html_to_markdown("<p>Use <code>x=1</code> and <strong>bold</strong>.</p>")
        assert "`x=1`" in md and "**bold**" in md


class TestExtraction:
    @pytest.fixture
    def cfg(self):
        return load_config(domain="https://example.com")

    def test_chrome_is_removed(self, cfg) -> None:
        html = """
        <html><head><title>Guide | MySite</title></head><body>
          <nav>Home About Contact</nav>
          <header>Site header</header>
          <div class="cookie-banner">We use cookies. Accept?</div>
          <aside class="sidebar">Sidebar links</aside>
          <main><h1>Configuration Guide</h1>
            <p>Set the timeout value in the configuration file to control how long
               the client waits before giving up on a slow request.</p>
            <p>The default timeout is thirty seconds, which suits most deployments
               but can be raised for very large uploads.</p>
          </main>
          <footer>Copyright 2024</footer>
        </body></html>
        """
        md, title, _, _, method = extract_page(html, "https://example.com/guide", cfg)
        assert "timeout" in md
        assert "cookies" not in md.lower()
        assert "Sidebar links" not in md
        assert method in ("trafilatura", "structural")

    def test_title_strips_site_suffix(self, cfg) -> None:
        html = (
            "<html><head><title>Configuration Guide | MySite</title></head>"
            "<body><main><h1>Configuration Guide</h1>"
            "<p>" + "Detailed configuration instructions follow below. " * 8 + "</p>"
            "</main></body></html>"
        )
        _, title, _, _, _ = extract_page(html, "https://example.com/g", cfg)
        assert title == "Configuration Guide"

    def test_hidden_elements_removed(self, cfg) -> None:
        html = (
            "<html><body><main>"
            '<div style="display:none">SECRETHIDDEN</div>'
            "<p>" + "This is the visible body text of the document. " * 10 + "</p>"
            "</main></body></html>"
        )
        md, _, _, _, _ = extract_page(html, "https://example.com/h", cfg)
        assert "SECRETHIDDEN" not in md

    def test_empty_page_yields_nothing(self, cfg) -> None:
        md, _, _, _, method = extract_page("<html><body></body></html>", "https://e.com/", cfg)
        assert md == "" and method == "empty"

    def test_nested_chrome_does_not_crash(self, cfg) -> None:
        """
        Regression: removal loops iterate a list captured before any node was
        removed, so decomposing a parent leaves stale children behind.
        Touching one raised AttributeError and killed the whole phase.
        """
        html = (
            "<html><body>"
            '<div class="sidebar"><div class="menu-item"><span class="cookie-notice">'
            '<a class="social-share" href="#">x</a></span></div></div>'
            '<nav><div class="breadcrumb"><span class="sr-only">nav</span></div></nav>'
            "<main><p>" + "Real documentation content here. " * 20 + "</p></main>"
            "</body></html>"
        )
        md, _, _, _, method = extract_page(html, "https://e.com/p", cfg)
        assert "Real documentation content" in md
        assert method != "empty"

    def test_deeply_nested_hidden_elements(self, cfg) -> None:
        html = (
            "<html><body>"
            '<div style="display:none"><div style="display:none">'
            '<p style="visibility:hidden">HIDDENTEXT</p></div></div>'
            "<main><p>" + "Visible body content. " * 20 + "</p></main>"
            "</body></html>"
        )
        md, _, _, _, _ = extract_page(html, "https://e.com/p", cfg)
        assert "HIDDENTEXT" not in md


class TestSimHash:
    def test_identical_text_same_fingerprint(self) -> None:
        text = "the quick brown fox jumps over the lazy dog repeatedly"
        assert simhash(text) == simhash(text)

    def test_realistic_near_duplicate_is_close(self) -> None:
        # The case that matters in practice: the same documentation page
        # republished under a different version prefix. One changed sentence
        # in a long document must stay within the dedupe threshold.
        base = (
            "The configuration file controls how the client connects to the server. "
            "Set the timeout value to control how long the client waits. "
            "The default is thirty seconds which suits most deployments. "
            "Raise it for very large uploads over slow links. "
            "Authentication uses a bearer token in the Authorization header. "
            "Tokens expire after one hour and must then be refreshed. "
        ) * 4
        a = base + "This page documents version 2.0 of the client library."
        b = base + "This page documents version 2.1 of the client library."
        assert hamming(simhash(a), simhash(b)) <= 3

    def test_different_text_is_far(self) -> None:
        a = "installation instructions for the command line interface tool"
        b = "frequently asked questions about billing and subscription plans"
        assert hamming(simhash(a), simhash(b)) > 8

    def test_short_sentence_edit_is_detected_as_different(self) -> None:
        # On a short string a single changed word is a large fraction of the
        # content, so it should NOT be collapsed as a duplicate.
        a = "the quick brown fox jumps over the lazy dog today"
        b = "the quick brown fox jumps over the lazy dog tomorrow"
        assert hamming(simhash(a), simhash(b)) > 3

    def test_word_order_matters(self) -> None:
        a = "alpha beta gamma delta epsilon zeta"
        b = "zeta epsilon delta gamma beta alpha"
        assert simhash(a) != simhash(b)


class TestNearDuplicateIndex:
    def test_finds_near_duplicate(self) -> None:
        index = NearDuplicateIndex(threshold=3)
        fingerprint = simhash("the configuration guide explains every available option")
        index.add(1, fingerprint)
        assert index.find(fingerprint) == 1

    def test_distinct_documents_not_matched(self) -> None:
        index = NearDuplicateIndex(threshold=3)
        index.add(1, simhash("installation instructions for the command line tool"))
        other = simhash("frequently asked questions about billing and refunds")
        assert index.find(other) is None


class TestBoilerplate:
    def test_repeated_lines_detected(self) -> None:
        docs = [f"# Page {i}\n\nEdit this page on GitHub\n\nUnique body {i}." for i in range(20)]
        found = find_boilerplate_lines(docs, ratio_threshold=0.35, min_docs=8)
        assert "Edit this page on GitHub" in found

    def test_headings_never_boilerplate(self) -> None:
        docs = ["# Overview\n\nbody " + str(i) for i in range(20)]
        found = find_boilerplate_lines(docs, ratio_threshold=0.35, min_docs=8)
        assert "# Overview" not in found

    def test_long_repeated_lines_kept(self) -> None:
        # Repeated text longer than max_line_chars is shared documentation
        # (a licence notice, a standard warning), not site chrome.
        long_line = "This standard warning about data loss applies to every operation " * 5
        assert len(long_line.strip()) > 200
        docs = [f"{long_line}\n\nbody {i}" for i in range(20)]
        found = find_boilerplate_lines(docs, ratio_threshold=0.35, min_docs=8, max_line_chars=200)
        assert long_line.strip() not in found

    def test_small_corpus_skipped(self) -> None:
        docs = ["Edit this page\n\nbody"] * 3
        assert find_boilerplate_lines(docs, min_docs=8) == set()

    def test_removal_skips_code_blocks(self) -> None:
        doc = "Copyright 2024\n\n```\nCopyright 2024\n```\n\nreal content"
        cleaned, removed = remove_lines(doc, {"Copyright 2024"})
        assert removed == 1
        assert "```\nCopyright 2024\n```" in cleaned


class TestNavStripping:
    def test_menu_run_removed(self) -> None:
        doc = (
            "# Real Heading\n\n"
            "Getting Started\nInstallation\nConfiguration\n"
            "API Reference\nExamples\nChangelog\n\n"
            "This is a real paragraph of prose that explains something useful."
        )
        cleaned, removed = strip_nav_runs(doc, min_run=6)
        assert removed >= 6
        assert "real paragraph" in cleaned

    def test_prose_list_preserved(self) -> None:
        doc = "\n".join(
            f"Item {i} does something specific and is described in a full sentence."
            for i in range(8)
        )
        cleaned, removed = strip_nav_runs(doc, min_run=6)
        assert removed == 0

    def test_code_untouched(self) -> None:
        doc = "```\nAlpha\nBeta\nGamma\nDelta\nEpsilon\nZeta\n```"
        cleaned, removed = strip_nav_runs(doc, min_run=6)
        assert removed == 0 and "Epsilon" in cleaned
