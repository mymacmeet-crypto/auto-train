"""
Phase 3 — content extraction.

Turns each stored HTML page into clean Markdown containing only the article
body: titles, headings, paragraphs, lists, tables, code, notes, warnings.
Everything structural — nav, header, footer, sidebar, cookie banner, ads,
tracking, hidden elements — is removed.

Three-stage strategy, because no single extractor is right for every page:

  1. Pre-strip. A CSS selector blacklist removes chrome that content-density
     scoring reliably keeps (cookie banners score well: they are dense text
     in a prominent block).
  2. trafilatura. Best-in-class boilerplate removal, tuned on news and docs.
     Handles the general case and emits Markdown directly.
  3. Fallback. When trafilatura returns too little — heavily templated pages,
     single-table API references, pages whose body is one <div> — we pick the
     densest plausible content root ourselves and convert it with the
     structure-first converter in markdown.py.

The fallback matters more than it sounds: on real documentation sites it
rescues 5-15% of pages, and those pages are disproportionately the API
reference tables that make the dataset valuable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from autotrain.extractor.markdown import html_to_markdown
from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import (
    normalize_text,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
)

logger = get_logger("extractor")

# Inline styles that hide an element from users; such content is either
# tracking, a11y filler, or an inactive tab we should not train on.
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)", re.I
)

# Class/id fragments that mark chrome on the long tail of sites that do not
# use semantic HTML. Matched as substrings, case-insensitively.
_CHROME_HINTS = (
    "cookie", "consent", "gdpr", "newsletter", "subscribe-",
    "advert", "adsense", "banner-ad", "sponsor",
    "sidebar", "side-bar", "site-nav", "navbar", "topbar", "menu-",
    "breadcrumb", "pagination", "pager", "footer", "site-header",
    "social-share", "sharethis", "addthis", "related-articles",
    "skip-to", "screen-reader", "sr-only", "visually-hidden",
    "back-to-top", "scroll-top", "modal", "popup", "overlay",
)

# Content roots, best first.
_MAIN_SELECTORS = (
    "main", "article", '[role="main"]', "#main-content", "#main", "#content",
    ".main-content", ".content", ".article-body", ".post-content",
    ".markdown-body", ".doc-content", ".documentation", ".rst-content",
    ".theme-doc-markdown", ".md-content",
)


@dataclass
class ExtractedDoc:
    """One page's extracted content."""

    index: int
    url: str
    title: str
    markdown: str
    lang: str = ""
    description: str = ""
    method: str = ""
    char_count: int = 0
    word_count: int = 0
    heading_count: int = 0
    code_block_count: int = 0
    table_count: int = 0
    content_hash: str = ""
    md_path: str = ""

    def to_json(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data.pop("markdown", None)
        return data


def _is_live(node: object) -> bool:
    """
    Is this node still attached to the tree?

    find_all() materialises its result before any removal happens, so
    decomposing a parent leaves stale children in the list we are iterating.
    Touching a decomposed node's attributes raises AttributeError, because
    bs4 sets `attrs` to None on teardown.
    """
    if not isinstance(node, Tag):
        return False
    if getattr(node, "decomposed", False):
        return False
    return getattr(node, "attrs", None) is not None


def _strip_elements(soup: BeautifulSoup, selectors: list[str]) -> int:
    """Remove blacklisted selectors, hidden nodes, and heuristic chrome."""
    removed = 0

    for selector in selectors:
        try:
            for node in soup.select(selector):
                if not _is_live(node):
                    continue
                node.decompose()
                removed += 1
        except Exception:
            # An invalid user-supplied selector must not abort extraction.
            continue

    # Inline-hidden elements.
    for node in soup.find_all(style=True):
        if not _is_live(node):
            continue
        style = node.get("style") or ""
        if _HIDDEN_STYLE.search(style):
            node.decompose()
            removed += 1

    # Heuristic chrome by class/id substring.
    for node in soup.find_all(True):
        if not _is_live(node):
            continue
        tokens: list[str] = []
        classes = node.get("class")
        if classes:
            tokens.extend(classes if isinstance(classes, list) else [classes])
        node_id = node.get("id")
        if node_id:
            tokens.append(str(node_id))
        if not tokens:
            continue
        blob = " ".join(tokens).lower()
        if any(hint in blob for hint in _CHROME_HINTS):
            # Never remove a node that holds most of the page's text — some
            # sites really do put the article inside .content-sidebar.
            text_len = len(node.get_text(" ", strip=True))
            if text_len < 4000:
                node.decompose()
                removed += 1

    return removed


def _page_metadata(soup: BeautifulSoup) -> tuple[str, str, str]:
    """(title, description, lang) from the usual places, best source first."""
    title = ""
    for getter in (
        lambda: (soup.find("meta", property="og:title") or {}).get("content"),
        lambda: (soup.find("meta", attrs={"name": "twitter:title"}) or {}).get("content"),
        lambda: soup.title.string if soup.title else None,
        lambda: soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else None,
    ):
        try:
            value = getter()
        except Exception:
            value = None
        if value and str(value).strip():
            title = str(value).strip()
            break

    # Strip the site-name suffix that almost every <title> carries.
    for sep in (" | ", " — ", " – ", " - ", " :: ", " » "):
        if sep in title:
            head, _, tail = title.rpartition(sep)
            # Only strip when the tail looks like a site name, not a real
            # part of the heading.
            if head and len(tail) < 40 and len(head) > 8:
                title = head.strip()
                break

    description = ""
    for attrs in ({"name": "description"}, {"property": "og:description"}):
        node = soup.find("meta", attrs=attrs)
        if node and node.get("content"):
            description = str(node["content"]).strip()
            break

    lang = ""
    html_tag = soup.find("html")
    if isinstance(html_tag, Tag):
        lang = (html_tag.get("lang") or "").strip()

    return title, description, lang


def _pick_content_root(soup: BeautifulSoup) -> Tag:
    """Choose the densest plausible article container."""
    for selector in _MAIN_SELECTORS:
        try:
            node = soup.select_one(selector)
        except Exception:
            continue
        if node is not None and len(node.get_text(" ", strip=True)) > 200:
            return node

    # No semantic root: score every div/section by text length, discounting
    # link-heavy blocks (a nav is dense text but almost entirely links).
    best: Tag | None = None
    best_score = 0.0
    for node in soup.find_all(("div", "section", "td")):
        text = node.get_text(" ", strip=True)
        length = len(text)
        if length < 200:
            continue
        link_chars = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
        link_density = link_chars / length if length else 1.0
        if link_density > 0.5:
            continue
        # Prefer shallow nodes: the outermost container that is still mostly
        # content, rather than one deeply nested paragraph wrapper.
        depth = len(list(node.parents))
        score = length * (1.0 - link_density) / (1.0 + depth * 0.08)
        if score > best_score:
            best_score = score
            best = node

    return best if best is not None else (soup.body or soup)


def _count_structures(markdown: str) -> tuple[int, int, int]:
    headings = len(re.findall(r"^#{1,6} ", markdown, re.MULTILINE))
    code_blocks = markdown.count("\n```") // 2 + (1 if markdown.startswith("```") else 0)
    tables = len(re.findall(r"^\|.+\|$", markdown, re.MULTILINE))
    return headings, max(0, code_blocks), tables


def extract_page(
    html: str,
    url: str,
    cfg: AutotrainConfig,
) -> tuple[str, str, str, str, str]:
    """
    Extract one page.

    Returns (markdown, title, description, lang, method).
    `method` records which strategy produced the text, which the extract
    report uses to show how often the fallback was needed.
    """
    ecfg = cfg.extractor

    soup = BeautifulSoup(html, "lxml")
    title, description, lang = _page_metadata(soup)
    _strip_elements(soup, ecfg.strip_selectors)
    cleaned_html = str(soup)

    markdown = ""
    method = "none"

    # ---- strategy 2: trafilatura --------------------------------------
    try:
        import trafilatura

        extracted = trafilatura.extract(
            cleaned_html,
            output_format="markdown",
            include_tables=ecfg.include_tables,
            include_formatting=ecfg.include_formatting,
            include_links=ecfg.include_links,
            include_images=ecfg.include_images,
            include_comments=ecfg.include_comments,
            favor_recall=ecfg.favour_recall,
            no_fallback=False,
            url=url,
        )
        if extracted and len(extracted.strip()) >= ecfg.min_extracted_chars:
            markdown = extracted
            method = "trafilatura"
    except Exception as exc:
        logger.debug(f"trafilatura failed on {url}: {type(exc).__name__}: {exc}")

    # ---- strategy 3: structural fallback --------------------------------
    if not markdown and ecfg.enable_fallback:
        try:
            root = _pick_content_root(soup)
            candidate = html_to_markdown(
                root,
                include_links=ecfg.include_links,
                include_images=ecfg.include_images,
            )
            if candidate and len(candidate.strip()) >= ecfg.min_extracted_chars:
                markdown = candidate
                method = "structural"
        except Exception as exc:
            logger.debug(f"structural fallback failed on {url}: {type(exc).__name__}: {exc}")

    if not markdown:
        return "", title, description, lang, "empty"

    markdown = normalize_text(markdown)

    # Guarantee the document opens with its title as an H1, so downstream
    # prompts always have a subject line to work with.
    if title and not markdown.lstrip().startswith("# "):
        first_line = markdown.split("\n", 1)[0].strip().lstrip("# ").strip()
        if first_line.lower() != title.lower():
            markdown = f"# {title}\n\n{markdown}"

    return markdown, title, description, lang, method


def run_extraction(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    """Extract every successfully fetched page into clean_docs/."""
    records = [
        r for r in read_jsonl(ws.fetch_index)
        if r.get("ok") and r.get("html_path")
    ]
    # A resumed crawl can append the same page twice; keep the newest.
    unique: dict[int, dict[str, Any]] = {}
    for record in records:
        unique[int(record["index"])] = record
    records = sorted(unique.values(), key=lambda r: r["index"])

    if not records:
        raise RuntimeError(
            "no fetched pages found — run the crawl phase first "
            f"(looked in {ws.fetch_index})"
        )

    ws.clean_docs.mkdir(parents=True, exist_ok=True)

    docs: list[ExtractedDoc] = []
    methods: dict[str, int] = {}
    empty: list[dict[str, str]] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("extracting", total=len(records))
        for record in records:
            index = int(record["index"])
            url = record.get("final_url") or record.get("requested_url") or ""
            html_file = ws.root / record["html_path"]

            try:
                html = html_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning(f"cannot read {html_file}: {exc}")
                progress.advance(task)
                continue

            markdown, title, description, lang, method = extract_page(html, url, cfg)
            methods[method] = methods.get(method, 0) + 1

            if not markdown:
                empty.append({"index": index, "url": url})
                progress.advance(task)
                continue

            headings, code_blocks, tables = _count_structures(markdown)
            md_path = ws.page_md(index)
            md_path.write_text(markdown, encoding="utf-8")

            docs.append(
                ExtractedDoc(
                    index=index,
                    url=url,
                    title=title or f"Page {index}",
                    markdown=markdown,
                    lang=lang,
                    description=description,
                    method=method,
                    char_count=len(markdown),
                    word_count=len(markdown.split()),
                    heading_count=headings,
                    code_block_count=code_blocks,
                    table_count=tables,
                    content_hash=sha256_text(markdown),
                    md_path=str(md_path.relative_to(ws.root)).replace("\\", "/"),
                )
            )
            progress.advance(task)

    write_jsonl((d.to_json() for d in docs), ws.clean_docs / "index.jsonl")

    total_chars = sum(d.char_count for d in docs)
    report = {
        "pages_considered": len(records),
        "documents_extracted": len(docs),
        "empty": len(empty),
        "empty_pages": empty[:200],
        "methods": methods,
        "total_chars": total_chars,
        "total_words": sum(d.word_count for d in docs),
        "avg_chars": round(total_chars / len(docs)) if docs else 0,
        "with_code_blocks": sum(1 for d in docs if d.code_block_count),
        "with_tables": sum(1 for d in docs if d.table_count),
        "total_code_blocks": sum(d.code_block_count for d in docs),
        "total_tables": sum(d.table_count for d in docs),
    }
    write_json(report, ws.extract_report)

    logger.info(
        f"extracted {len(docs):,}/{len(records):,} pages "
        f"({total_chars:,} chars) — methods: {methods}"
    )
    if empty:
        logger.info(f"{len(empty)} page(s) yielded no content and were dropped")

    return report
