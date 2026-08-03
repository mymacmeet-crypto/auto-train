"""
HTML -> Markdown conversion.

trafilatura handles the main path, but it optimises for prose and will
occasionally flatten a structure that matters on a documentation site
(definition lists, nested API parameter tables, multi-language code tabs).
This converter is the fallback, and it is deliberately structure-first:
every element type listed in the Phase 3 "keep" set has an explicit rule.

Correctness priorities, in order:
  1. Code blocks survive byte-exact, including indentation and language tag.
  2. Tables stay tables (GFM pipe syntax).
  3. Heading levels are preserved so the document keeps its outline.
  4. Lists keep their nesting depth.
Everything else is allowed to degrade to plain paragraphs.
"""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

# Elements that never contribute content.
_DROP_TAGS = frozenset(
    {
        "script", "style", "noscript", "template", "svg", "canvas", "math",
        "iframe", "object", "embed", "applet", "map", "area",
        "form", "input", "select", "option", "textarea", "button", "label",
        "video", "audio", "source", "track", "picture",
    }
)

_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

_INLINE_WS = re.compile(r"[ \t ]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _text_of(node: Tag | NavigableString) -> str:
    """Inline text with runs of whitespace collapsed (safe outside <pre>)."""
    if isinstance(node, NavigableString):
        return _INLINE_WS.sub(" ", str(node))
    return _INLINE_WS.sub(" ", node.get_text(" "))


def _escape_inline(text: str) -> str:
    """Escape Markdown control characters that would change rendering."""
    # Only the characters that actually start a construct at line level, and
    # only when they lead the line, are dangerous; inline escaping of every
    # asterisk makes prose unreadable. Keep this minimal.
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _language_of(element: Tag) -> str:
    """Recover the code language from the conventional class names."""
    candidates: list[str] = []
    for node in (element, element.find("code")):
        if not isinstance(node, Tag):
            continue
        classes = node.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        candidates.extend(classes)
        for attr in ("data-lang", "data-language", "lang"):
            value = node.get(attr)
            if value:
                candidates.append(str(value))

    for raw in candidates:
        value = raw.strip().lower()
        for prefix in ("language-", "lang-", "highlight-", "brush:", "syntax-"):
            if value.startswith(prefix):
                return value[len(prefix) :].strip(": ")
        if value in {
            "python", "py", "javascript", "js", "typescript", "ts", "java",
            "json", "yaml", "yml", "xml", "html", "css", "scss", "bash", "sh",
            "shell", "zsh", "sql", "go", "rust", "rb", "ruby", "php", "c",
            "cpp", "csharp", "cs", "swift", "kotlin", "scala", "r", "matlab",
            "dockerfile", "makefile", "toml", "ini", "diff", "graphql", "apex",
        }:
            return value
    return ""


class MarkdownConverter:
    """Recursive-descent HTML -> Markdown."""

    def __init__(self, *, include_links: bool = False, include_images: bool = False) -> None:
        self.include_links = include_links
        self.include_images = include_images

    # -- public ------------------------------------------------------------

    def convert(self, soup: Tag) -> str:
        blocks = list(self._blocks(soup, depth=0))
        text = "\n\n".join(b for b in blocks if b.strip())
        return _MULTI_NL.sub("\n\n", text).strip()

    # -- block level -------------------------------------------------------

    def _blocks(self, node: Tag, depth: int) -> Iterable[str]:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = _text_of(child).strip()
                if text:
                    yield _escape_inline(text)
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()
            if name in _DROP_TAGS:
                continue

            if name in _HEADINGS:
                level = _HEADINGS[name]
                text = self._inline(child).strip()
                if text:
                    yield f"{'#' * level} {text}"

            elif name == "p":
                text = self._inline(child).strip()
                if text:
                    yield text

            elif name == "pre":
                yield self._code_block(child)

            elif name in ("ul", "ol"):
                yield self._list(child, ordered=(name == "ol"), depth=0)

            elif name == "dl":
                yield self._definition_list(child)

            elif name == "table":
                yield self._table(child)

            elif name == "blockquote":
                inner = "\n\n".join(self._blocks(child, depth + 1))
                if inner.strip():
                    yield "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))

            elif name in ("hr",):
                yield "---"

            elif name == "br":
                continue

            elif name in ("figure", "figcaption"):
                text = self._inline(child).strip()
                if text:
                    yield text

            elif name == "img":
                if self.include_images:
                    alt = (child.get("alt") or "").strip()
                    src = (child.get("src") or "").strip()
                    if src:
                        yield f"![{alt}]({src})"

            elif name in ("div", "section", "article", "main", "aside", "details",
                          "summary", "span", "body", "html", "header", "footer",
                          "nav", "li", "td", "th", "tr", "tbody", "thead", "center"):
                # Structural wrapper: recurse. A wrapper that contains only
                # inline content is emitted as a paragraph so text is not lost.
                if self._has_block_children(child):
                    yield from self._blocks(child, depth + 1)
                else:
                    text = self._inline(child).strip()
                    if text:
                        yield text

            else:
                text = self._inline(child).strip()
                if text:
                    yield text

    # Anything that must be recursed into rather than flattened to inline
    # text. Must include the document wrappers: BeautifulSoup's lxml parser
    # inserts <html><body> around every fragment, so omitting them collapses
    # an entire document into one run of unstructured text.
    _BLOCK_NAMES = set(_HEADINGS) | {
        "p", "pre", "ul", "ol", "dl", "table", "blockquote", "hr",
        "div", "section", "article", "main", "aside", "details", "figure",
        "html", "body", "header", "footer", "nav", "li", "tbody", "thead",
        "tr", "td", "th", "center", "form", "fieldset",
    }

    @classmethod
    def _has_block_children(cls, node: Tag) -> bool:
        return any(
            isinstance(c, Tag) and c.name and c.name.lower() in cls._BLOCK_NAMES
            for c in node.children
        )

    # -- specific structures -----------------------------------------------

    def _code_block(self, element: Tag) -> str:
        """
        Fenced code block. Uses get_text() with no separator so that the
        original whitespace inside <pre> is preserved exactly — this is the
        one place where whitespace collapsing would be destructive.
        """
        code_el = element.find("code")
        source = code_el if isinstance(code_el, Tag) else element
        text = source.get_text()
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        if not text.strip():
            return ""
        lang = _language_of(element)
        # Use a longer fence when the code itself contains a triple backtick.
        fence = "```"
        while fence in text:
            fence += "`"
        return f"{fence}{lang}\n{text}\n{fence}"

    def _list(self, element: Tag, *, ordered: bool, depth: int) -> str:
        lines: list[str] = []
        indent = "  " * depth
        counter = 1

        try:
            start = int(element.get("start", 1))
        except (TypeError, ValueError):
            start = 1
        counter = start

        for item in element.find_all("li", recursive=False):
            marker = f"{counter}." if ordered else "-"
            # Nested lists are rendered separately and appended below the item.
            nested_out: list[str] = []
            for nested in item.find_all(("ul", "ol"), recursive=False):
                nested_out.append(
                    self._list(nested, ordered=(nested.name.lower() == "ol"), depth=depth + 1)
                )
                nested.extract()

            # A list item containing block content (code, tables) keeps it,
            # indented to stay inside the item.
            if self._has_block_children(item):
                inner_blocks = [b for b in self._blocks(item, depth + 1) if b.strip()]
                if inner_blocks:
                    head = inner_blocks[0]
                    lines.append(f"{indent}{marker} {head}")
                    for block in inner_blocks[1:]:
                        padded = "\n".join(
                            f"{indent}  {line}" for line in block.split("\n")
                        )
                        lines.append(padded)
            else:
                text = self._inline(item).strip()
                if text:
                    lines.append(f"{indent}{marker} {text}")

            for nested_md in nested_out:
                if nested_md.strip():
                    lines.append(nested_md)

            counter += 1

        return "\n".join(lines)

    def _definition_list(self, element: Tag) -> str:
        """<dl> is the canonical markup for API parameter documentation."""
        lines: list[str] = []
        for child in element.children:
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name == "dt":
                term = self._inline(child).strip()
                if term:
                    lines.append(f"**{term}**")
            elif name == "dd":
                definition = self._inline(child).strip()
                if definition:
                    lines.append(f": {definition}")
        return "\n".join(lines)

    def _table(self, element: Tag) -> str:
        """GFM pipe table. Falls back to plain text if the shape is irregular."""
        rows: list[list[str]] = []
        for tr in element.find_all("tr"):
            cells = tr.find_all(("td", "th"), recursive=False)
            if not cells:
                continue
            rows.append([self._inline(c).strip().replace("\n", " ") or " " for c in cells])

        if not rows:
            return ""

        width = max(len(r) for r in rows)
        if width == 0:
            return ""
        # Pad ragged rows so the pipe table stays well-formed.
        rows = [r + [" "] * (width - len(r)) for r in rows]

        header_present = bool(element.find("th"))
        header = rows[0] if header_present else [" "] * width
        body = rows[1:] if header_present else rows

        out = ["| " + " | ".join(header) + " |",
               "| " + " | ".join("---" for _ in range(width)) + " |"]
        out.extend("| " + " | ".join(r) + " |" for r in body)
        return "\n".join(out)

    # -- inline level ------------------------------------------------------

    def _inline(self, node: Tag) -> str:
        parts: list[str] = []
        for child in node.children:
            if isinstance(child, NavigableString):
                parts.append(_escape_inline(_text_of(child)))
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name.lower()
            if name in _DROP_TAGS:
                continue

            if name in ("code", "tt", "kbd", "samp"):
                text = child.get_text().strip()
                if text:
                    tick = "`"
                    while tick in text:
                        tick += "`"
                    parts.append(f"{tick}{text}{tick}")
            elif name in ("strong", "b"):
                text = self._inline(child).strip()
                if text:
                    parts.append(f"**{text}**")
            elif name in ("em", "i"):
                text = self._inline(child).strip()
                if text:
                    parts.append(f"*{text}*")
            elif name in ("del", "s", "strike"):
                text = self._inline(child).strip()
                if text:
                    parts.append(f"~~{text}~~")
            elif name == "br":
                parts.append("\n")
            elif name == "a":
                text = self._inline(child).strip()
                href = (child.get("href") or "").strip()
                if self.include_links and href and text and not href.startswith("#"):
                    parts.append(f"[{text}]({href})")
                elif text:
                    parts.append(text)
            elif name == "img":
                if self.include_images:
                    alt = (child.get("alt") or "").strip()
                    src = (child.get("src") or "").strip()
                    if src:
                        parts.append(f"![{alt}]({src})")
                else:
                    alt = (child.get("alt") or "").strip()
                    if alt:
                        parts.append(alt)
            else:
                parts.append(self._inline(child))

        return _INLINE_WS.sub(" ", "".join(parts)).strip()


def html_to_markdown(
    html: str | Tag,
    *,
    include_links: bool = False,
    include_images: bool = False,
) -> str:
    soup = html if isinstance(html, Tag) else BeautifulSoup(html, "lxml")
    return MarkdownConverter(
        include_links=include_links, include_images=include_images
    ).convert(soup)
