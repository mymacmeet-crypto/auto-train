"""
Phase 4 — corpus-level dataset cleaning.

Phase 3 works one page at a time and cannot see patterns. Phase 4 sees the
whole corpus at once, which is what makes three things possible:

  - Near-duplicate detection. Documentation sites publish the same page under
    /latest/, /v2.1/, and /stable/. Byte hashing misses these because the
    version string differs; SimHash catches them.
  - Boilerplate line removal. "Edit this page on GitHub", "Was this helpful?",
    breadcrumb trails and copyright lines survive per-page extraction because
    in isolation they look like content. Across 900 pages they are obviously
    chrome: they appear verbatim on most of them.
  - Menu residue removal. Navigation that extraction missed shows up as a
    long run of short link-like lines, usually at the top or bottom.

Everything here is conservative by design. Over-cleaning silently deletes
training signal, and unlike under-cleaning it leaves no trace in the output.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import (
    hamming,
    normalize_text,
    read_jsonl,
    sha256_text,
    simhash,
    write_json,
    write_jsonl,
)

logger = get_logger("cleaner")


@dataclass
class CleanDoc:
    index: int
    url: str
    title: str
    markdown: str
    char_count: int = 0
    word_count: int = 0
    heading_count: int = 0
    code_block_count: int = 0
    table_count: int = 0
    content_hash: str = ""
    simhash: str = ""
    md_path: str = ""
    lang: str = ""

    def to_record(self) -> dict[str, Any]:
        """The corpus.jsonl record consumed by Phase 5."""
        return {
            "index": self.index,
            "url": self.url,
            "title": self.title,
            "text": self.markdown,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "heading_count": self.heading_count,
            "code_block_count": self.code_block_count,
            "table_count": self.table_count,
            "content_hash": self.content_hash,
            "lang": self.lang,
            "md_path": self.md_path,
        }


@dataclass
class CleanStats:
    input_docs: int = 0
    output_docs: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_exact_dupe: int = 0
    dropped_near_dupe: int = 0
    boilerplate_lines_found: int = 0
    boilerplate_lines_removed: int = 0
    nav_blocks_removed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    boilerplate_samples: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Code-block-aware line iteration
# ---------------------------------------------------------------------------


def _split_fenced(markdown: str) -> list[tuple[str, bool]]:
    """
    Split a document into (segment, is_code) pieces.

    Every cleaning rule below runs only on is_code=False segments, so a
    fenced block is never rewritten — a stray "Copyright 2024" inside a
    licence-header code sample stays exactly as the author wrote it.
    """
    segments: list[tuple[str, bool]] = []
    buffer: list[str] = []
    in_code = False
    fence = ""

    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if not in_code and stripped.startswith("```"):
            if buffer:
                segments.append(("\n".join(buffer), False))
                buffer = []
            fence = stripped[: len(stripped) - len(stripped.lstrip("`"))]
            in_code = True
            buffer.append(line)
        elif in_code and stripped.startswith(fence) and stripped.rstrip() == fence:
            buffer.append(line)
            segments.append(("\n".join(buffer), True))
            buffer = []
            in_code = False
            fence = ""
        else:
            buffer.append(line)

    if buffer:
        segments.append(("\n".join(buffer), in_code))
    return segments


def _prose_lines(markdown: str) -> list[str]:
    """Non-empty lines outside fenced code, for frequency analysis."""
    out: list[str] = []
    for segment, is_code in _split_fenced(markdown):
        if is_code:
            continue
        for line in segment.split("\n"):
            stripped = line.strip()
            if stripped:
                out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------


def find_boilerplate_lines(
    documents: list[str],
    *,
    ratio_threshold: float = 0.35,
    min_docs: int = 8,
    max_line_chars: int = 200,
) -> set[str]:
    """
    Lines that appear verbatim on a large fraction of pages are site chrome.

    Guards against over-removal:
      - only runs when the corpus is big enough for the ratio to mean anything
      - never removes a long line (long repeated text is shared documentation,
        such as a licence notice or a standard warning, and is real content)
      - never removes a Markdown heading, list item, or table row: those carry
        structure even when repeated
    """
    if len(documents) < min_docs:
        return set()

    counts: Counter[str] = Counter()
    for doc in documents:
        # Count each distinct line once per document, so a page repeating a
        # phrase ten times does not inflate the corpus-wide ratio.
        counts.update(set(_prose_lines(doc)))

    threshold = max(min_docs, int(len(documents) * ratio_threshold))
    boilerplate: set[str] = set()

    for line, count in counts.items():
        if count < threshold:
            continue
        if len(line) > max_line_chars:
            continue
        # Structural Markdown is never chrome.
        if line.startswith(("#", "|", ">", "```")):
            continue
        if re.match(r"^\s*(?:[-*+]|\d+\.)\s", line) and len(line) > 60:
            continue
        boilerplate.add(line)

    return boilerplate


_NAV_LINE = re.compile(r"^\s*(?:[-*+]\s+)?[A-Z0-9][\w &/'.,()-]{0,48}$")


def strip_nav_runs(markdown: str, *, min_run: int = 6) -> tuple[str, int]:
    """
    Remove long runs of short, title-cased, link-like lines.

    A sidebar that survived extraction looks like this:

        - Getting Started
        - Installation
        - Configuration
        - API Reference

    Six or more consecutive lines of that shape, with no sentence punctuation
    among them, is a menu. Real content lists have longer items and contain
    prose. The run must also not contain a heading.
    """
    segments = _split_fenced(markdown)
    removed = 0
    out_segments: list[str] = []

    for segment, is_code in segments:
        if is_code:
            out_segments.append(segment)
            continue

        lines = segment.split("\n")
        keep: list[str] = []
        run: list[int] = []

        def flush(run_indices: list[int]) -> None:
            nonlocal removed
            if len(run_indices) >= min_run:
                block = [lines[i] for i in run_indices]
                # A menu has almost no sentence punctuation.
                punctuation = sum(1 for line in block if re.search(r"[.:;?!]\s*$", line.strip()))
                if punctuation <= len(block) // 4:
                    removed += len(block)
                    return
            keep.extend(lines[i] for i in run_indices)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and _NAV_LINE.match(stripped) and len(stripped) <= 50:
                run.append(i)
                continue
            flush(run)
            run = []
            keep.append(line)

        flush(run)
        out_segments.append("\n".join(keep))

    return "\n".join(out_segments), removed


def remove_lines(markdown: str, blacklist: set[str]) -> tuple[str, int]:
    """Delete blacklisted lines outside code fences."""
    if not blacklist:
        return markdown, 0

    removed = 0
    out_segments: list[str] = []
    for segment, is_code in _split_fenced(markdown):
        if is_code:
            out_segments.append(segment)
            continue
        kept: list[str] = []
        for line in segment.split("\n"):
            if line.strip() in blacklist:
                removed += 1
                continue
            kept.append(line)
        out_segments.append("\n".join(kept))
    return "\n".join(out_segments), removed


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class NearDuplicateIndex:
    """
    SimHash index with banding, so dedupe stays linear rather than O(n^2).

    A 64-bit fingerprint is split into 4 bands of 16 bits. Two fingerprints
    within Hamming distance 3 must share at least one identical band (pigeon-
    hole principle: 4 bands, at most 3 differing bits, so one band is clean).
    Candidates are looked up by band, then verified exactly.
    """

    BANDS = 4
    BAND_BITS = 16

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self.BANDS)]
        self._fingerprints: dict[int, int] = {}

    def _bands(self, fingerprint: int) -> list[int]:
        mask = (1 << self.BAND_BITS) - 1
        return [(fingerprint >> (i * self.BAND_BITS)) & mask for i in range(self.BANDS)]

    def find(self, fingerprint: int) -> int | None:
        seen: set[int] = set()
        for band_index, band_value in enumerate(self._bands(fingerprint)):
            for candidate in self._buckets[band_index].get(band_value, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if hamming(fingerprint, self._fingerprints[candidate]) <= self.threshold:
                    return candidate
        return None

    def add(self, doc_id: int, fingerprint: int) -> None:
        self._fingerprints[doc_id] = fingerprint
        for band_index, band_value in enumerate(self._bands(fingerprint)):
            self._buckets[band_index].setdefault(band_value, []).append(doc_id)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_cleaning(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    ccfg = cfg.cleaner
    stats = CleanStats()

    index_file = ws.clean_docs / "index.jsonl"
    entries = list(read_jsonl(index_file))
    if not entries:
        raise RuntimeError(
            f"no extracted documents found at {index_file} — run the extract phase first"
        )

    # ---- load ----------------------------------------------------------
    loaded: list[CleanDoc] = []
    for entry in entries:
        md_file = ws.root / entry["md_path"]
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        loaded.append(
            CleanDoc(
                index=int(entry["index"]),
                url=entry.get("url", ""),
                title=entry.get("title", ""),
                markdown=text,
                lang=entry.get("lang", ""),
            )
        )

    stats.input_docs = len(loaded)
    stats.chars_before = sum(len(d.markdown) for d in loaded)
    logger.info(f"cleaning {len(loaded):,} extracted documents")

    # ---- corpus-wide boilerplate ---------------------------------------
    boilerplate: set[str] = set()
    if len(loaded) >= ccfg.boilerplate_min_docs:
        boilerplate = find_boilerplate_lines(
            [d.markdown for d in loaded],
            ratio_threshold=ccfg.boilerplate_line_ratio,
            min_docs=ccfg.boilerplate_min_docs,
            max_line_chars=ccfg.boilerplate_max_line_chars,
        )
        stats.boilerplate_lines_found = len(boilerplate)
        if boilerplate:
            sample = sorted(boilerplate, key=len, reverse=True)[:40]
            stats.boilerplate_samples = [{"line": s} for s in sample]
            logger.info(f"identified {len(boilerplate)} corpus-wide boilerplate line(s)")

    # ---- per-document cleaning + dedupe --------------------------------
    exact_seen: dict[str, int] = {}
    near_index = NearDuplicateIndex(ccfg.simhash_threshold)
    output: list[CleanDoc] = []

    progress = make_progress()
    with progress:
        task = progress.add_task("cleaning", total=len(loaded))

        for doc in loaded:
            text = doc.markdown

            if boilerplate:
                text, removed = remove_lines(text, boilerplate)
                stats.boilerplate_lines_removed += removed

            if ccfg.strip_trailing_nav_lists:
                text, nav_removed = strip_nav_runs(text)
                stats.nav_blocks_removed += nav_removed

            if ccfg.normalize_whitespace:
                text = normalize_text(text)

            if not text.strip():
                stats.dropped_empty += 1
                progress.advance(task)
                continue

            word_count = len(text.split())
            if len(text) < ccfg.min_doc_chars or word_count < ccfg.min_doc_words:
                stats.dropped_short += 1
                progress.advance(task)
                continue

            content_hash = sha256_text(text)
            if content_hash in exact_seen:
                stats.dropped_exact_dupe += 1
                stats.duplicate_groups.append(
                    {"kind": "exact", "dropped": doc.index, "kept": exact_seen[content_hash],
                     "url": doc.url}
                )
                progress.advance(task)
                continue
            exact_seen[content_hash] = doc.index

            if ccfg.enable_near_dedupe:
                fingerprint = simhash(text)
                match = near_index.find(fingerprint)
                if match is not None:
                    stats.dropped_near_dupe += 1
                    stats.duplicate_groups.append(
                        {"kind": "near", "dropped": doc.index, "kept": match, "url": doc.url}
                    )
                    progress.advance(task)
                    continue
                near_index.add(doc.index, fingerprint)
                doc.simhash = f"{fingerprint:016x}"

            headings = len(re.findall(r"^#{1,6} ", text, re.MULTILINE))
            code_blocks = text.count("\n```") // 2
            tables = len(re.findall(r"^\|.+\|$", text, re.MULTILINE))

            doc.markdown = text
            doc.char_count = len(text)
            doc.word_count = word_count
            doc.heading_count = headings
            doc.code_block_count = max(0, code_blocks)
            doc.table_count = tables
            doc.content_hash = content_hash

            from autotrain.utils.paths import url_to_slug
            slug = url_to_slug(doc.url)
            out_path = ws.clean_dataset_docs / f"{slug}.md"
            if out_path.exists():
                out_path = ws.clean_dataset_docs / f"{slug}__{doc.index}.md"
            out_path.write_text(text, encoding="utf-8")
            doc.md_path = str(out_path.relative_to(ws.root)).replace("\\", "/")

            output.append(doc)
            progress.advance(task)

    stats.output_docs = len(output)
    stats.chars_after = sum(d.char_count for d in output)

    write_jsonl((d.to_record() for d in output), ws.corpus_jsonl)

    report = {
        "input_documents": stats.input_docs,
        "output_documents": stats.output_docs,
        "dropped": {
            "empty": stats.dropped_empty,
            "too_short": stats.dropped_short,
            "exact_duplicate": stats.dropped_exact_dupe,
            "near_duplicate": stats.dropped_near_dupe,
        },
        "boilerplate": {
            "lines_identified": stats.boilerplate_lines_found,
            "line_instances_removed": stats.boilerplate_lines_removed,
            "samples": stats.boilerplate_samples,
        },
        "nav_lines_removed": stats.nav_blocks_removed,
        "chars_before": stats.chars_before,
        "chars_after": stats.chars_after,
        "chars_removed": stats.chars_before - stats.chars_after,
        "reduction_pct": (
            round(100 * (stats.chars_before - stats.chars_after) / stats.chars_before, 2)
            if stats.chars_before else 0.0
        ),
        "corpus": {
            "total_words": sum(d.word_count for d in output),
            "avg_chars": round(stats.chars_after / len(output)) if output else 0,
            "with_code": sum(1 for d in output if d.code_block_count),
            "with_tables": sum(1 for d in output if d.table_count),
            "total_headings": sum(d.heading_count for d in output),
        },
        "duplicate_groups": stats.duplicate_groups[:500],
    }
    write_json(report, ws.clean_report)

    logger.info(
        f"cleaned corpus: {stats.output_docs:,} documents "
        f"({stats.chars_after:,} chars, -{report['reduction_pct']}%), "
        f"dropped {stats.dropped_exact_dupe + stats.dropped_near_dupe:,} duplicates "
        f"and {stats.dropped_short + stats.dropped_empty:,} thin pages"
    )

    if stats.output_docs == 0:
        raise RuntimeError(
            "cleaning removed every document — the thresholds are too aggressive "
            "for this site (try lowering cleaner.min_doc_chars)"
        )

    return report
