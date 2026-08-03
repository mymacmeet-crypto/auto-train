"""
Text utilities: hashing, fingerprinting, JSONL IO, token estimation.

SimHash is implemented here rather than pulled from a dependency because it
is 40 lines and the corpus-scale behaviour matters: we need a fingerprint
that is stable across whitespace changes and tolerant of a swapped sentence,
which is exactly what shingled SimHash gives.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha1_short(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:length]


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def simhash(text: str, shingle: int = 3, bits: int = 64) -> int:
    """
    64-bit SimHash over word shingles.

    Word shingles (rather than single words) make the fingerprint sensitive
    to word *order*, so two pages sharing a vocabulary but saying different
    things do not collide.
    """
    words = _tokens(text)
    if not words:
        return 0
    if len(words) < shingle:
        grams = [" ".join(words)]
    else:
        grams = [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]

    vector = [0] * bits
    mask = (1 << bits) - 1
    for gram in grams:
        h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big") & mask
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1

    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_NBSP = re.compile(r"[   ]")


def normalize_text(text: str, *, collapse_blank_lines: bool = True) -> str:
    """
    Unicode-normalise and tidy whitespace without touching structure.

    Deliberately does NOT collapse runs of spaces inside a line: that would
    corrupt indentation inside fenced code blocks.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ZERO_WIDTH)
    text = _NBSP.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    if collapse_blank_lines:
        text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


_CODE_FENCE = re.compile(r"^```", re.MULTILINE)


def split_code_and_prose(markdown: str) -> tuple[list[str], list[str]]:
    """Return (code_blocks, prose_segments) for a Markdown document."""
    parts = markdown.split("```")
    code: list[str] = []
    prose: list[str] = []
    for i, part in enumerate(parts):
        (code if i % 2 == 1 else prose).append(part)
    return code, prose


def iter_lines_outside_code(markdown: str) -> Iterator[tuple[int, str, bool]]:
    """Yield (index, line, in_code) so cleaners can skip fenced regions."""
    in_code = False
    for i, line in enumerate(markdown.split("\n")):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            yield i, line, True
            continue
        yield i, line, in_code


def estimate_tokens(text: str) -> int:
    """~4 chars per token; good enough for budgeting, never for truncation."""
    return max(1, len(text) // 4)


_CORRUPT_RE = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")


def corruption_ratio(text: str) -> float:
    """Fraction of characters that are replacement or control characters."""
    if not text:
        return 0.0
    return len(_CORRUPT_RE.findall(text)) / len(text)


_CONTENT_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those is are was were be been being
    to of in on at by for with from as it its into about over under after before between
    you your we our they their he she his her i me my do does did done have has had can
    could should would will shall may might must not no yes so such very more most other
    some any each which who whom what when where why how all both few many much own same
    only just also here there""".split()
)


def content_words(text: str) -> set[str]:
    """Lowercased non-stopword tokens, used for grounding/overlap scoring."""
    return {w.lower() for w in _CONTENT_WORD.findall(text)} - _STOPWORDS


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl_strict(path: Path) -> Iterator[tuple[int, str, dict[str, Any] | None]]:
    """Yield (lineno, raw, parsed_or_None) so the validator can report bad lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield lineno, raw, json.loads(raw)
            except json.JSONDecodeError:
                yield lineno, raw, None


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())
