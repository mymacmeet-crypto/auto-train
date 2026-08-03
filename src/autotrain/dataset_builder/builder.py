"""
Phase 5 — instruction dataset generation.

Reads the cleaned corpus and emits dataset.jsonl: thousands of
instruction/response pairs across the eight sample types.

Three things make this scale to a 10,000-page site:

  Windowing. A long API reference is split into overlapping windows on
  heading boundaries, so a 40-page document yields proportionally more
  samples than a one-paragraph page instead of being truncated to its first
  6,000 characters.

  Applicability gating. Every (window, sample_type) pair is checked against
  the window's structure first. Asking for a code explanation from prose
  produces a hallucination; skipping it produces nothing. Nothing is better.

  Resume. Completed (document, window, type) triples are recorded in
  generation_state.json and their pairs are already appended to dataset.jsonl.
  An interrupted run re-enters exactly where it stopped, which matters when
  generation against a local 7B takes hours.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from autotrain.dataset_builder.prompts import (
    SPECS,
    DocSignals,
    analyse,
    applicable_types,
)
from autotrain.dataset_builder.providers import (
    Pair,
    ResolvedProvider,
    SampleGenerator,
    resolve_provider,
)
from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import (
    append_jsonl,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
)

logger = get_logger("dataset")


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


@dataclass
class Window:
    doc_index: int
    window_index: int
    url: str
    title: str
    text: str
    signals: DocSignals

    @property
    def key(self) -> str:
        return f"{self.doc_index}:{self.window_index}"


_HEADING_LINE = re.compile(r"^#{1,6}\s+\S")


def split_into_windows(
    text: str, *, max_chars: int, overlap: int, max_windows: int
) -> list[str]:
    """
    Split on heading boundaries, never inside a fenced code block.

    Heading-aligned windows keep a section's explanation together with its
    code sample, which is what makes the generated pairs coherent. Falling
    back to a hard character split only happens for documents with no
    headings at all.
    """
    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    # Identify safe split points: heading lines outside code fences.
    in_code = False
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and _HEADING_LINE.match(line):
            boundaries.append(i)

    windows: list[str] = []

    if len(boundaries) >= 2:
        start_line = 0
        current: list[str] = []
        current_len = 0
        boundary_set = set(boundaries)

        for i, line in enumerate(lines):
            # Start a new window at a heading once we are over budget.
            if i in boundary_set and current_len >= max_chars and current:
                windows.append("\n".join(current))
                if len(windows) >= max_windows:
                    return windows
                # Carry the tail back as overlap for continuity.
                tail = "\n".join(current)[-overlap:] if overlap else ""
                current = ([tail] if tail else []) + [line]
                current_len = len(tail) + len(line)
                continue
            current.append(line)
            current_len += len(line) + 1

        if current and "\n".join(current).strip():
            windows.append("\n".join(current))
        if windows:
            return windows[:max_windows]

    # No usable headings: hard split with overlap.
    step = max(1, max_chars - overlap)
    for start in range(0, len(text), step):
        windows.append(text[start : start + max_chars])
        if len(windows) >= max_windows:
            break
    return windows


def build_windows(cfg: AutotrainConfig, records: list[dict[str, Any]]) -> list[Window]:
    dcfg = cfg.dataset
    windows: list[Window] = []
    for record in records:
        text = record.get("text", "")
        if not text.strip():
            continue
        chunks = split_into_windows(
            text,
            max_chars=dcfg.max_chars_per_window,
            overlap=dcfg.window_overlap_chars,
            max_windows=dcfg.max_windows_per_doc,
        )
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 200:
                continue
            windows.append(
                Window(
                    doc_index=int(record["index"]),
                    window_index=i,
                    url=record.get("url", ""),
                    title=record.get("title", ""),
                    text=chunk,
                    signals=analyse(chunk),
                )
            )
    return windows


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------


@dataclass
class GenerationState:
    completed: dict[str, list[str]] = field(default_factory=dict)
    total_pairs: int = 0
    backend: str = ""
    corpus_digest: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)
    failures: int = 0

    def is_done(self, window_key: str, sample_type: str) -> bool:
        return sample_type in self.completed.get(window_key, ())

    def mark(self, window_key: str, sample_type: str) -> None:
        self.completed.setdefault(window_key, [])
        if sample_type not in self.completed[window_key]:
            self.completed[window_key].append(sample_type)


def _load_state(ws: Workspace, corpus_digest: str) -> GenerationState:
    raw = read_json(ws.generation_state, default=None)
    if not raw:
        return GenerationState(corpus_digest=corpus_digest, started_at=time.time())
    try:
        state = GenerationState(**raw)
    except TypeError:
        return GenerationState(corpus_digest=corpus_digest, started_at=time.time())

    if state.corpus_digest and state.corpus_digest != corpus_digest:
        # The corpus changed under us: previously generated pairs no longer
        # correspond to the current documents, so start clean.
        logger.warning("corpus changed since the last run — regenerating the dataset")
        ws.dataset_jsonl.unlink(missing_ok=True)
        return GenerationState(corpus_digest=corpus_digest, started_at=time.time())
    return state


# ---------------------------------------------------------------------------
# Record shaping
# ---------------------------------------------------------------------------


def _site_name(cfg: AutotrainConfig) -> str:
    domain = cfg.domain if "://" in cfg.domain else f"https://{cfg.domain}"
    host = urlsplit(domain).hostname or cfg.domain
    return host.removeprefix("www.")


def pair_to_record(pair: Pair, system_prompt: str) -> dict[str, Any]:
    """
    The training record.

    `messages` is what TRL consumes. `instruction`/`output` are kept alongside
    it so the file is also usable by any Alpaca-style loader without a
    conversion step, and so the validator can check both shapes.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair.instruction},
            {"role": "assistant", "content": pair.output},
        ],
        "instruction": pair.instruction,
        "output": pair.output,
        "meta": {
            "sample_type": pair.sample_type,
            "source_index": pair.source_index,
            "source_url": pair.source_url,
            "source_title": pair.source_title,
            "window": pair.window,
            "generator": pair.generator,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _generate_async(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    dcfg = cfg.dataset
    records = list(read_jsonl(ws.corpus_jsonl))
    if not records:
        raise RuntimeError(
            f"cleaned corpus is empty at {ws.corpus_jsonl} — run the clean phase first"
        )

    corpus_digest = sha256_text(
        "".join(f"{r.get('index')}:{r.get('content_hash', '')}" for r in records)
    )[:16]

    windows = build_windows(cfg, records)
    logger.info(
        f"{len(records):,} documents -> {len(windows):,} windows "
        f"(<= {dcfg.max_chars_per_window:,} chars each)"
    )

    provider: ResolvedProvider = await resolve_provider(dcfg.provider)
    generator: SampleGenerator = provider.generator

    state = _load_state(ws, corpus_digest)
    state.backend = provider.backend_name
    if not state.started_at:
        state.started_at = time.time()

    system_prompt = dcfg.system_prompt.format(site_name=_site_name(cfg))

    # Build the full work list, skipping anything already done.
    tasks: list[tuple[Window, str]] = []
    skipped_inapplicable = 0
    for window in windows:
        specs = applicable_types(window.signals, dcfg.sample_types)
        applicable_keys = {s.key for s in specs}
        skipped_inapplicable += len(dcfg.sample_types) - len(applicable_keys)
        for spec in specs:
            if state.is_done(window.key, spec.key):
                continue
            tasks.append((window, spec.key))

    already = state.total_pairs
    if already:
        logger.info(f"resuming generation: {already:,} pairs already written")
    if not tasks:
        logger.info("nothing left to generate")
        return _finalise(ws, state, records, windows, skipped_inapplicable, provider)

    logger.info(
        f"{len(tasks):,} generation calls queued "
        f"({skipped_inapplicable:,} skipped as inapplicable)"
    )

    semaphore = asyncio.Semaphore(max(1, dcfg.provider.concurrency))
    write_lock = asyncio.Lock()
    site_name = _site_name(cfg)

    client_ctx = (
        httpx.AsyncClient(timeout=dcfg.provider.timeout)
        if provider.needs_client
        else _NullClientCtx()
    )

    progress = make_progress()

    async with client_ctx as client:
        with progress:
            bar = progress.add_task("generating", total=len(tasks))

            async def worker(window: Window, type_key: str) -> None:
                spec = SPECS[type_key]
                async with semaphore:
                    pairs = await generator.generate(
                        client,
                        spec,
                        site_name=site_name,
                        title=window.title,
                        url=window.url,
                        content=window.text,
                        signals=window.signals,
                        n=dcfg.pairs_per_type,
                        source_index=window.doc_index,
                        window=window.window_index,
                    )

                valid = [
                    p for p in pairs
                    if p.is_valid(dcfg.min_answer_chars, dcfg.max_answer_chars)
                ]

                async with write_lock:
                    if valid:
                        append_jsonl(
                            (pair_to_record(p, system_prompt) for p in valid),
                            ws.dataset_jsonl,
                        )
                        state.total_pairs += len(valid)
                        state.by_type[type_key] = state.by_type.get(type_key, 0) + len(valid)
                    else:
                        state.failures += 1
                    state.mark(window.key, type_key)
                    state.updated_at = time.time()
                    # Checkpoint periodically rather than every call: the
                    # state file is small but rewriting it 20,000 times is
                    # measurable overhead.
                    if (state.total_pairs + state.failures) % 25 == 0:
                        write_json(asdict(state), ws.generation_state)

                progress.advance(bar)

            batch: list[asyncio.Task[None]] = []
            for window, type_key in tasks:
                batch.append(asyncio.create_task(worker(window, type_key)))
                # Bound the number of pending tasks so a 10k-page site does
                # not allocate 200k coroutines up front.
                if len(batch) >= dcfg.provider.concurrency * 8:
                    await asyncio.gather(*batch, return_exceptions=True)
                    batch = []
            if batch:
                await asyncio.gather(*batch, return_exceptions=True)

    write_json(asdict(state), ws.generation_state)
    return _finalise(ws, state, records, windows, skipped_inapplicable, provider)


class _NullClientCtx:
    """Stand-in async context manager for generators that need no HTTP."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _finalise(
    ws: Workspace,
    state: GenerationState,
    records: list[dict[str, Any]],
    windows: list[Window],
    skipped_inapplicable: int,
    provider: ResolvedProvider,
) -> dict[str, Any]:
    write_json(asdict(state), ws.generation_state)

    total = state.total_pairs
    report = {
        "backend": state.backend,
        "documents": len(records),
        "windows": len(windows),
        "total_pairs": total,
        "by_type": dict(sorted(state.by_type.items(), key=lambda kv: -kv[1])),
        "failed_calls": state.failures,
        "skipped_inapplicable": skipped_inapplicable,
        "pairs_per_document": round(total / len(records), 2) if records else 0.0,
        "duration_seconds": round(time.time() - state.started_at, 2) if state.started_at else 0.0,
        "output": str(ws.dataset_jsonl),
    }

    logger.info(
        f"generated {total:,} instruction pairs via {state.backend} "
        f"across {len(state.by_type)} sample type(s)"
    )
    for type_key, count in report["by_type"].items():
        logger.info(f"   {type_key:<20} {count:>7,}")

    if total == 0:
        raise RuntimeError(
            "no instruction pairs were generated. Check that the corpus has "
            "content and that the provider backend is reachable."
        )
    return report


def run_generation(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    return asyncio.run(_generate_async(cfg, ws))
