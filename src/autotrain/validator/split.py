"""
Phase 7 — train/validation splitting.

The important decision here is grouping. Phase 5 generates several samples
from each source document, and those samples share vocabulary, facts, and
often whole sentences. Splitting them randomly puts near-copies of a training
sample into the validation set, and validation loss then measures memorisation
rather than generalisation — it drops smoothly and tells you nothing.

Grouping by source document fixes it: every sample derived from a page lands
entirely on one side of the split. Validation loss becomes a real signal about
unseen pages.

The split is deterministic given a seed: hashing the group key rather than
shuffling means adding new documents to the corpus does not reshuffle the
existing assignment, so validation loss stays comparable across runs.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger
from autotrain.utils.paths import Workspace
from autotrain.utils.text import read_jsonl, write_json, write_jsonl

logger = get_logger("splitter")


def _group_key(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    source = meta.get("source_index")
    if source is not None:
        return f"doc:{source}"
    url = meta.get("source_url")
    if url:
        return f"url:{url}"
    # Ungrouped records get their own group, which degrades to a random split
    # for those records only.
    instruction = next(
        (m["content"] for m in record.get("messages", []) if m.get("role") == "user"), ""
    )
    return f"rec:{hashlib.sha1(instruction.encode('utf-8', 'replace')).hexdigest()}"


def _bucket(key: str, seed: int) -> float:
    """Stable hash -> [0, 1). Adding documents never re-buckets old ones."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def run_split(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    scfg = cfg.split
    records = list(read_jsonl(ws.validated_jsonl))
    if not records:
        raise RuntimeError(
            f"no validated records at {ws.validated_jsonl} — run the validate phase first"
        )

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []

    if scfg.group_by_source:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[_group_key(record)].append(record)

        # Assign whole groups by stable hash.
        for key, items in groups.items():
            if _bucket(key, scfg.seed) < scfg.train_ratio:
                train.extend(items)
            else:
                validation.extend(items)

        # With few groups the hash can land badly (every group in train).
        # Force at least one group into validation so the trainer has an
        # eval set at all.
        if not validation and len(groups) > 1:
            ordered = sorted(groups.items(), key=lambda kv: _bucket(kv[0], scfg.seed), reverse=True)
            moved_key, moved = ordered[0]
            validation = moved
            train = [r for r in train if _group_key(r) != moved_key]
            logger.warning(
                f"hash split produced an empty validation set; moved group "
                f"'{moved_key}' ({len(moved)} samples) into validation"
            )
        group_count = len(groups)
    else:
        for record in records:
            key = _group_key(record) + str(id(record))
            if _bucket(key, scfg.seed) < scfg.train_ratio:
                train.append(record)
            else:
                validation.append(record)
        group_count = len(records)

    # Last resort for a single-group corpus: a record-level split, clearly
    # reported as leaking, because no eval set at all is worse.
    leaked = False
    if not validation and train:
        cut = max(1, int(len(train) * (1 - scfg.train_ratio)))
        ordered = sorted(train, key=lambda r: _bucket(_group_key(r) + r["messages"][1]["content"][:64], scfg.seed))
        validation = ordered[:cut]
        keep_ids = {id(r) for r in ordered[cut:]}
        train = [r for r in train if id(r) in keep_ids]
        leaked = True
        logger.warning(
            "corpus has only one source group — fell back to a record-level "
            "split. Validation loss will be optimistic (same-document leakage)."
        )

    write_jsonl(train, ws.train_jsonl)
    write_jsonl(validation, ws.val_jsonl)

    def _types(items: list[dict[str, Any]]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for record in items:
            counter[(record.get("meta") or {}).get("sample_type", "unknown")] += 1
        return dict(counter.most_common())

    def _docs(items: list[dict[str, Any]]) -> int:
        return len({_group_key(r) for r in items})

    total = len(train) + len(validation)
    report = {
        "total_records": total,
        "train_records": len(train),
        "validation_records": len(validation),
        "train_ratio_actual": round(len(train) / total, 4) if total else 0.0,
        "train_ratio_target": scfg.train_ratio,
        "seed": scfg.seed,
        "grouped_by_source": scfg.group_by_source,
        "source_groups": group_count,
        "train_groups": _docs(train),
        "validation_groups": _docs(validation),
        "leakage_warning": leaked,
        "train_by_type": _types(train),
        "validation_by_type": _types(validation),
        "files": {
            "train": str(ws.train_jsonl),
            "validation": str(ws.val_jsonl),
        },
    }
    write_json(report, ws.splits / "split_report.json")

    logger.info(
        f"split: {len(train):,} train / {len(validation):,} validation "
        f"({report['train_ratio_actual']:.1%} train) across "
        f"{group_count:,} source group(s)"
    )
    if scfg.group_by_source and not leaked:
        logger.info("grouped by source document — no same-page leakage across the split")

    return report
