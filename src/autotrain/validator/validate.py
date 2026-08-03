"""
Phase 6 — dataset validation.

Cheap checks that prevent expensive failures. A malformed record costs
nothing to drop here and costs an entire training run to discover at step
4,000.

Every check is a named `Issue` so validation_report.json says exactly what
was wrong and how often, rather than reporting one aggregate count. The
distinction between a *drop* (record removed) and a *warning* (record kept,
flagged) is deliberate: truncating a slightly-long answer is recoverable,
but a record with no assistant turn is not.
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
    corruption_ratio,
    normalize_text,
    read_jsonl_strict,
    sha256_text,
    write_json,
    write_jsonl,
)

logger = get_logger("validator")

# Phrases that mean the generator leaked its own scaffolding into the answer.
# These make the fine-tuned model reference documents the user cannot see.
_LEAK_PATTERNS = re.compile(
    r"\b(?:the (?:above )?(?:excerpt|passage|text|document|content|context)|"
    r"this (?:excerpt|passage|page|document|snippet) (?:says|states|shows|describes|mentions)|"
    r"based on the (?:excerpt|text|provided|given)|"
    r"according to the (?:excerpt|text|passage|content)|"
    r"as (?:mentioned|stated|shown|described) (?:above|in the (?:excerpt|text)))\b",
    re.I,
)

# A refusal or apology is the generator failing, not content.
_REFUSAL = re.compile(
    r"^\s*(?:i(?:'m| am) (?:sorry|unable|not able)|i cannot|i can't|"
    r"as an ai|unfortunately,? i|there (?:is|are) (?:no|not enough) "
    r"(?:information|content|context))",
    re.I,
)


@dataclass
class Issue:
    code: str
    severity: str  # "drop" | "warn"
    message: str
    count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def record(self, lineno: int, detail: str = "") -> None:
        self.count += 1
        if len(self.examples) < 10:
            self.examples.append({"line": lineno, "detail": detail[:300]})


class Validator:
    def __init__(self, cfg: AutotrainConfig) -> None:
        self.cfg = cfg.validation
        self.issues: dict[str, Issue] = {}
        for code, severity, message in (
            ("json_parse_error", "drop", "line is not valid JSON"),
            ("missing_messages", "drop", "record has no 'messages' list"),
            ("bad_message_shape", "drop", "a message is missing role or content"),
            ("missing_user_turn", "drop", "no user turn"),
            ("missing_assistant_turn", "drop", "no assistant turn"),
            ("empty_content", "drop", "a message has empty content"),
            ("instruction_too_short", "drop", "instruction below minimum length"),
            ("answer_too_short", "drop", "answer below minimum length"),
            ("sample_too_long", "drop", "record exceeds maximum character budget"),
            ("encoding_corrupted", "drop", "too many replacement/control characters"),
            ("duplicate", "drop", "identical instruction+output already seen"),
            ("refusal", "drop", "answer is a refusal or apology"),
            ("echo", "drop", "answer merely repeats the instruction"),
            ("context_leak", "warn", "answer refers to 'the excerpt'/'the text'"),
            ("missing_meta", "warn", "record has no meta.source_index"),
            ("truncated_answer", "warn", "answer appears cut off mid-sentence"),
        ):
            self.issues[code] = Issue(code, severity, message)

    def _hit(self, code: str, lineno: int, detail: str = "") -> str:
        issue = self.issues[code]
        issue.record(lineno, detail)
        return issue.severity

    def validate_record(
        self, record: dict[str, Any], lineno: int, seen: set[str]
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Return (cleaned_record_or_None, warning_codes)."""
        warnings: list[str] = []

        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            self._hit("missing_messages", lineno)
            return None, warnings

        roles: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                self._hit("bad_message_shape", lineno, str(message)[:120])
                return None, warnings
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                self._hit("bad_message_shape", lineno, f"role={role!r}")
                return None, warnings
            if not content.strip():
                self._hit("empty_content", lineno, f"role={role}")
                return None, warnings
            roles.append(role)

        if "user" not in roles:
            self._hit("missing_user_turn", lineno)
            return None, warnings
        if "assistant" not in roles:
            self._hit("missing_assistant_turn", lineno)
            return None, warnings

        instruction = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        ).strip()
        answer = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        ).strip()

        if len(instruction) < self.cfg.min_instruction_chars:
            self._hit("instruction_too_short", lineno, instruction[:80])
            return None, warnings

        if len(answer) < self.cfg.min_answer_chars:
            self._hit("answer_too_short", lineno, answer[:80])
            return None, warnings

        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > self.cfg.max_sample_chars:
            self._hit("sample_too_long", lineno, f"{total_chars} chars")
            return None, warnings

        combined = instruction + "\n" + answer
        if corruption_ratio(combined) > self.cfg.max_corruption_ratio:
            self._hit("encoding_corrupted", lineno, repr(combined[:80]))
            return None, warnings

        if _REFUSAL.search(answer):
            self._hit("refusal", lineno, answer[:100])
            return None, warnings

        if answer.strip().lower() == instruction.strip().lower():
            self._hit("echo", lineno, instruction[:80])
            return None, warnings

        if self.cfg.drop_duplicates:
            fingerprint = sha256_text(
                normalize_text(instruction).lower() + "\x00" + normalize_text(answer).lower()
            )
            if fingerprint in seen:
                self._hit("duplicate", lineno, instruction[:80])
                return None, warnings
            seen.add(fingerprint)

        # -- warnings: record is kept ---------------------------------------
        if _LEAK_PATTERNS.search(answer):
            self._hit("context_leak", lineno, answer[:120])
            warnings.append("context_leak")

        meta = record.get("meta")
        if not isinstance(meta, dict) or meta.get("source_index") is None:
            self._hit("missing_meta", lineno)
            warnings.append("missing_meta")

        # An answer ending without terminal punctuation mid-word is very
        # likely a max_tokens truncation.
        if len(answer) > 200 and not re.search(r"[.!?:;)\]`\"'»]\s*$", answer) and not answer.endswith("```"):
            self._hit("truncated_answer", lineno, answer[-80:])
            warnings.append("truncated_answer")

        return record, warnings


def run_validation(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    source = ws.dataset_jsonl
    if not source.exists():
        raise RuntimeError(f"dataset not found at {source} — run the generate phase first")

    validator = Validator(cfg)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []

    total_lines = 0
    parse_failures = 0
    type_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    generator_counts: Counter[str] = Counter()
    instruction_lengths: list[int] = []
    answer_lengths: list[int] = []
    warned = 0

    progress = make_progress()
    with progress:
        bar = progress.add_task("validating", total=None)

        for lineno, raw, parsed in read_jsonl_strict(source):
            total_lines += 1
            progress.advance(bar)

            if parsed is None:
                validator._hit("json_parse_error", lineno, raw[:120])
                parse_failures += 1
                continue

            record, warnings = validator.validate_record(parsed, lineno, seen)
            if record is None:
                continue

            if warnings:
                warned += 1
                record.setdefault("meta", {})["validation_warnings"] = warnings

            kept.append(record)

            meta = record.get("meta", {})
            type_counts[meta.get("sample_type", "unknown")] += 1
            generator_counts[meta.get("generator", "unknown")] += 1
            if meta.get("source_index") is not None:
                source_counts[meta["source_index"]] += 1

            instruction = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
            answer = next((m["content"] for m in record["messages"] if m["role"] == "assistant"), "")
            instruction_lengths.append(len(instruction))
            answer_lengths.append(len(answer))

    write_jsonl(kept, ws.validated_jsonl)

    dropped = total_lines - len(kept)
    invalid_ratio = dropped / total_lines if total_lines else 0.0

    def _stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 1),
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    report = {
        "input_file": str(source),
        "output_file": str(ws.validated_jsonl),
        "total_records": total_lines,
        "valid_records": len(kept),
        "dropped_records": dropped,
        "invalid_ratio": round(invalid_ratio, 4),
        "records_with_warnings": warned,
        "parse_failures": parse_failures,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "count": issue.count,
                "examples": issue.examples,
            }
            for issue in sorted(
                validator.issues.values(), key=lambda i: -i.count
            )
            if issue.count
        ],
        "distribution": {
            "by_sample_type": dict(type_counts.most_common()),
            "by_generator": dict(generator_counts.most_common()),
            "unique_source_documents": len(source_counts),
            "samples_per_document": round(
                len(kept) / len(source_counts), 2
            ) if source_counts else 0.0,
        },
        "lengths": {
            "instruction_chars": _stats(instruction_lengths),
            "answer_chars": _stats(answer_lengths),
        },
    }
    write_json(report, ws.validation_report)

    logger.info(
        f"validation: {len(kept):,}/{total_lines:,} records valid "
        f"({dropped:,} dropped, {warned:,} flagged)"
    )
    for issue in report["issues"][:8]:
        logger.info(f"   {issue['severity']:<5} {issue['code']:<24} {issue['count']:>6,}")

    if not kept:
        raise RuntimeError(
            "validation rejected every record. Inspect validation_report.json — "
            "this usually means the generator backend returned unusable output."
        )

    if invalid_ratio > cfg.validation.max_invalid_ratio:
        raise RuntimeError(
            f"{invalid_ratio:.1%} of records were invalid, above the configured "
            f"limit of {cfg.validation.max_invalid_ratio:.0%}. "
            f"See {ws.validation_report}."
        )

    return report
