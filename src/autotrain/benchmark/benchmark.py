"""
Phase 11 — automatic benchmarking against the base model.

The question this answers is narrow and important: did fine-tuning make the
model better *at this website's domain*, or did it just make it different?

Method: draw questions from the held-out validation split, ask both the base
model and the fine-tuned model, and score both answers the same way. Using
held-out questions matters — benchmarking on training questions measures
memorisation and will always show a large fake improvement.

Four scores per answer, all reference-free except overlap:
  reference_overlap   token F1 against the reference answer. Blunt, but it
                      is the only score that knows the right answer.
  grounding           content words present in the source corpus.
  instruction_follow  did it produce the requested shape.
  domain_vocabulary   does the answer use this site's terminology at all.

A base model that has never seen the site typically scores well on
instruction-following (it is already instruction-tuned) and badly on
grounding and domain vocabulary. That gap is the thing fine-tuning closes,
and it is what the report highlights.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import content_words, read_json, read_jsonl, write_json

logger = get_logger("benchmark")

_TOKEN = re.compile(r"[a-z0-9_.-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1. Insensitive to phrasing, sensitive to content."""
    pred = Counter(_tokens(prediction))
    ref = Counter(_tokens(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return round(2 * precision * recall / (precision + recall), 4)


def build_domain_vocabulary(ws: Workspace, top_n: int = 400) -> set[str]:
    """
    Terms that are characteristic of this site.

    Frequency alone would return English stopwords, so this takes words that
    are common in the corpus but structurally distinctive: multi-part
    identifiers, CamelCase, snake_case, and terms appearing in headings.
    """
    counts: Counter[str] = Counter()
    heading_terms: Counter[str] = Counter()

    for record in read_jsonl(ws.corpus_jsonl):
        text = record.get("text", "")
        counts.update(content_words(text))
        for match in re.finditer(r"^#{1,6}\s+(.+)$", text, re.MULTILINE):
            heading_terms.update(content_words(match.group(1)))

    distinctive: set[str] = set()
    for word, count in counts.items():
        if count < 3:
            continue
        looks_technical = (
            "_" in word
            or "." in word
            or "-" in word
            or bool(re.search(r"[a-z][A-Z]", word))
            or len(word) > 11
        )
        if looks_technical or heading_terms.get(word, 0) >= 2:
            distinctive.add(word)

    if len(distinctive) < 40:
        # Small or plain-prose corpus: fall back to the most frequent
        # non-stopword terms so the metric still means something.
        distinctive |= {w for w, _ in counts.most_common(top_n)}

    return distinctive


def domain_vocabulary_score(answer: str, vocabulary: set[str]) -> float:
    words = content_words(answer)
    if not words:
        return 0.0
    return round(len(words & vocabulary) / len(words), 4)


def _load_base(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return model, tokenizer


def run_benchmark(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    from autotrain.evaluator.evaluate import (
        generate,
        load_finetuned,
        score_instruction_following,
    )

    bcfg = cfg.benchmark
    ws.benchmark.mkdir(parents=True, exist_ok=True)

    if not bcfg.enabled:
        logger.info("benchmark disabled by configuration")
        return {"skipped": True, "reason": "benchmark.enabled is false"}

    # ---- questions -------------------------------------------------------
    questions: list[dict[str, Any]] = []
    if bcfg.source == "custom" and bcfg.custom_questions:
        questions = [{"instruction": q, "reference": "", "sample_type": "custom"}
                     for q in bcfg.custom_questions[: bcfg.num_questions]]
    else:
        for record in read_jsonl(ws.val_jsonl):
            messages = record.get("messages", [])
            instruction = next((m["content"] for m in messages if m["role"] == "user"), "")
            reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            if instruction and reference:
                questions.append({
                    "instruction": instruction,
                    "reference": reference,
                    "system": system,
                    "sample_type": (record.get("meta") or {}).get("sample_type", "unknown"),
                })
            if len(questions) >= bcfg.num_questions:
                break

    if not questions:
        return {"skipped": True, "reason": "no benchmark questions available"}

    logger.info(f"benchmarking on {len(questions)} held-out question(s)")

    vocabulary = build_domain_vocabulary(ws)
    corpus_vocabulary: set[str] = set()
    for record in read_jsonl(ws.corpus_jsonl):
        corpus_vocabulary |= content_words(record.get("text", ""))

    # ---- fine-tuned model -----------------------------------------------
    ft_model, tokenizer, model_id, device = load_finetuned(cfg, ws)

    rows: list[dict[str, Any]] = []

    def answer_with(model, tok, question: dict[str, Any]) -> str:
        messages = []
        if question.get("system"):
            messages.append({"role": "system", "content": question["system"]})
        messages.append({"role": "user", "content": question["instruction"]})
        try:
            return generate(
                model, tok, messages,
                max_new_tokens=bcfg.max_new_tokens,
                temperature=0.2, top_p=0.9, device=device,
            )
        except Exception as exc:
            logger.debug(f"generation failed: {exc}")
            return ""

    progress = make_progress()
    with progress:
        bar = progress.add_task("benchmark: fine-tuned", total=len(questions))
        for question in questions:
            rows.append({**question, "finetuned": answer_with(ft_model, tokenizer, question)})
            progress.advance(bar)

    del ft_model

    # ---- base model ------------------------------------------------------
    if bcfg.compare_with_base:
        logger.info(f"loading base model {model_id} for comparison")
        try:
            base_model, base_tokenizer = _load_base(model_id, device)
            progress = make_progress()
            with progress:
                bar = progress.add_task("benchmark: base", total=len(rows))
                for row in rows:
                    row["base"] = answer_with(base_model, base_tokenizer, row)
                    progress.advance(bar)
            del base_model
        except Exception as exc:
            logger.warning(f"base model comparison failed: {type(exc).__name__}: {exc}")
            for row in rows:
                row["base"] = ""

    # ---- scoring ---------------------------------------------------------
    def score(instruction: str, answer: str, reference: str) -> dict[str, float]:
        words = content_words(answer)
        return {
            "reference_overlap": token_f1(answer, reference) if reference else 0.0,
            "grounding": (
                round(len(words & corpus_vocabulary) / len(words), 4) if words else 0.0
            ),
            "instruction_following": score_instruction_following(instruction, answer)["score"],
            "domain_vocabulary": domain_vocabulary_score(answer, vocabulary),
            "length": float(len(answer)),
        }

    for row in rows:
        reference = row.get("reference", "")
        row["scores_finetuned"] = score(row["instruction"], row["finetuned"], reference)
        if "base" in row:
            row["scores_base"] = score(row["instruction"], row["base"], reference)

    def mean(key: str, which: str) -> float:
        values = [r[which][key] for r in rows if which in r]
        return round(sum(values) / len(values), 4) if values else 0.0

    metrics = ["reference_overlap", "grounding", "instruction_following", "domain_vocabulary", "length"]
    summary = {
        "finetuned": {m: mean(m, "scores_finetuned") for m in metrics},
    }
    if any("scores_base" in r for r in rows):
        summary["base"] = {m: mean(m, "scores_base") for m in metrics}
        summary["delta"] = {
            m: round(summary["finetuned"][m] - summary["base"][m], 4) for m in metrics
        }
        wins = sum(
            1 for r in rows
            if "scores_base" in r
            and r["scores_finetuned"]["reference_overlap"] > r["scores_base"]["reference_overlap"]
        )
        summary["finetuned_wins"] = wins
        summary["base_wins"] = len(rows) - wins
        summary["win_rate"] = round(wins / len(rows), 4)

    payload = {
        "model_id": model_id,
        "device": device,
        "questions": len(rows),
        "source": bcfg.source,
        "summary": summary,
        "rows": rows,
        "benchmarked_at": time.time(),
    }
    write_json(payload, ws.benchmark / "benchmark.json")
    _write_markdown(cfg, ws, payload)

    if "delta" in summary:
        logger.info(
            f"benchmark: fine-tuned wins {summary['finetuned_wins']}/{len(rows)} "
            f"({summary['win_rate']:.0%}); "
            f"grounding {summary['delta']['grounding']:+.3f}, "
            f"domain vocabulary {summary['delta']['domain_vocabulary']:+.3f}"
        )
    else:
        logger.info(f"benchmark complete on {len(rows)} questions (no base comparison)")

    return {k: v for k, v in payload.items() if k != "rows"}


def _write_markdown(cfg: AutotrainConfig, ws: Workspace, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows = payload["rows"]
    has_base = "base" in summary

    lines: list[str] = [
        f"# Benchmark report — {cfg.domain}",
        "",
        f"- **Base model:** `{payload['model_id']}`",
        f"- **Adapter:** `{ws.adapter}`",
        f"- **Questions:** {payload['questions']} (source: {payload['source']}, held out from training)",
        f"- **Device:** {payload['device']}",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(payload['benchmarked_at']))}",
        "",
        "## Summary",
        "",
    ]

    if has_base:
        lines += [
            "| Metric | Base | Fine-tuned | Delta |",
            "|---|---:|---:|---:|",
        ]
        labels = {
            "reference_overlap": "Reference overlap (token F1)",
            "grounding": "Grounding in source corpus",
            "instruction_following": "Instruction following",
            "domain_vocabulary": "Domain vocabulary use",
            "length": "Mean answer length (chars)",
        }
        for key, label in labels.items():
            base = summary["base"][key]
            fine = summary["finetuned"][key]
            delta = summary["delta"][key]
            arrow = "+" if delta > 0 else ""
            if key == "length":
                lines.append(f"| {label} | {base:.0f} | {fine:.0f} | {arrow}{delta:.0f} |")
            else:
                lines.append(f"| {label} | {base:.3f} | {fine:.3f} | {arrow}{delta:.3f} |")
        lines += [
            "",
            f"**Head-to-head on reference overlap:** fine-tuned wins "
            f"{summary['finetuned_wins']}/{payload['questions']} "
            f"({summary['win_rate']:.0%}), base wins {summary['base_wins']}.",
            "",
            "> Grounding and domain-vocabulary deltas are the meaningful ones here.",
            "> The base model is already instruction-tuned, so it scores well on shape;",
            "> what it cannot do is use this site's terminology or stay inside its facts.",
            "",
        ]
    else:
        lines += [
            "| Metric | Fine-tuned |",
            "|---|---:|",
        ]
        for key, value in summary["finetuned"].items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value:.3f} |")
        lines += ["", "_No base-model comparison was run._", ""]

    lines += ["## Side-by-side answers", ""]

    for i, row in enumerate(rows[:15], 1):
        lines += [
            f"### {i}. {row['instruction'][:160]}",
            "",
            f"*Sample type: `{row.get('sample_type', 'unknown')}`*",
            "",
        ]
        if row.get("reference"):
            lines += ["**Reference**", "", "```text", row["reference"][:800].strip(), "```", ""]
        lines += [
            "**Fine-tuned**"
            + (f" — overlap {row['scores_finetuned']['reference_overlap']:.3f}, "
               f"grounding {row['scores_finetuned']['grounding']:.3f}" if row.get("scores_finetuned") else ""),
            "",
            "```text",
            (row.get("finetuned") or "(empty)")[:800].strip(),
            "```",
            "",
        ]
        if has_base:
            lines += [
                "**Base**"
                + (f" — overlap {row['scores_base']['reference_overlap']:.3f}, "
                   f"grounding {row['scores_base']['grounding']:.3f}" if row.get("scores_base") else ""),
                "",
                "```text",
                (row.get("base") or "(empty)")[:800].strip(),
                "```",
                "",
            ]
        lines.append("---")
        lines.append("")

    ws.benchmark_report.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"benchmark report written: {ws.benchmark_report}")
