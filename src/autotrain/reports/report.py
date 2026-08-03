"""
Phase 12 — final report.

Reads every artifact the previous eleven phases produced and assembles one
document that answers: what was crawled, what was built from it, what was
trained, how good it is, and where everything ended up.

Two outputs, same content: final_report.json for machines and
final_report.md for humans. The Markdown includes an ASCII loss curve so the
report is self-contained — no image files, no plotting dependency, readable
over SSH.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger
from autotrain.utils.paths import Workspace
from autotrain.utils.state import PHASE_TITLES, PHASES, StateStore
from autotrain.utils.text import count_lines, read_json, write_json

logger = get_logger("reports")


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _fmt_bytes(size: float) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def ascii_loss_curve(
    points: list[dict[str, Any]], key: str, *, width: int = 64, height: int = 12
) -> list[str]:
    """
    Render a loss curve as ASCII.

    Keeps the report a single self-contained text file. Downsamples by
    bucketing so a 5,000-step run still fits in 64 columns.
    """
    values = [float(p[key]) for p in points if key in p]
    if len(values) < 2:
        return ["(not enough data points to plot)"]

    # Bucket-average down to `width` columns.
    if len(values) > width:
        bucket = len(values) / width
        sampled = []
        for i in range(width):
            chunk = values[int(i * bucket) : max(int((i + 1) * bucket), int(i * bucket) + 1)]
            sampled.append(sum(chunk) / len(chunk))
        values = sampled

    lowest, highest = min(values), max(values)
    span = highest - lowest or 1.0

    grid = [[" "] * len(values) for _ in range(height)]
    for x, value in enumerate(values):
        y = int((1 - (value - lowest) / span) * (height - 1))
        grid[y][x] = "•"

    lines: list[str] = []
    for y, row in enumerate(grid):
        label = ""
        if y == 0:
            label = f"{highest:6.3f}"
        elif y == height - 1:
            label = f"{lowest:6.3f}"
        else:
            label = " " * 6
        lines.append(f"{label} │{''.join(row)}")
    lines.append(" " * 6 + " └" + "─" * len(values))
    lines.append(" " * 8 + f"step 0{' ' * max(0, len(values) - 18)}step {len(points)}")
    return lines


def run_report(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    ws.reports.mkdir(parents=True, exist_ok=True)

    crawl = read_json(ws.crawl_report, default={}) or {}
    extract = read_json(ws.extract_report, default={}) or {}
    clean = read_json(ws.clean_report, default={}) or {}
    generation = read_json(ws.generation_state, default={}) or {}
    validation = read_json(ws.validation_report, default={}) or {}
    split = read_json(ws.splits / "split_report.json", default={}) or {}
    training = read_json(ws.train_metrics, default={}) or {}
    evaluation = read_json(ws.evaluation / "evaluation_report.json", default={}) or {}
    exports = read_json(ws.exports / "export_report.json", default={}) or {}
    benchmark = read_json(ws.benchmark / "benchmark.json", default={}) or {}

    phase_rows = []
    total_duration = 0.0
    for name in PHASES:
        state = store.state.phases.get(name)
        if state is None:
            phase_rows.append({"phase": name, "title": PHASE_TITLES[name],
                               "status": "pending", "duration_s": 0.0})
            continue
        total_duration += state.duration_s
        # This phase is the one writing the report, so it is necessarily still
        # "running" in the store. Reporting itself as incomplete is noise.
        status = "completed" if (name == "report" and state.status == "running") else state.status
        phase_rows.append({
            "phase": name,
            "title": PHASE_TITLES[name],
            "status": status,
            "duration_s": state.duration_s,
            "attempts": state.attempts,
            "error": state.error,
        })

    hardware = training.get("hardware", {})
    resolved = training.get("resolved", {})
    results = training.get("results", {})

    payload: dict[str, Any] = {
        "domain": cfg.domain,
        "run_name": store.state.run_name,
        "workspace": str(ws.root),
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_seconds": round(total_duration, 2),
        "phases": phase_rows,

        "crawl": {
            "pages_discovered": (crawl.get("pages") or {}).get("discovered", 0),
            "pages_fetched": (crawl.get("pages") or {}).get("fetched_this_run",
                             (crawl.get("pages") or {}).get("fetched", 0)),
            "pages_stored": (crawl.get("pages") or {}).get("stored_ok", 0),
            "pages_from_cache": (crawl.get("pages") or {}).get("from_cache", 0),
            "pages_failed": (crawl.get("pages") or {}).get("failed", 0),
            "left_unvisited": (crawl.get("pages") or {}).get("left_unvisited", 0),
            "skipped": crawl.get("skipped", {}),
            "bytes_downloaded": crawl.get("bytes_downloaded", 0),
            "duration_seconds": crawl.get("duration_seconds", 0),
            "pages_per_second": crawl.get("pages_per_second", 0),
            "sitemap_urls": (crawl.get("sitemap") or {}).get("url_count", 0),
            "robots_respected": (crawl.get("robots") or {}).get("enabled"),
        },
        "extraction": {
            "documents_extracted": extract.get("documents_extracted", 0),
            "empty_pages": extract.get("empty", 0),
            "methods": extract.get("methods", {}),
            "total_chars": extract.get("total_chars", 0),
            "with_code_blocks": extract.get("with_code_blocks", 0),
            "with_tables": extract.get("with_tables", 0),
        },
        "cleaning": {
            "input_documents": clean.get("input_documents", 0),
            "output_documents": clean.get("output_documents", 0),
            "dropped": clean.get("dropped", {}),
            "boilerplate_lines": (clean.get("boilerplate") or {}).get("lines_identified", 0),
            "chars_after": clean.get("chars_after", 0),
            "reduction_pct": clean.get("reduction_pct", 0),
        },
        "dataset": {
            "generator_backend": generation.get("backend", "unknown"),
            "total_pairs_generated": generation.get("total_pairs", 0),
            "by_sample_type": generation.get("by_type", {}),
            "failed_calls": generation.get("failures", 0),
            "valid_records": validation.get("valid_records", 0),
            "dropped_records": validation.get("dropped_records", 0),
            "invalid_ratio": validation.get("invalid_ratio", 0),
            "lengths": validation.get("lengths", {}),
            "train_records": split.get("train_records", 0),
            "validation_records": split.get("validation_records", 0),
            "source_groups": split.get("source_groups", 0),
            "grouped_split": split.get("grouped_by_source"),
            "leakage_warning": split.get("leakage_warning", False),
        },
        "training": {
            "base_model": training.get("model_id"),
            "device": resolved.get("device"),
            "device_name": hardware.get("device_name"),
            "quantised_4bit": resolved.get("load_in_4bit"),
            "dtype": resolved.get("torch_dtype"),
            "optimiser": resolved.get("optim"),
            "attention": resolved.get("attn_implementation"),
            "downgrades": resolved.get("downgrades", []),
            "lora": training.get("lora", {}),
            "schedule": training.get("schedule", {}),
            "train_loss": results.get("train_loss"),
            "eval_loss": results.get("eval_loss"),
            "eval_perplexity": results.get("eval_perplexity"),
            "runtime_seconds": results.get("train_runtime_seconds"),
            "samples_per_second": results.get("samples_per_second"),
            "interrupted": results.get("interrupted", False),
            "gpu": {
                "name": hardware.get("device_name"),
                "count": hardware.get("gpu_count", 0),
                "vram_gb": hardware.get("vram_gb", 0),
            },
            "ram_gb": hardware.get("ram_gb", 0),
            "memory": training.get("memory", {}),
        },
        "evaluation": {
            "perplexity": evaluation.get("perplexity"),
            "loss": evaluation.get("loss", {}),
            "quality": evaluation.get("quality", {}),
            "samples_evaluated": evaluation.get("samples_evaluated", 0),
        },
        "benchmark": benchmark.get("summary", {}),
        "exports": {
            key: value for key, value in exports.items() if key != "model_name"
        },
        "export_model_name": exports.get("model_name"),
    }

    write_json(payload, ws.reports / "final_report.json")
    _write_markdown(cfg, ws, payload, training)

    logger.info(f"final report written: {ws.reports / 'final_report.md'}")
    return payload


def _write_markdown(
    cfg: AutotrainConfig, ws: Workspace, p: dict[str, Any], training: dict[str, Any]
) -> None:
    crawl = p["crawl"]
    extraction = p["extraction"]
    cleaning = p["cleaning"]
    dataset = p["dataset"]
    train = p["training"]
    evaluation = p["evaluation"]
    benchmark = p["benchmark"]

    L: list[str] = [
        f"# autotrain report — {cfg.domain}",
        "",
        f"**Generated:** {p['generated_at_iso']}  ",
        f"**Workspace:** `{p['workspace']}`  ",
        f"**Total pipeline time:** {_fmt_duration(p['total_duration_seconds'])}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| | |",
        "|---|---|",
        f"| Pages crawled | {crawl['pages_stored']:,} stored / {crawl['pages_discovered']:,} discovered |",
        f"| Documents extracted | {extraction['documents_extracted']:,} |",
        f"| Documents after cleaning | {cleaning['output_documents']:,} |",
        f"| Instruction pairs generated | {dataset['total_pairs_generated']:,} |",
        f"| Valid training samples | {dataset['valid_records']:,} |",
        f"| Train / validation | {dataset['train_records']:,} / {dataset['validation_records']:,} |",
        f"| Base model | `{train['base_model']}` |",
        f"| Training device | {train['device']} ({train['device_name']}) |",
        f"| Final train loss | {train['train_loss']} |",
        f"| Final eval loss | {train['eval_loss']} |",
        f"| Eval perplexity | {train['eval_perplexity']} |",
        "",
    ]

    # ---- phases -----------------------------------------------------------
    L += ["## Phase timeline", "", "| # | Phase | Status | Duration |", "|---:|---|---|---:|"]
    for i, row in enumerate(p["phases"], 1):
        icon = {"completed": "done", "failed": "FAILED", "skipped": "skipped",
                "running": "running", "pending": "pending"}.get(row["status"], row["status"])
        L.append(f"| {i} | {row['title']} | {icon} | {_fmt_duration(row['duration_s'])} |")
    L.append("")

    failed = [r for r in p["phases"] if r["status"] == "failed"]
    if failed:
        L += ["> **Failed phases**", ""]
        for row in failed:
            L.append(f"> - `{row['phase']}`: {row.get('error', 'unknown error')}")
        L.append("")

    # ---- crawl ------------------------------------------------------------
    skipped = crawl.get("skipped", {})
    L += [
        "## 1-2. Crawl",
        "",
        f"- Corpus holds **{crawl['pages_stored']:,}** pages from "
        f"**{crawl['pages_discovered']:,}** discovered URLs",
        f"- This run fetched {crawl['pages_fetched']:,} "
        f"({crawl['pages_from_cache']:,} served from cache)",
        f"- Failed: {crawl['pages_failed']:,} · Left unvisited: {crawl['left_unvisited']:,}",
        f"- Downloaded {_fmt_bytes(crawl['bytes_downloaded'])} in {_fmt_duration(crawl['duration_seconds'])} "
        f"({crawl['pages_per_second']} pages/s)",
        f"- Sitemap contributed {crawl['sitemap_urls']:,} URLs · robots.txt respected: {crawl['robots_respected']}",
        "",
        "Skipped during discovery:",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in skipped.items():
        L.append(f"| {reason.replace('_', ' ')} | {count:,} |")
    L.append("")

    # ---- extraction / cleaning -------------------------------------------
    L += [
        "## 3-4. Extraction and cleaning",
        "",
        f"- Extracted **{extraction['documents_extracted']:,}** documents "
        f"({_fmt_bytes(extraction['total_chars'])} of text); {extraction['empty_pages']:,} pages yielded nothing",
        f"- Extraction methods: {extraction['methods']}",
        f"- {extraction['with_code_blocks']:,} documents contain code · "
        f"{extraction['with_tables']:,} contain tables",
        f"- Cleaning kept **{cleaning['output_documents']:,}** of {cleaning['input_documents']:,} documents "
        f"({cleaning['reduction_pct']}% of characters removed)",
        f"- Identified {cleaning['boilerplate_lines']:,} corpus-wide boilerplate lines",
        "",
        "Dropped during cleaning:",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in (cleaning.get("dropped") or {}).items():
        L.append(f"| {reason.replace('_', ' ')} | {count:,} |")
    L.append("")

    # ---- dataset ----------------------------------------------------------
    L += [
        "## 5-7. Dataset",
        "",
        f"- Generator backend: **{dataset['generator_backend']}**",
        f"- Generated **{dataset['total_pairs_generated']:,}** pairs "
        f"({dataset['failed_calls']:,} failed generation calls)",
        f"- Validation kept **{dataset['valid_records']:,}**, dropped {dataset['dropped_records']:,} "
        f"({dataset['invalid_ratio']:.1%} invalid)",
        f"- Split into **{dataset['train_records']:,}** train / **{dataset['validation_records']:,}** validation "
        f"across {dataset['source_groups']:,} source groups",
    ]
    if dataset.get("grouped_split"):
        L.append("- Split is grouped by source document — no same-page leakage between train and validation")
    if dataset.get("leakage_warning"):
        L.append("- **Warning:** corpus had one source group; validation loss is optimistic")
    L.append("")

    if dataset.get("by_sample_type"):
        L += ["Samples by type:", "", "| Type | Count |", "|---|---:|"]
        for sample_type, count in dataset["by_sample_type"].items():
            L.append(f"| {sample_type} | {count:,} |")
        L.append("")

    lengths = dataset.get("lengths") or {}
    if lengths.get("answer_chars"):
        answer = lengths["answer_chars"]
        instruction = lengths.get("instruction_chars", {})
        L += [
            "Length distribution (characters):",
            "",
            "| | min | p50 | mean | p95 | max |",
            "|---|---:|---:|---:|---:|---:|",
            f"| instruction | {instruction.get('min', 0)} | {instruction.get('p50', 0)} | "
            f"{instruction.get('mean', 0)} | {instruction.get('p95', 0)} | {instruction.get('max', 0)} |",
            f"| answer | {answer.get('min', 0)} | {answer.get('p50', 0)} | "
            f"{answer.get('mean', 0)} | {answer.get('p95', 0)} | {answer.get('max', 0)} |",
            "",
        ]

    # ---- training ---------------------------------------------------------
    lora = train.get("lora", {})
    schedule = train.get("schedule", {})
    L += [
        "## 8. Training",
        "",
        f"- Base model: `{train['base_model']}`",
        f"- Device: **{train['device']}** ({train['device_name']})"
        + (f", {train['gpu']['vram_gb']:.1f} GB VRAM" if train["gpu"].get("vram_gb") else "")
        + (f", {train['ram_gb']:.1f} GB RAM" if train.get("ram_gb") else ""),
        f"- Method: {'4-bit QLoRA' if train['quantised_4bit'] else 'LoRA'} "
        f"@ {train['dtype']}, optimiser `{train['optimiser']}`, attention `{train['attention']}`",
        f"- LoRA r={lora.get('r')} alpha={lora.get('alpha')} dropout={lora.get('dropout')} "
        f"on {len(lora.get('target_modules', []))} module type(s)",
        f"- Trainable parameters: **{lora.get('trainable_params', 0):,}** of "
        f"{lora.get('total_params', 0):,} ({lora.get('trainable_pct', 0)}%)",
        f"- Schedule: {schedule.get('epochs')} epochs, effective batch "
        f"{schedule.get('effective_batch_size')}, {schedule.get('completed_steps')} steps completed "
        f"of ~{schedule.get('planned_steps')} planned",
        f"- Sequence length {schedule.get('max_seq_length')}, "
        f"completion-only loss: {schedule.get('completion_only_loss')}",
        f"- Runtime: **{_fmt_duration(train.get('runtime_seconds') or 0)}** "
        f"({train.get('samples_per_second')} samples/s)",
    ]

    memory = train.get("memory") or {}
    if memory.get("peak_reserved_gb") is not None:
        line = (
            f"- Peak GPU memory: **{memory['peak_reserved_gb']} GB** reserved "
            f"({memory.get('peak_allocated_gb')} GB allocated)"
        )
        if memory.get("total_vram_gb"):
            line += (
                f" of {memory['total_vram_gb']} GB"
                + (f" — {memory['peak_utilisation_pct']}% utilised"
                   if memory.get("peak_utilisation_pct") is not None else "")
            )
        L.append(line)
    elif memory.get("peak_rss_gb") is not None:
        L.append(f"- Peak process memory: **{memory['peak_rss_gb']} GB** RSS")
    L.append("")

    if train.get("downgrades"):
        L += ["> **Hardware downgrades applied**", ""]
        for message in train["downgrades"]:
            L.append(f"> - {message}")
        L.append("")

    if train.get("interrupted"):
        L += ["> **Training was interrupted.** The adapter reflects partial training.", ""]

    # ---- loss curves ------------------------------------------------------
    train_log = training.get("train_log", [])
    eval_log = training.get("eval_log", [])
    if train_log:
        L += ["### Training loss", "", "```"]
        L += ascii_loss_curve(train_log, "loss")
        L += ["```", ""]
    if eval_log:
        L += ["### Validation loss", "", "```"]
        L += ascii_loss_curve(eval_log, "eval_loss")
        L += ["```", ""]

    # ---- evaluation -------------------------------------------------------
    quality = evaluation.get("quality") or {}
    loss = evaluation.get("loss") or {}
    L += [
        "## 9. Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Final train loss | {loss.get('final_train_loss')} |",
        f"| Final eval loss | {loss.get('final_eval_loss')} |",
        f"| Best eval loss | {loss.get('best_eval_loss')} |",
        f"| Train loss reduction | {loss.get('train_loss_reduction_pct')}% |",
        f"| Perplexity | {evaluation.get('perplexity')} |",
        f"| Instruction following | {quality.get('instruction_following_mean')} |",
        f"| Grounding in corpus | {quality.get('grounding_mean')} |",
        f"| Consistency | {quality.get('consistency_mean')} |",
        f"| Mean answer length | {quality.get('mean_answer_chars')} chars |",
        f"| Samples evaluated | {evaluation.get('samples_evaluated', 0)} |",
        "",
    ]
    if loss.get("overfitting_suspected"):
        L += [
            "> **Overfitting suspected** — validation loss rose while training loss fell.",
            "> Reduce `training.num_train_epochs` or raise `training.lora_dropout`.",
            "",
        ]
    if quality.get("low_grounding_pct", 0) > 25:
        L += [
            f"> **Grounding warning** — {quality['low_grounding_pct']}% of sampled answers "
            f"scored below the grounding threshold, meaning they used vocabulary absent from "
            f"the crawled corpus. This is the signature of hallucination on a closed-domain "
            f"fine-tune.",
            "",
        ]

    # ---- benchmark --------------------------------------------------------
    if benchmark:
        L += ["## 11. Benchmark vs base model", ""]
        if "delta" in benchmark:
            L += [
                "| Metric | Base | Fine-tuned | Delta |",
                "|---|---:|---:|---:|",
            ]
            for key in ("reference_overlap", "grounding", "instruction_following", "domain_vocabulary"):
                base_value = benchmark["base"][key]
                fine_value = benchmark["finetuned"][key]
                delta = benchmark["delta"][key]
                L.append(
                    f"| {key.replace('_', ' ')} | {base_value:.3f} | {fine_value:.3f} | "
                    f"{'+' if delta > 0 else ''}{delta:.3f} |"
                )
            L += [
                "",
                f"Fine-tuned wins **{benchmark.get('finetuned_wins', 0)}"
                f"/{benchmark.get('finetuned_wins', 0) + benchmark.get('base_wins', 0)}** "
                f"head-to-head ({benchmark.get('win_rate', 0):.0%}).",
                "",
                f"Full side-by-side answers: `{ws.benchmark_report}`",
                "",
            ]
        else:
            L += [f"Benchmark ran without a base comparison. See `{ws.benchmark_report}`.", ""]

    # ---- exports ----------------------------------------------------------
    exports = p.get("exports") or {}
    L += ["## 10. Exports", "", "| Artifact | Location | Size |", "|---|---|---:|"]
    for key in ("adapter", "merged", "gguf", "ollama"):
        entry = exports.get(key)
        if not isinstance(entry, dict):
            continue
        if entry.get("skipped"):
            L.append(f"| {key} | _skipped_ | — |")
            continue
        if entry.get("error"):
            L.append(f"| {key} | _failed_ | — |")
            continue
        location = entry.get("path") or entry.get("modelfile") or "—"
        size = entry.get("size_human", "—")
        L.append(f"| {key} | `{location}` | {size} |")
    L.append("")

    ollama = exports.get("ollama")
    if isinstance(ollama, dict) and ollama.get("install_command"):
        L += [
            "Install into Ollama:",
            "",
            "```bash",
            ollama["install_command"],
            f"ollama run {p.get('export_model_name')}",
            "```",
            "",
        ]
        if not ollama.get("ready_to_install"):
            L += [
                "> The Modelfile is written but the GGUF is missing. Build llama.cpp and "
                "re-run with `--set export.export_gguf=true`, then run the command above.",
                "",
            ]

    gguf = exports.get("gguf")
    if isinstance(gguf, dict) and gguf.get("skipped") and gguf.get("reason"):
        L += ["> GGUF export was skipped:", "", "```", str(gguf["reason"])[:600], "```", ""]

    # ---- artifacts --------------------------------------------------------
    L += [
        "## Artifact index",
        "",
        "```",
        f"{ws.root}",
        "├── discovery/       urls.json, sitemap.json, crawl_report.json",
        "├── dataset/raw_html/  raw pages + index.jsonl (status, headers, redirects)",
        "├── clean_docs/      per-page Markdown",
        "├── clean_dataset/   deduplicated corpus + corpus.jsonl",
        "├── generated/       dataset.jsonl",
        "├── validated/       dataset.jsonl + validation_report.json",
        "├── splits/          train.jsonl, validation.jsonl",
        "├── training/        checkpoints/, adapter/, metrics.json",
        "├── evaluation/      evaluation_report.json, sample_generations.json, loss_curves.json",
        "├── exports/         adapter/, merged/, gguf/, ollama/",
        "├── benchmark/       benchmark_report.md, benchmark.json",
        "└── reports/         final_report.md, final_report.json",
        "```",
        "",
        "---",
        "",
        f"_Generated by autotrain from a single input: `{cfg.domain}`_",
        "",
    ]

    (ws.reports / "final_report.md").write_text("\n".join(L), encoding="utf-8")
