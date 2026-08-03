"""
The orchestrator.

Walks the twelve phases in order, recording each one's status in run.json so
an interrupted run resumes where it stopped. A phase is skipped only when it
previously completed AND its declared output artifacts are still on disk —
deleting an artifact is therefore a supported way to force one stage to
re-run without touching the others.

Failure policy is per-phase, not global. The data phases (1-8) are hard
dependencies: if the crawl fails there is nothing to extract, so the run
stops. The post-training phases (9-12) are reporting, and a failure in one
should not destroy the trained model or prevent the others from running —
those are recorded as failed and the pipeline continues, with the final
report stating what went wrong.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, phase_logger, print_banner
from autotrain.utils.paths import Workspace, make_workspace
from autotrain.utils.state import PHASE_TITLES, PHASES, StateStore
from autotrain.utils.text import sha256_text

logger = get_logger("pipeline")


@dataclass
class PhaseDef:
    name: str
    title: str
    run: Callable[[AutotrainConfig, Workspace, StateStore], dict[str, Any]]
    outputs: Callable[[Workspace], list[Path]]
    # A critical phase stops the pipeline on failure; a non-critical one is
    # recorded and skipped past.
    critical: bool = True
    # Phases that need the heavy training stack installed.
    needs_torch: bool = False


# ---------------------------------------------------------------------------
# Phase adapters
# ---------------------------------------------------------------------------


def _discover_and_fetch(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.crawler.crawler import crawl_sync

    report = crawl_sync(cfg, ws)
    store.publish("pages_stored", (report.get("pages") or {}).get("stored_ok", 0))
    return report


def _extract(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.extractor.extract import run_extraction

    return run_extraction(cfg, ws)


def _clean(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.cleaner.clean import run_cleaning

    return run_cleaning(cfg, ws)


def _generate(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.dataset_builder.builder import run_generation

    report = run_generation(cfg, ws)
    store.publish("generator_backend", report.get("backend"))
    return report


def _validate(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.validator.validate import run_validation

    return run_validation(cfg, ws)


def _split(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.validator.split import run_split

    return run_split(cfg, ws)


def _train(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.trainer.train import run_training

    summary = run_training(cfg, ws)
    store.publish("base_model", summary.get("model_id"))
    store.publish("adapter_path", summary.get("adapter_path"))
    return summary


def _evaluate(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.evaluator.evaluate import run_evaluation

    return run_evaluation(cfg, ws)


def _export(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.exporter.export import run_export

    return run_export(cfg, ws)


def _benchmark(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.benchmark.benchmark import run_benchmark

    return run_benchmark(cfg, ws)


def _report(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.reports.report import run_report

    return run_report(cfg, ws, store)


# The crawl phase produces both discovery and fetch artifacts in one pass —
# you must fetch a page to read its links — so `fetch` is a no-op marker that
# exists to keep the twelve-phase contract visible in run.json and the report.
def _fetch_marker(cfg: AutotrainConfig, ws: Workspace, store: StateStore) -> dict[str, Any]:
    from autotrain.utils.text import count_lines

    stored = len(list(ws.raw_html.glob("page_*.html")))
    return {
        "raw_html_files": stored,
        "index_records": count_lines(ws.fetch_index),
        "note": "HTML is collected during discovery; one fetch per URL, no second pass",
    }


PHASE_DEFS: list[PhaseDef] = [
    PhaseDef("discover", PHASE_TITLES["discover"], _discover_and_fetch,
             lambda ws: [ws.urls_json, ws.sitemap_json, ws.crawl_report]),
    PhaseDef("fetch", PHASE_TITLES["fetch"], _fetch_marker,
             lambda ws: [ws.fetch_index]),
    PhaseDef("extract", PHASE_TITLES["extract"], _extract,
             lambda ws: [ws.extract_report, ws.clean_docs / "index.jsonl"]),
    PhaseDef("clean", PHASE_TITLES["clean"], _clean,
             lambda ws: [ws.corpus_jsonl, ws.clean_report]),
    PhaseDef("generate", PHASE_TITLES["generate"], _generate,
             lambda ws: [ws.dataset_jsonl]),
    PhaseDef("validate", PHASE_TITLES["validate"], _validate,
             lambda ws: [ws.validated_jsonl, ws.validation_report]),
    PhaseDef("split", PHASE_TITLES["split"], _split,
             lambda ws: [ws.train_jsonl, ws.val_jsonl]),
    PhaseDef("train", PHASE_TITLES["train"], _train,
             lambda ws: [ws.train_metrics, ws.adapter / "adapter_config.json"],
             needs_torch=True),
    PhaseDef("evaluate", PHASE_TITLES["evaluate"], _evaluate,
             lambda ws: [ws.evaluation / "evaluation_report.json"],
             critical=False, needs_torch=True),
    PhaseDef("export", PHASE_TITLES["export"], _export,
             lambda ws: [ws.exports / "export_report.json"],
             critical=False, needs_torch=True),
    PhaseDef("benchmark", PHASE_TITLES["benchmark"], _benchmark,
             lambda ws: [ws.benchmark / "benchmark.json"],
             critical=False, needs_torch=True),
    PhaseDef("report", PHASE_TITLES["report"], _report,
             lambda ws: [ws.reports / "final_report.md"],
             critical=False),
]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


class Pipeline:
    def __init__(self, cfg: AutotrainConfig, *, force: bool = False,
                 only: list[str] | None = None) -> None:
        if not cfg.domain:
            raise ValueError("a domain is required")

        self.cfg = cfg
        self.force = force
        self.only = set(only) if only else None

        self.ws = make_workspace(cfg.runtime.workspace, cfg.domain, cfg.runtime.run_name)
        self.store = StateStore(self.ws.run_file)
        self.store.state.domain = cfg.domain
        self.store.state.run_name = self.ws.root.name
        self.store.state.config_digest = sha256_text(cfg.model_dump_json())[:16]
        self.store.save()

    def phases_to_run(self) -> list[PhaseDef]:
        selected = self.cfg.runtime.phases or []
        wanted = self.only or (set(selected) if selected else None)
        if wanted is None:
            return PHASE_DEFS
        unknown = wanted - set(PHASES)
        if unknown:
            raise ValueError(
                f"unknown phase(s): {', '.join(sorted(unknown))}. "
                f"Valid phases: {', '.join(PHASES)}"
            )
        return [p for p in PHASE_DEFS if p.name in wanted]

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        phases = self.phases_to_run()

        print_banner(
            f"autotrain  ·  {self.cfg.domain}",
            f"{len(phases)} phase(s) → {self.ws.root}",
        )

        torch_ok = _torch_available()
        results: dict[str, Any] = {}
        failures: list[str] = []

        for i, phase in enumerate(phases, 1):
            outputs = phase.outputs(self.ws)

            if not self.store.should_run(
                phase.name, outputs, self.cfg.runtime.resume, self.force
            ):
                logger.info(f"[{i}/{len(phases)}] {phase.title}: already complete, skipping")
                results[phase.name] = self.store.state.phases[phase.name].summary
                continue

            if phase.needs_torch and not torch_ok:
                message = (
                    "the training stack is not installed — "
                    "pip install -e '.[train]' (and '.[cuda]' for 4-bit QLoRA)"
                )
                logger.warning(f"[{i}/{len(phases)}] {phase.title}: skipped — {message}")
                self.store.skip(phase.name, message)
                if phase.critical:
                    failures.append(phase.name)
                    break
                continue

            logger.info(f"[{i}/{len(phases)}] {phase.title}")
            self.store.start(phase.name)

            try:
                with phase_logger(self.ws.logs, phase.name):
                    summary = phase.run(self.cfg, self.ws, self.store)
            except KeyboardInterrupt:
                self.store.fail(phase.name, "interrupted by user")
                logger.warning(
                    f"interrupted during '{phase.name}'. "
                    f"Re-run the same command to resume from here."
                )
                raise
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                self.store.fail(phase.name, detail)
                failures.append(phase.name)
                logger.error(f"phase '{phase.name}' failed — {detail}")
                logger.debug(traceback.format_exc())

                if phase.critical:
                    logger.error(
                        f"'{phase.name}' is a required phase; stopping. "
                        f"Fix the cause and re-run to resume."
                    )
                    break
                logger.warning(f"'{phase.name}' is optional; continuing")
                continue

            # Keep the state summary small: full artifacts live in their own
            # files, and run.json is rewritten on every transition.
            compact = {
                k: v for k, v in (summary or {}).items()
                if not isinstance(v, (list, dict)) or len(str(v)) < 2000
            }
            self.store.complete(phase.name, compact)
            results[phase.name] = summary

        duration = time.monotonic() - started

        # The report phase reads everything, so run it even after a failure
        # further up — a failed run with a report is far more debuggable.
        if failures and "report" not in {p.name for p in phases if p.name in results}:
            try:
                from autotrain.reports.report import run_report

                with phase_logger(self.ws.logs, "report"):
                    run_report(self.cfg, self.ws, self.store)
                logger.info("wrote a final report covering the partial run")
            except Exception:
                pass

        self._summarise(duration, failures)

        return {
            "workspace": str(self.ws.root),
            "duration_seconds": round(duration, 2),
            "failed_phases": failures,
            "results": results,
        }

    def _summarise(self, duration: float, failures: list[str]) -> None:
        from autotrain.utils.logging import console

        c = console()
        c.print()
        c.rule("[bold]summary[/bold]")

        for name in PHASES:
            state = self.store.state.phases.get(name)
            if state is None:
                continue
            colour = {
                "completed": "green",
                "failed": "red",
                "skipped": "yellow",
                "running": "cyan",
            }.get(state.status, "dim")
            c.print(
                f"  [{colour}]{state.status:<10}[/{colour}] "
                f"{PHASE_TITLES[name]:<34} {state.duration_s:>8.1f}s"
            )

        c.print()
        if failures:
            c.print(f"  [red]{len(failures)} phase(s) failed:[/red] {', '.join(failures)}")
            c.print(f"  Re-run the same command to resume from the first failure.")
        else:
            c.print(f"  [green]pipeline complete[/green] in {duration / 60:.1f} min")

        report = self.ws.reports / "final_report.md"
        if report.exists():
            c.print(f"  report:    {report}")
        adapter = self.ws.adapter / "adapter_config.json"
        if adapter.exists():
            c.print(f"  adapter:   {self.ws.adapter}")
        if (self.ws.exports / "export_report.json").exists():
            c.print(f"  exports:   {self.ws.exports}")
        c.print(f"  workspace: {self.ws.root}")
        c.print()


def run_pipeline(
    cfg: AutotrainConfig, *, force: bool = False, only: list[str] | None = None
) -> dict[str, Any]:
    return Pipeline(cfg, force=force, only=only).run()
