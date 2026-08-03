"""
Command-line interface.

    autotrain https://example.com
    python main.py https://example.com

That single argument is the whole required input. Everything else has a
default, and the twelve phases run to completion without further interaction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from autotrain import __version__
from autotrain.utils.config import load_config
from autotrain.utils.logging import console, die, setup_logging
from autotrain.utils.paths import make_workspace, slugify_domain
from autotrain.utils.state import PHASE_TITLES, PHASES, StateStore

app = typer.Typer(
    name="autotrain",
    help="Crawl a website and fine-tune a Qwen model on its content. "
         "One argument in, a trained model out.",
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"autotrain {__version__}")
        raise typer.Exit()


@app.command("run")
def run(
    domain: str = typer.Argument(..., help="Domain or URL to crawl, e.g. https://example.com"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML or TOML config file layered over the defaults"
    ),
    set_: list[str] = typer.Option(
        [], "--set", "-s", metavar="KEY=VALUE",
        help="Override any config value, e.g. --set crawler.max_pages=500 (repeatable)",
    ),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Shortcut for crawler.max_pages"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Root directory for runs"),
    phases: Optional[str] = typer.Option(
        None, "--phases", "-p",
        help=f"Comma-separated subset to run. Valid: {','.join(PHASES)}",
    ),
    skip_training: bool = typer.Option(
        False, "--skip-training", help="Run phases 1-7 only (crawl through split)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-run phases even if their outputs already exist"
    ),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore previous run state"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Master seed for a reproducible run"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="DEBUG, INFO, WARNING, ERROR"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve the config, print the plan, and exit"
    ),
    version: bool = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show the version and exit",
    ),
) -> None:
    """Run the full pipeline for DOMAIN."""
    overrides = list(set_)
    if max_pages is not None:
        overrides.append(f"crawler.max_pages={max_pages}")
    if workspace:
        overrides.append(f"runtime.workspace={workspace}")
    if seed is not None:
        overrides.append(f"runtime.seed={seed}")
    if no_resume:
        overrides.append("runtime.resume=false")
    if no_progress:
        overrides.append("runtime.progress=false")
    overrides.append(f"runtime.log_level={log_level.upper()}")

    selected: list[str] | None = None
    if skip_training:
        selected = ["discover", "fetch", "extract", "clean", "generate", "validate", "split", "report"]
    if phases:
        selected = [p.strip() for p in phases.split(",") if p.strip()]

    try:
        cfg = load_config(config, overrides, domain=domain)
    except Exception as exc:
        die(f"configuration error: {exc}")
        return

    setup_logging(
        make_workspace(cfg.runtime.workspace, cfg.domain, cfg.runtime.run_name).logs,
        level=cfg.runtime.log_level,
        progress=cfg.runtime.progress,
    )

    if dry_run:
        _print_plan(cfg, selected)
        raise typer.Exit()

    from autotrain.pipeline import run_pipeline

    try:
        result = run_pipeline(cfg, force=force, only=selected)
    except KeyboardInterrupt:
        console().print("\n[yellow]interrupted[/yellow] — re-run the same command to resume")
        raise typer.Exit(code=130)
    except Exception as exc:
        die(f"{type(exc).__name__}: {exc}")
        return

    raise typer.Exit(code=1 if result["failed_phases"] else 0)


def _print_plan(cfg, selected: list[str] | None) -> None:
    c = console()
    ws = make_workspace(cfg.runtime.workspace, cfg.domain, cfg.runtime.run_name)
    c.rule("[bold]plan[/bold]")
    c.print(f"  domain      {cfg.domain}")
    c.print(f"  workspace   {ws.root}")
    c.print(f"  max pages   {cfg.crawler.max_pages:,}  (depth {cfg.crawler.max_depth}, "
            f"concurrency {cfg.crawler.concurrency})")
    c.print(f"  robots.txt  {'respected' if cfg.crawler.respect_robots else 'IGNORED'}")
    c.print(f"  generator   {cfg.dataset.provider.backend}")
    c.print(f"  types       {', '.join(cfg.dataset.sample_types)}")
    c.print(f"  base model  {cfg.training.base_model}")
    c.print(f"  device      {cfg.training.device}")
    c.print(f"  LoRA        r={cfg.training.lora_r} alpha={cfg.training.lora_alpha} "
            f"epochs={cfg.training.num_train_epochs}")
    c.print(f"  seed        {cfg.runtime.seed}")
    c.print()
    names = selected or list(PHASES)
    for i, name in enumerate(PHASES, 1):
        mark = "run " if name in names else "skip"
        c.print(f"   {i:>2}. [{mark}] {PHASE_TITLES[name]}")
    c.print()


@app.command("status")
def status(
    domain: str = typer.Argument(..., help="Domain of an existing run"),
    workspace: str = typer.Option("runs", "--workspace", "-w"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
) -> None:
    """Show phase status for a previous run."""
    ws = make_workspace(workspace, domain)
    if not ws.run_file.exists():
        die(f"no run found for {domain} in {workspace}/ (looked for {ws.run_file})")
        return

    store = StateStore(ws.run_file)

    if as_json:
        typer.echo(json.dumps(
            {
                "domain": store.state.domain,
                "workspace": str(ws.root),
                "phases": {
                    name: {
                        "status": state.status,
                        "duration_s": state.duration_s,
                        "error": state.error,
                    }
                    for name, state in store.state.phases.items()
                },
                "artifacts": store.state.artifacts,
            },
            indent=2,
        ))
        raise typer.Exit()

    c = console()
    c.rule(f"[bold]{store.state.domain}[/bold]")
    c.print(f"  workspace  {ws.root}")
    c.print(f"  created    {store.state.created_at}")
    c.print(f"  updated    {store.state.updated_at}")
    c.print()
    total = 0.0
    for i, name in enumerate(PHASES, 1):
        state = store.state.phases.get(name)
        if state is None:
            c.print(f"   {i:>2}. [dim]pending   [/dim] {PHASE_TITLES[name]}")
            continue
        total += state.duration_s
        colour = {
            "completed": "green", "failed": "red",
            "skipped": "yellow", "running": "cyan",
        }.get(state.status, "dim")
        line = (f"   {i:>2}. [{colour}]{state.status:<10}[/{colour}] "
                f"{PHASE_TITLES[name]:<34} {state.duration_s:>8.1f}s")
        c.print(line)
        if state.error:
            c.print(f"       [red]{state.error[:150]}[/red]")
    c.print()
    c.print(f"  total      {total / 60:.1f} min")
    report = ws.reports / "final_report.md"
    if report.exists():
        c.print(f"  report     {report}")
    c.print()


@app.command("clean")
def clean(
    domain: str = typer.Argument(..., help="Domain whose run should be cleaned"),
    workspace: str = typer.Option("runs", "--workspace", "-w"),
    phase: Optional[str] = typer.Option(
        None, "--phase", "-p",
        help="Reset only this phase and everything after it (re-runs on the next invocation)",
    ),
    everything: bool = typer.Option(False, "--all", help="Delete the entire run directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt"),
) -> None:
    """Reset phases so they re-run, or delete a run entirely."""
    import shutil

    ws = make_workspace(workspace, domain)
    if not ws.root.exists():
        die(f"no run directory at {ws.root}")
        return

    if everything:
        if not yes and not typer.confirm(f"Delete {ws.root} and everything in it?"):
            raise typer.Exit()
        shutil.rmtree(ws.root)
        console().print(f"deleted {ws.root}")
        raise typer.Exit()

    if not phase:
        die("specify --phase <name> or --all")
        return

    if phase not in PHASES:
        die(f"unknown phase '{phase}'. Valid: {', '.join(PHASES)}")
        return

    store = StateStore(ws.run_file)
    index = PHASES.index(phase)
    reset = PHASES[index:]

    if not yes and not typer.confirm(f"Reset {len(reset)} phase(s) starting at '{phase}'?"):
        raise typer.Exit()

    for name in reset:
        if name in store.state.phases:
            store.state.phases[name].status = "pending"
            store.state.phases[name].error = None
    store.save()
    console().print(f"reset: {', '.join(reset)}")
    console().print("re-run the pipeline to rebuild them")


@app.command("inspect")
def inspect(
    domain: str = typer.Argument(..., help="Domain of an existing run"),
    workspace: str = typer.Option("runs", "--workspace", "-w"),
    samples: int = typer.Option(5, "--samples", "-n", help="Training samples to print"),
) -> None:
    """Print a few generated training samples so you can eyeball dataset quality."""
    from autotrain.utils.text import read_jsonl

    ws = make_workspace(workspace, domain)
    source = ws.train_jsonl if ws.train_jsonl.exists() else ws.dataset_jsonl
    if not source.exists():
        die(f"no dataset found for {domain} (looked in {ws.generated} and {ws.splits})")
        return

    c = console()
    c.rule(f"[bold]{source.name}[/bold]")
    for i, record in enumerate(read_jsonl(source)):
        if i >= samples:
            break
        meta = record.get("meta", {})
        messages = record.get("messages", [])
        instruction = next((m["content"] for m in messages if m["role"] == "user"), "")
        answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        c.print()
        c.print(f"[dim]#{i + 1}  type={meta.get('sample_type')}  "
                f"generator={meta.get('generator')}  source={meta.get('source_url', '')[:60]}[/dim]")
        c.print(f"[bold cyan]Q[/bold cyan] {instruction[:400]}")
        c.print(f"[bold green]A[/bold green] {answer[:700]}")
    c.print()


def main() -> None:
    # A bare `autotrain https://example.com` should work without typing "run".
    argv = sys.argv[1:]
    commands = {"run", "status", "clean", "inspect"}
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
