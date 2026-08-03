"""
Structured logging and progress.

Two sinks per run:
  - console, via rich, human-readable and colourised
  - one JSON-lines file per phase under runs/<site>/logs/, machine-readable
    and complete (DEBUG level regardless of console verbosity)

The JSONL files are what the final report reads back to reconstruct timings
and error counts, so every phase logs its own summary through here.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_console: Console | None = None
_progress_enabled = True


def console() -> Console:
    global _console
    if _console is None:
        # stderr keeps stdout clean for piped JSON output.
        _console = Console(stderr=True, highlight=False, soft_wrap=False)
    return _console


class JsonlHandler(logging.Handler):
    """Write every record as one JSON object per line."""

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.DEBUG)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "extra_fields", None)
            if extra:
                payload["fields"] = extra
            if record.exc_info:
                payload["exception"] = self.format(record)
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:  # never let logging break the pipeline
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(log_dir: Path, level: str = "INFO", progress: bool = True) -> None:
    """Install the console handler. Phase files are attached by `phase_logger`."""
    global _progress_enabled
    _progress_enabled = progress

    root = logging.getLogger("autotrain")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    rich_handler = RichHandler(
        console=console(),
        show_time=True,
        show_path=False,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=False,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    root.addHandler(rich_handler)

    log_dir.mkdir(parents=True, exist_ok=True)
    combined = JsonlHandler(log_dir / "pipeline.jsonl")
    root.addHandler(combined)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"autotrain.{name}")


def log_fields(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a message carrying structured fields into the JSONL sink."""
    logger.log(level, message, extra={"extra_fields": fields})


@contextmanager
def phase_logger(log_dir: Path, phase: str) -> Iterator[logging.Logger]:
    """
    Attach a per-phase JSONL file for the duration of the phase and log
    start/finish with wall-clock duration.
    """
    root = logging.getLogger("autotrain")
    handler = JsonlHandler(log_dir / f"{phase}.jsonl")
    root.addHandler(handler)
    logger = get_logger(phase)
    started = time.monotonic()
    log_fields(logger, logging.INFO, f"phase '{phase}' started", phase=phase)
    try:
        yield logger
    except Exception as exc:
        log_fields(
            logger, logging.ERROR, f"phase '{phase}' failed: {exc}",
            phase=phase, duration_s=round(time.monotonic() - started, 2),
            error=str(exc), error_type=type(exc).__name__,
        )
        raise
    else:
        log_fields(
            logger, logging.INFO, f"phase '{phase}' finished",
            phase=phase, duration_s=round(time.monotonic() - started, 2),
        )
    finally:
        root.removeHandler(handler)
        handler.close()


def make_progress() -> Progress:
    """A progress bar with counts and ETA, disabled when --no-progress is set."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console(),
        disable=not _progress_enabled,
        transient=False,
    )


def print_banner(title: str, subtitle: str = "") -> None:
    c = console()
    c.rule(f"[bold]{title}[/bold]" + (f"  [dim]{subtitle}[/dim]" if subtitle else ""))


def die(message: str, code: int = 1) -> None:
    console().print(f"[bold red]error[/bold red] {message}")
    sys.exit(code)
