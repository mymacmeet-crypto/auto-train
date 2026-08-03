"""
Run state and resume.

`run.json` is the single source of truth for what has already been done.
Each phase records its status, timings, and a small result summary. On
restart the orchestrator skips any phase marked `completed` whose declared
output artifacts still exist on disk — so deleting an artifact is a valid
way to force one phase to re-run.

Writes are atomic (temp file + os.replace) because a Ctrl-C during a state
write would otherwise leave an unparseable run.json and strand the run.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

PhaseStatus = Literal["pending", "running", "completed", "failed", "skipped"]

# Canonical phase order. The orchestrator walks this list.
PHASES: tuple[str, ...] = (
    "discover",     # 1  website discovery
    "fetch",        # 2  HTML collection
    "extract",      # 3  content extraction
    "clean",        # 4  dataset cleaning
    "generate",     # 5  instruction dataset generation
    "validate",     # 6  dataset validation
    "split",        # 7  train/val split
    "train",        # 8  QLoRA fine-tuning
    "evaluate",     # 9  evaluation
    "export",       # 10 model export
    "benchmark",    # 11 automated testing
    "report",       # 12 final report
)

PHASE_TITLES: dict[str, str] = {
    "discover": "Website discovery",
    "fetch": "HTML collection",
    "extract": "Content extraction",
    "clean": "Dataset cleaning",
    "generate": "Instruction dataset generation",
    "validate": "Dataset validation",
    "split": "Dataset splitting",
    "train": "QLoRA fine-tuning",
    "evaluate": "Evaluation",
    "export": "Model export",
    "benchmark": "Automated benchmark",
    "report": "Final report",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseState:
    name: str
    status: PhaseStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float = 0.0
    attempts: int = 0
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    domain: str = ""
    run_name: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    autotrain_version: str = "0.1.0"
    config_digest: str = ""
    phases: dict[str, PhaseState] = field(default_factory=dict)
    # Free-form values phases publish for later phases and the final report.
    artifacts: dict[str, Any] = field(default_factory=dict)

    def phase(self, name: str) -> PhaseState:
        if name not in self.phases:
            self.phases[name] = PhaseState(name=name)
        return self.phases[name]


class StateStore:
    """Load/save `run.json` with atomic writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def _load(self) -> RunState:
        if not self.path.exists():
            return RunState()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt state file must not brick the run; start fresh but
            # keep the old one for forensics.
            backup = self.path.with_suffix(f".corrupt.{int(time.time())}.json")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return RunState()

        phases = {
            name: PhaseState(**data) for name, data in (raw.get("phases") or {}).items()
        }
        raw["phases"] = phases
        known = {f for f in RunState.__dataclass_fields__}
        return RunState(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        self.state.updated_at = _now()
        payload = asdict(self.state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- phase transitions -------------------------------------------------

    def start(self, name: str) -> PhaseState:
        ph = self.state.phase(name)
        ph.status = "running"
        ph.started_at = _now()
        ph.attempts += 1
        ph.error = None
        self.save()
        return ph

    def complete(self, name: str, summary: dict[str, Any] | None = None) -> None:
        ph = self.state.phase(name)
        ph.status = "completed"
        ph.finished_at = _now()
        ph.error = None
        if summary:
            ph.summary = summary
        if ph.started_at:
            started = datetime.fromisoformat(ph.started_at)
            ph.duration_s = round(
                (datetime.now(timezone.utc) - started).total_seconds(), 2
            )
        self.save()

    def fail(self, name: str, error: str) -> None:
        ph = self.state.phase(name)
        ph.status = "failed"
        ph.finished_at = _now()
        ph.error = error
        self.save()

    def skip(self, name: str, reason: str = "") -> None:
        ph = self.state.phase(name)
        ph.status = "skipped"
        ph.summary = {"reason": reason} if reason else ph.summary
        self.save()

    def publish(self, key: str, value: Any) -> None:
        """Record a cross-phase artifact (paths, counts, model ids)."""
        self.state.artifacts[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.artifacts.get(key, default)

    # -- resume ------------------------------------------------------------

    def should_run(
        self, name: str, outputs: list[Path], resume: bool, force: bool = False
    ) -> bool:
        """
        A phase is skipped only when it completed AND all of its declared
        outputs are still present. Deleting an output re-arms the phase.
        """
        if force or not resume:
            return True
        ph = self.state.phase(name)
        if ph.status != "completed":
            return True
        for path in outputs:
            if not path.exists():
                return True
            # An empty artifact is a failed artifact.
            try:
                if path.is_file() and path.stat().st_size == 0:
                    return True
            except OSError:
                return True
        return False
