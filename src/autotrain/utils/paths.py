"""
Workspace layout.

Every run lives under a single directory keyed by the domain, so re-running
the same site resumes in place and two different sites never collide.

    runs/example.com/
        run.json                  run metadata + phase state
        logs/                     one log file per phase
        discovery/
            urls.json
            sitemap.json
            crawl_report.json
        dataset/
            raw_html/page_0001.html
            raw_html/index.jsonl  per-page fetch metadata
        clean_docs/page_0001.md
        clean_dataset/
            docs/page_0001.md
            corpus.jsonl
            clean_report.json
        generated/
            dataset.jsonl
            generation_state.json
        validated/
            dataset.jsonl
            validation_report.json
        splits/
            train.jsonl
            validation.jsonl
        training/
            checkpoints/
            adapter/
            trainer_state.json
        evaluation/
        exports/
            adapter/ merged/ gguf/ ollama/
        benchmark/
            benchmark_report.md
        reports/
            final_report.md
            final_report.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


def url_to_slug(url: str, max_length: int = 80) -> str:
    """Turn a URL path into a safe, human-readable filename stem.

    https://example.com/about/team/ -> about__team
    https://example.com/            -> index
    https://example.com/products/x  -> products__x
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return "index"
    parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p).strip("._-") for p in path.split("/")]
    parts = [p for p in parts if p]
    slug = "__".join(parts) if parts else "index"
    return slug[:max_length] or "index"

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_domain(domain: str) -> str:
    """example.com/docs -> example.com_docs ; https://a.b/ -> a.b"""
    parsed = urlparse(domain if "://" in domain else f"https://{domain}")
    host = (parsed.netloc or parsed.path).strip("/")
    path = parsed.path.strip("/") if parsed.netloc else ""
    raw = f"{host}_{path}" if path else host
    slug = _SAFE.sub("_", raw).strip("._-")
    return slug or "site"


@dataclass(frozen=True)
class Workspace:
    """Resolved absolute paths for one run. All directories are created eagerly."""

    root: Path

    # -- top level ---------------------------------------------------------
    @property
    def run_file(self) -> Path:
        return self.root / "run.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # -- phase 1 -----------------------------------------------------------
    @property
    def discovery(self) -> Path:
        return self.root / "discovery"

    @property
    def urls_json(self) -> Path:
        return self.discovery / "urls.json"

    @property
    def sitemap_json(self) -> Path:
        return self.discovery / "sitemap.json"

    @property
    def crawl_report(self) -> Path:
        return self.discovery / "crawl_report.json"

    # -- phase 2 -----------------------------------------------------------
    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def raw_html(self) -> Path:
        return self.dataset / "raw_html"

    @property
    def fetch_index(self) -> Path:
        return self.raw_html / "index.jsonl"

    # -- phase 3 -----------------------------------------------------------
    @property
    def clean_docs(self) -> Path:
        return self.root / "clean_docs"

    @property
    def extract_report(self) -> Path:
        return self.clean_docs / "extract_report.json"

    # -- phase 4 -----------------------------------------------------------
    @property
    def clean_dataset(self) -> Path:
        return self.root / "clean_dataset"

    @property
    def clean_dataset_docs(self) -> Path:
        return self.clean_dataset / "docs"

    @property
    def corpus_jsonl(self) -> Path:
        return self.clean_dataset / "corpus.jsonl"

    @property
    def clean_report(self) -> Path:
        return self.clean_dataset / "clean_report.json"

    # -- phase 5 -----------------------------------------------------------
    @property
    def generated(self) -> Path:
        return self.root / "generated"

    @property
    def dataset_jsonl(self) -> Path:
        return self.generated / "dataset.jsonl"

    @property
    def generation_state(self) -> Path:
        return self.generated / "generation_state.json"

    # -- phase 6 -----------------------------------------------------------
    @property
    def validated(self) -> Path:
        return self.root / "validated"

    @property
    def validated_jsonl(self) -> Path:
        return self.validated / "dataset.jsonl"

    @property
    def validation_report(self) -> Path:
        return self.validated / "validation_report.json"

    # -- phase 7 -----------------------------------------------------------
    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def train_jsonl(self) -> Path:
        return self.splits / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.splits / "validation.jsonl"

    # -- phase 8 -----------------------------------------------------------
    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def checkpoints(self) -> Path:
        return self.training / "checkpoints"

    @property
    def adapter(self) -> Path:
        return self.training / "adapter"

    @property
    def train_metrics(self) -> Path:
        return self.training / "metrics.json"

    # -- phase 9 -----------------------------------------------------------
    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    # -- phase 10 ----------------------------------------------------------
    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def export_adapter(self) -> Path:
        return self.exports / "adapter"

    @property
    def export_merged(self) -> Path:
        return self.exports / "merged"

    @property
    def export_gguf(self) -> Path:
        return self.exports / "gguf"

    @property
    def export_ollama(self) -> Path:
        return self.exports / "ollama"

    # -- phase 11 ----------------------------------------------------------
    @property
    def benchmark(self) -> Path:
        return self.root / "benchmark"

    @property
    def benchmark_report(self) -> Path:
        return self.benchmark / "benchmark_report.md"

    # -- phase 12 ----------------------------------------------------------
    @property
    def reports(self) -> Path:
        return self.root / "reports"

    # ---------------------------------------------------------------------

    def ensure(self) -> Workspace:
        """Create every directory. Idempotent."""
        for path in (
            self.root, self.logs, self.discovery, self.dataset, self.raw_html,
            self.clean_docs, self.clean_dataset, self.clean_dataset_docs,
            self.generated, self.validated, self.splits, self.training,
            self.checkpoints, self.adapter, self.evaluation, self.exports,
            self.benchmark, self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def page_html(self, index: int) -> Path:
        return self.raw_html / f"page_{index:04d}.html"

    def page_md(self, index: int) -> Path:
        return self.clean_docs / f"page_{index:04d}.md"

    def clean_md(self, index: int) -> Path:
        return self.clean_dataset_docs / f"page_{index:04d}.md"


def make_workspace(workspace_root: str | Path, domain: str, run_name: str = "auto") -> Workspace:
    name = slugify_domain(domain) if run_name in ("auto", "", None) else run_name
    return Workspace(Path(workspace_root).expanduser().resolve() / name).ensure()
