"""
Typed configuration for the whole pipeline.

Every phase reads its settings from one `AutotrainConfig` object, which is
loaded from a YAML/TOML file and may be overridden per-run from the CLI.
Pydantic validates on construction, so a bad config fails immediately at
startup rather than four hours into a crawl.

Resolution order (later wins):
    1. Field defaults declared here
    2. configs/default.yaml
    3. --config <path> supplied on the CLI
    4. --set key.path=value overrides on the CLI
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class _ListCoercionMixin(BaseModel):
    """
    Accept a bare scalar wherever a list is expected.

    `--set crawler.include_patterns=^https://x\\.com/docs/` produces a string,
    because a single regex has no comma to split on and the CLI cannot know
    the target type. Rejecting it would make the most common single-value
    override fail for a reason the user cannot act on, so every list field
    accepts one value and wraps it.
    """

    @model_validator(mode="before")
    @classmethod
    def _wrap_scalars_into_lists(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name not in out:
                continue
            value = out[name]
            if value is None or isinstance(value, (list, tuple)):
                continue
            annotation = str(field.annotation)
            if annotation.startswith("list[") or annotation.startswith("typing.List["):
                out[name] = [value]
        return out


# ---------------------------------------------------------------------------
# Phase 1-2 — crawling
# ---------------------------------------------------------------------------


class CrawlerConfig(_ListCoercionMixin):
    """Website discovery and HTML collection."""

    respect_robots: bool = True
    user_agent: str = (
        "autotrain-bot/0.1 (+https://github.com/autotrain/autotrain; "
        "documentation corpus builder)"
    )

    # Budgets. max_pages is the hard stop; the crawler is designed to stay
    # responsive at 10k+ so this is a safety rail, not a target.
    max_pages: int = 10_000
    max_depth: int = 12
    max_page_bytes: int = 5_000_000
    max_url_length: int = 2048

    # Concurrency and politeness. delay_seconds is applied per-host between
    # request starts; robots.txt Crawl-delay overrides it when larger.
    concurrency: int = 8
    delay_seconds: float = 0.0
    request_timeout: float = 30.0
    retries: int = 3
    retry_backoff: float = 1.5

    # Scope control.
    include_subdomains: bool = False
    follow_sitemap: bool = True
    allowed_content_types: list[str] = Field(
        default_factory=lambda: ["text/html", "application/xhtml+xml"]
    )
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            # Binary and asset extensions that will never contain prose.
            r"\.(?:png|jpe?g|gif|webp|svg|ico|bmp|tiff?)(?:$|\?)",
            r"\.(?:css|js|mjs|map|woff2?|ttf|eot|otf)(?:$|\?)",
            r"\.(?:zip|tar|gz|bz2|xz|7z|rar|dmg|exe|msi|deb|rpm)(?:$|\?)",
            r"\.(?:mp4|webm|avi|mov|mp3|wav|ogg|flac)(?:$|\?)",
            # Endless-space traps: calendars, faceted search, print views.
            r"/(?:cdn-cgi|wp-json|xmlrpc\.php)(?:/|$)",
            r"[?&](?:replytocom|share|print)=",
        ]
    )

    # Loop detection. A URL whose path repeats the same segment more than
    # this many times is a crawler trap (/a/b/a/b/a/b/...).
    max_repeated_path_segments: int = 3
    max_path_depth: int = 24
    max_query_params: int = 8

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def _compilable(cls, v: list[str]) -> list[str]:
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return v

    @field_validator("concurrency")
    @classmethod
    def _sane_concurrency(cls, v: int) -> int:
        if not 1 <= v <= 64:
            raise ValueError("concurrency must be between 1 and 64")
        return v


# ---------------------------------------------------------------------------
# Phase 3 — content extraction
# ---------------------------------------------------------------------------


class ExtractorConfig(_ListCoercionMixin):
    """HTML -> clean Markdown."""

    # trafilatura favours recall at the cost of some boilerplate when
    # favour_recall is on. Documentation sites benefit: sidebars are already
    # stripped by its content-density model, and we would rather keep an
    # unusual <aside> containing a note than lose it.
    favour_recall: bool = True
    include_tables: bool = True
    include_formatting: bool = True
    include_links: bool = False
    include_images: bool = False
    include_comments: bool = False

    # Minimum extracted characters for a page to be considered real content.
    min_extracted_chars: int = 200

    # Fallback chain when trafilatura returns nothing (heavily templated or
    # JS-rendered pages). Each is tried in order.
    enable_fallback: bool = True

    # Selectors force-removed before extraction, on top of trafilatura's own
    # model. These catch cookie banners and chrome that density scoring keeps.
    strip_selectors: list[str] = Field(
        default_factory=lambda: [
            "nav", "header", "footer", "aside",
            "script", "style", "noscript", "svg", "iframe", "form",
            '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
            '[role="search"]', '[role="complementary"]',
            '[aria-hidden="true"]', "[hidden]",
            ".cookie", ".cookies", ".cookie-banner", ".cookie-consent",
            ".gdpr", ".consent", ".newsletter", ".subscribe",
            ".advert", ".advertisement", ".ads", ".ad-slot", ".promo",
            ".breadcrumb", ".breadcrumbs", ".pagination", ".pager",
            ".sidebar", ".side-nav", ".toc", ".table-of-contents",
            ".social", ".share", ".sharing", ".related-posts",
            ".skip-link", ".screen-reader-text", ".sr-only", ".visually-hidden",
            "#cookie-banner", "#onetrust-consent-sdk", "#CybotCookiebotDialog",
        ]
    )


# ---------------------------------------------------------------------------
# Phase 4 — dataset cleaning
# ---------------------------------------------------------------------------


class CleanerConfig(BaseModel):
    """Corpus-level cleaning across all extracted documents."""

    min_doc_chars: int = 300
    min_doc_words: int = 50

    # Exact duplicate removal via content hash is always on. Near-duplicate
    # removal uses SimHash: two docs whose 64-bit fingerprints differ by
    # <= threshold bits are considered the same document.
    enable_near_dedupe: bool = True
    simhash_threshold: int = 3

    # A line appearing verbatim on more than this fraction of pages is
    # site chrome that survived extraction (breadcrumbs, "Edit this page",
    # copyright). Removed corpus-wide.
    boilerplate_line_ratio: float = 0.35
    boilerplate_min_docs: int = 8
    # Never strip a repeated line longer than this — long repeated text is
    # more likely genuine shared documentation than chrome.
    boilerplate_max_line_chars: int = 200

    normalize_whitespace: bool = True
    strip_trailing_nav_lists: bool = True

    @field_validator("simhash_threshold")
    @classmethod
    def _threshold_range(cls, v: int) -> int:
        if not 0 <= v <= 16:
            raise ValueError("simhash_threshold must be between 0 and 16")
        return v


# ---------------------------------------------------------------------------
# Phase 5 — instruction dataset generation
# ---------------------------------------------------------------------------

SampleType = Literal[
    "doc_qa",
    "summarization",
    "explanation",
    "step_by_step",
    "api_understanding",
    "code_explanation",
    "concept_learning",
    "troubleshooting",
]

ALL_SAMPLE_TYPES: tuple[str, ...] = (
    "doc_qa",
    "summarization",
    "explanation",
    "step_by_step",
    "api_understanding",
    "code_explanation",
    "concept_learning",
    "troubleshooting",
)


class ProviderConfig(_ListCoercionMixin):
    """Which engine turns documents into instruction pairs."""

    # "auto" probes each backend in `fallback_order` and uses the first that
    # answers. "structural" needs no LLM at all and always works.
    backend: Literal["auto", "ollama", "openai", "anthropic", "structural"] = "auto"
    fallback_order: list[str] = Field(
        default_factory=lambda: ["ollama", "openai", "anthropic", "structural"]
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # OpenAI-compatible (OpenAI, vLLM, LM Studio, together.ai, ...)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"

    # Anthropic
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-sonnet-5"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"

    # Shared generation settings. Temperature is deliberately high: we want
    # varied phrasings of the same underlying fact, not the single most
    # likely question.
    temperature: float = 0.7
    max_output_tokens: int = 2048
    timeout: float = 180.0
    max_retries: int = 3
    concurrency: int = 4


class DatasetConfig(_ListCoercionMixin):
    """Phase 5 output shaping."""

    provider: ProviderConfig = Field(default_factory=ProviderConfig)

    sample_types: list[str] = Field(default_factory=lambda: list(ALL_SAMPLE_TYPES))
    # Pairs requested per sample type per document. The real yield is lower:
    # types that do not apply to a document (no code -> no code_explanation)
    # are skipped.
    pairs_per_type: int = 2

    # Documents longer than this are split into windows and each window is
    # processed separately, so a 40-page API reference yields proportionally
    # more samples than a one-paragraph page.
    max_chars_per_window: int = 6000
    window_overlap_chars: int = 400
    max_windows_per_doc: int = 8

    system_prompt: str = (
        "You are a precise technical assistant with deep knowledge of "
        "{site_name}. Answer using that knowledge. Use numbered steps for "
        "procedures, bullet points for lists, and fenced code blocks for code."
    )

    min_answer_chars: int = 40
    max_answer_chars: int = 8000


# ---------------------------------------------------------------------------
# Phase 6-7 — validation and splitting
# ---------------------------------------------------------------------------


class ValidationConfig(BaseModel):
    min_instruction_chars: int = 10
    min_answer_chars: int = 40
    max_sample_chars: int = 24_000
    # Fraction of non-ASCII replacement/control characters above which a
    # sample is treated as encoding-corrupted.
    max_corruption_ratio: float = 0.02
    drop_duplicates: bool = True
    # Fail the pipeline if more than this fraction of samples are invalid —
    # usually means the generator backend was misbehaving.
    max_invalid_ratio: float = 0.5


class SplitConfig(BaseModel):
    train_ratio: float = 0.9
    seed: int = 42
    # Group by source document so that pairs generated from the same page
    # never straddle the train/val boundary. Without this, validation loss is
    # measured on text the model memorised from the training half.
    group_by_source: bool = True

    @field_validator("train_ratio")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        if not 0.5 <= v <= 0.99:
            raise ValueError("train_ratio must be between 0.5 and 0.99")
        return v


# ---------------------------------------------------------------------------
# Phase 8 — training
# ---------------------------------------------------------------------------


class TrainingConfig(_ListCoercionMixin):
    """QLoRA / LoRA fine-tuning."""

    # Model. The CPU path needs a small model to be tractable; the CUDA path
    # defaults to 7B. `auto` picks based on detected hardware.
    base_model: str = "auto"
    cuda_model: str = "Qwen/Qwen2.5-7B-Instruct"
    cpu_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    # Quantisation. 4-bit requires bitsandbytes + CUDA; the trainer disables
    # it automatically elsewhere rather than crashing.
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_bias: Literal["none", "all", "lora_only"] = "none"

    # Schedule
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optim: str = "auto"  # paged_adamw_8bit on CUDA, adamw_torch elsewhere

    max_seq_length: int = 2048
    packing: bool = False
    # Compute loss on response tokens only. The model is not graded on
    # reproducing the prompt it was given.
    train_on_completions_only: bool = True

    # Memory / speed
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    use_flash_attention: bool = True  # silently ignored when unavailable

    # Checkpointing and logging
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 100
    eval_strategy: str = "steps"
    resume_from_checkpoint: bool = True

    seed: int = 42
    # Caps for smoke runs; 0 disables.
    max_steps: int = 0
    max_train_samples: int = 0
    max_eval_samples: int = 200


# ---------------------------------------------------------------------------
# Phase 9-11 — evaluation, export, benchmark
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    compute_perplexity: bool = True
    num_sample_generations: int = 12
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    # Hallucination proxy: fraction of the answer's content words that appear
    # in the source corpus. Low grounding on a closed-domain fine-tune means
    # the model is inventing.
    grounding_check: bool = True
    grounding_min_score: float = 0.55
    # Consistency: ask the same question twice at temperature > 0 and compare.
    consistency_check: bool = True
    consistency_trials: int = 2


class ExportConfig(BaseModel):
    export_adapter: bool = True
    export_merged: bool = True
    export_gguf: bool = False
    gguf_quantisation: str = "Q4_K_M"
    llama_cpp_dir: str = "~/llama.cpp"
    export_ollama: bool = True
    ollama_model_name: str = "auto"  # derived from the domain
    merged_dtype: Literal["float16", "bfloat16", "float32"] = "float16"


class BenchmarkConfig(_ListCoercionMixin):
    enabled: bool = True
    compare_with_base: bool = True
    num_questions: int = 20
    max_new_tokens: int = 384
    # Questions are drawn from the held-out validation split so the benchmark
    # measures generalisation, not recall of the training set.
    source: Literal["validation", "custom"] = "validation"
    custom_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


class RuntimeConfig(_ListCoercionMixin):
    workspace: str = "runs"
    # Named after the domain unless overridden, so re-running the same site
    # resumes into the same directory.
    run_name: str = "auto"
    resume: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    progress: bool = True
    seed: int = 42
    # Re-fetch pages whose cached copy is older than this. 0 = cache forever.
    cache_ttl_hours: int = 0
    # Phases to run. Useful for re-running one stage without the rest.
    phases: list[str] = Field(default_factory=list)  # empty = all


class AutotrainConfig(BaseModel):
    """Root configuration object."""

    domain: str = ""
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    cleaner: CleanerConfig = Field(default_factory=CleanerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _propagate_seed(self) -> AutotrainConfig:
        """One --seed on the CLI should make the whole run reproducible."""
        if self.runtime.seed != 42:
            self.split.seed = self.runtime.seed
            self.training.seed = self.runtime.seed
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into base, returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".toml":
        return tomllib.loads(text)
    return yaml.safe_load(text) or {}


def _coerce_scalar(raw: str) -> Any:
    """Turn a CLI override string into the most plausible Python value."""
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` strings onto a nested dict."""
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must be of the form key.path=value")
        key_path, _, raw = item.partition("=")
        cursor: dict[str, Any] = out
        parts = key_path.strip().split(".")
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = _coerce_scalar(raw.strip())
    return out


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    domain: str | None = None,
) -> AutotrainConfig:
    """
    Build the effective configuration.

    Reads configs/default.yaml when present, layers an explicit --config file
    on top, then applies --set overrides, then the positional domain.
    """
    data: dict[str, Any] = {}

    if DEFAULT_CONFIG_PATH.exists():
        data = _deep_merge(data, _read_structured(DEFAULT_CONFIG_PATH))

    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        data = _deep_merge(data, _read_structured(p))

    # Environment escape hatch for containers: AUTOTRAIN_SET="a.b=1;c.d=2"
    env_set = os.environ.get("AUTOTRAIN_SET", "").strip()
    if env_set:
        data = apply_overrides(data, [s for s in env_set.split(";") if s.strip()])

    if overrides:
        data = apply_overrides(data, overrides)

    if domain:
        data["domain"] = domain

    return AutotrainConfig.model_validate(data)
