"""
Generation backends.

Two layers:

  LLMBackend        raw text-in/text-out against Ollama, an OpenAI-compatible
                    endpoint, or Anthropic. Interchangeable; the builder never
                    knows which one it holds.

  SampleGenerator   turns a document window into instruction pairs. Either
                    LLMSampleGenerator (wraps a backend and parses its JSON)
                    or StructuralSampleGenerator (no LLM at all).

The structural generator exists so the pipeline is never blocked on an API
key, a GPU, or a running Ollama daemon. It is genuinely lower quality — it
recombines the document's own sentences rather than reasoning about them —
but it produces valid, grounded, non-hallucinated training data, and it is
free and deterministic. `backend: auto` prefers a real LLM and silently falls
back to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from autotrain.dataset_builder.prompts import (
    DocSignals,
    SampleTypeSpec,
    analyse,
    build_prompt,
)
from autotrain.utils.config import ProviderConfig
from autotrain.utils.logging import get_logger

logger = get_logger("dataset.providers")


@dataclass
class Pair:
    """One generated instruction/response pair."""

    instruction: str
    output: str
    sample_type: str
    source_index: int
    source_url: str
    source_title: str
    window: int = 0
    generator: str = ""

    def is_valid(self, min_answer_chars: int, max_answer_chars: int) -> bool:
        if not self.instruction.strip() or not self.output.strip():
            return False
        if len(self.output) < min_answer_chars:
            return False
        if len(self.output) > max_answer_chars:
            return False
        # A model echoing the prompt back is a common failure mode.
        if self.instruction.strip() == self.output.strip():
            return False
        return True


# ===========================================================================
# LLM backends
# ===========================================================================


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    async def available(self, client: httpx.AsyncClient) -> bool:
        """Cheap liveness probe. Must not raise."""

    @abstractmethod
    async def complete(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        """Return the model's raw text response."""

    def describe(self) -> str:
        return self.name


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: ProviderConfig) -> None:
        self.base_url = cfg.ollama_base_url.rstrip("/")
        self.model = cfg.ollama_model

    async def available(self, client: httpx.AsyncClient) -> bool:
        try:
            response = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
            if response.status_code != 200:
                return False
            tags = response.json().get("models", [])
            names = {m.get("name", "") for m in tags}
            if not names:
                return False
            # Accept an exact match or the same model under a different tag.
            base = self.model.split(":")[0]
            if self.model in names or any(n.split(":")[0] == base for n in names):
                return True
            logger.warning(
                f"ollama is running but '{self.model}' is not pulled "
                f"(available: {', '.join(sorted(names)[:5])})"
            )
            return False
        except Exception:
            return False

    async def complete(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 8192,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()

    def describe(self) -> str:
        return f"ollama:{self.model}"


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, cfg: ProviderConfig) -> None:
        self.base_url = cfg.openai_base_url.rstrip("/")
        self.model = cfg.openai_model
        self.api_key = os.environ.get(cfg.openai_api_key_env, "")

    async def available(self, client: httpx.AsyncClient) -> bool:
        # Local OpenAI-compatible servers (vLLM, LM Studio) often need no key.
        is_local = any(h in self.base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not self.api_key and not is_local:
            return False
        try:
            response = await client.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=8.0
            )
            return response.status_code < 400
        except Exception:
            return False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    def describe(self) -> str:
        return f"openai:{self.model}"


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, cfg: ProviderConfig) -> None:
        self.base_url = cfg.anthropic_base_url.rstrip("/")
        self.model = cfg.anthropic_model
        self.api_key = os.environ.get(cfg.anthropic_api_key_env, "")

    async def available(self, client: httpx.AsyncClient) -> bool:
        if not self.api_key:
            return False
        try:
            # A 1-token call is the cheapest reliable auth+model check.
            response = await client.post(
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=15.0,
            )
            return response.status_code < 400
        except Exception:
            return False

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    async def complete(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        response = await client.post(
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        blocks = response.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

    def describe(self) -> str:
        return f"anthropic:{self.model}"


BACKENDS: dict[str, type[LLMBackend]] = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
}


# ===========================================================================
# JSON recovery
# ===========================================================================

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_pairs(raw: str) -> list[dict[str, str]]:
    """
    Recover a list of {instruction, output} objects from a model response.

    Models fail at this in predictable ways, so we try progressively looser
    strategies rather than discarding the generation on the first error:
      1. the whole response is valid JSON
      2. it is wrapped in a ```json fence
      3. the first '[' to the last ']' is valid JSON
      4. concatenated bare objects, extracted one balanced {...} at a time
    """
    if not raw or not raw.strip():
        return []

    candidates: list[str] = [raw.strip()]

    fenced = _FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        pairs = _coerce(data)
        if pairs:
            return pairs

    # Last resort: scan for balanced objects.
    return _coerce(_scan_objects(raw))


def _scan_objects(raw: str) -> list[Any]:
    objects: list[Any] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        objects.append(json.loads(raw[start : i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return objects


# Key aliases models produce instead of the requested ones.
_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "input", "query", "task")
_OUTPUT_KEYS = ("output", "answer", "response", "completion", "result", "content")


def _coerce(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        # A single object, or {"samples": [...]}.
        for key in ("samples", "pairs", "data", "examples", "items", "results"):
            if key in data and isinstance(data[key], list):
                return _coerce(data[key])
        data = [data]

    if not isinstance(data, list):
        return []

    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = ""
        output = ""
        lowered = {str(k).lower(): v for k, v in item.items()}
        for key in _INSTRUCTION_KEYS:
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                instruction = value.strip()
                break
        for key in _OUTPUT_KEYS:
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                output = value.strip()
                break
        if instruction and output:
            out.append({"instruction": instruction, "output": output})
    return out


# ===========================================================================
# Sample generators
# ===========================================================================


class SampleGenerator(ABC):
    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        client: httpx.AsyncClient | None,
        spec: SampleTypeSpec,
        *,
        site_name: str,
        title: str,
        url: str,
        content: str,
        signals: DocSignals,
        n: int,
        source_index: int,
        window: int,
    ) -> list[Pair]:
        ...


class LLMSampleGenerator(SampleGenerator):
    """Prompts a backend and parses the JSON it returns."""

    def __init__(self, backend: LLMBackend, cfg: ProviderConfig) -> None:
        self.backend = backend
        self.cfg = cfg
        self.name = backend.describe()
        self.failures = 0
        self.calls = 0

    async def generate(
        self,
        client: httpx.AsyncClient | None,
        spec: SampleTypeSpec,
        *,
        site_name: str,
        title: str,
        url: str,
        content: str,
        signals: DocSignals,
        n: int,
        source_index: int,
        window: int,
    ) -> list[Pair]:
        assert client is not None
        prompt = build_prompt(
            spec, site_name=site_name, title=title, url=url, content=content, n=n
        )

        raw = ""
        for attempt in range(self.cfg.max_retries + 1):
            self.calls += 1
            try:
                raw = await self.backend.complete(
                    client,
                    prompt,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_output_tokens,
                    timeout=self.cfg.timeout,
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (429, 500, 502, 503, 529) and attempt < self.cfg.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 30.0))
                    continue
                logger.debug(f"{self.name} HTTP {status} on {url}")
                self.failures += 1
                return []
            except Exception as exc:
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(min(2.0 ** attempt, 30.0))
                    continue
                logger.debug(f"{self.name} failed on {url}: {type(exc).__name__}: {exc}")
                self.failures += 1
                return []

            parsed = parse_pairs(raw)
            if parsed:
                return [
                    Pair(
                        instruction=p["instruction"],
                        output=p["output"],
                        sample_type=spec.key,
                        source_index=source_index,
                        source_url=url,
                        source_title=title,
                        window=window,
                        generator=self.name,
                    )
                    for p in parsed[:n]
                ]

            # Unparseable: retry once, the temperature makes it a real reroll.
            if attempt < self.cfg.max_retries:
                continue

        self.failures += 1
        logger.debug(f"{self.name} produced no parseable pairs for {spec.key} @ {url}")
        return []


# ---------------------------------------------------------------------------
# Structural generator
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(`\[])")


def _sentences(text: str) -> list[str]:
    plain = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    plain = re.sub(r"^[|#>*\-\d.]+\s*", "", plain, flags=re.MULTILINE)
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(plain) if len(s.strip()) > 30]
    return parts


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split a document into (heading, body) sections."""
    lines = markdown.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current: list[str] = []
    in_code = False

    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                if current_heading or current:
                    sections.append((current_heading, current))
                current_heading = match.group(2).strip()
                current = []
                continue
        current.append(line)

    if current_heading or current:
        sections.append((current_heading, current))

    return [(h, "\n".join(b).strip()) for h, b in sections if "\n".join(b).strip()]


class StructuralSampleGenerator(SampleGenerator):
    """
    Derives instruction pairs from document structure — no LLM involved.

    Every pair is grounded by construction: the answer is verbatim text from
    the source document. That makes it hallucination-free but stylistically
    flat, which is the trade for needing no infrastructure.
    """

    name = "structural"

    async def generate(
        self,
        client: httpx.AsyncClient | None,
        spec: SampleTypeSpec,
        *,
        site_name: str,
        title: str,
        url: str,
        content: str,
        signals: DocSignals,
        n: int,
        source_index: int,
        window: int,
    ) -> list[Pair]:
        builder = getattr(self, f"_build_{spec.key}", None)
        if builder is None:
            return []
        try:
            raw_pairs = builder(site_name, title, content, signals, n)
        except Exception as exc:
            logger.debug(f"structural {spec.key} failed on {url}: {exc}")
            return []

        return [
            Pair(
                instruction=instruction,
                output=output,
                sample_type=spec.key,
                source_index=source_index,
                source_url=url,
                source_title=title,
                window=window,
                generator=self.name,
            )
            for instruction, output in raw_pairs[:n]
        ]

    # -- per-type builders ------------------------------------------------

    def _build_doc_qa(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        for heading, body in _sections(text):
            if not heading or len(body) < 120:
                continue
            subject = heading.rstrip("?:.")
            question = (
                f"What is {subject} in {site}?"
                if not heading.endswith("?")
                else heading
            )
            out.append((question, self._trim(body, 1400)))
            if len(out) >= n:
                break

        if len(out) < n:
            sentences = _sentences(text)
            if sentences:
                out.append(
                    (
                        f"What does the {title} documentation for {site} cover?",
                        self._trim(" ".join(sentences[:6]), 1200),
                    )
                )
        return out

    def _build_summarization(self, site: str, title: str, text: str, s: DocSignals, n: int):
        sentences = _sentences(text)
        if len(sentences) < 3:
            return []
        # Lead sentences plus the first sentence under each heading approximate
        # an extractive summary.
        picked = sentences[:3]
        for heading, body in _sections(text)[:4]:
            body_sentences = _sentences(body)
            if body_sentences and body_sentences[0] not in picked:
                picked.append(body_sentences[0])
        summary = " ".join(picked[:7])
        subject = title or site
        return [(f"Summarise {subject} in {site}.", self._trim(summary, 1400))]

    def _build_explanation(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        for heading, body in _sections(text):
            if not heading or len(body) < 200:
                continue
            out.append(
                (
                    f"Explain how {heading.rstrip('?:.')} works in {site}.",
                    self._trim(body, 1800),
                )
            )
            if len(out) >= n:
                break
        return out

    def _build_step_by_step(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        for heading, body in _sections(text):
            steps = re.findall(r"^\s*(?:\d+[.)]|[-*+])\s+(.+)$", body, re.MULTILINE)
            if len(steps) < 3:
                continue
            numbered = "\n".join(f"{i}. {step.strip()}" for i, step in enumerate(steps[:15], 1))
            subject = (heading or title).rstrip("?:.")
            out.append((f"How do I {subject.lower()} in {site}?", numbered))
            if len(out) >= n:
                break
        return out

    def _build_api_understanding(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        for match in re.finditer(
            r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/\S*)", text
        ):
            method, path = match.group(1).upper(), match.group(2)
            window = text[match.start() : match.start() + 1200]
            out.append(
                (
                    f"What does the {method} {path} endpoint do in {site}?",
                    self._trim(window, 1400),
                )
            )
            if len(out) >= n:
                break

        if len(out) < n:
            # Parameter tables are the other reliable API signal.
            for heading, body in _sections(text):
                if body.count("|") >= 6:
                    out.append(
                        (
                            f"What parameters does {(heading or title).rstrip('?:.')} accept in {site}?",
                            self._trim(body, 1600),
                        )
                    )
                    break
        return out

    def _build_code_explanation(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        for match in re.finditer(r"```([A-Za-z0-9+#_-]*)\n(.*?)```", text, re.DOTALL):
            lang, code = match.group(1) or "", match.group(2).rstrip()
            if len(code.strip()) < 40:
                continue
            # Context immediately before a code block is nearly always its
            # explanation on a documentation page.
            before = text[max(0, match.start() - 600) : match.start()].strip()
            explanation = " ".join(_sentences(before)[-3:]) or f"This {lang or 'code'} example is from {title}."
            out.append(
                (
                    f"Explain what this {lang or 'code'} example from {site} does:\n\n"
                    f"```{lang}\n{code}\n```",
                    self._trim(explanation, 1200),
                )
            )
            if len(out) >= n:
                break
        return out

    def _build_concept_learning(self, site: str, title: str, text: str, s: DocSignals, n: int):
        sections = [(h, b) for h, b in _sections(text) if h and len(b) >= 250]
        out: list[tuple[str, str]] = []
        for heading, body in sections[:n]:
            out.append(
                (
                    f"Teach me about {heading.rstrip('?:.')} in {site}.",
                    self._trim(body, 2000),
                )
            )
        return out

    def _build_troubleshooting(self, site: str, title: str, text: str, s: DocSignals, n: int):
        out: list[tuple[str, str]] = []
        pattern = re.compile(
            r"[^.\n]*\b(?:error|fails?|failure|exception|warning|cannot|unable to|"
            r"invalid|denied|timeout|troubleshoot)\b[^.\n]*\.", re.I
        )
        for heading, body in _sections(text):
            match = pattern.search(body)
            if not match:
                continue
            problem = match.group(0).strip()
            if len(problem) < 40:
                continue
            out.append(
                (
                    f"I ran into this issue with {site}: {problem} How do I resolve it?",
                    self._trim(body, 1600),
                )
            )
            if len(out) >= n:
                break
        return out

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        cut = text[:limit]
        # Prefer to end on a sentence boundary.
        for sep in (". ", "\n\n", "\n"):
            idx = cut.rfind(sep)
            if idx > limit * 0.6:
                return cut[: idx + 1].strip()
        return cut.strip()


# ===========================================================================
# Selection
# ===========================================================================


@dataclass
class ResolvedProvider:
    generator: SampleGenerator
    backend_name: str
    needs_client: bool
    probe_results: dict[str, bool] = field(default_factory=dict)


async def resolve_provider(cfg: ProviderConfig) -> ResolvedProvider:
    """
    Pick a generator.

    Explicit backend -> use it (structural never fails; the others error out
    loudly rather than silently degrading, because a user who asked for
    Anthropic wants to know their key is missing).

    backend: auto -> probe fallback_order and take the first live one.
    """
    if cfg.backend == "structural":
        logger.info("dataset generator: structural (no LLM)")
        return ResolvedProvider(StructuralSampleGenerator(), "structural", False)

    async with httpx.AsyncClient() as client:
        if cfg.backend != "auto":
            backend_cls = BACKENDS.get(cfg.backend)
            if backend_cls is None:
                raise ValueError(f"unknown provider backend: {cfg.backend}")
            backend = backend_cls(cfg)
            if not await backend.available(client):
                raise RuntimeError(
                    f"provider '{cfg.backend}' was requested but is not reachable. "
                    f"Check the endpoint, the model name, and the API key env var. "
                    f"Use dataset.provider.backend=auto to fall back automatically."
                )
            logger.info(f"dataset generator: {backend.describe()}")
            return ResolvedProvider(
                LLMSampleGenerator(backend, cfg), backend.describe(), True
            )

        probes: dict[str, bool] = {}
        for name in cfg.fallback_order:
            if name == "structural":
                break
            backend_cls = BACKENDS.get(name)
            if backend_cls is None:
                continue
            backend = backend_cls(cfg)
            ok = await backend.available(client)
            probes[name] = ok
            if ok:
                logger.info(f"dataset generator: {backend.describe()} (auto-selected)")
                return ResolvedProvider(
                    LLMSampleGenerator(backend, cfg), backend.describe(), True, probes
                )
            logger.debug(f"provider probe failed: {name}")

    logger.warning(
        "no LLM backend reachable (probed: "
        + ", ".join(f"{k}={'up' if v else 'down'}" for k, v in probes.items())
        + ") — falling back to the structural generator. "
        "Sample quality will be lower; start Ollama or set an API key for better data."
    )
    return ResolvedProvider(StructuralSampleGenerator(), "structural", False, probes)
