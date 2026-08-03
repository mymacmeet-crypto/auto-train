"""
The eight instruction sample types.

Each type declares three things:

  - `applies_to`: a cheap structural test. Asking for a code explanation from
    a page with no code produces a hallucinated answer, which is worse than
    producing nothing. Every type is gated on evidence that it is relevant.
  - `prompt`: the instruction sent to the generator LLM. All of them demand a
    bare JSON array and forbid preamble, because a chatty model wrapping its
    answer in prose is the single most common parse failure.
  - `weight`: relative sample budget. Documentation QA is the backbone of a
    docs fine-tune and gets the largest share.

The shared rule across every prompt is "answerable only from the excerpt".
A fine-tune trained on invented facts teaches the model to invent facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Structural signals used for applicability
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```")
_TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_ORDERED_STEP = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)

_HTTP_VERB = re.compile(
    r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b\s+/", re.I
)
_API_HINT = re.compile(
    r"\b(?:endpoint|api|request|response|payload|parameter|param|query string|"
    r"status code|authorization header|bearer token|rate limit|webhook|"
    r"returns?\s+(?:a\s+)?json|content-type)\b",
    re.I,
)
_PROCEDURE_HINT = re.compile(
    r"\b(?:step\s*\d|first,|then,|finally,|how to|to (?:install|configure|deploy|"
    r"enable|create|set ?up|add|remove|migrate)|follow these|procedure|"
    r"walkthrough|tutorial|getting started|quick ?start)\b",
    re.I,
)
_TROUBLE_HINT = re.compile(
    r"\b(?:error|exception|fail(?:s|ed|ure)?|troubleshoot|debug|issue|problem|"
    r"warning|caveat|gotcha|known limitation|not supported|invalid|denied|"
    r"timeout|cannot|unable to|fix|resolve|workaround)\b",
    re.I,
)
_CONCEPT_HINT = re.compile(
    r"\b(?:is a|are a|refers to|means that|concept|overview|introduction|"
    r"architecture|design|principle|model|what is|understanding|works by|"
    r"consists of|defined as)\b",
    re.I,
)


@dataclass(frozen=True)
class DocSignals:
    """Cheap structural facts about one document window."""

    chars: int
    words: int
    headings: list[str]
    code_blocks: int
    code_languages: list[str]
    tables: int
    ordered_steps: int
    has_http_verbs: bool
    api_score: int
    procedure_score: int
    trouble_score: int
    concept_score: int


def analyse(text: str) -> DocSignals:
    headings = [m.group(2).strip() for m in _HEADING.finditer(text)]
    fences = _CODE_FENCE.findall(text)
    code_blocks = len(fences) // 2

    languages: list[str] = []
    for match in re.finditer(r"```([A-Za-z0-9+#_-]+)", text):
        lang = match.group(1).lower()
        if lang and lang not in languages:
            languages.append(lang)

    return DocSignals(
        chars=len(text),
        words=len(text.split()),
        headings=headings,
        code_blocks=code_blocks,
        code_languages=languages,
        tables=len(_TABLE_ROW.findall(text)),
        ordered_steps=len(_ORDERED_STEP.findall(text)),
        has_http_verbs=bool(_HTTP_VERB.search(text)),
        api_score=len(_API_HINT.findall(text)),
        procedure_score=len(_PROCEDURE_HINT.findall(text)),
        trouble_score=len(_TROUBLE_HINT.findall(text)),
        concept_score=len(_CONCEPT_HINT.findall(text)),
    )


# ---------------------------------------------------------------------------
# Sample type definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleTypeSpec:
    key: str
    label: str
    weight: float
    applies_to: Callable[[DocSignals], bool]
    prompt: str


_JSON_CONTRACT = """\
Output rules (these override any other instinct you have):
- Output a JSON array and nothing else. No preamble, no explanation, no code fence.
- Every element must be an object with exactly two string keys: "instruction" and "output".
- Never invent facts. If the excerpt does not support {n} items, return fewer.
- Do not reference "the excerpt", "the text", "the documentation above", or
  "this page" in either field. The reader will not see the excerpt; questions
  and answers must stand alone.
"""

_CONTEXT_BLOCK = """\
SOURCE: {title}
{url_line}
CONTENT:
{content}
"""


DOC_QA = SampleTypeSpec(
    key="doc_qa",
    label="Documentation QA",
    weight=3.0,
    applies_to=lambda s: s.words >= 60,
    prompt="""\
You are building a question-answering dataset about {site_name}.

Write {n} distinct question-and-answer pairs drawn from the content below.

Requirements:
- Vary the question form: what-is, how-does, which, when, why, can-I.
- Each answer must be complete and self-contained, 2-8 sentences.
- Prefer specific, factual questions over vague ones. "What is the default
  timeout?" beats "Tell me about timeouts."
- Where the content gives exact names, values, or defaults, use them verbatim.
- "instruction" holds the question, "output" holds the answer.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


SUMMARIZATION = SampleTypeSpec(
    key="summarization",
    label="Summarization",
    weight=1.0,
    applies_to=lambda s: s.words >= 150,
    prompt="""\
You are building a summarization dataset about {site_name}.

Write {n} summarization example(s) from the content below.

Requirements:
- "instruction" must be a natural request to summarise this specific topic,
  naming the subject. For example: "Summarise how authentication works in
  {site_name}." Not just "Summarise this page."
- "output" is the summary: 3-6 sentences, covering the main points in order
  of importance, retaining concrete names and values.
- If you write more than one, vary the angle: one broad, one focused on the
  practical takeaway.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


EXPLANATION = SampleTypeSpec(
    key="explanation",
    label="Explanation",
    weight=1.5,
    applies_to=lambda s: s.words >= 100,
    prompt="""\
You are building an explanation dataset about {site_name}.

Write {n} explanation example(s) from the content below.

Requirements:
- "instruction" asks for an explanation of a specific topic named in the
  content, phrased as a user would: "Explain how X works", "Walk me through
  what X does and why it matters".
- "output" explains it properly: what it is, how it behaves, why it exists,
  and any constraint the content states. 4-10 sentences.
- Use a bulleted list only where the content itself enumerates things.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


STEP_BY_STEP = SampleTypeSpec(
    key="step_by_step",
    label="Step-by-step guide",
    weight=1.5,
    applies_to=lambda s: s.ordered_steps >= 2 or s.procedure_score >= 2,
    prompt="""\
You are building a procedural how-to dataset about {site_name}.

Write {n} step-by-step example(s) from the content below.

Requirements:
- "instruction" is a task-oriented question: "How do I deploy X?",
  "What are the steps to configure Y?"
- "output" is a numbered list of concrete steps in the correct order, each
  step a single action. Include exact commands, file paths, field names, or
  setting values wherever the content supplies them.
- Add a short sentence before the list stating the goal, and a short sentence
  after it stating how to verify success — but only if the content supports it.
- Only produce a guide for a procedure the content actually describes.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


API_UNDERSTANDING = SampleTypeSpec(
    key="api_understanding",
    label="API understanding",
    weight=1.5,
    applies_to=lambda s: s.has_http_verbs or s.api_score >= 3 or (s.tables >= 3 and s.api_score >= 1),
    prompt="""\
You are building an API reference dataset about {site_name}.

Write {n} API example(s) from the content below.

Requirements:
- "instruction" asks about a specific endpoint, method, parameter, or
  response: "What does the POST /v1/users endpoint do?", "What parameters
  does X accept?", "What does this API return on failure?"
- "output" answers precisely: purpose, HTTP method and path, required and
  optional parameters with their types and defaults, response shape, and
  error conditions — restricted to whatever the content states.
- Reproduce parameter names, types, and status codes exactly as written.
- Use a fenced code block for any request or response body.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


CODE_EXPLANATION = SampleTypeSpec(
    key="code_explanation",
    label="Code explanation",
    weight=1.5,
    applies_to=lambda s: s.code_blocks >= 1,
    prompt="""\
You are building a code-explanation dataset about {site_name}.

Write {n} code example(s) from the content below.

Requirements:
- "instruction" asks about a specific code sample that appears in the content:
  "What does this snippet do?", "How do I use X in code?", "Explain this
  configuration."
- Include the relevant code inside the "instruction" in a fenced code block
  when the question is about reading it; put the code in the "output" instead
  when the question is about writing it.
- "output" explains what the code does line by line or block by block, names
  the language, and states any prerequisite the content mentions.
- Reproduce code EXACTLY as it appears — never reformat, rename, or 'improve'
  it. Preserve indentation.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


CONCEPT_LEARNING = SampleTypeSpec(
    key="concept_learning",
    label="Concept learning",
    weight=1.0,
    applies_to=lambda s: s.concept_score >= 2 and s.words >= 120,
    prompt="""\
You are building a teaching dataset about {site_name}.

Write {n} concept-teaching example(s) from the content below.

Requirements:
- "instruction" is a learning request naming the concept: "Teach me about X",
  "I'm new to Y — explain the core idea", "Help me understand Z."
- "output" teaches it: start with a one-sentence definition, then the mental
  model, then how it fits with the surrounding system, then a concrete example
  if the content provides one. 5-12 sentences.
- Write for someone competent but unfamiliar. Do not condescend, and do not
  assume prior knowledge of this product's jargon without defining it.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


TROUBLESHOOTING = SampleTypeSpec(
    key="troubleshooting",
    label="Troubleshooting",
    weight=1.5,
    applies_to=lambda s: s.trouble_score >= 2,
    prompt="""\
You are building a troubleshooting dataset about {site_name}.

Write {n} troubleshooting example(s) from the content below.

Requirements:
- "instruction" states a problem in the user's voice: "I'm getting <error> —
  how do I fix it?", "Why does X fail when Y?", "What causes this warning?"
- "output" gives the likely cause first, then the fix as concrete steps, then
  how to confirm it worked. Reproduce exact error strings, codes, and setting
  names from the content.
- Only cover failures, errors, limitations, or caveats the content actually
  mentions. Do not invent plausible-sounding problems.

"""
    + _JSON_CONTRACT
    + "\n"
    + _CONTEXT_BLOCK,
)


SPECS: dict[str, SampleTypeSpec] = {
    spec.key: spec
    for spec in (
        DOC_QA,
        SUMMARIZATION,
        EXPLANATION,
        STEP_BY_STEP,
        API_UNDERSTANDING,
        CODE_EXPLANATION,
        CONCEPT_LEARNING,
        TROUBLESHOOTING,
    )
}


def build_prompt(
    spec: SampleTypeSpec,
    *,
    site_name: str,
    title: str,
    url: str,
    content: str,
    n: int,
) -> str:
    return spec.prompt.format(
        site_name=site_name,
        n=n,
        title=title or "Untitled",
        url_line=f"URL: {url}" if url else "",
        content=content,
    )


def applicable_types(
    signals: DocSignals, enabled: list[str]
) -> list[SampleTypeSpec]:
    """Which sample types this window can legitimately support."""
    out: list[SampleTypeSpec] = []
    for key in enabled:
        spec = SPECS.get(key)
        if spec is None:
            continue
        try:
            if spec.applies_to(signals):
                out.append(spec)
        except Exception:
            continue
    return out
