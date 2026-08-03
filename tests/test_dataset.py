"""Dataset generation (Phase 5), validation (6), and splitting (7)."""

from __future__ import annotations

import asyncio
import json

import pytest

from autotrain.dataset_builder.builder import split_into_windows
from autotrain.dataset_builder.prompts import SPECS, analyse, applicable_types
from autotrain.dataset_builder.providers import (
    Pair,
    StructuralSampleGenerator,
    parse_pairs,
)
from autotrain.utils.config import load_config
from autotrain.utils.paths import Workspace
from autotrain.utils.text import write_jsonl
from autotrain.validator.split import run_split
from autotrain.validator.validate import run_validation


class TestJsonRecovery:
    def test_clean_array(self) -> None:
        raw = '[{"instruction": "Q?", "output": "A."}]'
        assert parse_pairs(raw) == [{"instruction": "Q?", "output": "A."}]

    def test_fenced_json(self) -> None:
        raw = 'Sure!\n```json\n[{"instruction": "Q?", "output": "A."}]\n```\nHope that helps.'
        assert len(parse_pairs(raw)) == 1

    def test_preamble_and_trailer(self) -> None:
        raw = 'Here you go: [{"instruction": "Q?", "output": "A."}] Let me know!'
        assert len(parse_pairs(raw)) == 1

    def test_key_aliases(self) -> None:
        raw = '[{"question": "Q?", "answer": "A."}]'
        parsed = parse_pairs(raw)
        assert parsed[0]["instruction"] == "Q?" and parsed[0]["output"] == "A."

    def test_wrapper_object(self) -> None:
        raw = '{"samples": [{"instruction": "Q?", "output": "A."}]}'
        assert len(parse_pairs(raw)) == 1

    def test_single_object(self) -> None:
        assert len(parse_pairs('{"instruction": "Q?", "output": "A."}')) == 1

    def test_concatenated_objects_without_array(self) -> None:
        raw = '{"instruction": "Q1", "output": "A1"}\n{"instruction": "Q2", "output": "A2"}'
        assert len(parse_pairs(raw)) == 2

    def test_incomplete_entries_dropped(self) -> None:
        raw = '[{"instruction": "Q?"}, {"instruction": "Q2", "output": "A2"}]'
        assert len(parse_pairs(raw)) == 1

    @pytest.mark.parametrize("raw", ["", "   ", "no json here at all", "[[[", "{"])
    def test_garbage_returns_empty(self, raw: str) -> None:
        assert parse_pairs(raw) == []


class TestApplicability:
    def test_code_type_requires_code(self) -> None:
        prose = analyse("Just some prose about a topic. " * 20)
        keys = {s.key for s in applicable_types(prose, list(SPECS))}
        assert "code_explanation" not in keys

        with_code = analyse("Some prose.\n\n```python\nx = 1\n```\n" + "More prose. " * 20)
        keys = {s.key for s in applicable_types(with_code, list(SPECS))}
        assert "code_explanation" in keys

    def test_api_type_requires_api_signals(self) -> None:
        prose = analyse("A gentle introduction to gardening in temperate climates. " * 10)
        assert "api_understanding" not in {s.key for s in applicable_types(prose, list(SPECS))}

        api = analyse("Send a GET /v1/users request. The endpoint returns JSON. " * 5)
        assert "api_understanding" in {s.key for s in applicable_types(api, list(SPECS))}

    def test_steps_require_procedure(self) -> None:
        steps = analyse("How to install:\n\n1. Download it\n2. Run it\n3. Verify\n" * 3)
        assert "step_by_step" in {s.key for s in applicable_types(steps, list(SPECS))}

    def test_tiny_document_supports_nothing(self) -> None:
        assert applicable_types(analyse("Hi."), list(SPECS)) == []


class TestWindowing:
    def test_short_doc_single_window(self) -> None:
        assert len(split_into_windows("short", max_chars=6000, overlap=400, max_windows=8)) == 1

    def test_long_doc_splits(self) -> None:
        doc = "\n\n".join(f"## Section {i}\n\n" + ("body text. " * 200) for i in range(10))
        windows = split_into_windows(doc, max_chars=3000, overlap=200, max_windows=8)
        assert len(windows) > 1
        assert all(len(w) <= 3000 + 3000 for w in windows)

    def test_max_windows_respected(self) -> None:
        doc = "\n\n".join(f"## S{i}\n\n" + ("x " * 500) for i in range(50))
        assert len(split_into_windows(doc, max_chars=1000, overlap=100, max_windows=5)) <= 5

    def test_no_headings_still_splits(self) -> None:
        doc = "word " * 5000
        windows = split_into_windows(doc, max_chars=2000, overlap=100, max_windows=8)
        assert len(windows) > 1


class TestStructuralGenerator:
    @pytest.fixture
    def doc(self) -> str:
        return (
            "# Authentication\n\n"
            "Authentication controls who may call the API. Every request must carry "
            "a bearer token in the Authorization header. Tokens expire after one hour.\n\n"
            "## Obtaining a token\n\n"
            "1. Register an application in the console\n"
            "2. Copy the client secret\n"
            "3. Exchange the secret for a token\n"
            "4. Store the token securely\n\n"
            "## Example\n\n"
            "```python\nimport requests\nr = requests.get(url, headers=h)\n```\n\n"
            "## Errors\n\n"
            "A 401 error means the token is invalid or has expired. Refresh it and retry. "
            "This failure is common when the clock is skewed. Verify that the system "
            "clock is synchronised against a network time source before assuming the "
            "credentials themselves are wrong, because a drift of more than sixty "
            "seconds will invalidate an otherwise correct signature.\n"
        )

    def _run(self, doc: str, type_key: str) -> list[Pair]:
        generator = StructuralSampleGenerator()
        return asyncio.run(
            generator.generate(
                None, SPECS[type_key],
                site_name="example.com", title="Authentication",
                url="https://example.com/auth", content=doc,
                signals=analyse(doc), n=2, source_index=1, window=0,
            )
        )

    def test_doc_qa(self, doc: str) -> None:
        pairs = self._run(doc, "doc_qa")
        assert pairs and all(p.instruction and p.output for p in pairs)

    def test_steps_are_numbered(self, doc: str) -> None:
        pairs = self._run(doc, "step_by_step")
        assert pairs and "1." in pairs[0].output

    def test_code_pair_embeds_code(self, doc: str) -> None:
        pairs = self._run(doc, "code_explanation")
        assert pairs and "```" in pairs[0].instruction

    def test_troubleshooting_mentions_error(self, doc: str) -> None:
        pairs = self._run(doc, "troubleshooting")
        assert pairs and "401" in pairs[0].output

    def test_answers_are_grounded_in_source(self, doc: str) -> None:
        # Every structural answer is verbatim source text, so it cannot
        # hallucinate. Each generated answer must be a substring of the source.
        pairs = self._run(doc, "explanation")
        assert pairs
        normalised_source = " ".join(doc.split())
        for pair in pairs:
            assert " ".join(pair.output.split()) in normalised_source

    def test_short_sections_skipped_for_explanation(self) -> None:
        # An "explain this" task needs enough material to explain. A stub
        # section should yield nothing rather than a one-line non-answer.
        stub = "# Title\n\n## Bit\n\nToo short.\n"
        assert self._run(stub, "explanation") == []


def _record(topic: str, answer: str, source: int = 1, sample_type: str = "doc_qa") -> dict:
    """
    Build a realistic training record.

    `topic` is expanded into a full question rather than used verbatim:
    validation enforces a 10-character minimum on instructions, so bare
    placeholders like "Q?" would be dropped for the wrong reason and mask
    whatever the test is actually checking.
    """
    instruction = f"What does {topic} do and how is it configured?"
    return {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer},
        ],
        "instruction": instruction,
        "output": answer,
        "meta": {"sample_type": sample_type, "source_index": source,
                 "source_url": f"https://e.com/{source}", "generator": "test"},
    }


LONG = "This is a sufficiently detailed answer that clears the minimum length bar easily."


class TestValidation:
    @pytest.fixture
    def ws(self, tmp_path) -> Workspace:
        return Workspace(tmp_path).ensure()

    @pytest.fixture
    def cfg(self):
        return load_config(domain="https://e.com")

    def test_valid_records_pass(self, ws, cfg) -> None:
        write_jsonl([_record(f"widget {i}", LONG + str(i)) for i in range(5)], ws.dataset_jsonl)
        report = run_validation(cfg, ws)
        assert report["valid_records"] == 5 and report["dropped_records"] == 0

    def test_short_answer_dropped(self, ws, cfg) -> None:
        write_jsonl([_record("alpha", "no"), _record("beta", LONG)], ws.dataset_jsonl)
        report = run_validation(cfg, ws)
        assert report["valid_records"] == 1

    def test_short_instruction_dropped(self, ws, cfg) -> None:
        short = {
            "messages": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": LONG},
            ],
            "meta": {"source_index": 1},
        }
        write_jsonl([short, _record("beta", LONG)], ws.dataset_jsonl)
        assert run_validation(cfg, ws)["valid_records"] == 1

    def test_duplicates_dropped(self, ws, cfg) -> None:
        write_jsonl([_record("same", LONG)] * 3 + [_record("other", LONG)], ws.dataset_jsonl)
        report = run_validation(cfg, ws)
        assert report["valid_records"] == 2

    def test_missing_assistant_dropped(self, ws, cfg) -> None:
        bad = {"messages": [{"role": "user", "content": "What does alpha do here?"}], "meta": {}}
        write_jsonl([bad, _record("beta", LONG)], ws.dataset_jsonl)
        assert run_validation(cfg, ws)["valid_records"] == 1

    def test_malformed_json_line_reported(self, ws, cfg) -> None:
        ws.dataset_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with ws.dataset_jsonl.open("w", encoding="utf-8") as fh:
            fh.write("{not json\n")
            fh.write(json.dumps(_record("alpha", LONG)) + "\n")
        report = run_validation(cfg, ws)
        assert report["valid_records"] == 1
        assert report["parse_failures"] == 1

    def test_refusal_dropped(self, ws, cfg) -> None:
        write_jsonl(
            [_record("alpha", "I'm sorry, I cannot answer that based on what I have."),
             _record("beta", LONG)],
            ws.dataset_jsonl,
        )
        assert run_validation(cfg, ws)["valid_records"] == 1

    def test_context_leak_warns_but_keeps(self, ws, cfg) -> None:
        leaky = "According to the excerpt, the timeout is thirty seconds by default."
        write_jsonl([_record("alpha", leaky)], ws.dataset_jsonl)
        report = run_validation(cfg, ws)
        assert report["valid_records"] == 1
        assert report["records_with_warnings"] == 1

    def test_all_invalid_raises(self, ws, cfg) -> None:
        write_jsonl([_record(f"a{i}", "x") for i in range(5)], ws.dataset_jsonl)
        with pytest.raises(RuntimeError):
            run_validation(cfg, ws)


class TestSplit:
    @pytest.fixture
    def ws(self, tmp_path) -> Workspace:
        return Workspace(tmp_path).ensure()

    @pytest.fixture
    def cfg(self):
        return load_config(domain="https://e.com")

    def test_no_source_document_straddles_the_split(self, ws, cfg) -> None:
        records = [
            _record(f"widget {doc} part {i}", LONG, source=doc)
            for doc in range(30) for i in range(4)
        ]
        write_jsonl(records, ws.validated_jsonl)
        run_split(cfg, ws)

        from autotrain.utils.text import read_jsonl

        train_docs = {r["meta"]["source_index"] for r in read_jsonl(ws.train_jsonl)}
        val_docs = {r["meta"]["source_index"] for r in read_jsonl(ws.val_jsonl)}
        assert train_docs and val_docs
        assert not (train_docs & val_docs), "a document appeared in both splits"

    def test_deterministic(self, ws, cfg) -> None:
        records = [_record(f"widget {d}", LONG, source=d) for d in range(40)]
        write_jsonl(records, ws.validated_jsonl)
        first = run_split(cfg, ws)
        second = run_split(cfg, ws)
        assert first["train_records"] == second["train_records"]

    def test_ratio_approximately_honoured(self, ws, cfg) -> None:
        records = [_record(f"widget {d}", LONG, source=d) for d in range(200)]
        write_jsonl(records, ws.validated_jsonl)
        report = run_split(cfg, ws)
        assert 0.82 <= report["train_ratio_actual"] <= 0.97

    def test_single_group_falls_back_and_flags_leakage(self, ws, cfg) -> None:
        records = [_record(f"widget {i}", LONG, source=1) for i in range(20)]
        write_jsonl(records, ws.validated_jsonl)
        report = run_split(cfg, ws)
        assert report["validation_records"] > 0
        assert report["leakage_warning"] is True
