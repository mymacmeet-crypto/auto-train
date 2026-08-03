"""
Phase 9 — evaluation.

Six measurements, chosen because each catches a different failure mode that
the others miss:

  train/validation loss curves   Did it learn, and did it overfit? Divergence
                                 between the two curves is the signal.
  perplexity                     Loss in an interpretable unit. Comparable
                                 across runs on the same validation set.
  sample generations             What the model actually says. No metric
                                 substitutes for reading twenty answers.
  instruction-following          Did it answer the question asked, in the
                                 requested shape? A model that ignores "list
                                 the steps" and writes prose has regressed
                                 even if loss improved.
  grounding / hallucination      Are the answer's content words present in
                                 the source corpus? On a closed-domain
                                 fine-tune, low overlap means invention.
  consistency                    Same question twice at temperature > 0. A
                                 model that contradicts itself has not
                                 learned the fact, it has learned a
                                 distribution over plausible-sounding facts.

Grounding is a proxy, not proof. It cannot detect a confident wrong number
that happens to appear elsewhere in the corpus. It reliably catches
wholesale invention, which is the failure that matters most here.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger, make_progress
from autotrain.utils.paths import Workspace
from autotrain.utils.text import content_words, read_json, read_jsonl, write_json

logger = get_logger("evaluator")


# ---------------------------------------------------------------------------
# Instruction-following heuristics
# ---------------------------------------------------------------------------

_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
_CODE = re.compile(r"```")

_WANTS_STEPS = re.compile(r"\b(?:step|steps|how do i|how to|procedure|walk me through|guide)\b", re.I)
_WANTS_LIST = re.compile(r"\b(?:list|what are the|which \w+ are|enumerate|options|parameters)\b", re.I)
_WANTS_CODE = re.compile(r"\b(?:code|snippet|example|syntax|command|implement|script)\b", re.I)
_WANTS_SUMMARY = re.compile(r"\bsummar(?:ise|ize|y)\b", re.I)


def score_instruction_following(instruction: str, answer: str) -> dict[str, Any]:
    """
    Did the answer take the shape the instruction asked for?

    Each check only fires when the instruction actually requested that shape,
    so a factual question is never penalised for lacking bullet points.
    """
    checks: dict[str, bool] = {}

    if _WANTS_STEPS.search(instruction):
        checks["produced_steps"] = bool(_NUMBERED.search(answer))
    if _WANTS_LIST.search(instruction):
        checks["produced_list"] = bool(_BULLET.search(answer) or _NUMBERED.search(answer))
    if _WANTS_CODE.search(instruction):
        checks["produced_code"] = bool(_CODE.search(answer))
    if _WANTS_SUMMARY.search(instruction):
        # A summary longer than 2,000 characters is not a summary.
        checks["summary_is_concise"] = len(answer) <= 2000

    # Universal checks.
    checks["non_empty"] = len(answer.strip()) >= 20
    checks["not_truncated"] = bool(
        re.search(r"[.!?:;)\]`\"']\s*$", answer.strip()) or answer.strip().endswith("```")
    )
    checks["no_prompt_echo"] = instruction.strip().lower() not in answer.strip().lower()
    checks["no_refusal"] = not re.match(
        r"^\s*(?:i(?:'m| am) (?:sorry|unable)|i cannot|as an ai)", answer, re.I
    )

    passed = sum(1 for v in checks.values() if v)
    return {
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "score": round(passed / len(checks), 4) if checks else 0.0,
    }


def score_grounding(answer: str, corpus_vocabulary: set[str]) -> dict[str, Any]:
    """Fraction of the answer's content words that occur in the source corpus."""
    words = content_words(answer)
    if not words:
        return {"score": 0.0, "known": 0, "total": 0, "unknown_sample": []}
    known = words & corpus_vocabulary
    unknown = words - corpus_vocabulary
    return {
        "score": round(len(known) / len(words), 4),
        "known": len(known),
        "total": len(words),
        "unknown_sample": sorted(unknown)[:15],
    }


def score_consistency(answers: list[str]) -> dict[str, Any]:
    """Jaccard overlap of content words across repeated generations."""
    if len(answers) < 2:
        return {"score": 1.0, "trials": len(answers)}
    sets = [content_words(a) for a in answers if a.strip()]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return {"score": 0.0, "trials": len(answers)}

    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                scores.append(len(sets[i] & sets[j]) / len(union))
    return {
        "score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "trials": len(answers),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_finetuned(cfg: AutotrainConfig, ws: Workspace):
    """Load the base model with the trained adapter attached."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metrics = read_json(ws.train_metrics, default={}) or {}
    model_id = metrics.get("model_id") or (
        cfg.training.cpu_model if cfg.training.base_model == "auto" else cfg.training.base_model
    )
    device = (metrics.get("resolved") or {}).get("device", "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(str(ws.adapter), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, str(ws.adapter))
    model.eval()
    if device != "cpu":
        model = model.to(device)

    return model, tokenizer, model_id, device


def generate(model, tokenizer, messages: list[dict[str, str]], *,
             max_new_tokens: int, temperature: float, top_p: float, device: str) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_perplexity(model, tokenizer, records: list[dict[str, Any]],
                       *, device: str, max_len: int, limit: int = 100) -> float | None:
    """Token-weighted perplexity over the assistant turns of the eval set."""
    import torch

    total_nll = 0.0
    total_tokens = 0

    for record in records[:limit]:
        messages = record.get("messages")
        if not messages:
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # loss is a mean over (n-1) predicted tokens; re-weight to sum.
        n_tokens = input_ids.shape[1] - 1
        total_nll += float(out.loss) * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return None
    return round(math.exp(min(total_nll / total_tokens, 20)), 4)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_evaluation(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    ecfg = cfg.evaluation
    ws.evaluation.mkdir(parents=True, exist_ok=True)

    metrics = read_json(ws.train_metrics, default={}) or {}
    if not metrics:
        raise RuntimeError(
            f"no training metrics at {ws.train_metrics} — run the train phase first"
        )

    # ---- loss curves (no model load required) ---------------------------
    train_log = metrics.get("train_log", [])
    eval_log = metrics.get("eval_log", [])

    curves = {
        "train_loss": [{"step": e["step"], "loss": e["loss"]} for e in train_log],
        "eval_loss": [{"step": e["step"], "loss": e["eval_loss"]} for e in eval_log],
        "first_train_loss": train_log[0]["loss"] if train_log else None,
        "final_train_loss": train_log[-1]["loss"] if train_log else None,
        "best_eval_loss": min((e["eval_loss"] for e in eval_log), default=None),
        "final_eval_loss": eval_log[-1]["eval_loss"] if eval_log else None,
    }
    if curves["first_train_loss"] and curves["final_train_loss"]:
        curves["train_loss_reduction_pct"] = round(
            100 * (curves["first_train_loss"] - curves["final_train_loss"])
            / curves["first_train_loss"], 2
        )

    # Overfitting signal: eval loss rising while train loss falls.
    overfitting = False
    if len(eval_log) >= 3 and len(train_log) >= 3:
        eval_trend = eval_log[-1]["eval_loss"] - min(e["eval_loss"] for e in eval_log)
        train_trend = train_log[0]["loss"] - train_log[-1]["loss"]
        overfitting = eval_trend > 0.05 and train_trend > 0.1
    curves["overfitting_suspected"] = overfitting

    write_json(curves, ws.evaluation / "loss_curves.json")

    # ---- corpus vocabulary for grounding --------------------------------
    corpus_vocabulary: set[str] = set()
    if ecfg.grounding_check:
        for record in read_jsonl(ws.corpus_jsonl):
            corpus_vocabulary |= content_words(record.get("text", ""))
        logger.info(f"grounding vocabulary: {len(corpus_vocabulary):,} distinct content words")

    # ---- load the model --------------------------------------------------
    eval_records = list(read_jsonl(ws.val_jsonl))
    generations: list[dict[str, Any]] = []
    perplexity = None
    aggregate: dict[str, Any] = {}
    model_error: str | None = None

    try:
        logger.info("loading the fine-tuned model for generation checks")
        model, tokenizer, model_id, device = load_finetuned(cfg, ws)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        logger.warning(f"could not load the model for generation checks: {model_error}")
        model = tokenizer = None
        model_id = metrics.get("model_id", "unknown")
        device = "cpu"

    if model is not None:
        if ecfg.compute_perplexity and eval_records:
            logger.info("computing perplexity on the validation split")
            try:
                perplexity = compute_perplexity(
                    model, tokenizer, eval_records,
                    device=device, max_len=cfg.training.max_seq_length,
                )
            except Exception as exc:
                logger.warning(f"perplexity failed: {type(exc).__name__}: {exc}")

        sample_count = min(ecfg.num_sample_generations, len(eval_records))
        if sample_count:
            logger.info(f"generating {sample_count} sample answers")
            progress = make_progress()
            with progress:
                bar = progress.add_task("evaluating", total=sample_count)
                for record in eval_records[:sample_count]:
                    messages = record["messages"]
                    prompt_messages = [m for m in messages if m["role"] != "assistant"]
                    reference = next(
                        (m["content"] for m in messages if m["role"] == "assistant"), ""
                    )
                    instruction = next(
                        (m["content"] for m in messages if m["role"] == "user"), ""
                    )

                    try:
                        answer = generate(
                            model, tokenizer, prompt_messages,
                            max_new_tokens=ecfg.max_new_tokens,
                            temperature=ecfg.temperature,
                            top_p=ecfg.top_p,
                            device=device,
                        )
                    except Exception as exc:
                        answer = ""
                        logger.debug(f"generation failed: {exc}")

                    entry: dict[str, Any] = {
                        "instruction": instruction,
                        "reference": reference,
                        "generated": answer,
                        "sample_type": (record.get("meta") or {}).get("sample_type", "unknown"),
                        "source_url": (record.get("meta") or {}).get("source_url", ""),
                        "instruction_following": score_instruction_following(instruction, answer),
                    }

                    if ecfg.grounding_check and corpus_vocabulary:
                        entry["grounding"] = score_grounding(answer, corpus_vocabulary)

                    if ecfg.consistency_check and ecfg.consistency_trials > 1 and answer:
                        repeats = [answer]
                        for _ in range(ecfg.consistency_trials - 1):
                            try:
                                repeats.append(
                                    generate(
                                        model, tokenizer, prompt_messages,
                                        max_new_tokens=ecfg.max_new_tokens,
                                        temperature=max(ecfg.temperature, 0.5),
                                        top_p=ecfg.top_p,
                                        device=device,
                                    )
                                )
                            except Exception:
                                break
                        entry["consistency"] = score_consistency(repeats)

                    generations.append(entry)
                    progress.advance(bar)

        # ---- aggregate ---------------------------------------------------
        if generations:
            following = [g["instruction_following"]["score"] for g in generations]
            aggregate["instruction_following_mean"] = round(sum(following) / len(following), 4)

            grounded = [g["grounding"]["score"] for g in generations if "grounding" in g]
            if grounded:
                aggregate["grounding_mean"] = round(sum(grounded) / len(grounded), 4)
                below = sum(1 for s in grounded if s < ecfg.grounding_min_score)
                aggregate["low_grounding_count"] = below
                aggregate["low_grounding_pct"] = round(100 * below / len(grounded), 2)

            consistent = [g["consistency"]["score"] for g in generations if "consistency" in g]
            if consistent:
                aggregate["consistency_mean"] = round(sum(consistent) / len(consistent), 4)

            lengths = [len(g["generated"]) for g in generations]
            aggregate["mean_answer_chars"] = round(sum(lengths) / len(lengths), 1)
            aggregate["empty_generations"] = sum(1 for g in generations if not g["generated"].strip())

    write_json(
        {"generations": generations, "aggregate": aggregate},
        ws.evaluation / "sample_generations.json",
    )

    report = {
        "model_id": model_id,
        "adapter": str(ws.adapter),
        "device": device,
        "model_load_error": model_error,
        "loss": {
            "final_train_loss": curves.get("final_train_loss"),
            "final_eval_loss": curves.get("final_eval_loss"),
            "best_eval_loss": curves.get("best_eval_loss"),
            "train_loss_reduction_pct": curves.get("train_loss_reduction_pct"),
            "overfitting_suspected": overfitting,
        },
        "perplexity": perplexity,
        "quality": aggregate,
        "samples_evaluated": len(generations),
        "evaluated_at": time.time(),
    }
    write_json(report, ws.evaluation / "evaluation_report.json")

    logger.info(
        f"evaluation: train loss {report['loss']['final_train_loss']}, "
        f"eval loss {report['loss']['final_eval_loss']}, "
        f"perplexity {perplexity}"
    )
    if aggregate:
        logger.info(
            f"   instruction-following {aggregate.get('instruction_following_mean')}, "
            f"grounding {aggregate.get('grounding_mean')}, "
            f"consistency {aggregate.get('consistency_mean')}"
        )
    if overfitting:
        logger.warning(
            "eval loss rose while train loss fell — the model is overfitting. "
            "Reduce training.num_train_epochs or raise training.lora_dropout."
        )

    return report
