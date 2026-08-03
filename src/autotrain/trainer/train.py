"""
Phase 8 — QLoRA / LoRA fine-tuning.

One trainer, two paths, chosen by hardware:

  CUDA + bitsandbytes -> the spec'd QLoRA: 4-bit NF4 base with double
      quantisation, bf16 compute, paged 8-bit AdamW, flash-attention-2,
      gradient checkpointing.
  Anything else -> the same LoRA adapter over an unquantised base. Same
      dataset, same objective, same outputs; only the memory strategy differs.

Both paths share everything downstream: the adapter format, the checkpoint
layout, the metrics file, and the resume behaviour. Phase 10 does not know
or care which one ran.

Completion-only loss is the other decision worth stating. With
`train_on_completions_only`, the loss is masked over the prompt so the model
is graded solely on the assistant turn. Without it, a fine-tune on a corpus
with one repeated system prompt spends much of its capacity learning to
reproduce that prompt.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from autotrain.trainer.hardware import (
    Hardware,
    ResolvedTraining,
    detect_hardware,
    resolve_training_setup,
)
from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger
from autotrain.utils.paths import Workspace
from autotrain.utils.text import count_lines, read_jsonl, write_json

logger = get_logger("trainer")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism costs throughput; enable it only where it is cheap.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Drop keyword arguments the installed library version does not accept.

    transformers renames and removes TrainingArguments fields between minor
    releases (`evaluation_strategy` -> `eval_strategy` in 4.46,
    `overwrite_output_dir` removed in 5.0). Pinning a narrow version range
    would be the other fix, but this pipeline is meant to run on whatever the
    host already has, so the trainer adapts instead.
    """
    fields = set(getattr(cls, "__dataclass_fields__", {}) or {})
    if not fields:
        return kwargs, []
    kept = {k: v for k, v in kwargs.items() if k in fields}
    dropped = sorted(set(kwargs) - set(kept))
    return kept, dropped


def _dtype_kwarg() -> str:
    """`torch_dtype` was renamed to `dtype` in transformers 5."""
    try:
        import transformers

        major = int(str(transformers.__version__).split(".")[0])
        return "dtype" if major >= 5 else "torch_dtype"
    except Exception:
        return "torch_dtype"


def _find_latest_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = [
        p for p in directory.glob("checkpoint-*")
        if p.is_dir() and (p / "trainer_state.json").exists()
    ]
    if not candidates:
        return None

    def step_of(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    return max(candidates, key=step_of)


class MetricsCollector:
    """
    Captures the loss history from the Trainer.

    HF writes log_history into trainer_state.json, but only for the current
    run — a resumed run starts a fresh history. Collecting here and merging
    with any previous metrics.json keeps the loss curve continuous across
    interruptions, which is what Phase 9 plots.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.train_log: list[dict[str, float]] = []
        self.eval_log: list[dict[str, float]] = []
        previous = {}
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                previous = {}
        self.train_log = previous.get("train_log", [])
        self.eval_log = previous.get("eval_log", [])

    def ingest(self, log_history: list[dict[str, Any]]) -> None:
        seen_train = {(e.get("step"), e.get("epoch")) for e in self.train_log}
        seen_eval = {(e.get("step"), e.get("epoch")) for e in self.eval_log}
        for entry in log_history:
            key = (entry.get("step"), entry.get("epoch"))
            if "loss" in entry and key not in seen_train:
                self.train_log.append(
                    {
                        "step": entry.get("step", 0),
                        "epoch": round(float(entry.get("epoch", 0.0)), 4),
                        "loss": float(entry["loss"]),
                        "learning_rate": float(entry.get("learning_rate", 0.0)),
                        "grad_norm": float(entry.get("grad_norm", 0.0) or 0.0),
                    }
                )
                seen_train.add(key)
            if "eval_loss" in entry and key not in seen_eval:
                loss = float(entry["eval_loss"])
                self.eval_log.append(
                    {
                        "step": entry.get("step", 0),
                        "epoch": round(float(entry.get("epoch", 0.0)), 4),
                        "eval_loss": loss,
                        "eval_perplexity": round(math.exp(min(loss, 20)), 4),
                    }
                )
                seen_eval.add(key)

        self.train_log.sort(key=lambda e: e.get("step", 0))
        self.eval_log.sort(key=lambda e: e.get("step", 0))


def _load_split(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = []
    for record in read_jsonl(path):
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            records.append({"messages": messages})
        if limit and len(records) >= limit:
            break
    return records


def run_training(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    tcfg = cfg.training

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"the training stack is not installed ({exc}). "
            "Install it with:  pip install -e '.[train]'   "
            "(add '.[cuda]' on a CUDA host for 4-bit QLoRA)"
        ) from exc

    _set_seed(tcfg.seed)

    hardware = detect_hardware()
    resolved = resolve_training_setup(tcfg, hardware)

    logger.info(
        f"training on {resolved.device} ({hardware.device_name}) — "
        f"model {resolved.model_id}, "
        f"{'4-bit QLoRA' if resolved.load_in_4bit else 'LoRA'} "
        f"@ {resolved.torch_dtype}"
    )

    # ---- data ------------------------------------------------------------
    train_records = _load_split(ws.train_jsonl, tcfg.max_train_samples)
    eval_records = _load_split(ws.val_jsonl, tcfg.max_eval_samples)
    if not train_records:
        raise RuntimeError(f"no training records at {ws.train_jsonl} — run the split phase first")

    logger.info(f"dataset: {len(train_records):,} train / {len(eval_records):,} eval")

    # ---- tokenizer -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        resolved.model_id, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- tokenisation ----------------------------------------------------
    # Done manually rather than via SFTTrainer so that completion-only
    # masking is explicit and works identically across TRL versions, whose
    # collator APIs have changed repeatedly.
    max_len = tcfg.max_seq_length

    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        input_ids_batch: list[list[int]] = []
        labels_batch: list[list[int]] = []

        for messages in batch["messages"]:
            full = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            input_ids = tokenizer(
                full, add_special_tokens=False, truncation=True, max_length=max_len
            )["input_ids"]

            if tcfg.train_on_completions_only:
                # Re-render everything up to (and including) the generation
                # prompt, then mask that many tokens. Using the template for
                # both halves keeps the boundary exact regardless of the
                # model's special-token conventions.
                prompt_messages = [m for m in messages if m.get("role") != "assistant"]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_ids = tokenizer(
                    prompt_text, add_special_tokens=False, truncation=True, max_length=max_len
                )["input_ids"]
                mask_upto = min(len(prompt_ids), len(input_ids))
                labels = [-100] * mask_upto + input_ids[mask_upto:]
                # A sample whose answer was entirely truncated away teaches
                # nothing and produces a NaN loss; drop it.
                if all(label == -100 for label in labels):
                    continue
            else:
                labels = list(input_ids)

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        return {"input_ids": input_ids_batch, "labels": labels_batch}

    train_ds = Dataset.from_list(train_records).map(
        tokenize, batched=True, batch_size=64, remove_columns=["messages"],
        desc="tokenizing train",
    )
    eval_ds = None
    if eval_records:
        eval_ds = Dataset.from_list(eval_records).map(
            tokenize, batched=True, batch_size=64, remove_columns=["messages"],
            desc="tokenizing eval",
        )

    if len(train_ds) == 0:
        raise RuntimeError(
            "every training sample was dropped during tokenisation — "
            f"max_seq_length={max_len} is probably too small for this dataset"
        )

    # ---- model -----------------------------------------------------------
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[resolved.torch_dtype]

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        _dtype_kwarg(): torch_dtype,
        "attn_implementation": resolved.attn_implementation,
    }

    if resolved.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=tcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=tcfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype_map[tcfg.bnb_4bit_compute_dtype],
        )
        model_kwargs["device_map"] = {"": 0}

    logger.info(f"loading {resolved.model_id} (this downloads the weights on first run)")
    model = AutoModelForCausalLM.from_pretrained(resolved.model_id, **model_kwargs)

    if not resolved.load_in_4bit and resolved.device != "cpu":
        model = model.to(resolved.device)

    model.config.use_cache = False  # incompatible with gradient checkpointing
    if getattr(model.config, "pretraining_tp", 1) != 1:
        model.config.pretraining_tp = 1

    if resolved.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=resolved.gradient_checkpointing
        )

    # ---- LoRA ------------------------------------------------------------
    # Filter the requested target modules to those the architecture actually
    # has, so a config tuned for Qwen does not explode on another model.
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [m for m in tcfg.lora_target_modules if m in present]
    if not targets:
        targets = [m for m in ("q_proj", "k_proj", "v_proj", "o_proj") if m in present]
    if not targets:
        raise RuntimeError(
            f"none of the configured LoRA target modules exist in {resolved.model_id}. "
            f"Set training.lora_target_modules explicitly."
        )
    missing = set(tcfg.lora_target_modules) - set(targets)
    if missing:
        logger.info(f"LoRA targets not present in this model, skipped: {sorted(missing)}")

    resume_checkpoint = None
    if tcfg.resume_from_checkpoint:
        resume_checkpoint = _find_latest_checkpoint(ws.checkpoints)

    if resume_checkpoint is not None:
        logger.info(f"resuming from {resume_checkpoint.name}")
        model = PeftModel.from_pretrained(model, str(resume_checkpoint), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=tcfg.lora_r,
                lora_alpha=tcfg.lora_alpha,
                lora_dropout=tcfg.lora_dropout,
                bias=tcfg.lora_bias,
                task_type="CAUSAL_LM",
                target_modules=targets,
            ),
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA r={tcfg.lora_r} alpha={tcfg.lora_alpha} on {len(targets)} module type(s): "
        f"{trainable:,} trainable / {total:,} total ({100 * trainable / total:.3f}%)"
    )

    # ---- schedule --------------------------------------------------------
    effective_batch = tcfg.per_device_train_batch_size * tcfg.gradient_accumulation_steps
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    planned_steps = tcfg.max_steps if tcfg.max_steps > 0 else int(steps_per_epoch * tcfg.num_train_epochs)
    logger.info(
        f"schedule: {len(train_ds):,} samples, effective batch {effective_batch}, "
        f"{steps_per_epoch} steps/epoch x {tcfg.num_train_epochs} epochs = ~{planned_steps} steps"
    )

    args_kwargs: dict[str, Any] = {
        "output_dir": str(ws.checkpoints),
        "overwrite_output_dir": False,
        "num_train_epochs": tcfg.num_train_epochs,
        "per_device_train_batch_size": tcfg.per_device_train_batch_size,
        "per_device_eval_batch_size": tcfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
        "learning_rate": tcfg.learning_rate,
        "lr_scheduler_type": tcfg.lr_scheduler_type,
        "warmup_ratio": tcfg.warmup_ratio,
        "weight_decay": tcfg.weight_decay,
        "max_grad_norm": tcfg.max_grad_norm,
        "optim": resolved.optim,
        "logging_steps": tcfg.logging_steps,
        "logging_first_step": True,
        "save_steps": tcfg.save_steps,
        "save_total_limit": tcfg.save_total_limit,
        "save_strategy": "steps",
        "bf16": resolved.bf16,
        "fp16": resolved.fp16,
        "gradient_checkpointing": resolved.gradient_checkpointing,
        "seed": tcfg.seed,
        "data_seed": tcfg.seed,
        "report_to": [],
        "disable_tqdm": False,
        "remove_unused_columns": False,
        "label_names": ["labels"],
    }

    # Live loss curves when tensorboard is installed. Opt-in by presence
    # rather than by config: if the user installed it they want it, and if
    # they did not, enabling it would crash the run at step 0.
    if importlib.util.find_spec("tensorboard") is not None:
        tb_dir = ws.training / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        args_kwargs["report_to"] = ["tensorboard"]
        args_kwargs["logging_dir"] = str(tb_dir)
        logger.info(f"tensorboard logging enabled — tensorboard --logdir {tb_dir}")
    if tcfg.max_steps > 0:
        args_kwargs["max_steps"] = tcfg.max_steps
    if resolved.gradient_checkpointing:
        args_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if eval_ds is not None and len(eval_ds) > 0:
        args_kwargs["eval_steps"] = tcfg.eval_steps
        args_kwargs["load_best_model_at_end"] = False
        # Renamed in transformers 4.46; set both and let the filter below
        # keep whichever this version actually has.
        args_kwargs["eval_strategy"] = "steps"
        args_kwargs["evaluation_strategy"] = "steps"

    args_kwargs, dropped = _supported_kwargs(TrainingArguments, args_kwargs)
    if dropped:
        logger.debug(f"TrainingArguments options not supported by this transformers version: {dropped}")

    training_args = TrainingArguments(**args_kwargs)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # ---- train -----------------------------------------------------------
    metrics = MetricsCollector(ws.train_metrics)
    started = time.time()
    interrupted = False

    # Reset the allocator's high-water mark so peak usage reflects this run
    # rather than anything the model load left behind.
    if resolved.device == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    try:
        result = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
        train_output = result.metrics
    except KeyboardInterrupt:
        interrupted = True
        train_output = {}
        logger.warning("training interrupted — saving the adapter as-is")

    duration = time.time() - started
    metrics.ingest(trainer.state.log_history)

    # Peak memory. On CUDA this is what tells you whether the batch size and
    # sequence length actually fit, or merely happened to fit this time.
    memory: dict[str, Any] = {}
    if resolved.device == "cuda":
        try:
            memory = {
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                "total_vram_gb": round(hardware.vram_gb, 2),
                "peak_utilisation_pct": (
                    round(100 * torch.cuda.max_memory_reserved() / (hardware.vram_gb * 1e9), 1)
                    if hardware.vram_gb else None
                ),
            }
        except Exception:
            memory = {}
    else:
        try:
            import resource  # POSIX only

            memory = {
                "peak_rss_gb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2
                )
            }
        except ImportError:
            try:
                import psutil

                memory = {
                    "peak_rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2)
                }
            except ImportError:
                memory = {}

    # ---- final eval ------------------------------------------------------
    final_eval: dict[str, float] = {}
    if eval_ds is not None and len(eval_ds) > 0 and not interrupted:
        try:
            final_eval = trainer.evaluate()
            metrics.ingest(trainer.state.log_history)
        except Exception as exc:
            logger.warning(f"final evaluation failed: {type(exc).__name__}: {exc}")

    # ---- save ------------------------------------------------------------
    ws.adapter.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(ws.adapter))
    tokenizer.save_pretrained(str(ws.adapter))
    logger.info(f"adapter saved to {ws.adapter}")

    eval_loss = final_eval.get("eval_loss")
    if eval_loss is None and metrics.eval_log:
        eval_loss = metrics.eval_log[-1]["eval_loss"]

    # A resume that finds a checkpoint already past max_steps runs zero steps
    # and HuggingFace reports train_loss = 0.0. Taking that at face value
    # would overwrite the real loss from the original run with a zero, so
    # fall back to the merged log history whenever nothing actually trained.
    train_loss = train_output.get("train_loss")
    if not train_loss and metrics.train_log:
        train_loss = metrics.train_log[-1]["loss"]

    payload = {
        "model_id": resolved.model_id,
        "adapter_path": str(ws.adapter),
        "hardware": hardware.summary(),
        "resolved": resolved.summary(),
        "dataset": {
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds else 0,
            "train_file": str(ws.train_jsonl),
            "eval_file": str(ws.val_jsonl),
        },
        "lora": {
            "r": tcfg.lora_r,
            "alpha": tcfg.lora_alpha,
            "dropout": tcfg.lora_dropout,
            "target_modules": targets,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
        },
        "schedule": {
            "epochs": tcfg.num_train_epochs,
            "effective_batch_size": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "planned_steps": planned_steps,
            "completed_steps": int(trainer.state.global_step),
            "learning_rate": tcfg.learning_rate,
            "lr_scheduler": tcfg.lr_scheduler_type,
            "max_seq_length": max_len,
            "completion_only_loss": tcfg.train_on_completions_only,
        },
        "results": {
            "train_loss": round(float(train_loss), 6) if train_loss is not None else None,
            "eval_loss": round(float(eval_loss), 6) if eval_loss is not None else None,
            "eval_perplexity": (
                round(math.exp(min(float(eval_loss), 20)), 4) if eval_loss is not None else None
            ),
            "train_runtime_seconds": round(duration, 2),
            "samples_per_second": (
                round(len(train_ds) * tcfg.num_train_epochs / duration, 3) if duration > 0 else 0.0
            ),
            "interrupted": interrupted,
        },
        "memory": memory,
        "train_log": metrics.train_log,
        "eval_log": metrics.eval_log,
        "seed": tcfg.seed,
        "finished_at": time.time(),
    }
    write_json(payload, ws.train_metrics)

    logger.info(
        f"training complete in {duration / 60:.1f} min — "
        f"{int(trainer.state.global_step)} steps, "
        f"train loss {payload['results']['train_loss']}, "
        f"eval loss {payload['results']['eval_loss']}"
    )

    # Strip the full logs from the phase summary; they live in metrics.json.
    summary = {k: v for k, v in payload.items() if k not in ("train_log", "eval_log")}
    return summary
