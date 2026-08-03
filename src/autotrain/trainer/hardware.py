"""
Hardware detection and capability-aware training defaults.

The spec asks for QLoRA with bitsandbytes. That path requires CUDA:
bitsandbytes has no working CPU 4-bit kernel, and no Metal one. Rather than
crash on machines without a GPU, this module resolves what is *actually*
possible and reports every downgrade explicitly, so the training log states
what ran instead of silently doing something different from the config.

Resolution rules:
    CUDA  -> 4-bit QLoRA, bf16 (or fp16 on pre-Ampere), paged 8-bit optimiser,
             flash-attention-2 when installed
    MPS   -> LoRA in fp16, no quantisation, standard AdamW
    CPU   -> LoRA in fp32, no quantisation, small model, standard AdamW
"""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass, field
from typing import Any

from autotrain.utils.logging import get_logger

logger = get_logger("trainer.hardware")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Hardware:
    device: str = "cpu"
    device_name: str = "cpu"
    gpu_count: int = 0
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    supports_bf16: bool = False
    has_bitsandbytes: bool = False
    has_flash_attention: bool = False
    torch_version: str = ""
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")

    def summary(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "gpu_count": self.gpu_count,
            "vram_gb": round(self.vram_gb, 1),
            "ram_gb": round(self.ram_gb, 1),
            "compute_capability": (
                ".".join(map(str, self.compute_capability)) if self.compute_capability else None
            ),
            "supports_bf16": self.supports_bf16,
            "has_bitsandbytes": self.has_bitsandbytes,
            "has_flash_attention": self.has_flash_attention,
            "torch_version": self.torch_version,
            "platform": self.platform,
        }


def detect_hardware() -> Hardware:
    hardware = Hardware()

    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed — install the 'train' extra")
        return hardware

    hardware.torch_version = torch.__version__

    try:
        import psutil  # optional

        hardware.ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        try:
            if hasattr(os, "sysconf"):
                hardware.ram_gb = (
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                )
        except (ValueError, OSError):
            pass

    if torch.cuda.is_available():
        hardware.device = "cuda"
        hardware.gpu_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        hardware.device_name = props.name
        hardware.vram_gb = props.total_memory / 1e9
        hardware.compute_capability = (props.major, props.minor)
        # bf16 needs Ampere (SM 8.0) or newer.
        hardware.supports_bf16 = props.major >= 8
        hardware.has_bitsandbytes = _module_available("bitsandbytes")
        hardware.has_flash_attention = _module_available("flash_attn")

    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        hardware.device = "mps"
        hardware.device_name = "Apple Silicon (MPS)"
        hardware.supports_bf16 = False

    else:
        hardware.device = "cpu"
        hardware.device_name = platform.processor() or "cpu"

    return hardware


@dataclass
class ResolvedTraining:
    """Config values after reconciling the request with the hardware."""

    device: str
    model_id: str
    load_in_4bit: bool
    torch_dtype: str
    bf16: bool
    fp16: bool
    optim: str
    attn_implementation: str
    gradient_checkpointing: bool
    downgrades: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "model_id": self.model_id,
            "load_in_4bit": self.load_in_4bit,
            "torch_dtype": self.torch_dtype,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "optim": self.optim,
            "attn_implementation": self.attn_implementation,
            "gradient_checkpointing": self.gradient_checkpointing,
            "downgrades": self.downgrades,
            "notes": self.notes,
        }


def resolve_training_setup(training_cfg: Any, hardware: Hardware) -> ResolvedTraining:
    """Reconcile the requested training config with what this box can do."""
    downgrades: list[str] = []
    notes: list[str] = []

    # -- device ----------------------------------------------------------
    requested = training_cfg.device
    if requested == "auto":
        device = hardware.device
    else:
        device = requested
        if requested == "cuda" and hardware.device != "cuda":
            raise RuntimeError(
                "training.device=cuda was requested but no CUDA device is visible. "
                "Set device=auto, or run on a GPU host / inside the CUDA container."
            )
        if requested != hardware.device:
            notes.append(f"device forced to {requested} (detected {hardware.device})")

    # -- model -----------------------------------------------------------
    model_id = training_cfg.base_model
    if model_id == "auto":
        model_id = training_cfg.cuda_model if device == "cuda" else training_cfg.cpu_model
        notes.append(f"base_model=auto resolved to {model_id} for {device}")

    # -- quantisation -----------------------------------------------------
    load_in_4bit = bool(training_cfg.load_in_4bit)
    if load_in_4bit and device != "cuda":
        load_in_4bit = False
        downgrades.append(
            f"4-bit quantisation disabled: bitsandbytes requires CUDA, this run is on {device}. "
            "Training in full precision with LoRA instead."
        )
    elif load_in_4bit and not hardware.has_bitsandbytes:
        load_in_4bit = False
        downgrades.append(
            "4-bit quantisation disabled: bitsandbytes is not installed. "
            "pip install bitsandbytes to enable QLoRA."
        )

    # -- dtype ------------------------------------------------------------
    bf16 = bool(training_cfg.bf16)
    fp16 = bool(training_cfg.fp16)

    if device == "cpu":
        if bf16 or fp16:
            downgrades.append("mixed precision disabled: CPU training runs in float32")
        bf16 = fp16 = False
        torch_dtype = "float32"
    elif device == "mps":
        if bf16:
            downgrades.append("bf16 disabled on MPS; using fp16")
        bf16 = False
        fp16 = True
        torch_dtype = "float16"
    else:
        if bf16 and not hardware.supports_bf16:
            bf16 = False
            fp16 = True
            downgrades.append(
                f"bf16 unsupported on {hardware.device_name} "
                f"(compute capability {hardware.compute_capability}); using fp16"
            )
        if bf16 and fp16:
            fp16 = False
        torch_dtype = "bfloat16" if bf16 else "float16"

    # -- optimiser ---------------------------------------------------------
    optim = training_cfg.optim
    if optim == "auto":
        if device == "cuda" and hardware.has_bitsandbytes:
            optim = "paged_adamw_8bit"
        else:
            optim = "adamw_torch"
        notes.append(f"optim=auto resolved to {optim}")
    elif optim.startswith("paged_") and not (device == "cuda" and hardware.has_bitsandbytes):
        downgrades.append(f"optimiser {optim} needs bitsandbytes+CUDA; using adamw_torch")
        optim = "adamw_torch"

    # -- attention ---------------------------------------------------------
    attn = "eager"
    if training_cfg.use_flash_attention:
        if device == "cuda" and hardware.has_flash_attention:
            attn = "flash_attention_2"
        elif device == "cuda":
            attn = "sdpa"
            notes.append("flash-attn not installed; using PyTorch SDPA")
        else:
            attn = "sdpa" if device == "mps" else "eager"
            notes.append(f"flash attention unavailable on {device}; using {attn}")

    # -- gradient checkpointing -------------------------------------------
    gradient_checkpointing = bool(training_cfg.gradient_checkpointing)
    if gradient_checkpointing and device == "cpu":
        # On CPU this trades ~30% speed for memory we are not short of, and
        # CPU training is already the slow path.
        gradient_checkpointing = False
        notes.append("gradient checkpointing disabled on CPU (costs speed, saves nothing useful)")

    resolved = ResolvedTraining(
        device=device,
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
        bf16=bf16,
        fp16=fp16,
        optim=optim,
        attn_implementation=attn,
        gradient_checkpointing=gradient_checkpointing,
        downgrades=downgrades,
        notes=notes,
    )

    for message in downgrades:
        logger.warning(message)
    for message in notes:
        logger.info(message)

    return resolved
