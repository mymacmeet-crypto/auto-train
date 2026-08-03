"""
Phase 10 — model export.

Produces up to five artifacts from one trained adapter:

  adapter/    the LoRA weights alone (~10-200 MB). Smallest thing that is
              still the fine-tune. Requires the base model at load time.
  merged/     base + adapter folded into one HuggingFace checkpoint. Larger,
              but self-contained and the input every other tool expects.
  gguf/       llama.cpp binary format, quantised. Needed by Ollama,
              llama.cpp, LM Studio.
  ollama/     a Modelfile pointing at the GGUF, plus a one-line install
              script. Optional GGUF; if absent, the Modelfile is still
              written so it works the moment the GGUF appears.
  Hugging Face format is what merged/ already is — a directory with
  config.json, tokenizer files, and safetensors shards, loadable by
  `from_pretrained` and pushable with `huggingface-cli upload`.

Merging is the dependency root: GGUF conversion needs a merged checkpoint,
and Ollama needs the GGUF. Requesting GGUF without merging enables merging
automatically rather than failing late.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from autotrain.utils.config import AutotrainConfig
from autotrain.utils.logging import get_logger
from autotrain.utils.paths import Workspace, slugify_domain
from autotrain.utils.text import read_json, write_json

logger = get_logger("exporter")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def export_adapter(ws: Workspace) -> dict[str, Any]:
    if not ws.adapter.exists() or not any(ws.adapter.iterdir()):
        raise RuntimeError(f"no trained adapter at {ws.adapter} — run the train phase first")

    ws.export_adapter.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in ws.adapter.iterdir():
        target = ws.export_adapter / item.name
        if item.is_file():
            shutil.copy2(item, target)
            copied += 1
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            copied += 1

    size = _dir_size_bytes(ws.export_adapter)
    logger.info(f"adapter exported: {ws.export_adapter} ({_human(size)})")
    return {
        "path": str(ws.export_adapter),
        "files": copied,
        "size_bytes": size,
        "size_human": _human(size),
    }


# ---------------------------------------------------------------------------
# Merged model
# ---------------------------------------------------------------------------


def export_merged(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    """
    Fold the LoRA weights into the base model.

    W_merged = W_base + (alpha / r) * (B @ A), computed once per adapted
    layer by peft's merge_and_unload(), producing a plain checkpoint with no
    adapter dependency.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metrics = read_json(ws.train_metrics, default={}) or {}
    model_id = metrics.get("model_id")
    if not model_id:
        model_id = (
            cfg.training.cpu_model
            if cfg.training.base_model == "auto"
            else cfg.training.base_model
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[cfg.export.merged_dtype]

    logger.info(f"merging adapter into {model_id} (loading in {cfg.export.merged_dtype})")

    # Always merge on CPU: it avoids VRAM pressure and the result is written
    # to disk anyway, so device placement gains nothing.
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, device_map=None
    )
    model = PeftModel.from_pretrained(base, str(ws.adapter), torch_dtype=dtype)
    model = model.merge_and_unload()
    model = model.to(dtype)

    ws.export_merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ws.export_merged), safe_serialization=True)

    tokenizer_source = ws.adapter if (ws.adapter / "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    tokenizer.save_pretrained(str(ws.export_merged))

    del model, base

    size = _dir_size_bytes(ws.export_merged)
    shards = len(list(ws.export_merged.glob("*.safetensors")))
    logger.info(f"merged model exported: {ws.export_merged} ({_human(size)}, {shards} shard(s))")

    return {
        "path": str(ws.export_merged),
        "base_model": model_id,
        "dtype": cfg.export.merged_dtype,
        "shards": shards,
        "size_bytes": size,
        "size_human": _human(size),
        "huggingface_format": True,
    }


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


def export_gguf(cfg: AutotrainConfig, ws: Workspace, model_name: str) -> dict[str, Any]:
    """
    Convert merged -> fp16 GGUF -> quantised GGUF using llama.cpp.

    Kept as a subprocess call to llama.cpp rather than a Python dependency:
    the conversion script tracks upstream model architectures closely, and
    vendoring a copy would go stale.
    """
    llama_dir = Path(os.path.expanduser(cfg.export.llama_cpp_dir))
    if not llama_dir.exists():
        return {
            "skipped": True,
            "reason": (
                f"llama.cpp not found at {llama_dir}. Clone and build it:\n"
                f"  git clone https://github.com/ggerganov/llama.cpp {llama_dir}\n"
                f"  cmake -B build -S {llama_dir} && cmake --build {llama_dir}/build -j\n"
                f"Then re-run with export.export_gguf=true."
            ),
        }

    convert_script = None
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        candidate = llama_dir / name
        if candidate.exists():
            convert_script = candidate
            break
    if convert_script is None:
        return {"skipped": True, "reason": f"convert_hf_to_gguf.py not found in {llama_dir}"}

    quantize_bin = None
    exe = ".exe" if platform.system() == "Windows" else ""
    for candidate in (
        llama_dir / "build" / "bin" / f"llama-quantize{exe}",
        llama_dir / "build" / "bin" / "Release" / f"llama-quantize{exe}",
        llama_dir / f"llama-quantize{exe}",
    ):
        if candidate.exists():
            quantize_bin = candidate
            break

    ws.export_gguf.mkdir(parents=True, exist_ok=True)
    quant = cfg.export.gguf_quantisation
    fp16_path = ws.export_gguf / f"{model_name}.fp16.gguf"
    quant_path = ws.export_gguf / f"{model_name}-{quant}.gguf"

    if not fp16_path.exists():
        logger.info("converting merged model to fp16 GGUF (this takes a few minutes)")
        result = subprocess.run(
            [sys.executable, str(convert_script), str(ws.export_merged),
             "--outtype", "f16", "--outfile", str(fp16_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {
                "skipped": True,
                "reason": f"GGUF conversion failed:\n{result.stderr[-1500:]}",
            }

    if quantize_bin is None:
        logger.warning("llama-quantize binary not found — keeping the fp16 GGUF only")
        return {
            "fp16_path": str(fp16_path),
            "fp16_size_human": _human(fp16_path.stat().st_size),
            "quantised": False,
            "reason": "llama-quantize not built; run cmake --build in the llama.cpp directory",
        }

    if not quant_path.exists():
        logger.info(f"quantising to {quant}")
        result = subprocess.run(
            [str(quantize_bin), str(fp16_path), str(quant_path), quant],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return {
                "fp16_path": str(fp16_path),
                "quantised": False,
                "reason": f"quantisation failed:\n{result.stderr[-1500:]}",
            }

    # The fp16 intermediate is several times larger than the quantised file
    # and has no further use once quantisation succeeds.
    fp16_size = fp16_path.stat().st_size if fp16_path.exists() else 0
    fp16_path.unlink(missing_ok=True)

    size = quant_path.stat().st_size
    logger.info(f"GGUF exported: {quant_path} ({_human(size)})")
    return {
        "path": str(quant_path),
        "quantisation": quant,
        "quantised": True,
        "size_bytes": size,
        "size_human": _human(size),
        "fp16_intermediate_removed_bytes": fp16_size,
    }


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

_MODELFILE = """\
# Ollama Modelfile — {model_name}
# Fine-tuned from {base_model} on content crawled from {domain}
# Generated by autotrain on {generated_at}
#
# Install:
#   cd {ollama_dir}
#   ollama create {model_name} -f Modelfile
#
# Run:
#   ollama run {model_name}

FROM {gguf_reference}

# Qwen uses ChatML. Ollama reads the template embedded in the GGUF when one
# is present; this block is the explicit fallback.
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
<|im_start|>assistant
{{{{ else if eq .Role "assistant" }}}}{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}{{{{ end }}}}\"\"\"

SYSTEM \"\"\"{system_prompt}\"\"\"

# Low temperature: this model exists to answer factual questions about
# {domain}, where variety is a defect rather than a feature.
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Must be at least the training sequence length ({max_seq_length}).
PARAMETER num_ctx {num_ctx}

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""


def export_ollama(
    cfg: AutotrainConfig, ws: Workspace, model_name: str, gguf: dict[str, Any]
) -> dict[str, Any]:
    ws.export_ollama.mkdir(parents=True, exist_ok=True)

    metrics = read_json(ws.train_metrics, default={}) or {}
    base_model = metrics.get("model_id", "unknown")

    gguf_path = gguf.get("path") or gguf.get("fp16_path")
    if gguf_path:
        try:
            reference = os.path.relpath(gguf_path, ws.export_ollama).replace("\\", "/")
        except ValueError:
            reference = str(gguf_path)
    else:
        reference = f"../gguf/{model_name}-{cfg.export.gguf_quantisation}.gguf"

    # num_ctx must cover the training length; round up to a sane power of two.
    num_ctx = max(4096, cfg.training.max_seq_length * 2)

    system_prompt = cfg.dataset.system_prompt.format(
        site_name=slugify_domain(cfg.domain)
    ).replace('"""', "'''")

    content = _MODELFILE.format(
        model_name=model_name,
        base_model=base_model,
        domain=cfg.domain,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        ollama_dir=ws.export_ollama,
        gguf_reference=reference,
        system_prompt=system_prompt,
        max_seq_length=cfg.training.max_seq_length,
        num_ctx=num_ctx,
    )

    modelfile = ws.export_ollama / "Modelfile"
    modelfile.write_text(content, encoding="utf-8")

    install_sh = ws.export_ollama / "install.sh"
    install_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "${BASH_SOURCE[0]}")"\n'
        f'ollama rm {model_name} 2>/dev/null || true\n'
        f'ollama create {model_name} -f Modelfile\n'
        f'echo "installed: {model_name}"\n'
        f'ollama run {model_name} "What is this documentation about?"\n',
        encoding="utf-8",
    )

    install_ps1 = ws.export_ollama / "install.ps1"
    install_ps1.write_text(
        "Set-Location $PSScriptRoot\n"
        f"try {{ ollama rm {model_name} }} catch {{}}\n"
        f"ollama create {model_name} -f Modelfile\n"
        f'Write-Output "installed: {model_name}"\n',
        encoding="utf-8",
    )

    ready = bool(gguf_path and Path(gguf_path).exists())
    logger.info(
        f"Ollama Modelfile written: {modelfile}"
        + ("" if ready else " (GGUF missing — build it, then run install.sh)")
    )

    return {
        "path": str(ws.export_ollama),
        "modelfile": str(modelfile),
        "model_name": model_name,
        "gguf_reference": reference,
        "ready_to_install": ready,
        "install_command": f"cd {ws.export_ollama} && ollama create {model_name} -f Modelfile",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_export(cfg: AutotrainConfig, ws: Workspace) -> dict[str, Any]:
    xcfg = cfg.export
    ws.exports.mkdir(parents=True, exist_ok=True)

    model_name = xcfg.ollama_model_name
    if model_name == "auto":
        slug = slugify_domain(cfg.domain).replace(".", "-").replace("_", "-").lower()
        model_name = f"qwen-{slug}"

    results: dict[str, Any] = {"model_name": model_name}

    # GGUF and Ollama both require a merged checkpoint.
    need_merged = xcfg.export_merged or xcfg.export_gguf
    if xcfg.export_gguf and not xcfg.export_merged:
        logger.info("GGUF export requires a merged model — enabling merge")

    if xcfg.export_adapter:
        results["adapter"] = export_adapter(ws)

    if need_merged:
        try:
            results["merged"] = export_merged(cfg, ws)
        except Exception as exc:
            logger.error(f"merge failed: {type(exc).__name__}: {exc}")
            results["merged"] = {"error": f"{type(exc).__name__}: {exc}"}

    gguf: dict[str, Any] = {"skipped": True, "reason": "export.export_gguf is false"}
    if xcfg.export_gguf:
        merged_ok = isinstance(results.get("merged"), dict) and "error" not in results["merged"]
        if merged_ok:
            try:
                gguf = export_gguf(cfg, ws, model_name)
            except Exception as exc:
                gguf = {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
        else:
            gguf = {"skipped": True, "reason": "merged model unavailable"}
        if gguf.get("skipped"):
            logger.warning(f"GGUF export skipped: {gguf.get('reason', '')[:200]}")
    results["gguf"] = gguf

    if xcfg.export_ollama:
        results["ollama"] = export_ollama(cfg, ws, model_name, gguf)

    results["total_size_human"] = _human(_dir_size_bytes(ws.exports))
    results["exports_root"] = str(ws.exports)
    write_json(results, ws.exports / "export_report.json")

    logger.info(f"exports complete: {ws.exports} ({results['total_size_human']})")
    return results
