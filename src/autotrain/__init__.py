"""
autotrain — autonomous website-to-fine-tuned-Qwen training pipeline.

Give it a domain; it crawls the site, extracts and cleans the content,
synthesises an instruction dataset, fine-tunes a Qwen model with LoRA/QLoRA,
evaluates it, exports it, benchmarks it against the base model, and writes
a report. No steps in between require a human.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
