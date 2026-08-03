"""Phase 5: instruction dataset generation."""

from autotrain.dataset_builder.builder import (
    Window,
    build_windows,
    pair_to_record,
    run_generation,
    split_into_windows,
)
from autotrain.dataset_builder.prompts import SPECS, DocSignals, analyse, applicable_types
from autotrain.dataset_builder.providers import (
    LLMSampleGenerator,
    Pair,
    StructuralSampleGenerator,
    parse_pairs,
    resolve_provider,
)

__all__ = [
    "Window",
    "build_windows",
    "pair_to_record",
    "run_generation",
    "split_into_windows",
    "SPECS",
    "DocSignals",
    "analyse",
    "applicable_types",
    "LLMSampleGenerator",
    "Pair",
    "StructuralSampleGenerator",
    "parse_pairs",
    "resolve_provider",
]
