"""Phase 4: corpus-level cleaning, deduplication, boilerplate removal."""

from autotrain.cleaner.clean import (
    CleanDoc,
    NearDuplicateIndex,
    find_boilerplate_lines,
    remove_lines,
    run_cleaning,
    strip_nav_runs,
)

__all__ = [
    "CleanDoc",
    "NearDuplicateIndex",
    "find_boilerplate_lines",
    "remove_lines",
    "run_cleaning",
    "strip_nav_runs",
]
