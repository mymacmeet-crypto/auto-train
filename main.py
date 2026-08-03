#!/usr/bin/env python3
"""
Entry point so the pipeline runs without installing the package:

    python main.py https://example.com

Equivalent to the `autotrain` console script created by `pip install -e .`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running straight from a checkout.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from autotrain.cli.app import main

if __name__ == "__main__":
    main()
