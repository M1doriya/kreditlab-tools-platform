# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration — ensure `src/` is importable without an install."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
