"""
conftest.py
-----------
pytest configuration for the Water-Smart Crop Planner test suite.

Ensures all tests run with the project root as the working directory so that
relative paths in config/config.yaml resolve correctly.
"""

from __future__ import annotations

import os
from pathlib import Path

# Change to project root before any test runs
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
