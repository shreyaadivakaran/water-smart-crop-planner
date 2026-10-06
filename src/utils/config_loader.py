"""
config_loader.py
----------------
Loads and validates the master YAML configuration file.

Usage
-----
    from src.utils.config_loader import load_config
    cfg = load_config()                        # uses default path
    cfg = load_config("config/config.yaml")    # explicit path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Keys that must be present at the top level of the config file.
_REQUIRED_TOP_LEVEL_KEYS: list[str] = [
    "district",
    "study_period",
    "crops",
    "seasons",
    "data",
    "models",
    "risk",
    "optimization",
    "weights_default",
]

# Required sub-keys within data.use_synthetic
_REQUIRED_SYNTHETIC_KEYS: list[str] = [
    "weather",
    "yield",
    "prices",
    "crop_water",
    "cost",
]


def load_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the YAML configuration file and validate required keys.

    Parameters
    ----------
    path:
        Absolute or relative path to the YAML config file.
        Defaults to ``config/config.yaml`` relative to the project root,
        which is resolved as two levels up from this file's location.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    ValueError
        If any required top-level key or sub-key is missing.
    """
    if path is None:
        # Resolve project root (two levels up from src/utils/)
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "config" / "config.yaml"

    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Create config/config.yaml or pass an explicit path to load_config()."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError if required keys are absent.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary.
    """
    missing_top = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing_top:
        raise ValueError(
            f"Config is missing required top-level keys: {missing_top}"
        )

    # Check per-dataset synthetic flags
    use_syn = cfg.get("data", {}).get("use_synthetic", {})
    if not isinstance(use_syn, dict):
        raise ValueError(
            "config.data.use_synthetic must be a mapping of dataset names to "
            "booleans, e.g. {weather: true, yield: true, ...}"
        )
    missing_syn = [k for k in _REQUIRED_SYNTHETIC_KEYS if k not in use_syn]
    if missing_syn:
        raise ValueError(
            f"config.data.use_synthetic is missing keys: {missing_syn}"
        )

    # Validate weights sum to ~1.0
    weights = cfg.get("weights_default", {})
    total = sum(weights.values()) if weights else 0.0
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"weights_default values must sum to 1.0, got {total:.4f}."
        )


def get_synthetic_flags(cfg: dict[str, Any]) -> dict[str, bool]:
    """Return the per-dataset synthetic flags as a plain dict.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary from :func:`load_config`.

    Returns
    -------
    dict[str, bool]
        Keys: weather, yield, prices, crop_water, cost.
    """
    return dict(cfg["data"]["use_synthetic"])


def any_synthetic(cfg: dict[str, Any]) -> bool:
    """Return True if at least one dataset is flagged as synthetic.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary from :func:`load_config`.
    """
    return any(get_synthetic_flags(cfg).values())


def synthetic_dataset_names(cfg: dict[str, Any]) -> list[str]:
    """Return the names of datasets currently flagged as synthetic.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary from :func:`load_config`.
    """
    return [k for k, v in get_synthetic_flags(cfg).items() if v]
