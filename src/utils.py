"""Shared utilities for NutriNLP."""
from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import nltk
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_global_seed(seed: int = 42) -> None:
    """Set reproducible seeds where supported."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    """Create all output directories used by the project."""
    for rel in [
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "reports/figures",
        "reports/confusion_matrices",
        "reports/classification_reports",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """Convert NumPy, pandas, and pathlib objects to JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary as formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(obj: Any, path: Path) -> None:
    """Persist a Python object with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def library_versions() -> dict[str, str]:
    """Return reproducibility-relevant runtime versions."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "nltk": nltk.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
