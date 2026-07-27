"""Raw dataset validation and cleaning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "dish_id",
    "ingr_id",
    "ingr_name",
    "grams",
    "calories",
    "fat",
    "carb",
    "protein",
]
NUMERIC_COLUMNS = ["grams", "calories", "fat", "carb", "protein"]


@dataclass
class ValidationResult:
    """Container for cleaned data and an audit report."""

    data: pd.DataFrame
    audit: dict[str, Any]


def validate_raw_dataset(path: str | Path) -> ValidationResult:
    """Load, validate, and minimally clean the ingredient-level CSV.

    Rows are excluded only if an identifier is missing, a numeric nutrition field is
    non-numeric/missing, or a numeric value is negative. A missing ingredient name is
    retained for nutrition aggregation but contributes no text token.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    original_columns = list(df.columns)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    raw_rows = len(df)
    missing_before = df.isna().sum().to_dict()
    exact_duplicates = int(df.duplicated().sum())

    invalid_reasons = pd.Series("", index=df.index, dtype="object")
    invalid_id = df["dish_id"].isna() | df["ingr_id"].isna()
    invalid_reasons.loc[invalid_id] += "missing_identifier;"

    numeric_non_numeric: dict[str, int] = {}
    numeric_missing: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        original_missing = df[col].isna()
        converted = pd.to_numeric(df[col], errors="coerce")
        numeric_non_numeric[col] = int((converted.isna() & ~original_missing).sum())
        numeric_missing[col] = int(converted.isna().sum())
        negative_counts[col] = int((converted < 0).sum())
        invalid_reasons.loc[converted.isna()] += f"invalid_{col};"
        invalid_reasons.loc[converted < 0] += f"negative_{col};"
        df[col] = converted

    invalid_mask = invalid_reasons.ne("")
    excluded = df.loc[invalid_mask].copy()
    if not excluded.empty:
        excluded["exclusion_reason"] = invalid_reasons.loc[invalid_mask]

    cleaned = df.loc[~invalid_mask].copy()
    cleaned["ingr_name"] = cleaned["ingr_name"].astype("string")
    cleaned["dish_id"] = cleaned["dish_id"].astype(str)
    cleaned["ingr_id"] = cleaned["ingr_id"].astype(str)

    audit = {
        "input_path": str(path),
        "raw_row_count": raw_rows,
        "valid_row_count": int(len(cleaned)),
        "excluded_row_count": int(invalid_mask.sum()),
        "original_columns": original_columns,
        "expected_columns": REQUIRED_COLUMNS,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "dtypes_after_conversion": {k: str(v) for k, v in cleaned.dtypes.items()},
        "missing_values_before_cleaning": {k: int(v) for k, v in missing_before.items()},
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_dish_ingredient_pairs": int(cleaned.duplicated(["dish_id", "ingr_id"]).sum()),
        "numeric_non_numeric_counts": numeric_non_numeric,
        "numeric_missing_counts": numeric_missing,
        "negative_value_counts": negative_counts,
        "missing_ingredient_name_rows_retained": int(cleaned["ingr_name"].isna().sum()),
        "unique_dishes": int(cleaned["dish_id"].nunique()),
        "unique_ingredient_names": int(cleaned["ingr_name"].nunique(dropna=True)),
        "units": {
            "grams": "g",
            "calories": "kcal",
            "fat": "g",
            "carb": "g",
            "protein": "g",
        },
        "exclusion_reasons": excluded.get("exclusion_reason", pd.Series(dtype=str)).value_counts().to_dict(),
    }
    return ValidationResult(cleaned, audit)
