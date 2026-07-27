from pathlib import Path

import pandas as pd
import pytest

from src.data_validation import REQUIRED_COLUMNS, validate_raw_dataset


def test_required_csv_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({c: [] for c in REQUIRED_COLUMNS})
    path = tmp_path / "ok.csv"
    df.to_csv(path, index=False)
    result = validate_raw_dataset(path)
    assert result.audit["missing_columns"] == []


def test_missing_required_column_fails(tmp_path: Path) -> None:
    df = pd.DataFrame({c: [] for c in REQUIRED_COLUMNS if c != "protein"})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_dataset(path)


def test_numeric_validation_excludes_negative(tmp_path: Path) -> None:
    row = {
        "dish_id": "d1",
        "ingr_id": "i1",
        "ingr_name": "rice",
        "grams": 10,
        "calories": -1,
        "fat": 0,
        "carb": 2,
        "protein": 1,
    }
    path = tmp_path / "negative.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    result = validate_raw_dataset(path)
    assert result.audit["excluded_row_count"] == 1
    assert result.data.empty
