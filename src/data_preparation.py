"""Dish-level aggregation, train/test splitting, and calorie labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = {0: "Low Calorie", 1: "Medium Calorie", 2: "High Calorie"}


def assign_calorie_class(calories: float, low_threshold: float, high_threshold: float) -> int:
    """Assign a reproducible numeric calorie class using fixed thresholds."""
    if calories <= low_threshold:
        return 0
    if calories <= high_threshold:
        return 1
    return 2


def aggregate_to_dish_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ingredient rows into one record per dish_id."""
    working = df.copy()
    working["ingredient_text_part"] = working["ingr_name"].fillna("").astype(str).str.strip()

    def join_names(values: pd.Series) -> str:
        names = [v for v in values.tolist() if v]
        return ", ".join(names)

    dish = (
        working.groupby("dish_id", sort=False)
        .agg(
            dish_text=("ingredient_text_part", join_names),
            total_grams=("grams", "sum"),
            total_calories=("calories", "sum"),
            total_fat=("fat", "sum"),
            total_carb=("carb", "sum"),
            total_protein=("protein", "sum"),
            ingredient_count=("ingr_id", "size"),
            named_ingredient_count=("ingredient_text_part", lambda s: int((s != "").sum())),
        )
        .reset_index()
    )
    if dish["dish_id"].duplicated().any():
        raise AssertionError("Dish aggregation produced duplicate dish_id values")
    if len(dish) != df["dish_id"].nunique():
        raise AssertionError("Dish-level row count does not match unique dish count")
    if (dish["dish_text"].str.strip() == "").any():
        raise ValueError("At least one dish has no usable ingredient-name text")
    return dish


def create_split_and_labels(
    dish_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    quantiles: tuple[float, float] = (0.3333, 0.6667),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a fixed split, derive thresholds from training only, and label all rows."""
    train_ids, test_ids = train_test_split(
        dish_df["dish_id"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_id_set = set(train_ids.tolist())
    test_id_set = set(test_ids.tolist())
    if train_id_set & test_id_set:
        raise AssertionError("dish_id leakage detected between train and test partitions")

    labeled = dish_df.copy()
    labeled["split"] = np.where(labeled["dish_id"].isin(train_id_set), "train", "test")
    train_calories = labeled.loc[labeled["split"] == "train", "total_calories"]
    low_threshold = float(train_calories.quantile(quantiles[0]))
    high_threshold = float(train_calories.quantile(quantiles[1]))
    if not low_threshold < high_threshold:
        raise ValueError("Calculated class thresholds are not strictly increasing")

    labeled["calorie_class_id"] = labeled["total_calories"].apply(
        lambda x: assign_calorie_class(float(x), low_threshold, high_threshold)
    )
    labeled["calorie_class"] = labeled["calorie_class_id"].map(CLASS_NAMES)

    metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "quantiles": list(quantiles),
        "low_threshold_kcal": low_threshold,
        "high_threshold_kcal": high_threshold,
        "train_dishes": int((labeled["split"] == "train").sum()),
        "test_dishes": int((labeled["split"] == "test").sum()),
        "class_names": CLASS_NAMES,
        "train_class_distribution": labeled.loc[labeled["split"] == "train", "calorie_class"].value_counts().to_dict(),
        "test_class_distribution": labeled.loc[labeled["split"] == "test", "calorie_class"].value_counts().to_dict(),
    }
    return labeled, metadata


def save_processed_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Write the final dish-level dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
