"""Train the separate, quantity-aware nutrition estimator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from .nutrition_estimator import HybridNutritionEstimator, ingredient_phrase_analyzer
from .text_preprocessing import FoodTextPreprocessor, normalize_text
from .utils import save_joblib

TARGETS = ["total_calories", "total_protein", "total_carb", "total_fat"]
RAW_TARGETS = ["calories", "protein", "carb", "fat"]
DISPLAY = {
    "total_calories": "Calories (kcal)",
    "total_protein": "Protein (g)",
    "total_carb": "Carbohydrates (g)",
    "total_fat": "Fat (g)",
}


@dataclass
class NutritionTrainingOutput:
    pipeline: HybridNutritionEstimator
    metrics: pd.DataFrame
    best_params: dict[str, Any]
    residual_quantiles: dict[str, dict[str, float]]


def _negative_average_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -float(np.mean(np.mean(np.abs(y_true - y_pred), axis=0)))


def _build_profiles(
    raw_df: pd.DataFrame,
    train_dish_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build ingredient profiles from training rows only."""
    train_raw = raw_df[raw_df["dish_id"].astype(str).isin(train_dish_ids)].copy()
    train_raw = train_raw[train_raw["ingr_name"].notna()].copy()
    train_raw["normalized_ingredient"] = train_raw["ingr_name"].map(normalize_text)
    train_raw = train_raw[train_raw["normalized_ingredient"] != ""].copy()

    profiles: dict[str, dict[str, Any]] = {}
    for key, group in train_raw.groupby("normalized_ingredient", sort=False):
        display_counts = group["ingr_name"].astype(str).str.strip().value_counts()
        display_name = str(display_counts.index[0])
        mean_contribution = group[RAW_TARGETS].mean().to_numpy(dtype=float)
        valid = group[group["grams"] > 0].copy()
        per_gram = valid[RAW_TARGETS].div(valid["grams"], axis=0)
        # Median density is robust to unusual portion sizes and recording outliers.
        median_per_gram = per_gram.replace([np.inf, -np.inf], np.nan).median().fillna(0.0).to_numpy(dtype=float)
        profiles[key] = {
            "display_name": display_name,
            "frequency": int(len(group)),
            "mean_contribution": mean_contribution.tolist(),
            "median_per_gram": median_per_gram.tolist(),
            "median_grams": float(group["grams"].median()),
        }

    # Single-token aliases map common generic words to the most frequent matching
    # ingredient. Exact one-token ingredients always take priority.
    candidates: dict[str, list[tuple[int, str]]] = {}
    preparation_tokens = {
        "fried", "grilled", "roasted", "boiled", "baked", "fresh", "cooked",
        "raw", "chopped", "sliced", "mixed", "country", "plain", "white", "brown",
    }
    for key, profile in profiles.items():
        for token in key.split():
            if len(token) >= 3 and token not in preparation_tokens:
                candidates.setdefault(token, []).append((int(profile["frequency"]), key))
    aliases: dict[str, str] = {}
    for token, options in candidates.items():
        exact = [item for item in options if item[1] == token]
        selected = max(exact or options, key=lambda item: item[0])[1]
        aliases[token] = selected
    return profiles, aliases


def train_nutrition_estimator(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    vectorizer_settings: dict[str, Any],
    reports_dir: Path,
    models_dir: Path,
    mode: str = "full",
    seed: int = 42,
    cv_folds: int = 5,
    n_jobs: int = 1,
) -> NutritionTrainingOutput:
    """Train and evaluate the hybrid ingredient nutrition estimator."""
    X_train = train_df["dish_text"].astype(str)
    y_train = train_df[TARGETS].astype(float).to_numpy()
    X_test = test_df["dish_text"].astype(str)
    y_test = test_df[TARGETS].astype(float).to_numpy()

    # Exact ingredient-phrase model. Unlike L2-normalized TF-IDF, phrase-presence
    # features preserve the additive structure of meals containing more ingredients.
    phrase_vectorizer = CountVectorizer(
        analyzer=ingredient_phrase_analyzer,
        lowercase=False,
        binary=False,
    )
    phrase_train = phrase_vectorizer.fit_transform(X_train).toarray()
    trees = 80 if mode == "quick" else 250
    ingredient_model = RandomForestRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=seed,
        n_jobs=n_jobs,
    )
    print("[nutrition] Training ingredient-phrase Random Forest ...", flush=True)
    ingredient_model.fit(phrase_train, y_train)

    # Word-level fallback for descriptions where no exact ingredient can be mapped.
    fallback = Pipeline(
        [
            ("preprocess", FoodTextPreprocessor()),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    token_pattern=r"(?u)\b[a-z][a-z]+\b",
                    ngram_range=tuple(vectorizer_settings["ngram_range"]),
                    min_df=int(vectorizer_settings["min_df"]),
                    max_df=float(vectorizer_settings["max_df"]),
                    max_features=int(vectorizer_settings["max_features"]),
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            ("model", MultiOutputRegressor(Ridge(solver="lsqr"), n_jobs=1)),
        ]
    )
    alphas = [1.0] if mode == "quick" else [0.1, 1.0, 10.0, 50.0]
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        fallback,
        {"model__estimator__alpha": alphas},
        scoring=make_scorer(_negative_average_mae, greater_is_better=True),
        refit=True,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    print("[nutrition] Training word-level fallback ...", flush=True)
    search.fit(X_train, y_train)

    profiles, aliases = _build_profiles(raw_df, set(train_df["dish_id"].astype(str)))
    estimator = HybridNutritionEstimator(
        phrase_vectorizer=phrase_vectorizer,
        ingredient_model=ingredient_model,
        fallback_model=search.best_estimator_,
        ingredient_profiles=profiles,
        token_aliases=aliases,
        ingredient_model_weight=0.75,
        quantity_profile_weight=0.85,
    )

    pred = estimator.predict(X_test)
    rows: list[dict[str, float | str]] = []
    residual_quantiles: dict[str, dict[str, float]] = {}
    for idx, target in enumerate(TARGETS):
        errors = np.abs(y_test[:, idx] - pred[:, idx])
        mae = mean_absolute_error(y_test[:, idx], pred[:, idx])
        rmse = mean_squared_error(y_test[:, idx], pred[:, idx]) ** 0.5
        r2 = r2_score(y_test[:, idx], pred[:, idx])
        q80 = float(np.quantile(errors, 0.80))
        q90 = float(np.quantile(errors, 0.90))
        residual_quantiles[target] = {"q80": q80, "q90": q90}
        rows.append(
            {
                "target": target,
                "display_name": DISPLAY[target],
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "absolute_error_q80": q80,
                "absolute_error_q90": q90,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports_dir / "nutrition_regression_metrics.csv", index=False)
    prediction_data: dict[str, Any] = {}
    for idx, target in enumerate(TARGETS):
        prediction_data[f"actual_{target}"] = y_test[:, idx]
        prediction_data[f"predicted_{target}"] = pred[:, idx]
    pd.DataFrame(prediction_data).to_csv(reports_dir / "nutrition_test_predictions.csv", index=False)
    save_joblib(estimator, models_dir / "nutrition_estimator.joblib")

    params = {
        "ingredient_model": "RandomForestRegressor",
        "n_estimators": trees,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "ingredient_model_weight": 0.75,
        "quantity_profile_weight": 0.85,
        "fallback_best_params": search.best_params_,
    }
    return NutritionTrainingOutput(estimator, metrics, params, residual_quantiles)
