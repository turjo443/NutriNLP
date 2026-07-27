"""Training and evaluation of the eight required calorie classifiers."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data_preparation import CLASS_NAMES
from .text_preprocessing import FoodTextPreprocessor
from .utils import save_json, save_joblib


@dataclass
class ClassifierTrainingOutput:
    """Artifacts and tables produced by classifier training."""

    comparison: pd.DataFrame
    best_model_name: str
    best_pipeline: Pipeline
    vectorizer_settings: dict[str, Any]
    all_models: dict[str, Pipeline]
    reports: dict[str, dict[str, Any]]


def _tfidf(settings: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
        ngram_range=tuple(settings["ngram_range"]),
        min_df=int(settings["min_df"]),
        max_df=float(settings["max_df"]),
        max_features=int(settings["max_features"]),
        sublinear_tf=bool(settings.get("sublinear_tf", True)),
        norm="l2",
    )


def select_vectorizer_settings(
    X_train: pd.Series,
    y_train: pd.Series,
    cv: StratifiedKFold,
    mode: str,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select TF-IDF settings using only training folds and Logistic Regression."""
    pipe = Pipeline(
        [
            ("preprocess", FoodTextPreprocessor()),
            ("tfidf", _tfidf({"ngram_range": (1, 2), "min_df": 1, "max_df": 1.0, "max_features": 5000, "sublinear_tf": True})),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="saga",
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    if mode == "quick":
        grid = {
            "tfidf__min_df": [1],
            "tfidf__max_df": [1.0],
            "tfidf__max_features": [3000],
            "tfidf__ngram_range": [(1, 2)],
        }
    else:
        grid = [
            {"tfidf__min_df": [1], "tfidf__max_df": [1.0], "tfidf__max_features": [5000], "tfidf__ngram_range": [(1, 2)]},
            {"tfidf__min_df": [2], "tfidf__max_df": [1.0], "tfidf__max_features": [5000], "tfidf__ngram_range": [(1, 2)]},
            {"tfidf__min_df": [1], "tfidf__max_df": [0.95], "tfidf__max_features": [3000], "tfidf__ngram_range": [(1, 2)]},
            {"tfidf__min_df": [2], "tfidf__max_df": [0.95], "tfidf__max_features": [3000], "tfidf__ngram_range": [(1, 2)]},
        ]
    search = GridSearchCV(
        pipe,
        grid,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best = {
        "ngram_range": list(search.best_params_["tfidf__ngram_range"]),
        "min_df": int(search.best_params_["tfidf__min_df"]),
        "max_df": float(search.best_params_["tfidf__max_df"]),
        "max_features": int(search.best_params_["tfidf__max_features"]),
        "sublinear_tf": True,
    }
    cols = [
        "params",
        "mean_test_f1_macro",
        "std_test_f1_macro",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_fit_time",
    ]
    results = pd.DataFrame(search.cv_results_)[cols].sort_values("mean_test_f1_macro", ascending=False)
    results["params"] = results["params"].apply(json.dumps)
    return best, results


def _model_specs(settings: dict[str, Any], seed: int, mode: str) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    def base(model: Any) -> Pipeline:
        return Pipeline([("preprocess", FoodTextPreprocessor()), ("tfidf", _tfidf(settings)), ("model", model)])

    specs: dict[str, tuple[Pipeline, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            base(LogisticRegression(max_iter=1500, solver="saga", tol=1e-3, class_weight="balanced", random_state=seed)),
            {"model__C": [1.0] if mode == "quick" else [0.5, 1.0, 2.0]},
        ),
        "Multinomial Naive Bayes": (
            base(MultinomialNB()),
            {"model__alpha": [1.0] if mode == "quick" else [0.25, 0.5, 1.0]},
        ),
        "Decision Tree": (
            base(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            {
                "model__max_depth": [20] if mode == "quick" else [12, 20, None],
                "model__min_samples_leaf": [2] if mode == "quick" else [1, 3],
            },
        ),
        "Random Forest": (
            base(
                RandomForestClassifier(
                    n_estimators=120,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                )
            ),
            {
                "model__max_depth": [None] if mode == "quick" else [20, None],
                "model__min_samples_leaf": [1],
                "model__max_features": ["sqrt"],
            },
        ),
        "Support Vector Machine": (
            base(LinearSVC(class_weight="balanced", random_state=seed, dual="auto", max_iter=10000)),
            {"model__C": [1.0] if mode == "quick" else [0.25, 0.5, 1.0, 2.0]},
        ),
        "K-Nearest Neighbors": (
            Pipeline(
                [
                    ("preprocess", FoodTextPreprocessor()),
                    ("tfidf", _tfidf(settings)),
                    ("svd", TruncatedSVD(n_components=60, random_state=seed)),
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier(algorithm="brute", metric="euclidean")),
                ]
            ),
            {
                "model__n_neighbors": [9] if mode == "quick" else [7, 15],
                "model__weights": ["distance"],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                [
                    ("preprocess", FoodTextPreprocessor()),
                    ("tfidf", _tfidf(settings)),
                    ("svd", TruncatedSVD(n_components=40, random_state=seed)),
                    ("scale", StandardScaler()),
                    ("model", GradientBoostingClassifier(random_state=seed)),
                ]
            ),
            {
                "model__n_estimators": [60] if mode == "quick" else [60, 100],
                "model__learning_rate": [0.1],
                "model__max_depth": [2],
            },
        ),
        "XGBoost": (
            base(
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=0,
                )
            ),
            {
                "model__n_estimators": [150] if mode == "quick" else [150, 250],
                "model__max_depth": [4] if mode == "quick" else [3, 5],
                "model__learning_rate": [0.08],
                "model__subsample": [0.9],
                "model__colsample_bytree": [0.9],
            },
        ),
    }
    return specs


def _metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray, search: GridSearchCV, prediction_time: float) -> dict[str, Any]:
    idx = search.best_index_
    cv = search.cv_results_
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cv_macro_f1_mean": float(cv["mean_test_f1_macro"][idx]),
        "cv_macro_f1_std": float(cv["std_test_f1_macro"][idx]),
        "cv_accuracy_mean": float(cv["mean_test_accuracy"][idx]),
        "cv_accuracy_std": float(cv["std_test_accuracy"][idx]),
        "training_time_seconds": float(search.refit_time_),
        "prediction_time_seconds": prediction_time,
        "best_params": json.dumps(search.best_params_, sort_keys=True),
    }


def _save_confusion_matrix(cm: np.ndarray, model_name: str, out_path: Path) -> None:
    labels = [CLASS_NAMES[i] for i in range(3)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(3), labels=labels, rotation=25, ha="right")
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_all_classifiers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reports_dir: Path,
    models_dir: Path,
    mode: str = "full",
    seed: int = 42,
    cv_folds: int = 5,
    n_jobs: int = 1,
) -> ClassifierTrainingOutput:
    """Tune, fit, and evaluate all eight required models."""
    X_train = train_df["dish_text"].astype(str)
    y_train = train_df["calorie_class_id"].astype(int)
    X_test = test_df["dish_text"].astype(str)
    y_test = test_df["calorie_class_id"].astype(int)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    vectorizer_settings, vectorizer_results = select_vectorizer_settings(X_train, y_train, cv, mode, seed, n_jobs)
    vectorizer_results.to_csv(reports_dir / "vectorizer_search_results.csv", index=False)

    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}

    for name, (pipeline, param_grid) in _model_specs(vectorizer_settings, seed, mode).items():
        print(f"[classification] Training {name} ...", flush=True)
        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            refit="f1_macro",
            cv=cv,
            n_jobs=n_jobs,
            verbose=0,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        start = time.perf_counter()
        pred = search.predict(X_test)
        prediction_time = time.perf_counter() - start
        rows.append(_metric_row(name, y_test, pred, search, prediction_time))

        report_dict = classification_report(
            y_test,
            pred,
            labels=[0, 1, 2],
            target_names=[CLASS_NAMES[i] for i in range(3)],
            output_dict=True,
            zero_division=0,
        )
        reports[name] = report_dict
        report_df = pd.DataFrame(report_dict).transpose()
        safe = name.lower().replace(" ", "_").replace("-", "_")
        report_df.to_csv(reports_dir / "classification_reports" / f"{safe}.csv")
        save_json(report_dict, reports_dir / "classification_reports" / f"{safe}.json")
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
        _save_confusion_matrix(cm, name, reports_dir / "confusion_matrices" / f"{safe}.png")
        fitted[name] = search.best_estimator_

    comparison = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_mean", "macro_f1", "accuracy"], ascending=False
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    best_model_name = str(comparison.iloc[0]["model"])
    best_pipeline = fitted[best_model_name]
    save_joblib(best_pipeline, models_dir / "best_calorie_classifier.joblib")
    save_joblib(fitted, models_dir / "all_classifiers.joblib")

    return ClassifierTrainingOutput(
        comparison=comparison,
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        vectorizer_settings=vectorizer_settings,
        all_models=fitted,
        reports=reports,
    )
