"""End-to-end NutriNLP data, model, report, and artifact pipeline."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from src.data_preparation import aggregate_to_dish_level, create_split_and_labels, save_processed_dataset
from src.data_validation import validate_raw_dataset
from src.evaluation import generate_eda_reports, generate_model_charts, write_documentation
from src.train_classifiers import train_all_classifiers
from src.train_nutrition_estimator import TARGETS, train_nutrition_estimator
from src.utils import (
    PROJECT_ROOT,
    ensure_directories,
    library_versions,
    save_json,
    set_global_seed,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the complete NutriNLP project")
    parser.add_argument("--input", required=True, help="Path to the ingredient-level CSV")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for GridSearchCV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    seed = int(config["project"]["random_seed"])
    set_global_seed(seed)

    input_path = Path(args.input).expanduser().resolve()
    raw_copy = PROJECT_ROOT / "data" / "raw" / "dish_ingredients.csv"
    if input_path != raw_copy.resolve():
        shutil.copy2(input_path, raw_copy)
    print(f"[data] Validating {input_path}", flush=True)
    validation = validate_raw_dataset(input_path)
    raw_df = validation.data
    audit = validation.audit
    save_json(audit, PROJECT_ROOT / "reports" / "raw_data_audit.json")

    print("[data] Aggregating ingredient rows to dish level", flush=True)
    dish_df = aggregate_to_dish_level(raw_df)
    labeled, split_meta = create_split_and_labels(
        dish_df,
        test_size=float(config["project"]["test_size"]),
        random_state=seed,
        quantiles=tuple(config["project"]["class_quantiles"]),
    )
    save_processed_dataset(labeled, PROJECT_ROOT / "data" / "processed" / "dish_level_dataset.csv")
    save_json(split_meta, PROJECT_ROOT / "models" / "class_thresholds.json")

    data_summary = generate_eda_reports(raw_df, labeled, PROJECT_ROOT / "reports")
    save_json(data_summary, PROJECT_ROOT / "reports" / "dataset_summary.json")

    train_df = labeled[labeled["split"] == "train"].copy()
    test_df = labeled[labeled["split"] == "test"].copy()
    if set(train_df["dish_id"]) & set(test_df["dish_id"]):
        raise AssertionError("dish_id leakage detected before model training")

    classifier_output = train_all_classifiers(
        train_df,
        test_df,
        reports_dir=PROJECT_ROOT / "reports",
        models_dir=PROJECT_ROOT / "models",
        mode=args.mode,
        seed=seed,
        cv_folds=int(config["training"]["cv_folds"]),
        n_jobs=args.n_jobs,
    )
    generate_model_charts(classifier_output.comparison, PROJECT_ROOT / "reports")

    nutrition_output = train_nutrition_estimator(
        train_df,
        test_df,
        raw_df=raw_df,
        vectorizer_settings=classifier_output.vectorizer_settings,
        reports_dir=PROJECT_ROOT / "reports",
        models_dir=PROJECT_ROOT / "models",
        mode=args.mode,
        seed=seed,
        cv_folds=int(config["training"]["cv_folds"]),
        n_jobs=args.n_jobs,
    )

    ingredient_vocabulary = sorted(raw_df["ingr_name"].dropna().astype(str).str.strip().unique().tolist())
    save_json({"ingredients": ingredient_vocabulary}, PROJECT_ROOT / "models" / "ingredient_vocabulary.json")

    nutrition_mae = dict(zip(nutrition_output.metrics["target"], nutrition_output.metrics["mae"]))
    mean_r2 = float(nutrition_output.metrics["r2"].mean())
    low_r2_count = int((nutrition_output.metrics["r2"] < 0.2).sum())
    reliability = "low" if low_r2_count >= 2 else "moderate" if mean_r2 < 0.5 else "higher"
    best_row = classifier_output.comparison.loc[
        classifier_output.comparison["model"] == classifier_output.best_model_name
    ].iloc[0]

    metadata = {
        "project_name": "NutriNLP",
        "training_mode": args.mode,
        "training_date_utc": utc_now_iso(),
        "random_seed": seed,
        "dataset": {
            "raw_rows": audit["raw_row_count"],
            "valid_rows": audit["valid_row_count"],
            "excluded_rows": audit["excluded_row_count"],
            "unique_dishes": data_summary["unique_dishes"],
            "train_dishes": split_meta["train_dishes"],
            "test_dishes": split_meta["test_dishes"],
        },
        "selected_model_name": classifier_output.best_model_name,
        "selected_model_metrics": {
            k: (float(best_row[k]) if isinstance(best_row[k], (int, float)) else best_row[k])
            for k in [
                "accuracy",
                "macro_precision",
                "weighted_precision",
                "macro_recall",
                "weighted_recall",
                "macro_f1",
                "weighted_f1",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "training_time_seconds",
                "prediction_time_seconds",
            ]
        },
        "class_names": {0: "Low Calorie", 1: "Medium Calorie", 2: "High Calorie"},
        "class_thresholds": split_meta,
        "feature_settings": classifier_output.vectorizer_settings,
        "nutrition_targets": TARGETS,
        "nutrition_uncertainty_mae": nutrition_mae,
        "nutrition_uncertainty_quantiles": nutrition_output.residual_quantiles,
        "nutrition_estimator_metrics": nutrition_output.metrics.to_dict(orient="records"),
        "nutrition_estimator_best_params": nutrition_output.best_params,
        "nutrition_estimator_reliability": reliability,
        "library_versions": library_versions(),
        "notes": {
            "classification_input": "ingredient-name text only",
            "nutrition_estimation": "separate hybrid ingredient-phrase Random Forest + training-only ingredient profiles + Ridge fallback; values are estimates",
            "portion_handling": "explicit g/kg/ml quantities are used when provided; otherwise training-derived typical ingredient portions are assumed",
            "negative_prediction_display": "clipped to zero only at presentation time",
        },
    }
    save_json(metadata, PROJECT_ROOT / "models" / "model_metadata.json")

    write_documentation(
        root=PROJECT_ROOT,
        audit=audit,
        data_summary=data_summary,
        split_meta=split_meta,
        comparison=classifier_output.comparison,
        nutrition_metrics=nutrition_output.metrics,
        best_model_name=classifier_output.best_model_name,
        vectorizer_settings=classifier_output.vectorizer_settings,
        metadata=metadata,
    )
    save_json(
        {
            "status": "success",
            "best_model": classifier_output.best_model_name,
            "raw_rows": audit["raw_row_count"],
            "unique_dishes": data_summary["unique_dishes"],
            "mode": args.mode,
        },
        PROJECT_ROOT / "reports" / "pipeline_summary.json",
    )
    print(f"[done] Best model: {classifier_output.best_model_name}", flush=True)
    print("[done] Artifacts and reports saved successfully", flush=True)


if __name__ == "__main__":
    main()
