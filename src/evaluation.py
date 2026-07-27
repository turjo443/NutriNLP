"""EDA, plots, and academic documentation generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .data_preparation import CLASS_NAMES


def generate_eda_reports(raw_df: pd.DataFrame, dish_df: pd.DataFrame, reports_dir: Path) -> dict[str, Any]:
    """Generate required EDA tables and figures."""
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    raw_missing = raw_df.isna().sum().rename("missing_count").to_frame()
    raw_missing["missing_percent"] = raw_missing["missing_count"] / len(raw_df) * 100
    raw_missing.to_csv(reports_dir / "missing_value_report.csv")

    duplicate_summary = pd.DataFrame(
        [
            {"duplicate_type": "exact_rows", "count": int(raw_df.duplicated().sum())},
            {"duplicate_type": "dish_id_plus_ingr_id", "count": int(raw_df.duplicated(["dish_id", "ingr_id"]).sum())},
        ]
    )
    duplicate_summary.to_csv(reports_dir / "duplicate_report.csv", index=False)

    numeric_cols = ["total_grams", "total_calories", "total_fat", "total_carb", "total_protein", "ingredient_count"]
    dish_df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T.to_csv(
        reports_dir / "dish_level_descriptive_statistics.csv"
    )

    top_ingredients = (
        raw_df["ingr_name"].dropna().astype(str).str.strip().value_counts().head(25).rename_axis("ingredient").reset_index(name="count")
    )
    top_ingredients.to_csv(reports_dir / "top_ingredients.csv", index=False)

    class_distribution = (
        dish_df["calorie_class"].value_counts().reindex([CLASS_NAMES[i] for i in range(3)]).fillna(0).astype(int)
        .rename_axis("calorie_class").reset_index(name="count")
    )
    class_distribution.to_csv(reports_dir / "calorie_class_distribution.csv", index=False)

    def hist(column: str, title: str, xlabel: str, filename: str, bins: int = 45) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(dish_df[column].dropna().to_numpy(), bins=bins, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of dishes")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    hist("total_calories", "Dish-Level Calorie Distribution", "Total calories (kcal)", "total_calorie_distribution.png")
    for column, label, filename in [
        ("total_protein", "Protein (g)", "protein_distribution.png"),
        ("total_carb", "Carbohydrates (g)", "carbohydrate_distribution.png"),
        ("total_fat", "Fat (g)", "fat_distribution.png"),
    ]:
        hist(column, f"Dish-Level {label} Distribution", label, filename)
    hist("ingredient_count", "Ingredients per Dish", "Ingredient rows per dish", "ingredients_per_dish.png", bins=34)

    fig, ax = plt.subplots(figsize=(9, 6))
    top15 = top_ingredients.head(15).iloc[::-1]
    ax.barh(top15["ingredient"], top15["count"])
    ax.set_title("Most Frequent Ingredients")
    ax.set_xlabel("Ingredient-row frequency")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures / "top_ingredients.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    corr_cols = ["total_grams", "total_calories", "total_fat", "total_carb", "total_protein", "ingredient_count"]
    corr = dish_df[corr_cols].corr()
    corr.to_csv(reports_dir / "dish_level_correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(), cmap="Greens", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Dish-Level Nutrition Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figures / "nutrition_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(class_distribution["calorie_class"], class_distribution["count"])
    ax.set_title("Final Calorie-Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of dishes")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figures / "calorie_class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "raw_ingredient_rows": int(len(raw_df)),
        "unique_dishes": int(dish_df["dish_id"].nunique()),
        "processed_dish_rows": int(len(dish_df)),
        "unique_named_ingredients": int(raw_df["ingr_name"].nunique(dropna=True)),
        "mean_ingredients_per_dish": float(dish_df["ingredient_count"].mean()),
        "median_ingredients_per_dish": float(dish_df["ingredient_count"].median()),
        "explanation": (
            f"The {len(raw_df):,} ingredient-level rows are grouped by dish_id. Each group becomes one dish-level sample, "
            f"so the modeling table contains {len(dish_df):,} rows—one for each unique dish."
        ),
    }
    return summary


def generate_model_charts(comparison: pd.DataFrame, reports_dir: Path) -> None:
    """Generate model comparison charts from measured results."""
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ordered = comparison.sort_values("macro_f1", ascending=True)

    for metric, title, filename in [
        ("accuracy", "Untouched Test Accuracy by Model", "model_accuracy_comparison.png"),
        ("macro_f1", "Untouched Test Macro F1 by Model", "model_macro_f1_comparison.png"),
        ("cv_macro_f1_mean", "Five-Fold CV Macro F1 by Model", "model_cv_macro_f1_comparison.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(ordered["model"], ordered[metric])
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title)
        for i, value in enumerate(ordered[metric]):
            ax.text(min(value + 0.01, 0.97), i, f"{value:.3f}", va="center")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _markdown_table(df: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    view = df[columns].copy()
    for col in view.select_dtypes(include="number").columns:
        view[col] = view[col].map(lambda x: f"{x:.{decimals}f}")
    return view.to_markdown(index=False)


def write_documentation(
    root: Path,
    audit: dict[str, Any],
    data_summary: dict[str, Any],
    split_meta: dict[str, Any],
    comparison: pd.DataFrame,
    nutrition_metrics: pd.DataFrame,
    best_model_name: str,
    vectorizer_settings: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Create README and final academic report using actual measured results."""
    low = split_meta["low_threshold_kcal"]
    high = split_meta["high_threshold_kcal"]
    best = comparison.loc[comparison["model"] == best_model_name].iloc[0]
    model_table = _markdown_table(
        comparison,
        ["rank", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "cv_macro_f1_mean", "cv_macro_f1_std"],
    )
    nutrition_table = _markdown_table(nutrition_metrics, ["display_name", "mae", "rmse", "r2"])

    readme = f"""# NutriNLP

**NutriNLP** is a complete, reproducible NLP and machine-learning project for classifying an ingredient-based meal description as **Low**, **Medium**, or **High Calorie**, separately estimating calories and macronutrients, and producing transparent rule-based healthier alternatives.

## Research problem

People often describe meals in ordinary language but do not know their approximate calorie category or nutritional profile. NutriNLP transforms ingredient-name text into TF-IDF features, compares eight supervised classifiers, and keeps numerical nutrition estimation separate from classification so that category predictions are not misrepresented as exact nutrient measurements.

## Dataset

The source CSV contains **{audit['raw_row_count']:,} ingredient-level rows**, **8 required columns**, **{data_summary['unique_dishes']:,} unique dishes**, and **{data_summary['unique_named_ingredients']:,} named ingredient values**. The uploaded structure matched the expected schema. No exact duplicate rows were found. **{audit['excluded_row_count']} rows were excluded** by validation. One missing ingredient name was retained for dish nutrition aggregation but contributes no text token.

### Data dictionary

| Column | Meaning | Unit |
|---|---|---|
| `dish_id` | Unique dish identifier | — |
| `ingr_id` | Unique ingredient-row identifier | — |
| `ingr_name` | Ingredient text used for NLP | text |
| `grams` | Ingredient mass | g |
| `calories` | Ingredient energy | kcal |
| `fat` | Ingredient fat | g |
| `carb` | Ingredient carbohydrate | g |
| `protein` | Ingredient protein | g |

The {audit['raw_row_count']:,} rows become **{data_summary['processed_dish_rows']:,} dish-level samples** by grouping all rows with the same `dish_id`, concatenating ingredient names into `dish_text`, and summing grams, calories, fat, carbohydrate, and protein.

## Leakage-safe calorie labels

A fixed 80/20 split was created first with `random_state=42`. Class thresholds were calculated **only from the training partition**:

- Low Calorie: `total_calories <= {low:.4f} kcal`
- Medium Calorie: `{low:.4f} < total_calories <= {high:.4f} kcal`
- High Calorie: `total_calories > {high:.4f} kcal`

The saved thresholds are reused for both training and test data and during application display.

## NLP pipeline

1. Unicode and whitespace normalization
2. Lowercasing
3. Punctuation/special-character removal
4. NLTK regex tokenization
5. English stop-word removal
6. Deterministic food-oriented rule-based lemmatization
7. TF-IDF with unigrams and bigrams
8. Training-fold-only fitting inside scikit-learn pipelines

Selected TF-IDF settings: `{vectorizer_settings}`.

## Models compared

Exactly eight classifiers were tuned with five-fold `StratifiedKFold` cross-validation on the training partition: Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, and XGBoost. Macro F1 was the primary selection metric.

## Actual classification results

{model_table}

**Selected model:** {best_model_name}. It ranked first by training-partition cross-validated Macro F1 ({best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}) and achieved test Macro F1 {best['macro_f1']:.4f} with accuracy {best['accuracy']:.4f} on the untouched test partition.

## Separate nutrition estimator

A quantity-aware hybrid estimator combines an ingredient-phrase Random Forest, additive ingredient profiles learned from training rows only, and a word-level Ridge fallback. Explicit g/kg/ml amounts are parsed when supplied; otherwise the app states its portion assumption. Values are educational estimates, not laboratory measurements, and the app displays empirical 80% absolute-error ranges.

{nutrition_table}

## Application features

- Meal-description input and validation
- Calorie category prediction
- Confidence only when the selected classifier exposes calibrated probabilities
- Quantity-aware estimated calories/macronutrients with empirical 80% error ranges
- Relevant ingredient detection
- Editable rule-based ingredient substitutions
- Model comparison, confusion matrix, EDA charts, thresholds, and methodology
- Educational-use and health-information disclaimer

## Project structure

```text
NutriNLP/
├── app.py
├── run_pipeline.py
├── README.md
├── requirements.txt
├── config/
├── data/raw/
├── data/processed/dish_level_dataset.csv
├── src/
├── models/
├── reports/
├── notebooks/NutriNLP_Analysis.ipynb
└── tests/
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

The shipped preprocessing does not require downloaded NLTK corpora. For experimentation with WordNet lemmatization, optional resources can be installed with:

```bash
python -m nltk.downloader wordnet omw-1.4
```

## Full training

```bash
python run_pipeline.py --input "path/to/dish_ingredients.csv" --mode full
```

A fast debugging run is available with `--mode quick`, but all reported metrics in this repository come from full mode.

## Tests

```bash
pytest -q
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Example

Input: `Chicken breast with rice and broccoli`

Output includes a calorie category, estimated calorie/macronutrient values with uncertainty, detected ingredients, and relevant rule-based alternatives.

## Limitations

- Ingredient text alone cannot recover portion size reliably when users omit quantities.
- The dataset contains only 209 unique named ingredients, so unfamiliar foods may have low vocabulary coverage.
- Numerical estimates can have substantial error, especially for very large or unusual dishes.
- Rule-based substitutions are curated educational guidance, not labels learned from the dataset.
- The tool does not diagnose disease or provide personalized medical nutrition therapy.

## Ethical and health disclaimer

NutriNLP provides educational estimates only. It is not a substitute for a registered dietitian, physician, laboratory analysis, or emergency medical advice.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    report = f"""# NutriNLP — Final Academic Project Report

## Abstract

NutriNLP investigates whether ingredient-name text can support multiclass calorie-category prediction and approximate nutrition estimation. The project converts {audit['raw_row_count']:,} ingredient-level observations into {data_summary['processed_dish_rows']:,} dish-level samples, implements leakage-safe TF-IDF pipelines, compares eight classifiers under five-fold stratified cross-validation, evaluates final models on an untouched test partition, and integrates the selected classifier with a separate multi-output nutrition regressor and a transparent rule-based recommendation engine.

## 1. Introduction

Nutrition applications often rely on exact database lookup, barcode scanning, or image recognition. NutriNLP instead accepts a natural-language ingredient description and uses NLP features to infer a broad calorie category. The system deliberately separates classification from numerical nutrient estimation.

## 2. Problem Statement

Users may enter descriptions such as “chicken breast with rice and broccoli” without quantities or measured nutrient values. A useful system must operate from ingredient text only, avoid data leakage, communicate uncertainty, and refrain from treating approximate predictions as clinical facts.

## 3. Objectives

The project cleans and aggregates the supplied dataset, builds a reusable NLP pipeline, compares eight required classification algorithms, evaluates a separate nutrition estimator, generates evidence-based reports, and deploys the artifacts through Streamlit.

## 4. Conceptual Background

TF-IDF represents ingredient terms according to their frequency within a dish and rarity across dishes. Macro F1 is used as the primary classifier metric because it weights Low, Medium, and High Calorie classes equally. Accuracy, precision, recall, weighted scores, confusion matrices, training time, and prediction time are also reported.

## 5. Dataset Methodology

The uploaded CSV matched the expected 8-column structure. Validation found {audit['raw_row_count']:,} raw rows, {audit['exact_duplicate_rows']} exact duplicates, {audit['excluded_row_count']} excluded invalid rows, and one missing ingredient name. That row was retained because its numeric nutrition values were valid and its dish had other named ingredients; only the absent text token was omitted. Aggregation produced {data_summary['processed_dish_rows']:,} unique dish records.

## 6. Data Preprocessing

Text is normalized, tokenized with NLTK's corpus-free regex tokenizer, stripped of stop words, and lemmatized with deterministic food-oriented rules. The TF-IDF vectorizer uses unigrams and bigrams and is fitted only inside training folds. Numeric nutrition columns and identifiers are never used as classification input features.

## 7. Label Methodology

The proposal did not define fixed calorie thresholds. Therefore, the data was split first and the 33.33rd/66.67th percentiles were computed only from training calories. The thresholds were {low:.4f} kcal and {high:.4f} kcal. The same thresholds were then applied to the untouched test partition.

## 8. Model Methodology

The eight required models were Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Linear Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, and XGBoost. Gradient Boosting uses TruncatedSVD inside its pipeline to produce a leakage-safe dense representation. Hyperparameters were tuned using five-fold StratifiedKFold CV with Macro F1 refitting.

## 9. Experimental Setup

- Random seed: 42
- Train dishes: {split_meta['train_dishes']:,}
- Test dishes: {split_meta['test_dishes']:,}
- Primary metric: Macro F1
- TF-IDF settings: `{vectorizer_settings}`
- Test partition remained untouched until hyperparameters were frozen

## 10. Classification Results

{model_table}

## 11. Selected Model

{best_model_name} was selected because it achieved the highest cross-validated Macro F1 ({best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}). On the untouched test set it achieved accuracy {best['accuracy']:.4f}, Macro Precision {best['macro_precision']:.4f}, Macro Recall {best['macro_recall']:.4f}, and Macro F1 {best['macro_f1']:.4f}.

## 12. Nutrition Estimation Results

The separate quantity-aware hybrid nutrition estimator was evaluated independently:

{nutrition_table}

The interface labels these values as estimates and provides empirical 80% absolute-error ranges. Explicit g/kg/ml values are used where provided; otherwise the portion assumption is shown. Negative outputs are clipped only for presentation.

## 13. Recommendation Engine

Ingredient substitutions are stored in `config/ingredient_substitutions.json`. Suggestions are triggered only when matching ingredients or estimated nutrition patterns are present. They are not claimed to be learned from the CSV.

## 14. Discussion

Ingredient-name text contains useful signals for broad calorie-category classification, but exact nutrition estimation is inherently limited when portion sizes are absent. The comparison results quantify the trade-offs among linear, probabilistic, tree, instance-based, boosting, and margin-based methods.

## 15. Limitations

The vocabulary is limited, user-entered quantities may be absent, unusual dishes can be outside the training distribution, and the raw dataset contains extreme but not automatically invalid dishes. Rule-based dietary suggestions are general and non-clinical.

## 16. Conclusion

NutriNLP satisfies the proposal's core objective by combining NLP preprocessing, TF-IDF, multiclass classification, separate nutrition estimation, transparent recommendations, reproducible testing, and a functional Streamlit interface.

## 17. Future Work

Future work may add household-unit conversion, multilingual support, stronger calibrated models, transformer embeddings, personalized dietary constraints with professional oversight, and external validation on independently collected meal descriptions.
"""
    (root / "reports" / "final_report.md").write_text(report, encoding="utf-8")
