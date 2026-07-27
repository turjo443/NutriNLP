"""Streamlit user interface for NutriNLP."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import NutriNLPInference


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="NutriNLP",
    page_icon="🥗",
    layout="wide",
)


# =========================
# Custom UI design and CSS
# =========================
st.markdown(
    """
<style>
:root {
    --green: #1f7a4d;
    --light: #eef8f1;
    --ink: #173528;
}

.stApp {
    background: #fbfdfb;
    color: var(--ink);
}

.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #e7f7ec, #ffffff);
    border: 1px solid #cae8d3;
    margin-bottom: 1rem;
}

.hero h1 {
    color: var(--green);
    margin: 0;
}

.result-box {
    padding: 1rem;
    border: 1px solid #d4eadb;
    border-radius: 14px;
    background: white;
}

.small-note {
    color: #52675c;
    font-size: 0.9rem;
}

[data-testid="stMetricValue"] {
    color: #173528 !important;
}

[data-testid="stMetricLabel"] {
    color: #385a49 !important;
}

/* Food-description text box visibility fix */
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    color: #173528 !important;
    -webkit-text-fill-color: #173528 !important;
    caret-color: #173528 !important;
    border-color: #9fcbb0 !important;
    border-radius: 10px !important;
}

[data-testid="stTextArea"] textarea:focus {
    background-color: #ffffff !important;
    color: #173528 !important;
    -webkit-text-fill-color: #173528 !important;
    caret-color: #173528 !important;
}

[data-testid="stTextArea"] textarea::placeholder {
    color: #718078 !important;
    -webkit-text-fill-color: #718078 !important;
    opacity: 1 !important;
}

[data-testid="stTextArea"] label,
[data-testid="stTextArea"] p {
    color: #173528 !important;
    font-weight: 600 !important;
}

/* Selectbox readability */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #173528 !important;
    border-color: #9fcbb0 !important;
}

[data-testid="stSelectbox"] input {
    color: #173528 !important;
    -webkit-text-fill-color: #173528 !important;
}

/* Analyze button */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Alert readability */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #173528 !important;
}

/* Tabs readability */
button[data-baseweb="tab"] {
    color: #385a49 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1f7a4d !important;
    font-weight: 700 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Header section
# =========================
st.markdown(
    """
<div class="hero">
    <h1>🥗 NutriNLP</h1>
    <p>
        Natural-language calorie-category prediction,
        separate nutrition estimation, and transparent
        healthier alternatives.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Required trained artifacts
# =========================
artifact_paths = [
    ROOT / "models" / "best_calorie_classifier.joblib",
    ROOT / "models" / "nutrition_estimator.joblib",
    ROOT / "models" / "model_metadata.json",
]

if not all(path.exists() for path in artifact_paths):
    st.error("Trained artifacts are missing.")
    st.code(
        'python run_pipeline.py --input "data/raw/dish_ingredients.csv" --mode full'
    )
    st.stop()


# =========================
# Cached loading functions
# =========================
@st.cache_resource
def load_inference() -> NutriNLPInference:
    """Load the saved classification and nutrition models."""
    return NutriNLPInference(ROOT)


@st.cache_data
def load_comparison() -> pd.DataFrame:
    """Load the actual eight-model comparison results."""
    return pd.read_csv(ROOT / "reports" / "model_comparison.csv")


engine = load_inference()
metadata = engine.metadata
comparison = load_comparison()


# =========================
# Main application tabs
# =========================
prediction_tab, performance_tab, data_tab, about_tab = st.tabs(
    [
        "Meal Analysis",
        "Model Performance",
        "Dataset Insights",
        "About & Methodology",
    ]
)


# =================================================
# Tab 1: Meal prediction and nutrition estimation
# =================================================
with prediction_tab:
    st.subheader("Analyze a meal description")

    examples = [
        "Chicken breast with white rice and broccoli",
        "Fried potatoes with cheese and mayonnaise",
        "Lentil soup with spinach, tomato, and olive oil",
        "100 g mango with 150 ml milk and 100 g yogurt",
        "100 g goat cheese",
        "100 g salmon with 100 g broccoli",
        "120 g mutton with 150 g cooked rice",
    ]

    selected = st.selectbox(
        "Example descriptions",
        ["Choose an example..."] + examples,
    )

    default_text = "" if selected == "Choose an example..." else selected

    text = st.text_area(
        "Food description",
        value=default_text,
        height=120,
        placeholder="Example: 100 g salmon with 100 g broccoli",
    )

    if st.button(
        "Analyze Meal",
        type="primary",
        use_container_width=True,
    ):
        try:
            result = engine.analyze(text)

            thresholds = metadata["class_thresholds"]
            estimated = result["estimated_nutrition"]
            text_model_category = result["calorie_class"]
            has_explicit_quantities = bool(result.get("explicit_quantities_grams"))

            # When the user supplies grams/kg/ml, the final displayed category
            # must agree with the quantity-aware numerical calorie estimate.
            if has_explicit_quantities:
                display_category = result.get(
                    "nutrition_estimate_category",
                    text_model_category,
                )
                category_source = "Quantity-aware nutrition estimate"
            else:
                display_category = text_model_category
                category_source = "Text-only machine-learning classifier"

            st.markdown(
                "### Predicted category: "
                f"**{display_category}**"
            )

            st.caption(f"Category source: {category_source}")

            st.caption(
                "Training-only thresholds: "
                f"Low ≤ {thresholds['low_threshold_kcal']:.2f} kcal; "
                f"Medium ≤ {thresholds['high_threshold_kcal']:.2f} kcal; "
                "High above that."
            )

            if has_explicit_quantities and display_category != text_model_category:
                st.info(
                    "The text-only classifier predicted "
                    f"**{text_model_category}**, but the supplied quantities produced "
                    f"an estimated **{estimated['total_calories']:.1f} kcal**. "
                    "Therefore, the final displayed category was adjusted to "
                    f"**{display_category}** so that the category and numerical "
                    "estimate remain consistent."
                )

            if result["confidence"] is not None:
                st.progress(
                    result["confidence"],
                    text=(
                        "Text-classifier probability: "
                        f"{result['confidence']:.1%}"
                    ),
                )

            ranges = result["uncertainty_ranges"]
            col1, col2, col3, col4 = st.columns(4)

            nutrition_cards = [
                (col1, "Estimated calories", "total_calories", "kcal"),
                (col2, "Estimated protein", "total_protein", "g"),
                (col3, "Estimated carbohydrates", "total_carb", "g"),
                (col4, "Estimated fat", "total_fat", "g"),
            ]

            for column, label, key, unit in nutrition_cards:
                with column:
                    st.metric(label, f"{estimated[key]:.1f} {unit}")
                    st.caption(
                        "Approx. range: "
                        f"{ranges[key]['low']:.1f}–"
                        f"{ranges[key]['high']:.1f} {unit}"
                    )

            st.caption(
                "Known-vocabulary coverage: "
                f"{result['vocabulary_coverage']:.0%}"
            )

            if result.get("inference_aliases"):
                alias_text = "; ".join(
                    f"{item['source']} → {item['proxy']} (dataset proxy)"
                    for item in result["inference_aliases"]
                )
                st.warning("**Proxy mapping used:** " + alias_text)

            st.info(
                "**Nutrition estimate method:** "
                f"{result['nutrition_estimation_method']}\n\n"
                "**Portion basis:** "
                f"{result['portion_basis']}"
            )

            if result["explicit_quantities_grams"]:
                quantity_text = ", ".join(
                    f"{name}: {grams:.1f} g"
                    for name, grams in result[
                        "explicit_quantities_grams"
                    ].items()
                )
                st.caption("Parsed quantities — " + quantity_text)

            for warning in result["warnings"]:
                # The old generic disagreement warning is redundant after the
                # quantity-aware final category has already been corrected.
                if (
                    has_explicit_quantities
                    and "classifier and numerical nutrition estimator disagree"
                    in warning.lower()
                ):
                    continue
                st.warning(warning)

            left_column, right_column = st.columns(2)

            with left_column:
                st.markdown("#### Recognized dataset ingredients")

                if result["detected_ingredients"]:
                    st.write(", ".join(result["detected_ingredients"]))
                else:
                    st.write(
                        "No exact dataset ingredient phrase was recognized."
                    )

                if result["detected_rule_ingredients"]:
                    st.caption(
                        "Substitution triggers: "
                        + ", ".join(result["detected_rule_ingredients"])
                    )

                st.markdown("#### Healthier alternatives")

                if result["healthier_alternatives"]:
                    for item in result["healthier_alternatives"]:
                        st.write(f"• {item}")
                else:
                    st.write(
                        "No ingredient-specific substitution is available "
                        "for this description."
                    )

            with right_column:
                st.markdown("#### General dietary suggestions")

                for item in result["dietary_suggestions"]:
                    st.write(f"• {item}")

            st.markdown("### Calorie-reduction plan")
            reduction_plan = result.get("calorie_reduction_plan", [])

            if reduction_plan:
                plan_rows = []
                for item in reduction_plan:
                    saving = item.get("estimated_saving_kcal")
                    plan_rows.append(
                        {
                            "Detected item": item.get("trigger", ""),
                            "Recommended change": item.get("suggestion", ""),
                            "Estimated kcal saved": (
                                f"{saving:.1f}" if saving is not None else "Add quantity"
                            ),
                            "Calculation basis": item.get("basis", ""),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(plan_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                total_saving = result.get("estimated_total_calorie_saving")
                revised_calories = result.get("estimated_revised_calories")
                revised_category = result.get("estimated_revised_category")

                if total_saving is not None and revised_calories is not None:
                    current_col, saving_col, revised_col = st.columns(3)
                    current_col.metric(
                        "Current estimated calories",
                        f"{estimated['total_calories']:.1f} kcal",
                    )
                    saving_col.metric(
                        "Potential calorie reduction",
                        f"{total_saving:.1f} kcal",
                    )
                    revised_col.metric(
                        "Estimated calories after changes",
                        f"{revised_calories:.1f} kcal",
                    )
                    if revised_category:
                        st.success(
                            "Estimated category after applying the listed changes: "
                            f"**{revised_category}**"
                        )
                else:
                    st.info(
                        "Specific lower-calorie substitutions were found. Add amounts "
                        "such as `30 g mayonnaise` or `150 g white rice` to calculate "
                        "the approximate calories that could be saved."
                    )

                st.caption(result.get("calorie_reduction_note", ""))
            else:
                st.write(
                    "No matching calorie-reduction rule was found for the entered ingredients."
                )

            st.info(result["disclaimer"])

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# =================================
# Tab 2: Model performance results
# =================================
with performance_tab:
    st.subheader("Eight-model comparison")

    display_columns = [
        "rank",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "training_time_seconds",
        "prediction_time_seconds",
    ]

    st.dataframe(
        comparison[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    best_model_name = metadata["selected_model_name"]
    best_model_metrics = metadata["selected_model_metrics"]

    st.success(
        f"Selected model: {best_model_name}. "
        "CV Macro F1 = "
        f"{best_model_metrics['cv_macro_f1_mean']:.4f} ± "
        f"{best_model_metrics['cv_macro_f1_std']:.4f}; "
        "untouched-test Macro F1 = "
        f"{best_model_metrics['macro_f1']:.4f}."
    )

    chart_data = comparison.set_index("model")[[
        "accuracy",
        "macro_f1",
        "cv_macro_f1_mean",
    ]]

    st.bar_chart(chart_data)

    confusion_matrix_path = (
        ROOT
        / "reports"
        / "confusion_matrices"
        / (
            best_model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".png"
        )
    )

    if confusion_matrix_path.exists():
        st.image(
            str(confusion_matrix_path),
            caption=f"Confusion matrix — {best_model_name}",
        )


# ============================
# Tab 3: Dataset information
# ============================
with data_tab:
    dataset_summary = metadata["dataset"]

    st.subheader("Dataset summary")

    column1, column2, column3, column4 = st.columns(4)

    column1.metric(
        "Raw ingredient rows",
        f"{dataset_summary['raw_rows']:,}",
    )

    column2.metric(
        "Valid rows",
        f"{dataset_summary['valid_rows']:,}",
    )

    column3.metric(
        "Unique dishes",
        f"{dataset_summary['unique_dishes']:,}",
    )

    column4.metric(
        "Excluded invalid rows",
        f"{dataset_summary['excluded_rows']:,}",
    )

    st.write(
        "Ingredient-level rows are grouped by `dish_id`; ingredient names "
        "become one `dish_text` field, while grams and nutrients are summed."
    )

    figure_names = [
        ("total_calorie_distribution.png", "Total calorie distribution"),
        ("calorie_class_distribution.png", "Calorie class distribution"),
        ("top_ingredients.png", "Most frequent ingredients"),
        (
            "nutrition_correlation_heatmap.png",
            "Nutrition correlation heatmap",
        ),
    ]

    figure_columns = st.columns(2)

    for index, (filename, caption) in enumerate(figure_names):
        figure_path = ROOT / "reports" / "figures" / filename

        if figure_path.exists():
            figure_columns[index % 2].image(
                str(figure_path),
                caption=caption,
            )


# ===================================
# Tab 4: About and methodology
# ===================================
with about_tab:
    st.subheader("Methodology")

    st.markdown(
        """
- **Objective:** classify ingredient-text descriptions into Low, Medium, or High Calorie categories.
- **Preprocessing:** lowercase and Unicode normalization, punctuation cleanup, regex tokenization, stop-word removal, deterministic lemmatization, and TF-IDF unigrams/bigrams.
- **Leakage control:** the split is created before calorie thresholds; TF-IDF and all model-specific transformations are fitted only on training folds.
- **Eight classifiers:** Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, KNN, Gradient Boosting, and XGBoost.
- **Quantity-aware category:** when explicit `g`, `kg`, or `ml` quantities are supplied, the final displayed category is derived from the numerical calorie estimate so the category and calories do not contradict each other.
- **Separate estimation:** numerical calories and macronutrients come from an independently evaluated hybrid ingredient-phrase Random Forest, training-only ingredient profiles, and a Ridge fallback—not from the category classifier.
- **Portions:** explicit `g`, `kg`, or `ml` quantities are used when provided; otherwise dataset-derived typical portions are assumed.
- **Recommendations:** substitutions are explicit rules in an editable JSON file and were not learned from the CSV.
"""
    )

    st.warning(
        "NutriNLP provides educational estimates only. "
        "It is not a substitute for a registered dietitian, physician, "
        "laboratory analysis, or medical advice."
    )
