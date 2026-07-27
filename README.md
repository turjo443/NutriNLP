# NutriNLP

**NutriNLP** is a complete, reproducible NLP and machine-learning project for classifying an ingredient-based meal description as **Low**, **Medium**, or **High Calorie**, separately estimating calories and macronutrients, and producing transparent rule-based healthier alternatives.

## Research problem

People often describe meals in ordinary language but do not know their approximate calorie category or nutritional profile. NutriNLP transforms ingredient-name text into TF-IDF features, compares eight supervised classifiers, and keeps numerical nutrition estimation separate from classification so that category predictions are not misrepresented as exact nutrient measurements.

## Dataset

The source CSV contains **27,225 ingredient-level rows**, **8 required columns**, **4,768 unique dishes**, and **209 named ingredient values**. The uploaded structure matched the expected schema. No exact duplicate rows were found. **0 rows were excluded** by validation. One missing ingredient name was retained for dish nutrition aggregation but contributes no text token.

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

The 27,225 rows become **4,768 dish-level samples** by grouping all rows with the same `dish_id`, concatenating ingredient names into `dish_text`, and summing grams, calories, fat, carbohydrate, and protein.

## Leakage-safe calorie labels

A fixed 80/20 split was created first with `random_state=42`. Class thresholds were calculated **only from the training partition**:

- Low Calorie: `total_calories <= 87.4924 kcal`
- Medium Calorie: `87.4924 < total_calories <= 260.8691 kcal`
- High Calorie: `total_calories > 260.8691 kcal`

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

Selected TF-IDF settings: `{'ngram_range': [1, 2], 'min_df': 1, 'max_df': 1.0, 'max_features': 5000, 'sublinear_tf': True}`.

## Models compared

Exactly eight classifiers were tuned with five-fold `StratifiedKFold` cross-validation on the training partition: Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, and XGBoost. Macro F1 was the primary selection metric.

## Actual classification results

|   rank | model                   |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   weighted_f1 |   cv_macro_f1_mean |   cv_macro_f1_std |
|-------:|:------------------------|-----------:|------------------:|---------------:|-----------:|--------------:|-------------------:|------------------:|
|      1 | XGBoost                 |     0.7893 |            0.7859 |         0.7908 |     0.7867 |        0.7857 |             0.7579 |            0.0091 |
|      2 | Random Forest           |     0.7935 |            0.7912 |         0.7947 |     0.7923 |        0.7913 |             0.7545 |            0.0135 |
|      3 | K-Nearest Neighbors     |     0.7862 |            0.7832 |         0.7875 |     0.784  |        0.783  |             0.732  |            0.0099 |
|      4 | Decision Tree           |     0.7547 |            0.7544 |         0.7556 |     0.7546 |        0.7537 |             0.7274 |            0.0085 |
|      5 | Support Vector Machine  |     0.7474 |            0.743  |         0.7489 |     0.7447 |        0.7436 |             0.7083 |            0.022  |
|      6 | Gradient Boosting       |     0.74   |            0.7431 |         0.7415 |     0.739  |        0.7378 |             0.7069 |            0.0117 |
|      7 | Logistic Regression     |     0.7285 |            0.7239 |         0.7301 |     0.7258 |        0.7246 |             0.6967 |            0.0166 |
|      8 | Multinomial Naive Bayes |     0.7096 |            0.7063 |         0.7109 |     0.708  |        0.7069 |             0.6567 |            0.0141 |

**Selected model:** XGBoost. It ranked first by training-partition cross-validated Macro F1 (0.7579 ± 0.0091) and achieved test Macro F1 0.7867 with accuracy 0.7893 on the untouched test partition.

## Separate nutrition estimator

The numerical component is now a **quantity-aware hybrid estimator**. It combines an ingredient-phrase Random Forest, additive ingredient nutrition profiles learned from training rows only, and a word-level Ridge fallback. Explicit `g`, `kg`, and `ml` quantities are parsed when supplied. Without quantities, the app clearly states the assumed dataset-derived portion basis. A one-ingredient input such as `mango` uses a transparent energy-density-based default serving instead of a garnish-sized training amount.

This change fixes unrealistic outputs such as a sub-1-kcal estimate for `Mango with yogurt and milk`. The improved estimator is marked **moderate reliability** and uses empirical 80% absolute-error ranges.

| display_name      |    mae |    rmse |     r2 |
|:------------------|-------:|--------:|-------:|
| Calories (kcal)   | 81.440 | 276.835 | 0.4233 |
| Protein (g)       |  5.384 |   9.638 | 0.7557 |
| Carbohydrates (g) |  6.977 |  17.230 | 0.3779 |
| Fat (g)           |  4.903 |  24.735 | 0.3685 |

These values remain estimates rather than laboratory measurements because ingredient text alone cannot reveal exact portion sizes unless the user provides them.

## Application features

- Meal-description input and validation
- Calorie category prediction
- Confidence only when the selected classifier exposes calibrated probabilities
- Estimated calories/macronutrients with empirical 80% error ranges
- Explicit g/kg/ml quantity parsing and transparent portion assumptions
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
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Optional notebook support:

```bash
pip install jupyter
```

Jupyter is intentionally optional so Windows users do not hit unnecessary long-path installation errors when only running the Streamlit app.

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

Input: `100 g mango with 150 ml milk and 100 g yogurt`

Output includes a calorie category, quantity-aware estimated calorie/macronutrient values with uncertainty, parsed quantities, detected ingredients, and relevant rule-based alternatives.

## Limitations

- Ingredient text alone cannot recover exact portion size when users omit quantities; the app therefore shows its default-serving assumption.
- The dataset contains only 209 unique named ingredients, so unfamiliar foods may have low vocabulary coverage.
- Numerical estimates can have substantial error, especially for very large or unusual dishes.
- Rule-based substitutions are curated educational guidance, not labels learned from the dataset.
- The tool does not diagnose disease or provide personalized medical nutrition therapy.

## Ethical and health disclaimer

NutriNLP provides educational estimates only. It is not a substitute for a registered dietitian, physician, laboratory analysis, or emergency medical advice.

## Build verification

The full training pipeline completed successfully on the uploaded dataset. The updated automated suite includes realistic-estimate, single-ingredient serving, and explicit-quantity tests in addition to the original validation suite. The skipped test is the Streamlit runtime smoke test because Streamlit could not be installed in the restricted build environment; see `reports/verification_summary.md`.
