# NutriNLP — Final Academic Project Report

## Abstract

NutriNLP investigates whether ingredient-name text can support multiclass calorie-category prediction and approximate nutrition estimation. The project converts 27,225 ingredient-level observations into 4,768 dish-level samples, implements leakage-safe TF-IDF pipelines, compares eight classifiers under five-fold stratified cross-validation, evaluates final models on an untouched test partition, and integrates the selected classifier with a separate quantity-aware hybrid nutrition estimator and a transparent rule-based recommendation engine.

## 1. Introduction

Nutrition applications often rely on exact database lookup, barcode scanning, or image recognition. NutriNLP instead accepts a natural-language ingredient description and uses NLP features to infer a broad calorie category. The system deliberately separates classification from numerical nutrient estimation.

## 2. Problem Statement

Users may enter descriptions such as “chicken breast with rice and broccoli” without quantities or measured nutrient values. A useful system must operate from ingredient text only, avoid data leakage, communicate uncertainty, and refrain from treating approximate predictions as clinical facts.

## 3. Objectives

The project cleans and aggregates the supplied dataset, builds a reusable NLP pipeline, compares eight required classification algorithms, evaluates a separate nutrition estimator, generates evidence-based reports, and deploys the artifacts through Streamlit.

## 4. Conceptual Background

TF-IDF represents ingredient terms according to their frequency within a dish and rarity across dishes. Macro F1 is used as the primary classifier metric because it weights Low, Medium, and High Calorie classes equally. Accuracy, precision, recall, weighted scores, confusion matrices, training time, and prediction time are also reported.

## 5. Dataset Methodology

The uploaded CSV matched the expected 8-column structure. Validation found 27,225 raw rows, 0 exact duplicates, 0 excluded invalid rows, and one missing ingredient name. That row was retained because its numeric nutrition values were valid and its dish had other named ingredients; only the absent text token was omitted. Aggregation produced 4,768 unique dish records.

## 6. Data Preprocessing

Text is normalized, tokenized with NLTK's corpus-free regex tokenizer, stripped of stop words, and lemmatized with deterministic food-oriented rules. The TF-IDF vectorizer uses unigrams and bigrams and is fitted only inside training folds. Numeric nutrition columns and identifiers are never used as classification input features.

## 7. Label Methodology

The proposal did not define fixed calorie thresholds. Therefore, the data was split first and the 33.33rd/66.67th percentiles were computed only from training calories. The thresholds were 87.4924 kcal and 260.8691 kcal. The same thresholds were then applied to the untouched test partition.

## 8. Model Methodology

The eight required models were Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Linear Support Vector Machine, K-Nearest Neighbors, Gradient Boosting, and XGBoost. Gradient Boosting uses TruncatedSVD inside its pipeline to produce a leakage-safe dense representation. Hyperparameters were tuned using five-fold StratifiedKFold CV with Macro F1 refitting.

## 9. Experimental Setup

- Random seed: 42
- Train dishes: 3,814
- Test dishes: 954
- Primary metric: Macro F1
- TF-IDF settings: `{'ngram_range': [1, 2], 'min_df': 1, 'max_df': 1.0, 'max_features': 5000, 'sublinear_tf': True}`
- Test partition remained untouched until hyperparameters were frozen

## 10. Classification Results

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

## 11. Selected Model

XGBoost was selected because it achieved the highest cross-validated Macro F1 (0.7579 ± 0.0091). On the untouched test set it achieved accuracy 0.7893, Macro Precision 0.7859, Macro Recall 0.7908, and Macro F1 0.7867.

## 12. Nutrition Estimation Results

The separate numerical component was upgraded to a quantity-aware hybrid estimator containing an ingredient-phrase Random Forest, additive ingredient profiles learned only from the training partition, and a word-level Ridge fallback. It was evaluated independently on the untouched test partition:

| display_name      |    mae |    rmse |     r2 |
|:------------------|-------:|--------:|-------:|
| Calories (kcal)   | 81.440 | 276.835 | 0.4233 |
| Protein (g)       |  5.384 |   9.638 | 0.7557 |
| Carbohydrates (g) |  6.977 |  17.230 | 0.3779 |
| Fat (g)           |  4.903 |  24.735 | 0.3685 |

The interface labels these values as estimates and provides empirical 80% absolute-error ranges. Explicit `g`, `kg`, and `ml` amounts are parsed when present. If quantities are absent, the interface states the portion assumption. The estimator is rated moderate rather than high reliability because exact portions cannot be recovered from ingredient names alone.

## 13. Recommendation Engine

Ingredient substitutions are stored in `config/ingredient_substitutions.json`. Suggestions are triggered only when matching ingredients or estimated nutrition patterns are present. They are not claimed to be learned from the CSV.

## 14. Discussion

Ingredient-name text contains useful signals for broad calorie-category classification, but exact nutrition estimation is inherently limited when portion sizes are absent. The comparison results quantify the trade-offs among linear, probabilistic, tree, instance-based, boosting, and margin-based methods.

## 15. Limitations

The vocabulary is limited, user-entered quantities may be absent, unusual dishes can be outside the training distribution, and the raw dataset contains extreme but not automatically invalid dishes. Rule-based dietary suggestions are general and non-clinical.

## 16. Conclusion

NutriNLP satisfies the proposal's core objective by combining NLP preprocessing, TF-IDF, multiclass classification, separate nutrition estimation, transparent recommendations, reproducible testing, and a functional Streamlit interface.

## 17. Future Work

Future work may add quantity extraction, multilingual support, stronger calibrated models, transformer embeddings, personalized dietary constraints with professional oversight, and external validation on independently collected meal descriptions.
