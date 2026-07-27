# Proposal-to-Implementation Mapping

| Proposal requirement | Implemented component |
|---|---|
| Ingredient-text NLP pipeline | `src/text_preprocessing.py` and TF-IDF inside every model pipeline |
| Low/Medium/High calorie prediction | `src/data_preparation.py`, `src/train_classifiers.py`, saved thresholds and classifier |
| Eight required classifiers | Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, SVM, KNN, Gradient Boosting, XGBoost |
| Accuracy, precision, recall, F1 and confusion matrices | `reports/model_comparison.csv`, per-model reports and images |
| Nutrition summary | Separate `src/train_nutrition_estimator.py` multi-output Ridge pipeline |
| Healthier substitutions | Editable `config/ingredient_substitutions.json` and `src/recommendation_engine.py` |
| User interface | `app.py` Streamlit application with prediction, performance, dataset and methodology tabs |
| EDA and dataset understanding | Tables and figures in `reports/` |
| Testing and reproducibility | `run_pipeline.py`, `pytest.ini`, and `tests/` |
