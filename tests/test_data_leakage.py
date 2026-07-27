from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.text_preprocessing import FoodTextPreprocessor


def test_processed_split_has_no_dish_overlap() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "dish_level_dataset.csv")
    train = set(df.loc[df["split"] == "train", "dish_id"])
    test = set(df.loc[df["split"] == "test", "dish_id"])
    assert train.isdisjoint(test)
    assert len(train) + len(test) == df["dish_id"].nunique()


def test_saved_thresholds_equal_training_only_quantiles() -> None:
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "dish_level_dataset.csv")
    thresholds = pd.read_json(root / "models" / "class_thresholds.json", typ="series")
    train_calories = df.loc[df["split"] == "train", "total_calories"]
    assert abs(float(thresholds["low_threshold_kcal"]) - float(train_calories.quantile(0.3333))) < 1e-9
    assert abs(float(thresholds["high_threshold_kcal"]) - float(train_calories.quantile(0.6667))) < 1e-9


def test_tfidf_does_not_learn_test_only_token() -> None:
    pipeline = Pipeline(
        [
            ("preprocess", FoodTextPreprocessor()),
            ("tfidf", TfidfVectorizer(lowercase=False, ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=200, solver="saga")),
        ]
    )
    x_train = ["rice chicken", "broccoli tomato", "rice tomato", "chicken broccoli", "rice broccoli", "tomato chicken"]
    y_train = [0, 1, 0, 1, 0, 1]
    pipeline.fit(x_train, y_train)
    vocab = pipeline.named_steps["tfidf"].vocabulary_
    assert "dragonfruit" not in vocab
    pipeline.predict(["dragonfruit rice"])
    assert "dragonfruit" not in pipeline.named_steps["tfidf"].vocabulary_
