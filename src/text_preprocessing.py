"""Leakage-safe, reusable NLP text preprocessing."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from nltk.tokenize import RegexpTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")
_IRREGULAR = {
    "tomatoes": "tomato",
    "potatoes": "potato",
    "leaves": "leaf",
    "loaves": "loaf",
    "knives": "knife",
    "berries": "berry",
    "cherries": "cherry",
    "fries": "fry",
    "eggs": "egg",
    "onions": "onion",
    "peppers": "pepper",
    "mushrooms": "mushroom",
    "beans": "bean",
    "lentils": "lentil",
    "oats": "oat",
}


def simple_lemma(token: str) -> str:
    """Apply deterministic lightweight English lemmatization rules.

    The food vocabulary is noun-heavy. These conservative rules normalize common
    plurals without requiring an external corpus at runtime.
    """
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ves"):
        return token[:-3] + "f"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_text(text: object) -> str:
    """Normalize a value into clean, tokenized, lemmatized text."""
    if text is None:
        return ""
    value = str(text).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return ""
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[_/\\-]+", " ", value)
    value = re.sub(r"[^a-z\s']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    tokens = _TOKENIZER.tokenize(value)
    cleaned = [simple_lemma(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(cleaned)


class FoodTextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that applies NutriNLP text normalization."""

    def fit(self, X: Iterable[object], y: object = None) -> "FoodTextPreprocessor":
        return self

    def transform(self, X: Iterable[object]) -> list[str]:
        return [normalize_text(item) for item in X]
