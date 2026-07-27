"""Hybrid, quantity-aware nutrition estimation for NutriNLP.

The estimator combines:
1. an ingredient-phrase Random Forest trained on dish-level ingredient presence,
2. additive ingredient nutrition profiles learned from training rows only, and
3. a word-level Ridge fallback for unfamiliar descriptions.

This design produces substantially more realistic additive estimates than an
L2-normalized TF-IDF regressor while preserving a completely separate numerical
estimation component from the calorie-category classifier.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .text_preprocessing import normalize_text, simple_lemma

TARGET_ORDER = ["total_calories", "total_protein", "total_carb", "total_fat"]
_PROFILE_COLUMNS = ["calories", "protein", "carb", "fat"]
_UNIT_TO_GRAMS = {"g": 1.0, "gram": 1.0, "kg": 1000.0, "ml": 1.0}


def ingredient_phrase_analyzer(text: object) -> list[str]:
    """Convert comma-separated dish text into normalized ingredient phrases."""
    return [
        normalized
        for part in str(text).split(",")
        if (normalized := normalize_text(part))
    ]


def _normalize_preserving_numbers(text: object) -> str:
    """Normalize words while preserving numbers and supported unit tokens."""
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[_/\\-]+", " ", value)
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", value)
    output: list[str] = []
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            output.append(token)
        else:
            output.append(simple_lemma(token))
    return " ".join(output)


@dataclass
class NutritionPredictionDetails:
    """One nutrition prediction plus transparent inference details."""

    prediction: np.ndarray
    method: str
    detected_ingredients: list[str]
    explicit_quantities_grams: dict[str, float]
    portion_basis: str
    canonical_text: str


class HybridNutritionEstimator:
    """Quantity-aware hybrid estimator with a scikit-learn-like ``predict`` API."""

    def __init__(
        self,
        phrase_vectorizer: CountVectorizer,
        ingredient_model: Any,
        fallback_model: Any,
        ingredient_profiles: dict[str, dict[str, Any]],
        token_aliases: dict[str, str],
        ingredient_model_weight: float = 0.75,
        quantity_profile_weight: float = 0.85,
    ) -> None:
        self.phrase_vectorizer = phrase_vectorizer
        self.ingredient_model = ingredient_model
        self.fallback_model = fallback_model
        self.ingredient_profiles = ingredient_profiles
        self.token_aliases = token_aliases
        self.ingredient_model_weight = float(ingredient_model_weight)
        self.quantity_profile_weight = float(quantity_profile_weight)
        self._sorted_keys = sorted(
            ingredient_profiles,
            key=lambda item: (len(item.split()), len(item)),
            reverse=True,
        )

    def _detect_ingredients(self, text: object) -> list[str]:
        """Detect known ingredient phrases, then resolve safe single-token aliases."""
        raw = str(text)
        normalized = normalize_text(raw)
        if not normalized:
            return []

        # Dish-level training/evaluation text is comma separated, so preserve its
        # exact ingredient boundaries when available.
        comma_parts = [normalize_text(part) for part in raw.split(",") if normalize_text(part)]
        comma_matches = [part for part in comma_parts if part in self.ingredient_profiles]
        # Treat commas as exact training boundaries only when nearly every part is
        # a known ingredient. Natural user sentences may also contain commas.
        if len(comma_parts) > 1 and len(comma_matches) / len(comma_parts) >= 0.8:
            return list(dict.fromkeys(comma_matches))

        padded = f" {normalized} "
        matches: list[str] = []
        occupied_tokens: set[str] = set()
        for key in self._sorted_keys:
            if f" {key} " not in padded:
                continue
            key_tokens = set(key.split())
            if key_tokens and key_tokens.issubset(occupied_tokens):
                continue
            matches.append(key)
            occupied_tokens.update(key_tokens)

        # Resolve generic user words such as "rice" to the most frequent matching
        # training ingredient (for example, "white rice").
        for token in normalized.split():
            if token in occupied_tokens:
                continue
            key = self.token_aliases.get(token)
            if key and key not in matches:
                matches.append(key)
                occupied_tokens.update(key.split())
        return matches

    def _extract_quantities(self, text: object, detected: Iterable[str]) -> dict[str, float]:
        """Parse explicit g/kg/ml quantities placed before or after ingredients."""
        normalized = _normalize_preserving_numbers(text)
        quantities: dict[str, float] = {}
        for key in detected:
            phrase_pattern = r"\s+".join(re.escape(token) for token in key.split())
            before = re.search(
                rf"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|gram|ml)\s+(?:of\s+)?{phrase_pattern}\b",
                normalized,
            )
            after = re.search(
                rf"\b{phrase_pattern}\s+(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>kg|g|gram|ml)\b",
                normalized,
            )
            match = before or after
            if not match:
                continue
            grams = float(match.group("amount")) * _UNIT_TO_GRAMS[match.group("unit")]
            if 0 < grams <= 5000:
                quantities[key] = grams
        return quantities

    def _profile_prediction(
        self,
        detected: list[str],
        explicit_quantities: dict[str, float],
    ) -> np.ndarray:
        total = np.zeros(len(TARGET_ORDER), dtype=float)
        for key in detected:
            profile = self.ingredient_profiles[key]
            if key in explicit_quantities:
                grams = explicit_quantities[key]
                contribution = np.asarray(profile["median_per_gram"], dtype=float) * grams
            else:
                contribution = np.asarray(profile["mean_contribution"], dtype=float)
            total += contribution
        return total

    def predict_with_details(self, X: Iterable[object]) -> list[NutritionPredictionDetails]:
        """Predict nutrition and return a transparent explanation for each input."""
        results: list[NutritionPredictionDetails] = []
        for text in X:
            detected = self._detect_ingredients(text)
            explicit_quantities = self._extract_quantities(text, detected)
            if detected:
                canonical_text = ", ".join(detected)
                matrix = self.phrase_vectorizer.transform([canonical_text]).toarray()
                model_prediction = np.asarray(self.ingredient_model.predict(matrix)[0], dtype=float)
                profile_prediction = self._profile_prediction(detected, explicit_quantities)
                if explicit_quantities:
                    profile_weight = self.quantity_profile_weight
                    portion_basis = (
                        "Explicit g/kg/ml quantities were used where provided; "
                        "unquantified ingredients use training-derived typical portions."
                    )
                    method = "quantity-aware hybrid ingredient estimator"
                    prediction = (1.0 - profile_weight) * model_prediction + profile_weight * profile_prediction
                elif len(detected) == 1:
                    # A one-word food such as "mango" usually means a serving, not
                    # the tiny garnish-sized amount sometimes present in a mixed dish.
                    # Use a transparent energy-density-based default portion.
                    density = np.asarray(self.ingredient_profiles[detected[0]]["median_per_gram"], dtype=float)
                    kcal_per_gram = float(density[0])
                    if kcal_per_gram <= 1.5:
                        default_grams = 100.0
                    elif kcal_per_gram <= 3.0:
                        default_grams = 75.0
                    elif kcal_per_gram <= 5.0:
                        default_grams = 40.0
                    else:
                        default_grams = 15.0
                    prediction = density * default_grams
                    portion_basis = f"Single-ingredient default serving of {default_grams:.0f} g (quantity not supplied)."
                    method = "single-ingredient density estimator"
                else:
                    profile_weight = 1.0 - self.ingredient_model_weight
                    portion_basis = "Training-derived typical ingredient portions (quantity not supplied)."
                    method = "hybrid ingredient estimator"
                    prediction = (1.0 - profile_weight) * model_prediction + profile_weight * profile_prediction
                display_names = [self.ingredient_profiles[key]["display_name"] for key in detected]
                quantity_display = {
                    self.ingredient_profiles[key]["display_name"]: value
                    for key, value in explicit_quantities.items()
                }
            else:
                canonical_text = normalize_text(text)
                prediction = np.asarray(self.fallback_model.predict([text])[0], dtype=float)
                display_names = []
                quantity_display = {}
                portion_basis = "Word-level fallback; no exact dataset ingredient phrase was recognized."
                method = "word-level Ridge fallback"

            prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(
                NutritionPredictionDetails(
                    prediction=prediction,
                    method=method,
                    detected_ingredients=display_names,
                    explicit_quantities_grams=quantity_display,
                    portion_basis=portion_basis,
                    canonical_text=canonical_text,
                )
            )
        return results

    def predict(self, X: Iterable[object]) -> np.ndarray:
        """Return predictions in target order for scikit-learn compatibility."""
        return np.vstack([item.prediction for item in self.predict_with_details(X)])
