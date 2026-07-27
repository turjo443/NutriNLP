"""Saved-artifact loading and end-to-end inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import joblib
import numpy as np

from .recommendation_engine import RecommendationEngine
from .text_preprocessing import normalize_text
from .utils import load_json


class NutriNLPInference:
    """Load trained artifacts once and analyze user-entered meal descriptions."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        classifier_path = self.root / "models" / "best_calorie_classifier.joblib"
        nutrition_path = self.root / "models" / "nutrition_estimator.joblib"
        metadata_path = self.root / "models" / "model_metadata.json"
        for path in [classifier_path, nutrition_path, metadata_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required artifact missing: {path}. Train first with: python run_pipeline.py --input data/raw/dish_ingredients.csv --mode full"
                )
        self.classifier = joblib.load(classifier_path)
        self.nutrition = joblib.load(nutrition_path)
        self.metadata = load_json(metadata_path)
        self.engine = RecommendationEngine(self.root / "config" / "ingredient_substitutions.json")
        aliases_path = self.root / "config" / "inference_aliases.json"
        self.inference_aliases = load_json(aliases_path) if aliases_path.exists() else {}
        self.class_names = {int(k): v for k, v in self.metadata["class_names"].items()}
        vocabulary_data = load_json(self.root / "models" / "ingredient_vocabulary.json")
        self.ingredient_vocabulary = sorted(vocabulary_data.get("ingredients", []), key=lambda x: len(normalize_text(x)), reverse=True)


    def _apply_inference_aliases(self, text: str) -> tuple[str, list[dict[str, str]]]:
        """Map unsupported food terms to transparent in-dataset proxies.

        Aliases are applied only at inference time and are reported to the user.
        Longer phrases are processed first so ``goat meat`` never becomes
        confused with the dataset ingredient ``goat cheese``.
        """
        mapped = str(text)
        applied: list[dict[str, str]] = []
        for source in sorted(self.inference_aliases, key=lambda item: (len(item.split()), len(item)), reverse=True):
            rule = self.inference_aliases[source]
            pattern = re.compile(rf"(?<![a-z]){re.escape(source)}(?![a-z])", flags=re.IGNORECASE)
            if not pattern.search(mapped):
                continue
            proxy = str(rule["proxy"])
            mapped = pattern.sub(proxy, mapped)
            applied.append(
                {
                    "source": str(rule.get("display_name", source)),
                    "proxy": proxy,
                    "reason": str(rule.get("reason", "An in-dataset proxy was used.")),
                }
            )
        return mapped, applied

    def _vocabulary_coverage(self, text: str) -> tuple[float, list[str]]:
        normalized = normalize_text(text)
        tokens = normalized.split()
        if not tokens:
            return 0.0, []
        tfidf = self.classifier.named_steps.get("tfidf")
        vocab = getattr(tfidf, "vocabulary_", {})
        known = [token for token in tokens if token in vocab]
        return len(known) / len(tokens), known


    def _detect_dataset_ingredients(self, text: str) -> list[str]:
        """Match known dataset ingredient phrases in normalized user text."""
        normalized = f" {normalize_text(text)} "
        matches: list[str] = []
        matched_token_sets: list[set[str]] = []
        for ingredient in self.ingredient_vocabulary:
            key = normalize_text(ingredient)
            if not key:
                continue
            key_tokens = set(key.split())
            if any(key_tokens.issubset(existing) for existing in matched_token_sets):
                continue
            if f" {key} " in normalized:
                matches.append(ingredient)
                matched_token_sets.append(key_tokens)
        return matches

    def analyze(self, text: str) -> dict[str, Any]:
        if text is None or len(str(text).strip()) < 3:
            raise ValueError("Please enter a meaningful food description with at least one ingredient.")
        normalized = normalize_text(text)
        if len(normalized.split()) == 0:
            raise ValueError("No usable ingredient words were found in the input.")

        model_text, applied_aliases = self._apply_inference_aliases(str(text))

        class_id = int(self.classifier.predict([model_text])[0])
        class_name = self.class_names[class_id]
        confidence = None
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba([model_text])[0]
            confidence = float(np.max(probs))

        if hasattr(self.nutrition, "predict_with_details"):
            nutrition_details = self.nutrition.predict_with_details([model_text])[0]
            raw_est = nutrition_details.prediction
            nutrition_method = nutrition_details.method
            portion_basis = nutrition_details.portion_basis
            explicit_quantities = nutrition_details.explicit_quantities_grams
            estimator_ingredients = nutrition_details.detected_ingredients
        else:
            raw_est = self.nutrition.predict([model_text])[0]
            nutrition_method = "legacy word-level estimator"
            portion_basis = "Quantity was not supplied; interpret values as broad estimates."
            explicit_quantities = {}
            estimator_ingredients = []

        target_order = self.metadata["nutrition_targets"]
        estimated = {target: float(max(0.0, raw_est[i])) for i, target in enumerate(target_order)}
        uncertainty = self.metadata["nutrition_uncertainty_mae"]
        quantiles = self.metadata.get("nutrition_uncertainty_quantiles", {})
        ranges = {}
        for target in target_order:
            interval = float(quantiles.get(target, {}).get("q80", uncertainty[target]))
            ranges[target] = {
                "low": max(0.0, estimated[target] - interval),
                "high": estimated[target] + interval,
                "mae": float(uncertainty[target]),
                "interval_error_q80": interval,
            }
        coverage, known_tokens = self._vocabulary_coverage(model_text)
        warnings = []
        for alias in applied_aliases:
            warnings.append(
                f"{alias['source'].title()} is not represented as a meat ingredient in the uploaded dataset, "
                f"so the system used {alias['proxy']} as a transparent proxy. {alias['reason']}"
            )
        if coverage < 0.5:
            warnings.append("Much of the input is outside the trained ingredient vocabulary; interpret results cautiously.")
        if confidence is not None and confidence < 0.55:
            warnings.append("The classifier confidence is low; the meal may be ambiguous or unfamiliar.")
        if self.metadata.get("nutrition_estimator_reliability") == "low":
            warnings.append("Numerical nutrition estimation has limited test-set reliability; use the displayed ranges rather than treating point values as exact measurements.")
        if not explicit_quantities:
            if nutrition_method == "single-ingredient density estimator":
                warnings.append(f"No quantity was supplied, so the estimator used this transparent assumption: {portion_basis}")
            else:
                warnings.append("No ingredient quantity was supplied, so the estimator used training-derived typical portions. Add values such as '100 g mango' for a more specific estimate.")

        thresholds = self.metadata["class_thresholds"]
        estimate_calories = estimated["total_calories"]
        if estimate_calories <= float(thresholds["low_threshold_kcal"]):
            estimated_category = "Low Calorie"
        elif estimate_calories <= float(thresholds["high_threshold_kcal"]):
            estimated_category = "Medium Calorie"
        else:
            estimated_category = "High Calorie"
        if estimated_category != class_name:
            warnings.append(
                "The independent category classifier and numerical nutrition estimator disagree. This is expected in some cases; use the category and wide uncertainty range as separate evidence."
            )

        recommendations = self.engine.recommend(text, estimated, class_name)
        dataset_ingredients = estimator_ingredients or self._detect_dataset_ingredients(text)
        return {
            "input_text": str(text),
            "normalized_text": normalized,
            "model_input_text": model_text,
            "inference_aliases": applied_aliases,
            "calorie_class_id": class_id,
            "calorie_class": class_name,
            "confidence": confidence,
            "estimated_nutrition": estimated,
            "uncertainty_ranges": ranges,
            "nutrition_estimation_method": nutrition_method,
            "portion_basis": portion_basis,
            "explicit_quantities_grams": explicit_quantities,
            "vocabulary_coverage": coverage,
            "known_tokens": known_tokens,
            "detected_ingredients": dataset_ingredients,
            "detected_rule_ingredients": recommendations["detected_rule_ingredients"],
            "nutrition_estimate_category": estimated_category,
            "healthier_alternatives": recommendations["substitutions"],
            "dietary_suggestions": recommendations["dietary_suggestions"],
            "warnings": warnings,
            "disclaimer": "Educational estimates only; not professional medical or dietetic advice.",
        }
