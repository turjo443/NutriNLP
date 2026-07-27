"""Transparent rule-based ingredient and dietary recommendations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .text_preprocessing import normalize_text


class RecommendationEngine:
    """Match curated substitution rules and nutrition-pattern suggestions."""

    def __init__(self, rules_path: str | Path) -> None:
        self.rules_path = Path(rules_path)
        self.rules: dict[str, dict[str, Any]] = json.loads(self.rules_path.read_text(encoding="utf-8"))

    def detect_rule_ingredients(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        padded = f" {normalized} "
        found = []
        for ingredient in self.rules:
            key = normalize_text(ingredient)
            if f" {key} " in padded or key in normalized:
                found.append(ingredient)
        return found

    def recommend(
        self,
        text: str,
        estimated: dict[str, float] | None = None,
        calorie_class: str | None = None,
    ) -> dict[str, list[str]]:
        substitutions: list[str] = []
        detected = self.detect_rule_ingredients(text)
        for ingredient in detected:
            rule = self.rules[ingredient]
            alternatives = ", ".join(rule["alternatives"])
            substitutions.append(f"For {ingredient}, consider {alternatives}. {rule['reason']}")

        suggestions: list[str] = []
        estimated = estimated or {}
        calories = float(estimated.get("total_calories", 0.0))
        fat = float(estimated.get("total_fat", 0.0))
        carb = float(estimated.get("total_carb", 0.0))
        protein = float(estimated.get("total_protein", 0.0))

        if calorie_class == "High Calorie" or calories > 600:
            suggestions.append("Consider a smaller portion, fewer calorie-dense toppings, or adding non-starchy vegetables.")
        if fat > 30:
            suggestions.append("The estimated fat is relatively high; reduce added oil, butter, cream, or fried components where practical.")
        if carb > 65:
            suggestions.append("The estimated carbohydrate level is high; balance the meal with vegetables, protein, and a smaller refined-grain portion.")
        if protein < 15 and calories > 150:
            suggestions.append("The meal may be relatively low in protein; consider beans, lentils, tofu, eggs, fish, or lean poultry as appropriate.")
        if not suggestions:
            suggestions.append("Aim for a balanced plate with vegetables, a protein source, and an appropriate portion of whole-food carbohydrates.")

        return {"detected_rule_ingredients": detected, "substitutions": substitutions, "dietary_suggestions": suggestions}
