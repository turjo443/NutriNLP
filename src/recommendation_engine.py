"""Transparent rule-based ingredient and calorie-reduction recommendations.

Numerical calorie-saving estimates use median ingredient calorie density learned
from the uploaded CSV through the saved nutrition-estimator artifact. The rules
themselves are curated and editable; they are not learned substitution labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .text_preprocessing import normalize_text


class RecommendationEngine:
    """Match curated substitutions and estimate optional calorie savings."""

    def __init__(
        self,
        rules_path: str | Path,
        ingredient_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.rules: dict[str, dict[str, Any]] = json.loads(
            self.rules_path.read_text(encoding="utf-8")
        )
        self.ingredient_profiles: dict[str, Mapping[str, Any]] = {
            normalize_text(key): value
            for key, value in (ingredient_profiles or {}).items()
        }

    def detect_rule_ingredients(self, text: str) -> list[str]:
        """Return matching rule triggers, with longer phrases checked first."""
        normalized = normalize_text(text)
        padded = f" {normalized} "
        found: list[str] = []
        for ingredient in sorted(
            self.rules,
            key=lambda item: (len(normalize_text(item).split()), len(item)),
            reverse=True,
        ):
            key = normalize_text(ingredient)
            if not key:
                continue
            # Use token boundaries to avoid accidental substring matches.
            if f" {key} " in padded:
                found.append(ingredient)
                continue
            # Generic triggers such as "cheese" should also match "goat cheese".
            if len(key.split()) == 1 and key in normalized.split():
                found.append(ingredient)
        return found

    def _resolve_profile_key(
        self,
        trigger: str,
        rule: Mapping[str, Any],
        detected_ingredients: Sequence[str] | None,
    ) -> str | None:
        explicit_key = normalize_text(rule.get("profile_key", trigger))
        if explicit_key in self.ingredient_profiles:
            return explicit_key

        trigger_key = normalize_text(trigger)
        candidates = [normalize_text(item) for item in (detected_ingredients or [])]
        candidates = [
            item
            for item in candidates
            if item in self.ingredient_profiles
            and (
                trigger_key == item
                or trigger_key in item.split()
                or item in trigger_key
            )
        ]
        if candidates:
            return max(candidates, key=lambda item: (len(item.split()), len(item)))
        return None

    def _kcal_per_gram(self, profile_key: str | None) -> float | None:
        if not profile_key:
            return None
        profile = self.ingredient_profiles.get(profile_key)
        if not profile:
            return None
        values = profile.get("median_per_gram")
        if not values:
            return None
        value = float(values[0])
        return value if value >= 0 else None

    @staticmethod
    def _normalized_quantities(
        explicit_quantities: Mapping[str, float] | None,
    ) -> dict[str, float]:
        return {
            normalize_text(name): float(value)
            for name, value in (explicit_quantities or {}).items()
            if float(value) > 0
        }

    def _quantity_for_profile(
        self,
        profile_key: str | None,
        quantities: Mapping[str, float],
    ) -> float | None:
        if not profile_key:
            return None
        if profile_key in quantities:
            return float(quantities[profile_key])
        # Match a generic rule such as cheese to a quantified phrase such as
        # goat cheese or mozzarella cheese.
        matches = [
            (name, value)
            for name, value in quantities.items()
            if profile_key == name
            or profile_key in name.split()
            or name in profile_key
        ]
        if not matches:
            return None
        return float(max(matches, key=lambda item: len(item[0]))[1])

    def _calculate_reduction(
        self,
        trigger: str,
        rule: Mapping[str, Any],
        profile_key: str | None,
        amount_g: float | None,
    ) -> dict[str, Any] | None:
        options = rule.get("calorie_reduction_options") or []
        if not options:
            return None

        original_density = self._kcal_per_gram(profile_key)
        best: dict[str, Any] | None = None

        for option in options:
            option_type = str(option.get("type", "replacement"))
            suggestion = str(option.get("label", "Use the lower-calorie option."))
            saving: float | None = None
            basis: str

            if option_type == "preparation":
                basis = str(
                    option.get(
                        "basis",
                        "Preparation-method suggestion; exact saving depends on added oil.",
                    )
                )
            elif amount_g is None or original_density is None:
                basis = "Add an ingredient amount in g or ml to calculate estimated calorie savings."
            elif option_type == "portion_reduction":
                multiplier = float(option.get("portion_multiplier", 0.5))
                multiplier = min(max(multiplier, 0.0), 1.0)
                saving = amount_g * original_density * (1.0 - multiplier)
                basis = (
                    f"Compared with the supplied {amount_g:.0f} g amount; "
                    f"the suggested portion is approximately {amount_g * multiplier:.0f} g."
                )
            elif option_type in {"replacement", "blend"}:
                replacement_key = normalize_text(option.get("replacement_profile", ""))
                replacement_density = self._kcal_per_gram(replacement_key)
                if replacement_density is None:
                    basis = "The replacement is qualitative because no compatible CSV calorie profile was available."
                else:
                    fraction = float(option.get("replacement_fraction", 1.0))
                    fraction = min(max(fraction, 0.0), 1.0)
                    saving = amount_g * fraction * max(0.0, original_density - replacement_density)
                    replacement_name = str(option.get("replacement_name", replacement_key))
                    basis = (
                        f"Same-weight comparison for {amount_g * fraction:.0f} g using "
                        f"CSV-derived median calorie densities: {profile_key} "
                        f"{original_density * 100:.0f} kcal/100 g vs {replacement_name} "
                        f"{replacement_density * 100:.0f} kcal/100 g."
                    )
            else:
                basis = "Qualitative rule; no numeric saving was calculated."

            candidate = {
                "trigger": trigger,
                "original_ingredient": profile_key or trigger,
                "suggestion": suggestion,
                "estimated_saving_kcal": None if saving is None else round(max(0.0, saving), 1),
                "basis": basis,
                "is_rule_based": True,
            }
            # Prefer the option with the greatest computable saving. If none is
            # computable, keep the first transparent qualitative option.
            if best is None:
                best = candidate
            elif candidate["estimated_saving_kcal"] is not None and (
                best["estimated_saving_kcal"] is None
                or candidate["estimated_saving_kcal"] > best["estimated_saving_kcal"]
            ):
                best = candidate

        return best

    def recommend(
        self,
        text: str,
        estimated: dict[str, float] | None = None,
        calorie_class: str | None = None,
        explicit_quantities: Mapping[str, float] | None = None,
        detected_ingredients: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        substitutions: list[str] = []
        detected = self.detect_rule_ingredients(text)
        quantities = self._normalized_quantities(explicit_quantities)
        reduction_plan: list[dict[str, Any]] = []

        for ingredient in detected:
            rule = self.rules[ingredient]
            alternatives = ", ".join(rule.get("alternatives", []))
            reason = str(rule.get("reason", ""))
            if alternatives:
                substitutions.append(
                    f"For {ingredient}, consider {alternatives}. {reason}".strip()
                )

            profile_key = self._resolve_profile_key(
                ingredient,
                rule,
                detected_ingredients,
            )
            amount_g = self._quantity_for_profile(profile_key, quantities)
            recommendation = self._calculate_reduction(
                ingredient,
                rule,
                profile_key,
                amount_g,
            )
            if recommendation:
                reduction_plan.append(recommendation)

        suggestions: list[str] = []
        estimated = estimated or {}
        calories = float(estimated.get("total_calories", 0.0))
        fat = float(estimated.get("total_fat", 0.0))
        carb = float(estimated.get("total_carb", 0.0))
        protein = float(estimated.get("total_protein", 0.0))

        if calorie_class == "High Calorie" or calories > 600:
            suggestions.append(
                "Consider a smaller portion, fewer calorie-dense toppings, or more non-starchy vegetables."
            )
        if fat > 30:
            suggestions.append(
                "The estimated fat is relatively high; reduce added oil, butter, cream, or fried components where practical."
            )
        if carb > 65:
            suggestions.append(
                "The estimated carbohydrate level is high; use a smaller refined-grain portion and add vegetables or protein."
            )
        if protein < 15 and calories > 150:
            suggestions.append(
                "The meal may be relatively low in protein; consider beans, lentils, tofu, eggs, fish, or lean poultry as appropriate."
            )
        if not suggestions:
            suggestions.append(
                "Aim for a balanced plate with vegetables, a protein source, and an appropriate portion of whole-food carbohydrates."
            )

        numeric_savings = [
            float(item["estimated_saving_kcal"])
            for item in reduction_plan
            if item.get("estimated_saving_kcal") is not None
        ]
        total_saving = round(sum(numeric_savings), 1)
        revised_calories = round(max(0.0, calories - total_saving), 1) if numeric_savings else None

        return {
            "detected_rule_ingredients": detected,
            "substitutions": substitutions,
            "dietary_suggestions": suggestions,
            "calorie_reduction_plan": reduction_plan,
            "estimated_total_calorie_saving": total_saving if numeric_savings else None,
            "estimated_revised_calories": revised_calories,
            "calorie_reduction_note": (
                "Savings are educational estimates. Numeric values use supplied quantities and "
                "median calorie densities learned from the uploaded CSV; substitution rules are curated."
            ),
        }
