from pathlib import Path

import pytest

from src.inference import NutriNLPInference


@pytest.fixture(scope="module")
def engine() -> NutriNLPInference:
    return NutriNLPInference(Path(__file__).resolve().parents[1])


def test_saved_models_load(engine: NutriNLPInference) -> None:
    assert engine.classifier is not None
    assert engine.nutrition is not None


def test_end_to_end_prediction_structure(engine: NutriNLPInference) -> None:
    result = engine.analyze("chicken with rice and broccoli")
    required = {
        "calorie_class",
        "estimated_nutrition",
        "uncertainty_ranges",
        "vocabulary_coverage",
        "healthier_alternatives",
        "dietary_suggestions",
        "warnings",
    }
    assert required.issubset(result)
    assert result["calorie_class"] in {"Low Calorie", "Medium Calorie", "High Calorie"}
    assert set(result["estimated_nutrition"]) == {"total_calories", "total_protein", "total_carb", "total_fat"}


def test_blank_inference_rejected(engine: NutriNLPInference) -> None:
    with pytest.raises(ValueError):
        engine.analyze(" ")


def test_mango_yogurt_milk_estimate_is_realistic(engine: NutriNLPInference) -> None:
    result = engine.analyze("Mango with yogurt and milk")
    nutrition = result["estimated_nutrition"]
    assert 50 <= nutrition["total_calories"] <= 350
    assert nutrition["total_protein"] > 0
    assert nutrition["total_carb"] > 0
    assert nutrition["total_fat"] >= 0
    assert "hybrid" in result["nutrition_estimation_method"]


def test_single_mango_uses_transparent_default_serving(engine: NutriNLPInference) -> None:
    result = engine.analyze("mango")
    calories = result["estimated_nutrition"]["total_calories"]
    assert 45 <= calories <= 80
    assert result["nutrition_estimation_method"] == "single-ingredient density estimator"
    assert "100 g" in result["portion_basis"]


def test_explicit_quantities_are_parsed(engine: NutriNLPInference) -> None:
    result = engine.analyze("100 g mango with 150 ml milk and 100 g yogurt")
    quantities = result["explicit_quantities_grams"]
    assert quantities["mangos"] == 100.0
    assert quantities["milk"] == 150.0
    assert quantities["yogurt"] == 100.0
    assert result["nutrition_estimation_method"] == "quantity-aware hybrid ingredient estimator"


def test_mutton_uses_transparent_proxy_and_is_not_low(engine: NutriNLPInference) -> None:
    result = engine.analyze("mutton meal")
    assert result["calorie_class"] != "Low Calorie"
    assert result["inference_aliases"]
    assert result["inference_aliases"][0]["proxy"] == "beef"


def test_goat_meat_does_not_match_goat_cheese(engine: NutriNLPInference) -> None:
    result = engine.analyze("goat meat meal")
    assert result["calorie_class"] != "Low Calorie"
    assert "goat cheese" not in [item.lower() for item in result["detected_ingredients"]]
    assert result["inference_aliases"]
