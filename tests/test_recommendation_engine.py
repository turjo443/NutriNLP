from pathlib import Path

from src.recommendation_engine import RecommendationEngine


def test_rule_matching() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = RecommendationEngine(root / "config" / "ingredient_substitutions.json")
    result = engine.recommend(
        "fried chicken with white rice and butter",
        {"total_calories": 700, "total_fat": 35, "total_carb": 80, "total_protein": 20},
        "High Calorie",
    )
    assert "white rice" in result["detected_rule_ingredients"]
    assert "butter" in result["detected_rule_ingredients"]
    assert "fried" in result["detected_rule_ingredients"]
    assert result["substitutions"]
    assert result["dietary_suggestions"]
