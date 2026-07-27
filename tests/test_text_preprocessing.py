from src.text_preprocessing import FoodTextPreprocessor, normalize_text


def test_deterministic_preprocessing() -> None:
    text = "Chicken breasts, TOMATOES & potatoes!!!"
    first = normalize_text(text)
    second = normalize_text(text)
    assert first == second
    assert first == "chicken breast tomato potato"


def test_empty_input_handling() -> None:
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""
    transformer = FoodTextPreprocessor()
    assert transformer.fit_transform([None, "Rice"]) == ["", "rice"]
