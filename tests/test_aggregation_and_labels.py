import pandas as pd

from src.data_preparation import aggregate_to_dish_level, assign_calorie_class, create_split_and_labels


def test_dish_level_aggregation() -> None:
    raw = pd.DataFrame(
        [
            {"dish_id": "d1", "ingr_id": "i1", "ingr_name": "rice", "grams": 100, "calories": 130, "fat": 1, "carb": 28, "protein": 3},
            {"dish_id": "d1", "ingr_id": "i2", "ingr_name": "chicken", "grams": 50, "calories": 90, "fat": 2, "carb": 0, "protein": 18},
            {"dish_id": "d2", "ingr_id": "i3", "ingr_name": "broccoli", "grams": 80, "calories": 30, "fat": 0, "carb": 6, "protein": 2},
        ]
    )
    dish = aggregate_to_dish_level(raw)
    d1 = dish.set_index("dish_id").loc["d1"]
    assert len(dish) == 2
    assert d1["dish_text"] == "rice, chicken"
    assert d1["total_grams"] == 150
    assert d1["total_calories"] == 220
    assert d1["total_protein"] == 21


def test_calorie_label_boundaries() -> None:
    assert assign_calorie_class(100, 100, 200) == 0
    assert assign_calorie_class(100.01, 100, 200) == 1
    assert assign_calorie_class(200, 100, 200) == 1
    assert assign_calorie_class(201, 100, 200) == 2


def test_train_test_dish_separation() -> None:
    dish = pd.DataFrame(
        {
            "dish_id": [f"d{i}" for i in range(30)],
            "dish_text": [f"food {i}" for i in range(30)],
            "total_grams": range(30),
            "total_calories": [i * 10 + 1 for i in range(30)],
            "total_fat": range(30),
            "total_carb": range(30),
            "total_protein": range(30),
            "ingredient_count": [1] * 30,
            "named_ingredient_count": [1] * 30,
        }
    )
    labeled, _ = create_split_and_labels(dish, random_state=42)
    train_ids = set(labeled.loc[labeled.split == "train", "dish_id"])
    test_ids = set(labeled.loc[labeled.split == "test", "dish_id"])
    assert not train_ids & test_ids
