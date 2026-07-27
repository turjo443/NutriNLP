# Verification Summary

## Completed executions

- Raw CSV validation and dish-level aggregation: **passed**
- Original full-mode eight-classifier training/evaluation: **passed**
- Improved hybrid nutrition-estimator training on the fixed training partition: **passed**
- Untouched-test nutrition evaluation and empirical interval generation: **passed**
- Saved-artifact loading and sample end-to-end inference: **passed**
- `Mango with yogurt and milk` regression check: **103.7 kcal**, realistic non-negative macros
- Explicit quantity parsing check (`100 g mango`, `150 ml milk`, `100 g yogurt`): **passed**
- Single-ingredient default-serving check (`mango`): **60.6 kcal for stated 100 g assumption**
- Python syntax compilation for project modules and `app.py`: **passed**
- `pytest -q`: **18 passed, 1 skipped**

## Streamlit verification note

The Streamlit runtime smoke test is skipped in this build container because Streamlit is not installed here. The app source compiles successfully, and the user has already run the original Streamlit application locally. The updated app uses the same launch command and saved-artifact loading path.

## Improved nutrition-estimator results

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Calories | 81.440 kcal | 276.835 kcal | 0.4233 |
| Protein | 5.384 g | 9.638 g | 0.7557 |
| Carbohydrates | 6.977 g | 17.230 g | 0.3779 |
| Fat | 4.903 g | 24.735 g | 0.3685 |

The estimator is rated **moderate reliability**. Exact results still require exact ingredient quantities; the interface now makes that limitation and every portion assumption visible.
