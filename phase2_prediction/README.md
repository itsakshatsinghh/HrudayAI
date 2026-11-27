# Phase 3 Integration: HrudayAI -> MedPredict (`phase2_prediction/`)

Standalone adapter and local XGBoost inference layer connecting HrudayAI's Phase 2.5 validated symptom JSON to MedPredict's pre-trained 131-feature disease prediction model—without Flask, TensorFlow, paid APIs, or model retraining.

## Components

- `configs/medpredict_symptom_map.json` — Explicit, non-guessing mapping between Phase 2.5 canonical symptoms and MedPredict's 131 features.
- `src/adapter.py` — Filters eligible symptoms (`temporal_status == "current"`, `certainty == "confirmed"`, `negated == False`), records excluded/ambiguous/unmatched symptoms, preserves exact 131-feature order, and prevents inference when 0 valid features match.
- `src/predictor.py` — Loads standalone `disease predictor/model.json` and `disease predictor/label_encoder.pkl`, executes local `XGBClassifier.predict_proba()`, and renders plain-text reviewer reports.
- `notebooks/02_hrudayai_reviewer_demo.ipynb` — End-to-end Google Colab demo (upload 2 audio files → Phase 1 transcription → Phase 2.5 extraction → Adapter → MedPredict top-3 plain-text predictions).
- `tests/test_medpredict_adapter.py` — Unit and artifact-loading tests.

## Mapping & Safety Policy

- **Strict eligibility:** Historical, negated, and uncertain symptoms never become positive model features.
- **No guessing on ambiguous terms:**
  - Generic `fever` maps to `high_fever` only when `severity == "severe"` and `mild_fever` when `severity == "mild"`; unspecified `fever` (`severity: null`) is flagged for clarification.
  - Generic `weakness`, `swelling`, and `frequent_urination` are flagged for clarification rather than guessed.
- **Zero-match guardrail:** If 0 valid features match, `predict_top_conditions()` skips inference and returns a clarification message.
- **Disclaimer:** Outputs are model-generated candidate conditions from an educational prototype—not medical diagnoses.

## Installation & Testing

```bash
pip install -r phase2_prediction/requirements.txt
python -m unittest discover -s phase2_prediction/tests -v
```

In Google Colab, open `phase2_prediction/notebooks/02_hrudayai_reviewer_demo.ipynb` directly from GitHub (`https://github.com/itsakshatsinghh/HrudayAI`). Section 1 automatically clones `HrudayAI` and `MedPredict` from their public repositories and installs the required dependencies.
