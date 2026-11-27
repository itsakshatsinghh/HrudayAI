"""Standalone MedPredict XGBoost loader, inference runner, and plain-text formatter.

Loads ONLY `disease predictor/model.json` and `disease predictor/label_encoder.pkl`
without importing `MedPredict/app.py` (avoiding Flask, TensorFlow, chatbot, and skin deps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from phase2_prediction.src.adapter import adapt_phase2_to_medpredict

NON_DIAGNOSTIC_DISCLAIMER = (
    "DISCLAIMER: This is an educational prototype, not a clinically validated diagnostic system. "
    "Listed conditions are model-generated candidate conditions with raw XGBoost probability scores—"
    "not medical diagnoses or clinical certainty. Consult a qualified healthcare professional."
)

_MODEL_CACHE: dict[str, dict[str, Any]] = {}


class MedPredictLoadError(RuntimeError):
    """Raised when MedPredict model artifacts cannot be located or loaded."""


def resolve_medpredict_dir(
    medpredict_dir: str | Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Locate the directory containing `model.json` and `label_encoder.pkl`."""
    candidates: list[Path] = []
    if medpredict_dir is not None:
        base = Path(medpredict_dir)
        candidates.extend([base, base / "disease predictor"])

    if repo_root is not None:
        candidates.append(Path(repo_root).parent / "MedPredict" / "disease predictor")

    candidates.extend(
        [
            Path("/content/MedPredict/disease predictor"),
            Path(__file__).resolve().parents[3] / "MedPredict" / "disease predictor",
        ]
    )

    for candidate in candidates:
        if (candidate / "model.json").is_file() and (candidate / "label_encoder.pkl").is_file():
            return candidate.resolve()

    searched = ", ".join(str(c) for c in candidates)
    raise MedPredictLoadError(
        f"Could not locate MedPredict `model.json` and `label_encoder.pkl`. Searched: {searched}"
    )


def load_medpredict_artifacts(
    medpredict_dir: str | Path | None = None,
    repo_root: Path | None = None,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load standalone XGBoost model.json and label_encoder.pkl with in-memory caching."""
    artifacts_dir = resolve_medpredict_dir(medpredict_dir=medpredict_dir, repo_root=repo_root)
    cache_key = str(artifacts_dir)
    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_path = artifacts_dir / "model.json"
    encoder_path = artifacts_dir / "label_encoder.pkl"

    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except Exception as exc:
        raise MedPredictLoadError(f"Failed to load XGBoost model from {model_path}: {exc}") from exc

    # Extract exact feature names in trained order from model or model.json learner metadata
    if hasattr(model, "feature_names_in_") and len(model.feature_names_in_) > 0:
        feature_names = [str(f) for f in model.feature_names_in_]
    else:
        raw_model = json.loads(model_path.read_text(encoding="utf-8"))
        feature_names = [str(f) for f in raw_model.get("learner", {}).get("feature_names", [])]

    if not feature_names:
        raise MedPredictLoadError(f"No feature_names found in {model_path}.")

    try:
        label_encoder = joblib.load(encoder_path)
    except Exception as exc:
        raise MedPredictLoadError(
            f"Failed to load label_encoder.pkl from {encoder_path}: {exc}"
        ) from exc

    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "artifacts_dir": artifacts_dir,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def predict_top_conditions(
    phase2_result: dict[str, Any],
    model_bundle: dict[str, Any],
    top_k: int = 3,
    mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the Phase 2.5 -> MedPredict adapter and local XGBoost inference."""
    feature_names: list[str] = model_bundle["feature_names"]
    adapted = adapt_phase2_to_medpredict(
        phase2_result=phase2_result,
        model_features=feature_names,
        mapping_path=mapping_path,
    )

    # JSON-serializable copy of adapter audit (omitting ndarray)
    audit = {k: v for k, v in adapted.items() if k != "feature_vector"}

    if not adapted["can_predict"]:
        return {
            "status": "clarification_required",
            "predictions": [],
            "message": adapted["clarification_message"],
            "adapter_audit": audit,
            "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
        }

    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    vector: np.ndarray = adapted["feature_vector"]

    probabilities = model.predict_proba(vector)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        class_idx = int(idx)
        condition_name = str(label_encoder.inverse_transform([class_idx])[0])
        raw_prob = float(probabilities[class_idx])
        predictions.append(
            {
                "rank": rank,
                "condition": condition_name,
                "model_score": round(raw_prob, 6),
                "score_percent": round(raw_prob * 100.0, 2),
            }
        )

    return {
        "status": "predicted",
        "predictions": predictions,
        "message": None,
        "adapter_audit": audit,
        "disclaimer": NON_DIAGNOSTIC_DISCLAIMER,
    }


def format_plain_text_report(
    audio_label: str,
    phase1_result: dict[str, Any],
    phase2_result: dict[str, Any],
    prediction_result: dict[str, Any],
) -> str:
    """Render a clean, reviewer-ready plain-text summary for one recording."""
    audit = prediction_result["adapter_audit"]
    lines: list[str] = [
        f"=== {audio_label} ===",
        f"Detected Language   : {phase1_result.get('detected_language', phase2_result.get('language', 'unknown'))}",
        f"Original Transcript : {phase2_result.get('original_transcript', phase1_result.get('text', ''))}",
        "",
        "Extracted Symptoms (Phase 2.5):",
    ]

    symptoms = phase2_result.get("symptoms", [])
    if not symptoms:
        lines.append("  (none extracted)")
    else:
        for sym in symptoms:
            lines.append(
                f"  - {sym.get('canonical_name')} | term='{sym.get('original_term')}' | "
                f"status={sym.get('temporal_status')} | certainty={sym.get('certainty')} | "
                f"negated={sym.get('negated')} | duration={sym.get('duration')} | "
                f"severity={sym.get('severity')}"
            )

    lines.extend(
        [
            "",
            f"Matched Model Features ({len(audit['matched_features'])}): "
            + (", ".join(audit["matched_features"]) if audit["matched_features"] else "(none)"),
        ]
    )

    if audit["excluded_symptoms"]:
        excluded_str = "; ".join(
            f"{e['canonical_name']} ({', '.join(e['reasons'])})" for e in audit["excluded_symptoms"]
        )
        lines.append(f"Excluded Symptoms        : {excluded_str}")
    else:
        lines.append("Excluded Symptoms        : (none)")

    if audit["ambiguous_symptoms"]:
        amb_str = "; ".join(
            f"{a['canonical_name']} -> needs clarification ({', '.join(a['candidates'])})"
            for a in audit["ambiguous_symptoms"]
        )
        lines.append(f"Needs Clarification      : {amb_str}")
    else:
        lines.append("Needs Clarification      : (none)")

    if audit["unmatched_symptoms"]:
        unm_str = ", ".join(u["canonical_name"] for u in audit["unmatched_symptoms"])
        lines.append(f"Unmatched Symptoms       : {unm_str}")
    else:
        lines.append("Unmatched Symptoms       : (none)")

    lines.append("")
    if prediction_result["status"] == "predicted":
        lines.append("Top-3 Model-Generated Candidate Conditions (XGBoost Raw Scores):")
        for pred in prediction_result["predictions"]:
            lines.append(
                f"  {pred['rank']}. {pred['condition']} — model score: {pred['model_score']:.4f} ({pred['score_percent']:.2f}%)"
            )
    else:
        lines.append(f"Prediction Status        : SKIPPED — {prediction_result['message']}")

    lines.extend(["", prediction_result["disclaimer"]])
    return "\n".join(lines)
