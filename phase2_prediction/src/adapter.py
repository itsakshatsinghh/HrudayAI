"""Adapter converting Phase 2.5 validated symptom output into MedPredict's 131-feature vector.

Strictly preserves the trained model's feature ordering and never turns historical,
negated, uncertain, or ambiguous symptoms into positive model features.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "medpredict_symptom_map.json"
)


class AdapterConfigError(Exception):
    """Raised when the MedPredict symptom mapping configuration is missing or invalid."""


def load_symptom_mapping(mapping_path: str | Path | None = None) -> dict[str, Any]:
    """Load the explicit canonical-to-MedPredict feature mapping JSON."""
    path = Path(mapping_path) if mapping_path is not None else DEFAULT_MAPPING_PATH
    if not path.exists():
        raise AdapterConfigError(f"Mapping configuration not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict) or "direct_mappings" not in data:
        raise AdapterConfigError(f"Mapping file {path} must define 'direct_mappings'.")
    return data


def _exclusion_reasons(symptom: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    temporal_status = str(symptom.get("temporal_status", "current")).strip().lower()
    certainty = str(symptom.get("certainty", "confirmed")).strip().lower()
    negated = bool(symptom.get("negated", False))

    if temporal_status != "current":
        reasons.append(f"temporal_status={temporal_status}")
    if certainty != "confirmed":
        reasons.append(f"certainty={certainty}")
    if negated:
        reasons.append("negated=true")
    return reasons


def adapt_phase2_to_medpredict(
    phase2_result: dict[str, Any],
    model_features: list[str],
    mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Filter, map, and vectorize Phase 2.5 symptom output into MedPredict's exact feature order.

    Rules enforced:
      - Preserves `phase2_result` untouched.
      - Only `temporal_status == "current"`, `certainty == "confirmed"`, and `negated == False`
        symptoms are eligible to activate positive model features.
      - Never guesses ambiguous symptoms (e.g. severity-less `fever` or generic `weakness`).
      - Preserves `model_features` order verbatim (never sorts independently).
      - Sets `can_predict = False` with a clarification message when 0 features match.
    """
    if not model_features:
        raise ValueError("model_features must be a non-empty list preserving the model's exact order.")

    mapping_cfg = load_symptom_mapping(mapping_path)
    direct_map: dict[str, str] = mapping_cfg.get("direct_mappings", {})
    severity_cond_map: dict[str, dict[str, Any]] = mapping_cfg.get("severity_conditional_mappings", {})
    ambiguous_map: dict[str, dict[str, Any]] = mapping_cfg.get("ambiguous_clarifications", {})

    feature_index = {feature: idx for idx, feature in enumerate(model_features)}
    vector = np.zeros((1, len(model_features)), dtype=np.float32)

    preserved_phase2 = copy.deepcopy(phase2_result)
    symptoms = preserved_phase2.get("symptoms", [])
    if not isinstance(symptoms, list):
        raise ValueError("phase2_result['symptoms'] must be a list.")

    matched_features: list[str] = []
    matched_set: set[str] = set()
    mapping_details: list[dict[str, Any]] = []
    excluded_symptoms: list[dict[str, Any]] = []
    ambiguous_symptoms: list[dict[str, Any]] = []
    unmatched_symptoms: list[dict[str, Any]] = []

    for sym in symptoms:
        canonical = str(sym.get("canonical_name", "")).strip()
        original_term = sym.get("original_term")
        severity = sym.get("severity")

        reasons = _exclusion_reasons(sym)
        if reasons:
            excluded_symptoms.append(
                {
                    "canonical_name": canonical,
                    "original_term": original_term,
                    "reasons": reasons,
                }
            )
            continue

        # Eligible: current + confirmed + not negated
        target_feature: str | None = None
        mapping_type: str = ""

        if canonical in severity_cond_map:
            rule = severity_cond_map[canonical]
            by_sev = rule.get("by_severity", {})
            sev_key = str(severity).strip().lower() if severity is not None else ""
            candidate = by_sev.get(sev_key)
            if candidate and candidate in feature_index:
                target_feature = candidate
                mapping_type = f"severity_conditional ({sev_key})"
            else:
                ambiguous_symptoms.append(
                    {
                        "canonical_name": canonical,
                        "original_term": original_term,
                        "severity": severity,
                        "candidates": list(rule.get("clarification_candidates", [])),
                        "clarification_prompt": rule.get(
                            "clarification_prompt",
                            f"Clarify details for '{canonical}' before mapping.",
                        ),
                    }
                )
                continue
        elif canonical in direct_map:
            candidate = direct_map[canonical]
            if candidate in feature_index:
                target_feature = candidate
                mapping_type = "direct_exact" if candidate == canonical else "direct_alias"
            else:
                unmatched_symptoms.append(
                    {
                        "canonical_name": canonical,
                        "original_term": original_term,
                        "reason": f"Mapped target '{candidate}' not present in model features.",
                    }
                )
                continue
        elif canonical in ambiguous_map:
            rule = ambiguous_map[canonical]
            ambiguous_symptoms.append(
                {
                    "canonical_name": canonical,
                    "original_term": original_term,
                    "severity": severity,
                    "candidates": list(rule.get("clarification_candidates", [])),
                    "clarification_prompt": rule.get(
                        "clarification_prompt",
                        f"Clarify '{canonical}' before mapping.",
                    ),
                }
            )
            continue
        elif canonical in feature_index:
            target_feature = canonical
            mapping_type = "direct_model_feature"
        else:
            unmatched_symptoms.append(
                {
                    "canonical_name": canonical,
                    "original_term": original_term,
                    "reason": "No equivalent feature in MedPredict 131-feature schema.",
                }
            )
            continue

        if target_feature is not None:
            idx = feature_index[target_feature]
            vector[0, idx] = 1.0
            if target_feature not in matched_set:
                matched_set.add(target_feature)
                matched_features.append(target_feature)
            mapping_details.append(
                {
                    "canonical_name": canonical,
                    "original_term": original_term,
                    "model_feature": target_feature,
                    "feature_index": idx,
                    "mapping_type": mapping_type,
                }
            )

    can_predict = len(matched_features) > 0
    clarification_message: str | None = None
    if not can_predict:
        parts = [
            "Insufficient unambiguous current confirmed symptoms mapped to MedPredict's 131 features; "
            "model inference was not executed."
        ]
        if ambiguous_symptoms:
            prompts = "; ".join(a["clarification_prompt"] for a in ambiguous_symptoms)
            parts.append(f"Clarification needed: {prompts}")
        if unmatched_symptoms:
            names = ", ".join(u["canonical_name"] for u in unmatched_symptoms)
            parts.append(f"Unmatched symptoms: {names}")
        if excluded_symptoms:
            names = ", ".join(
                f"{e['canonical_name']} ({', '.join(e['reasons'])})" for e in excluded_symptoms
            )
            parts.append(f"Excluded non-eligible symptoms: {names}")
        clarification_message = " ".join(parts)

    return {
        "can_predict": can_predict,
        "clarification_message": clarification_message,
        "matched_features": matched_features,
        "mapping_details": mapping_details,
        "excluded_symptoms": excluded_symptoms,
        "ambiguous_symptoms": ambiguous_symptoms,
        "unmatched_symptoms": unmatched_symptoms,
        "feature_vector": vector,
        "feature_count": len(model_features),
        "phase2_output": preserved_phase2,
    }
