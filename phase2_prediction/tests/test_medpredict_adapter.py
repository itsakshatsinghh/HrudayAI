"""Unit and integration tests for Phase 2.5 -> MedPredict adapter and standalone predictor."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from phase2_prediction.src import (
    NON_DIAGNOSTIC_DISCLAIMER,
    adapt_phase2_to_medpredict,
    format_plain_text_report,
    load_medpredict_artifacts,
    predict_top_conditions,
)
from phase2_symptom_extraction.src import extract_symptoms

SAMPLE_ORDERED_FEATURES = [
    "headache",
    "mild_fever",
    "high_fever",
    "chills",
    "cough",
    "muscle_pain",
    "fatigue",
]


class _DummyModel:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.called = False

    def predict_proba(self, vec: np.ndarray) -> np.ndarray:
        self.called = True
        assert vec.shape == (1, self.n_features)
        return np.array([[0.10, 0.55, 0.25, 0.10]], dtype=np.float32)


class _DummyEncoder:
    classes_ = ["CondA", "CondB", "CondC", "CondD"]

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]


class MedPredictAdapterTests(unittest.TestCase):
    def test_filters_historical_negated_and_uncertain_symptoms(self) -> None:
        phase2_out = {
            "language": "en",
            "original_transcript": "synthetic",
            "symptoms": [
                {
                    "canonical_name": "headache",
                    "original_term": "headache",
                    "duration": None,
                    "severity": "severe",
                    "negated": False,
                    "temporal_status": "current",
                    "certainty": "confirmed",
                },
                {
                    "canonical_name": "chills",
                    "original_term": "chills",
                    "duration": "last week",
                    "severity": None,
                    "negated": False,
                    "temporal_status": "historical",
                    "certainty": "confirmed",
                },
                {
                    "canonical_name": "cough",
                    "original_term": "cough",
                    "duration": None,
                    "severity": None,
                    "negated": True,
                    "temporal_status": "current",
                    "certainty": "confirmed",
                },
                {
                    "canonical_name": "body_ache",
                    "original_term": "body ache",
                    "duration": None,
                    "severity": None,
                    "negated": False,
                    "temporal_status": "current",
                    "certainty": "uncertain",
                },
            ],
        }
        adapted = adapt_phase2_to_medpredict(phase2_out, SAMPLE_ORDERED_FEATURES)
        self.assertTrue(adapted["can_predict"])
        self.assertEqual(adapted["matched_features"], ["headache"])
        self.assertEqual(len(adapted["excluded_symptoms"]), 3)
        np.testing.assert_array_equal(
            adapted["feature_vector"],
            np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        # Original input preserved untouched
        self.assertEqual(len(adapted["phase2_output"]["symptoms"]), 4)

    def test_ambiguous_fever_and_weakness_are_never_guessed(self) -> None:
        phase2_out = extract_symptoms(
            "I have been experiencing fevers since yesterday along with a headache and some weaknesses.",
            language="en",
        )
        adapted = adapt_phase2_to_medpredict(phase2_out, SAMPLE_ORDERED_FEATURES)
        self.assertTrue(adapted["can_predict"])
        self.assertEqual(adapted["matched_features"], ["headache"])
        amb_names = {a["canonical_name"] for a in adapted["ambiguous_symptoms"]}
        self.assertEqual(amb_names, {"fever", "weakness"})
        # Neither mild_fever nor high_fever nor fatigue may be activated
        vec = adapted["feature_vector"][0]
        self.assertEqual(vec[SAMPLE_ORDERED_FEATURES.index("mild_fever")], 0.0)
        self.assertEqual(vec[SAMPLE_ORDERED_FEATURES.index("high_fever")], 0.0)
        self.assertEqual(vec[SAMPLE_ORDERED_FEATURES.index("fatigue")], 0.0)

    def test_severity_conditional_fever_maps_explicitly(self) -> None:
        severe_out = extract_symptoms("I have high fever and cough.", language="en")
        adapted_sev = adapt_phase2_to_medpredict(severe_out, SAMPLE_ORDERED_FEATURES)
        self.assertEqual(set(adapted_sev["matched_features"]), {"high_fever", "cough"})

        mild_out = extract_symptoms("I have mild fever.", language="en")
        adapted_mild = adapt_phase2_to_medpredict(mild_out, SAMPLE_ORDERED_FEATURES)
        self.assertEqual(adapted_mild["matched_features"], ["mild_fever"])

    def test_unmatched_symptom_recorded_and_zero_matches_skip_model(self) -> None:
        phase2_out = {
            "language": "en",
            "original_transcript": "I have wheezing.",
            "symptoms": [
                {
                    "canonical_name": "wheezing",
                    "original_term": "wheezing",
                    "duration": None,
                    "severity": None,
                    "negated": False,
                    "temporal_status": "current",
                    "certainty": "confirmed",
                }
            ],
        }
        dummy_model = _DummyModel(len(SAMPLE_ORDERED_FEATURES))
        bundle = {
            "model": dummy_model,
            "label_encoder": _DummyEncoder(),
            "feature_names": SAMPLE_ORDERED_FEATURES,
        }
        pred = predict_top_conditions(phase2_out, bundle)
        self.assertEqual(pred["status"], "clarification_required")
        self.assertFalse(dummy_model.called)
        self.assertEqual(pred["predictions"], [])
        self.assertEqual(pred["adapter_audit"]["unmatched_symptoms"][0]["canonical_name"], "wheezing")

    def test_preserves_custom_non_sorted_feature_order(self) -> None:
        custom_order = ["cough", "headache", "muscle_pain"]
        phase2_out = extract_symptoms("I have body ache.", language="en")
        adapted = adapt_phase2_to_medpredict(phase2_out, custom_order)
        self.assertEqual(adapted["matched_features"], ["muscle_pain"])
        np.testing.assert_array_equal(
            adapted["feature_vector"],
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_real_medpredict_artifacts_load_and_predict(self) -> None:
        med_dir = Path(__file__).resolve().parents[3] / "MedPredict" / "disease predictor"
        if not (med_dir / "model.json").exists():
            self.skipTest("Sibling MedPredict repository not present in this environment.")

        bundle = load_medpredict_artifacts(medpredict_dir=med_dir)
        self.assertEqual(len(bundle["feature_names"]), 131)
        self.assertEqual(len(bundle["label_encoder"].classes_), 150)

        # Synthetic Phase 2.5 output matching the actual Hindi transcript + one clear feature
        p2_hindi = extract_symptoms(
            "मुजे कल से बुखार है, सिर में दर्द हो रहा है, और तोडी कमजोरी भी महसुस हो रही है",
            language="hi",
        )
        res = predict_top_conditions(p2_hindi, bundle, top_k=3)
        self.assertEqual(res["status"], "predicted")
        self.assertEqual(len(res["predictions"]), 3)
        self.assertEqual(res["adapter_audit"]["matched_features"], ["headache"])
        self.assertEqual(
            {a["canonical_name"] for a in res["adapter_audit"]["ambiguous_symptoms"]},
            {"fever", "weakness"},
        )
        report = format_plain_text_report(
            "Audio 1 (Hindi)",
            {"detected_language": "hi", "text": p2_hindi["original_transcript"]},
            p2_hindi,
            res,
        )
        self.assertIn("Top-3 Model-Generated Candidate Conditions", report)
        self.assertIn(NON_DIAGNOSTIC_DISCLAIMER, report)


if __name__ == "__main__":
    unittest.main()
