"""Unit tests for Phase 2 Hindi & English open-source symptom extraction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phase2_symptom_extraction.src import (
    SymptomExtractionError,
    extract_from_phase1_json_file,
    extract_from_phase1_payload,
    extract_symptoms,
)


class SymptomExtractionTests(unittest.TestCase):
    def test_hindi_phase1_actual_transcript(self) -> None:
        text = "मुजे कल से बुखार है, सिर में दर्द हो रहा है, और तोडी कमजोरी भी महसुस हो रही है"
        res = extract_symptoms(text, language="hi")
        self.assertEqual(res["language"], "hi")
        self.assertEqual(res["original_transcript"], text)

        by_name = {s["canonical_name"]: s for s in res["symptoms"]}
        self.assertEqual(set(by_name.keys()), {"fever", "headache", "weakness"})
        self.assertEqual(by_name["fever"]["original_term"], "बुखार")
        self.assertEqual(by_name["fever"]["duration"], "1 day")
        self.assertIsNone(by_name["fever"]["severity"])
        self.assertFalse(by_name["fever"]["negated"])

        self.assertEqual(by_name["headache"]["original_term"], "सिर में दर्द")
        self.assertIsNone(by_name["headache"]["duration"])
        self.assertFalse(by_name["headache"]["negated"])

        self.assertEqual(by_name["weakness"]["original_term"], "कमजोरी")
        self.assertEqual(by_name["weakness"]["severity"], "mild")
        self.assertFalse(by_name["weakness"]["negated"])

    def test_english_phase1_actual_transcript(self) -> None:
        text = "I have been experiencing fevers since yesterday along with a headache and some weaknesses."
        res = extract_symptoms(text, language="en")
        self.assertEqual(res["language"], "en")
        self.assertEqual(res["original_transcript"], text)

        by_name = {s["canonical_name"]: s for s in res["symptoms"]}
        self.assertEqual(set(by_name.keys()), {"fever", "headache", "weakness"})
        self.assertEqual(by_name["fever"]["original_term"], "fevers")
        self.assertEqual(by_name["fever"]["duration"], "1 day")
        self.assertIsNone(by_name["fever"]["severity"])
        self.assertFalse(by_name["fever"]["negated"])

        self.assertEqual(by_name["headache"]["original_term"], "headache")
        self.assertIsNone(by_name["headache"]["duration"])
        self.assertIsNone(by_name["headache"]["severity"])
        self.assertFalse(by_name["headache"]["negated"])

        self.assertEqual(by_name["weakness"]["original_term"], "weaknesses")
        self.assertIsNone(by_name["weakness"]["duration"])
        self.assertIsNone(by_name["weakness"]["severity"])
        self.assertFalse(by_name["weakness"]["negated"])

    def test_english_negation_duration_and_severity(self) -> None:
        text = "I have severe chest pain for 3 days, mild cough, and no fever or vomiting."
        res = extract_symptoms(text, language="en")
        by_name = {s["canonical_name"]: s for s in res["symptoms"]}

        self.assertEqual(by_name["chest_pain"]["severity"], "severe")
        self.assertEqual(by_name["chest_pain"]["duration"], "3 days")
        self.assertFalse(by_name["chest_pain"]["negated"])

        self.assertEqual(by_name["cough"]["severity"], "mild")
        self.assertIsNone(by_name["cough"]["duration"])
        self.assertFalse(by_name["cough"]["negated"])

        self.assertTrue(by_name["fever"]["negated"])
        self.assertTrue(by_name["vomiting"]["negated"])

    def test_hindi_mixed_positive_and_negated_symptoms(self) -> None:
        text = "मुझे दो दिन से तेज बुखार है और सिर में दर्द नहीं है, उल्टी भी नहीं हो रही।"
        res = extract_symptoms(text, language="hi")
        by_name = {s["canonical_name"]: s for s in res["symptoms"]}

        self.assertEqual(by_name["fever"]["duration"], "2 days")
        self.assertEqual(by_name["fever"]["severity"], "severe")
        self.assertFalse(by_name["fever"]["negated"])

        self.assertTrue(by_name["headache"]["negated"])
        self.assertTrue(by_name["vomiting"]["negated"])

    def test_hindi_shared_negation_without_positive_verb(self) -> None:
        text = "बुखार और खांसी नहीं है"
        res = extract_symptoms(text, language="hi")
        by_name = {s["canonical_name"]: s for s in res["symptoms"]}
        self.assertTrue(by_name["fever"]["negated"])
        self.assertTrue(by_name["cough"]["negated"])

    def test_unknown_duration_and_severity_remain_null(self) -> None:
        res = extract_symptoms("I have dizziness and nausea.", language="en")
        for item in res["symptoms"]:
            self.assertIsNone(item["duration"])
            self.assertIsNone(item["severity"])
            self.assertFalse(item["negated"])

    def test_auto_language_detection_and_invalid_input(self) -> None:
        hi_res = extract_symptoms("मुझे कल से बुखार है और सिर में दर्द है")
        self.assertEqual(hi_res["language"], "hi")
        with self.assertRaises(SymptomExtractionError):
            extract_symptoms("   ")
        with self.assertRaises(SymptomExtractionError):
            extract_symptoms("hello", language="fr")

    def test_phase1_payload_and_json_file_adapters(self) -> None:
        payload = {
            "hindi": {
                "text": "मुझे कल से बुखार है और सिर में दर्द है",
                "detected_language": "hi",
            },
            "english": {
                "text": "I have mild cough for 2 weeks and no fever.",
                "detected_language": "en",
            },
        }
        results = extract_from_phase1_payload(payload)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["language"], "hi")
        self.assertEqual(results[1]["language"], "en")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "transcription_results.json"
            json_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            file_results = extract_from_phase1_json_file(json_file)
            self.assertEqual(file_results, results)


if __name__ == "__main__":
    unittest.main()
