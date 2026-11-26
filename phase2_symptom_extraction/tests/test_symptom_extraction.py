"""Unit tests for Phase 2 / Milestone 2.5 Hindi & English symptom extraction."""

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
        self.assertEqual(by_name["fever"]["temporal_status"], "current")
        self.assertEqual(by_name["fever"]["certainty"], "confirmed")

        self.assertEqual(by_name["headache"]["original_term"], "सिर में दर्द")
        self.assertIsNone(by_name["headache"]["duration"])
        self.assertFalse(by_name["headache"]["negated"])
        self.assertEqual(by_name["headache"]["temporal_status"], "current")

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
        self.assertEqual(by_name["fever"]["temporal_status"], "current")
        self.assertEqual(by_name["fever"]["certainty"], "confirmed")

        self.assertEqual(by_name["headache"]["original_term"], "headache")
        self.assertIsNone(by_name["headache"]["duration"])
        self.assertIsNone(by_name["headache"]["severity"])
        self.assertFalse(by_name["headache"]["negated"])

        self.assertEqual(by_name["weakness"]["original_term"], "weaknesses")
        self.assertIsNone(by_name["weakness"]["duration"])
        self.assertIsNone(by_name["weakness"]["severity"])
        self.assertFalse(by_name["weakness"]["negated"])

    def test_english_negation_cases(self) -> None:
        for text, expected_name in [
            ("I don't have fever.", "fever"),
            ("No headache.", "headache"),
            ("I do not have cough.", "cough"),
        ]:
            res = extract_symptoms(text, language="en")
            self.assertEqual(len(res["symptoms"]), 1)
            item = res["symptoms"][0]
            self.assertEqual(item["canonical_name"], expected_name)
            self.assertTrue(item["negated"], msg=f"Failed negation for: {text}")
            self.assertEqual(item["temporal_status"], "current")
            self.assertEqual(item["certainty"], "confirmed")

    def test_hindi_negation_cases(self) -> None:
        for text, expected_name in [
            ("मुझे बुखार नहीं है।", "fever"),
            ("सिर में दर्द नहीं है।", "headache"),
        ]:
            res = extract_symptoms(text, language="hi")
            self.assertEqual(len(res["symptoms"]), 1)
            item = res["symptoms"][0]
            self.assertEqual(item["canonical_name"], expected_name)
            self.assertTrue(item["negated"], msg=f"Failed negation for: {text}")
            self.assertEqual(item["temporal_status"], "current")
            self.assertEqual(item["certainty"], "confirmed")

    def test_historical_symptoms_hindi_and_english(self) -> None:
        en_res = extract_symptoms("I had fever last week.", language="en")
        self.assertEqual(len(en_res["symptoms"]), 1)
        en_item = en_res["symptoms"][0]
        self.assertEqual(en_item["canonical_name"], "fever")
        self.assertEqual(en_item["duration"], "last week")
        self.assertFalse(en_item["negated"])
        self.assertEqual(en_item["temporal_status"], "historical")
        self.assertEqual(en_item["certainty"], "confirmed")

        hi_res = extract_symptoms("मुझे पिछले हफ्ते बुखार था।", language="hi")
        self.assertEqual(len(hi_res["symptoms"]), 1)
        hi_item = hi_res["symptoms"][0]
        self.assertEqual(hi_item["canonical_name"], "fever")
        self.assertEqual(hi_item["duration"], "last week")
        self.assertFalse(hi_item["negated"])
        self.assertEqual(hi_item["temporal_status"], "historical")
        self.assertEqual(hi_item["certainty"], "confirmed")

    def test_current_symptoms_hindi_and_english(self) -> None:
        en_res = extract_symptoms("I have had fever for 3 days.", language="en")
        self.assertEqual(en_res["symptoms"][0]["temporal_status"], "current")
        self.assertEqual(en_res["symptoms"][0]["duration"], "3 days")

        hi_res = extract_symptoms("मुझे अभी बुखार है।", language="hi")
        self.assertEqual(hi_res["symptoms"][0]["temporal_status"], "current")
        self.assertFalse(hi_res["symptoms"][0]["negated"])

    def test_severity_levels_explicit_only(self) -> None:
        cases = [
            ("en", "I have severe headache.", "headache", "severe"),
            ("hi", "मुझे बहुत तेज सिर दर्द है।", "headache", "severe"),
            ("en", "I have mild weakness.", "weakness", "mild"),
            ("hi", "थोड़ी कमजोरी है।", "weakness", "mild"),
            ("en", "I have moderate chest pain.", "chest_pain", "moderate"),
        ]
        for lang, text, expected_name, expected_sev in cases:
            res = extract_symptoms(text, language=lang)
            self.assertEqual(res["symptoms"][0]["canonical_name"], expected_name)
            self.assertEqual(res["symptoms"][0]["severity"], expected_sev)

    def test_uncertainty_and_question_statements(self) -> None:
        cases = [
            ("en", "I think I might have a fever."),
            ("hi", "शायद मुझे बुखार है।"),
            ("en", "Do I have a fever?"),
            ("hi", "क्या मुझे बुखार है?"),
        ]
        for lang, text in cases:
            res = extract_symptoms(text, language=lang)
            self.assertEqual(len(res["symptoms"]), 1)
            item = res["symptoms"][0]
            self.assertEqual(item["canonical_name"], "fever")
            self.assertEqual(item["certainty"], "uncertain", msg=f"Expected uncertain for: {text}")
            self.assertFalse(item["negated"])

    def test_duration_extraction_hindi_and_english(self) -> None:
        self.assertEqual(extract_symptoms("Fever for three days.", language="en")["symptoms"][0]["duration"], "3 days")
        self.assertEqual(extract_symptoms("तीन दिन से बुखार है।", language="hi")["symptoms"][0]["duration"], "3 days")
        self.assertEqual(extract_symptoms("Fever since yesterday.", language="en")["symptoms"][0]["duration"], "1 day")
        self.assertEqual(extract_symptoms("कल से बुखार है।", language="hi")["symptoms"][0]["duration"], "1 day")

    def test_temporal_changes_past_to_current(self) -> None:
        en_res = extract_symptoms("I had fever yesterday, but I don't have it now.", language="en")
        self.assertEqual(len(en_res["symptoms"]), 2)
        past_item, curr_item = en_res["symptoms"]
        self.assertEqual(past_item["canonical_name"], "fever")
        self.assertEqual(past_item["temporal_status"], "historical")
        self.assertFalse(past_item["negated"])
        self.assertEqual(past_item["duration"], "yesterday")
        self.assertEqual(curr_item["canonical_name"], "fever")
        self.assertEqual(curr_item["temporal_status"], "current")
        self.assertTrue(curr_item["negated"])

        hi_res = extract_symptoms("कल बुखार था, लेकिन अब नहीं है।", language="hi")
        self.assertEqual(len(hi_res["symptoms"]), 2)
        hi_past, hi_curr = hi_res["symptoms"]
        self.assertEqual(hi_past["canonical_name"], "fever")
        self.assertEqual(hi_past["temporal_status"], "historical")
        self.assertFalse(hi_past["negated"])
        self.assertEqual(hi_curr["canonical_name"], "fever")
        self.assertEqual(hi_curr["temporal_status"], "current")
        self.assertTrue(hi_curr["negated"])

    def test_english_and_hindi_mixed_positive_and_negative(self) -> None:
        en_res = extract_symptoms(
            "I have severe chest pain for 3 days, mild cough, and no fever or vomiting.",
            language="en",
        )
        by_name = {s["canonical_name"]: s for s in en_res["symptoms"]}
        self.assertEqual(by_name["chest_pain"]["severity"], "severe")
        self.assertEqual(by_name["chest_pain"]["duration"], "3 days")
        self.assertFalse(by_name["chest_pain"]["negated"])
        self.assertEqual(by_name["cough"]["severity"], "mild")
        self.assertFalse(by_name["cough"]["negated"])
        self.assertTrue(by_name["fever"]["negated"])
        self.assertTrue(by_name["vomiting"]["negated"])

        hi_res = extract_symptoms(
            "मुझे दो दिन से तेज बुखार है और सिर में दर्द नहीं है, उल्टी भी नहीं हो रही।",
            language="hi",
        )
        hi_by = {s["canonical_name"]: s for s in hi_res["symptoms"]}
        self.assertEqual(hi_by["fever"]["duration"], "2 days")
        self.assertEqual(hi_by["fever"]["severity"], "severe")
        self.assertFalse(hi_by["fever"]["negated"])
        self.assertTrue(hi_by["headache"]["negated"])
        self.assertTrue(hi_by["vomiting"]["negated"])

    def test_hindi_shared_negation_without_positive_verb(self) -> None:
        res = extract_symptoms("बुखार और खांसी नहीं है", language="hi")
        by_name = {s["canonical_name"]: s for s in res["symptoms"]}
        self.assertTrue(by_name["fever"]["negated"])
        self.assertTrue(by_name["cough"]["negated"])

    def test_ambiguous_statements_preserve_uncertainty(self) -> None:
        for lang, text in [
            ("en", "I am not sure if I have fever or not."),
            ("hi", "पता नहीं बुखार है या नहीं।"),
        ]:
            res = extract_symptoms(text, language=lang)
            self.assertEqual(len(res["symptoms"]), 1)
            item = res["symptoms"][0]
            self.assertEqual(item["canonical_name"], "fever")
            self.assertEqual(item["certainty"], "uncertain")
            self.assertFalse(item["negated"])

    def test_unknown_duration_and_severity_remain_null(self) -> None:
        res = extract_symptoms("I have dizziness and nausea.", language="en")
        for item in res["symptoms"]:
            self.assertIsNone(item["duration"])
            self.assertIsNone(item["severity"])
            self.assertFalse(item["negated"])
            self.assertEqual(item["temporal_status"], "current")
            self.assertEqual(item["certainty"], "confirmed")

    def test_auto_language_detection_and_invalid_input(self) -> None:
        hi_res = extract_symptoms("मुझे कल से बुखार है और सिर में दर्द है")
        self.assertEqual(hi_res["language"], "hi")
        with self.assertRaises(SymptomExtractionError):
            extract_symptoms("   ")
        with self.assertRaises(SymptomExtractionError):
            extract_symptoms("")
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
