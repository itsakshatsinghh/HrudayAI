# Phase 2 Symptom Extraction (`phase2_symptom_extraction/`)

Open-source, locally executable Hindi (`hi`) and English (`en`) symptom validation and clinical language extraction module for HrudayAI.

Accepts Phase 1 speech-to-text JSON (`phase1_voice/outputs/transcription_results.json`) or raw transcripts and produces structured symptom JSON distinguishing positive, negated, historical, current, uncertain, and temporal-change statements.

> **Scope & Safety Note:** This module performs rule-based symptom extraction and normalization only. It does **not** connect to disease prediction or perform clinical diagnosis, and clinical accuracy has **not** yet been evaluated on patient cohorts.

## JSON Output Schema (Milestone 2.5)

```json
{
  "language": "en",
  "original_transcript": "I had fever last week, but I don't have it now.",
  "symptoms": [
    {
      "canonical_name": "fever",
      "original_term": "fever",
      "duration": "last week",
      "severity": null,
      "negated": false,
      "temporal_status": "historical",
      "certainty": "confirmed"
    },
    {
      "canonical_name": "fever",
      "original_term": "fever",
      "duration": null,
      "severity": null,
      "negated": true,
      "temporal_status": "current",
      "certainty": "confirmed"
    }
  ]
}
```

### Field Definitions
- `canonical_name` (`str`): Normalized symptom label from `configs/symptom_lexicon.json`.
- `original_term` (`str`): Exact matched surface term from the transcript (or resolved antecedent term for elliptical follow-up clauses).
- `duration` (`str | null`): Explicit duration or temporal anchor (`"1 day"`, `"3 days"`, `"last week"`, etc.), or `null` when not mentioned.
- `severity` (`"mild" | "moderate" | "severe" | null`): Extracted only when explicitly stated; otherwise `null`.
- `negated` (`bool`): `true` when explicitly negated, `false` otherwise.
- `temporal_status` (`"current" | "historical"`): Distinguishes active/present symptoms (`"current"`) from resolved/past symptoms (`"historical"`).
- `certainty` (`"confirmed" | "uncertain"`): Marks hedged, ambiguous, or question-mentioned symptoms as `"uncertain"`.

## Directory Layout

- `src/config_loader.py` — Validates and loads lexicon and extraction rule JSON configs.
- `src/normalizer.py` — Unicode/whitespace normalization + longest-first exact and fuzzy (`rapidfuzz` / `difflib`) symptom matching.
- `src/extractor.py` — Clause splitting, negation scoping, temporal classification, certainty/question detection, ellipsis resolution, duration & severity extraction, and Phase 1 JSON adapters.
- `configs/symptom_lexicon.json` — Configurable Hindi-English symptom synonym dictionary.
- `configs/extraction_rules.json` — Configurable negation, temporal, uncertainty, ellipsis, severity, and matching rules.
- `notebooks/01_symptom_extraction.ipynb` — Google Colab CPU-compatible notebook using `shared/colab_bootstrap.py`.
- `tests/test_symptom_extraction.py` — Unit tests covering Hindi/English negation, historical vs. current, severity, uncertainty, duration, temporal transitions, ambiguity, and Phase 1 JSON adapters.

## Installation & Execution

Free, open-source, CPU-only dependency (`rapidfuzz`, MIT License; falls back to Python stdlib `difflib` if omitted):

```bash
pip install -r phase2_symptom_extraction/requirements.txt
python -m unittest discover -s phase2_symptom_extraction/tests -v
```

### Google Colab

1. Open `phase2_symptom_extraction/notebooks/01_symptom_extraction.ipynb` directly from GitHub in Google Colab (CPU runtime is sufficient).
2. Run **Section 1** to auto-clone/pull `https://github.com/itsakshatsinghh/HrudayAI.git` via `shared/colab_bootstrap.py` and install `rapidfuzz`.
3. In **Section 2**, load `phase1_voice/outputs/transcription_results.json` (or upload it via `UPLOAD_PHASE1_JSON = True`, or enable `USE_DEMO_SAMPLE = True` for explicitly labeled demo testing).
4. Run **Sections 3–5** to inspect extracted JSON, Milestone 2.5 clinical language cases, and Colab unit test results.

## Limitations & Licensing

- **Languages:** Currently supports Hindi (`hi`) and English (`en`) only.
- **Rule & Lexicon Scope:** Rule-based extraction covers symptoms in `configs/symptom_lexicon.json` and explicit linguistic cues in `configs/extraction_rules.json`. Complex multi-symptom anaphora (e.g., `"it"` after listing multiple different symptoms) is intentionally left unresolved rather than guessed.
- **Duration vs. Historical Anchors:** Ongoing durations (`"कल से"`, `"for three days"`) normalize to counts (`"1 day"`, `"3 days"`), while past anchors (`"last week"`, `"पिछले हफ्ते"`, `"yesterday"`) preserve anchor labels (`"last week"`, `"yesterday"`).
- **Clinical Disclaimer:** Not clinically validated; must not be used for medical diagnosis without clinical evaluation.
- **Licensing:** Pure open-source Python (`rapidfuzz` is MIT-licensed). Zero paid APIs or external inference endpoints.
