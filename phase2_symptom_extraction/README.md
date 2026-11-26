# Phase 2 Symptom Extraction (`phase2_symptom_extraction/`)

Open-source, locally executable Hindi (`hi`) and English (`en`) symptom extraction module for HrudayAI.

Takes Phase 1 speech-to-text JSON (`phase1_voice/outputs/transcription_results.json` or raw transcripts) and extracts structured symptom records containing:
- `canonical_name`: Normalized symptom label
- `original_term`: Matched Hindi or English phrase from the transcript
- `duration`: Normalized duration when explicitly stated (otherwise `null`)
- `severity`: `"mild"`, `"moderate"`, or `"severe"` when explicitly stated (otherwise `null`)
- `negated`: `true` for explicitly negated symptoms, `false` otherwise

> **Scope Note:** This module performs symptom extraction and normalization only. It does **not** run disease prediction or clinical diagnosis, and clinical accuracy has **not** yet been formally evaluated on patient cohorts.

## Directory Layout

- `src/config_loader.py` — Loads lexicon and extraction rule JSON configs.
- `src/normalizer.py` — Unicode/whitespace normalization + longest-first exact and fuzzy (`rapidfuzz` / `difflib`) symptom matching.
- `src/extractor.py` — Clause splitting, duration parsing, severity extraction, negation scoping, and Phase 1 JSON adapters (`extract_symptoms`, `extract_from_phase1_payload`, `extract_from_phase1_json_file`).
- `configs/symptom_lexicon.json` — Configurable Hindi-English symptom synonym dictionary.
- `configs/extraction_rules.json` — Configurable negation triggers, severity modifiers, clause separators, and matching thresholds.
- `notebooks/01_symptom_extraction.ipynb` — Google Colab CPU-compatible test notebook.
- `tests/test_symptom_extraction.py` — Unit tests covering Hindi, English, negation, duration, severity, and Phase 1 JSON adaptation.

## Installation

Free, open-source, CPU-only dependency (`rapidfuzz`, MIT License; falls back to Python stdlib `difflib` if omitted):

```bash
pip install -r phase2_symptom_extraction/requirements.txt
```

## Execution

### Local / Unit Tests

```bash
python -m unittest discover -s phase2_symptom_extraction/tests -v
```

### Google Colab

1. Open `phase2_symptom_extraction/notebooks/01_symptom_extraction.ipynb` in Google Colab (CPU runtime is sufficient).
2. Clone or pull `https://github.com/itsakshatsinghh/HrudayAI` into `/content/HrudayAI`.
3. Install `phase2_symptom_extraction/requirements.txt`.
4. Load `phase1_voice/outputs/transcription_results.json` (or use the notebook's Phase 1 sample fallback if running before uploading/generating the file) and run all cells.

## Limitations & Licensing

- **Languages:** Currently supports Hindi (`hi`) and English (`en`) only.
- **Rule & Lexicon Scope:** Extraction covers symptoms defined in `configs/symptom_lexicon.json` and explicit duration/severity/negation patterns in `configs/extraction_rules.json`; unseen colloquialisms or complex multi-clause negations may require expanding the configs.
- **Clinical Disclaimer:** Preliminary research tooling only; not clinically validated or intended as a medical device.
- **Licensing:** All code and configuration in this module use open-source Python (`rapidfuzz` is MIT-licensed). No paid APIs, external endpoints, or proprietary dependencies are used.
