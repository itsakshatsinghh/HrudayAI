# Phase 1: Hindi & English Audio-to-Text (`phase1_voice/`)

Phase 1 converts uploaded Hindi (`hi`) and English (`en`) audio clips into text transcripts using configurable, free open-source speech recognition models.

> **Status:** Audio utilities, model configuration, and error-handling tests are verified locally on CPU. End-to-end ASR transcription has **not** yet been tested in Google Colab.

## Structure

- `src/audio_utils.py` — File validation, audio decoding, mono/16 kHz preprocessing, and error handling (`UnsupportedAudioFormatError`, `EmptyAudioError`, `AudioLoadError`).
- `src/transcription.py` — Configurable open-source ASR model loader and Hindi/English inference (`transcribe_audio`, `load_asr_model`, `UnsupportedLanguageError`, `TranscriptionError`).
- `configs/asr_models.json` — Registry of open-source Whisper / Hugging Face models for easy comparison.
- `notebooks/01_audio_to_text.ipynb` — Google Colab workflow for setup, audio upload, Hindi/English tests, and benchmark logging.
- `tests/test_phase1_voice.py` — Offline CPU unit tests for validation, preprocessing, and transcription contract checks.

## How to Run `01_audio_to_text.ipynb` in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `phase1_voice/notebooks/01_audio_to_text.ipynb` (or open it directly from your private GitHub repository).
2. *(Optional but recommended)* Choose **Runtime → Change runtime type → T4 GPU** (CPU runtime also works for `faster_whisper_base` / `faster_whisper_small`).
3. Run **Section 1 (Setup)** to clone/pull `HrudayAI` into `/content/HrudayAI` and install `phase1_voice/requirements.txt`.
4. Run **Section 2 (Model Selection)** to choose a model key from `phase1_voice/configs/asr_models.json` (default: `faster_whisper_small`).
5. Run **Section 3 (Audio Upload & Preprocessing)** to upload temporary Hindi and/or English audio clips (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
6. Run **Sections 4–6** to execute Hindi, English, and auto-detected transcription tests and record results in the notebook's comparison log.

> **Privacy:** Never commit uploaded audio recordings, patient samples, or model caches to Git.
