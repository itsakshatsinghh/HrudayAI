"""Configurable open-source Hindi & English speech-to-text module for Phase 1.

Designed for heavy model execution in Google Colab while keeping configuration,
validation, and device selection CPU-compatible and free of paid APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase1_voice.src.audio_utils import DEFAULT_SAMPLE_RATE, preprocess_audio

TARGET_LANGUAGES: dict[str, str] = {
    "hi": "Hindi",
    "en": "English",
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "asr_models.json"

_FALLBACK_MODELS: dict[str, dict[str, Any]] = {
    "faster_whisper_small": {
        "backend": "faster_whisper",
        "model_id": "small",
        "display_name": "Whisper Small (faster-whisper)",
        "supported_languages": ["hi", "en"],
        "notes": "Recommended Phase 1 starting point balancing Hindi/English accuracy and Colab speed.",
    },
    "faster_whisper_base": {
        "backend": "faster_whisper",
        "model_id": "base",
        "display_name": "Whisper Base (faster-whisper)",
        "supported_languages": ["hi", "en"],
        "notes": "Lightweight baseline for CPU and GPU comparison.",
    },
    "hf_whisper_small": {
        "backend": "transformers_whisper",
        "model_id": "openai/whisper-small",
        "display_name": "OpenAI Whisper Small (Transformers)",
        "supported_languages": ["hi", "en"],
        "notes": "Hugging Face Transformers ASR baseline.",
    },
}


class TranscriptionError(Exception):
    """Base error raised for ASR model loading or transcription failures."""


class UnsupportedLanguageError(TranscriptionError):
    """Raised when a caller requests a language outside Phase 1 scope (hi, en)."""


_MODEL_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def clear_model_cache() -> None:
    """Clear the in-memory loaded ASR model cache."""
    _MODEL_CACHE.clear()


@dataclass(frozen=True)
class ASRModelConfig:
    """Configuration metadata for an open-source ASR model candidate."""

    key: str
    backend: str
    model_id: str
    display_name: str
    supported_languages: tuple[str, ...]
    notes: str = ""


def _load_registry(config_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models")
        if isinstance(models, dict) and models:
            return models
    return _FALLBACK_MODELS


def list_available_models(config_path: Path | None = None) -> dict[str, ASRModelConfig]:
    """Return all configured open-source ASR model candidates."""
    raw_models = _load_registry(config_path)
    return {
        key: ASRModelConfig(
            key=key,
            backend=str(entry["backend"]),
            model_id=str(entry["model_id"]),
            display_name=str(entry.get("display_name", key)),
            supported_languages=tuple(entry.get("supported_languages", ["hi", "en"])),
            notes=str(entry.get("notes", "")),
        )
        for key, entry in raw_models.items()
    }


def get_model_config(model_key: str, config_path: Path | None = None) -> ASRModelConfig:
    """Look up a single ASR model configuration by key."""
    models = list_available_models(config_path)
    if model_key not in models:
        valid = ", ".join(sorted(models.keys()))
        raise ValueError(f"Unknown model_key '{model_key}'. Available models: {valid}")
    return models[model_key]


def resolve_device(preferred_device: str = "auto") -> tuple[str, str]:
    """Resolve execution device ('cuda' or 'cpu') and compute precision without assuming GPU."""
    choice = preferred_device.strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{preferred_device}'. Choose 'auto', 'cpu', or 'cuda'.")

    cuda_available = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda_available = bool(torch.cuda.is_available())

    if choice == "cuda" and not cuda_available:
        raise TranscriptionError("CUDA requested, but no GPU is available in this runtime.")

    use_cuda = cuda_available if choice == "auto" else (choice == "cuda")
    if use_cuda:
        return "cuda", "float16"
    return "cpu", "int8"


def load_asr_model(
    model_key: str = "faster_whisper_small",
    device: str = "auto",
    config_path: Path | None = None,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load and cache an open-source ASR model bundle for Phase 1 inference."""
    config = get_model_config(model_key, config_path=config_path)
    resolved_device, compute_type = resolve_device(device)
    cache_key = (config.key, resolved_device)

    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if config.backend == "faster_whisper":
        if importlib.util.find_spec("faster_whisper") is None:
            raise TranscriptionError(
                "faster-whisper is not installed. Run `pip install -r phase1_voice/requirements.txt` in Colab."
            )
        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(
                config.model_id,
                device=resolved_device,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load faster-whisper model '{config.model_id}': {exc}"
            ) from exc

        bundle = {
            "config": config,
            "backend": config.backend,
            "device": resolved_device,
            "compute_type": compute_type,
            "model": model,
        }
        _MODEL_CACHE[cache_key] = bundle
        return bundle

    if config.backend == "transformers_whisper":
        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            raise TranscriptionError(
                "transformers/torch are not installed. Run `pip install -r phase1_voice/requirements.txt` in Colab."
            )
        import torch
        from transformers import pipeline

        device_index = 0 if resolved_device == "cuda" else -1
        torch_dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        try:
            asr_pipe = pipeline(
                task="automatic-speech-recognition",
                model=config.model_id,
                device=device_index,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Failed to load transformers ASR model '{config.model_id}': {exc}"
            ) from exc

        bundle = {
            "config": config,
            "backend": config.backend,
            "device": resolved_device,
            "compute_type": str(torch_dtype),
            "model": asr_pipe,
        }
        _MODEL_CACHE[cache_key] = bundle
        return bundle

    raise TranscriptionError(f"Unsupported ASR backend '{config.backend}' for model '{model_key}'.")


def _validate_language_hint(language: str | None, config: ASRModelConfig) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized not in TARGET_LANGUAGES:
        allowed = ", ".join(sorted(TARGET_LANGUAGES.keys()))
        raise UnsupportedLanguageError(
            f"Unsupported language '{language}'. Phase 1 currently supports only: {allowed} (or 'auto')."
        )
    if normalized not in config.supported_languages:
        raise UnsupportedLanguageError(
            f"Model '{config.key}' supports {config.supported_languages}, not '{normalized}'."
        )
    return normalized


def _transcribe_faster_whisper(
    waveform: np.ndarray,
    model_bundle: dict[str, Any],
    language_hint: str | None,
) -> dict[str, Any]:
    model = model_bundle["model"]

    def _run(use_vad: bool):
        transcribe_kwargs: dict[str, Any] = {
            "beam_size": 5,
            "task": "transcribe",
            "vad_filter": use_vad,
        }
        if language_hint is not None:
            transcribe_kwargs["language"] = language_hint
        segments_iter, info = model.transcribe(waveform, **transcribe_kwargs)
        segments = [
            {
                "start": round(float(seg.start), 2),
                "end": round(float(seg.end), 2),
                "text": seg.text.strip(),
            }
            for seg in segments_iter
            if seg.text and seg.text.strip()
        ]
        full_text = " ".join(seg["text"] for seg in segments).strip()
        return full_text, segments, info

    full_text, segments, info = _run(use_vad=True)
    # Retry without VAD if aggressive silence filtering removed quiet speech
    if not full_text:
        full_text, segments, info = _run(use_vad=False)

    detected_lang = str(getattr(info, "language", None) or language_hint or "unknown").lower()
    lang_prob = getattr(info, "language_probability", None)
    return {
        "text": full_text,
        "detected_language": detected_lang,
        "language_probability": round(float(lang_prob), 4) if lang_prob is not None else None,
        "segments": segments,
    }


def _transcribe_transformers_whisper(
    waveform: np.ndarray,
    sample_rate: int,
    model_bundle: dict[str, Any],
    language_hint: str | None,
) -> dict[str, Any]:
    asr_pipe = model_bundle["model"]
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language_hint is not None:
        generate_kwargs["language"] = TARGET_LANGUAGES[language_hint].lower()

    output = asr_pipe(
        {"array": waveform, "sampling_rate": sample_rate},
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )
    text = str(output.get("text", "")).strip()
    return {
        "text": text,
        "detected_language": language_hint or "auto",
        "language_probability": None,
        "segments": [],
    }


def transcribe_audio(
    waveform: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    model_bundle: dict[str, Any] | None = None,
    model_key: str = "faster_whisper_small",
    language: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Transcribe a preprocessed audio waveform into Hindi or English text."""
    audio_array = np.asarray(waveform, dtype=np.float32)
    if audio_array.size == 0:
        raise TranscriptionError("Cannot transcribe an empty waveform.")

    audio_16k, sr_16k = preprocess_audio(
        audio_array,
        sample_rate=sample_rate,
        target_sr=DEFAULT_SAMPLE_RATE,
        mono=True,
        normalize=False,
    )

    bundle = model_bundle or load_asr_model(model_key=model_key, device=device)
    config: ASRModelConfig = bundle["config"]
    language_hint = _validate_language_hint(language, config)

    try:
        if bundle["backend"] == "faster_whisper":
            raw = _transcribe_faster_whisper(audio_16k, bundle, language_hint)
        elif bundle["backend"] == "transformers_whisper":
            raw = _transcribe_transformers_whisper(audio_16k, sr_16k, bundle, language_hint)
        elif bundle["backend"] == "custom_callable":
            raw = bundle["model"](audio_16k, sr_16k, language_hint)
        else:
            raise TranscriptionError(f"Unsupported backend in bundle: {bundle['backend']}")
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Speech-to-text inference failed: {exc}") from exc

    text = str(raw.get("text", "")).strip()
    if not text:
        raise TranscriptionError(
            "Transcription produced empty text. Check audio volume, clarity, or selected language."
        )

    detected_lang = str(raw.get("detected_language") or language_hint or "unknown").lower()
    warnings: list[str] = []
    target_supported = detected_lang in TARGET_LANGUAGES
    if detected_lang != "auto" and not target_supported:
        warnings.append(
            f"Detected language '{detected_lang}' is outside Phase 1 scope (hi, en)."
        )

    duration_seconds = round(float(audio_16k.shape[0]) / float(sr_16k), 3)
    return {
        "text": text,
        "detected_language": detected_lang,
        "language_probability": raw.get("language_probability"),
        "target_language_supported": target_supported,
        "duration_seconds": duration_seconds,
        "model_key": config.key,
        "model_id": config.model_id,
        "backend": config.backend,
        "device": bundle["device"],
        "segments": raw.get("segments", []),
        "warnings": warnings,
    }
