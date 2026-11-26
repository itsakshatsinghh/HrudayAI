"""Phase 1 voice intake and Hindi/English audio-to-text modules."""

from phase1_voice.src.audio_utils import (
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
    AudioLoadError,
    EmptyAudioError,
    UnsupportedAudioFormatError,
    get_audio_metadata,
    load_audio,
    preprocess_audio,
    validate_audio_path,
)
from phase1_voice.src.transcription import (
    TARGET_LANGUAGES,
    ASRModelConfig,
    TranscriptionError,
    UnsupportedLanguageError,
    get_model_config,
    list_available_models,
    load_asr_model,
    resolve_device,
    transcribe_audio,
)

__all__ = [
    "ASRModelConfig",
    "AudioLoadError",
    "DEFAULT_SAMPLE_RATE",
    "EmptyAudioError",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "TARGET_LANGUAGES",
    "TranscriptionError",
    "UnsupportedAudioFormatError",
    "UnsupportedLanguageError",
    "get_audio_metadata",
    "get_model_config",
    "list_available_models",
    "load_asr_model",
    "load_audio",
    "preprocess_audio",
    "resolve_device",
    "transcribe_audio",
    "validate_audio_path",
]
