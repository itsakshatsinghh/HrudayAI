"""Audio validation, loading, and preprocessing utilities for Phase 1.

Kept strictly independent from ASR model inference so audio handling can be
tested and reused cleanly across CPU and Google Colab environments.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
import wave

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SECONDS = 0.05
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioLoadError(Exception):
    """Base error raised when validating, loading, or preprocessing audio."""


class UnsupportedAudioFormatError(AudioLoadError):
    """Raised when an audio file uses an unsupported file extension."""


class EmptyAudioError(AudioLoadError):
    """Raised when an audio file or waveform is empty or silent."""


def validate_audio_path(audio_path: str | Path) -> Path:
    """Validate that an audio path exists, is non-empty, and has a supported extension."""
    if not audio_path or not str(audio_path).strip():
        raise AudioLoadError("Audio path cannot be empty.")

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise AudioLoadError(f"Audio path is not a file: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext or '<none>'}'. Supported formats: {supported}"
        )

    if path.stat().st_size == 0:
        raise EmptyAudioError(f"Audio file is empty (0 bytes): {path}")

    return path


def _load_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    """Load uncompressed PCM WAV audio using Python's standard library."""
    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
    except wave.Error as exc:
        raise AudioLoadError(f"Failed to read WAV file '{path.name}': {exc}") from exc

    if n_frames == 0 or not raw_bytes or sample_rate <= 0:
        raise EmptyAudioError(f"WAV file contains no audio frames: {path.name}")

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise AudioLoadError(
            f"Unsupported WAV sample width ({sample_width} bytes). Install soundfile/librosa to decode."
        )

    samples = np.frombuffer(raw_bytes, dtype=dtype_map[sample_width]).astype(np.float32)
    if sample_width == 1:
        waveform = (samples - 128.0) / 128.0
    elif sample_width == 2:
        waveform = samples / 32768.0
    else:
        waveform = samples / 2147483648.0

    if channels > 1:
        waveform = waveform.reshape(-1, channels)

    return waveform, sample_rate


def _resample_linear(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio with numpy linear interpolation when librosa is unavailable."""
    if orig_sr == target_sr or waveform.size == 0:
        return waveform
    target_length = max(1, int(round(waveform.shape[0] * float(target_sr) / float(orig_sr))))
    old_indices = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
    new_indices = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(new_indices, old_indices, waveform).astype(np.float32)


def preprocess_audio(
    waveform: np.ndarray,
    sample_rate: int,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
    trim_silence: bool = False,
) -> tuple[np.ndarray, int]:
    """Convert waveform to mono float32, resample to target_sr, and optionally normalize/trim."""
    if sample_rate <= 0:
        raise AudioLoadError(f"Invalid sample rate: {sample_rate}")

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        raise EmptyAudioError("Audio waveform is empty.")

    if audio.ndim == 2:
        if mono:
            # Handle both (samples, channels) from soundfile and (channels, samples) from librosa
            axis = 1 if audio.shape[1] <= 8 and audio.shape[0] > audio.shape[1] else 0
            audio = np.mean(audio, axis=axis, dtype=np.float32)
    elif audio.ndim > 2:
        raise AudioLoadError(f"Expected 1D or 2D audio array, got shape {audio.shape}")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(np.abs(audio))) <= 1e-7:
        raise EmptyAudioError("Audio waveform contains only silence.")

    if trim_silence and importlib.util.find_spec("librosa") is not None:
        import librosa

        trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if trimmed.size > 0:
            audio = trimmed

    if sample_rate != target_sr:
        if importlib.util.find_spec("librosa") is not None:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        else:
            audio = _resample_linear(audio, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    duration = audio.shape[0] / float(sample_rate)
    if duration < MIN_AUDIO_DURATION_SECONDS:
        raise EmptyAudioError(
            f"Audio clip is too short ({duration:.3f}s < {MIN_AUDIO_DURATION_SECONDS}s minimum)."
        )

    if normalize:
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            audio = (audio / peak).astype(np.float32)

    return audio.astype(np.float32), sample_rate


def load_audio(
    audio_path: str | Path,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
    trim_silence: bool = False,
) -> tuple[np.ndarray, int]:
    """Load and preprocess an audio file into a mono float32 waveform at target_sr."""
    path = validate_audio_path(audio_path)
    ext = path.suffix.lower()

    waveform: np.ndarray
    sample_rate: int

    if importlib.util.find_spec("librosa") is not None:
        import librosa

        try:
            waveform, sample_rate = librosa.load(str(path), sr=None, mono=mono)
        except Exception as exc:
            raise AudioLoadError(f"Failed to decode audio '{path.name}': {exc}") from exc
    elif importlib.util.find_spec("soundfile") is not None and ext in {".wav", ".flac", ".ogg"}:
        import soundfile as sf

        try:
            waveform, sample_rate = sf.read(str(path), dtype="float32")
        except Exception as exc:
            raise AudioLoadError(f"Failed to read audio '{path.name}': {exc}") from exc
    elif ext == ".wav":
        waveform, sample_rate = _load_wav_stdlib(path)
    else:
        raise AudioLoadError(
            f"Cannot decode '{ext}' without librosa/ffmpeg installed. "
            "Install Phase 1 dependencies in Google Colab (phase1_voice/requirements.txt)."
        )

    return preprocess_audio(
        waveform=waveform,
        sample_rate=int(sample_rate),
        target_sr=target_sr,
        mono=mono,
        normalize=normalize,
        trim_silence=trim_silence,
    )


def get_audio_metadata(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return summary audio properties without exposing raw samples."""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        raise EmptyAudioError("Cannot compute metadata for empty audio.")

    return {
        "sample_rate_hz": int(sample_rate),
        "num_samples": int(audio.shape[0]),
        "duration_seconds": round(float(audio.shape[0]) / float(sample_rate), 3),
        "peak_amplitude": round(float(np.max(np.abs(audio))), 4),
        "rms_amplitude": round(float(np.sqrt(np.mean(np.square(audio)))), 4),
    }
