"""Offline CPU unit tests for Phase 1 audio loading and transcription interfaces.

These tests verify audio validation, preprocessing, configuration loading, and
error handling without downloading or executing large ASR model weights.
"""

from __future__ import annotations

import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

import numpy as np

from phase1_voice.src.audio_utils import (
    EmptyAudioError,
    UnsupportedAudioFormatError,
    get_audio_metadata,
    load_audio,
    preprocess_audio,
    validate_audio_path,
)
from phase1_voice.src.transcription import (
    TranscriptionError,
    UnsupportedLanguageError,
    get_model_config,
    list_available_models,
    transcribe_audio,
)


def _write_test_wav(path: Path, sample_rate: int = 16000, duration_s: float = 0.25, amplitude: float = 0.5) -> None:
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = amplitude * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)
            frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, val)) * 32767)))
        wf.writeframes(bytes(frames))


class Phase1AudioAndTranscriptionTests(unittest.TestCase):
    def test_unsupported_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "notes.txt"
            bad_file.write_text("not audio", encoding="utf-8")
            with self.assertRaises(UnsupportedAudioFormatError):
                validate_audio_path(bad_file)

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_wav = Path(tmpdir) / "empty.wav"
            empty_wav.write_bytes(b"")
            with self.assertRaises(EmptyAudioError):
                validate_audio_path(empty_wav)

    def test_silent_wav_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            silent_wav = Path(tmpdir) / "silent.wav"
            _write_test_wav(silent_wav, amplitude=0.0)
            with self.assertRaises(EmptyAudioError):
                load_audio(silent_wav)

    def test_load_and_preprocess_synthetic_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "tone_8k.wav"
            _write_test_wav(wav_path, sample_rate=8000, duration_s=0.3, amplitude=0.25)
            waveform, sr = load_audio(wav_path, target_sr=16000)
            self.assertEqual(sr, 16000)
            self.assertEqual(waveform.dtype, np.float32)
            meta = get_audio_metadata(waveform, sr)
            self.assertAlmostEqual(meta["duration_seconds"], 0.3, places=1)
            self.assertAlmostEqual(meta["peak_amplitude"], 1.0, places=2)

    def test_preprocess_stereo_to_mono(self) -> None:
        stereo = np.ones((1600, 2), dtype=np.float32) * 0.25
        mono, sr = preprocess_audio(stereo, sample_rate=16000, target_sr=16000)
        self.assertEqual(mono.ndim, 1)
        self.assertEqual(sr, 16000)

    def test_model_registry_loads_from_config(self) -> None:
        models = list_available_models()
        self.assertIn("faster_whisper_small", models)
        cfg = get_model_config("faster_whisper_small")
        self.assertEqual(cfg.backend, "faster_whisper")
        self.assertEqual(cfg.supported_languages, ("hi", "en"))

    def test_unsupported_language_raises(self) -> None:
        cfg = get_model_config("faster_whisper_small")
        fake_bundle = {
            "config": cfg,
            "backend": "custom_callable",
            "device": "cpu",
            "model": lambda *_args: {"text": "unused", "detected_language": "hi"},
        }
        dummy_audio = np.ones(1600, dtype=np.float32) * 0.1
        with self.assertRaises(UnsupportedLanguageError):
            transcribe_audio(dummy_audio, 16000, model_bundle=fake_bundle, language="ta")

    def test_empty_transcription_output_raises(self) -> None:
        cfg = get_model_config("faster_whisper_small")
        fake_bundle = {
            "config": cfg,
            "backend": "custom_callable",
            "device": "cpu",
            "model": lambda *_args: {"text": "   ", "detected_language": "en"},
        }
        dummy_audio = np.ones(1600, dtype=np.float32) * 0.1
        with self.assertRaises(TranscriptionError):
            transcribe_audio(dummy_audio, 16000, model_bundle=fake_bundle, language="en")

    def test_transcribe_contract_with_mock_backend(self) -> None:
        cfg = get_model_config("faster_whisper_small")
        fake_bundle = {
            "config": cfg,
            "backend": "custom_callable",
            "device": "cpu",
            "model": lambda *_args: {
                "text": "mujhe kal se bukhar hai",
                "detected_language": "hi",
                "language_probability": 0.96,
                "segments": [{"start": 0.0, "end": 1.0, "text": "mujhe kal se bukhar hai"}],
            },
        }
        dummy_audio = np.ones(16000, dtype=np.float32) * 0.1
        result = transcribe_audio(dummy_audio, 16000, model_bundle=fake_bundle, language="hi")
        self.assertEqual(result["detected_language"], "hi")
        self.assertTrue(result["target_language_supported"])
        self.assertEqual(result["text"], "mujhe kal se bukhar hai")
        self.assertEqual(result["model_key"], "faster_whisper_small")


if __name__ == "__main__":
    unittest.main()
