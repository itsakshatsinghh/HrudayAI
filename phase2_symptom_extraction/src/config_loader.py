"""Configuration loading utilities for Phase 2 symptom extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_LEXICON_PATH = CONFIG_DIR / "symptom_lexicon.json"
DEFAULT_RULES_PATH = CONFIG_DIR / "extraction_rules.json"


class ExtractionConfigError(Exception):
    """Raised when symptom extraction configuration files are missing or invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExtractionConfigError(f"Configuration file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionConfigError(f"Invalid JSON in configuration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionConfigError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def load_symptom_lexicon(lexicon_path: str | Path | None = None) -> dict[str, Any]:
    """Load the Hindi-English symptom normalization lexicon."""
    path = Path(lexicon_path) if lexicon_path is not None else DEFAULT_LEXICON_PATH
    data = _read_json(path)
    symptoms = data.get("symptoms")
    if not isinstance(symptoms, dict) or not symptoms:
        raise ExtractionConfigError(f"Lexicon {path} must define a non-empty 'symptoms' mapping.")
    return data


def load_extraction_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    """Load clause splitting, negation, severity, and matching rules."""
    path = Path(rules_path) if rules_path is not None else DEFAULT_RULES_PATH
    data = _read_json(path)
    for required_key in ("matching", "negation", "severity"):
        if required_key not in data or not isinstance(data[required_key], dict):
            raise ExtractionConfigError(f"Rules file {path} is missing '{required_key}' section.")
    return data
