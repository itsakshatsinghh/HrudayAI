"""Phase 2 open-source Hindi & English symptom extraction module."""

from phase2_symptom_extraction.src.config_loader import (
    DEFAULT_LEXICON_PATH,
    DEFAULT_RULES_PATH,
    ExtractionConfigError,
    load_extraction_rules,
    load_symptom_lexicon,
)
from phase2_symptom_extraction.src.extractor import (
    SymptomExtractionError,
    extract_from_phase1_json_file,
    extract_from_phase1_payload,
    extract_symptoms,
)
from phase2_symptom_extraction.src.normalizer import (
    SymptomNormalizer,
    build_normalizer,
    normalize_text,
)

__all__ = [
    "DEFAULT_LEXICON_PATH",
    "DEFAULT_RULES_PATH",
    "ExtractionConfigError",
    "SymptomExtractionError",
    "SymptomNormalizer",
    "build_normalizer",
    "extract_from_phase1_json_file",
    "extract_from_phase1_payload",
    "extract_symptoms",
    "load_extraction_rules",
    "load_symptom_lexicon",
    "normalize_text",
]
