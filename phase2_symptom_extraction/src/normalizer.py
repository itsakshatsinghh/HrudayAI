"""Hindi and English symptom phrase normalization utilities.

Uses configurable exact + optional fuzzy matching (rapidfuzz if installed,
difflib fallback otherwise) with zero external API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import importlib.util
from pathlib import Path
import re
import unicodedata
from typing import Any

from phase2_symptom_extraction.src.config_loader import (
    load_extraction_rules,
    load_symptom_lexicon,
)

_HAS_RAPIDFUZZ = importlib.util.find_spec("rapidfuzz") is not None


def normalize_text(text: str) -> str:
    """Normalize whitespace, Unicode NFC, and casing while preserving Devanagari."""
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass(frozen=True)
class LexiconEntry:
    """Single surface form mapped to a canonical symptom label."""

    surface: str
    canonical_name: str
    language: str
    token_count: int


class SymptomNormalizer:
    """Normalizes Hindi and English surface phrases into canonical symptom labels."""

    def __init__(
        self,
        lexicon: dict[str, Any],
        rules: dict[str, Any],
    ) -> None:
        matching_cfg = rules.get("matching", {})
        self.enable_fuzzy: bool = bool(matching_cfg.get("enable_fuzzy", True))
        self.fuzzy_score_cutoff: float = float(matching_cfg.get("fuzzy_score_cutoff", 88.0))
        self.min_fuzzy_term_length: int = int(matching_cfg.get("min_fuzzy_term_length", 5))
        self.max_ngram_tokens: int = int(matching_cfg.get("max_ngram_tokens", 5))

        self.entries: list[LexiconEntry] = []
        self.exact_lookup: dict[str, LexiconEntry] = {}

        symptoms_map: dict[str, dict[str, list[str]]] = lexicon.get("symptoms", {})
        for canonical_name, lang_map in symptoms_map.items():
            for lang_code, terms in lang_map.items():
                for term in terms:
                    cleaned = normalize_text(term)
                    if not cleaned:
                        continue
                    entry = LexiconEntry(
                        surface=cleaned,
                        canonical_name=canonical_name,
                        language=lang_code,
                        token_count=len(cleaned.split()),
                    )
                    self.entries.append(entry)
                    # Prefer longest or first registered canonical mapping on exact surface
                    self.exact_lookup.setdefault(cleaned, entry)

        # Sort longest phrases first so multi-word phrases win over sub-phrases
        self.entries.sort(key=lambda e: (e.token_count, len(e.surface)), reverse=True)
        self.fuzzy_candidates_by_len: dict[int, list[LexiconEntry]] = {}
        for entry in self.entries:
            if len(entry.surface) >= self.min_fuzzy_term_length:
                self.fuzzy_candidates_by_len.setdefault(entry.token_count, []).append(entry)

    def match_phrase(
        self,
        phrase: str,
        language: str | None = None,
    ) -> tuple[str, float, str] | None:
        """Return (canonical_name, score, match_type) for an exact or fuzzy phrase match."""
        cleaned = normalize_text(phrase)
        if not cleaned:
            return None

        exact = self.exact_lookup.get(cleaned)
        if exact is not None:
            return exact.canonical_name, 100.0, "exact"

        if not self.enable_fuzzy or len(cleaned) < self.min_fuzzy_term_length:
            return None

        token_count = len(cleaned.split())
        candidates = self.fuzzy_candidates_by_len.get(token_count, [])
        if not candidates:
            return None

        # Restrict to matching language when known, falling back to all if empty
        if language in {"hi", "en"}:
            lang_filtered = [c for c in candidates if c.language == language]
            if lang_filtered:
                candidates = lang_filtered

        best_entry: LexiconEntry | None = None
        best_score = 0.0

        if _HAS_RAPIDFUZZ:
            from rapidfuzz import fuzz

            for candidate in candidates:
                score = float(fuzz.ratio(cleaned, candidate.surface))
                if score > best_score:
                    best_score = score
                    best_entry = candidate
        else:
            for candidate in candidates:
                score = SequenceMatcher(None, cleaned, candidate.surface).ratio() * 100.0
                if score > best_score:
                    best_score = score
                    best_entry = candidate

        if best_entry is not None and best_score >= self.fuzzy_score_cutoff:
            return best_entry.canonical_name, round(best_score, 2), "fuzzy"

        return None


def build_normalizer(
    lexicon_path: str | Path | None = None,
    rules_path: str | Path | None = None,
) -> SymptomNormalizer:
    """Convenience factory loading lexicon and extraction rules."""
    lexicon = load_symptom_lexicon(lexicon_path)
    rules = load_extraction_rules(rules_path)
    return SymptomNormalizer(lexicon=lexicon, rules=rules)
