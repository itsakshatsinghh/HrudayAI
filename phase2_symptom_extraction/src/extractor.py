"""Open-source Hindi & English symptom, duration, severity, temporal, and certainty extractor.

Accepts raw text or Phase 1 transcription JSON outputs and produces structured
symptom JSON without invoking any paid APIs or disease prediction models.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from phase2_symptom_extraction.src.config_loader import (
    load_extraction_rules,
    load_symptom_lexicon,
)
from phase2_symptom_extraction.src.normalizer import SymptomNormalizer, normalize_text

SUPPORTED_LANGUAGES = {"hi", "en"}


class SymptomExtractionError(Exception):
    """Raised when input payload or transcript cannot be processed."""


@dataclass(frozen=True)
class _TokenSpan:
    text: str
    norm: str
    start: int
    end: int


@dataclass
class _MatchedSymptom:
    canonical_name: str
    original_term: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int


def _tokenize_with_spans(text: str) -> list[_TokenSpan]:
    """Tokenize preserving Devanagari matras and internal apostrophes (e.g. don't)."""
    pattern = re.compile(r"[^\s,;.\u0964!?()\[\]\"]+")
    tokens: list[_TokenSpan] = []
    for match in pattern.finditer(text):
        raw = match.group(0)
        start = match.start()
        end = match.end()
        while raw.startswith("'") and len(raw) > 1:
            raw = raw[1:]
            start += 1
        while raw.endswith("'") and len(raw) > 1:
            raw = raw[:-1]
            end -= 1
        if raw == "'":
            continue
        tokens.append(
            _TokenSpan(
                text=raw,
                norm=normalize_text(raw),
                start=start,
                end=end,
            )
        )
    return tokens


def _split_sentence_units(text: str) -> list[str]:
    """Split text into sentence-level units while keeping terminal '?' attached for question detection."""
    raw_parts = re.split(r"([.!\u0964?\n]+)", text)
    sentences: list[str] = []
    buf = ""
    for part in raw_parts:
        if not part:
            continue
        if re.fullmatch(r"[.!\u0964?\n]+", part):
            if "?" in part:
                buf += "?"
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf += part
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def _split_major_segments(text: str, separator_regex: str) -> list[str]:
    parts = re.split(separator_regex, text)
    return [part.strip() for part in parts if part and part.strip()]


def _split_conjunction_subclauses(segment: str, conjunction_terms: list[str]) -> list[str]:
    """Split a comma/sentence segment at coordinating conjunctions for attribute scoping."""
    sorted_terms = sorted(conjunction_terms, key=len, reverse=True)
    escaped = [re.escape(term) for term in sorted_terms]
    if not escaped:
        return [segment]
    conj_re = re.compile(rf"(?<!\S)(?:{'|'.join(escaped)})(?!\S)", flags=re.IGNORECASE)
    parts = [p.strip() for p in conj_re.split(segment) if p and p.strip()]
    return parts or [segment]


def _contains_phrase(norm_text: str, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    return bool(re.search(rf"(?<!\S){re.escape(phrase_norm)}(?!\S)", norm_text))


def _find_symptom_matches(
    subclause: str,
    normalizer: SymptomNormalizer,
    language: str | None,
) -> list[_MatchedSymptom]:
    """Find non-overlapping symptom matches within a subclause using longest-first n-grams."""
    tokens = _tokenize_with_spans(subclause)
    if not tokens:
        return []

    occupied = [False] * len(tokens)
    matches: list[_MatchedSymptom] = []
    max_n = min(normalizer.max_ngram_tokens, len(tokens))

    for n in range(max_n, 0, -1):
        for i in range(0, len(tokens) - n + 1):
            if any(occupied[i : i + n]):
                continue
            candidate_norm = " ".join(tok.norm for tok in tokens[i : i + n])
            hit = normalizer.match_phrase(candidate_norm, language=language)
            if hit is None:
                continue
            canonical_name, _score, _match_type = hit
            char_start = tokens[i].start
            char_end = tokens[i + n - 1].end
            original_term = subclause[char_start:char_end]
            for idx in range(i, i + n):
                occupied[idx] = True
            matches.append(
                _MatchedSymptom(
                    canonical_name=canonical_name,
                    original_term=original_term,
                    token_start=i,
                    token_end=i + n,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

    matches.sort(key=lambda m: m.token_start)
    return matches


def _extract_duration(clause_text: str, number_words: dict[str, int]) -> str | None:
    """Extract and normalize explicit duration or historical time expressions; None when absent."""
    text = normalize_text(clause_text)

    # Ongoing relative durations (since/from/से -> duration from past into present)
    ongoing_patterns: list[tuple[str, str]] = [
        (r"\b(?:since|from)\s+yesterday\b", "1 day"),
        (r"(?<!\S)कल\s+से(?!\S)", "1 day"),
        (r"(?<!\S)परसों\s+से(?!\S)", "2 days"),
        (r"\b(?:since\s+today|since\s+this\s+morning|this\s+morning)\b", "today"),
        (r"(?<!\S)(?:आज\s+से|आज\s+सुबह\s+से|सुबह\s+से)(?!\S)", "today"),
        (r"\b(?:for\s+)?(?:a\s+)?couple\s+of\s+days\b", "2 days"),
        (r"\b(?:for\s+)?a\s+week\b|\bsince\s+last\s+week\b", "1 week"),
        (r"(?<!\S)(?:एक\s+हफ्ते\s+से|एक\s+सप्ताह\s+से|पिछले\s+हफ्ते\s+से)(?!\S)", "1 week"),
        (r"\b(?:for\s+)?a\s+month\b|\bsince\s+last\s+month\b", "1 month"),
        (r"(?<!\S)(?:एक\s+महीने\s+से|पिछले\s+महीने\s+से)(?!\S)", "1 month"),
    ]
    for pattern, normalized_value in ongoing_patterns:
        if re.search(pattern, text):
            return normalized_value

    # Numeric or word-number + unit patterns (English & Hindi)
    num_word_alt = "|".join(re.escape(k) for k in number_words.keys())
    unit_map = {
        "hour": "hour",
        "hours": "hour",
        "hr": "hour",
        "hrs": "hour",
        "घंटा": "hour",
        "घंटे": "hour",
        "घंटों": "hour",
        "day": "day",
        "days": "day",
        "दिन": "day",
        "दिनों": "day",
        "week": "week",
        "weeks": "week",
        "हफ्ता": "week",
        "हफ्ते": "week",
        "हफ्तों": "week",
        "सप्ताह": "week",
        "month": "month",
        "months": "month",
        "महीना": "month",
        "महीने": "month",
        "महीनों": "month",
        "year": "year",
        "years": "year",
        "साल": "year",
        "वर्ष": "year",
    }
    unit_alt = "|".join(re.escape(u) for u in unit_map.keys())
    num_unit_re = re.compile(
        rf"(?:^|\s)(?P<num>\d+|{num_word_alt})\s+(?P<unit>{unit_alt})(?:\s+(?:से|पहले|ago))?(?=\s|$)"
    )
    match = num_unit_re.search(text)
    if match:
        raw_num = match.group("num")
        raw_unit = match.group("unit")
        count = int(raw_num) if raw_num.isdigit() else number_words.get(raw_num)
        canonical_unit = unit_map.get(raw_unit)
        if count is not None and canonical_unit is not None:
            suffix = "" if count == 1 else "s"
            return f"{count} {canonical_unit}{suffix}"

    # Historical point-in-time phrases (when not paired with since/से)
    historical_anchors: list[tuple[str, str]] = [
        (r"\blast\s+week\b", "last week"),
        (r"(?<!\S)पिछले\s+(?:हफ्ते|सप्ताह)(?!\S)", "last week"),
        (r"\blast\s+month\b", "last month"),
        (r"(?<!\S)पिछले\s+महीने(?!\S)", "last month"),
        (r"\blast\s+night\b", "last night"),
        (r"\byesterday\b", "yesterday"),
        (r"(?<!\S)कल(?!\S)", "yesterday"),
    ]
    for pattern, label in historical_anchors:
        if re.search(pattern, text):
            return label

    return None


def _extract_severity(
    tokens: list[_TokenSpan],
    match: _MatchedSymptom,
    severity_rules: dict[str, list[str]],
) -> str | None:
    """Look for an explicit severity modifier immediately adjacent to a symptom match."""
    left_start = max(0, match.token_start - 3)
    right_end = min(len(tokens), match.token_end + 2)
    left_window = " ".join(tok.norm for tok in tokens[left_start : match.token_start])
    right_window = " ".join(tok.norm for tok in tokens[match.token_end : right_end])
    matched_norm = " ".join(tok.norm for tok in tokens[match.token_start : match.token_end])

    for level in ("severe", "moderate", "mild"):
        modifiers = sorted(severity_rules.get(level, []), key=len, reverse=True)
        for mod in modifiers:
            mod_norm = normalize_text(mod)
            if not mod_norm:
                continue
            mod_re = re.compile(rf"(?<!\S){re.escape(mod_norm)}(?!\S)")
            if mod_re.search(left_window) or mod_re.search(right_window) or mod_re.search(matched_norm):
                return level
    return None


def _detect_negation(
    tokens: list[_TokenSpan],
    match: _MatchedSymptom,
    neg_rules: dict[str, Any],
    inherited_negation: bool,
    is_ambiguous: bool,
) -> bool:
    """Detect whether a matched symptom is negated in its local context."""
    # Ambiguous statements ("not sure if I have fever or not", "पता नहीं बुखार है या नहीं")
    # must stay uncertain rather than being marked as confirmed negation.
    if is_ambiguous:
        return False

    pre_triggers = sorted(
        [normalize_text(t) for t in neg_rules.get("pre_triggers", [])],
        key=len,
        reverse=True,
    )
    post_triggers = sorted(
        [normalize_text(t) for t in neg_rules.get("post_triggers", [])],
        key=len,
        reverse=True,
    )
    scope_breakers = {normalize_text(t) for t in neg_rules.get("scope_breakers", [])}
    conditional_verbs = {normalize_text(t) for t in neg_rules.get("conditional_verb_breakers", [])}
    neg_auxiliaries = {normalize_text(t) for t in neg_rules.get("negated_auxiliaries", [])}
    pre_win = int(neg_rules.get("pre_window_tokens", 5))
    post_win = int(neg_rules.get("post_window_tokens", 4))

    left_tokens: list[str] = []
    for idx in range(match.token_start - 1, max(-1, match.token_start - pre_win - 1), -1):
        tok_norm = tokens[idx].norm
        if tok_norm in scope_breakers:
            break
        if tok_norm in conditional_verbs:
            prev_tok = tokens[idx - 1].norm if idx > 0 else ""
            # Allow "don't have", "do not have", "didn't have", "haven't had" to cross verb
            if prev_tok not in neg_auxiliaries:
                break
        left_tokens.insert(0, tok_norm)

    left_phrase = " ".join(left_tokens)
    for trigger in pre_triggers:
        if re.search(rf"(?<!\S){re.escape(trigger)}(?!\S)", left_phrase):
            return True

    right_tokens = [
        tokens[idx].norm
        for idx in range(match.token_end, min(len(tokens), match.token_end + post_win))
    ]
    right_phrase = " ".join(right_tokens)
    for trigger in post_triggers:
        if re.search(rf"(?<!\S){re.escape(trigger)}(?!\S)", right_phrase):
            return True

    return inherited_negation


def _detect_temporal_status(subclause: str, temporal_rules: dict[str, list[str]]) -> str:
    """Classify symptom temporal_status as 'historical' or 'current'."""
    norm = normalize_text(subclause)

    # Ongoing duration anchors ("since yesterday", "कल से", "for 3 days", "तीन दिन से")
    # indicate a symptom active up to the present unless overridden by explicit past-resolved cues.
    has_ongoing_since = bool(
        re.search(
            r"\b(?:since|for)\b|(?<!\S)(?:कल|परसों|सुबह|हफ्ते|सप्ताह|महीने|दिन|दिनों|घंटे|घंटों)\s+से(?!\S)",
            norm,
        )
    )

    current_cues = temporal_rules.get("current_cues", [])
    historical_cues = temporal_rules.get("historical_cues", [])

    has_explicit_now = any(
        _contains_phrase(norm, cue)
        for cue in ("now", "right now", "currently", "today", "still", "अब", "अभी", "अभी भी", "अब भी", "आज", "फिलहाल")
    )
    if has_explicit_now:
        return "current"

    # Ongoing "have had ... for/since ..." is current
    if re.search(r"\b(?:have|has)\s+had\b", norm) and has_ongoing_since:
        return "current"

    # Past-time anchor words without "since"/"से" (e.g. "yesterday", "कल", "परसों", "2 days ago")
    has_past_without_since = bool(
        not has_ongoing_since
        and (
            re.search(r"\byesterday\b|\bago\b", norm)
            or re.search(r"(?<!\S)(?:कल|परसों)(?!\S)", norm)
            or re.search(r"(?:दिन|हफ्ते|सप्ताह|महीने)\s+पहले", norm)
        )
    )
    has_historical_cue = has_past_without_since or any(_contains_phrase(norm, cue) for cue in historical_cues)
    has_present_copula = any(_contains_phrase(norm, cue) for cue in current_cues)

    if has_historical_cue and not has_explicit_now:
        return "historical"
    if has_ongoing_since and not has_historical_cue:
        return "current"
    if has_present_copula:
        return "current"
    return "current"


def _detect_certainty(
    subclause: str,
    sentence_is_question: bool,
    uncertainty_rules: dict[str, list[str]],
) -> tuple[str, bool]:
    """Return (certainty, is_ambiguous) where certainty is 'confirmed' or 'uncertain'."""
    norm = normalize_text(subclause)
    ambiguity_phrases = uncertainty_rules.get("ambiguity_phrases", [])
    hedge_triggers = uncertainty_rules.get("hedge_triggers", [])
    question_cues = uncertainty_rules.get("question_cues", [])

    is_ambiguous = any(_contains_phrase(norm, phrase) for phrase in ambiguity_phrases)
    if is_ambiguous:
        return "uncertain", True

    if sentence_is_question or any(_contains_phrase(norm, q) for q in question_cues):
        return "uncertain", False

    if any(_contains_phrase(norm, hedge) for hedge in hedge_triggers):
        return "uncertain", False

    return "confirmed", False


def _subclause_has_positive_hindi_verb(subclause: str) -> bool:
    norm = normalize_text(subclause)
    if re.search(r"नहीं\s+(?:है|हैं|था|थी|थे|हो\s+रहा|हो\s+रही)", norm):
        return False
    return bool(
        re.search(
            r"(?:^|\s)(?:है|हैं|था|थी|थे|हो\s+रहा\s+है|हो\s+रही\s+है|लग\s+रहा\s+है|लग\s+रही\s+है)\s*$",
            norm,
        )
    )


def _segment_starts_with_pre_negation(subclause: str, neg_rules: dict[str, Any]) -> bool:
    tokens = _tokenize_with_spans(subclause)
    if not tokens:
        return False
    pre_triggers = {normalize_text(t) for t in neg_rules.get("pre_triggers", [])}
    head_two = " ".join(tok.norm for tok in tokens[:2])
    return tokens[0].norm in pre_triggers or head_two in pre_triggers


def _is_elliptical_followup(
    subclause: str,
    ellipsis_rules: dict[str, list[str]],
    neg_rules: dict[str, Any],
    temporal_rules: dict[str, list[str]],
) -> bool:
    """Detect pronoun or temporal/negation follow-up clauses referring to a prior single symptom."""
    norm = normalize_text(subclause)
    pronouns = ellipsis_rules.get("pronouns", [])
    status_cues = ellipsis_rules.get("status_cues", [])
    has_pronoun = any(_contains_phrase(norm, p) for p in pronouns)
    has_status_cue = any(_contains_phrase(norm, c) for c in status_cues)
    has_post_neg = any(_contains_phrase(norm, p) for p in neg_rules.get("post_triggers", []))
    has_temporal = any(
        _contains_phrase(norm, t)
        for t in ("now", "today", "earlier", "last week", "अब", "अभी", "आज", "पहले")
    )
    has_hindi_state_verb = any(
        _contains_phrase(norm, v) for v in ("नहीं है", "नहीं हैं", "ठीक है", "कम है", "है", "था")
    )

    if has_pronoun and (has_status_cue or has_post_neg or has_temporal):
        return True
    # Hindi frequently drops subject pronouns: "लेकिन अब नहीं है" / "अब ठीक है"
    if has_temporal and (has_post_neg or has_hindi_state_verb):
        return True
    return False


def _detect_language_fallback(text: str) -> str:
    """Infer 'hi' if Devanagari script appears, otherwise 'en'."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def extract_symptoms(
    transcript: str,
    language: str | None = None,
    normalizer: SymptomNormalizer | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract structured symptoms, duration, severity, negation, temporal status, and certainty."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise SymptomExtractionError("Transcript must be a non-empty string.")

    loaded_rules = rules or load_extraction_rules()
    if normalizer is None:
        lexicon = load_symptom_lexicon()
        normalizer = SymptomNormalizer(lexicon=lexicon, rules=loaded_rules)

    resolved_lang = (language or "").strip().lower()
    if resolved_lang in {"", "auto", "unknown"}:
        resolved_lang = _detect_language_fallback(transcript)
    elif resolved_lang not in SUPPORTED_LANGUAGES:
        raise SymptomExtractionError(
            f"Unsupported language '{language}'. Phase 2 supports only: {sorted(SUPPORTED_LANGUAGES)}."
        )

    conj_terms: list[str] = list(loaded_rules.get("conjunction_split_terms", []))
    neg_rules: dict[str, Any] = loaded_rules.get("negation", {})
    temporal_rules: dict[str, list[str]] = loaded_rules.get("temporal", {})
    uncertainty_rules: dict[str, list[str]] = loaded_rules.get("uncertainty", {})
    ellipsis_rules: dict[str, list[str]] = loaded_rules.get("ellipsis", {})
    severity_rules: dict[str, list[str]] = loaded_rules.get("severity", {})
    number_words: dict[str, int] = {
        normalize_text(k): int(v) for k, v in loaded_rules.get("number_words", {}).items()
    }

    extracted_by_key: dict[tuple[str, str, bool, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str, bool, str]] = []

    for sentence in _split_sentence_units(transcript):
        sentence_is_question = sentence.rstrip().endswith("?")
        clean_sentence = sentence.rstrip("?").strip()
        if not clean_sentence:
            continue

        major_segments = _split_major_segments(clean_sentence, r"[,;]+")
        last_unambiguous_match: _MatchedSymptom | None = None

        for segment in major_segments:
            norm_seg = normalize_text(segment)
            trailing_hindi_neg = bool(
                re.search(
                    r"(?:नहीं\s+है|नहीं\s+हैं|नहीं\s+था|नहीं\s+थी|नहीं\s+हो\s+रहा|नहीं\s+हो\s+रही)\s*$",
                    norm_seg,
                )
            )
            subclauses = _split_conjunction_subclauses(segment, conj_terms)

            carry_negation = False
            for idx, subclause in enumerate(subclauses):
                sub_tokens = _tokenize_with_spans(subclause)
                matches = _find_symptom_matches(subclause, normalizer, language=resolved_lang)
                certainty, is_ambiguous = _detect_certainty(
                    subclause,
                    sentence_is_question=sentence_is_question,
                    uncertainty_rules=uncertainty_rules,
                )
                temporal_status = _detect_temporal_status(subclause, temporal_rules)
                sub_duration = _extract_duration(subclause, number_words)

                if not matches:
                    if (
                        last_unambiguous_match is not None
                        and _is_elliptical_followup(subclause, ellipsis_rules, neg_rules, temporal_rules)
                    ):
                        # Locate pronoun if present; otherwise place synthetic span at start so
                        # right-window post-negation triggers ("अब नहीं है") inspect the full clause.
                        pronoun_set = {normalize_text(p) for p in ellipsis_rules.get("pronouns", [])}
                        syn_start = 0
                        syn_end = 0
                        for t_idx, tok in enumerate(sub_tokens):
                            if tok.norm in pronoun_set:
                                syn_start = t_idx
                                syn_end = t_idx + 1
                                break
                        synthetic_match = _MatchedSymptom(
                            canonical_name=last_unambiguous_match.canonical_name,
                            original_term=last_unambiguous_match.original_term,
                            token_start=syn_start,
                            token_end=syn_end,
                            char_start=0,
                            char_end=0,
                        )
                        negated = _detect_negation(
                            tokens=sub_tokens,
                            match=synthetic_match,
                            neg_rules=neg_rules,
                            inherited_negation=False,
                            is_ambiguous=is_ambiguous,
                        )
                        severity = _extract_severity(sub_tokens, synthetic_match, severity_rules)
                        item = {
                            "canonical_name": synthetic_match.canonical_name,
                            "original_term": synthetic_match.original_term,
                            "duration": sub_duration,
                            "severity": severity,
                            "negated": negated,
                            "temporal_status": temporal_status,
                            "certainty": certainty,
                        }
                        dedup_key = (
                            item["canonical_name"],
                            item["temporal_status"],
                            item["negated"],
                            item["certainty"],
                        )
                        if dedup_key not in extracted_by_key:
                            extracted_by_key[dedup_key] = item
                            ordered_keys.append(dedup_key)
                        continue

                    sub_norm = normalize_text(subclause)
                    if any(b in sub_norm for b in ("have ", "has ", "है", "हैं")):
                        carry_negation = False
                    continue

                if len(matches) == 1:
                    last_unambiguous_match = matches[0]
                else:
                    last_unambiguous_match = None

                starts_negated = _segment_starts_with_pre_negation(subclause, neg_rules)
                is_or_continuation = bool(
                    idx > 0
                    and re.search(r"\b(?:or|nor)\b|(?<!\S)(?:या|न)(?!\S)", norm_seg)
                )
                applies_trailing_hindi = (
                    trailing_hindi_neg
                    and not is_ambiguous
                    and not _subclause_has_positive_hindi_verb(subclause)
                )
                inherited_neg = (carry_negation and is_or_continuation) or applies_trailing_hindi

                for match in matches:
                    negated = _detect_negation(
                        tokens=sub_tokens,
                        match=match,
                        neg_rules=neg_rules,
                        inherited_negation=inherited_neg,
                        is_ambiguous=is_ambiguous,
                    )
                    severity = _extract_severity(sub_tokens, match, severity_rules)

                    item = {
                        "canonical_name": match.canonical_name,
                        "original_term": match.original_term,
                        "duration": sub_duration,
                        "severity": severity,
                        "negated": negated,
                        "temporal_status": temporal_status,
                        "certainty": certainty,
                    }

                    dedup_key = (
                        item["canonical_name"],
                        item["temporal_status"],
                        item["negated"],
                        item["certainty"],
                    )
                    existing = extracted_by_key.get(dedup_key)
                    if existing is None:
                        extracted_by_key[dedup_key] = item
                        ordered_keys.append(dedup_key)
                    else:
                        if existing["duration"] is None and item["duration"] is not None:
                            existing["duration"] = item["duration"]
                        if existing["severity"] is None and item["severity"] is not None:
                            existing["severity"] = item["severity"]

                carry_negation = starts_negated or (carry_negation and is_or_continuation)

    return {
        "language": resolved_lang,
        "original_transcript": transcript,
        "symptoms": [extracted_by_key[k] for k in ordered_keys],
    }


def _coerce_phase1_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize single-dict, keyed-dict, or list Phase 1 JSON payloads into records."""
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_coerce_phase1_records(item))
        return records

    if isinstance(payload, dict):
        if "text" in payload or "transcript" in payload or "original_transcript" in payload:
            return [payload]
        for container_key in ("results", "transcriptions", "items"):
            if container_key in payload:
                return _coerce_phase1_records(payload[container_key])
        nested: list[dict[str, Any]] = []
        for val in payload.values():
            if isinstance(val, (dict, list)):
                nested.extend(_coerce_phase1_records(val))
        if nested:
            return nested

    raise SymptomExtractionError(
        "Unrecognized Phase 1 JSON format. Expected transcript object(s) with 'text' or 'transcript'."
    )


def extract_from_phase1_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    normalizer: SymptomNormalizer | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract structured symptoms from a Phase 1 transcription dict or list of dicts."""
    loaded_rules = rules or load_extraction_rules()
    if normalizer is None:
        lexicon = load_symptom_lexicon()
        normalizer = SymptomNormalizer(lexicon=lexicon, rules=loaded_rules)

    records = _coerce_phase1_records(payload)
    results: list[dict[str, Any]] = []
    for rec in records:
        text = rec.get("text") or rec.get("transcript") or rec.get("original_transcript")
        lang = rec.get("detected_language") or rec.get("language")
        extracted = extract_symptoms(
            transcript=str(text or ""),
            language=str(lang) if lang is not None else None,
            normalizer=normalizer,
            rules=loaded_rules,
        )
        results.append(extracted)

    if isinstance(payload, dict) and (
        "text" in payload or "transcript" in payload or "original_transcript" in payload
    ):
        return results[0]
    return results


def extract_from_phase1_json_file(
    json_path: str | Path,
    normalizer: SymptomNormalizer | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Load Phase 1 JSON output from disk and run structured symptom extraction."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 JSON file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SymptomExtractionError(f"Invalid JSON in {path}: {exc}") from exc
    return extract_from_phase1_payload(payload, normalizer=normalizer, rules=rules)
