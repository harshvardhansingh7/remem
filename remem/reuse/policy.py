from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from remem.models.execution_context import ExecutionContext

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(
    r"(?<!\w)(?:[$€£₹]\s*)?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:usd|eur|gbp|inr))?(?!\w)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_QUOTED_RE = re.compile(r"(?:\"([^\"]+)\"|'([^']+)')")
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_-]*|[A-Za-z]+[-_][A-Za-z0-9_-]+|[A-Za-z]+\d+[A-Za-z0-9_-]*)\b"
)
_PROPER_NAME_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][a-z]{2,})\b")
_TEMPORAL_RE = re.compile(
    r"\b(?:today|current|currently|latest|now|recent|this\s+week|yesterday|tomorrow|"
    r"q[1-4]|quarter|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|\d{4})\b",
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:[?.!,;]|$)", re.IGNORECASE)

_LEADING_WORDS = {
    "A",
    "An",
    "Calculate",
    "Classify",
    "Compare",
    "Create",
    "Describe",
    "Disable",
    "Draft",
    "Enable",
    "Explain",
    "Extract",
    "Generate",
    "Give",
    "How",
    "Include",
    "List",
    "Please",
    "Return",
    "Rewrite",
    "Summarize",
    "Tell",
    "The",
    "Translate",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "Write",
}


def _normalize(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(_TOKEN_RE.findall(value.casefold()))
    return str(value).casefold()


def _as_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {_normalize(item) for item in value if _normalize(item)}
    normalized = _normalize(value)
    return {normalized} if normalized else set()


def _metadata_override(context: ExecutionContext, *keys: str) -> Any:
    for key in keys:
        if key in context.metadata:
            return context.metadata[key]
    return None


def _detect_operation(text: str) -> str:
    normalized = _normalize(text)
    patterns = [
        ("summarization", r"\bsummari[sz](?:e|ation)\b"),
        ("translation", r"\btranslat(?:e|ion)\b"),
        ("classification", r"\bclassif(?:y|ication)|\bcategori[sz]e\b|\bsentiment\b"),
        ("comparison", r"\bcompare\b|\bcomparison\b|\bdifference\b|\bversus\b|\bvs\b"),
        ("extraction", r"\bextract\b|\bidentify\b"),
        ("rewriting", r"\brewrite\b|\brephrase\b|\bparaphrase\b"),
        ("coding", r"\bcode\b|\bfunction\b|\bpython\b|\bjava\b|\bsql\b"),
        ("calculation", r"\bcalculate\b|\bcompute\b|\bhow many\b|\bsum\b"),
        ("explanation", r"\bexplain\b|\bwhy\b|\bhow does\b"),
        ("generation", r"\bgenerate\b|\bcreate\b|\bdraft\b|\bwrite\b"),
    ]
    for operation, pattern in patterns:
        if re.search(pattern, normalized):
            return operation
    if re.search(
        r"\b(?:what|when|where|which|who|how|is|are|does|do|can)\b", normalized
    ):
        return "question_answering"
    return "unknown"


def _question_focus(text: str) -> str | None:
    normalized = _normalize(text)
    match = re.match(
        r"(?:please\s+)?(how many|how much|what|when|where|which|who|why|how)\b",
        normalized,
    )
    return match.group(1) if match else None


def _critical_entities(context: ExecutionContext, text: str) -> set[str]:
    explicit = _metadata_override(context, "entities", "critical_entities")
    if explicit is not None:
        return _as_values(explicit)
    values = {
        _normalize(match.group(0).rstrip(".,;:!?")) for match in _URL_RE.finditer(text)
    }
    values.update(_normalize(match.group(0)) for match in _EMAIL_RE.finditer(text))
    values.update(
        _normalize(next(group for group in match.groups() if group is not None))
        for match in _QUOTED_RE.finditer(text)
    )
    values.update(_normalize(match.group(0)) for match in _IDENTIFIER_RE.finditer(text))
    values.update(
        _normalize(match.group(0))
        for match in _PROPER_NAME_RE.finditer(text)
        if match.group(0) not in _LEADING_WORDS
    )
    return {value for value in values if value}


def _numeric_values(context: ExecutionContext, text: str) -> set[str]:
    explicit = _metadata_override(context, "critical_values", "numbers")
    if explicit is not None:
        return _as_values(explicit)
    return {_normalize(match.group(0)) for match in _NUMBER_RE.finditer(text)}


def _temporal_values(context: ExecutionContext, text: str) -> set[str]:
    explicit = _metadata_override(context, "temporal_scope", "as_of")
    if explicit is not None:
        return _as_values(explicit)
    return {_normalize(match.group(0)) for match in _TEMPORAL_RE.finditer(text)}


def _polarity(text: str) -> dict[str, str]:
    normalized = _normalize(text)
    dimensions = {
        "enabled": (
            r"\benabl(?:e|es|ed|ing)\b|\bactivat(?:e|es|ed|ing)\b",
            r"\bdisabl(?:e|es|ed|ing)\b|\bdeactivat(?:e|es|ed|ing)\b",
        ),
        "included": (
            r"\binclud(?:e|es|ed|ing)\b|\bwith\b",
            r"\bexclud(?:e|es|ed|ing)\b|\bwithout\b",
        ),
        "change": (
            r"\bincreas(?:e|es|ed|ing)\b|\brais(?:e|es|ed|ing)\b",
            r"\bdecreas(?:e|es|ed|ing)\b|\blower(?:s|ed|ing)?\b",
        ),
        "ordering": (r"\bafter\b", r"\bbefore\b"),
        "comparison": (
            r"\bgreater than\b|\bmore than\b",
            r"\bless than\b|\bfewer than\b",
        ),
        "sentiment": (r"\bpositive\b", r"\bnegative\b"),
    }
    result: dict[str, str] = {}
    for name, (positive, negative) in dimensions.items():
        if re.search(positive, normalized):
            result[name] = "positive"
        elif re.search(negative, normalized):
            result[name] = "negative"
    if re.search(r"\b(?:not|no|never)\b", normalized):
        result["explicit_negation"] = "negative"
    return result


def _direction(text: str) -> tuple[str, str] | None:
    match = _DIRECTION_RE.search(text)
    if not match:
        return None
    return _normalize(match.group(1)), _normalize(match.group(2))


def _output_constraints(context: ExecutionContext, text: str) -> set[str]:
    explicit = _metadata_override(context, "output_format", "response_format")
    language = _metadata_override(context, "language", "response_language")
    if explicit is not None or language is not None:
        return _as_values(explicit) | {
            f"language:{value}" for value in _as_values(language)
        }
    normalized = _normalize(text)
    constraints = set()
    formats = {
        "json": r"\bjson\b",
        "csv": r"\bcsv\b",
        "markdown_table": r"\bmarkdown table\b",
        "table": r"\btable\b",
        "bullet_list": r"\bbullet(?:ed)? list\b|\bbullets\b",
        "python": r"\bpython\b",
        "java": r"\bjava\b",
        "code": r"\bcode\b",
        "short_answer": r"\bshort answer\b|\bbrief(?:ly)?\b",
        "detailed_answer": r"\bdetailed answer\b|\bin detail\b",
    }
    for name, pattern in formats.items():
        if re.search(pattern, normalized):
            constraints.add(name)
    language_match = re.search(
        r"\b(?:in|into)\s+(english|spanish|french|german|hindi|chinese|japanese)\b",
        normalized,
    )
    if language_match:
        constraints.add(f"language:{language_match.group(1)}")
    item_count = re.search(r"\b(?:return|give|list|provide)\s+(\d+)\b", normalized)
    if item_count:
        constraints.add(f"items:{item_count.group(1)}")
    return constraints


def _check(passed: bool, detail: str, *, applied: bool = True) -> dict[str, Any]:
    return {"passed": passed, "applied": applied, "detail": detail}


@dataclass
class ReusePolicy:
    """Configurable compatibility and lightweight reuse-safety policy."""

    retrieval_threshold: float = 0.80
    response_threshold: float = 0.95
    require_same_namespace: bool = True
    require_same_kb_version: bool = True
    require_same_prompt_version: bool = True
    require_same_model: bool = True
    required_metadata_keys: tuple[str, ...] = ()
    enable_required_metadata_check: bool = True
    enable_intent_check: bool = True
    enable_entity_check: bool = True
    enable_numeric_check: bool = True
    enable_temporal_check: bool = True
    enable_negation_check: bool = True
    enable_directional_check: bool = True
    enable_output_format_check: bool = True
    enable_freshness_check: bool = True
    enable_candidate_ambiguity_check: bool = True
    max_response_age_seconds: float | None = None
    max_retrieval_age_seconds: float | None = None
    minimum_response_score_margin: float | None = None
    query_metadata_key: str = "query"

    def __post_init__(self) -> None:
        for name in (
            "max_response_age_seconds",
            "max_retrieval_age_seconds",
            "minimum_response_score_margin",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when configured")

    def is_compatible(
        self, current: ExecutionContext, cached: ExecutionContext
    ) -> bool:
        """Evaluate existing isolation fields and configured metadata matches."""
        return self.compatibility_check(current, cached)[0]

    def compatibility_check(
        self, current: ExecutionContext, cached: ExecutionContext
    ) -> tuple[bool, list[str]]:
        failures = []
        fields = (
            ("namespace", self.require_same_namespace),
            ("kb_version", self.require_same_kb_version),
            ("prompt_version", self.require_same_prompt_version),
            ("model", self.require_same_model),
        )
        for field_name, required in fields:
            if required and getattr(current, field_name) != getattr(cached, field_name):
                failures.append(f"{field_name} differs")
        if self.enable_required_metadata_check:
            for key in self.required_metadata_keys:
                if key not in current.metadata or key not in cached.metadata:
                    failures.append(f"required metadata {key!r} is missing")
                elif current.metadata[key] != cached.metadata[key]:
                    failures.append(f"required metadata {key!r} differs")
        return not failures, failures

    def retrieval_freshness_check(
        self, current: ExecutionContext, created_at: datetime
    ) -> dict[str, Any]:
        return self._freshness_check(
            current, created_at, self.max_retrieval_age_seconds, "retrieval"
        )

    def response_checks(
        self,
        current: ExecutionContext,
        cached: ExecutionContext,
        created_at: datetime,
        score_margin: float | None,
    ) -> dict[str, dict[str, Any]]:
        text_check_names = (
            "intent",
            "critical_entities",
            "numeric_values",
            "temporal_scope",
            "negation",
            "direction",
            "output_format",
        )
        if not any(
            (
                self.enable_intent_check,
                self.enable_entity_check,
                self.enable_numeric_check,
                self.enable_temporal_check,
                self.enable_negation_check,
                self.enable_directional_check,
                self.enable_output_format_check,
            )
        ):
            text_checks = {
                name: _check(True, f"{name} check disabled", applied=False)
                for name in text_check_names
            }
            text_checks["freshness"] = self._freshness_check(
                current, created_at, self.max_response_age_seconds, "response"
            )
            text_checks["candidate_margin"] = self._margin_check(score_margin)
            return text_checks
        current_text = str(current.metadata.get(self.query_metadata_key, "")).strip()
        cached_text = str(cached.metadata.get(self.query_metadata_key, "")).strip()
        explicit_signal_keys = {
            "operation",
            "intent",
            "entities",
            "critical_entities",
            "critical_values",
            "numbers",
            "temporal_scope",
            "as_of",
            "output_format",
            "response_format",
            "language",
            "response_language",
        }
        has_explicit_signals = bool(
            explicit_signal_keys & (current.metadata.keys() | cached.metadata.keys())
        )
        if (not current_text or not cached_text) and not has_explicit_signals:
            unavailable = _check(
                True,
                f"query text unavailable in metadata[{self.query_metadata_key!r}]",
                applied=False,
            )
            text_checks = {
                "intent": dict(unavailable),
                "critical_entities": dict(unavailable),
                "numeric_values": dict(unavailable),
                "temporal_scope": dict(unavailable),
                "negation": dict(unavailable),
                "direction": dict(unavailable),
                "output_format": dict(unavailable),
            }
        else:
            text_checks = self._text_checks(current, cached, current_text, cached_text)
        text_checks["freshness"] = self._freshness_check(
            current, created_at, self.max_response_age_seconds, "response"
        )
        text_checks["candidate_margin"] = self._margin_check(score_margin)
        return text_checks

    def _text_checks(
        self,
        current: ExecutionContext,
        cached: ExecutionContext,
        current_text: str,
        cached_text: str,
    ) -> dict[str, dict[str, Any]]:
        current_operation = _metadata_override(current, "operation", "intent")
        cached_operation = _metadata_override(cached, "operation", "intent")
        current_operation = (
            _normalize(current_operation)
            if current_operation is not None
            else _detect_operation(current_text)
        )
        cached_operation = (
            _normalize(cached_operation)
            if cached_operation is not None
            else _detect_operation(cached_text)
        )
        current_focus, cached_focus = (
            _question_focus(current_text),
            _question_focus(cached_text),
        )
        intent_passed = (
            not self.enable_intent_check
            or current_operation == "unknown"
            or cached_operation == "unknown"
            or (
                current_operation == cached_operation
                and not (
                    current_focus and cached_focus and current_focus != cached_focus
                )
            )
        )

        current_entities = _critical_entities(current, current_text)
        cached_entities = _critical_entities(cached, cached_text)
        entity_passed = (
            not self.enable_entity_check or current_entities == cached_entities
        )

        current_numbers = _numeric_values(current, current_text)
        cached_numbers = _numeric_values(cached, cached_text)
        numeric_passed = (
            not self.enable_numeric_check or current_numbers == cached_numbers
        )

        current_temporal = _temporal_values(current, current_text)
        cached_temporal = _temporal_values(cached, cached_text)
        temporal_passed = (
            not self.enable_temporal_check or current_temporal == cached_temporal
        )

        current_polarity, cached_polarity = (
            _polarity(current_text),
            _polarity(cached_text),
        )
        polarity_conflicts = {
            name
            for name in current_polarity.keys() & cached_polarity.keys()
            if current_polarity[name] != cached_polarity[name]
        }
        if ("explicit_negation" in current_polarity) != (
            "explicit_negation" in cached_polarity
        ):
            polarity_conflicts.add("explicit_negation")
        negation_passed = not self.enable_negation_check or not polarity_conflicts

        current_direction, cached_direction = (
            _direction(current_text),
            _direction(cached_text),
        )
        reversed_direction = (
            current_direction is not None
            and cached_direction is not None
            and current_direction == tuple(reversed(cached_direction))
        )
        direction_passed = not self.enable_directional_check or not reversed_direction

        current_output = _output_constraints(current, current_text)
        cached_output = _output_constraints(cached, cached_text)
        output_passed = (
            not self.enable_output_format_check or current_output == cached_output
        )

        return {
            "intent": _check(
                intent_passed,
                f"operation/focus {cached_operation}/{cached_focus or '-'} -> {current_operation}/{current_focus or '-'}",
                applied=self.enable_intent_check,
            ),
            "critical_entities": _check(
                entity_passed,
                f"entities {sorted(cached_entities)} -> {sorted(current_entities)}",
                applied=self.enable_entity_check,
            ),
            "numeric_values": _check(
                numeric_passed,
                f"values {sorted(cached_numbers)} -> {sorted(current_numbers)}",
                applied=self.enable_numeric_check,
            ),
            "temporal_scope": _check(
                temporal_passed,
                f"temporal {sorted(cached_temporal)} -> {sorted(current_temporal)}",
                applied=self.enable_temporal_check,
            ),
            "negation": _check(
                negation_passed,
                f"polarity conflicts {sorted(polarity_conflicts)}",
                applied=self.enable_negation_check,
            ),
            "direction": _check(
                direction_passed,
                f"direction {cached_direction} -> {current_direction}",
                applied=self.enable_directional_check,
            ),
            "output_format": _check(
                output_passed,
                f"output {sorted(cached_output)} -> {sorted(current_output)}",
                applied=self.enable_output_format_check,
            ),
        }

    def _freshness_check(
        self,
        current: ExecutionContext,
        created_at: datetime,
        configured_max_age: float | None,
        tier: str,
    ) -> dict[str, Any]:
        if not self.enable_freshness_check:
            return _check(True, "freshness check disabled", applied=False)
        requested = _metadata_override(
            current, "freshness_requirement", "max_age_seconds"
        )
        requested_max_age: float | None = None
        if isinstance(requested, (int, float)) and not isinstance(requested, bool):
            requested_max_age = float(requested)
        limits = [
            value
            for value in (configured_max_age, requested_max_age)
            if value is not None
        ]
        if not limits:
            return _check(True, f"no {tier} age limit configured", applied=False)
        maximum_age = min(limits)
        normalized_created_at = (
            created_at.replace(tzinfo=timezone.utc)
            if created_at.tzinfo is None
            else created_at.astimezone(timezone.utc)
        )
        age_seconds = max(
            0.0, (datetime.now(timezone.utc) - normalized_created_at).total_seconds()
        )
        return _check(
            age_seconds <= maximum_age,
            f"{tier} age {age_seconds:.3f}s <= {maximum_age:.3f}s",
        )

    def _margin_check(self, score_margin: float | None) -> dict[str, Any]:
        if (
            not self.enable_candidate_ambiguity_check
            or self.minimum_response_score_margin is None
        ):
            return _check(True, "candidate margin check disabled", applied=False)
        if score_margin is None:
            return _check(True, "only one compatible candidate", applied=True)
        return _check(
            score_margin >= self.minimum_response_score_margin,
            f"top score margin {score_margin:.4f} >= {self.minimum_response_score_margin:.4f}",
        )
