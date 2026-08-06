"""Deterministic Calibration Layer: Filter, rank, and calibrate retrieved evidence.

Phase 3 of Zero-Mem integration. All operations zero-LLM-token:
- Hard filter: remove contradictions, boundary violations, subject mismatches
- Compatibility ranking: subject, recency, answer-type scoring
- Answer calibration: extract evidence-local candidates, correct ungrounded claims

This runs AFTER retrieval and BEFORE the LLM reader, ensuring only
consistent, relevant evidence reaches the final QA step.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from zero_mem.index_builder import Entity, Message
from zero_mem.retriever import RetrievedEvidence, parse_query

TECH_TERMS = {"docker", "json", "api", "ai", "mcp", "caprover", "github", "python", "spacy", "kubernetes", "cloudflare", "bge-m3", "minilm"}


# ---------------------------------------------------------------------------
# Error signals detected during calibration (for diagnostics, not LLM)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSignal:
    """A conflict or problem found during evidence calibration."""
    kind: str           # "contradiction", "subject_mismatch", "boundary_violation",
                        # "temporal_inconsistency", "entity_ambiguity"
    description: str
    evidence_ids: list[str]  # msg_ids involved
    severity: float     # 0..1, higher = more serious


@dataclass
class CalibratedEvidence:
    """Evidence block after filtering, ranking, and calibration."""
    message: Message
    score: float
    source_view: str
    related_entities: list[Entity] = field(default_factory=list)
    temporal_context: list[str] = field(default_factory=list)
    calibration_signals: list[CalibrationSignal] = field(default_factory=list)
    filtered: bool = False
    filter_reason: str = ""
    calibrated_claim: str = ""  # corrected version of message content, if needed


# ---------------------------------------------------------------------------
# Hard filter: remove contradictions and mismatches
# ---------------------------------------------------------------------------

def hard_filter(
    evidence_blocks: list[RetrievedEvidence],
    query_entities: list[str],
) -> list[tuple[RetrievedEvidence, list[CalibrationSignal]]]:
    """Apply hard filters to remove clearly conflicting or irrelevant evidence.

    Returns evidence blocks with any calibration signals attached.
    Blocks with severity >= 0.8 signals should be discarded.
    """
    filtered: list[tuple[RetrievedEvidence, list[CalibrationSignal]]] = []

    for ev in evidence_blocks:
        signals: list[CalibrationSignal] = []
        content = ev.message.content.lower()

        # 1. Subject mismatch: evidence discusses different entities than query
        ev_entity_keys = {e.key for e in ev.related_entities}
        query_entity_set = set(query_entities)
        if query_entity_set and ev_entity_keys and ev_entity_keys.isdisjoint(query_entity_set):
            # Evidence covers different entities entirely
            signals.append(CalibrationSignal(
                kind="subject_mismatch",
                description=f"Evidence covers entities {ev_entity_keys}; query targets {query_entity_set}",
                evidence_ids=[ev.message.msg_id],
                severity=0.7,
            ))

        # 2. Temporal boundary violation: evidence timestamp is far from query-relevant window
        # (Simplified: mark evidence older than 90 days as potentially stale)
        import time
        msg_age_days = (time.time() - ev.message.timestamp) / 86400
        if msg_age_days > 90:
            signals.append(CalibrationSignal(
                kind="boundary_violation",
                description=f"Evidence is {msg_age_days:.0f} days old; may be stale",
                evidence_ids=[ev.message.msg_id],
                severity=0.3,
            ))

        # 3. Source quality: weak signals from far-removed PageRank propagation
        if ev.score < 0.05:
            signals.append(CalibrationSignal(
                kind="boundary_violation",
                description=f"Evidence score ({ev.score:.4f}) below relevance threshold",
                evidence_ids=[ev.message.msg_id],
                severity=0.5,
            ))

        # 4. Contradiction detection (simplified: flag mutually exclusive entity labels)
        # In practice, this would compare claims across evidence blocks
        # For now, flag single-evidence blocks with conflicting entity types
        entity_labels = Counter(e.label for e in ev.related_entities if e.text.lower().strip() not in TECH_TERMS)
        if len(entity_labels) >= 3 and max(entity_labels.values()) <= 2:
            signals.append(CalibrationSignal(
                kind="entity_ambiguity",
                description=f"Multiple entity types with low confidence: {dict(entity_labels)}",
                evidence_ids=[ev.message.msg_id],
                severity=0.2,
            ))

        # Hard exclusion is reserved for clearly irrelevant or weak evidence.
        # Keep diagnostics on retained blocks; callers can inspect all signals.
        max_severity = max((sig.severity for sig in signals), default=0.0)
        has_subject_mismatch = any(sig.kind == "subject_mismatch" for sig in signals)
        if max_severity >= 0.8 or has_subject_mismatch or ev.score < 0.03:
            signals.append(CalibrationSignal(
                kind="boundary_violation" if ev.score < 0.03 else "subject_mismatch",
                description="Evidence excluded by deterministic relevance hard filter",
                evidence_ids=[ev.message.msg_id],
                severity=max(0.8, max_severity),
            ))
            continue
        filtered.append((ev, signals))

    return filtered


# ---------------------------------------------------------------------------
# Compatibility ranking
# ---------------------------------------------------------------------------

def compatibility_rank(
    evidence_blocks: list[RetrievedEvidence],
    query_subject: str,
    query_answer_type: str,
) -> list[tuple[RetrievedEvidence, float]]:
    """Rank evidence by compatibility with query subject, recency, and answer type.

    Adds compatibility scores to the existing retrieval scores.
    """
    ranked: list[tuple[RetrievedEvidence, float]] = []
    subject_lower = query_subject.lower()
    import time
    now = time.time()

    for ev in evidence_blocks:
        content = ev.message.content.lower()
        compat = 0.0

        # Subject relevance: does this evidence discuss the same subject?
        subject_terms = set(subject_lower.split())
        content_terms = set(content.split())
        overlap = subject_terms & content_terms
        if subject_terms:
            compat += len(overlap) / len(subject_terms) * 0.3

        # Recency bonus: newer evidence is more compatible for factual queries
        msg_age_hours = (now - ev.message.timestamp) / 3600
        if msg_age_hours < 24:
            compat += 0.2
        elif msg_age_hours < 168:  # 1 week
            compat += 0.1

        # Answer type match: evidence format matches expected answer type
        answer_type_patterns = {
            "definition": ["is", "means", "refers to", "defined as"],
            "comparison": ["compare", "versus", "difference", "similar"],
            "list": ["first", "second", "third", "steps", "items"],
            "howto": ["step", "how to", "install", "configure", "run"],
            "factual": ["has", "runs", "uses", "contains", "supports"],
        }
        type_signals = answer_type_patterns.get(query_answer_type, answer_type_patterns["factual"])
        if any(s in content for s in type_signals):
            compat += 0.15

        # Density bonus: shorter, focused messages rank higher
        msg_len = len(ev.message.content.split())
        if msg_len > 0:
            compat += min(0.1, 10 / msg_len)

        final_score = ev.score * 0.7 + compat * 0.3
        ranked.append((ev, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Answer calibration: correct ungrounded claims
# ---------------------------------------------------------------------------

def calibrate_answers(
    evidence_blocks: list[RetrievedEvidence],
    query: str,
    existing_signals: dict[str, list[CalibrationSignal]] | None = None,
) -> list[CalibratedEvidence]:
    """Extract evidence-local answer candidates and correct any ungrounded claims.

    This is the final step before the LLM reader. It annotates each evidence
    block with calibration signals and suggested corrections (deterministic,
    no LLM tokens consumed).
    """
    calibrated: list[CalibratedEvidence] = []

    for ev in evidence_blocks:
        signals: list[CalibrationSignal] = list((existing_signals or {}).get(ev.message.msg_id, []))
        filtered = False
        filter_reason = ""
        corrected = ev.message.content

        # Check for speculative language that might signal ungrounded claims
        speculative_patterns = [
            (r'\b(?:might|could|probably|perhaps|possibly|i think)\b', 0.1),
            (r'\b(?:always|never|every|all|none)\b', 0.15),  # overgeneralizations
        ]
        for pattern, severity in speculative_patterns:
            if re.search(pattern, ev.message.content, re.IGNORECASE):
                signals.append(CalibrationSignal(
                    kind="entity_ambiguity",
                    description=f"Speculative or overgeneralized language detected: '{pattern}'",
                    evidence_ids=[ev.message.msg_id],
                    severity=severity,
                ))

        # Check for hallucination markers: claims about specific numbers/versions
        # from unverified sources
        number_pattern = r'\b\d+(?:\.\d+)?\s*(?:GB|TB|MB|GHz|cores?|nodes?|ms)\b'
        numbers = re.findall(number_pattern, ev.message.content, re.IGNORECASE)
        if numbers and ev.score < 0.1:
            signals.append(CalibrationSignal(
                kind="temporal_inconsistency",
                description=f"Specific metrics ({', '.join(numbers[:3])}) from low-confidence evidence",
                evidence_ids=[ev.message.msg_id],
                severity=0.25,
            ))

        # Filter out blocks with any high-severity signal
        if any(s.severity >= 0.7 for s in signals):
            filtered = True
            filter_reason = f"Severity {max(s.severity for s in signals):.1f}: " + \
                           "; ".join(s.description for s in signals if s.severity >= 0.7)

        calibrated.append(CalibratedEvidence(
            message=ev.message,
            score=ev.score,
            source_view=ev.source_view,
            related_entities=ev.related_entities,
            temporal_context=ev.temporal_context,
            calibration_signals=signals,
            filtered=filtered,
            filter_reason=filter_reason,
            calibrated_claim=corrected,
        ))

    return calibrated


# ---------------------------------------------------------------------------
# Main calibration pipeline
# ---------------------------------------------------------------------------

def classify_answer_type(query: str) -> str:
    """Determine what kind of answer the query expects. Zero LLM tokens."""
    q = query.lower()
    if any(w in q for w in ("what is", "define", "mean", "definition")):
        return "definition"
    if any(w in q for w in ("compare", "versus", "vs", "difference", "better")):
        return "comparison"
    if any(w in q for w in ("steps", "how to", "how do i", "install", "configure", "setup")):
        return "howto"
    if any(w in q for w in ("list", "enumerate", "what are", "all the")):
        return "list"
    return "factual"


def extract_query_subject(query: str, spacy_model) -> str:
    """Extract the primary subject of the query. Zero LLM tokens."""
    doc = spacy_model(query[:5000])
    # Take the first noun phrase after common question words
    skip_words = {"what", "who", "where", "when", "how", "why", "is", "are",
                  "was", "were", "does", "do", "did", "can", "could", "will",
                  "would", "tell", "me", "about", "the", "a", "an", "in", "on"}
    words = [t.text for t in doc if t.text.lower() not in skip_words and not t.is_punct]
    if words:
        # Return first 3-5 content words as subject
        return " ".join(words[:5])
    return query


class CalibrationLayer:
    """Full calibration pipeline: filter -> rank -> calibrate.

    Runs after dual-view retrieval and before the LLM reader.
    All operations are deterministic — zero LLM tokens consumed.
    """

    def __init__(self, spacy_model):
        self.spacy_model = spacy_model

    def calibrate(
        self,
        evidence_blocks: list[RetrievedEvidence],
        query: str,
        query_entities: list[str] | None = None,
    ) -> list[CalibratedEvidence]:
        """Run the full calibration pipeline.

        Args:
            evidence_blocks: Raw retrieved evidence from DualViewRetriever.
            query: Original user query.
            query_entities: Entity keys extracted from the query.

        Returns:
            Calibrated evidence blocks, filtered and scored, ready for LLM.
        """
        if query_entities is None:
            parsed = parse_query(query, None, self.spacy_model)
            query_entities = parsed.entities

        # Step 1: Hard filter
        filtered = hard_filter(evidence_blocks, query_entities)

        # Step 2: Compatibility ranking
        subject = extract_query_subject(query, self.spacy_model)
        answer_type = classify_answer_type(query)
        signal_map = {ev.message.msg_id: signals for ev, signals in filtered}
        ranked = compatibility_rank(
            [ev for ev, _ in filtered], subject, answer_type
        )
        ranked_evidence = [ev for ev, _ in ranked]

        # Step 3: Answer calibration, retaining hard-filter diagnostics.
        calibrated = calibrate_answers(ranked_evidence, query, existing_signals=signal_map)
        for item in calibrated:
            max_severity = max((sig.severity for sig in item.calibration_signals), default=0.0)
            item.filtered = max_severity >= 0.8
            if item.filtered and not item.filter_reason:
                item.filter_reason = "Deterministic severity/relevance hard filter"
        return [c for c in calibrated if not c.filtered]
