"""Append-only report merger.

The baseline report is never modified. New findings from the steered report are
appended to it:

  1. CheXbert labels the full baseline report and the full steered report.
  2. A pathology is new if its probe fired, the steered report is positive for it,
     and the baseline report is not positive for it.
  3. For each new pathology, the first steered sentence that CheXbert labels
     positive for it is appended to the baseline. Each sentence is appended once.

safe_merge_negation_aware, the merger used in the paper, also skips pathologies
that the baseline explicitly negates, so appended text never contradicts it.
Appended sentences can still add false positives or mention other findings.
If CheXbert fails, the baseline is returned unchanged.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# CheXbert 14-label schema
LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]
LABEL_TO_IDX = {p: i for i, p in enumerate(LABELS_14)}


def _chexbert_label_report(text: str, scorer: Any) -> Optional[Dict[str, int]]:
    """Label a full report with CheXbert. Returns None on failure."""
    try:
        raw = scorer.get_label(text, mode="rrg")
        return {
            p: int(raw[i]) if i < len(raw) and raw[i] != "" else 0
            for p, i in LABEL_TO_IDX.items()
        }
    except Exception as e:
        logger.warning(f"CheXbert failed on report: {e}")
        return None  # Caller must abort merge


def _split_sentences(text: str) -> List[str]:
    """Split report into sentences."""
    if not text or not text.strip():
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [s.strip() for s in sents if s.strip() and len(s.strip()) > 5]


def _label_sentence(sentence: str, scorer: Any) -> Optional[Dict[str, int]]:
    """CheXbert-label a single sentence. Returns None on failure."""
    try:
        raw = scorer.get_label(sentence, mode="rrg")
        return {
            p: int(raw[i]) if i < len(raw) and raw[i] != "" else 0
            for p, i in LABEL_TO_IDX.items()
        }
    except Exception:
        return None  # Sentence skipped in evidence extraction


def safe_merge(
    baseline_text: str,
    steered_text: str,
    fired_pathologies: List[str],
    scorer: Any,
) -> Tuple[str, Dict[str, str]]:
    """Append-only merge; the baseline text is never modified.

    Returns (merged_text, actions_dict).

    Actions per pathology:
      - 'already_in_baseline': BL already has this finding -> no change
      - 'appended': genuinely new finding from PARE -> appended
      - 'no_new_evidence': PARE didn't produce this finding either
      - 'not_fired': pathology wasn't in the fired set
    """
    actions: Dict[str, str] = {}

    if not baseline_text.strip() or not steered_text.strip():
        for p in fired_pathologies:
            actions[p] = "no_parseable_text"
        return baseline_text, actions

    # Step 1: label the full reports with CheXbert
    bl_labels = _chexbert_label_report(baseline_text, scorer)
    st_labels = _chexbert_label_report(steered_text, scorer)

    # On CheXbert failure, return the baseline unchanged
    if bl_labels is None or st_labels is None:
        for p in fired_pathologies:
            actions[p] = "chexbert_failure_aborted"
        return baseline_text, actions

    # Step 2: new findings are fired, positive in the steered report and not positive in the baseline
    new_pathologies = []
    for p in fired_pathologies:
        if p not in LABEL_TO_IDX:
            actions[p] = "unknown_pathology"
            continue
        if bl_labels.get(p, 0) == 1:
            actions[p] = "already_in_baseline"
            continue
        if st_labels.get(p, 0) == 1:
            new_pathologies.append(p)
        else:
            actions[p] = "no_new_evidence"

    # Step 3: If nothing new, return baseline unchanged
    if not new_pathologies:
        return baseline_text, actions

    # Step 4: Extract evidence sentences from PARE report
    st_sents = _split_sentences(steered_text)
    if not st_sents:
        for p in new_pathologies:
            actions[p] = "no_parseable_sentences"
        return baseline_text, actions

    # Label each PARE sentence to find which carries the evidence
    st_sent_labels = [_label_sentence(s, scorer) for s in st_sents]

    append_sents: List[str] = []
    used_sent_idx: set = set()  # avoid duplicating sentences

    for p in new_pathologies:
        # Find PARE sentences positive for this pathology
        # Skip sentences where CheXbert labeling failed (None)
        candidates = [
            j for j in range(len(st_sents))
            if st_sent_labels[j] is not None
            and st_sent_labels[j].get(p, 0) == 1
            and j not in used_sent_idx
        ]
        if candidates:
            # Take the first positive sentence (most relevant by position)
            best_j = candidates[0]
            append_sents.append(st_sents[best_j])
            used_sent_idx.add(best_j)
            actions[p] = "appended"
        else:
            # The full report was positive but no single sentence is; append nothing
            actions[p] = "no_sentence_evidence"

    # Step 5: append to the baseline (the baseline text is never modified)
    merged = baseline_text.rstrip()
    if append_sents:
        if not merged.endswith("."):
            merged += "."
        merged += " " + " ".join(append_sents)
        if not merged.endswith("."):
            merged += "."

    return merged, actions


def safe_merge_negation_aware(
    baseline_text: str,
    steered_text: str,
    fired_pathologies: List[str],
    scorer: Any,
) -> Tuple[str, Dict[str, str]]:
    """Merger used in the paper (Sec. 3.5).

    Identical to `safe_merge`, plus condition (iii): nothing is appended for a
    pathology that the baseline explicitly negates (CheXbert class "negative").
    Together with the `safe_merge` check that the baseline is not positive or
    uncertain, sentences are appended only for pathologies the baseline does
    not mention, so appended text never contradicts the baseline.
    """
    negated = set()
    if baseline_text.strip():
        try:
            cls = scorer.get_label(baseline_text.strip(), mode="classification")
            negated = {p for p in fired_pathologies
                       if p in LABEL_TO_IDX and cls[LABEL_TO_IDX[p]] == 0}
        except Exception as e:  # On CheXbert failure, append nothing
            logger.warning(f"CheXbert failed on baseline: {e}")
            return baseline_text, {p: "chexbert_failure_aborted" for p in fired_pathologies}
    keep = [p for p in fired_pathologies if p not in negated]
    merged, actions = safe_merge(baseline_text, steered_text, keep, scorer)
    for p in negated:
        actions[p] = "baseline_negated"
    return merged, actions
