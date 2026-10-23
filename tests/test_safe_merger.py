"""Tests for pare/safe_merger.py with a deterministic stand-in for CheXbert.

The stand-in labels a text by keywords, using CheXbert's output conventions:
  mode="rrg":            1 for positive or uncertain, 0 otherwise
  mode="classification": 1 positive, 0 negative, -1 uncertain, "" not mentioned
Run with:  python -m pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pare.safe_merger import LABELS_14, safe_merge, safe_merge_negation_aware

KEYWORDS = {
    "Cardiomegaly": "enlarged",
    "Pleural Effusion": "effusion",
    "Edema": "edema",
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
}


class FakeCheXbert:
    def __init__(self, fail=False):
        self.fail = fail

    def get_label(self, text, mode="rrg"):
        if self.fail:
            raise RuntimeError("labeler failure")
        t = text.lower()
        out = []
        for label in LABELS_14:
            k = KEYWORDS.get(label)
            if k is None or k not in t:
                cls = ""
            elif f"no {k}" in t or f"no pleural {k}" in t or f"not {k}" in t:
                cls = 0
            elif f"possible {k}" in t:
                cls = -1
            else:
                cls = 1
            out.append(cls)
        if mode == "rrg":
            return [1 if c in (1, -1) else 0 for c in out]
        return out


S = FakeCheXbert()


def test_appends_new_finding_and_keeps_baseline_verbatim():
    base = "The lungs are clear."
    merged, acts = safe_merge(
        base, "The lungs are clear. There is a small right effusion.", ["Pleural Effusion"], S
    )
    assert merged.startswith(base)
    assert merged.endswith("There is a small right effusion.")
    assert acts["Pleural Effusion"] == "appended"


def test_no_append_when_baseline_already_positive():
    base = "There is a small effusion."
    merged, acts = safe_merge(base, "There is a large effusion.", ["Pleural Effusion"], S)
    assert merged == base and acts["Pleural Effusion"] == "already_in_baseline"


def test_uncertain_baseline_counts_as_present():
    base = "Possible effusion."
    merged, acts = safe_merge(base, "There is an effusion.", ["Pleural Effusion"], S)
    assert merged == base and acts["Pleural Effusion"] == "already_in_baseline"


def test_negation_aware_merger_skips_contradictions():
    base = "No effusion."
    plain, _ = safe_merge(base, "There is a small effusion.", ["Pleural Effusion"], S)
    assert "small effusion" in plain  # the plain merger would contradict the baseline
    merged, acts = safe_merge_negation_aware(
        base, "There is a small effusion.", ["Pleural Effusion"], S
    )
    assert merged == base and acts["Pleural Effusion"] == "baseline_negated"


def test_sentence_with_two_findings_is_appended_once():
    merged, acts = safe_merge(
        "The lungs are clear.", "There is edema and atelectasis.", ["Edema", "Atelectasis"], S
    )
    assert merged.count("There is edema and atelectasis.") == 1
    assert acts["Edema"] == "appended"
    assert acts["Atelectasis"] == "no_sentence_evidence"  # the sentence was already used once


def test_only_gated_pathologies_are_appended():
    merged, acts = safe_merge(
        "The lungs are clear.", "There is edema. The heart is enlarged.", ["Edema"], S
    )
    assert "edema" in merged and "enlarged" not in merged and set(acts) == {"Edema"}


def test_steered_report_without_the_finding_appends_nothing():
    base = "The lungs are clear."
    merged, acts = safe_merge(base, "The lungs remain clear.", ["Edema"], S)
    assert merged == base and acts["Edema"] == "no_new_evidence"


@pytest.mark.parametrize("merge", [safe_merge, safe_merge_negation_aware])
def test_labeler_failure_returns_baseline_unchanged(merge):
    base = "The lungs are clear."
    merged, acts = merge(
        base, "There is an effusion.", ["Pleural Effusion"], FakeCheXbert(fail=True)
    )
    assert merged == base and acts["Pleural Effusion"] == "chexbert_failure_aborted"


def test_empty_reports_are_handled():
    merged, acts = safe_merge("", "There is an effusion.", ["Pleural Effusion"], S)
    assert merged == "" and acts["Pleural Effusion"] == "no_parseable_text"
