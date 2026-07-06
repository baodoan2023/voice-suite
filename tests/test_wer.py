"""WER normalization and scoring: diacritics, folding, edge cases."""
from __future__ import annotations

import pytest

from voice_suite.scoring.wer import norm_text, score_wer


def test_norm_preserves_vietnamese_diacritics():
    assert norm_text("Đây là tiếng Việt.") == "đây là tiếng việt"


def test_norm_folds_case_punct_whitespace():
    assert norm_text("  Xin CHÀO,   bạn!  ") == "xin chào bạn"


def test_identical_after_normalization_is_zero():
    assert score_wer("Xin chào, bạn!", "xin chào bạn") == 0.0


def test_missing_diacritics_count_as_substitution():
    # "chao" ≠ "chào": one substitution out of two reference words.
    assert score_wer("xin chào", "xin chao") == pytest.approx(0.5)


def test_empty_hypothesis_is_one():
    assert score_wer("một hai ba", "") == 1.0


def test_empty_reference():
    assert score_wer("", "") == 0.0
    assert score_wer("", "gì đó") == 1.0


def test_wer_capped_at_one():
    assert score_wer("một", "a b c d e f") == 1.0
