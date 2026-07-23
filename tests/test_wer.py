"""WER normalization and scoring: diacritics, folding, edge cases."""
from __future__ import annotations

import unicodedata

import pytest

from voice_suite.scoring.wer import adjudicated_wer, align, norm_text, score_wer


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


def test_align_sub_del_ins():
    # Each case has a unique minimal edit path (Levenshtein paths are not
    # unique in general, so ambiguous cases must not pin exact ops).
    assert align("một hai ba", "một hay ba") == [("sub", "hai", "hay")]
    assert align("một hai ba", "một ba") == [("del", "hai", "")]
    assert align("một hai", "một hai ba") == [("ins", "", "ba")]


def test_align_normalizes_like_score_wer():
    assert align("Xin chào, bạn!", "xin chào bạn") == []


def test_align_empty_sides():
    assert align("", "") == []
    assert align("một hai", "") == [("del", "một", ""), ("del", "hai", "")]
    assert align("", "gì đó") == [("ins", "", "gì"), ("ins", "", "đó")]


def test_align_op_count_matches_score_wer():
    ref, hyp = "xe buýt số 403 chạy đến sintra", "xe buýt số bốn chạy tới xina"
    assert len(align(ref, hyp)) / 7 == pytest.approx(score_wer(ref, hyp))


def test_adjudicated_wer_no_accepts_equals_score_wer():
    ref, hyp = "một hai ba bốn", "một hay bốn năm"
    assert adjudicated_wer(ref, hyp, set()) == pytest.approx(score_wer(ref, hyp))


def test_adjudicated_wer_accepting_ops_lowers_score():
    ref, hyp = "một hai ba bốn", "một hay bốn năm"  # 3 ops, 4 ref words
    assert adjudicated_wer(ref, hyp, {0}) == pytest.approx(2 / 4)
    assert adjudicated_wer(ref, hyp, {0, 1, 2}) == 0.0


def test_adjudicated_wer_ignores_stale_indices():
    # Decisions referencing ops that no longer exist must not crash or count.
    assert adjudicated_wer("một hai", "một hai", {5}) == 0.0
    # Negative indices must not deflate the error count either.
    assert adjudicated_wer("một hai", "một ba", {-1}) == pytest.approx(0.5)


def test_adjudicated_wer_empty_reference():
    assert adjudicated_wer("", "gì đó", set()) == 1.0
    assert adjudicated_wer("", "gì đó", {0, 1}) == 0.0


def test_norm_text_nfc_nfd_invariant():
    # Verify that norm_text produces identical output regardless of
    # whether input is in NFC (precomposed) or NFD (decomposed) form.
    # This is critical for Vietnamese text with diacritics.
    original = "tiếng Việt"
    nfc_form = unicodedata.normalize("NFC", original)
    nfd_form = unicodedata.normalize("NFD", original)

    # Both forms should normalize to the same result
    assert norm_text(nfc_form) == norm_text(nfd_form)

    # Expected normalized result: lowercase, no diacritics preserved
    # (diacritics should be preserved as-is, just lowercased)
    expected = "tiếng việt"
    assert norm_text(nfc_form) == expected
    assert norm_text(nfd_form) == expected
