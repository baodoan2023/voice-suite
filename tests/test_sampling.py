"""Deterministic sampling invariants."""
from __future__ import annotations

from voice_suite.sampling import sample_items


def test_no_limit_returns_all_id_sorted(make_utt):
    items = [make_utt(3), make_utt(1), make_utt(2)]
    assert [u.id for u in sample_items(items, None)] == ["u001", "u002", "u003"]


def test_same_seed_same_subset(make_utt):
    items = [make_utt(i) for i in range(1, 51)]
    a = sample_items(items, 10, seed=7)
    b = sample_items(items, 10, seed=7)
    assert a == b
    assert len(a) == 10


def test_different_seed_different_subset(make_utt):
    items = [make_utt(i) for i in range(1, 51)]
    assert sample_items(items, 10, seed=0) != sample_items(items, 10, seed=1)


def test_limit_zero_or_negative_empty(make_utt):
    items = [make_utt(1)]
    assert sample_items(items, 0) == []
    assert sample_items(items, -5) == []


def test_limit_larger_than_population(make_utt):
    items = [make_utt(2), make_utt(1)]
    assert [u.id for u in sample_items(items, 99)] == ["u001", "u002"]
