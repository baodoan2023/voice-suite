"""Join and sampling logic against fake FLEURS id columns (offline)."""
from __future__ import annotations

from voice_suite.ingest_fleurs import join_ids, sample_triplets, utt_id


def test_join_keeps_only_ids_on_both_sides():
    trips = join_ids([1, 2, 3], [2, 3, 4])
    assert [t[0] for t in trips] == [2, 3]


def test_join_dedupes_repeated_ids_keeping_first_index():
    # FLEURS repeats ids when several speakers read one sentence.
    trips = join_ids([5, 5, 6], [6, 5])
    assert trips == [(5, 0, 1), (6, 2, 0)]


def test_join_result_is_id_sorted():
    trips = join_ids([9, 1, 4], [4, 9, 1])
    assert [t[0] for t in trips] == [1, 4, 9]


def test_sample_is_deterministic_and_id_sorted():
    trips = join_ids(list(range(100)), list(range(100)))
    a = sample_triplets(trips, 10, seed=0)
    b = sample_triplets(trips, 10, seed=0)
    c = sample_triplets(trips, 10, seed=1)
    assert a == b
    assert a != c
    assert len(a) == 10
    assert [t[0] for t in a] == sorted(t[0] for t in a)


def test_sample_larger_than_population_returns_all():
    trips = join_ids([1, 2], [1, 2])
    assert sample_triplets(trips, 10, seed=0) == trips


def test_utt_id_zero_padded():
    assert utt_id(7) == "fleurs-000007"
