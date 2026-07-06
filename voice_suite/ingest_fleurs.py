"""FLEURS → ground-truth manifest + input WAVs.

FLEURS sentences are n-way parallel (FLoRes heritage): joining the vi_vn and
en_us configs on the FLEURS ``id`` field yields (vi audio, vi transcript,
en text) triplets. Some ids appear several times (multiple speakers reading
one sentence); we keep the first occurrence per id on both sides.
"""
from __future__ import annotations

import random
from pathlib import Path

from voice_suite.protocol import Utterance

FLEURS_DATASET = "google/fleurs"
SRC_CONFIG = "vi_vn"
DST_CONFIG = "en_us"
SPLIT = "test"


def _first_index_per_id(ids: list[int]) -> dict[int, int]:
    """Map each FLEURS id to the index of its first occurrence."""
    out: dict[int, int] = {}
    for i, fid in enumerate(ids):
        out.setdefault(int(fid), i)
    return out


def join_ids(vi_ids: list[int], en_ids: list[int]) -> list[tuple[int, int, int]]:
    """(fleurs_id, vi_index, en_index) for ids present on both sides, id-sorted."""
    vi_first = _first_index_per_id(vi_ids)
    en_first = _first_index_per_id(en_ids)
    return [(fid, vi_first[fid], en_first[fid])
            for fid in sorted(vi_first.keys() & en_first.keys())]


def sample_triplets(triplets: list[tuple[int, int, int]], n: int,
                    seed: int) -> list[tuple[int, int, int]]:
    """Deterministic sample of n triplets, id-sorted (tuples sort by id first)."""
    if n >= len(triplets):
        return list(triplets)
    rng = random.Random(seed)
    return sorted(rng.sample(triplets, n))


def utt_id(fleurs_id: int) -> str:
    """Format a FLEURS id as a zero-padded utterance id."""
    return f"fleurs-{fleurs_id:06d}"


def ingest(n: int, seed: int, wav_dir: Path, manifest_path: Path) -> int:
    """Download FLEURS, join vi⋈en on id, sample, write WAVs + manifest.

    Returns the number of manifest rows written. Only sampled rows have
    their audio decoded. Requires the ``ingest`` extra (datasets/soundfile).
    """
    import soundfile as sf              # heavy: ingest extra
    from datasets import load_dataset   # heavy: ingest extra

    vi = load_dataset(FLEURS_DATASET, SRC_CONFIG, split=SPLIT,
                      trust_remote_code=True)
    en = load_dataset(FLEURS_DATASET, DST_CONFIG, split=SPLIT,
                      trust_remote_code=True)
    triplets = sample_triplets(join_ids(vi["id"], en["id"]), n, seed)
    en_texts = en["transcription"]

    wav_dir = Path(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for fid, vi_i, en_i in triplets:
            row = vi[vi_i]                       # decodes audio for this row only
            audio = row["audio"]
            wav_path = wav_dir / f"{utt_id(fid)}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"],
                     subtype="PCM_16")
            utt = Utterance(
                id=utt_id(fid),
                audio_path=str(wav_path),
                src_lang="vi",
                dst_lang="en",
                ref_transcript=row["transcription"],
                ref_translation=en_texts[en_i],
            )
            fh.write(utt.model_dump_json() + "\n")
    return len(triplets)
