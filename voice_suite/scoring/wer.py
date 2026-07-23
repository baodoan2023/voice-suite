"""ASR word-error-rate on normalized text (Vietnamese-safe)."""
from __future__ import annotations

import re
import unicodedata

import jiwer

# Everything that is not a word character or whitespace counts as
# punctuation. ``\w`` is unicode-aware in Python 3: Vietnamese letters with
# diacritics (à, đ, ệ, …) are preserved; casing and punctuation are folded.
_PUNCT_RE = re.compile(r"[^\w\s]")


def norm_text(s: str) -> str:
    """Lowercase, punctuation → space, collapse whitespace."""
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT_RE.sub(" ", s.lower())
    return " ".join(s.split())


WordOp = tuple[str, str, str]  # (op, ref_word, hyp_word); op ∈ sub|del|ins

_OP_NAMES = {"substitute": "sub", "delete": "del", "insert": "ins"}


def align(ref: str, hyp: str) -> list[WordOp]:
    """Word-level error ops of hyp against ref, on normalized text.

    Deterministic and in sentence order: review decisions reference ops by
    index, so the order must be stable across runs. len(align()) equals the
    uncapped edit count score_wer() is built on.
    """
    ref_n, hyp_n = norm_text(ref), norm_text(hyp)
    if not ref_n and not hyp_n:
        return []
    if not ref_n:
        return [("ins", "", w) for w in hyp_n.split()]
    if not hyp_n:
        return [("del", w, "") for w in ref_n.split()]
    out = jiwer.process_words(ref_n, hyp_n)
    refs, hyps = out.references[0], out.hypotheses[0]
    ops: list[WordOp] = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        op = _OP_NAMES[chunk.type]
        r_words = refs[chunk.ref_start_idx:chunk.ref_end_idx]
        h_words = hyps[chunk.hyp_start_idx:chunk.hyp_end_idx]
        for i in range(max(len(r_words), len(h_words))):
            ops.append((op,
                        r_words[i] if i < len(r_words) else "",
                        h_words[i] if i < len(h_words) else ""))
    return ops


def adjudicated_wer(ref: str, hyp: str, accepted: set[int]) -> float:
    """WER after a human reviewer accepts some ops as not-real-errors.

    ``accepted`` holds indices into align(ref, hyp); stale indices are
    ignored. Same cap and empty-ref conventions as score_wer.
    """
    ops = align(ref, hyp)
    errors = sum(1 for i in range(len(ops)) if i not in accepted)
    n_ref = len(norm_text(ref).split())
    if n_ref == 0:
        return 0.0 if errors == 0 else 1.0
    return min(errors / n_ref, 1.0)


def score_wer(ref: str, hyp: str) -> float:
    """WER of hyp against ref after normalization, capped at 1.0.

    The cap keeps one long hallucination from dominating a mean (insertions
    can push raw WER above 1.0). Empty ref: 0.0 if hyp is also empty, else
    1.0 (jiwer cannot divide by zero reference words).
    """
    ref_n, hyp_n = norm_text(ref), norm_text(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return min(jiwer.wer(ref_n, hyp_n), 1.0)
