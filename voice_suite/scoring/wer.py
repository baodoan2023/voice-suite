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
