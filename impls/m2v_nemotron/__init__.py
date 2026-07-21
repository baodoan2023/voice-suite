"""Nemotron my-2nd-voice pipeline config: NVIDIA Nemotron streaming ASR + Marian vi-en + StyleTTS2."""
from impls.m2v_nemotron.adapter import M2vNemotronImpl

IMPL = M2vNemotronImpl()
