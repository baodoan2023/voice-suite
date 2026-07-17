"""Default my-2nd-voice pipeline config: PhoWhisper-small + Marian vi-en + Supertonic."""
from impls.m2v_phowhisper.adapter import M2vBatchImpl

IMPL = M2vBatchImpl()
