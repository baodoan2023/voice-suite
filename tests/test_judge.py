"""Judge plumbing: prompt content, score mapping, lenient JSON, backends."""
from __future__ import annotations

import types

import pytest

from voice_suite.scoring.judge import (
    _anthropic_judge, _loads_lenient, build_judge_prompt, get_judge,
    score_utterance,
)

GOOD = {"mt_adequacy": 0.9, "mt_fluency": 0.8, "e2e_adequacy": 0.7,
        "verdict": "good", "rationale": "close match"}


def test_prompt_contains_all_five_blocks_and_no_impl_name(make_utt, make_result):
    utt = make_utt(ref_transcript="nguồn tiếng việt", ref_translation="english ref")
    res = make_result(asr_text="asr nghe được", mt_text="mt english out")
    prompt = build_judge_prompt(utt, res, "back transcript here")
    for needle in ("nguồn tiếng việt", "english ref", "asr nghe được",
                   "mt english out", "back transcript here"):
        assert needle in prompt
    assert "m2v" not in prompt  # bias-blind: no impl identity


def test_prompt_marks_missing_output_audio(make_utt, make_result):
    prompt = build_judge_prompt(make_utt(), make_result(), "")
    assert "(no output audio)" in prompt


def test_score_utterance_maps_fields(make_utt, make_result):
    score = score_utterance(make_utt(), make_result(), "bt",
                            call_judge=lambda prompt: dict(GOOD))
    assert score.mt_adequacy == 0.9
    assert score.mt_fluency == 0.8
    assert score.e2e_adequacy == 0.7
    assert score.verdict == "good"
    assert score.error is None


def test_score_utterance_captures_judge_exception(make_utt, make_result):
    def boom(prompt):
        raise RuntimeError("api down")
    score = score_utterance(make_utt(), make_result(), "bt", call_judge=boom)
    assert score.error == "api down"
    assert score.verdict == "error"
    assert score.mt_adequacy == 0.0


def test_score_utterance_missing_key_is_error(make_utt, make_result):
    score = score_utterance(make_utt(), make_result(), "bt",
                            call_judge=lambda p: {"mt_adequacy": 1.0})
    assert score.error is not None


def test_loads_lenient_plain_and_fenced():
    assert _loads_lenient('{"a": 1}') == {"a": 1}
    assert _loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}


def test_loads_lenient_prose_and_trailing_brace():
    assert _loads_lenient('score: {"a": 1} done {oops') == {"a": 1}


def test_loads_lenient_no_json_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        _loads_lenient("nothing here")


def test_get_judge_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown judge backend"):
        get_judge("smoke-signals")


def test_cli_judge_missing_binary_raises(monkeypatch):
    from voice_suite.scoring import judge as judge_mod
    monkeypatch.setattr(judge_mod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="claude CLI not found"):
        judge_mod._cli_judge("prompt")


def test_anthropic_judge_parses_fenced_reply_and_reuses_client(monkeypatch):
    import anthropic

    from voice_suite.scoring import judge as judge_mod
    monkeypatch.setattr(judge_mod, "_anthropic_client", None)

    fenced_reply = (
        "Sure, here are the scores:\n```json\n"
        '{"mt_adequacy": 0.9, "mt_fluency": 0.8, "e2e_adequacy": 0.7, '
        '"verdict": "good", "rationale": "close match"}\n```'
    )
    constructions = []

    class FakeAnthropic:
        def __init__(self):
            constructions.append(self)
            self.messages = self

        def create(self, **kwargs):
            block = types.SimpleNamespace(type="text", text=fenced_reply)
            return types.SimpleNamespace(content=[block])

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    first = judge_mod._anthropic_judge("prompt one")
    second = judge_mod._anthropic_judge("prompt two")

    assert first == GOOD
    assert second == GOOD
    assert len(constructions) == 1
