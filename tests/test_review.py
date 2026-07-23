"""Manual WER review: row building, adjudicated report, decision persistence,
and the review HTTP server (GET allow-list, POST validation, CSRF check)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from voice_suite.protocol import Utterance
from voice_suite.review import (build_rows, load_decisions, render_page,
                                render_wer_report, save_decisions,
                                serve_review)


def _utt(uid: str, ref: str) -> Utterance:
    return Utterance(id=uid, audio_path=f"data/utterances/{uid}.wav",
                     src_lang="vi", dst_lang="en",
                     ref_transcript=ref, ref_translation="x")


def test_build_rows_skips_missing_and_sorts_worst_first():
    utts = [_utt("good", "một hai ba"), _utt("bad", "một hai ba"),
            _utt("norun", "một")]
    asr = {"good": {"asr_text": "một hai ba"},
           "bad": {"asr_text": "bốn năm sáu"}}
    rows = build_rows(utts, asr, {"bad": [0]})
    assert [r["utt_id"] for r in rows] == ["bad", "good"]
    assert rows[0]["wer"] == 1.0
    assert rows[0]["accepted"] == [0]
    assert len(rows[0]["ops"]) == 3
    assert rows[1]["ops"] == []


def test_render_wer_report_adjudicates():
    rows = build_rows([_utt("u", "một hai ba bốn")],
                      {"u": {"asr_text": "một hay ba bốn"}},  # 1 sub / 4 words
                      {"u": [0, 99]})  # stale index 99 must be ignored
    text = render_wer_report(rows)
    assert "mean WER (raw): **0.2500**" in text
    assert "mean WER (adjudicated): **0.0000**" in text
    assert "sub 1 (1 accepted)" in text


def test_render_wer_report_empty():
    assert "No ASR results" in render_wer_report([])


def test_decisions_roundtrip(tmp_path):
    path = tmp_path / "d.json"
    assert load_decisions(path) == {}
    save_decisions({"u1": [2, 0]}, path)
    assert load_decisions(path) == {"u1": [2, 0]}


def test_render_page_embeds_rows():
    rows = build_rows([_utt("u", "một hai")], {"u": {"asr_text": "một hay"}}, {})
    html = render_page(rows)
    assert "WER review" in html
    assert '"utt_id": "u"' in html
    assert "const DATA" in html


def test_render_page_escapes_script_close():
    rows = build_rows([_utt("u", "a </script> b")],
                      {"u": {"asr_text": "a b"}}, {})
    assert "</script><" not in render_page(rows).replace("\n", "")


def test_render_wer_report_ignores_negative_indices():
    rows = build_rows([_utt("u", "một hai ba bốn")],
                      {"u": {"asr_text": "một hay ba bốn"}},  # 1 sub / 4 words
                      {"u": [-1]})
    text = render_wer_report(rows)
    assert "mean WER (adjudicated): **0.2500**" in text  # -1 accepts nothing


@pytest.fixture()
def review_server(tmp_path):
    rows = build_rows([_utt("u", "một hai ba")],
                      {"u": {"asr_text": "một hay ba"}}, {})
    decisions_path = tmp_path / "decisions.json"
    server = serve_review(rows, 0, decisions_path=decisions_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", decisions_path
    server.shutdown()


def _post(url: str, body: bytes, headers: dict | None = None) -> int:
    req = urllib.request.Request(url + "/save", data=body, method="POST",
                                 headers=headers or {})
    try:
        return urllib.request.urlopen(req).status
    except urllib.error.HTTPError as err:
        return err.code


def test_server_serves_page_and_blocks_repo_files(review_server):
    url, _ = review_server
    assert "WER review" in urllib.request.urlopen(url + "/").read().decode()
    for path in ("/pyproject.toml", "/impls/m2v_sherpa/local.toml",
                 "/out/scored.jsonl"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url + path)
        assert exc.value.code == 404


def test_server_save_roundtrip(review_server):
    url, decisions_path = review_server
    assert _post(url, json.dumps({"u": [1, 0]}).encode()) == 204
    assert load_decisions(decisions_path) == {"u": [0, 1]}


def test_server_save_rejects_malformed(review_server):
    url, decisions_path = review_server
    assert _post(url, b"[1,2]") == 400              # not an object
    assert _post(url, b"not json") == 400
    assert _post(url, json.dumps({"u": [-1]}).encode()) == 400   # negative idx
    assert _post(url, json.dumps({"u": "xx"}).encode()) == 400   # not a list
    assert _post(url, b"{}", {"Content-Length": "zz"}) == 400
    assert not decisions_path.exists()


def test_server_save_rejects_cross_origin(review_server):
    url, decisions_path = review_server
    body = json.dumps({"u": [0]}).encode()
    assert _post(url, body, {"Origin": "http://evil.example"}) == 403
    assert not decisions_path.exists()
    # Same-origin browser fetch carries the server's own origin: allowed.
    assert _post(url, body, {"Origin": url}) == 204
