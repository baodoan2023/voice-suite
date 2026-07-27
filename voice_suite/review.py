"""Manual WER adjudication: a local review page for accepting non-errors.

The reviewer sees ref vs. hypothesis with word-level diff ops, listens to the
source audio, and ticks ops that are not real errors (spelling variants,
spoken-form numbers, reference mistakes). Decisions persist to a JSON file
that `wer-report` folds into an adjudicated WER.
"""
from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from voice_suite.protocol import Utterance
from voice_suite.scoring.wer import adjudicated_wer, align, norm_text, score_wer

def decisions_path(engine: str) -> Path:
    """Per-engine decision file: op indices only make sense per hypothesis."""
    return Path("out") / f"wer_decisions_{engine}.json"


DECISIONS = decisions_path("sherpa")


def load_decisions(path: Path = DECISIONS) -> dict[str, list[int]]:
    """{utt_id: [accepted op indices]}; missing file means no decisions."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_decisions(decisions: dict[str, list[int]],
                   path: Path = DECISIONS) -> None:
    """Atomic write: a crash mid-save must not corrupt existing decisions."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(decisions, indent=1, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, p)


def build_rows(utts: list[Utterance], asr: dict[str, dict],
               decisions: dict[str, list[int]]) -> list[dict]:
    """One JSON-safe row per utterance that has an ASR result."""
    rows = []
    for u in utts:
        if u.id not in asr:
            continue
        hyp = asr[u.id]["asr_text"]
        rows.append({
            "utt_id": u.id,
            "audio": Path(u.audio_path).as_posix(),
            "ref": u.ref_transcript,
            "hyp": hyp,
            "n_ref": len(norm_text(u.ref_transcript).split()),
            "ops": [list(op) for op in align(u.ref_transcript, hyp)],
            "accepted": decisions.get(u.id, []),
            "wer": round(score_wer(u.ref_transcript, hyp), 4),
            "asr_ms": asr[u.id].get("asr_ms", 0),
        })
    rows.sort(key=lambda r: -r["wer"])
    return rows


_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>WER review — m2v_sherpa</title>
<style>
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1a1d21; }}
  header {{ position: sticky; top: 0; background: #1a1d21; color: #fff; padding: .7rem 1.2rem;
           display: flex; gap: 2rem; align-items: baseline; z-index: 1; }}
  header b {{ font-size: 1.2rem; }}
  main {{ max-width: 60rem; margin: 1rem auto; padding: 0 1rem; }}
  .utt {{ background: #fff; border: 1px solid #e2e5e9; border-radius: 8px;
         padding: 1rem 1.2rem; margin-bottom: 1rem; }}
  .utt h3 {{ margin: 0 0 .4rem; font-size: .95rem; display: flex; gap: 1rem; align-items: baseline; }}
  .wer {{ font-variant-numeric: tabular-nums; color: #b3261e; }}
  .adj {{ font-variant-numeric: tabular-nums; color: #1a7f37; }}
  .txt {{ margin: .15rem 0; }} .txt b {{ display: inline-block; width: 2.6rem; color: #6a7180; font-weight: 600; }}
  audio {{ width: 100%; height: 2rem; margin: .4rem 0; }}
  .ops {{ display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .5rem; }}
  .op {{ border: 1px solid #d0d4da; border-radius: 6px; padding: .15rem .5rem;
        cursor: pointer; user-select: none; background: #fff; }}
  .op.acc {{ background: #e7f4ea; border-color: #1a7f37; opacity: .75; text-decoration: line-through; }}
  .op kbd {{ background: #eef0f3; border-radius: 3px; padding: 0 .3rem; font-size: .75rem; margin-right: .3rem; }}
  .clean {{ color: #1a7f37; }}
  #status {{ margin-left: auto; font-size: .85rem; color: #9aa3af; }}
</style>
<header>
  <b>WER review</b>
  <span>raw <span class="wer" id="mraw"></span></span>
  <span>adjudicated <span class="adj" id="madj"></span></span>
  <span id="status">saved</span>
</header>
<main id="list"></main>
<script>
const DATA = {data};
const acc = Object.fromEntries(DATA.map(r => [r.utt_id, new Set(r.accepted)]));

function adjWer(r) {{
  const errs = r.ops.filter((_, i) => !acc[r.utt_id].has(i)).length;
  return r.n_ref === 0 ? (errs ? 1 : 0) : Math.min(errs / r.n_ref, 1);
}}
function fmt(x) {{ return x.toFixed(3); }}
function refreshTotals() {{
  const n = DATA.length;
  document.getElementById('mraw').textContent =
    fmt(DATA.reduce((s, r) => s + r.wer, 0) / n);
  document.getElementById('madj').textContent =
    fmt(DATA.reduce((s, r) => s + adjWer(r), 0) / n);
}}
let saveTimer;
function save() {{
  document.getElementById('status').textContent = 'saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {{
    const body = Object.fromEntries(
      Object.entries(acc).map(([k, v]) => [k, [...v].sort((a, b) => a - b)]));
    try {{
      const res = await fetch('/save', {{method: 'POST', body: JSON.stringify(body)}});
      document.getElementById('status').textContent = res.ok ? 'saved' : 'SAVE FAILED';
    }} catch (e) {{
      document.getElementById('status').textContent = 'SAVE FAILED (server down?)';
    }}
  }}, 300);
}}
// Corpus/ASR text goes through textContent only — never innerHTML.
const adjEls = {{}};
const list = document.getElementById('list');
for (const r of DATA) {{
  const div = document.createElement('div');
  div.className = 'utt';
  div.innerHTML = `<h3><span class="id"></span>
      <span>raw <span class="wer"></span></span>
      <span>adj <span class="adj"></span></span></h3>
    <audio controls preload="none"></audio>
    <div class="txt"><b>ref</b> <span class="ref"></span></div>
    <div class="txt"><b>asr</b> <span class="hyp"></span></div>
    <div class="ops"></div>`;
  div.querySelector('.id').textContent = r.utt_id;
  div.querySelector('.wer').textContent = fmt(r.wer);
  div.querySelector('audio').src = '/' + r.audio;
  div.querySelector('.ref').textContent = r.ref;
  div.querySelector('.hyp').textContent = r.hyp;
  adjEls[r.utt_id] = div.querySelector('.adj');
  list.appendChild(div);
  const ops = div.querySelector('.ops');
  if (!r.ops.length) {{
    const clean = document.createElement('span');
    clean.className = 'clean';
    clean.textContent = 'no errors';
    ops.appendChild(clean);
  }}
  r.ops.forEach((op, i) => {{
    const chip = document.createElement('span');
    chip.className = 'op' + (acc[r.utt_id].has(i) ? ' acc' : '');
    chip.title = 'click to toggle: accepted ops do not count as errors';
    const kbd = document.createElement('kbd');
    kbd.textContent = op[0];
    chip.append(kbd, op[0] === 'sub'
      ? `${{op[1]}} → ${{op[2]}}` : (op[0] === 'del' ? op[1] : op[2]));
    chip.onclick = () => {{
      acc[r.utt_id].has(i) ? acc[r.utt_id].delete(i) : acc[r.utt_id].add(i);
      chip.classList.toggle('acc');
      refreshRow(r); refreshTotals(); save();
    }};
    ops.appendChild(chip);
  }});
  refreshRow(r);
}}
function refreshRow(r) {{
  adjEls[r.utt_id].textContent = fmt(adjWer(r));
}}
refreshTotals();
</script>
"""


def render_page(rows: list[dict]) -> str:
    # `</` escaping stops corpus text containing "</script>" from ending
    # the inline script block early.
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.format(data=data)


_AUDIO_PREFIX = "/data/utterances/"


class _Handler(SimpleHTTPRequestHandler):
    """Serve the page at /, audio under data/utterances/, and accept /save."""

    page: str = ""
    decisions_path: Path = DECISIONS
    allowed_origins: frozenset[str] = frozenset()

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path in ("/", "/index.html"):
            body = self.page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith(_AUDIO_PREFIX):
            # Only the audio dir is exposed — never the rest of the repo
            # (local.toml, .git, …). translate_path drops any `..` parts,
            # so this prefix cannot be escaped.
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path != "/save":
            self.send_error(404)
            return
        # Loopback binding does not stop another browser tab from POSTing
        # cross-origin; browsers always attach Origin to fetch POSTs.
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.allowed_origins:
            self.send_error(403, "cross-origin save rejected")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            decisions = json.loads(self.rfile.read(length))
            if not isinstance(decisions, dict):
                raise ValueError("expected a JSON object")
            clean: dict[str, list[int]] = {}
            for utt_id, indices in decisions.items():
                idxs = sorted({int(i) for i in indices})
                if idxs and idxs[0] < 0:
                    raise ValueError("negative op index")
                clean[str(utt_id)] = idxs
        except (ValueError, TypeError):
            self.send_error(400, "malformed decisions payload")
            return
        save_decisions(clean, self.decisions_path)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):  # quiet: one line per audio seek is noise
        pass


def serve_review(rows: list[dict], port: int,
                 decisions_path: Path = DECISIONS) -> ThreadingHTTPServer:
    """Start (but do not block on) the review server; caller runs forever."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    bound_port = server.server_address[1]  # real port when asked for 0
    _Handler.page = render_page(rows)
    _Handler.decisions_path = decisions_path
    _Handler.allowed_origins = frozenset(
        f"http://{host}:{bound_port}" for host in ("127.0.0.1", "localhost"))
    return server


def _adjudicated(row: dict) -> float:
    # Single source of truth: the tested scoring.wer implementation, which
    # ignores stale and negative indices by construction.
    return adjudicated_wer(row["ref"], row["hyp"], set(row["accepted"]))


def _engine_summary_row(name: str, rows: list[dict]) -> str:
    n = len(rows)
    mean_raw = sum(r["wer"] for r in rows) / n
    mean_adj = sum(_adjudicated(r) for r in rows) / n
    by_type = {"sub": 0, "del": 0, "ins": 0}
    accepted = 0
    for r in rows:
        accepted += sum(1 for i in r["accepted"] if 0 <= i < len(r["ops"]))
        for op in r["ops"]:
            by_type[op[0]] += 1
    p50 = sorted(r["asr_ms"] for r in rows)[n // 2]
    return (f"| {name} | {n} | {mean_raw:.4f} | {mean_adj:.4f} "
            f"| {by_type['sub']} | {by_type['del']} | {by_type['ins']} "
            f"| {accepted} | {p50} |")


def render_wer_report(engines: dict[str, list[dict]]) -> str:
    """WER-only markdown report: per-engine summary, cross-engine matrix."""
    engines = {k: v for k, v in engines.items() if v}
    if not engines:
        return ("# WER report\n\nNo ASR results yet — run "
                "`voice-suite asr-run` first.\n")
    lines = [
        "# WER report",
        "",
        "Adjudicated = after a reviewer accepts non-errors via "
        "`voice-suite review --engine <name>`. Cloud engines' asr_ms "
        "includes network time.",
        "",
        "| engine | n | WER raw | WER adj | sub | del | ins | accepted | asr_ms P50 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_engine_summary_row(name, rows)
              for name, rows in sorted(engines.items())]
    lines.append("")
    if len(engines) == 1:
        ((_, rows),) = engines.items()
        lines += [
            "| utt | WER raw | WER adj | ops | accepted |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(rows, key=lambda r: (-_adjudicated(r), -r["wer"])):
            lines.append(
                f"| {r['utt_id']} | {r['wer']:.4f} | {_adjudicated(r):.4f}"
                f" | {len(r['ops'])} | {len(r['accepted'])} |")
    else:
        names = sorted(engines)
        by_utt: dict[str, dict[str, float]] = {}
        for name, rows in engines.items():
            for r in rows:
                by_utt.setdefault(r["utt_id"], {})[name] = r["wer"]
        lines += [
            "Raw WER per utterance (— = not run on that engine):",
            "",
            "| utt | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        worst_first = sorted(
            by_utt, key=lambda u: -max(by_utt[u].values()))
        for utt_id in worst_first:
            cells = [f"{by_utt[utt_id][n]:.4f}" if n in by_utt[utt_id] else "—"
                     for n in names]
            lines.append(f"| {utt_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
