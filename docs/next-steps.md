# Next Steps — decided 2026-07-22 (from `out/report.md`, n=15)

**Decision: ship `m2v_sherpa`.** Quality is tied with nemotron once sherpa's one infra crash is excluded (e2e ~0.475 vs 0.465, WER ~0.13 vs 0.178), and its ASR is ~98x faster on CPU (394 ms vs 38.7 s P50). Revisit nemotron only if a CUDA path lands.

## 1. MT — Priority 1 (the bottleneck)

- `mt_adequacy` 0.445: half the meaning is lost at the MT stage, even on clean input.
- Proof: `fleurs-001676` — WER 0.04 (near-perfect transcript) still scored e2e 0.10
  ("công viên quốc gia của nam phi" → "male-pattern male- state park").
  Numbers break too: "số 403" → "four-hundred-year-old bus".
- Action: swap Marian for a stronger vi→en model — VietAI `envit5-translation` or
  `facebook/nllb-200-distilled-600M` — as a new impl (e.g. `m2v_sherpa_envit5`) so the
  leaderboard arbitrates.
- Do first: score Marian on the *reference* transcripts (not ASR output) to isolate its
  ceiling and set the baseline for judging replacements.

## 2. TTS — Priority 2 (stability, then latency)

- Fix the StyleTTS2 "server exited unexpectedly" crash — a single infra failure counted
  as 0.0 and flipped rank 1↔2 at n=15.
- Latency: 9.2 s P50 is ~90% of pipeline wall-clock (ASR+MT combined ≈ 1.2 s).
- Quality is fine: e2e 0.443 ≈ mt_adequacy 0.445 — synthesis loses no meaning.

## 3. ASR — Priority 3 (leave as-is)

- Sherpa at WER ~0.13 (excluding the crashed record) / 394 ms P50 is already the sweet spot.
- Further ASR gains are masked by MT — don't spend here until MT improves.

Also: scale the test set 15 → 200 sentences so one record can't flip a ranking again.
