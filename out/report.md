# voice-suite Leaderboard

Ranked by e2e_adequacy (desc), then WER (asc).

| Rank | Impl | n | e2e_adequacy↑ | WER↓ | mt_adequacy↑ | mt_fluency↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | errors |
|------|------|---|---------------|------|--------------|-------------|----------------|---------------|----------------|--------|
| 1 | m2v_sherpa | 15 | 0.443 | 0.186 | 0.445 | 0.640 | 394/596 | 784/1171 | 9166/15272 | 1 |
| 2 | m2v_phowhisper | 15 | 0.301 | 0.271 | 0.301 | 0.533 | 58427/67021 | 512/1043 | 4171/5521 | 0 |

**Note:** Judge means exclude judge-errored records; pipeline errors count as zeros.
- `m2v_sherpa` = sherpa-onnx (ASR) + Marian ONNX (MT) + StyleTTS2 (TTS)
- `m2v_phowhisper` = PhoWhisper (ASR) + Marian ONNX (MT) + Supertonic (TTS)

**Best implementation: `m2v_sherpa`**

## Metric definitions

ASR (automatic speech recognition), MT (machine translation), ms (milliseconds), P50/P95 (50th/95th percentile). Rows marked (LLM judge) are scored by Claude (a large language model) reading the text and rating it 0.0-1.0, rather than a fixed formula like WER; run with `score --no-judge` and they're always 0.0.

| Metric | Meaning | Direction |
|---|---|---|
| `n` | Number of evaluated utterances | — |
| `WER` | ASR transcript vs. reference transcript word error rate: edits ÷ reference word count, capped at 1.0 (1.0 = 100%, i.e. fully wrong) | lower is better |
| `mt_adequacy` | Does the MT output preserve the reference translation's meaning? (LLM judge) | higher is better |
| `mt_fluency` | Is the MT output natural, grammatical English? (LLM judge) | higher is better |
| `e2e_adequacy` | Does the synthesized output audio (re-heard via back-transcription) still convey the reference meaning? (LLM judge) | higher is better |
| `asr_ms` / `mt_ms` / `tts_ms` | Per-utterance wall-clock time per pipeline stage, P50/P95 across non-crashed runs | lower is faster |
| `errors` | Utterances where the impl itself crashed/failed to produce output | lower is better |

Pipeline-error and judge-error records count as `0.0` in the judge-score means (crashing configs rank worse, not excluded). Full depth: `docs/metrics.md`.

## Analysis

- **WER**: `m2v_sherpa` leads at 0.186 vs `m2v_phowhisper` at 0.271 (0.085 gap).
- **mt_adequacy**: `m2v_sherpa` leads at 0.445 vs `m2v_phowhisper` at 0.301 (0.143 gap).
- **mt_fluency**: `m2v_sherpa` leads at 0.640 vs `m2v_phowhisper` at 0.533 (0.107 gap).
- **e2e_adequacy**: `m2v_sherpa` leads at 0.443 vs `m2v_phowhisper` at 0.301 (0.142 gap).
- **asr_ms P50**: `m2v_sherpa` is 148.3x faster than `m2v_phowhisper` (394 vs 58427 ms).
- **mt_ms P50**: `m2v_phowhisper` is 1.5x faster than `m2v_sherpa` (512 vs 784 ms).
- **tts_ms P50**: `m2v_phowhisper` is 2.2x faster than `m2v_sherpa` (4171 vs 9166 ms).
- **errors**: `m2v_sherpa`: 1/15 (6.7%).
- **errors**: `m2v_phowhisper`: 0/15 (0.0%).
- **shared difficulty**: 3 utterance(s) challenged more than one impl (not impl-specific): fleurs-001805_male_southern_mix_sentence, fleurs-001891_male_southern_complex_sentence, fleurs-001997_male_southern.


## Worst utterances (lowest e2e_adequacy, pooled across all impls)

### fleurs-001997_male_southern — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): tuy nhiên dường như tỷ lệ xdr-tb trong tổng số người mắc bệnh lao vẫn còn ở mức thấp chỉ có 6.000 trong tổng số 330.000 người bị nhiễm bệnh tại một thời điểm cụ thể bất kỳ tại nam phi
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): however the percentage of xdr-tb in the entire group of people with tuberculosis still seems to be low 6,000 of the total 330,000 people infected at any particular moment in south africa
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001891_male_southern_complex_sentence — m2v_phowhisper (e2e 0.05, WER 1.00)
- ref (vi): lakkha singh cũng trình bày chhappan bhog bhajan ca kĩ raju khandelwal hát cùng ông
- asr (vi): Lát kha sinh cũng trình bài, chắp ban, hông, há danh, ca sĩ gai du, khán đeu át hét cùng ông.
- mt (en): Later, he was presented, with a rash, hip, famed, a busy singer, and a loud audience.
- audio back-transcript (en): Later he was presented with a rash, hip, famed, a busy singer, and a loud audience.
- ref translation (en): lakkha singh presented the chhappan bhog bhajan as well singer raju khandelwal was accompanying him

### fleurs-001891_male_southern_complex_sentence — m2v_sherpa (e2e 0.10, WER 0.67)
- ref (vi): lakkha singh cũng trình bày chhappan bhog bhajan ca kĩ raju khandelwal hát cùng ông
- asr (vi): lat kha xin cũng trình bày ban hong han ca sĩ gaiu khan dat hát cùng ông.
- mt (en): In fact, the singers sang along with him.
- audio back-transcript (en): In fact, the singers sang along with him.
- ref translation (en): lakkha singh presented the chhappan bhog bhajan as well singer raju khandelwal was accompanying him

### fleurs-001805_male_southern_mix_sentence — m2v_phowhisper (e2e 0.10, WER 0.60)
- ref (vi): xe buýt scotturb số 403 thường xuyên chạy đến sintra dừng ở cabo da roca
- asr (vi): Sabit. Có thất số 403 thường xuyên chạy đến sinh cha dựng ở Kabul, Daraka.
- mt (en): There's a 403-problem failure that regularly runs to the birth of a father in The City of The City of - This is the true God, - and the United States.
- audio back-transcript (en): There's a 403 problem failure that regularly runs to the birth of a father in the city of the city of, this is the true God, and the United States.
- ref translation (en): scotturb bus 403 travels regularly to sintra stopping at cabo da roca

### fleurs-001805_male_southern_mix_sentence — m2v_sherpa (e2e 0.10, WER 0.60)
- ref (vi): xe buýt scotturb số 403 thường xuyên chạy đến sintra dừng ở cabo da roca
- asr (vi): xe buýt số bốn trăm linh ba thường xuyên chạy đến xina dừng ở kboda rốt ca.
- mt (en): The four hundred-year-old bus regularly runs to the back of the shift at the kboda.
- audio back-transcript (en): the four-hundred-year-old bus regularly runs to the back of the shift at the kboda
- ref translation (en): scotturb bus 403 travels regularly to sintra stopping at cabo da roca

### fleurs-001997_male_southern — m2v_phowhisper (e2e 0.10, WER 0.32)
- ref (vi): tuy nhiên dường như tỷ lệ xdr-tb trong tổng số người mắc bệnh lao vẫn còn ở mức thấp chỉ có 6.000 trong tổng số 330.000 người bị nhiễm bệnh tại một thời điểm cụ thể bất kỳ tại nam phi
- asr (vi): Tuy nhiên, dù như tới lại, ít dì gà tới bay, trong tổng số người mắc bạn nào vẫn còn ở mức thách, chỉ có 6.000 trong tổng số 330.000 người bị nhiễm bệnh tại một thời điểm cựu thải, bác kỳ tại Nam Phi.
- mt (en): However, even if they do come, few of them will fly, and of all those with whom you are still at the test, only 6,000 of the 30,000 people who are infected at a time of waste, in South Africa.
- audio back-transcript (en): However, even if they do come few of them will fly, and of all those with whom you are still at the test, only 6,000 of the 30,000 people who are infected at a time of waste in South Africa.
- ref translation (en): however the percentage of xdr-tb in the entire group of people with tuberculosis still seems to be low 6,000 of the total 330,000 people infected at any particular moment in south africa

### fleurs-001835_male_southern_mix_sentence — m2v_phowhisper (e2e 0.10, WER 0.29)
- ref (vi): không có cảnh báo sóng thần nào được đưa ra và theo cơ quan địa vật lý jakarta sẽ không có cảnh báo sóng thần vì cơn địa chấn chưa đạt tới tiêu chuẩn 6.5 độ richter
- asr (vi): Không có cảnh báo sống thần nào được đưa ra và theo cơ quan địa vật lý, Daraka sẽ không có cảnh báo sống thần vì cơ địa chống chưa đã cơ tư trưởng 6.5 độ hít tơ.
- mt (en): No warning of any spirit life or spirit is given and, according to the geophysics, Daraka will not have any warning of life because the anti-restrial facility has not yet taken a 6.5-degree shot of silk.
- audio back-transcript (en): No warning of any spirit life or spirit is given and, according to the geophysics, Duraka will not have any warning of life because the antirestrial facility has not yet taken a 6.5-degree shot of silk.
- ref translation (en): no tsunami warning has been issued and according to the jakarta geophysics agency no tsunami warning will be issued because the quake did not meet the magnitude 6.5 requirement

### fleurs-001757_male_southern — m2v_phowhisper (e2e 0.10, WER 0.18)
- ref (vi): chỉ các đột biến ở tế bào dòng vi khuẩn mới có thể di truyền sang con cái trong khi đột biến ở nơi khác có thể gây chết tế bào hay ung thư
- asr (vi): Chỉ các độc biến ở tế bào dòng vi phẩm mới có thể di chuyển sang con cái. Trong khi độc biến ở nơi, khác có thể gây chết tới bào hay ông thư.
- mt (en): Only toxins in microcellular cells can move to females, while toxins are in different places, which can kill cells or mailmen.
- audio back-transcript (en): Only toxins in microcellular cells can move to females, while toxins are in different places, which can kill cells or mailmen.
- ref translation (en): only mutations in germ-line cells can be passed on to children while mutations elsewhere can cause cell-death or cancer

### fleurs-001676_male_southern — m2v_sherpa (e2e 0.10, WER 0.04)
- ref (vi): cũng như tất cả mọi công viên quốc gia của nam phi hoạt động bảo tồn diễn ra hàng ngày và phải mua vé vào cửa công viên
- asr (vi): cũng như tất cả mọi công viên quốc gia của nam phi hoạt động bảo tồn diễn ra hằng ngày và phải mua vé vào cửa công viên.
- mt (en): Just like every male-pattern male- state park is going on every day and has to buy tickets to the park door.
- audio back-transcript (en): just like every male pattern male state park is going on every day and has to buy tickets to the park door
- ref translation (en): as with all south african national parks there are daily conservation and entry fees for the park

### fleurs-001714_male_southern — m2v_phowhisper (e2e 0.15, WER 0.34)
- ref (vi): vệ tinh được đưa vào vũ trụ bằng tên lửa các nhà khoa học sử dụng kính thiên văn trong không gian vì bầu khí quyển của trái đất làm biến dạng một số ánh sáng và hình ảnh quan sát của chúng ta
- asr (vi): Vệ tin được đưa vào vũ trụ văn tin lỡ các nhà phá học sử dụng kiến tiên nhân trong phong gian vì bầu khuy biển cổ trái đất làm biến nhà một số ánh sáng và hình ảnh hoa xác của chúng ta.
- mt (en): The Guardians are being brought into the text universe, and they're missing the scientists using the eye seers in space because the ancient Earth sea-spot changes home some of our light and image of our bodies.
- audio back-transcript (en): The Guardians are being brought into the text universe, and they're missing the scientists using the eye seers in space, because the ancient Earth, C-Spot changes home some of our light and image of our bodies.
- ref translation (en): the satellite was sent into space by a rocket scientists use telescopes in space because the earth's atmosphere distorts some of our light and view
