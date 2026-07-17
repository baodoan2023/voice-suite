# voice-suite Leaderboard

Ranked by e2e_adequacy (desc), then WER (asc). Judge means exclude judge-errored records; pipeline errors count as zeros.

| Rank | Impl | n | WER↓ | mt_adequacy↑ | mt_fluency↑ | e2e_adequacy↑ | asr_ms P50/P95 | mt_ms P50/P95 | tts_ms P50/P95 | errors |
|------|------|---|------|--------------|-------------|---------------|----------------|---------------|----------------|--------|
| 1 | m2v_default | 30 | 0.237 | 0.299 | 0.532 | 0.299 | 65982/80736 | 546/911 | 3289/5206 | 0 |
| 2 | m2v_sherpa | 30 | 0.501 | 0.285 | 0.338 | 0.285 | 299/682 | 746/1049 | 7250/13069 | 13 |

**Best implementation: `m2v_default`**

## Worst utterances (lowest e2e_adequacy)

### fleurs-001844 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): bề mặt của mặt trăng được tạo nên bởi đá và bụi lớp bên ngoài của mặt trăng được gọi là lớp vỏ
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): the surface of the moon is made of rocks and dust the outer layer of the moon is called the crust
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001876 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): phòng quản lý tình trạng khẩn cấp miền bắc marianas cho biết không có báo cáo nào về tổn thất ở quốc gia này
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): the northern marianas emergency management office said that there were no damages reported in the nation
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001878 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): cơ quan trung ương của giáo hội đã tọa lạc ở rome hơn một ngàn năm qua và sự tập trung quyền lực và tiền bạc này khiến nhiều người thắc mắc là liệu giáo lý nói trên có được tuân theo hay không
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): the central authority of the church had been in rome for over a thousand years and this concentration of power and money led many to question whether this tenet was being met
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001882 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): các nhà khoa học hy vọng sẽ hiểu được cách thức hình thành của các hành tinh đặc biệt là trái đất được hình thành như thế nào bởi các sao chổi đã va chạm vào trái đất từ xa xưa
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): scientists hope to understand how planets form especially how the earth formed since comets collided with the earth long ago
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001889 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): nó mang đến cho chúng ta xe lửa xe hơi và nhiều phương tiện di chuyển khác
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): it has brought us the train the car and many other transportation devices
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001891 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): lakkha singh cũng trình bày chhappan bhog bhajan ca kĩ raju khandelwal hát cùng ông
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): lakkha singh presented the chhappan bhog bhajan as well singer raju khandelwal was accompanying him
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001903 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): khi tất cả các nguồn lực hiện có được sử dụng một cách hiệu quả ở khắp các phòng ban có chức năng của một tổ chức khả năng sáng tạo và sự khôn khéo có thể được phát huy
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): when all available resources are effectively used across the functional departments of an organization creativity and ingenuity can transpire
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001916 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): nhiều món bánh nướng của đức còn có hạnh nhân quả phỉ và các loại hạt khác bánh càng ngon hơn khi ăn kèm một tách cà phê đậm đà
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): many german baked goods also feature almonds hazelnuts and other tree nuts popular cakes often pair particularly well with a cup of strong coffee
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001926 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): chấp nhận là quan điểm của aristotle về tất cả mọi vấn đề liên quan đến khoa học kể cả tâm lý học
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): accepted were aristotle's views on all matters of science including psychology
- error: StyleTTS2 server exited unexpectedly during synthesis

### fleurs-001934 — m2v_sherpa (e2e 0.00, WER 1.00)
- ref (vi): thường bạn luôn nghe tiếng của các du khách và người bán hàng rong câu chuyện về âm thanh và ánh sáng giống như một quyển sách truyện
- asr (vi): 
- mt (en): 
- audio back-transcript (en): 
- ref translation (en): usually you always here the sound of tourists and vendors the story of the sound and light is just like a story book
- error: StyleTTS2 server exited unexpectedly during synthesis
