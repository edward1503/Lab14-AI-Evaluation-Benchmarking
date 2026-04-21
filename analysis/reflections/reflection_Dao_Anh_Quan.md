# 📝 Báo cáo Cá nhân (Rút gọn) — Lab 14

- **Họ tên:** Đào Anh Quân
- **Mã số:** 2A202600028
- **Nhiệm vụ chính:** Phase 1 — Data / SDG: Tạo và kiểm tra Golden Dataset

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất) |
| :--- | :--- | :---: | :--- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).`<br>`- Chứng minh qua Git commits và giải trình kỹ thuật. | 14/15 | **Phụ trách Golden Dataset trong `data/synthetic_gen.py` (~513 dòng):**`<br>`• Thiết kế `SyntheticDataGenerator` chạy hoàn toàn async với `AsyncOpenAI` + `asyncio.Semaphore(5)` để sinh 71 cases song song, không bị rate limit.`<br>`• Xây dựng **7 loại hard case prompts** theo `HARD_CASES_GUIDE.md`: adversarial (prompt injection, goal hijacking), edge case (out-of-context, ambiguous, conflicting info), multi-turn (context carry-over, correction).`<br>`• Triển khai `_pick_one_per_doc()` và `_select_conflicting_pairs()` để đảm bảo đa dạng nguồn tài liệu và tính thực tế của test cases.`<br>`• Verify và fix ChromaDB (commit `3da9ce3`): phát hiện `data_level0.bin` phình 100× (62.84MB → 628KB) do ingest trùng lặp, re-ingest và validate lại toàn bộ chunk IDs.`<br>`• Fix `agent/main_agent.py` và `main.py` (commit `2448953`) để pipeline end-to-end chạy đúng. |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.`<br>`- Hiểu về trade-off giữa Chi phí và Chất lượng. | 14/15 | • **MRR vs Hit Rate:** Hit Rate chỉ đo "có tìm được không", MRR đo "tìm được ở vị trí nào" — chunk đúng càng lên đầu, LLM càng có context tốt để trả lời chính xác. Hệ thống đạt Hit Rate = 0.90.`<br>`• **Cohen's Kappa:** Loại bỏ yếu tố chance agreement ra khỏi Agreement Rate — κ = (Po − Pe) / (1 − Pe); nếu 2 judges đều hay chấm điểm trung bình, Agreement Rate ảo cao nhưng κ thấp.`<br>`• **Position Bias:** LLM judge có xu hướng ưu tiên câu trả lời xuất hiện trước (primacy bias) — phát hiện bằng cách hoán đổi thứ tự (A,B) → (B,A) và so sánh kết quả.`<br>`• **Trade-off SDG:** Dùng `temperature=0.8` cho generation (cần đa dạng) khác với judge dùng `0.0` (cần deterministic); seed `random.Random(42)` cố định đảm bảo reproducibility cho Regression Testing. |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 9/10 | • **ChromaDB phình kích thước:** `data_level0.bin` tăng 100× do ingest lặp không deduplicate → phát hiện qua `git diff --stat`, fix bằng xóa sạch collection và re-ingest có kiểm soát.`<br>`• **Chunk IDs bị stale:** Sau re-ingest, IDs mới khiến `golden_set.jsonl` cũ trỏ sai → viết cross-reference validation so sánh IDs trong ChromaDB vs JSONL, regenerate các cases bị lệch.`<br>`• **Conflicting info cases không thực sự mâu thuẫn:** Khi chunk được chọn ngẫu nhiên, 2 chunks thường nói về chủ đề khác nhau → sửa `_select_conflicting_pairs()` chỉ ghép chunk từ cùng source doc, ưu tiên chunk liền kề. |

---

## Trả lời 3 câu hỏi cá nhân theo Rubric

### 1. Engineering Contribution

Đóng góp chính của em nằm ở phần **Golden Dataset**. Em xây dựng toàn bộ `data/synthetic_gen.py` để sinh và validate bộ dữ liệu đánh giá:

- `SyntheticDataGenerator` dùng `AsyncOpenAI` + `asyncio.Semaphore` để sinh 71 QA cases song song với rate limit tự kiểm soát.
- RAG baseline (49 cases) phân bổ difficulty round-robin: easy → mixed → hard theo `(chunk_idx + pair_idx) % 3`.
- 22 hard cases chia thành 7 loại với prompt chuyên biệt cho từng loại, dùng `_pick_one_per_doc()` đảm bảo đa dạng nguồn.
- Sau khi sinh, kiểm tra toàn bộ `source_chunk_id` map sang ChromaDB để đảm bảo Hit Rate có thể tính đúng.

### 2. Technical Depth

Trong task này, em tập trung vào mối liên hệ giữa **chất lượng dataset** và **độ chính xác của evaluation pipeline**:

- **MRR** cho biết chunk đúng nằm ở vị trí nào trong kết quả retrieval — không chỉ "có hay không" như Hit Rate. Nếu chunk đúng bị đẩy xuống vị trí 3-4, LLM vẫn có thể hallucinate dù Hit Rate = 1.
- **Cohen's Kappa** loại bỏ phần đồng ý do ngẫu nhiên, phản ánh đúng hơn chất lượng đồng thuận thực sự giữa các judges.
- **Position Bias** là lỗi hệ thống của LLM-as-Judge cần phát hiện chủ động, không chỉ tin vào Agreement Rate.
- **Golden Dataset xấu** gây hỏng toàn bộ pipeline: ground truth sai → judge sai → release gate sai → production fail.

### 3. Problem Solving

Vấn đề phức tạp nhất là **dữ liệu ChromaDB bị inconsistent** do pipeline chạy nhiều lần. Hướng xử lý:

- Phát hiện qua `git diff --stat` thấy `data_level0.bin` tăng bất thường → kiểm tra kích thước thực tế → xác nhận duplicate.
- Re-ingest có kiểm soát: xóa sạch collection trước khi ingest mới, không dùng `upsert` mù quáng.
- Sau re-ingest, toàn bộ `source_chunk_id` trong JSONL cũ bị stale → so sánh cross-reference và regenerate cases bị affected.
- Bài học: **validate sớm, validate thường** — lỗi ở data layer nếu không bắt kịp sẽ lan xuống toàn bộ Retrieval Eval, LLM Judge và Regression Testing.

---

**Tổng điểm tự đánh giá:** 37/40 điểm
