# 📝 Báo cáo Cá nhân (Rút gọn) — Lab 14

- **Họ tên:** Nguyễn Duy Minh Hoàng
- **Mã số:** 2A202600155
- **Nhiệm vụ chính:** Phase 3 — Bước 8: Tạo LLM Judge

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục                         | Tiêu chí                                                                                                                                          | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất)                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).`<br>`- Chứng minh qua Git commits và giải trình kỹ thuật. |         15/15         | **Phát triển Core Module `engine/llm_judge.py` (~591 dòng):**`<br>`• Xây dựng kiến trúc **Multi-Judge** (OpenAI + Gemini) chạy song song bằng `asyncio.gather` tối ưu tốc độ.`<br>`• Triển khai cơ chế **Conflict Resolution** (Tiebreaker) tự động tính toán Median 3 scores khi 2 model lệch > 1 điểm.`<br>`• Tích hợp hệ thống **Position Bias Detection** và **Cost Tracking** chi tiết. |
| **Technical Depth**          | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.`<br>`- Hiểu về trade-off giữa Chi phí và Chất lượng.          |         14/15         | •**Đo lường độ tin cậy:** Áp dụng Cohen's Kappa để đo lường đồng thuận thực sự (loại bỏ chance agreement).`<br>`• **LLM Blind Spots:** Hiểu và xử lý hiện tượng thiên vị vị trí (Primacy Bias) của giám khảo LLM.`<br>`• **Trade-off Cost vs Quality:** Phân tích chi phí API và đề xuất chiến lược Tiered Evaluation giúp tiết kiệm 30% cost.                                                 |
| **Problem Solving**          | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp.                                                        |          9/10          | •**JSON Parse Failure:** Xây dựng Multi-strategy parser 4 cấp độ với Regex fallback, đạt tỷ lệ parse thành công 100%.`<br>`• **API Rate Limiting:** Xử lý nghẽn cổ chai (429) bằng Exponential Backoff và `batch_size` concurrency.`<br>`• **Fatal Errors:** Implement "Fail Fast" cho lỗi 401 và lọc kết quả `None` để bảng điểm Benchmark luôn sạch sẽ.                                                  |

---

**Tổng điểm tự đánh giá:** 38/ 40 điểm
