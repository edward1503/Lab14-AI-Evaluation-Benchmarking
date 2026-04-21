# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Tỉ lệ Pass/Fail:** 100.0% / 0.0%
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.97
  - Relevancy: 0.57
  - Hit Rate: 0.97
  - MRR: 0.96
- **Điểm LLM-Judge trung bình:** 4.51 / 5.0
- **Regression Decision:** APPROVE

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| paraphrase | 1 | Retriever phụ thuộc quá nhiều vào keyword literal. |
| ambiguous | 1 | Agent chưa hỏi làm rõ đủ tốt khi ngữ cảnh thiếu. |
| factoid | 1 | Pipeline nền tảng vẫn còn lỗi ở câu hỏi cơ bản. |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case case-046: Diễn giải lại giúp mình: Chính sách remote cũ khác gì bản 2026?
1. **Symptom:** Điểm judge chỉ đạt 3.76 và status là `pass`.
2. **Why 1:** Câu trả lời tạo ra chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa đủ tốt.
4. **Why 3:** Case thuộc nhóm `paraphrase` nên đòi hỏi guardrail hoặc reasoning chính sách tốt hơn câu hỏi thường.
5. **Why 4:** Prompting/generation hiện tại tối ưu cho trả lời nhanh hơn là kiểm tra độ chắc chắn trước khi trả lời.
6. **Root Cause:** Retriever và answer composer chưa phối hợp ổn định ở case khó.

### Case case-056: Muốn làm từ xa thêm thì hỏi ai?
1. **Symptom:** Điểm judge chỉ đạt 4.04 và status là `pass`.
2. **Why 1:** Câu trả lời tạo ra chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa đủ tốt.
4. **Why 3:** Case thuộc nhóm `ambiguous` nên đòi hỏi guardrail hoặc reasoning chính sách tốt hơn câu hỏi thường.
5. **Why 4:** Prompting/generation hiện tại tối ưu cho trả lời nhanh hơn là kiểm tra độ chắc chắn trước khi trả lời.
6. **Root Cause:** Thiếu cơ chế xác định ngưỡng mơ hồ để buộc agent hỏi lại.

### Case case-045: Memo remote năm 2024 quy định mấy ngày và còn hiệu lực không?
1. **Symptom:** Điểm judge chỉ đạt 4.16 và status là `pass`.
2. **Why 1:** Câu trả lời tạo ra chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa đủ tốt.
4. **Why 3:** Case thuộc nhóm `factoid` nên đòi hỏi guardrail hoặc reasoning chính sách tốt hơn câu hỏi thường.
5. **Why 4:** Prompting/generation hiện tại tối ưu cho trả lời nhanh hơn là kiểm tra độ chắc chắn trước khi trả lời.
6. **Root Cause:** Retriever và answer composer chưa phối hợp ổn định ở case khó.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Thêm retrieval theo synonym và ưu tiên tài liệu hiện hành để giảm lỗi paraphrase/conflict.
- [x] Thêm cơ chế abstain cho câu hỏi ambiguous và out-of-context.
- [x] Bổ sung multi-judge consensus và regression gate để chặn release khi chất lượng giảm.
- [ ] Tiếp tục thêm reranker semantic hoặc embedding thực để tăng MRR ở case khó.
