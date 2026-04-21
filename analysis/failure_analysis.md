# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 71
- **Tỉ lệ Pass/Fail:** 54.9% / 45.1%
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.81
  - Relevancy: 0.75
  - Hit Rate: 0.93
  - MRR: 0.85
- **Điểm LLM-Judge trung bình:** 3.62 / 5.0
- **Regression Decision:** APPROVE

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| synthetic-rag | 17 | Câu hỏi chuẩn vẫn còn case retrieval hoặc answer synthesis chưa đủ chính xác. |
| edge_ambiguous | 4 | Agent chưa hỏi làm rõ nhất quán khi ngữ cảnh thiếu. |
| adversarial_prompt_injection | 3 | Guardrail chống prompt injection còn yếu. |
| edge_out_of_context | 3 | Agent chưa từ chối suy đoán đủ mạnh ở câu hỏi ngoài tài liệu. |
| edge_conflicting_info | 3 | Cần trình bày song song hai nguồn thay vì chọn một vế. |
| adversarial_goal_hijacking | 1 | Agent dễ bị kéo ra ngoài scope hỗ trợ tài liệu. |
| multiturn_correction | 1 | Agent chưa xử lý phản biện/đính chính đủ mượt. |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case case-016: Nếu một nhân viên làm thêm vào ngày lễ, với lương giờ tiêu chuẩn là 100.000 VNĐ, thì tổng lương cho 5 giờ làm thêm trong ngày lễ sẽ là bao nhiêu?
1. **Symptom:** Điểm judge chỉ đạt 2.05 và status là `fail`.
2. **Why 1:** Câu trả lời chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa tối ưu ở case này.
4. **Why 3:** Case thuộc nhóm `synthetic-rag` nên đòi hỏi reasoning hoặc guardrail khác với câu hỏi fact đơn giản.
5. **Why 4:** Prompt/generation hiện tại ưu tiên trả lời nhanh, đôi lúc chưa kiểm tra đủ độ chắc chắn trước khi trả lời.
6. **Root Cause:** Câu hỏi chuẩn vẫn còn case retrieval hoặc answer synthesis chưa đủ chính xác.

### Case case-029: Số điện thoại hotline của IT Helpdesk là gì?
1. **Symptom:** Điểm judge chỉ đạt 2.05 và status là `fail`.
2. **Why 1:** Câu trả lời chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa tối ưu ở case này.
4. **Why 3:** Case thuộc nhóm `synthetic-rag` nên đòi hỏi reasoning hoặc guardrail khác với câu hỏi fact đơn giản.
5. **Why 4:** Prompt/generation hiện tại ưu tiên trả lời nhanh, đôi lúc chưa kiểm tra đủ độ chắc chắn trước khi trả lời.
6. **Root Cause:** Câu hỏi chuẩn vẫn còn case retrieval hoặc answer synthesis chưa đủ chính xác.

### Case case-028: Nếu tôi cần hỗ trợ IT vào thứ Bảy lúc 10:00 sáng, tôi nên liên hệ ai?
1. **Symptom:** Điểm judge chỉ đạt 2.13 và status là `fail`.
2. **Why 1:** Câu trả lời chưa khớp hoàn toàn với ground truth mong đợi.
3. **Why 2:** Retrieval metric có hit_rate=1.00, cho thấy grounding chưa tối ưu ở case này.
4. **Why 3:** Case thuộc nhóm `synthetic-rag` nên đòi hỏi reasoning hoặc guardrail khác với câu hỏi fact đơn giản.
5. **Why 4:** Prompt/generation hiện tại ưu tiên trả lời nhanh, đôi lúc chưa kiểm tra đủ độ chắc chắn trước khi trả lời.
6. **Root Cause:** Câu hỏi chuẩn vẫn còn case retrieval hoặc answer synthesis chưa đủ chính xác.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Nối retrieval metric thật vào benchmark thay vì placeholder.
- [x] Thêm offline multi-judge consensus để pipeline luôn benchmark được không cần API.
- [x] Chuẩn hóa report `summary.json`, `benchmark_results.json`, và spot-check report để phục vụ submit.
- [ ] Tiếp tục cải thiện câu trả lời ở edge cases bằng reranker semantic hoặc prompt clarify chuyên biệt.
