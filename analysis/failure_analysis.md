# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark

- **Tổng số cases:** 71
- **Tỉ lệ Pass/Fail:** 52/71 pass, 19/71 fail
- **Điểm Retrieval trung bình:**
  - Hit Rate: **0.901**
  - Agreement Rate của Judge: **0.897**
- **Điểm LLM-Judge trung bình:**
  - V1: **3.993 / 5.0**
  - V2: **4.063 / 5.0**
  - Delta: **+0.070**
- **Quyết định Regression Gate:** `APPROVE`

### Nhận định nhanh

Pipeline hiện đã vượt qua ngưỡng submit vì:

- Benchmark chạy được cho toàn bộ 71 cases.
- Retrieval stage nhìn chung mạnh, Hit Rate đạt hơn 90%.
- V2 tốt hơn V1 cả về điểm trung bình lẫn độ đồng thuận giữa hai judges.

Tuy vậy, 19 case fail cho thấy chất lượng agent chưa đồng đều ở các nhóm câu hỏi khó hơn, đặc biệt là:

- câu hỏi suy luận nhiều điều kiện,
- câu hỏi yêu cầu arithmetic theo ngày làm việc,
- prompt adversarial hoặc ngoài phạm vi,
- multi-turn carry-over / correction,
- và các case cần trả lời rõ ngoại lệ thay vì từ chối chung chung.

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Dấu hiệu | Nguyên nhân dự kiến |
|----------|----------|----------|---------------------|
| **Over-Abstention trên câu hỏi in-scope** | 10 | Agent trả lời "Không tìm thấy trong tài liệu" hoặc "Vui lòng liên hệ bộ phận liên quan" dù retrieval đúng | Prompt fallback quá bảo thủ, answer synthesis không tận dụng được context đã retrieve |
| **Adversarial / Out-of-Scope Handling chưa tinh** | 6 | Agent từ chối chung chung, không tách phần hợp lệ và phần ngoài phạm vi | Guardrail có nhưng chưa biết "giữ phần hợp lệ, từ chối phần lệch mục tiêu" |
| **Reasoning theo lịch làm việc / thời gian** | 2 | Sai ở câu cần tính ngày làm việc như thứ Sáu → thứ Hai hoặc deadline xin nghỉ phép | Retrieval đúng nhưng reasoning layer chưa chuẩn về business-day logic |
| **Incomplete Policy Reasoning** | 1 | Trả lời đúng một phần nhưng suy diễn sai ở điều kiện đặc biệt | Agent nối các policy fragments chưa đủ chặt, bỏ sót ràng buộc trong exception handling |

### Kết luận từ clustering

Điểm đáng chú ý là **retrieval không phải vấn đề lớn nhất** của repo hiện tại. Phần lớn fail không đến từ việc không tìm thấy chunk đúng, mà đến từ việc:

1. Agent **không dám trả lời** dù đã có context đúng.
2. Agent **không tổng hợp được nhiều điều kiện** trong cùng một câu hỏi.
3. Prompt fallback hiện tại tối ưu cho an toàn hơn là coverage.

Điều này nghĩa là vòng cải tiến tiếp theo nên ưu tiên **generation policy và reasoning control**, không chỉ tối ưu vector retrieval.

---

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Nếu sản phẩm bị lỗi do nhà sản xuất nhưng khách hàng đã mở seal sản phẩm trước khi yêu cầu hoàn tiền, khách hàng có được hoàn tiền không?

1. **Symptom:** Agent fail với điểm judge **1.0/5**, dù đây là câu hỏi quan trọng về exception trong refund policy.
2. **Why 1:** Agent trả lời fallback "không tìm thấy" thay vì kết luận rõ là **không được hoàn tiền**.
3. **Why 2:** Câu hỏi yêu cầu ghép hai điều kiện nằm ở nhiều phần của policy: "lỗi do nhà sản xuất" và "đã mở seal".
4. **Why 3:** Agent retrieve được context liên quan nhưng không tổng hợp đủ logic AND/OR giữa điều kiện đủ và điều kiện loại trừ.
5. **Why 4:** Prompt hiện tại ưu tiên tránh hallucination, nên khi gặp câu có nhiều điều kiện chồng nhau agent chọn abstain.
6. **Root Cause:** **Generation reasoning chưa đủ mạnh cho policy exception handling**; agent chưa có chiến lược tổng hợp điều kiện từ nhiều bullet/section trong cùng tài liệu.

### Case #2: [Lịch sử hội thoại] ... Người dùng: Ai là những đối tượng có thể yêu cầu quyền truy cập vào nó?

1. **Symptom:** Agent fail với điểm judge **1.0/5** ở case multi-turn carry-over.
2. **Why 1:** Agent không dùng được ngữ cảnh của lượt trước, nên trả lời như một câu hỏi rời rạc.
3. **Why 2:** Câu hỏi hiện tại chỉ dùng đại từ "nó", phụ thuộc hoàn toàn vào lịch sử hội thoại.
4. **Why 3:** Pipeline benchmark đã đưa history vào trường `test_case`, nhưng agent vẫn chưa có logic tách và hiểu explicit conversation state.
5. **Why 4:** Retrieval hiện chủ yếu bám keyword ở câu cuối, nên mất lợi thế của turn trước.
6. **Root Cause:** **Agent chưa hỗ trợ multi-turn reasoning một cách thực sự**, dù dataset đã có loại case này.

### Case #3: Nếu một nhân viên cần nghỉ phép 1 ngày vào thứ Hai, họ phải gửi yêu cầu vào ngày nào để đảm bảo yêu cầu được xử lý đúng hạn?

1. **Symptom:** Agent fail với điểm judge **2.0/5** vì đưa ra mốc thời gian chưa chính xác.
2. **Why 1:** Agent hiểu đúng policy "ít nhất 3 ngày làm việc trước ngày nghỉ", nhưng suy luận theo lịch chưa đúng kỳ vọng của ground truth.
3. **Why 2:** Đây không còn là bài toán retrieval đơn thuần mà là **calendar/business-day reasoning**.
4. **Why 3:** Agent không có module chuyên xử lý ngày làm việc, cuối tuần và backward counting.
5. **Why 4:** Prompt generation hiện dựa vào văn bản thuần, không có bước xác minh kết quả suy luận thời gian trước khi trả lời.
6. **Root Cause:** **Thiếu reasoning layer cho temporal / business-day calculation**, khiến các câu hỏi policy theo lịch dễ sai dù grounded context đúng.

---

## 4. Kế hoạch cải tiến (Action Plan)

- [ ] **Giảm over-abstention:** sửa prompt để agent chỉ fallback khi thực sự không có bằng chứng, không fallback khi retrieval đã có chunk đúng.
- [ ] **Thêm answer synthesis theo rule:** với policy documents, cho phép agent tổng hợp nhiều bullet/section liên quan trước khi kết luận.
- [ ] **Tách rõ adversarial handling:** nếu câu hỏi gồm cả phần hợp lệ và phần ngoài phạm vi, agent phải trả lời phần hợp lệ trước rồi mới từ chối phần còn lại.
- [ ] **Bổ sung multi-turn parser:** trích lịch sử hội thoại thành context riêng thay vì nhét toàn bộ vào một chuỗi query.
- [ ] **Thêm temporal reasoning helper:** chuẩn hóa các case có "ngày làm việc", "thứ Hai", "trong vòng N ngày" để tránh sai logic lịch.
- [ ] **Ưu tiên hard-case regression set:** sau mỗi lần chỉnh prompt/agent, rerun riêng các nhóm fail hiện tại trước khi chạy full 71-case benchmark.

---

## 5. Kết luận cuối

Repo hiện tại đã đạt mức **submit được** vì:

- có benchmark results,
- có summary,
- có failure analysis,
- có regression decision rõ ràng,
- và benchmark chạy ra số liệu thật.

Tuy nhiên, failure analysis cho thấy roadmap cải tiến tiếp theo của team không nên tập trung quá nhiều vào chunking hay retrieval nữa. Với **Hit Rate ~90%**, điểm nghẽn lớn hơn nằm ở:

- prompt design,
- policy reasoning,
- multi-turn handling,
- và khả năng phân biệt giữa "không có thông tin" với "có thông tin nhưng khó tổng hợp".
