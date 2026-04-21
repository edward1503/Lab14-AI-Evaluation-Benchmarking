# 📊 Failure Analysis Report - AI Evaluation Factory (Full Dataset Run)

## 1. Executive Summary
Hệ thống đã hoàn tất Benchmark trên toàn bộ **73 câu hỏi**. 
*   **V1 (Baseline)**: 3.95/5
*   **V2 (Optimized)**: 4.08/5
*   **Định hướng**: Quyết định **APPROVE** cho phiên bản V2 do có sự cải thiện về độ chính xác và tính trung thực (Faithfulness).

---

## 2. Failure Clustering (Phân nhóm lỗi chi tiết)

### ❌ Cụm 1: Lỗi tính toán Số học (Arithmetic failure)
*   **Tần suất**: 5.4% (4/73 cases)
*   **Ví dụ**: Tính lương làm thêm ngày lễ (100k/giờ * 5 giờ * 300%).
    *   **Agent**: 300,000 VNĐ.
    *   **Lý do**: LLM đôi khi bị "hallucination" ở bước nhân kết quả cuối cùng mặc dù các bước trung gian có vẻ đúng.
*   **Hậu quả**: Judge chấm 1 điểm cho Accuracy.

### ❌ Cụm 2: Quy trình thời gian & Ngày làm việc (Temporal Reasoning)
*   **Tần suất**: 8.2% (6/73 cases)
*   **Ví dụ**: Yêu cầu nghỉ phép/Cấp quyền gửi trước X ngày làm việc.
    *   **Agent**: Thường cộng trực tiếp số ngày vào ngày hiện tại mà không loại trừ Thứ 7, Chủ Nhật.
*   **Nguyên nhân**: Thiếu logic kiểm tra lịch làm việc (Calendar-aware reasoning).

### ❌ Cụm 3: Từ chối trả lời sai (False Refusals)
*   **Tần suất**: 4.1% (3/73 cases)
*   **Ví dụ**: Câu hỏi về điều kiện Level 2 cho nhân viên mới dưới 30 ngày.
    *   **Agent**: "Xin lỗi, tôi không thấy thông tin này..."
    *   **Thực tế**: Thông tin có trong Section 1 của tài liệu Access Control.
*   **Nguyên nhân**: Do Chunking có thể đã cắt nhỏ đoạn văn bản khiến Agent không xâu chuỗi được logic "Nếu < 30 ngày -> Level 1".

---

## 3. Judge Disagreement Analysis (Xung đột Judge)
Hệ thống ghi nhận **12 trường hợp Conflict** (Agreement Rate ~95% cho V2).

*   **Đặc điểm**: GPT-4o-mini thường trừ điểm nặng các lỗi sai số liệu (Accuracy = 1), trong khi GPT-3.5-turbo lại chấm 4-5 điểm nếu thấy Agent có nỗ lực giải thích (Reasoning).
*   **Kết luận**: GPT-4o-mini đóng vai trò "Judge khắt khe" giúp bảo vệ chất lượng dữ liệu, trong khi GPT-3.5-turbo giúp đánh giá tính thân thiện của phản hồi.

---

## 4. Đề xuất kỹ thuật (Next Steps)
1.  **Chain-of-Thought Prompting**: Nâng cấp System Prompt của Agent V2 để yêu cầu tính toán nháp trước khi đưa ra kết quả cuối.
2.  **Hybrid Search**: Kết hợp Keyword Search để đảm bảo các cụm từ như "30 ngày", "Level 2" luôn được truy xuất đúng chunk.
3.  **Refined Chunking**: Tăng kích thước chunk hoặc sử dụng overlapping chunks lên 20% để giữ ngữ cảnh logic tốt hơn.

---
*Báo cáo được cập nhật dựa trên kết quả Full Benchmark 73 cases.*
