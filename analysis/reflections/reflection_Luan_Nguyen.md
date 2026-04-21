# Báo cáo Cá nhân — Lab 14: AI Evaluation Factory

- **Họ tên:** Luan Nguyen
- **Vai trò dự kiến:** Thành viên nhóm Lab 14
- **Trạng thái:** Bản nháp cần cá nhân hóa trước khi nộp chính thức

## 1. Nhiệm vụ chính

Tôi tham gia vào quá trình xây dựng hệ thống benchmark cho AI Agent, phối hợp cùng nhóm ở các phần chuẩn bị dữ liệu, kiểm tra chất lượng pipeline và rà soát kết quả benchmark trước khi chốt bản nộp.

## 2. Đóng góp kỹ thuật

- Hỗ trợ kiểm tra corpus và các chunk tài liệu trong `data/doc/` và `data/chunks.json`.
- Phối hợp rà soát chất lượng của bộ `golden_set.jsonl` và hard cases.
- Tham gia xác nhận đầu ra benchmark, đặc biệt là các chỉ số retrieval, judge agreement và regression gate.

## 3. Hiểu biết kỹ thuật

- MRR giúp đo vị trí xuất hiện đầu tiên của chunk đúng trong kết quả retrieval.
- Agreement Rate phản ánh mức độ đồng thuận giữa hai judges.
- Regression gate giúp quyết định có nên chấp nhận agent mới hay chặn release khi chất lượng giảm.

## 4. Bài học rút ra

- Chỉ benchmark answer mà không benchmark retrieval thì khó xác định đúng nguyên nhân lỗi.
- Cần chuẩn hóa artifacts nộp bài như `reports/summary.json`, `reports/benchmark_results.json`, và `analysis/failure_analysis.md`.

## 5. Việc cần cá nhân hóa trước khi nộp

- Bổ sung module/file cụ thể mà tôi phụ trách.
- Điền thêm ví dụ kỹ thuật hoặc commit tiêu biểu của bản thân.
