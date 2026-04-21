# Báo cáo Cá nhân — Lab 14: AI Evaluation Factory

- **Họ tên:** edward1503
- **Vai trò dự kiến:** Thành viên nhóm Lab 14
- **Trạng thái:** Bản nháp cần cá nhân hóa trước khi nộp chính thức

## 1. Nhiệm vụ chính

Tôi tham gia vào việc hoàn thiện pipeline benchmark của nhóm, phối hợp ở các phần dữ liệu, runner và kiểm tra tính đầy đủ của bài nộp.

## 2. Đóng góp kỹ thuật

- Hỗ trợ kiểm tra cấu trúc `golden_set.jsonl` và metadata của test cases.
- Tham gia xác minh benchmark artifacts và rà soát sự khác biệt giữa V1 và V2.
- Góp phần kiểm tra các failure cases để chuẩn bị cho báo cáo 5 Whys.

## 3. Hiểu biết kỹ thuật

- Golden dataset cần gắn được ground-truth retrieval IDs thì retrieval eval mới có ý nghĩa.
- Async runner giúp xử lý nhiều test case nhanh hơn và phù hợp với benchmark quy mô lớn.
- Root cause analysis phải chỉ ra được lỗi nằm ở retrieval, prompt, hay answer synthesis.

## 4. Bài học rút ra

- Một repo có đủ file nhưng chưa chạy được end-to-end thì vẫn chưa sẵn sàng để submit.
- Benchmark tốt cần đồng thời đo answer quality, retrieval quality và judge reliability.

## 5. Việc cần cá nhân hóa trước khi nộp

- Cập nhật họ tên chính thức theo danh sách lớp.
- Bổ sung commit hoặc module thể hiện đóng góp cá nhân cụ thể.
