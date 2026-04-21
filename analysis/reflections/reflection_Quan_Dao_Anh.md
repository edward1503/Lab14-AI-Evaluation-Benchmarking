# Báo cáo Cá nhân — Lab 14: AI Evaluation Factory

- **Họ tên:** Quan Dao Anh
- **Vai trò dự kiến:** Thành viên nhóm Lab 14
- **Trạng thái:** Bản nháp cần cá nhân hóa trước khi nộp chính thức

## 1. Nhiệm vụ chính

Tôi tham gia vào quá trình hoàn thiện pipeline evaluation của nhóm, đặc biệt ở khâu kiểm thử end-to-end và rà soát artifact trước khi submit.

## 2. Đóng góp kỹ thuật

- Tham gia kiểm tra flow `data/synthetic_gen.py` → `main.py` → `check_lab.py`.
- Hỗ trợ xác minh tính nhất quán giữa retrieval IDs, dataset metadata và report cuối cùng.
- Rà soát các hard cases như ambiguous, out-of-context, conflicting info để đảm bảo benchmark bao phủ các failure modes quan trọng.

## 3. Hiểu biết kỹ thuật

- Hit Rate cho biết chunk đúng có xuất hiện trong top-k retrieval hay không.
- Failure clustering giúp gom lỗi theo pattern để tìm root cause nhanh hơn.
- Spot check report hữu ích để xác định cases mà judge consensus chưa thực sự đáng tin cậy.

## 4. Bài học rút ra

- Chất lượng nộp bài phụ thuộc nhiều vào khả năng tái chạy pipeline chứ không chỉ code có sẵn.
- Các artifact phân tích cần được sinh từ dữ liệu thật, không thể để placeholder.

## 5. Việc cần cá nhân hóa trước khi nộp

- Bổ sung tên module hoặc commit tôi trực tiếp phụ trách.
- Thêm ví dụ vấn đề tôi đã xử lý trong quá trình làm lab.
