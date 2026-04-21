# 📝 Báo cáo Cá nhân — Lab 14

- **Họ tên:** Nguyễn Đôn Đức
- **Mã số:** 2A202600145
- **GitHub:** [edward1503](https://github.com/edward1503)
- **Nhiệm vụ chính:** Phase 1 — Data Pipeline & **Phase 4 — Expert Features Optimization** (Async, Cost Tracking, Error Attribution).

Minh chứng: 
- Commit ID: 4d4b82fd3171f1063d04441269071d735c1b6f9c, message: chunking text
- Branch: [duc](https://github.com/edward1503/Lab14-AI-Evaluation-Benchmarking/tree/duc), nơi tổng hợp và finalize toàn bộ mã nguồn Expert (Async, Cost, Attribution) được merge vào main.
---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất) |
| :--- | :--- | :---: | :--- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).<br>- Chứng minh qua Git commits và giải trình kỹ thuật. | 15/15 | • **Data Pipeline:** Xây dựng `ingest.py`, xử lý Recursive Chunking (600 characters/50 overlap) và nạp vector vào ChromaDB đồng nhất với `chunks.json`.<br>• **Expert Upgrades:** Tối ưu hóa toàn diện Async (~1s/case), triển khai Multi-model Judge với Consensus logic. Tích hợp theo dõi `token_usage` và tính toán chi phí (USD) cho từng case chạy. |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.<br>- Hiểu về trade-off giữa Chi phí và Chất lượng. | 15/15 | • **Trade-off Chi phí vs Tốc độ:** Phân tích sự chênh lệch giữa gpt-4o-mini và gpt-3.5-turbo; sử dụng Async batching (size=10) để cân bằng giữa throughput và giới hạn RPM của API.<br>• **Error Attribution Logic:** Phát triển thuật toán đối chiếu tự động giữa `hit_rate` và `faithfulness` để cô lập lỗi nằm ở Retrieval hay Generation (Hallucination origin). |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 10/10 | • **Đảm bảo tính nhất quán:** Xử lý triệt để vấn đề "Chunk ID drift" giữa Vector DB và Golden Dataset.<br>• **Hệ thống Observability:** Giải quyết bài toán thiếu thông tin chi phí bằng cách bóc tách `usage` metadata từ API responses, tổng hợp báo cáo chuyên nghiệp ngay tại console cho `main.py` và `check_lab.py`. |

---

## Trả lời 3 câu hỏi cá nhân theo Rubric

### 1. Engineering Contribution

Đóng góp chính của em nằm ở việc xây dựng **Nền tảng dữ liệu** và **Tối ưu hóa hệ thống Expert**:
1. **Data Ingestion:** Load, chunk (RecursiveSplitter) và gán ID nhất quán cho tài liệu, tạo tiền đề để tính Hit Rate/MRR chính xác.
2. **Async Optimization:** Triển khai cơ chế xử lý song song thông minh, giúp benchmark 73 cases chỉ mất ~150 giây (giảm 70% latency).
3. **Observability:** Hiện thực hóa tính năng theo dõi chi phí thực tế cho từng lần evaluate, giúp người dùng kiểm soát ngân sách API hiệu quả.

### 2. Technical Depth

Em tập trung sâu vào việc phân tích lỗi và tối ưu chi phí:
- **Error Attribution:** Hiểu rằng nếu `faithfulness` thấp nhưng `hit_rate` = 1, lỗi hoàn toàn do LLM "bịa đặt" (Generation Error). Nếu `hit_rate` = 0, lỗi do hệ thống tìm kiếm (Retrieval Error). Việc tự động hóa phân loại này giúp team debug hệ thống RAG nhanh hơn.
- **Metric Insights:** Hiểu rõ MRR giúp đánh giá khả năng đưa thông tin đúng lên đầu trang của Retriever, từ đó giảm áp lực cho LLM khi phải lọc nhiễu từ các chunk sai.

### 3. Problem Solving

Vấn đề phức tạp nhất là **Đồng bộ hóa kết quả và Chi phí**:
- **Giải pháp:** Xây dựng một Wrapper bao quanh quá trình gọi API để thu thập metadata. Kết hợp dữ liệu từ Agent (v1/v2) và Judge (Multi-model) thành một luồng dữ liệu duy nhất, sau đó quy đổi ra USD dựa trên bảng giá chính thức. 
- **Kết quả:** Hệ thống `check_lab.py` hiện tại có thể in ra báo cáo tài chính cực kỳ chuyên nghiệp, giúp bài Lab đạt chuẩn doanh nghiệp thực tế.

---

**Tổng điểm tự đánh giá:** 40/40 điểm
