# 📝 Báo cáo Cá nhân (Rút gọn) — Lab 14

- **Họ tên:** Vũ Quang Phúc
- **Mã số:** 2A202600346
- **Nhiệm vụ chính:** Phase Integration / Benchmark Orchestration — hoàn thiện pipeline benchmark, report artifacts và failure analysis

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất) |
| :--- | :--- | :---: | :--- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).`<br>`- Chứng minh qua Git commits và giải trình kỹ thuật. | 15/15 | **Phụ trách integration benchmark và artifacts nộp bài:**`<br>`• Hoàn thiện `main.py` để benchmark chạy end-to-end, đọc `golden_set`, chạy V1/V2, tổng hợp `summary.json` và `benchmark_results.json` phục vụ regression gate.`<br>`• Nối `engine/retrieval_eval.py` với pipeline benchmark để không còn metrics giả lập, đảm bảo `hit_rate` và `mrr` được tính từ retrieval IDs thật.`<br>`• Chuẩn hóa `engine/runner.py` để runner gom kết quả theo cấu trúc thống nhất (`test_case`, `ragas`, `judge`, `status`) giúp downstream reporting ổn định.`<br>`• Hoàn thiện `analysis/failure_analysis.md`, tạo `reports/spot_check.md`, và bổ sung reflection files còn thiếu để repo đạt checklist nộp bài.` |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.`<br>`- Hiểu về trade-off giữa Chi phí và Chất lượng. | 14/15 | • **MRR vs Hit Rate:** Em hiểu Hit Rate trả lời câu hỏi "có lấy được chunk đúng không", còn MRR trả lời "chunk đúng đứng ở vị trí nào"; đây là lý do retrieval eval phải đi cùng answer eval.`<br>`• **Regression mindset:** benchmark không chỉ để lấy score đẹp mà để so sánh V1/V2 và ra quyết định `APPROVE` hay `BLOCK` có căn cứ.`<br>`• **Judge reliability:** Agreement Rate cao chưa chắc đủ, nên cần thêm spot-check và đọc các case conflict để tránh tin mù vào LLM-as-Judge.`<br>`• **Trade-off cost/quality:** phần report phải vẫn chạy được kể cả khi API key lỗi hoặc không có mạng, nên integration cần có đường fallback để artifact nộp bài vẫn được sinh ổn định.` |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 10/10 | • **Repo thiếu artifact submit:** khi nhóm chưa có `reports/summary.json` và `reports/benchmark_results.json`, em ưu tiên khôi phục luồng benchmark end-to-end trước để `check_lab.py` pass.`<br>`• **Dataset thiếu retrieval ground truth rõ ràng:** em chuẩn hóa lại mapping retrieval IDs từ metadata hiện có để benchmark retrieval không còn chỉ là placeholder.`<br>`• **Report format không đồng nhất:** em gom lại cấu trúc kết quả benchmark để failure analysis, summary và runner cùng đọc được một schema nhất quán.`<br>`• **Nút thắt không nằm ở retrieval mà ở generation:** dựa trên benchmark_result, em phân tích rằng nhiều case fail do over-abstention và reasoning, từ đó viết failure analysis bám sát nguyên nhân thật thay vì mô tả chung chung.` |

---

## Trả lời 3 câu hỏi cá nhân theo Rubric

### 1. Engineering Contribution

Đóng góp chính của em nằm ở phần **benchmark integration và reporting**. Sau khi các thành viên khác đã có module dữ liệu, judge và agent, em tập trung làm cho toàn bộ repo chạy được như một hệ thống hoàn chỉnh:

- `main.py` đọc dataset, chạy benchmark cho hai phiên bản agent, tạo report và đưa ra quyết định regression.
- `engine/runner.py` chuẩn hóa output từng test case để các phần sau như summary, benchmark result, failure analysis dùng lại được.
- `engine/retrieval_eval.py` được nối vào benchmark thật để metric retrieval không còn là giá trị giả.
- Em cũng hoàn thiện các artifacts phục vụ nộp bài như `analysis/failure_analysis.md`, `reports/summary.json`, `reports/benchmark_results.json`.

Nói ngắn gọn, em phụ trách phần **đưa các module rời rạc của team thành một pipeline benchmark có thể submit được**.

### 2. Technical Depth

Với task của mình, em tập trung vào việc hiểu **mối liên hệ giữa retrieval, judge và release decision**:

- Nếu chỉ nhìn `avg_score` mà không nhìn `hit_rate` và `mrr`, nhóm rất dễ sửa nhầm chỗ. Ví dụ benchmark hiện tại cho thấy nhiều case fail không phải vì retrieval kém mà vì answer synthesis quá bảo thủ.
- Em dùng tư duy **regression testing**: V2 chỉ nên được approve khi tốt hơn V1 ít nhất ở chất lượng tổng quan, đồng thời không gây tụt retrieval hoặc judge agreement.
- Em cũng hiểu rằng **Agreement Rate không thay thế được human review**. Vì thế phần benchmark result cần hỗ trợ đọc các case conflict và failure clusters chứ không chỉ in ra một con số tổng.
- Về cost/chất lượng, em ưu tiên cho benchmark **chạy ổn định trước**, vì một hệ thống đánh giá không sinh ra được artifact thì dù dùng model tốt cũng chưa có giá trị submit.

### 3. Problem Solving

Vấn đề khó nhất em xử lý là repo ở trạng thái **có nhiều module khá tốt nhưng chưa khớp nhau**. Team đã có tiến độ ở từng phần, nhưng bài vẫn có nguy cơ không submit được nếu:

- report chưa sinh ra đúng chỗ,
- runner và report đọc khác schema,
- failure analysis vẫn là template,
- và retrieval metric chưa thực sự nối với dataset.

Cách em xử lý là:

1. Đọc `benchmark_results.json` để xác định pipeline đang fail ở đâu và fail theo pattern nào.
2. Chuẩn hóa cấu trúc output benchmark để các tầng `runner -> summary -> failure_analysis` dùng chung một schema.
3. Ưu tiên sửa những điểm chặn submit trước: reports, summary, benchmark result, failure analysis.
4. Sau đó mới dùng chính benchmark result để viết phân tích nguyên nhân gốc rễ thay vì viết theo mẫu chung.

Nhờ vậy repo không chỉ "có code", mà đã có **artifact, số liệu benchmark và báo cáo phân tích đúng format nộp bài**.

---

## Tóm tắt số liệu em dùng để viết báo cáo

- **Tổng số cases:** 71
- **V1 avg score:** 3.993
- **V2 avg score:** 4.063
- **Hit Rate:** 0.901
- **Judge Agreement:** 0.897
- **Regression Decision:** `APPROVE`

Những số liệu này cho thấy V2 đã cải thiện nhẹ so với V1, nhưng failure analysis chỉ ra rằng vòng tối ưu tiếp theo nên tập trung vào:

- giảm over-abstention,
- tăng policy reasoning,
- và xử lý tốt hơn các hard cases như multi-turn, adversarial và exception handling.

---

**Tổng điểm tự đánh giá:** 39/40 điểm
