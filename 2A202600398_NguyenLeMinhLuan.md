# 📝 Báo cáo Cá nhân (Rút gọn) — Lab 14

- **Họ tên:** Nguyễn Lê Minh Luân
- **Mã số:** 2A202600398
- **Nhiệm vụ chính:** Phase — Agent Version: xây dựng và so sánh Agent V1/V2 trong `agent/main_agent.py`

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất) |
| :--- | :--- | :---: | :--- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).`<br>`- Chứng minh qua Git commits và giải trình kỹ thuật. | 14/15 | **Phụ trách Agent Version trong `agent/main_agent.py`:**`<br>`• Xây dựng RAG Agent có khả năng kết nối ChromaDB, dùng `OpenAIEmbeddings` đồng bộ với pipeline ingest và trả về đầy đủ `answer`, `contexts`, `retrieved_ids`, `metadata` để phục vụ benchmark.`<br>`• Thiết kế baseline **V1** theo hướng yếu hơn: retrieval top-k đơn giản, prompt cơ bản, final answer chưa tối ưu để làm mốc regression.`<br>`• Thiết kế **V2** theo hướng tối ưu hơn: dùng model tốt hơn (`gpt-4o-mini`), tăng chất lượng grounding, ưu tiên context liên quan, prompt chặt chẽ hơn và output phù hợp hơn cho LLM Judge.`<br>`• Tích hợp với `main.py` để chạy so sánh V1/V2, tạo `reports/summary.json` và `reports/benchmark_results.json` cho release gate. |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.`<br>`- Hiểu về trade-off giữa Chi phí và Chất lượng. | 14/15 | • **MRR và Hit Rate:** Hiểu retrieval không chỉ cần lấy đúng tài liệu mà còn cần đưa chunk đúng lên vị trí cao; MRR cao giúp LLM thấy bằng chứng quan trọng sớm hơn, giảm trả lời sai.`<br>`• **RAG grounding:** Agent trả lời dựa trực tiếp trên context, nếu không có thông tin thì fallback thay vì hallucinate.`<br>`• **Regression mindset:** V1 đóng vai trò baseline; V2 chỉ nên được approve khi điểm judge, retrieval và agreement không giảm.`<br>`• **Cost vs Quality:** V2 dùng model tốt hơn để tăng chất lượng câu trả lời, nhưng cần release gate để kiểm soát chi phí/latency và tránh nâng cấp model khi chất lượng không cải thiện rõ. |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 9/10 | • **Sai context làm sai câu trả lời:** Trả về `retrieved_ids` và `contexts` để truy vết lỗi nằm ở retrieval hay generation.`<br>`• **V2 chưa chắc tốt hơn V1:** Dùng benchmark hồi quy trong `main.py`; nếu delta âm thì decision là `BLOCK`, tránh release cảm tính.`<br>`• **Agent cần chạy trong pipeline async:** Hàm `query()` dùng `ainvoke()` để tương thích `BenchmarkRunner.run_all()` chạy theo batch bằng `asyncio.gather`.`<br>`• **Thiếu dữ liệu trong context:** Prompt yêu cầu trả lời fallback rõ ràng khi tài liệu không chứa đáp án, giảm nguy cơ bịa thông tin. |

---

## Trả lời 3 câu hỏi cá nhân theo Rubric

### 1. Engineering Contribution

Đóng góp chính của em nằm ở phần **Agent Version**. Em xây dựng agent RAG trong `agent/main_agent.py` để benchmark có thể gọi cùng một interface cho cả V1 và V2:

- `query(question)` nhận câu hỏi và trả về câu trả lời, context, danh sách `retrieved_ids` và metadata.
- Retrieval dùng ChromaDB để lấy các chunk liên quan.
- Generation dùng `ChatOpenAI` và prompt RAG để trả lời dựa trên context.
- `main.py` khởi tạo hai phiên bản agent để so sánh regression: V1 baseline và V2 optimized.

V1 được dùng làm mốc yếu hơn: retrieval đơn giản, prompt chưa tối ưu, model baseline. V2 hướng tới chất lượng tốt hơn: model tốt hơn, prompt chặt hơn, câu trả lời bám context hơn và dễ được judge đánh giá chính xác hơn.

### 2. Technical Depth

Trong task này, em tập trung vào mối liên hệ giữa **Retrieval Quality** và **Answer Quality**. Nếu retrieval lấy sai chunk, LLM có thể vẫn trả lời trôi chảy nhưng sai nội dung. Vì vậy agent phải trả ra `retrieved_ids` để hệ thống tính Hit Rate/MRR và phân tích lỗi.

- **Hit Rate** cho biết agent có lấy được chunk đúng hay không.
- **MRR** cho biết chunk đúng nằm ở vị trí thứ mấy; chunk đúng càng lên cao thì xác suất trả lời đúng càng tốt.
- **LLM Judge Agreement** giúp kiểm tra độ ổn định khi nhiều judge cùng đánh giá một câu trả lời.
- **Cost vs Quality** là trade-off quan trọng: V2 tốt hơn nhưng phải chứng minh bằng benchmark, không chỉ đổi model đắt hơn.

### 3. Problem Solving

Vấn đề lớn nhất của Agent Version là **không thể giả định V2 luôn tốt hơn V1**. Vì vậy em dùng quy trình regression: chạy cùng golden set cho V1 và V2, tính `avg_score`, `hit_rate`, `judge_agreement`, sau đó đưa ra quyết định `APPROVE` hoặc `BLOCK`.

Khi V2 chưa vượt V1, hướng xử lý là:

- kiểm tra `retrieved_ids` để xác định retrieval có lấy đúng tài liệu không;
- tăng chất lượng retrieval bằng top-k phù hợp hoặc reranking;
- sửa prompt để yêu cầu trả lời đủ điều kiện, ngoại lệ, thời hạn và nguồn thông tin;
- giữ fallback khi context không đủ để tránh hallucination;
- chỉ approve V2 khi benchmark chứng minh V2 > V1.

---

**Tổng điểm tự đánh giá:** 37/40 điểm
