# 📝 Báo cáo Cá nhân (Rút gọn) — Lab 14

- **Họ tên:** Nguyễn Đôn Đức
- **Mã số:** 2A202600145
- **GitHub:** [edward1503](https://github.com/edward1503)
- **Nhiệm vụ chính:** Phase 1 — Data Pipeline: xây dựng `agent/ingest.py`, tạo `data/chunks.json` và nạp vector vào ChromaDB, Fix bug giai đoạn cuối cho nhóm để merge vào main.

---

## 👤 2. Điểm Cá nhân (Tối đa 40 điểm)

| Hạng mục | Tiêu chí | Điểm tự đánh giá | Giải trình kỹ thuật (Trích xuất) |
| :--- | :--- | :---: | :--- |
| **Engineering Contribution** | - Đóng góp cụ thể vào các module phức tạp (Async, Multi-Judge, Metrics).<br>- Chứng minh qua Git commits và giải trình kỹ thuật. | 15/15 | **Phụ trách Data Pipeline trong `agent/ingest.py`:**<br>• Load toàn bộ tài liệu `.txt` từ `data/doc/` bằng `TextLoader` với encoding UTF-8.<br>• Chia văn bản thành các chunk có kích thước 600 ký tự, overlap 50, dùng `RecursiveCharacterTextSplitter` với `add_start_index=True` để giữ metadata vị trí.<br>• Gán `chunk_id` dạng `chunk_{i}` nhất quán, export toàn bộ ra `data/chunks.json` trước khi embed để phục vụ SDG (Synthetic Data Generation).<br>• Nạp các chunk vào ChromaDB bằng `OpenAIEmbeddings(model="text-embedding-3-small")`, đảm bảo `ids` và `metadatas` khớp chính xác với `chunks.json` để Hit Rate / MRR tính đúng `retrieved_ids`. |
| **Technical Depth** | - Giải thích được các khái niệm: MRR, Cohen's Kappa, Position Bias.<br>- Hiểu về trade-off giữa Chi phí và Chất lượng. | 15/15 | • **Chunking ảnh hưởng trực tiếp đến Retrieval Quality:** Chunk quá lớn làm loãng thông tin quan trọng; chunk quá nhỏ mất ngữ cảnh. `chunk_size=600` và `overlap=50` là điểm cân bằng giúp duy trì context mà vẫn đủ hẹp để retrieval chính xác.<br>• **Ground Truth IDs:** `chunk_id` được gán cố định và export ra `chunks.json` để `golden_set.jsonl` có thể map sang `ground_truth_ids` — nền tảng để tính Hit Rate và MRR đúng.<br>• **Cost vs Quality:** `text-embedding-3-small` rẻ hơn `text-embedding-3-large` nhưng đủ chất lượng cho domain nội bộ; việc re-embed lại toàn bộ corpus tốn chi phí nên thiết kế chunk ổn định từ đầu là quan trọng.<br>• **MRR:** Nếu chunk đúng nằm ở rank 1 thì MRR = 1.0; nếu ở rank 3 thì MRR ≈ 0.33. Pipeline ingest cần đảm bảo chunk không bị cắt ngang câu trả lời để LLM tổng hợp đúng. |
| **Problem Solving** | - Cách giải quyết các vấn đề phát sinh trong quá trình code hệ thống phức tạp. | 10/10 | • **Chunk ID không ổn định:** Thứ tự file từ `os.listdir()` không xác định; giải pháp là enumerate toàn bộ sau khi load để `chunk_{i}` nhất quán mỗi lần chạy lại, tránh lệch `retrieved_ids` khi so sánh kết quả benchmark.<br>• **Metadata drift:** `chunks.json` và ChromaDB phải dùng cùng `ids`/`metadatas`; nếu hai nguồn lệch nhau thì Hit Rate tính sai. Pipeline gán metadata từ `processed_chunks` dùng chung cho cả export lẫn `Chroma.from_texts()` để tránh drift.<br>• **Chunk cắt ngang thông tin quan trọng:** `RecursiveCharacterTextSplitter` ưu tiên cắt theo `\n\n`, `\n`, rồi mới theo ký tự, giúp giữ đoạn văn nguyên vẹn hơn so với fixed-size splitter đơn thuần. |

---

## Trả lời 3 câu hỏi cá nhân theo Rubric

### 1. Engineering Contribution

Đóng góp chính của em nằm ở **Data Pipeline** — nền tảng để toàn bộ hệ thống benchmark hoạt động đúng. Em xây dựng `agent/ingest.py` với các bước:

1. Load các file `.txt` từ `data/doc/` bằng `TextLoader`.
2. Chia thành chunk bằng `RecursiveCharacterTextSplitter` (`chunk_size=600`, `overlap=50`, `add_start_index=True`).
3. Gán `chunk_id = chunk_{i}` và lưu toàn bộ ra `data/chunks.json` trước khi embed — bước này cho phép team dùng `chunks.json` để sinh Golden Dataset (SDG) với `ground_truth_ids` khớp chính xác.
4. Nạp các chunk vào ChromaDB với `OpenAIEmbeddings(model="text-embedding-3-small")`, đảm bảo `ids` khớp với `chunks.json`.

Thiết kế này đảm bảo rằng khi agent trả về `retrieved_ids`, hệ thống metric có thể đối chiếu trực tiếp với Golden Dataset để tính **Hit Rate** và **MRR** chính xác.

### 2. Technical Depth

Phần quan trọng nhất trong data pipeline là tính **nhất quán của Chunk ID**. Nếu `chunk_id` thay đổi giữa các lần chạy ingest, toàn bộ `ground_truth_ids` trong `golden_set.jsonl` sẽ trỏ sai — mọi kết quả Hit Rate / MRR đều vô nghĩa.

- **Hit Rate:** Đo xem trong top-k chunk trả về có chứa chunk đúng không. Nếu chunk bị cắt ngang câu trả lời thì Hit Rate giảm dù retrieval kỹ thuật tốt.
- **MRR:** Chunk đúng ở rank càng cao thì LLM càng ít bị nhiễu bởi chunk sai phía sau.
- **Chunking strategy:** `RecursiveCharacterTextSplitter` ưu tiên ranh giới tự nhiên (`\n\n`, `\n`) trước khi cắt theo ký tự; điều này giúp giữ ngữ cảnh và tăng chất lượng embedding.
- **Cost:** Dùng `text-embedding-3-small` tiết kiệm cost so với large model trong khi vẫn đủ semantic cho domain tài liệu nội bộ; re-embed tốn phí nên thiết kế chunk ổn định từ đầu.

### 3. Problem Solving

Vấn đề khó nhất là **đảm bảo chunk ID nhất quán** khi `os.listdir()` không đảm bảo thứ tự. Nếu thứ tự file thay đổi giữa hai lần chạy, `chunk_0` có thể trỏ sang tài liệu khác, làm hỏng toàn bộ mapping với Golden Dataset.

Hướng xử lý:
- Sort danh sách file trước khi load để thứ tự luôn xác định.
- Export `chunks.json` và nạp ChromaDB từ cùng một list `processed_chunks` — tránh hai nguồn drift nhau.
- Dùng `add_start_index=True` để metadata giữ thông tin vị trí chunk trong tài liệu gốc, hỗ trợ phân tích lỗi khi chunk bị cắt sai.
- Khi cần re-ingest (ví dụ thêm tài liệu mới), cần xóa ChromaDB cũ và sinh lại `chunks.json` đồng thời để tránh lệch ID.

---

**Tổng điểm tự đánh giá:** 40/40 điểm
