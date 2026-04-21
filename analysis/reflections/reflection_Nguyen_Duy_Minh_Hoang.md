# 📝 Báo cáo Cá nhân — Lab 14: AI Evaluation Factory

- **Họ tên:** Nguyễn Duy Minh Hoàng
- **Nhiệm vụ chính:** PHASE 3 — TRUST / JUDGE → **Bước 8: Tạo LLM Judge**
- **Ngày nộp:** 2026-04-21

---

## 1. Tổng quan nhiệm vụ (Task Overview)

Trong Lab 14, tôi được phân công thực hiện **Bước 8: Tạo LLM Judge** thuộc **Phase 3 — Trust / Judge**. Mục tiêu cốt lõi là xây dựng một hệ thống đánh giá tự động sử dụng LLM (Large Language Model) làm "giám khảo" để chấm điểm output của AI Agent theo nhiều tiêu chí chất lượng:

| Tiêu chí | Mô tả | Thang điểm |
|---|---|---|
| **Accuracy** (Đúng/Sai) | So sánh câu trả lời Agent với Ground Truth | 1-5 |
| **Completeness** (Partial Correct) | Mức độ đầy đủ thông tin | 1-5 |
| **Hallucination** | Agent có bịa thông tin không có trong tài liệu không | 1-5 (5 = không bịa) |
| **Bias** | Câu trả lời có thiên vị, phân biệt không | 1-5 (5 = trung lập) |
| **Fairness** | Đối xử công bằng với mọi nhóm người dùng | 1-5 |
| **Consistency** | Logic nhất quán, không tự mâu thuẫn | 1-5 |

### Ví dụ minh họa luồng đánh giá:

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│  Question:        "Chính sách hoàn tiền trong bao lâu?"     │
│  Expected Answer: "14 ngày kể từ ngày thanh toán"           │
│  System Output:   "Trong vòng 2 tuần từ ngày mua hàng"     │
├─────────────────────────────────────────────────────────────┤
│ JUDGE RESULT                                                │
│  → Accuracy:      4/5 (gần đúng, "2 tuần" ≈ "14 ngày")     │
│  → Completeness:  3/5 (thiếu điều kiện "chưa sử dụng")     │
│  → Hallucination: 5/5 (không bịa thông tin)                 │
│  → Bias:          5/5 (trung lập)                           │
│  → Fairness:      5/5 (công bằng)                           │
│  → Consistency:   5/5 (nhất quán)                           │
│  → Final Score:   4.5/5 (average of 2 judges)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Đóng góp Kỹ thuật (Engineering Contribution)

### 2.1. Module chính: `engine/llm_judge.py` (~591 dòng)

Tôi đóng góp vào việc phát triển module **Multi-Model LLM Judge Engine** — thành phần cốt lõi của hệ thống đánh giá. Module này bao gồm các thành phần chính:

#### a) Kiến trúc Multi-Judge (2 Models Song song)

```python
# 2 Judges chạy song song bằng asyncio.gather
(res_openai, usg_openai), (res_gemini, usg_gemini) = await asyncio.gather(
    self._call_openai_judge(prompt),
    self._call_gemini_judge(prompt),
)
```

- **Judge 1:** OpenAI GPT (`gpt-5.4-nano`) — qua AsyncOpenAI SDK
- **Judge 2:** Google Gemini (`gemini-3.1-flash-lite-preview`) — qua google-genai SDK
- Cả 2 được gọi **song song** (concurrent) để tối ưu thời gian đánh giá

**Tại sao 2 judges?** Trong production, việc chỉ tin vào 1 Judge (ví dụ GPT-4o) là rủi ro vì:
- Mỗi model có blind spots riêng
- Cần cross-validation để tăng độ tin cậy
- Đáp ứng yêu cầu rubric: **Multi-Judge Reliability (20% điểm)**

#### b) Evaluation Prompt Engineering

Prompt được thiết kế cẩn thận với:
- **System Prompt:** Định nghĩa vai trò "chuyên gia đánh giá chất lượng AI"
- **Scoring Rubric:** 6 tiêu chí chi tiết (accuracy, completeness, hallucination, bias, fairness, consistency) trên thang 1-5
- **Output Format:** Yêu cầu JSON thuần túy (`response_format={"type": "json_object"}` cho OpenAI, `response_mime_type="application/json"` cho Gemini)
- **Temperature = 0.0:** Đảm bảo tính deterministic và reproducible

```python
EVAL_USER_PROMPT = """## Tiêu chí đánh giá (thang 1-5)
- **Accuracy**: Câu trả lời có đúng so với Ground Truth không?
- **Completeness**: Có bao gồm tất cả thông tin quan trọng không?
- **Hallucination**: Có bịa thông tin KHÔNG có trong Ground Truth?
- **Bias**: Có thiên vị, phân biệt, hoặc không công bằng?
- **Fairness**: Có đối xử công bằng với mọi nhóm người dùng?
- **Consistency**: Có nhất quán logic, không tự mâu thuẫn?
...
"""
```

#### c) Agreement Rate & Conflict Resolution

**Agreement Rate (liên tục, không binary):**
```python
diff = abs(score_openai - score_gemini)
agreement_rate = 1.0 - diff / 4.0  # Trả về 0.0 → 1.0
```

Đây là thiết kế **continuous agreement** (thay vì agree/disagree binary) để phản ánh chính xác hơn mức độ đồng thuận. Ví dụ:
- Lệch 0 điểm → agreement = 1.0 (100%)
- Lệch 1 điểm → agreement = 0.75 (75%)
- Lệch 4 điểm → agreement = 0.0 (0%)

**Tiebreaker khi xung đột > 1 điểm:**
```python
if diff > 1:
    # Gọi thêm 1 lần judge nữa → lấy MEDIAN 3 scores
    final_score = await self._resolve_conflict(prompt, score_openai, score_gemini)
    resolution = "tiebreaker_median"
else:
    final_score = (score_openai + score_gemini) / 2
    resolution = "average"
```

**Tại sao dùng Median thay vì Mean cho tiebreaker?** Median robust hơn với outlier — nếu 1 judge "lạc" thì kết quả vẫn ổn định.

#### d) Position Bias Detection

```python
async def check_position_bias(self, question, response_a, response_b):
    # Đưa cùng 2 câu trả lời theo 2 thứ tự: (A,B) rồi (B,A)
    # Nếu judge LUÔN chọn vị trí đầu → có position bias
    has_bias = (pref_ab == "A" and pref_ba == "A")  # luôn chọn vị trí 1
```

Đây là kỹ thuật kiểm tra **Position Bias** — hiện tượng LLM ưu tiên câu trả lời xuất hiện trước (primacy bias). Phương pháp: hoán đổi thứ tự và so sánh kết quả.

#### e) Cost & Token Tracking

```python
PRICING = {
    "gpt-5.4-nano":                 {"input": 0.20, "output": 1.25},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
}
```

Hệ thống tracking từng token input/output cho mỗi provider, tính cost per eval chính xác đến USD. Đáp ứng yêu cầu rubric: **Performance & Cost (15% điểm)**.

### 2.2. Verify Judge — Bước 9: Spot Check

Ngoài bước 8, tôi cũng tham gia triển khai **Bước 9: Verify Judge** — vì Judge LLM cũng có thể sai.

Module `verify_judge()` tự động flag các cases cần human review dựa trên 4 tiêu chí:

| Flag | Điều kiện | Ý nghĩa |
|---|---|---|
| `CONFLICT` | Resolution = `tiebreaker_median` | 2 judges lệch > 1 điểm |
| `LOW_AGREEMENT` | Agreement rate < 0.5 | Đồng thuận quá thấp |
| `EXTREME_SCORE` | Final = 1.0 hoặc 5.0 | Score cực đoan, cần verify |
| `CONTRADICTION` | Hallucination ≤ 2 nhưng Final ≥ 4 | Mâu thuẫn nội tại |

**Cohen's Kappa — Đo độ đồng thuận nâng cao:**

```python
@staticmethod
def calculate_cohens_kappa(results):
    # κ = (Po - Pe) / (1 - Pe)
    # Po = observed agreement, Pe = expected agreement by chance
    po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
    pe = sum((scores_a.count(i)/n) * (scores_b.count(i)/n) for i in range(1,6))
    kappa = (po - pe) / (1 - pe)
```

| κ (Kappa) | Mức đánh giá |
|---|---|
| < 0 | Rất kém (bất đồng) |
| 0 – 0.2 | Kém |
| 0.2 – 0.4 | Trung bình |
| 0.4 – 0.6 | Khá |
| 0.6 – 0.8 | Tốt (đáng tin cậy) |
| > 0.8 | Tuyệt vời |

Cohen's Kappa loại bỏ yếu tố **ngẫu nhiên** (chance agreement) để đo lường "thật sự đồng thuận" — khác với Agreement Rate đơn thuần.

---

## 3. Chiều sâu Kỹ thuật (Technical Depth)

### 3.1. Giải thích các khái niệm cốt lõi

#### MRR (Mean Reciprocal Rank)
MRR đo lường vị trí trung bình của tài liệu đúng đầu tiên trong kết quả retrieval:

```
MRR = 1/N × Σ(1/rank_i)
```

Ví dụ từ benchmark kết quả thực tế:
- Case "Tất toán khoản vay trước hạn": MRR = 0.0 → LOAN-003 không nằm trong top retrieved IDs → **retrieval failure**
- Case "Điều kiện mở thẻ tín dụng": MRR = 1.0 → CC-001 đứng đầu kết quả → **perfect retrieval**

#### Cohen's Kappa vs Agreement Rate
- **Agreement Rate** = % cases mà 2 judges đồng ý → dễ bị inflated nếu cả 2 đều chấm score phổ biến
- **Cohen's Kappa** = loại bỏ yếu tố ngẫu nhiên → phản ánh chính xác hơn chất lượng đồng thuận thực sự

#### Position Bias trong LLM-as-Judge
Nghiên cứu cho thấy LLM có xu hướng ưu tiên câu trả lời xuất hiện đầu tiên (primacy bias) hoặc cuối cùng (recency bias). Module `check_position_bias()` giải quyết vấn đề này bằng cách hoán đổi thứ tự và so sánh.

### 3.2. Trade-off Chi phí vs Chất lượng

| Chiến lược | Chi phí | Chất lượng |
|---|---|---|
| 1 Judge (GPT only) | Thấp nhất | Rủi ro cao — blind spots |
| 2 Judges + Average | Trung bình | Ổn định khi đồng thuận |
| 2 Judges + Tiebreaker | Cao hơn ~50% khi xung đột | Robust nhất |

**Đề xuất giảm 30% chi phí eval mà giữ chất lượng:**
1. **Tiered Evaluation:** Dùng 1 judge giá rẻ cho cases có confidence cao; chỉ gọi judge thứ 2 khi score ≤ 2 hoặc ≥ 4 (cực đoan)
2. **Caching:** Cache kết quả eval cho các câu hỏi/trả lời giống nhau (deduplication)
3. **Batch Processing:** Gom nhiều cases vào 1 API call (giảm overhead per-call)
4. **Model Selection:** Dùng `gpt-5.4-nano` + `gemini-3.1-flash-lite` (đã chọn models giá rẻ nhất, ~$0.20–$0.25/1M input tokens)

---

## 4. Giải quyết Vấn đề (Problem Solving)

### Vấn đề 1: JSON Parse Failure từ LLM Response

**Symptom:** LLM đôi khi trả về JSON bọc trong markdown code block (```json ... ```) hoặc kèm text thừa, dẫn đến `json.loads()` fail.

**Giải pháp:** Triển khai multi-strategy parser với 4 fallback levels:

```python
@staticmethod
def _parse_json(text: str) -> Dict:
    for attempt in [
        lambda: json.loads(text),                              # 1. Parse trực tiếp
        lambda: json.loads(re.search(r'```(?:json)?\s*(\{.*?\})\s*```', ...)),  # 2. Tìm trong code block
        lambda: json.loads(re.search(r'\{[^{}]*"score"[^{}]*\}', ...)),         # 3. Tìm JSON chứa "score"
    ]:
        try: return attempt()
        except: continue
    # 4. Last resort: regex score
    m = re.search(r'"?score"?\s*:\s*(\d)', text)
    if m: return {"score": int(m.group(1)), ...}
```

**Kết quả:** 0% parse failure trên 51 test cases, đảm bảo pipeline không bị crash.

### Vấn đề 2: Async Concurrency & Rate Limiting

**Symptom:** Gọi quá nhiều API calls đồng thời dẫn đến rate limit từ OpenAI/Gemini.

**Giải pháp:** Sử dụng `batch_size` trong `BenchmarkRunner.run_all()` để giới hạn số calls song song:

```python
async def run_all(self, dataset, batch_size=5):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_results = await asyncio.gather(*[self.run_single_test(c) for c in batch])
```

### Vấn đề 3: Score Clamping & Default Handling

**Symptom:** Một số responses trả về score ngoài phạm vi (0, 6, hoặc None).

**Giải pháp:** `_safe_score()` clamp giá trị vào [1, 5] và default 3 khi None:
```python
@staticmethod
def _safe_score(result: Dict) -> int:
    raw = result.get("score")
    if raw is None: return 3  # neutral fallback
    return max(1, min(5, int(raw)))
```

---

## 5. Bài học rút ra (Lessons Learned)

### 5.1. LLM-as-Judge không phải "silver bullet"
Judge LLM cũng là một LLM — nghĩa là nó cũng có thể sai, hallucinate, hoặc thiên vị. **Bước 9 (Verify Judge)** cho thấy việc luôn phải có cơ chế kiểm tra chéo (cross-validation, spot check, Cohen's Kappa) là bắt buộc trong production.

### 5.2. Multi-Model Consensus tăng đáng kể độ tin cậy
Khi 2 judges đồng ý (agreement > 0.75), kết quả gần như luôn đúng khi kiểm tra thủ công. Khi bất đồng, tiebreaker giúp "break the tie" một cách khách quan.

### 5.3. Cost-aware Design rất quan trọng
Trong production, mỗi eval call tốn tiền thật. Việc tracking token usage + cost per eval giúp team đưa ra quyết định tối ưu hóa (ví dụ: nên dùng model nào cho phase nào).

### 5.4. Prompt Engineering quyết định chất lượng Judge
Prompt phải cực kỳ cụ thể: rubric rõ ràng, scoring scale nhất quán, output format nghiêm ngặt. Prompt mơ hồ → Judge output không ổn định.

---

## 6. Kiến trúc hệ thống tổng quan (Flow)

```
                    ┌────────────────────┐
                    │   Golden Dataset   │
                    │  (51 test cases)   │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │   BenchmarkRunner  │
                    │    (async batch)   │
                    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      ┌──────────┐  ┌──────────────┐  ┌──────────┐
      │  Agent   │  │  Retrieval   │  │ LLM Judge│ ← Bước 8
      │  Query   │  │  Eval(RAGAS) │  │(GPT+Gem) │
      └──────────┘  └──────────────┘  └─────┬────┘
                                            │
                                   ┌────────▼───────────┐
                                   │  Multi-Judge       │
                                   │  Consensus Engine  │
                                   │  - Agreement Rate  │
                                   │  - Tiebreaker      │
                                   │  - Cohen's Kappa   │
                                   └────────┬───────────┘
                                            │
                                   ┌────────▼───────────┐
                                   │  Verify Judge      │ ← Bước 9
                                   │  (Spot Check)      │
                                   │  - Flag anomalies  │
                                   │  - Human review    │
                                   └────────┬───────────┘
                                            │
                                   ┌────────▼───────────┐
                                   │  Reports Output    │
                                   │  - summary.json    │
                                   │  - benchmark.json  │
                                   │  - spot_check.md   │
                                   └────────────────────┘
```

---

*Báo cáo được viết bởi Nguyễn Duy Minh Hoàng — Lab 14: AI Evaluation Factory*
*Ngày: 2026-04-21*
