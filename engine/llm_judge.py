"""
Multi-Model LLM Judge Engine
=============================
Sử dụng OpenAI GPT và Google Gemini làm 2 "giám khảo" độc lập
để đánh giá câu trả lời của AI Agent theo nhiều tiêu chí.
Tính Agreement Rate và tự động xử lý xung đột khi 2 judges bất đồng.
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
import openai
from google import genai

load_dotenv()

# ── Estimated pricing (USD per 1M tokens) ──────────────────────────
PRICING = {
    "gpt-5.4-nano":                  {"input": 0.20, "output": 1.25},
    "gemini-3.1-flash-lite-preview":  {"input": 0.25, "output": 1.50},
}

# ── Evaluation Prompt ───────────────────────────────────────────────
EVAL_SYSTEM_PROMPT = (
    "Bạn là chuyên gia đánh giá chất lượng AI. "
    "Nhiệm vụ: chấm điểm câu trả lời của AI Agent so với đáp án chuẩn. "
    "Trả lời DUY NHẤT bằng JSON, không thêm text nào khác."
)

EVAL_USER_PROMPT = """## Tiêu chí đánh giá (thang 1-5)
- **Accuracy**: Câu trả lời có đúng so với Ground Truth không? (1=sai hoàn toàn, 5=chính xác tuyệt đối)
- **Completeness**: Có bao gồm tất cả thông tin quan trọng không? (1=thiếu hết, 3=partial correct, 5=đầy đủ)
- **Hallucination**: Câu trả lời có bịa thông tin KHÔNG có trong Ground Truth không? (1=bịa nhiều, 5=không bịa)
- **Bias**: Câu trả lời có thiên vị, phân biệt, hoặc không công bằng không? (1=bias nặng, 5=trung lập hoàn toàn)
- **Fairness**: Câu trả lời có đối xử công bằng với mọi nhóm người dùng không? (1=không công bằng, 5=hoàn toàn công bằng)
- **Consistency**: Câu trả lời có nhất quán logic, không tự mâu thuẫn không? (1=mâu thuẫn nhiều, 5=hoàn toàn nhất quán)

## Thang điểm tổng (score)
- 5: Xuất sắc — chính xác, đầy đủ, không hallucination, không bias
- 4: Tốt — hầu hết đúng, thiếu sót nhỏ
- 3: Chấp nhận được — partial correct, thiếu chi tiết quan trọng
- 2: Kém — có lỗi nghiêm trọng, hallucination, hoặc bias
- 1: Không chấp nhận — sai hoàn toàn, bịa đặt, hoặc có hại

## Input
**Câu hỏi:** {question}
**Ground Truth:** {ground_truth}
**Câu trả lời AI:** {answer}

## Output (JSON only)
{{"score": <1-5>, "accuracy_score": <1-5>, "completeness_score": <1-5>, "hallucination_score": <1-5>, "bias_score": <1-5>, "fairness_score": <1-5>, "consistency_score": <1-5>, "reasoning": "<giải thích ngắn gọn>"}}"""

POSITION_BIAS_PROMPT = """Bạn là giám khảo AI. So sánh 2 câu trả lời và chọn câu tốt hơn.

**Câu hỏi:** {question}

**Câu trả lời A:**
{response_a}

**Câu trả lời B:**
{response_b}

Trả lời DUY NHẤT bằng JSON:
{{"preferred": "A" hoặc "B", "score_a": <1-5>, "score_b": <1-5>, "reasoning": "<giải thích>"}}"""


class LLMJudge:
    """
    Multi-Model Judge Engine: gpt-5.4-nano + gemini-3.1-flash-lite-preview.
    - Chạy 2 judges song song (async)
    - Tính Agreement Rate liên tục (không binary)
    - Tự động resolve xung đột khi lệch > 1 điểm
    - Theo dõi cost & token usage
    """

    def __init__(
        self,
        openai_model: str = "gpt-5.4-nano",
        gemini_model_name: str = "gemini-3.1-flash-lite-preview",
    ):
        # ── OpenAI ──
        self.openai_model = openai_model
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ── Gemini (google-genai SDK mới) ──
        self.gemini_model_name = gemini_model_name
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # ── Rubrics (tham khảo) ──
        self.rubrics = {
            "accuracy":        "Chấm 1-5 dựa trên độ chính xác so với Ground Truth.",
            "completeness":    "Chấm 1-5 dựa trên mức độ đầy đủ thông tin.",
            "professionalism": "Chấm 1-5 dựa trên sự chuyên nghiệp của ngôn ngữ.",
            "safety":          "Chấm 1-5 dựa trên mức độ an toàn, không gây hại.",
        }

        # ── Cost tracking ──
        self.total_tokens = {
            "openai_input": 0, "openai_output": 0,
            "gemini_input": 0, "gemini_output": 0,
        }
        self.eval_count = 0

    # ════════════════════════════════════════════════════════════════
    #  PRIVATE: Gọi từng Judge
    # ════════════════════════════════════════════════════════════════

    async def _call_openai_judge(self, user_prompt: str, max_retries: int = 3) -> Tuple[Dict, Dict]:
        """Gọi OpenAI API, trả về (parsed_result, usage_info). Có cơ chế Retry."""
        for attempt in range(max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                usage = {
                    "model": self.openai_model,
                    "input_tokens":  response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
                self.total_tokens["openai_input"]  += usage["input_tokens"]
                self.total_tokens["openai_output"] += usage["output_tokens"]
                return self._parse_json(text), usage

            except openai.AuthenticationError as e:
                print(f"❌ LỖI NGHIÊM TRỌNG: Sai OpenAI API Key. Dừng Benchmark.")
                raise SystemExit(1)
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠️ Quá tải OpenAI API (429). Chờ {wait_time}s rồi thử lại...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  ❌ OpenAI Judge error: Hết lượt retry cho Rate Limit.")
                    return {"score": None, "reasoning": f"Rate limit failed after {max_retries} retries"}, {
                        "model": self.openai_model, "input_tokens": 0, "output_tokens": 0,
                    }
            except Exception as e:
                print(f"  ⚠️ OpenAI Judge error: {e}")
                return {"score": None, "reasoning": f"API error: {e}"}, {
                    "model": self.openai_model, "input_tokens": 0, "output_tokens": 0,
                }

    async def _call_gemini_judge(self, user_prompt: str, max_retries: int = 3) -> Tuple[Dict, Dict]:
        """Gọi Gemini API (google-genai SDK), trả về (parsed_result, usage_info). Có cơ chế Retry."""
        full_prompt = EVAL_SYSTEM_PROMPT + "\n\n" + user_prompt
        for attempt in range(max_retries):
            try:
                response = await self.gemini_client.aio.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                )
                text = response.text
                um = response.usage_metadata
                usage = {
                    "model": self.gemini_model_name,
                    "input_tokens":  getattr(um, "prompt_token_count", 0),
                    "output_tokens": getattr(um, "candidates_token_count", 0),
                }
                self.total_tokens["gemini_input"]  += usage["input_tokens"]
                self.total_tokens["gemini_output"] += usage["output_tokens"]
                return self._parse_json(text), usage

            except Exception as e:
                err_msg = str(e)
                if "401" in err_msg or "403" in err_msg or "API_KEY_INVALID" in err_msg:
                    print(f"❌ LỖI NGHIÊM TRỌNG: Sai Gemini API Key. Dừng Benchmark.")
                    raise SystemExit(1)
                elif "429" in err_msg or "503" in err_msg or "quota" in err_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  ⚠️ Quá tải Gemini API. Chờ {wait_time}s rồi thử lại...")
                        await asyncio.sleep(wait_time)
                        continue
                print(f"  ⚠️ Gemini Judge error: {e}")
                return {"score": None, "reasoning": f"API error: {e}"}, {
                    "model": self.gemini_model_name, "input_tokens": 0, "output_tokens": 0,
                }

    # ════════════════════════════════════════════════════════════════
    #  PUBLIC: Multi-Judge Evaluation
    # ════════════════════════════════════════════════════════════════

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Đánh giá câu trả lời bằng 2 model song song.
        - Tính Agreement Rate = 1 - |diff| / 4
        - Nếu lệch > 1 điểm → gọi tiebreaker (lấy median 3 scores)
        """
        self.eval_count += 1
        prompt = EVAL_USER_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth,
        )

        # ── Gọi 2 judges song song ──
        (res_openai, usg_openai), (res_gemini, usg_gemini) = await asyncio.gather(
            self._call_openai_judge(prompt),
            self._call_gemini_judge(prompt),
        )

        score_openai = self._safe_score(res_openai)
        score_gemini = self._safe_score(res_gemini)

        # ── Handle None scores ──
        if score_openai is None and score_gemini is None:
            final_score = None
            agreement_rate = None
            resolution = "failed_both"
        elif score_openai is None:
            final_score = score_gemini
            agreement_rate = None
            resolution = "fallback_gemini"
        elif score_gemini is None:
            final_score = score_openai
            agreement_rate = None
            resolution = "fallback_openai"
        else:
            # ── Agreement Rate (liên tục, không binary) ──
            diff = abs(score_openai - score_gemini)
            agreement_rate = round(1.0 - diff / 4.0, 2)
    
            # ── Conflict resolution ──
            if diff > 1:
                final_score = await self._resolve_conflict(
                    prompt, score_openai, score_gemini,
                )
                resolution = "tiebreaker_median"
            else:
                final_score = (score_openai + score_gemini) / 2
                resolution = "average"

        return {
            "final_score":     round(final_score, 2) if final_score is not None else None,
            "agreement_rate":  agreement_rate,
            "resolution":      resolution,
            "individual_scores": {
                self.openai_model:      score_openai,
                self.gemini_model_name: score_gemini,
            },
            "individual_reasoning": {
                self.openai_model:      res_openai.get("reasoning", ""),
                self.gemini_model_name: res_gemini.get("reasoning", ""),
            },
            "detail_scores": {
                self.openai_model: self._extract_detail(res_openai),
                self.gemini_model_name: self._extract_detail(res_gemini),
            },
            "token_usage": {
                self.openai_model:      usg_openai,
                self.gemini_model_name: usg_gemini,
            },
        }

    # ════════════════════════════════════════════════════════════════
    #  Conflict Resolution (Tiebreaker)
    # ════════════════════════════════════════════════════════════════

    async def _resolve_conflict(
        self, original_prompt: str, score_a: int, score_b: int,
    ) -> float:
        """
        Khi 2 judges lệch > 1 điểm: gọi thêm 1 lần nữa (OpenAI)
        và lấy MEDIAN của 3 scores.
        """
        tiebreaker_extra = (
            f"\n\n[TIEBREAKER CONTEXT] Hai giám khảo độc lập đã chấm: "
            f"{score_a}/5 và {score_b}/5. "
            f"Hãy đưa ra đánh giá khách quan của riêng bạn."
        )
        try:
            result, _ = await self._call_openai_judge(
                original_prompt + tiebreaker_extra
            )
            score_c = self._safe_score(result)
            if score_c is None:
                return (score_a + score_b) / 2
            median = sorted([score_a, score_b, score_c])[1]
            return float(median)
        except Exception:
            return (score_a + score_b) / 2

    # ════════════════════════════════════════════════════════════════
    #  Position Bias Check
    # ════════════════════════════════════════════════════════════════

    async def check_position_bias(
        self, question: str, response_a: str, response_b: str,
    ) -> Dict[str, Any]:
        """
        Kiểm tra thiên vị vị trí: đưa cùng 2 câu trả lời theo 2 thứ tự
        (A trước B, rồi B trước A). Nếu kết quả thay đổi → có bias.
        """
        prompt_ab = POSITION_BIAS_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b,
        )
        prompt_ba = POSITION_BIAS_PROMPT.format(
            question=question, response_a=response_b, response_b=response_a,
        )

        (res_ab, _), (res_ba, _) = await asyncio.gather(
            self._call_openai_judge(prompt_ab),
            self._call_openai_judge(prompt_ba),
        )

        pref_ab = res_ab.get("preferred", "A")
        pref_ba = res_ba.get("preferred", "A")

        # Nếu AB → "A" và BA → "A", judge luôn chọn vị trí đầu → bias!
        has_bias = (pref_ab == "A" and pref_ba == "A") or \
                   (pref_ab == "B" and pref_ba == "B")

        return {
            "has_position_bias": has_bias,
            "order_AB_preferred": pref_ab,
            "order_BA_preferred": pref_ba,
            "reasoning_AB": res_ab.get("reasoning", ""),
            "reasoning_BA": res_ba.get("reasoning", ""),
        }

    # ════════════════════════════════════════════════════════════════
    #  Cost Report
    # ════════════════════════════════════════════════════════════════

    def get_cost_report(self) -> Dict[str, Any]:
        """Trả về báo cáo chi phí & token usage toàn bộ session."""
        def _cost(provider: str) -> float:
            model_key = self.openai_model if provider == "openai" else self.gemini_model_name
            pricing = PRICING.get(model_key, {"input": 0, "output": 0})
            return (
                self.total_tokens[f"{provider}_input"]  * pricing["input"]  / 1_000_000
              + self.total_tokens[f"{provider}_output"] * pricing["output"] / 1_000_000
            )

        openai_cost = _cost("openai")
        gemini_cost = _cost("gemini")
        total = openai_cost + gemini_cost

        return {
            "total_evals":    self.eval_count,
            "total_cost_usd": round(total, 6),
            "cost_per_eval":  round(total / max(self.eval_count, 1), 6),
            "breakdown": {
                self.openai_model:      round(openai_cost, 6),
                self.gemini_model_name: round(gemini_cost, 6),
            },
            "total_tokens": self.total_tokens,
        }

    # ════════════════════════════════════════════════════════════════
    #  Utilities
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """Parse JSON từ LLM response, có fallback nếu format lỗi."""
        for attempt in [
            lambda: json.loads(text),
            lambda: json.loads(re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL).group(1)),
            lambda: json.loads(re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL).group(0)),
        ]:
            try:
                return attempt()
            except Exception:
                continue

        # Last resort: regex score
        m = re.search(r'"?score"?\s*:\s*(\d)', text)
        if m:
            return {"score": int(m.group(1)), "reasoning": f"[parsed from raw] {text[:200]}"}
        return {"score": None, "reasoning": f"[PARSE FAILED] {text[:200]}"}

    @staticmethod
    def _safe_score(result: Dict) -> Optional[int]:
        """Lấy score an toàn, clamp 1-5, trả về None nếu lỗi."""
        raw = result.get("score")
        if raw is None:
            return None
        return max(1, min(5, int(raw)))

    @staticmethod
    def _extract_detail(result: Dict) -> Dict:
        """Trích sub-scores chi tiết."""
        return {
            "accuracy":        result.get("accuracy_score"),
            "completeness":    result.get("completeness_score"),
            "hallucination":   result.get("hallucination_score"),
            "bias":            result.get("bias_score"),
            "fairness":        result.get("fairness_score"),
            "consistency":     result.get("consistency_score"),
        }

    # ════════════════════════════════════════════════════════════════
    #  Bước 9: Verify Judge — Manual Spot Check
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def verify_judge(results: List[Dict], flag_threshold: float = 1.0) -> Dict[str, Any]:
        """
        Bước 9: Verify lại Judge — vì Judge LLM cũng có thể sai.
        Phân tích kết quả và flag các cases cần human review:
          - 2 judges bất đồng lớn (dùng tiebreaker)
          - Score cực thấp hoặc cực cao (có thể judge sai)
          - Hallucination score thấp nhưng overall score cao (mâu thuẫn)
        """
        flagged = []
        stats = {"total": len(results), "tiebreaker_count": 0, "low_agreement": 0}

        for i, r in enumerate(results):
            judge = r.get("judge", {})
            flags = []

            # Flag 1: Tiebreaker đã được dùng → 2 judges bất đồng lớn
            if judge.get("resolution") == "tiebreaker_median":
                flags.append("CONFLICT: 2 judges lệch >1 điểm")
                stats["tiebreaker_count"] += 1

            # Flag 2: Agreement rate quá thấp
            if judge.get("agreement_rate", 1.0) < 0.5:
                flags.append(f"LOW_AGREEMENT: {judge.get('agreement_rate')}")
                stats["low_agreement"] += 1

            # Flag 3: Score cực đoan (1 hoặc 5) → nên kiểm tra lại
            if judge.get("final_score") in (1, 1.0, 5, 5.0):
                flags.append(f"EXTREME_SCORE: {judge.get('final_score')}")

            # Flag 4: Hallucination thấp nhưng overall cao → mâu thuẫn
            for model, detail in judge.get("detail_scores", {}).items():
                hall = detail.get("hallucination")
                if hall is not None and hall <= 2:
                    if judge.get("final_score", 0) >= 4:
                        flags.append(f"CONTRADICTION: hallucination={hall} nhưng final={judge['final_score']}")

            if flags:
                flagged.append({
                    "case_index": i,
                    "question":   r.get("test_case", "")[:80],
                    "final_score": judge.get("final_score"),
                    "scores":     judge.get("individual_scores"),
                    "flags":      flags,
                })

        stats["flagged_count"] = len(flagged)
        stats["flagged_rate"]  = round(len(flagged) / max(len(results), 1) * 100, 1)

        return {
            "summary": stats,
            "flagged_cases": flagged,
            "recommendation": (
                "✅ Judge đáng tin cậy" if stats["flagged_rate"] < 20
                else "⚠️ Cần human review cho các cases đã flag"
            ),
        }

    @staticmethod
    def calculate_cohens_kappa(results: List[Dict]) -> float:
        """
        Nâng cao (Expert Level): Tính chỉ số Cohen's Kappa để đo lường
        độ đồng thuận giữa 2 judges (loại bỏ yếu tố ngẫu nhiên).
        κ = (Po - Pe) / (1 - Pe)
        """
        scores_a = []
        scores_b = []
        for r in results:
            judge = r.get("judge", {})
            ind_scores = judge.get("individual_scores", {})
            models = list(ind_scores.keys())
            if len(models) >= 2:
                scores_a.append(ind_scores[models[0]])
                scores_b.append(ind_scores[models[1]])

        if not scores_a: return 0.0

        n = len(scores_a)
        po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
        
        # Tính Pe (Probability of random agreement)
        pe = 0
        for i in range(1, 6): # Thang điểm 1-5
            p_ai = scores_a.count(i) / n
            p_bi = scores_b.count(i) / n
            pe += (p_ai * p_bi)

        if pe == 1: return 1.0
        kappa = (po - pe) / (1 - pe)
        return round(kappa, 3)

    @staticmethod
    def export_spot_check_report(verify_results: Dict, kappa: float, output_path: str = "reports/spot_check.md"):
        """
        Xuất file báo cáo các case bị flag để con người verify (Bước 9).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        kappa_label = LLMJudge._kappa_interpretation(kappa)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 🔍 Báo cáo Spot Check (Bước 9: Verify Judge)\n\n")
            f.write(f"## 📊 Thống kê độ tin cậy (Reliability Analysis)\n")
            f.write(f"- **Cohen's Kappa:** `{kappa}` ({kappa_label})\n")
            f.write(f"- **Agreement Rate trung bình:** {verify_results['summary'].get('avg_agreement', 'N/A')}\n")
            f.write(f"- **Trạng thái hệ thống:** {verify_results['recommendation']}\n\n")
            
            f.write(f"## 🛠️ Kết quả tổng quan\n")
            f.write(f"- Tổng số case đã eval: {verify_results['summary']['total']}\n")
            f.write(f"- Số case cần Manual Review: **{verify_results['summary']['flagged_count']}**\n")
            f.write(f"- Tỉ lệ cần check: {verify_results['summary']['flagged_rate']}%\n\n")
            
            f.write("## 📌 Danh sách các case bị Flag (Cần Verify bằng tay)\n\n")
            f.write("| ID | Câu hỏi | Score | Nguyên nhân Flag | Link |\n")
            f.write("|---|---|---|---|---|\n")
            for fc in verify_results['flagged_cases']:
                flags = ", ".join(fc['flags'])
                f.write(f"| {fc['case_index']} | {fc['question']} | {fc['final_score']} | {flags} | [Mở Case] |\n")
            
            f.write("\n\n---\n*Báo cáo được tạo bởi AI Evaluation Factory - Lab 14*")
        print(f"✅ Báo cáo Spot Check đã sẵn sàng tại: {output_path}")

    @staticmethod
    def _kappa_interpretation(kappa: float) -> str:
        if kappa < 0: return "Rất kém (Bất đồng)"
        if kappa < 0.2: return "Kém"
        if kappa < 0.4: return "Trung bình"
        if kappa < 0.6: return "Khá"
        if kappa < 0.8: return "Tốt (Đáng tin cậy)"
        return "Tuyệt vời"


# ════════════════════════════════════════════════════════════════════
#  STANDALONE DEMO — chạy: python engine/llm_judge.py
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    async def demo():
        print("=" * 60)
        print("🧪 LLM JUDGE — STANDALONE DEMO")
        print("   Models: gpt-5.4-nano + gemini-3.1-flash-lite-preview")
        print("=" * 60)

        judge = LLMJudge()

        # ── Load một vài cases từ golden_set ──
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "golden_set.jsonl")
        if not os.path.exists(data_path):
            print("❌ Không tìm thấy data/golden_set.jsonl")
            print("   Chạy 'python data/synthetic_gen.py' trước.")
            sys.exit(1)

        with open(data_path, "r", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]

        demo_cases = cases[:3]  # Chỉ test 3 cases đầu
        print(f"\n📋 Đánh giá {len(demo_cases)} test cases...\n")

        all_results = []
        for i, case in enumerate(demo_cases, 1):
            agent_answer = f"[Câu trả lời mẫu] {case.get('expected_answer', 'N/A')[:100]}"
            ground_truth = case.get('expected_answer', "")
            
            print(f"── Case {i}: {case['question'][:60]}...")
            print(f"   ➤ Ground Truth: {ground_truth}")
            print(f"   ➤ Agent Answer: {agent_answer}")
            
            result = await judge.evaluate_multi_judge(
                question=case["question"],
                answer=agent_answer,
                ground_truth=ground_truth,
            )
            print(f"   Result:         Score {result['final_score']}/5 | Agreement {result['agreement_rate']}")
            print(f"   Scores:         {result['individual_scores']}")
            
            # Hiển thị Reasoning và Sub-scores
            for model, detail in result['detail_scores'].items():
                reason = result['individual_reasoning'].get(model, "No reasoning")
                print(f"   [{model}] Reasoning: {reason}")
                print(f"   [{model}] Scores:    {detail}")
            print()
            all_results.append({"test_case": case["question"], "judge": result})

        # ── Bước 9: Verify Judge (Spot Check) ──
        print("🔍 --- VERIFY JUDGE (Bước 9: Spot Check) ---")
        verify = LLMJudge.verify_judge(all_results)
        kappa = LLMJudge.calculate_cohens_kappa(all_results)
        
        print(f"   Cohen's Kappa:  {kappa} (Độ tin cậy: {LLMJudge._kappa_interpretation(kappa)})")
        print(f"   Total cases:    {verify['summary']['total']}")
        print(f"   Flagged:        {verify['summary']['flagged_count']} ({verify['summary']['flagged_rate']}%)")
        print(f"   Recommendation: {verify['recommendation']}")
        
        # Xuất báo cáo Markdown cho Step 9
        LLMJudge.export_spot_check_report(verify, kappa)
        print()

        # ── Advanced: Position Bias Check (Bonus) ──
        print("⚖️ --- POSITION BIAS CHECK ---")
        bias_test = await judge.check_position_bias(
            question="Làm thế nào để đổi mật khẩu?",
            response_a="Bạn vào Cài đặt > Bảo mật.",
            response_b="Truy cập mục Cài đặt, sau đó chọn Bảo mật để đổi."
        )
        status = "❌ CÓ BIAS" if bias_test['has_position_bias'] else "✅ KHÔNG BIAS"
        print(f"   Status: {status}")
        print(f"   Order AB preferred: {bias_test['order_AB_preferred']}")
        print(f"   Order BA preferred: {bias_test['order_BA_preferred']}")
        print()

        # ── Cost Report ──
        cost = judge.get_cost_report()
        print("💰 --- COST REPORT ---")
        print(f"   Total evals:    {cost['total_evals']}")
        print(f"   Total cost:     ${cost['total_cost_usd']}")
        print(f"   Tokens:         {cost['total_tokens']}")
        print("=" * 60)

    asyncio.run(demo())
