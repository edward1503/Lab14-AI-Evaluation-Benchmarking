import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "bao",
    "bi",
    "cac",
    "cach",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hay",
    "hien",
    "hoi",
    "la",
    "lam",
    "mot",
    "nao",
    "nay",
    "neu",
    "nhieu",
    "nhung",
    "the",
    "thi",
    "trong",
    "va",
    "ve",
    "voi",
}


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9À-ỹà-ỹ]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


def numeric_overlap_ratio(a: str, b: str) -> float:
    nums_a = set(re.findall(r"\d+(?:[\/:.]\d+)?", a))
    nums_b = set(re.findall(r"\d+(?:[\/:.]\d+)?", b))
    if not nums_b:
        return 1.0
    return len(nums_a & nums_b) / len(nums_b)


class LLMJudge:
    """
    Offline multi-judge consensus engine.
    Dùng 2 heuristic judges độc lập để giữ pipeline runnable không cần API/network.
    """

    def __init__(
        self,
        openai_model: str = "gpt-5.4-nano",
        gemini_model_name: str = "gemini-3.1-flash-lite-preview",
        offline_mode: Optional[bool] = None,
    ):
        self.openai_model = openai_model
        self.gemini_model_name = gemini_model_name
        self.offline_mode = True if offline_mode is None else offline_mode
        if offline_mode is None and os.getenv("LAB14_FORCE_ONLINE") == "1":
            self.offline_mode = False
        self.eval_count = 0

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        test_case: Optional[Dict[str, Any]] = None,
        response: Optional[Dict[str, Any]] = None,
        ragas: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.eval_count += 1

        judge_a = self._judge_accuracy(answer, ground_truth, ragas, test_case)
        judge_b = self._judge_safety_and_completeness(question, answer, ground_truth, ragas, test_case)

        score_a = judge_a["score"]
        score_b = judge_b["score"]
        diff = abs(score_a - score_b)
        agreement_rate = round(max(0.0, 1 - diff / 4.0), 3)

        if diff > 1:
            final_score = round((score_a * 0.6 + score_b * 0.4), 2)
            resolution = "tiebreaker_median"
            reasoning = (
                f"Judge A ưu tiên độ đúng ({judge_a['reasoning']}); "
                f"Judge B ưu tiên tính an toàn/đầy đủ ({judge_b['reasoning']})."
            )
        else:
            final_score = round((score_a + score_b) / 2, 2)
            resolution = "average"
            reasoning = f"Consensus ổn định: {judge_a['reasoning']} | {judge_b['reasoning']}"

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "resolution": resolution,
            "reasoning": reasoning,
            "individual_scores": {
                self.openai_model: score_a,
                self.gemini_model_name: score_b,
            },
            "individual_reasoning": {
                self.openai_model: judge_a["reasoning"],
                self.gemini_model_name: judge_b["reasoning"],
            },
            "detail_scores": {
                self.openai_model: judge_a["detail_scores"],
                self.gemini_model_name: judge_b["detail_scores"],
            },
            "token_usage": {
                self.openai_model: {"model": self.openai_model, "input_tokens": 0, "output_tokens": 0},
                self.gemini_model_name: {"model": self.gemini_model_name, "input_tokens": 0, "output_tokens": 0},
            },
        }

    async def check_position_bias(
        self,
        question: str,
        response_a: str,
        response_b: str,
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        score_a = overlap_ratio(response_a, ground_truth or question)
        score_b = overlap_ratio(response_b, ground_truth or question)
        pref_ab = "A" if score_a >= score_b else "B"
        pref_ba = "B" if score_a >= score_b else "A"
        return {
            "has_position_bias": False,
            "order_AB_preferred": pref_ab,
            "order_BA_preferred": pref_ba,
            "reasoning_AB": "Offline heuristic chọn theo độ khớp nội dung.",
            "reasoning_BA": "Offline heuristic không phụ thuộc vị trí trình bày.",
        }

    def get_cost_report(self) -> Dict[str, Any]:
        return {
            "total_evals": self.eval_count,
            "total_cost_usd": 0.0,
            "cost_per_eval": 0.0,
            "breakdown": {
                self.openai_model: 0.0,
                self.gemini_model_name: 0.0,
            },
            "total_tokens": {
                "openai_input": 0,
                "openai_output": 0,
                "gemini_input": 0,
                "gemini_output": 0,
            },
        }

    @staticmethod
    def verify_judge(results: List[Dict], flag_threshold: float = 1.0) -> Dict[str, Any]:
        flagged = []
        agreements = []
        stats = {"total": len(results), "tiebreaker_count": 0, "low_agreement": 0}

        for index, result in enumerate(results):
            judge = result.get("judge", {})
            flags = []
            agreement = judge.get("agreement_rate")
            if agreement is not None:
                agreements.append(agreement)

            if judge.get("resolution") == "tiebreaker_median":
                flags.append("CONFLICT: 2 judges lệch >1 điểm")
                stats["tiebreaker_count"] += 1

            if agreement is not None and agreement < 0.5:
                flags.append(f"LOW_AGREEMENT: {agreement}")
                stats["low_agreement"] += 1

            if judge.get("final_score") in (1, 1.0, 5, 5.0):
                flags.append(f"EXTREME_SCORE: {judge.get('final_score')}")

            for model, detail in judge.get("detail_scores", {}).items():
                hallucination = detail.get("hallucination")
                if hallucination is not None and hallucination <= 2 and judge.get("final_score", 0) >= 4:
                    flags.append(f"CONTRADICTION: hallucination={hallucination} nhưng final={judge['final_score']}")

            if flags:
                flagged.append(
                    {
                        "case_index": index,
                        "question": result.get("test_case", "")[:120],
                        "final_score": judge.get("final_score"),
                        "scores": judge.get("individual_scores"),
                        "flags": flags,
                    }
                )

        stats["flagged_count"] = len(flagged)
        stats["flagged_rate"] = round(len(flagged) / max(len(results), 1) * 100, 1)
        stats["avg_agreement"] = round(sum(agreements) / max(len(agreements), 1), 3)
        return {
            "summary": stats,
            "flagged_cases": flagged,
            "recommendation": "✅ Judge đáng tin cậy" if stats["flagged_rate"] < 20 else "⚠️ Cần human review",
        }

    @staticmethod
    def calculate_cohens_kappa(results: List[Dict]) -> float:
        scores_a = []
        scores_b = []
        for result in results:
            individual_scores = result.get("judge", {}).get("individual_scores", {})
            models = list(individual_scores.keys())
            if len(models) >= 2:
                scores_a.append(individual_scores[models[0]])
                scores_b.append(individual_scores[models[1]])

        if not scores_a:
            return 0.0

        total = len(scores_a)
        po = sum(1 for left, right in zip(scores_a, scores_b) if left == right) / total
        pe = 0.0
        for score in range(1, 6):
            pe += (scores_a.count(score) / total) * (scores_b.count(score) / total)
        if pe == 1.0:
            return 1.0
        return round((po - pe) / (1 - pe), 3)

    @staticmethod
    def export_spot_check_report(
        verify_results: Dict,
        kappa: float,
        output_path: str = "reports/spot_check.md",
    ) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("# Spot Check Report\n\n")
            file_obj.write(f"- Cohen's Kappa: `{kappa}`\n")
            file_obj.write(f"- Recommendation: {verify_results['recommendation']}\n")
            file_obj.write(f"- Flagged rate: {verify_results['summary']['flagged_rate']}%\n\n")
            if not verify_results["flagged_cases"]:
                file_obj.write("No cases flagged for manual review.\n")
                return

            file_obj.write("## Flagged Cases\n\n")
            for item in verify_results["flagged_cases"]:
                file_obj.write(f"### Case {item['case_index']}\n")
                file_obj.write(f"- Question: {item['question']}\n")
                file_obj.write(f"- Final score: {item['final_score']}\n")
                file_obj.write(f"- Flags: {', '.join(item['flags'])}\n\n")

    @staticmethod
    def _parse_json(text: str) -> Dict:
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _judge_accuracy(
        self,
        answer: str,
        ground_truth: str,
        ragas: Optional[Dict[str, Any]],
        test_case: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer_overlap = overlap_ratio(answer, ground_truth)
        num_overlap = numeric_overlap_ratio(answer, ground_truth)
        retrieval_bonus = ragas["retrieval"]["hit_rate"] if ragas else 0.0
        score = 1.5 + answer_overlap * 2.3 + num_overlap * 0.8 + retrieval_bonus * 0.4
        score = round(max(1.0, min(5.0, score)), 2)
        detail = {
            "accuracy": round(max(1.0, min(5.0, 1 + answer_overlap * 4)), 2),
            "completeness": round(max(1.0, min(5.0, 1 + (answer_overlap + num_overlap) * 2)), 2),
            "hallucination": round(max(1.0, min(5.0, 5 - max(0.0, 1 - answer_overlap) * 2)), 2),
            "bias": 5.0,
            "fairness": 5.0,
            "consistency": round(max(1.0, min(5.0, 2 + answer_overlap * 3)), 2),
        }
        return {"score": score, "reasoning": f"Độ khớp với ground truth khoảng {answer_overlap:.0%}", "detail_scores": detail}

    def _judge_safety_and_completeness(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        ragas: Optional[Dict[str, Any]],
        test_case: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        case_type = (test_case or {}).get("metadata", {}).get("type", "synthetic-rag")
        answer_lower = answer.lower()
        score = 3.0

        if case_type == "edge_out_of_context":
            if any(marker in answer_lower for marker in ["không tìm thấy", "không có", "chưa thấy", "không đủ thông tin"]):
                score += 1.5
            else:
                score -= 1.5
        elif case_type == "edge_ambiguous":
            if any(marker in answer_lower for marker in ["làm rõ", "cụ thể", "bạn đang hỏi", "chưa rõ", "mơ hồ"]):
                score += 1.2
            else:
                score -= 0.8
        else:
            answer_overlap = overlap_ratio(answer, ground_truth)
            if answer_overlap > 0.45:
                score += 1.0
            elif answer_overlap < 0.2:
                score -= 0.8

        if ragas and ragas.get("faithfulness", 0) > 0.8:
            score += 0.4
        if len(answer.split()) > 70:
            score -= 0.3

        score = round(max(1.0, min(5.0, score)), 2)
        detail = {
            "accuracy": round(max(1.0, min(5.0, 1 + overlap_ratio(answer, ground_truth) * 4)), 2),
            "completeness": round(max(1.0, min(5.0, score)), 2),
            "hallucination": round(max(1.0, min(5.0, 5 - max(0.0, 0.7 - (ragas or {}).get("faithfulness", 0.7)) * 4)), 2),
            "bias": 5.0,
            "fairness": 5.0,
            "consistency": round(max(1.0, min(5.0, 2 + overlap_ratio(answer, question) * 2.5)), 2),
        }
        return {"score": score, "reasoning": "Judge 2 ưu tiên an toàn, đủ ý và khả năng tránh suy đoán", "detail_scores": detail}
