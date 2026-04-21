import asyncio
import os
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """
    Hệ thống đánh giá đa mô hình (Multi-Judge) chuẩn Lab.
    Sử dụng gpt-4o-mini và gpt-3.5-turbo để chấm điểm và tính toán Consensus.
    """
    def __init__(self, models: List[str] = ["gpt-4o-mini", "gpt-3.5-turbo"]):
        self.models = models
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Rubrics chi tiết theo thang điểm 1-5
        self.rubrics = {
            "accuracy": "5: Hoàn hảo, khớp ground truth. 1: Hoàn toàn sai.",
            "faithfulness": "5: Dựa hoàn toàn vào context. 1: Bịa đặt thông tin (Hallucination).",
            "relevancy": "5: Câu trả lời giải quyết trực tiếp câu hỏi. 1: Lạc đề, không liên quan."
        }

    async def _get_single_score(self, model: str, question: str, answer: str, ground_truth: str, context: str = "") -> Dict:
        prompt = f"""Bạn là một chuyên gia thẩm định AI. Hãy chấm điểm câu trả lời dựa trên:
1. Accuracy: {self.rubrics['accuracy']}
2. Faithfulness: {self.rubrics['faithfulness']}
3. Relevancy: {self.rubrics['relevancy']}

Câu hỏi: {question}
Context: {context}
Câu trả lời: {answer}
Ground Truth: {ground_truth}

Chỉ trả về JSON:
{{
    "accuracy": <1-5>,
    "faithfulness": <1-5>,
    "relevancy": <1-5>,
    "reasoning": "<giải thích>"
}}
"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            # Lấy data và gán thêm usage thông tin
            data = json.loads(response.choices[0].message.content)
            data["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return data
        except Exception as e:
            return {
                "accuracy": 1, "faithfulness": 1, "relevancy": 1, 
                "reasoning": f"Error {model}: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str, context: str = "") -> Dict[str, Any]:
        tasks = [self._get_single_score(model, question, answer, ground_truth, context) for model in self.models]
        results = await asyncio.gather(*tasks)
        
        # Mapping theo cấu trúc sample
        individual_results = {self.models[i]: results[i] for i in range(len(self.models))}
        
        # Tính Accuracy trung bình làm final_score
        acc_scores = [r.get("accuracy", 1) for r in results]
        avg_acc = sum(acc_scores) / len(acc_scores)
        
        # Consensus status
        diff = abs(acc_scores[0] - acc_scores[1]) if len(acc_scores) >= 2 else 0
        status = "consensus" if diff <= 1 else "conflict"
        agreement = 1.0 if diff == 0 else (0.5 if diff > 1 else 0.8)

        # Trích xuất lý do tổng hợp
        combined_reasoning = " | ".join([f"{m}: {r.get('reasoning')}" for m, r in individual_results.items()])

        # Tổng hợp Usage cho Judge
        total_usage = {
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in results),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in results),
            "total_tokens": sum(r["usage"]["total_tokens"] for r in results)
        }

        return {
            "final_score": avg_acc,
            "agreement_rate": agreement,
            "faithfulness_avg": sum(r.get("faithfulness", 1) for r in results) / len(results),
            "relevancy_avg": sum(r.get("relevancy", 1) for r in results) / len(results),
            "individual_results": individual_results,
            "status": status,
            "reasoning": combined_reasoning,
            "usage": total_usage
        }

if __name__ == "__main__":
    async def main():
        judge = LLMJudge()
        res = await judge.evaluate_multi_judge("Câu hỏi?", "Trả lời sai", "Đáp án đúng", "Ngữ cảnh")
        print(json.dumps(res, indent=2, ensure_ascii=False))
    asyncio.run(main())
