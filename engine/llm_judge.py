import asyncio
import os
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    """
    Hệ thống đánh giá đa mô hình (Multi-Judge) để đảm bảo tính khách quan.
    Sử dụng ít nhất 2 model để chấm điểm và tính toán sự đồng thuận.
    """
    def __init__(self, models: List[str] = ["gpt-4o", "gpt-4o-mini"]):
        self.models = models
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Định nghĩa Rubrics chi tiết theo thang điểm 1-5
        self.rubrics = {
            "accuracy": (
                "5: Hoàn hảo, khớp hoàn toàn với Ground Truth.\n"
                "4: Đúng ý chính, có thể thiếu chi tiết nhỏ không quan trọng.\n"
                "3: Đúng một phần, có sai sót nhỏ hoặc thiếu thông tin quan trọng.\n"
                "2: Sai sót nghiêm trọng, chỉ đúng một vài từ khóa.\n"
                "1: Hoàn toàn sai hoặc không liên quan."
            ),
            "faithfulness": (
                "5: Tuyệt đối trung thực, mọi thông tin đều có trong Context.\n"
                "3: Có dấu hiệu suy luận quá đà nhưng không sai lệch nghiêm trọng.\n"
                "1: Hallucination - Bịa đặt thông tin không có trong Context."
            )
        }

    async def _get_single_score(self, model: str, question: str, answer: str, ground_truth: str, context: str = "") -> Dict:
        """
        Gọi 1 model cụ thể để chấm điểm.
        """
        prompt = f"""Bạn là một chuyên gia thẩm định chất lượng AI Agent. 
Hãy chấm điểm câu trả lời của Agent dựa trên Ground Truth và Context được cung cấp.

[Tiêu chí chấm điểm Accuracy]:
{self.rubrics['accuracy']}

[Tiêu chí chấm điểm Faithfulness]:
{self.rubrics['faithfulness']}

Câu hỏi: {question}
Context: {context}
Câu trả lời của Agent: {answer}
Ground Truth (Đáp án đúng): {ground_truth}

Hãy trả về kết quả dưới dạng JSON với cấu trúc:
{{
    "accuracy_score": <1-5>,
    "faithfulness_score": <1-5>,
    "reasoning": "<Giải thích ngắn gọn tại sao chấm điểm này>"
}}
"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an objective AI Judge. Always output valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2 # Độ nhiễu thấp để kết quả ổn định
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [!] Lỗi khi gọi Judge model {model}: {e}")
            return {"accuracy_score": 0, "faithfulness_score": 0, "reasoning": f"Error: {str(e)}"}

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str, context: str = "") -> Dict[str, Any]:
        """
        Thực hiện đánh giá bằng nhiều model và tính toán Consensus.
        """
        # Chạy song song các Judge models
        tasks = [self._get_single_score(model, question, answer, ground_truth, context) for model in self.models]
        results = await asyncio.gather(*tasks)
        
        # Tổng hợp điểm
        accuracy_scores = [r["accuracy_score"] for r in results if r["accuracy_score"] > 0]
        faithfulness_scores = [r["faithfulness_score"] for r in results if r["faithfulness_score"] > 0]
        
        if not accuracy_scores:
            return {"final_score": 0, "agreement_rate": 0, "reasoning": "Tất cả các Judge đều lỗi."}

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        
        # Tính Agreement Rate (sự đồng nhất)
        # Logic: Nếu lệch điểm giữa 2 model <= 1 thì coi là đồng thuận (1.0), nếu lệch > 1 thì là 0.5
        agreement = 1.0
        if len(accuracy_scores) >= 2:
            diff = abs(accuracy_scores[0] - accuracy_scores[1])
            if diff > 1:
                agreement = 0.5
            elif diff == 0:
                agreement = 1.0
            else:
                agreement = 0.8 # Lệch đúng 1 điểm

        # Tổng hợp reasoning
        reasoning_combined = " | ".join([f"{self.models[i]}: {results[i].get('reasoning')}" for i in range(len(results))])

        return {
            "final_score": avg_accuracy,
            "faithfulness_avg": avg_faithfulness,
            "agreement_rate": agreement,
            "individual_scores": {self.models[i]: results[i] for i in range(len(results))},
            "reasoning": reasoning_combined
        }

if __name__ == "__main__":
    async def test_judge():
        judge = LLMJudge()
        print("🚀 Đang test LLM Judge...")
        
        # Giả lập một case
        q = "Công ty có bao nhiêu ngày nghỉ phép cho nhân viên 4 năm kinh nghiệm?"
        ans = "Nhân viên có 4 năm kinh nghiệm được nghỉ 12 ngày phép năm." # Sai (thực tế là 15)
        gt = "Nhân viên có 3-5 năm kinh nghiệm được nghỉ 15 ngày phép năm."
        ctx = "Level 1: 12 ngày (<3 năm). Level 2: 15 ngày (3-5 năm)."
        
        result = await judge.evaluate_multi_judge(q, ans, gt, ctx)
        print("\n--- Kết quả đánh giá ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(test_judge())
