import asyncio
import time
from typing import List, Dict
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        
        # Bảng giá OpenAI chính thức (Cost per 1M tokens)
        self.pricing = {
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        }

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        prices = self.pricing.get(model, self.pricing["gpt-4o-mini"]) # Default to mini
        cost = (prompt_tokens * prices["prompt"] + completion_tokens * prices["completion"]) / 1_000_000
        return cost

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy Evaluation
        retrieval_metrics = self.evaluator.score(test_case, response)
        
        # Lấy expected chunk ID để so sánh Attribution
        source_id = test_case.get("metadata", {}).get("source_chunk_id")
        expected_ids = [source_id] if isinstance(source_id, str) else (source_id or [])
        is_retrieved = any(doc_id in response.get("retrieved_ids", []) for doc_id in expected_ids)

        context_text = "\n\n".join(response.get("contexts", []))
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"], 
            answer=response["answer"], 
            ground_truth=test_case["expected_answer"],
            context=context_text
        )
        
        # 3. Định vị lỗi Hallucination (Error Attribution)
        error_attribution = "N/A"
        if judge_result["faithfulness_avg"] < 3.5:
            if not is_retrieved:
                error_attribution = f"Retrieval Error: Relevant chunk(s) {expected_ids} not found in Top-K."
            else:
                error_attribution = "Generation Error: Relevant chunk was retrieved but LLM failed to use it correctly (Hallucination)."
        elif judge_result["final_score"] < 3.5:
            error_attribution = "Reasoning Error: Truthful to context but failed to answer the question correctly."

        # 4. Tính toán Chi phí & Usage
        agent_usage = response["metadata"]["usage"]
        judge_usage = judge_result["usage"]
        
        # Tính chi phí Agent
        agent_cost = self._calculate_cost(response["metadata"]["model"], agent_usage["prompt_tokens"], agent_usage["completion_tokens"])
        
        # Tính chi phí Judge (Tổng của các model trong judge)
        judge_cost = 0
        for model, res in judge_result["individual_results"].items():
            u = res.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
            judge_cost += self._calculate_cost(model, u["prompt_tokens"], u["completion_tokens"])
        
        total_case_cost = agent_cost + judge_cost

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "cost_usd": total_case_cost,
            "error_attribution": error_attribution,
            "ragas": {
                "hit_rate": float(retrieval_metrics["hit_rate"]),
                "mrr": float(retrieval_metrics["mrr"]),
                "faithfulness": float(judge_result["faithfulness_avg"]),
                "relevancy": float(judge_result["relevancy_avg"])
            },
            "judge": {
                "final_score": judge_result["final_score"],
                "agreement_rate": judge_result["agreement_rate"],
                "individual_results": judge_result["individual_results"],
                "status": judge_result["status"]
            },
            "usage": {
                "agent": agent_usage,
                "judge": judge_usage,
                "total_tokens": agent_usage["total_tokens"] + judge_usage["total_tokens"]
            },
            "status": "fail" if judge_result["final_score"] <= 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
