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

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy Retrieval metrics (Hit Rate, MRR)
        retrieval_scores = self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge (Accuracy, Faithfulness)
        # Nối tất cả context tìm được để Judge đối chiếu
        context_text = "\n\n".join(response.get("contexts", []))
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"], 
            answer=response["answer"], 
            ground_truth=test_case["expected_answer"],
            context=context_text
        )
        
        return {
            "test_case": test_case["question"],
            "ground_truth": test_case["expected_answer"],
            "agent_response": response["answer"],
            "latency": latency,
            "retrieval": retrieval_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] <= 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
