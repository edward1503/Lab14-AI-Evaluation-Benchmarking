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
        
        # 2. Chạy Evaluation
        retrieval_metrics = self.evaluator.score(test_case, response)
        
        context_text = "\n\n".join(response.get("contexts", []))
        judge_result = await self.judge.evaluate_multi_judge(
            question=test_case["question"], 
            answer=response["answer"], 
            ground_truth=test_case["expected_answer"],
            context=context_text
        )
        
        # 3. Gom nhóm theo chuẩn Sample Submission (Tên key là 'ragas' cho cả Retrieval và Faithfulness)
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
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
