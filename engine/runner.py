import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        latency = round(time.perf_counter() - start_time, 4)

        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
            test_case=test_case,
            response=response,
            ragas=ragas_scores,
        )

        metadata = response.get("metadata", {})
        return {
            "case_id": test_case["id"],
            "test_case": test_case["question"],
            "case_type": test_case.get("metadata", {}).get("type", "factoid"),
            "expected_answer": test_case["expected_answer"],
            "agent_response": response["answer"],
            "latency": latency,
            "tokens_used": metadata.get("tokens_used", 0),
            "cost_usd": metadata.get("cost_usd", 0.0),
            "sources": metadata.get("sources", []),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "pass" if judge_result["final_score"] >= 3.5 else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 8) -> List[Dict]:
        results = []
        for index in range(0, len(dataset), batch_size):
            batch = dataset[index:index + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
