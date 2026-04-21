import asyncio
import json
import os
import time
from typing import Dict, List, Tuple
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

async def run_single_version(version_name: str, agent: MainAgent, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    """Chạy benchmark cho 1 phiên bản cụ thể."""
    print(f"⏳ Đang chạy Benchmark cho phiên bản: {version_name}...")
    evaluator = RetrievalEvaluator()
    judge = LLMJudge(models=["gpt-4o-mini", "gpt-3.5-turbo"])
    runner = BenchmarkRunner(agent, evaluator, judge)
    
    results = await runner.run_all(dataset, batch_size=5)
    
    # Tính toán metrics cho version này
    total = len(results)
    metrics = {
        "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
        "hit_rate": sum(r["ragas"]["hit_rate"] for r in results) / total,
        "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
    }
    return results, metrics

async def main():
    print("🚀 Bắt đầu quy trình Đánh giá hồi quy (Regression Benchmark)...")
    
    # 1. Load Data
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    # Chạy trên toàn bộ bộ dữ liệu vàng (73 câu)
    print(f"📝 Chế độ Full Benchmark: Chạy trên {len(dataset)} câu hỏi.")

    # 2. Chạy V1 (Baseline)
    v1_agent = MainAgent(model_name="gpt-3.5-turbo")
    v1_results, v1_metrics = await run_single_version("V1 (Baseline)", v1_agent, dataset)

    # 3. Chạy V2 (Optimized)
    v2_agent = MainAgent(model_name="gpt-4o-mini")
    v2_results, v2_metrics = await run_single_version("V2 (Optimized)", v2_agent, dataset)

    # 4. Phân tích hồi quy & Quyết định
    delta = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    decision = "APPROVE" if delta >= 0 else "BLOCK"
    
    print(f"\n📊 --- KẾT QUẢ SO SÁNH ---")
    print(f"V1 Score: {v1_metrics['avg_score']:.2f} | V2 Score: {v2_metrics['avg_score']:.2f}")
    print(f"Delta: {delta:+.2f} -> QUYẾT ĐỊNH: {decision}")

    # 5. Lưu kết quả chuẩn Sample Submission
    os.makedirs("reports", exist_ok=True)
    
    # summary.json
    summary = {
        "metadata": {
            "total": len(dataset),
            "version": "REGRESSION_RUN",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions_compared": ["V1", "V2"]
        },
        "metrics": v2_metrics, # Mặc định hiển thị metrics của bản mới nhất
        "regression": {
            "v1": {
                "score": v1_metrics["avg_score"],
                "hit_rate": v1_metrics["hit_rate"],
                "judge_agreement": v1_metrics["agreement_rate"]
            },
            "v2": {
                "score": v2_metrics["avg_score"],
                "hit_rate": v2_metrics["hit_rate"],
                "judge_agreement": v2_metrics["agreement_rate"]
            },
            "decision": decision
        }
    }
    
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # benchmark_results.json
    results_combined = {
        "v1": v1_results,
        "v2": v2_results
    }
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_combined, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã tạo báo cáo tại reports/. Hãy chạy 'python check_lab.py' để xác nhận.")

if __name__ == "__main__":
    asyncio.run(main())