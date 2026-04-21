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
    
    results = await runner.run_all(dataset, batch_size=10)
    
    # Tính toán metrics & Tích lũy Usage/Cost
    total = len(results)
    total_cost = sum(r["cost_usd"] for r in results)
    total_prompt_tokens = sum(r["usage"]["agent"]["prompt_tokens"] + r["usage"]["judge"]["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["usage"]["agent"]["completion_tokens"] + r["usage"]["judge"]["completion_tokens"] for r in results)
    
    metrics = {
        "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
        "hit_rate": sum(r["ragas"]["hit_rate"] for r in results) / total,
        "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        "total_cost_usd": total_cost,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
    }
    
    print(f"✅ Hoàn thành {version_name}. Chi phí: ${total_cost:.4f} | Token: {metrics['usage']['total_tokens']}")
    return results, metrics

async def main():
    print("🚀 Bắt đầu quy trình Đánh giá hồi quy Expert (Regression Benchmark)...")
    
    # 1. Load Data
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    # Chế độ Full Benchmark (73 câu)
    print(f"📝 Số lượng câu hỏi: {len(dataset)}")

    start_run = time.perf_counter()

    # 2. Chạy V1 (Baseline)
    v1_agent = MainAgent(model_name="gpt-3.5-turbo")
    v1_results, v1_metrics = await run_single_version("V1 (Baseline)", v1_agent, dataset)

    # 3. Chạy V2 (Optimized)
    v2_agent = MainAgent(model_name="gpt-4o-mini")
    v2_results, v2_metrics = await run_single_version("V2 (Optimized)", v2_agent, dataset)

    total_duration = time.perf_counter() - start_run

    # 4. Phân tích hồi quy & Quyết định
    delta = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    decision = "APPROVE" if delta >= 0 else "BLOCK"
    
    # 5. In bảng thống kê Expert ra console
    print("\n" + "="*50)
    print("📊 --- BÁO CÁO TỔNG KẾT EXPERT ---")
    print(f"⏱️ Tổng thời gian chạy: {total_duration:.1f}s ({total_duration/len(dataset)/2:.2f}s/case)")
    print(f"💸 Tổng chi phí (V1+V2): ${v1_metrics['total_cost_usd'] + v2_metrics['total_cost_usd']:.4f}")
    print(f"💎 Tổng Token (V1+V2): {v1_metrics['usage']['total_tokens'] + v2_metrics['usage']['total_tokens']}")
    print("-" * 50)
    print(f"V1 Score: {v1_metrics['avg_score']:.2f} | V2 Score: {v2_metrics['avg_score']:.2f}")
    print(f"Delta: {delta:+.2f} -> QUYẾT ĐỊNH: {decision}")
    print("="*50 + "\n")

    # 6. Lưu kết quả chuẩn Sample Submission
    os.makedirs("reports", exist_ok=True)
    
    summary = {
        "metadata": {
            "total": len(dataset),
            "version": "EXPERT_REGRESSION_RUN",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": total_duration,
            "total_cost_usd": v1_metrics['total_cost_usd'] + v2_metrics['total_cost_usd']
        },
        "metrics": v2_metrics,
        "regression": {
            "v1": {
                "score": v1_metrics["avg_score"],
                "cost": v1_metrics["total_cost_usd"],
                "usage": v1_metrics["usage"]
            },
            "v2": {
                "score": v2_metrics["avg_score"],
                "cost": v2_metrics["total_cost_usd"],
                "usage": v2_metrics["usage"]
            },
            "decision": decision
        }
    }
    
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    results_combined = {"v1": v1_results, "v2": v2_results}
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_combined, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo báo cáo Expert tại reports/. Chạy 'python check_lab.py' để xác nhận.")

if __name__ == "__main__":
    asyncio.run(main())
