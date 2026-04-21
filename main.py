import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

async def run_benchmark_with_results(agent_version: str, num_cases: int = None):
    """
    Điều phối toàn bộ quy trình Benchmark.
    """
    print(f"\n🚀 Khởi động Benchmark cho [{agent_version}]...")

    # 1. Load Dataset
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    # Tùy chọn chạy một phần dataset để tiết kiệm thời gian/cost trong lúc debug
    if num_cases:
        dataset = dataset[:num_cases]
        print(f"📝 Chạy thử nghiệm trên {num_cases} cases đầu tiên...")

    # 2. Khởi tạo các components thực tế
    agent = MainAgent()
    evaluator = RetrievalEvaluator()
    judge = LLMJudge(models=["gpt-4o", "gpt-4o-mini"]) # Multi-Judge

    runner = BenchmarkRunner(agent, evaluator, judge)

    # 3. Chạy Benchmark
    print(f"⏳ Đang xử lý {len(dataset)} câu hỏi...")
    results = await runner.run_all(dataset, batch_size=5)

    # 4. Tính toán Metrics tổng quát
    total = len(results)
    avg_score = sum(r["judge"]["final_score"] for r in results) / total
    avg_hit_rate = sum(r["retrieval"]["hit_rate"] for r in results) / total
    avg_mrr = sum(r["retrieval"]["mrr"] for r in results) / total
    avg_agreement = sum(r["judge"]["agreement_rate"] for r in results) / total
    avg_latency = sum(r["latency"] for r in results) / total

    summary = {
        "metadata": {
            "agent_version": agent_version,
            "total_cases": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_accuracy_score": round(avg_score, 2),
            "hit_rate": round(avg_hit_rate, 4),
            "mrr": round(avg_mrr, 4),
            "agreement_rate": round(avg_agreement, 4),
            "avg_latency_sec": round(avg_latency, 2)
        }
    }

    return results, summary

async def main():
    # Trong thực tế Lab, bạn có thể chạy so sánh giữa 2 phiên bản Agent
    # Ở đây chúng ta chạy bản hiện tại và lưu báo cáo.
    
    # CHÚ Ý: Bộ dữ liệu vàng của bạn có 73 câu. 
    # Nếu muốn chạy toàn bộ, hãy bỏ tham số num_cases.
    results, summary = await run_benchmark_with_results("MainAgent_RAG_v1", num_cases=50)
    
    if not results:
        return

    print("\n📊 --- KẾT QUẢ BENCHMARK ---")
    print(json.dumps(summary["metrics"], indent=2))

    # 5. Lưu báo cáo (Artifacts)
    os.makedirs("reports", exist_ok=True)
    
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu báo cáo tại reports/summary.json")
    print(f"👉 Tiếp theo, hãy chạy 'python check_lab.py' để kiểm tra tiêu chuẩn nộp bài.")

if __name__ == "__main__":
    asyncio.run(main())
