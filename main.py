import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from agent.main_agent import MainAgent
from data.synthetic_gen import generate_dataset, save_dataset
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


DATASET_PATH = Path("data/golden_set.jsonl")
REPORT_DIR = Path("reports")
SUMMARY_PATH = REPORT_DIR / "summary.json"
BENCHMARK_RESULTS_PATH = REPORT_DIR / "benchmark_results.json"
FAILURE_ANALYSIS_PATH = Path("analysis/failure_analysis.md")


def ensure_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        dataset = generate_dataset()
        save_dataset(dataset)
        return dataset

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = [json.loads(line) for line in file if line.strip()]

    if dataset:
        return dataset

    dataset = generate_dataset()
    save_dataset(dataset)
    return dataset


def summarize_results(version: str, results: List[Dict]) -> Dict:
    total = len(results)
    metrics = {
        "avg_score": round(sum(item["judge"]["final_score"] for item in results) / total, 3),
        "pass_rate": round(sum(1 for item in results if item["status"] == "pass") / total, 3),
        "hit_rate": round(sum(item["ragas"]["retrieval"]["hit_rate"] for item in results) / total, 3),
        "mrr": round(sum(item["ragas"]["retrieval"]["mrr"] for item in results) / total, 3),
        "agreement_rate": round(sum(item["judge"]["agreement_rate"] for item in results) / total, 3),
        "faithfulness": round(sum(item["ragas"]["faithfulness"] for item in results) / total, 3),
        "relevancy": round(sum(item["ragas"]["relevancy"] for item in results) / total, 3),
        "avg_latency_sec": round(sum(item["latency"] for item in results) / total, 4),
        "avg_tokens": round(sum(item["tokens_used"] for item in results) / total, 1),
        "total_cost_usd": round(sum(item["cost_usd"] for item in results), 6),
        "avg_cost_usd": round(sum(item["cost_usd"] for item in results) / total, 6),
    }

    breakdown = {}
    grouped = defaultdict(list)
    for item in results:
        grouped[item["case_type"]].append(item)

    for case_type, group in grouped.items():
        breakdown[case_type] = {
            "count": len(group),
            "pass_rate": round(sum(1 for item in group if item["status"] == "pass") / len(group), 3),
            "avg_score": round(sum(item["judge"]["final_score"] for item in group) / len(group), 3),
        }

    return {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
        "case_breakdown": breakdown,
    }


def compare_versions(v1_summary: Dict, v2_summary: Dict) -> Dict:
    delta_score = round(v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"], 3)
    delta_hit_rate = round(v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"], 3)
    delta_pass_rate = round(v2_summary["metrics"]["pass_rate"] - v1_summary["metrics"]["pass_rate"], 3)
    delta_cost = round(v2_summary["metrics"]["avg_cost_usd"] - v1_summary["metrics"]["avg_cost_usd"], 6)
    delta_latency = round(v2_summary["metrics"]["avg_latency_sec"] - v1_summary["metrics"]["avg_latency_sec"], 4)

    decision = "APPROVE"
    reasons = []

    if delta_score < 0.15:
        decision = "BLOCK"
        reasons.append("Điểm judge chưa cải thiện đủ mạnh.")
    if delta_hit_rate < 0:
        decision = "BLOCK"
        reasons.append("Retrieval quality bị giảm.")
    if delta_pass_rate < 0:
        decision = "BLOCK"
        reasons.append("Pass rate thấp hơn baseline.")

    if not reasons:
        reasons.append("V2 cải thiện chất lượng mà không gây regression đáng kể.")

    return {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "delta": {
            "avg_score": delta_score,
            "hit_rate": delta_hit_rate,
            "pass_rate": delta_pass_rate,
            "avg_cost_usd": delta_cost,
            "avg_latency_sec": delta_latency,
        },
        "decision": decision,
        "reasons": reasons,
    }


def build_failure_analysis(summary: Dict, comparison: Dict, results: List[Dict]) -> str:
    failing_results = [item for item in results if item["status"] == "fail"]
    if not failing_results:
        failing_results = sorted(results, key=lambda item: item["judge"]["final_score"])[:3]

    cluster_counter = Counter(item["case_type"] for item in failing_results)
    cluster_reason_map = {
        "adversarial": "Guardrail hoặc chống prompt injection còn yếu.",
        "ambiguous": "Agent chưa hỏi làm rõ đủ tốt khi ngữ cảnh thiếu.",
        "out_of_context": "Abstention logic chưa ổn định, dễ trả lời theo suy đoán.",
        "scenario": "Retrieval đúng tài liệu nhưng answer synthesis còn chưa bám hết chi tiết.",
        "paraphrase": "Retriever phụ thuộc quá nhiều vào keyword literal.",
        "factoid": "Pipeline nền tảng vẫn còn lỗi ở câu hỏi cơ bản.",
    }

    lines = [
        "# Báo cáo Phân tích Thất bại (Failure Analysis Report)",
        "",
        "## 1. Tổng quan Benchmark",
        f"- **Tổng số cases:** {summary['metadata']['total']}",
        f"- **Tỉ lệ Pass/Fail:** {round(summary['metrics']['pass_rate'] * 100, 1)}% / {round((1 - summary['metrics']['pass_rate']) * 100, 1)}%",
        "- **Điểm RAGAS trung bình:**",
        f"  - Faithfulness: {summary['metrics']['faithfulness']:.2f}",
        f"  - Relevancy: {summary['metrics']['relevancy']:.2f}",
        f"  - Hit Rate: {summary['metrics']['hit_rate']:.2f}",
        f"  - MRR: {summary['metrics']['mrr']:.2f}",
        f"- **Điểm LLM-Judge trung bình:** {summary['metrics']['avg_score']:.2f} / 5.0",
        f"- **Regression Decision:** {comparison['decision']}",
        "",
        "## 2. Phân nhóm lỗi (Failure Clustering)",
        "| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |",
        "|----------|----------|---------------------|",
    ]

    if cluster_counter:
        for case_type, count in cluster_counter.most_common():
            lines.append(f"| {case_type} | {count} | {cluster_reason_map.get(case_type, 'Cần điều tra thêm.')} |")
    else:
        lines.append("| none | 0 | Không có lỗi nghiêm trọng trong lượt benchmark này. |")

    lines.extend(["", "## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)", ""])

    for item in sorted(failing_results, key=lambda result: result["judge"]["final_score"])[:3]:
        case_type = item["case_type"]
        if case_type == "adversarial":
            root_cause = "Prompt defense chưa được ưu tiên ngay tại bước answer synthesis."
        elif case_type == "ambiguous":
            root_cause = "Thiếu cơ chế xác định ngưỡng mơ hồ để buộc agent hỏi lại."
        elif case_type == "out_of_context":
            root_cause = "Retriever hoặc generator chưa đủ tự tin để abstain sạch."
        else:
            root_cause = "Retriever và answer composer chưa phối hợp ổn định ở case khó."

        lines.extend(
            [
                f"### Case {item['case_id']}: {item['test_case']}",
                f"1. **Symptom:** Điểm judge chỉ đạt {item['judge']['final_score']:.2f} và status là `{item['status']}`.",
                f"2. **Why 1:** Câu trả lời tạo ra chưa khớp hoàn toàn với ground truth mong đợi.",
                f"3. **Why 2:** Retrieval metric có hit_rate={item['ragas']['retrieval']['hit_rate']:.2f}, cho thấy grounding chưa đủ tốt.",
                f"4. **Why 3:** Case thuộc nhóm `{case_type}` nên đòi hỏi guardrail hoặc reasoning chính sách tốt hơn câu hỏi thường.",
                "5. **Why 4:** Prompting/generation hiện tại tối ưu cho trả lời nhanh hơn là kiểm tra độ chắc chắn trước khi trả lời.",
                f"6. **Root Cause:** {root_cause}",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Kế hoạch cải tiến (Action Plan)",
            "- [x] Thêm retrieval theo synonym và ưu tiên tài liệu hiện hành để giảm lỗi paraphrase/conflict.",
            "- [x] Thêm cơ chế abstain cho câu hỏi ambiguous và out-of-context.",
            "- [x] Bổ sung multi-judge consensus và regression gate để chặn release khi chất lượng giảm.",
            "- [ ] Tiếp tục thêm reranker semantic hoặc embedding thực để tăng MRR ở case khó.",
        ]
    )

    return "\n".join(lines) + "\n"


async def run_benchmark_with_results(agent_version: str, dataset: List[Dict]) -> Tuple[List[Dict], Dict]:
    runner = BenchmarkRunner(
        MainAgent(agent_version),
        RetrievalEvaluator(),
        LLMJudge(),
    )
    results = await runner.run_all(dataset)
    summary = summarize_results(agent_version, results)
    return results, summary


async def main() -> None:
    dataset = ensure_dataset()

    print(f"🚀 Chạy benchmark với {len(dataset)} test cases...")
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", dataset)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", dataset)
    comparison = compare_versions(v1_summary, v2_summary)
    v2_summary["regression"] = comparison

    REPORT_DIR.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(v2_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    BENCHMARK_RESULTS_PATH.write_text(
        json.dumps(
            {
                "versions": {
                    "Agent_V1_Base": {
                        "summary": v1_summary,
                        "results": v1_results,
                    },
                    "Agent_V2_Optimized": {
                        "summary": v2_summary,
                        "results": v2_results,
                    },
                },
                "comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    FAILURE_ANALYSIS_PATH.write_text(
        build_failure_analysis(v2_summary, comparison, v2_results),
        encoding="utf-8",
    )

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.3f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.3f}")
    print(f"Delta Score: {comparison['delta']['avg_score']:+.3f}")
    print(f"Delta Hit Rate: {comparison['delta']['hit_rate']:+.3f}")
    print(f"Delta Pass Rate: {comparison['delta']['pass_rate']:+.3f}")
    print(f"Avg Cost V2: ${v2_summary['metrics']['avg_cost_usd']:.6f}")
    print(f"Decision: {comparison['decision']}")
    print(f"Saved reports -> {SUMMARY_PATH}, {BENCHMARK_RESULTS_PATH}")
    print(f"Updated analysis -> {FAILURE_ANALYSIS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
