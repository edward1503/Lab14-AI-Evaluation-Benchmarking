import asyncio
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


REPO_ROOT = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "data" / "golden_set.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.json"
REPORT_DIR = REPO_ROOT / "reports"
SUMMARY_PATH = REPORT_DIR / "summary.json"
BENCHMARK_RESULTS_PATH = REPORT_DIR / "benchmark_results.json"
FAILURE_ANALYSIS_PATH = REPO_ROOT / "analysis" / "failure_analysis.md"


def build_chunk_id_map() -> Dict[str, str]:
    if not CHUNKS_PATH.exists():
        return {}
    raw_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    mapping = {}
    for index, item in enumerate(raw_chunks):
        raw_id = str(item.get("id", f"chunk_{index}"))
        source = str(item.get("metadata", {}).get("source", f"chunk_{index}")).replace("\\", "/")
        source_stem = Path(source).stem or "doc"
        mapping[raw_id] = f"{source_stem}::{raw_id}"
    return mapping


def load_dataset() -> List[Dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Thiếu data/golden_set.jsonl. Hãy chạy `python data/synthetic_gen.py` trước.")

    chunk_map = build_chunk_id_map()
    dataset = []
    with DATASET_PATH.open("r", encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.setdefault("metadata", {})
            item.setdefault("id", f"case-{index:03d}")

            expected_ids = item.get("expected_retrieval_ids")
            if not expected_ids:
                raw_ids = []
                if metadata.get("source_chunk_id"):
                    raw_ids.append(str(metadata["source_chunk_id"]))
                if metadata.get("source_chunk_ids"):
                    raw_ids.extend(str(raw) for raw in metadata["source_chunk_ids"])
                expected_ids = [chunk_map.get(raw_id, raw_id) for raw_id in raw_ids]
                item["expected_retrieval_ids"] = expected_ids

            dataset.append(item)
    return dataset


def summarize_results(version: str, results: List[Dict], judge: LLMJudge) -> Dict:
    total = len(results)
    cost_report = judge.get_cost_report()
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
        "total_cost_usd": cost_report["total_cost_usd"],
        "avg_cost_usd": cost_report["cost_per_eval"],
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
    delta = {
        "avg_score": round(v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"], 3),
        "hit_rate": round(v2_summary["metrics"]["hit_rate"] - v1_summary["metrics"]["hit_rate"], 3),
        "mrr": round(v2_summary["metrics"]["mrr"] - v1_summary["metrics"]["mrr"], 3),
        "pass_rate": round(v2_summary["metrics"]["pass_rate"] - v1_summary["metrics"]["pass_rate"], 3),
        "avg_latency_sec": round(v2_summary["metrics"]["avg_latency_sec"] - v1_summary["metrics"]["avg_latency_sec"], 4),
        "avg_tokens": round(v2_summary["metrics"]["avg_tokens"] - v1_summary["metrics"]["avg_tokens"], 1),
        "avg_cost_usd": round(v2_summary["metrics"]["avg_cost_usd"] - v1_summary["metrics"]["avg_cost_usd"], 6),
    }

    reasons = []
    decision = "APPROVE"
    if delta["avg_score"] < 0.1:
        decision = "BLOCK"
        reasons.append("Điểm judge chưa cải thiện đủ rõ so với baseline.")
    if delta["hit_rate"] < 0:
        decision = "BLOCK"
        reasons.append("Retrieval quality bị giảm.")
    if delta["pass_rate"] < 0:
        decision = "BLOCK"
        reasons.append("Pass rate thấp hơn baseline.")
    if not reasons:
        reasons.append("V2 cải thiện chất lượng và không tạo regression đáng kể.")

    return {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "delta": delta,
        "decision": decision,
        "reasons": reasons,
    }


def build_failure_analysis(summary: Dict, comparison: Dict, results: List[Dict]) -> str:
    failing = [item for item in results if item["status"] == "fail"]
    if not failing:
        failing = sorted(results, key=lambda item: item["judge"]["final_score"])[:3]

    clusters = Counter(item["case_type"] for item in failing)
    reason_map = {
        "synthetic-rag": "Câu hỏi chuẩn vẫn còn case retrieval hoặc answer synthesis chưa đủ chính xác.",
        "edge_out_of_context": "Agent chưa từ chối suy đoán đủ mạnh ở câu hỏi ngoài tài liệu.",
        "edge_ambiguous": "Agent chưa hỏi làm rõ nhất quán khi ngữ cảnh thiếu.",
        "edge_conflicting_info": "Cần trình bày song song hai nguồn thay vì chọn một vế.",
        "multiturn_context_carryover": "Agent chưa mang ngữ cảnh nhiều lượt ổn định.",
        "multiturn_correction": "Agent chưa xử lý phản biện/đính chính đủ mượt.",
        "adversarial_prompt_injection": "Guardrail chống prompt injection còn yếu.",
        "adversarial_goal_hijacking": "Agent dễ bị kéo ra ngoài scope hỗ trợ tài liệu.",
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

    if clusters:
        for case_type, count in clusters.most_common():
            lines.append(f"| {case_type} | {count} | {reason_map.get(case_type, 'Cần điều tra thêm.')} |")
    else:
        lines.append("| none | 0 | Không có lỗi nghiêm trọng trong lượt benchmark này. |")

    lines.extend(["", "## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)", ""])
    for item in sorted(failing, key=lambda row: row["judge"]["final_score"])[:3]:
        case_type = item["case_type"]
        lines.extend(
            [
                f"### Case {item['case_id']}: {item['test_case']}",
                f"1. **Symptom:** Điểm judge chỉ đạt {item['judge']['final_score']:.2f} và status là `{item['status']}`.",
                f"2. **Why 1:** Câu trả lời chưa khớp hoàn toàn với ground truth mong đợi.",
                f"3. **Why 2:** Retrieval metric có hit_rate={item['ragas']['retrieval']['hit_rate']:.2f}, cho thấy grounding chưa tối ưu ở case này.",
                f"4. **Why 3:** Case thuộc nhóm `{case_type}` nên đòi hỏi reasoning hoặc guardrail khác với câu hỏi fact đơn giản.",
                "5. **Why 4:** Prompt/generation hiện tại ưu tiên trả lời nhanh, đôi lúc chưa kiểm tra đủ độ chắc chắn trước khi trả lời.",
                f"6. **Root Cause:** {reason_map.get(case_type, 'Cần điều tra thêm ở tầng retrieval/prompting.')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Kế hoạch cải tiến (Action Plan)",
            "- [x] Nối retrieval metric thật vào benchmark thay vì placeholder.",
            "- [x] Thêm offline multi-judge consensus để pipeline luôn benchmark được không cần API.",
            "- [x] Chuẩn hóa report `summary.json`, `benchmark_results.json`, và spot-check report để phục vụ submit.",
            "- [ ] Tiếp tục cải thiện câu trả lời ở edge cases bằng reranker semantic hoặc prompt clarify chuyên biệt.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_benchmark(version: str, dataset: List[Dict]) -> Tuple[List[Dict], Dict, LLMJudge]:
    agent = MainAgent(version=version)
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)
    summary = summarize_results(f"Agent_{version.upper()}", results, judge)
    return results, summary, judge


async def main() -> None:
    dataset = load_dataset()
    print(f"🚀 Chạy benchmark với {len(dataset)} test cases...")

    v1_results, v1_summary, v1_judge = await run_benchmark("v1", dataset)
    v2_results, v2_summary, v2_judge = await run_benchmark("v2", dataset)

    comparison = compare_versions(v1_summary, v2_summary)
    v2_summary["regression"] = comparison

    verify = LLMJudge.verify_judge(v2_results)
    kappa = LLMJudge.calculate_cohens_kappa(v2_results)

    REPORT_DIR.mkdir(exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(v2_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    BENCHMARK_RESULTS_PATH.write_text(
        json.dumps(
            {
                "versions": {
                    "v1": {"summary": v1_summary, "results": v1_results},
                    "v2": {"summary": v2_summary, "results": v2_results},
                },
                "comparison": comparison,
                "spot_check": verify,
                "cohens_kappa": kappa,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LLMJudge.export_spot_check_report(verify, kappa, str(REPORT_DIR / "spot_check.md"))
    FAILURE_ANALYSIS_PATH.write_text(build_failure_analysis(v2_summary, comparison, v2_results), encoding="utf-8")

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.3f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.3f}")
    print(f"Delta Score: {comparison['delta']['avg_score']:+.3f}")
    print(f"Delta Hit Rate: {comparison['delta']['hit_rate']:+.3f}")
    print(f"Delta Pass Rate: {comparison['delta']['pass_rate']:+.3f}")
    print(f"Decision: {comparison['decision']}")
    print(f"Spot Check Kappa: {kappa}")
    print(f"Saved -> {SUMMARY_PATH.relative_to(REPO_ROOT)}, {BENCHMARK_RESULTS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
