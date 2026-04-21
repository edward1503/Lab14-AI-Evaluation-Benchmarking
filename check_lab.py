import json
import os

def validate_lab():
    print("\n" + "="*50)
    print("🔍 Đang kiểm tra định dạng bài nộp (EXPERT MODE)...")
    print("="*50)

    required_files = [
        "reports/summary.json",
        "reports/benchmark_results.json",
        "analysis/failure_analysis.md"
    ]

    # 1. Kiểm tra sự tồn tại của tất cả file
    missing = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✅ Tìm thấy: {f}")
        else:
            print(f"❌ Thiếu file: {f}")
            missing.append(f)

    if missing:
        print(f"\n❌ Thiếu {len(missing)} file. Hãy bổ sung trước khi nộp bài.")
        return

    # 2. Kiểm tra nội dung summary.json
    try:
        with open("reports/summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ File reports/summary.json không phải JSON hợp lệ: {e}")
        return

    if "metrics" not in data or "metadata" not in data:
        print("❌ File summary.json thiếu trường 'metrics' hoặc 'metadata'.")
        return

    metrics = data["metrics"]
    metadata = data["metadata"]

    print(f"\n--- Thống kê hiệu năng & Chi phí ---")
    print(f"📊 Tổng số cases: {metadata.get('total', 'N/A')}")
    print(f"⏱️ Thời gian thực thi: {metadata.get('duration_seconds', 0):.1f}s")
    
    # In thông tin Cost & Token (EXPERT)
    total_cost = metadata.get("total_cost_usd", 0)
    print(f"💸 Tổng chi phí Evaluation: ${total_cost:.4f}")
    
    usage = metrics.get("usage", {})
    print(f"💎 Token Usage: {usage.get('prompt_tokens', 0)} (Input) / {usage.get('completion_tokens', 0)} (Output)")
    print(f"⭐️ Điểm trung bình (V2): {metrics.get('avg_score', 0):.2f}/5.0")

    # METRICS CHECKS
    print(f"\n--- Chất lượng RAG ---")
    has_retrieval = "hit_rate" in metrics
    if has_retrieval:
        print(f"✅ Hit Rate: {metrics['hit_rate']*100:.1f}%")
        
    has_multi_judge = "agreement_rate" in metrics
    if has_multi_judge:
        print(f"✅ Judge Agreement Rate: {metrics['agreement_rate']*100:.1f}%")

    if metadata.get("version") == "EXPERT_REGRESSION_RUN":
        print(f"✅ Chế độ Regression: Đã so sánh V1 vs V2")
        decision = data.get("regression", {}).get("decision", "N/A")
        print(f"📢 Quyết định cuối cùng: {decision}")

    print("\n" + "="*50)
    print("🚀 BÀI LAB ĐÃ SẴN SÀNG ĐỂ CHẤM ĐIỂM (Chuẩn Expert)!")
    print("="*50 + "\n")

if __name__ == "__main__":
    validate_lab()
