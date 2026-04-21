import re
from typing import Dict, List


STOPWORDS = {
    "bị",
    "cho",
    "có",
    "của",
    "đã",
    "để",
    "gì",
    "khi",
    "không",
    "là",
    "làm",
    "một",
    "này",
    "nếu",
    "theo",
    "thì",
    "trong",
    "và",
    "với",
}


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9À-ỹ]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


class RetrievalEvaluator:
    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        if not expected_ids:
            return 1.0 if not top_retrieved else 0.0
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for index, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (index + 1)
        return 0.0

    async def score(self, case: Dict, response: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])
        answer = response.get("answer", "")
        expected_answer = case.get("expected_answer", "")
        case_type = case.get("metadata", {}).get("type", "factoid")

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        if case_type in {"out_of_context", "ambiguous"}:
            safe_markers = [
                "chưa",
                "không thấy",
                "không có",
                "cần nói rõ",
                "ngữ cảnh",
                "không suy đoán",
            ]
            faithfulness = 1.0 if any(marker in answer.lower() for marker in safe_markers) else 0.35
        else:
            similarity = overlap_ratio(answer, expected_answer)
            faithfulness = min(1.0, 0.45 + similarity * 0.55)
            faithfulness = round((faithfulness + hit_rate) / 2, 3)

        relevancy = round(min(1.0, overlap_ratio(answer, case.get("question", "")) + 0.25), 3)

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": relevancy,
            "retrieval": {
                "hit_rate": round(hit_rate, 3),
                "mrr": round(mrr, 3),
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected_ids,
            },
        }

    async def evaluate_batch(self, results: List[Dict]) -> Dict:
        if not results:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        hit_rate = sum(item["ragas"]["retrieval"]["hit_rate"] for item in results) / len(results)
        mrr = sum(item["ragas"]["retrieval"]["mrr"] for item in results) / len(results)
        return {"avg_hit_rate": round(hit_rate, 3), "avg_mrr": round(mrr, 3)}
