import re
from typing import Dict, List


STOPWORDS = {
    "a",
    "an",
    "and",
    "bao",
    "bi",
    "cac",
    "cach",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hay",
    "hien",
    "hoi",
    "la",
    "lam",
    "mot",
    "nao",
    "nay",
    "neu",
    "nhieu",
    "nhung",
    "the",
    "thi",
    "trong",
    "va",
    "ve",
    "voi",
}


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z0-9À-ỹà-ỹ]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_b)


class RetrievalEvaluator:
    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: int = 3,
    ) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids[:top_k] else 0.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(self._ids_match(expected_id, retrieved_id) for expected_id in expected_ids for retrieved_id in top_retrieved)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        for index, retrieved_id in enumerate(retrieved_ids):
            if any(self._ids_match(expected_id, retrieved_id) for expected_id in expected_ids):
                return 1.0 / (index + 1)
        return 0.0

    async def score(self, case: Dict, response: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids") or self._derive_expected_ids(case)
        retrieved_ids = response.get("retrieved_ids", []) or response.get("metadata", {}).get("retrieved_ids", [])
        answer = response.get("answer", "")
        expected_answer = case.get("expected_answer", "")
        case_type = case.get("metadata", {}).get("type", "synthetic-rag")

        hit_rate = self.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.calculate_mrr(expected_ids, retrieved_ids)

        if case_type == "edge_out_of_context":
            safe_markers = ["không tìm thấy", "không có", "chưa thấy", "không đủ thông tin"]
            faithfulness = 1.0 if any(marker in answer.lower() for marker in safe_markers) else 0.35
        elif case_type == "edge_ambiguous":
            clarify_markers = ["làm rõ", "cụ thể", "bạn đang hỏi", "chưa rõ", "mơ hồ"]
            faithfulness = 1.0 if any(marker in answer.lower() for marker in clarify_markers) else 0.45
        else:
            answer_overlap = overlap_ratio(answer, expected_answer)
            faithfulness = min(1.0, 0.4 + answer_overlap * 0.6)
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
        avg_hit_rate = sum(item["ragas"]["retrieval"]["hit_rate"] for item in results) / len(results)
        avg_mrr = sum(item["ragas"]["retrieval"]["mrr"] for item in results) / len(results)
        return {"avg_hit_rate": round(avg_hit_rate, 3), "avg_mrr": round(avg_mrr, 3)}

    def _derive_expected_ids(self, case: Dict) -> List[str]:
        metadata = case.get("metadata", {})
        ids = []
        if metadata.get("source_chunk_id"):
            ids.append(str(metadata["source_chunk_id"]))
        if metadata.get("source_chunk_ids"):
            ids.extend(str(item) for item in metadata["source_chunk_ids"])
        return ids

    def _ids_match(self, expected_id: str, retrieved_id: str) -> bool:
        expected = str(expected_id)
        retrieved = str(retrieved_id)
        return expected == retrieved or retrieved.endswith(expected)
