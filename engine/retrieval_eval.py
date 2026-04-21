from typing import List, Dict

class RetrievalEvaluator:
    """
    Chuyên tính toán các metrics cho bài toán Retrieval: Hit Rate, MRR.
    """
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids (vị trí 1-indexed).
        """
        if not expected_ids or not retrieved_ids:
            return 0.0
            
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score(self, test_case: Dict, agent_response: Dict) -> Dict:
        """
        Tính toán metrics cho một case đơn lẻ.
        'expected_ids' lấy từ test_case['metadata']['source_chunk_id']
        'retrieved_ids' lấy từ agent_response['retrieved_ids']
        """
        # Hỗ trợ cả 1 ID hoặc List ID
        source_id = test_case.get("metadata", {}).get("source_chunk_id")
        expected_ids = [source_id] if isinstance(source_id, str) else (source_id or [])
        
        retrieved_ids = agent_response.get("retrieved_ids", [])
        
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids)
        }
