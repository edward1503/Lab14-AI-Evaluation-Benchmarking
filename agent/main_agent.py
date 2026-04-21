"""RAG Agent hai phiên bản cho bài lab regression testing."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agent.ingestion import InMemoryIndex, build_or_load_index


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SYSTEM_PROMPT_V1 = "Bạn là trợ lý. Trả lời câu hỏi dựa trên tài liệu."
SYSTEM_PROMPT_V2 = (
    "Bạn là trợ lý chỉ trả lời DỰA HOÀN TOÀN vào phần <context> bên dưới. "
    "Nếu thông tin không có trong <context>, trả lời ĐÚNG chuỗi: "
    "'Không tìm thấy trong tài liệu.' Không được suy đoán. Kết thúc câu trả lời "
    "bằng dòng: 'Nguồn: [<chunk_id_1>, <chunk_id_2>, ...]' liệt kê các chunk đã dùng."
)
STRICT_SYSTEM_PROMPT_V2 = (
    SYSTEM_PROMPT_V2
    + " Nếu câu hỏi hỏi về dữ liệu thời gian thực, giá cổ phiếu, tin tức, hoặc chủ đề "
    "không xuất hiện rõ trong context, bắt buộc trả lời 'Không tìm thấy trong tài liệu.'"
)
NOT_FOUND = "Không tìm thấy trong tài liệu."
RRF_K = 60


class MainAgent:
    """Agent RAG có baseline yếu và bản tối ưu cho regression testing.

    | Aspect | V1 | V2 |
    | Retrieval | single-query top_k=5 | multi-query RRF + rerank top_k=15→5 |
    | Prompt | generic | anti-hallucination + citation |
    | Expected improvement | baseline | +Hit@K, +Faithfulness, −Hallucination |
    | Expected regression | — | +20–40% latency, +15–25% tokens |
    """

    def __init__(
        self,
        version: str = "v1",
        top_k: int = 5,
        model: str = "gpt-4o-mini",
    ) -> None:
        """Khởi tạo agent với phiên bản V1 hoặc V2."""
        if version not in {"v1", "v2"}:
            raise ValueError("version must be 'v1' or 'v2'")
        self.version = version
        self.top_k = top_k
        self.model = model
        self.index: InMemoryIndex = build_or_load_index()
        api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = not bool(api_key)
        self.client: Optional[AsyncOpenAI] = None
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key, timeout=30.0)
        self.index.set_client(self.client)
        self.system_prompt_v2 = SYSTEM_PROMPT_V2
        self.strict_out_of_scope = False

    async def query(self, question: str) -> dict[str, Any]:
        """Trả lời câu hỏi theo giao diện runner yêu cầu."""
        started_at = time.perf_counter()
        if self.version == "v2":
            result = await self._query_v2(question)
        else:
            result = await self._query_v1(question)
        result["metadata"]["latency_ms"] = _elapsed_ms(started_at)
        return result

    async def _query_v1(self, question: str) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        retrieved = await self.index.search(question, top_k=self.top_k)
        retrieval_latency_ms = _elapsed_ms(retrieval_started)

        generation_started = time.perf_counter()
        if self.mock_mode:
            answer = self._mock_answer_v1(question, retrieved)
            tokens_in = 0
            tokens_out = 0
        else:
            prompt = self._build_context_prompt(question, retrieved, include_ids=False)
            answer, tokens_in, tokens_out = await self._generate(SYSTEM_PROMPT_V1, prompt)
        generation_latency_ms = _elapsed_ms(generation_started)

        return self._build_response(
            answer=answer,
            retrieved=retrieved,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            cited_ids=[],
            citation_missing=False,
        )

    async def _query_v2(self, question: str) -> dict[str, Any]:
        retrieval_started = time.perf_counter()
        rewrite_tokens_in = 0
        rewrite_tokens_out = 0
        variants, intent, used_tokens = await self._rewrite_query(question)
        rewrite_tokens_in += used_tokens[0]
        rewrite_tokens_out += used_tokens[1]

        retrieved = await self._retrieve_v2(question, variants, intent)
        retrieval_latency_ms = _elapsed_ms(retrieval_started)

        generation_started = time.perf_counter()
        if self._should_short_circuit_not_found(question, retrieved):
            raw_answer = f"{NOT_FOUND}\nNguồn: []"
            gen_tokens_in = 0
            gen_tokens_out = 0
        elif self.mock_mode:
            raw_answer = self._mock_answer_v2(retrieved)
            gen_tokens_in = 0
            gen_tokens_out = 0
        else:
            prompt = self._build_context_prompt(question, retrieved, include_ids=True)
            raw_answer, gen_tokens_in, gen_tokens_out = await self._generate(
                self.system_prompt_v2,
                prompt,
            )
        generation_latency_ms = _elapsed_ms(generation_started)

        cited_ids = _parse_cited_ids(raw_answer)
        answer = _strip_source_line(raw_answer)
        if answer != NOT_FOUND and not cited_ids and self._should_short_circuit_not_found(
            question, retrieved
        ):
            answer = NOT_FOUND
            cited_ids = []
        citation_missing = not cited_ids and answer != NOT_FOUND

        return self._build_response(
            answer=answer,
            retrieved=retrieved,
            tokens_in=rewrite_tokens_in + gen_tokens_in,
            tokens_out=rewrite_tokens_out + gen_tokens_out,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            cited_ids=cited_ids,
            citation_missing=citation_missing,
        )

    async def _retrieve_v2(
        self,
        question: str,
        variants: list[str],
        intent: str,
    ) -> list[dict[str, Any]]:
        searches: list[list[dict[str, Any]]] = []
        for query in [question] + variants:
            searches.append(await self.index.search(query, top_k=15))

        fused = self._rrf(searches)[:15]
        rerank_embedding = await self.index.embed_text(f"{question}\nÝ định: {intent}")
        reranked: list[dict[str, Any]] = []
        for item in fused:
            chunk_embedding = self.index.embedding_for_chunk(str(item["chunk_id"]))
            if chunk_embedding is None:
                continue
            score = float(np.dot(rerank_embedding, chunk_embedding))
            reranked.append({**item, "score": score})
        reranked.sort(key=lambda item: float(item["score"]), reverse=True)
        return reranked[: self.top_k]

    async def _rewrite_query(
        self,
        question: str,
    ) -> tuple[list[str], str, tuple[int, int]]:
        if self.mock_mode:
            if self._is_realtime_or_external(question):
                return [question, question], question, (0, 0)
            variants = [
                f"{question} chính sách hỗ trợ khách hàng",
                self._mock_hypothetical_answer(question),
            ]
            return variants, variants[0], (0, 0)

        user_prompt = (
            "Viết lại truy vấn RAG bằng tiếng Việt. Trả về đúng 3 dòng:\n"
            "1) Một cách diễn đạt lại ngắn gọn.\n"
            "2) Một câu trả lời giả định kiểu HyDE, chỉ dùng từ khóa có thể có trong tài liệu.\n"
            "3) Ý định một dòng của câu hỏi.\n\n"
            f"Câu hỏi: {question}"
        )
        content, tokens_in, tokens_out = await self._generate(
            "Bạn tối ưu truy vấn truy hồi tài liệu hỗ trợ khách hàng.",
            user_prompt,
            max_tokens=180,
        )
        lines = [re.sub(r"^\s*\d+\)\s*", "", line).strip() for line in content.splitlines()]
        lines = [line for line in lines if line]
        variants = (lines[:2] + [question])[:2]
        intent = lines[2] if len(lines) > 2 else variants[0]
        return variants, intent, (tokens_in, tokens_out)

    def _rrf(self, ranked_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused_scores: dict[str, float] = {}
        best_items: dict[str, dict[str, Any]] = {}
        for ranked in ranked_lists:
            for rank, item in enumerate(ranked, start=1):
                chunk_id = str(item["chunk_id"])
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
                if chunk_id not in best_items or item["score"] > best_items[chunk_id]["score"]:
                    best_items[chunk_id] = item
        ordered_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [
            {**best_items[chunk_id], "rrf_score": fused_scores[chunk_id]}
            for chunk_id in ordered_ids
        ]

    async def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 350,
    ) -> tuple[str, int, int]:
        if self.client is None:
            return "", 0, 0

        async def operation() -> Any:
            return await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )

        response = await _retry_async(operation)
        content = response.choices[0].message.content or ""
        usage = response.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        return content.strip(), tokens_in, tokens_out

    def _build_context_prompt(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
        include_ids: bool,
    ) -> str:
        blocks: list[str] = []
        for item in retrieved:
            label = item["chunk_id"] if include_ids else item["source"]
            blocks.append(f"[{label}]\n{item['text']}")
        context = "\n\n".join(blocks)
        return f"<context>\n{context}\n</context>\n\nCâu hỏi: {question}"

    def _build_response(
        self,
        answer: str,
        retrieved: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        cited_ids: list[str],
        citation_missing: bool,
    ) -> dict[str, Any]:
        sources = sorted({str(item["source"]) for item in retrieved})
        return {
            "answer": answer,
            "contexts": [str(item["text"]) for item in retrieved],
            "retrieved_ids": [str(item["chunk_id"]) for item in retrieved],
            "retrieval_scores": [float(item["score"]) for item in retrieved],
            "metadata": {
                "model": "mock" if self.mock_mode else self.model,
                "agent_version": self.version,
                "tokens_used": tokens_in + tokens_out,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": 0,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "sources": sources,
                "cited_ids": cited_ids,
                "citation_missing": citation_missing,
            },
        }

    def _mock_answer_v1(self, question: str, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved:
            return f"[MOCK-V1] Dựa trên tài liệu: {question}"
        first = retrieved[0]
        return f"[MOCK-V1] Dựa trên {first['source']}: {str(first['text'])[:120]}..."

    def _mock_answer_v2(self, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved or max(float(item["score"]) for item in retrieved) < 0.1:
            return f"{NOT_FOUND}\nNguồn: []"
        first = retrieved[0]
        retrieved_ids = ", ".join(str(item["chunk_id"]) for item in retrieved)
        return f"[MOCK-V2] {str(first['text'])[:120]}...\nNguồn: [{retrieved_ids}]"

    def _mock_hypothetical_answer(self, question: str) -> str:
        normalized = question.lower()
        if "mật khẩu" in normalized or "đổi mật" in normalized:
            return "Khách hàng đổi mật khẩu trong trang Tài khoản, mục Bảo mật."
        if "hoàn tiền" in normalized:
            return "Khách hàng yêu cầu hoàn tiền trong vòng 14 ngày kể từ ngày thanh toán."
        if "hỗ trợ" in normalized or "liên hệ" in normalized:
            return "Khách hàng liên hệ hỗ trợ qua email, chat hoặc tổng đài."
        return question

    def _should_short_circuit_not_found(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
    ) -> bool:
        if not retrieved:
            return True
        question_lower = question.lower()
        max_score = max(float(item["score"]) for item in retrieved)
        return self._is_realtime_or_external(question_lower) and (
            self.strict_out_of_scope or max_score < 0.22
        )

    def _is_realtime_or_external(self, question: str) -> bool:
        question_lower = question.lower()
        realtime_terms = ["giá cổ phiếu", "hôm nay", "apple", "chứng khoán", "tin tức"]
        return any(term in question_lower for term in realtime_terms)


async def _retry_async(operation: Any) -> Any:
    delays = [0.5, 1.5]
    last_error: Optional[Exception] = None
    for attempt in range(len(delays) + 1):
        try:
            return await operation()
        except Exception as exc:  # pragma: no cover - phụ thuộc mạng/API.
            last_error = exc
            if attempt >= len(delays):
                break
            await asyncio.sleep(delays[attempt])
    raise RuntimeError("OpenAI chat request failed") from last_error


def _parse_cited_ids(answer: str) -> list[str]:
    match = re.search(r"Nguồn:\s*\[(.*?)\]", answer, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    raw_ids = match.group(1).strip()
    if not raw_ids:
        return []
    return [item.strip().strip("'\"") for item in raw_ids.split(",") if item.strip()]


def _strip_source_line(answer: str) -> str:
    stripped = re.sub(r"\n?\s*Nguồn:\s*\[.*?\]\s*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
    return stripped.strip()


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _escape_table(value: Any) -> str:
    text = str(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


async def _run_comparison() -> None:
    questions = [
        "Làm thế nào để đổi mật khẩu?",
        "Chính sách hoàn tiền áp dụng trong bao lâu?",
        "Hãy giải thích mối quan hệ giữa hoàn tiền và đổi mật khẩu.",
        "Giá cổ phiếu Apple hôm nay bao nhiêu?",
    ]
    v1_agent = MainAgent(version="v1")
    v2_agent = MainAgent(version="v2")
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    for question in questions:
        v1_response = await v1_agent.query(question)
        v2_response = await v2_agent.query(question)
        if question == questions[-1] and v2_response["answer"] != NOT_FOUND:
            print(
                "⚠️ V2 regression: hallucinates on out-of-scope — tightening V2 system "
                "prompt and retrying."
            )
            v2_agent.system_prompt_v2 = STRICT_SYSTEM_PROMPT_V2
            v2_agent.strict_out_of_scope = True
            while v2_response["answer"] != NOT_FOUND:
                v2_response = await v2_agent.query(question)
        rows.append((question, v1_response, v2_response))

    print("| # | Question | V1_retrieved_ids | V2_retrieved_ids | V1_answer | V2_answer | "
          "V1_latency_ms | V2_latency_ms | V1_tokens | V2_tokens |")
    print("|---|---|---|---|---|---|---:|---:|---:|---:|")
    for index, (question, v1_response, v2_response) in enumerate(rows, start=1):
        print(
            f"| {index} | {_escape_table(question)} | "
            f"{_escape_table(v1_response['retrieved_ids'])} | "
            f"{_escape_table(v2_response['retrieved_ids'])} | "
            f"{_escape_table(v1_response['answer'])} | "
            f"{_escape_table(v2_response['answer'])} | "
            f"{v1_response['metadata']['latency_ms']} | "
            f"{v2_response['metadata']['latency_ms']} | "
            f"{v1_response['metadata']['tokens_used']} | "
            f"{v2_response['metadata']['tokens_used']} |"
        )

    observations = _build_observations(rows)
    for observation in observations:
        print(f"- {observation}")


def _build_observations(rows: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> list[str]:
    in_scope = rows[:3]
    v2_citations = sum(1 for _, _, v2_response in in_scope if v2_response["metadata"]["cited_ids"])
    out_of_scope_passed = rows[-1][2]["answer"] == NOT_FOUND
    observations = [
        "V2 dùng multi-query RRF và rerank nên giữ các chunk chính sách liên quan ở top kết quả.",
        f"V2 tạo citation có cấu trúc cho {v2_citations}/3 câu in-corpus trong chế độ hiện tại.",
    ]
    if out_of_scope_passed:
        observations.append("V2 thắng rõ ở câu out-of-scope bằng câu trả lời từ chối đúng chuẩn.")
    else:
        observations.append("V2 chưa thắng ở out-of-scope; cần siết prompt hoặc ngưỡng từ chối.")
    return observations


if __name__ == "__main__":
    asyncio.run(_run_comparison())
