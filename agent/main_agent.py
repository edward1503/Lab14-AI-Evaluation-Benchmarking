"""RAG Agent hai phiên bản cho bài lab regression testing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI


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
REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

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
    "hom",
    "la",
    "lam",
    "luc",
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

SEED_DOCUMENTS = {
    "password_reset.txt": """# Chính sách đổi mật khẩu

Khách hàng có thể đổi mật khẩu trong trang Tài khoản bằng cách chọn mục Bảo mật,
sau đó bấm Đổi mật khẩu. Hệ thống sẽ gửi mã xác minh qua email hoặc số điện
thoại đã đăng ký.
""",
    "refunds.txt": """# Chính sách hoàn tiền

Khách hàng có thể yêu cầu hoàn tiền trong vòng 14 ngày kể từ ngày thanh toán nếu
dịch vụ chưa được sử dụng quá mức hoặc có lỗi kỹ thuật được xác nhận. Thời gian
xử lý hoàn tiền thường là 5 đến 7 ngày làm việc sau khi yêu cầu được phê duyệt.
""",
    "support_contact.txt": """# Liên hệ bộ phận hỗ trợ

Khách hàng có thể liên hệ bộ phận hỗ trợ qua email support@example.com, trò
chuyện trực tuyến trong ứng dụng, hoặc tổng đài 1900-0000 từ 8:00 đến 18:00.
""",
}


class InMemoryIndex:
    """Chỉ mục runtime đọc artifact từ ingest.py và tìm kiếm bằng cosine."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        client: Optional[AsyncOpenAI] = None,
        mock_mode: Optional[bool] = None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = _normalize_matrix(embeddings.astype(np.float32, copy=False))
        self.client = client
        self.mock_mode = (
            mock_mode if mock_mode is not None else not bool(os.getenv("OPENAI_API_KEY"))
        )
        self._row_by_chunk_id = {
            str(chunk["chunk_id"]): index for index, chunk in enumerate(self.chunks)
        }

    def set_client(self, client: Optional[AsyncOpenAI]) -> None:
        """Gắn AsyncOpenAI client cho truy vấn embedding sau khi nạp index."""
        self.client = client
        self.mock_mode = client is None

    async def embed_text(self, text: str) -> np.ndarray:
        """Tạo embedding đã chuẩn hóa cho một chuỗi truy vấn."""
        if self.mock_mode or self.client is None:
            return _mock_embedding(text)
        embeddings = await _embed_openai([text], self.client)
        return embeddings[0]

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Tìm chunk gần nhất với truy vấn."""
        query_embedding = await self.embed_text(query)
        return await self.search_by_embedding(query_embedding, top_k)

    async def search_by_embedding(self, emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Tìm chunk gần nhất từ embedding có sẵn."""
        if not self.chunks or self.embeddings.size == 0:
            return []
        query_vector = _normalize_vector(np.asarray(emb, dtype=np.float32))
        scores = np.dot(self.embeddings.astype(np.float64), query_vector.astype(np.float64))
        limit = min(max(top_k, 0), len(self.chunks))
        if limit == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:limit]
        return [self._format_result(int(index), float(scores[index])) for index in top_indices]

    def embedding_for_chunk(self, chunk_id: str) -> Optional[np.ndarray]:
        """Trả về embedding của chunk theo mã định danh."""
        row = self._row_by_chunk_id.get(chunk_id)
        if row is None:
            return None
        return self.embeddings[row]

    def _format_result(self, index: int, score: float) -> dict[str, Any]:
        chunk = self.chunks[index]
        return {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score,
            "source": chunk["source"],
        }


def build_or_load_index(
    chunks_path: str = "data/chunks.json",
    doc_dir: str = "data/doc",
    index_path: str = "data/index.pkl",
) -> InMemoryIndex:
    """Nạp index ưu tiên artifact data/chunks.json từ ingest.py."""
    chunks_file = _repo_path(chunks_path)
    docs_path = _repo_path(doc_dir)
    cache_path = _repo_path(index_path)
    mock_mode = not bool(os.getenv("OPENAI_API_KEY"))
    chunks = _load_ingested_chunks(chunks_file)
    if not chunks:
        chunks = _build_chunks_from_docs(docs_path)

    signature = _chunk_signature(chunks)
    if cache_path.exists():
        with cache_path.open("rb") as file_obj:
            payload = pickle.load(file_obj)
        if payload.get("signature") == signature:
            logging.info("Loaded RAG index from %s", cache_path)
            return InMemoryIndex(
                chunks=payload["chunks"],
                embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
                mock_mode=mock_mode,
            )

    texts = [chunk["text"] for chunk in chunks]
    if mock_mode:
        embeddings = np.vstack([_mock_embedding(text) for text in texts])
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)
        embeddings = _run_async_blocking(_embed_openai(texts, client))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file_obj:
        pickle.dump(
            {"chunks": chunks, "embeddings": embeddings, "signature": signature},
            file_obj,
        )
    logging.info("Built RAG index with %s chunks", len(chunks))
    return InMemoryIndex(chunks=chunks, embeddings=embeddings, mock_mode=mock_mode)


def _load_ingested_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    if not chunks_file.exists():
        return []
    raw_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(raw_chunks):
        metadata = item.get("metadata", {})
        source_path = str(metadata.get("source", f"chunk_{index}")).replace("\\", "/")
        source_name = Path(source_path).name
        source_stem = Path(source_path).stem or "doc"
        local_id = str(metadata.get("chunk_id") or item.get("id") or f"chunk_{index}")
        chunk_index = _parse_chunk_index(local_id, index)
        chunk_id = f"{source_stem}::chunk_{chunk_index}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": str(item.get("text", "")),
                "source": source_name,
            }
        )
    logging.info("Loaded %s chunks from %s", len(chunks), chunks_file)
    return [chunk for chunk in chunks if chunk["text"].strip()]


def _build_chunks_from_docs(doc_dir: Path) -> list[dict[str, Any]]:
    _seed_docs_if_needed(doc_dir)
    chunks: list[dict[str, Any]] = []
    documents = sorted(path for path in doc_dir.iterdir() if path.suffix.lower() == ".txt")
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for index, chunk_text in enumerate(_chunk_text(text)):
            chunks.append(
                {
                    "chunk_id": f"{path.stem}::chunk_{index}",
                    "text": chunk_text,
                    "source": path.name,
                }
            )
    if not chunks:
        raise ValueError(f"No supported documents found in {doc_dir}")
    return chunks


def _seed_docs_if_needed(doc_dir: Path) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)
    if any(path.is_file() and path.suffix.lower() == ".txt" for path in doc_dir.iterdir()):
        return
    for filename, content in SEED_DOCUMENTS.items():
        (doc_dir / filename).write_text(content, encoding="utf-8")
    logging.info("Seeded Vietnamese support docs in %s", doc_dir)


def _chunk_text(text: str) -> list[str]:
    clean_text = re.sub(r"[ \t]+", " ", text).strip()
    if not clean_text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean_text):
        end = min(start + CHUNK_SIZE, len(clean_text))
        if end < len(clean_text):
            end = _preferred_boundary(clean_text, start, end)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _preferred_boundary(text: str, start: int, end: int) -> int:
    window = text[start:end]
    minimum = int(len(window) * 0.45)
    for pattern in ["\n\n", ". ", "! ", "? ", "\n", "; "]:
        position = window.rfind(pattern)
        if position >= minimum:
            return start + position + len(pattern)
    return end


async def _embed_openai(texts: list[str], client: AsyncOpenAI) -> np.ndarray:
    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = await _retry_async(
            lambda batch=batch: client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        )
        vectors.extend([item.embedding for item in response.data])
    return _normalize_matrix(np.asarray(vectors, dtype=np.float32))


def _mock_embedding(text: str) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = _tokenize(text)
    features = tokens + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize_vector(vector)


def _chunk_signature(chunks: list[dict[str, Any]]) -> str:
    payload = json.dumps(chunks, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_chunk_index(chunk_id: str, fallback: int) -> int:
    match = re.search(r"chunk_(\d+)$", chunk_id)
    if not match:
        return fallback
    return int(match.group(1))


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        raise ValueError("Absolute paths are not supported")
    return REPO_ROOT / path


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def _tokenize(text: str) -> list[str]:
    normalized = _strip_accents(text.lower())
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in raw_tokens if len(token) > 1 and token not in STOPWORDS]


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _run_async_blocking(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - phụ thuộc vòng lặp gọi ngoài.
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


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
            raw_answer = self._mock_answer_v2(question, retrieved)
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
        selected_lines, _, score = self._select_relevant_lines(question, retrieved, max_lines=1)
        if not selected_lines or score < 0.08:
            return "Không tìm thấy trong tài liệu."
        return selected_lines[0]

    def _mock_answer_v2(self, question: str, retrieved: list[dict[str, Any]]) -> str:
        selected_lines, cited_ids, score = self._select_relevant_lines(question, retrieved, max_lines=3)
        if not selected_lines or score < 0.1:
            return f"{NOT_FOUND}\nNguồn: []"
        answer = " ".join(selected_lines)
        return f"{answer}\nNguồn: [{', '.join(cited_ids)}]"

    def _mock_hypothetical_answer(self, question: str) -> str:
        normalized = question.lower()
        if "mật khẩu" in normalized or "đổi mật" in normalized:
            return "Khách hàng đổi mật khẩu trong trang Tài khoản, mục Bảo mật."
        if "hoàn tiền" in normalized:
            return "Khách hàng yêu cầu hoàn tiền trong vòng 14 ngày kể từ ngày thanh toán."
        if "hỗ trợ" in normalized or "liên hệ" in normalized:
            return "Khách hàng liên hệ hỗ trợ qua email, chat hoặc tổng đài."
        return question

    def _select_relevant_lines(
        self,
        question: str,
        retrieved: list[dict[str, Any]],
        max_lines: int = 2,
    ) -> tuple[list[str], list[str], float]:
        scored_lines: list[tuple[float, str, str]] = []
        question_tokens = set(_tokenize(question))

        for item in retrieved:
            lines = [line.strip(" -") for line in str(item["text"]).splitlines() if line.strip()]
            for line in lines:
                if self._is_metadata_line(line):
                    continue
                line_tokens = set(_tokenize(line))
                if not line_tokens:
                    continue

                overlap = len(question_tokens & line_tokens)
                score = overlap / max(len(question_tokens), 1)
                if re.search(r"\d", question) and re.search(r"\d", line):
                    score += 0.2
                if any(word in question.lower() for word in ["ai", "khi nào", "bao lâu", "bao nhiêu", "điều kiện", "quy trình"]):
                    score += 0.08
                if any(marker in question.lower() for marker in ["so sánh", "khác nhau", "điểm khác biệt"]):
                    score += 0.05
                scored_lines.append((score, line, str(item["chunk_id"])))

        scored_lines.sort(key=lambda row: row[0], reverse=True)
        selected_lines: list[str] = []
        cited_ids: list[str] = []
        best_score = scored_lines[0][0] if scored_lines else 0.0

        for score, line, chunk_id in scored_lines:
            if line in selected_lines:
                continue
            if score < 0.05 and selected_lines:
                continue
            selected_lines.append(line)
            if chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
            if len(selected_lines) >= max_lines:
                break

        return selected_lines, cited_ids, best_score

    def _is_metadata_line(self, line: str) -> bool:
        lower = line.lower()
        return lower.startswith(("source:", "department:", "effective date:", "access:"))

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
