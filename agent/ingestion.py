"""Nạp tài liệu và xây dựng chỉ mục vector trong bộ nhớ."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - môi trường CI có thể chỉ chạy mock với .md.
    PdfReader = None  # type: ignore[assignment]


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

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
    "password_reset.md": """# Chính sách đổi mật khẩu

Khách hàng có thể đổi mật khẩu trong trang Tài khoản bằng cách chọn mục Bảo mật,
sau đó bấm Đổi mật khẩu. Hệ thống sẽ gửi mã xác minh qua email hoặc số điện
thoại đã đăng ký.

Nếu khách hàng quên mật khẩu, hãy chọn Quên mật khẩu trên màn hình đăng nhập.
Liên kết đặt lại mật khẩu có hiệu lực trong 30 phút. Nhân viên hỗ trợ không bao
giờ yêu cầu khách hàng cung cấp mật khẩu hiện tại.
""",
    "refunds.md": """# Chính sách hoàn tiền

Khách hàng có thể yêu cầu hoàn tiền trong vòng 14 ngày kể từ ngày thanh toán nếu
dịch vụ chưa được sử dụng quá mức hoặc có lỗi kỹ thuật được xác nhận. Khoản hoàn
tiền được xử lý về phương thức thanh toán ban đầu.

Thời gian xử lý hoàn tiền thường là 5 đến 7 ngày làm việc sau khi yêu cầu được
phê duyệt. Các phí phát sinh từ ngân hàng hoặc bên thanh toán thứ ba có thể không
được hoàn lại.
""",
    "support_contact.md": """# Liên hệ bộ phận hỗ trợ

Khách hàng có thể liên hệ bộ phận hỗ trợ qua email support@example.com, trò
chuyện trực tuyến trong ứng dụng, hoặc tổng đài 1900-0000 từ 8:00 đến 18:00 các
ngày làm việc.

Khi liên hệ hỗ trợ, khách hàng nên cung cấp mã tài khoản, email đăng ký, mô tả
vấn đề và ảnh chụp màn hình nếu có. Không gửi mật khẩu, mã OTP hoặc thông tin thẻ
thanh toán qua kênh hỗ trợ.
""",
}


class InMemoryIndex:
    """Chỉ mục vector trong bộ nhớ cho truy hồi RAG."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        client: Optional[AsyncOpenAI] = None,
        mock_mode: Optional[bool] = None,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.chunks = chunks
        self.embeddings = _normalize_matrix(embeddings.astype(np.float32, copy=False))
        self.client = client
        self.embedding_model = embedding_model
        self.mock_mode = (
            mock_mode if mock_mode is not None else not bool(os.getenv("OPENAI_API_KEY"))
        )
        self._row_by_chunk_id = {
            str(chunk["chunk_id"]): index for index, chunk in enumerate(self.chunks)
        }

    def set_client(self, client: Optional[AsyncOpenAI]) -> None:
        """Gắn AsyncOpenAI client cho truy vấn embedding sau khi nạp cache."""
        self.client = client
        self.mock_mode = client is None

    async def embed_text(self, text: str) -> np.ndarray:
        """Tạo embedding đã chuẩn hóa cho một chuỗi truy vấn."""
        if self.mock_mode or self.client is None:
            return mock_embedding(text)
        embeddings = await _embed_openai([text], self.client, self.embedding_model)
        return embeddings[0]

    async def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Tìm các chunk gần nhất với truy vấn bằng cosine similarity."""
        query_embedding = await self.embed_text(query)
        return await self.search_by_embedding(query_embedding, top_k)

    async def search_by_embedding(self, emb: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """Tìm các chunk gần nhất bằng embedding đã có sẵn."""
        if not self.chunks or self.embeddings.size == 0:
            return []
        query_vector = _normalize_vector(np.asarray(emb, dtype=np.float32))
        scores = self.embeddings @ query_vector
        limit = min(max(top_k, 0), len(self.chunks))
        if limit == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:limit]
        return [self._format_result(int(index), float(scores[index])) for index in top_indices]

    def embedding_for_chunk(self, chunk_id: str) -> Optional[np.ndarray]:
        """Trả về embedding của chunk theo mã định danh ổn định."""
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
    corpus_dir: str = "data/corpus",
    index_path: str = "data/index.pkl",
) -> InMemoryIndex:
    """Xây dựng hoặc nạp chỉ mục vector từ cache pickle."""
    corpus_path = _repo_path(corpus_dir)
    cache_path = _repo_path(index_path)
    api_key = os.getenv("OPENAI_API_KEY")
    mock_mode = not bool(api_key)

    if cache_path.exists():
        with cache_path.open("rb") as file_obj:
            payload = pickle.load(file_obj)
        logging.info("Loaded RAG index from %s", cache_path)
        return InMemoryIndex(
            chunks=payload["chunks"],
            embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
            mock_mode=mock_mode,
        )

    _seed_corpus_if_needed(corpus_path)
    chunks = _load_chunks(corpus_path)
    if not chunks:
        raise ValueError(f"No supported documents found in {corpus_path}")

    texts = [chunk["text"] for chunk in chunks]
    if mock_mode:
        embeddings = np.vstack([mock_embedding(text) for text in texts])
    else:
        client = AsyncOpenAI(api_key=api_key, timeout=30.0)
        embeddings = _run_async_blocking(_embed_openai(texts, client, EMBEDDING_MODEL))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file_obj:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, file_obj)
    logging.info("Built RAG index with %s chunks", len(chunks))
    return InMemoryIndex(chunks=chunks, embeddings=embeddings, mock_mode=mock_mode)


def mock_embedding(text: str) -> np.ndarray:
    """Tạo embedding giả lập ổn định bằng SHA-256 cho môi trường không API key."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tokens = _tokenize(text)
    features = tokens + [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize_vector(vector)


async def _embed_openai(
    texts: list[str],
    client: AsyncOpenAI,
    model: str,
) -> np.ndarray:
    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = await _retry_async(
            lambda batch=batch: client.embeddings.create(model=model, input=batch)
        )
        vectors.extend([item.embedding for item in response.data])
    return _normalize_matrix(np.asarray(vectors, dtype=np.float32))


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
    raise RuntimeError("OpenAI embedding request failed") from last_error


def _seed_corpus_if_needed(corpus_path: Path) -> None:
    corpus_path.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in corpus_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"}
    ]
    if existing:
        return
    for filename, content in SEED_DOCUMENTS.items():
        (corpus_path / filename).write_text(content, encoding="utf-8")
    logging.info("Seeded Vietnamese support corpus in %s", corpus_path)


def _load_chunks(corpus_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    documents = sorted(
        path
        for path in corpus_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"}
    )
    for path in documents:
        text = _read_document(path)
        for index, chunk_text in enumerate(_chunk_text(text)):
            chunks.append(
                {
                    "chunk_id": f"{path.stem}::chunk_{index}",
                    "text": chunk_text,
                    "source": path.name,
                }
            )
    return chunks


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        if PdfReader is None:
            logging.warning("Skipping PDF because pypdf is unavailable: %s", path.name)
            return ""
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


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
    boundary_patterns = ["\n\n", ". ", "! ", "? ", "\n", "; "]
    for pattern in boundary_patterns:
        position = window.rfind(pattern)
        if position >= minimum:
            return start + position + len(pattern)
    return end


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
