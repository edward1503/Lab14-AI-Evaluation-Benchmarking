import json
import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Giả lập việc gọi LLM để tạo dữ liệu (Students will implement this)

class SyntheticDataGenerator:
    def __init__(self):
        self.model = os.getenv("SYNTHETIC_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def call_llm(self, prompt: str) -> Dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a specialized AI for generating high-quality RAG evaluation datasets. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

def _build_prompt(text: str, num_pairs: int = 1, difficulty: str = "easy") -> str:
    if difficulty == "easy":
        prompt = f"""
        Tạo {num_pairs} câu hỏi và câu trả lời đơn giản và trực tiếp từ văn bản dưới đây.
        Câu hỏi phải nằm trực tiếp trong nội dung văn bản.
        
        Văn bản: {text}
        
        Định dạng trả về JSON:
        {{
            "pairs": [
                {{"question": "...", "expected_answer": "..."}}
            ]
        }}
        """

    elif difficulty == "mixed":
        prompt = f"""
        Tạo {num_pairs} câu hỏi và câu trả lời mức độ trung bình từ văn bản dưới đây.
        Câu hỏi nên yêu cầu suy luận nhẹ hoặc kết nối các ý.
        
        Văn bản: {text}
        
        Định dạng trả về JSON:
        {{
            "pairs": [
                {{"question": "...", "expected_answer": "..."}}
            ]
        }}
        """
    
    elif difficulty == "hard":
        prompt = f"""
        Tạo {num_pairs} câu hỏi và câu trả lời khó hoặc lừa (adversarial) từ văn bản dưới đây.
        Câu hỏi có thể về chi tiết nhỏ hoặc yêu cầu hiểu sâu quy trình.
        
        Văn bản: {text}
        
        Định dạng trả về JSON:
        {{
            "pairs": [
                {{"question": "...", "expected_answer": "..."}}
            ]
        }}
        """
    
    else:
        raise ValueError("Invalid difficulty level")
    
    return prompt

async def generate_qa_from_text(text: str, num_pairs: int = 1) -> List[Dict]:
    """
    Sử dụng OpenAI/Anthropic API để tạo các cặp (Question, Expected Answer, Context)
    từ đoạn văn bản cho trước.
    """
    sdg = SyntheticDataGenerator()
    # Chọn độ khó ngẫu nhiên hoặc xen kẽ
    difficulties = ["easy", "mixed", "hard"]
    results = []
    
    for _ in range(num_pairs):
        import random
        diff = random.choice(difficulties)
        prompt = _build_prompt(text, 1, diff)
        
        try:
            content = await sdg.call_llm(prompt)
            for pair in content.get("pairs", []):
                results.append({
                    "question": pair["question"],
                    "expected_answer": pair["expected_answer"],
                    "context": text,
                    "metadata": {"difficulty": diff, "type": "synthetic-rag"}
                })
        except Exception as e:
            print(f"Error generating QA: {e}")
            
    return results

async def main():
    # Kiểm tra và đọc dữ liệu từ chunks.json
    chunks_path = "data/chunks.json"
    if not os.path.exists(chunks_path):
        print(f"❌ File {chunks_path} không tồn tại. Đang sử dụng text mặc định...")
        raw_text = "AI Evaluation là một quy trình kỹ thuật nhằm đo lường chất lượng..."
        qa_pairs = await generate_qa_from_text(raw_text, num_pairs=5)
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        print(f"🚀 Đang sinh dữ liệu từ {len(chunks)} chunks...")
        qa_pairs = []
        
        # Để nhanh chóng và đủ 50 cases, chúng ta có thể lấy mỗi chunk 2 câu hỏi
        for chunk in chunks[:25]: # Giới hạn 25 chunk đầu để demo
            pairs = await generate_qa_from_text(chunk["text"], num_pairs=2)
            # Thêm metadata chunk_id để tính Hit Rate
            for p in pairs:
                p["metadata"]["source_chunk_id"] = chunk.get("id")
            qa_pairs.extend(pairs)
    
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"Done! Saved {len(qa_pairs)} pairs to data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
