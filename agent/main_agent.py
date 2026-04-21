import os
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    """
    RAG Agent thực tế kết nối với ChromaDB để phục vụ benchmarking.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.name = f"SupportAgent-{model_name}"
        # Cấu hình Embeddings (phải khớp với ingest.py)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Kết nối tới Vector DB hiện có
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings,
            collection_name="langchain"
        )
        
        # Khởi tạo LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Định nghĩa System Prompt cho RAG
        self.template = """Bạn là một trợ lý hỗ trợ nội bộ chuyên nghiệp. 
Hãy trả lời câu hỏi dựa TRỰC TIẾP vào các đoạn văn bản (context) được cung cấp dưới đây. 

Nếu không thể tìm thấy câu trả lời trong context, hãy trả lời: 
"Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu hệ thống. Vui lòng liên hệ bộ phận liên quan."

Context: 
{context}

Câu hỏi: {question}
"""
        self.prompt = ChatPromptTemplate.from_template(self.template)

    async def query(self, question: str) -> Dict:
        """
        Quy trình RAG thực tế:
        1. Retrieval: Tìm kiếm top-K chunks liên quan.
        2. Generation: Sinh câu trả lời dựa trên context.
        """
        # 1. Retrieval (lấy top 3 chunks)
        docs = self.vectorstore.similarity_search(question, k=3)
        
        contexts = [doc.page_content for doc in docs]
        retrieved_ids = [doc.metadata.get("chunk_id", "unknown") for doc in docs]
        sources = list(set([os.path.basename(doc.metadata.get("source", "unknown")) for doc in docs]))

        # 2. Generation
        context_text = "\n\n".join(contexts)
        full_prompt = self.prompt.format(context=context_text, question=question)
        
        # Gọi LLM (sử dụng ainvoke cho async)
        response = await self.llm.ainvoke(full_prompt)
        
        return {
            "answer": response.content,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": self.llm.model_name,
                "sources": sources
            }
        }

if __name__ == "__main__":
    import asyncio
    
    async def test():
        agent = MainAgent()
        print(f"🚀 Đang test Agent: {agent.name}")
        
        test_questions = [
            "Chính sách nghỉ phép năm của nhân viên có 4 năm kinh nghiệm?",
            "Làm thế nào để reset mật khẩu?"
        ]
        
        for q in test_questions:
            print(f"\n--- Q: {q} ---")
            resp = await agent.query(q)
            print(f"A: {resp['answer']}")
            print(f"Retrieved Chunks: {resp['retrieved_ids']}")

    asyncio.run(test())
