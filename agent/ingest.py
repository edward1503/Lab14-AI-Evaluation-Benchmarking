import json
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def ingest_docs():
    # 1. Load documents
    doc_dir = "data/doc"
    documents = []
    for file in os.listdir(doc_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(doc_dir, file), encoding="utf-8")
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} documents.")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    # Process chunks for export and vector store
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        chunk_data = {
            "id": chunk_id,
            "text": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "chunk_id": chunk_id,
            },
        }
        processed_chunks.append(chunk_data)

    print(f"Created {len(processed_chunks)} chunks.")

    # 3. Export chunks for SDG (before embedding as requested)
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
    print("Exported chunks to data/chunks.json")

    # 4. Embed and store in ChromaDB
    # We use processed_chunks to ensure the metadata matches exactly what we exported
    texts = [c["text"] for c in processed_chunks]
    metadatas = [c["metadata"] for c in processed_chunks]
    ids = [c["id"] for c in processed_chunks]

    Chroma.from_texts(
        texts=texts,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        metadatas=metadatas,
        ids=ids,
        persist_directory="chroma_db",
    )
    print("Stored chunks in ChromaDB at 'chroma_db/'")


if __name__ == "__main__":
    ingest_docs()
