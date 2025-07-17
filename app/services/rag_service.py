import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
from .ollama import stream_ollama
import os
from config import Config

class RAGService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

    def chunk_text(self, text, chunk_size=500):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def add_document(self, file_path, doc_id, chunk_size=500):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        chunks = self.chunk_text(text, chunk_size)
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=chunk_ids)

    async def query(self, query_text, system_prompt="You are a helpful assistant.", n_results=3):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        context = " ".join(results['documents'][0]) if results['documents'] else ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query_text}"}
        ]
        async for chunk in stream_ollama(messages):
            yield chunk