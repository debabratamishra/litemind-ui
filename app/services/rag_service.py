import chromadb
from chromadb.utils import embedding_functions
import pypdf
from .ollama import stream_ollama
import os
from config import Config
from crewai import Agent, LLM
import os

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
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        chunks = self.chunk_text(text, chunk_size)
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=chunk_ids)

    # Old (single agent) RAG for fallback or simple use
    async def query(self, query_text, system_prompt="You are a helpful assistant. You need to answer the user based on the context of the document. If the user asks anything which is not there in the context of the uploaded document, then just answer that you can't help with anything outside of the context of the document.",
                     n_results=3):
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        context = " ".join(results['documents'][0]) if results['documents'] else ""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query_text}"}
        ]
        async for chunk in stream_ollama(messages):
            yield chunk

# ===================== CREWAI MULTI-AGENT ORCHESTRATION =====================

class CrewAIRAGOrchestrator:
    def __init__(self, rag_service: RAGService, model_name="gemma3n:e2b"):
        self.rag_service = rag_service

        # Ensure the model string is properly prefixed
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"

        self.ollama_llm = LLM(
            provider="ollama",
            model=model_name,
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=0.0
        )

        self.refiner = Agent(
            role="Query Refiner",
            goal="Clarify user questions for retrieval.",
            backstory="Understands intent and rewrites prompts.",
            llm=self.ollama_llm
        )

        self.composer = Agent(
            role="Answer Composer",
            goal="Craft final answers with citations.",
            backstory="Synthesises context into helpful replies.",
            llm=self.ollama_llm
        )

    # ----------------------------- async generator -----------------------------
    async def query(self, user_query: str, system_prompt: str, n_results: int = 3):
        """Multi-step RAG implemented as an *async generator* so callers can
        `async for` over the outgoing chunks."""
        # 1) refine the question
        refine_prompt = (
            "Refine the following user query so it is precise, self-contained "
            "and uses full nouns:\n\n"
            f"{user_query}"
        )
        refined = await self.refiner.kickoff_async(refine_prompt)
        refined_query = refined.raw.strip()          # LiteAgentOutput → text

        # 2) retrieve context from the vector-DB
        res = self.rag_service.collection.query(
            query_texts=[refined_query], n_results=n_results
        )
        context = " ".join(res["documents"][0]) if res["documents"] else ""

        # 3) compose the answer
        compose_prompt = (
            f"Context:\n{context}\n\nOriginal question:\n{user_query}\n\n"
            f"Follow these instructions when you answer:\n{system_prompt}"
        )
        final = await self.composer.kickoff_async(compose_prompt)
        answer = final.raw

        # 4) stream back in ~400-character chunks
        for i in range(0, len(answer), 400):
            yield answer[i : i + 400]

# ===================== END CREWAI ORCHESTRATION =====================