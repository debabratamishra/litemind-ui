import chromadb
from chromadb.utils import embedding_functions
import pypdf
from .ollama import stream_ollama
import os
from config import Config
from crewai import Agent, LLM
import os
import httpx
import shutil
import logging  # Add logging for warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Wipe entire Chroma DB directory for a true reset
        if os.path.exists(Config.CHROMA_DB_PATH):
            shutil.rmtree(Config.CHROMA_DB_PATH)
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")  # Default
        collection_name = "documents"
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        self.file_paths = []  # Track uploaded files for re-indexing

    def chunk_text(self, text, chunk_size=500):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def add_document(self, file_path, doc_id, chunk_size=500):
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty page text
                        text += page_text + "\n"
            
            if not text.strip():  # Key Fix: Check if text is empty after extraction
                logger.warning(f"No text extracted from {doc_id}. Skipping indexing.")
                return  # Skip adding to collection
            
            chunks = self.chunk_text(text, chunk_size)
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            
            if not chunks:  # Double-check for empty chunks
                logger.warning(f"Empty chunks for {doc_id}. Skipping.")
                return
            
            self.collection.add(documents=chunks, ids=chunk_ids)
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
            logger.info(f"Successfully indexed {doc_id} with {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error indexing {doc_id}: {str(e)}")
            raise  # Re-raise to propagate to backend for error response

    def recreate_collection(self):
        """Recreate collection with current embedding_function and re-add documents"""
        self.client.delete_collection(name="documents")
        self.collection = self.client.create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )
        for file_path in self.file_paths:
            doc_id = os.path.basename(file_path)
            self.add_document(file_path, doc_id)  # Note: Sync call for simplicity; make async if needed

    async def add_document(self, file_path, doc_id, chunk_size=500):
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        chunks = self.chunk_text(text, chunk_size)
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]  # Unique per file and chunk
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
        self.model_name = model_name
        self.context_length = self._get_context_length(model_name)

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

    def _get_context_length(self, model_name: str) -> int:
        """Synchronously fetch the model's context length from Ollama."""
        model = model_name.replace("ollama/", "") if "ollama/" in model_name else model_name
        with httpx.Client(timeout=10.0) as client:
            try:
                resp = client.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/show",
                    json={"name": model}
                )
                resp.raise_for_status()
                data = resp.json()
                params = data.get('model_info', '')
                # If params is a string, parse to dict
                if isinstance(params, str):
                    import ast
                    params = ast.literal_eval(params)
                # Find key ending with "context_length"
                for key, value in params.items():
                    if key.endswith("context_length"):
                        return int(value)
                return 4096  # default if not found
            except Exception:
                return 4096

    async def _generate_summary(self, text: str, system_prompt: str) -> str:
        """Generate a summary using stream_ollama."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        model = self.model_name.replace("ollama/", "")
        response = ""
        async for chunk in stream_ollama(messages, model=model):
            response += chunk
        return response

    def chunk_text(self, text: str, chunk_size: int):
        """Split text into chunks."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def summarize_context(self, text: str, target_length: int) -> str:
        """Recursively summarize text using map-reduce to fit target length."""
        if len(text) <= target_length:
            return text

        # Split into smaller chunks (approx half target for recursion)
        chunk_size = target_length // 2
        chunks = self.chunk_text(text, chunk_size)
        summaries = []
        for chunk in chunks:
            sum_prompt = "Provide a concise summary of the following text:"
            summ = await self._generate_summary(chunk, sum_prompt)
            summaries.append(summ)

        combined = " ".join(summaries)
        # Recurse if still too long
        return await self.summarize_context(combined, target_length)

    async def query(self, user_query: str, system_prompt: str, n_results: int = 3):
        """Multi-step RAG implemented as an async generator."""
        # 1) Refine the question
        refine_prompt = (
            "Refine the following user query so it is precise, self-contained "
            "and uses full nouns:\n\n"
            f"{user_query}"
        )
        refined = await self.refiner.kickoff_async(refine_prompt)
        refined_query = refined.raw.strip()

        # 2) Retrieve context from the vector-DB
        res = self.rag_service.collection.query(
            query_texts=[refined_query], n_results=n_results
        )
        context = " ".join(res["documents"][0]) if res["documents"] else ""

        # 3) Summarize context if it exceeds model limits
        approx_tokens = len(context) // 4 + 1  # Rough token estimate (1 token ~ 4 chars)
        target_tokens = self.context_length * 0.6  # Conservative target (leave buffer)
        if approx_tokens > target_tokens:
            target_chars = target_tokens * 4
            context = await self.summarize_context(context, target_chars)

        # 4) Compose the answer
        compose_prompt = (
            f"Context:\n{context}\n\nOriginal question:\n{user_query}\n\n"
            f"Follow these instructions when you answer:\n{system_prompt}"
        )
        final = await self.composer.kickoff_async(compose_prompt)
        answer = final.raw

        # 5) Stream back in ~400-character chunks
        for i in range(0, len(answer), 400):
            yield answer[i : i + 400]
