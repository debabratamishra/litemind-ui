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
import logging
import asyncio
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from typing import List, Dict, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class RAGService:
    def __init__(self):
        # Wipe entire Chroma DB directory for a true reset
        if os.path.exists(Config.CHROMA_DB_PATH):
            shutil.rmtree(Config.CHROMA_DB_PATH)
        
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.default_chunk_size = 500
        
        collection_name = "documents"
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        self.file_paths = []
        
        # BM25 related attributes
        self.bm25_corpus = []  # Store tokenized documents for BM25
        self.bm25_model = None
        self.document_chunks = []  # Store original chunks for retrieval
        self.chunk_ids = []  # Store chunk IDs for mapping
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25: tokenize, lowercase, remove stopwords and punctuation"""
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        tokens = [token for token in tokens 
                 if token not in self.stop_words and token not in string.punctuation]
        
        return tokens

    def chunk_text(self, text, chunk_size=500):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def add_document(self, file_path, doc_id, chunk_size=None):
        """Add document with specified chunk size and build BM25 index"""
        if chunk_size is None:
            chunk_size = self.default_chunk_size

        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if not text.strip():
                logger.warning(f"No text extracted from {doc_id}. Skipping indexing.")
                return

            chunks = self.chunk_text(text, chunk_size)
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

            if not chunks:
                logger.warning(f"Empty chunks for {doc_id}. Skipping.")
                return

            # Add to ChromaDB for vector search
            self.collection.add(documents=chunks, ids=chunk_ids)

            # Add to BM25 corpus
            for i, chunk in enumerate(chunks):
                processed_tokens = self.preprocess_text(chunk)
                self.bm25_corpus.append(processed_tokens)
                self.document_chunks.append(chunk)
                self.chunk_ids.append(chunk_ids[i])

            # Rebuild BM25 model
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)

            if file_path not in self.file_paths:
                self.file_paths.append(file_path)

            logger.info(f"Successfully indexed {doc_id} with {len(chunks)} chunks (chunk_size: {chunk_size}).")

        except Exception as e:
            logger.error(f"Error indexing {doc_id}: {str(e)}")
            raise

    async def recreate_collection(self):
        """Recreate collection with current embedding_function and re-add documents"""
        try:
            self.client.delete_collection(name="documents")
        except:
            pass

        self.collection = self.client.create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

        # Reset BM25 components
        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []
        self.chunk_ids = []

        # Re-index existing files with current chunk size
        for file_path in self.file_paths:
            doc_id = os.path.basename(file_path)
            await self.add_document(file_path, doc_id, self.default_chunk_size)  # Use await instead of asyncio.run()


    def bm25_search(self, query: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Perform BM25 search and return results with scores"""
        if not self.bm25_model or not self.document_chunks:
            return []

        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25_model.get_scores(query_tokens)
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append((
                    self.chunk_ids[idx],
                    self.document_chunks[idx],
                    float(scores[idx])
                ))
        
        return results

    def vector_search(self, query: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Perform vector search and return results with scores"""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            vector_results = []
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0], 
                results['distances'][0]
            )):
                # Convert distance to similarity score (assuming cosine distance)
                similarity = 1 - distance
                vector_results.append((doc_id, document, similarity))
            
            return vector_results
        except Exception as e:
            logger.error(f"Vector search error: {str(e)}")
            return []

    def reciprocal_rank_fusion(self, bm25_results: List[Tuple], vector_results: List[Tuple], k: int = 60) -> List[str]:
        """Combine BM25 and vector search results using Reciprocal Rank Fusion"""
        
        # Create score dictionaries
        bm25_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, _, _) in enumerate(bm25_results)}
        vector_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, _, _) in enumerate(vector_results)}
        
        # Combine scores
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            combined_scores[doc_id] = bm25_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, _ in sorted_results]

    def hybrid_search(self, query: str, n_results: int = 3) -> List[str]:
        """Perform hybrid search combining BM25 and vector search"""
        
        # Perform both searches
        bm25_results = self.bm25_search(query, n_results * 2)  # Get more candidates
        vector_results = self.vector_search(query, n_results * 2)
        
        if not bm25_results and not vector_results:
            return []
        
        # Use RRF to combine results
        fused_doc_ids = self.reciprocal_rank_fusion(bm25_results, vector_results)
        
        # Get the actual document content for top results
        top_doc_ids = fused_doc_ids[:n_results]
        
        # Create mapping from doc_id to content
        id_to_content = {}
        for doc_id, content, _ in bm25_results + vector_results:
            if doc_id not in id_to_content:
                id_to_content[doc_id] = content
        
        # Return documents in fused order
        result_documents = []
        for doc_id in top_doc_ids:
            if doc_id in id_to_content:
                result_documents.append(id_to_content[doc_id])
        
        return result_documents

    async def query(self, query_text, system_prompt="You are a helpful assistant.", messages=[], n_results=3, use_hybrid_search=False):
        """Query using either hybrid search or traditional vector search"""
        
        # Build context from history
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] != 'system'])
        full_query = f"{history_context}\nUser: {query_text}" if history_context else query_text
        
        # Choose search method
        if use_hybrid_search and self.bm25_model:
            logger.info("Using hybrid search")
            documents = self.hybrid_search(full_query, n_results)
            context = " ".join(documents)
        else:
            logger.info("Using vector search")
            results = self.collection.query(query_texts=[full_query], n_results=n_results)
            context = " ".join(results['documents'][0]) if results['documents'] else ""

        # Generate response
        llm_messages = messages + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query_text}"}
        ]

        async for chunk in stream_ollama(llm_messages):
            yield chunk

# Update CrewAIRAGOrchestrator to support hybrid search
class CrewAIRAGOrchestrator:
    def __init__(self, rag_service: RAGService, model_name="gemma3n:e2b"):
        self.rag_service = rag_service
        
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        
        self.ollama_llm = LLM(
            provider="ollama",
            model=model_name,
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=0.0
        )
        
        self.model_name = model_name
        self.context_length = 4096  # Default value, will be set properly when needed
        
        self.refiner = Agent(
            role="Query Refiner",
            goal="Clarify user questions for retrieval.",
            backstory="Understands intent of the prompt and rewrites prompts with more details.",
            llm=self.ollama_llm
        )
        
        self.composer = Agent(
            role="Answer Composer",
            goal="Craft final answers with citations.",
            backstory="Synthesises context into helpful replies.",
            llm=self.ollama_llm
        )

    async def _initialize_context_length(self):
        """Initialize context length asynchronously when needed"""
        if self.context_length == 4096:  # Only if still default
            self.context_length = await self._get_context_length(self.model_name)


    async def _get_context_length(self, model_name: str) -> int:
        """Asynchronously fetch the model's context length from Ollama."""
        model = model_name.replace("ollama/", "") if "ollama/" in model_name else model_name
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{os.getenv('OLLAMA_API_BASE', 'http://localhost:11434')}/api/show",
                    json={"name": model}
                )
                resp.raise_for_status()
                data = resp.json()
                params = data.get('model_info', '')
                
                if isinstance(params, str):
                    import ast
                    params = ast.literal_eval(params)
                
                for key, value in params.items():
                    if key.endswith("context_length"):
                        return int(value)
                
                return 4096
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
        
        chunk_size = target_length // 2
        chunks = self.chunk_text(text, chunk_size)
        summaries = []
        
        for chunk in chunks:
            sum_prompt = "Provide a concise summary of the following text:"
            summ = await self._generate_summary(chunk, sum_prompt)
            summaries.append(summ)
        
        combined = " ".join(summaries)
        return await self.summarize_context(combined, target_length)

    async def query(self, user_query: str, system_prompt: str, messages=[], n_results: int = 3, use_hybrid_search: bool = False):
        """Multi-step RAG with optional hybrid search"""
        
        # Incorporate history into refinement
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] != 'system'])
        refine_prompt = (
            f"Previous conversation:\n{history_context}\n\n"
            "Refine the following user query so it is precise, self-contained and uses full nouns:\n\n"
            f"{user_query}"
        )
        
        refined = await self.refiner.kickoff_async(refine_prompt)
        refined_query = refined.raw.strip()
        logger.info(f"Refined query: {refined_query}")
        
        # Retrieve using chosen method
        if use_hybrid_search and self.rag_service.bm25_model:
            documents = self.rag_service.hybrid_search(refined_query, n_results)
            context = " ".join(documents)
        else:
            res = self.rag_service.collection.query(query_texts=[refined_query], n_results=n_results)
            context = " ".join(res["documents"][0]) if res["documents"] else ""
        
        # Summarize context if needed
        approx_tokens = len(context) // 4 + 1
        target_tokens = self.context_length * 0.6
        
        if approx_tokens > target_tokens:
            target_chars = target_tokens * 4
            context = await self.summarize_context(context, target_chars)
        
        # Compose answer
        compose_prompt = (
            f"Previous conversation:\n{history_context}\n\n"
            f"Context:\n{context}\n\nOriginal question:\n{user_query}\n\n"
            f"Follow these instructions when you answer:\n{system_prompt}"
        )
        
        final = await self.composer.kickoff_async(compose_prompt)
        answer = final.raw
        
        # Stream back
        for i in range(0, len(answer), 400):
            yield answer[i : i + 400]
