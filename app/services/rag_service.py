import chromadb
from chromadb.utils import embedding_functions
import pypdf  # retained for legacy compatibility
from .ollama import stream_ollama
import os
from config import Config
from crewai import Agent, LLM
import httpx
import shutil
import logging
import asyncio
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64

from .file_ingest import ingest_file

# Configuration flags
ENABLE_SIMPLE_IMAGE_INDEXING = os.getenv("ENABLE_SIMPLE_IMAGE_INDEXING", "true").lower() == "true"
MAX_IMAGES_PER_DOC = int(os.getenv("MAX_IMAGES_PER_DOC", "10"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class RAGService:
    def __init__(self):
        # Wipe Chroma DB on startup for true reset
        if os.path.exists(Config.CHROMA_DB_PATH):
            shutil.rmtree(Config.CHROMA_DB_PATH)

        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)

        # Text embedding function (CPU-friendly)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.default_chunk_size = 500

        # Single TEXT collection for all content
        self.text_collection = self.client.create_collection(
            name="documents_text",
            embedding_function=self.embedding_function
        )

        # Storage for uploaded file paths
        self.file_paths = []

        # BM25 attributes for text chunks
        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []  # original text chunks
        self.chunk_ids = []

        self.stop_words = set(stopwords.words('english'))

        # Directory to persist extracted images
        self.image_cache_dir = Path("./uploads/imgcache")
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]
        return tokens

    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks with slight overlap."""
        overlap = min(50, chunk_size // 10)
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _create_simple_image_references(self, images: List[dict], doc_id: str) -> List[str]:
        """Create simple, clean image references for indexing."""
        if not (ENABLE_SIMPLE_IMAGE_INDEXING and images):
            return []
        
        references = []
        for i, rec in enumerate(images[:MAX_IMAGES_PER_DOC]):
            try:
                meta = rec.get('metadata', {})
                filename = meta.get('filename', 'unknown')
                
                # Create clean reference without noisy captions
                if 'page_number' in meta:
                    ref = f"[IMAGE REFERENCE] Document: {filename}, Page: {meta['page_number']}, Image: {i+1}"
                elif 'slide_number' in meta:
                    ref = f"[IMAGE REFERENCE] Presentation: {filename}, Slide: {meta['slide_number']}, Image: {i+1}"
                else:
                    ref = f"[IMAGE REFERENCE] File: {filename}, Image: {i+1}"
                
                references.append(ref)
                
            except Exception as e:
                logger.warning(f"Failed to create image reference {i}: {e}")
                continue
        
        logger.info(f"Created {len(references)} image references")
        return references

    async def add_document(self, file_path, doc_id, chunk_size=None):
        """
        Add document of any supported format with clean image handling.
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size

        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return

            logger.info(f"Processing document: {doc_id}")

            # Parse using unified ingest pipeline
            text_chunks, image_records, table_texts = ingest_file(path)

            # Merge tables into text stream
            for t in table_texts:
                if t["content"] and t["content"].strip():
                    text_chunks.append({
                        "content": t["content"],
                        "metadata": {**t.get("metadata", {}), "is_table": True}
                    })

            # Chunk long text into manageable pieces
            flat_texts = []
            for blk in text_chunks:
                content = blk["content"]
                meta = blk.get("metadata", {})
                if not content or not content.strip():
                    continue
                
                # Skip very short or very long chunks that might be noise
                content = content.strip()
                if len(content) < 20 or len(content) > 10000:
                    continue
                
                for ch in self.chunk_text(content, chunk_size):
                    flat_texts.append({"content": ch, "metadata": meta})

            # Add simple image references (clean, no AI captioning)
            img_references = self._create_simple_image_references(image_records, doc_id)
            for ref in img_references:
                flat_texts.append({
                    "content": ref,
                    "metadata": {"is_image_reference": True, "filename": path.name}
                })

            # Add to Chroma TEXT collection and BM25
            if flat_texts:
                ids = [f"{doc_id}_text_{i}" for i in range(len(flat_texts))]
                docs = [x["content"] for x in flat_texts]
                metadatas = [x["metadata"] for x in flat_texts]
                
                self.text_collection.add(documents=docs, ids=ids, metadatas=metadatas)

                # Update BM25 index
                for i, d in enumerate(docs):
                    tokens = self.preprocess_text(d)
                    self.bm25_corpus.append(tokens)
                    self.document_chunks.append(d)
                    self.chunk_ids.append(ids[i])
                
                if self.bm25_corpus:
                    self.bm25_model = BM25Okapi(self.bm25_corpus)

            # Save images to disk for potential future VLM usage
            saved_images = 0
            if image_records:
                for i, rec in enumerate(image_records):
                    try:
                        img_bytes = rec["image_bytes"]
                        img_name = f"{doc_id}_img_{i}.png"
                        out_path = self.image_cache_dir / img_name
                        
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        saved_images += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to save image {i}: {e}")
                        continue

            if file_path not in self.file_paths:
                self.file_paths.append(file_path)

            logger.info(f"Successfully indexed {doc_id}: {len(flat_texts)} text chunks, {saved_images} images saved")

        except Exception as e:
            logger.error(f"Error indexing {doc_id}: {str(e)}")
            raise

    async def recreate_collection(self):
        """Recreate text collection and rebuild indexes."""
        try:
            self.client.delete_collection(name="documents_text")
        except Exception:
            pass

        self.text_collection = self.client.create_collection(
            name="documents_text",
            embedding_function=self.embedding_function
        )

        # Reset BM25
        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []
        self.chunk_ids = []

        # Re-index all files
        for file_path in self.file_paths:
            doc_id = os.path.basename(file_path)
            await self.add_document(file_path, doc_id, self.default_chunk_size)

    def bm25_search(self, query: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Perform BM25 keyword search."""
        if not self.bm25_model or not self.document_chunks:
            return []
        
        q_tokens = self.preprocess_text(query)
        if not q_tokens:
            return []
        
        scores = self.bm25_model.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], self.document_chunks[idx], float(scores[idx])))
        
        return results

    def vector_search_text(self, query: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Perform semantic vector search over text chunks."""
        try:
            results = self.text_collection.query(query_texts=[query], n_results=n_results)
            if not results['documents'] or not results['documents'][0]:
                return []
            
            out = []
            for doc_id, document, distance in zip(
                results['ids'][0], 
                results['documents'][0], 
                results['distances'][0]
            ):
                similarity = 1 - distance
                out.append((doc_id, document, similarity))
            
            return out
        except Exception as e:
            logger.error(f"Vector text search error: {str(e)}")
            return []

    def reciprocal_rank_fusion(self, bm25_results: List[Tuple], vector_results: List[Tuple], k: int = 60) -> List[str]:
        """Combine BM25 and vector search results using Reciprocal Rank Fusion."""
        bm25_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, *_rest) in enumerate(bm25_results)}
        vector_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, *_rest) in enumerate(vector_results)}
        
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            combined_scores[doc_id] = bm25_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_results]

    def hybrid_search(self, query: str, n_results: int = 3) -> List[str]:
        """Perform hybrid search combining BM25 and vector search."""
        bm25_results = self.bm25_search(query, n_results * 2)
        vec_text_results = self.vector_search_text(query, n_results * 2)

        if not (bm25_results or vec_text_results):
            return []

        fused_text_ids = self.reciprocal_rank_fusion(bm25_results, vec_text_results)
        top_text_ids = fused_text_ids[:n_results]

        id_to_content = {}
        for doc_id, content, _ in bm25_results + vec_text_results:
            if doc_id not in id_to_content:
                id_to_content[doc_id] = content

        result_documents = [id_to_content[i] for i in top_text_ids if i in id_to_content]
        return result_documents

    async def query(self, query_text, system_prompt="You are a helpful assistant.", messages=[], n_results=3, use_hybrid_search=False):
        """Query with hybrid search or simple vector search."""
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] != 'system'])
        full_query = f"{history_context}\nUser: {query_text}" if history_context else query_text

        if use_hybrid_search and self.bm25_model:
            logger.info("Using hybrid search (BM25 + semantic)")
            text_docs = self.hybrid_search(full_query, n_results)
            context = " ".join(text_docs)
        else:
            logger.info("Using semantic vector search")
            results = self.text_collection.query(query_texts=[full_query], n_results=n_results)
            context = " ".join(results['documents'][0]) if results.get('documents') else ""

        llm_messages = messages + [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query_text}"}
        ]
        
        async for chunk in stream_ollama(llm_messages):
            yield chunk


# CrewAI Orchestrator (unchanged from previous version)
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
        self.context_length = 4096
        
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

    async def _get_context_length(self, model_name: str) -> int:
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
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def summarize_context(self, text: str, target_length: int) -> str:
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
        if self.context_length == 4096:
            self.context_length = await self._get_context_length(self.model_name)
        
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages if msg['role'] != 'system'])
        refine_prompt = (
            f"Previous conversation:\n{history_context}\n\n"
            "Refine the following user query so it is precise, self-contained and uses full nouns:\n\n"
            f"{user_query}"
        )
        
        refined = await self.refiner.kickoff_async(refine_prompt)
        refined_query = refined.raw.strip()
        logger.info(f"Refined query: {refined_query}")
        
        if use_hybrid_search and self.rag_service.bm25_model:
            documents = self.rag_service.hybrid_search(refined_query, n_results)
            context = " ".join(documents)
        else:
            res = self.rag_service.text_collection.query(query_texts=[refined_query], n_results=n_results)
            context = " ".join(res["documents"][0]) if res["documents"] else ""
        
        approx_tokens = len(context) // 4 + 1
        target_tokens = self.context_length * 0.6
        
        if approx_tokens > target_tokens:
            target_chars = int(target_tokens * 4)
            context = await self.summarize_context(context, target_chars)
        
        compose_prompt = (
            f"Previous conversation:\n{history_context}\n\n"
            f"Context:\n{context}\n\nOriginal question:\n{user_query}\n\n"
            f"Follow these instructions when you answer:\n{system_prompt}"
        )
        
        final = await self.composer.kickoff_async(compose_prompt)
        answer = final.raw
        
        for i in range(0, len(answer), 400):
            yield answer[i : i + 400]
