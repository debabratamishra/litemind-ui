"""RAG service and CrewAI orchestration utilities.
Provides ingestion, indexing, hybrid retrieval, and answer composition using ChromaDB,
BM25, and an Ollama-backed LLM.
"""
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
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
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import gc
import json
import importlib.util

# Import enhanced extractors
from app.ingestion.enhanced_extractors import extract_csv_enhanced, process_images_enhanced, get_image_processor, get_csv_processor
from app.ingestion.file_ingest import ingest_file
from app.ingestion.enhanced_document_processor import get_document_processor, extract_pdf_enhanced, extract_docx_enhanced, extract_epub_enhanced

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

def _flatten_metadata(meta, prefix=""):
    """Flatten a (possibly nested) metadata mapping into a single-level dict.
    Nested keys are joined with underscores. Iterables are JSON-encoded when possible.
    """
    out = {}
    if not isinstance(meta, dict):
        return out
    for k, v in meta.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, dict):
            out.update(_flatten_metadata(v, key))
        elif isinstance(v, (list, tuple, set)):
            try:
                out[key] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[key] = str(v)
        else:
            out[key] = str(v)
    return out


class RAGService:
    """High-level service for ingesting documents, building indexes, and answering queries.
    Manages ChromaDB persistence, BM25 keyword indexing, and hybrid retrieval for text and
    lightweight image references.
    """
    def __init__(self):
        """Initialize storage, embedding models, collections, and in-memory indexes."""
        if os.path.exists(Config.CHROMA_DB_PATH):
            shutil.rmtree(Config.CHROMA_DB_PATH)

        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False))

        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "900"))

        self.text_collection = self.client.create_collection(
            name="documents_text",
            embedding_function=self.embedding_function
        )

        self.file_paths = []
        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []  # original text chunks
        self.chunk_ids = []

        self.stop_words = set(stopwords.words('english'))

        self.image_cache_dir = Path("./uploads/imgcache")
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_capabilities(self) -> dict:
        """Report processing capabilities and supported extensions detected at runtime."""
        ocr_available = any(
            importlib.util.find_spec(name) is not None
            for name in ("pytesseract", "easyocr")
        )

        memory_optimized = True
        enhanced_csv = True

        return {
            "status": "ready",
            "message": "Enhanced processing available",
            "capabilities": {
                "enhanced_csv": enhanced_csv,
                "ocr_available": ocr_available,
                "memory_optimized": memory_optimized
            },
            "supported_extensions": [
                "pdf","doc","docx","ppt","pptx","rtf","odt","epub",
                "xls","xlsx","csv","tsv",
                "txt","md","html","htm","org","rst",
                "png","jpg","jpeg","bmp","tiff","webp","gif","heic","svg"
            ]
        }


    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize, lowercase, and remove stop words/punctuation for BM25 indexing."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]
        return tokens

    def chunk_text(self, text, chunk_size=500):
        """Split text into fixed-size chunks with a small overlap to preserve context."""
        overlap = min(50, chunk_size // 10)
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _create_simple_image_references(self, images: List[dict], doc_id: str) -> List[str]:
        """Create lightweight, human-readable references for extracted images to index with text."""
        if not (ENABLE_SIMPLE_IMAGE_INDEXING and images):
            return []
        
        references = []
        for i, rec in enumerate(images[:MAX_IMAGES_PER_DOC]):
            try:
                meta = rec.get('metadata', {})
                filename = meta.get('filename', 'unknown')
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

    def _index_chunk_batch(self, chunk_batch: List[dict], doc_id: str):
        """Index a batch of text chunks into ChromaDB and update the BM25 corpus."""
        try:
            if not chunk_batch:
                return
                
            texts = [c["content"] for c in chunk_batch if c.get("content", "").strip()]
            raw_metas = [c.get("metadata", {}) for c in chunk_batch if c.get("content", "").strip()]
            metadatas = [_flatten_metadata(m) for m in raw_metas]
            
            if not texts:
                logger.warning("No valid texts in chunk batch")
                return
            
            base_id = f"{doc_id}_batch_{len(self.chunk_ids)}"
            ids = [f"{base_id}_{i}" for i in range(len(texts))]
            
            self.text_collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            
            for i, text in enumerate(texts):
                tokens = self.preprocess_text(text)
                self.bm25_corpus.append(tokens)
                self.document_chunks.append(text)
                self.chunk_ids.append(ids[i])
            
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)
            
            logger.info(f"Indexed batch of {len(texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error indexing batch: {e}")
            raise

    async def _index_single_chunk(self, chunk_data: dict):
        """Index a single text chunk into ChromaDB and update the BM25 corpus."""
        try:
            if not chunk_data.get("content", "").strip():
                return
                
            content = chunk_data["content"]
            metadata = _flatten_metadata(chunk_data.get("metadata", {}))
            
            chunk_id = f"single_{len(self.chunk_ids)}_{hash(content) % 1000000}"
            
            self.text_collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[chunk_id],
            )
            
            tokens = self.preprocess_text(content)
            self.bm25_corpus.append(tokens)
            self.document_chunks.append(content)
            self.chunk_ids.append(chunk_id)
            
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)
                
        except Exception as e:
            logger.error(f"Error indexing single chunk: {e}")
            raise

    def _ingest_file_with_enhancement(self, path: Path):
        """Extract text, tables, and image metadata using enhanced processors when available."""
        ext = path.suffix.lower()
        
        if ext == '.pdf':
            logger.info(f"Using enhanced PDF processing for {path.name}")
            text_chunks = extract_pdf_enhanced(path)
            image_records = []
            table_texts = []
        elif ext in ['.docx', '.doc']:
            logger.info(f"Using enhanced DOCX processing for {path.name}")
            text_chunks = extract_docx_enhanced(path)
            image_records = []
            table_texts = []
        elif ext == '.epub':
            logger.info(f"Using enhanced EPUB processing for {path.name}")
            text_chunks = extract_epub_enhanced(path)
            image_records = []
            table_texts = []
        else:
            text_chunks, image_records, table_texts = ingest_file(path)
            if image_records:
                logger.info(f"Processing {len(image_records)} images with enhanced extraction")
                enhanced_image_content = process_images_enhanced(image_records)
                text_chunks.extend(enhanced_image_content)
                image_records.clear()
                gc.collect()

        return text_chunks, image_records, table_texts



    async def add_document(self, file_path, doc_id, chunk_size=None):
        """Ingest a file, extract content using enhanced loaders, and index text/table data.
        Optionally processes CSV/TSV via optimized path. Uses batched writes for ChromaDB.
        """
        if chunk_size is None:
            chunk_size = self.default_chunk_size

        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return

            logger.info(f"Processing document with enhanced extraction: {doc_id}")
            
            ext = path.suffix.lower()
            
            if ext in {'.csv', '.tsv'}:
                logger.info(f"Using enhanced CSV processing for {path.name}")
                chunks = await asyncio.to_thread(extract_csv_enhanced, path)
                text_chunks, image_records, table_texts = chunks, [], []
                gc.collect()
            else:
                text_chunks, image_records, table_texts = await asyncio.to_thread(
                    self._ingest_file_with_enhancement, path
                )
            
            all_chunks = []
            
            for chunk in text_chunks:
                if isinstance(chunk, dict):
                    chunk_data = {
                        "content": chunk["content"],
                        "metadata": {
                            **chunk.get("metadata", {}),
                            "doc_id": doc_id,
                            "file_path": str(file_path)
                        }
                    }
                else:
                    chunk_data = {
                        "content": str(chunk),
                        "metadata": {
                            "doc_id": doc_id,
                            "file_path": str(file_path),
                            "content_type": "legacy_text"
                        }
                    }
                all_chunks.append(chunk_data)
            
            for table in table_texts:
                if isinstance(table, dict):
                    table_content = table.get("content", str(table))
                else:
                    table_content = str(table)
                all_chunks.append({
                    "content": table_content,
                    "metadata": {
                        "doc_id": doc_id,
                        "file_path": str(file_path),
                        "content_type": "table"
                    }
                })
            
            logger.info(f"Total chunks to index: {len(all_chunks)}")
            
            batch_size = int(os.getenv("INDEX_BATCH_SIZE", "128"))
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                self._index_chunk_batch(batch, doc_id)
                if i % 200 == 0:
                    gc.collect()
            
            if str(file_path) not in self.file_paths:
                self.file_paths.append(str(file_path))
            
            logger.info(f"Successfully indexed {len(all_chunks)} chunks for {doc_id}")
            
        except Exception as e:
            logger.error(f"Error indexing {doc_id}: {str(e)}")
            raise

    async def recreate_collection(self):
        """Rebuild the text collection and BM25 index, then re-index all known files."""
        try:
            self.client.delete_collection(name="documents_text")
        except Exception:
            pass

        self.text_collection = self.client.create_collection(
            name="documents_text",
            embedding_function=self.embedding_function
        )

        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []
        self.chunk_ids = []

        for file_path in self.file_paths:
            doc_id = os.path.basename(file_path)
            await self.add_document(file_path, doc_id, self.default_chunk_size)

    def bm25_search(self, query: str, n_results: int = 3) -> List[Tuple[str, str, float]]:
        """Run a BM25 keyword search and return (chunk_id, text, score) tuples for top hits."""
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
        """Run semantic search over text chunks and return (chunk_id, text, similarity)."""
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
        """Fuse BM25 and vector results using Reciprocal Rank Fusion and return sorted IDs."""
        bm25_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, *_rest) in enumerate(bm25_results)}
        vector_scores = {doc_id: 1 / (k + rank + 1) for rank, (doc_id, *_rest) in enumerate(vector_results)}
        
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            combined_scores[doc_id] = bm25_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_results]

    def hybrid_search(self, query: str, n_results: int = 3) -> List[str]:
        """Combine BM25 and semantic search via RRF and return the top document texts."""
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
        """Answer a query using semantic or hybrid retrieval and stream model tokens via Ollama."""
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

class CrewAIRAGOrchestrator:
    """Coordinates refined querying and answer composition using an Ollama-backed model."""
    def __init__(self, rag_service: RAGService, model_name="gemma3n:e2b"):
        """Configure the Ollama LLM, agent roles, and default context parameters."""
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
        """Query Ollama for the model's context length; fall back to a sensible default."""
        model = model_name.replace("ollama/", "") if "ollama/" in model_name else model_name
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
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
        """Generate a short summary for a text segment using the configured model."""
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
        """Chunk a string into fixed-size pieces without overlap."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def summarize_context(self, text: str, target_length: int) -> str:
        """Recursively summarize text until it fits within a target character budget."""
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
        """Refine the user query, retrieve/summarize context, and compose a final streamed answer."""
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