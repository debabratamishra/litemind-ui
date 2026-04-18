"""RAG service and CrewAI orchestration utilities.
Provides ingestion, indexing, hybrid retrieval, and answer composition using ChromaDB,
BM25, and an Ollama-backed LLM.
"""
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from .ollama import stream_ollama
import os
import re
import hashlib
from config import Config
from crewai import Agent, LLM
import httpx
import shutil
import logging
import asyncio
import base64
from rank_bm25 import BM25Okapi
import string
from typing import Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
from PIL import Image
import gc
import json
import importlib
from typing import Optional

from app.core.rag_formats import SUPPORTED_EXTENSIONS
from app.ingestion.enhanced_extractors import process_images_enhanced
from app.ingestion.file_ingest import get_ingestion_capabilities, ingest_file

# Configuration flags
ENABLE_SIMPLE_IMAGE_INDEXING = os.getenv("ENABLE_SIMPLE_IMAGE_INDEXING", "true").lower() == "true"
MAX_IMAGES_PER_DOC = int(os.getenv("MAX_IMAGES_PER_DOC", "10"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "so",
    "that", "the", "their", "theirs", "them", "they", "this", "those", "to", "too",
    "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "yours",
}

_stopwords_fallback_logged = False
_tokenizer_fallback_logged = False


def _module_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _load_stop_words() -> set[str]:
    global _stopwords_fallback_logged

    if not _stopwords_fallback_logged:
        logger.info("Using built-in stopword list for BM25 preprocessing")
        _stopwords_fallback_logged = True
    return set(DEFAULT_STOP_WORDS)


def _tokenize_text(text: str) -> List[str]:
    global _tokenizer_fallback_logged

    if not _tokenizer_fallback_logged:
        logger.info("Using regex tokenizer for BM25 preprocessing")
        _tokenizer_fallback_logged = True
    return re.findall(r"\b\w+\b", text)

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
    """Retrieval-augmented generation service with enhanced document processing capabilities."""
    
    # Class-level ChromaDB client management to avoid conflicts
    _shared_client = None
    _shared_client_path = None
    
    @classmethod
    def _get_or_create_client(cls, chroma_db_path: str):
        """Get or create a shared ChromaDB client to avoid conflicts."""
        if cls._shared_client is None or cls._shared_client_path != chroma_db_path:
            if cls._shared_client is not None:
                # Clean up previous client if path changed
                try:
                    cls._shared_client.reset()
                except:
                    pass
            
            cls._shared_client = chromadb.PersistentClient(
                path=chroma_db_path, 
                settings=Settings(anonymized_telemetry=False)
            )
            cls._shared_client_path = chroma_db_path
            
        return cls._shared_client
    
    def __init__(self):
        # Get the appropriate ChromaDB path based on environment
        self.chroma_db_path = self._get_chroma_db_path()
        
        # In containerized environments, we can't remove mounted volumes
        # Instead, clean the contents if the directory exists
        if os.path.exists(self.chroma_db_path):
            try:
                shutil.rmtree(self.chroma_db_path)
            except OSError as e:
                # If we can't remove the directory (e.g., mounted volume),
                # try to clean its contents instead
                logger.warning(f"Could not remove ChromaDB directory {self.chroma_db_path}: {e}")
                try:
                    for item in os.listdir(self.chroma_db_path):
                        item_path = os.path.join(self.chroma_db_path, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    logger.info(f"Cleaned contents of ChromaDB directory: {self.chroma_db_path}")
                except Exception as clean_error:
                    logger.warning(f"Could not clean ChromaDB directory contents: {clean_error}")

        self.client = self._get_or_create_client(self.chroma_db_path)

        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"SentenceTransformerEmbeddingFunction not available ({e}), using ChromaDB default")
            try:
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            except Exception:
                self.embedding_function = None

        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "900"))

        self.text_collection = self.client.get_or_create_collection(
            name="documents_text",
            embedding_function=self.embedding_function
        )

        self.file_paths = []
        self.bm25_corpus = []
        self.bm25_model = None
        self.document_chunks = []  # original text chunks
        self.chunk_ids = []
        self.chunk_metadata = []
        self.chunk_metadata_by_id = {}
        
        # Track processed files to prevent duplicates
        self.processed_files = {}  # filename -> {hash, chunk_count, timestamp}
        self.file_hashes = {}  # hash -> filename (for duplicate detection)

        self.stop_words = _load_stop_words()

        # Initialize image cache directory using environment-aware path
        upload_dir = self._get_upload_directory()
        self.image_cache_dir = Path(upload_dir) / "imgcache"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image cache directory initialized at: {self.image_cache_dir}")
        
        # Rebuild indexes from existing collection data
        self._rebuild_indexes_from_collection()
    
    def _get_chroma_db_path(self) -> str:
        """Get the appropriate ChromaDB path based on execution environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            path = str(host_service_manager.environment_config.chroma_db_dir)
            logger.debug(f"Using ChromaDB path from host service manager: {path}")
            return path
        except ImportError:
            logger.warning("Host service manager not available, using fallback config")
            # Fallback to config-based detection
            from config import Config
            path = Config.get_chroma_db_path()
            logger.debug(f"Using fallback ChromaDB path: {path}")
            return path
    
    def _get_upload_directory(self) -> str:
        """Get the appropriate upload directory based on execution environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            path = str(host_service_manager.environment_config.upload_dir)
            logger.debug(f"Using upload directory from host service manager: {path}")
            return path
        except ImportError:
            logger.warning("Host service manager not available, using fallback config")
            # Fallback to config-based detection
            from config import Config
            path = Config.get_upload_folder()
            logger.debug(f"Using fallback upload directory: {path}")
            return path
    
    def _get_ollama_url(self) -> str:
        """Get the appropriate Ollama URL based on execution environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            url = host_service_manager.environment_config.ollama_url
            logger.debug(f"Using Ollama URL from host service manager: {url}")
            return url
        except ImportError:
            logger.warning("Host service manager not available, using fallback config")
            url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            logger.debug(f"Using fallback Ollama URL: {url}")
            return url
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file for duplicate detection."""
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def _is_file_already_processed(self, file_path: str, filename: str) -> tuple[bool, str]:
        """Check if file is already processed. Returns (is_duplicate, reason)."""
        # Check by filename first
        if filename in self.processed_files:
            file_info = self.processed_files[filename]
            return True, f"File '{filename}' already processed ({file_info['chunk_count']} chunks)"
        
        # Check by content hash to detect renamed duplicates
        file_hash = self._calculate_file_hash(file_path)
        if file_hash and file_hash in self.file_hashes:
            original_name = self.file_hashes[file_hash]
            return True, f"File content already processed as '{original_name}' (duplicate content detected)"
        
        return False, ""
    
    def _register_processed_file(self, file_path: str, filename: str, chunk_count: int):
        """Register a file as processed to prevent future duplicates."""
        import time
        
        file_hash = self._calculate_file_hash(file_path)
        
        file_info = {
            "hash": file_hash,
            "chunk_count": chunk_count,
            "timestamp": time.time(),
            "file_path": file_path
        }
        
        self.processed_files[filename] = file_info
        if file_hash:
            self.file_hashes[file_hash] = filename
        
        logger.info(f"Registered file: {filename} ({chunk_count} chunks)")
    
    def get_processed_files_info(self) -> dict:
        """Get information about all processed files."""
        import time
        
        files_info = []
        for filename, info in self.processed_files.items():
            files_info.append({
                "filename": filename,
                "chunk_count": info["chunk_count"],
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["timestamp"])),
                "file_path": info.get("file_path", "unknown")
            })
        
        return {
            "total_files": len(self.processed_files),
            "total_chunks": sum(info["chunk_count"] for info in self.processed_files.values()),
            "files": files_info
        }
    
    def remove_processed_file(self, filename: str) -> bool:
        """Remove a file from processed files tracking and from collections."""
        if filename not in self.processed_files:
            return False
        
        try:
            # Remove from ChromaDB collection
            # Get all chunks for this file
            results = self.text_collection.get(
                where={"filename": filename}
            )
            
            if results and results['ids']:
                self.text_collection.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} chunks for file: {filename}")
            
            # Remove from tracking
            file_info = self.processed_files[filename]
            if file_info.get("hash") and file_info["hash"] in self.file_hashes:
                del self.file_hashes[file_info["hash"]]
            
            del self.processed_files[filename]
            
            # Rebuild BM25 index
            self._rebuild_bm25_index()
            
            logger.info(f"Successfully removed file: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing file {filename}: {e}")
            return False
    
    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index from current collection."""
        try:
            # Get all documents from collection
            all_docs = self.text_collection.get()
            
            if all_docs and all_docs['documents']:
                self.document_chunks = all_docs['documents']
                self.chunk_ids = all_docs['ids']
                raw_metadatas = all_docs.get('metadatas') or []
                if len(raw_metadatas) < len(self.document_chunks):
                    raw_metadatas = list(raw_metadatas) + [{}] * (len(self.document_chunks) - len(raw_metadatas))
                self.chunk_metadata = [(metadata or {}) for metadata in raw_metadatas[:len(self.document_chunks)]]
                self.chunk_metadata_by_id = {
                    chunk_id: metadata
                    for chunk_id, metadata in zip(self.chunk_ids, self.chunk_metadata)
                }
                
                # Rebuild BM25 corpus
                self.bm25_corpus = [self.preprocess_text(doc) for doc in self.document_chunks]
                
                if self.bm25_corpus:
                    from rank_bm25 import BM25Okapi
                    self.bm25_model = BM25Okapi(self.bm25_corpus)
                    logger.info(f"Rebuilt BM25 index with {len(self.bm25_corpus)} documents")
                else:
                    self.bm25_model = None
            else:
                self.document_chunks = []
                self.chunk_ids = []
                self.chunk_metadata = []
                self.chunk_metadata_by_id = {}
                self.bm25_corpus = []
                self.bm25_model = None
                
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index: {e}")

    def _rebuild_indexes_from_collection(self):
        """Rebuild all indexes and tracking from existing collection data."""
        try:
            # First rebuild the BM25 index which also populates document_chunks and chunk_ids
            self._rebuild_bm25_index()
            
            # Rebuild processed files tracking from metadata
            all_docs = self.text_collection.get()
            if all_docs and all_docs['metadatas']:
                processed_files = {}
                file_hashes = {}
                file_paths_set = set()
                
                for metadata in all_docs['metadatas']:
                    doc_id = metadata.get('doc_id')
                    file_path = metadata.get('file_path')
                    
                    if doc_id and file_path:
                        file_paths_set.add(file_path)
                        
                        if doc_id not in processed_files:
                            # Calculate basic info for this file
                            chunk_count = sum(1 for m in all_docs['metadatas'] if m.get('doc_id') == doc_id)
                            processed_files[doc_id] = {
                                'hash': None,  # We don't store hash in metadata, will calculate if needed
                                'chunk_count': chunk_count,
                                'timestamp': 0,  # Unknown timestamp for existing files
                                'file_path': file_path
                            }
                
                self.processed_files = processed_files
                self.file_hashes = file_hashes  # We can't rebuild this without recalculating hashes
                self.file_paths = list(file_paths_set)
                
                if processed_files:
                    logger.info(f"Rebuilt tracking for {len(processed_files)} processed files")
                    
        except Exception as e:
            logger.error(f"Error rebuilding indexes from collection: {e}")
            # Don't fail completely, just use empty state
            pass

    def get_capabilities(self) -> dict:
        """Report processing capabilities and supported extensions detected at runtime."""
        ocr_available = _module_importable("pytesseract") or _module_importable("easyocr")

        ingestion_capabilities = get_ingestion_capabilities()

        return {
            "status": "ready",
            "message": "Enhanced local extraction pipeline available",
            "capabilities": {
                "ocr_available": ocr_available,
                "memory_optimized": True,
                "local_pipeline": ingestion_capabilities.get("local_pipeline", {}),
            },
            "supported_extensions": SUPPORTED_EXTENSIONS,
        }


    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize, lowercase, and remove stop words/punctuation for BM25 indexing."""
        if not text or not text.strip():
            return []
        
        # Clean text before tokenization
        text = self._clean_text_for_indexing(text)
        
        tokens = _tokenize_text(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation and len(t) > 1]
        return tokens
    
    def _clean_text_for_indexing(self, text: str) -> str:
        """Clean text for better indexing and search."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive formatting artifacts
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        
        # Clean up common document artifacts
        text = re.sub(r'PAGE\s+\d+\s+(?:HEADERS?|LISTS?|ANALYSIS):', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?:Document|Filename|Page):\s*[^\n]+', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def chunk_text(self, text, chunk_size=500):
        """Split text into paragraph-aware chunks with overlap for indexing."""
        if not text or not text.strip():
            return []

        text = self._clean_text_for_chunking(text)

        overlap = max(60, min(200, chunk_size // 6))
        chunks = []

        paragraphs = [paragraph.strip() for paragraph in re.split(r'\n\s*\n+', text) if paragraph.strip()]
        if not paragraphs:
            return self._fallback_character_chunking(text, chunk_size, overlap)

        current_parts = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_parts = self._split_paragraph_for_chunking(paragraph, chunk_size)
            for part in paragraph_parts:
                projected_length = current_length + len(part) + (2 if current_parts else 0)
                if current_parts and projected_length > chunk_size:
                    chunks.append("\n\n".join(current_parts).strip())
                    overlap_parts = self._select_overlap_paragraphs(current_parts, overlap)
                    current_parts = overlap_parts[:]
                    current_length = len("\n\n".join(current_parts))
                current_parts.append(part)
                current_length = len("\n\n".join(current_parts))

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())

        if not chunks:
            return self._fallback_character_chunking(text, chunk_size, overlap)

        return chunks
    
    def _clean_text_for_chunking(self, text: str) -> str:
        """Clean text before chunking to improve formatting."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?m)^ +', '', text)
        return text.strip()

    def _split_paragraph_for_chunking(self, paragraph: str, chunk_size: int) -> List[str]:
        """Split oversized paragraphs while preserving sentence boundaries where possible."""
        paragraph = paragraph.strip()
        if not paragraph:
            return []

        if len(paragraph) <= chunk_size:
            return [paragraph]

        sentences = [segment.strip() for segment in re.split(r'(?<=[.!?])\s+', paragraph) if segment.strip()]
        if not sentences:
            return self._fallback_character_chunking(paragraph, chunk_size, max(40, chunk_size // 10))

        parts = []
        current = ""
        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    parts.append(current.strip())
                    current = ""
                parts.extend(self._fallback_character_chunking(sentence, chunk_size, max(40, chunk_size // 10)))
                continue

            if current and len(current) + len(sentence) + 1 > chunk_size:
                parts.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence

        if current:
            parts.append(current.strip())

        return parts or [paragraph]

    def _select_overlap_paragraphs(self, paragraphs: List[str], overlap_chars: int) -> List[str]:
        """Keep a short tail overlap to improve recall across chunk boundaries."""
        selected = []
        total = 0
        for paragraph in reversed(paragraphs):
            candidate = len(paragraph) + (2 if selected else 0)
            if selected and total + candidate > overlap_chars:
                break
            selected.insert(0, paragraph)
            total += candidate
            if total >= overlap_chars:
                break
        return selected

    def _derive_section_title(self, metadata: dict, content: str) -> str:
        """Infer a stable section title to carry across derived chunks."""
        content_type = str(metadata.get("content_type", "")).lower()
        skip_generated_titles_for = {
            "ocr_text",
            "pdf_page_ocr",
            "pdf_text_header",
            "pdf_text_footer",
            "pdf_image_enhanced",
            "docx_image",
            "image_analysis",
            "image_reference",
            "structured_content",
        }

        for key in ("section_title", "slide_title", "title", "heading", "sheet"):
            value = metadata.get(key)
            if isinstance(value, str) and self._is_meaningful_section_title(value, metadata):
                return value.strip()

        if content_type in skip_generated_titles_for:
            return ""

        first_line = content.splitlines()[0].strip() if content else ""
        if (
            first_line
            and len(first_line) <= 120
            and not first_line.endswith('.')
            and self._is_meaningful_section_title(first_line, metadata)
        ):
            return first_line
        return ""

    def _apply_section_prefix(self, content: str, section_title: str) -> str:
        """Prefix chunks with their section title when it adds grounding."""
        if not section_title:
            return content.strip()
        stripped_content = content.strip()
        if stripped_content.lower().startswith(section_title.lower()):
            return stripped_content
        return f"{section_title}\n\n{stripped_content}"

    def _dedupe_chunk_key(self, content: str, metadata: dict) -> str:
        """Generate a stable dedupe key for extracted content blocks."""
        normalized_content = re.sub(r'\s+', ' ', content).strip().lower()
        identity_parts = [
            str(metadata.get('content_type', '')),
            str(metadata.get('page_number', '')),
            str(metadata.get('slide_number', '')),
            str(metadata.get('sheet', '')),
            normalized_content,
        ]
        return hashlib.sha1('|'.join(identity_parts).encode('utf-8')).hexdigest()

    def _prepare_extracted_chunks(self, extracted_chunks: List, doc_id: str, file_path: str, chunk_size: int) -> List[dict]:
        """Normalize, split, and deduplicate extracted blocks before indexing."""
        prepared_chunks = []
        seen_keys = set()

        for source_index, chunk in enumerate(extracted_chunks):
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                metadata = {**chunk.get('metadata', {})}
            else:
                content = str(chunk)
                metadata = {'content_type': 'legacy_text'}

            clean_content = self._clean_text_for_chunking(content)
            if not clean_content:
                continue

            section_title = self._derive_section_title(metadata, clean_content)
            subchunks = self.chunk_text(clean_content, chunk_size=chunk_size)
            if not subchunks:
                subchunks = [clean_content]

            for chunk_index, subchunk in enumerate(subchunks):
                final_content = self._apply_section_prefix(subchunk, section_title)
                dedupe_key = self._dedupe_chunk_key(final_content, metadata)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)

                prepared_chunks.append({
                    'content': final_content,
                    'metadata': {
                        **metadata,
                        'doc_id': doc_id,
                        'file_path': str(file_path),
                        'source_index': source_index,
                        'chunk_index': chunk_index,
                        'section_title': section_title,
                    },
                })

        return prepared_chunks
    
    def _fallback_character_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback character-based chunking when sentence-based fails."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
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

    def _collect_embedded_image_records(self, extracted_chunks: List[dict]) -> List[dict]:
        """Recover embedded document images so they can pass through the OCR pipeline."""
        embedded_records = []

        for chunk in extracted_chunks or []:
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get("metadata", {}) or {}
            content_type = metadata.get("content_type", "")
            image_b64 = metadata.get("image_bytes")

            if content_type not in {"pdf_image_enhanced", "docx_image"} or not image_b64:
                continue

            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as exc:
                logger.warning(f"Failed to decode embedded image bytes for {metadata.get('filename', 'unknown')}: {exc}")
                continue

            clean_metadata = {k: v for k, v in metadata.items() if k != "image_bytes"}
            clean_metadata["embedded_image"] = True
            clean_metadata["source_content_type"] = content_type

            embedded_records.append({
                "image_bytes": image_bytes,
                "metadata": clean_metadata,
            })

        return embedded_records

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
                metadata = metadatas[i] if i < len(metadatas) else {}
                self.chunk_metadata.append(metadata)
                self.chunk_metadata_by_id[ids[i]] = metadata
            
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
            self.chunk_metadata.append(metadata)
            self.chunk_metadata_by_id[chunk_id] = metadata
            
            if self.bm25_corpus:
                self.bm25_model = BM25Okapi(self.bm25_corpus)
                
        except Exception as e:
            logger.error(f"Error indexing single chunk: {e}")
            raise

    def _ingest_file_with_enhancement(self, path: Path):
        """Extract text, tables, and image metadata using enhanced processors when available."""
        text_chunks, image_records, table_texts = ingest_file(path)
        text_chunks = list(text_chunks or [])
        image_records = list(image_records or [])
        table_texts = list(table_texts or [])

        embedded_image_records = self._collect_embedded_image_records(text_chunks)
        if embedded_image_records:
            logger.info(f"Recovered {len(embedded_image_records)} embedded document images for OCR")
            image_records.extend(embedded_image_records)

        text_chunks = self._filter_enhanced_content(text_chunks)
        table_texts = self._filter_enhanced_content(table_texts)

        if image_records:
            logger.info(f"Processing {len(image_records)} images with enhanced extraction")
            enhanced_image_content = process_images_enhanced(image_records)
            enhanced_image_content = self._filter_enhanced_content(enhanced_image_content)
            text_chunks.extend(enhanced_image_content)
            image_records.clear()
            gc.collect()

        return text_chunks, image_records, table_texts

    def _filter_enhanced_content(self, text_chunks: List[dict]) -> List[dict]:
        """Filter out or deprioritize metadata-heavy content to improve RAG relevance."""
        if not text_chunks:
            return text_chunks
            
        # Define metadata-heavy content types that should be excluded or deprioritized
        metadata_content_types = {
            # PDF metadata types
            'pdf_document_overview',
            'page_layout_analysis', 
            'document_structure_analysis',
            'pdf_document_analysis',
            'document_metadata',
            'document_overview',
            'pdf_image_enhanced',
            
            # Image metadata types
            'image_analysis',
            'image_reference',
            'partial_structure',
            
            # CSV/Excel metadata types
            'dataset_overview',
            'statistical_summary',
            'column_analysis',
            'large_dataset_overview',
            
            # DOCX metadata types
            'document_analysis',
            'docx_metadata',
            'docx_image',
            
            # EPUB metadata types  
            'epub_metadata',
            'book_structure',
            
            # Generic metadata types
            'metadata_summary',
            'file_analysis'
        }
        
        # Define content that should be prioritized (actual document content)
        priority_content_types = {
            # PDF content types
            'pdf_text_body',
            'pdf_text_header', 
            'pdf_text_footer',
            'pdf_headers',
            'pdf_table_camelot',
            'pdf_table_pdfplumber',
            'pdf_content_block',
            'pdf_page_ocr',
            
            # Image content types
            'ocr_text',
            'structured_content',
            
            # CSV/Excel content types
            'data_chunk',
            'table_content',
            
            # DOCX content types
            'docx_section',
            'docx_paragraph',
            'docx_table',
            'docx_content',
            
            # EPUB content types
            'epub_chapter',
            'epub_content',
            'epub_text',
            
            # Generic content types
            'text_content',
            'main_content'
        }
        
        filtered_chunks = []
        
        for chunk in text_chunks:
            if not isinstance(chunk, dict):
                # Handle legacy string chunks
                filtered_chunks.append(chunk)
                continue
                
            content_type = chunk.get('metadata', {}).get('content_type', '')
            content = chunk.get('content', '')
            
            # Skip metadata-heavy content types entirely
            if content_type.lower() in metadata_content_types:
                logger.debug(f"Filtering out metadata content type: {content_type}")
                continue
                
            # Also filter based on content patterns (in case content_type is missing)
            metadata_patterns = [
                'document metadata',
                'document statistics', 
                'document overview',
                'statistical summary',
                'column analysis',
                'dataset overview',
                'image analysis',
                'layout analysis',
                'document analysis',
                'total pages:',
                'file size:',
                'creation date:',
                'document characteristics:',
                'layout characteristics:',
                'page size:',
                'text blocks:',
                'word count:',
                'reading time:',
                'document type is categorized',
                'dimensions:',
                'aspect ratio:',
                'average brightness:',
                'color variance:',
                'unique values:',
                'missing values:',
                'data types breakdown:',
                'rows processed:',
                'columns (',
                'processing mode:'
            ]
            
            # Check if content is metadata-heavy
            content_lower = content.lower()
            is_metadata_heavy = any(pattern in content_lower for pattern in metadata_patterns)
            
            # Skip if it's primarily metadata and short
            if is_metadata_heavy and len(content) < 1000:  # Allow longer content even if it mentions metadata
                logger.debug(f"Filtering out metadata-heavy content: {content[:100]}...")
                continue
                
            # Additional filtering for very short non-informative chunks
            if len(content.strip()) < 50 and content_type not in priority_content_types:
                logger.debug(f"Filtering out too-short content: {content[:50]}...")
                continue
                
            # Keep the chunk
            filtered_chunks.append(chunk)
            
        logger.info(f"Filtered {len(text_chunks) - len(filtered_chunks)} metadata chunks, keeping {len(filtered_chunks)} content chunks")
        return filtered_chunks



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
                return {"status": "error", "message": f"File not found: {file_path}"}

            # Check for duplicates before processing
            is_duplicate, reason = self._is_file_already_processed(str(path), doc_id)
            if is_duplicate:
                logger.warning(f"Duplicate file detected: {reason}")
                return {"status": "duplicate", "message": reason, "filename": doc_id}

            logger.info(f"Processing document with enhanced extraction: {doc_id}")
            
            text_chunks, image_records, table_texts = await asyncio.to_thread(
                self._ingest_file_with_enhancement, path
            )
            gc.collect()

            all_chunks = self._prepare_extracted_chunks(text_chunks, doc_id, str(file_path), chunk_size)
            all_chunks.extend(self._prepare_extracted_chunks(table_texts, doc_id, str(file_path), chunk_size))

            if not all_chunks:
                logger.warning(f"No extractable content found in {doc_id}")
                return {
                    "status": "error",
                    "message": f"No extractable content found in {doc_id}",
                    "chunks_created": 0,
                }
            
            logger.info(f"Total chunks to index: {len(all_chunks)}")
            
            batch_size = int(os.getenv("INDEX_BATCH_SIZE", "128"))
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                self._index_chunk_batch(batch, doc_id)
                if i % 200 == 0:
                    gc.collect()
            
            if str(file_path) not in self.file_paths:
                self.file_paths.append(str(file_path))
            
            # Register the file as processed to prevent future duplicates
            self._register_processed_file(str(file_path), doc_id, len(all_chunks))
            
            logger.info(f"Successfully indexed {len(all_chunks)} chunks for {doc_id}")
            return {"status": "success", "message": f"Successfully processed {doc_id}", "chunks_created": len(all_chunks)}
            
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
        self.chunk_metadata = []
        self.chunk_metadata_by_id = {}

        for file_path in self.file_paths:
            doc_id = os.path.basename(file_path)
            await self.add_document(file_path, doc_id, self.default_chunk_size)

    async def reset_system(self):
        """Completely reset the RAG system by clearing all collections, indexes, and file references."""
        try:
            # Delete the collection
            try:
                self.client.delete_collection(name="documents_text")
                logger.info("Deleted existing text collection")
            except Exception as e:
                logger.debug(f"Collection deletion failed (may not exist): {e}")

            # Recreate empty collection
            self.text_collection = self.client.create_collection(
                name="documents_text",
                embedding_function=self.embedding_function
            )

            # Clear all in-memory indexes and references
            self.file_paths.clear()
            self.bm25_corpus.clear()
            self.bm25_model = None
            self.document_chunks.clear()
            
            # Clear chunk_ids if it exists
            if hasattr(self, 'chunk_ids'):
                self.chunk_ids.clear()
            if hasattr(self, 'chunk_metadata'):
                self.chunk_metadata.clear()
            if hasattr(self, 'chunk_metadata_by_id'):
                self.chunk_metadata_by_id.clear()

            # Clear file tracking to allow re-uploads after reset
            self.processed_files.clear()
            self.file_hashes.clear()

            # Force garbage collection to free memory
            gc.collect()
            
            logger.info("RAG system reset completed successfully")

        except Exception as e:
            logger.error(f"Error during RAG system reset: {e}")
            raise

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

    def _build_search_record(
        self,
        chunk_id: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
        retrieval_method: str = "semantic",
    ) -> Dict[str, Any]:
        """Normalize search results into a source-aware record."""
        return {
            "id": chunk_id,
            "content": content,
            "score": float(score),
            "metadata": metadata or {},
            "retrieval_method": retrieval_method,
        }

    def bm25_search_records(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Run BM25 keyword search and preserve metadata for citation formatting."""
        raw_results = self.bm25_search(query, n_results)
        return [
            self._build_search_record(
                chunk_id,
                content,
                score,
                self.chunk_metadata_by_id.get(chunk_id, {}),
                retrieval_method="bm25",
            )
            for chunk_id, content, score in raw_results
        ]

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

    def vector_search_records(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Run semantic search over text chunks and preserve metadata for citation formatting."""
        try:
            results = self.text_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            ids = results.get("ids") or [[]]
            documents = results.get("documents") or [[]]
            metadatas = results.get("metadatas") or [[]]
            distances = results.get("distances") or [[]]

            if not ids[0] or not documents[0]:
                return []

            out = []
            for index, doc_id in enumerate(ids[0]):
                document = documents[0][index] if index < len(documents[0]) else ""
                metadata = metadatas[0][index] if metadatas and index < len(metadatas[0]) else {}
                distance = distances[0][index] if distances and index < len(distances[0]) else None
                similarity = 1 - distance if distance is not None else 0.0
                out.append(
                    self._build_search_record(
                        doc_id,
                        document,
                        similarity,
                        metadata,
                        retrieval_method="semantic",
                    )
                )

            return out
        except Exception as e:
            logger.error(f"Vector text search error: {str(e)}")
            return []

    def reciprocal_rank_fusion(self, bm25_results: List[Tuple], vector_results: List[Tuple], k: int = 60) -> List[str]:
        """Fuse BM25 and vector results using Reciprocal Rank Fusion and return sorted IDs."""
        def extract_id(result: Any) -> str:
            if isinstance(result, dict):
                return result.get("id", "")
            return result[0]

        bm25_scores = {extract_id(result): 1 / (k + rank + 1) for rank, result in enumerate(bm25_results) if extract_id(result)}
        vector_scores = {extract_id(result): 1 / (k + rank + 1) for rank, result in enumerate(vector_results) if extract_id(result)}
        
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        combined_scores = {}
        
        for doc_id in all_doc_ids:
            combined_scores[doc_id] = bm25_scores.get(doc_id, 0) + vector_scores.get(doc_id, 0)
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_results]

    def hybrid_search_records(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Combine BM25 and semantic search via RRF and keep source metadata intact."""
        bm25_results = self.bm25_search_records(query, n_results * 2)
        vec_text_results = self.vector_search_records(query, n_results * 2)

        if not (bm25_results or vec_text_results):
            return []

        fused_text_ids = self.reciprocal_rank_fusion(bm25_results, vec_text_results)
        top_text_ids = fused_text_ids[:n_results]

        id_to_record = {}
        for record in vec_text_results + bm25_results:
            if record["id"] not in id_to_record:
                id_to_record[record["id"]] = record

        return [id_to_record[chunk_id] for chunk_id in top_text_ids if chunk_id in id_to_record]

    def hybrid_search(self, query: str, n_results: int = 3) -> List[str]:
        """Combine BM25 and semantic search via RRF and return the top document texts."""
        return [record["content"] for record in self.hybrid_search_records(query, n_results)]

    def build_retrieval_query(self, query_text: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Build a retrieval query that includes non-system conversation history."""
        messages = messages or []
        history_context = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in messages
            if msg.get('role') != 'system' and msg.get('content')
        )
        return f"{history_context}\nUser: {query_text}" if history_context else query_text

    def get_retrieval_records(self, query: str, n_results: int = 3, use_hybrid_search: bool = False) -> List[Dict[str, Any]]:
        """Retrieve source-aware records for grounded prompt construction."""
        if use_hybrid_search and self.bm25_model:
            logger.info("Using hybrid search (BM25 + semantic)")
            return self.hybrid_search_records(query, n_results)

        logger.info("Using semantic vector search")
        return self.vector_search_records(query, n_results)

    def _format_source_locator(self, label: str, value: Any) -> Optional[str]:
        """Format a single source location field for display."""
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"{label} {value}"

    def _is_meaningful_section_title(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Filter out boilerplate section titles that make citations noisy."""
        normalized = re.sub(r"\s+", " ", str(title or "")).strip().strip(":").strip()
        if not normalized:
            return False

        filename = str((metadata or {}).get("filename") or "").strip().lower()
        generic_patterns = [
            r"^ocr extracted text\b",
            r"^ocr page text\b",
            r"^page\s+\d+\s+headers?\b",
            r"^page\s+\d+\s+footers?\b",
            r"^table\s+\d+\b",
            r"^enhanced pdf image analysis\b",
            r"^word document image\b",
            r"^image analysis\b",
            r"^document structure analysis\b",
            r"^document overview\b",
        ]

        if any(re.match(pattern, normalized, re.IGNORECASE) for pattern in generic_patterns):
            return False
        if filename and filename in normalized.lower():
            return False
        return True

    def _format_source_kind(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Add a short, human-readable content hint when a section title is not useful."""
        content_type = str((metadata or {}).get("content_type") or "").lower()
        content_type_labels = {
            "ocr_text": "OCR text",
            "pdf_page_ocr": "OCR page",
            "pdf_table_camelot": "table",
            "pdf_table_pdfplumber": "table",
            "pdf_table_structured": "table",
            "docx_table": "table",
            "structured_content": "structured content",
        }
        return content_type_labels.get(content_type)

    def _format_source_label(self, metadata: Optional[Dict[str, Any]], fallback_index: int) -> str:
        """Create a compact label for a retrieved chunk."""
        metadata = metadata or {}
        filename = str(metadata.get("filename") or metadata.get("doc_id") or f"Source {fallback_index}")

        location_parts = []
        for label, key in (("page", "page_number"), ("slide", "slide_number"), ("sheet", "sheet")):
            locator = self._format_source_locator(label, metadata.get(key))
            if locator:
                location_parts.append(locator)

        section_title = str(metadata.get("section_title") or "").strip()
        if self._is_meaningful_section_title(section_title, metadata):
            location_parts.append(f"section {section_title[:80]}")
        else:
            source_kind = self._format_source_kind(metadata)
            if source_kind:
                location_parts.append(source_kind)

        if location_parts:
            return f"{filename}, {', '.join(location_parts)}"
        return filename

    def _truncate_source_content(self, content: str, max_chars: int = 1600) -> str:
        """Keep prompt source blocks readable and bounded."""
        normalized = re.sub(r'\n{3,}', '\n\n', content.strip())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def build_grounded_user_prompt(self, query_text: str, source_records: List[Dict[str, Any]], voice_mode: bool = False) -> str:
        """Create a grounded prompt without exposing citation markers in the answer."""
        if voice_mode:
            instruction_block = (
                "Answer only from the provided context passages. "
                "Do not mention source numbers, filenames, citations, or a sources section unless the user explicitly asks. "
                "If the context is insufficient, say so briefly."
            )
        else:
            instruction_block = (
                "Answer only from the provided context passages. "
                "Write a clean, direct answer and do not mention source numbers, filenames, citations, or a sources section unless the user explicitly asks. "
                "If the context is insufficient, say so clearly."
            )

        if not source_records:
            return (
                f"{instruction_block}\n\n"
                "No relevant context was retrieved from the uploaded documents.\n\n"
                f"User question: {query_text}"
            )

        context_blocks = [
            self._truncate_source_content(record.get("content", ""))
            for record in source_records
            if record.get("content", "").strip()
        ]

        return (
            f"{instruction_block}\n\n"
            f"Context passages:\n\n---\n{('\n\n---\n').join(context_blocks)}\n\n"
            f"User question: {query_text}"
        )

    def build_cited_user_prompt(self, query_text: str, source_records: List[Dict[str, Any]], voice_mode: bool = False) -> str:
        """Backward-compatible alias for grounded prompts after citation removal."""
        return self.build_grounded_user_prompt(query_text, source_records, voice_mode=voice_mode)

    def build_sources_footer(self, source_records: List[Dict[str, Any]], voice_mode: bool = False) -> str:
        """RAG citations are disabled, so no footer is appended."""
        return ""

    async def query(self, query_text, system_prompt="You are a helpful assistant.", messages=None, n_results=3, use_hybrid_search=False, model: Optional[str] = None, conversation_summary: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2048, top_p: float = 0.9, frequency_penalty: float = 0.0, repetition_penalty: float = 1.0, is_voice_mode: bool = False):
        """Answer a query using semantic or hybrid retrieval and stream model tokens via Ollama.
        
        Args:
            query_text: The user's query
            system_prompt: System prompt for the LLM
            messages: Conversation history
            n_results: Number of documents to retrieve
            use_hybrid_search: Whether to use hybrid BM25 + semantic search
            model: Model name to use
            conversation_summary: Summary of earlier conversation for context
            temperature: Temperature for LLM response generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalize frequent tokens (-2.0 to 2.0)
            repetition_penalty: Penalize repeated tokens (0.0 to 2.0)
            is_voice_mode: Whether this is voice mode (uses shorter, conversational responses)
        """
        messages = messages or []
        model_name = (model or os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:1b")).replace("ollama/","")
        full_query = self.build_retrieval_query(query_text, messages)
        source_records = self.get_retrieval_records(full_query, n_results, use_hybrid_search)

        # Build messages with conversation memory support
        llm_messages = []
        
        # Add conversation summary if available
        if conversation_summary:
            llm_messages.append({
                "role": "system",
                "content": f"Summary of previous conversation:\n{conversation_summary}"
            })
        
        # Add conversation history
        llm_messages.extend(messages)
        
        # Use voice-optimized system prompt if in voice mode
        if is_voice_mode and system_prompt == "You are a helpful assistant.":
            from app.frontend.config import DEFAULT_RAG_SYSTEM_PROMPT_VOICE
            system_prompt = DEFAULT_RAG_SYSTEM_PROMPT_VOICE
        
        # Adjust max_tokens for voice mode
        if is_voice_mode:
            max_tokens = min(max_tokens, 300)
        
        # Add system prompt and current query with context
        llm_messages.append({"role": "system", "content": system_prompt})
        llm_messages.append({
            "role": "system",
            "content": "Do not include citations, source numbers, filenames, bracketed references, or a Sources section unless the user explicitly asks for sources.",
        })
        llm_messages.append({
            "role": "user",
            "content": self.build_grounded_user_prompt(query_text, source_records, voice_mode=is_voice_mode),
        })
        
        async for chunk in stream_ollama(
            llm_messages, 
            model=model_name, 
            temperature=temperature, 
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty
        ):
            yield chunk


class CrewAIRAGOrchestrator:
    """Coordinates refined querying and answer composition using an Ollama-backed model."""
    def __init__(self, rag_service: RAGService, model_name="gemma3:1b"):
        """Configure the Ollama LLM, agent roles, and default context parameters."""
        self.rag_service = rag_service
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        
        self.ollama_llm = LLM(
            provider="ollama",
            model=model_name,
            api_base=self._get_ollama_url(),
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

    def _get_ollama_url(self) -> str:
        """Get the appropriate Ollama URL based on execution environment."""
        try:
            from app.services.host_service_manager import host_service_manager
            url = host_service_manager.environment_config.ollama_url
            logger.debug(f"Using Ollama URL from host service manager: {url}")
            return url
        except ImportError:
            logger.warning("Host service manager not available, using fallback config")
            url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            logger.debug(f"Using fallback Ollama URL: {url}")
            return url

    async def _get_context_length(self, model_name: str) -> int:
        """Query Ollama for the model's context length; fall back to a sensible default."""
        model = model_name.replace("ollama/", "") if "ollama/" in model_name else model_name
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._get_ollama_url()}/api/show",
                    json={"name": model}
                )
                resp.raise_for_status()
                data = resp.json()
                params = data.get('model_info', '')
                
                # Security: Use JSON parsing instead of ast.literal_eval
                if isinstance(params, str):
                    import json
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try ast.literal_eval as fallback
                        # ast.literal_eval is safe for literal structures only
                        import ast
                        try:
                            params = ast.literal_eval(params)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Failed to parse model_info: {e}")
                            return 4096
                
                if not isinstance(params, dict):
                    return 4096
                
                for key, value in params.items():
                    if key.endswith("context_length"):
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid context_length value: {value}")
                            return 4096
                
                return 4096
        except Exception as e:
            logger.warning(f"Error getting context length: {e}")
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
        """Simple character-based chunking without overlap (used for specific cases)."""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

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

    async def query(self, user_query: str, system_prompt: str, messages=[], n_results: int = 3, use_hybrid_search: bool = False, model: Optional[str] = None):
        """Refine the user query, retrieve/summarize context, and compose a final streamed answer."""
        model_name = (model or os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:1b")).replace("ollama/","")
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