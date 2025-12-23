import io
import mimetypes
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import warnings
from app.ingestion.enhanced_document_processor import extract_pdf_enhanced, extract_docx_enhanced, extract_epub_enhanced

# Suppress pypdf warnings about malformed PDF objects
logging.getLogger('pypdf._reader').setLevel(logging.ERROR)
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='pypdf')

try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Table, NarrativeText, Title, Header, Footer, FigureCaption
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

import fitz
from PIL import Image
import csv

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Comprehensive file type support
SUPPORTED_TEXT_EXT = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".rtf", ".html", ".htm", 
    ".txt", ".md", ".epub", ".odt", ".org", ".rst", ".xlsx", ".xls"
}
SUPPORTED_IMAGE_EXT = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".gif", ".heic", ".svg"
}
SUPPORTED_DATA_EXT = {".csv", ".tsv", ".xls", ".xlsx"}

def _read_file_bytes(path: Path) -> bytes:
    """Read file as bytes."""
    with open(path, "rb") as f:
        return f.read()

def _guess_mime(path: Path) -> str:
    """Guess MIME type from file extension."""
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"

def _pil_from_pix(pix):
    """Convert PyMuPDF pixmap to PIL Image bytes."""
    if pix.n > 3:  # Convert CMYK to RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def extract_with_unstructured(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Parse document with Unstructured library.
    Returns: text_chunks, image_records, table_texts
    """
    if not UNSTRUCTURED_AVAILABLE:
        logger.warning("Unstructured library not available, falling back to basic parsing")
        return [], [], []
    
    try:
        elements = partition(filename=str(path))
        text_chunks, image_records, table_texts = [], [], []
        
        for el in elements:
            meta = {
                "filename": path.name,
                "filetype": _guess_mime(path),
                "category": getattr(el, "category", None),
                "page_number": getattr(el.metadata, "page_number", None) if hasattr(el, "metadata") else None,
            }
            
            # Tables
            if isinstance(el, Table):
                table_texts.append({"content": el.text, "metadata": meta})
            
            # Skip figure captions to avoid noisy text
            elif isinstance(el, FigureCaption):
                continue
            
            # General text content
            elif isinstance(el, (NarrativeText, Title, Header, Footer)) or hasattr(el, "text"):
                txt = getattr(el, "text", None)
                if txt and txt.strip() and len(txt.strip()) > 10:  # Filter very short fragments
                    text_chunks.append({"content": txt, "metadata": meta})
        
        # Handle standalone image files
        if path.suffix.lower() in SUPPORTED_IMAGE_EXT:
            img_bytes = _read_file_bytes(path)
            image_records.append({
                "image_bytes": img_bytes, 
                "caption": f"Image file: {path.stem}", 
                "metadata": {"filename": path.name, "filetype": _guess_mime(path)}
            })
        
        return text_chunks, image_records, table_texts
    
    except Exception as e:
        logger.error(f"Unstructured parsing failed for {path}: {e}")
        return [], [], []

def extract_pdf_with_pymupdf(path: Path) -> Tuple[List[dict], List[dict]]:
    """
    Clean PDF extraction with PyMuPDF - avoid noisy image captions.
    """
    try:
        doc = fitz.open(str(path))
        text_chunks, image_records = [], []
        
        for page_index in range(len(doc)):
            page = doc[page_index]
            
            # Extract clean text blocks (not full page text)
            text_dict = page.get_text("dict")
            clean_text_blocks = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text and len(text) > 3:  # Filter very short fragments
                                block_text += text + " "
                    
                    if block_text.strip() and len(block_text.strip()) > 20:
                        clean_text_blocks.append(block_text.strip())
            
            # Add clean text blocks as separate chunks
            for i, text_block in enumerate(clean_text_blocks):
                text_chunks.append({
                    "content": text_block,
                    "metadata": {
                        "filename": path.name,
                        "page_number": page_index + 1,
                        "block_number": i,
                        "filetype": "application/pdf"
                    }
                })
            
            # Extract images without using page text as caption
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip very small images (likely icons/decorations)
                    if pix.width < 100 or pix.height < 100:
                        pix = None
                        continue
                    
                    img_bytes = _pil_from_pix(pix)
                    pix = None  # Free memory
                    
                    # Use clean, simple caption without page text
                    caption = f"{path.stem} - Page {page_index + 1}, Image {img_index + 1}"
                    
                    image_records.append({
                        "image_bytes": img_bytes,
                        "caption": caption,
                        "metadata": {
                            "filename": path.name,
                            "page_number": page_index + 1,
                            "image_index": img_index,
                            "filetype": "application/pdf"
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_index + 1}: {e}")
                    continue
        
        doc.close()
        return text_chunks, image_records
    
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {path}: {e}")
        return [], []

def extract_excel(path: Path) -> List[dict]:
    """Extract text content from Excel files."""
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas not available for Excel parsing. Install with: uv pip install pandas openpyxl")
        return []
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(path)
        blocks = []
        
        for sheet_name in excel_file.sheet_names:
            try:
                # Read sheet with string dtype to preserve formatting
                df = excel_file.parse(sheet_name, dtype=str).fillna("")
                
                # Skip empty sheets
                if df.empty:
                    continue
                
                # Limit rows to avoid memory issues
                df = df.head(100)
                
                # Convert to text format
                content_lines = [f"SHEET: {sheet_name}"]
                
                # Add headers
                headers = df.columns.tolist()
                content_lines.append("HEADERS: " + " | ".join(str(h) for h in headers))
                
                # Add rows
                for idx, row in df.iterrows():
                    row_text = " | ".join(str(val) for val in row.values)
                    content_lines.append(f"ROW {idx + 1}: {row_text}")
                
                content = "\n".join(content_lines)
                
                blocks.append({
                    "content": content,
                    "metadata": {
                        "filename": path.name,
                        "sheet": sheet_name,
                        "filetype": "application/vnd.ms-excel",
                        "rows": len(df)
                    }
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse sheet '{sheet_name}' in {path}: {e}")
                continue
        
        return blocks
    
    except Exception as e:
        logger.error(f"Excel extraction failed for {path}: {e}")
        return []

def extract_csv(path: Path) -> List[dict]:
    """Extract text content from CSV files."""
    try:
        blocks = []
        with open(path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader, [])
            except:
                return []
            
            rows = []
            for i, row in enumerate(reader):
                if i >= 1000:  # Limit rows
                    break
                rows.append(row)
        
        # Create text blocks
        chunk_size = 50
        for i in range(0, len(rows), chunk_size):
            chunk_rows = rows[i:i + chunk_size]
            
            content_lines = [f"CSV FILE: {path.name}"]
            content_lines.append("HEADERS: " + " | ".join(headers))
            
            for j, row in enumerate(chunk_rows, i + 1):
                row_text = " | ".join(str(val) for val in row)
                content_lines.append(f"ROW {j}: {row_text}")
            
            content = "\n".join(content_lines)
            
            blocks.append({
                "content": content,
                "metadata": {
                    "filename": path.name,
                    "filetype": "text/csv",
                    "chunk": i // chunk_size + 1,
                    "rows": len(chunk_rows)
                }
            })
        
        return blocks
    
    except Exception as e:
        logger.error(f"CSV extraction failed for {path}: {e}")
        return []

def extract_txt(path: Path) -> List[dict]:
    """Extract text from plain text files."""
    try:
        content = _read_file_bytes(path).decode("utf-8", errors="ignore")
        return [{
            "content": content,
            "metadata": {
                "filename": path.name,
                "filetype": "text/plain"
            }
        }]
    except Exception as e:
        logger.error(f"TXT extraction failed for {path}: {e}")
        return []

def extract_presentation_images(path: Path) -> List[dict]:
    """Extract images from PowerPoint presentations."""
    if not PPTX_AVAILABLE:
        return []
    
    try:
        prs = Presentation(str(path))
        image_records = []
        
        for slide_idx, slide in enumerate(prs.slides):
            for shape_idx, shape in enumerate(slide.shapes):
                if hasattr(shape, "image"):
                    try:
                        img = shape.image
                        img_bytes = img.blob
                        caption = f"{path.stem} - Slide {slide_idx + 1}, Image {shape_idx + 1}"
                        
                        image_records.append({
                            "image_bytes": img_bytes,
                            "caption": caption,
                            "metadata": {
                                "filename": path.name,
                                "slide_number": slide_idx + 1,
                                "shape_index": shape_idx,
                                "filetype": "application/vnd.ms-powerpoint"
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image from slide {slide_idx + 1}: {e}")
                        continue
        
        return image_records
    
    except Exception as e:
        logger.error(f"PowerPoint image extraction failed for {path}: {e}")
        return []

def extract_standalone_image(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Handle standalone image files."""
    try:
        img_bytes = _read_file_bytes(path)
        
        # Validate it's actually an image
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()
        except:
            logger.warning(f"Invalid image file: {path}")
            return [], [], []
        
        image_record = {
            "image_bytes": img_bytes,
            "caption": f"Image file: {path.stem}",
            "metadata": {
                "filename": path.name,
                "filetype": _guess_mime(path)
            }
        }
        
        return [], [image_record], []
    
    except Exception as e:
        logger.error(f"Image file processing failed for {path}: {e}")
        return [], [], []

# Replace the PDF processing section in ingest_file function
def ingest_file(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Main file ingestion function with enhanced document processing.
    Returns: (text_chunks, image_records, table_texts)
    """
    ext = path.suffix.lower()
    logger.info(f"Processing file: {path} (extension: {ext})")

    # Excel files
    if ext in {".xls", ".xlsx"}:
        return extract_excel(path), [], []

    # CSV files
    if ext in {".csv", ".tsv"}:
        return extract_csv(path), [], []

    # Plain text files
    if ext in {".txt", ".md"}:
        return extract_txt(path), [], []

    # Standalone image files
    if ext in SUPPORTED_IMAGE_EXT:
        return extract_standalone_image(path)

    # Enhanced PDF processing
    if ext == ".pdf":
        logger.info("Using enhanced PDF extraction")
        enhanced_content = extract_pdf_enhanced(path)
        return enhanced_content, [], []  # Images are processed within the enhanced content

    # Enhanced Word document processing
    if ext in {".docx", ".doc"}:
        logger.info("Using enhanced DOCX extraction")
        enhanced_content = extract_docx_enhanced(path)
        return enhanced_content, [], []

    # Enhanced EPUB processing
    if ext == ".epub":
        logger.info("Using enhanced EPUB extraction")
        enhanced_content = extract_epub_enhanced(path)
        return enhanced_content, [], []

    # Other document formats - use enhanced processing if possible, otherwise fall back to Unstructured
    if ext in SUPPORTED_TEXT_EXT:
        # Try enhanced processing first for known formats
        if ext in {".pptx", ".ppt"}:
            # For presentations, we can add enhanced processing later
            text_chunks, image_records, table_texts = extract_with_unstructured(path)
            
            # Supplement with image extraction for presentations
            if not image_records:
                presentation_images = extract_presentation_images(path)
                image_records.extend(presentation_images)
            
            return text_chunks, image_records, table_texts
        else:
            # Use existing unstructured processing for other formats
            text_chunks, image_records, table_texts = extract_with_unstructured(path)
            return text_chunks, image_records, table_texts

    # Unsupported format
    logger.warning(f"Unsupported file format: {ext}")
    return [], [], []