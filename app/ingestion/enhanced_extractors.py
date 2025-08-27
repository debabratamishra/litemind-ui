import pandas as pd
import numpy as np
import io
import logging
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator

# Optional imports for enhanced features
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    easyocr = None
    EASYOCR_AVAILABLE = False
    logging.warning(f"EasyOCR not available: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    cv2 = None
    CV2_AVAILABLE = False
    logging.warning(f"OpenCV not available: {e}")

from .enhanced_document_processor import extract_pdf_enhanced, extract_docx_enhanced, extract_epub_enhanced, get_document_processor

logger = logging.getLogger(__name__)

class EnhancedImageProcessor:
    """Enhanced image processor with OCR and content analysis."""
    
    def __init__(self):
        self.ocr_reader = None
        self.initialized = False
        
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                self.initialized = True
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"OCR initialization failed: {e}. Falling back to basic processing.")
                self.ocr_reader = None
                self.initialized = False
        else:
            logger.warning("EasyOCR not available. Using basic image processing only.")
    
    def extract_image_content(self, image_bytes: bytes, metadata: dict) -> List[dict]:
        """Extract meaningful content from images using OCR and analysis."""
        extracted_content = []
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize large images to save memory
            if image.width > 2000 or image.height > 2000:
                image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # 1. OCR Text Extraction
            if self.initialized and self.ocr_reader:
                ocr_content = self._extract_ocr_text(img_array, metadata)
                if ocr_content:
                    extracted_content.append(ocr_content)
            
            # 2. Basic Image Analysis
            analysis = self._analyze_image_properties(image, metadata)
            if analysis:
                extracted_content.append(analysis)
            
            # 3. Detect structured content
            structured_content = self._detect_structured_content(img_array, metadata)
            if structured_content:
                extracted_content.extend(structured_content)
                
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
            # Fallback to basic reference
            extracted_content.append({
                "content": f"IMAGE REFERENCE: {metadata.get('filename', 'unknown')} - Unable to extract content",
                "metadata": {**metadata, "content_type": "image_reference", "error": str(e)}
            })
        
        return extracted_content if extracted_content else [self._create_fallback_content(metadata)]
    
    def _extract_ocr_text(self, img_array: np.ndarray, metadata: dict) -> Optional[dict]:
        """Extract text using OCR."""
        try:
            ocr_results = self.ocr_reader.readtext(img_array)
            
            if ocr_results:
                extracted_text = []
                high_confidence_text = []
                
                for (bbox, text, confidence) in ocr_results:
                    if confidence > 0.3:  # Include lower confidence for more text
                        extracted_text.append(text)
                    if confidence > 0.7:  # High confidence text
                        high_confidence_text.append(text)
                
                if extracted_text:
                    all_text = " ".join(extracted_text)
                    high_conf_text = " ".join(high_confidence_text)
                    
                    content = f"""OCR EXTRACTED TEXT from {metadata.get('filename', 'image')}:
                        HIGH CONFIDENCE TEXT: {high_conf_text}
                        ALL EXTRACTED TEXT: {all_text}
                        Total text blocks found: {len(extracted_text)}
                        """
                    
                    return {
                        "content": content,
                        "metadata": {
                            **metadata,
                            "content_type": "ocr_text",
                            "text_blocks_count": len(extracted_text),
                            "high_confidence_blocks": len(high_confidence_text)
                        }
                    }
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        
        return None
    
    def _analyze_image_properties(self, image: Image.Image, metadata: dict) -> dict:
        """Extract basic image properties and characteristics."""
        try:
            width, height = image.size
            aspect_ratio = width / height
            
            # Convert to numpy for analysis
            img_array = np.array(image)
            
            # Basic analysis
            avg_brightness = np.mean(img_array)
            color_variance = np.var(img_array)
            
            # Determine likely content type
            image_type = self._classify_image_type(aspect_ratio, color_variance, avg_brightness)
            
            content = f"""IMAGE ANALYSIS for {metadata.get('filename', 'image')}:
                Dimensions: {width}x{height} pixels
                Aspect Ratio: {aspect_ratio:.2f}
                Average Brightness: {avg_brightness:.1f}/255
                Color Variance: {color_variance:.1f}
                Likely Content Type: {image_type}
                File Size: {metadata.get('file_size', 'unknown')}
                """
            
            return {
                "content": content,
                "metadata": {
                    **metadata,
                    "content_type": "image_analysis",
                    "image_type": image_type,
                    "dimensions": f"{width}x{height}",
                    "aspect_ratio": aspect_ratio
                }
            }
            
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return None
    
    def _classify_image_type(self, aspect_ratio: float, color_variance: float, avg_brightness: float) -> str:
        """Classify image type based on characteristics."""
        if aspect_ratio > 3 or aspect_ratio < 0.3:
            return "banner_or_wide_chart"
        elif color_variance < 1000 and avg_brightness > 200:
            return "text_document_or_screenshot"
        elif color_variance > 5000:
            return "photograph_or_complex_image"
        elif avg_brightness > 220:
            return "diagram_or_whiteboard"
        else:
            return "mixed_content"
    
    def _detect_structured_content(self, img_array: np.ndarray, metadata: dict) -> List[dict]:
        """Detect tables, charts, or other structured content."""
        content = []
        
        if not CV2_AVAILABLE:
            logger.debug("OpenCV not available, skipping structured content detection")
            return content
        
        try:
            # Convert to grayscale for line detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Count significant lines
            h_lines = np.sum(horizontal_lines > 50)
            v_lines = np.sum(vertical_lines > 50)
            
            if h_lines > 100 and v_lines > 100:
                content.append({
                    "content": f"STRUCTURED CONTENT DETECTED in {metadata.get('filename', 'image')}: This image contains tabular data or structured layout with {h_lines} horizontal and {v_lines} vertical line elements. This suggests the presence of tables, forms, or organized data that may require special attention during analysis.",
                    "metadata": {
                        **metadata,
                        "content_type": "structured_content",
                        "detected_type": "table_or_form",
                        "line_counts_horizontal": int(h_lines),
                        "line_counts_vertical": int(v_lines)
                    }
                })
            elif h_lines > 50 or v_lines > 50:
                content.append({
                    "content": f"PARTIAL STRUCTURE DETECTED in {metadata.get('filename', 'image')}: Some organized elements found with {h_lines} horizontal and {v_lines} vertical elements.",
                    "metadata": {
                        **metadata,
                        "content_type": "partial_structure",
                        "line_counts_horizontal": int(h_lines),
                        "line_counts_vertical": int(v_lines)
                    }
                })
            
        except Exception as e:
            logger.warning(f"Structured content detection failed: {e}")
        
        return content
    
    def _create_fallback_content(self, metadata: dict) -> dict:
        """Create fallback content when processing fails."""
        return {
            "content": f"IMAGE FILE: {metadata.get('filename', 'unknown')} - Basic image reference (content extraction not available)",
            "metadata": {**metadata, "content_type": "image_reference"}
        }


class EnhancedCSVProcessor:
    """Enhanced CSV processor with intelligent data analysis."""
    
    def __init__(self, max_rows_analysis: int = 10000):
        self.max_rows_analysis = max_rows_analysis
    
    def extract_csv_enhanced(self, path: Path) -> List[dict]:
        """Enhanced CSV extraction with intelligent data analysis."""
        try:
            # Read with efficient data types and sampling for large files
            file_size = path.stat().st_size
            
            if file_size > 50 * 1024 * 1024:  # Files larger than 50MB
                return self._extract_large_csv_streaming(path)
            else:
                return self._extract_csv_full(path)
                
        except Exception as e:
            logger.error(f"Enhanced CSV extraction failed for {path}: {e}")
            return self._create_fallback_csv_content(path, str(e))
    
    def _extract_csv_full(self, path: Path) -> List[dict]:
        """Extract full CSV for smaller files."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not decode CSV with any standard encoding")
            
            if df.empty:
                return [self._create_empty_csv_content(path)]
            
            # Limit analysis for very large datasets
            analysis_df = df.head(self.max_rows_analysis) if len(df) > self.max_rows_analysis else df
            
            blocks = []
            
            # 1. Dataset Overview
            overview = self._create_dataset_overview(df, analysis_df, path.name)
            blocks.append(overview)
            
            # 2. Column Analysis
            for col in analysis_df.columns[:20]:  # Limit to first 20 columns
                col_analysis = self._analyze_column(analysis_df, col, path.name)
                if col_analysis:
                    blocks.append(col_analysis)
            
            # 3. Smart Chunking
            chunks = self._create_intelligent_chunks(df, path.name)
            blocks.extend(chunks)
            
            # 4. Statistical Summary
            stats_summary = self._create_statistical_summary(analysis_df, path.name)
            if stats_summary:
                blocks.append(stats_summary)
            
            return blocks
            
        except Exception as e:
            logger.error(f"Full CSV extraction failed: {e}")
            return self._create_fallback_csv_content(path, str(e))
    
    def _extract_large_csv_streaming(self, path: Path) -> List[dict]:
        """Handle large CSV files with streaming approach."""
        blocks = []
        chunk_size = 1000
        total_rows = 0
        
        try:
            # First, get basic info
            first_chunk = pd.read_csv(path, nrows=100, encoding='utf-8')
            
            # Create overview from first chunk
            overview = {
                "content": f"""LARGE CSV DATASET: {path.name}
                File Size: {path.stat().st_size / (1024*1024):.1f} MB
                Estimated Rows: Large file (processed in chunks)
                Columns ({len(first_chunk.columns)}): {', '.join(first_chunk.columns[:10])}
                Note: This is a large file processed in streaming mode for memory efficiency.
                """,
                "metadata": {
                    "filename": path.name,
                    "filetype": "text/csv",
                    "content_type": "large_dataset_overview",
                    "processing_mode": "streaming"
                }
            }
            blocks.append(overview)
            
            # Process in chunks
            chunk_num = 0
            for chunk_df in pd.read_csv(path, chunksize=chunk_size, encoding='utf-8'):
                if chunk_num < 5:  # Process only first 5 chunks for analysis
                    chunk_content = self._create_chunk_content(
                        chunk_df, path.name, 
                        f"Chunk {chunk_num + 1} (Rows {total_rows + 1}-{total_rows + len(chunk_df)})"
                    )
                    blocks.append(chunk_content)
                
                total_rows += len(chunk_df)
                chunk_num += 1
                
                if chunk_num >= 10:  # Limit processing for very large files
                    break
            
            # Update overview with actual count
            blocks[0]["content"] += f"\nActual rows processed: {total_rows}"
            
        except Exception as e:
            logger.error(f"Streaming CSV extraction failed: {e}")
            return self._create_fallback_csv_content(path, str(e))
        
        return blocks
    
    def _create_dataset_overview(self, df: pd.DataFrame, analysis_df: pd.DataFrame, filename: str) -> dict:
        """Create comprehensive dataset overview."""
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = analysis_df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = analysis_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Detect potential date columns that weren't auto-detected
        potential_date_cols = []
        for col in categorical_cols[:5]:  # Check first 5 categorical columns
            if analysis_df[col].dtype == 'object':
                sample_values = analysis_df[col].dropna().head(3).astype(str)
                if any(self._looks_like_date(val) for val in sample_values):
                    potential_date_cols.append(col)
        
        overview_text = f"""CSV DATASET OVERVIEW: {filename}
            Total Rows: {len(df):,}
            Total Columns: {len(df.columns)}
            Data Types Breakdown:
            • Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}
            • Text/Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
            • Date/Time Columns ({len(datetime_cols)}): {', '.join(datetime_cols)}
            • Potential Date Columns: {', '.join(potential_date_cols) if potential_date_cols else 'None detected'}

            Data Quality:
            • Missing Values: {analysis_df.isnull().sum().sum():,} total missing values
            • Complete Rows: {len(analysis_df.dropna()):,} ({(len(analysis_df.dropna())/len(analysis_df)*100):.1f}%)
            • Duplicate Rows: {analysis_df.duplicated().sum():,}

            Sample Column Names: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}
            """
        
        return {
            "content": overview_text,
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "dataset_overview",
                "importance": "high",
                "row_count": len(df),
                "column_count": len(df.columns)
            }
        }
    
    def _looks_like_date(self, value: str) -> bool:
        """Simple heuristic to detect date-like strings."""
        if not isinstance(value, str) or len(value) < 6:
            return False
        
        date_indicators = ['-', '/', '.', '2020', '2021', '2022', '2023', '2024', '2025']
        return any(indicator in value for indicator in date_indicators)
    
    def _analyze_column(self, df: pd.DataFrame, column: str, filename: str) -> dict:
        """Analyze individual column characteristics."""
        col_data = df[column]
        
        try:
            if col_data.dtype in ['object', 'category']:
                return self._analyze_categorical_column(col_data, column, filename)
            elif col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                return self._analyze_numerical_column(col_data, column, filename)
            elif col_data.dtype.name.startswith('datetime'):
                return self._analyze_datetime_column(col_data, column, filename)
            else:
                return self._analyze_other_column(col_data, column, filename)
        except Exception as e:
            logger.warning(f"Column analysis failed for {column}: {e}")
            return None
    
    def _analyze_categorical_column(self, col_data: pd.Series, column: str, filename: str) -> dict:
        """Analyze categorical column."""
        value_counts = col_data.value_counts().head(10)
        unique_count = col_data.nunique()
        null_count = col_data.isnull().sum()
        
        # Detect if this might be an ID column
        is_likely_id = unique_count == len(col_data) and null_count == 0
        
        content = f"""COLUMN ANALYSIS: {column}
Type: Categorical/Text
Unique Values: {unique_count:,} ({(unique_count/len(col_data)*100):.1f}% of total)
Missing Values: {null_count:,} ({(null_count/len(col_data)*100):.1f}%)
Likely ID Column: {'Yes' if is_likely_id else 'No'}

Most Common Values:
{value_counts.to_string()}

Data Quality Notes:
  • Average text length: {col_data.astype(str).str.len().mean():.1f} characters
  • Empty strings: {(col_data == '').sum():,}
"""
        
        return {
            "content": content,
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "column_analysis",
                "column_name": column,
                "column_type": "categorical",
                "unique_values": unique_count,
                "is_likely_id": is_likely_id
            }
        }
    
    def _analyze_numerical_column(self, col_data: pd.Series, column: str, filename: str) -> dict:
        """Analyze numerical column."""
        stats = col_data.describe()
        null_count = col_data.isnull().sum()
        
        # Detect potential issues
        has_outliers = self._detect_outliers(col_data)
        is_integer_like = col_data.dropna().apply(lambda x: float(x).is_integer()).all()
        
        content = f"""COLUMN ANALYSIS: {column}
Type: Numerical ({'Integer-like' if is_integer_like else 'Decimal'})
Missing Values: {null_count:,} ({(null_count/len(col_data)*100):.1f}%)

Statistical Summary:
  • Count: {stats['count']:,.0f}
  • Mean: {stats['mean']:.2f}
  • Median: {stats['50%']:.2f}
  • Range: {stats['min']:.2f} to {stats['max']:.2f}
  • Standard Deviation: {stats['std']:.2f}
  • Quartiles: Q1={stats['25%']:.2f}, Q3={stats['75%']:.2f}

Data Quality:
  • Potential Outliers: {'Detected' if has_outliers else 'None detected'}
  • Zero Values: {(col_data == 0).sum():,}
  • Negative Values: {(col_data < 0).sum():,}
"""
        
        return {
            "content": content,
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "column_analysis",
                "column_name": column,
                "column_type": "numerical",
                "has_outliers": has_outliers,
                "is_integer_like": is_integer_like
            }
        }
    
    def _detect_outliers(self, col_data: pd.Series) -> bool:
        """Simple outlier detection using IQR method."""
        try:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            return len(outliers) > 0
        except:
            return False
    
    def _analyze_datetime_column(self, col_data: pd.Series, column: str, filename: str) -> dict:
        """Analyze datetime column."""
        null_count = col_data.isnull().sum()
        valid_dates = col_data.dropna()
        
        if len(valid_dates) > 0:
            date_range = f"{valid_dates.min()} to {valid_dates.max()}"
            span_days = (valid_dates.max() - valid_dates.min()).days
        else:
            date_range = "No valid dates"
            span_days = 0
        
        content = f"""COLUMN ANALYSIS: {column}
Type: Date/Time
Missing Values: {null_count:,} ({(null_count/len(col_data)*100):.1f}%)

Date Range: {date_range}
Time Span: {span_days:,} days
Valid Dates: {len(valid_dates):,}
"""
        
        return {
            "content": content,
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "column_analysis",
                "column_name": column,
                "column_type": "datetime",
                "date_span_days": span_days
            }
        }
    
    def _analyze_other_column(self, col_data: pd.Series, column: str, filename: str) -> dict:
        """Analyze other column types."""
        content = f"""COLUMN ANALYSIS: {column}
Type: {col_data.dtype}
Missing Values: {col_data.isnull().sum():,}
Unique Values: {col_data.nunique():,}
Sample Values: {', '.join(col_data.dropna().head(3).astype(str))}
"""
        
        return {
            "content": content,
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "column_analysis",
                "column_name": column,
                "column_type": str(col_data.dtype)
            }
        }
    
    def _create_intelligent_chunks(self, df: pd.DataFrame, filename: str, chunk_size: int = 100) -> List[dict]:
        """Create intelligent data chunks based on data patterns."""
        chunks = []
        
        # Try to find a natural grouping column
        grouping_col = self._find_grouping_column(df)
        
        if grouping_col and df[grouping_col].nunique() <= 50:  # Reasonable number of groups
            # Group by natural categories
            for group_name, group_df in df.groupby(grouping_col):
                if len(group_df) > chunk_size:
                    # Split large groups
                    for i in range(0, len(group_df), chunk_size):
                        chunk_df = group_df.iloc[i:i+chunk_size]
                        content = self._create_chunk_content(
                            chunk_df, filename, 
                            f"{grouping_col}={group_name}, Part {i//chunk_size + 1}"
                        )
                        chunks.append(content)
                else:
                    content = self._create_chunk_content(group_df, filename, f"{grouping_col}={group_name}")
                    chunks.append(content)
        else:
            # Sequential chunking with meaningful breaks
            for i in range(0, min(len(df), 1000), chunk_size):  # Limit to first 1000 rows
                chunk_df = df.iloc[i:i+chunk_size]
                content = self._create_chunk_content(
                    chunk_df, filename, 
                    f"Rows {i+1}-{min(i+chunk_size, len(df))}"
                )
                chunks.append(content)
        
        return chunks[:10]  # Limit to 10 chunks to avoid overload
    
    def _find_grouping_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find a suitable column for grouping data."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 50:  # Good grouping range
                return col
        
        return None
    
    def _create_chunk_content(self, chunk_df: pd.DataFrame, filename: str, chunk_info: str) -> dict:
        """Create formatted content from DataFrame chunk."""
        content_lines = [f"DATA CHUNK: {chunk_info}"]
        content_lines.append(f"HEADERS: {' | '.join(chunk_df.columns)}")
        content_lines.append("")
        
        for idx, row in chunk_df.head(50).iterrows():  # Limit to 50 rows per chunk
            row_text = " | ".join([
                str(val) if pd.notna(val) and str(val).strip() != '' else "NULL" 
                for val in row.values
            ])
            content_lines.append(f"ROW {idx}: {row_text}")
        
        if len(chunk_df) > 50:
            content_lines.append(f"... and {len(chunk_df) - 50} more rows in this chunk")
        
        return {
            "content": "\n".join(content_lines),
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "data_chunk",
                "rows": len(chunk_df),
                "chunk_info": chunk_info
            }
        }
    
    def _create_statistical_summary(self, df: pd.DataFrame, filename: str) -> Optional[dict]:
        """Create statistical summary for numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        # Calculate correlations if multiple numeric columns
        summary_lines = [f"STATISTICAL SUMMARY: {filename}"]
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            # Find highest correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation
                        high_corr_pairs.append(
                            f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}"
                        )
            
            if high_corr_pairs:
                summary_lines.append("Strong Correlations Found:")
                summary_lines.extend(high_corr_pairs[:5])  # Top 5 correlations
                summary_lines.append("")
        
        # Summary statistics
        summary_lines.append("Numerical Columns Summary:")
        for col in numeric_df.columns[:10]:  # First 10 numeric columns
            series = numeric_df[col]
            summary_lines.append(
                f"  • {col}: Mean={series.mean():.2f}, "
                f"Std={series.std():.2f}, Range=[{series.min():.2f}, {series.max():.2f}]"
            )
        
        return {
            "content": "\n".join(summary_lines),
            "metadata": {
                "filename": filename,
                "filetype": "text/csv",
                "content_type": "statistical_summary",
                "numeric_columns": len(numeric_df.columns)
            }
        }
    
    def _create_fallback_csv_content(self, path: Path, error: str) -> List[dict]:
        """Create fallback content when CSV processing fails."""
        return [{
            "content": f"CSV FILE: {path.name} - Error processing file: {error}",
            "metadata": {
                "filename": path.name,
                "filetype": "text/csv",
                "content_type": "error",
                "error": error
            }
        }]
    
    def _create_empty_csv_content(self, path: Path) -> dict:
        """Create content for empty CSV files."""
        return {
            "content": f"EMPTY CSV FILE: {path.name} - File contains no data",
            "metadata": {
                "filename": path.name,
                "filetype": "text/csv",
                "content_type": "empty_file"
            }
        }


# Global instances
_image_processor = None
_csv_processor = None

def get_image_processor():
    """Get or create image processor instance."""
    global _image_processor
    if _image_processor is None:
        _image_processor = EnhancedImageProcessor()
    return _image_processor

def get_csv_processor():
    """Get or create CSV processor instance."""
    global _csv_processor
    if _csv_processor is None:
        _csv_processor = EnhancedCSVProcessor()
    return _csv_processor

# Enhanced extraction functions to replace existing ones
def extract_csv_enhanced(path: Path) -> List[dict]:
    """Enhanced CSV extraction function."""
    processor = get_csv_processor()
    return processor.extract_csv_enhanced(path)

def process_images_enhanced(image_records: List[dict]) -> List[dict]:
    """Enhanced image processing function."""
    processor = get_image_processor()
    enhanced_content = []
    
    for img_record in image_records:
        try:
            extracted = processor.extract_image_content(
                img_record.get("image_bytes", b""), 
                img_record.get("metadata", {})
            )
            enhanced_content.extend(extracted)
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")
            # Add fallback content
            enhanced_content.append({
                "content": f"IMAGE: {img_record.get('metadata', {}).get('filename', 'unknown')} - Processing failed",
                "metadata": {
                    **img_record.get("metadata", {}),
                    "content_type": "image_error",
                    "error": str(e)
                }
            })
    
    return enhanced_content

def process_documents_enhanced(document_records: List[dict]) -> List[dict]:
    """Enhanced document processing function for mixed-content documents."""
    processor = get_document_processor()
    enhanced_content = []
    
    for doc_record in document_records:
        try:
            file_path = Path(doc_record.get("file_path", ""))
            ext = file_path.suffix.lower()
            
            if ext == '.pdf':
                extracted = extract_pdf_enhanced(file_path)
            elif ext in ['.docx', '.doc']:
                extracted = extract_docx_enhanced(file_path)
            elif ext == '.epub':
                extracted = extract_epub_enhanced(file_path)
            else:
                continue
            
            enhanced_content.extend(extracted)
            
        except Exception as e:
            logger.warning(f"Failed to process document: {e}")
            # Add fallback content
            enhanced_content.append({
                "content": f"DOCUMENT: {doc_record.get('filename', 'unknown')} - Enhanced processing failed",
                "metadata": {
                    **doc_record.get("metadata", {}),
                    "content_type": "document_error",
                    "error": str(e)
                }
            })
    
    return enhanced_content
