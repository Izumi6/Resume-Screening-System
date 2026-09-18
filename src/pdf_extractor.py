"""
pdf_extractor.py
----------------
Handles extracting text content from uploaded PDF files.

Uses PyMuPDF (fitz) as the primary extraction engine because it's fast
and handles most PDF layouts well. Falls back to pdfplumber if fitz
gives us an empty result (happens sometimes with oddly formatted PDFs).
"""

import io
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF file object.
    
    Parameters
    ----------
    uploaded_file : file-like object
        The uploaded PDF file (e.g., from Streamlit's file_uploader).
    
    Returns
    -------
    str
        The extracted text, or an empty string if extraction failed.
    """
    text = ""
    
    # read the file bytes once so we can try multiple extractors
    file_bytes = uploaded_file.read()
    
    if not file_bytes:
        logger.warning("Received empty file, nothing to extract.")
        return ""
    
    # try PyMuPDF first -- it's generally faster and more reliable
    text = _extract_with_fitz(file_bytes)
    
    # if fitz didn't get anything useful, try pdfplumber as fallback
    if len(text.strip()) < 50:
        logger.info("fitz extraction was too short, trying pdfplumber...")
        fallback_text = _extract_with_pdfplumber(file_bytes)
        if len(fallback_text.strip()) > len(text.strip()):
            text = fallback_text
    
    if not text.strip():
        logger.error("Could not extract any text from the PDF.")
    
    return text


def _extract_with_fitz(file_bytes):
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text:
                pages.append(page_text)
        doc.close()
        
        return "\n".join(pages)
    
    except ImportError:
        logger.warning("PyMuPDF not installed, skipping fitz extraction.")
        return ""
    except Exception as e:
        logger.error(f"fitz extraction error: {e}")
        return ""


def _extract_with_pdfplumber(file_bytes):
    """Extract text using pdfplumber as a fallback."""
    try:
        import pdfplumber
        
        pdf = pdfplumber.open(io.BytesIO(file_bytes))
        pages = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        pdf.close()
        
        return "\n".join(pages)
    
    except ImportError:
        logger.warning("pdfplumber not installed, skipping fallback.")
        return ""
    except Exception as e:
        logger.error(f"pdfplumber extraction error: {e}")
        return ""
