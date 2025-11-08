"""
📘 PDF Loader for AskBharat
---------------------------------
Reads all PDFs from the local 'data/' folder
and extracts raw text using PyMuPDF.

Designed for teaching purposes:
- No metadata
- No OCR
- No complex error handling

Usage:
    from loaders.pdf_loader import load_all_pdfs
    texts = load_all_pdfs("data/")
"""

import os
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

# ---------------------------------
# Single PDF Extractor
# ---------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts readable text from a single PDF file using PyMuPDF."""
    text = ""
    try:
        logger.info("Opening PDF: %s", pdf_path)
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                text += page_text + "\n"
    except Exception as e:
        logger.warning("Could not read %s: %s", pdf_path, e)
    return text.strip()

# ---------------------------------
# Folder Loader
# ---------------------------------
def load_all_pdfs(folder_path: str = "data/", return_filenames: bool = False):
    """
    Loads and extracts text from all PDFs in a folder.
    Returns a list of text strings (one per PDF).
    If return_filenames=True, returns a tuple: (texts, filenames)
    """
    all_texts = []
    filenames = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning("No PDF files found in folder: %s", folder_path)
        return ([], []) if return_filenames else []

    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        logger.info("Loading PDF: %s", file_name)
        text = extract_text_from_pdf(file_path)
        if text:
            all_texts.append(text)
            filenames.append(file_name)

    logger.info("Loaded %d PDF(s) successfully", len(all_texts))
    return (all_texts, filenames) if return_filenames else all_texts

# ---------------------------------
# Optional Quick Test
# ---------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [pdf_loader] %(message)s")
    texts = load_all_pdfs("data/")
    preview = texts[0][:500] if texts else "No PDFs loaded."
    logger.info("First 500 characters of first PDF: %s", preview)
