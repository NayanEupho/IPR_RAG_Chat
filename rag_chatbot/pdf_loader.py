import fitz  # PyMuPDF
import pdfplumber
import re
import os
from typing import List, Tuple


def clean_text(text: str) -> str:
    """Cleans up extracted text by removing unwanted characters and spaces."""
    text = text.replace('\xa0', ' ')  # Replace non-breaking space
    text = re.sub(r'\n{2,}', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
    return text.strip()


def extract_text_and_tables(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extracts both text and tables from each page of the PDF.
    Returns a list of (combined_text, page_number).
    """
    pages = []
    with pdfplumber.open(pdf_path) as plumber_pdf, fitz.open(pdf_path) as pymupdf_doc:
        for i, (plumber_page, pymupdf_page) in enumerate(zip(plumber_pdf.pages, pymupdf_doc), start=1):
            text = clean_text(pymupdf_page.get_text())

            # Extract tables as strings if present
            tables = plumber_page.extract_tables()
            table_strings = []
            for table in tables:
                table_str = "\n".join(
                    ["\t".join(cell or "" for cell in row) for row in table]
                )
                table_strings.append(table_str.strip())

            # Combine plain text and table data
            full_text = text
            if table_strings:
                full_text += "\n\n[Table Data]\n" + "\n\n".join(table_strings)

            if full_text.strip():
                pages.append((full_text.strip(), i))

    return pages


def load_multiple_pdfs(pdf_paths: List[str]) -> List[Tuple[str, str, int, str]]:
    """
    Loads and cleans multiple PDFs.
    Returns a list of tuples: (cleaned_text, source_filename, page_number, section).
    """
    all_pages = []
    for path in pdf_paths:
        file_pages = extract_text_and_tables(path)
        for text, page_num in file_pages:
            section = "N/A"  # Default section placeholder
            all_pages.append((text, os.path.basename(path), page_num, section))
    return all_pages
