import pdfplumber
import pandas as pd
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from config import logger  # Import logger từ config

def extract_text_from_file(file_name: str) -> str:
    file_path = f"./files/{file_name}"
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                return "".join(page.extract_text() or "" for page in pdf.pages)
        elif file_path.endswith(".xlsx") or file_path.endswith(".csv"):
            df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
            return df.to_string()
        elif file_path.endswith(".docx"):
            doc = DocxDocument(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif file_path.endswith(".html"):
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                return soup.get_text()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""