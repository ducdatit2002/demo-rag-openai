import os
import unicodedata
import re
from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from config import get_api_keys, logger  # Import từ config
from file_utils import extract_text_from_file  # Import từ file_utils


PINECONE_API_KEY, OPENAI_API_KEY = get_api_keys()

def normalize_id(id_str: str) -> str:
    normalized = unicodedata.normalize('NFKD', id_str).encode('ASCII', 'ignore').decode('ASCII')
    normalized = re.sub(r'[^a-zA-Z0-9._-]', '_', normalized)
    return normalized

def initialize_vector_store(files: List[str]) -> PineconeVectorStore:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for file_path in files:
        text = extract_text_from_file(file_path)
        if text:
            docs = splitter.create_documents([text], metadatas=[{"source": file_path, "type": file_path.split(".")[-1]}])
            documents.extend(docs)

    index_name = "project-30-rag"
    embeddings = HuggingFaceEmbeddings(model_name="alikia2x/jina-embedding-v3-m2v-1024", model_kwargs={"device": "cpu"})

    # Khởi tạo Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Kiểm tra và tạo chỉ mục nếu chưa tồn tại
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024,  # Phù hợp với mô hình nhúng jina-embedding-v3-m2v-1024
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

    # Kiểm tra xem chỉ mục đã có dữ liệu chưa
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) == 0:  # Nếu chỉ mục rỗng
        logger.info("Indexing new documents...")
        ids = [normalize_id(f"{doc.metadata['source']}#chunk{num}") for num, doc in enumerate(documents)]
        vector_store.add_documents(documents=documents, ids=ids)
    else:
        logger.info("Using existing index with data.")

    return vector_store
