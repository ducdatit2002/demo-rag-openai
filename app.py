import os
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from contextlib import asynccontextmanager
from config import get_api_keys, logger  # Import từ config
from pinecone_utils import initialize_vector_store  # Import từ pinecone_utils
from models import Query  # Import từ models
import uvicorn

# Lấy API keys
PINECONE_API_KEY, OPENAI_API_KEY = get_api_keys()

# Lifespan handler để khởi tạo vector_store và llm
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Khởi tạo vector_store và llm
    logger.info("Initializing vector store and LLM...")
    files = os.listdir("files")
    app.state.vector_store = initialize_vector_store(files)
    app.state.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    logger.info("Initialization complete.")
    yield
    # Shutdown: Có thể thêm logic dọn dẹp nếu cần
    logger.info("Shutting down application...")

# Khởi tạo ứng dụng FastAPI với lifespan
app = FastAPI(lifespan=lifespan)

# Endpoint truy vấn
@app.post("/query")
async def query_rag(query: Query):
    try:
        retrieved_docs = app.state.vector_store.similarity_search(query.text, k=5)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = f"Answer the query based on the following context:\n{docs_content}\n\nQuery: {query.text}\n\nAnswer:"
        answer = app.state.llm.invoke(prompt)
        return {"answer": answer.content, "sources": set([doc.metadata["source"] for doc in retrieved_docs])}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)