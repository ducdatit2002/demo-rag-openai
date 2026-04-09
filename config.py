import os
from getpass import getpass
from dotenv import load_dotenv
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thiết lập API key
load_dotenv()

def get_api_keys():
    pinecone_api_key = os.environ.get("PINECONE_API_KEY") or getpass("Enter Pinecone API key: ")
    openai_api_key = os.environ.get("OPENAI_API_KEY") or getpass("Enter OpenAI API key: ")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return pinecone_api_key, openai_api_key
