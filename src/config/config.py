"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # API Key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # LLM Configuration
    LLM_MODEL = "llama-3.3-70b-versatile"

    # Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize Groq LLM"""
        return ChatGroq(
            groq_api_key=cls.GROQ_API_KEY,
            model_name=cls.LLM_MODEL,
            temperature=0,
        )

    @classmethod
    def get_embeddings(cls):
        """Initialize embedding model"""
        return HuggingFaceEmbeddings(
            model_name=cls.EMBEDDING_MODEL
        )