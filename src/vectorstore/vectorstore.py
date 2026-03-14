"""Vector store module for document embedding and retrieval"""

from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config.config import Config


class VectorStore:
    """Manages vector store operations"""

    def __init__(self, index_path: str = "vectorstore_index"):
        """
        Initialize vector store

        Args:
            index_path: Path to save/load FAISS index
        """

        self.embedding = Config.get_embeddings()
        self.vectorstore = None
        self.retriever = None
        self.index_path = index_path

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents
        """

        self.vectorstore = FAISS.from_documents(documents, self.embedding)

        # Save index for reuse
        self.vectorstore.save_local(self.index_path)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    def load_vectorstore(self):
        """
        Load existing vectorstore if it exists
        """

        if not Path(self.index_path).exists():
            raise ValueError("Vectorstore not found. Run create_vectorstore first.")

        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embedding,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    def get_retriever(self):
        """
        Get the retriever instance
        """

        if self.retriever is None:
            raise ValueError("Vector store not initialized.")

        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents
        """

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")

        return self.vectorstore.similarity_search(query, k=k)