"""Advanced retrieval module (multi-query retrieval)"""

from typing import List
from langchain_core.documents import Document


class Retriever:
    """Handles document retrieval logic"""

    def __init__(self, retriever, llm):
        """
        Args:
            retriever: FAISS retriever
            llm: language model
        """
        self.retriever = retriever
        self.llm = llm

    def generate_queries(self, question: str) -> List[str]:
        """
        Generate multiple search queries from a single question
        """

        prompt = f"""
Generate 3 different search queries to retrieve relevant information
for answering the question.

Question:
{question}

Return each query on a new line.
"""

        response = self.llm.invoke(prompt)

        queries = response.content.split("\n")

        queries = [q.strip() for q in queries if q.strip()]

        return queries

    def retrieve(self, question: str) -> List[Document]:
        """
        Perform multi-query retrieval
        """

        queries = self.generate_queries(question)

        all_docs = []

        for q in queries:
            docs = self.retriever.invoke(q)
            all_docs.extend(docs)

        # Remove duplicate documents
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

        return unique_docs