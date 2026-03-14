"""RAG evaluation module"""

from typing import List
from langchain_core.documents import Document


class RAGEvaluator:
    """Evaluates RAG outputs for quality"""

    def __init__(self, llm):
        """
        Args:
            llm: Language model used for evaluation
        """
        self.llm = llm

    def evaluate_context_relevance(self, question: str, docs: List[Document]) -> str:
        """
        Evaluate if retrieved documents are relevant
        """

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Evaluate whether the retrieved context is relevant to the question.

Question:
{question}

Context:
{context}

Respond with one of the following:
- Highly Relevant
- Partially Relevant
- Not Relevant
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    def evaluate_groundedness(self, question: str, answer: str, docs: List[Document]) -> str:
        """
        Check if the answer is supported by the retrieved context
        """

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Check whether the answer is grounded in the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Respond with:
- Fully Grounded
- Partially Grounded
- Not Grounded
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    def evaluate_answer_length(self, answer: str) -> str:
        """
        Simple heuristic evaluation
        """

        length = len(answer.split())

        if length < 10:
            return "Too Short"
        elif length < 50:
            return "Good Length"
        else:
            return "Long Answer"

    def evaluate(self, question: str, answer: str, docs: List[Document]) -> dict:
        """
        Run full evaluation pipeline
        """

        return {
            "context_relevance": self.evaluate_context_relevance(question, docs),
            "groundedness": self.evaluate_groundedness(question, answer, docs),
            "answer_length": self.evaluate_answer_length(answer)
        }