"""LangGraph nodes for RAG workflow"""

from src.states.rag_state import RAGState


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Advanced retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm

    def rewrite_query(self, state: RAGState) -> RAGState:
        """
        Rewrite user query to improve retrieval
        """

        prompt = f"""
Rewrite the following user question into a clear standalone search query.

Question:
{state.question}

Rewritten query:
"""

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            rewritten_query=response.content.strip()
        )

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using advanced retriever
        """

        # Use rewritten query if available
        query = state.rewritten_query if state.rewritten_query else state.question

        docs = self.retriever.retrieve(query)

        return RAGState(
            question=state.question,
            rewritten_query=query,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents
        """

        context = "\n\n".join(
            [doc.page_content for doc in state.retrieved_docs]
        )

        prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{state.question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            rewritten_query=state.rewritten_query,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )