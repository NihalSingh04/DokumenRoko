"""Query rewrite node for improving retrieval queries"""

from src.states.rag_state import RAGState
from src.prompts.rag_prompts import QUERY_REWRITE_PROMPT


class RewriteNode:
    """Node responsible for rewriting user queries"""

    def __init__(self, llm):
        """
        Args:
            llm: Language model instance
        """
        self.llm = llm

    def rewrite(self, state: RAGState) -> RAGState:
        """
        Rewrite user query to improve document retrieval
        """

        # Include last few messages from chat history
        history = "\n".join(state.chat_history[-4:]) if state.chat_history else ""

        prompt = QUERY_REWRITE_PROMPT.format(
            question=state.question,
            chat_history=history
        )

        response = self.llm.invoke(prompt)

        rewritten_query = response.content.strip()

        return RAGState(
            question=state.question,
            rewritten_query=rewritten_query,
            retrieved_docs=state.retrieved_docs,
            answer=state.answer,
            chat_history=state.chat_history  # preserve history
        )