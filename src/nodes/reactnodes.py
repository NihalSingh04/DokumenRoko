"""LangGraph nodes for RAG workflow"""

from typing import List
from src.states.rag_state import RAGState
from langchain_core.documents import Document


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):

        self.retriever = retriever
        self.llm = llm

    # -------------------------
    # Query Rewrite Node
    # -------------------------

    def rewrite_query(
        self,
        state: RAGState
    ) -> RAGState:

        history = "\n".join(
            state.chat_history[-4:]
        )

        prompt = f"""
Rewrite the user question for document retrieval.

Conversation History:
{history}

Current Question:
{state.question}

Rewritten Query:
"""

        response = self.llm.invoke(
            prompt
        )

        rewritten_query = response.content.strip()

        return RAGState(

            question=state.question,

            rewritten_query=rewritten_query,

            chat_history=state.chat_history,

            selected_documents=state.selected_documents
        )

    # -------------------------
    # Retrieval Node
    # -------------------------

    def retrieve_docs(
        self,
        state: RAGState
    ) -> RAGState:

        query = (
            state.rewritten_query
            or state.question
        )

        selected_docs = (
            state.selected_documents
        )

        print("QUERY:", query)
        print("SELECTED DOCS:", selected_docs)

        docs: List[Document] = (
            self.retriever.retrieve(
                query,
                document_names=selected_docs
            )
        )

        print("RETRIEVED DOC COUNT:", len(docs))


        return RAGState(

            question=state.question,

            rewritten_query=query,

            retrieved_docs=docs,

            chat_history=state.chat_history,

            selected_documents=selected_docs
        )

    # -------------------------
    # Generate Answer Node
    # -------------------------

    def generate_answer(
        self,
        state: RAGState,
    ) -> RAGState:
        docs = state.retrieved_docs or []

        if not docs:
            return RAGState(
                question=state.question,
                retrieved_docs=[],
                answer="No relevant information found in the selected documents.",
                chat_history=state.chat_history,
                selected_documents=state.selected_documents,
            )

        # -------------------------
        # Build context
        # -------------------------

        context = "\n\n".join(doc.page_content for doc in docs)

        history = "\n".join(state.chat_history[-4:])

        question = state.question

        # -------------------------
        # Detect length instruction
        # -------------------------

        length_instruction = ""

        if "lines" in question.lower():

            length_instruction = """
IMPORTANT:
- If the user asks for a specific number of lines, you MUST produce that many lines.
- Do NOT shorten the explanation.
- Provide a detailed explanation.
"""

        if "brief" in question.lower():

            length_instruction = """
IMPORTANT:
- Provide a short concise answer.
"""

        if "detail" in question.lower():

            length_instruction = """
IMPORTANT:
- Provide a detailed explanation with examples.
"""

        # -------------------------
        # Final prompt
        # -------------------------

        prompt = f"""
You are a helpful AI assistant answering questions about documents.

Use ONLY the provided context.

Conversation History:
{history}

Context:
{context}

User Question:
{question}

{length_instruction}

Rules:

- Follow the user's instruction about length.
- If the user asks for 50 lines, produce approximately 50 lines.
- Be informative and structured.
- Do not say you lack context if context exists.
- Do not hallucinate outside the documents.

Answer:
"""

        response = self.llm.invoke(prompt)
        answer = response.content

        updated_history = state.chat_history + [
            f"User: {question}",
            f"Assistant: {answer}",
        ]

        return RAGState(
            question=question,
            rewritten_query=state.rewritten_query,
            retrieved_docs=docs,
            answer=answer,
            chat_history=updated_history,
            selected_documents=state.selected_documents,
        )


# """LangGraph nodes for RAG workflow"""

# from typing import List
# from src.states.rag_state import RAGState
# from langchain_core.documents import Document


# class RAGNodes:
#     """Contains node functions for RAG workflow"""

#     def __init__(self, retriever, llm):
#         self.retriever = retriever
#         self.llm = llm

#     # -------------------------
#     # Query Rewrite Node
#     # -------------------------

#     def rewrite_query(self, state: RAGState) -> RAGState:
#         """Rewrite query to improve retrieval"""

#         history = "\n".join(state.chat_history[-4:])

#         prompt = f"""
# Rewrite the user question for document retrieval.

# Conversation History:
# {history}

# Current Question:
# {state.question}

# Rewritten Query:
# """

#         response = self.llm.invoke(prompt)

#         rewritten_query = response.content.strip()

#         return RAGState(
#             question=state.question,
#             rewritten_query=rewritten_query,
#             chat_history=state.chat_history
#         )

#     # -------------------------
#     # Retrieval Node
#     # -------------------------

#     def retrieve_docs(self, state: RAGState) -> RAGState:
#         """Retrieve documents from vector store"""

#         query = state.rewritten_query or state.question

#         docs: List[Document] = self.retriever.invoke(query)

#         return RAGState(
#             question=state.question,
#             rewritten_query=query,
#             retrieved_docs=docs,
#             chat_history=state.chat_history
#         )

#     # -------------------------
#     # Generate Answer Node
#     # -------------------------

#     def generate_answer(self, state: RAGState) -> RAGState:
#         """Generate answer using retrieved documents"""

#         print("CHAT HISTORY:", state.chat_history)
#         print("GENERATE ANSWER NODE CALLED")

#         docs = state.retrieved_docs or []

#         context = "\n\n".join([doc.page_content for doc in docs])
#         history = "\n".join(state.chat_history[-4:])

#         prompt = f"""
# You are a helpful AI assistant answering questions about documents.

# Conversation History:
# {history}

# Context:
# {context}

# Current Question:
# {state.question}

# Answer clearly using the context and conversation.
# """

#         response = self.llm.invoke(prompt)
#         answer = response.content

#         updated_history = state.chat_history + [
#             f"User: {state.question}",
#             f"Assistant: {answer}"
#         ]

#         return RAGState(
#             question=state.question,
#             retrieved_docs=docs,
#             answer=answer,
#             chat_history=updated_history
#         )
       

        