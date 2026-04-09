"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.states.rag_state import RAGState
from src.nodes.reactnodes import RAGNodes


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm):

        self.nodes = RAGNodes(
            retriever,
            llm
        )

        self.graph = None

    # -------------------------
    # Build Graph
    # -------------------------

    def build(self):

        builder = StateGraph(
            RAGState
        )

        builder.add_node(
            "rewrite",
            self.nodes.rewrite_query
        )

        builder.add_node(
            "retriever",
            self.nodes.retrieve_docs
        )

        builder.add_node(
            "responder",
            self.nodes.generate_answer
        )

        builder.set_entry_point(
            "rewrite"
        )

        builder.add_edge(
            "rewrite",
            "retriever"
        )

        builder.add_edge(
            "retriever",
            "responder"
        )

        builder.add_edge(
            "responder",
            END
        )

        self.graph = builder.compile()

        return self.graph

    # -------------------------
    # Run Graph
    # -------------------------

    def run(
        self,
        question: str,
        history: list,
        selected_documents=None
    ) -> dict:

        if self.graph is None:

            self.build()

        # -------------------------
        # Initial State
        # -------------------------

        initial_state = RAGState(

            question=question,

            chat_history=history,

            # NEW FIELD
            selected_documents=selected_documents
        )

        result = self.graph.invoke(
            initial_state
        )

        return {

            "answer": result["answer"],

            "documents": result["retrieved_docs"],

            "history": result["chat_history"]

        }




# """Graph builder for LangGraph workflow"""

# from langgraph.graph import StateGraph, END
# from src.states.rag_state import RAGState
# from src.nodes.reactnodes import RAGNodes


# class GraphBuilder:
#     """Builds and manages the LangGraph workflow"""

#     def __init__(self, retriever, llm):
#         self.nodes = RAGNodes(retriever, llm)
#         self.graph = None

#     def build(self):

#         builder = StateGraph(RAGState)

#         builder.add_node("rewrite", self.nodes.rewrite_query)
#         builder.add_node("retriever", self.nodes.retrieve_docs)
#         builder.add_node("responder", self.nodes.generate_answer)

#         builder.set_entry_point("rewrite")

#         builder.add_edge("rewrite", "retriever")
#         builder.add_edge("retriever", "responder")
#         builder.add_edge("responder", END)

#         self.graph = builder.compile()
#         return self.graph

#     def run(self, question: str, history: list) -> dict:

#         if self.graph is None:
#             self.build()

#         initial_state = RAGState(
#             question=question,
#             chat_history=history
#         )

#         result = self.graph.invoke(initial_state)

#         return {
#             "answer": result["answer"],
#             "documents": result["retrieved_docs"],
#             "history": result["chat_history"]
#         }