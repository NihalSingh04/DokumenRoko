from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RAGState(BaseModel):

    # -------------------------
    # Core Fields
    # -------------------------

    question: str

    rewritten_query: Optional[str] = None

    retrieved_docs: Optional[List[Document]] = None

    answer: Optional[str] = None

    chat_history: List[str] = Field(
        default_factory=list
    )

    # -------------------------
    # NEW FIELD (Solution 3)
    # -------------------------

    selected_documents: Optional[List[str]] = None

# from typing import List, Optional
# from langchain_core.documents import Document
# from pydantic import BaseModel, Field


# class RAGState(BaseModel):

#     question: str
#     rewritten_query: Optional[str] = None
#     retrieved_docs: Optional[List[Document]] = None
#     answer: Optional[str] = None

#     chat_history: List[str] = Field(default_factory=list)