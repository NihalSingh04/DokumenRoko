"""Vector store module for document embedding and retrieval"""

from typing import List
from pathlib import Path
import shutil

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config.config import Config


class VectorStore:
    """Manages vector store operations with robust document filtering"""

    def __init__(self, index_path: str = "vectorstore_index"):

        self.embedding = Config.get_embeddings()

        self.vectorstore = None
        self.retriever = None

        self.index_path = Path(index_path)

    # -------------------------
    # Create or Append Vectorstore
    # -------------------------

    def create_vectorstore(self, documents: List[Document]):

        if not documents:
            print("No documents provided.")
            return

        try:

            if self.index_path.exists():

                print("Loading existing vectorstore...")

                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embedding,
                    allow_dangerous_deserialization=True
                )

                print("Adding documents to existing index...")

                self.vectorstore.add_documents(documents)

            else:

                print("Creating new vectorstore...")

                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embedding
                )

        except Exception as e:

            print("Vectorstore incompatible. Rebuilding index...")
            print("Error:", e)

            if self.index_path.exists():

                shutil.rmtree(
                    self.index_path,
                    ignore_errors=True
                )

            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding
            )

        # ensure directory exists

        self.index_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # save index

        self.vectorstore.save_local(
            str(self.index_path)
        )

        # retriever config

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 6
            }
        )

        print("Vectorstore ready.")

    # -------------------------
    # Load Existing Index
    # -------------------------

    def load_vectorstore(self):

        if not self.index_path.exists():

            raise ValueError(
                "Vectorstore not found. Upload a document first."
            )

        print("Loading vectorstore from disk...")

        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embedding,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 6
            }
        )

    # -------------------------
    # Get Retriever
    # -------------------------

    def get_retriever(self):

        if self.retriever is None:

            self.load_vectorstore()

        return self.retriever

    # -------------------------
    # Retrieve Documents
    # -------------------------

    def retrieve(
        self,
        query: str,
        k: int = 4,
        document_names=None,
    ):
        if self.vectorstore is None:
            self.load_vectorstore()

        print("QUERY:", query)

        # -------------------------
        # STEP 1 — deep retrieval
        # -------------------------

        search_depth = max(k * 8, 50)

        docs = self.vectorstore.similarity_search(
            query,
            k=search_depth,
        )

        print("Total candidates retrieved:", len(docs))

        # -------------------------
        # STEP 2 — robust normalization
        # -------------------------

        def normalize(name: str) -> str:
            if not name:
                return ""

            name = str(name)
            name = name.split("/")[-1]  # remove path if present
            name = name.replace("temp_", "")  # remove temp prefix
            name = name.strip()
            name = name.lower()
            return name

        # -------------------------
        # STEP 3 — strict filtering
        # -------------------------

        if document_names:
            print("Selected documents:", document_names)

            normalized_selected = [normalize(n) for n in document_names]
            filtered_docs = []

            for doc in docs:
                doc_name = doc.metadata.get("document_name", "")
                normalized_doc = normalize(doc_name)
                if normalized_doc in normalized_selected:
                    filtered_docs.append(doc)

            print("Filtered doc count:", len(filtered_docs))

            # STRICT behavior
            if not filtered_docs:
                print("No matching documents found in selected files.")
                return []

            return filtered_docs[:k]

        # -------------------------
        # No filtering case
        # -------------------------

        return docs[:k]
    
# """Vector store module for document embedding and retrieval"""

# from typing import List
# from pathlib import Path

# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document

# from src.config.config import Config


# class VectorStore:
#     """Manages vector store operations"""

#     def __init__(self, index_path: str = "vectorstore_index"):
#         """
#         Initialize vector store

#         Args:
#             index_path: Path to save/load FAISS index
#         """

#         self.embedding = Config.get_embeddings()
#         self.vectorstore = None
#         self.retriever = None
#         self.index_path = index_path

#     def create_vectorstore(self, documents: List[Document]):
#         """
#         Create vector store from documents
#         """

#         self.vectorstore = FAISS.from_documents(documents, self.embedding)

#         # Save index for reuse
#         self.vectorstore.save_local(self.index_path)

#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": 4}
#         )

#     def load_vectorstore(self):
#         """
#         Load existing vectorstore if it exists
#         """

#         if not Path(self.index_path).exists():
#             raise ValueError("Vectorstore not found. Run create_vectorstore first.")

#         self.vectorstore = FAISS.load_local(
#             self.index_path,
#             self.embedding,
#             allow_dangerous_deserialization=True
#         )

#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": 4}
#         )

#     def get_retriever(self):
#         """
#         Get the retriever instance
#         """

#         if self.retriever is None:
#             raise ValueError("Vector store not initialized.")

#         return self.retriever

#     def retrieve(self, query: str, k: int = 4) -> List[Document]:
#         """
#         Retrieve relevant documents
#         """

#         if self.vectorstore is None:
#             raise ValueError("Vector store not initialized.")

#         return self.vectorstore.similarity_search(query, k=k)