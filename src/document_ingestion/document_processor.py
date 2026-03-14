"""Document processing module for loading and splitting documents"""

from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:

        loader = WebBaseLoader(url)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = url

        return docs

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:

        loader = PyPDFLoader(str(file_path))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = str(file_path)

        return docs

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:

        loader = PyPDFDirectoryLoader(str(directory))
        docs = loader.load()

        return docs

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = str(file_path)

        return docs

    def load_documents(self, sources: List[str]) -> List[Document]:

        docs: List[Document] = []

        for src in sources:

            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            else:

                path = Path(src)

                if path.is_dir():
                    docs.extend(self.load_from_pdf_dir(path))

                elif path.suffix.lower() == ".pdf":
                    docs.extend(self.load_from_pdf(path))

                elif path.suffix.lower() == ".txt":
                    docs.extend(self.load_from_txt(path))

                else:
                    raise ValueError(f"Unsupported source type: {src}")

        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:

        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[str]) -> List[Document]:

        docs = self.load_documents(sources)
        chunks = self.split_documents(docs)

        return chunks