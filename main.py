"""Main application entry point for Agentic RAG system"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


class AgenticRAG:
    """Main Agentic RAG application"""

    def __init__(self, urls=None):

        print("🚀 Initializing Agentic RAG System...")

        # Use default URLs if none provided
        self.urls = urls or Config.DEFAULT_URLS

        # Initialize components
        self.llm = Config.get_llm()

        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        self.vector_store = VectorStore()

        # Setup vectorstore
        self._setup_vectorstore()

        # Build graph
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )

        self.graph_builder.build()

        print("✅ System initialized successfully!\n")

    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""

        index_path = Path("vectorstore_index")

        if index_path.exists():
            print("📂 Loading existing vector store...")
            self.vector_store.load_vectorstore()
            return

        print(f"📄 Processing {len(self.urls)} sources...")

        documents = self.doc_processor.process_sources(self.urls)

        print(f"📊 Created {len(documents)} document chunks")

        print("🔍 Creating vector store...")

        self.vector_store.create_vectorstore(documents)

    def ask(self, question: str) -> str:
        """Ask a question to the RAG system"""

        print(f"\n❓ Question: {question}")
        print("🤔 Processing...\n")

        result = self.graph_builder.run(question)

        answer = result["answer"]

        print(f"✅ Answer:\n{answer}\n")

        return answer

    def interactive_mode(self):
        """Run interactive question answering"""

        print("💬 Interactive Mode (type 'quit' to exit)\n")

        while True:

            question = input("Enter your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if question:
                self.ask(question)
                print("-" * 80)


def main():

    urls_file = Path("data/urls.txt")
    urls = None

    if urls_file.exists():
        with open(urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

    rag = AgenticRAG(urls=urls)

    example_questions = [
        "What is the concept of agent loop in autonomous agents?",
        "What are the key components of LLM-powered agents?",
        "Explain diffusion models for video generation."
    ]

    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80)

    for q in example_questions:
        rag.ask(q)

    print("\n" + "=" * 80)

    user_input = input("Enter interactive mode? (y/n): ")

    if user_input.lower() == "y":
        rag.interactive_mode()


if __name__ == "__main__":
    main()