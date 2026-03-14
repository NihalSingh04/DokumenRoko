"""Streamlit UI for Conversational Agentic RAG"""

import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="DokumenRoko",
    page_icon="📚",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------

st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */

[data-testid="stAppViewContainer"] {
    background-color: #30364F;
    color: #F0F0DB;
}

/* ---------- TITLE ---------- */

h1 {
    text-align: center;
    color: #F0F0DB;
    font-weight: 700;
}

/* ---------- SUBHEADERS ---------- */

h3 {
    color: #F0F0DB;
}

/* ---------- CHAT BUBBLES ---------- */

[data-testid="stChatMessage"] {
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 12px;
    font-size: 16px;
}

/* USER MESSAGE */

[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: #F0F0DB;
    color: #30364F;
}

/* ASSISTANT MESSAGE */

[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background-color: #E1D9BC;
    color: #30364F;
}

/* ---------- CHAT INPUT (FIXED COLOR) ---------- */

[data-testid="stChatInput"] textarea {
    background-color: #ACBAC4 !important;
    color: white !important;
    border-radius: 12px;
    border: none;
}

/* when typing */

[data-testid="stChatInput"] textarea:focus {
    background-color: #ACBAC4 !important;
    color: white !important;
}

/* placeholder text */

[data-testid="stChatInput"] textarea::placeholder {
    color: #30364F;
}

/* ---------- FILE UPLOADER ---------- */

[data-testid="stFileUploader"] {
    background-color: #ACBAC4;
    padding: 12px;
    border-radius: 10px;
}

/* ---------- EXPANDER ---------- */

details {
    background-color: #F0F0DB;
    border-radius: 10px;
    padding: 10px;
}

/* ---------- BUTTONS ---------- */

.stButton button {
    background-color: #E1D9BC;
    color: #30364F;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

/* ---------- SOURCES SECTION ---------- */

details summary {
    color: #30364F !important;
    font-weight: 600;
}

details p, 
details li, 
details span, 
details div {
    color: #30364F !important;
}

/* ---------- PAGE SPACING ---------- */

.block-container {
    padding-top: 2rem;
}

/* ---------- CHAT CONTAINER ---------- */

[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

</style>
""", unsafe_allow_html=True)


# -------------------------
# Session State
# -------------------------

def init_session_state():

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "doc_loaded" not in st.session_state:
        st.session_state.doc_loaded = False


# -------------------------
# Build RAG
# -------------------------

def build_rag_from_docs(docs):

    llm = Config.get_llm()

    vector_store = VectorStore()
    vector_store.create_vectorstore(docs)

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )

    graph_builder.build()

    return graph_builder


# -------------------------
# Main App
# -------------------------

def main():

    init_session_state()

    st.title("📚 DokumenRoko")
    st.markdown(
        "<p style='text-align:center font-family:roboto'>Upload a document and chat with it instantly</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -------------------------
    # Layout Columns
    # -------------------------

    col1, col2 = st.columns([1,2])

    # -------------------------
    # Sidebar / Upload Panel
    # -------------------------

    with col1:

        st.subheader("📂 Upload Document")

        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf","txt"]
        )

        if uploaded_file:

            file_path = Path("temp_uploaded_file")

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )

            if uploaded_file.name.endswith(".pdf"):
                docs = processor.load_from_pdf(file_path)
            else:
                docs = processor.load_from_txt(file_path)

            chunks = processor.split_documents(docs)

            st.session_state.rag_system = build_rag_from_docs(chunks)

            st.session_state.chat_history = []
            st.session_state.messages = []

            st.session_state.doc_loaded = True

            st.success("✅ Document indexed successfully")

            st.info(f"Pages / Chunks indexed: {len(chunks)}")

        st.markdown("---")

        st.markdown("### ℹ️ Instructions")

        st.markdown("""
        1. Upload a document  
        2. Ask questions about the content  
        3. Follow-up questions will remember context
        """)

    # -------------------------
    # Chat Panel
    # -------------------------

    with col2:

        st.subheader("💬 Hola! It's me DokumenRoko")

        chat_container = st.container()

        with chat_container:

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        question = st.chat_input("Ask a question about the document...")

        if question:

            if st.session_state.rag_system is None:
                st.warning("Upload a document first.")
                return

            st.session_state.messages.append({
                "role":"user",
                "content":question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Thinking..."):

                start_time = time.time()

                result = st.session_state.rag_system.run(
                    question,
                    st.session_state.chat_history
                )

                elapsed = time.time() - start_time

                answer = result["answer"]
                documents = result["documents"]

                st.session_state.chat_history = result["history"]

            with st.chat_message("assistant"):

                st.markdown(answer)

                with st.expander("📚 Sources"):

                    if documents:

                        for i, doc in enumerate(documents[:5], start=1):

                            source = doc.metadata.get("source","unknown")

                            st.markdown(f"**Source {i}: {source}**")

                            st.write(doc.page_content[:350] + "...")

                    else:
                        st.write("No sources available")

                st.caption(f"⏱️ {elapsed:.2f} sec")

            st.session_state.messages.append({
                "role":"assistant",
                "content":answer
            })


if __name__ == "__main__":
    main()