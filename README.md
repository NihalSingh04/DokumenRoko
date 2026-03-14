
# DokumenRoko📚

DokumenRoko is an Agentic Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask intelligent questions about them. It is designed for students, researchers, and professionals who want to quickly extract insights from large documents using AI.

## Application Features

#### Upload Documents

Users can upload documents (PDF/text) which are processed and converted into embeddings for semantic search.

| Input | Type | Description |
|------|------|-------------|
| document | file | Document uploaded by the user |

---

#### Ask Question

Users can ask questions about the uploaded documents. The system retrieves relevant chunks and generates answers using an LLM.

| Parameter | Type | Description |
|-----------|------|-------------|
| question | string | User query about the document |

---

#### Retrieve Context

The system performs vector similarity search to retrieve the most relevant document chunks.

| Parameter | Type | Description |
|-----------|------|-------------|
| query | string | Processed query used for retrieval |

---

#### Generate Answer

The LLM generates a response based on retrieved context and returns the answer along with the sources used.

| Parameter | Type | Description |
|-----------|------|-------------|
| context | text | Retrieved document chunks used to generate the answer |# DokumenRoko

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://dokumenroko.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
## Demo

Try the live application here:

👉 https://dokumenroko-nihalsingh04.streamlit.app/

Upload a document and start asking questions.  
The system retrieves relevant content and generates answers using an Agentic RAG pipeline.

