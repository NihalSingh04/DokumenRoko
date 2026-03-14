"""Prompt templates for the RAG system"""

# -------------------------------
# Query Rewriting Prompt
# -------------------------------

QUERY_REWRITE_PROMPT = """
Rewrite the user question into a clear standalone search query
that will help retrieve relevant documents.

User Question:
{question}

Rewritten Query:
"""

# -------------------------------
# Multi Query Generation Prompt
# -------------------------------

MULTI_QUERY_PROMPT = """
Generate 3 different search queries that could retrieve useful
information for answering the question.

Question:
{question}

Return each query on a new line.
"""

# -------------------------------
# RAG Answer Prompt
# -------------------------------

RAG_ANSWER_PROMPT = """
You are a helpful assistant answering questions using the provided context.

Use ONLY the information from the context. 
If the context does not contain the answer, say "The information is not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

# -------------------------------
# Groundedness Evaluation Prompt
# -------------------------------

GROUNDEDNESS_PROMPT = """
Check whether the answer is supported by the context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Respond with one of:
- Fully Grounded
- Partially Grounded
- Not Grounded
"""