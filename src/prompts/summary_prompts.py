"""Prompt templates for summarization tasks"""


# ---------------------------------------
# Document Summary Prompt
# ---------------------------------------

DOCUMENT_SUMMARY_PROMPT = """
You are an expert document analyst.

Read the document below and produce a concise summary.

Document:
{document}

Provide:
1. A short summary (3-5 sentences)
2. Key topics covered
3. Important insights
"""


# ---------------------------------------
# Context Summary Prompt
# ---------------------------------------

CONTEXT_SUMMARY_PROMPT = """
Summarize the following context extracted from multiple documents.

Context:
{context}

Provide a clear and concise summary that captures the main ideas.
"""


# ---------------------------------------
# Bullet Point Summary Prompt
# ---------------------------------------

BULLET_SUMMARY_PROMPT = """
Summarize the following content into bullet points.

Content:
{content}

Bullet Summary:
"""


# ---------------------------------------
# Executive Summary Prompt
# ---------------------------------------

EXECUTIVE_SUMMARY_PROMPT = """
Create an executive summary of the following content.

Content:
{content}

The summary should:
- Highlight the main purpose
- Mention key findings
- Be concise and easy to read
"""