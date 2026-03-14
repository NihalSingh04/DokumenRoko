"""LLM initialization module"""

from src.config.config import Config


class LLMProvider:
    """Handles LLM initialization"""

    def __init__(self):
        self.llm = Config.get_llm()

    def get_llm(self):
        """Return initialized LLM"""
        return self.llm