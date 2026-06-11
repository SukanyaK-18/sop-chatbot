from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ChatbotConfig:
    top_n: int = 10
    confidence_threshold: float = 0.1
    max_history_pairs: int = 10
    chunk_size: int = 1500
    chunk_overlap: int = 200
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"
    response_timeout_seconds: int = 30
