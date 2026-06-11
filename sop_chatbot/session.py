from __future__ import annotations

from sop_chatbot.config import ChatbotConfig
from sop_chatbot.models import QAPair


class SessionContext:
    def __init__(self, config: ChatbotConfig) -> None:
        self._config = config
        self._sessions: dict[str, list[QAPair]] = {}

    def get_history(self, session_id: str) -> list[QAPair]:
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, query: str, answer: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(QAPair(query=query, answer=answer))
        if len(self._sessions[session_id]) > self._config.max_history_pairs:
            self._sessions[session_id].pop(0)

    def reset(self, session_id: str) -> None:
        self._sessions[session_id] = []
