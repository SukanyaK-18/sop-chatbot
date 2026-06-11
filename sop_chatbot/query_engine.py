from __future__ import annotations

import groq as groq_sdk

from sop_chatbot.config import ChatbotConfig
from sop_chatbot.index import SOPIndex
from sop_chatbot.models import (
    LLMError,
    QueryResult,
    QueryTimeoutError,
    SourceReference,
)
from sop_chatbot.session import SessionContext


class QueryEngine:
    def __init__(
        self,
        index: SOPIndex,
        session: SessionContext,
        config: ChatbotConfig,
        llm_client=None,
    ) -> None:
        self._index = index
        self._session = session
        self._config = config
        self._llm_client = llm_client  # None means lazy-init on first query

    def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = groq_sdk.Groq()
        return self._llm_client

    def query(self, query_text: str, session_id: str = "default") -> QueryResult:
        history = self._session.get_history(session_id)

        query_embedding = self._index._model.encode(query_text, convert_to_numpy=True).tolist()

        scored_chunks = self._index.search(
            query_embedding,
            top_n=self._config.top_n,
            threshold=self._config.confidence_threshold,
        )

        if not scored_chunks:
            return QueryResult(
                answer="I could not find any relevant information in the ingested SOPs.",
                sources=[],
                confidence_score=0.0,
                chunks_used=[],
            )

        context_text = "\n\n---\n\n".join(sc.chunk.text for sc in scored_chunks)

        history_text = "\n".join(
            f"Q: {pair.query}\nA: {pair.answer}" for pair in history
        )

        prompt = (
            "You are an assistant that answers questions based solely on the provided SOP documents.\n"
            "Do not use any knowledge outside of the provided context.\n"
            "Give a complete and detailed answer. Include all steps, sub-steps, conditions, and details from the context that relate to the question.\n"
            "Do not summarize or abbreviate. If there are numbered steps, list them all with their full details.\n"
            "Stay focused on the topic asked — but within that topic, be exhaustive.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Question: {query_text}\n"
            "Answer (complete and detailed):"
        )

        try:
            response = self._get_llm_client().chat.completions.create(
                model=self._config.llm_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._config.response_timeout_seconds,
            )
            answer = response.choices[0].message.content
        except groq_sdk.APITimeoutError as exc:
            raise QueryTimeoutError(str(exc)) from exc
        except groq_sdk.APIError as exc:
            raise LLMError(str(exc)) from exc

        seen: set[tuple[str, str]] = set()
        sources: list[SourceReference] = []
        for sc in scored_chunks:
            key = (sc.chunk.source, sc.chunk.section_id)
            if key not in seen:
                seen.add(key)
                sources.append(SourceReference(source=sc.chunk.source, section_id=sc.chunk.section_id))

        confidence_score = max(sc.confidence_score for sc in scored_chunks)
        chunks_used = [sc.chunk for sc in scored_chunks]

        self._session.append(session_id, query_text, answer)

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence_score=confidence_score,
            chunks_used=chunks_used,
        )
