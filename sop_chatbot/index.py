from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from sop_chatbot.config import ChatbotConfig
from sop_chatbot.models import Chunk, ScoredChunk


class SOPIndex:
    def __init__(self, config: ChatbotConfig, collection_name: str = "sop_chunks") -> None:
        self._config = config
        self._collection_name = collection_name
        self._model = SentenceTransformer(config.embedding_model)
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True).tolist()
        ids = [f"{c.source}_{c.chunk_index}" for c in chunks]
        metadatas = [
            {"source": c.source, "section_id": c.section_id, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], top_n: int, threshold: float) -> list[ScoredChunk]:
        count = self._collection.count()
        if count == 0:
            return []
        n_results = min(top_n, count)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        scored: list[ScoredChunk] = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - dist
            if score < threshold:
                continue
            chunk = Chunk(
                text=doc,
                source=meta["source"],
                section_id=meta["section_id"],
                chunk_index=int(meta["chunk_index"]),
            )
            scored.append(ScoredChunk(chunk=chunk, confidence_score=score))
        scored.sort(key=lambda sc: sc.confidence_score, reverse=True)
        return scored

    def delete_by_source(self, file_name: str) -> None:
        self._collection.delete(where={"source": file_name})

    def list_sources(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in result["metadatas"]:
            sources.add(meta["source"])
        return list(sources)

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            self._collection_name, metadata={"hnsw:space": "cosine"}
        )
