from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str          # Original, unmodified text of the chunk
    source: str        # File name of the originating SOP document
    section_id: str    # Section identifier (e.g., heading or chunk index)
    chunk_index: int   # Position of this chunk within the document


@dataclass
class ScoredChunk:
    chunk: Chunk
    confidence_score: float  # Cosine similarity score in [0.0, 1.0]


@dataclass
class SourceReference:
    source: str
    section_id: str


@dataclass
class QueryResult:
    answer: str
    sources: list[SourceReference]
    confidence_score: float  # Score of the highest-ranked chunk
    chunks_used: list[Chunk]


@dataclass
class QAPair:
    query: str
    answer: str


@dataclass
class IngestResult:
    file_name: str
    chunk_count: int


class IngestError(Exception): ...
class DocumentNotFoundError(Exception): ...
class QueryTimeoutError(Exception): ...
class LLMError(Exception): ...
