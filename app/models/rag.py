from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    user_role: str = Field(default="customer")
    retrieval_mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    answer_mode: Literal["brief", "detailed", "step_by_step", "agent"] = "brief"
    # Last N turns passed as [{"role": "user"|"assistant", "content": "..."}]
    # TODO (Phase 3): summarize history beyond 3 turns to control token cost
    chat_history: list[dict] = Field(default_factory=list)


class Citation(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    score: float
    source_path: str


class TokenUsage(BaseModel):
    context: int    # tokens used for retrieved context
    answer: int     # tokens in the generated answer
    total: int      # context + system + answer


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieval_mode: Literal["keyword", "vector", "hybrid"]
    tokens: TokenUsage
    confidence: Literal["grounded", "cautious", "refused"]
    latency_ms: float
