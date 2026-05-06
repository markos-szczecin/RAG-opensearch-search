import tiktoken

from app.models.document import Chunk, DocumentMetadata


class RecursiveTextChunker:
    """
    Splits text into token-bounded chunks with configurable overlap.

    Splitting strategy (in priority order):
      1. Double newline (paragraph boundary)
      2. Single newline (line boundary)
      3. Space (word boundary)
      4. Character (last resort — avoids token-count overshoot)

    This mirrors LangChain's RecursiveCharacterTextSplitter but is tiktoken-
    aware so chunk sizes are measured in tokens, not characters.

    TODO (Phase 1):
      - Implement _split_recursive() that tries each separator in order.
      - Ensure overlap tokens come from the end of the previous chunk.
      - Emit a warning (not an error) when a single paragraph exceeds chunk_size
        (force-split at character level).
      - Add a min_chunk_tokens guard to discard tiny trailing chunks (< 20 tokens).
      - Expose a count_tokens(text) helper for the context_budgeter node to reuse.
    """

    # Separators tried in order; first match that keeps chunks within budget wins.
    _SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, text: str, metadata: DocumentMetadata) -> list[Chunk]:
        """
        Split `text` into Chunk objects.

        chunk_id format: "{doc_id}::chunk-{n:03d}"

        TODO: implement _split_recursive() and replace the stub below.
        """
        # --- STUB: one chunk per document (replace with real splitting) ---
        return [
            Chunk(
                chunk_id=f"{metadata.doc_id}::chunk-000",
                doc_id=metadata.doc_id,
                content=text,
                metadata=metadata,
            )
        ]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        TODO: recursively split text using the first separator that keeps
        segments within chunk_size tokens.  Return a flat list of raw text
        segments (without overlap).  The caller applies overlap in chunk().
        """
        raise NotImplementedError
