"""poma-memory: Persistent context for AI coding agents."""

__version__ = "0.3.3"

# Re-export chunking primitives from poma-primecut-nano
from poma_primecut_nano import (
    chunk,
    chunks_to_chunksets,
    chunks_to_chunksets_optimized,
    expand_chunk_ids,
    expand_chunk_ids_deep,
    assemble_context,
    normalize_for_embedding,
)

# poma-memory's own API
from poma_memory.api import index, search, status

__all__ = [
    # From poma-primecut-nano (re-exported for convenience)
    "chunk",
    "chunks_to_chunksets",
    "chunks_to_chunksets_optimized",
    "expand_chunk_ids",
    "expand_chunk_ids_deep",
    "assemble_context",
    "normalize_for_embedding",
    # poma-memory API
    "index",
    "search",
    "status",
]
