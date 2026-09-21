"""poma-memory: Persistent context for AI coding agents."""

__version__ = "0.6.0"

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
from poma_memory.api import index, index_file, search, status
from poma_memory.metadata import MetadataNotIndexed, MetadataRulesError

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
    "index_file",
    "search",
    "status",
    "MetadataNotIndexed",
    "MetadataRulesError",
]
