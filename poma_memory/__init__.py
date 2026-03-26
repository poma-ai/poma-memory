"""poma-memory: Structure-preserving memory for AI agents."""

__version__ = "0.2.0"

from poma_memory.chunker import indent_light
from poma_memory.chunksets import chunks_to_chunksets, chunks_to_chunksets_optimized
from poma_memory.retrieval import expand_chunk_ids, expand_chunk_ids_deep, assemble_context
from poma_memory.normalize import normalize_for_embedding
from poma_memory.api import index, search, status

__all__ = [
    "indent_light",
    "chunks_to_chunksets",
    "chunks_to_chunksets_optimized",
    "expand_chunk_ids",
    "expand_chunk_ids_deep",
    "assemble_context",
    "normalize_for_embedding",
    "index",
    "search",
    "status",
]
