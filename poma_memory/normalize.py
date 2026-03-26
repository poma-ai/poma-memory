"""Text normalization for embedding-ready text."""
from __future__ import annotations

import html
import re
import unicodedata


def normalize_for_embedding(text: str) -> str:
    """Produce embedding-ready text from chunk/chunkset contents.

    Matches poma-core's normalize_for_embedding():
    HTML strip (table-aware) -> NFKD -> whitespace collapse -> thousand-separator removal.
    """
    if not text:
        return text
    # Strip HTML (table-aware)
    if "<table" in text.lower():
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", "", text, flags=re.I)
        text = re.sub(r"</t[dh]>\s*", "\t", text, flags=re.I)
        text = re.sub(r"</tr>\s*", "\n", text, flags=re.I)
        text = re.sub(r"<tr[^>]*>\s*", "", text, flags=re.I)
        text = re.sub(r"<t[dh][^>]*>\s*", "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
    else:
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
    # Unicode NFKD
    text = unicodedata.normalize("NFKD", text)
    # Whitespace normalization
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Thousand separator removal (improves numeric embedding quality)
    for _ in range(5):
        text = re.sub(r"(\d),(\d{3})(?=[,.\s\)\]\}]|$)", r"\1\2", text)
    for _ in range(5):
        text = re.sub(r"(\d)\.(\d{3})(?=[.,\s\)\]\}]|$)", r"\1\2", text)
    return text.strip()
