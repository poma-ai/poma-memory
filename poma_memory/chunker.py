"""Heuristic markdown chunker — structure-preserving, zero ML.

Extracted from poma-core/indentation_light.py. This is the canonical
upstream location for the heuristic-only chunking path.

The full POMA pipeline (LLM-assisted indentation, depth shortcuts,
parallel/sequential modes) remains in poma-core. This module contains
only the deterministic, regex-based heuristic path.
"""

import logging
import re

import tiktoken

# --------------- configuration ---------------

_TOKEN_LIMIT = 500
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Lazy-init token chunker (avoid import-time dep on chonkie if not needed)
_TOKEN_CHUNKER = None


def _get_token_chunker():
    global _TOKEN_CHUNKER
    if _TOKEN_CHUNKER is None:
        try:
            from chonkie import TokenChunker
            _TOKEN_CHUNKER = TokenChunker(
                tokenizer=_ENCODING, chunk_size=_TOKEN_LIMIT, chunk_overlap=0
            )
        except ImportError:
            _TOKEN_CHUNKER = None
    return _TOKEN_CHUNKER


# --------------- regex patterns ---------------

# Atomic placeholders (must NEVER be split)
_PLACEHOLDER_RE = re.compile(r"【[^】]+】")
_PLACEHOLDER_LINE_RE = re.compile(r"^\s*(【[^】]+】)\s*$")
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\[\.\.\.\]\s*$")

# Markdown structure
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_THEMATIC_BREAK_RE = re.compile(r"^[ \t]{0,3}((\*\s*){3,}|(-\s*){3,}|(_\s*){3,})\s*$")
_UL_ITEM_RE = re.compile(r"^([ \t]{0,})([-*+•])[ \t]+(.*)$")
_OL_ITEM_RE = re.compile(r"^([ \t]{0,})(\d+)[.)][ \t]+(.*)$")
_ALPHA_ITEM_RE = re.compile(r"^([ \t]*)([a-zA-ZÀ-ÿ])[.)][ \t]+(.*)$")
_ROMAN_ITEM_RE = re.compile(r"^([ \t]*)([ivxlcdm]+)[.)][ \t]+(.*)$", re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^[ \t]*(?P<tick>`{3,}|~{3,})(?P<info>.*)$")
_CAPTIONISH_RE = re.compile(r"^\s*([*_]{1,2}).+?\1\s*$")

# Standalone links
_LINK_LINE_RE = re.compile(r"^\s*(?:https?://\S+|\[[^\]]+\]\([^)]+\))\s*$")

# Table detection
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_GRID_TABLE_RE = re.compile(r"^\s*\+[-=+]{3,}\+\s*$")
_HTML_TABLE_OPEN_RE = re.compile(r"(?i)^\s*<table\b")
_HTML_TABLE_CLOSE_RE = re.compile(r"(?i)</table>\s*$")

# Indented code blocks
_INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t)")

# Sentence splitting
_SENT_SPLIT_RE = re.compile(
    r"([.!?。！？]+)\s+(?=(?:[A-ZÀ-ÝÄÖÜÇÑ0-9\"\'\"\"\'\'\(\[\{]|[-*•]))"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"(?s)(.*?)([;:,，、；：]+)(\s+|$)")

# Common abbreviations that should NOT trigger sentence splits
_ABBR = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "etc.", "e.g.", "i.e.",
    "no.", "fig.", "eq.", "sec.", "ch.", "vol.", "vs.", "inc.", "ltd.", "corp.",
    "st.", "ave.", "blvd.", "rd.", "apt.", "tel.", "fax.", "ref.", "pp.", "ed.",
    "rev.", "gen.", "col.", "lt.", "sgt.", "gov.", "sen.", "rep.", "hon.",
    "cal.", "civ.", "colo.", "stat.", "ann.", "u.s.", "v.", "cf.", "seq.",
    "para.", "art.", "amend.", "reg.", "supp.", "app.", "op.", "cit.",
}


# --------------- public API ---------------


def indent_light(text: str, *, extract_title: bool = True) -> str:
    """Parse markdown into depth-annotated arrow-prefixed lines.

    Each output line is prefixed with arrow characters (→) indicating
    its depth in the document hierarchy. Depth 0 = document title,
    depth 1+ = sections/subsections/body.

    Args:
        text: Markdown text to chunk.
        extract_title: If True (default), extract the first heading as
            depth-0 title. Set to False for incremental chunks where
            the caller injects heading context.

    Returns:
        Arrow-prefixed text. Example::

            Title
            →Section heading
            →→Body text sentence one.
            →→Body text sentence two.
            →→→- List item
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    events = _scan_events(text)

    if extract_title:
        maybe_title, events = _extract_title(events)
    else:
        maybe_title = None

    out: list[str] = []

    def emit(depth: int, s: str) -> None:
        s = s.rstrip("\n")
        if s:
            out.append(("→" * max(0, depth)) + s)

    # State
    current_heading_depth = 0
    colon_parent_depth: int | None = None
    pseudo_parent_depth: int | None = None

    last_anchor_depth = 0
    after_placeholder_return_depth: int | None = None
    last_placeholder_depth: int | None = None
    last_was_placeholder = False

    if extract_title:
        emit(0, maybe_title or "")
        current_heading_depth = 0
        last_anchor_depth = 0

    def body_depth() -> int:
        d = max(1, current_heading_depth + 1)
        if pseudo_parent_depth is not None:
            d = max(d, pseudo_parent_depth + 1)
        if colon_parent_depth is not None:
            d = max(d, colon_parent_depth + 1)
        if after_placeholder_return_depth is not None:
            d = max(d, after_placeholder_return_depth)
        return d

    for ev in events:
        kind = ev[0]
        line = ev[1]

        if kind == "blank":
            colon_parent_depth = None
            pseudo_parent_depth = None
            after_placeholder_return_depth = None
            last_was_placeholder = False
            continue

        if kind == "thematic":
            colon_parent_depth = None
            pseudo_parent_depth = None
            after_placeholder_return_depth = None
            last_was_placeholder = False
            continue

        if kind == "code_fence":
            d = body_depth() + 1
            for piece, delta in _enforce_limit_anchor(line):
                emit(d + delta, piece)
            last_anchor_depth = d
            last_was_placeholder = False
            after_placeholder_return_depth = None
            continue

        if kind == "table_block":
            d = body_depth() + 1
            for piece, delta in _enforce_limit_anchor(line):
                emit(d + delta, piece)
            last_anchor_depth = d
            last_was_placeholder = False
            after_placeholder_return_depth = None
            continue

        if kind == "heading":
            colon_parent_depth = None
            pseudo_parent_depth = None
            after_placeholder_return_depth = None
            last_was_placeholder = False

            md_level = ev[2]
            if not md_level:
                raise ValueError("Heading level missing")
            depth = max(1, md_level - 1)
            current_heading_depth = depth
            emit(depth, line)
            last_anchor_depth = depth
            continue

        if kind == "pseudo_heading":
            d = body_depth()
            pseudo_parent_depth = d
            colon_parent_depth = None
            after_placeholder_return_depth = None
            last_was_placeholder = False
            emit(d, line)
            last_anchor_depth = d
            continue

        if kind == "placeholder":
            parent = last_anchor_depth
            last_placeholder_depth = parent + 1
            emit(last_placeholder_depth, line)
            last_was_placeholder = True
            after_placeholder_return_depth = parent
            continue

        if kind == "line":
            if (
                last_was_placeholder
                and last_placeholder_depth is not None
                and _CAPTIONISH_RE.match(line.strip())
            ):
                for piece, delta in _enforce_limit_anchor(line):
                    emit(last_placeholder_depth + 1 + delta, piece)
                last_was_placeholder = False
                continue

            if last_was_placeholder:
                last_was_placeholder = False

            # Lists
            mul = _UL_ITEM_RE.match(line)
            mol = _OL_ITEM_RE.match(line)
            mal = _ALPHA_ITEM_RE.match(line)
            mrl = _ROMAN_ITEM_RE.match(line)
            if mul or mol or mal or mrl:
                base = body_depth() + (0 if colon_parent_depth is not None else 1)
                if mul:
                    indent_raw, marker, rest = mul.group(1), mul.group(2), mul.group(3)
                    nest = _list_nesting(_expand_indent(indent_raw))
                    d = base + nest
                    _emit_list_item(d, f"{marker} {rest}", emit)
                elif mol:
                    indent_raw, num, rest = mol.group(1), mol.group(2), mol.group(3)
                    nest = _list_nesting(_expand_indent(indent_raw))
                    d = base + nest
                    _emit_list_item(d, f"{num}. {rest}", emit)
                elif mal:
                    indent_raw, letter, rest = mal.group(1), mal.group(2), mal.group(3)
                    nest = _list_nesting(_expand_indent(indent_raw))
                    d = base + nest
                    _emit_list_item(d, f"{letter}. {rest}", emit)
                else:
                    indent_raw, roman, rest = mrl.group(1), mrl.group(2), mrl.group(3)  # type: ignore
                    nest = _list_nesting(_expand_indent(indent_raw))
                    d = base + nest
                    _emit_list_item(d, f"{roman}. {rest}", emit)

                last_anchor_depth = base
                colon_parent_depth = None
                after_placeholder_return_depth = None
                continue

            # Standalone link
            if _LINK_LINE_RE.match(line):
                d = body_depth()
                emit(d, line)
                last_anchor_depth = d
                colon_parent_depth = None
                after_placeholder_return_depth = None
                continue

            # Normal text: sentence split
            d = body_depth()
            sentences = _split_sentences_linewise(line)
            if not sentences:
                for piece, delta in _enforce_limit_anchor(line):
                    emit(d + delta, piece)
            else:
                for s in sentences:
                    for piece, delta in _enforce_limit_anchor(s):
                        emit(d + delta, piece)
                    if s.rstrip().endswith((":", "：")):
                        colon_parent_depth = d
                    else:
                        if colon_parent_depth == d:
                            colon_parent_depth = None

            last_anchor_depth = d
            after_placeholder_return_depth = None
            continue

    return "\n".join(out)


# --------------- scanning ---------------


def _scan_events(text: str) -> list[tuple[str, str, int | None]]:
    """Scan the text for events."""
    lines = text.split("\n")
    events: list[tuple[str, str, int | None]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            events.append(("blank", "", None))
            i += 1
            continue

        if _THEMATIC_BREAK_RE.match(line):
            events.append(("thematic", "", None))
            i += 1
            continue

        m_open = _FENCE_OPEN_RE.match(line)
        if m_open:
            tick = m_open.group("tick")
            fence_char = tick[0]
            fence_len = len(tick)
            buf = [line]
            i += 1
            while i < len(lines):
                ln = lines[i]
                buf.append(ln)
                if re.match(
                    rf"^[ \t]*{re.escape(fence_char)}{{{fence_len},}}[ \t]*$",
                    ln.strip(),
                ):
                    i += 1
                    break
                i += 1
            events.append(("code_fence", "\n".join(buf), None))
            continue

        mh = _ATX_HEADING_RE.match(line)
        if mh:
            level = len(mh.group(1))
            heading_text = mh.group(2)
            events.append(("heading", heading_text, level))
            i += 1
            continue

        if _PLACEHOLDER_LINE_RE.match(line) or _ELLIPSIS_LINE_RE.match(line):
            events.append(("placeholder", line.strip(), None))
            i += 1
            continue

        # HTML <table> block
        if _HTML_TABLE_OPEN_RE.match(line):
            buf = [line]
            i += 1
            if not _HTML_TABLE_CLOSE_RE.search(line):
                while i < len(lines):
                    ln = lines[i]
                    buf.append(ln)
                    i += 1
                    if _HTML_TABLE_CLOSE_RE.search(ln):
                        break
            events.append(("table_block", "\n".join(buf), None))
            continue

        # Markdown pipe/grid tables
        if _MD_TABLE_ROW_RE.match(line) or _GRID_TABLE_RE.match(line):
            buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip():
                ln = lines[i]
                if (
                    _MD_TABLE_ROW_RE.match(ln)
                    or _MD_TABLE_SEP_RE.match(ln)
                    or _GRID_TABLE_RE.match(ln)
                ):
                    buf.append(ln)
                    i += 1
                    continue
                break
            events.append(("table_block", "\n".join(buf), None))
            continue

        # Indented code blocks
        if _INDENTED_CODE_RE.match(line):
            buf = [line]
            i += 1
            while i < len(lines) and (
                _INDENTED_CODE_RE.match(lines[i]) or not lines[i].strip()
            ):
                buf.append(lines[i])
                i += 1
            events.append(("code_fence", "\n".join(buf), None))
            continue

        # Pseudo heading
        if _looks_like_pseudo_heading_line(lines, i):
            events.append(("pseudo_heading", line.strip(), None))
            i += 1
            continue

        events.append(("line", line, None))
        i += 1

    return events


def _extract_title(
    events: list[tuple[str, str, int | None]],
) -> tuple[str | None, list[tuple[str, str, int | None]]]:
    """Extract the title from the events."""
    for idx, ev in enumerate(events):
        if ev[0] == "heading":
            return ev[1].strip(), events[:idx] + events[idx + 1:]
        if ev[0] == "line" and ev[1].strip():
            sents = _split_sentences_linewise(ev[1].strip())
            if sents:
                title = sents[0].strip()
                rest = sents[1].strip() if len(sents) > 1 else ""
                new_events = (
                    events[:idx]
                    + ([("line", rest, None)] if rest else [])
                    + events[idx + 1:]
                )
                return title, new_events
            return ev[1].strip(), events[:idx] + events[idx + 1:]
    logging.warning("(indent_light) No title found in document.")
    return None, events


def _looks_like_pseudo_heading_line(lines: list[str], i: int) -> bool:
    """Check if a line looks like a pseudo heading."""
    s = lines[i].strip()
    if not s or len(s) > 120:
        return False
    if _PLACEHOLDER_RE.search(s):
        return False
    if _UL_ITEM_RE.match(s) or _OL_ITEM_RE.match(s):
        return False
    if _ALPHA_ITEM_RE.match(s) or _ROMAN_ITEM_RE.match(s):
        return False
    if s.endswith((".", "!", "?", "。", "！", "？")):
        return False
    before_blank = i - 1 >= 0 and not lines[i - 1].strip()
    after_blank = i + 1 < len(lines) and not lines[i + 1].strip()
    return before_blank and after_blank


# --------------- splitting + limit enforcement ---------------


def _split_sentences_linewise(line: str) -> list[str]:
    """Split a line into sentences using heuristic punctuation rules."""
    s = line.strip("\n")
    parts: list[str] = []
    start = 0
    for m in _SENT_SPLIT_RE.finditer(s):
        prefix = s[start: m.start(1)].rstrip()
        words = prefix.split()
        last_word = words[-1].lower() if words else ""
        last_with_punct = last_word + m.group(1)[0] if last_word else ""
        if last_with_punct in _ABBR:
            continue
        end = m.end(1)
        parts.append(s[start:end] + " ")
        start = m.end()
    tail = s[start:]
    if tail:
        parts.append(tail)
    return [p for p in parts if p.strip()]


def _enforce_limit_anchor(text: str) -> list[tuple[str, int]]:
    """Split text that exceeds token limit. Returns [(piece, delta_depth)]."""
    if _toklen(text) <= _TOKEN_LIMIT:
        return [(text, 0)]

    if _PLACEHOLDER_RE.search(text):
        return _split_placeholder_safe_anchor(text)

    clauses = _split_clauses(text)
    if len(clauses) > 1 and all(_toklen(c) <= _TOKEN_LIMIT for c in clauses):
        return [(clauses[0], 0)] + [(c, 1) for c in clauses[1:]]

    # Last resort: token chunk
    chunker = _get_token_chunker()
    if chunker:
        chunks = [c.text for c in chunker.chunk(text)]
        if len(chunks) > 1:
            return [(chunks[0], 0)] + [(c, 1) for c in chunks[1:]]

    # Fallback without chonkie: simple split by token limit
    return _simple_token_split(text)


def _simple_token_split(text: str) -> list[tuple[str, int]]:
    """Fallback token splitter when chonkie is not available."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if _toklen(" ".join(current)) > _TOKEN_LIMIT:
            if len(current) > 1:
                current.pop()
                pieces.append(" ".join(current))
                current = [word]
            else:
                pieces.append(" ".join(current))
                current = []
    if current:
        pieces.append(" ".join(current))
    if not pieces:
        return [(text, 0)]
    return [(pieces[0], 0)] + [(p, 1) for p in pieces[1:]]


def _split_clauses(text: str) -> list[str]:
    """Split the text into clauses."""
    s = text
    out: list[str] = []
    pos = 0
    while pos < len(s):
        m = _CLAUSE_BOUNDARY_RE.match(s, pos)
        if not m:
            tail = s[pos:]
            if tail:
                out.append(tail)
            break
        body, punct, ws = m.group(1), m.group(2), m.group(3)
        seg = body + punct + ws
        if seg:
            out.append(seg)
        pos = m.end()
        if pos >= len(s):
            break
    return [x for x in out if x]


def _split_placeholder_safe_anchor(text: str) -> list[tuple[str, int]]:
    """Split the text into placeholder-safe anchor."""
    parts: list[tuple[str, bool]] = []
    last = 0
    for m in _PLACEHOLDER_RE.finditer(text):
        if m.start() > last:
            parts.append((text[last: m.start()], False))
        parts.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))

    buf = ""
    pieces: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        if _toklen(buf) <= _TOKEN_LIMIT:
            pieces.append(buf)
        else:
            chunker = _get_token_chunker()
            if chunker:
                pieces.extend([c.text for c in chunker.chunk(buf)])
            else:
                # Fallback
                for p, _ in _simple_token_split(buf):
                    pieces.append(p)
        buf = ""

    for seg, is_ph in parts:
        if is_ph:
            flush_buf()
            pieces.append(seg)
            continue
        if not seg:
            continue
        if _toklen(seg) <= _TOKEN_LIMIT:
            if _toklen(buf + seg) <= _TOKEN_LIMIT:
                buf += seg
            else:
                flush_buf()
                buf = seg
        else:
            chunker = _get_token_chunker()
            if chunker:
                seg_chunks = [c.text for c in chunker.chunk(seg)]
            else:
                seg_chunks = [p for p, _ in _simple_token_split(seg)]
            for sc in seg_chunks:
                if _toklen(buf + sc) <= _TOKEN_LIMIT:
                    buf += sc
                else:
                    flush_buf()
                    buf = sc
    flush_buf()

    if len(pieces) <= 1:
        return [(pieces[0] if pieces else text, 0)]
    return [(pieces[0], 0)] + [(p, 1) for p in pieces[1:]]


def _emit_list_item(depth: int, item_text: str, emit) -> None:
    """Emit a list item."""
    parts = item_text.split(" ", 1)
    if len(parts) == 1:
        for piece, delta in _enforce_limit_anchor(item_text):
            emit(depth + delta, piece)
        return
    marker, rest = parts[0], parts[1]
    sents = _split_sentences_linewise(rest)
    if not sents:
        for piece, delta in _enforce_limit_anchor(item_text):
            emit(depth + delta, piece)
        return

    first = marker + " " + sents[0]
    for piece, delta in _enforce_limit_anchor(first):
        emit(depth + delta, piece)

    for s in sents[1:]:
        for piece, delta in _enforce_limit_anchor(s):
            emit(depth + delta, piece)


def _expand_indent(s: str) -> int:
    """Expand the indent."""
    return len(s.replace("\t", "    "))


def _list_nesting(indent_spaces: int) -> int:
    """Get the list nesting."""
    return max(0, indent_spaces // 2)


def _toklen(s: str) -> int:
    """Get the token length."""
    return len(_ENCODING.encode(s))
