"""Restricted YAML front-matter parser.

Deliberately not a YAML parser. It reads the subset that the corpora this
package targets actually write, and it *refuses* anything else rather than
guessing. A hand-rolled YAML subset is normally a bad trade because its
failures are silent; here an out-of-grammar block is reported as unparsed and
surfaced by ``status()``, which is the same discipline as the empty-result
problem in search: never hand back something indistinguishable from a correct
answer.

Grammar, between a ``---`` fence at byte 0 and the next ``---`` line:

    key: value                  scalar
    key: [a, b]                 inline list
    key:                        block list
      - a
      - b
    key:                        one level of nesting, flattened to "key.sub"
      sub: value

A trailing ``# comment`` outside quotes is dropped, as YAML does. Values stay
strings otherwise. ``true``/``yes``/``1`` are NOT coerced: the search
predicate is case-sensitive string equality, so any coercion opens a silent gap
between what an author wrote and what they can filter on. Matching outer quotes
are stripped, which is spelling rather than meaning.

Everything else -- block scalars (``|``, ``>``), anchors, flow mappings, tags,
multi-document streams, duplicate keys, inconsistent indentation, an
unterminated fence -- makes the whole block unparsed.
"""

from __future__ import annotations

FENCE = "---"

# A front-matter block past this is not front-matter. Bounds the head-read that
# metadata backfill does on files it does not otherwise need to open.
MAX_BYTES = 8192

# A fence that has not closed within this much text is not front-matter. Both
# the parser and the bounded head read in `incremental` use it, so the two
# cannot disagree about whether a block parsed, and a large file that merely
# opens with `---` is never read whole to find that out.
MAX_BLOCK_BYTES = 65536

_REJECT_VALUE_PREFIXES = ("|", ">", "&", "*", "!", "{")


def extract_block(text: str) -> str | None:
    """Return the raw text between the fences, or None if there is no fence.

    Absent fence and unterminated fence are different answers: None means "this
    file has no front-matter" (fine), while an unterminated fence returns the
    remainder so that `parse` can report it as unparsed rather than as absent.
    """
    # A BOM before the fence would otherwise make this look like a file with
    # no front-matter at all: no metadata, not flagged unparsed, invisible in
    # `status()`. Editors write one without being asked.
    text = text.lstrip("\ufeff")
    if not text.startswith(FENCE):
        return None
    after = text[len(FENCE):]
    if after[:1] not in ("\n", "\r"):
        return None  # "---foo", a horizontal rule, something else entirely
    lines = text[:MAX_BLOCK_BYTES].splitlines()
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == FENCE:
            return "\n".join(lines[1:i])
    return "\n".join(lines[1:])  # unterminated -> parse() will reject it


def is_terminated(text: str) -> bool:
    text = text.lstrip("\ufeff")
    if not text.startswith(FENCE):
        return False
    lines = text[:MAX_BLOCK_BYTES].splitlines()
    return any(line.rstrip() == FENCE for line in lines[1:])


def parse(text: str) -> tuple[dict, bool]:
    """Parse front-matter from the head of a document.

    Returns ``(metadata, ok)``. ``ok`` is False only when a fence is present but
    its contents fall outside the grammar -- the caller stores ``{}`` and records
    the path. A document with no fence is ``({}, True)``: nothing to parse is not
    a failure.
    """
    text = text.lstrip("\ufeff")
    block = extract_block(text)
    if block is None:
        return {}, True
    if not is_terminated(text):
        return {}, False
    return _parse_block(block)


def _parse_block(block: str) -> tuple[dict, bool]:
    out: dict[str, str | list[str]] = {}
    lines = block.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        if raw[:1].isspace():
            return {}, False  # indented line with no key above it
        if raw.startswith("- "):
            return {}, False  # top-level sequence: not a mapping
        if ":" not in raw:
            return {}, False

        split = _split_key(raw)
        if split is None:
            return {}, False
        key, rest = split
        key = _unquote_key(key.strip())
        if key is None:
            return {}, False
        # Strip before testing emptiness: `key:  # note` is an empty value with
        # a comment, not a value of "# note".
        value = strip_comment(rest.strip())
        if not key or key in out:
            return {}, False  # empty or duplicate key

        if value:
            parsed = _scalar_or_inline_list(value)
            if parsed is None:
                return {}, False
            out[key] = parsed
            i += 1
            continue

        # Empty value: a block list or a nested mapping follows, or nothing.
        body, i = _take_indented(lines, i + 1)
        if body is None:
            return {}, False  # inconsistent indentation
        if not body:
            out[key] = ""
            continue
        if all(b.startswith("- ") for b in body):
            items = []
            for b in body:
                item = _scalar(b[2:].strip())
                if item is None:
                    return {}, False
                items.append(item)
            out[key] = items
            continue
        for b in body:
            if ":" not in b or b.startswith("- "):
                return {}, False
            sub, _, subval = b.partition(":")
            sub = _unquote_key(sub.strip())
            subval = _scalar(subval.strip())
            if not sub or subval is None:
                return {}, False
            dotted = f"{key}.{sub}"
            if dotted in out:
                return {}, False
            out[dotted] = subval

    return out, True


def _take_indented(lines: list[str], start: int) -> tuple[list[str] | None, int]:
    """Collect the run of indented lines at `start`, requiring one indent width.

    Returns (stripped lines, index of the first line not consumed). A run that
    mixes indent widths returns (None, _): ambiguous nesting is rejected rather
    than guessed at.
    """
    body: list[str] = []
    indent: str | None = None
    i = start
    while i < len(lines):
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue
        if not raw[:1].isspace():
            break
        prefix = raw[: len(raw) - len(raw.lstrip())]
        if indent is None:
            indent = prefix
        elif prefix != indent:
            return None, i
        body.append(raw.strip())
        i += 1
    return body, i


def _split_key(raw: str) -> tuple[str, str] | None:
    """Split a mapping line at the first colon OUTSIDE quotes.

    `raw.partition(":")` cuts `"my:key": value` in the middle of the key and
    the line is then rejected — visible rather than silent, but still a valid
    quoted key refused for no reason but a naive split.
    """
    quote = None
    for i, ch in enumerate(raw):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"') and not raw[:i].strip():
            quote = ch
        elif ch == ":":
            return raw[:i], raw[i + 1:]
    return None


def _unquote_key(key: str) -> str | None:
    """Strip matching outer quotes from a key. None means out of grammar.

    A quoted key is ordinary YAML, and keeping the quotes produces a key no
    predicate can ever match. `<<` is YAML's merge key and means something this
    parser does not implement, so it is refused rather than taken literally.
    """
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        key = key[1:-1]
    if key.startswith("<<"):
        return None
    if key[:1] in ("'", '"'):
        return None
    return key


def strip_comment(value: str) -> str:
    """Drop a YAML trailing comment. A '#' inside quotes is literal.

    Without this, `kind: event  # primary` stores "event  # primary" and the
    author cannot filter on the value they wrote — a silent wrong parse, which
    is the one failure this parser is built to avoid.
    """
    quote = None
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"') and i == 0:
            # YAML quotes a scalar only from its start. Treating an apostrophe
            # anywhere as an opener swallows the rest of the line, so
            # `title: Don't ship  # decided` kept the comment in the value.
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value


def _split_inline(inner: str) -> list[str] | None:
    """Split an inline list on commas that are not inside quotes.

    `inner.split(",")` cuts `[api, "auth, login"]` into three garbage items and
    reports success. None means an unterminated quote: unparsed, not guessed.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"') and not "".join(buf).strip():
            # Only at the start of an item, for the same reason as
            # `strip_comment`: `[don't, can't]` is two items, not one.
            quote = ch
            buf.append(ch)
        elif ch in "[]{}":
            return None  # nested flow collections are outside this grammar
        elif ch == ",":
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if quote is not None:
        return None
    parts.append("".join(buf).strip())
    if parts and parts[-1] == "":
        parts.pop()  # `[a, ]` is ['a'] in YAML, not ['a', '']
    if any(part == "" for part in parts):
        return None
    return parts


def _scalar(value: str) -> str | None:
    """A single scalar, kept as a string. None means out of grammar."""
    value = strip_comment(value)
    if value[:1] in _REJECT_VALUE_PREFIXES:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"' and "\\" in inner:
            # YAML unescapes these; this parser does not. Returning the
            # backslashes literally is a wrong value, and `\"` also closes the
            # quote early in strip_comment and silently truncates the rest.
            return None
        if value[0] == "'" and "''" in inner:
            return None  # single-quoted YAML escapes a quote by doubling it
        return inner
    if value[:1] in ("'", '"'):
        return None  # opened a quote and never closed it
    return value


def _scalar_or_inline_list(value: str) -> str | list[str] | None:
    if value.startswith("[") and not value.endswith("]"):
        return None  # an unterminated bracket is not a scalar
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = _split_inline(inner)
        if parts is None:
            return None
        items = []
        for part in parts:
            item = _scalar(part)
            if item is None:
                return None
            items.append(item)
        return items
    return _scalar(value)
