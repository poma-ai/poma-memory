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

Values stay strings. ``true``/``yes``/``1`` are NOT coerced: the search
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

_REJECT_VALUE_PREFIXES = ("|", ">", "&", "*", "!", "{")


def extract_block(text: str) -> str | None:
    """Return the raw text between the fences, or None if there is no fence.

    Absent fence and unterminated fence are different answers: None means "this
    file has no front-matter" (fine), while an unterminated fence returns the
    remainder so that `parse` can report it as unparsed rather than as absent.
    """
    if not text.startswith(FENCE):
        return None
    after = text[len(FENCE):]
    if after[:1] not in ("\n", "\r"):
        return None  # "---foo", a horizontal rule, something else entirely
    lines = text.splitlines()
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == FENCE:
            return "\n".join(lines[1:i])
    return "\n".join(lines[1:])  # unterminated -> parse() will reject it


def is_terminated(text: str) -> bool:
    if not text.startswith(FENCE):
        return False
    lines = text.splitlines()
    return any(line.rstrip() == FENCE for line in lines[1:])


def parse(text: str) -> tuple[dict, bool]:
    """Parse front-matter from the head of a document.

    Returns ``(metadata, ok)``. ``ok`` is False only when a fence is present but
    its contents fall outside the grammar -- the caller stores ``{}`` and records
    the path. A document with no fence is ``({}, True)``: nothing to parse is not
    a failure.
    """
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

        key, _, rest = raw.partition(":")
        key = key.strip()
        value = rest.strip()
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
            sub = sub.strip()
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


def _scalar(value: str) -> str | None:
    """A single scalar, kept as a string. None means out of grammar."""
    if value[:1] in _REJECT_VALUE_PREFIXES:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _scalar_or_inline_list(value: str) -> str | list[str] | None:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = []
        for part in inner.split(","):
            item = _scalar(part.strip())
            if item is None:
                return None
            items.append(item)
        return items
    return _scalar(value)
