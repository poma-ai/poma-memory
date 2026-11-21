"""Guard: no tracked file may refer to a removed provider. No exemptions."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Built from parts so this file does not match its own pattern.
_PATTERN = re.compile("|".join(("open" + "ai", "gem" + "ini")), re.IGNORECASE)


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [f for f in out.split("\0") if f]


def test_no_reference_to_removed_providers():
    hits = []
    for rel in _tracked():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if _PATTERN.search(line):
                hits.append(f"{rel}:{n}")
    assert not hits, hits
