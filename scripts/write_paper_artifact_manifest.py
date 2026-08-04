"""Write stable SHA-256 checksums for the public reviewer package."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path("paper_artifacts")
OUTPUT = ROOT / "MANIFEST.sha256"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".sha256", ".txt", ".yaml", ".yml"}


def portable_bytes(path: Path) -> bytes:
    """Hash text with canonical LF endings and binary artifacts byte-for-byte."""
    data = path.read_bytes()
    if path.suffix.casefold() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def main():
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        digest = hashlib.sha256(portable_bytes(path)).hexdigest()
        entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    print(f"{OUTPUT}: {len(entries)} files")


if __name__ == "__main__":
    main()
