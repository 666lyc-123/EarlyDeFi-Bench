"""Regenerate MANIFEST.json with SHA-256 hashes for every released file.

The manifest intentionally excludes itself (its own hash would change the
moment it is written) and all version-control internals under ``.git``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PATHS = {"MANIFEST.json"}
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="EarlyDeFi-Bench-GitHub-Ready-20260925")
    parser.add_argument("--output", default="MANIFEST.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED_PATHS:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})

    manifest = {
        "package": args.package,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.name}: {len(files)} files, manifest excludes itself.")


if __name__ == "__main__":
    main()
