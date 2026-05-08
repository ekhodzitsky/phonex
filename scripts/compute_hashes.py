#!/usr/bin/env python3
"""Download model files and compute SHA-256 hashes for manifest.json."""

import hashlib
import json
import os
import sys
import tempfile
import urllib.request

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "manifest.json")
CHUNK_SIZE = 8192


def compute_sha256(url: str) -> str:
    """Download file to a temp location and compute SHA-256."""
    print(f"Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "phonex-manifest/1.0"})
    sha256 = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            total += len(chunk)
            if total % (10 * 1024 * 1024) == 0:
                print(f"  ... {total / 1024 / 1024:.1f} MB downloaded")
    print(f"  Done. Total: {total / 1024 / 1024:.1f} MB")
    return sha256.hexdigest()


def main() -> int:
    manifest_path = os.path.abspath(MANIFEST_PATH)
    with open(manifest_path) as f:
        data = json.load(f)

    updated = False
    for entry in data["models"]:
        if entry.get("sha256"):
            print(f"SKIP (already hashed): {entry['name']}")
            continue
        url = entry["url"]
        try:
            h = compute_sha256(url)
            entry["sha256"] = h
            updated = True
            print(f"  SHA-256: {h}")
        except Exception as e:
            print(f"ERROR downloading {entry['name']}: {e}", file=sys.stderr)
            continue

    if updated:
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Updated {manifest_path}")
    else:
        print("No updates needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
