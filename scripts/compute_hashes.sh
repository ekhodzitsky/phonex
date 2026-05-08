#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/../models/manifest.json"
TMPDIR="${SCRIPT_DIR}/../.tmp_models"
mkdir -p "$TMPDIR"

echo "Reading manifest: $MANIFEST"

# Extract URLs that need hashing
urls=$(python3 -c "
import json, sys
with open('$MANIFEST') as f:
    data = json.load(f)
for m in data['models']:
    if not m.get('sha256'):
        print(m['url'])
")

if [ -z "$urls" ]; then
    echo "All entries already have SHA-256 hashes."
    exit 0
fi

echo "Models to hash: $(echo "$urls" | wc -l | tr -d ' ')"

for url in $urls; do
    filename=$(basename "$url")
    tmpfile="${TMPDIR}/${filename}"
    
    echo ""
    echo "Downloading: $filename"
    if [ -f "$tmpfile" ]; then
        echo "  (using cached file)"
    else
        curl -fsSL --progress-bar "$url" -o "$tmpfile"
    fi
    
    hash=$(shasum -a 256 "$tmpfile" | awk '{print $1}')
    echo "  SHA-256: $hash"
    
    # Update manifest in-place
    python3 -c "
import json
with open('$MANIFEST') as f:
    data = json.load(f)
for m in data['models']:
    if m['url'] == '$url':
        m['sha256'] = '$hash'
with open('$MANIFEST', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write('\n')
"
    
    rm -f "$tmpfile"
done

rmdir "$TMPDIR" 2>/dev/null || true
echo ""
echo "Done. Manifest updated."
