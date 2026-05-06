#!/usr/bin/env bash
# Sequential training for all 7 target languages on RunPod L40S
# Usage: bash run_all_languages.sh

set -euo pipefail

cd "$(dirname "$0")"

# Mapping based on Common Voice data size
declare -A MODEL_MAP=(
    [ar]="large"   # ~1000h
    [hi]="medium"  # ~200h
    [tr]="medium"  # ~400h
    [id]="medium"  # ~150h
    [fa]="medium"  # ~200h
    [he]="small"   # ~80h
    [sw]="small"   # ~30h
)

TOTAL_START=$(date +%s)

for lang in ar hi tr id fa he sw; do
    size=${MODEL_MAP[$lang]}
    echo ""
    echo "=========================================="
    echo "  STARTING: $lang → Zipformer-$size"
    echo "=========================================="
    START=$(date +%s)

    bash train_language.sh "$lang" "$size"

    END=$(date +%s)
    DURATION=$((END - START))
    echo "  COMPLETED: $lang in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m"
    echo "=========================================="
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
echo ""
echo "================================================"
echo "  ALL LANGUAGES TRAINED"
echo "  Total time: $((TOTAL_DURATION / 3600))h $(((TOTAL_DURATION % 3600) / 60))m"
echo "================================================"

# Export all to ONNX
echo ""
echo "Exporting ONNX models..."
for lang in ar hi tr id fa he sw; do
    size=${MODEL_MAP[$lang]}
    bash export_onnx.sh "$lang" "$size" || echo "WARNING: $lang export failed"
done

echo ""
echo "Done. Models are in /workspace/icefall_training/exp/"
