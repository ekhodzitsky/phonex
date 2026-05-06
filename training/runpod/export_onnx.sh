#!/usr/bin/env bash
# Export trained Zipformer to ONNX for phonex inference
# Usage: ./export_onnx.sh <lang> [model_size]

set -euo pipefail

LANG=${1:-}
MODEL_SIZE=${2:-medium}

if [ -z "$LANG" ]; then
    echo "Usage: $0 <lang_code> [model_size]"
    exit 1
fi

WORKSPACE="/workspace/icefall_training"
EXP_DIR="$WORKSPACE/exp/zipformer_${MODEL_SIZE}_${LANG}"
ICEFALL="$WORKSPACE/icefall"
RECIPE="$ICEFALL/egs/commonvoice/ASR/zipformer"

if [ ! -d "$EXP_DIR" ]; then
    echo "ERROR: Experiment directory not found: $EXP_DIR"
    exit 1
fi

# Detect export script
EXPORT_SCRIPT=""
cd "$RECIPE"
for name in export-onnx.py export.py export-onnx-encoder-decoder-joiner.py; do
    [ -f "$name" ] && EXPORT_SCRIPT="$name" && break
done

if [ -z "$EXPORT_SCRIPT" ]; then
    echo "WARNING: No export script found in $RECIPE"
    echo "Falling back to sherpa-onnx export..."
    
    # Fallback: use sherpa-onnx scripts
    SHERPA_EXPORT="$WORKSPACE/sherpa-onnx/scripts/icefall/export-onnx-encoder-decoder-joiner.py"
    if [ -f "$SHERPA_EXPORT" ]; then
        python3 "$SHERPA_EXPORT" \
            --epoch 50 \
            --avg 10 \
            --exp-dir "$EXP_DIR" \
            --tokens "$EXP_DIR/../data/$LANG/lang/tokens.txt" \
            2>&1 | tee "$EXP_DIR/export_onnx.log"
    else
        echo "ERROR: Cannot find export script. Install sherpa-onnx or check icefall recipe."
        exit 1
    fi
else
    echo "Using export script: $EXPORT_SCRIPT"
    python3 "$EXPORT_SCRIPT" \
        --epoch 50 \
        --avg 10 \
        --exp-dir "$EXP_DIR" \
        --lang-dir "$EXP_DIR/../data/$LANG/lang" \
        2>&1 | tee "$EXP_DIR/export_onnx.log"
fi

echo "ONNX export complete. Check $EXP_DIR for encoder.onnx, decoder.onnx, joiner.onnx"
