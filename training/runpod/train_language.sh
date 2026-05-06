#!/usr/bin/env bash
# Train a single-language Zipformer model on RunPod L40S
# Usage: ./train_language.sh <lang> [small|medium|large]
# Example: ./train_language.sh ar large

set -euo pipefail

LANG=${1:-}
MODEL_SIZE=${2:-medium}
CV_VERSION="16.1"

if [ -z "$LANG" ]; then
    echo "Usage: $0 <lang_code> [model_size]"
    echo "  lang_code : ar, hi, tr, id, fa, he, sw, ..."
    echo "  model_size: small | medium | large  (default: medium)"
    echo ""
    echo "Recommended mapping for highest quality:"
    echo "  large : ar   (~1000h Common Voice)"
    echo "  medium: hi, tr, id, fa  (~100-500h)"
    echo "  small : he, sw          (~20-100h)"
    exit 1
fi

WORKSPACE="/workspace/icefall_training"
DATA_DIR="$WORKSPACE/data/$LANG"
EXP_DIR="$WORKSPACE/exp/zipformer_${MODEL_SIZE}_${LANG}"
ICEFALL="$WORKSPACE/icefall"
RECIPE="$ICEFALL/egs/commonvoice/ASR/zipformer"

mkdir -p "$DATA_DIR" "$EXP_DIR"

echo "================================================"
echo "  Language : $LANG"
echo "  Model    : Zipformer-$MODEL_SIZE"
echo "  Data     : $DATA_DIR"
echo "  Exp      : $EXP_DIR"
echo "================================================"

# -------------------------------------------------
# Presets tuned for L40S 48GB — maximum quality
# -------------------------------------------------
case $MODEL_SIZE in
    small)
        # ~37M params — low-resource (he, sw). Conservative VRAM for stability.
        NUM_ENCODER_LAYERS="2,2,3,4,3,2"
        FEEDFORWARD_DIMS="384,576,1152,1536,1152,576"
        ENCODER_DIMS="128,192,384,512,384,192"
        ATTENTION_DIMS="64,96,192,256,192,96"
        ENCODER_UNMASKED_DIMS="112,136,224,256,224,136"
        DECODER_DIM=512
        JOINER_DIM=512
        MAX_DURATION=600
        NUM_EPOCHS=60
        BASE_LR=0.05
        WARMUP_STEPS=5000
        DROPOUT=0.15
        ;;
    medium)
        # ~85M params — balanced quality/speed (hi, tr, id, fa)
        # ~30-35 GB VRAM on L40S
        NUM_ENCODER_LAYERS="2,2,4,5,4,2"
        FEEDFORWARD_DIMS="512,768,1536,2048,1536,768"
        ENCODER_DIMS="192,256,512,768,512,256"
        ATTENTION_DIMS="96,128,256,384,256,128"
        ENCODER_UNMASKED_DIMS="168,180,300,320,300,180"
        DECODER_DIM=768
        JOINER_DIM=768
        MAX_DURATION=450
        NUM_EPOCHS=50
        BASE_LR=0.045
        WARMUP_STEPS=8000
        DROPOUT=0.12
        ;;
    large)
        # ~147M params — maximum quality (ar)
        # ~40-45 GB VRAM on L40S 48GB — tight but works with FP16
        NUM_ENCODER_LAYERS="2,2,5,6,5,2"
        FEEDFORWARD_DIMS="768,1024,2048,3072,2048,1024"
        ENCODER_DIMS="256,384,768,1024,768,384"
        ATTENTION_DIMS="128,192,384,512,384,192"
        ENCODER_UNMASKED_DIMS="224,240,448,512,448,240"
        DECODER_DIM=1024
        JOINER_DIM=1024
        MAX_DURATION=350
        NUM_EPOCHS=50
        BASE_LR=0.04
        WARMUP_STEPS=10000
        DROPOUT=0.10
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"; exit 1
        ;;
esac

# -------------------------------------------------
# 1. Download Common Voice
# -------------------------------------------------
echo "[1/5] Downloading Common Voice $CV_VERSION for '$LANG'..."
cd "$WORKSPACE"

CV_TARBALL="cv-corpus-${CV_VERSION}-2023-12-06-${LANG}.tar.gz"
CV_URL="https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-${CV_VERSION}-2023-12-06/${CV_TARBALL}"

if [ ! -f "$CV_TARBALL" ]; then
    wget -q --show-progress "$CV_URL" -O "$CV_TARBALL" || {
        echo "Primary URL failed, trying alternate date..."
        wget -q --show-progress \
            "https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-${CV_VERSION}-2022-12-07/cv-corpus-${CV_VERSION}-2022-12-07-${LANG}.tar.gz" \
            -O "$CV_TARBALL"
    }
fi

if [ ! -d "cv-corpus-${CV_VERSION}-2023-12-06" ]; then
    echo "Extracting Common Voice..."
    tar -xzf "$CV_TARBALL"
fi

CV_DIR=$(find . -maxdepth 1 -type d -name "cv-corpus-*-${LANG}" | head -1)
if [ -z "$CV_DIR" ]; then
    echo "ERROR: Could not find extracted Common Voice directory"
    exit 1
fi

# -------------------------------------------------
# 2. Lhotse manifests
# -------------------------------------------------
echo "[2/5] Preparing Lhotse manifests..."
python3 - "$CV_DIR" "$DATA_DIR" "$LANG" <<'PY'
import sys
from lhotse.recipes import prepare_common_voice
prepare_common_voice(
    corpus_dir=sys.argv[1],
    output_dir=sys.argv[2],
    language=sys.argv[3],
    num_jobs=8,
)
PY

# -------------------------------------------------
# 3. Train SentencePiece BPE (500 tokens)
# -------------------------------------------------
echo "[3/5] Training BPE tokenizer (vocab=500)..."
LANG_DIR="$DATA_DIR/lang"
mkdir -p "$LANG_DIR"

python3 - "$DATA_DIR" "$LANG_DIR" <<'PY'
import sys, glob, sentencepiece as spm
import lhotse

data_dir, lang_dir = sys.argv[1], sys.argv[2]

# Find supervision manifest
pattern = f"{data_dir}/commonvoice_supervisions_*.jsonl*"
files = glob.glob(pattern)
if not files:
    pattern = f"{data_dir}/cv*supervisions*.jsonl*"
    files = glob.glob(pattern)

cuts = lhotse.load_manifest_lazy(files[0])
texts = [c.text for c in cuts if hasattr(c, 'text') and c.text.strip()]

txt_path = f"{lang_dir}/train_text.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t.strip() + "\n")

spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix=f"{lang_dir}/bpe",
    vocab_size=500,
    character_coverage=0.9995,
    model_type="bpe",
    num_threads=8,
    input_sentence_size=200000,
    shuffle_input_sentence=True,
)
print(f"BPE trained. Vocab: {lang_dir}/bpe.model")
PY

# Build tokens.txt for icefall
python3 - "$LANG_DIR" <<'PY'
import sys
import sentencepiece as spm

lang_dir = sys.argv[1]
sp = spm.SentencePieceProcessor(model_file=f"{lang_dir}/bpe.model")

with open(f"{lang_dir}/tokens.txt", "w", encoding="utf-8") as f:
    f.write("<blk> 0\n")
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        # Escape spaces for icefall
        if piece == " ":
            piece = "<space>"
        f.write(f"{piece} {i + 1}\n")
print(f"tokens.txt written with {sp.vocab_size()} tokens")
PY

# -------------------------------------------------
# 4. Train Zipformer
# -------------------------------------------------
echo "[4/5] Training Zipformer-$MODEL_SIZE..."
cd "$RECIPE"

# Detect whether recipe has export-onnx.py (to know naming)
EXPORT_SCRIPT=""
for name in export-onnx.py export.py export-onnx-encoder-decoder-joiner.py; do
    [ -f "$name" ] && EXPORT_SCRIPT="$name" && break
done

python3 train.py \
    --world-size 1 \
    --num-epochs $NUM_EPOCHS \
    --start-epoch 1 \
    --exp-dir "$EXP_DIR" \
    --lang-dir "$LANG_DIR" \
    --manifest-dir "$DATA_DIR" \
    --max-duration $MAX_DURATION \
    --num-encoder-layers "$NUM_ENCODER_LAYERS" \
    --feedforward-dims "$FEEDFORWARD_DIMS" \
    --encoder-dims "$ENCODER_DIMS" \
    --attention-dims "$ATTENTION_DIMS" \
    --encoder-unmasked-dims "$ENCODER_UNMASKED_DIMS" \
    --decoder-dim $DECODER_DIM \
    --joiner-dim $JOINER_DIM \
    --causal False \
    --use-transducer True \
    --use-ctc True \
    --ctc-loss-scale 0.2 \
    --base-lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --bucketing-sampler True \
    --num-buckets 50 \
    --concatenate-cuts False \
    --on-the-fly-feats False \
    --shuffle True \
    --return-cuts True \
    --spec-aug-time-warp-factor 80 \
    --spec-aug-num-time-mask 2 \
    --spec-aug-max-time-mask-ratio 0.05 \
    --spec-aug-num-freq-mask 2 \
    --spec-aug-max-freq-mask-ratio 0.15 \
    --label-smoothing 0.1 \
    --dropout $DROPOUT \
    --seed 42 \
    --save-every-n 1000 \
    --keep-last-k 20 \
    --average-period 100 \
    --use-fp16 True \
    2>&1 | tee "$EXP_DIR/train.log"

# -------------------------------------------------
# 5. Decode with averaged model
# -------------------------------------------------
echo "[5/5] Decoding test set..."

AVG=10
[ "$NUM_EPOCHS" -lt 20 ] && AVG=5

# Some recipes use --avg, others --avg-last-n
python3 decode.py \
    --epoch $NUM_EPOCHS \
    --avg $AVG \
    --exp-dir "$EXP_DIR" \
    --lang-dir "$LANG_DIR" \
    --manifest-dir "$DATA_DIR" \
    --max-duration 600 \
    --decoding-method greedy_search \
    2>&1 | tee "$EXP_DIR/decode.log"

echo ""
echo "================================================"
echo "  $LANG ($MODEL_SIZE) DONE"
echo "  Exp dir: $EXP_DIR"
echo "  WER:     grep -i wer $EXP_DIR/decode.log"
echo "================================================"

# Metadata
cat > "$EXP_DIR/model_info.txt" <<EOF
language: $LANG
model_size: $MODEL_SIZE
num_epochs: $NUM_EPOCHS
max_duration: $MAX_DURATION
base_lr: $BASE_LR
warmup_steps: $WARMUP_STEPS
dropout: $DROPOUT
avg_epochs: $AVG
finished_at: $(date -Iseconds)
EOF
