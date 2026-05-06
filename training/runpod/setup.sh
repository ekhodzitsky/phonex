#!/usr/bin/env bash
# Setup script for RunPod L40S Zipformer training
# Run once per fresh Pod

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}================================================${NC}"
echo -e "${YELLOW}  RunPod L40S Setup — Zipformer Training${NC}"
echo -e "${YELLOW}================================================${NC}"

echo -e "${YELLOW}[1/6] System dependencies...${NC}"
apt-get update -qq
apt-get install -y -qq \
    git wget sox libsox-dev libsox-fmt-all ffmpeg \
    libsndfile1-dev build-essential cmake pigz \
    > /dev/null 2>&1

echo -e "${YELLOW}[2/6] Workspace...${NC}"
WORKSPACE="/workspace/icefall_training"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

echo -e "${YELLOW}[3/6] PyTorch (CUDA 12.1)...${NC}"
pip install -q --upgrade pip
pip install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

echo -e "${YELLOW}[4/6] k2 (CUDA 12.1)...${NC}"
pip install -q k2==1.24.4.dev20240309+cuda12.1.torch2.2.0 \
    -f https://k2-fsa.github.io/k2/cuda.html || {
    echo -e "${RED}Pinned k2 failed, trying latest...${NC}"
    pip install -q k2 -f https://k2-fsa.github.io/k2/cuda.html
}

echo -e "${YELLOW}[5/6] lhotse + tooling...${NC}"
pip install -q lhotse sentencepiece onnx onnxruntime

echo -e "${YELLOW}[6/6] Cloning repos...${NC}"
[ ! -d "icefall" ] && git clone --depth 1 https://github.com/k2-fsa/icefall.git
[ ! -d "sherpa-onnx" ] && git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx.git

echo -e "${GREEN}Done!${NC} Workspace: ${WORKSPACE}"
