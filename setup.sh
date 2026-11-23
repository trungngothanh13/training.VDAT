#!/bin/bash
# One-command setup for training.VDAT on ThunderCompute (or local Linux GPU host)
# Usage: bash setup.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Installing system tools (unzip, tmux, libgl1) ==="
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq unzip tmux libgl1 libglib2.0-0
fi

echo "=== 2. Installing / upgrading Python dependencies ==="
pip install -q -U -r "$ROOT_DIR/requirements.txt"

echo "=== 3. Checking GPU & downloading stock yolo26l.pt ==="
cd "$ROOT_DIR"
python -c "
import torch
from ultralytics import YOLO
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)')
YOLO('yolo26l.pt')
print('Stock yolo26l.pt ready.')
"

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Download dataset from Roboflow:"
echo "       bash download_dataset.sh \"<ROBOFLOW_TERMINAL_CURL_URL>\""
echo "  2. Start training inside tmux:"
echo "       tmux new -s train"
echo "       python train.py"
echo "  3. Evaluate & generate Markdown report:"
echo "       python evaluate_compare.py"
echo "  4. Launch side-by-side comparison web app:"
echo "       python compare_app.py --port 7860"
