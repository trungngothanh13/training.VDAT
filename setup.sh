#!/bin/bash
# Fast, apt-free setup for training.VDAT on ThunderCompute (takes ~10 seconds)
# Usage: bash setup.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Installing Python dependencies ==="
pip install -q -U -r "$ROOT_DIR/requirements.txt"

echo "=== 2. Enforcing headless OpenCV (avoids libgl1/libxcb1 apt-get completely) ==="
pip uninstall -y -q opencv-python >/dev/null 2>&1 || true
pip install -q --force-reinstall --no-deps opencv-python-headless

echo "=== 3. Checking GPU & downloading stock yolo26l.pt ==="
cd "$ROOT_DIR"
python -c "
import cv2, torch
from ultralytics import YOLO
print('OpenCV:', cv2.__version__, '| PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
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
echo "  2. Start training in background (survives SSH disconnect without tmux):"
echo "       nohup python -u train.py > train.log 2>&1 &"
echo "       tail -f train.log"
echo "  3. Evaluate & generate Markdown report:"
echo "       python evaluate_compare.py"
echo "  4. Launch side-by-side comparison web app:"
echo "       python compare_app.py --port 7860"
