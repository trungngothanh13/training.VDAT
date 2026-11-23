#!/bin/bash
# Downloads and prepares a Roboflow YOLO dataset directly into ./dataset/
# Usage:
#   bash download_dataset.sh "https://app.roboflow.com/ds/XXXX?key=YYYY"
#
# Or using Roboflow Python SDK with env var:
#   ROBOFLOW_API_KEY=... bash download_dataset.sh --sdk <workspace> <project> <version>

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$ROOT_DIR/dataset"

if [ "$1" = "--sdk" ]; then
    WORKSPACE="$2"
    PROJECT="$3"
    VERSION="${4:-2}"
    if [ -z "$ROBOFLOW_API_KEY" ] || [ -z "$WORKSPACE" ] || [ -z "$PROJECT" ]; then
        echo "Usage: ROBOFLOW_API_KEY=<key> bash download_dataset.sh --sdk <workspace> <project> [version]"
        exit 1
    fi
    rm -rf "$DATASET_DIR"
    mkdir -p "$DATASET_DIR"
    python -c "
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
ds = rf.workspace('$WORKSPACE').project('$PROJECT').version(int('$VERSION')).download('yolov8', location='$DATASET_DIR')
print('Downloaded dataset to:', ds.location)
"
else
    URL="$1"
    if [ -z "$URL" ]; then
        echo "Usage: bash download_dataset.sh \"<ROBOFLOW_EXPORT_URL>\""
        echo "Tip: In Roboflow -> Download Dataset -> Format: YOLOv8/YOLOv11 -> Show download code -> Terminal tab -> copy the https://app.roboflow.com/ds/... URL."
        exit 1
    fi
    rm -rf "$DATASET_DIR"
    mkdir -p "$DATASET_DIR"
    echo "=== Downloading Roboflow dataset into $DATASET_DIR ==="
    curl -L "$URL" -o "$DATASET_DIR/roboflow.zip"
    unzip -qo "$DATASET_DIR/roboflow.zip" -d "$DATASET_DIR"
    rm -f "$DATASET_DIR/roboflow.zip"
fi

echo "=== Normalizing dataset/data.yaml paths ==="
python -c "
from model_utils import normalize_data_yaml
yaml_path, cfg = normalize_data_yaml('$DATASET_DIR/data.yaml')
print('Normalized:', yaml_path)
print('Classes:', cfg.get('names'))
for split in ('train', 'val', 'test'):
    print(f'  {split}: {cfg.get(split)}')
"

echo "=== Dataset ready! Run: python train.py ==="
