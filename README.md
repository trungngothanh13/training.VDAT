# training.VDAT — YOLO26 Fine-Tuning & Model Comparison Toolkit

Self-contained training, benchmarking, and side-by-side comparison suite for fine-tuning **YOLO26l** on ThunderCompute without uploading datasets to GitHub.

## 1. Upload `training.VDAT` to ThunderCompute
From your local machine (using your training ThunderCompute instance, e.g. `tnr-0` or `tnr-1`):
```bash
scp -r /home/trung/Projects/training.VDAT tnr-0:~/
```

## 2. Install Dependencies (One Command)
On ThunderCompute:
```bash
cd ~/training.VDAT
bash setup.sh
```

## 3. Pull Dataset Directly from Roboflow
In Roboflow → **Download Dataset** → Format: **YOLOv8 / YOLOv11** → **Show download code** → **Terminal** tab → copy the `https://app.roboflow.com/ds/...` URL:
```bash
bash download_dataset.sh "https://app.roboflow.com/ds/YOUR_EXPORT_LINK"
```
This downloads into `./dataset/` and automatically fixes `dataset/data.yaml` paths.

## 4. Train YOLO26l in Background (Survives SSH Disconnect)
```bash
nohup python -u train.py > train.log 2>&1 &
tail -f train.log
```
*(Or inside `tmux new -s train` if `tmux` is installed. Press `Ctrl+C` on `tail -f` anytime — training keeps running in the background.)*
Optional overrides:
```bash
python train.py --epochs 100 --batch 16 --imgsz 864
python train.py --resume
```
When training finishes, the best weights are copied automatically to `./weights/best.pt` and official `val`/`test` metrics are logged to `./weights/training_summary.json`.

## 5. Generate Detailed Comparison Report (Stock vs Trained)
Runs both **Stock `yolo26l.pt`** and **Trained `weights/best.pt`** on:
1. Labeled `test` & `valid` splits in `./dataset/data.yaml` (canonical class-name aligned **mAP@50, mAP@50-95, Precision, Recall, F1, TP/FP/FN, IoU**).
2. `test_1.mp4`, `test_2.mp4`, and `test_3.mp4` (detection counts, per-class breakdown, confidence stats, small-object counts, box agreement, latency/FPS, and saved side-by-side frames).

```bash
python evaluate_compare.py
```
Outputs:
- `reports/comparison_report.md` (easy-to-read Markdown benchmark report)
- `reports/comparison_report.json` (raw metrics JSON)
- `reports/frames/*.jpg` (side-by-side annotated snapshots)

## 6. Launch Side-by-Side Interactive Comparison App
On ThunderCompute:
```bash
python compare_app.py --port 7860
```
Forward port `7860` from your local laptop:
```bash
ssh -L 7860:localhost:7860 tnr-0
```
Then open `http://localhost:7860` in your browser to inspect random or specific frames from `test_1.mp4`–`test_3.mp4` side-by-side and save PNG snapshots for your thesis.
