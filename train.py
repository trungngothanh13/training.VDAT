#!/usr/bin/env python3
"""
Ready-to-run YOLO26 fine-tuning script for training.VDAT on ThunderCompute.
Usage:
    python train.py
    python train.py --epochs 100 --batch 32 --imgsz 864
    python train.py --resume
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

from model_utils import (
    ROOT_DIR,
    get_device_info,
    normalize_data_yaml,
    parse_names_dict,
    resolve_data_yaml,
)


def count_images(ds_root: Path, rel_path: str) -> int:
    if not rel_path:
        return 0
    folder = (ds_root / rel_path).resolve()
    if not folder.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom YOLO26 model for VDAT on ThunderCompute")
    parser.add_argument("--model", type=str, default="yolo26l.pt", help="Base YOLO26 weights (default: yolo26l.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (default: auto-detect dataset/data.yaml or ./data.yaml)")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--imgsz", type=int, default=864, help="Training image size matching VDAT stream (default: 864)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto; default: 16 for A6000)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker threads (default: 8)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs (default: 20)")
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"], help="Dataset cache mode (default: ram)")
    parser.add_argument("--name", type=str, default="vdat_yolo26l", help="Run name under ./runs/detect/")
    parser.add_argument("--project", type=str, default=str(ROOT_DIR / "runs" / "detect"), help="Output runs project dir")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Use cosine LR schedule (default: True)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest last.pt")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. '0' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    auto_device, device_desc = get_device_info()
    device = args.device if args.device is not None else auto_device

    found_yaml = resolve_data_yaml(args.data)
    if found_yaml is None:
        raise FileNotFoundError(
            "Dataset data.yaml not found in ./dataset/data.yaml or ./data.yaml. "
            "Run: bash download_dataset.sh \"<ROBOFLOW_EXPORT_URL>\" first."
        )
    data_yaml, cfg = normalize_data_yaml(str(found_yaml))
    ds_root = Path(cfg["path"])
    names = parse_names_dict(cfg.get("names", {}))

    train_n = count_images(ds_root, cfg.get("train", ""))
    val_n = count_images(ds_root, cfg.get("val", ""))
    test_n = count_images(ds_root, cfg.get("test", ""))

    print("=" * 68)
    print("VDAT YOLO26 Training Pipeline")
    print("=" * 68)
    print(f"Device:         {device_desc} (arg={device})")
    print(f"Base Model:     {args.model}")
    print(f"Dataset YAML:   {data_yaml}")
    print(f"Classes ({len(names)}):   {names}")
    print(f"Split Counts:   train={train_n}, val={val_n}, test={test_n}")
    print(f"Hyperparams:    epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, patience={args.patience}, cache={args.cache}")
    print("=" * 68)

    if train_n == 0:
        raise RuntimeError(
            f"No training images found under {ds_root / cfg.get('train', 'train/images')}. "
            "Run: bash download_dataset.sh \"<ROBOFLOW_URL>\" first."
        )

    cache_val = False if args.cache == "none" else args.cache

    if args.resume:
        last_pt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(f"Cannot resume: {last_pt} not found.")
        model = YOLO(str(last_pt))
        t0 = time.time()
        results = model.train(resume=True, device=device)
    else:
        model = YOLO(args.model)
        t0 = time.time()
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            patience=args.patience,
            cache=cache_val,
            cos_lr=args.cos_lr,
            lr0=args.lr0,
            project=args.project,
            name=args.name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
        )

    elapsed_min = (time.time() - t0) / 60.0
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (Path(args.project) / args.name)
    best_src = save_dir / "weights" / "best.pt"
    last_src = save_dir / "weights" / "last.pt"

    weights_dir = ROOT_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    exported_best = weights_dir / "best.pt"

    if best_src.exists():
        shutil.copy2(best_src, exported_best)
        print(f"\nSaved best checkpoint -> {exported_best}")
    if last_src.exists():
        shutil.copy2(last_src, weights_dir / "last.pt")

    summary = {
        "base_model": args.model,
        "trained_weights": str(exported_best if exported_best.exists() else best_src),
        "run_dir": str(save_dir),
        "device": device_desc,
        "elapsed_minutes": round(elapsed_min, 2),
        "epochs_configured": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "classes": names,
        "splits": {"train": train_n, "val": val_n, "test": test_n},
        "validation": {},
    }

    eval_weights = exported_best if exported_best.exists() else best_src
    if eval_weights.exists():
        best_model = YOLO(str(eval_weights))
        for split_name, split_count in [("val", val_n), ("test", test_n)]:
            if split_count <= 0:
                continue
            print(f"\n=== Running official Ultralytics evaluation on '{split_name}' split ({split_count} images) ===")
            metrics = best_model.val(
                data=str(data_yaml),
                split=split_name,
                imgsz=args.imgsz,
                device=device,
                plots=True,
                verbose=False,
            )
            box = metrics.box
            summary["validation"][split_name] = {
                "images": split_count,
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
                "mAP50": round(float(box.map50), 4),
                "mAP50_95": round(float(box.map), 4),
            }
            print(
                f"  [{split_name}] P={box.mp:.4f} | R={box.mr:.4f} | "
                f"mAP@50={box.map50:.4f} | mAP@50-95={box.map:.4f}"
            )

    summary_path = weights_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print(f"Training completed in {elapsed_min:.2f} min.")
    print(f"Best weights:   {exported_best}")
    print(f"Run artifacts:  {save_dir}")
    print(f"Summary JSON:   {summary_path}")
    print("Next steps:")
    print("  1. Generate full Stock vs Trained report:  python evaluate_compare.py")
    print("  2. Launch side-by-side comparison app:     python compare_app.py --port 7860")
    print("=" * 68)


if __name__ == "__main__":
    main()
