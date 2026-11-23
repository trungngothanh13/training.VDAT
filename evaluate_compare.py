#!/usr/bin/env python3
"""
Comprehensive Stock YOLO26 vs Custom Trained YOLO26 evaluation & Markdown report generator.

Evaluates BOTH models on:
  1. Labeled dataset splits (test and valid from ./dataset/data.yaml, if present),
     using canonical class-name alignment so Stock COCO YOLO26 and Custom Trained YOLO26
     are measured apples-to-apples (Precision, Recall, F1, mAP@50, mAP@50-95, TP/FP/FN, IoU).
  2. Real traffic test videos (test_1.mp4, test_2.mp4, test_3.mp4):
     detection volume, per-class breakdown, confidence distribution, small-object counts,
     model-to-model IoU agreement, inference latency/FPS, and saved side-by-side PNG frames.

Outputs a readable Markdown report to ./reports/comparison_report.md.
"""

import argparse
import datetime
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from model_utils import (
    ROOT_DIR,
    canonical_class_name,
    compute_iou_matrix,
    draw_detections_bgr,
    get_device_info,
    get_precision_kwargs,
    list_test_videos,
    make_side_by_side_bgr,
    normalize_data_yaml,
    parse_names_dict,
    resolve_data_yaml,
    resolve_trained_weights,
    run_model_detections,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Stock YOLO26 vs Trained YOLO26 and generate Markdown report")
    parser.add_argument("--stock", type=str, default="yolo26l.pt", help="Stock YOLO26 weights (default: yolo26l.pt)")
    parser.add_argument("--trained", type=str, default=None, help="Trained weights path (default: auto-find weights/best.pt)")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML for ground-truth eval (default: auto-detect dataset/data.yaml or ./data.yaml)")
    parser.add_argument("--videos", nargs="*", default=None, help="Video paths (default: all *.mp4 in training.VDAT)")
    parser.add_argument("--video-samples", type=int, default=60, help="Evenly sampled frames per video (default: 60)")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
    parser.add_argument("--imgsz", type=int, default=864, help="Inference image size (default: 864)")
    parser.add_argument("--all-coco-classes", action="store_true", help="Do not filter Stock YOLO26 to COCO vehicle classes on videos")
    parser.add_argument("--out", type=str, default=str(ROOT_DIR / "reports" / "comparison_report.md"), help="Output Markdown report path")
    return parser.parse_args()


def load_yolo_ground_truth(label_path: Path, img_w: int, img_h: int, names: Dict[int, str]) -> List[Dict[str, Any]]:
    gts: List[Dict[str, Any]] = []
    if not label_path.exists():
        return gts
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h
            cname = names.get(cid, str(cid))
            gts.append({
                "xyxy": [x1, y1, x2, y2],
                "class_id": cid,
                "class_name": cname,
                "canon_name": canonical_class_name(cname),
            })
    return gts


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point COCO-style interpolated Average Precision."""
    if len(recalls) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def evaluate_aligned_dataset_split(
    model,
    ds_root: Path,
    images_rel: str,
    ds_names: Dict[int, str],
    imgsz: int,
    conf_oper: float,
    precision_kwargs: Dict[str, Any],
    is_stock_coco: bool,
) -> Dict[str, Any]:
    """
    Ground-truth evaluator matching predictions to dataset labels by canonical class name,
    allowing fair Stock COCO YOLO26 vs Custom Trained YOLO26 metrics on identical splits.
    """
    images_dir = (ds_root / images_rel).resolve()
    labels_dir = images_dir.parent / "labels"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    target_canon_classes = sorted({canonical_class_name(n) for n in ds_names.values()})
    iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95

    # Per-class prediction records for AP curve: list of (conf, matched_per_iou_bool_list)
    class_preds_ap: Dict[str, List[Tuple[float, List[bool]]]] = defaultdict(list)
    class_gt_count: Counter = Counter()

    # Operating-point metrics at conf >= conf_oper & IoU >= 0.50
    class_tp: Counter = Counter()
    class_fp: Counter = Counter()
    matched_ious: List[float] = []
    oper_confs: List[float] = []
    latencies_ms: List[float] = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        ih, iw = frame.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        gts = load_yolo_ground_truth(label_path, iw, ih, ds_names)

        # Predict at low conf (0.01) for full PR/AP curve, then slice >= conf_oper for P/R/F1
        preds_all, infer_ms = run_model_detections(
            model,
            frame,
            conf=0.01,
            imgsz=imgsz,
            vehicle_only_coco=is_stock_coco,
            precision_kwargs=precision_kwargs,
        )
        latencies_ms.append(infer_ms)

        # Keep only predictions belonging to classes represented in this dataset
        preds = [p for p in preds_all if p["canon_name"] in target_canon_classes]

        for cls in target_canon_classes:
            cls_gts = [g for g in gts if g["canon_name"] == cls]
            cls_preds = sorted(
                [p for p in preds if p["canon_name"] == cls],
                key=lambda x: x["conf"],
                reverse=True,
            )
            class_gt_count[cls] += len(cls_gts)

            if not cls_preds:
                continue

            if not cls_gts:
                for p in cls_preds:
                    class_preds_ap[cls].append((p["conf"], [False] * len(iou_thresholds)))
                    if p["conf"] >= conf_oper:
                        class_fp[cls] += 1
                        oper_confs.append(p["conf"])
                continue

            gt_boxes = np.array([g["xyxy"] for g in cls_gts], dtype=np.float32)
            pred_boxes = np.array([p["xyxy"] for p in cls_preds], dtype=np.float32)
            ious = compute_iou_matrix(pred_boxes, gt_boxes)

            # Track matched GT per IoU threshold for AP curve
            matched_gt_by_thr = [set() for _ in iou_thresholds]
            matched_gt_oper: set = set()

            for pi, p in enumerate(cls_preds):
                row_ious = ious[pi]
                best_gi = int(np.argmax(row_ious))
                best_iou = float(row_ious[best_gi])

                # AP matching across 0.50..0.95
                thr_flags = []
                for ti, thr in enumerate(iou_thresholds):
                    ok = False
                    # Find best unmatched GT >= thr
                    order = np.argsort(-row_ious)
                    for gi in order:
                        if float(row_ious[gi]) < thr:
                            break
                        if int(gi) not in matched_gt_by_thr[ti]:
                            matched_gt_by_thr[ti].add(int(gi))
                            ok = True
                            break
                    thr_flags.append(ok)
                class_preds_ap[cls].append((p["conf"], thr_flags))

                # Operating-point matching (conf >= conf_oper, IoU >= 0.50)
                if p["conf"] >= conf_oper:
                    oper_confs.append(p["conf"])
                    order = np.argsort(-row_ious)
                    matched = False
                    for gi in order:
                        if float(row_ious[gi]) < 0.50:
                            break
                        if int(gi) not in matched_gt_oper:
                            matched_gt_oper.add(int(gi))
                            matched_ious.append(float(row_ious[gi]))
                            matched = True
                            break
                    if matched:
                        class_tp[cls] += 1
                    else:
                        class_fp[cls] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    ap50_list: List[float] = []
    ap50_95_list: List[float] = []
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for cls in target_canon_classes:
        n_gt = int(class_gt_count[cls])
        tp = int(class_tp[cls])
        fp = int(class_fp[cls])
        fn = max(0, n_gt - tp)
        total_tp += tp
        total_fp += fp
        total_gt += n_gt

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        records = sorted(class_preds_ap[cls], key=lambda x: x[0], reverse=True)
        ap_per_thr: List[float] = []
        if n_gt > 0 and records:
            for ti in range(len(iou_thresholds)):
                tp_flags = np.array([1.0 if r[1][ti] else 0.0 for r in records], dtype=np.float64)
                fp_flags = 1.0 - tp_flags
                tp_cum = np.cumsum(tp_flags)
                fp_cum = np.cumsum(fp_flags)
                rec_curve = tp_cum / n_gt
                prec_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
                ap_per_thr.append(compute_ap_from_pr(rec_curve, prec_curve))
        else:
            ap_per_thr = [0.0] * len(iou_thresholds)

        ap50 = ap_per_thr[0]
        ap50_95 = float(np.mean(ap_per_thr))
        if n_gt > 0:
            ap50_list.append(ap50)
            ap50_95_list.append(ap50_95)

        per_class[cls] = {
            "gt": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "ap50": ap50,
            "ap50_95": ap50_95,
        }

    total_fn = max(0, total_gt - total_tp)
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "images": len(image_paths),
        "gt_boxes": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1": overall_f1,
        "mAP50": float(np.mean(ap50_list)) if ap50_list else 0.0,
        "mAP50_95": float(np.mean(ap50_95_list)) if ap50_95_list else 0.0,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_conf": float(np.mean(oper_confs)) if oper_confs else 0.0,
        "avg_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "per_class": per_class,
    }


def match_between_models(
    stock_dets: List[Dict[str, Any]],
    trained_dets: List[Dict[str, Any]],
    iou_thr: float = 0.50,
) -> Tuple[int, int, int, List[float]]:
    """Returns (matched_both, only_stock, only_trained, matched_ious) using class-agnostic IoU>=0.50."""
    if not stock_dets and not trained_dets:
        return 0, 0, 0, []
    if not stock_dets:
        return 0, 0, len(trained_dets), []
    if not trained_dets:
        return 0, len(stock_dets), 0, []

    b_s = np.array([d["xyxy"] for d in stock_dets], dtype=np.float32)
    b_t = np.array([d["xyxy"] for d in trained_dets], dtype=np.float32)
    ious = compute_iou_matrix(b_s, b_t)

    used_t = set()
    matched_ious: List[float] = []
    # Greedy match by highest IoU pairs
    pairs = []
    for si in range(len(stock_dets)):
        for ti in range(len(trained_dets)):
            v = float(ious[si, ti])
            if v >= iou_thr:
                pairs.append((v, si, ti))
    pairs.sort(reverse=True)
    used_s = set()
    for v, si, ti in pairs:
        if si in used_s or ti in used_t:
            continue
        used_s.add(si)
        used_t.add(ti)
        matched_ious.append(v)

    matched = len(matched_ious)
    return matched, len(stock_dets) - matched, len(trained_dets) - matched, matched_ious


def summarize_video_model_stats(per_frame_dets: List[List[Dict[str, Any]]], latencies: List[float]) -> Dict[str, Any]:
    all_dets = [d for frame_dets in per_frame_dets for d in frame_dets]
    counts = [len(fd) for fd in per_frame_dets]
    confs = [d["conf"] for d in all_dets]
    class_counts = Counter(d["canon_name"] for d in all_dets)

    small_n = sum(1 for d in all_dets if d["area"] < 32 * 32)
    med_n = sum(1 for d in all_dets if 32 * 32 <= d["area"] < 96 * 96)
    large_n = sum(1 for d in all_dets if d["area"] >= 96 * 96)
    high_conf_n = sum(1 for c in confs if c >= 0.50)

    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    p95_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps = (1000.0 / avg_ms) if avg_ms > 0 else 0.0

    return {
        "frames": len(per_frame_dets),
        "total_detections": len(all_dets),
        "avg_per_frame": float(np.mean(counts)) if counts else 0.0,
        "max_per_frame": int(max(counts)) if counts else 0,
        "frames_with_det": sum(1 for c in counts if c > 0),
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
        "median_conf": float(np.median(confs)) if confs else 0.0,
        "p95_conf": float(np.percentile(confs, 95)) if confs else 0.0,
        "min_conf": float(min(confs)) if confs else 0.0,
        "high_conf_count": high_conf_n,
        "small_count": small_n,
        "medium_count": med_n,
        "large_count": large_n,
        "class_counts": dict(class_counts),
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def evaluate_video(
    video_path: Path,
    stock_model,
    trained_model,
    num_samples: int,
    conf: float,
    imgsz: int,
    vehicle_only_coco: bool,
    precision_kwargs: Dict[str, Any],
    frames_out_dir: Path,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_sample = min(max(1, num_samples), max(1, total_frames))
    indices = np.linspace(0, max(0, total_frames - 1), num=n_sample, dtype=int).tolist()
    indices = sorted(set(indices))

    stock_per_frame: List[List[Dict[str, Any]]] = []
    trained_per_frame: List[List[Dict[str, Any]]] = []
    stock_ms_list: List[float] = []
    trained_ms_list: List[float] = []

    matched_total = 0
    only_stock_total = 0
    only_trained_total = 0
    pair_ious_all: List[float] = []

    # Keep candidate frames to save visual comparison snapshots
    best_gain_record = None     # highest (len(trained) - len(stock))
    densest_record = None       # highest total detections
    mid_record = None           # middle representative frame

    mid_idx = indices[len(indices) // 2]

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        s_dets, s_ms = run_model_detections(
            stock_model,
            frame,
            conf=conf,
            imgsz=imgsz,
            vehicle_only_coco=vehicle_only_coco,
            precision_kwargs=precision_kwargs,
        )
        t_dets, t_ms = run_model_detections(
            trained_model,
            frame,
            conf=conf,
            imgsz=imgsz,
            vehicle_only_coco=False,
            precision_kwargs=precision_kwargs,
        )

        stock_per_frame.append(s_dets)
        trained_per_frame.append(t_dets)
        stock_ms_list.append(s_ms)
        trained_ms_list.append(t_ms)

        m, os_n, ot_n, p_ious = match_between_models(s_dets, t_dets)
        matched_total += m
        only_stock_total += os_n
        only_trained_total += ot_n
        pair_ious_all.extend(p_ious)

        gain = len(t_dets) - len(s_dets)
        density = len(t_dets) + len(s_dets)

        if best_gain_record is None or gain > best_gain_record[0]:
            best_gain_record = (gain, fi, frame.copy(), s_dets, t_dets, s_ms, t_ms)
        if densest_record is None or density > densest_record[0]:
            densest_record = (density, fi, frame.copy(), s_dets, t_dets, s_ms, t_ms)
        if fi == mid_idx:
            mid_record = (0, fi, frame.copy(), s_dets, t_dets, s_ms, t_ms)

    cap.release()

    saved_images: List[Dict[str, Any]] = []
    seen_saved_frames = set()
    frames_out_dir.mkdir(parents=True, exist_ok=True)

    for tag, rec in [
        ("max_gain", best_gain_record),
        ("densest_traffic", densest_record),
        ("midpoint", mid_record),
    ]:
        if rec is None:
            continue
        _, fi, fr, s_dets, t_dets, s_ms, t_ms = rec
        if fi in seen_saved_frames:
            continue
        seen_saved_frames.add(fi)
        left = draw_detections_bgr(fr, s_dets, title=f"Stock YOLO26 | {len(s_dets)} det | {s_ms:.1f}ms")
        right = draw_detections_bgr(fr, t_dets, title=f"Trained YOLO26 | {len(t_dets)} det | {t_ms:.1f}ms")
        combo = make_side_by_side_bgr(left, right)
        fname = f"{video_path.stem}_{tag}_f{fi}.jpg"
        out_img = frames_out_dir / fname
        cv2.imwrite(str(out_img), combo, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved_images.append({
            "tag": tag,
            "frame": fi,
            "filename": fname,
            "rel_path": f"frames/{fname}",
            "stock_count": len(s_dets),
            "trained_count": len(t_dets),
        })

    return {
        "video": video_path.name,
        "resolution": f"{w}x{h}",
        "source_fps": round(vid_fps, 2),
        "total_video_frames": total_frames,
        "sampled_frames": len(stock_per_frame),
        "stock": summarize_video_model_stats(stock_per_frame, stock_ms_list),
        "trained": summarize_video_model_stats(trained_per_frame, trained_ms_list),
        "agreement": {
            "matched_both": matched_total,
            "only_stock": only_stock_total,
            "only_trained": only_trained_total,
            "mean_pair_iou": float(np.mean(pair_ious_all)) if pair_ious_all else 0.0,
        },
        "saved_images": saved_images,
    }


def fmt_delta(new_v: float, old_v: float, fmt: str = "+.2f", pct: bool = False) -> str:
    diff = new_v - old_v
    if pct:
        return f"{diff * 100:+.2f}%"
    return format(diff, fmt)


def write_markdown_report(
    out_path: Path,
    meta: Dict[str, Any],
    dataset_results: Dict[str, Any],
    official_val_results: Dict[str, Any],
    video_results: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# VDAT Model Comparison Report: Stock YOLO26 vs Custom Trained YOLO26")
    lines.append("")
    lines.append(f"- **Generated:** {meta['generated_at']}")
    lines.append(f"- **Hardware:** `{meta['device']}` (`precision={meta['precision']}`)")
    lines.append(f"- **Stock Model:** `{meta['stock_model']}` (Vehicle filter: `{meta['vehicle_only_coco']}`)")
    lines.append(f"- **Trained Model:** `{meta['trained_model']}`")
    lines.append(f"- **Trained Classes:** `{meta['trained_classes']}`")
    lines.append(f"- **Evaluation Settings:** `imgsz={meta['imgsz']}`, `conf={meta['conf']}`")
    lines.append("")

    # 1. Ground-Truth Dataset Evaluation
    if dataset_results:
        lines.append("## 1. Ground-Truth Dataset Benchmark (Class-Name-Aligned)")
        lines.append("")
        lines.append(
            "> Both models are evaluated on the exact same labeled images with canonical class-name "
            "alignment (e.g. COCO `motorcycle` ↔ dataset `motorcycle`) so Stock YOLO26 and Trained YOLO26 "
            "are compared apples-to-apples."
        )
        lines.append("")

        for split_name, pair in dataset_results.items():
            s = pair["stock"]
            t = pair["trained"]
            lines.append(f"### Split: `{split_name}` ({t['images']} images, {t['gt_boxes']} ground-truth boxes)")
            lines.append("")
            lines.append("| Metric | Stock YOLO26 | Trained YOLO26 | Δ (Trained − Stock) |")
            lines.append("| :--- | ---: | ---: | ---: |")
            lines.append(f"| **mAP@50** | {s['mAP50']*100:.2f}% | **{t['mAP50']*100:.2f}%** | {fmt_delta(t['mAP50'], s['mAP50'], pct=True)} |")
            lines.append(f"| **mAP@50-95** | {s['mAP50_95']*100:.2f}% | **{t['mAP50_95']*100:.2f}%** | {fmt_delta(t['mAP50_95'], s['mAP50_95'], pct=True)} |")
            lines.append(f"| **Precision (conf≥{meta['conf']})** | {s['precision']*100:.2f}% | **{t['precision']*100:.2f}%** | {fmt_delta(t['precision'], s['precision'], pct=True)} |")
            lines.append(f"| **Recall (conf≥{meta['conf']})** | {s['recall']*100:.2f}% | **{t['recall']*100:.2f}%** | {fmt_delta(t['recall'], s['recall'], pct=True)} |")
            lines.append(f"| **F1-Score** | {s['f1']*100:.2f}% | **{t['f1']*100:.2f}%** | {fmt_delta(t['f1'], s['f1'], pct=True)} |")
            lines.append(f"| **True Positives (TP)** | {s['tp']} | {t['tp']} | {t['tp'] - s['tp']:+d} |")
            lines.append(f"| **False Positives (FP)** | {s['fp']} | {t['fp']} | {t['fp'] - s['fp']:+d} |")
            lines.append(f"| **False Negatives (FN / Missed)** | {s['fn']} | {t['fn']} | {t['fn'] - s['fn']:+d} |")
            lines.append(f"| **Mean Matched IoU** | {s['mean_iou']:.3f} | {t['mean_iou']:.3f} | {t['mean_iou'] - s['mean_iou']:+.3f} |")
            lines.append(f"| **Mean Confidence** | {s['mean_conf']:.3f} | {t['mean_conf']:.3f} | {t['mean_conf'] - s['mean_conf']:+.3f} |")
            lines.append(f"| **Avg Inference Time** | {s['avg_ms']:.2f} ms | {t['avg_ms']:.2f} ms | {t['avg_ms'] - s['avg_ms']:+.2f} ms |")
            lines.append("")

            # Per-class breakdown
            all_classes = sorted(t["per_class"].keys())
            if all_classes:
                lines.append(f"#### Per-Class Breakdown (`{split_name}`)")
                lines.append("")
                lines.append("| Class | GT | Stock P / R / F1 / AP@50 | Trained P / R / F1 / AP@50 | AP@50 Δ |")
                lines.append("| :--- | ---: | ---: | ---: | ---: |")
                for cls in all_classes:
                    sc = s["per_class"][cls]
                    tc = t["per_class"][cls]
                    s_str = f"{sc['precision']*100:.1f}% / {sc['recall']*100:.1f}% / {sc['f1']*100:.1f}% / {sc['ap50']*100:.1f}%"
                    t_str = f"{tc['precision']*100:.1f}% / {tc['recall']*100:.1f}% / {tc['f1']*100:.1f}% / {tc['ap50']*100:.1f}%"
                    lines.append(f"| `{cls}` | {tc['gt']} | {s_str} | **{t_str}** | {fmt_delta(tc['ap50'], sc['ap50'], pct=True)} |")
                lines.append("")

        if official_val_results:
            lines.append("### Official Ultralytics `model.val()` Summary (Trained Model)")
            lines.append("")
            lines.append("| Split | Precision | Recall | mAP@50 | mAP@50-95 |")
            lines.append("| :--- | ---: | ---: | ---: | ---: |")
            for split_name, vals in official_val_results.items():
                lines.append(
                    f"| `{split_name}` | {vals['precision']*100:.2f}% | {vals['recall']*100:.2f}% | "
                    f"**{vals['mAP50']*100:.2f}%** | **{vals['mAP50_95']*100:.2f}%** |"
                )
            lines.append("")
    else:
        lines.append("## 1. Ground-Truth Dataset Benchmark")
        lines.append("")
        lines.append("_Skipped: `./dataset/data.yaml` not found. Download dataset with `bash download_dataset.sh` to include mAP/Precision/Recall ground-truth evaluation._")
        lines.append("")

    # 2. Real Traffic Video Benchmark
    if video_results:
        lines.append("## 2. Real Traffic Video Benchmark")
        lines.append("")

        # Aggregate totals across all videos
        tot_frames = sum(v["sampled_frames"] for v in video_results)
        tot_s_det = sum(v["stock"]["total_detections"] for v in video_results)
        tot_t_det = sum(v["trained"]["total_detections"] for v in video_results)
        tot_s_small = sum(v["stock"]["small_count"] for v in video_results)
        tot_t_small = sum(v["trained"]["small_count"] for v in video_results)
        tot_s_hc = sum(v["stock"]["high_conf_count"] for v in video_results)
        tot_t_hc = sum(v["trained"]["high_conf_count"] for v in video_results)
        avg_s_ms = float(np.mean([v["stock"]["avg_ms"] for v in video_results]))
        avg_t_ms = float(np.mean([v["trained"]["avg_ms"] for v in video_results]))
        avg_s_conf = float(np.mean([v["stock"]["mean_conf"] for v in video_results]))
        avg_t_conf = float(np.mean([v["trained"]["mean_conf"] for v in video_results]))
        tot_matched = sum(v["agreement"]["matched_both"] for v in video_results)
        tot_only_s = sum(v["agreement"]["only_stock"] for v in video_results)
        tot_only_t = sum(v["agreement"]["only_trained"] for v in video_results)

        lines.append(f"### Overall Video Summary ({len(video_results)} videos, {tot_frames} sampled frames)")
        lines.append("")
        lines.append("| Metric | Stock YOLO26 | Trained YOLO26 | Δ (Trained − Stock) |")
        lines.append("| :--- | ---: | ---: | ---: |")
        lines.append(f"| **Total Detections** | {tot_s_det} | **{tot_t_det}** | {tot_t_det - tot_s_det:+d} |")
        lines.append(f"| **Avg Detections / Frame** | {tot_s_det/max(1,tot_frames):.2f} | **{tot_t_det/max(1,tot_frames):.2f}** | {(tot_t_det-tot_s_det)/max(1,tot_frames):+.2f} |")
        lines.append(f"| **High-Confidence Detections (≥0.50)** | {tot_s_hc} | **{tot_t_hc}** | {tot_t_hc - tot_s_hc:+d} |")
        lines.append(f"| **Small Objects (<32×32 px)** | {tot_s_small} | **{tot_t_small}** | {tot_t_small - tot_s_small:+d} |")
        lines.append(f"| **Mean Confidence** | {avg_s_conf:.3f} | **{avg_t_conf:.3f}** | {avg_t_conf - avg_s_conf:+.3f} |")
        lines.append(f"| **Avg Latency (ms/frame)** | {avg_s_ms:.2f} ms | **{avg_t_ms:.2f} ms** | {avg_t_ms - avg_s_ms:+.2f} ms |")
        lines.append(f"| **Throughput (FPS)** | {1000.0/avg_s_ms if avg_s_ms>0 else 0:.1f} FPS | **{1000.0/avg_t_ms if avg_t_ms>0 else 0:.1f} FPS** | {(1000.0/avg_t_ms - 1000.0/avg_s_ms) if avg_s_ms>0 and avg_t_ms>0 else 0:+.1f} FPS |")
        lines.append(f"| **Agreement (IoU≥0.50)** | Matched Both: **{tot_matched}** | Only Trained: **{tot_only_t}** | Only Stock: **{tot_only_s}** |")
        lines.append("")

        # Per-video detailed tables
        for vr in video_results:
            s = vr["stock"]
            t = vr["trained"]
            ag = vr["agreement"]
            lines.append(
                f"### Video: `{vr['video']}` ({vr['resolution']} @ {vr['source_fps']} FPS, "
                f"{vr['sampled_frames']} sampled / {vr['total_video_frames']} total frames)"
            )
            lines.append("")
            lines.append("| Metric | Stock YOLO26 | Trained YOLO26 | Δ |")
            lines.append("| :--- | ---: | ---: | ---: |")
            lines.append(f"| Total Detections | {s['total_detections']} | **{t['total_detections']}** | {t['total_detections'] - s['total_detections']:+d} |")
            lines.append(f"| Avg / Max per Frame | {s['avg_per_frame']:.2f} / {s['max_per_frame']} | **{t['avg_per_frame']:.2f} / {t['max_per_frame']}** | {t['avg_per_frame'] - s['avg_per_frame']:+.2f} |")
            lines.append(f"| Confidence (Mean / Median / P95) | {s['mean_conf']:.3f} / {s['median_conf']:.3f} / {s['p95_conf']:.3f} | **{t['mean_conf']:.3f} / {t['median_conf']:.3f} / {t['p95_conf']:.3f}** | {t['mean_conf'] - s['mean_conf']:+.3f} |")
            lines.append(f"| High-Conf Detections (≥0.50) | {s['high_conf_count']} | **{t['high_conf_count']}** | {t['high_conf_count'] - s['high_conf_count']:+d} |")
            lines.append(f"| Size Breakdown (Small / Med / Large) | {s['small_count']} / {s['medium_count']} / {s['large_count']} | **{t['small_count']} / {t['medium_count']} / {t['large_count']}** | Small: {t['small_count'] - s['small_count']:+d} |")
            lines.append(f"| Latency (Mean / P95) & FPS | {s['avg_ms']:.2f} / {s['p95_ms']:.2f} ms ({s['fps']:.1f} FPS) | **{t['avg_ms']:.2f} / {t['p95_ms']:.2f} ms ({t['fps']:.1f} FPS)** | {t['avg_ms'] - s['avg_ms']:+.2f} ms |")
            lines.append(f"| Box Agreement (IoU≥0.50) | Matched: {ag['matched_both']} (IoU {ag['mean_pair_iou']:.3f}) | Only Trained: {ag['only_trained']} | Only Stock: {ag['only_stock']} |")
            lines.append("")

            # Per-class counts on this video
            cls_keys = sorted(set(s["class_counts"].keys()) | set(t["class_counts"].keys()))
            if cls_keys:
                lines.append("| Class (Canonical) | Stock Count | Trained Count | Δ |")
                lines.append("| :--- | ---: | ---: | ---: |")
                for ck in cls_keys:
                    sc = s["class_counts"].get(ck, 0)
                    tc = t["class_counts"].get(ck, 0)
                    lines.append(f"| `{ck}` | {sc} | **{tc}** | {tc - sc:+d} |")
                lines.append("")

            if vr["saved_images"]:
                lines.append("#### Saved Side-by-Side Snapshots")
                lines.append("")
                for img in vr["saved_images"]:
                    lines.append(
                        f"- **{img['tag']}** (Frame `{img['frame']}` — Stock: {img['stock_count']} vs Trained: {img['trained_count']}): "
                        f"`{img['rel_path']}`"
                    )
                    lines.append(f"  ![{vr['video']} {img['tag']} frame {img['frame']}]({img['rel_path']})")
                lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    trained_path = resolve_trained_weights(args.trained)
    if trained_path is None:
        raise FileNotFoundError(
            "Could not find trained weights (`weights/best.pt`). "
            "Run `python train.py` first or pass `--trained /path/to/best.pt`."
        )

    device_arg, device_desc = get_device_info()
    precision_kwargs = get_precision_kwargs(use_cuda=(device_arg != "cpu"))
    vehicle_only_coco = not args.all_coco_classes

    print("=" * 68)
    print("VDAT Model Evaluation & Comparison Benchmark")
    print("=" * 68)
    print(f"Device:         {device_desc} | precision={precision_kwargs or 'FP32'}")
    print(f"Stock Model:    {args.stock}")
    print(f"Trained Model:  {trained_path}")
    print(f"Settings:       imgsz={args.imgsz}, conf={args.conf}, video_samples={args.video_samples}")
    print("=" * 68)

    stock_model = YOLO(args.stock)
    trained_model = YOLO(str(trained_path))

    # Warm up both models so timing metrics are clean
    dummy = np.zeros((480, 854, 3), dtype=np.uint8)
    run_model_detections(stock_model, dummy, conf=args.conf, imgsz=args.imgsz, vehicle_only_coco=vehicle_only_coco, precision_kwargs=precision_kwargs)
    run_model_detections(trained_model, dummy, conf=args.conf, imgsz=args.imgsz, vehicle_only_coco=False, precision_kwargs=precision_kwargs)

    # 1. Ground-truth dataset benchmark (if data.yaml exists)
    dataset_results: Dict[str, Any] = {}
    official_val_results: Dict[str, Any] = {}
    data_yaml_path = resolve_data_yaml(args.data)
    if data_yaml_path is not None and data_yaml_path.exists():
        data_yaml, cfg = normalize_data_yaml(str(data_yaml_path))
        ds_root = Path(cfg["path"])
        ds_names = parse_names_dict(cfg.get("names", {}))

        for split_name in ("test", "val"):
            rel = cfg.get(split_name)
            if not rel or not (ds_root / rel).is_dir():
                continue
            print(f"\n[1/2] Evaluating Stock vs Trained on labeled '{split_name}' split...")
            stock_eval = evaluate_aligned_dataset_split(
                stock_model,
                ds_root,
                rel,
                ds_names,
                imgsz=args.imgsz,
                conf_oper=args.conf,
                precision_kwargs=precision_kwargs,
                is_stock_coco=True,
            )
            trained_eval = evaluate_aligned_dataset_split(
                trained_model,
                ds_root,
                rel,
                ds_names,
                imgsz=args.imgsz,
                conf_oper=args.conf,
                precision_kwargs=precision_kwargs,
                is_stock_coco=False,
            )
            dataset_results[split_name] = {"stock": stock_eval, "trained": trained_eval}
            print(
                f"  [{split_name}] Stock mAP@50={stock_eval['mAP50']*100:.2f}% (P={stock_eval['precision']*100:.1f}%, R={stock_eval['recall']*100:.1f}%) "
                f"vs Trained mAP@50={trained_eval['mAP50']*100:.2f}% (P={trained_eval['precision']*100:.1f}%, R={trained_eval['recall']*100:.1f}%)"
            )

            # Also run official Ultralytics .val() for trained checkpoint
            try:
                val_res = trained_model.val(
                    data=str(data_yaml),
                    split=split_name,
                    imgsz=args.imgsz,
                    device=device_arg,
                    plots=False,
                    verbose=False,
                )
                official_val_results[split_name] = {
                    "precision": float(val_res.box.mp),
                    "recall": float(val_res.box.mr),
                    "mAP50": float(val_res.box.map50),
                    "mAP50_95": float(val_res.box.map),
                }
            except Exception as exc:
                print(f"  Warning: official val() on '{split_name}' skipped: {exc}")

    # 2. Video benchmark
    videos = [Path(v) for v in args.videos] if args.videos else list_test_videos(ROOT_DIR)
    out_path = Path(args.out).resolve()
    frames_out_dir = out_path.parent / "frames"
    video_results: List[Dict[str, Any]] = []

    for vp in videos:
        if not vp.exists():
            print(f"  Skipping missing video: {vp}")
            continue
        print(f"\n[2/2] Benchmarking video '{vp.name}' ({args.video_samples} sampled frames)...")
        vr = evaluate_video(
            vp,
            stock_model,
            trained_model,
            num_samples=args.video_samples,
            conf=args.conf,
            imgsz=args.imgsz,
            vehicle_only_coco=vehicle_only_coco,
            precision_kwargs=precision_kwargs,
            frames_out_dir=frames_out_dir,
        )
        video_results.append(vr)
        print(
            f"  [{vp.name}] Stock={vr['stock']['total_detections']} det ({vr['stock']['avg_ms']:.1f}ms) | "
            f"Trained={vr['trained']['total_detections']} det ({vr['trained']['avg_ms']:.1f}ms) | "
            f"Only Trained={vr['agreement']['only_trained']}, Only Stock={vr['agreement']['only_stock']}"
        )

    meta = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_desc,
        "precision": precision_kwargs or "FP32",
        "stock_model": args.stock,
        "trained_model": str(trained_path),
        "trained_classes": parse_names_dict(getattr(trained_model, "names", {})),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "vehicle_only_coco": vehicle_only_coco,
    }

    write_markdown_report(out_path, meta, dataset_results, official_val_results, video_results)

    # Also save structured JSON alongside Markdown for programmatic analysis
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "dataset": dataset_results,
                "official_val": official_val_results,
                "videos": video_results,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 68)
    print(f"Markdown report saved -> {out_path}")
    print(f"Raw JSON stats saved  -> {json_path}")
    print(f"Side-by-side frames   -> {frames_out_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
