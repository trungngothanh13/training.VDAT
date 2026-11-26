import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent

# Standard COCO vehicle classes used by stock yolo26l.pt in VDAT
STOCK_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
STOCK_VEHICLE_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Canonical class synonyms to align Stock COCO labels <-> Custom Roboflow dataset labels
CLASS_SYNONYMS: Dict[str, str] = {
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "moto": "motorcycle",
    "scooter": "motorcycle",
    "moped": "motorcycle",
    "two-wheeler": "motorcycle",
    "2-wheeler": "motorcycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "car": "car",
    "cars": "car",
    "auto": "car",
    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "van": "car",
    "pickup": "car",
    "bus": "bus",
    "buses": "bus",
    "coach": "bus",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "trailer": "truck",
}

# Deterministic BGR colors per canonical class for crisp side-by-side overlays
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "motorcycle": (255, 165, 30),   # Vivid blue-orange contrast (BGR -> bright blue/cyan-ish -> let's use bright orange/gold)
    "car": (80, 220, 100),          # Emerald green
    "truck": (60, 215, 255),        # Amber / yellow
    "bus": (235, 110, 210),         # Magenta
    "bicycle": (245, 210, 60),      # Cyan
}

FALLBACK_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (255, 140, 40),
    (80, 220, 100),
    (60, 215, 255),
    (235, 110, 210),
    (245, 210, 60),
    (90, 90, 255),
    (180, 240, 120),
]


def canonical_class_name(name: str) -> str:
    cleaned = str(name).strip().lower().replace("_", " ").replace("-", " ")
    if cleaned in CLASS_SYNONYMS:
        return CLASS_SYNONYMS[cleaned]
    compact = cleaned.replace(" ", "")
    if compact in CLASS_SYNONYMS:
        return CLASS_SYNONYMS[compact]
    return cleaned


def color_for_class(class_name: str, class_id: int = 0) -> Tuple[int, int, int]:
    canon = canonical_class_name(class_name)
    if canon == "motorcycle":
        return (255, 130, 35)  # High-visibility sky blue/royal blue in BGR
    if canon in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[canon]
    return FALLBACK_PALETTE_BGR[int(class_id) % len(FALLBACK_PALETTE_BGR)]


def get_precision_kwargs(use_cuda: Optional[bool] = None) -> Dict[str, Any]:
    """Returns FP16 kwargs compatible with Ultralytics 8.4+ (`quantize=16`) or older (`half=True`), falling back to FP32 on GPUs prone to FP16 NaN overflow."""
    if use_cuda is None:
        try:
            import torch
            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
    if not use_cuda:
        return {}
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0).upper()
        cap = torch.cuda.get_device_capability(0)
        if cap < (7, 0) or any(tok in gpu_name for tok in ("GTX 16", "T500", "T600", "T1000", "T1200", "T2000")):
            return {}
    except Exception:
        pass
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except Exception:
        pass
    return {"half": True}


def get_device_info() -> Tuple[str, str]:
    """Returns (device_arg, human_readable_description)."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "0", f"CUDA: {gpu} ({vram_gb:.1f} GB VRAM)"
    except Exception:
        pass
    return "cpu", "CPU"


def normalize_data_yaml(yaml_path: str) -> Tuple[Path, Dict[str, Any]]:
    """
    Rewrites Roboflow-exported data.yaml so `path` is an absolute path to the dataset
    directory and `train`/`val`/`test` point reliably to existing image folders.
    """
    p = Path(yaml_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ds_root = p.parent.resolve()
    cfg["path"] = str(ds_root)

    split_candidates = {
        "train": ["train/images", "images/train"],
        "val": ["valid/images", "val/images", "images/val", "images/valid"],
        "test": ["test/images", "images/test"],
    }

    for key, candidates in split_candidates.items():
        found = None
        for rel in candidates:
            if (ds_root / rel).is_dir():
                found = rel
                break
        if found is not None:
            cfg[key] = found
        elif key == "test" and "test" in cfg:
            # Remove broken test path if dataset has no test split
            raw_test = str(cfg.get("test", "")).lstrip("../")
            if not (ds_root / raw_test).is_dir():
                cfg.pop("test", None)

    # Normalize names into a clean dict {0: name0, ...} or list preserved
    if isinstance(cfg.get("names"), list):
        cfg["nc"] = len(cfg["names"])
    elif isinstance(cfg.get("names"), dict):
        cfg["nc"] = len(cfg["names"])

    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return p, cfg


def parse_names_dict(names_obj: Any) -> Dict[int, str]:
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, list):
        return {idx: str(v) for idx, v in enumerate(names_obj)}
    return {}


def resolve_data_yaml(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Finds dataset/data.yaml or root data.yaml (if Roboflow zip was unzipped directly in project root)."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([
        ROOT_DIR / "dataset" / "data.yaml",
        ROOT_DIR / "data.yaml",
    ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def resolve_trained_weights(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Finds the trained YOLO weights file in standard output locations."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend([
        ROOT_DIR / "weights" / "best.pt",
        ROOT_DIR / "best.pt",
    ])
    # Also check newest runs/**/weights/best.pt
    run_weights = sorted(
        ROOT_DIR.glob("runs/**/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(run_weights)

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def list_test_videos(search_dir: Optional[Path] = None) -> List[Path]:
    base = search_dir or ROOT_DIR
    found: List[Path] = []
    seen = set()
    for pattern in ("*.mp4", "testvideos/*.mp4", "videos/*.mp4", "test_videos/*.mp4"):
        for p in sorted(glob.glob(str(base / pattern))):
            rp = Path(p).resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(rp)
    return found


def run_model_detections(
    model,
    frame_bgr: np.ndarray,
    conf: float = 0.30,
    imgsz: int = 864,
    vehicle_only_coco: bool = False,
    precision_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Runs single-frame YOLO prediction and returns (detections_list, infer_ms).
    Each detection: {xyxy: [x1,y1,x2,y2], conf: float, class_id: int, class_name: str, canon_name: str, area: float}
    """
    import time

    kwargs = dict(precision_kwargs or {})
    if vehicle_only_coco:
        kwargs["classes"] = sorted(STOCK_VEHICLE_CLASS_IDS)

    t0 = time.perf_counter()
    results = model.predict(
        frame_bgr,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        **kwargs,
    )
    infer_ms = (time.perf_counter() - t0) * 1000.0

    names = parse_names_dict(getattr(model, "names", {}))
    detections: List[Dict[str, Any]] = []
    res = results[0]
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for box, score, cid in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cname = names.get(int(cid), str(int(cid)))
            detections.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": float(score),
                "class_id": int(cid),
                "class_name": cname,
                "canon_name": canonical_class_name(cname),
                "area": w * h,
            })

    return detections, infer_ms


def draw_detections_bgr(
    frame_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> np.ndarray:
    """Draws clean, legible bounding boxes + optional top header bar on a frame copy."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = color_for_class(det["class_name"], det["class_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']} {det['conf']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = ty1 + th + baseline + 4
        cv2.rectangle(out, (x1, ty1), (min(w - 1, x1 + tw + 6), ty2), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, ty2 - baseline - 2),
            font,
            scale,
            (12, 12, 12),
            thickness,
            cv2.LINE_AA,
        )

    if title:
        bar_h = 34
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
        cv2.putText(
            out,
            title,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 248, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def make_side_by_side_bgr(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    divider_width: int = 4,
) -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    divider = np.full((h, divider_width, 3), (45, 55, 72), dtype=np.uint8)
    return np.hstack([left_bgr, divider, right_bgr])


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU matrix [N, M] between xyxy box arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)
