"""
评测脚本：支持检测（detect）和分割（segment）模式，自动从 data.yaml 识别。

预测结果和 GT 标注都先转换为 class name，在名称空间统一比较，
无需关心不同模型的 class index 差异。

用法:
  python eval_detect.py --data <data.yaml>                   # 评测原始 COCO 模型
  python eval_detect.py --data <data.yaml> --model best.pt   # 评测微调后模型
"""

import argparse
import sys
import yaml
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

IOU_THRESH  = 0.5
MASK_SIZE   = 160   # 计算 mask IoU 时的归一化分辨率
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default=str(Path(__file__).parent / "models/yolo26x.pt"), help="模型权重路径")
    p.add_argument("--data",   required=True,               help="数据集配置文件（data.yaml）")
    p.add_argument("--imgsz",  type=int,   default=1920,     help="推理图片尺寸")
    p.add_argument("--conf",   type=float, default=0.5,    help="置信度阈值")
    p.add_argument("--device", default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    p.add_argument("--limit",  type=int,   default=0,       help="最多评测多少张图片（0 = 全部）")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


# ── 标注加载 ──────────────────────────────────────────────────────────────────

def load_labels_detect(label_dir: Path, idx_to_name: dict) -> dict:
    """检测模式：{stem: [(class_name, cx, cy, w, h), ...]}"""
    labels = {}
    for f in label_dir.glob("*.txt"):
        items = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls_idx = int(parts[0])
                name = idx_to_name.get(cls_idx)
                if name:
                    items.append((name, *map(float, parts[1:])))
        labels[f.stem] = items
    return labels


def load_labels_seg(label_dir: Path, idx_to_name: dict) -> dict:
    """分割模式：{stem: [(class_name, [x1,y1,x2,y2,...]), ...]}（归一化坐标）"""
    labels = {}
    for f in label_dir.glob("*.txt"):
        items = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 7:   # 至少 class + 3 个点（6 个坐标）
                continue
            cls_idx = int(parts[0])
            name = idx_to_name.get(cls_idx)
            if name:
                coords = list(map(float, parts[1:]))
                items.append((name, coords))
        labels[f.stem] = items
    return labels


# ── IoU 工具 ──────────────────────────────────────────────────────────────────

def iou_box(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _poly_to_mask(coords_norm, size=MASK_SIZE):
    """归一化多边形坐标 → 二值 mask（size×size）"""
    pts = np.array(coords_norm, dtype=np.float32).reshape(-1, 2)
    pts = (pts * size).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def iou_mask(coords_a, coords_b):
    """两个归一化多边形的 mask IoU"""
    ma = _poly_to_mask(coords_a)
    mb = _poly_to_mask(coords_b)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / union if union > 0 else 0.0


# ── 评测主逻辑 ────────────────────────────────────────────────────────────────

def xywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    return (cx - w/2)*img_w, (cy - h/2)*img_h, (cx + w/2)*img_w, (cy + h/2)*img_h


def compute_ap(precisions, recalls):
    """AP（11-point interpolation）"""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0)
        ap += p / 11
    return ap


def evaluate_detect(model, val_img_dir, val_label_dir, args, gt_names):
    idx_to_name  = {i: name for i, name in enumerate(gt_names)}
    gt_labels    = load_labels_detect(val_label_dir, idx_to_name)
    model_names  = model.names
    eval_names   = sorted(gt_names)

    detections = defaultdict(list)
    gt_counts  = defaultdict(int)

    img_paths = sorted(p for p in val_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    print(f"标注文件数: {len(gt_labels)}  评测图片数: {len(img_paths)}")

    for img_path in tqdm(img_paths, unit="img"):
        stem = img_path.stem
        gts  = gt_labels.get(stem, [])

        gt_by_name = defaultdict(list)
        for (name, cx, cy, w, h) in gts:
            gt_counts[name] += 1
            gt_by_name[name].append((cx, cy, w, h))

        result  = model.predict(str(img_path), imgsz=args.imgsz,
                                conf=args.conf, device=args.device or None,
                                verbose=False, save=False)[0]
        img_h, img_w = result.orig_shape

        preds_by_name = defaultdict(list)
        for box in result.boxes:
            name = model_names.get(int(box.cls.item()))
            if name not in eval_names:
                continue
            preds_by_name[name].append((float(box.conf.item()), box.xyxy[0].tolist()))

        for name in eval_names:
            gt_boxes = [xywh_to_xyxy(cx, cy, w, h, img_w, img_h)
                        for (cx, cy, w, h) in gt_by_name[name]]
            matched  = [False] * len(gt_boxes)
            for conf, pred_xyxy in sorted(preds_by_name[name], key=lambda x: -x[0]):
                best_iou, best_j = 0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    v = iou_box(pred_xyxy, gt_box)
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_iou >= IOU_THRESH and best_j >= 0
                if is_tp:
                    matched[best_j] = True
                detections[name].append((conf, int(is_tp)))

    _print_results(eval_names, detections, gt_counts)


def evaluate_seg(model, val_img_dir, val_label_dir, args, gt_names):
    idx_to_name  = {i: name for i, name in enumerate(gt_names)}
    gt_labels    = load_labels_seg(val_label_dir, idx_to_name)
    model_names  = model.names
    eval_names   = sorted(gt_names)

    detections = defaultdict(list)
    gt_counts  = defaultdict(int)

    img_paths = sorted(p for p in val_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    print(f"标注文件数: {len(gt_labels)}  评测图片数: {len(img_paths)}")

    for img_path in tqdm(img_paths, unit="img"):
        stem = img_path.stem
        gts  = gt_labels.get(stem, [])

        gt_by_name = defaultdict(list)
        for (name, coords) in gts:
            gt_counts[name] += 1
            gt_by_name[name].append(coords)

        result = model.predict(str(img_path), imgsz=args.imgsz,
                               conf=args.conf, device=args.device or None,
                               verbose=False, save=False)[0]

        preds_by_name = defaultdict(list)
        if result.masks is not None:
            for i, box in enumerate(result.boxes):
                name = model_names.get(int(box.cls.item()))
                if name not in eval_names:
                    continue
                # xyn: 归一化多边形坐标，flat list
                coords = result.masks.xyn[i].flatten().tolist()
                preds_by_name[name].append((float(box.conf.item()), coords))

        for name in eval_names:
            gt_polys = gt_by_name[name]
            matched  = [False] * len(gt_polys)
            for conf, pred_coords in sorted(preds_by_name[name], key=lambda x: -x[0]):
                best_iou, best_j = 0, -1
                for j, gt_coords in enumerate(gt_polys):
                    if matched[j]:
                        continue
                    v = iou_mask(pred_coords, gt_coords)
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_iou >= IOU_THRESH and best_j >= 0
                if is_tp:
                    matched[best_j] = True
                detections[name].append((conf, int(is_tp)))

    _print_results(eval_names, detections, gt_counts)


def _print_results(eval_names, detections, gt_counts):
    print(f"\n{'类别':<18} {'GT':>6} {'Pred':>6} {'P':>8} {'R':>8} {'mAP50':>8}")
    print("-" * 60)
    aps = []
    for name in eval_names:
        dets = sorted(detections[name], key=lambda x: -x[0])
        n_gt = gt_counts[name]
        if not dets or n_gt == 0:
            print(f"{name:<18} {n_gt:>6} {len(dets):>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        tp_cum = np.cumsum([d[1] for d in dets])
        fp_cum = np.cumsum([1 - d[1] for d in dets])
        prec   = tp_cum / (tp_cum + fp_cum)
        rec    = tp_cum / n_gt
        ap     = compute_ap(prec.tolist(), rec.tolist())
        aps.append(ap)
        print(f"{name:<18} {n_gt:>6} {len(dets):>6} "
              f"{float(prec[-1]):>8.4f} {float(rec[-1]):>8.4f} {ap:>8.4f}")
    mean_ap = np.mean(aps) if aps else 0.0
    print("-" * 60)
    print(f"{'all':<18} {sum(gt_counts.values()):>6} {'':>6} {'':>8} {'':>8} {mean_ap:>8.4f}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def find_label_dir(data_root: Path, val_rel: str) -> Path:
    sub = Path(val_rel).name
    candidate = data_root / "labels" / sub
    if candidate.exists():
        return candidate
    return data_root / "labels"


def main():
    args = parse_args()

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    is_seg    = cfg.get("task") == "segment"
    mode_str  = "segment" if is_seg else "detect"

    data_root     = Path(cfg["path"]) if Path(cfg["path"]).is_absolute() \
                    else (Path(args.data).parent / cfg["path"]).resolve()
    val_img_dir   = data_root / cfg["val"]
    val_label_dir = find_label_dir(data_root, cfg["val"])
    gt_names      = cfg["names"]

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}  [{mode_str}]")
    print(f"  imgsz    {args.imgsz:<10}  conf     {args.conf:<10}  device  {args.device or 'auto'}")
    print(f"  limit    {args.limit or 'all'}")
    print("─" * 60, flush=True)

    model = YOLO(args.model)
    print(f"[   model] classes={list(model.names.values())}")
    print(f"[    data] gt_classes={gt_names}  label_dir={val_label_dir}")

    if is_seg:
        evaluate_seg(model, val_img_dir, val_label_dir, args, gt_names)
    else:
        evaluate_detect(model, val_img_dir, val_label_dir, args, gt_names)


if __name__ == "__main__":
    main()
