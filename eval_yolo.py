"""
评测脚本：支持任意模型（原始 COCO、微调后等）。

预测结果和 GT 标注都先转换为 class name，在名称空间统一比较，
无需关心不同模型的 class index 差异。

用法:
  python eval_yolo.py --data <data.yaml>                   # 评测原始 COCO 模型
  python eval_yolo.py --data <data.yaml> --model best.pt   # 评测微调后模型
"""

import argparse
import sys
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

IOU_THRESH = 0.5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default=str(Path(__file__).parent / "models/yolo26x.pt"), help="模型权重路径")
    p.add_argument("--data",   required=True,               help="数据集配置文件（data.yaml）")
    p.add_argument("--imgsz",  type=int,   default=640,     help="推理图片尺寸")
    p.add_argument("--conf",   type=float, default=0.25,    help="置信度阈值")
    p.add_argument("--device", default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    p.add_argument("--limit",  type=int,   default=0,       help="最多评测多少张图片（0 = 全部）")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def load_labels(label_dir: Path, idx_to_name: dict[int, str]) -> dict:
    """读取 YOLO 格式 label 文件，返回 {stem: [(class_name, cx, cy, w, h), ...]}"""
    labels = {}
    for f in label_dir.glob("*.txt"):
        boxes = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls_idx = int(parts[0])
                name = idx_to_name.get(cls_idx)
                if name:
                    boxes.append((name, *map(float, parts[1:])))
        labels[f.stem] = boxes
    return labels


def find_label_dir(data_root: Path, val_rel: str) -> Path:
    sub = Path(val_rel).name
    candidate = data_root / "labels" / sub
    if candidate.exists():
        return candidate
    return data_root / "labels"


def xywh_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions, recalls):
    """AP（11-point interpolation）"""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max((precisions[i] for i, r in enumerate(recalls) if r >= t), default=0)
        ap += p / 11
    return ap


def evaluate(model, val_img_dir, val_label_dir, args, gt_names: list[str]):
    # GT: class index → name（来自 data.yaml）
    gt_idx_to_name = {i: name for i, name in enumerate(gt_names)}
    gt_labels = load_labels(val_label_dir, gt_idx_to_name)
    print(f"标注文件数: {len(gt_labels)}")

    # 模型自带的 class index → name 映射
    model_names: dict[int, str] = model.names   # e.g. {0:'person', 38:'tennis racket', ...}

    # 只评测 GT 中出现的类别
    eval_names = sorted(gt_names)

    detections = defaultdict(list)   # name → [(conf, is_tp)]
    gt_counts  = defaultdict(int)    # name → count

    img_paths = sorted(p for p in val_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        img_paths = img_paths[:args.limit]
    print(f"评测图片数: {len(img_paths)}")

    for img_path in tqdm(img_paths, unit="img"):
        stem = img_path.stem
        gts  = gt_labels.get(stem, [])

        gt_by_name = defaultdict(list)
        for (name, cx, cy, w, h) in gts:
            gt_counts[name] += 1
            gt_by_name[name].append((cx, cy, w, h))

        result = model.predict(str(img_path), imgsz=args.imgsz,
                               conf=args.conf, device=args.device or None,
                               verbose=False, save=False)[0]
        img_h, img_w = result.orig_shape

        preds_by_name = defaultdict(list)
        for box in result.boxes:
            raw_cls = int(box.cls.item())
            name    = model_names.get(raw_cls)
            if name not in eval_names:
                continue
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            preds_by_name[name].append((conf, xyxy))

        for name in eval_names:
            gt_boxes = [xywh_to_xyxy(cx, cy, w, h, img_w, img_h)
                        for (cx, cy, w, h) in gt_by_name[name]]
            matched  = [False] * len(gt_boxes)
            preds = sorted(preds_by_name[name], key=lambda x: -x[0])
            for conf, pred_xyxy in preds:
                best_iou, best_j = 0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    v = iou(pred_xyxy, gt_box)
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_iou >= IOU_THRESH and best_j >= 0
                if is_tp:
                    matched[best_j] = True
                detections[name].append((conf, int(is_tp)))

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
        ap = compute_ap(prec.tolist(), rec.tolist())
        aps.append(ap)
        print(f"{name:<18} {n_gt:>6} {len(dets):>6} "
              f"{float(prec[-1]):>8.4f} {float(rec[-1]):>8.4f} {ap:>8.4f}")

    mean_ap = np.mean(aps) if aps else 0.0
    print("-" * 60)
    print(f"{'all':<18} {sum(gt_counts.values()):>6} {'':>6} {'':>8} {'':>8} {mean_ap:>8.4f}")


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}")
    print(f"  imgsz    {args.imgsz:<10}  conf     {args.conf:<10}  device  {args.device or 'auto'}")
    print(f"  limit    {args.limit or 'all'}")
    print("─" * 60, flush=True)

    with open(args.data) as f:
        cfg = yaml.safe_load(f)

    data_root     = (Path(args.data).parent / cfg["path"]).resolve() if not Path(cfg["path"]).is_absolute() else Path(cfg["path"])
    val_img_dir   = data_root / cfg["val"]
    val_label_dir = find_label_dir(data_root, cfg["val"])
    gt_names      = cfg["names"]

    model = YOLO(args.model)

    print(f"[  model] classes={list(model.names.values())}")
    print(f"[   data] gt_classes={gt_names}  label_dir={val_label_dir}")

    evaluate(model, val_img_dir, val_label_dir, args, gt_names)


if __name__ == "__main__":
    main()
