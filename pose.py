"""
第四点五阶段：读取 parse.py 输出的 JSON，逐帧运行 YOLO 姿态估计，
将关键点附加到匹配的有效球员检测上，输出新的 COCO JSON。

用法：
    python pose.py -i <video>_parsed.json -v <video>
    python pose.py -i <video>_parsed.json -v <video> -o <video>_posed.json
输出：
    <video>_posed.json（默认：将 _parsed 后缀替换为 _posed）
"""

import argparse
import os
import sys
import time

import numpy as np
from ultralytics import YOLO

from utils import iter_frames, load_detections, save_coco, video_info


def _iou(a, b):
    """计算两个 [x1,y1,x2,y2] bbox 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True,
                   help='parse.py 输出的 JSON 路径（_parsed.json）')
    p.add_argument('-v', '--video',  required=True,
                   help='原始视频路径')
    p.add_argument('-m', '--model',  default='models/yolo26x-pose.pt',
                   help='YOLO 姿态估计模型路径')
    p.add_argument('-o', '--output', default=None,
                   help='输出 JSON 路径（默认：将 _parsed 替换为 _posed）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def _default_output(input_path):
    base = input_path
    if base.endswith('_parsed.json'):
        base = base[:-len('_parsed.json')] + '_posed.json'
    else:
        root, _ = os.path.splitext(base)
        base = root + '_posed.json'
    return base


def main():
    args = parse_args()
    output_path = args.output or _default_output(args.input)

    print("─" * 60)
    print(f"  input       {args.input}")
    print(f"  video       {args.video}")
    print(f"  model       {args.model}")
    print(f"  output      {output_path}")
    print("─" * 60, flush=True)

    # ── 加载检测 JSON ──────────────────────────────────────────────────────────
    fps, width, height, court, players, rackets, balls = \
        load_detections(args.input)

    # ── 加载姿态模型 ───────────────────────────────────────────────────────────
    model = YOLO(args.model)

    # ── 逐帧处理 ───────────────────────────────────────────────────────────────
    _, _, _, n_frames = video_info(args.video)
    frame_num_width = len(str(n_frames)) if n_frames else 6
    t0 = time.time()
    total_with_kps = 0

    for fi, frame in enumerate(iter_frames(args.video)):
        # 运行姿态估计
        results = model(frame, conf=0.3, verbose=False)

        # 收集本帧所有姿态检测的 bbox + keypoints
        pose_boxes = []   # list of [x1,y1,x2,y2]
        pose_kps   = []   # list of [[x,y,conf], ...] (17 points)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            kps_data = results[0].keypoints  # shape (N, 17, 3) or None
            if kps_data is not None:
                for bi in range(len(boxes)):
                    xyxy = boxes.xyxy[bi].cpu().numpy().tolist()
                    kp = kps_data.data[bi].cpu().numpy().tolist()  # 17 x 3
                    pose_boxes.append(xyxy)
                    pose_kps.append(kp)

        # 为每个有效球员匹配最佳姿态
        if fi < len(players):
            for det in players[fi]:
                if not det.get('valid', True):
                    continue
                best_iou  = -1.0
                best_idx  = -1
                for pi, pb in enumerate(pose_boxes):
                    iou = _iou(det['bbox'], pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = pi
                if best_idx >= 0 and best_iou >= 0.3:
                    det['keypoints'] = pose_kps[best_idx]
                    total_with_kps += 1

        count = fi + 1
        if n_frames:
            pct = count * 100 // n_frames
            print(f"[   pose] {count:>{frame_num_width}}/{n_frames} frames  ({pct:>3}%)",
                  end='\r', flush=True)
        else:
            print(f"[   pose] {count} frames", end='\r', flush=True)

    print(f"[   pose] {count:>{frame_num_width}}/{count} frames  (100%)  done: {time.time()-t0:>6.1f}s")
    print(f"[   pose] {total_with_kps} players with keypoints", flush=True)

    # ── 保存 ───────────────────────────────────────────────────────────────────
    save_coco(width, height, players, rackets, balls, output_path, fps=fps, court=court)


if __name__ == '__main__':
    main()
