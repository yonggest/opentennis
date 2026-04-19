"""
第一阶段：球场检测 + 物体检测，结果保存为 COCO JSON。

用法：
    python detect.py -i <video>
    python detect.py -i <video> -o results/my.json -m models/yolo26x.pt -s models/court_seg.pt
输出：
    <video>.detected.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys

import cv2

from utils import video_info, iter_frames, save_coco
from court_detector import (CourtDetector, compute_H_from_kps,
                             CLEARANCE_BACK, CLEARANCE_SIDE)
from objects_detector import ObjectsDetector

# 缓冲区尺寸：取 ITF 标准的一半（适配俱乐部球场）
_FILTER_BACK   = CLEARANCE_BACK / 2   # 底线后方 3.20 m
_FILTER_SIDE   = CLEARANCE_SIDE / 2   # 侧线外侧 1.83 m
_FILTER_HEIGHT = 2.0                  # 球员最大站立高度（米）


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',        required=True,                help='输入视频路径')
    p.add_argument('-o', '--output',        default=None,                 help='输出 JSON 路径（默认：输入同名加 _detected）')
    p.add_argument('-m', '--object-model', default='models/yolo26x.pt',  help='物体检测模型路径（球员/球拍/球）')
    p.add_argument('-s', '--court-model',  default='models/court_seg.pt', help='球场分割模型路径')
    p.add_argument('-c', '--conf',         type=float, default=0.5,      help='检测置信度阈值')
    p.add_argument('-z', '--imgsz',        type=int,   default=1920,     help='推理图片尺寸')
    p.add_argument('-d', '--device',       default=None,                  help='推理设备：cpu / cuda / mps（默认自动）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '.detected.json'

    print("─" * 60)
    print(f"  input         {args.input}")
    print(f"  output        {output_path}")
    print(f"  object-model  {args.object_model}")
    print(f"  court-model   {args.court_model}")
    print(f"  conf          {args.conf:<10}  imgsz   {args.imgsz}")
    print(f"  device        {args.device or 'auto'}")
    print("─" * 60, flush=True)

    fps, width, height, n_frames = video_info(args.input)

    # ── 球场检测（仅第一帧）──────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    _, first_frame = cap.read()
    cap.release()

    court_detector = CourtDetector(seg_model=args.court_model)
    kps            = court_detector.predict(first_frame)

    # 计算缓冲区凸包，写入 JSON，供后续模块直接使用（无需重新依赖常量）
    H = compute_H_from_kps(kps)
    court_det      = CourtDetector.from_H(H)
    ground_hull    = court_det.get_clearance_hull(back=_FILTER_BACK, side=_FILTER_SIDE)
    volume_hull, vol_bottom_pts, vol_top_pts = court_det.get_clearance_volume_hull(
        (height, width), back=_FILTER_BACK, side=_FILTER_SIDE, height=_FILTER_HEIGHT)
    _, court_bottom_pts, court_top_pts = court_det.get_clearance_volume_hull(
        (height, width), back=_FILTER_BACK, side=0, height=_FILTER_HEIGHT)

    court = {
        'keypoints':        kps,
        'ground_hull':      ground_hull,
        'volume_hull':      volume_hull,
        'vol_bottom_pts':   vol_bottom_pts,
        'vol_top_pts':      vol_top_pts,
        'court_bottom_pts': court_bottom_pts,
        'court_top_pts':    court_top_pts,
    }

    # ── 物体检测（全部帧，全图推理）──────────────────────────────────────────
    obj_detector = ObjectsDetector(args.object_model, conf=args.conf, imgsz=args.imgsz,
                                    device=args.device)
    players, rackets, balls = obj_detector.run(
        iter_frames(args.input),
        total=n_frames,
    )

    video_rel = os.path.relpath(
        os.path.abspath(args.input),
        os.path.dirname(os.path.abspath(output_path)),
    )
    save_coco(width, height, players, rackets, balls, output_path,
              fps=fps, court=court, video=video_rel)

    n_players = sum(len(v) for v in players)
    n_rackets = sum(len(v) for v in rackets)
    n_balls   = sum(len(v) for v in balls)
    print("\n── 检测结果摘要 " + "─" * 44)
    print(f"  帧数          {n_frames}")
    print(f"  person        {n_players}  ({n_players/n_frames:.1f}/帧)")
    print(f"  tennis racket {n_rackets}  ({n_rackets/n_frames:.1f}/帧)")
    print(f"  sports ball   {n_balls}  ({n_balls/n_frames:.1f}/帧)")
    print(f"  输出          {output_path}")
    print("─" * 60)


if __name__ == '__main__':
    main()
