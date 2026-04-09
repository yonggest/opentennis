"""
第二阶段：读取原始视频和检测 JSON，完成球跟踪、遮罩、绘制，输出视频。

用法：
    python render.py -i <video> -j <video>.json
    python render.py -i <video> -j <video>.json -o output.mp4
输出：
    <video>_out.mp4（默认）或 -o 指定的路径
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from utils import read_video, load_detections, save_video, text_params
from ball_tracker import BallTracker

# ITF 标准球场宽度（m），与 court_detector.py 中的 COURT_W 相同
_COURT_W = 10.97

# 每条球轨迹按 track_id 循环取色（BGR）
_TRAJ_COLORS = [
    (0, 255, 255), (255, 100, 0), (180, 0, 255),
    (0, 255, 100), (255, 200, 0), (0, 100, 255),
]


def _traj_color(track_id):
    return _TRAJ_COLORS[track_id % len(_TRAJ_COLORS)]


def _px_per_meter(court_kps):
    """
    从 court_kps (ndarray, 14×2 展平) 估算像素/米比例。
    取远端底线（kps[0]→[1]）和近端底线（kps[2]→[3]）宽度的平均值，
    消除透视压缩带来的误差。两端真实宽度均为 _COURT_W。
    """
    kps      = court_kps.reshape(14, 2)
    far_ppm  = float(np.linalg.norm(kps[1] - kps[0])) / _COURT_W
    near_ppm = float(np.linalg.norm(kps[3] - kps[2])) / _COURT_W
    return (far_ppm + near_ppm) / 2.0


def apply_hull_mask(frames, valid_hull):
    """凸包外区域置黑。"""
    fh, fw = frames[0].shape[:2]
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(mask, [valid_hull], 255)
    total = len(frames)
    nw = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(frames):
        frame[mask == 0] = 0
        pct = (i + 1) * 100 // total
        print(f"[    mask] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    print(f"[    mask] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")


def draw_preview(frames, valid_hull, players, rackets, balls):
    """叠加球场轮廓、检测框、网球轨迹箭头和帧号。"""
    fh = frames[0].shape[0]
    scale, thick             = text_params(fh)
    scale_large, thick_large = text_params(fh, base_height=1080)
    margin = int(fh * 0.028)

    # 预计算各轨迹的全部中心点：{track_id: [(frame_idx, cx, cy), ...]}
    traj = defaultdict(list)
    for fi, frame_balls in enumerate(balls):
        for det in frame_balls:
            tid = det.get('track_id')
            if tid is not None:
                x1, y1, x2, y2 = det['bbox']
                traj[tid].append((fi, int((x1 + x2) / 2), int((y1 + y2) / 2)))

    total = len(frames)
    nw = len(str(total))
    t0 = time.time()

    for i, frame in enumerate(frames):
        cv2.polylines(frame, [valid_hull], True, (0, 255, 255), 2)

        for det in players[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if det.get('track_id') is not None:
                cv2.putText(frame, f"P{det['track_id']}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)

        for det in rackets[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        for tid, pts in traj.items():
            color   = _traj_color(tid)
            visible = [(cx, cy) for fi, cx, cy in pts if fi <= i]
            for k in range(len(visible) - 1):
                cv2.arrowedLine(frame, visible[k], visible[k + 1], color, 2, tipLength=0.4)

        for det in balls[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            tid   = det.get('track_id')
            color = _traj_color(tid) if tid is not None else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        cv2.putText(frame, str(i), (margin, fh - margin),
                    cv2.FONT_HERSHEY_SIMPLEX, scale_large * 1.5, (0, 255, 0), thick_large)

        pct = (i + 1) * 100 // total
        print(f"[    draw] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)

    print(f"[    draw] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True, help='原始输入视频路径')
    p.add_argument('-j', '--json',   required=True, help='detect.py 输出的检测 JSON 路径')
    p.add_argument('-o', '--output', default=None,  help='输出视频路径（默认：输入视频同名加 _out.mp4）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '_out.mp4'

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  json    {args.json}")
    print(f"  output  {output_path}")
    print("─" * 60, flush=True)

    frames, _  = read_video(args.input)
    fps, _, _, court_kps, valid_hull, players, rackets, balls = load_detections(args.json)

    balls = BallTracker.from_video(fps, _px_per_meter(court_kps)).run(balls)

    apply_hull_mask(frames, valid_hull)
    draw_preview(frames, valid_hull, players, rackets, balls)
    save_video(frames, output_path, fps=fps)


if __name__ == '__main__':
    main()
