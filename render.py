"""
第二阶段：读取检测 JSON，逐帧流式处理原始视频，输出注释视频。

用法：
    python render.py -i <video> -j <video>.json
    python render.py -i <video> -j <video>.json -o output.mp4
输出：
    <video>.mp4（默认，输入为 .mp4 时需用 -o 指定不同路径）

内存占用：恒定（单帧 + JSON）——适合超大视频文件（>10 GB）。
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from utils import video_info, iter_frames, open_video_writer, load_detections, text_params
from ball_tracker import BallTracker
from court_detector import MODEL_KPS_M, COURT_LINES, COURT_W as _COURT_W, NET_Y as _NET_Y


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


def _build_hull_mask(valid_hull, height, width):
    """预计算凸包掩膜，用于逐帧快速遮黑场外区域。"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [valid_hull], 255)
    return mask


def _compute_H_from_kps(court_kps):
    """从 14 个关键点反推单应矩阵 H（球场米坐标 → 图像像素）。"""
    kps_2d = court_kps.reshape(14, 2).astype(np.float32)
    H, _   = cv2.findHomography(MODEL_KPS_M, kps_2d)
    return H


def _project_line(H, x1, y1, x2, y2):
    """用 H 将一段球场米坐标线投影为图像像素坐标，返回 (pt1, pt2)。"""
    pts = cv2.perspectiveTransform(
        np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32), H)
    return tuple(pts[0, 0].astype(int)), tuple(pts[1, 0].astype(int))


def _draw_court_kps(frame, court_kps, H):
    """绘制 14 个球场关键点、球场线条两侧边缘及网线。"""
    color = (80, 200, 255)   # 淡橙黄，低调但与白色可区分

    if H is not None:
        # 每条球场线绘制两条边缘（内缘 + 外缘）
        for (p1, p2, lw_m) in COURT_LINES:
            half = lw_m / 2
            if p1[1] == p2[1]:   # 水平线 → 沿 y 方向偏移
                for dy in (-half, +half):
                    pt1, pt2 = _project_line(H, p1[0], p1[1]+dy, p2[0], p2[1]+dy)
                    cv2.line(frame, pt1, pt2, color, 1)
            else:                 # 垂直线 → 沿 x 方向偏移
                for dx in (-half, +half):
                    pt1, pt2 = _project_line(H, p1[0]+dx, p1[1], p2[0]+dx, p2[1])
                    cv2.line(frame, pt1, pt2, color, 1)

        # 网线
        pt1, pt2 = _project_line(H, 0, _NET_Y, _COURT_W, _NET_Y)
        cv2.line(frame, pt1, pt2, color, 1)

    # 关键点小圆
    kps = court_kps.reshape(14, 2).astype(int)
    for pt in kps:
        cv2.circle(frame, tuple(pt), 4, color, -1)


def _build_traj(balls):
    """预计算各轨迹的全部中心点：{track_id: [(frame_idx, cx, cy), ...]}。"""
    traj = defaultdict(list)
    for fi, frame_balls in enumerate(balls):
        for det in frame_balls:
            tid = det.get('track_id')
            if tid is not None:
                x1, y1, x2, y2 = det['bbox']
                traj[tid].append((fi, int((x1 + x2) / 2), int((y1 + y2) / 2)))
    return traj


def _draw_frame(frame, fi, court_kps, H, valid_hull, hull_mask,
                players, rackets, balls, traj,
                scale, thick, scale_large, thick_large, margin):
    """原地修改单帧：遮黑 → 绘制球场轮廓、检测框、球轨迹、帧号。"""
    # 遮黑场外区域
    frame[hull_mask == 0] = 0

    # 球场关键点 + 线条（含网线）
    _draw_court_kps(frame, court_kps, H)

    # 球场凸包轮廓
    cv2.polylines(frame, [valid_hull], True, (0, 255, 255), 2)

    # 球员框
    for det in players[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if det.get('track_id') is not None:
            cv2.putText(frame, f"P{det['track_id']}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)

    # 球拍框
    for det in rackets[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

    # 球轨迹（已出现的点）
    for tid, pts in traj.items():
        color   = _traj_color(tid)
        visible = [(cx, cy) for fj, cx, cy in pts if fj <= fi]
        for k in range(len(visible) - 1):
            cv2.arrowedLine(frame, visible[k], visible[k + 1], color, 2, tipLength=0.4)

    # 当前帧球框
    for det in balls[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        tid   = det.get('track_id')
        color = _traj_color(tid) if tid is not None else (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # 帧号
    fh = frame.shape[0]
    cv2.putText(frame, str(fi), (margin, fh - margin),
                cv2.FONT_HERSHEY_SIMPLEX, scale_large * 1.5, (0, 255, 0), thick_large)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True, help='原始输入视频路径')
    p.add_argument('-j', '--json',   required=True, help='detect.py 输出的检测 JSON 路径')
    p.add_argument('-o', '--output', default=None,  help='输出视频路径（默认：输入视频同名 .mp4）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '.mp4'
    if os.path.abspath(output_path) == os.path.abspath(args.input):
        print(f"错误: 输出路径与输入路径相同: {output_path}", file=sys.stderr)
        sys.exit(1)

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  json    {args.json}")
    print(f"  output  {output_path}")
    print("─" * 60, flush=True)

    # ── 预加载：JSON（小）＋ 球跟踪 ────────────────────────────────────────────
    fps, width, height, court_kps, valid_hull, players, rackets, balls = \
        load_detections(args.json)

    balls     = BallTracker.from_video(fps, _px_per_meter(court_kps)).run(balls)
    H         = _compute_H_from_kps(court_kps)
    hull_mask = _build_hull_mask(valid_hull, height, width)
    traj      = _build_traj(balls)

    scale, thick             = text_params(height)
    scale_large, thick_large = text_params(height, base_height=1080)
    margin = int(height * 0.028)

    # ── 单遍流式处理：读帧 → 处理 → 写帧 ─────────────────────────────────────
    _, _, _, n_frames = video_info(args.input)
    nw = len(str(n_frames)) if n_frames else 6
    t0 = time.time()
    count = 0

    with open_video_writer(output_path, fps, width, height) as pipe:
        for fi, frame in enumerate(iter_frames(args.input)):
            _draw_frame(frame, fi, court_kps, H, valid_hull, hull_mask,
                        players, rackets, balls, traj,
                        scale, thick, scale_large, thick_large, margin)
            pipe.write(frame.tobytes())
            count = fi + 1
            if n_frames:
                pct = count * 100 // n_frames
                print(f"[  render] {count:>{nw}}/{n_frames} frames  ({pct:>3}%)", end='\r', flush=True)
            else:
                print(f"[  render] {count} frames", end='\r', flush=True)

    print(f"[  render] {count:>{nw}}/{count} frames  (100%)  done: {time.time()-t0:>6.1f}s")
    print(f"[  render] saved → {output_path}", flush=True)


if __name__ == '__main__':
    main()
