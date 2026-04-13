"""
第二阶段：读取 detect.py 输出的全量检测 JSON，
进行球追踪和缓冲区过滤，输出供 render.py 使用的处理后 JSON。

用法：
    python parse.py -i <video>.json
    python parse.py -i <video>.json -o <video>_parsed.json
输出：
    <video>_parsed.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from utils import load_detections, save_coco
from ball_tracker import BallTracker
from court_detector import COURT_W as _COURT_W


_STATIC_BALL_THRESH_PX = 5.0   # 静止球判定阈值：帧间平均位移（像素）


def _px_per_meter(court_kps):
    """从 court_kps（展平 28 维）估算像素/米比例，取远近底线宽度的平均值。"""
    kps      = court_kps.reshape(14, 2)
    far_ppm  = float(np.linalg.norm(kps[1] - kps[0])) / _COURT_W
    near_ppm = float(np.linalg.norm(kps[3] - kps[2])) / _COURT_W
    return (far_ppm + near_ppm) / 2.0


def _in_hull(hull, x, y):
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def _bbox_overlaps_hull(hull, x1, y1, x2, y2):
    """bbox 的中心或任一角点在凸包内则返回 True。"""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for pt in [(cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        if _in_hull(hull, *pt):
            return True
    return False


def _filter_players(players, ground_hull):
    """返回 (kept, removed)，底部中心在地面缓冲区内为有效球员。"""
    kept, removed = [], []
    for frame in players:
        k, r = [], []
        for d in frame:
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            (k if _in_hull(ground_hull, cx, d['bbox'][3]) else r).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _filter_rackets(rackets, volume_hull):
    """返回 (kept, removed)，bbox 与缓冲区立方体凸包有重叠为有效球拍。"""
    kept, removed = [], []
    for frame in rackets:
        k, r = [], []
        for d in frame:
            (k if _bbox_overlaps_hull(volume_hull, *d['bbox']) else r).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _filter_balls(balls, volume_hull):
    """
    返回 (kept, removed)。
    静止球（轨迹平均帧间位移 < _STATIC_BALL_THRESH_PX）必须落在缓冲区内；
    运动球起点必须在缓冲区内，否则整条轨迹无效。
    无 track_id 的孤立检测视为静止球处理。
    """
    track_pts = defaultdict(list)
    for frame in balls:
        for d in frame:
            tid = d.get('track_id')
            if tid is not None:
                cx = (d['bbox'][0] + d['bbox'][2]) / 2
                cy = (d['bbox'][1] + d['bbox'][3]) / 2
                track_pts[tid].append((cx, cy))

    static_tracks  = set()
    invalid_tracks = set()
    for tid, pts in track_pts.items():
        if len(pts) < 2:
            static_tracks.add(tid)
            continue
        dists = [np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
                 for i in range(len(pts)-1)]
        if np.mean(dists) < _STATIC_BALL_THRESH_PX:
            static_tracks.add(tid)
        elif not _in_hull(volume_hull, pts[0][0], pts[0][1]):
            invalid_tracks.add(tid)

    kept, removed = [], []
    for frame in balls:
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            cx  = (d['bbox'][0] + d['bbox'][2]) / 2
            cy  = (d['bbox'][1] + d['bbox'][3]) / 2
            if tid in invalid_tracks:
                r.append(d)
            elif tid is None or tid in static_tracks:
                (k if _in_hull(volume_hull, cx, cy) else r).append(d)
            else:
                k.append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True, help='detect.py 输出的 JSON 路径')
    p.add_argument('-o', '--output', default=None,  help='输出 JSON 路径（默认：输入同名加 _parsed）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '_parsed.json'

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  output  {output_path}")
    print("─" * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)

    # 球追踪
    balls = BallTracker.from_video(fps, _px_per_meter(court['keypoints'])).run(balls)

    # 缓冲区过滤
    ground_hull = court['ground_hull']
    volume_hull = court['volume_hull']

    n_players_before = sum(len(f) for f in players)
    n_rackets_before = sum(len(f) for f in rackets)
    n_balls_before   = sum(len(f) for f in balls)
    players, players_inv = _filter_players(players, ground_hull)
    rackets, rackets_inv = _filter_rackets(rackets, volume_hull)
    balls,   balls_inv   = _filter_balls(balls, volume_hull)
    print(f"[  filter] players: {n_players_before} → {sum(len(f) for f in players)}")
    print(f"[  filter] rackets: {n_rackets_before} → {sum(len(f) for f in rackets)}")
    print(f"[  filter] balls:   {n_balls_before} → {sum(len(f) for f in balls)}")

    # 合并 valid/invalid，写入 valid 标记
    n = len(players)
    players_out = [[dict(d, valid=True)  for d in players[fi]] +
                   [dict(d, valid=False) for d in players_inv[fi]] for fi in range(n)]
    rackets_out = [[dict(d, valid=True)  for d in rackets[fi]] +
                   [dict(d, valid=False) for d in rackets_inv[fi]] for fi in range(n)]
    balls_out   = [[dict(d, valid=True)  for d in balls[fi]] +
                   [dict(d, valid=False) for d in balls_inv[fi]] for fi in range(n)]

    save_coco(width, height, players_out, rackets_out, balls_out,
              output_path, fps=fps, court=court)


if __name__ == '__main__':
    main()
