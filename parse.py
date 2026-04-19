"""
第三阶段：读取 track.py 输出的 JSON（含 track_id），
进行缓冲区过滤，输出供 render.py 使用的处理后 JSON。

也可直接读取 detect.py 输出（跳过追踪阶段），此时球均视为无轨迹检测处理。

用法：
    python parse.py -i <video>.tracked.json
    python parse.py -i <video>.tracked.json -o <video>.parsed.json
输出：
    <video>.parsed.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from utils import load_detections, save_coco, propagate_video


_STATIC_BBOX_DIAG_PX = 20.0  # 静止球判定阈值：轨迹全局包围盒对角线（像素）
_STOPPED_TIMEOUT_S   = 3.0   # 运动球停止后多久开始标记无效（秒）
_STOPPED_MOVE_PX     = 15.0  # 判定"仍在移动"的最小位移（像素）



def _in_hull(hull, x, y):
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def _bboxes_overlap(a, b):
    """两个 [x1,y1,x2,y2] bbox 是否有交叠。"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _bbox_overlaps_hull(hull, x1, y1, x2, y2):
    """bbox 的中心或任一角点在凸包内则返回 True。"""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for pt in [(cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        if _in_hull(hull, *pt):
            return True
    return False


def _filter_players(players, left_player_wall, right_player_wall, ground_hull):
    """返回 (kept, removed)。
    按 track_id 分组，同时满足以下两个条件才认为是球员：
    1. 轨迹大部分（>50%）底部中心在 ground_hull（含缓冲区球场范围）内
    2. 轨迹大部分不落在双打侧线外（左墙 + 右墙合计 <= 50%）
    """
    track_stats = defaultdict(lambda: {'total': 0, 'in_ground': 0, 'out_side': 0})

    for frame in players:
        for d in frame:
            tid = d.get('track_id')
            if tid is None:
                continue
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            cy = d['bbox'][3]
            s = track_stats[tid]
            s['total'] += 1
            if _in_hull(ground_hull, cx, cy):
                s['in_ground'] += 1
            if _in_hull(left_player_wall, cx, cy) or _in_hull(right_player_wall, cx, cy):
                s['out_side'] += 1

    invalid_tracks = set()
    for tid, s in track_stats.items():
        total = s['total']
        majority_in_ground = s['in_ground'] / total > 0.5
        majority_out_side  = s['out_side']  / total > 0.5
        if not majority_in_ground or majority_out_side:
            invalid_tracks.add(tid)

    kept, removed = [], []
    for frame in players:
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            (r if tid in invalid_tracks else k).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _filter_rackets(rackets, volume_hull, valid_players):
    """返回 (kept, removed)。
    有效条件（两者同时满足）：
    1. bbox 与立方体缓冲区（volume hull）有交叠；
    2. bbox 与当前帧至少一个有效球员的 bbox 有交叠。
    """
    kept, removed = [], []
    for frame, players in zip(rackets, valid_players):
        k, r = [], []
        for d in frame:
            in_volume = _bbox_overlaps_hull(volume_hull, *d['bbox'])
            has_player = any(_bboxes_overlap(d['bbox'], p['bbox']) for p in players)
            (k if in_volume and has_player else r).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _make_wall_quads(vol_bottom_pts, vol_top_pts, img_height):
    """构造左右侧边墙延伸到天空的四边形 (4,1,2) float32。
    球起点落在墙面四边形内 → 出界。
    vol_bottom/top 顺序: [远左, 远右, 近右, 近左]
    """
    bpts = np.array(vol_bottom_pts, dtype=np.float64)
    tpts = np.array(vol_top_pts,    dtype=np.float64)
    fl_b, fr_b, nr_b, nl_b = bpts
    fl_t, fr_t, nr_t, nl_t = tpts
    sky_y = float(-img_height)

    def to_sky(p_b, p_t):
        dy = p_t[1] - p_b[1]
        if abs(dy) < 1e-6:
            return p_t.copy()
        t = (sky_y - p_t[1]) / dy
        return p_t + t * (p_t - p_b)

    def quad(a, b, c, d):
        return np.array([a[:2], b[:2], c[:2], d[:2]],
                        dtype=np.float32).reshape(-1, 1, 2)

    # 左墙四边形：远左底 → 近左底 → 近左天 → 远左天
    left_q  = quad(fl_b, nl_b, to_sky(nl_b, nl_t), to_sky(fl_b, fl_t))
    # 右墙四边形：远右底 → 近右底 → 近右天 → 远右天
    right_q = quad(fr_b, nr_b, to_sky(nr_b, nr_t), to_sky(fr_b, fr_t))
    return left_q, right_q


def _find_stop_frame(pts, timeout_frames, move_px):
    """在运动轨迹中找到球"永久停止"的帧号。

    pts       : [(frame_idx, cx, cy), ...]，按帧号升序排列
    返回第一个在后续 timeout_frames 内位移均 < move_px 的帧号；
    若整条轨迹始终在运动则返回 None。
    """
    n          = len(pts)
    frame_idxs = [p[0] for p in pts]
    for i in range(n):
        fi0       = frame_idxs[i]
        cx0, cy0  = pts[i][1], pts[i][2]
        # 收集 timeout 窗口内的所有点
        window = [k for k in range(i, n) if frame_idxs[k] - fi0 <= timeout_frames]
        if len(window) < 2:
            break  # 轨迹末尾，数据不足，不判定
        moved = any(
            np.hypot(pts[k][1] - cx0, pts[k][2] - cy0) >= move_px
            for k in window
        )
        if not moved:
            return fi0
    return None


def _filter_balls(balls, left_wall_q, right_wall_q, volume_hull, fps=25.0):
    """
    返回 (kept, removed)。

    静止轨迹（全局包围盒对角线 < _STATIC_BBOX_DIAG_PX）→ 整条轨迹无效（场地噪点/假阳性）。
    运动球起点在侧墙外 → 整条轨迹无效（场外球）。
    运动球在停止后超过 _STOPPED_TIMEOUT_S 秒 → 停止帧之后的检测无效（遗留球）。

    track_id 处理：
    - 来自 track.py：track_id=None 的检测直接移除。
    - 来自 detect.py（所有 track_id 均为 None）：按位置过滤（兼容跳过 track 的用法）。
    """
    has_tracked = any(d.get('track_id') is not None
                      for frame in balls for d in frame)

    # 收集每条轨迹的 (frame_idx, cx, cy)
    track_pts = defaultdict(list)
    for fi, frame in enumerate(balls):
        for d in frame:
            tid = d.get('track_id')
            if tid is not None:
                cx = (d['bbox'][0] + d['bbox'][2]) / 2
                cy = (d['bbox'][1] + d['bbox'][3]) / 2
                track_pts[tid].append((fi, cx, cy))

    timeout_frames = max(1, int(fps * _STOPPED_TIMEOUT_S))

    invalid_tracks = set()            # 整条轨迹无效
    stopped_frames: dict = {}         # tid → 停止帧号（之后的检测无效）

    for tid, pts in track_pts.items():
        if len(pts) < 2:
            invalid_tracks.add(tid)
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        bbox_diag = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if bbox_diag < _STATIC_BBOX_DIAG_PX:
            invalid_tracks.add(tid)                          # 始终静止：噪点
        elif (_in_hull(left_wall_q, pts[0][1], pts[0][2]) or
              _in_hull(right_wall_q, pts[0][1], pts[0][2])):
            invalid_tracks.add(tid)                          # 起点在侧墙外
        else:
            stop_fi = _find_stop_frame(pts, timeout_frames, _STOPPED_MOVE_PX)
            if stop_fi is not None:
                stopped_frames[tid] = stop_fi                # 曾运动后停止

    kept, removed = [], []
    for fi, frame in enumerate(balls):
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            cx  = (d['bbox'][0] + d['bbox'][2]) / 2
            cy  = (d['bbox'][1] + d['bbox'][3]) / 2
            if tid in invalid_tracks:
                r.append(d)
            elif tid is None:
                if has_tracked:
                    r.append(d)
                else:
                    (k if _in_hull(volume_hull, cx, cy) else r).append(d)
            elif tid in stopped_frames and fi >= stopped_frames[tid]:
                r.append(d)                                  # 遗留球：停止超时
            else:
                k.append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True, help='track.py（或 detect.py）输出的 JSON 路径')
    p.add_argument('-o', '--output', default=None,  help='输出 JSON 路径（默认：输入同名加 _parsed）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    if stem.endswith('.tracked'):
        stem = stem[:-len('.tracked')]
    output_path = args.output or stem + '.parsed.json'

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  output  {output_path}")
    print("─" * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)

    # 缓冲区过滤
    ground_hull = court['ground_hull']
    volume_hull = court['volume_hull']
    left_wall_q, right_wall_q = _make_wall_quads(
        court['vol_bottom_pts'], court['vol_top_pts'], height)
    left_player_wall, right_player_wall = _make_wall_quads(
        court['court_bottom_pts'], court['court_top_pts'], height)

    n_players_before = sum(len(f) for f in players)
    n_rackets_before = sum(len(f) for f in rackets)
    n_balls_before   = sum(len(f) for f in balls)
    players, players_inv = _filter_players(players, left_player_wall, right_player_wall, ground_hull)
    rackets, rackets_inv = _filter_rackets(rackets, volume_hull, players)
    balls,   balls_inv   = _filter_balls(balls, left_wall_q, right_wall_q, volume_hull, fps=fps)
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
              output_path, fps=fps, court=court,
              video=propagate_video(args.input, output_path))


if __name__ == '__main__':
    main()
