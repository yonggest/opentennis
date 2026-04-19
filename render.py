"""
第四阶段：读取 parse.py 输出的处理后 JSON，逐帧流式处理原始视频，输出注释视频。

用法：
    python render.py -i <video> -j <video>.json           # detect.py 输出（无过滤/追踪）
    python render.py -i <video> -j <video>.parsed.json   # parse.py 输出（含过滤/追踪）
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

from utils import video_info, iter_frames, open_video_writer, load_detections, text_params, load_video_path
from court_detector import (COURT_LINES, COURT_W as _COURT_W, NET_Y as _NET_Y,
                             compute_H_from_kps)


# ── 绘制颜色（BGR）────────────────────────────────────────────────────────────
_COLOR_COURT      = (80, 200, 255)   # 球场线 / 关键点
_COLOR_VOLUME     = (0, 220, 255)    # 缓冲区立方体线框
_COLOR_SIDELINE   = (0, 180, 0)      # 双打侧线立方体线框（球员过滤边界）
_COLOR_PLAYER     = (0, 255, 0)      # 有效球员
_COLOR_RACKET     = (255, 165, 0)    # 有效球拍
_COLOR_BALL_NONE  = (128, 128, 128)  # 无 track_id 的球
_COLOR_INV_PLAYER = (0, 60, 0)       # 无效球员
_COLOR_INV_RACKET = (40, 50, 80)     # 无效球拍
_COLOR_INV_BALL   = (40, 40, 40)     # 无效网球

# 每条球轨迹按 track_id 循环取色（BGR）
_BALL_TRAJ_COLORS = [
    (0, 255, 255), (255, 100, 0), (180, 0, 255),
    (0, 255, 100), (255, 200, 0), (0, 100, 255),
]

# 每条球员轨迹按 track_id 循环取色（BGR）
_PLAYER_TRAJ_COLORS = [
    (68, 68, 255), (68, 255, 68), (68, 255, 255), (255, 68, 255),
]

# 每条球拍轨迹按 track_id 循环取色（BGR）
_RACKET_TRAJ_COLORS = [
    (0, 136, 255), (255, 0, 170), (0, 255, 187), (170, 0, 255),
]

# COCO 17点骨架连接
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 头部
    (5, 7), (7, 9), (6, 8), (8, 10),          # 手臂
    (5, 6), (5, 11), (6, 12), (11, 12),       # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16),   # 腿部
]
_COLOR_SKELETON = (0, 255, 128)   # 骨架颜色
_KP_CONF_THRESH = 0.3             # 关键点置信度阈值

# 帧号字体相对于 scale_large 的放大倍数
_FRAME_NUM_SCALE = 1.5

# 帧号距边缘的比例（相对于帧高）
_MARGIN_RATIO = 0.028


def _ball_traj_color(track_id):
    return _BALL_TRAJ_COLORS[track_id % len(_BALL_TRAJ_COLORS)]


def _player_traj_color(track_id):
    return _PLAYER_TRAJ_COLORS[track_id % len(_PLAYER_TRAJ_COLORS)]


def _racket_traj_color(track_id):
    return _RACKET_TRAJ_COLORS[track_id % len(_RACKET_TRAJ_COLORS)]


def _project_line(H, x1, y1, x2, y2):
    """用 H 将一段球场米坐标线投影为图像像素坐标，返回 (pt1, pt2)。"""
    pts = cv2.perspectiveTransform(
        np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32), H)
    return tuple(pts[0, 0].astype(int)), tuple(pts[1, 0].astype(int))


def _draw_court_kps(frame, court_kps, H):
    """绘制 14 个球场关键点、球场线条两侧边缘及网线。"""
    if H is not None:
        for (p1, p2, lw_m) in COURT_LINES:
            half = lw_m / 2
            if p1[1] == p2[1]:   # 水平线 → 沿 y 方向偏移
                for dy in (-half, +half):
                    pt1, pt2 = _project_line(H, p1[0], p1[1]+dy, p2[0], p2[1]+dy)
                    cv2.line(frame, pt1, pt2, _COLOR_COURT, 1)
            else:                 # 垂直线 → 沿 x 方向偏移
                for dx in (-half, +half):
                    pt1, pt2 = _project_line(H, p1[0]+dx, p1[1], p2[0]+dx, p2[1])
                    cv2.line(frame, pt1, pt2, _COLOR_COURT, 1)
        pt1, pt2 = _project_line(H, 0, _NET_Y, _COURT_W, _NET_Y)
        cv2.line(frame, pt1, pt2, _COLOR_COURT, 1)

    kps = court_kps.reshape(14, 2).astype(int)
    for pt in kps:
        cv2.circle(frame, tuple(pt), 4, _COLOR_COURT, -1)


def _build_ball_traj(balls):
    """预计算各轨迹的全部中心点：{track_id: [(frame_idx, cx, cy), ...]}。"""
    traj = defaultdict(list)
    for fi, frame_balls in enumerate(balls):
        for det in frame_balls:
            tid = det.get('track_id')
            if tid is not None:
                x1, y1, x2, y2 = det['bbox']
                traj[tid].append((fi, int((x1 + x2) / 2), int((y1 + y2) / 2)))
    return traj


def _build_player_traj(players):
    """预计算各球员轨迹的脚步位置：{track_id: [(frame_idx, fx, fy), ...]}。
    使用检测框底部中心点（脚步位置）作为轨迹锚点，仅含有效检测。
    """
    traj = defaultdict(list)
    for fi, frame_players in enumerate(players):
        for det in frame_players:
            tid = det.get('track_id')
            if tid is not None and det.get('valid', True):
                if 'foot' in det:
                    fx, fy = det['foot']
                else:
                    x1, y1, x2, y2 = det['bbox']
                    fx, fy = (x1 + x2) / 2, y2
                traj[tid].append((fi, int(fx), int(fy)))
    return traj


def _build_racket_traj(rackets):
    """预计算各球拍轨迹的中心位置：{track_id: [(frame_idx, cx, cy), ...]}，仅含有效检测。"""
    traj = defaultdict(list)
    for fi, frame_rackets in enumerate(rackets):
        for det in frame_rackets:
            tid = det.get('track_id')
            if tid is not None and det.get('valid', True):
                if 'center' in det:
                    cx, cy = det['center']
                else:
                    x1, y1, x2, y2 = det['bbox']
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                traj[tid].append((fi, int(cx), int(cy)))
    return traj


def _draw_invalid_bbox(frame, bbox, color, thick=1):
    """用暗色调矩形 + 对角线 X 标注无效物体。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.line(frame, (x1, y1), (x2, y2), color, thick)
    cv2.line(frame, (x2, y1), (x1, y2), color, thick)


def _draw_outside_masks(frame, pts_bot, pts_top, alpha=0.4):
    """在缓冲区侧边外部绘制半透明遮罩，侧边沿墙竖边方向延伸到天空。"""
    h, w = frame.shape[:2]
    sky_y   = -h
    floor_y =  h * 2

    fl_b, fr_b, nr_b, nl_b = pts_bot
    fl_t, fr_t, nr_t, nl_t = pts_top

    def wall_to_sky(p_b, p_t):
        dy = float(p_t[1]) - float(p_b[1])
        if abs(dy) < 1e-6:
            return float(p_t[0])
        return float(p_t[0]) + (sky_y - float(p_t[1])) / dy * (float(p_t[0]) - float(p_b[0]))

    def x_at_y(p1, p2, y):
        dy = float(p2[1]) - float(p1[1])
        if abs(dy) < 1e-6:
            return float(p1[0])
        return float(p1[0]) + (float(p2[0]) - float(p1[0])) / dy * (y - float(p1[1]))

    left_pts = np.array([
        [-w,                       sky_y  ],
        [wall_to_sky(fl_b, fl_t),  sky_y  ],
        [fl_t[0],                  fl_t[1]],
        [fl_b[0],                  fl_b[1]],
        [nl_b[0],                  nl_b[1]],
        [x_at_y(fl_b, nl_b, floor_y), floor_y],
        [-w,                       floor_y],
    ], dtype=np.int32)

    right_pts = np.array([
        [w * 2,                    sky_y  ],
        [wall_to_sky(fr_b, fr_t),  sky_y  ],
        [fr_t[0],                  fr_t[1]],
        [fr_b[0],                  fr_b[1]],
        [nr_b[0],                  nr_b[1]],
        [x_at_y(fr_b, nr_b, floor_y), floor_y],
        [w * 2,                    floor_y],
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [left_pts],  (0, 0, 0))
    cv2.fillPoly(overlay, [right_pts], (0, 0, 0))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_volume_wireframe(frame, pts_bot, pts_top, color):
    """绘制缓冲区立方体线框：底面4条、顶面4条、竖边4条，共12条边。"""
    pts_b = pts_bot.astype(int)
    pts_t = pts_top.astype(int)
    n = len(pts_b)
    for i in range(n):
        j = (i + 1) % n
        cv2.line(frame, tuple(pts_b[i]), tuple(pts_b[j]), color, 1)
        cv2.line(frame, tuple(pts_t[i]), tuple(pts_t[j]), color, 1)
        cv2.line(frame, tuple(pts_b[i]), tuple(pts_t[i]), color, 1)



def _draw_skeleton(frame, keypoints, thick):
    """绘制 COCO 17 关键点骨架。"""
    for (i, j) in _SKELETON:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]
        if ci >= _KP_CONF_THRESH and cj >= _KP_CONF_THRESH:
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)),
                     _COLOR_SKELETON, thick)
    for kp in keypoints:
        x, y, c = kp
        if c >= _KP_CONF_THRESH:
            cv2.circle(frame, (int(x), int(y)), 3, _COLOR_SKELETON, -1)


def _draw_frame(frame, fi, court_kps, H, pts_vol_bot, pts_vol_top,
                pts_court_bot, pts_court_top,
                players, rackets, balls, ball_traj, player_traj, racket_traj,
                players_inv, rackets_inv, balls_inv,
                scale, thick, scale_large, thick_large, margin):
    """原地修改单帧：绘制球场轮廓、缓冲区立方体、检测框、球/球员/球拍轨迹、帧号。"""
    _draw_court_kps(frame, court_kps, H)
    _draw_outside_masks(frame, pts_court_bot, pts_court_top)
    _draw_volume_wireframe(frame, pts_court_bot, pts_court_top, _COLOR_SIDELINE)
    _draw_outside_masks(frame, pts_vol_bot, pts_vol_top)
    _draw_volume_wireframe(frame, pts_vol_bot, pts_vol_top, _COLOR_VOLUME)

    # 无效物体（暗色 + X）
    for det in players_inv[fi]:
        _draw_invalid_bbox(frame, det['bbox'], _COLOR_INV_PLAYER, 1)
    for det in rackets_inv[fi]:
        _draw_invalid_bbox(frame, det['bbox'], _COLOR_INV_RACKET, 1)
    for det in balls_inv[fi]:
        _draw_invalid_bbox(frame, det['bbox'], _COLOR_INV_BALL, 1)

    # 有效球员
    for det in players[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_PLAYER, thick)
        if det.get('track_id') is not None:
            cv2.putText(frame, f"P{det['track_id']}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, _COLOR_PLAYER, thick)

    # 球员姿态骨架
    for det in players[fi]:
        if 'keypoints' in det:
            _draw_skeleton(frame, det['keypoints'], thick)

    # 有效球拍
    for det in rackets[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_RACKET, thick)
        if det.get('track_id') is not None:
            cv2.putText(frame, f"R{det['track_id']}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, _COLOR_RACKET, thick)

    # 球拍中心轨迹
    for tid, pts in racket_traj.items():
        color   = _racket_traj_color(tid)
        visible = [(cx, cy) for fj, cx, cy in pts if fj <= fi]
        for k in range(len(visible) - 1):
            cv2.arrowedLine(frame, visible[k], visible[k + 1], color, thick, tipLength=0.3)

    # 球员脚步全量历史轨迹（不含将来帧，无帧数窗口上限）
    for tid, pts in player_traj.items():
        color   = _player_traj_color(tid)
        visible = [(fx, fy) for fj, fx, fy in pts if fj <= fi]
        for k in range(len(visible) - 1):
            cv2.arrowedLine(frame, visible[k], visible[k + 1], color, thick, tipLength=0.3)

    # 球轨迹（已出现的点）
    for tid, pts in ball_traj.items():
        color   = _ball_traj_color(tid)
        visible = [(cx, cy) for fj, cx, cy in pts if fj <= fi]
        for k in range(len(visible) - 1):
            cv2.arrowedLine(frame, visible[k], visible[k + 1], color, thick, tipLength=0.3)

    # 有效球框
    for det in balls[fi]:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        tid   = det.get('track_id')
        color = _ball_traj_color(tid) if tid is not None else _COLOR_BALL_NONE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    # 帧号
    cv2.putText(frame, str(fi), (margin, frame.shape[0] - margin),
                cv2.FONT_HERSHEY_SIMPLEX, scale_large * _FRAME_NUM_SCALE,
                _COLOR_PLAYER, thick_large)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',      default=None,                help='原始输入视频路径（默认：从 JSON 的 video 字段读取）')
    p.add_argument('-j', '--json',       required=True,               help='detect.py 或 parse.py 输出的 JSON 路径')
    p.add_argument('-o', '--output',     default=None,                help='输出视频路径（默认：输入视频同名 .mp4）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.input is None:
        args.input = load_video_path(args.json)
    if args.input is None:
        print("错误: 未指定 -i，且 JSON 中未包含 video 字段", file=sys.stderr)
        sys.exit(1)
    output_path = args.output or os.path.splitext(args.input)[0] + '.mp4'
    if os.path.abspath(output_path) == os.path.abspath(args.input):
        print(f"错误: 输出路径与输入路径相同: {output_path}", file=sys.stderr)
        sys.exit(1)

    print("─" * 60)
    print(f"  input       {args.input}")
    print(f"  json        {args.json}")
    print(f"  output      {output_path}")
    print("─" * 60, flush=True)

    # ── 预加载：JSON（小）─────────────────────────────────────────────────────
    fps, width, height, court, players_raw, rackets_raw, balls_raw = \
        load_detections(args.json)

    court_kps    = court['keypoints']
    pts_vol_bot  = court['vol_bottom_pts']
    pts_vol_top  = court['vol_top_pts']
    pts_court_bot = np.array(court['court_bottom_pts'])
    pts_court_top = np.array(court['court_top_pts'])
    H            = compute_H_from_kps(court_kps)

    players     = [[d for d in f if     d['valid']] for f in players_raw]
    players_inv = [[d for d in f if not d['valid']] for f in players_raw]
    rackets     = [[d for d in f if     d['valid']] for f in rackets_raw]
    rackets_inv = [[d for d in f if not d['valid']] for f in rackets_raw]
    balls       = [[d for d in f if     d['valid']] for f in balls_raw]
    balls_inv   = [[d for d in f if not d['valid']] for f in balls_raw]

    ball_traj   = _build_ball_traj(balls)
    player_traj = _build_player_traj(players_raw)
    racket_traj = _build_racket_traj(rackets_raw)

    scale, thick             = text_params(height)
    scale_large, thick_large = text_params(height, base_height=1080)
    margin = int(height * _MARGIN_RATIO)

    # ── 单遍流式处理：读帧 → 处理 → 写帧 ─────────────────────────────────────
    _, _, _, n_frames = video_info(args.input)
    frame_num_width = len(str(n_frames)) if n_frames else 6
    t0 = time.time()
    count = 0

    with open_video_writer(output_path, fps, width, height) as pipe:
        for fi, frame in enumerate(iter_frames(args.input)):
            _draw_frame(frame, fi, court_kps, H, pts_vol_bot, pts_vol_top,
                        pts_court_bot, pts_court_top,
                        players, rackets, balls, ball_traj, player_traj, racket_traj,
                        players_inv, rackets_inv, balls_inv,
                        scale, thick, scale_large, thick_large, margin)
            pipe.write(frame.tobytes())
            count = fi + 1
            if n_frames:
                pct = count * 100 // n_frames
                print(f"[  render] {count:>{frame_num_width}}/{n_frames} frames  ({pct:>3}%)", end='\r', flush=True)
            else:
                print(f"[  render] {count} frames", end='\r', flush=True)

    print(f"[  render] {count:>{frame_num_width}}/{count} frames  (100%)  done: {time.time()-t0:>6.1f}s")
    print(f"[  render] saved → {output_path}", flush=True)


if __name__ == '__main__':
    main()
