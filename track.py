"""
第二阶段：读取 detect.py 输出的 JSON，对球员、球拍、网球进行追踪与空间过滤，
输出含 track_id 的干净 JSON。

用法：
    python track.py -i <video>.detected.json
    python track.py -i <video>.detected.json -o <video>.tracked.json
输出：
    <video>.tracked.json（默认，去掉 _detected 后缀后加 _tracked）
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils import load_detections, load_video_path, save_coco, iter_frames, propagate_video
from tracker import BallTracker, PlayerTracker, RacketTracker
from court_detector import COURT_W as _COURT_W, compute_H_from_kps, compute_court_polygons

_VIDEO_EXTENSIONS     = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
_SMOOTH_SIGMA_SECONDS = 0.1   # 轨迹平滑高斯核标准差（秒）


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _px_per_meter(court_kps):
    """从球场关键点（展平 28 维）估算像素/米比例。

    取远端底线和近端底线的宽度各自换算，再取均值，以减小透视畸变的影响。
    """
    kps      = court_kps.reshape(14, 2)
    far_ppm  = float(np.linalg.norm(kps[1] - kps[0])) / _COURT_W
    near_ppm = float(np.linalg.norm(kps[3] - kps[2])) / _COURT_W
    return (far_ppm + near_ppm) / 2.0


def _split_continuous_segments(frames):
    """将 (frame_idx, det) 列表按帧号连续性拆分为若干段。

    帧号相邻（间隔 <= 1）归入同一段；间隔 > 1 表示遮挡或丢失，切为新段。
    各段独立平滑，避免跨间隙插值。
    """
    segments = [[frames[0]]]
    for k in range(1, len(frames)):
        if frames[k][0] - frames[k - 1][0] <= 1:
            segments[-1].append(frames[k])
        else:
            segments.append([frames[k]])
    return segments


# ── 轨迹平滑 ──────────────────────────────────────────────────────────────────

def _smooth_player_tracks(players, fps):
    """对每条球员轨迹的脚点坐标做高斯平滑，结果写入 det['foot']。

    脚点 = bbox 底边中点，用于后续球场坐标投影。
    按 track_id 分组，各连续段独立平滑，遮挡间隙两侧不相互影响。
    """
    sigma = max(1.0, fps * _SMOOTH_SIGMA_SECONDS)

    # 按 track_id 收集 (frame_idx, det)，忽略未追踪的检测
    tracks: dict = {}
    for fi, frame_dets in enumerate(players):
        for det in frame_dets:
            tid = det.get('track_id')
            if tid is not None:
                tracks.setdefault(tid, []).append((fi, det))

    for frames in tracks.values():
        frames.sort(key=lambda x: x[0])
        for seg in _split_continuous_segments(frames):
            fxs = np.array([(d['bbox'][0] + d['bbox'][2]) / 2 for _, d in seg])
            fys = np.array([d['bbox'][3] for _, d in seg])
            if len(seg) >= 3:
                fxs = gaussian_filter1d(fxs, sigma)
                fys = gaussian_filter1d(fys, sigma)
            for k, (_, det) in enumerate(seg):
                det['foot'] = [float(fxs[k]), float(fys[k])]

    return players


def _smooth_racket_tracks(rackets, fps):
    """对每条球拍轨迹的中心点坐标做高斯平滑，结果写入 det['center']。

    bbox 本身不修改；center 用于后续可视化和分析。
    """
    sigma = max(1.0, fps * _SMOOTH_SIGMA_SECONDS)

    tracks: dict = {}
    for fi, frame_dets in enumerate(rackets):
        for det in frame_dets:
            tid = det.get('track_id')
            if tid is not None:
                tracks.setdefault(tid, []).append((fi, det))

    for frames in tracks.values():
        frames.sort(key=lambda x: x[0])
        for seg in _split_continuous_segments(frames):
            cxs = np.array([(d['bbox'][0] + d['bbox'][2]) / 2 for _, d in seg])
            cys = np.array([(d['bbox'][1] + d['bbox'][3]) / 2 for _, d in seg])
            if len(seg) >= 3:
                cxs = gaussian_filter1d(cxs, sigma)
                cys = gaussian_filter1d(cys, sigma)
            for k, (_, det) in enumerate(seg):
                det['center'] = [float(cxs[k]), float(cys[k])]

    return rackets


# ── 空间过滤 ──────────────────────────────────────────────────────────────────

_STATIC_BBOX_DIAG_PX = 20.0  # 静止球判定阈值：轨迹全局包围盒对角线（像素）


def _in_hull(hull, x, y):
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def _bboxes_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _bbox_overlaps_hull(hull, x1, y1, x2, y2):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for pt in [(cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        if _in_hull(hull, *pt):
            return True
    return False


def _make_out_zones(floor_pts, ceil_pts, img_height):
    """构造左右场外区（侧线外延伸到天空）的四边形 (4,1,2) float32。"""
    bpts = np.array(floor_pts, dtype=np.float64)
    tpts = np.array(ceil_pts,  dtype=np.float64)
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

    left_q  = quad(fl_b, nl_b, to_sky(nl_b, nl_t), to_sky(fl_b, fl_t))
    right_q = quad(fr_b, nr_b, to_sky(nr_b, nr_t), to_sky(fr_b, fr_t))
    return left_q, right_q


def _filter_players(players, left_out, right_out, ground_poly):
    """返回 (kept, removed)。

    按 track_id 分组，同时满足以下两个条件才认为是球员：
    1. 轨迹大部分（>50%）底部中心在 ground_poly 内
    2. 轨迹大部分不落在双打侧线外（左场外区 + 右场外区合计 <= 50%）
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
            if _in_hull(ground_poly, cx, cy):
                s['in_ground'] += 1
            if _in_hull(left_out, cx, cy) or _in_hull(right_out, cx, cy):
                s['out_side'] += 1

    invalid_tracks = set()
    for tid, s in track_stats.items():
        total = s['total']
        if s['in_ground'] / total <= 0.5 or s['out_side'] / total > 0.5:
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


def _filter_rackets(rackets, clearance_poly, valid_players):
    """返回 (kept, removed)。

    按 track_id 分组，轨迹中 >50% 的帧同时满足：
    1. bbox 与 clearance_poly 有交叠
    2. bbox 与当前帧至少一个有效球员重叠
    无 track_id 的检测直接移除。
    """
    track_total: dict = defaultdict(int)
    track_valid: dict = defaultdict(int)
    for frame, players in zip(rackets, valid_players):
        for d in frame:
            tid = d.get('track_id')
            if tid is None:
                continue
            track_total[tid] += 1
            if (_bbox_overlaps_hull(clearance_poly, *d['bbox']) and
                    any(_bboxes_overlap(d['bbox'], p['bbox']) for p in players)):
                track_valid[tid] += 1

    invalid_tracks = {
        tid for tid, total in track_total.items()
        if track_valid[tid] / total <= 0.5
    }

    kept, removed = [], []
    for frame in rackets:
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            if tid is None:
                r.append(d)
                continue
            (r if tid in invalid_tracks else k).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


def _filter_balls(balls, clearance_poly, left_out, right_out):
    """返回 (kept, removed)。

    静止轨迹（包围盒对角线 < _STATIC_BBOX_DIAG_PX）→ 无效（场地噪点）。
    运动轨迹起始点在场外区且不在 clearance_poly 内 → 无效（边线外噪点）。
    运动轨迹从未进入 clearance_poly → 无效（场外噪点）。
    无 track_id 的检测直接移除。
    """
    track_pts = defaultdict(list)
    for fi, frame in enumerate(balls):
        for d in frame:
            tid = d.get('track_id')
            if tid is not None and not d.get('interpolated'):
                cx = (d['bbox'][0] + d['bbox'][2]) / 2
                cy = (d['bbox'][1] + d['bbox'][3]) / 2
                track_pts[tid].append((fi, cx, cy))

    invalid_tracks = set()
    for tid, pts in track_pts.items():
        if len(pts) < 2:
            invalid_tracks.add(tid)
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        if np.hypot(max(xs) - min(xs), max(ys) - min(ys)) < _STATIC_BBOX_DIAG_PX:
            invalid_tracks.add(tid)
            continue
        # 运动轨迹：起始点在场外区且不在 clearance_poly 内 → 场外噪点
        sx, sy = pts[0][1], pts[0][2]
        if (not _in_hull(clearance_poly, sx, sy)
                and (_in_hull(left_out, sx, sy) or _in_hull(right_out, sx, sy))):
            invalid_tracks.add(tid)
            continue
        if not any(_in_hull(clearance_poly, p[1], p[2]) for p in pts):
            invalid_tracks.add(tid)

    kept, removed = [], []
    for frame in balls:
        k, r = [], []
        for d in frame:
            tid = d.get('track_id')
            (r if tid is None or tid in invalid_tracks else k).append(d)
        kept.append(k)
        removed.append(r)
    return kept, removed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',          required=True,       help='detect.py 输出的 JSON 路径')
    p.add_argument('-o', '--output',         default=None,        help='输出 JSON 路径（默认：输入同名加 _tracked）')
    p.add_argument('--conf-high',            type=float, default=0.5,  help='高置信度阈值：>= 此值的检测可新建轨迹')
    p.add_argument('--conf-low',             type=float, default=0.0,  help='低置信度下限：[low,high) 的检测仅续接已有轨迹')
    p.add_argument('--search-diameters',     type=float, default=3.0,  help='球追踪搜索半径 = N × 球径（px）')
    p.add_argument('--min-aspect-h',         type=float, default=0.15, help='水平拉长（w≥h）长宽比下限：运动模糊允许较大拉伸')
    p.add_argument('--min-aspect-v',         type=float, default=0.5,  help='垂直拉长（h>w）长宽比下限：竖向模糊罕见，严格限制')
    p.add_argument('--sub-model',            default=None,             help='次检测器模型路径；传入后启用次检测器进行 recall 补检')
    p.add_argument('--sub-save-dir',         default=None,             help='调试：将每次 recall 的 patch 图存入该目录')
    p.add_argument('--debug-frame',          type=int,   default=-1,   help='打印指定帧的追踪器内部状态（-1 关闭）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 推断输出路径：去掉 _detected 后缀，加 _tracked
    stem = os.path.splitext(args.input)[0]
    if stem.endswith('.detected'):
        stem = stem[:-len('.detected')]
    output_path = args.output or stem + '.tracked.json'

    # 加载次检测器模型（可选，仅在显式传入 --sub-model 时启用）
    sub_model = None
    if args.sub_model:
        if not Path(args.sub_model).exists():
            print(f"[sub] 警告：模型不存在 {args.sub_model}，次检测器已禁用")
        else:
            from ultralytics import YOLO as _YOLO
            from ultralytics.utils import LOGGER as _ul_logger
            _prev_level = _ul_logger.level
            _ul_logger.setLevel(logging.WARNING)
            sub_model = _YOLO(args.sub_model, verbose=False)
            _ul_logger.setLevel(_prev_level)
            print(f"[sub] 已加载 {args.sub_model}")

    print("─" * 60)
    print(f"  input      {args.input}")
    print(f"  output     {output_path}")
    print(f"  conf       [{args.conf_low}, {args.conf_high})")
    print(f"  search     {args.search_diameters}× ball_d")
    print(f"  sub-model  {args.sub_model if sub_model else '禁用'}")
    if args.sub_save_dir:
        print(f"  sub-dir    {args.sub_save_dir}")
    print("─" * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)
    ppm = _px_per_meter(court['keypoints'])
    H     = compute_H_from_kps(court['keypoints'])
    H_inv = np.linalg.inv(H.astype(np.float64))

    # 计算背景板 / 网带图像多边形，存入 court 供后续阶段使用
    try:
        backdrop_poly, net_poly = compute_court_polygons(H, width, height)
        court['backdrop_poly'] = backdrop_poly
        court['net_poly']      = net_poly
        print(f"[ court ] backdrop_poly={[f'({p[0]:.0f},{p[1]:.0f})' for p in backdrop_poly]}")
        print(f"[ court ] net_poly     ={[f'({p[0]:.0f},{p[1]:.0f})' for p in net_poly]}")
    except Exception as e:
        print(f"[ court ] 警告：无法计算背景板/网带多边形：{e}")

    # 查找视频文件：优先读 JSON 的 video 字段，再按扩展名枚举
    video_path = load_video_path(args.input)
    if video_path is None or not os.path.exists(video_path):
        video_path = None
        for ext in _VIDEO_EXTENSIONS:
            candidate = stem + ext
            if os.path.exists(candidate):
                video_path = candidate
                break
    if video_path:
        print(f"[ player ] video → {video_path}  颜色直方图外观匹配已启用")
    else:
        print(f"[ player ] no video found alongside JSON  外观匹配已禁用")

    # 球员追踪：颜色直方图 Re-ID 可选（需要视频文件）
    players = PlayerTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
    ).run(players, frames=iter_frames(video_path) if video_path else None)
    players = _smooth_player_tracks(players, fps)

    # 球拍追踪
    rackets = RacketTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
    ).run(rackets)
    rackets = _smooth_racket_tracks(rackets, fps)

    # 网球追踪：recall 补检 + gap 插值（均在 BallTracker 内逐帧完成）
    # 无次检测器时不需要读取视频帧
    ball_frames = iter_frames(video_path) if (video_path and sub_model) else None
    balls, frame_predictions = BallTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
        search_diameters=args.search_diameters,
        min_aspect_h=args.min_aspect_h, min_aspect_v=args.min_aspect_v,
        H_inv=H_inv,
        backdrop_poly=court.get('backdrop_poly'),
        sub_model=sub_model,
        sub_save_dir=args.sub_save_dir,
    ).run(balls,
          rackets=rackets, players=players, court=court,
          debug_frame=args.debug_frame, frames=ball_frames)

    # 空间过滤：无效检测标 valid=False，全部保留在输出中
    ground_poly    = court['ground_poly']
    clearance_poly = court['clearance_poly']
    player_left_out,  player_right_out  = _make_out_zones(
        court['court_floor_pts'], court['court_ceil_pts'], height)
    ball_left_out, ball_right_out = _make_out_zones(
        court['floor_pts'], court['ceil_pts'], height)

    n_p = sum(len(f) for f in players)
    n_r = sum(len(f) for f in rackets)
    n_b = sum(len(f) for f in balls)
    players, players_inv = _filter_players(players, player_left_out, player_right_out, ground_poly)
    rackets, rackets_inv = _filter_rackets(rackets, clearance_poly, players)
    balls,   balls_inv   = _filter_balls(balls, clearance_poly, ball_left_out, ball_right_out)
    print(f"[  filter] players: {n_p} → {sum(len(f) for f in players)}"
          f"  (invalid={sum(len(f) for f in players_inv)})")
    print(f"[  filter] rackets: {n_r} → {sum(len(f) for f in rackets)}"
          f"  (invalid={sum(len(f) for f in rackets_inv)})")
    print(f"[  filter] balls:   {n_b} → {sum(len(f) for f in balls)}"
          f"  (invalid={sum(len(f) for f in balls_inv)})")

    for frame in players_inv:
        for d in frame: d['valid'] = False
    for frame in rackets_inv:
        for d in frame: d['valid'] = False
    for frame in balls_inv:
        for d in frame: d['valid'] = False

    n_frames = len(players)
    players = [players[fi] + players_inv[fi] for fi in range(n_frames)]
    rackets = [rackets[fi] + rackets_inv[fi] for fi in range(n_frames)]
    balls   = [balls[fi]   + balls_inv[fi]   for fi in range(n_frames)]

    save_coco(width, height, players, rackets, balls,
              output_path, fps=fps, court=court,
              video=propagate_video(args.input, output_path),
              frame_predictions=frame_predictions)


if __name__ == '__main__':
    main()
