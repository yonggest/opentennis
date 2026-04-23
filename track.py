"""
第二阶段：读取 detect.py 输出的 JSON，对球员、球拍、网球进行追踪，输出含 track_id 的 JSON。

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
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils import load_detections, load_video_path, save_coco, iter_frames, propagate_video
from tracker import BallTracker, PlayerTracker, RacketTracker
from court_detector import COURT_W as _COURT_W

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
    balls = BallTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
        search_diameters=args.search_diameters,
        sub_model=sub_model,
        sub_save_dir=args.sub_save_dir,
    ).run(balls, debug_frame=args.debug_frame, frames=ball_frames)

    save_coco(width, height, players, rackets, balls,
              output_path, fps=fps, court=court,
              video=propagate_video(args.input, output_path))


if __name__ == '__main__':
    main()
