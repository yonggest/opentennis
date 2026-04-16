"""
第二阶段：读取 detect.py 输出的 JSON，对球员和网球进行追踪，输出含 track_id 的 JSON。

用法：
    python track.py -i <video>_detected.json
    python track.py -i <video>_detected.json -o <video>_tracked.json
输出：
    <video>_tracked.json（默认，去掉 _detected 后缀后加 _tracked）
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

from utils import load_detections, save_coco, iter_frames
from tracker import BallTracker, PlayerTracker, RacketTracker
from court_detector import COURT_W as _COURT_W

_VIDEO_EXTENSIONS  = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
_SMOOTH_SIGMA_SECONDS  = 0.1   # 球员 bbox 平滑高斯核标准差（秒）


def _smooth_player_tracks(players, fps):
    """对每条球员轨迹的 bbox 坐标做高斯平滑。
    按 track_id 分组，帧号间隔 > 1 的位置切段，各段独立平滑，避免跨遮挡间隙平滑。
    """
    sigma = max(1.0, fps * _SMOOTH_SIGMA_SECONDS)

    # 按 track_id 收集 (frame_idx, det) 列表
    tracks: dict = {}
    for fi, frame_dets in enumerate(players):
        for det in frame_dets:
            tid = det.get('track_id')
            if tid is None:
                continue
            tracks.setdefault(tid, []).append((fi, det))

    for frames in tracks.values():
        frames.sort(key=lambda x: x[0])
        # 分割连续段（帧号相邻则归入同一段）
        segments = [[frames[0]]]
        for k in range(1, len(frames)):
            if frames[k][0] - frames[k - 1][0] <= 1:
                segments[-1].append(frames[k])
            else:
                segments.append([frames[k]])

        for seg in segments:
            if len(seg) < 3:
                # 段太短，直接用原始脚点
                for _, det in seg:
                    x1, _, x2, y2 = det['bbox']
                    det['foot'] = [float((x1 + x2) / 2), float(y2)]
                continue
            fxs = np.array([(d['bbox'][0] + d['bbox'][2]) / 2 for _, d in seg])
            fys = np.array([d['bbox'][3] for _, d in seg])

            fxs = gaussian_filter1d(fxs, sigma)
            fys = gaussian_filter1d(fys, sigma)

            for k, (_, det) in enumerate(seg):
                det['foot'] = [float(fxs[k]), float(fys[k])]

    return players


def _smooth_racket_tracks(rackets, fps):
    """对每条球拍轨迹的中心点坐标做高斯平滑，结果写入 det['center']，bbox 不变。"""
    sigma = max(1.0, fps * _SMOOTH_SIGMA_SECONDS)

    tracks: dict = {}
    for fi, frame_dets in enumerate(rackets):
        for det in frame_dets:
            tid = det.get('track_id')
            if tid is None:
                continue
            tracks.setdefault(tid, []).append((fi, det))

    for frames in tracks.values():
        frames.sort(key=lambda x: x[0])
        segments = [[frames[0]]]
        for k in range(1, len(frames)):
            if frames[k][0] - frames[k - 1][0] <= 1:
                segments[-1].append(frames[k])
            else:
                segments.append([frames[k]])

        for seg in segments:
            cxs = np.array([(d['bbox'][0] + d['bbox'][2]) / 2 for _, d in seg])
            cys = np.array([(d['bbox'][1] + d['bbox'][3]) / 2 for _, d in seg])
            if len(seg) >= 3:
                cxs = gaussian_filter1d(cxs, sigma)
                cys = gaussian_filter1d(cys, sigma)
            for k, (_, det) in enumerate(seg):
                det['center'] = [float(cxs[k]), float(cys[k])]

    return rackets


def _px_per_meter(court_kps):
    """从 court_kps（展平 28 维）估算像素/米比例，取远近底线宽度的平均值。"""
    kps      = court_kps.reshape(14, 2)
    far_ppm  = float(np.linalg.norm(kps[1] - kps[0])) / _COURT_W
    near_ppm = float(np.linalg.norm(kps[3] - kps[2])) / _COURT_W
    return (far_ppm + near_ppm) / 2.0


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',     required=True,        help='detect.py 输出的 JSON 路径')
    p.add_argument('-o', '--output',    default=None,         help='输出 JSON 路径（默认：输入同名加 _tracked）')
    p.add_argument('--conf-high',        type=float, default=0.5,  help='高置信度阈值：>= 此值的检测可新建轨迹')
    p.add_argument('--conf-low',         type=float, default=0.1,  help='低置信度下限：[low,high) 的检测仅续接已有轨迹')
    p.add_argument('--search-diameters', type=float, default=2.0,  help='搜索半径 = N × 球径（px）')
    p.add_argument('--debug-frame',      type=int,   default=-1,   help='打印指定帧的追踪器内部状态（-1 关闭）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    stem = os.path.splitext(args.input)[0]
    if stem.endswith('_detected'):
        stem = stem[:-len('_detected')]
    output_path = args.output or stem + '_tracked.json'

    print("─" * 60)
    print(f"  input      {args.input}")
    print(f"  output     {output_path}")
    print(f"  conf       [{args.conf_low}, {args.conf_high})")
    print(f"  search     {args.search_diameters}× ball_d")
    print("─" * 60, flush=True)

    fps, width, height, court, players, rackets, balls = load_detections(args.input)
    ppm = _px_per_meter(court['keypoints'])

    # 自动查找与 JSON 同目录同名的视频文件（供 Re-ID 使用）
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

    # 球员追踪（颜色直方图外观匹配可选）
    players = PlayerTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
    ).run(players,
          frames=iter_frames(video_path) if video_path else None)
    players = _smooth_player_tracks(players, fps)

    # 球拍追踪
    rackets = RacketTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
    ).run(rackets)
    rackets = _smooth_racket_tracks(rackets, fps)

    # 网球追踪
    balls = BallTracker.from_video(
        fps, ppm,
        conf_high=args.conf_high, conf_low=args.conf_low,
        search_diameters=args.search_diameters,
    ).run(balls, debug_frame=args.debug_frame)

    save_coco(width, height, players, rackets, balls,
              output_path, fps=fps, court=court)


if __name__ == '__main__':
    main()
