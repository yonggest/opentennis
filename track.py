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

from utils import load_detections, save_coco, iter_frames
from tracker import BallTracker, PlayerTracker
from court_detector import COURT_W as _COURT_W

_VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')


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
