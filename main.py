import argparse
import os
import sys
import time
import cv2
import numpy as np
from collections import defaultdict
from utils import read_video, save_video, save_coco, text_params
from objects_detector import ObjectsDetector
from court_detector import CourtDetector
from ball_tracker import BallTracker

# 每条球轨迹的颜色，按 track_id 循环使用（BGR）
_TRAJ_COLORS = [
    (0, 255, 255), (255, 100, 0), (180, 0, 255),
    (0, 255, 100), (255, 200, 0), (0, 100, 255),
]


def _traj_color(track_id):
    return _TRAJ_COLORS[track_id % len(_TRAJ_COLORS)]


def draw_preview(frames, valid_hull, players, rackets, balls):
    """在每帧上叠加球场轮廓、人物/球拍检测框、网球轨迹箭头和帧号。"""
    fh = frames[0].shape[0]
    scale, thick       = text_params(fh)
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
        # 球场轮廓
        cv2.polylines(frame, [valid_hull], True, (0, 255, 255), 2)

        # 人物检测框 + ID
        for det in players[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if det.get('track_id') is not None:
                cv2.putText(frame, f"P{det['track_id']}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick)

        # 球拍检测框
        for det in rackets[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        # 网球轨迹：当前帧及之前的所有点，相邻点之间画箭头
        for tid, pts in traj.items():
            color   = _traj_color(tid)
            visible = [(cx, cy) for fi, cx, cy in pts if fi <= i]
            for k in range(len(visible) - 1):
                cv2.arrowedLine(frame, visible[k], visible[k + 1], color, 2, tipLength=0.4)

        # 当前帧网球检测框
        for det in balls[i]:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            tid   = det.get('track_id')
            color = _traj_color(tid) if tid is not None else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # 帧号
        cv2.putText(frame, str(i), (margin, fh - margin),
                    cv2.FONT_HERSHEY_SIMPLEX, scale_large * 1.5, (0, 255, 0), thick_large)

        pct = (i + 1) * 100 // total
        print(f"[    draw] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)

    print(f"[    draw] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  default='input_videos/input_video.mp4')
    parser.add_argument('-m', '--model',  default='models/yolo26x.pt')
    parser.add_argument('-c', '--conf',   type=float, default=0.5)
    parser.add_argument('--imgsz',        type=int,   default=1920)
    parser.add_argument('-d', '--device', default=None,
                        help='推理设备：cpu / cuda / mps / 0 / 1 ...（默认自动）')
    parser.add_argument('--annotate',     action='store_true',
                        help='输出凸包外置黑干净视频 + COCO JSON（默认为叠加标注的预览视频）')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    mode = 'annotate' if args.annotate else 'preview'
    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  model   {args.model}")
    print(f"  conf    {args.conf:<12}  imgsz   {args.imgsz}")
    print(f"  device  {args.device or 'auto':<12}  mode    {mode}")
    print("─" * 60, flush=True)

    input_dir   = os.path.dirname(args.input) or '.'
    input_name  = os.path.splitext(os.path.basename(args.input))[0]
    output_stem = os.path.join(input_dir, input_name + '_out')

    frames, fps = read_video(args.input)

    court = CourtDetector()
    court.predict(frames[0])
    valid_hull = court.get_valid_zone_hull(frames[0].shape, height=6.0)

    objects = ObjectsDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    players, rackets, balls = objects.run(frames, valid_hull=valid_hull)

    balls = BallTracker.from_video(fps, court.get_px_per_meter()).run(balls)

    # 凸包外区域置黑
    fh, fw = frames[0].shape[:2]
    hull_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [valid_hull], 255)
    total = len(frames)
    nw = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(frames):
        frame[hull_mask == 0] = 0
        pct = (i + 1) * 100 // total
        print(f"[    mask] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    print(f"[    mask] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")

    if args.annotate:
        save_video(frames, output_stem + '.mp4', fps=fps)
        save_coco(frames, players, rackets, balls, output_stem + '.json')
    else:
        draw_preview(frames, valid_hull, players, rackets, balls)
        save_video(frames, output_stem + '.mp4', fps=fps)


if __name__ == "__main__":
    main()
