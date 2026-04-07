import argparse
import os
import sys
import time
import cv2
import numpy as np
from utils import read_video, save_video, save_coco, text_params
from objects_detector import ObjectsDetector
from court_detector import CourtDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',       default='input_videos/input_video.mp4')
    parser.add_argument('-m', '--model',       default='models/yolo26x.pt')
    parser.add_argument('-c', '--conf',        type=float, default=0.1)
    parser.add_argument('--imgsz',             type=int,   default=1920)
    parser.add_argument('-d', '--device',       default=None,
                        help='推理设备：cpu / cuda / mps / 0 / 1 ...（GPU编号，默认自动选择）')
    parser.add_argument('--annotate',          action='store_true',
                        help='输出凸包外置黑干净视频 + COCO JSON（默认为叠加检测框的预览视频）')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    input_dir   = os.path.dirname(args.input) or '.'
    input_name  = os.path.splitext(os.path.basename(args.input))[0]
    output_stem = os.path.join(input_dir, input_name + '_out')

    frames, fps = read_video(args.input)

    court = CourtDetector()
    court.predict(frames[0])
    valid_hull = court.get_valid_zone_hull(frames[0].shape, height=6.0)

    objects = ObjectsDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    players, rackets, balls = objects.run(frames, valid_hull=valid_hull)

    fh, fw = frames[0].shape[:2]
    hull_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [valid_hull], 255)

    total = len(frames)
    nw    = len(str(total))
    t0    = time.time()
    for i, frame in enumerate(frames):
        frame[hull_mask == 0] = 0
        pct = (i + 1) * 100 // total
        print(f"[    mask] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    print(f"[    mask] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")

    if args.annotate:
        save_video(frames, output_stem + '.mp4', fps=fps)
        save_coco(frames, players, rackets, balls, output_stem + '.json')
    else:
        scale, thick = text_params(fh)
        scale_large, thick_large = text_params(fh, base_height=1080)
        margin = int(fh * 0.028)
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
            for det in balls[i]:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thick)
                tid   = det.get('track_id')
                label = (f"B{tid}" if tid is not None else "B?") + f" {det['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick)
            cv2.putText(frame, str(i), (margin, fh - margin),
                        cv2.FONT_HERSHEY_SIMPLEX, scale_large * 1.5, (0, 255, 0), thick_large)
            pct = (i + 1) * 100 // total
            print(f"[    draw] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
        print(f"[    draw] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")
        save_video(frames, output_stem + '.mp4', fps=fps)


if __name__ == "__main__":
    main()
