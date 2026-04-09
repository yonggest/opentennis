"""
第一阶段：球场检测 + 物体检测，结果保存为 COCO JSON。

用法：
    python detect.py -i <video> -m models/yolo26x.pt
    python detect.py -i <video> -o results/my.json -m models/yolo26x.pt
输出：
    <video>.json（默认）或 -o 指定的路径
"""

import argparse
import os
import sys

from utils import video_info, iter_frames, save_coco
from court_detector import CourtDetector
from objects_detector import ObjectsDetector


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-i', '--input',  required=True,               help='输入视频路径')
    p.add_argument('-o', '--output', default=None,                help='输出 JSON 路径（默认：输入同名，后缀改为 .json）')
    p.add_argument('-m', '--model',  default='models/yolo26x.pt', help='YOLO 模型路径')
    p.add_argument('-c', '--conf',   type=float, default=0.5,     help='检测置信度阈值')
    p.add_argument('--imgsz',        type=int,   default=1920,    help='推理图片尺寸')
    p.add_argument('-d', '--device', default=None,                help='推理设备：cpu / cuda / mps（默认自动）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or os.path.splitext(args.input)[0] + '.json'

    print("─" * 60)
    print(f"  input   {args.input}")
    print(f"  output  {output_path}")
    print(f"  model   {args.model}")
    print(f"  conf    {args.conf:<12}  imgsz   {args.imgsz}")
    print(f"  device  {args.device or 'auto'}")
    print("─" * 60, flush=True)

    fps, width, height, n_frames = video_info(args.input)
    first_frame = next(iter_frames(args.input))

    court = CourtDetector()
    kps   = court.predict(first_frame)
    hull  = court.get_valid_zone_hull(first_frame.shape, height=6.0)

    objects = ObjectsDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    players, rackets, balls = objects.run(
        iter_frames(args.input),
        valid_hull=hull,
        frame_shape=(height, width),
        total=n_frames,
    )

    save_coco(width, height, players, rackets, balls, output_path,
              fps=fps, court_kps=kps, valid_hull=hull)


if __name__ == '__main__':
    main()
