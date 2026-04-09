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

from utils import read_video, save_coco
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

    frames, fps = read_video(args.input)

    court = CourtDetector()
    kps   = court.predict(frames[0])
    hull  = court.get_valid_zone_hull(frames[0].shape, height=6.0)

    objects = ObjectsDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=args.device)
    players, rackets, balls = objects.run(frames, valid_hull=hull)

    save_coco(frames, players, rackets, balls, output_path,
              fps=fps, court_kps=kps, valid_hull=hull)


if __name__ == '__main__':
    main()
