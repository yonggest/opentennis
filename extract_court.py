#!/usr/bin/env python3
"""
从输入视频中均匀采样帧，运行球场检测，输出图像 + COCO 标注文件。

标注格式与 tennis-court-26 数据集一致：
  - segmentation : 球场四角多边形（顺时针：远左→远右→近右→近左）
  - bbox         : 多边形外接矩形 [x, y, w, h]
  - keypoints    : 14 个关键点（扁平列表 [x0,y0, x1,y1, ...]，图像像素坐标）

检测失败的帧保存为无标注负样本（图像保留，不写标注）。

用法：
    python extract_court.py -i video.mp4  -o datasets/mycourt --pos 90 --neg 10
    python extract_court.py -i videos/    -o datasets/mycourt --pos 90 --neg 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from court_detector import CourtDetector

_JPEG_QUALITY = 95

# 球场四角在 MODEL_KPS_M 中的索引（顺时针：远左→远右→近右→近左）
_CORNER_IDX = [0, 1, 3, 2]


def _sample_indices(total: int, n: int) -> list[int]:
    """在 [0, total) 中均匀取最多 n 个帧索引。"""
    if n >= total:
        return list(range(total))
    step = total / n
    return sorted({int(i * step) for i in range(n)})


def _corner_polygon(kps_px: np.ndarray) -> list[float]:
    """从 14×2 关键点取 4 个外角，返回平坦坐标 [x0,y0,x1,y1,...]。"""
    return [float(v) for v in kps_px[_CORNER_IDX].flatten()]


def _bbox_from_poly(poly: list[float], w: int, h: int) -> list[float]:
    """平坦多边形坐标 → COCO bbox [x, y, w, h]，裁剪到图像边界。"""
    xs = poly[0::2]
    ys = poly[1::2]
    x1 = max(0.0, min(xs));  x2 = min(float(w), max(xs))
    y1 = max(0.0, min(ys));  y2 = min(float(h), max(ys))
    return [x1, y1, x2 - x1, y2 - y1]


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.strip(),
    )
    p.add_argument('-i', '--input',  required=True, metavar='PATH',
                   help='输入视频文件或目录（.mp4 / .avi）')
    p.add_argument('-o', '--output', required=True, metavar='DIR',
                   help='输出目录（图像 + _annotations.coco.json）')
    p.add_argument('--pos', type=int, required=True, metavar='N',
                   help='每个视频提取的有标注正样本帧数（检测成功）')
    p.add_argument('--neg', type=int, default=0, metavar='N',
                   help='每个视频提取的无标注负样本帧数（检测失败，默认 0）')
    p.add_argument('--seg-model', default=None, metavar='PT',
                   help='球场 seg 模型路径（默认 models/yolo26n-seg-court.pt）')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output)

    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(input_path.glob('*.mp4')) + sorted(input_path.glob('*.avi'))
        if not videos:
            print(f"错误: 未找到 .mp4/.avi 文件: {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"错误: 路径不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    print('─' * 60)
    print(f'  input     {input_path}')
    print(f'  output    {out_dir}')
    print(f'  pos       {args.pos} per video')
    print(f'  neg       {args.neg} per video')
    print(f'  videos    {len(videos)}')
    print('─' * 60, flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    detector = CourtDetector(seg_model=args.seg_model)

    coco_images      = []
    coco_annotations = []
    total_pos = total_neg = 0

    for video_path in videos:
        print(f'\n── {video_path.name}')

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print('  [ERROR] 无法打开视频', file=sys.stderr)
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total == 0:
            print('  [ERROR] 帧数为 0', file=sys.stderr)
            continue

        stem = video_path.stem
        pad  = len(str(total - 1))
        pos = neg = 0

        # 正样本：均匀索引
        pos_indices = set(_sample_indices(total, args.pos))

        # 负样本候选：随机抽取大池子（20 倍配额），排序后顺序读取避免频繁 seek
        neg_pool = sorted(random.sample(range(total), min(total, args.neg * 20)))
        neg_pool_set = set(neg_pool)

        # 合并两类候选，顺序扫描
        scan_indices = sorted(pos_indices | neg_pool_set)

        cap = cv2.VideoCapture(str(video_path))
        prev_idx = -1
        for fi in scan_indices:
            if pos >= args.pos and neg >= args.neg:
                break

            if fi != prev_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            prev_idx = fi
            if not ret:
                continue

            try:
                kps_flat = detector.predict(frame)
                det_ok = True
            except Exception as e:
                print(f'  [neg ] frame {fi}: {e}', flush=True)
                det_ok = False

            if det_ok:
                if fi in pos_indices and pos < args.pos:
                    kps_px = np.array(kps_flat, dtype=np.float32).reshape(14, 2)
                    poly   = _corner_polygon(kps_px)
                    bbox   = _bbox_from_poly(poly, fw, fh)
                    img_id = len(coco_images) + 1
                    fname  = f'{stem}-{str(fi).zfill(pad)}-pos.jpg'
                    cv2.imwrite(str(out_dir / fname), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                    coco_images.append({
                        'id': img_id, 'file_name': fname, 'width': fw, 'height': fh,
                    })
                    coco_annotations.append({
                        'id':           len(coco_annotations) + 1,
                        'image_id':     img_id,
                        'category_id':  1,
                        'bbox':         bbox,
                        'area':         bbox[2] * bbox[3],
                        'segmentation': [poly],
                        'keypoints':    [float(v) for v in kps_px.flatten()],
                        'iscrowd':      0,
                        'modified':     False,
                    })
                    pos += 1
            elif fi in neg_pool_set and neg < args.neg:
                img_id = len(coco_images) + 1
                fname  = f'{stem}-{str(fi).zfill(pad)}-neg.jpg'
                cv2.imwrite(str(out_dir / fname), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
                coco_images.append({
                    'id': img_id, 'file_name': fname, 'width': fw, 'height': fh,
                })
                # 不写标注，COCO 训练器将其视为背景帧
                neg += 1

        cap.release()
        total_pos += pos
        total_neg += neg
        print(f'  → pos={pos}  neg={neg}', flush=True)

    coco = {
        'categories': [{'id': 1, 'name': 'court', 'supercategory': 'court'}],
        'images':      coco_images,
        'annotations': coco_annotations,
    }
    dst = out_dir / '_annotations.coco.json'
    with open(dst, 'w') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    n_img = len(coco_images)
    n_ann = len(coco_annotations)
    print()
    print('─' * 60)
    print(f'  写出 {n_img} 帧  ({n_ann} 有标注 + {n_img - n_ann} 无标注负样本)')
    print(f'  → {dst}')
    print('─' * 60)


if __name__ == '__main__':
    main()
