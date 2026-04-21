#!/usr/bin/env python3
"""
从流水线 JSON（detect/track/parse 输出）中提取视频帧，生成 COCO 格式标注数据集。

视频路径从 JSON 的 `video` 字段读取。
帧过滤：至少含一个满足条件的标注的帧才被提取。

用法：
    python build_coco_dataset.py -i video.parsed.json -o datasets/mydata
    python build_coco_dataset.py -i video.parsed.json -o datasets/mydata \\
        --filter valid --category "sports ball"
    python build_coco_dataset.py -i video.parsed.json -o datasets/mydata \\
        --filter all --category "sports ball" "person"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

_STRIP_FIELDS = {'score', 'track_id', 'valid', 'interpolated',
                 'foot', 'center', 'backward_found', '_backward'}
_JPEG_QUALITY = 95


def _resolve_video(json_path: Path) -> Path:
    with open(json_path) as f:
        rel = json.load(f).get('video')
    if rel is None:
        raise RuntimeError("JSON 中没有 video 字段，无法定位视频文件")
    abs_path = (json_path.parent / rel).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {abs_path}")
    return abs_path


def _ann_matches(ann: dict, cat_ids: set | None, filter_mode: str) -> bool:
    """判断一个标注是否满足过滤条件。"""
    if cat_ids is not None and ann.get('category_id') not in cat_ids:
        return False
    if filter_mode == 'all':
        return True
    if filter_mode == 'valid':
        return ann.get('valid', True) and not ann.get('interpolated', False)
    if filter_mode == 'invalid':
        return not ann.get('valid', True)
    if filter_mode == 'interpolated':
        return ann.get('valid', True) and ann.get('interpolated', False)
    return True


def build_coco_dataset(json_path: Path, out_dir: Path,
                       filter_mode: str, categories: list[str] | None) -> None:
    with open(json_path) as f:
        src = json.load(f)

    # 解析类别
    all_cats = {c['name']: c['id'] for c in src.get('categories', [])}
    if categories:
        missing = [c for c in categories if c not in all_cats]
        if missing:
            raise ValueError(f"JSON 中不存在类别: {missing}  可用: {list(all_cats)}")
        cat_ids = {all_cats[c] for c in categories}
        out_cats = [c for c in src['categories'] if c['name'] in categories]
    else:
        cat_ids = None
        out_cats = src.get('categories', [])

    # 按 image_id 整理标注
    images_map = {img['id']: img for img in src.get('images', [])}
    anns_by_image: dict[int, list] = {}
    for ann in src.get('annotations', []):
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    # 找出满足条件的 frame_idx（image_id 即 frame_idx）
    kept_frame_ids = sorted(
        iid for iid, anns in anns_by_image.items()
        if any(_ann_matches(a, cat_ids, filter_mode) for a in anns)
    )
    if not kept_frame_ids:
        print("没有满足条件的帧，退出。")
        return

    print(f"满足条件的帧: {len(kept_frame_ids)} / {len(images_map)}")

    # 定位视频
    video_path = _resolve_video(json_path)
    print(f"视频: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # 逐帧读取，只保存目标帧
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pad     = len(str(max(total - 1, 0)))

    target_set = set(kept_frame_ids)
    coco_images = []
    new_id_map: dict[int, int] = {}   # old frame_id → new 1-based image_id

    with tqdm(total=len(kept_frame_ids), unit='frame') as pbar:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx not in target_set:
                continue
            new_id = len(coco_images) + 1
            new_id_map[frame_idx] = new_id
            fname = f"{stem}-{str(frame_idx).zfill(pad)}.jpg"
            cv2.imwrite(str(out_dir / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            coco_images.append({
                'id':        new_id,
                'file_name': fname,
                'width':     frame_w,
                'height':    frame_h,
            })
            pbar.update(1)
    cap.release()

    # 迁移标注：与帧选择标准一致，只写出满足 filter 条件的标注
    coco_annotations = []
    for frame_id in kept_frame_ids:
        new_img_id = new_id_map.get(frame_id)
        if new_img_id is None:
            continue
        for ann in anns_by_image.get(frame_id, []):
            if not _ann_matches(ann, cat_ids, filter_mode):
                continue
            clean = {k: v for k, v in ann.items() if k not in _STRIP_FIELDS}
            clean['id']       = len(coco_annotations) + 1
            clean['image_id'] = new_img_id
            coco_annotations.append(clean)

    coco = {
        'categories':  out_cats,
        'images':      coco_images,
        'annotations': coco_annotations,
    }
    dst = out_dir / '_annotations.coco.json'
    with open(dst, 'w') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"写出 {len(coco_images)} 帧  {len(coco_annotations)} 个标注 → {dst}")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.strip(),
    )
    p.add_argument('-i', '--input',    required=True, metavar='JSON',
                   help='流水线输出 JSON（detect/track/parse 任意阶段）')
    p.add_argument('-o', '--output',   required=True, metavar='DIR',
                   help='输出目录（图像 + _annotations.coco.json）')
    p.add_argument('--filter', default='valid',
                   choices=['valid', 'invalid', 'interpolated', 'all'],
                   help='标注过滤模式：'
                        'valid=有效非插值  invalid=无效  interpolated=插值  all=全量')
    p.add_argument('--category', nargs='+', metavar='NAME',
                   help='只保留指定类别（如 "sports ball" "person"），不传则不限')
    if len(sys.argv) == 1:
        p.print_help(); sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.input)
    out_dir   = Path(args.output)

    if not json_path.exists():
        print(f"Error: 找不到 {json_path}", file=sys.stderr); sys.exit(1)

    print("─" * 60)
    print(f"  input      {json_path}")
    print(f"  output     {out_dir}")
    print(f"  filter     {args.filter}")
    print(f"  category   {args.category or '(全部)'}")
    print("─" * 60, flush=True)

    try:
        build_coco_dataset(json_path, out_dir, args.filter, args.category)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == '__main__':
    main()
