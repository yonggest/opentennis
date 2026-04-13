#!/usr/bin/env python3
"""
从视频中提取帧，保存为 JPEG 图片，并生成标准 COCO 格式 JSON。

文件名格式：{视频名}-{帧序号}.jpg，帧序号补零至统一长度。

如果通过 -j 指定或在视频目录下找到同名 .json，会将有效标注（valid=True）迁移到
新 COCO JSON，image_id 重新映射。track_id / valid / score 等运行时字段不写入输出。

用法：
    python build_coco.py -i <视频文件> [-o <输出目录>] [-j <检测JSON>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# 迁移时从 annotation 中剔除的运行时字段（非标准 COCO）
_STRIP_FIELDS = {"score", "track_id", "valid"}


def build_coco(video_path: str, output_dir: str | None = None,
               json_src_path: str | None = None) -> None:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"{video_path} not found")

    out_dir = Path(output_dir) if output_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pad     = len(str(max(total - 1, 0)))
    stem    = path.stem

    # ── 读取源 JSON ───────────────────────────────────────────────────────────
    src_data: dict = {}
    json_src = Path(json_src_path) if json_src_path else path.parent / (stem + ".json")
    if json_src.exists():
        with open(json_src) as f:
            src_data = json.load(f)
        print(f"源 JSON: {json_src}")

    # ── 提取帧 ────────────────────────────────────────────────────────────────
    coco_images = []
    with tqdm(total=total, unit="frame") as pbar:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            fname = f"{stem}-{str(frame_idx).zfill(pad)}.jpg"
            cv2.imwrite(str(out_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            coco_images.append({
                "id":        frame_idx + 1,   # COCO 惯例：1-based
                "file_name": fname,
                "width":     frame_w,
                "height":    frame_h,
            })
            pbar.update(1)
    cap.release()
    print(f"提取 {len(coco_images)} 帧到 {out_dir}/")

    # ── 迁移源 JSON 中的有效标注 ──────────────────────────────────────────────
    src_annotations = src_data.get("annotations", [])
    src_categories  = src_data.get("categories",  [])

    # detect.py / parse.py 的 image.id 等于 0-based frame_id，新 COCO 用 1-based
    src_id_to_new_id = {
        img["id"]: img.get("frame_id", img["id"]) + 1
        for img in src_data.get("images", [])
    }

    if src_annotations:
        print(f"源标注已加载: {len(src_annotations)} 个标注")

    coco_annotations = []
    skipped_invalid = 0
    skipped_remap   = 0
    for ann_id, ann in enumerate(src_annotations, start=1):
        if not ann.get("valid", True):
            skipped_invalid += 1
            continue
        new_img_id = src_id_to_new_id.get(ann["image_id"])
        if new_img_id is None:
            skipped_remap += 1
            continue
        new_ann = {k: v for k, v in ann.items() if k not in _STRIP_FIELDS}
        new_ann["id"]       = ann_id
        new_ann["image_id"] = new_img_id
        coco_annotations.append(new_ann)

    if skipped_invalid:
        print(f"  跳过无效标注（valid=False）: {skipped_invalid} 个")
    if skipped_remap:
        print(f"  警告: {skipped_remap} 个标注无法映射到帧（已跳过）")

    # ── 写出 COCO JSON ────────────────────────────────────────────────────────
    coco = {
        "categories":  src_categories,
        "images":      coco_images,
        "annotations": coco_annotations,
    }
    dst = out_dir / "_annotations.coco.json"
    with open(dst, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    print(f"COCO saved: {dst}  ({len(coco_images)} 张图片，{len(coco_annotations)} 个标注)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python build_coco.py -i video.mp4\n"
            "  python build_coco.py -i video.mp4 -o frames/ -j video.parse.json\n"
        ),
    )
    parser.add_argument("-i", "--input",  required=True, metavar="FILE",
                        help="输入视频文件")
    parser.add_argument("-o", "--output", metavar="DIR",
                        help="输出目录（默认：视频所在目录）")
    parser.add_argument("-j", "--json",   metavar="JSON",
                        help="detect.py 或 parse.py 输出的 JSON（默认：视频同名 .json）")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    try:
        build_coco(args.input, args.output, args.json)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
