#!/usr/bin/env python3
"""
从流水线 JSON（track/parse 输出）中提取视频帧，生成 COCO 格式标注数据集。

视频路径从 JSON 的 `video` 字段读取。
帧过滤：至少含一个网球标注落在指定空间区域的帧才被提取。
--category 控制写出哪些类别的标注（不传则全部写出）。

用法：
    python extract_dataset.py -i video.tracked.json -o datasets/mydata
    python extract_dataset.py -i video.tracked.json -o datasets/mydata -p net
    python extract_dataset.py -i video.tracked.json -o datasets/mydata -p racket \\
        --category "sports ball" "tennis racket"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize_scalar

from court_detector import (MODEL_KPS_M, NET_Y, COURT_L, COURT_W,
                            CLEARANCE_BACK, CLEARANCE_SIDE)

# ── 空间过滤参数 ───────────────────────────────────────────────────────────────
_NET_MARGIN_M  = 3.5   # 距球网 ±N 米以内（覆盖发球区附近）
_BACK_MARGIN_M = 4.0   # 距远端底线 N 米以内（含底线内侧）
_NET_H_POST    = 1.07  # ITF 标准：网带立柱高度（m）
_NET_POST_OFF  = 0.914 # ITF 标准：立柱在双打侧线外的距离（m）
_BACKDROP_H    = 4.0   # 远端背景板高度（m）

# 写入 COCO 时剔除的运行时字段
_STRIP_FIELDS = {'score', 'track_id', 'valid', 'foot', 'center'}

_JPEG_QUALITY = 95


# ── 空间过滤工具 ───────────────────────────────────────────────────────────────

def _compute_homographies(court_kps_raw):
    """从 JSON court.keypoints（14×2 列表）返回 (H_inv, H)。
    H_inv : 图像坐标 → 球场米坐标
    H     : 球场米坐标 → 图像坐标（地面单应）
    """
    kps_2d = np.array(court_kps_raw, dtype=np.float32).reshape(14, 2)
    H_inv, _ = cv2.findHomography(kps_2d, MODEL_KPS_M)
    H,     _ = cv2.findHomography(MODEL_KPS_M, kps_2d)
    return H_inv, H


def _recover_camera_P(H, image_shape):
    """从地面单应矩阵 H（球场坐标→图像坐标）恢复 3×4 投影矩阵 P。"""
    h_img, w_img = image_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    h1, h2 = H[:, 0], H[:, 1]

    def cost(f):
        Ki = np.array([[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1.0]])
        r1 = Ki @ h1;  r2 = Ki @ h2
        return (r1 @ r2) ** 2 + (r1 @ r1 - r2 @ r2) ** 2

    f   = minimize_scalar(cost, bounds=(w_img * 0.3, w_img * 20), method='bounded').x
    K   = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki  = np.linalg.inv(K)
    lam = 1.0 / np.linalg.norm(Ki @ h1)
    r1  = lam * (Ki @ H[:, 0])
    r2  = lam * (Ki @ H[:, 1])
    r3  = np.cross(r1, r2)
    t   = lam * (Ki @ H[:, 2])
    if t[2] < 0:
        r1, r2, r3, t = -r1, -r2, -r3, -t
    R = np.column_stack([r1, r2, r3])
    # 确保 z 向上（高处投影 v 更小）
    mid       = np.array([COURT_W / 2, COURT_L / 2, 0.0])
    cam_mid   = R @ mid + t
    cam_above = R @ (mid + [0, 0, 1]) + t
    if (K[1, 1] * cam_above[1] / cam_above[2] + K[1, 2] >
            K[1, 1] * cam_mid[1]   / cam_mid[2]   + K[1, 2]):
        r3 = -r3
        R  = np.column_stack([r1, r2, r3])
    return K @ np.hstack([R, t[:, None]])


def _project_3d(pts3d, P):
    """将 (N,3) 世界坐标通过投影矩阵 P 投影为 (N,2) 图像坐标。"""
    ph  = np.hstack([pts3d, np.ones((len(pts3d), 1))])
    uv  = (P @ ph.T).T
    return (uv[:, :2] / uv[:, 2:]).astype(np.float32)


def _filter_region_polygon(filter_mode: str, H, frame_w: int, frame_h: int):
    """把过滤区域投影到图像坐标，返回 (bbox, polygon)。
    bbox    : [x, y, w, h]  — 梯形的外接矩形
    polygon : [[x,y], ...]  — 4 个角点（顺序：左下、右下、右上、左上）
    仅 net / backdrop 有固定区域；其他模式返回 None。

    net      : 网带立柱位置（双打侧线外 0.914 m），高度 1.07 m（ITF 立柱高度）
    backdrop : 球场缓冲区远端（CLEARANCE_BACK），宽度含侧向缓冲（CLEARANCE_SIDE），高 3 m
    """
    P = _recover_camera_P(H, (frame_h, frame_w))

    if filter_mode == 'net':
        x0 = -_NET_POST_OFF
        x1 = COURT_W + _NET_POST_OFF
        pts3d = np.array([
            [x0, NET_Y, 0          ],   # 左立柱底
            [x1, NET_Y, 0          ],   # 右立柱底
            [x1, NET_Y, _NET_H_POST],   # 右立柱顶
            [x0, NET_Y, _NET_H_POST],   # 左立柱顶
        ], dtype=np.float32)
    elif filter_mode == 'backdrop':
        x0 = -CLEARANCE_SIDE
        x1 = COURT_W + CLEARANCE_SIDE
        y_back = -CLEARANCE_BACK        # 远端缓冲区末端（y < 0，球场外）
        pts3d = np.array([
            [x0, y_back, 0          ],  # 左底
            [x1, y_back, 0          ],  # 右底
            [x1, y_back, _BACKDROP_H],  # 右顶
            [x0, y_back, _BACKDROP_H],  # 左顶
        ], dtype=np.float32)
    else:
        return None

    pts2d = _project_3d(pts3d, P)
    polygon = [[float(p[0]), float(p[1])] for p in pts2d]
    xs, ys  = pts2d[:, 0], pts2d[:, 1]
    bx1 = float(max(0,       xs.min()))
    by1 = float(max(0,       ys.min()))
    bx2 = float(min(frame_w, xs.max()))
    by2 = float(min(frame_h, ys.max()))
    return [bx1, by1, bx2 - bx1, by2 - by1], polygon


def _to_court(cx: float, cy: float, H_inv: np.ndarray) -> np.ndarray:
    """把图像像素坐标投影到球场米坐标，返回 (x_m, y_m)。"""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H_inv)[0][0]


def _ball_in_region(ann: dict, racket_anns: list,
                    filter_mode: str, region_poly=None) -> bool:
    """判断单个标注是否落在目标空间区域内。
    net/backdrop : 用预计算的图像空间多边形做点包含测试（与可视化完全一致）
    racket       : bbox 与任意球拍重叠
    all          : 无过滤，始终返回 True
    """
    if filter_mode == 'all':
        return True
    x, y, w, h = ann['bbox']
    cx, cy = x + w / 2, y + h / 2
    if filter_mode in ('net', 'backdrop'):
        if region_poly is None:
            return False
        return cv2.pointPolygonTest(region_poly, (cx, cy), False) >= 0
    if filter_mode == 'racket':
        bx1, by1, bx2, by2 = x, y, x + w, y + h
        return any(bx1 < rx + rw and bx2 > rx and by1 < ry + rh and by2 > ry
                   for rann in racket_anns
                   for rx, ry, rw, rh in [rann['bbox']])
    return False


def _frame_matches(ball_anns: list, racket_anns: list,
                   filter_mode: str, region_poly=None,
                   conf_high: float = 0.5,
                   sample_mode: str = 'interpolated') -> bool:
    """判断当前帧中是否有目标网球落在目标空间区域。

    sample_mode 控制哪类球标注计入候选：
      interpolated : 插值点（recall 补插，主检测器未检出）
      low-conf     : 低置信度（score < conf_high）
      high-conf    : 高置信度（score >= conf_high）
    """
    if sample_mode == 'interpolated':
        candidates = [a for a in ball_anns if a.get('interpolated')]
    elif sample_mode == 'low-conf':
        candidates = [a for a in ball_anns if a.get('score', 1.0) < conf_high]
    else:  # high-conf
        candidates = [a for a in ball_anns if a.get('score', 0.0) >= conf_high]
    if not candidates:
        return False
    if filter_mode == 'all':
        return True
    return any(_ball_in_region(a, racket_anns, filter_mode, region_poly)
               for a in candidates)


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def _resolve_video(json_path: Path) -> Path:
    with open(json_path) as f:
        rel = json.load(f).get('video')
    if rel is None:
        raise RuntimeError("JSON 中没有 video 字段，无法定位视频文件")
    abs_path = (json_path.parent / rel).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {abs_path}")
    return abs_path


def extract_dataset(json_path: Path, out_dir: Path,
                    filter_mode: str, categories: list[str] | None,
                    conf_high: float = 0.5,
                    sample_mode: str = 'interpolated',
                    max_frames: int = 200) -> None:
    with open(json_path) as f:
        src = json.load(f)

    # 读取球场数据（net/backdrop 过滤需要）
    court_raw = src.get('court', {})
    H = None
    if filter_mode in ('net', 'backdrop'):
        _, H = _compute_homographies(court_raw.get('keypoints', []))

    # 类别映射
    all_cats    = src.get('categories', [])
    cat_name    = {c['id']: c['name'] for c in all_cats}
    ball_cids   = {cid for cid, n in cat_name.items() if 'ball'   in n}
    racket_cids = {cid for cid, n in cat_name.items() if 'racket' in n}

    # 输出类别：--category 限定，否则全部
    if categories:
        missing = [c for c in categories if c not in cat_name.values()]
        if missing:
            raise ValueError(f"JSON 中不存在类别: {missing}  可用: {list(cat_name.values())}")
        out_cat_ids = {cid for cid, n in cat_name.items() if n in categories}
        out_cats    = [c for c in all_cats if c['name'] in categories]
    else:
        out_cat_ids = None   # None 表示全部
        out_cats    = all_cats

    # 按帧整理标注
    anns_by_frame: dict[int, list] = {}
    for ann in src.get('annotations', []):
        anns_by_frame.setdefault(ann['image_id'], []).append(ann)

    images_map = {img['id']: img for img in src.get('images', [])}

    # 从第一帧获取图像尺寸（后续打开视频时会再次赋值，此处用于提前计算多边形）
    first_img = next(iter(images_map.values()), {})
    frame_w = first_img.get('width',  0)
    frame_h = first_img.get('height', 0)

    # 提前计算过滤区域多边形（net/backdrop），供帧选择和标注输出共用
    region_result = _filter_region_polygon(filter_mode, H, frame_w, frame_h) if H is not None else None
    if region_result is not None:
        region_poly = np.array(region_result[1], dtype=np.float32).reshape(-1, 1, 2)
    else:
        region_poly = None

    # 帧选择：至少一个目标网球落在空间过滤区域
    kept_frame_ids = sorted(
        fid for fid, anns in anns_by_frame.items()
        if _frame_matches(
            [a for a in anns if a['category_id'] in ball_cids],
            [a for a in anns if a['category_id'] in racket_cids],
            filter_mode, region_poly, conf_high, sample_mode,
        )
    )
    if not kept_frame_ids:
        print("没有满足条件的帧，退出。")
        return

    total_matched = len(kept_frame_ids)
    n = min(total_matched, max_frames)
    kept_frame_ids = sorted(random.sample(kept_frame_ids, n))
    print(f"随机抽取: {n} 帧 / 满足条件 {total_matched} 帧")

    # 定位视频
    video_path = _resolve_video(json_path)
    print(f"视频: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem  = video_path.stem
    total = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
    pad   = len(str(max(total - 1, 0)))

    target_set  = set(kept_frame_ids)
    coco_images = []
    new_id_map: dict[int, int] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or frame_w
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame_h

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

    # 输出标注：只保留落在区域内的指定类别标注
    coco_annotations = []
    for frame_id in kept_frame_ids:
        new_img_id = new_id_map.get(frame_id)
        if new_img_id is None:
            continue
        frame_anns  = anns_by_frame.get(frame_id, [])
        racket_anns = [a for a in frame_anns if a['category_id'] in racket_cids]
        for ann in frame_anns:
            if out_cat_ids is not None and ann['category_id'] not in out_cat_ids:
                continue
            if not _ball_in_region(ann, racket_anns, filter_mode, region_poly):
                continue
            clean = {k: v for k, v in ann.items() if k not in _STRIP_FIELDS}
            clean['id']       = len(coco_annotations) + 1
            clean['image_id'] = new_img_id
            coco_annotations.append(clean)

    # 过滤区域标注：把空间区域投影到图像坐标，每帧写一个 region 多边形，供可视化核查
    region_cat_id = max((c['id'] for c in out_cats), default=0) + 1
    if region_result is not None:
        region_bbox, region_polygon = region_result
        out_cats = out_cats + [{'id': region_cat_id,
                                'name': f'filter: {filter_mode}',
                                'supercategory': 'filter'}]
        for new_img_id in new_id_map.values():
            coco_annotations.append({
                'id':          len(coco_annotations) + 1,
                'image_id':    new_img_id,
                'category_id': region_cat_id,
                'bbox':        region_bbox,
                'area':        region_bbox[2] * region_bbox[3],
                'iscrowd':     0,
                'region':      True,
                'segmentation': [[coord for pt in region_polygon for coord in pt]],
                'modified':    False,
            })

    coco = {
        'categories':  out_cats,
        'images':      coco_images,
        'annotations': coco_annotations,
    }
    dst = out_dir / '_annotations.coco.json'
    with open(dst, 'w') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    print(f"写出 {len(coco_images)} 帧  {len(coco_annotations)} 个标注 → {dst}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.strip(),
    )
    p.add_argument('-i', '--input',  required=True, metavar='JSON',
                   help='track.py 或 parse.py 输出的 JSON')
    p.add_argument('-o', '--output', required=True, metavar='DIR',
                   help='输出目录（图像 + _annotations.coco.json）')
    p.add_argument('--position', '-p', default='all',
                   choices=['all', 'net', 'backdrop', 'racket'],
                   help='位置过滤：all=全量  net=球网附近  '
                        'backdrop=远端背景板  racket=球与球拍重叠')
    p.add_argument('--category', nargs='+', metavar='NAME',
                   help='输出标注的类别（如 "sports ball" "tennis racket"），不传则全部输出')
    p.add_argument('--sample', default='interpolated',
                   choices=['interpolated', 'low-conf', 'high-conf'],
                   help='帧选择依据：interpolated=插值点  low-conf=低置信度  high-conf=高置信度')
    p.add_argument('--num-frames', type=int, default=200, metavar='N',
                   help='随机抽取帧数（默认 200），不足时取全部')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args     = parse_args()
    json_path = Path(args.input)
    out_dir   = Path(args.output)

    if not json_path.exists():
        print(f"Error: 找不到 {json_path}", file=sys.stderr)
        sys.exit(1)

    print("─" * 60)
    print(f"  input   {json_path}")
    print(f"  output  {out_dir}")
    print(f"  position  {args.position}")
    print(f"  sample     {args.sample}")
    print(f"  num-frames {args.num_frames}")
    print(f"  category   {args.category or '(全部)'}")
    print("─" * 60, flush=True)

    try:
        extract_dataset(json_path, out_dir, args.position, args.category,
                        sample_mode=args.sample, max_frames=args.num_frames)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
