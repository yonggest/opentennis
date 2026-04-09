import cv2
import json
import os
import subprocess
import time

import numpy as np


def text_params(frame_height, base_height=1080):
    """根据帧高度返回 (font_scale, thickness)，基准为 1080p。"""
    scale = frame_height / base_height
    return scale * 0.6, max(1, round(scale))


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def save_video(frames, path, fps=24):
    out_path = os.path.splitext(path)[0] + '.mp4'
    h, w = frames[0].shape[:2]
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-vcodec', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = len(frames)
    nw = len(str(total))
    t0 = time.time()
    for i, frame in enumerate(frames):
        proc.stdin.write(frame.tobytes())
        pct = (i + 1) * 100 // total
        print(f"[   video] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"[   video] {total:>{nw}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")
    print(f"[   video] saved → {out_path}", flush=True)


# ── COCO JSON 格式 ────────────────────────────────────────────────────────────
# detect.py 保存、render.py 和 browse.py 读取的统一格式。
# 顶层结构：
#   images       : [{id, width, height, frame_id}, ...]
#   annotations  : [{id, image_id, category_id, bbox [x,y,w,h], area, iscrowd, score}, ...]
#   categories   : [{id, name, supercategory}, ...]
#   fps          : float
#   court        : {keypoints: [[x,y]×14], valid_hull: [[x,y]×N]}  （可选）

_CATEGORIES = [
    {'id': 1, 'name': 'person',        'supercategory': 'person'},
    {'id': 2, 'name': 'tennis racket', 'supercategory': 'sports'},
    {'id': 3, 'name': 'sports ball',   'supercategory': 'sports'},
]
_CAT_ID = {'person': 1, 'tennis racket': 2, 'sports ball': 3}
_CAT_NAME = {v: k for k, v in _CAT_ID.items()}


def save_coco(frames, players, rackets, balls, path, fps=None, court_kps=None, valid_hull=None):
    """
    将检测结果保存为 COCO JSON。

    court_kps  : ndarray (28,)   — CourtDetector.predict() 的返回值（可选）
    valid_hull : ndarray (N,1,2) — CourtDetector.get_valid_zone_hull() 的返回值（可选）
    """
    fh, fw = frames[0].shape[:2]
    images, annotations = [], []
    ann_id = 0

    for frame_id, (p_list, r_list, b_list) in enumerate(zip(players, rackets, balls)):
        images.append({'id': frame_id, 'width': fw, 'height': fh, 'frame_id': frame_id})
        for cat_name, dets in [('person', p_list), ('tennis racket', r_list), ('sports ball', b_list)]:
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                bw, bh = x2 - x1, y2 - y1
                annotations.append({
                    'id':          ann_id,
                    'image_id':    frame_id,
                    'category_id': _CAT_ID[cat_name],
                    'bbox':        [x1, y1, bw, bh],
                    'area':        bw * bh,
                    'iscrowd':     0,
                    'score':       det['conf'],
                })
                ann_id += 1

    result = {
        'images':      images,
        'annotations': annotations,
        'categories':  _CATEGORIES,
    }
    if fps is not None:
        result['fps'] = fps
    if court_kps is not None:
        result['court'] = {
            'keypoints':  np.array(court_kps).reshape(14, 2).tolist(),
            'valid_hull': valid_hull.reshape(-1, 2).tolist() if valid_hull is not None else [],
        }

    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[    coco] saved → {path}  ({ann_id} annotations, {len(images)} frames)", flush=True)


def load_detections(path):
    """
    读取 save_coco() 保存的检测 JSON。

    返回：fps, width, height,
          court_kps  (ndarray float32, shape (28,)),
          valid_hull (ndarray int32,   shape (N, 1, 2)),
          players, rackets, balls  (list[list[dict]])
          每个 det dict 含 bbox [x1,y1,x2,y2] / conf / track_id
    """
    with open(path) as f:
        data = json.load(f)

    cat_name = {c['id']: c['name'] for c in data.get('categories', [])}
    images   = {img['id']: img for img in data.get('images', [])}
    fps      = float(data.get('fps', 25.0))
    first    = next(iter(images.values())) if images else {}
    width    = first.get('width',  0)
    height   = first.get('height', 0)

    n_frames   = len(images)
    players    = [[] for _ in range(n_frames)]
    rackets    = [[] for _ in range(n_frames)]
    balls      = [[] for _ in range(n_frames)]
    for ann in data.get('annotations', []):
        fi = ann['image_id']
        x, y, w, h = ann['bbox']
        det = {'bbox': [x, y, x + w, y + h], 'conf': ann.get('score', 1.0), 'track_id': None}
        name = cat_name.get(ann['category_id'], '')
        if name == 'person':
            players[fi].append(det)
        elif name == 'tennis racket':
            rackets[fi].append(det)
        elif name == 'sports ball':
            balls[fi].append(det)

    court      = data.get('court', {})
    court_kps  = np.array(court.get('keypoints', [[0, 0]] * 14), dtype=np.float32).flatten()
    valid_hull = np.array(court.get('valid_hull', [[0, 0]]),      dtype=np.int32).reshape(-1, 1, 2)

    print(f"[    json] loaded ← {path}  ({n_frames} frames)", flush=True)
    return fps, width, height, court_kps, valid_hull, players, rackets, balls
