import contextlib
import cv2
import json
import os
import subprocess
import numpy as np


def text_params(frame_height, base_height=1080):
    """根据帧高度返回 (font_scale, thickness)，基准为 1080p。"""
    scale = frame_height / base_height
    return scale * 0.6, max(1, round(scale))


def video_info(video_path):
    """返回 (fps, width, height, frame_count)，不读取任何帧。"""
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, count


def iter_frames(video_path):
    """逐帧生成器，每次只有一帧在内存中。适合大文件。"""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


@contextlib.contextmanager
def open_video_writer(path, fps, width, height):
    """
    上下文管理器，返回可写入原始 BGR24 帧字节的 stdin 管道。
    退出时自动关闭管道并等待 ffmpeg 完成。

    用法：
        with open_video_writer(path, fps, w, h) as pipe:
            pipe.write(frame.tobytes())
    """
    out_path = os.path.splitext(path)[0] + '.mp4'
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', 'pipe:0',
        '-vcodec', 'libx264', '-crf', '18', '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield proc.stdin
    finally:
        proc.stdin.close()
        proc.wait()


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


def save_coco(width, height, players, rackets, balls, path, fps=None, court_kps=None, valid_hull=None):
    """
    将检测结果保存为 COCO JSON。

    width/height : 视频帧尺寸（像素）
    court_kps    : ndarray (28,)   — CourtDetector.predict() 的返回值（可选）
    valid_hull   : ndarray (N,1,2) — CourtDetector.get_valid_zone_hull() 的返回值（可选）
    """
    fw, fh = width, height
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
