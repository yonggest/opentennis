import contextlib
import cv2
import json
import os
import subprocess
import numpy as np


def pick_free_gpu():
    """返回空闲显存最多的 GPU 索引字符串；无 CUDA（如 macOS）时返回 None 让框架自动选。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            text=True,
        )
        best = max(
            ((int(idx.strip()), int(free.strip())) for idx, free in
             (line.split(',') for line in out.strip().splitlines())),
            key=lambda x: x[1],
        )
        return str(best[0])
    except Exception:
        return None


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
#   court        : {keypoints, ground_hull, volume_hull, vol_bottom_pts, vol_top_pts, court_bottom_pts, court_top_pts}  （可选）

_CATEGORIES = [
    {'id': 1, 'name': 'person',        'supercategory': 'person'},
    {'id': 2, 'name': 'tennis racket', 'supercategory': 'sports'},
    {'id': 3, 'name': 'sports ball',   'supercategory': 'sports'},
]
_CAT_ID = {'person': 1, 'tennis racket': 2, 'sports ball': 3}


def _serialize_court(court):
    """将 court dict 的 numpy 数组序列化为 JSON 可写格式。"""
    return {
        'keypoints':        np.array(court['keypoints']).reshape(14, 2).tolist(),
        'ground_hull':      np.array(court['ground_hull']).reshape(-1, 2).tolist(),
        'volume_hull':      np.array(court['volume_hull']).reshape(-1, 2).tolist(),
        'vol_bottom_pts':   np.array(court['vol_bottom_pts']).tolist(),
        'vol_top_pts':      np.array(court['vol_top_pts']).tolist(),
        'court_bottom_pts': np.array(court['court_bottom_pts']).tolist(),
        'court_top_pts':    np.array(court['court_top_pts']).tolist(),
    }


def _deserialize_court(raw):
    """将 JSON 中的 court dict 恢复为 numpy 数组。"""
    return {
        'keypoints':        np.array(raw['keypoints'],        dtype=np.float32).flatten(),
        'ground_hull':      np.array(raw['ground_hull'],      dtype=np.float32).reshape(-1, 1, 2),
        'volume_hull':      np.array(raw['volume_hull'],      dtype=np.float32).reshape(-1, 1, 2),
        'vol_bottom_pts':   np.array(raw['vol_bottom_pts'],   dtype=np.float32),
        'vol_top_pts':      np.array(raw['vol_top_pts'],      dtype=np.float32),
        'court_bottom_pts': np.array(raw['court_bottom_pts'], dtype=np.float32),
        'court_top_pts':    np.array(raw['court_top_pts'],    dtype=np.float32),
    }


def save_coco(width, height, players, rackets, balls, path, fps=None, court=None, video=None):
    """
    将检测结果保存为 COCO JSON。

    width/height : 视频帧尺寸（像素）
    court        : dict，包含球场关键点和缓冲区凸包（可选）：
                     keypoints      ndarray (28,)    — 14 个关键点
                     ground_hull    ndarray (4,1,2)  — 地面缓冲区四边形
                     volume_hull    ndarray (N,1,2)  — 立方体凸包
                     vol_bottom_pts ndarray (4,2)    — 立方体底面角点
                     vol_top_pts    ndarray (4,2)    — 立方体顶面角点
    """
    images, annotations = [], []
    annotation_id = 0

    for frame_id, (player_dets, racket_dets, ball_dets) in enumerate(zip(players, rackets, balls)):
        images.append({'id': frame_id, 'width': width, 'height': height, 'frame_id': frame_id})
        for category_name, dets in [('person', player_dets), ('tennis racket', racket_dets), ('sports ball', ball_dets)]:
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                box_w, box_h = x2 - x1, y2 - y1
                ann = {
                    'id':          annotation_id,
                    'image_id':    frame_id,
                    'category_id': _CAT_ID[category_name],
                    'bbox':        [x1, y1, box_w, box_h],
                    'area':        box_w * box_h,
                    'iscrowd':     0,
                    'score':       det['conf'],
                }
                if det.get('track_id') is not None:
                    ann['track_id'] = det['track_id']
                if det.get('interpolated'):
                    ann['interpolated'] = True
                if det.get('backward_found'):
                    ann['backward_found'] = True
                if det.get('_rescue'):
                    ann['rescue'] = True
                if 'valid' in det:
                    ann['valid'] = det['valid']
                if 'foot' in det:
                    ann['foot'] = det['foot']
                if 'center' in det:
                    ann['center'] = det['center']
                if 'keypoints' in det:
                    ann['keypoints'] = det['keypoints']
                annotations.append(ann)
                annotation_id += 1

    result = {
        'images':      images,
        'annotations': annotations,
        'categories':  _CATEGORIES,
    }
    if fps is not None:
        result['fps'] = fps
    if court is not None:
        result['court'] = _serialize_court(court)
    if video is not None:
        result['video'] = video

    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"[    coco] saved → {path}  ({annotation_id} annotations, {len(images)} frames)", flush=True)


def load_video_path(json_path):
    """从 JSON 读取 video 字段，返回绝对路径（字段不存在则返回 None）。"""
    with open(json_path) as f:
        rel = json.load(f).get('video')
    if rel is None:
        return None
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(json_path)), rel))


def propagate_video(from_json_path, to_json_path):
    """从 from_json 读取 video 字段，转换为相对于 to_json 目录的路径（不存在则返回 None）。"""
    with open(from_json_path) as f:
        rel = json.load(f).get('video')
    if rel is None:
        return None
    from_dir = os.path.dirname(os.path.abspath(from_json_path))
    to_dir   = os.path.dirname(os.path.abspath(to_json_path))
    abs_path = os.path.normpath(os.path.join(from_dir, rel))
    return os.path.relpath(abs_path, to_dir)


def load_detections(path):
    """
    读取 save_coco() 保存的检测 JSON。

    返回：fps, width, height,
          court_kps  (ndarray float32, shape (28,)),
          court      (dict，含 keypoints / ground_hull / volume_hull / vol_bottom_pts / vol_top_pts),
          players, rackets, balls  (list[list[dict]])
          每个 det dict 含 bbox [x1,y1,x2,y2] / conf / track_id / valid
    """
    with open(path) as f:
        data = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in data.get('categories', [])}
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
        frame_idx = ann['image_id']
        x, y, w, h = ann['bbox']
        det = {
            'bbox':     [x, y, x + w, y + h],
            'conf':     ann.get('score', 1.0),
            'track_id': ann.get('track_id'),
            'valid':    ann.get('valid', True),
        }
        if ann.get('interpolated'):
            det['interpolated'] = True
        if ann.get('backward_found'):
            det['backward_found'] = True
        if ann.get('rescue'):
            det['rescue'] = True
        if 'foot' in ann:
            det['foot'] = ann['foot']
        if 'center' in ann:
            det['center'] = ann['center']
        if 'keypoints' in ann:
            det['keypoints'] = ann['keypoints']
        name = cat_id_to_name.get(ann['category_id'], '')
        if name == 'person':
            players[frame_idx].append(det)
        elif name == 'tennis racket':
            rackets[frame_idx].append(det)
        elif name == 'sports ball':
            balls[frame_idx].append(det)

    court = _deserialize_court(data.get('court', {}))

    print(f"[    json] loaded ← {path}  ({n_frames} frames)", flush=True)
    return fps, width, height, court, players, rackets, balls


