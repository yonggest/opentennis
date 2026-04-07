import cv2
import os
import subprocess
import time
import json


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


def save_coco(frames, players, rackets, balls, path):
    """将检测结果保存为 COCO JSON 格式。"""
    CATEGORIES = [
        {'id': 1, 'name': 'person',        'supercategory': 'person'},
        {'id': 2, 'name': 'tennis racket', 'supercategory': 'sports'},
        {'id': 3, 'name': 'sports ball',   'supercategory': 'sports'},
    ]
    CAT_ID = {'person': 1, 'tennis racket': 2, 'sports ball': 3}

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
                    'category_id': CAT_ID[cat_name],
                    'bbox':        [x1, y1, bw, bh],
                    'area':        bw * bh,
                    'iscrowd':     0,
                    'score':       det['conf'],
                })
                ann_id += 1

    with open(path, 'w') as f:
        json.dump({'images': images, 'annotations': annotations, 'categories': CATEGORIES}, f, indent=2)
    print(f"[    coco] saved → {path}  ({ann_id} annotations, {len(images)} frames)")
