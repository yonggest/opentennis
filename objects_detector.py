import time

import torch


def _fmt(seconds):
    """将秒数格式化为 0:05、1:23、2:07:45 等形式。"""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
from ultralytics import YOLO

# 每类独立 NMS IoU 阈值
_NMS_IOU_PERSON  = 0.45   # 体型稳定，标准值
_NMS_IOU_RACKET  = 0.35   # 形变大，宁可少保留
_NMS_IOU_BALL    = 0.65   # 目标小，避免误删


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(dets, iou_thresh):
    """贪心 NMS：按置信度降序，IoU > iou_thresh 的重叠框移除。"""
    if len(dets) <= 1:
        return dets
    dets = sorted(dets, key=lambda d: d['conf'], reverse=True)
    keep = []
    for d in dets:
        if not any(_iou(d['bbox'], k['bbox']) > iou_thresh for k in keep):
            keep.append(d)
    return keep


class ObjectsDetector:
    """一次推断提取 person、tennis racket、sports ball 三类目标。"""

    def __init__(self, model_path, conf=0.1, imgsz=960, device=None):
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or self._auto_device(model_path)
        print(f"[  detect] loading model: {model_path}  device={self.device}", flush=True)
        self.model = YOLO(model_path, task='detect')
        name_to_id = {v: k for k, v in self.model.names.items()}
        self.class_ids = [name_to_id[n] for n in ('person', 'tennis racket', 'sports ball')
                          if n in name_to_id]

    @staticmethod
    def _auto_device(model_path):
        if model_path.endswith('.mlpackage'):
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def run(self, frames, total=None):
        """
        frames : list 或任意可迭代对象（生成器）
        total  : 总帧数，用于进度显示
        """
        total = total or (len(frames) if isinstance(frames, list) else 0)

        player_detections, racket_detections, ball_detections = [], [], []
        frame_num_width = len(str(total)) if total else 6
        t0 = time.time()

        for i, frame in enumerate(frames):
            results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                         classes=self.class_ids, device=self.device,
                                         iou=1.0,
                                         verbose=False, save=False)[0]
            players, rackets, balls = self._parse(results)
            player_detections.append(players)
            racket_detections.append(rackets)
            ball_detections.append(balls)
            if total:
                done    = i + 1
                pct     = done * 100 // total
                elapsed = time.time() - t0
                eta     = elapsed / done * (total - done)
                print(f"[  detect] {done:>{frame_num_width}}/{total}  {pct:>3}%"
                      f"  elapsed {_fmt(elapsed):>7}  ETA {_fmt(eta):>7}", end='\r', flush=True)
            else:
                print(f"[  detect] {i+1} frames  elapsed {_fmt(time.time()-t0)}", end='\r', flush=True)

        n_frames = len(player_detections)
        print(f"[  detect] {n_frames}/{n_frames}  100%  elapsed {_fmt(time.time()-t0):>7}  ETA    0:00  done")
        return player_detections, racket_detections, ball_detections

    def _parse(self, results):
        names = results.names
        players, rackets, balls = [], [], []
        for box in results.boxes:
            cls_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            det = {'bbox': [x1, y1, x2, y2], 'conf': float(box.conf[0]), 'track_id': None}
            if cls_name == "person":
                players.append(det)
            elif cls_name == "tennis racket":
                rackets.append(det)
            elif cls_name == "sports ball":
                balls.append(det)
        players = _nms(players, _NMS_IOU_PERSON)
        rackets = _nms(rackets, _NMS_IOU_RACKET)
        balls   = _nms(balls,   _NMS_IOU_BALL)
        return players, rackets, balls
