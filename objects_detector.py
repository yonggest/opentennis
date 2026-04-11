import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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

    def run(self, frames, valid_hull=None, frame_shape=None, total=None):
        """
        frames       : list 或任意可迭代对象（生成器）
        frame_shape  : (height, width)，frames 为生成器时必须提供
        total        : 总帧数，用于进度显示；frames 为生成器时必须提供
        """
        if isinstance(frames, list):
            fh, fw = frames[0].shape[:2]
            total  = total or len(frames)
        else:
            fh, fw = frame_shape
            total  = total or 0  # 未知时进度显示为帧计数

        if valid_hull is not None:
            bx, by, bw, bh = cv2.boundingRect(valid_hull)
            cx1 = max(0, bx)
            cy1 = max(0, by)
            cx2 = min(fw, bx + bw)
            cy2 = min(fh, by + bh)
            mask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.fillPoly(mask, [valid_hull], 255)
        else:
            cx1 = cy1 = 0
            cx2, cy2 = fw, fh
            mask = None

        player_detections, racket_detections, ball_detections = [], [], []
        nw = len(str(total)) if total else 6
        t0 = time.time()

        for i, frame in enumerate(frames):
            crop = cv2.bitwise_and(frame, frame, mask=mask)[cy1:cy2, cx1:cx2] \
                   if mask is not None else frame[cy1:cy2, cx1:cx2]
            results = self.model.predict(crop, conf=self.conf, imgsz=self.imgsz,
                                         classes=self.class_ids, device=self.device,
                                         verbose=False, save=False)[0]
            p, r, b = self._parse(results, offset=(cx1, cy1))
            player_detections.append(p)
            racket_detections.append(r)
            ball_detections.append(b)
            if total:
                pct = (i + 1) * 100 // total
                print(f"[  detect] {i+1:>{nw}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)
            else:
                print(f"[  detect] {i+1} frames", end='\r', flush=True)

        n = len(player_detections)
        print(f"[  detect] {n} frames  (100%)  done: {time.time()-t0:>6.1f}s")
        return player_detections, racket_detections, ball_detections

    def _parse(self, results, offset=(0, 0)):
        ox, oy = offset
        names = results.names
        players, rackets, balls = [], [], []
        for box in results.boxes:
            cls_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            track_id = int(box.id[0]) if box.id is not None else None
            det = {'bbox': [x1+ox, y1+oy, x2+ox, y2+oy], 'conf': float(box.conf[0]), 'track_id': track_id}
            if cls_name == "person":
                players.append(det)
            elif cls_name == "tennis racket":
                rackets.append(det)
            elif cls_name == "sports ball":
                balls.append(det)
        return players, rackets, balls
