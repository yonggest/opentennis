from ultralytics import YOLO
import pickle
import time
import cv2


class YOLODetector:
    """一次推断提取 person、tennis racket、sports ball 三类目标。"""

    def __init__(self, model_paths, conf=0.1, imgsz=960, device='cpu'):
        # model_paths: {imgsz: path}，主推断用 imgsz 对应的模型
        self.model_paths = model_paths
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._models = {}   # imgsz → YOLO 实例（懒加载）

    def _get_model(self, imgsz):
        if imgsz not in self._models:
            path = self.model_paths[imgsz]
            print(f"[  detect] loading model imgsz={imgsz}: {path}", flush=True)
            self._models[imgsz] = YOLO(path, task='detect')
        return self._models[imgsz]

    def _select_imgsz(self, crop_max):
        """选择能容纳 crop_max 的最小模型尺寸，无合适尺寸返回 None。"""
        return next((s for s in sorted(self.model_paths) if s >= crop_max), None)

    @property
    def model(self):
        return self._get_model(self.imgsz)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'balls' in data:
                print(f"[  detect] loaded from cache: {stub_path}")
                return data['players'], data['rackets'], data['balls']
            print("[  detect] old cache format, re-running inference")

        player_detections, racket_detections, ball_detections = [], [], []
        total = len(frames)
        w = len(str(total))
        t0 = time.time()
        for i, frame in enumerate(frames):
            results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                         classes=[0, 38, 32], device=self.device, verbose=False)[0]
            p, r, b = self._parse_results(results)
            player_detections.append(p)
            racket_detections.append(r)
            ball_detections.append(b)
            pct = (i + 1) * 100 // total
            print(f"[  detect] {i+1:>{w}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)

        print(f"[  detect] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump({'players': player_detections, 'rackets': racket_detections, 'balls': ball_detections}, f)
            print(f"[  detect] saved to {stub_path}")

        return player_detections, racket_detections, ball_detections

    def filter_detections(self, player_detections, racket_detections, ball_detections,
                          valid_hull):
        """
        基于 3D 三棱柱投影凸包（valid_hull）过滤原始检测结果：
        - 球员：足部位置在 valid_hull 内
        - 球拍：与已过滤球员 bbox 有重叠
        - 网球：中心在 valid_hull 内
        """
        def in_hull(cx, cy):
            return cv2.pointPolygonTest(valid_hull, (float(cx), float(cy)), False) >= 0

        out_players, out_rackets, out_balls = [], [], []
        for player_dict, racket_dict, ball_dict in zip(
                player_detections, racket_detections, ball_detections):

            # 球员：足部在 valid_hull 内
            fp, idx = {}, 1
            for det in player_dict.values():
                fx = (det['bbox'][0] + det['bbox'][2]) / 2
                fy = det['bbox'][3]
                if in_hull(fx, fy):
                    fp[idx] = det
                    idx += 1
            out_players.append(fp)

            # 球拍：与已过滤球员 bbox 有重叠
            fr, idx = {}, 1
            for det in racket_dict.values():
                rb = det['bbox']
                overlaps = any(self._bboxes_overlap(rb, p['bbox']) for p in fp.values())
                if overlaps:
                    fr[idx] = det
                    idx += 1
            out_rackets.append(fr)

            # 网球：中心在 valid_hull 内
            fb, idx = {}, 1
            for det in ball_dict.values():
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                if in_hull(cx, cy):
                    fb[idx] = det
                    idx += 1
            out_balls.append(fb)

        return out_players, out_rackets, out_balls

    def augment_rackets(self, video_frames, player_detections, racket_detections, conf=None,
                        read_from_stub=False, stub_path=None):
        """对每帧每个球员，若无重叠球拍则裁剪球员区域用合适尺寸的模型搜索。"""
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                print(f"[  rpatch] loaded from cache: {stub_path}")
                return pickle.load(f)
        total = len(video_frames)
        w = len(str(total))
        t0 = time.time()
        conf = conf if conf is not None else self.conf
        result = []
        for i, (frame, player_dict, racket_dict) in enumerate(
                zip(video_frames, player_detections, racket_detections)):
            augmented = dict(racket_dict)
            next_idx = max(augmented.keys(), default=0) + 1
            fh, fw = frame.shape[:2]
            for pdet in player_dict.values():
                has_overlap = any(
                    self._bboxes_overlap(pdet['bbox'], rdet['bbox'])
                    for rdet in augmented.values()
                )
                if has_overlap:
                    continue
                x1, y1, x2, y2 = [int(v) for v in pdet['bbox']]
                bw, bh = x2 - x1, y2 - y1
                # 左右上各扩 50%，不向下扩
                cx1 = max(0, x1 - bw // 2)
                cx2 = min(fw, x2 + bw // 2)
                cy1 = max(0, y1 - bh // 2)
                cy2 = y2
                crop_max = max(cy2 - cy1, cx2 - cx1)
                imgsz = self._select_imgsz(crop_max)
                if imgsz is None:
                    continue
                # 扩展到 imgsz×imgsz
                exp_w = max(0, imgsz - (cx2 - cx1))
                exp_h = max(0, imgsz - (cy2 - cy1))
                cx1 = max(0, cx1 - exp_w // 2)
                cx2 = min(fw, cx2 + (exp_w - exp_w // 2))
                cy1 = max(0, cy1 - exp_h)
                crop = frame[cy1:cy2, cx1:cx2]
                res = self._get_model(imgsz).predict(
                    crop, conf=conf, imgsz=imgsz,
                    classes=[38], device=self.device, verbose=False)[0]
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy.tolist()[0]
                    bbox = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
                    augmented[next_idx] = {'bbox': bbox, 'conf': float(box.conf[0]), 'from_patch': True}
                    next_idx += 1
            result.append(augmented)
            pct = (i + 1) * 100 // total
            print(f"[  rpatch] {i+1:>{w}}/{total} frames  ({pct:>3}%)\r", end='', flush=True)
        orig_total  = sum(len(d) for d in racket_detections)
        patch_total = sum(sum(1 for d in f.values() if d.get('from_patch')) for f in result)
        aug_total   = orig_total + patch_total
        print(f"[  rpatch] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s  "
              f"orig={orig_total:>4}  patch={patch_total:>4}  total={aug_total:>4}")
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"[  rpatch] saved to {stub_path}")
        return result

    @staticmethod
    def _bboxes_overlap(b1, b2):
        return b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]

    def augment_balls(self, video_frames, ball_detections, max_dist=100, conf=None):
        """
        前向追踪网球：对每帧每个网球，检查下一帧附近是否有网球；
        没有则在下一帧对应位置裁剪并用合适尺寸的模型搜索。
        新找到的球继续参与后续帧的追踪。
        """
        conf = conf if conf is not None else self.conf
        result = [dict(d) for d in ball_detections]
        total = len(video_frames)
        w = len(str(total))
        t0 = time.time()

        for i in range(total - 1):
            for det in list(result[i].values()):
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2

                # 下一帧附近是否已有网球
                nearby = any(
                    ((d['bbox'][0] + d['bbox'][2]) / 2 - cx) ** 2 +
                    ((d['bbox'][1] + d['bbox'][3]) / 2 - cy) ** 2 <= max_dist ** 2
                    for d in result[i + 1].values()
                )
                if nearby:
                    continue

                # 在下一帧以当前球位置为中心裁剪并检测
                fh, fw = video_frames[i + 1].shape[:2]
                x1 = max(0, int(cx - max_dist))
                y1 = max(0, int(cy - max_dist))
                x2 = min(fw, int(cx + max_dist))
                y2 = min(fh, int(cy + max_dist))
                crop_max = max(y2 - y1, x2 - x1)
                imgsz = self._select_imgsz(crop_max)
                if crop_max == 0 or imgsz is None:
                    continue
                # 扩展到 imgsz×imgsz，充分利用推断算力
                exp_w = max(0, imgsz - (x2 - x1))
                exp_h = max(0, imgsz - (y2 - y1))
                x1 = max(0, x1 - exp_w // 2)
                x2 = min(fw, x2 + (exp_w - exp_w // 2))
                y1 = max(0, y1 - exp_h // 2)
                y2 = min(fh, y2 + (exp_h - exp_h // 2))
                crop = video_frames[i + 1][y1:y2, x1:x2]

                res = self._get_model(imgsz).predict(
                    crop, conf=conf, imgsz=imgsz,
                    classes=[32], device=self.device, verbose=False)[0]

                next_idx = max(result[i + 1].keys(), default=0) + 1
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy.tolist()[0]
                    bbox = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                    result[i + 1][next_idx] = {'bbox': bbox, 'conf': float(box.conf[0]), 'from_patch': True}
                    next_idx += 1

            pct = (i + 1) * 100 // total
            print(f"[  bpatch] {i+1:>{w}}/{total} frames  ({pct:>3}%)\r", end='', flush=True)
        orig_total  = sum(len(d) for d in ball_detections)
        patch_total = sum(sum(1 for d in f.values() if d.get('from_patch')) for f in result)
        aug_total   = orig_total + patch_total
        print(f"[  bpatch] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s  "
              f"orig={orig_total:>4}  patch={patch_total:>4}  total={aug_total:>4}")
        return result

    def augment_balls_backward(self, video_frames, ball_detections, max_dist=100, conf=None):
        """
        后向追踪网球：对每帧每个网球，检查上一帧附近是否有网球；
        没有则在上一帧对应位置裁剪并用合适尺寸的模型搜索。
        新找到的球继续参与前序帧的追踪。
        """
        conf = conf if conf is not None else self.conf
        result = [dict(d) for d in ball_detections]
        total = len(video_frames)
        w = len(str(total))
        t0 = time.time()

        for i in range(total - 1, 0, -1):
            for det in list(result[i].values()):
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2

                # 上一帧附近是否已有网球
                nearby = any(
                    ((d['bbox'][0] + d['bbox'][2]) / 2 - cx) ** 2 +
                    ((d['bbox'][1] + d['bbox'][3]) / 2 - cy) ** 2 <= max_dist ** 2
                    for d in result[i - 1].values()
                )
                if nearby:
                    continue

                # 在上一帧以当前球位置为中心裁剪并检测
                fh, fw = video_frames[i - 1].shape[:2]
                x1 = max(0, int(cx - max_dist))
                y1 = max(0, int(cy - max_dist))
                x2 = min(fw, int(cx + max_dist))
                y2 = min(fh, int(cy + max_dist))
                crop_max = max(y2 - y1, x2 - x1)
                imgsz = self._select_imgsz(crop_max)
                if crop_max == 0 or imgsz is None:
                    continue
                exp_w = max(0, imgsz - (x2 - x1))
                exp_h = max(0, imgsz - (y2 - y1))
                x1 = max(0, x1 - exp_w // 2)
                x2 = min(fw, x2 + (exp_w - exp_w // 2))
                y1 = max(0, y1 - exp_h // 2)
                y2 = min(fh, y2 + (exp_h - exp_h // 2))
                crop = video_frames[i - 1][y1:y2, x1:x2]

                res = self._get_model(imgsz).predict(
                    crop, conf=conf, imgsz=imgsz,
                    classes=[32], device=self.device, verbose=False)[0]

                next_idx = max(result[i - 1].keys(), default=0) + 1
                for box in res.boxes:
                    bx1, by1, bx2, by2 = box.xyxy.tolist()[0]
                    bbox = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                    result[i - 1][next_idx] = {'bbox': bbox, 'conf': float(box.conf[0]), 'from_patch': True}
                    next_idx += 1

            pct = (total - i) * 100 // total
            print(f"[ bpatch2] {total-i:>{w}}/{total} frames  ({pct:>3}%)\r", end='', flush=True)

        orig_total  = sum(len(d) for d in ball_detections)
        patch_total = sum(sum(1 for d in f.values() if d.get('from_patch')) for f in result)
        aug_total   = orig_total + patch_total
        print(f"[ bpatch2] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s  "
              f"orig={orig_total:>4}  patch={patch_total:>4}  total={aug_total:>4}")
        return result

    def _parse_results(self, results):
        id_name_dict = results.names
        player_dict = {}
        racket_dict = {}
        ball_dict = {}
        player_idx = racket_idx = ball_idx = 1
        for box in results.boxes:
            cls_name = id_name_dict[int(box.cls[0])]
            bbox = box.xyxy.tolist()[0]
            conf = float(box.conf[0])
            det = {'bbox': bbox, 'conf': conf}
            if cls_name == "person":
                player_dict[player_idx] = det
                player_idx += 1
            elif cls_name == "tennis racket":
                racket_dict[racket_idx] = det
                racket_idx += 1
            elif cls_name == "sports ball":
                ball_dict[ball_idx] = det
                ball_idx += 1
        return player_dict, racket_dict, ball_dict
