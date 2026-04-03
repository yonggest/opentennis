import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from utils import measure_distance, get_center_of_bbox, get_text_params
import constants


class PlayerTracker:
    """球员后处理：从原始检测中筛选球员、分配稳定 ID、绘制。"""

    def __init__(self, n_players=4, max_match_dist=200):
        self.n_players = n_players
        self.max_match_dist = max_match_dist

    def select_and_track_players(self, valid_hull, player_detections, video_frames=None):
        """
        第一帧从凸包内的球员中选出 n_players 名（按距凸包重心最近排序），分配稳定 ID 1..n。
        后续帧用匈牙利算法（Kalman 预测位置）维持稳定 ID。
        """
        import cv2 as _cv2
        # ── 第一帧初始化 ──
        first_dets = player_detections[0]
        first_frame = video_frames[0] if video_frames is not None else None

        # 凸包重心作为参考点
        hull_center = valid_hull.reshape(-1, 2).mean(axis=0)

        distances = []
        for pid, det in first_dets.items():
            center = get_center_of_bbox(det['bbox'])
            dist = measure_distance(center, (hull_center[0], hull_center[1]))
            distances.append((pid, dist))
        distances.sort(key=lambda x: x[1])

        n = min(self.n_players, len(distances))
        stable_tracks = {}
        for stable_id, (pid, _) in enumerate(distances[:n], 1):
            det = first_dets[pid]
            stable_tracks[stable_id] = {
                'det': det,
                'kf': self._init_kalman(det['bbox']),
            }

        result = [{sid: t['det'] for sid, t in stable_tracks.items()}]

        # ── 逐帧匹配 ──
        total = len(player_detections)
        t0 = time.time()
        for frame_idx, frame_dets in enumerate(player_detections[1:], 1):
            w = len(str(total))
            pct = frame_idx * 100 // total
            print(f"[  player] {frame_idx:>{w}}/{total} frames  ({pct:>3}%)", end='\r', flush=True)

            frame = video_frames[frame_idx] if video_frames is not None else None

            track_ids = list(stable_tracks.keys())
            predicted_centers = []
            for sid in track_ids:
                pred = stable_tracks[sid]['kf'].predict()
                predicted_centers.append((float(pred[0][0]), float(pred[1][0])))

            dets = list(frame_dets.values())
            if not dets:
                result.append({sid: t['det'] for sid, t in stable_tracks.items()})
                continue

            det_centers = [get_center_of_bbox(d['bbox']) for d in dets]

            cost = np.array([
                [measure_distance(predicted_centers[i], det_centers[j]) for j in range(len(dets))]
                for i in range(len(track_ids))
            ])

            row_ind, col_ind = linear_sum_assignment(cost)

            frame_result = {}
            matched = set()
            for r, c in zip(row_ind, col_ind):
                sid = track_ids[r]
                if cost[r, c] < self.max_match_dist:
                    new_det = dets[c]
                    cx, cy = get_center_of_bbox(new_det['bbox'])
                    stable_tracks[sid]['kf'].correct(np.array([[cx], [cy]], dtype=np.float32))
                    stable_tracks[sid]['det'] = new_det
                    frame_result[sid] = new_det
                    matched.add(sid)

            for sid in track_ids:
                if sid not in matched:
                    frame_result[sid] = stable_tracks[sid]['det']

            result.append(frame_result)

        w = len(str(total))
        print(f"[  player] {total:>{w}}/{total} frames  (100%)  done: {time.time()-t0:>6.1f}s")
        return result


    def draw_bboxes_frame(self, frame, player_dict):
        fs, ft = get_text_params(frame.shape[0])
        for track_id, det in player_dict.items():
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = constants.PLAYER_COLORS.get(track_id, (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, ft)
            cv2.putText(frame, f'P{track_id} {det["conf"]:.2f}', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)

    # ── 内部工具 ────────────────────────────────────────────────────

    def _init_kalman(self, bbox):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost      = np.eye(4, dtype=np.float32)
        cx, cy = get_center_of_bbox(bbox)
        kf.statePost = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
        return kf

    def _compute_hist(self, frame, bbox):
        if frame is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _hist_dist(self, h1, h2):
        if h1 is None or h2 is None:
            return 0.5
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    def _court_polygon(self, court_keypoints):
        corners = [(court_keypoints[i*2], court_keypoints[i*2+1]) for i in [0, 1, 3, 2]]
        return np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
