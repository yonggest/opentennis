import cv2
import pandas as pd
import numpy as np
from utils import get_text_params


class BallTracker:
    def __init__(self):
        pass

    def filter_by_court(self, ball_detections, court_keypoints, margin=150):
        corners = [(court_keypoints[i*2], court_keypoints[i*2+1]) for i in [0, 1, 3, 2]]
        poly = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)

        filtered = []
        for frame_dets in ball_detections:
            kept = {}
            for k, det in frame_dets.items():
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                if cv2.pointPolygonTest(poly, (float(cx), float(cy)), True) >= -margin:
                    kept[k] = det
            filtered.append(kept)
        return filtered

    def select_best_ball(self, ball_detections):
        result = []
        for frame_dets in ball_detections:
            if not frame_dets:
                result.append({})
                continue
            best = max(frame_dets.values(), key=lambda d: d['conf'])
            result.append({1: best})
        return result

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        return [{1: {'bbox': x, 'conf': None}} for x in df.to_numpy().tolist()]

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df) - int(minimum_change_frames_for_hit * 1.2)):
            negative_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
            positive_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0
            if negative_change or positive_change:
                change_count = 0
                for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    neg_following = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[change_frame] < 0
                    pos_following = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[change_frame] > 0
                    if negative_change and neg_following:
                        change_count += 1
                    elif positive_change and pos_following:
                        change_count += 1
                if change_count > minimum_change_frames_for_hit - 1:
                    df.loc[i, 'ball_hit'] = 1
        return df[df['ball_hit'] == 1].index.tolist()

    def filter_static_detections(self, ball_detections, static_thresh=5, min_static_frames=10):
        """
        排除连续 min_static_frames 帧内位置几乎不变（移动 < static_thresh px）的检测框。
        返回与输入格式相同的 ball_detections。
        """
        n = len(ball_detections)
        # 收集每个检测 key 在各帧的位置
        key_positions = {}  # key -> list of (frame_idx, cx, cy)
        for fi, frame_dets in enumerate(ball_detections):
            for k, det in frame_dets.items():
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                key_positions.setdefault(k, []).append((fi, cx, cy))

        # 找出静止的 (frame_idx, key) 对
        static_pairs = set()
        for k, appearances in key_positions.items():
            if len(appearances) < min_static_frames:
                continue
            for i in range(len(appearances)):
                run = [appearances[i]]
                for j in range(i + 1, len(appearances)):
                    if appearances[j][0] - appearances[j-1][0] > 1:
                        break
                    dx = appearances[j][1] - run[0][1]
                    dy = appearances[j][2] - run[0][2]
                    if (dx*dx + dy*dy) ** 0.5 < static_thresh:
                        run.append(appearances[j])
                    else:
                        break
                if len(run) >= min_static_frames:
                    for fi, _, _ in run:
                        static_pairs.add((fi, k))

        result = []
        for fi, frame_dets in enumerate(ball_detections):
            result.append({k: v for k, v in frame_dets.items() if (fi, k) not in static_pairs})
        return result

    def find_ball_by_longest_trajectory(self, ball_detections, max_dist=60, max_gap=30):
        """
        贪心帧间连接，按总移动距离选出最长轨迹，返回只保留该轨迹的 ball_detections。
        """
        # 每帧取置信度最高的检测
        candidates = []
        for fi, frame_dets in enumerate(ball_detections):
            if frame_dets:
                best = max(frame_dets.values(), key=lambda d: d['conf'])
                cx = (best['bbox'][0] + best['bbox'][2]) / 2
                cy = (best['bbox'][1] + best['bbox'][3]) / 2
                candidates.append((fi, cx, cy, best))

        if not candidates:
            return ball_detections

        # 贪心构建 tracklets
        tracklets = []
        used = [False] * len(candidates)
        for i, (fi, cx, cy, det) in enumerate(candidates):
            if used[i]:
                continue
            track = [(fi, cx, cy, det)]
            used[i] = True
            last_fi, last_cx, last_cy = fi, cx, cy
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                fj, cxj, cyj, detj = candidates[j]
                if fj - last_fi > max_gap:
                    break
                dist = ((cxj - last_cx)**2 + (cyj - last_cy)**2) ** 0.5
                if dist <= max_dist * (fj - last_fi):
                    track.append((fj, cxj, cyj, detj))
                    used[j] = True
                    last_fi, last_cx, last_cy = fj, cxj, cyj
            tracklets.append(track)

        if not tracklets:
            return [{} for _ in ball_detections]

        def total_distance(track):
            d = 0
            for k in range(1, len(track)):
                dx = track[k][1] - track[k-1][1]
                dy = track[k][2] - track[k-1][2]
                d += (dx*dx + dy*dy) ** 0.5
            return d

        best_track = max(tracklets, key=total_distance)
        frame_to_det = {fi: det for fi, _, _, det in best_track}

        return [{1: frame_to_det[fi]} if fi in frame_to_det else {} for fi in range(len(ball_detections))]

    def _build_tracklets(self, ball_detections, max_dist=100, max_gap=10):
        """Phase 1+2: 每帧取最高置信度检测，贪心构建 tracklet 列表。"""
        candidates = []
        for fi, frame_dets in enumerate(ball_detections):
            if not frame_dets:
                continue
            best = max(frame_dets.values(), key=lambda d: d['conf'] if d['conf'] is not None else 0)
            cx = (best['bbox'][0] + best['bbox'][2]) / 2
            cy = (best['bbox'][1] + best['bbox'][3]) / 2
            candidates.append((fi, cx, cy, best))

        tracklets = []
        used = [False] * len(candidates)
        for i, (fi, cx, cy, det) in enumerate(candidates):
            if used[i]:
                continue
            track = [(fi, cx, cy, det)]
            used[i] = True
            last_fi, last_cx, last_cy = fi, cx, cy
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                fj, cxj, cyj, detj = candidates[j]
                gap = fj - last_fi
                if gap > max_gap:
                    break
                dist = ((cxj - last_cx)**2 + (cyj - last_cy)**2) ** 0.5
                if dist <= max_dist * gap:
                    track.append((fj, cxj, cyj, detj))
                    used[j] = True
                    last_fi, last_cx, last_cy = fj, cxj, cyj
            tracklets.append(track)
        return tracklets

    def draw_tracklets(self, video_frames, ball_detections, max_dist=100, max_gap=10):
        """
        在每帧上用不同颜色绘制各 tracklet：
        - 检测框 + tracklet 编号
        - 截止当前帧已发生的所有箭头（历史轨迹保留）
        """
        palette = [
            (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
            (255, 255,   0), (  0, 255, 255), (255,   0, 255),
            (255, 128,   0), (128,   0, 255), (  0, 255, 128),
            (255,  64, 128), ( 64, 255,  64), (128, 128, 255),
        ]
        tracklets = self._build_tracklets(ball_detections, max_dist, max_gap)
        fs, ft = get_text_params(video_frames[0].shape[0])

        # 预处理：每个 tracklet 按帧号排好的点列表
        track_pts = [[(fi, cx, cy) for fi, cx, cy, _ in track] for track in tracklets]
        # 每个 tracklet 在各帧的检测框
        track_dets = [{fi: det for fi, _, _, det in track} for track in tracklets]

        for fi, frame in enumerate(video_frames):
            for idx, pts in enumerate(track_pts):
                color = palette[idx % len(palette)]
                # 截止当前帧的点
                visible = [(cx, cy) for fj, cx, cy in pts if fj <= fi]
                # 检测框 + 编号
                if fi in track_dets[idx]:
                    det = track_dets[idx][fi]
                    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, ft)
                    cv2.putText(frame, f'T{idx}', (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)
                # 历史箭头
                for k in range(1, len(visible)):
                    cv2.arrowedLine(frame,
                                    (int(visible[k-1][0]), int(visible[k-1][1])),
                                    (int(visible[k][0]),   int(visible[k][1])),
                                    color, ft, tipLength=0.3)

    def find_rally(self, ball_detections, video_fps, max_dist=100, max_gap=10,
                   smooth_window=7, min_reversal_frames=8, min_reversals=1, min_duration_s=1.0):
        """
        从网球检测结果中找出最可能的 rally。
        评分标准：y 方向反转次数（击球次数），需满足最短持续时间。
        """
        tracklets = self._build_tracklets(ball_detections, max_dist, max_gap)
        if not tracklets:
            return ball_detections

        # Phase 3: 评分（y 方向反转次数）
        def count_reversals(track):
            fis = [fi for fi, _, _, _ in track]
            if fis[-1] - fis[0] < video_fps * min_duration_s:
                return 0
            ys = np.array([cy for _, _, cy, _ in track], dtype=float)
            w = min(smooth_window, len(ys))
            if w < 3:
                return 0
            ys_s = np.convolve(ys, np.ones(w) / w, mode='valid')
            dy = np.diff(ys_s)
            reversals = 0
            cur_dir = None
            run = 0
            for d in dy:
                s = 1 if d > 0 else (-1 if d < 0 else 0)
                if s == 0:
                    continue
                if s == cur_dir:
                    run += 1
                else:
                    if cur_dir is not None and run >= min_reversal_frames:
                        reversals += 1
                    cur_dir = s
                    run = 1
            if cur_dir is not None and run >= min_reversal_frames:
                reversals += 1
            return reversals

        # Phase 4: 选出最佳 tracklet
        scores = [count_reversals(t) for t in tracklets]
        for idx, (t, s) in enumerate(zip(tracklets, scores)):
            print(f"[    ball] tracklet {idx:>3}  frames {t[0][0]:>4}–{t[-1][0]:>4}  len={len(t):>4}  reversals={s}")
        best_score = max(scores)
        if best_score < min_reversals:
            return ball_detections  # 无满足条件的 rally，返回原始结果

        best_track = tracklets[scores.index(best_score)]
        print(f"[    ball] rally: {len(best_track)} detections, {best_score} reversals, "
              f"frames {best_track[0][0]}–{best_track[-1][0]}")
        frame_to_det = {fi: det for fi, _, _, det in best_track}
        return [{1: frame_to_det[fi]} if fi in frame_to_det else {} for fi in range(len(ball_detections))]

    def draw_bboxes_frame(self, frame, ball_dict):
        fs, ft = get_text_params(frame.shape[0])
        for track_id, det in ball_dict.items():
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = (0, 0, 255) if det.get('from_patch') else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, ft)
#            conf = det['conf']
#            label = f'B{track_id} {conf:.2f}' if conf is not None else f'B{track_id}'
#            cv2.putText(frame, label, (x1, y1 - 6),
#                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)

