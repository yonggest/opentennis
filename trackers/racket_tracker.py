import cv2
from utils import get_center_of_bbox, measure_distance, get_text_params
import constants


class RacketTracker:
    """将球拍检测结果按最近距离分配给球员，并负责绘图。"""

    def assign_rackets_to_players(self, player_detections, racket_detections):
        """
        以球拍为中心：每个球拍分配给距离最近的球员。
        若多个球拍最近球员相同，只保留距离最近的那个。

        返回 [{player_id: det_or_None}, ...]
        """
        assigned = []
        for player_dict, racket_dict in zip(player_detections, racket_detections):
            frame_assignment = {pid: None for pid in player_dict}

            if not racket_dict or not player_dict:
                assigned.append(frame_assignment)
                continue

            player_ids = list(player_dict.keys())
            player_centers = {pid: get_center_of_bbox(player_dict[pid]['bbox']) for pid in player_ids}

            racket_candidates = {}  # player_id → (dist, det)
            for det in racket_dict.values():
                rb = det['bbox']
                # 无重叠球员的球拍丢弃
                overlapping = [pid for pid in player_ids
                               if self._bboxes_overlap(rb, player_dict[pid]['bbox'])]
                if not overlapping:
                    continue
                rc = get_center_of_bbox(rb)
                nearest_pid = min(overlapping,
                                  key=lambda pid: measure_distance(rc, player_centers[pid]))
                dist = measure_distance(rc, player_centers[nearest_pid])
                if nearest_pid not in racket_candidates or dist < racket_candidates[nearest_pid][0]:
                    racket_candidates[nearest_pid] = (dist, det)

            for pid, (_, det) in racket_candidates.items():
                frame_assignment[pid] = det

            assigned.append(frame_assignment)
        return assigned

    @staticmethod
    def _bboxes_overlap(b1, b2):
        return b1[0] < b2[2] and b2[0] < b1[2] and b1[1] < b2[3] and b2[1] < b1[3]

    def draw_bboxes_frame(self, frame, assignment):
        fs, ft = get_text_params(frame.shape[0])
        for player_id, det in assignment.items():
            if det is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = (0, 0, 255) if det.get('from_patch') else constants.PLAYER_COLORS.get(player_id, (255, 165, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, ft)
            cv2.putText(frame, f'R{player_id} {det["conf"]:.2f}', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)


