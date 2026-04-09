import numpy as np

# ── 物理常量 ────────────────────────────────────────────────────────────
_BALL_D_M     = 0.067   # ITF 标准网球直径（m），范围 6.54~6.86 cm
_MAX_SPEED_MS = 70.0    # 职业发球上限 ~250 km/h，换算为 m/s

# ── from_video 推算系数 ──────────────────────────────────────────────────
# 搜索半径 = 每帧最大位移 × 此系数（留安全裕量）
_RADIUS_MARGIN   = 1.3
# bbox 面积下/上限 = (球径_px × 系数)²
# 下限宽松：YOLO 对远端小球 bbox 偏小
# 上限宽松：覆盖近景大框，同时排除人头等超大目标
_BBOX_MIN_FACTOR = 0.5
_BBOX_MAX_FACTOR = 15.0
# max_gap / min_detections 对应的时长（秒）
_GAP_SECONDS     = 0.12   # 运动模糊通常 < 0.10 s，留 0.02 s 裕量
_MIN_DET_SECONDS = 0.08   # 真实球可见段 > 0.1 s；误检通常仅 1~2 帧

# ── _link 内部系数 ───────────────────────────────────────────────────────
_VEL_ALPHA        = 0.5   # 速度指数滑动平均系数（0=不更新，1=瞬时速度）
_DYN_RADIUS_SCALE = 1.5   # 动态半径 = 当前速度 × dt × 此系数


# ── bbox 工具函数 ────────────────────────────────────────────────────────

def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def _aspect(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return min(w, h) / max(w, h) if max(w, h) > 0 else 0


# ── 主类 ─────────────────────────────────────────────────────────────────

class BallTracker:
    """
    基于 YOLO predict 输出的网球轨迹跟踪器。

    流程
    ----
    1. 逐帧候选预过滤（bbox 尺寸、长宽比）
    2. 贪心跨帧连接：速度预测位置 + 动态搜索半径
    3. 轨迹段验证：丢弃真实检测帧数不足的短段（误检）
    4. 段内 gap 线性插值，赋予连续 track_id

    推荐用 BallTracker.from_video(fps, px_per_meter) 构造，
    自动从物理量推算参数；也可手动传入各参数覆盖默认值。

    参数
    ----
    max_gap        : 段内允许的最大连续空白帧数
    min_detections : 有效段至少需要的真实检测帧数
    base_radius    : 基础搜索半径（px），速度快时自动放大
    min_area       : 候选球 bbox 最小面积（px²）
    max_area       : 候选球 bbox 最大面积（px²）
    min_aspect     : 候选球 bbox 最小长宽比（0~1）
    """

    def __init__(self, max_gap=5, min_detections=4,
                 base_radius=80, min_area=20, max_area=8000, min_aspect=0.4):
        self.max_gap        = max_gap
        self.min_detections = min_detections
        self.base_radius    = base_radius
        self.min_area       = min_area
        self.max_area       = max_area
        self.min_aspect     = min_aspect

    @classmethod
    def from_video(cls, fps: float, px_per_meter: float, min_aspect: float = 0.4):
        """
        根据帧率和像素/米比例推算各参数。

        ball_d_px  = _BALL_D_M × px_per_meter
        base_radius = (最大球速 ÷ fps × px_per_meter) × _RADIUS_MARGIN
        min_area   = (ball_d_px × _BBOX_MIN_FACTOR)²
        max_area   = (ball_d_px × _BBOX_MAX_FACTOR)²
        max_gap    = max(3, round(fps × _GAP_SECONDS))
        min_det    = max(3, round(fps × _MIN_DET_SECONDS))
        """
        ball_d_px   = _BALL_D_M * px_per_meter
        max_disp_px = _MAX_SPEED_MS / fps * px_per_meter

        base_radius    = max_disp_px * _RADIUS_MARGIN
        min_area       = max(10.0, (ball_d_px * _BBOX_MIN_FACTOR) ** 2)
        max_area       = (ball_d_px * _BBOX_MAX_FACTOR) ** 2
        max_gap        = max(3, round(fps * _GAP_SECONDS))
        min_detections = max(3, round(fps * _MIN_DET_SECONDS))

        print(f"[ tracker] fps={fps:.1f}  px/m={px_per_meter:.1f}  "
              f"ball_d={ball_d_px:.1f}px  radius={base_radius:.1f}px  "
              f"area=[{min_area:.0f},{max_area:.0f}]  "
              f"gap={max_gap}f  min_det={min_detections}f")

        return cls(max_gap=max_gap, min_detections=min_detections,
                   base_radius=base_radius, min_area=min_area,
                   max_area=max_area, min_aspect=min_aspect)

    # ── 公开接口 ──────────────────────────────────────────────────────────

    def run(self, ball_detections):
        """
        输入：  ball_detections[i] = [{'bbox', 'conf', 'track_id'}, ...]
        输出：  同结构，每帧至多一个球（插值帧 conf=0.0），track_id 为段编号。
        """
        n = len(ball_detections)

        candidates    = self._prefilter(ball_detections)
        raw_segments  = self._link(candidates, n)
        valid_segments = [s for s in raw_segments if len(s) >= self.min_detections]

        seg_lens = [len(s) for s in raw_segments]
        print(f"[ tracker] segments raw={len(raw_segments)} valid={len(valid_segments)}  "
              f"len_range=[{min(seg_lens) if seg_lens else 0}, "
              f"{max(seg_lens) if seg_lens else 0}]")

        output = [[] for _ in range(n)]
        for seg_id, seg in enumerate(valid_segments):
            for frame_idx, det in self._fill_gaps(seg):
                output[frame_idx].append({
                    'bbox':     det['bbox'],
                    'conf':     det['conf'],
                    'track_id': seg_id,
                })
        return output

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _prefilter(self, ball_detections):
        """过滤掉尺寸或形状不符的候选框。"""
        return [
            [d for d in dets
             if self.min_area <= _area(d['bbox']) <= self.max_area
             and _aspect(d['bbox']) >= self.min_aspect]
            for dets in ball_detections
        ]

    def _link(self, candidates, n):
        """贪心逐帧连接，返回轨迹段列表，每段为 [(frame_idx, det), ...]。"""
        active = []   # 活跃轨迹，每条: {frames, vx, vy, gap}
        closed = []

        for i, cands in enumerate(candidates):
            matched = set()

            for track in active:
                last_fi, last_det = track['frames'][-1]
                lx, ly = _center(last_det['bbox'])
                dt = i - last_fi

                # 预测位置（匀速外推）
                pred_x = lx + track['vx'] * dt
                pred_y = ly + track['vy'] * dt

                # 动态搜索半径：速度越快或间隔越长，半径越大
                speed  = (track['vx'] ** 2 + track['vy'] ** 2) ** 0.5
                radius = max(self.base_radius, speed * dt * _DYN_RADIUS_SCALE)

                # 找距预测位置最近的未匹配候选
                best_j, best_dist = None, radius
                for j, c in enumerate(cands):
                    if j in matched:
                        continue
                    cx, cy = _center(c['bbox'])
                    dist = ((cx - pred_x) ** 2 + (cy - pred_y) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist, best_j = dist, j

                if best_j is not None:
                    det = cands[best_j]
                    cx, cy = _center(det['bbox'])
                    # 指数滑动平均更新速度（平衡历史与瞬时）
                    track['vx'] = _VEL_ALPHA * (cx - lx) / dt + (1 - _VEL_ALPHA) * track['vx']
                    track['vy'] = _VEL_ALPHA * (cy - ly) / dt + (1 - _VEL_ALPHA) * track['vy']
                    track['frames'].append((i, det))
                    track['gap'] = 0
                    matched.add(best_j)
                else:
                    track['gap'] += 1

            # 关闭超出 max_gap 的轨迹，保留其余
            still_active = []
            for track in active:
                if track['gap'] <= self.max_gap:
                    still_active.append(track)
                else:
                    closed.append(track['frames'])
            active = still_active

            # 未匹配的候选各自开启新轨迹
            for j, c in enumerate(cands):
                if j not in matched:
                    active.append({'frames': [(i, c)], 'vx': 0.0, 'vy': 0.0, 'gap': 0})

        # 关闭所有剩余活跃轨迹
        closed.extend(track['frames'] for track in active)
        return closed

    def _fill_gaps(self, seg):
        """段内 gap 线性插值，返回含插值帧的 [(frame_idx, det), ...] 列表。"""
        result = []
        for k in range(len(seg) - 1):
            fi, di = seg[k]
            fj, dj = seg[k + 1]
            result.append((fi, di))
            gap = fj - fi
            if gap > 1:
                cx_i, cy_i = _center(di['bbox'])
                cx_j, cy_j = _center(dj['bbox'])
                # 插值框的宽高取两端平均
                w = ((di['bbox'][2] - di['bbox'][0]) + (dj['bbox'][2] - dj['bbox'][0])) / 2
                h = ((di['bbox'][3] - di['bbox'][1]) + (dj['bbox'][3] - dj['bbox'][1])) / 2
                for t in range(1, gap):
                    alpha = t / gap
                    cx = cx_i + alpha * (cx_j - cx_i)
                    cy = cy_i + alpha * (cy_j - cy_i)
                    result.append((fi + t, {
                        'bbox': [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        'conf': 0.0,
                        'track_id': None,
                    }))
        result.append(seg[-1])
        return result
