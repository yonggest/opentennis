"""
网球追踪器：线性预测 + 匈牙利算法匹配。

追踪流程（每帧）：
  1. 每条轨迹用最近 _LINEAR_WINDOW 个检测点线性外推预测下一帧位置
     （2+ 点线性；1 点静止）
  2. 匈牙利算法将检测与轨迹最优匹配（欧氏距离，逐轨迹动态门限）
  3. 匹配成功 → 更新历史；累计命中 >= min_hits → TENTATIVE 升为 CONFIRMED
  4. 未匹配的轨迹 → 丢失计数 +1；超出 max_age → 删除
  5. 未匹配的高置信度检测 → 创建新 TENTATIVE 轨迹

类
----
  Tracker       — 多目标追踪器，逐帧调用 step(detections, frame_idx)
  BallTracker   — 封装 Tracker，加网球专用预过滤 + gap 线性插值
  PlayerTracker — 封装 Tracker，以脚点（bbox 底部中心）为追踪锚点
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ── 物理常量 ──────────────────────────────────────────────────────────────────

_BALL_D_M        = 0.067   # ITF 网球直径（m）
_MAX_SPEED_MS    = 70.0    # 职业发球上限（m/s），用于 max_dist 兜底截断
_RADIUS_MARGIN   = 1.3     # max_dist 安全裕量系数
_BBOX_MIN_FACTOR = 0.5     # min_area = (ball_d_px × 系数)²
_BBOX_MAX_FACTOR = 15.0    # max_area = (ball_d_px × 系数)²
_GAP_SECONDS      = 0.5    # max_age 对应时长（s）
_MIN_HIT_SECONDS  = 0.05   # min_hits 对应时长（s）
_SEARCH_DIAMETERS = 2.0    # 搜索半径 = N × 球径
_LINEAR_WINDOW    = 4      # 预测时只使用最近 N 个历史点估计速度方向
_HIST_BINS        = 16     # HSV H 通道直方图 bin 数
_HIST_MOMENTUM    = 0.8    # 直方图 EMA 系数：新值权重 = 1 - momentum


# ── 状态枚举 ─────────────────────────────────────────────────────────────────

class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1
    LOST      = 2   # 保留，供外部状态展示用


# ── bbox 工具 ────────────────────────────────────────────────────────────────

def _center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def _aspect(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

def _center_det(det):
    """检测框中心（anchor_fn 默认值，接受 det dict）。"""
    return _center(det['bbox'])

def _foot_center(det):
    """检测框底部中点（球员脚点）。"""
    x1, y1, x2, y2 = det['bbox']
    return (x1 + x2) / 2.0, float(y2)

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0.0 else 0.0


# ── 轨迹 ────────────────────────────────────────────────────────────────────

class _LinearTrack:
    """
    单条轨迹：用最近 _LINEAR_WINDOW 个历史检测点线性外推预测下一帧位置。

    history : [(frame_idx, cx, cy), ...]
    predict()  → 更新 _pred，age+1
    update()   → 追加历史，重置 age

    search_radius 根据最近一次检测的 bbox 均值尺寸动态计算（search_diameters × bbox_d_px），
    以自动适应近大远小的透视变化；search_diameters=None 时退化为固定门限（max_dist）。
    """
    _next_id = 0

    def __init__(self, det, frame_idx, min_hits, search_diameters, max_dist,
                 anchor_fn=None, use_prediction=True):
        anchor_fn = anchor_fn or _center_det
        ax, ay = anchor_fn(det)
        x1, y1, x2, y2 = det['bbox']
        self.id                 = _LinearTrack._next_id
        _LinearTrack._next_id += 1
        self.state              = TrackState.TENTATIVE
        self.hits               = 1
        self.age                = 0
        self._min_hits          = min_hits
        self._search_diameters  = search_diameters
        self._bbox_d_px         = (x2 - x1 + y2 - y1) / 2.0
        self._max_dist          = max_dist
        self._anchor_fn         = anchor_fn
        self._use_prediction    = use_prediction
        self.history            = [(frame_idx, ax, ay)]
        self.last_det           = det
        self._next_frame        = frame_idx + 1
        self._pred              = (ax, ay)
        self.hist               = det.get('hist')  # HSV H-channel histogram, or None

    @property
    def search_radius(self) -> float:
        """当前搜索半径：search_diameters × bbox 均值尺寸（px）。
        search_diameters=None 时直接返回 max_dist（固定门限模式，用于球员）。"""
        if self._search_diameters is None:
            return self._max_dist
        return self._search_diameters * self._bbox_d_px

    @property
    def effective_gate(self) -> float:
        """搜索门限：search_diameters=None（固定门限模式）或仅 1 个历史点时用 max_dist；
        否则用 search_radius。"""
        if self._search_diameters is None or (len(self.history) == 1 and self._max_dist is not None):
            return self._max_dist
        return self.search_radius

    def predict(self):
        """更新预测位置，age+1。
        use_prediction=True：线性外推；False：停在上一帧位置。"""
        if not self._use_prediction:
            self._pred = (self.history[-1][1], self.history[-1][2])
            self.age        += 1
            self._next_frame += 1
            return
        h  = self.history
        ts = np.array([p[0] for p in h], dtype=float)
        xs = np.array([p[1] for p in h], dtype=float)
        ys = np.array([p[2] for p in h], dtype=float)
        t0 = ts[-1]
        tn = ts - t0
        tp = self._next_frame - t0
        w  = _LINEAR_WINDOW
        deg = min(1, len(h) - 1)
        px = np.polyfit(tn[-w:], xs[-w:], deg)
        py = np.polyfit(tn[-w:], ys[-w:], deg)
        self._pred       = (float(np.polyval(px, tp)), float(np.polyval(py, tp)))
        self.age        += 1
        self._next_frame += 1

    def update(self, det, frame_idx):
        ax, ay = self._anchor_fn(det)
        x1, y1, x2, y2 = det['bbox']
        self.history.append((frame_idx, ax, ay))
        self.last_det    = det
        self._bbox_d_px  = (x2 - x1 + y2 - y1) / 2.0
        self.hits       += 1
        self.age         = 0
        self._next_frame = frame_idx + 1
        if self.state == TrackState.TENTATIVE and self.hits >= self._min_hits:
            self.state = TrackState.CONFIRMED
        # 直方图 EMA 更新（仅当检测携带直方图时）
        new_hist = det.get('hist')
        if new_hist is not None:
            if self.hist is None:
                self.hist = new_hist.copy()
            else:
                self.hist = _HIST_MOMENTUM * self.hist + (1.0 - _HIST_MOMENTUM) * new_hist
                self.hist /= self.hist.sum()  # 保持归一化

    @property
    def predicted_center(self):
        return self._pred


# ── 匹配 ─────────────────────────────────────────────────────────────────────

def _bbox_d(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1 + y2 - y1) / 2.0

def _extract_hist(frame, bbox):
    """从 bbox 上半部分（球衣区域）提取 HSV H 通道归一化直方图。"""
    x1, y1, x2, y2 = max(0, int(bbox[0])), max(0, int(bbox[1])), \
                     min(frame.shape[1], int(bbox[2])), min(frame.shape[0], int(bbox[3]))
    mid_y = (y1 + y2) // 2
    crop = frame[y1:mid_y, x1:x2]
    if crop.size == 0:
        return np.ones(_HIST_BINS, dtype=np.float32) / _HIST_BINS
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [_HIST_BINS], [0, 180]).flatten()
    hist += 1e-6
    hist /= hist.sum()
    return hist.astype(np.float32)

def _hist_dist(h1, h2):
    """直方图交叉距离：0 = 完全相同，1 = 完全不同。h1/h2 为 None 时返回 0.0。"""
    if h1 is None or h2 is None:
        return 0.0
    return float(1.0 - np.sum(np.minimum(h1, h2)))


def _match(tracks, dets, max_dist, min_iou=None, anchor_fn=None,
           size_gate=None, hist_weight=0.0, hist_gate=None):
    """
    匈牙利算法匹配轨迹与检测。

    max_dist != None → 欧氏锚点距离模式
                       可为标量或长度等于 tracks 的列表（逐轨迹门限）
    max_dist == None → IoU 模式
    size_gate        → bbox 尺寸比例上限，超出则拒绝
    hist_weight      → 颜色直方图代价权重：cost = dist × (1 + hist_weight × hist_dist)
    hist_gate        → 直方图交叉距离上限，超出则硬拒绝；None 关闭

    返回 (matched[(ti,di)], unmatched_tracks[ti], unmatched_dets[di])

    注意：距离门限在构建代价矩阵时即硬拒绝（cost 保持 1e9），而非仅在匈牙利分配后
    再过滤。若门限在分配后才检查，匈牙利全局最优化会把大门限轨迹分配给本属于小门限
    轨迹的检测（整体总代价更小），导致小门限轨迹被迫与远处噪点匹配或失配。
    例：ball tracker 的 track A 门限 30px、track B 门限 500px，当某检测距 A 18px、
    距 B 257px 时，匈牙利会把它分配给 B（腾出更近的检测给其他轨迹），A 反而失配。
    """
    if anchor_fn is None:
        anchor_fn = _center_det
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    n_t, n_d = len(tracks), len(dets)
    cost = np.full((n_t, n_d), 1e9)

    if max_dist is not None:
        gates = max_dist if isinstance(max_dist, list) else [max_dist] * n_t
        for i, t in enumerate(tracks):
            tcx, tcy = t.predicted_center
            track_d = _bbox_d(t.last_det['bbox'])
            for j, d in enumerate(dets):
                # 尺寸门限
                if size_gate is not None:
                    det_d = _bbox_d(d['bbox'])
                    ratio = max(track_d, det_d) / max(min(track_d, det_d), 1e-3)
                    if ratio > size_gate:
                        continue

                # 颜色直方图门限（硬拒绝）
                hd = _hist_dist(t.hist, d.get('hist'))
                if hist_gate is not None and t.hist is not None and d.get('hist') is not None:
                    if hd > hist_gate:
                        continue

                dcx, dcy = anchor_fn(d)
                dist = ((tcx - dcx)**2 + (tcy - dcy)**2) ** 0.5
                if dist > gates[i]:
                    continue        # 超出距离门限，硬拒绝，不参与全局分配
                # 颜色代价调制
                cost[i, j] = dist * (1.0 + hist_weight * hd)
        valid = lambda r, v: v <= gates[r]
    else:
        for i, t in enumerate(tracks):
            # 用上次检测框做 IoU（_LinearTrack 没有 predicted_bbox，退化到 last_det）
            pb = t.last_det['bbox']
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - _iou(pb, d['bbox'])
        thr = min_iou if min_iou is not None else 0.3
        valid = lambda r, v: (1.0 - v) >= thr

    row_ind, col_ind = linear_sum_assignment(cost)
    matched, mr, mc = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if valid(r, cost[r, c]):
            matched.append((r, c))
            mr.add(r); mc.add(c)

    return (matched,
            [i for i in range(n_t) if i not in mr],
            [j for j in range(n_d) if j not in mc])


# ── 追踪器 ────────────────────────────────────────────────────────────────────

class Tracker:
    """
    在线多目标追踪器（ByteTrack 风格两阶段匹配）。

    参数
    ----
    min_hits         : 累计命中帧数阈值，达到后 TENTATIVE → CONFIRMED
    max_age          : 连续丢失帧数上限，超出则删除轨迹
    conf_high        : 高置信度阈值；>= 此值的检测可新建轨迹
    conf_low         : 低置信度下限；[conf_low, conf_high) 的检测仅续接已有轨迹
    search_diameters : 搜索半径 = search_diameters × 当前检测球径（px）；
                       球径取最近一次检测 bbox 的均值宽高，自动适应透视缩放
    max_dist         : 欧氏距离硬截断（px），防止丢失帧时搜索范围失控；
                       None 则改用 IoU 匹配（用于球员 / 球拍）
    size_gate        : bbox 尺寸比例上限；超出则拒绝匹配（None 关闭）
    hist_weight      : 颜色直方图代价权重 cost = dist × (1 + hist_weight × hist_dist)
    hist_gate        : 直方图交叉距离上限；超出则硬拒绝（None 关闭）
    """

    def __init__(self, min_hits=3, max_age=5,
                 conf_high=0.5, conf_low=0.1,
                 search_diameters=_SEARCH_DIAMETERS, max_dist=None,
                 anchor_fn=None, size_gate=None,
                 hist_weight=0.0, hist_gate=None,
                 use_prediction=True):
        self.min_hits         = min_hits
        self.max_age          = max_age
        self.conf_high        = conf_high
        self.conf_low         = conf_low
        self.search_diameters = search_diameters
        self.max_dist         = max_dist
        self._anchor_fn       = anchor_fn or _center_det
        self._size_gate       = size_gate
        self._hist_weight     = hist_weight
        self._hist_gate       = hist_gate
        self._use_prediction  = use_prediction
        self._tracks: list[_LinearTrack] = []

    def reset(self):
        self._tracks = []
        _LinearTrack._next_id = 0

    def step(self, detections, frame_idx):
        """
        处理单帧检测（两阶段 ByteTrack 风格）。

        输入 : detections = [{'bbox':[x1,y1,x2,y2], 'conf':float, ...}, ...]
               frame_idx  = 当前帧号（int）
        输出 : 同结构，每条检测新增两个字段：
               track_id — CONFIRMED 轨迹的 ID(int)；未确认或未匹配时为 None
               _tid     — 内部字段，含 TENTATIVE 轨迹的 ID；BallTracker.run() 用于
                          回填历史、随后由 _clean() 从 JSON 输出中剥离
        """
        idx_high = [i for i, d in enumerate(detections)
                    if d.get('conf', 1.0) >= self.conf_high]
        idx_low  = [i for i, d in enumerate(detections)
                    if self.conf_low <= d.get('conf', 1.0) < self.conf_high]
        dets_high = [detections[i] for i in idx_high]
        dets_low  = [detections[i] for i in idx_low]

        # 1. 预测
        for t in self._tracks:
            t.predict()

        # 2. 阶段一：所有轨迹 vs 高置信度检测（逐轨迹门限）
        gates1 = [t.effective_gate for t in self._tracks]
        matched1, unmatched_t1, unmatched_d_high = _match(
            self._tracks, dets_high, gates1,
            anchor_fn=self._anchor_fn, size_gate=self._size_gate,
            hist_weight=self._hist_weight, hist_gate=self._hist_gate)
        for ti, di in matched1:
            self._tracks[ti].update(dets_high[di], frame_idx)

        # 3. 阶段二：阶段一未匹配的轨迹 vs 低置信度检测（逐轨迹门限）
        matched2 = []
        if unmatched_t1 and dets_low:
            tracks2 = [self._tracks[ti] for ti in unmatched_t1]
            gates2  = [t.effective_gate for t in tracks2]
            matched2_local, _, _ = _match(
                tracks2, dets_low, gates2,
                anchor_fn=self._anchor_fn, size_gate=self._size_gate,
                hist_weight=self._hist_weight, hist_gate=self._hist_gate)
            for i2, di2 in matched2_local:
                ti = unmatched_t1[i2]
                self._tracks[ti].update(dets_low[di2], frame_idx)
                matched2.append((ti, idx_low[di2]))

        # 4. 记录输出映射（CONFIRMED 轨迹才输出 track_id）
        orig_i_to_tid = {}
        all_i_to_tid  = {}
        for ti, di in matched1:
            all_i_to_tid[idx_high[di]] = self._tracks[ti].id
            if self._tracks[ti].state == TrackState.CONFIRMED:
                orig_i_to_tid[idx_high[di]] = self._tracks[ti].id
        for ti, orig_i in matched2:
            all_i_to_tid[orig_i] = self._tracks[ti].id
            if self._tracks[ti].state == TrackState.CONFIRMED:
                orig_i_to_tid[orig_i] = self._tracks[ti].id

        # 5. 删除超出 max_age 的轨迹
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        # 6. 未匹配的高置信度检测 → 新 TENTATIVE 轨迹（种子也记入 all_i_to_tid）
        for di in unmatched_d_high:
            new_track = _LinearTrack(
                dets_high[di], frame_idx,
                self.min_hits, self.search_diameters, self.max_dist,
                anchor_fn=self._anchor_fn, use_prediction=self._use_prediction)
            self._tracks.append(new_track)
            all_i_to_tid[idx_high[di]] = new_track.id

        return [dict(det, track_id=orig_i_to_tid.get(i), _tid=all_i_to_tid.get(i))
                for i, det in enumerate(detections)]


# ── 网球追踪器 ────────────────────────────────────────────────────────────────

class BallTracker:
    """
    网球追踪器：预过滤（尺寸 / 形状）+ Tracker + gap 线性插值。

    推荐用 BallTracker.from_video(fps, px_per_meter) 构造。
    搜索半径 = search_diameters × 轨迹最近一帧检测 bbox 的均值宽高，随透视自动缩放。
    max_dist 为单帧最大球速对应的像素位移：轨迹仅有 1 个历史点时（无方向信息）用此值作为搜索门限；
    有多个历史点时改用 search_radius。
    """

    def __init__(self, min_hits=3, max_age=5,
                 conf_high=0.5, conf_low=0.1,
                 search_diameters=_SEARCH_DIAMETERS, max_dist=None,
                 min_area=20.0, max_area=8000.0, min_aspect=0.3):
        self._tracker   = Tracker(min_hits=min_hits, max_age=max_age,
                                  conf_high=conf_high, conf_low=conf_low,
                                  search_diameters=search_diameters,
                                  max_dist=max_dist)
        self.min_area   = min_area
        self.max_area   = max_area
        self.min_aspect = min_aspect

    @classmethod
    def from_video(cls, fps: float, px_per_meter: float,
                   conf_high: float = 0.5, conf_low: float = 0.1,
                   min_aspect: float = 0.3,
                   search_diameters: float = _SEARCH_DIAMETERS):
        """
        根据帧率和像素/米比例推算各参数。

        search_radius 动态：每条轨迹用最近检测的 bbox 球径实时计算，不在此处固定。
        max_dist      = 单帧最大球速（物理兜底截断，hist=1 时使用）
        """
        ball_d_px   = _BALL_D_M * px_per_meter
        max_disp_px = _MAX_SPEED_MS / fps * px_per_meter

        # search_radius 不再是固定值，每条轨迹根据最新检测 bbox 动态计算。
        # 这里仍用标称 ball_d_px 打印参考值。
        max_dist = max_disp_px * _RADIUS_MARGIN
        min_area = max(10.0, (ball_d_px * _BBOX_MIN_FACTOR) ** 2)
        max_area = (ball_d_px * _BBOX_MAX_FACTOR) ** 2
        max_age  = max(3, round(fps * _GAP_SECONDS))
        min_hits = max(3, round(fps * _MIN_HIT_SECONDS))

        print(f"[ tracker] fps={fps:.1f}  px/m={px_per_meter:.1f}  "
              f"ball_d(ref)={ball_d_px:.1f}px  search_r(ref)={search_diameters*ball_d_px:.1f}px  "
              f"max_dist={max_dist:.0f}px  "
              f"area=[{min_area:.0f},{max_area:.0f}]  "
              f"max_age={max_age}f  min_hits={min_hits}f  "
              f"conf=[{conf_low},{conf_high})")

        return cls(min_hits=min_hits, max_age=max_age,
                   conf_high=conf_high, conf_low=conf_low,
                   search_diameters=search_diameters, max_dist=max_dist,
                   min_area=min_area, max_area=max_area, min_aspect=min_aspect)

    def run(self, ball_detections, debug_frame: int = -1):
        """
        输入：ball_detections[i] = [{'bbox', 'conf', 'track_id'}, ...]
        输出：同结构，CONFIRMED 轨迹含 track_id(int)，gap 帧线性插值（conf=0.0，interpolated=True）
        """
        n = len(ball_detections)
        self._tracker.reset()

        # ── 预过滤（尺寸 / 形状）────────────────────────────────────────────
        candidates   = []
        dropped_dets = []
        for fi, dets in enumerate(ball_detections):
            passed, dropped = [], []
            for d in dets:
                a, asp = _area(d['bbox']), _aspect(d['bbox'])
                if self.min_area <= a <= self.max_area and asp >= self.min_aspect:
                    passed.append(d)
                else:
                    dropped.append(d)
            candidates.append(passed)
            dropped_dets.append(dropped)
            if fi == debug_frame and (passed or dropped):
                print(f"[dbg f{fi}] prefilter: {len(passed)} passed, {len(dropped)} dropped"
                      f"  (area∈[{self.min_area:.0f},{self.max_area:.0f}]"
                      f"  asp>={self.min_aspect})")
                for d in dropped:
                    a2, asp2 = _area(d['bbox']), _aspect(d['bbox'])
                    print(f"           DROPPED  conf={d['conf']:.3f}  area={a2:.0f}  asp={asp2:.2f}")
                for d in passed:
                    print(f"           passed   conf={d['conf']:.3f}  area={_area(d['bbox']):.0f}"
                          f"  asp={_aspect(d['bbox']):.2f}")

        # ── 逐帧追踪 ─────────────────────────────────────────────────────────
        tracked = []
        for fi, frame_dets in enumerate(candidates):
            result = self._tracker.step(frame_dets, fi)
            tracked.append(result)
            if fi == debug_frame:
                tr = self._tracker
                idx_high = [i for i, d in enumerate(frame_dets)
                            if d.get('conf', 1.0) >= tr.conf_high]
                idx_low  = [i for i, d in enumerate(frame_dets)
                            if tr.conf_low <= d.get('conf', 1.0) < tr.conf_high]
                print(f"[dbg f{fi}] conf split: high(>={tr.conf_high})={len(idx_high)}"
                      f"  low([{tr.conf_low},{tr.conf_high}))={len(idx_low)}")
                states = {0: 'TENTATIVE', 1: 'CONFIRMED', 2: 'LOST'}
                for t in tr._tracks:
                    print(f"           track id={t.id}  {states[t.state]}"
                          f"  hits={t.hits}  age={t.age}"
                          f"  pred=({t.predicted_center[0]:.0f},{t.predicted_center[1]:.0f})"
                          f"  hist={len(t.history)}")
                for det in result:
                    print(f"           output: conf={det['conf']:.3f}"
                          f"  track_id={det.get('track_id')}"
                          f"  bbox={[round(v) for v in det['bbox']]}")

        # ── 收集各 track_id 的检测点（含 TENTATIVE 回填）────────────────────
        tid_frames: dict[int, list] = {}
        tentative_hist: dict[int, list] = {}
        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                tid  = det.get('track_id')
                _tid = det.get('_tid')
                if tid is not None:
                    tid_frames.setdefault(tid, []).append((fi, det))
                elif _tid is not None:
                    tentative_hist.setdefault(_tid, []).append((fi, det))

        for tid in list(tid_frames.keys()):
            if tid not in tentative_hist:
                continue
            confirmed_frames = {fi for fi, _ in tid_frames[tid]}
            prepend = [(fi, det) for fi, det in tentative_hist[tid]
                       if fi not in confirmed_frames]
            if prepend:
                tid_frames[tid] = sorted(prepend + tid_frames[tid], key=lambda x: x[0])

        # ── Gap 线性插值，写入 output ─────────────────────────────────────────
        def _clean(det, **overrides):
            d = {k: v for k, v in det.items() if k != '_tid'}
            d.update(overrides)
            return d

        output = [[] for _ in range(n)]
        for tid, frames in tid_frames.items():
            for k in range(len(frames) - 1):
                fi_a, det_a = frames[k]
                fi_b, det_b = frames[k + 1]
                output[fi_a].append(_clean(det_a, track_id=tid))
                gap = fi_b - fi_a
                if gap > 1:
                    cx_a, cy_a = _center(det_a['bbox'])
                    cx_b, cy_b = _center(det_b['bbox'])
                    w = ((det_a['bbox'][2] - det_a['bbox'][0]) +
                         (det_b['bbox'][2] - det_b['bbox'][0])) / 2
                    h = ((det_a['bbox'][3] - det_a['bbox'][1]) +
                         (det_b['bbox'][3] - det_b['bbox'][1])) / 2
                    for t in range(1, gap):
                        alpha = t / gap
                        cx = cx_a + alpha * (cx_b - cx_a)
                        cy = cy_a + alpha * (cy_b - cy_a)
                        output[fi_a + t].append({
                            'bbox': [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                            'conf': 0.0, 'track_id': tid, 'interpolated': True,
                        })
            fi_last, det_last = frames[-1]
            output[fi_last].append(_clean(det_last, track_id=tid))

        # ── 保留未被追踪的检测（供可视化）───────────────────────────────────
        backfilled: set[int] = set()
        for tid, frames in tid_frames.items():
            for fi, det in frames:
                if det.get('track_id') is None:
                    backfilled.add(id(det))

        for fi, dets in enumerate(dropped_dets):
            for det in dets:
                output[fi].append(_clean(det, track_id=None))
        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                if det.get('track_id') is None and id(det) not in backfilled:
                    output[fi].append(_clean(det, track_id=None))

        n_raw     = sum(len(f) for f in ball_detections)
        n_filtered = sum(len(f) for f in candidates)
        n_tracked  = sum(len(v) for v in tid_frames.values())
        print(f"[ tracker] raw={n_raw}  filtered={n_filtered}  "
              f"tracks={len(tid_frames)}  confirmed_dets={n_tracked}")

        return output


# ── 球员追踪器 ────────────────────────────────────────────────────────────────

_PLAYER_MAX_SPEED_MS   = 8.0   # 球员冲刺最大速度（m/s）
_PLAYER_RADIUS_MARGIN  = 1.5   # 搜索门限安全裕量
_PLAYER_GAP_SECONDS    = 0.5   # max_age 对应时长（s），用于遮挡续接
_PLAYER_MIN_HIT_SECONDS = 0.04 # min_hits 对应时长（s）


class PlayerTracker:
    """
    球员追踪器：以检测框底部中心（脚点）为追踪锚点，
    搜索门限基于球员最大移动速度（单帧像素位移）。

    推荐用 PlayerTracker.from_video(fps, px_per_meter) 构造。
    遮挡时轨迹最多保活 _PLAYER_GAP_SECONDS，重新出现后自动续接。
    """

    def __init__(self, min_hits=2, max_age=13,
                 conf_high=0.5, conf_low=0.1, max_dist=None,
                 size_gate=3.0, hist_weight=1.5, hist_gate=0.6):
        self._tracker = Tracker(
            min_hits=min_hits, max_age=max_age,
            conf_high=conf_high, conf_low=conf_low,
            search_diameters=None, max_dist=max_dist,
            anchor_fn=_foot_center, size_gate=size_gate,
            hist_weight=hist_weight, hist_gate=hist_gate,
            use_prediction=False,
        )

    @classmethod
    def from_video(cls, fps: float, px_per_meter: float,
                   conf_high: float = 0.5, conf_low: float = 0.1,
                   size_gate: float = 3.0,
                   hist_weight: float = 1.5, hist_gate: float = 0.6):
        """根据帧率和像素/米比例推算各参数。

        size_gate   : bbox 尺寸比例上限，防止近端/远端球员 ID 互换
        hist_weight : 颜色直方图代价权重（需提供 frames 才生效）
        hist_gate   : 直方图交叉距离上限，超出则拒绝匹配
        """
        max_dist = _PLAYER_MAX_SPEED_MS / fps * px_per_meter * _PLAYER_RADIUS_MARGIN
        max_age  = max(3, round(fps * _PLAYER_GAP_SECONDS))
        min_hits = max(2, round(fps * _PLAYER_MIN_HIT_SECONDS))
        sg_str = f"{size_gate:.1f}×" if size_gate is not None else "off"
        print(f"[ player ] fps={fps:.1f}  px/m={px_per_meter:.1f}  "
              f"max_dist={max_dist:.0f}px  size_gate={sg_str}  "
              f"hist_weight={hist_weight}  hist_gate={hist_gate}  "
              f"max_age={max_age}f  min_hits={min_hits}f  "
              f"conf=[{conf_low},{conf_high})")
        return cls(min_hits=min_hits, max_age=max_age,
                   conf_high=conf_high, conf_low=conf_low,
                   max_dist=max_dist, size_gate=size_gate,
                   hist_weight=hist_weight, hist_gate=hist_gate)

    def run(self, player_detections, frames=None):
        """
        输入：player_detections[i] = [{'bbox', 'conf', 'track_id'}, ...]
              frames — 可选，视频帧迭代器（BGR ndarray）；提供时启用颜色直方图外观匹配
        输出：同结构，CONFIRMED 轨迹含 track_id(int)；TENTATIVE 阶段回填。
        遮挡间隙帧无输出（track 保活但不插值，重现后续接同一 track_id）。
        """
        n = len(player_detections)
        self._tracker.reset()

        # ── 颜色直方图注入 ────────────────────────────────────────────────────
        if frames is not None:
            for fi, frame in enumerate(frames):
                if fi >= n:
                    break
                for det in player_detections[fi]:
                    det['hist'] = _extract_hist(frame, det['bbox'])

        # ── 逐帧追踪 ─────────────────────────────────────────────────────────
        tracked = []
        for fi, frame_dets in enumerate(player_detections):
            result = self._tracker.step(frame_dets, fi)
            tracked.append(result)

        # ── 收集各 track_id 的检测点（含 TENTATIVE 回填）────────────────────
        tid_frames: dict[int, list] = {}
        tentative_hist: dict[int, list] = {}
        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                tid  = det.get('track_id')
                _tid = det.get('_tid')
                if tid is not None:
                    tid_frames.setdefault(tid, []).append((fi, det))
                elif _tid is not None:
                    tentative_hist.setdefault(_tid, []).append((fi, det))

        for tid in list(tid_frames.keys()):
            if tid not in tentative_hist:
                continue
            confirmed_frames = {fi for fi, _ in tid_frames[tid]}
            prepend = [(fi, det) for fi, det in tentative_hist[tid]
                       if fi not in confirmed_frames]
            if prepend:
                tid_frames[tid] = sorted(prepend + tid_frames[tid], key=lambda x: x[0])

        # ── 写入 output ───────────────────────────────────────────────────────
        def _clean(det, **overrides):
            d = {k: v for k, v in det.items() if k != '_tid'}
            d.update(overrides)
            return d

        output = [[] for _ in range(n)]
        for tid, frames in tid_frames.items():
            for fi, det in frames:
                output[fi].append(_clean(det, track_id=tid))

        # ── 保留未被追踪的检测（供可视化）───────────────────────────────────
        backfilled: set[int] = set()
        for frames in tid_frames.values():
            for fi, det in frames:
                if det.get('track_id') is None:
                    backfilled.add(id(det))

        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                if det.get('track_id') is None and id(det) not in backfilled:
                    output[fi].append(_clean(det, track_id=None))

        n_tracked = sum(len(v) for v in tid_frames.values())
        print(f"[ player ] tracks={len(tid_frames)}  confirmed_dets={n_tracked}")

        return output


# ── 球拍追踪器 ────────────────────────────────────────────────────────────────

_RACKET_MAX_SPEED_MS    = 12.0  # 球拍中心最大速度（球员移动 + 挥拍）
_RACKET_RADIUS_MARGIN   = 1.5   # 搜索门限安全裕量
_RACKET_GAP_SECONDS     = 0.3   # max_age 对应时长（s），遮挡续接容忍时长
_RACKET_MIN_HIT_SECONDS = 0.04  # min_hits 对应时长（s）


class RacketTracker:
    """
    球拍追踪器：以检测框中心为追踪锚点，
    搜索门限基于球拍最大移动速度（单帧像素位移）。

    推荐用 RacketTracker.from_video(fps, px_per_meter) 构造。
    """

    def __init__(self, min_hits=2, max_age=8,
                 conf_high=0.5, conf_low=0.1, max_dist=None,
                 size_gate=4.0):
        self._tracker = Tracker(
            min_hits=min_hits, max_age=max_age,
            conf_high=conf_high, conf_low=conf_low,
            search_diameters=None, max_dist=max_dist,
            anchor_fn=_center_det, size_gate=size_gate,
            use_prediction=False,
        )

    @classmethod
    def from_video(cls, fps: float, px_per_meter: float,
                   conf_high: float = 0.5, conf_low: float = 0.1,
                   size_gate: float = 4.0):
        """根据帧率和像素/米比例推算各参数。"""
        max_dist = _RACKET_MAX_SPEED_MS / fps * px_per_meter * _RACKET_RADIUS_MARGIN
        max_age  = max(3, round(fps * _RACKET_GAP_SECONDS))
        min_hits = max(2, round(fps * _RACKET_MIN_HIT_SECONDS))
        sg_str   = f"{size_gate:.1f}×" if size_gate is not None else "off"
        print(f"[ racket ] fps={fps:.1f}  px/m={px_per_meter:.1f}  "
              f"max_dist={max_dist:.0f}px  size_gate={sg_str}  "
              f"max_age={max_age}f  min_hits={min_hits}f  "
              f"conf=[{conf_low},{conf_high})")
        return cls(min_hits=min_hits, max_age=max_age,
                   conf_high=conf_high, conf_low=conf_low,
                   max_dist=max_dist, size_gate=size_gate)

    def run(self, racket_detections):
        """
        输入：racket_detections[i] = [{'bbox', 'conf', 'track_id'}, ...]
        输出：同结构，CONFIRMED 轨迹含 track_id(int)；TENTATIVE 阶段回填。
        遮挡间隙帧无输出（track 保活但不插值，重现后续接同一 track_id）。
        """
        n = len(racket_detections)
        self._tracker.reset()

        # ── 逐帧追踪 ─────────────────────────────────────────────────────────
        tracked = []
        for fi, frame_dets in enumerate(racket_detections):
            result = self._tracker.step(frame_dets, fi)
            tracked.append(result)

        # ── 收集各 track_id 的检测点（含 TENTATIVE 回填）────────────────────
        tid_frames: dict[int, list] = {}
        tentative_hist: dict[int, list] = {}
        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                tid  = det.get('track_id')
                _tid = det.get('_tid')
                if tid is not None:
                    tid_frames.setdefault(tid, []).append((fi, det))
                elif _tid is not None:
                    tentative_hist.setdefault(_tid, []).append((fi, det))

        for tid in list(tid_frames.keys()):
            if tid not in tentative_hist:
                continue
            confirmed_frames = {fi for fi, _ in tid_frames[tid]}
            prepend = [(fi, det) for fi, det in tentative_hist[tid]
                       if fi not in confirmed_frames]
            if prepend:
                tid_frames[tid] = sorted(prepend + tid_frames[tid], key=lambda x: x[0])

        # ── 写入 output ───────────────────────────────────────────────────────
        def _clean(det, **overrides):
            d = {k: v for k, v in det.items() if k != '_tid'}
            d.update(overrides)
            return d

        output = [[] for _ in range(n)]
        for tid, frames in tid_frames.items():
            for fi, det in frames:
                output[fi].append(_clean(det, track_id=tid))

        # ── 保留未被追踪的检测（供可视化）───────────────────────────────────
        backfilled: set[int] = set()
        for frames in tid_frames.values():
            for fi, det in frames:
                if det.get('track_id') is None:
                    backfilled.add(id(det))

        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                if det.get('track_id') is None and id(det) not in backfilled:
                    output[fi].append(_clean(det, track_id=None))

        n_tracked = sum(len(v) for v in tid_frames.values())
        print(f"[ racket ] tracks={len(tid_frames)}  confirmed_dets={n_tracked}")

        return output
