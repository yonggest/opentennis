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
  BallTracker   — 离线网球追踪：前过滤（形状/孤立点/静态误检）→ 运动追踪 → 后过滤
  PlayerTracker — 球员追踪器，以检测框底部中心（脚点）为追踪锚点
  RacketTracker — 球拍追踪器
"""

import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── 物理常量（网球）─────────────────────────────────────────────────────────

_BALL_D_M        = 0.067   # ITF 网球直径（m）
_MAX_SPEED_MS    = 41.7    # 最大球速 150 km/h（m/s），用于 max_dist 兜底截断
_RADIUS_MARGIN   = 1.3     # max_dist 安全裕量系数
_BBOX_MIN_FACTOR = 0.5     # min_area = (ball_d_px × 系数)²
_BBOX_MAX_FACTOR = 15.0    # max_area = (ball_d_px × 系数)²
_GAP_SECONDS     = 0.25    # max_age 对应时长（s）
_MIN_HIT_SECONDS = 0.05    # min_hits 对应时长（s）
_SEARCH_DIAMETERS = 3.0    # 搜索半径 = N × 球径

# ── 追踪器通用常量 ────────────────────────────────────────────────────────────

_LINEAR_WINDOW = 3      # 线性预测：仅取最近 N 个历史点估计速度方向
_HIST_BINS     = 16     # HSV H 通道直方图 bin 数
_HIST_MOMENTUM = 0.8    # 直方图 EMA 系数：旧值权重

# ── 前过滤：静态误检 ──────────────────────────────────────────────────────────

_STATIC_IOU_THRESH = 0.5   # 背景板静态误检：bbox IoU 下限
_STATIC_MIN_GAP_S  = 2.0   # 背景板静态误检：帧差下限（秒），排除同一次击球的连续帧
_STATIC_MIN_COUNT  = 5     # 背景板静态误检：远距离匹配次数下限（避免偶发误判）

# ── 次检测器（recall）────────────────────────────────────────────────────────

_RECALL_PATCH = 96   # recall 裁图边长及推断尺寸（px），与 yolo26n-ball.pt 训练 imgsz 一致


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
        self._pred2             = (ax, ay)
        self.near_racket: bool  = False   # 本帧预测中心落在球拍 bbox 内，匹配门限扩至 max_dist
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
        """搜索门限：
        - near_racket=True → 可能发生击球转折，退化到 max_dist 全向搜索
        - search_diameters=None（固定门限模式）或仅 1 个历史点 → max_dist
        - 否则 → search_radius（物理约束小圆）
        """
        if self.near_racket and self._max_dist is not None:
            return self._max_dist
        if self._search_diameters is None or (len(self.history) == 1 and self._max_dist is not None):
            return self._max_dist
        return self.search_radius

    def predict(self):
        """更新预测位置，age+1。
        use_prediction=True：线性外推；False：停在上一帧位置。"""
        if not self._use_prediction:
            self._pred  = (self.history[-1][1], self.history[-1][2])
            self._pred2 = self._pred
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
        self._pred = (float(np.polyval(px, tp)), float(np.polyval(py, tp)))
        # 次预测：仅取最近2点外推，对转折（落地弹起、击球）更敏感
        if len(h) >= 2:
            px2 = np.polyfit(tn[-2:], xs[-2:], 1)
            py2 = np.polyfit(tn[-2:], ys[-2:], 1)
            self._pred2 = (float(np.polyval(px2, tp)), float(np.polyval(py2, tp)))
        else:
            self._pred2 = self._pred
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

    @property
    def predicted_center2(self):
        """次预测中心：最近2点线性外推，仅在3+历史点时与 predicted_center 不同。"""
        return self._pred2


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
           size_gate=None, hist_weight=0.0, hist_gate=None,
           secondary_centers=None):
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
                # 次预测：取两个预测圆中较近的距离（并集搜索）
                if secondary_centers is not None and secondary_centers[i] is not None:
                    scx, scy = secondary_centers[i]
                    dist2 = ((scx - dcx)**2 + (scy - dcy)**2) ** 0.5
                    dist = min(dist, dist2)
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

    def predict_all(self):
        """对所有轨迹执行线性外推预测，更新 predicted_center 和 age。"""
        for t in self._tracks:
            t.predict()

    def step(self, detections, frame_idx, skip_predict=False):
        """
        处理单帧检测（两阶段 ByteTrack 风格）。

        输入 : detections    = [{'bbox':[x1,y1,x2,y2], 'conf':float, ...}, ...]
               frame_idx     = 当前帧号（int）
               skip_predict  = True 时跳过预测步骤（由调用方提前调用 predict_all()）
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

        # 1. 预测（调用方已预测时跳过）
        if not skip_predict:
            self.predict_all()

        # 2. 阶段一：所有轨迹 vs 高置信度检测（逐轨迹门限）
        gates1 = [t.effective_gate for t in self._tracks]
        sec1   = [t.predicted_center2 for t in self._tracks]
        matched1, unmatched_t1, unmatched_d_high = _match(
            self._tracks, dets_high, gates1,
            anchor_fn=self._anchor_fn, size_gate=self._size_gate,
            hist_weight=self._hist_weight, hist_gate=self._hist_gate,
            secondary_centers=sec1)
        for ti, di in matched1:
            self._tracks[ti].update(dets_high[di], frame_idx)

        # 3. 阶段二：阶段一未匹配的 CONFIRMED 轨迹 vs 低置信度检测（逐轨迹门限）
        # 仅 CONFIRMED 轨迹参与：TENTATIVE 轨迹不抢低置信度检测，防止"抢占"导致
        # CONFIRMED 轨迹丢失检测点，进而触发 recall 产生重复轨迹。
        matched2 = []
        if unmatched_t1 and dets_low:
            unmatched_t1_confirmed = [ti for ti in unmatched_t1
                                      if self._tracks[ti].state == TrackState.CONFIRMED]
            tracks2 = [self._tracks[ti] for ti in unmatched_t1_confirmed]
            gates2  = [t.effective_gate for t in tracks2]
            sec2    = [t.predicted_center2 for t in tracks2]
            matched2_local, _, _ = _match(
                tracks2, dets_low, gates2,
                anchor_fn=self._anchor_fn, size_gate=self._size_gate,
                hist_weight=self._hist_weight, hist_gate=self._hist_gate,
                secondary_centers=sec2)
            for i2, di2 in matched2_local:
                ti = unmatched_t1_confirmed[i2]
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


# ── Recall 辅助 ───────────────────────────────────────────────────────────────


def _sub_detect_crop(frame, cx, cy, model, fi, tid, save_dir=None):
    """以 (cx, cy) 为中心裁 _RECALL_PATCH×_RECALL_PATCH，用次检测器检测网球。

    以极低置信度（0.01）获取全部候选，返回置信度最高的检测（原图坐标 det dict）；
    无检测则返回 None。

    fi / tid  仅用于打印和文件命名。
    save_dir  不为 None 时，将 patch 图（叠加预测中心 + 检测框）存入该目录。
    """
    h, w = frame.shape[:2]
    patch = _RECALL_PATCH
    if w < patch or h < patch:
        return None
    half = patch // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, w - patch))
    y0 = max(0, min(y0, h - patch))
    crop = frame[y0:y0 + patch, x0:x0 + patch]

    # 以极低置信度推断，获取全部候选（坐标保留 patch 内相对坐标，方便存图标注）
    results = model.predict(crop, imgsz=patch, conf=0.01, verbose=False)
    candidates_crop  = []   # (conf, x1, y1, x2, y2) — patch 内坐标
    candidates_frame = []   # (conf, x1, y1, x2, y2) — 原图坐标
    for r in results:
        for box in r.boxes:
            c = float(box.conf)
            px1, py1, px2, py2 = box.xyxy[0].tolist()
            candidates_crop.append((c, px1, py1, px2, py2))
            candidates_frame.append((c, px1 + x0, py1 + y0, px2 + x0, py2 + y0))
    candidates_crop.sort(key=lambda x: -x[0])
    candidates_frame.sort(key=lambda x: -x[0])

    # 保存调试 patch 图（可选）
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        vis = crop.copy()
        # 预测中心在 patch 内的实际像素坐标（crop 被夹到边界时不再是正中心）
        pred_px = int(round(cx)) - x0
        pred_py = int(round(cy)) - y0
        cv2.drawMarker(vis, (pred_px, pred_py), (255, 100, 0),
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        # 最佳检测框（有检测→绿色，无检测→不画）
        if candidates_crop:
            bc, bx1, by1, bx2, by2 = candidates_crop[0]
            cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 200, 0), 1)
            cv2.putText(vis, f"{bc:.2f}", (int(bx1), max(int(by1) - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1, cv2.LINE_AA)
        tag  = f"{candidates_crop[0][0]:.3f}" if candidates_crop else "none"
        name = f"sub_f{fi:05d}_tid{tid}_conf{tag}.jpg"
        cv2.imwrite(os.path.join(save_dir, name), vis)

    # 返回置信度最高的检测（原图坐标），无检测则返回 None
    if candidates_frame:
        c, x1, y1, x2, y2 = candidates_frame[0]
        return {'bbox': [x1, y1, x2, y2], 'conf': c}
    return None


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
                 conf_high=0.5, conf_low=0.0,
                 search_diameters=_SEARCH_DIAMETERS, max_dist=None,
                 min_area=20.0, max_area=8000.0,
                 min_aspect_h=0.15, min_aspect_v=0.5,
                 ball_d_px=0.0, fps=25.0, H_inv=None,
                 backdrop_poly=None,
                 sub_model=None, sub_save_dir=None):
        self._tracker = Tracker(min_hits=min_hits, max_age=max_age,
                                conf_high=conf_high, conf_low=conf_low,
                                search_diameters=search_diameters,
                                max_dist=max_dist,
                                size_gate=2.0)
        self.min_area     = min_area
        self.max_area     = max_area
        self.min_aspect_h = min_aspect_h   # 水平拉长（w≥h）下限：运动模糊允许较大拉伸
        self.min_aspect_v = min_aspect_v   # 垂直拉长（h>w）下限：竖向模糊罕见，严格限制
        self._ball_d_px = ball_d_px
        self._fps       = fps
        # 背景板判断：优先用 3D 投影多边形，备用单应矩阵反投影
        self._backdrop_contour = (
            np.array(backdrop_poly, dtype=np.float32) if backdrop_poly is not None else None
        )
        self._H_inv        = H_inv
        self._sub_model    = sub_model
        self._sub_save_dir = sub_save_dir

    @classmethod
    def from_video(cls, fps: float, px_per_meter: float,
                   conf_high: float = 0.5, conf_low: float = 0.0,
                   min_aspect_h: float = 0.15, min_aspect_v: float = 0.5,
                   search_diameters: float = _SEARCH_DIAMETERS,
                   H_inv=None, backdrop_poly=None,
                   sub_model=None, sub_save_dir=None):
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
                   min_area=min_area, max_area=max_area,
                   min_aspect_h=min_aspect_h, min_aspect_v=min_aspect_v,
                   ball_d_px=ball_d_px, fps=fps, H_inv=H_inv,
                   backdrop_poly=backdrop_poly,
                   sub_model=sub_model,
                   sub_save_dir=sub_save_dir)

    def _in_backdrop(self, det) -> bool:
        """检测中心是否在远端背景板区域内。"""
        x1, y1, x2, y2 = det['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if self._backdrop_contour is not None:
            return cv2.pointPolygonTest(self._backdrop_contour,
                                        (float(cx), float(cy)), False) >= 0
        if self._H_inv is not None:
            pt = cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32), self._H_inv)[0][0]
            return float(pt[1]) < 0   # y_court < 0 → 远端底线以外
        return True   # 无几何信息时对所有点应用

    # ── 1. 前过滤 ──────────────────────────────────────────────────────────────

    def _prefilter(self, ball_detections, debug_frame=-1):
        """前过滤：① 尺寸/形状过滤  ② 静态误检过滤。

        ① 尺寸/形状：面积和长宽比不合规的检测直接丢弃。
        ② 静态误检：背景板区域内的检测，若在整个视频中同位置（IoU > 阈值）
           累计出现 ≥ MIN_COUNT 次（帧差 > min_gap，且跨度 ≥ min_gap），视为固定误检丢弃。
        """
        n = len(ball_detections)

        # ── ① 尺寸/形状过滤 ──────────────────────────────────────────────
        shape_ok = []
        dropped  = []
        for fi, dets in enumerate(ball_detections):
            passed, fail = [], []
            for d in dets:
                a = _area(d['bbox'])
                x1, y1, x2, y2 = d['bbox']
                w, h = x2 - x1, y2 - y1
                # 方向感知长宽比：水平拉长（运动模糊）宽松，垂直拉长严格
                if w >= h:
                    asp_ok = (h / w >= self.min_aspect_h) if w > 0 else False
                else:
                    asp_ok = (w / h >= self.min_aspect_v) if h > 0 else False
                if self.min_area <= a <= self.max_area and asp_ok:
                    passed.append(d)
                else:
                    fail.append(d)
            shape_ok.append(passed)
            dropped.append(fail)
            if fi == debug_frame and (passed or fail):
                print(f"[dbg f{fi}] prefilter shape: {len(passed)} passed, {len(fail)} dropped"
                      f"  (area∈[{self.min_area:.0f},{self.max_area:.0f}]"
                      f"  asp_h>={self.min_aspect_h} asp_v>={self.min_aspect_v})")
                for d in fail:
                    print(f"           DROPPED  conf={d['conf']:.3f}"
                          f"  area={_area(d['bbox']):.0f}  asp={_aspect(d['bbox']):.2f}")
                for d in passed:
                    print(f"           passed   conf={d['conf']:.3f}"
                          f"  area={_area(d['bbox']):.0f}  asp={_aspect(d['bbox']):.2f}")

        n_after_shape = sum(len(f) for f in shape_ok)
        # 孤立点过滤已移除：孤立检测无法满足 min_hits 形成 CONFIRMED 轨迹，
        # tracker 自然淘汰；parse.py 再移除无 track_id 的检测。
        # 保留孤立检测还能帮助已有轨迹在漏检帧续接。
        iso_ok = shape_ok

        # ── ③ 静态误检过滤（仅限远端背景板区域）────────────────────────
        # 背景板是固定误检高发区：广告牌、场地标志等在整个视频中反复出现在相近位置。
        # 判断条件：同位置（IoU > 阈值）在帧差 > min_gap 的帧中累计匹配 >= MIN_COUNT 次
        min_gap = max(1, round(self._fps * _STATIC_MIN_GAP_S))

        # 收集背景板区域内的检测：(frame_idx, x1, y1, x2, y2, det_id)
        bd_rows = []   # List of [fi, x1, y1, x2, y2, det_id]
        for fi, dets in enumerate(iso_ok):
            for d in dets:
                if not self._in_backdrop(d):
                    continue
                x1, y1, x2, y2 = d['bbox']
                bd_rows.append((fi, x1, y1, x2, y2, id(d)))

        if bd_rows:
            arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in bd_rows],
                           dtype=np.float32)
            arr_fi, arr_x1, arr_y1, arr_x2, arr_y2 = arr.T
            arr_area = (arr_x2 - arr_x1) * (arr_y2 - arr_y1)
            det_ids  = [r[5] for r in bd_rows]
            static_ids: set[int] = set()
            for i in range(len(det_ids)):
                far   = np.abs(arr_fi - arr_fi[i]) > min_gap
                ix1   = np.maximum(arr_x1, arr_x1[i])
                iy1   = np.maximum(arr_y1, arr_y1[i])
                ix2   = np.minimum(arr_x2, arr_x2[i])
                iy2   = np.minimum(arr_y2, arr_y2[i])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                union = arr_area + arr_area[i] - inter
                iou   = inter / np.maximum(union, 1e-6)
                mask = far & (iou > _STATIC_IOU_THRESH)
                # 匹配帧必须在时间上分散（跨度 >= min_gap），排除"另一条真实球轨迹恰好经过同位置"的情况
                if np.sum(mask) >= _STATIC_MIN_COUNT and (arr_fi[mask].max() - arr_fi[mask].min()) >= min_gap:
                    static_ids.add(det_ids[i])
            n_static = len(static_ids)
            candidates = []
            for fi in range(len(iso_ok)):
                keep, fail = [], []
                for d in iso_ok[fi]:
                    (fail if id(d) in static_ids else keep).append(d)
                candidates.append(keep)
                dropped[fi].extend(fail)
        else:
            n_static = 0
            candidates = iso_ok

        n_raw          = sum(len(f) for f in ball_detections)
        n_after_static = sum(len(f) for f in candidates)
        print(f"[prefilter] raw={n_raw}"
              f"  →shape→ {n_after_shape} (-{n_raw - n_after_shape})"
              f"  →static→ {n_after_static} (-{n_static})"
              f"  (iou>{_STATIC_IOU_THRESH}  min_gap={min_gap}f  min_count={_STATIC_MIN_COUNT})")

        return candidates, dropped

    # ── 2. 运动跟踪 ────────────────────────────────────────────────────────────

    @staticmethod
    def _point_in_racket(cx, cy, frame_rackets):
        """判断点 (cx, cy) 是否落在任意球拍 bbox 内。"""
        for r in (frame_rackets or []):
            x1, y1, x2, y2 = r['bbox']
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _track(self, candidates, debug_frame, frames, n, rackets=None):
        """运动跟踪：逐帧 Tracker + Recall 补检 + TENTATIVE 回填。

        rackets : list[list[det]]，逐帧球拍检测（可为 None）。
                  当球的预测中心落在球拍 bbox 内时，视为可能发生击球转折，
                  匹配门限扩至 max_dist，并在 frame_predictions 中记录大圆 r_max。

        返回 (tid_frames, tracked, frame_predictions)：
          tid_frames        — {track_id: [(frame_idx, det), ...]}（已含 TENTATIVE 回填）
          tracked           — 逐帧 Tracker 原始输出，供后续保留未追踪检测使用
          frame_predictions — 每帧活跃轨迹的预测圆列表
                              [  # frame 0
                                [{"tid":1,"cx":x,"cy":y,"r":r,
                                  "cx2":x2,"cy2":y2,         # 可选，次预测
                                  "r_max":rmax               # 可选，击球转折大圆
                                 }, ...],
                                ...
                              ]
        """
        tracked           = []
        frame_predictions = []
        frame_iter = iter(frames) if frames is not None else None
        use_sub    = frame_iter is not None and self._sub_model is not None
        rcl_tried  = rcl_found = 0

        for fi, frame_dets in enumerate(candidates):
            frame = next(frame_iter) if frame_iter is not None else None
            frame_rackets = rackets[fi] if rackets is not None and fi < len(rackets) else None

            self._tracker.predict_all()

            # predict_all() 之后：标记球拍感知，再记录搜索圆
            for t in self._tracker._tracks:
                cx, cy = t.predicted_center
                t.near_racket = self._point_in_racket(cx, cy, frame_rackets)

            # 记录所有活跃轨迹的预测圆（含 TENTATIVE）
            preds = []
            for t in self._tracker._tracks:
                cx,  cy  = t.predicted_center
                cx2, cy2 = t.predicted_center2
                # 显示半径：用正常搜索半径（不受 near_racket 影响），保留方向信息
                if t._search_diameters is not None and len(t.history) >= 2:
                    display_r = t.search_radius
                else:
                    display_r = t._max_dist or t.search_radius
                entry = {
                    'tid': t.id,
                    'cx':  round(float(cx),  1),
                    'cy':  round(float(cy),  1),
                    'r':   round(float(display_r), 1),
                }
                # 次预测圆：仅在与主预测有明显差异时（≥3 个历史点）才存储
                if len(t.history) >= 3 and (abs(cx2 - cx) > 0.5 or abs(cy2 - cy) > 0.5):
                    entry['cx2'] = round(float(cx2), 1)
                    entry['cy2'] = round(float(cy2), 1)
                # 球拍感知大圆：击球转折时额外叠加全向 max_dist 搜索圆
                if t.near_racket and t._max_dist is not None:
                    entry['r_max'] = round(float(t._max_dist), 1)
                preds.append(entry)
            frame_predictions.append(preds)

            if use_sub:
                h, w = frame.shape[:2]
                for t in self._tracker._tracks:
                    if t.state != TrackState.CONFIRMED:
                        continue
                    tcx, tcy = t.predicted_center
                    if any(((tcx - _center(d['bbox'])[0])**2 +
                            (tcy - _center(d['bbox'])[1])**2) ** 0.5 <= t.effective_gate
                           for d in frame_dets):
                        continue
                    rcl_tried += 1
                    rdet = _sub_detect_crop(
                        frame, tcx, tcy, self._sub_model,
                        fi, t.id, save_dir=self._sub_save_dir)
                    if rdet is not None:
                        overlap = next((d for d in frame_dets
                                        if _iou(rdet['bbox'], d['bbox']) > 0.3), None)
                        b = [round(v) for v in rdet['bbox']]
                        bbox_str = f"[{b[0]:4d},{b[1]:4d},{b[2]:4d},{b[3]:4d}]"
                        hdr = (f"\033[1;32m[recall] f{fi:<5} tid={t.id:<3}"
                               f"  bbox={bbox_str}  conf={rdet['conf']:.3f}")
                        if overlap is not None:
                            print(f"{hdr}  iou={_iou(rdet['bbox'], overlap['bbox']):.3f}"
                                  f"  → 已知点，不续接\033[0m")
                            continue
                        near = next((d for d in frame_dets
                                     if 0 < _iou(rdet['bbox'], d['bbox']) <= 0.3), None)
                        if near is not None:
                            print(f"{hdr}  WARNING low iou={_iou(rdet['bbox'], near['bbox']):.3f}"
                                  f"  existing conf={near.get('conf', 0.0):.3f}\033[0m")
                        rcl_found += 1
                        print(f"{hdr}\033[0m")
                        frame_dets.append(dict(rdet, _recall=True))

            result = self._tracker.step(frame_dets, fi, skip_predict=True)
            tracked.append(result)

            if fi == debug_frame:
                tr = self._tracker
                n_high = sum(1 for d in frame_dets if d.get('conf', 1.0) >= tr.conf_high)
                n_low  = sum(1 for d in frame_dets
                             if tr.conf_low <= d.get('conf', 1.0) < tr.conf_high)
                print(f"[dbg f{fi}] conf split: high(>={tr.conf_high})={n_high}"
                      f"  low([{tr.conf_low},{tr.conf_high}))={n_low}")
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

        if use_sub:
            print(f"[ recall ] tried={rcl_tried}  found={rcl_found}"
                  + (f"  ({rcl_found/rcl_tried*100:.1f}%)" if rcl_tried else ""))

        # ── TENTATIVE 回填 ────────────────────────────────────────────────
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
            # 确认帧 = tid_frames 中最早的帧（轨迹首次 CONFIRMED 的帧）
            confirmation_frame = min(fi for fi, _ in tid_frames[tid])
            confirmed_frames   = {fi for fi, _ in tid_frames[tid]}
            prepend = [(fi, det) for fi, det in tentative_hist[tid]
                       if fi not in confirmed_frames]
            if prepend:
                for _, det in prepend:
                    det['backfill']    = True
                    det['revealed_at'] = confirmation_frame
                tid_frames[tid] = sorted(prepend + tid_frames[tid], key=lambda x: x[0])

        n_cands   = sum(len(f) for f in candidates)
        n_tracked = sum(len(v) for v in tid_frames.values())
        print(f"[  track  ] candidates={n_cands}  tracks={len(tid_frames)}"
              f"  dets={n_tracked}")

        return tid_frames, tracked, frame_predictions

    # ── 3. 后过滤 ──────────────────────────────────────────────────────────────

    def _postfilter(self, tid_frames):
        """后过滤：（未实现）"""
        return tid_frames

    # ── 组装输出 ────────────────────────────────────────────────────────────────

    def _build_output(self, tid_frames, tracked, dropped_dets, n):
        """Gap 线性插值 + 合并 dropped / 未追踪检测，组装逐帧输出列表。"""
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
                            'conf': 0.0, 'track_id': tid,
                            'interpolated': True, 'revealed_at': fi_b,
                        })
            fi_last, det_last = frames[-1]
            output[fi_last].append(_clean(det_last, track_id=tid))

        backfilled: set[int] = set()
        for frames in tid_frames.values():
            for fi, det in frames:
                if det.get('track_id') is None:
                    backfilled.add(id(det))

        for fi, dets in enumerate(dropped_dets):
            for det in dets:
                output[fi].append(_clean(det, track_id=None, valid=False))
        for fi, frame_dets in enumerate(tracked):
            for det in frame_dets:
                if det.get('track_id') is None and id(det) not in backfilled:
                    output[fi].append(_clean(det, track_id=None))

        return output

    # ── 主入口 ──────────────────────────────────────────────────────────────────

    def run(self, ball_detections,
            rackets=None, players=None, court=None,
            debug_frame: int = -1, frames=None):
        """
        输入：
          ball_detections — 逐帧网球检测列表
          rackets         — 逐帧球拍检测列表（可为 None）；用于球拍感知预测扩圆
          players         — 逐帧球员检测列表（可为 None）；预留，暂未使用
          court           — 球场信息 dict（可为 None）；预留，暂未使用
        输出：(balls, frame_predictions)
          balls             — 同输入结构，CONFIRMED 轨迹含 track_id(int)，gap 帧线性插值
          frame_predictions — 每帧轨迹的预测圆列表，供 check_json.py 直接渲染
        """
        n = len(ball_detections)
        self._tracker.reset()

        # ── 1. 前过滤 ────────────────────────────────────────────────────
        candidates, dropped_dets = self._prefilter(ball_detections, debug_frame)

        # ── 2. 运动跟踪 ──────────────────────────────────────────────────
        tid_frames, tracked, frame_predictions = self._track(
            candidates, debug_frame, frames, n, rackets=rackets)

        # ── 3. 后过滤 ────────────────────────────────────────────────────
        tid_frames = self._postfilter(tid_frames)

        return self._build_output(tid_frames, tracked, dropped_dets, n), frame_predictions


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
