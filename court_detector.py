"""
TemplateHomographyDetector
把网球场标准线条绘制成模板图像（白线/黑底），
通过迭代优化单应矩阵 H，最小化模板白线像素
到实际图像白线边缘的距离残差。

核心思路：
  cost(H) = mean( dist_transform[proj(template_white_pixels)] )
其中 dist_transform 是实际图像白线边缘的距离场，
     proj 是用 H 把模板米坐标投影到图像像素坐标。
"""
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# ── 球场尺寸（国际网联标准，米）─────────────────────────────────
COURT_W     = 10.97
COURT_L     = 23.77
SINGLE_W    =  8.23
SINGLE_OFF  = (COURT_W - SINGLE_W) / 2   # 1.37 m
NET_Y       = COURT_L / 2
SVC_L       =  6.40
SVC_Y_FAR   = NET_Y - SVC_L              # 5.485 m
SVC_Y_NEAR  = NET_Y + SVC_L              # 18.285 m
CTR_X       = COURT_W / 2

# 14 个模型关键点（与其他检测器对齐）
MODEL_KPS_M = np.array([
    [0,          0       ], [COURT_W,    0       ],
    [0,          COURT_L ], [COURT_W,    COURT_L ],
    [SINGLE_OFF, 0       ], [SINGLE_OFF, COURT_L ],
    [COURT_W-SINGLE_OFF, 0       ], [COURT_W-SINGLE_OFF, COURT_L ],
    [SINGLE_OFF,           SVC_Y_FAR  ], [COURT_W-SINGLE_OFF, SVC_Y_FAR  ],
    [SINGLE_OFF,           SVC_Y_NEAR ], [COURT_W-SINGLE_OFF, SVC_Y_NEAR ],
    [CTR_X,                SVC_Y_FAR  ], [CTR_X,              SVC_Y_NEAR ],
], dtype=np.float32)

# ── 线宽常量（ITF 标准）─────────────────────────────────────────
# 尺寸均指外缘。坐标原点 = 远端左双打角外缘。
BASELINE_W    = 0.10   # 底线：最大 10 cm
LINE_W        = 0.05   # 其余所有线：5 cm（取上限）
CENTER_MARK_W = 0.05   # 中心标志宽 5 cm
CENTER_MARK_L = 0.10   # 中心标志长 10 cm（向场内延伸）

# ── ITF 标准场地缓冲区（比赛场地最小净空）────────────────────────
CLEARANCE_BACK = 6.40  # 底线后方缓冲（米）
CLEARANCE_SIDE = 3.66  # 侧线外侧缓冲（米）

# ── 球场线段定义：(端点1, 端点2, 线宽_m) ─────────────────────
# 注：坐标为线条中心线坐标（各线已向场内偏移半个线宽，使外缘对齐尺寸）
_BASELINE_HALF_W = BASELINE_W / 2   # 底线半宽
_LINE_HALF_W     = LINE_W / 2       # 普通线半宽

COURT_LINES = [
    # 底线（外缘在 y=0 / y=COURT_L，中心线向场内偏移半宽）
    ([0,       _BASELINE_HALF_W],          [COURT_W, _BASELINE_HALF_W],          BASELINE_W),  # 远底线
    ([0,       COURT_L-_BASELINE_HALF_W],  [COURT_W, COURT_L-_BASELINE_HALF_W],  BASELINE_W),  # 近底线
    # 双打侧线（外缘在 x=0 / x=COURT_W）
    ([_LINE_HALF_W,     0],            [_LINE_HALF_W,     COURT_L],       LINE_W),      # 左双打线
    ([COURT_W-_LINE_HALF_W, 0],        [COURT_W-_LINE_HALF_W, COURT_L],  LINE_W),      # 右双打线
    # 单打侧线
    ([SINGLE_OFF+_LINE_HALF_W, 0],     [SINGLE_OFF+_LINE_HALF_W, COURT_L],       LINE_W),
    ([COURT_W-SINGLE_OFF-_LINE_HALF_W, 0],[COURT_W-SINGLE_OFF-_LINE_HALF_W, COURT_L], LINE_W),
    # 发球线
    ([SINGLE_OFF, SVC_Y_FAR],  [COURT_W-SINGLE_OFF, SVC_Y_FAR],  LINE_W),
    ([SINGLE_OFF, SVC_Y_NEAR], [COURT_W-SINGLE_OFF, SVC_Y_NEAR], LINE_W),
    # 中线（发球区中线）
    ([CTR_X, SVC_Y_FAR],      [CTR_X, SVC_Y_NEAR],    LINE_W),
    # 中心标志（底线中央，向场内延伸10cm，宽5cm）
    ([CTR_X, BASELINE_W],             [CTR_X, BASELINE_W+CENTER_MARK_L],    CENTER_MARK_W),
    ([CTR_X, COURT_L-BASELINE_W-CENTER_MARK_L], [CTR_X, COURT_L-BASELINE_W], CENTER_MARK_W),
]


def compute_H_from_kps(court_kps):
    """从 14 个关键点计算单应矩阵 H（球场米坐标 → 图像像素）。"""
    kps_2d = np.array(court_kps).reshape(14, 2).astype(np.float32)
    H, _   = cv2.findHomography(MODEL_KPS_M, kps_2d)
    return H


class CourtDetector:
    """
    主接口：
        kps  = detector.predict(frame)            # shape (28,) — 14 关键点
        hull = detector.get_valid_zone_hull(...)   # 球场有效区域凸包
    """

    @classmethod
    def from_H(cls, H):
        """从已知单应矩阵创建轻量实例（不加载模型），用于投影计算。"""
        obj = object.__new__(cls)
        obj._last_H = H
        return obj

    def __init__(self, scale: int = 40, seg_model: str = None):
        """
        scale: 模板分辨率（像素/米），越大越精细但越慢
        seg_model: YOLO seg 模型路径，用于检测球场多边形作为初始估计
        """
        self.scale = scale
        # 预计算：模板中所有白线像素的米坐标
        self._template_img, self._template_pts_m = self._build_template(scale)

        # YOLO seg 模型（球场区域检测）
        self._seg_model = None
        seg_path = seg_model or str(Path(__file__).parent / 'models/court_seg.pt')
        if Path(seg_path).exists():
            from ultralytics import YOLO
            self._seg_model = YOLO(seg_path, verbose=False)
            print(f"[   court] court seg model: {seg_path}")
        # 帧级推理缓存（同帧只推理一次）
        self._seg_cache_id = None   # id(frame)
        self._seg_cache_res = None  # 上次推理结果

    # ── 1. 构建模板 ────────────────────────────────────────────────
    def _build_template(self, scale: int):
        """
        按 ITF 标准线宽绘制球场白线：
          - 底线：10 cm
          - 其余线（侧线/发球线/中线）：5 cm
          - 中心标志：5 cm 宽 × 10 cm 长
        返回：
          template_img   : uint8 二值图（白=255/黑=0）
          template_pts_m : (N,2) float32，所有白线像素的米坐标
        """
        pad_m  = 1.0
        pad_px = int(pad_m * scale)
        W = int(COURT_W * scale) + 2 * pad_px
        H = int(COURT_L * scale) + 2 * pad_px
        tmpl = np.zeros((H, W), dtype=np.uint8)

        def m2p(xm, ym):
            return (int(round(xm * scale)) + pad_px,
                    int(round(ym * scale)) + pad_px)

        for (p1, p2, lw_m) in COURT_LINES:
            lw_px = max(1, int(round(lw_m * scale)))
            half = lw_px // 2
            px1 = m2p(p1[0], p1[1])
            px2 = m2p(p2[0], p2[1])
            if p1[1] == p2[1]:  # 水平线
                tl = (px1[0], px1[1] - half)
                br = (px2[0], px2[1] + (lw_px - half - 1))
            else:               # 垂直线
                tl = (px1[0] - half, px1[1])
                br = (px2[0] + (lw_px - half - 1), px2[1])
            cv2.rectangle(tmpl, tl, br, 255, -1)

        # 提取白线像素坐标 → 转换为米坐标
        ys, xs = np.where(tmpl > 0)
        pts_m = np.column_stack([
            (xs.astype(np.float32) - pad_px) / scale,
            (ys.astype(np.float32) - pad_px) / scale,
        ])
        print(f"[   court] template white pixels: {len(pts_m)}, "
              f"size: {W}x{H} px @ {scale}px/m")
        return tmpl, pts_m.astype(np.float32)

    # ── 2. 主入口 ──────────────────────────────────────────────────
    def predict(self, frame):
        """
        三阶段球场检测流程：

        步骤1  YOLO seg 初始化
               用 court_seg 模型分割球场区域，提取四边形四角，
               getPerspectiveTransform → 粗略初始单应矩阵 H_init。

        步骤2  粗 dist_map + Nelder-Mead 精调
               用 YOLO seg 多边形膨胀 mask（_get_court_mask）限定白线检测范围，
               对范围内的白线像素做距离变换得到 dist_map。
               以 dist_map 为目标，Nelder-Mead 精调 4 个角点坐标 → H_opt。
               此阶段 mask 范围较宽松（含场外余量），dist_map 可能含
               场地表面的假白点，但足以把 H 拉到正确区域。

        步骤3  精 dist_map + 再次精调
               用步骤2的 H_opt 把球场线条反投影回图像，得到仅覆盖
               实际白线附近的带状 line_mask，重建更精确的 dist_map2。
               再次 Nelder-Mead 从 H_opt 出发做精调 → H_opt2。
               line_mask 排除了大面积场地表面的假白点，代价函数
               梯度更锐利，优化结果更准确。
        """
        # 步骤1：YOLO seg 粗略初始化
        court_mask = self._get_court_mask(frame)
        dist_map   = self._build_dist_map(frame, court_mask)
        H_init, c_init = self._yolo_seg_init(frame, dist_map)
        print(f"[   court] YOLO seg init:   cost={c_init:.3f}")

        # 步骤2：粗 dist_map 下的 Nelder-Mead 精调
        H_opt = self._optimize(H_init, dist_map, frame.shape)

        # 步骤3：用 H_opt 重建精 dist_map，再次精调
        line_mask = self._build_line_mask(H_opt, frame.shape)
        dist_map2 = self._build_dist_map(frame, line_mask)
        H_opt2    = self._optimize(H_opt, dist_map2, frame.shape)

        self._last_H = H_opt2
        kps = self._project_keypoints(H_opt2)
        return kps.flatten()

    # ── 3. 球场 mask ────────────────────────────────────────────────
    def _run_seg(self, frame):
        """对 frame 做 YOLO seg 推理，带帧级缓存（同帧只推理一次）。"""
        fid = id(frame)
        if fid != self._seg_cache_id:
            if self._seg_model is None:
                raise RuntimeError("court seg model not loaded")
            self._seg_cache_res = self._seg_model(frame, verbose=False, conf=0.1)
            self._seg_cache_id  = fid
        return self._seg_cache_res

    def _get_court_mask(self, frame):
        """
        用 YOLO seg 多边形填充 + 膨胀作为 court_mask。
        膨胀量 ~5% 图像高度，保证边界线条完整落在 mask 内。
        YOLO seg 必须成功；若失败则抛出异常。
        """
        h, w = frame.shape[:2]
        results = self._run_seg(frame)
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            raise RuntimeError("YOLO seg: no court detected")
        best = int(results[0].boxes.conf.argmax())
        poly = results[0].masks.xy[best].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        ks = max(3, int(h * 0.05) | 1)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        return cv2.dilate(mask, k)

    # ── 4. 基于优化后 H 构建球场线条 mask ──────────────────────────
    def _build_line_mask(self, H, img_shape):
        """
        将所有球场线条（COURT_LINES + 网线）通过单应矩阵 H 投影到图像坐标，
        按自然线宽绘制，再用形态学膨胀扩展容差带，返回带状二值 mask。

        膨胀半径 = LINE_W（5 cm）换算成像素，吸收 H 的残余误差和透视不均匀，
        同时保持 mask 足够紧致，排除远离线条的场地表面（假白点来源）。

        px/m 估算：取远端底线（y=0）和近端底线（y=COURT_L）在图像中的
        像素宽度各除以 COURT_W，取平均，抵消透视压缩的影响。
        """
        h_img, w_img = img_shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        # 估算平均像素/米比例：用远近两端底线宽度取平均
        corners_m = np.array([[[0, 0], [COURT_W, 0],
                                [0, COURT_L], [COURT_W, COURT_L]]], dtype=np.float32)
        px = cv2.perspectiveTransform(corners_m, H)[0]
        ppm = (np.linalg.norm(px[1] - px[0]) / COURT_W +
               np.linalg.norm(px[3] - px[2]) / COURT_W) / 2.0

        # 按自然线宽绘制
        all_lines = list(COURT_LINES) + [([0, NET_Y], [COURT_W, NET_Y], LINE_W)]
        for (p1, p2, lw_m) in all_lines:
            pts = np.array([[[p1[0], p1[1]], [p2[0], p2[1]]]], dtype=np.float32)
            proj = cv2.perspectiveTransform(pts, H)[0].astype(int)
            thickness = max(1, round(lw_m * ppm))
            cv2.line(mask, tuple(proj[0]), tuple(proj[1]), 255, thickness)

        # 膨胀：扩展 LINE_W（5 cm）对应的像素距离
        r  = max(1, round(LINE_W * ppm))
        ks = 2 * r + 1
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        return cv2.dilate(mask, k)

    # ── 5. 构建距离场 ──────────────────────────────────────────────
    def _build_dist_map(self, frame, court_mask, cap=None):
        """
        在 court_mask 范围内检测白线像素，对非白线像素做距离变换，
        返回每个像素到最近白线的距离（上限 cap）。

        用法：
          - 步骤2 传入 _get_court_mask 的粗粒度 mask（整个球场区域）
          - 步骤3 传入 _build_line_mask 的细粒度 mask（线条带状区域）
          细粒度 mask 排除了大面积场地表面，dist_map 更干净，
          代价函数对 H 的偏移更敏感，优化收敛到更准确的结果。

        白线判定：HSV 高亮度（V > 180）+ 低饱和度（S < 50）。
        cap 限制最大距离，防止远离线条的区域主导代价均值。
        """
        if cap is None:
            cap = frame.shape[0] * 0.046    # ~4.6% 图像高度
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white    = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        white_in = cv2.bitwise_and(white, court_mask)
        dist = cv2.distanceTransform(
                   cv2.bitwise_not(white_in), cv2.DIST_L2, 5)
        return np.minimum(dist, cap).astype(np.float32)

    # ── 6. 代价函数：模板白线像素到实际边缘的平均距离 ─────────────
    def _compute_weights(self, H):
        """
        计算模板各点的透视权重，正比于该点投影到图像后的局部面积放大倍数
        （单应矩阵在该点的 Jacobian 行列式绝对值）。

        近端点在图像中占更大面积 → 权重更高，与肉眼感知一致。
        权重归一化为均值 = 1，使代价函数量纲不变。
        """
        pts  = self._template_pts_m                          # (N, 2)
        ones = np.ones((len(pts), 1), dtype=np.float32)
        ph   = (H @ np.column_stack([pts, ones]).T).T        # (N, 3)
        w    = ph[:, 2]
        u    = ph[:, 0] / w
        v    = ph[:, 1] / w
        # Jacobian 元素 ∂(u,v)/∂(x,y)
        J00  = (H[0, 0] - u * H[2, 0]) / w
        J01  = (H[0, 1] - u * H[2, 1]) / w
        J10  = (H[1, 0] - v * H[2, 0]) / w
        J11  = (H[1, 1] - v * H[2, 1]) / w
        jac  = np.abs(J00 * J11 - J01 * J10).astype(np.float32)
        mean = jac.mean()
        return (jac / mean).astype(np.float32) if mean > 0 else None

    def _cost(self, H, dist_map, weights=None):
        """
        计算代价：模板白线像素投影后在 dist_map 上的（加权）平均距离。
        weights: 与 _template_pts_m 等长的权重数组（None = 均匀权重）。
        """
        h_img, w_img = dist_map.shape

        # 检查投影关键点合法性
        kps_m = MODEL_KPS_M.reshape(-1, 1, 2)
        kps_proj = cv2.perspectiveTransform(kps_m, H).reshape(-1, 2)

        # 远端角点（0,1）必须严格在图像内
        far_c = kps_proj[:2]
        if not (np.all(far_c[:, 0] >= 0) and np.all(far_c[:, 0] < w_img) and
                np.all(far_c[:, 1] >= 0) and np.all(far_c[:, 1] < h_img)):
            return 1e6

        # 近端角点（2,3）：允许超出图像（俯视时近端底线可能在画面上方以外）
        # 但不能偏得太远
        near_c = kps_proj[2:4]
        nc_m = max(w_img, h_img) * 0.7
        if not (np.all(near_c[:, 0] > -nc_m) and np.all(near_c[:, 0] < w_img + nc_m) and
                np.all(near_c[:, 1] > -nc_m) and np.all(near_c[:, 1] < h_img + nc_m)):
            return 1e6

        # 拓扑：远端必须在近端上方（远端 y 更小 = 图像中更靠上）
        if far_c[:, 1].mean() >= near_c[:, 1].mean():
            return 1e6

        # 主代价：模板白线像素到最近边缘的（加权）平均距离
        pts = self._template_pts_m.reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        valid = ((proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) &
                 (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1))
        if valid.sum() < 100:
            return 1e6
        p    = proj[valid]
        xi   = p[:, 0].astype(np.int32)
        yi   = p[:, 1].astype(np.int32)
        vals = dist_map[yi, xi]
        if weights is not None:
            w = weights[valid]
            ws = w.sum()
            return float((vals * w).sum() / ws) if ws > 0 else 1e6
        return float(vals.mean())

    # ── 7. 优化：角点参数化 + Nelder-Mead 精调 ──────────────────────
    def _optimize(self, H_hint, dist_map, img_shape):
        """
        把 4 个双打角的图像坐标作为优化变量（8个参数），
        用 Nelder-Mead 从 YOLO seg 给出的初始 H 精调。
        """
        h_img, w_img = dist_map.shape

        # 四个双打角的世界坐标：tl/tr/bl/br
        SRC_M = np.array([[0,       0      ],
                          [COURT_W, 0      ],
                          [0,       COURT_L],
                          [COURT_W, COURT_L]], dtype=np.float32)

        def corners_to_H(params):
            dst = np.array(params, dtype=np.float32).reshape(4, 2)
            H, _ = cv2.findHomography(SRC_M, dst, method=0)
            return H

        if H_hint is None:
            raise RuntimeError("所有初始化方法均失败，无法检测球场")

        # 权重固定在优化开始前计算（用 H_hint），整个过程不变
        weights = self._compute_weights(H_hint)

        def corners_cost(params):
            """4-角参数化代价（含拓扑约束）"""
            dst = np.array(params, dtype=np.float32).reshape(4, 2)
            tl, tr, bl, br = dst
            margin   = int(h_img * 0.046)
            topo_gap = int(h_img * 0.046)
            for px, py in [tl, tr, bl, br]:
                if px < -margin or px > w_img + margin: return 1e6
                if py < -margin or py > h_img + margin: return 1e6
            if tl[1] >= bl[1] - topo_gap or tr[1] >= br[1] - topo_gap: return 1e6
            if tl[0] >= tr[0] - topo_gap or bl[0] >= br[0] - topo_gap: return 1e6
            poly = np.array([tl, tr, br, bl])
            n = len(poly)
            area = 0.5 * abs(sum(
                poly[i][0] * poly[(i+1) % n][1] - poly[(i+1) % n][0] * poly[i][1]
                for i in range(n)))
            if area < 0.05 * w_img * h_img: return 1e6
            H = corners_to_H(params)
            return 1e6 if H is None else self._cost(H, dist_map, weights)

        best_H    = H_hint
        best_cost = self._cost(H_hint, dist_map, weights)
        print(f"[   court] init H cost: {best_cost:.3f}")

        # Nelder-Mead 精调：用 4-角参数化，防止 H 漂移到退化解
        init_corners = cv2.perspectiveTransform(
            SRC_M.reshape(-1, 1, 2), best_H).reshape(-1, 2).flatten()
        r = minimize(corners_cost, init_corners, method='Nelder-Mead',
                     options={'maxiter': 8000, 'xatol': 0.5,
                              'fatol': 0.01, 'adaptive': True})
        H_nm = corners_to_H(r.x)
        c_nm = self._cost(H_nm, dist_map, weights) if H_nm is not None else 1e9
        print(f"[   court] Nelder-Mead refine: {best_cost:.3f} -> {c_nm:.3f}  ({r.nit} iters)")

        return (H_nm if c_nm < best_cost else best_H).reshape(3, 3)

    # ── 8. 初始单应矩阵：YOLO seg ────────────────────────────────────
    def _yolo_seg_init(self, frame, dist_map):
        """
        用 YOLO seg 模型检测球场多边形 mask，
        从 mask 的凸包提取 4 个角点，计算初始 H。
        返回 (H, cost)，失败时返回 (None, 1e9)。
        """
        try:
            results = self._run_seg(frame)
        except RuntimeError:
            return None, 1e9
        if not results or results[0].masks is None or len(results[0].masks) == 0:
            return None, 1e9

        # 选置信度最高的检测
        best = int(results[0].boxes.conf.argmax())
        poly = results[0].masks.xy[best].astype(np.float32)  # (N,2) pixel coords

        # 凸包 → 近似四边形
        hull = cv2.convexHull(poly.astype(np.int32))
        eps = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(np.float32)

        if len(approx) < 4:
            return None, 1e9

        # 若顶点 > 4 个，按面积选最优四边形
        if len(approx) != 4:
            approx = self._best_quad(approx)

        # 排序：tl, tr, br, bl（按 y 分上下两行，各行按 x 排序）
        corners = self._sort_quad(approx)

        src = np.array([
            [0,        0       ],
            [COURT_W,  0       ],
            [COURT_W,  COURT_L ],
            [0,        COURT_L ],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, corners)
        cost = self._cost(H, dist_map)
        return H.astype(np.float32), cost

    @staticmethod
    def _sort_quad(pts):
        """将 4 点排列为 tl, tr, br, bl 顺序。"""
        pts = pts[np.argsort(pts[:, 1])]   # 按 y 排序
        top, bot = pts[:2], pts[2:]
        top = top[np.argsort(top[:, 0])]   # 按 x 排序
        bot = bot[np.argsort(bot[:, 0])]
        return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

    @staticmethod
    def _best_quad(pts):
        """从多边形顶点中选出面积最大的四边形组合。"""
        from itertools import combinations
        best_area, best_quad = 0, pts[:4]
        for combo in combinations(range(len(pts)), 4):
            q = pts[list(combo)]
            hull = cv2.convexHull(q.astype(np.int32))
            area = cv2.contourArea(hull)
            if area > best_area:
                best_area = area
                best_quad = q
        return best_quad

    # ── 9. 投影关键点 ────────────────────────────────────────────────
    def _project_keypoints(self, H):
        pts = MODEL_KPS_M.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def _recover_camera(self, image_shape):
        """
        从单应矩阵 H 恢复相机内参 K 和外参 R, t。
        返回 (K, P)，P = K @ [R|t] 为 3×4 投影矩阵，可将世界坐标 (x,y,z,1) 投影到图像。
        """
        from scipy.optimize import minimize_scalar

        H = self._last_H
        h_img, w_img = image_shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0
        h1, h2 = H[:, 0], H[:, 1]

        def cost(f):
            Ki = np.array([[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1.0]])
            r1 = Ki @ h1;  r2 = Ki @ h2
            return (r1 @ r2) ** 2 + (r1 @ r1 - r2 @ r2) ** 2

        f  = minimize_scalar(cost, bounds=(w_img * 0.3, w_img * 20), method='bounded').x
        K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Ki = np.linalg.inv(K)

        lam = 1.0 / np.linalg.norm(Ki @ h1)
        r1  = lam * (Ki @ H[:, 0])
        r2  = lam * (Ki @ H[:, 1])
        r3  = np.cross(r1, r2)
        t   = lam * (Ki @ H[:, 2])
        if t[2] < 0:
            r1, r2, r3, t = -r1, -r2, -r3, -t

        R = np.column_stack([r1, r2, r3])

        # 确保 z 向上（高处 v 更小）
        mid = np.array([COURT_W / 2, COURT_L / 2, 0.0])
        cam_mid   = R @ mid + t
        cam_above = R @ (mid + [0, 0, 1]) + t
        if (K[1, 1] * cam_above[1] / cam_above[2] + K[1, 2] >
                K[1, 1] * cam_mid[1] / cam_mid[2] + K[1, 2]):
            r3 = -r3
            R  = np.column_stack([r1, r2, r3])

        P = K @ np.hstack([R, t[:, None]])
        return K, P

    def _project_3d(self, pts3d, P):
        """将 (N,3) 世界坐标通过投影矩阵 P 投影为 (N,2) 图像坐标。"""
        ph  = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        uv  = (P @ ph.T).T
        return (uv[:, :2] / uv[:, 2:]).astype(np.float32)

    def get_valid_zone_hull(self, image_shape, expand=1.5, height=7.0):
        """
        将 3D 三棱柱（帐篷形）投影到图像，返回凸包多边形（用于有效区域过滤）。
        三棱柱定义：
          底面 = 双打球场四角向外扩展 expand 米（z=0）
          脊线 = 底线中点（网位置）左右两端，高度 height 米
        返回 numpy int32 数组，形状 (N,1,2)，可直接传给 cv2.pointPolygonTest。
        """
        _, P = self._recover_camera(image_shape)

        x0, x1 = -expand, COURT_W + expand
        y0, y1 = -expand, COURT_L + expand
        ym = COURT_L / 2
        pts3d = np.array([
            [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
            [x0, ym, height], [x1, ym, height],
        ])

        pts2d = self._project_3d(pts3d, P)
        hull  = cv2.convexHull(pts2d.reshape(-1, 1, 2))
        return hull.astype(np.int32)

    def get_clearance_volume_hull(self, image_shape, back, side, height=2.0):
        """
        将缓冲区立方体（8 个顶点）投影到图像，返回凸包多边形和各顶点像素坐标。
        底面 = 缓冲区矩形（z=0），顶面 = 同一矩形上移 height 米（z=height）。
        返回 (hull, pts2d_bottom, pts2d_top)：
          hull          : (N,1,2) int32，可用于 pointPolygonTest / fillPoly
          pts2d_bottom  : (4,2) float32，底面四角像素坐标（顺序 tl,tr,br,bl）
          pts2d_top     : (4,2) float32，顶面四角像素坐标
        必须在 predict() 之后调用。
        """
        _, P = self._recover_camera(image_shape)

        x0, x1 = -side, COURT_W + side
        y0, y1 = -back, COURT_L + back
        bottom = np.array([[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]], dtype=np.float32)
        top    = np.array([[x0,y0,height],[x1,y0,height],[x1,y1,height],[x0,y1,height]], dtype=np.float32)

        pts2d_bottom = self._project_3d(bottom, P)
        pts2d_top    = self._project_3d(top,    P)
        all_pts      = np.vstack([pts2d_bottom, pts2d_top])
        hull         = cv2.convexHull(all_pts.reshape(-1, 1, 2))
        return hull.astype(np.int32), pts2d_bottom, pts2d_top

    def get_clearance_hull(self, back=CLEARANCE_BACK, side=CLEARANCE_SIDE):
        """
        将缓冲区矩形通过单应矩阵 H 投影到图像，返回四边形顶点 (4,1,2) int32。
        back : 底线后方缓冲（米），默认 ITF 标准
        side : 侧线外侧缓冲（米），默认 ITF 标准
        """
        pts_m = np.array([
            [-side,          -back          ],
            [COURT_W + side, -back          ],
            [COURT_W + side,  COURT_L + back],
            [-side,           COURT_L + back],
        ], dtype=np.float32)
        pts_px = cv2.perspectiveTransform(pts_m.reshape(-1, 1, 2), self._last_H)
        return pts_px.reshape(-1, 1, 2).astype(np.int32)

