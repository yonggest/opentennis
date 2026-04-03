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
from scipy.optimize import minimize, differential_evolution

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

# ── 球场线段定义：(端点1, 端点2, 线宽_m) ─────────────────────
# 注：坐标为线条中心线坐标（各线已向场内偏移半个线宽，使外缘对齐尺寸）
_BH = BASELINE_W / 2   # 底线半宽
_LH = LINE_W / 2       # 普通线半宽

COURT_LINES = [
    # 底线（外缘在 y=0 / y=COURT_L，中心线向场内偏移半宽）
    ([0,       _BH],          [COURT_W, _BH],          BASELINE_W),  # 远底线
    ([0,       COURT_L-_BH],  [COURT_W, COURT_L-_BH],  BASELINE_W),  # 近底线
    # 双打侧线（外缘在 x=0 / x=COURT_W）
    ([_LH,     0],            [_LH,     COURT_L],       LINE_W),      # 左双打线
    ([COURT_W-_LH, 0],        [COURT_W-_LH, COURT_L],  LINE_W),      # 右双打线
    # 单打侧线
    ([SINGLE_OFF+_LH, 0],     [SINGLE_OFF+_LH, COURT_L],       LINE_W),
    ([COURT_W-SINGLE_OFF-_LH, 0],[COURT_W-SINGLE_OFF-_LH, COURT_L], LINE_W),
    # 发球线
    ([SINGLE_OFF, SVC_Y_FAR],  [COURT_W-SINGLE_OFF, SVC_Y_FAR],  LINE_W),
    ([SINGLE_OFF, SVC_Y_NEAR], [COURT_W-SINGLE_OFF, SVC_Y_NEAR], LINE_W),
    # 中线（发球区中线）
    ([CTR_X, SVC_Y_FAR],      [CTR_X, SVC_Y_NEAR],    LINE_W),
    # 中心标志（底线中央，向场内延伸10cm，宽5cm）
    ([CTR_X, BASELINE_W],             [CTR_X, BASELINE_W+CENTER_MARK_L],    CENTER_MARK_W),
    ([CTR_X, COURT_L-BASELINE_W-CENTER_MARK_L], [CTR_X, COURT_L-BASELINE_W], CENTER_MARK_W),
]


class TemplateHomographyDetector:
    """
    接口与其他检测器相同：
        kps = detector.predict(frame)   # shape (28,)
        img = detector.draw_keypoints_on_video(frames, kps)
    """

    def __init__(self, scale: int = 40):
        """
        scale: 模板分辨率（像素/米），越大越精细但越慢
        """
        self.scale = scale
        # 预计算：模板中所有白线像素的米坐标
        self._template_img, self._template_pts_m = self._build_template(scale)

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
        print(f"[template] template white pixels: {len(pts_m)}, "
              f"size: {W}x{H} px @ {scale}px/m")
        return tmpl, pts_m.astype(np.float32)

    # ── 2. 主入口 ──────────────────────────────────────────────────
    def predict(self, frame):
        court_mask = self._get_court_mask(frame)
        dist_map   = self._build_dist_map(frame, court_mask)

        # 初始单应矩阵（角点检测）
        # 用 Hough 角点做初始 H（仅作参考起点，全局搜索不依赖它）
        H_init = self._get_hough_H(frame, court_mask, dist_map)

        # 全局搜索 + 精调
        H_opt = self._optimize(H_init, dist_map, frame.shape)

        self._last_H = H_opt   # 供调试用
        kps = self._project_keypoints(H_opt, frame.shape)
        return kps.flatten()

    # ── 3. 球场颜色分割 ────────────────────────────────────────────
    def _get_court_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ks  = max(3, int(frame.shape[0] * 0.023) | 1)  # ~2.3% 图像高度
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        broad = cv2.inRange(hsv, np.array([80, 15, 60]),
                                  np.array([135, 220, 230]))
        broad = cv2.morphologyEx(broad, cv2.MORPH_CLOSE, k)
        broad = cv2.morphologyEx(broad, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(broad, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return broad
        roi = np.zeros_like(broad)
        cv2.drawContours(roi, [max(cnts, key=cv2.contourArea)], -1, 255, -1)

        pixels = hsv[roi > 0]
        if len(pixels) < 1000:
            return roi
        lo, hi = [], []
        for ch in range(3):
            m, s = pixels[:, ch].mean(), pixels[:, ch].std()
            lo.append(max(0,   int(m - 2.0 * s)))
            hi.append(min(255, int(m + 2.0 * s)))
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [max(cnts, key=cv2.contourArea)],
                             -1, 255, -1)
            mask = clean
        return mask

    # ── 4. 构建距离场 ──────────────────────────────────────────────
    def _build_dist_map(self, frame, court_mask, white_thresh=160, cap=None):
        """
        对实际图像中的白线像素直接求距离变换（不做 Canny）。
        每个像素的值 = 到最近白线像素的距离（上限 cap）。
        模板像素落在白线内部或边缘均得到低代价，更鲁棒。
        """
        if cap is None:
            cap = frame.shape[0] * 0.046    # ~4.6% 图像高度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
        white_in = cv2.bitwise_and(white, court_mask)
        dist = cv2.distanceTransform(
                   cv2.bitwise_not(white_in), cv2.DIST_L2, 5)
        return np.minimum(dist, cap).astype(np.float32)

    # ── 5. 代价函数：模板白线像素到实际边缘的平均距离 ─────────────
    def _cost(self, H, dist_map):
        h_img, w_img = dist_map.shape
        margin = 200  # 允许少量越界

        # 检查投影关键点合法性
        kps_m = MODEL_KPS_M.reshape(-1, 1, 2)
        kps_proj = cv2.perspectiveTransform(kps_m, H).reshape(-1, 2)

        # 四个双打角必须严格在图像内（无额外容差）
        corners = kps_proj[:4]
        if not (np.all(corners[:, 0] >= 0) and np.all(corners[:, 0] < w_img) and
                np.all(corners[:, 1] >= 0) and np.all(corners[:, 1] < h_img)):
            return 1e6

        # 远端（y 小）必须在近端（y 大）上方
        p0, p1, p2, p3 = corners
        if (p0[1] + p1[1]) / 2 >= (p2[1] + p3[1]) / 2:
            return 1e6

        # 主代价：模板白线像素到最近边缘的平均距离
        pts = self._template_pts_m.reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        valid = ((proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) &
                 (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1))
        if valid.sum() < 100:
            return 1e6
        p  = proj[valid]
        xi = np.clip(p[:, 0].astype(np.int32), 0, w_img - 1)
        yi = np.clip(p[:, 1].astype(np.int32), 0, h_img - 1)
        return float(dist_map[yi, xi].mean())

    # ── 6. 优化：角点参数化 + 差分进化全局搜索 + Nelder-Mead 精调 ──
    def _optimize(self, H_hint, dist_map, img_shape):
        """
        把 4 个双打角的图像坐标作为优化变量（8个参数），
        在合理范围内用差分进化做全局搜索，再用 Nelder-Mead 精调。
        这样避免直接优化 H 的 9 个参数导致的局部最优问题。
        """
        h_img, w_img = dist_map.shape

        # 为加速差分进化，对模板点降采样
        idx = np.random.choice(len(self._template_pts_m),
                               min(3000, len(self._template_pts_m)),
                               replace=False)
        pts_sub = self._template_pts_m[idx].reshape(-1, 1, 2)

        def cost_sub(H):
            proj = cv2.perspectiveTransform(pts_sub, H).reshape(-1, 2)
            valid = ((proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) &
                     (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1))
            if valid.sum() < 50:
                return 1e6
            p  = proj[valid]
            xi = np.clip(p[:, 0].astype(np.int32), 0, w_img - 1)
            yi = np.clip(p[:, 1].astype(np.int32), 0, h_img - 1)
            return float(dist_map[yi, xi].mean())

        # 仅使用远端底线 + 近端底线的对应关系
        # （摄像机2米高，从近端拍摄，远近两条底线都可见）
        SRC_CASES = [
            np.array([[0,       0      ],
                      [COURT_W, 0      ],
                      [0,       COURT_L],
                      [COURT_W, COURT_L]], dtype=np.float32),
        ]

        # 搜索范围用图像尺寸的比例表达，自动适配任意分辨率
        def bx(r): return int(r * w_img)
        def by(r): return int(r * h_img)
        # [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
        BOUNDS = [
            (bx(0.18), bx(0.42)),   # tl_x  远-左
            (by(0.28), by(0.52)),   # tl_y  远端底线在上半部
            (bx(0.55), bx(0.86)),   # tr_x  远-右
            (by(0.28), by(0.52)),   # tr_y
            (0,        bx(0.21)),   # bl_x  近-左
            (by(0.69), by(0.91)),   # bl_y  近端底线靠近下边
            (bx(0.78), bx(1.00)),   # br_x  近-右
            (by(0.69), by(0.91)),   # br_y
        ]

        best_H, best_cost = H_hint, self._cost(H_hint, dist_map)
        print(f"[template] Hough initial cost: {best_cost:.3f}")

        for case_i, src_m in enumerate(SRC_CASES):
            def corners_to_H(params, _src=src_m):
                dst = np.array(params, dtype=np.float32).reshape(4, 2)
                H, _ = cv2.findHomography(_src, dst, method=0)
                return H

            def de_cost(params, _src=src_m):
                dst = np.array(params, dtype=np.float32).reshape(4, 2)
                tl, tr, bl, br = dst  # 顺序: 远左, 远右, 近左, 近右

                # 1. 所有角点在图像范围内
                margin = int(h_img * 0.046)   # ~4.6% 图像高度
                for px, py in [tl, tr, bl, br]:
                    if px < -margin or px > w_img + margin:
                        return 1e6
                    if py < -margin or py > h_img + margin:
                        return 1e6

                # 2. 拓扑正确：远端在近端上方，左边在右边左侧
                topo_gap = int(h_img * 0.046)   # ~4.6% 图像高度
                if tl[1] >= bl[1] - topo_gap or tr[1] >= br[1] - topo_gap:
                    return 1e6
                if tl[0] >= tr[0] - topo_gap or bl[0] >= br[0] - topo_gap:
                    return 1e6

                # 3. 用 shoelace 公式算四边形面积（按顺序 tl→tr→br→bl）
                poly = np.array([tl, tr, br, bl])
                n = len(poly)
                area = 0.5 * abs(sum(
                    poly[i][0] * poly[(i+1) % n][1] -
                    poly[(i+1) % n][0] * poly[i][1]
                    for i in range(n)
                ))
                if area < 0.05 * w_img * h_img:  # 至少占图像面积 5%
                    return 1e6

                H = corners_to_H(params, _src)
                if H is None:
                    return 1e6
                return cost_sub(H)

            result = differential_evolution(
                de_cost, BOUNDS,
                maxiter=600, popsize=15, tol=0.001,
                seed=42, workers=1, polish=False,
                mutation=(0.5, 1.5), recombination=0.7)

            H_de = corners_to_H(result.x)
            c_de = self._cost(H_de, dist_map)
            print(f"[template] case{case_i} diff-evolution: {c_de:.3f}  ({result.nit} iters)")

            if c_de < best_cost:
                best_cost, best_H = c_de, H_de

        # Nelder-Mead 精调
        cost_fn = lambda h: self._cost(h.reshape(3, 3), dist_map)
        r = minimize(cost_fn, best_H.flatten(), method='Nelder-Mead',
                     options={'maxiter': 8000, 'xatol': 0.1,
                              'fatol': 0.01, 'adaptive': True})
        c_final = cost_fn(r.x)
        print(f"[template] Nelder-Mead refine: {best_cost:.3f} -> {c_final:.3f}  ({r.nit} iters)")

        return (r.x if c_final < best_cost else best_H).reshape(3, 3)

    # ── 8. Hough 直线检测四角点（仅作初始参考）────────────────────
    def _get_hough_H(self, frame, court_mask, dist_map):
        corners = self._detect_corners(frame, court_mask)
        if corners is None:
            # 无法检测角点时，返回单位矩阵（差分进化会从 BOUNDS 里自行搜索）
            return np.eye(3, dtype=np.float32)
        src = np.array([[0, 0], [COURT_W, 0], [0, COURT_L], [COURT_W, COURT_L]],
                       dtype=np.float32)
        H, _ = cv2.findHomography(src, corners, method=0)
        return H if H is not None else np.eye(3, dtype=np.float32)

    def _detect_corners(self, frame, court_mask):
        h_img, w_img = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        white_in = cv2.bitwise_and(white, court_mask)
        edges_w  = cv2.Canny(cv2.dilate(white_in, np.ones((3,3)), 1), 30, 100)
        boundary = court_mask - cv2.erode(court_mask, np.ones((5,5),np.uint8), 4)
        edges    = cv2.bitwise_or(edges_w, cv2.Canny(boundary, 30, 100))

        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=60, minLineLength=60, maxLineGap=20)
        if lines is None:
            return None

        ys = np.where(court_mask.any(axis=1))[0]
        if len(ys) == 0:
            return None
        mask_top = int(ys.min())

        horiz, diag = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle  = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.hypot(x2 - x1, y2 - y1)
            if angle < 25:
                horiz.append(((y1+y2)/2, length, l[0]))
            elif angle > 40:
                diag.append(l[0])

        if len(horiz) < 2:
            return None

        horiz.sort(key=lambda x: x[0])
        all_ys = np.array([h[0] for h in horiz])
        split  = int(np.argmax(np.diff(all_ys))) + 1
        top_g, bot_g = horiz[:split], horiz[split:]
        if not top_g or not bot_g:
            return None

        _, _, top_seg = max(top_g,  key=lambda x: x[1])
        _, _, bot_seg = max(bot_g,  key=lambda x: x[1])
        top_y = (top_seg[1] + top_seg[3]) / 2
        bot_y = (bot_seg[1] + bot_seg[3]) / 2
        if bot_y - top_y < frame.shape[0] * 0.10:
            return None

        def seg_to_virtual(seg):
            x1, y1, x2, y2 = seg
            if abs(y2-y1) < 5: return None
            xa = x1 + (mask_top - y1)*(x2-x1)/(y2-y1)
            xb = x1 + (bot_y    - y1)*(x2-x1)/(y2-y1)
            return [int(xa), int(mask_top), int(xb), int(bot_y)]

        def mask_side(side):
            xs_list = []
            y_range = range(int(mask_top), int(bot_y), 4)
            for y in y_range:
                row = court_mask[y]
                pts = np.where(row > 0)[0]
                if len(pts) == 0: continue
                xs_list.append(pts.min() if side=='left' else pts.max())
            if len(xs_list) < 20: return None
            ys_a = np.array(list(y_range)[:len(xs_list)], np.float32)
            xs_a = np.array(xs_list, np.float32)
            c = np.polyfit(ys_a, xs_a, 1)
            y1_, y2_ = int(mask_top), int(bot_y)
            return [int(np.polyval(c, y1_)), y1_, int(np.polyval(c, y2_)), y2_]

        left_v  = [s for s in diag if (s[0]+s[2])/2 < w_img/2
                   and abs(s[3]-s[1]) > (bot_y-mask_top)*0.20]
        right_v = [s for s in diag if (s[0]+s[2])/2 > w_img/2
                   and abs(s[3]-s[1]) > (bot_y-mask_top)*0.20]

        left_line  = (seg_to_virtual(max(left_v,  key=lambda s:abs(s[3]-s[1])))
                      if left_v  else mask_side('left'))
        right_line = (seg_to_virtual(max(right_v, key=lambda s:abs(s[3]-s[1])))
                      if right_v else mask_side('right'))

        if left_line is None or right_line is None:
            return None

        def line_eq(seg):
            x1,y1,x2,y2 = seg
            a,b = y2-y1, x1-x2
            return np.array([a, b, x2*y1-x1*y2], dtype=float)

        def intersect(s1, s2):
            e1,e2 = line_eq(s1), line_eq(s2)
            c = np.cross(e1, e2)
            if abs(c[2]) < 1e-6: return None
            return np.array([c[0]/c[2], c[1]/c[2]])

        tl = intersect(top_seg, left_line)
        tr = intersect(top_seg, right_line)
        bl = intersect(bot_seg, left_line)
        br = intersect(bot_seg, right_line)
        if any(p is None for p in [tl,tr,bl,br]):
            return None
        return np.array([tl,tr,bl,br], dtype=np.float32)

    # ── 9. 投影关键点 & 可视化 ─────────────────────────────────────
    def _project_keypoints(self, H, img_shape):
        pts = MODEL_KPS_M.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def draw_keypoints_on_video(self, video_frames, keypoints):
        return [self.draw_keypoints(f, keypoints) for f in video_frames]

    def get_valid_zone_hull(self, image_shape, expand=1.5, height=7.0):
        """
        将 3D 三棱柱（帐篷形）投影到图像，返回凸包多边形（用于有效区域过滤）。
        三棱柱定义：
          底面 = 双打球场四角向外扩展 expand 米（z=0）
          脊线 = 底线中点（网位置）左右两端，高度 height 米
        返回 numpy int32 数组，形状 (N,1,2)，可直接传给 cv2.pointPolygonTest。
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

        f = minimize_scalar(cost, bounds=(w_img * 0.3, w_img * 20), method='bounded').x
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
        def v_of(p):
            cam = R @ p + t
            return K[1, 1] * cam[1] / cam[2] + K[1, 2]
        mid = np.array([COURT_W / 2, COURT_L / 2, 0.0])
        if v_of(mid + [0, 0, 1]) > v_of(mid):
            r3 = -r3
            R  = np.column_stack([r1, r2, r3])

        # 三棱柱 6 个顶点
        x0, x1 = -expand,       COURT_W + expand
        y0, y1 = -expand,       COURT_L + expand
        ym     = COURT_L / 2
        pts3d = np.array([
            [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
            [x0, ym, height], [x1, ym, height],
        ])

        P  = K @ np.hstack([R, t[:, None]])
        ph = np.hstack([pts3d, np.ones((6, 1))])
        uv = (P @ ph.T).T
        pts2d = (uv[:, :2] / uv[:, 2:]).astype(np.float32)

        hull = cv2.convexHull(pts2d.reshape(-1, 1, 2))
        return hull.astype(np.int32)

    def draw_frame(self, frame, alpha=0.8):
        """原地把球场线条叠加到 frame 上（与 tracker.draw_bboxes_frame 接口一致）。
        alpha: 线条不透明度（0=完全透明，1=完全覆盖）
        """
        if not hasattr(self, '_court_overlay') or self._court_overlay.shape != frame.shape:
            self._court_overlay = self.draw_court(np.zeros_like(frame))
            self._court_mask = self._court_overlay.any(axis=2)  # 线条区域 bool mask
        frame[self._court_mask] = cv2.addWeighted(
            frame, 1 - alpha, self._court_overlay, alpha, 0
        )[self._court_mask]

    def draw_court(self, image, color=(0, 200, 0)):
        """用 H 把 COURT_LINES 每条线的实际宽度矩形投影到图像上。"""
        if not hasattr(self, '_last_H'):
            return image.copy()
        img = image.copy()
        H = self._last_H
        for (p1, p2, w) in COURT_LINES:
            x1, y1 = p1
            x2, y2 = p2
            hw = w / 2
            # 所有线条均为水平或垂直，按方向计算四角
            if abs(x2 - x1) > abs(y2 - y1):   # 水平线
                corners = np.array([[x1, y1 - hw], [x2, y2 - hw],
                                    [x2, y2 + hw], [x1, y1 + hw]], dtype=np.float32)
            else:                               # 垂直线
                corners = np.array([[x1 - hw, y1], [x2 - hw, y2],
                                    [x2 + hw, y2], [x1 + hw, y1]], dtype=np.float32)
            proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
            pts = proj.astype(np.int32)
            cv2.fillPoly(img, [pts], color)
        return img

    def draw_keypoints(self, image, keypoints, color=(0, 255, 0)):
        img = image.copy()
        # Draw court lines (1px)
        line_pairs = [
            (0, 2), (1, 3),           # doubles sidelines
            (4, 5), (6, 7),           # singles sidelines
            (0, 1), (2, 3),           # baselines
            (8, 9), (10, 11),         # service lines
            (12, 13),                 # center service line
        ]
        for p1, p2 in line_pairs:
            x1, y1 = int(keypoints[p1*2]), int(keypoints[p1*2+1])
            x2, y2 = int(keypoints[p2*2]), int(keypoints[p2*2+1])
            cv2.line(img, (x1, y1), (x2, y2), color, 1)
        # Draw cross markers at each keypoint (1px)
        s = 6
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            cv2.line(img, (x-s, y), (x+s, y), color, 1)
            cv2.line(img, (x, y-s), (x, y+s), color, 1)
        return img

    # ── 10. 调试：保存可视化图 ────────────────────────────────────
    def debug_overlay(self, frame, H, path="output_videos/template_debug.jpg"):
        h_img, w_img = frame.shape[:2]
        vis = frame.copy()

        # 把模板白线投影到图像上
        pts = self._template_pts_m.reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        valid = ((proj[:,0] >= 0) & (proj[:,0] < w_img) &
                 (proj[:,1] >= 0) & (proj[:,1] < h_img))
        for x, y in proj[valid]:
            vis[int(y), int(x)] = (0, 255, 0)

        cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[template] 调试图: {path}")
        return vis
