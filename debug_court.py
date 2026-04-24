"""
可视化球场检测每一步的中间结果。

用法：
    python debug_court.py -i <video>
输出：
    <video>_debug/  目录，包含各步骤可视化图
"""

import argparse
import os
import sys

import cv2
import numpy as np

from court_detector import (CourtDetector, MODEL_KPS_M, COURT_LINES,
                             COURT_W, COURT_L, NET_Y,
                             CLEARANCE_BACK, CLEARANCE_SIDE)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--input',  required=True, help='输入视频路径')
    p.add_argument('-s', '--court-model', default='models/yolo26n-seg-court.pt', help='球场分割模型路径')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def draw_volume_wireframe(vis, pts_bot, pts_top, color):
    """绘制立方体线框：底面 + 顶面 + 四条竖边。"""
    n = len(pts_bot)
    for i in range(n):
        cv2.line(vis, tuple(pts_bot[i].astype(int)), tuple(pts_bot[(i+1)%n].astype(int)), color, 1)
        cv2.line(vis, tuple(pts_top[i].astype(int)), tuple(pts_top[(i+1)%n].astype(int)), color, 1)
        cv2.line(vis, tuple(pts_bot[i].astype(int)), tuple(pts_top[i].astype(int)),        color, 1)


def min_circumscribed_quad(approx_pts):
    """从 approxPolyDP 多边形的边中，枚举所有 C(n,4) 组合，
    找能包住所有顶点的面积最小四边形。
    返回 (4,2) float32 角点，失败返回 None。
    """
    from itertools import combinations
    pts = approx_pts.reshape(-1, 2).astype(np.float64)
    n   = len(pts)
    if n == 4:
        return pts.astype(np.float32)

    def edge_line(i):
        """第 i 条边延长线：(a, b, c) 满足 a*x + b*y = c。"""
        p1, p2 = pts[i], pts[(i + 1) % n]
        dx, dy = p2 - p1
        a, b = -dy, dx
        return a, b, a * p1[0] + b * p1[1]

    def intersect(l1, l2):
        a1, b1, c1 = l1;  a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        return np.array([(c1*b2 - c2*b1) / det,
                         (a1*c2 - a2*c1) / det])

    best_area    = float('inf')
    best_corners = None

    for idx in combinations(range(n), 4):
        lines = [edge_line(i) for i in idx]

        # 按法线角度排序，保证四边形循环顺序
        order = np.argsort([np.arctan2(l[1], l[0]) for l in lines])
        lines = [lines[o] for o in order]

        # 求相邻边交点
        corners = [intersect(lines[k], lines[(k + 1) % 4]) for k in range(4)]
        if any(c is None for c in corners):
            continue
        corners = np.array(corners)

        # 判断多边形所有点是否在四边形内
        # 以四边形重心为内部参考点确定每条线的"内侧"方向
        ctr = corners.mean(axis=0)
        ok  = True
        for a, b, c in lines:
            inside_sign = np.sign(a * ctr[0] + b * ctr[1] - c)
            if any(inside_sign * (a * p[0] + b * p[1] - c) < -1e-4 for p in pts):
                ok = False
                break
        if not ok:
            continue

        area = 0.5 * abs(sum(
            corners[k, 0] * corners[(k+1)%4, 1] - corners[(k+1)%4, 0] * corners[k, 1]
            for k in range(4)
        ))
        if area < best_area:
            best_area    = area
            best_corners = corners

    return best_corners.astype(np.float32) if best_corners is not None else None


def save(path, img, label):
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  → {path}  [{label}]")


def draw_court_lines(frame, H, color=(0, 220, 100), thickness=1):
    """把球场线条（含线宽两侧边缘）投影到 frame 上。"""
    vis = frame.copy()
    for (p1, p2, lw_m) in COURT_LINES:
        half = lw_m / 2
        if p1[1] == p2[1]:   # 水平线
            for dy in (-half, +half):
                pts = np.array([[[p1[0], p1[1]+dy], [p2[0], p2[1]+dy]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pts, H)[0].astype(int)
                cv2.line(vis, tuple(proj[0]), tuple(proj[1]), color, thickness)
        else:                 # 垂直线
            for dx in (-half, +half):
                pts = np.array([[[p1[0]+dx, p1[1]], [p2[0]+dx, p2[1]]], ], dtype=np.float32)
                proj = cv2.perspectiveTransform(pts, H)[0].astype(int)
                cv2.line(vis, tuple(proj[0]), tuple(proj[1]), color, thickness)
    # 网线
    net = np.array([[[0, NET_Y]], [[COURT_W, NET_Y]]], dtype=np.float32)
    net_px = cv2.perspectiveTransform(net, H)
    cv2.line(vis, tuple(net_px[0,0].astype(int)), tuple(net_px[1,0].astype(int)), color, thickness)
    return vis


def main():
    args = parse_args()

    # ── 读第一帧 ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("无法读取视频")
        sys.exit(1)

    out_dir = os.path.splitext(args.input)[0] + '_debug'
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n输出目录: {out_dir}\n")

    # ── 步骤 0：原始第一帧 ──────────────────────────────────────────
    save(f"{out_dir}/0_original.jpg", frame, "原始第一帧")

    # ── 步骤 1：球场颜色分割 ────────────────────────────────────────
    detector = CourtDetector(seg_model=args.court_model)
    court_mask = detector._get_court_mask(frame)

    vis_mask = frame.copy()
    vis_mask[court_mask == 0] = (vis_mask[court_mask == 0] * 0.35).astype(np.uint8)
    cv2.polylines(vis_mask,
                  [cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                   if cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else np.array([])],
                  True, (0, 255, 255), 2)
    save(f"{out_dir}/1_court_mask.jpg", vis_mask, "球场颜色分割（场外变暗）")

    # ── 步骤 2：全帧白色像素（顶帽，不依赖 YOLO mask）──────────────
    full_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    white_in = detector._detect_white_pixels(frame, full_mask)

    vis_white = frame.copy()
    vis_white[white_in > 0] = (0, 200, 255)   # 橙色标出白线像素
    n_white = int((white_in > 0).sum())
    n_total = int(full_mask.size)
    pct = 100 * n_white / n_total
    print(f"  全帧白线像素: {n_white}/{n_total} = {pct:.1f}%")
    save(f"{out_dir}/2_white_pixels.jpg", vis_white, "全帧白色像素（橙色）")

    # ── 步骤 3：距离场 ──────────────────────────────────────────────
    dist_map = detector._build_dist_map(frame, full_mask)
    cap_val   = float(dist_map.max())
    dist_vis  = (dist_map / cap_val * 255).astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_INFERNO)
    save(f"{out_dir}/3_dist_map.jpg", dist_color, "距离场（亮=远离白线，暗=靠近白线）")

    # ── 步骤 4：YOLO seg 初始化 ─────────────────────────────────────
    H_seg, c_seg = detector._yolo_seg_init(frame, dist_map)
    print(f"  YOLO seg init cost: {c_seg:.3f}")

    if H_seg is not None:
        vis_seg = draw_court_lines(frame, H_seg, color=(0, 100, 255), thickness=1)
        # 画4个初始角点
        corners_m = np.array([[[0,0],[COURT_W,0],[COURT_W,COURT_L],[0,COURT_L]]], dtype=np.float32).reshape(-1,1,2)
        corners_px = cv2.perspectiveTransform(corners_m, H_seg).reshape(-1,2).astype(int)
        for pt in corners_px:
            cv2.circle(vis_seg, tuple(pt), 8, (0, 0, 255), -1)
        cv2.putText(vis_seg, f"seg init cost={c_seg:.3f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
        save(f"{out_dir}/4_seg_init.jpg", vis_seg, f"YOLO seg 初始 H（cost={c_seg:.3f}）")
    else:
        print("  YOLO seg 初始化失败，跳过步骤 4")

    # ── 步骤 4b：凸包多边形近似 + 最优四边形 + 外接四边形 ──────────
    seg_results = detector._run_seg(frame)
    if seg_results and seg_results[0].masks is not None and len(seg_results[0].masks) > 0:
        best = int(seg_results[0].boxes.conf.argmax())
        poly = seg_results[0].masks.xy[best].astype(np.float32)

        hull   = cv2.convexHull(poly.astype(np.int32))
        eps    = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(np.float32)
        quad   = approx if len(approx) == 4 else detector._best_quad(approx)
        quad_sorted = detector._sort_quad(quad)

        # 最小面积外接四边形（包住 approx 的所有顶点）
        min_circ = min_circumscribed_quad(approx)

        vis_quad = frame.copy()

        # seg mask（半透明）
        mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_img, [poly.astype(np.int32)], 255)
        vis_quad[mask_img > 0] = (vis_quad[mask_img > 0] * 0.5
                                  + np.array([100, 60, 0]) * 0.5).astype(np.uint8)

        # 凸包近似多边形（黄色）
        for pt in approx.astype(int):
            cv2.circle(vis_quad, tuple(pt), 5, (0, 220, 255), -1)
        cv2.polylines(vis_quad, [approx.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 220, 255), 1)

        # 最优四边形（红色）
        cv2.polylines(vis_quad, [quad_sorted.astype(np.int32).reshape(-1, 1, 2)],
                      True, (0, 0, 255), 2)
        for i, pt in enumerate(quad_sorted.astype(int)):
            cv2.circle(vis_quad, tuple(pt), 8, (0, 0, 255), -1)

        # 最小面积外接四边形（绿色大圆 + 编号）
        if min_circ is not None:
            circ = detector._sort_quad(min_circ)
            cv2.polylines(vis_quad, [circ.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 255, 80), 2)
            for i, pt in enumerate(circ.astype(int)):
                cv2.circle(vis_quad, tuple(pt), 10, (0, 255, 80), -1)
                cv2.putText(vis_quad, str(i), (pt[0] + 12, pt[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)

        cv2.putText(vis_quad,
                    f"approx {len(approx)}pts  best-quad(red)  min-circ(green)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
        save(f"{out_dir}/4b_best_quad.jpg", vis_quad,
             f"approx {len(approx)} 顶点（黄）→ best quad（红）→ 最小外接四边形（绿）")
    else:
        print("  seg 结果不可用，跳过步骤 4b")

    # ── 步骤 5：优化后结果 ──────────────────────────────────────────
    H_opt = detector._optimize(H_seg, dist_map, frame.shape)
    c_opt = detector._cost(H_opt, dist_map)
    print(f"  optimized cost: {c_opt:.3f}")

    vis_opt = draw_court_lines(frame, H_opt, color=(80, 200, 255), thickness=1)
    kps = cv2.perspectiveTransform(MODEL_KPS_M.reshape(-1,1,2), H_opt).reshape(-1,2).astype(int)
    for pt in kps:
        cv2.circle(vis_opt, tuple(pt), 4, (80, 200, 255), -1)
    cv2.putText(vis_opt, f"optimized cost={c_opt:.3f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 200, 255), 2)
    save(f"{out_dir}/5_optimized.jpg", vis_opt, f"优化后 H（cost={c_opt:.3f}）")

    # ── 步骤 6：模板像素投影（散点图）──────────────────────────────
    vis_tmpl = frame.copy()
    pts = detector._template_pts_m.reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H_opt).reshape(-1, 2)
    h_img, w_img = frame.shape[:2]
    valid = ((proj[:,0] >= 0) & (proj[:,0] < w_img) &
             (proj[:,1] >= 0) & (proj[:,1] < h_img))
    for x, y in proj[valid].astype(int):
        vis_tmpl[y, x] = (0, 255, 0)
    save(f"{out_dir}/6_template_proj.jpg", vis_tmpl, "模板白线像素投影（绿点）")


    # ── 步骤 10：ITF 标准缓冲区可视化 ────────────────────────────
    # 缓冲区：底线后 CLEARANCE_BACK m，侧线外 CLEARANCE_SIDE m
    buf_m = np.array([
        [-CLEARANCE_SIDE,          -CLEARANCE_BACK          ],
        [COURT_W + CLEARANCE_SIDE, -CLEARANCE_BACK          ],
        [COURT_W + CLEARANCE_SIDE,  COURT_L + CLEARANCE_BACK],
        [-CLEARANCE_SIDE,           COURT_L + CLEARANCE_BACK],
    ], dtype=np.float32)
    buf_px = cv2.perspectiveTransform(buf_m.reshape(-1, 1, 2), H_opt).reshape(-1, 1, 2).astype(np.int32)

    vis_buf = draw_court_lines(frame, H_opt, color=(0, 255, 128), thickness=1)
    # 缓冲区外变暗
    buf_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(buf_mask, [buf_px], 255)
    vis_buf[buf_mask == 0] = (vis_buf[buf_mask == 0] * 0.35).astype(np.uint8)
    # 缓冲区边界（黄色）
    cv2.polylines(vis_buf, [buf_px], True, (0, 220, 255), 2)
    cv2.putText(vis_buf, f"clearance: back={CLEARANCE_BACK}m  side={CLEARANCE_SIDE}m",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2)
    save(f"{out_dir}/10_clearance_zone.jpg", vis_buf,
         f"ITF 缓冲区（后侧{CLEARANCE_BACK}m，边线{CLEARANCE_SIDE}m）")

    # ── 步骤 11：私人/俱乐部场地缩减缓冲区可视化 ─────────────────
    cb_back, cb_side = 5.5, 3.0
    buf2_m = np.array([
        [-cb_side,          -cb_back          ],
        [COURT_W + cb_side, -cb_back          ],
        [COURT_W + cb_side,  COURT_L + cb_back],
        [-cb_side,           COURT_L + cb_back],
    ], dtype=np.float32)
    buf2_px = cv2.perspectiveTransform(buf2_m.reshape(-1, 1, 2), H_opt).reshape(-1, 1, 2).astype(np.int32)

    vis_buf2 = draw_court_lines(frame, H_opt, color=(0, 255, 128), thickness=1)
    buf2_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(buf2_mask, [buf2_px], 255)
    vis_buf2[buf2_mask == 0] = (vis_buf2[buf2_mask == 0] * 0.35).astype(np.uint8)
    cv2.polylines(vis_buf2, [buf2_px], True, (0, 180, 255), 2)
    cv2.putText(vis_buf2, f"clearance: back={cb_back}m  side={cb_side}m",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 255), 2)
    save(f"{out_dir}/11_clearance_zone_small.jpg", vis_buf2,
         f"缩减缓冲区（后侧{cb_back}m，边线{cb_side}m）")

    # ── 步骤 12：缓冲区立方体凸包（两种尺寸对比）────────────────────
    def draw_clearance_volume(frame, H_opt, back, side, color, label):
        detector._last_H = H_opt
        hull, pts_bot, pts_top = detector.get_clearance_volume_hull(
            frame.shape, back=back, side=side, height=2.0)
        vis = draw_court_lines(frame, H_opt, color=(0, 255, 128), thickness=1)
        # 凸包外变暗
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hull_i32 = hull.reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [hull_i32], 255)
        vis[mask == 0] = (vis[mask == 0] * 0.35).astype(np.uint8)
        # 凸包轮廓
        cv2.polylines(vis, [hull_i32], True, color, 2)
        # 立方体线框
        draw_volume_wireframe(vis, pts_bot, pts_top, color)
        cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        return vis

    vis_v1 = draw_clearance_volume(
        frame, H_opt,
        back=CLEARANCE_BACK, side=CLEARANCE_SIDE,
        color=(0, 220, 255),
        label=f"ITF 标准: back={CLEARANCE_BACK}m side={CLEARANCE_SIDE}m h=2m")
    save(f"{out_dir}/12a_volume_itf.jpg", vis_v1, "ITF 缓冲区立方体凸包")

    vis_v2 = draw_clearance_volume(
        frame, H_opt,
        back=5.5, side=3.0,
        color=(0, 140, 255),
        label="俱乐部: back=5.5m side=3.0m h=2m")
    save(f"{out_dir}/12b_volume_small.jpg", vis_v2, "俱乐部缓冲区立方体凸包")

    print("\n完成。")


if __name__ == '__main__':
    main()
