"""
调试脚本：从单应性矩阵估算相机参数，在视频帧上绘制有效球场 3D 区域。
有效区域：双打线和底线向外扩展 2.75m，高度 12.5m。
"""
import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from utils import read_video
from court_line_detector import TemplateHomographyDetector

# 球场参数（ITF 标准，米）
COURT_W = 10.97
COURT_L = 23.77
EXPAND  = 1.5    # 四边向外扩展
HEIGHT  = 7.0    # 有效高度


def estimate_camera(H, image_shape):
    """
    从地面单应性矩阵 H（米→像素）估算相机内参 K 和外参 R, t。
    假设等焦距（fx=fy=f），主点在图像中心。
    """
    h_img, w_img = image_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    h1, h2 = H[:, 0], H[:, 1]

    def cost(f):
        Ki = np.array([[1/f, 0, -cx/f], [0, 1/f, -cy/f], [0, 0, 1.0]])
        r1 = Ki @ h1
        r2 = Ki @ h2
        # 约束：r1·r2=0，|r1|=|r2|
        return (r1 @ r2) ** 2 + (r1 @ r1 - r2 @ r2) ** 2

    f = minimize_scalar(cost, bounds=(w_img * 0.3, w_img * 20), method='bounded').x
    K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)

    lam = 1.0 / np.linalg.norm(Ki @ h1)
    r1  = lam * (Ki @ H[:, 0])
    r2  = lam * (Ki @ H[:, 1])
    r3  = np.cross(r1, r2)
    t   = lam * (Ki @ H[:, 2])

    # 确保 t[2] > 0（场地在相机前方）
    if t[2] < 0:
        r1, r2, r3, t = -r1, -r2, -r3, -t

    R = np.column_stack([r1, r2, r3])

    # 验证 z 方向：高处的点应在图像中更高（v 更小）
    # 取球场中心地面点和 1m 高点对比
    cx_world = np.array([COURT_W / 2, COURT_L / 2, 0.0])
    cu_world = np.array([COURT_W / 2, COURT_L / 2, 1.0])
    def v_proj(p):
        cam = R @ p + t
        return K[1, 1] * cam[1] / cam[2] + K[1, 2]
    if v_proj(cu_world) > v_proj(cx_world):   # 高处反而在图像下方 → 翻转 z
        r3 = -r3
        R  = np.column_stack([r1, r2, r3])

    return K, R, t


def project(pts3d, K, R, t):
    """3D 世界点（米）投影到图像像素坐标。"""
    P = K @ np.hstack([R, t[:, None]])           # 3×4
    ph = np.hstack([pts3d, np.ones((len(pts3d), 1))])  # Nx4
    uv = (P @ ph.T).T                             # Nx3
    return (uv[:, :2] / uv[:, 2:]).astype(np.float32)


def draw_box_edges(frame, bp, tp, color_bottom, color_top, color_vert, thickness=3):
    """绘制 3D box 的底面、顶面和竖边。"""
    bp = bp.reshape(-1, 1, 2).astype(np.int32)
    tp = tp.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(frame, [bp], True, color_bottom, thickness)
    cv2.polylines(frame, [tp], True, color_top,    thickness)
    for i in range(4):
        cv2.line(frame, tuple(bp[i, 0]), tuple(tp[i, 0]), color_vert, thickness)


if __name__ == "__main__":
    frames, _ = read_video("input_videos/input_video.mp4")
    frame = frames[0].copy()

    det = TemplateHomographyDetector()
    det.predict(frame)
    H = det._last_H

    K, R, t = estimate_camera(H, frame.shape)
    print(f"estimated f = {K[0, 0]:.1f} px  (image {frame.shape[1]}×{frame.shape[0]})")
    print(f"t = {t.round(2)}")

    # 3D 三棱柱顶点（球场坐标系，米）
    # 底面 4 角（z=0），脊线 2 点（y=净中央，z=HEIGHT）
    x0, x1 = -EXPAND,       COURT_W + EXPAND
    y0, y1 = -EXPAND,       COURT_L + EXPAND
    ym     = COURT_L / 2    # 底线中点（网的位置）

    base = np.array([[x0, y0, 0], [x1, y0, 0],
                     [x1, y1, 0], [x0, y1, 0]])          # 底面 4 点
    ridge = np.array([[x0, ym, HEIGHT], [x1, ym, HEIGHT]])  # 脊线 2 点

    bp = project(base,  K, R, t)   # 4×2
    rp = project(ridge, K, R, t)   # 2×2

    # 6 个投影点的凸包 = 有效球场区域
    all_pts = np.vstack([bp, rp]).reshape(-1, 1, 2).astype(np.int32)
    hull = cv2.convexHull(all_pts)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [hull], (0, 255, 255))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [hull], True, (0, 255, 255), 3)

    # 绘制棱线
    bp_i = bp.astype(np.int32)
    rp_i = rp.astype(np.int32)
    # 底面矩形
    for a, b in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.line(frame, tuple(bp_i[a]), tuple(bp_i[b]), (0, 255, 0), 2)
    # 脊线
    cv2.line(frame, tuple(rp_i[0]), tuple(rp_i[1]), (0, 220, 255), 2)
    # 底角到脊线
    for bi, ri in [(0,0),(3,0),(1,1),(2,1)]:
        cv2.line(frame, tuple(bp_i[bi]), tuple(rp_i[ri]), (0, 140, 255), 2)

    # 也绘制原始球场凸包（地面，红色参考）
    kps = det.predict(frames[0])
    pts = np.array([(kps[i*2], kps[i*2+1]) for i in range(len(kps)//2)], dtype=np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [hull], True, (0, 0, 255), 2)

    # 图例
    h = frame.shape[0]
    s = max(1, h // 1080)
    fs, ft = 0.7 * s, 2 * s
    cv2.putText(frame, "Yellow: valid zone (tent projected)",  (20, h - 210*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), ft)
    cv2.putText(frame, "Green:  ground rectangle",            (20, h - 160*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0),   ft)
    cv2.putText(frame, "Cyan:   ridge (net center, 8m)",      (20, h - 110*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 220, 255), ft)
    cv2.putText(frame, "Orange: slopes to ridge",             (20, h -  60*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 140, 255), ft)
    cv2.putText(frame, "Red:    court hull (ref)",            (20, h -  10*s), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255),   ft)

    out_path = "output_videos/debug_3d_valid_zone.jpg"
    cv2.imwrite(out_path, frame)
    print(f"saved → {out_path}")
