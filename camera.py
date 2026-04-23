"""
从地面单应矩阵 H 恢复相机矩阵 P，并拟合网球三维轨迹（抛物线模型）。

坐标系：ITF 球场坐标（米）
  X : 横向（0 → COURT_W = 10.97 m）
  Y : 纵向（0 → COURT_L = 23.77 m）
  Z : 高度（0 = 地面）

数学原理
--------
单应矩阵 H 将地面平面（Z=0）映射到图像：
  H = K [r₁ | r₂ | t]  （差一个公比）

假设方形像素（fx=fy=f）、主点在图像中心，利用两个约束求 f：
  1. r₁ ⊥ r₂  →  (K⁻¹ h₁) · (K⁻¹ h₂) = 0
  2. |r₁| = |r₂|  →  |K⁻¹ h₁|² = |K⁻¹ h₂|²

二者各自给出 f² 的线性方程，取均值后得焦距 f。
再由 r₃ = r₁ × r₂ 补全旋转矩阵，构造 3×4 投影矩阵 P。

3D 轨迹拟合
-----------
网球在空中做抛体运动（忽略空气阻力）：
  X(t) = X₀ + vx·t
  Y(t) = Y₀ + vy·t
  Z(t) = Z₀ + vz·t - ½·g·t²

对 N 帧图像观测，最小化重投影误差（像素），用 scipy least_squares 求解
6 个参数 (X₀, Y₀, Z₀, vx, vy, vz)。
"""

import warnings
import numpy as np
from scipy.optimize import least_squares


# ── 重力常数 ──────────────────────────────────────────────────────────────────
_G = 9.81   # m/s²

# ── ITF 球场尺寸 ───────────────────────────────────────────────────────────────
COURT_W = 10.97   # 宽（米），双打线
COURT_L = 23.77   # 长（米）
_COURT_MARGIN = 6.0  # 地面投影超出此裕量则认为是噪声段，跳过拟合


# ── 相机恢复 ──────────────────────────────────────────────────────────────────

def recover_camera_from_H(H, img_w, img_h):
    """
    从地面单应矩阵 H（球场米坐标 → 图像像素）恢复相机矩阵。

    假设：方形像素（fx = fy = f）、主点在图像中心、无畸变。

    Parameters
    ----------
    H      : (3,3) 单应矩阵，compute_H_from_kps 的输出
    img_w  : int  图像宽度（像素）
    img_h  : int  图像高度（像素）

    Returns
    -------
    K  : (3,3) 相机内参矩阵  [[f,0,cx],[0,f,cy],[0,0,1]]
    P  : (3,4) 投影矩阵，将 [X,Y,Z,1] 映射到图像像素 [u,v,w]
    R  : (3,3) 旋转矩阵（球场 → 相机）
    t  : (3,)  平移向量（球场 → 相机）
    """
    H = np.asarray(H, dtype=np.float64)
    cx, cy = img_w / 2.0, img_h / 2.0

    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    # K⁻¹ h = [(h₀ - cx·h₂)/f,  (h₁ - cy·h₂)/f,  h₂]
    # 令 aᵢₓ = hᵢ[0] - cx·hᵢ[2]，aᵢᵧ = hᵢ[1] - cy·hᵢ[2]，aᵢᵤ = hᵢ[2]
    a1x = h1[0] - cx * h1[2];  a1y = h1[1] - cy * h1[2];  a1z = h1[2]
    a2x = h2[0] - cx * h2[2];  a2y = h2[1] - cy * h2[2];  a2z = h2[2]

    # 约束 1：r₁ · r₂ = 0
    #   (a1x·a2x + a1y·a2y)/f² + a1z·a2z = 0
    #   → f² = -(a1x·a2x + a1y·a2y) / (a1z·a2z)
    num1 = -(a1x * a2x + a1y * a2y)
    den1 = a1z * a2z
    f2_ortho = (num1 / den1) if abs(den1) > 1e-12 else None

    # 约束 2：|r₁|² = |r₂|²
    #   (a1x²+a1y²)/f² + a1z² = (a2x²+a2y²)/f² + a2z²
    #   → f² = (a1x²+a1y² − a2x²−a2y²) / (a2z² − a1z²)
    num2 = (a1x**2 + a1y**2) - (a2x**2 + a2y**2)
    den2 = a2z**2 - a1z**2
    f2_equal = (num2 / den2) if abs(den2) > 1e-12 else None

    candidates = [v for v in (f2_ortho, f2_equal) if v is not None and v > 0]
    if candidates:
        f = float(np.sqrt(np.mean(candidates)))
    else:
        # 退化情况（极少见）：用图像对角线作为粗略估计
        f = float(np.sqrt(img_w**2 + img_h**2))
        warnings.warn(
            f"[camera] 焦距估计退化（f2_ortho={f2_ortho}, f2_equal={f2_equal}），"
            f"使用启发式 f={f:.0f} px"
        )

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], dtype=np.float64)
    K_inv = np.array([[1/f,   0, -cx/f],
                      [  0, 1/f, -cy/f],
                      [  0,   0,     1]], dtype=np.float64)

    # H = K [r₁ | r₂ | t]，归一化使 |r₁| = 1
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    t  = lam * (K_inv @ h3)
    r3 = np.cross(r1, r2)

    # 注意：不做 SVD 正交化，保持 P[:,{0,1,3}] = λH 的等价关系，
    # 确保地面（Z=0）投影与 H 完全一致。
    # r3 = r1 × r2 已是单位向量（r1、r2 单位正交），只做符号修正。
    if np.linalg.det(np.column_stack([r1, r2, r3])) < 0:
        r3 = -r3

    R = np.column_stack([r1, r2, r3])
    P = K @ np.column_stack([R, t])
    return K, P, R, t


# ── 3D 投影 ───────────────────────────────────────────────────────────────────

def project_3d(P, xyz):
    """
    将三维球场坐标投影到图像。

    Parameters
    ----------
    P   : (3,4) 相机投影矩阵
    xyz : (N,3) 或 (3,) — [X, Y, Z] 球场坐标（米）

    Returns
    -------
    (N,2) 或 (2,) — 图像坐标 [u, v]（像素）
    """
    P = np.asarray(P, dtype=np.float64)
    single = np.ndim(xyz) == 1
    pts = np.atleast_2d(xyz).astype(np.float64)
    ones = np.ones((len(pts), 1))
    proj = (P @ np.hstack([pts, ones]).T).T   # (N, 3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv[0] if single else uv


# ── 单段抛物线拟合 ────────────────────────────────────────────────────────────

def fit_parabola_3d(obs_uv, obs_frames, fps, P, H_inv=None, g=_G):
    """
    对一段网球轨迹拟合三维抛体运动模型。

    最小化图像重投影误差（像素）。

    Parameters
    ----------
    obs_uv     : (N,2) 观测到的图像球心（像素）
    obs_frames : (N,)  对应帧号
    fps        : float 帧率
    P          : (3,4) 相机投影矩阵
    H_inv      : (3,3) H 的逆矩阵（图像 → 球场地面），用于初值；None 则从 P 推导
    g          : float 重力加速度（m/s²）

    Returns
    -------
    params     : (6,) [X₀, Y₀, Z₀, vx, vy, vz]
    t_arr      : (N,) 时间数组（秒，t₀=0）
    xyz        : (N,3) 每帧三维坐标（米）
    reproj_err : float 平均重投影误差（像素）
    success    : bool  优化是否收敛
    """
    obs_uv = np.asarray(obs_uv, dtype=np.float64)
    obs_frames = np.asarray(obs_frames, dtype=np.float64)
    t_arr = (obs_frames - obs_frames[0]) / fps

    def xyz_at(params, t):
        X0, Y0, Z0, vx, vy, vz = params
        return np.column_stack([
            X0 + vx * t,
            Y0 + vy * t,
            np.maximum(0.0, Z0 + vz * t - 0.5 * g * t**2),
        ])

    def residuals(params):
        uv_proj = project_3d(P, xyz_at(params, t_arr))
        return (uv_proj - obs_uv).ravel()

    # ── 初值估计 ──────────────────────────────────────────────────────────────
    # 用 H 逆矩阵把第一帧/最后帧映射到地面，估计水平速度
    if H_inv is not None:
        def to_ground(uv_pt):
            pt = H_inv @ np.array([uv_pt[0], uv_pt[1], 1.0])
            return pt[:2] / pt[2]
    else:
        # 从 P 的地面列（Z=0）导出：H_approx = P[:, [0,1,3]]
        Pg_inv = np.linalg.inv(P[:, [0, 1, 3]])
        def to_ground(uv_pt):
            pt = Pg_inv @ np.array([uv_pt[0], uv_pt[1], 1.0])
            return pt[:2] / pt[2]

    xy0 = to_ground(obs_uv[0])
    xy1 = to_ground(obs_uv[-1])
    dt  = float(t_arr[-1]) if t_arr[-1] > 0 else 1.0

    vx0 = float(np.clip((xy1[0] - xy0[0]) / dt, -80, 80))
    vy0 = float(np.clip((xy1[1] - xy0[1]) / dt, -80, 80))
    vz0 = float(np.clip(0.5 * g * dt, 0.0, 30.0))

    x0 = [float(xy0[0]), float(xy0[1]), 1.0, vx0, vy0, vz0]

    # ── 物理约束 bound ────────────────────────────────────────────────────────
    # 球不可能在地下（Z≥0）、也不可能上升到 12m 以上；速度限制在物理范围内
    # 同时约束 X/Y 在球场加宽裕范围内，防止尺度退化（单目相机固有歧义）
    margin_xy = 20.0
    lo = [xy0[0] - margin_xy, xy0[1] - margin_xy,  0.0, -80, -80, -30]
    hi = [xy0[0] + margin_xy, xy0[1] + margin_xy, 12.0,  80,  80,  30]

    result = least_squares(residuals, x0, method='trf', bounds=(lo, hi),
                           max_nfev=2000, ftol=1e-6, xtol=1e-6)

    params  = result.x
    xyz     = xyz_at(params, t_arr)
    uv_proj = project_3d(P, xyz)
    err     = float(np.mean(np.linalg.norm(uv_proj - obs_uv, axis=1)))

    return params, t_arr, xyz, err, bool(result.success)


# ── 轨迹分段 ─────────────────────────────────────────────────────────────────

def _segment_by_bounce(frames, uv, fps=30.0, min_seg_frames=6):
    """
    将轨迹分段：先在帧间隙 > 0.25s 处切断（tracker gap-fill 残留），
    再在图像 Y 极大值（球触地 → cy 最大）处切断。

    返回 [(start_idx, end_idx), ...] 的列表（索引针对 frames/uv）。
    """
    n = len(frames)
    if n < min_seg_frames:
        return [(0, n)]

    # ── 1. 按帧间隙切断（> 0.25 s = tracker max gap）────────────────────────
    max_gap = max(3, round(fps * 0.25))
    gap_splits = [0]
    for i in range(1, n):
        if frames[i] - frames[i - 1] > max_gap:
            gap_splits.append(i)
    gap_splits.append(n)

    # ── 2. 在每个间隙段内再按 cy 极大值切断 ─────────────────────────────────
    all_segs = []
    for g_s, g_e in zip(gap_splits[:-1], gap_splits[1:]):
        seg_n = g_e - g_s
        if seg_n < min_seg_frames:
            continue

        cy = uv[g_s:g_e, 1]
        w  = min(5, max(1, seg_n // 6))
        cy_s = np.convolve(cy, np.ones(w) / w, mode='same')
        vy   = np.diff(cy_s)

        sign  = np.sign(vy)
        peaks = [i + 1 for i in range(1, len(sign))
                 if sign[i - 1] > 0 and sign[i] <= 0]

        inner = [g_s] + [g_s + p for p in peaks
                         if min_seg_frames <= p <= seg_n - min_seg_frames] + [g_e]
        for s, e in zip(inner[:-1], inner[1:]):
            if e - s >= min_seg_frames:
                all_segs.append((s, e))

    return all_segs if all_segs else [(0, n)]


# ── 完整轨迹多段拟合 ──────────────────────────────────────────────────────────

def build_3d_trajectories(ball_traj_2d, H, fps, img_w, img_h,
                          min_frames=6, max_reproj_px=15.0):
    """
    对所有球轨迹分段拟合抛物线，输出每段的三维参数。

    Parameters
    ----------
    ball_traj_2d : {track_id: [(frame_idx, cx, cy, ball_d_px), ...]}
                   来自 check_json._build_ball_trajectories()
    H            : (3,3) 地面单应矩阵（球场米 → 图像像素）
    fps          : float  帧率
    img_w, img_h : int    图像尺寸（像素）
    min_frames   : int    最少帧数才拟合
    max_reproj_px: float  重投影误差超过此值标记为 success=False

    Returns
    -------
    {track_id: {
        'focal_len' : float          # 估计焦距（像素），供验证用
        'segments'  : [{
            'params'      : list[6]  # [X0,Y0,Z0,vx,vy,vz]
            't0_frame'    : float    # 该段 t=0 对应的帧号
            'frame_range' : (s, e)   # 覆盖帧号范围
            'reproj_err'  : float    # 平均重投影误差（像素）
            'success'     : bool
        }, ...]
    }}
    """
    try:
        K, P, R, t = recover_camera_from_H(H, img_w, img_h)
    except Exception as e:
        warnings.warn(f"[camera] 相机恢复失败: {e}")
        return {}

    H_inv  = np.linalg.inv(H.astype(np.float64))
    focal  = float(K[0, 0])
    result = {}

    for tid, pts in ball_traj_2d.items():
        all_frames = np.array([p[0] for p in pts], dtype=np.float64)
        all_uv     = np.array([[p[1], p[2]] for p in pts], dtype=np.float64)

        segs = _segment_by_bounce(all_frames, all_uv, fps=fps, min_seg_frames=min_frames)
        seg_results = []

        for s_idx, e_idx in segs:
            frames_seg = all_frames[s_idx:e_idx]
            uv_seg     = all_uv[s_idx:e_idx]

            if len(frames_seg) < min_frames:
                continue   # 帧数不足，无法可靠拟合 6 参数模型

            # ── 球场边界预检：地面投影不能远在球场之外 ────────────────────────
            uv0 = uv_seg[0]
            pt0 = H_inv @ np.array([uv0[0], uv0[1], 1.0])
            gx0, gy0 = pt0[0] / pt0[2], pt0[1] / pt0[2]
            out_of_court = (
                gx0 < -_COURT_MARGIN or gx0 > COURT_W + _COURT_MARGIN or
                gy0 < -_COURT_MARGIN or gy0 > COURT_L + _COURT_MARGIN
            )
            if out_of_court:
                seg_results.append({
                    'params':      [gx0, gy0, 0.0, 0.0, 0.0, 0.0],
                    't0_frame':    float(frames_seg[0]),
                    'frame_range': (int(frames_seg[0]), int(frames_seg[-1])),
                    'reproj_err':  float('inf'),
                    'success':     False,
                    'skip_reason': 'out_of_court',
                })
                continue

            params, _, xyz, err, ok = fit_parabola_3d(
                uv_seg, frames_seg, fps, P, H_inv=H_inv
            )
            ok = ok and err < max_reproj_px

            # ── 拟合后检验：X0/Y0 不能远在球场之外 ──────────────────────────
            X0f, Y0f = params[0], params[1]
            if (X0f < -_COURT_MARGIN or X0f > COURT_W + _COURT_MARGIN or
                    Y0f < -_COURT_MARGIN or Y0f > COURT_L + _COURT_MARGIN):
                ok = False

            seg_results.append({
                'params':      params.tolist(),
                't0_frame':    float(frames_seg[0]),
                'frame_range': (int(frames_seg[0]), int(frames_seg[-1])),
                'reproj_err':  err,
                'success':     ok,
            })

        result[tid] = {
            'focal_len': focal,
            'segments':  seg_results,
        }

    return result


def eval_at_frame(traj_entry, frame_idx, fps, g=_G):
    """
    在指定帧号处求三维坐标和速度（m/s）。

    traj_entry : build_3d_trajectories 返回的单条轨迹 dict
    返回 (X, Y, Z, speed_ms) 或 None（帧号不在任何段内）
    """
    for seg in traj_entry.get('segments', []):
        s, e = seg['frame_range']
        if s <= frame_idx <= e:
            t = (frame_idx - seg['t0_frame']) / fps
            X0, Y0, Z0, vx, vy, vz = seg['params']
            X = X0 + vx * t
            Y = Y0 + vy * t
            Z = max(0.0, Z0 + vz * t - 0.5 * g * t**2)
            vz_t = vz - g * t
            speed = float(np.sqrt(vx**2 + vy**2 + vz_t**2))
            return float(X), float(Y), float(Z), speed
    return None
