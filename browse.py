#!/usr/bin/env python3
"""
视频标注浏览器 - 逐帧查看视频标注结果。

用法:
    python browse.py -v <视频文件> -j <COCO标注JSON>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from court_detector import (compute_H_from_kps, COURT_LINES,
                             COURT_W as _COURT_W, NET_Y as _NET_Y)
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QSlider, QSplitter, QTextBrowser, QVBoxLayout, QWidget,
)

# ── 常量 ──────────────────────────────────────────────────────────────────────
_PALETTE = [
    ("#5588ff", "#1e3a6e"),
    ("#e05555", "#7a2222"),
    ("#44bb55", "#1e5c30"),
    ("#ffaa33", "#7a5500"),
    ("#aa44ff", "#4a1e6e"),
    ("#33cccc", "#1e5c5c"),
]
_COURT_COLOR    = "#f0c040"
_VOLUME_COLOR   = "#00dcff"
_SIDELINE_COLOR = "#00b400"   # 双打侧线墙（球员过滤边界）
_LIST_W      = 80    # 左侧帧号列表宽度（像素）

_BALL_TRAJ_COLORS = [
    QColor("#00ffff"), QColor("#ff6400"), QColor("#b400ff"),
    QColor("#00ff64"), QColor("#ffc800"), QColor("#0064ff"),
]
_BALL_UNTRACKED_COLOR = QColor("#804020")   # 未被 track 的网球（暗橙）
_BALL_TRAJ_FADE_FRAMES = 60    # 轨迹淡出窗口（帧数）
_PLAYER_TRAJ_COLORS = [
    QColor("#ff4444"), QColor("#44ff44"), QColor("#ffff44"), QColor("#ff44ff"),
]
_PLAYER_TRAJ_FADE_FRAMES = 120   # 球员轨迹淡出窗口（帧数，较长）
_RACKET_TRAJ_COLORS = [
    QColor("#ff8800"), QColor("#aa00ff"), QColor("#00ffbb"), QColor("#ff0099"),
]
_RACKET_TRAJ_FADE_FRAMES = 60    # 球拍轨迹淡出窗口（帧数）
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
_KP_CONF_THRESH = 0.3
_SEARCH_DIAMETERS  = 2.0   # 搜索半径倍数，与 tracker.py 默认值一致
_GAP_SECONDS       = 0.5   # 轨迹最大间隙时长（s），与 tracker.py _GAP_SECONDS 保持一致
_MAX_SPEED_MS      = 70.0  # 职业发球上限（m/s），与 tracker.py 一致
_RADIUS_MARGIN     = 1.3   # max_dist 安全裕量系数，与 tracker.py 一致
_LINEAR_WINDOW = 4   # 预测时只使用最近 N 个历史点估计速度方向


def _add_arrowed_line(scene, x1, y1, x2, y2, pen, tip_ratio=0.3, z=10):
    """带箭头的线段：主线 + 箭头三角形，与 cv2.arrowedLine tipLength 语义一致。"""
    scene.addLine(x1, y1, x2, y2, pen).setZValue(z)
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 4:
        return
    tip = length * tip_ratio
    ux, uy = dx / length, dy / length          # 方向单位向量
    px, py = -uy, ux                            # 垂直单位向量
    hw = tip * 0.35                             # 箭头半宽
    bx = x2 - ux * tip;  by = y2 - uy * tip   # 箭头底边中点
    poly = QPolygonF([
        QPointF(x2, y2),
        QPointF(bx + px * hw, by + py * hw),
        QPointF(bx - px * hw, by - py * hw),
    ])
    arrow_pen = QPen(pen.color(), 1); arrow_pen.setCosmetic(True)
    scene.addPolygon(poly, arrow_pen, QBrush(pen.color())).setZValue(z)


def _project_line(H, x1, y1, x2, y2):
    """用 H 将球场米坐标线段投影为图像像素坐标，返回 (pt1, pt2)。"""
    pts = cv2.perspectiveTransform(
        np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32), H)
    return tuple(pts[0, 0].astype(int)), tuple(pts[1, 0].astype(int))


# ── JSON 加载 ─────────────────────────────────────────────────────────────────

def load_annotations(json_path: Path) -> tuple[dict, dict, dict | None]:
    """
    读取 COCO 格式 JSON，返回 (frame_anns, categories, court)
      frame_anns : {frame_idx: [{"bbox":[x,y,w,h], "category_id":int, "score":float}, ...]}
      categories : {cat_id: name}
      court      : {"keypoints":[[x,y],...], "ground_hull":[[x,y],...], ...} 或 None
    image.id 直接作为帧号。
    """
    with open(json_path) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data.get("categories", [])}

    frame_anns: dict[int, list] = {}
    for ann in data.get("annotations", []):
        entry = {
            "bbox":         ann["bbox"],
            "category_id":  ann["category_id"],
            "score":        ann.get("score", 1.0),
            "track_id":     ann.get("track_id"),
            "valid":        ann.get("valid", True),
            "interpolated":   ann.get("interpolated", False),
            "backward_found": ann.get("backward_found", False),
            "rescue":         ann.get("rescue", False),
        }
        if "foot" in ann:
            entry["foot"] = ann["foot"]
        if "center" in ann:
            entry["center"] = ann["center"]
        if "keypoints" in ann:
            entry["keypoints"] = ann["keypoints"]
        frame_anns.setdefault(ann["image_id"], []).append(entry)

    court     = data.get("court")
    video_rel = data.get("video")
    video_abs = (json_path.parent / video_rel).resolve() if video_rel else None
    return frame_anns, cats, court, video_abs


# ── View（缩放/平移） ─────────────────────────────────────────────────────────

class FrameView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = max(0.5, min(1.0 + delta / 1200.0, 2.0))
        self.scale(factor, factor)


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class BrowseApp(QMainWindow):
    def __init__(self, video_path: Path, frame_anns: dict, categories: dict, court: dict | None):
        super().__init__()
        self.video_path  = video_path
        self.frame_anns  = frame_anns
        self.categories  = categories
        self.court       = court

        cat_ids = sorted(categories.keys())
        self.category_colors: dict[int, QColor] = {
            cid: QColor(_PALETTE[i % len(_PALETTE)][0])
            for i, cid in enumerate(cat_ids)
        }
        self.category_dark_colors: dict[int, QColor] = {
            cid: QColor(_PALETTE[i % len(_PALETTE)][1])
            for i, cid in enumerate(cat_ids)
        }
        self.category_labels: dict[int, str] = {
            cid: (words[-1][0].upper() if (words := categories[cid].split()) else "?")
            for cid in cat_ids
        }
        self.visible_cats: set = set(cat_ids)
        self.ball_cids: set = {cid for cid, name in categories.items()
                               if "ball" in name.lower()}
        self.player_cids: set = {cid for cid, name in categories.items()
                                 if "person" in name.lower()}
        self.racket_cids: set = {cid for cid, name in categories.items()
                                 if "racket" in name.lower()}
        # 是否来自 track 阶段：有任意球标注的 track_id != None 则为 True
        # detected JSON 中所有 track_id 均为 None，不应把 None 视为"被追踪器拒绝"
        self._has_tracked: bool = any(
            ann.get("track_id") is not None
            for anns in frame_anns.values()
            for ann in anns
            if ann.get("category_id") in self.ball_cids
        )
        # parse.py 之后的 JSON 会出现 valid=False 的标注；有 False 说明已过滤，不显示搜索圆
        self._has_parsed: bool = any(
            ann.get("valid") is False
            for anns in frame_anns.values()
            for ann in anns
        )
        self.show_court: bool       = court is not None
        self.show_ball_traj:  bool       = True
        self.show_player_traj: bool = True
        self.show_racket_traj: bool = True
        self.show_pose: bool        = True
        self._ball_traj          = self._build_ball_trajectories()
        self._player_traj   = self._build_player_trajectories()
        self._racket_traj   = self._build_racket_trajectories()

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}", file=sys.stderr)
            sys.exit(1)
        self.total_frames  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps           = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        # tracker 的 max_age（帧数），用于轨迹间隙判断和预测范围，与 tracker.py 计算方式一致
        self._ball_traj_max_age = max(3, round(self.fps * _GAP_SECONDS))
        # hist=1 时的搜索门限：单帧最大球速对应的像素位移（与 tracker.py 的 effective_gate 一致）
        self._max_dist = self._compute_max_dist()
        self.current_frame: int = -1

        self._current_pixmap: QPixmap | None = None
        self._img_w = self._img_h = 0
        self._view_fitted = False

        self._playing    = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_next)

        self._build_ui()
        self._goto_frame(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle(f"Browse — {self.video_path.name}")
        self.resize(1440, 820)
        self.setStyleSheet("QMainWindow { background:#1e1e1e; }")

        # 工具栏（类别可见性切换）
        toolbar = QWidget(); toolbar.setStyleSheet("background:#2d2d2d;")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 6, 12, 6); tl.setSpacing(6)
        tl.addWidget(QLabel("显示:", styleSheet="color:#858585; font:11pt Menlo;"))

        self.vis_buttons: dict[int, QPushButton] = {}
        for i, (cid, cname) in enumerate(sorted(self.categories.items())):
            active = _PALETTE[i % len(_PALETTE)][0]
            dark   = _PALETTE[i % len(_PALETTE)][1]
            btn = QPushButton(f"  {cname}  ")
            btn.setCheckable(True); btn.setChecked(True)
            btn.setStyleSheet(f"""
                QPushButton         {{ background:#2a2a2a; color:#666; border:none;
                                      padding:4px 10px; font:11pt Menlo; }}
                QPushButton:checked {{ background:{dark};  color:#ccc;
                                      border:1px solid {active}; }}
                QPushButton:hover   {{ background:{dark};  color:white; }}
            """)
            btn.clicked.connect(lambda _, c=cid: self._toggle_category(c))
            tl.addWidget(btn)
            self.vis_buttons[cid] = btn

        if self.court:
            self.court_btn = QPushButton("  球场  ")
            self.court_btn.setCheckable(True); self.court_btn.setChecked(True)
            self.court_btn.setStyleSheet(f"""
                QPushButton         {{ background:#2a2a2a; color:#666; border:none;
                                      padding:4px 10px; font:11pt Menlo; }}
                QPushButton:checked {{ background:#5a4a00; color:#ccc;
                                      border:1px solid {_COURT_COLOR}; }}
                QPushButton:hover   {{ background:#5a4a00; color:white; }}
            """)
            self.court_btn.clicked.connect(self._toggle_court)
            tl.addWidget(self.court_btn)

        self.ball_traj_btn = QPushButton("  网球轨迹  ")
        self.ball_traj_btn.setCheckable(True); self.ball_traj_btn.setChecked(True)
        self.ball_traj_btn.setStyleSheet("""
            QPushButton         { background:#2a2a2a; color:#666; border:none;
                                  padding:4px 10px; font:11pt Menlo; }
            QPushButton:checked { background:#004040; color:#ccc;
                                  border:1px solid #00ffff; }
            QPushButton:hover   { background:#004040; color:white; }
        """)
        self.ball_traj_btn.clicked.connect(self._toggle_ball_traj)
        tl.addWidget(self.ball_traj_btn)

        self.player_traj_btn = QPushButton("  球员轨迹  ")
        self.player_traj_btn.setCheckable(True); self.player_traj_btn.setChecked(True)
        self.player_traj_btn.setStyleSheet("""
            QPushButton         { background:#2a2a2a; color:#666; border:none;
                                  padding:4px 10px; font:11pt Menlo; }
            QPushButton:checked { background:#3a0020; color:#ccc;
                                  border:1px solid #ff4444; }
            QPushButton:hover   { background:#3a0020; color:white; }
        """)
        self.player_traj_btn.clicked.connect(self._toggle_player_traj)
        tl.addWidget(self.player_traj_btn)

        self.racket_traj_btn = QPushButton("  球拍轨迹  ")
        self.racket_traj_btn.setCheckable(True); self.racket_traj_btn.setChecked(True)
        self.racket_traj_btn.setStyleSheet("""
            QPushButton         { background:#2a2a2a; color:#666; border:none;
                                  padding:4px 10px; font:11pt Menlo; }
            QPushButton:checked { background:#3a2000; color:#ccc;
                                  border:1px solid #ff8800; }
            QPushButton:hover   { background:#3a2000; color:white; }
        """)
        self.racket_traj_btn.clicked.connect(self._toggle_racket_traj)
        tl.addWidget(self.racket_traj_btn)

        self.pose_btn = QPushButton("  球员姿态  ")
        self.pose_btn.setCheckable(True); self.pose_btn.setChecked(True)
        self.pose_btn.setStyleSheet("""
            QPushButton         { background:#2a2a2a; color:#666; border:none;
                                  padding:4px 10px; font:11pt Menlo; }
            QPushButton:checked { background:#003a20; color:#ccc;
                                  border:1px solid #00ff80; }
            QPushButton:hover   { background:#003a20; color:white; }
        """)
        self.pose_btn.clicked.connect(self._toggle_pose)
        tl.addWidget(self.pose_btn)

        tl.addStretch()
        self.status_lbl = QLabel("", styleSheet="color:#4ec9b0; font:11pt Menlo;")
        tl.addWidget(self.status_lbl)

        # 左侧帧号列表
        self.frame_list = QListWidget()
        self.frame_list.setFixedWidth(_LIST_W)
        self.frame_list.setUniformItemSizes(True)
        self.frame_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_list.setStyleSheet("""
            QListWidget { background:#141414; border:none; outline:none; }
            QListWidget::item { color:#555; font:9pt Menlo; text-align:right;
                                border-bottom:1px solid #1c1c1c; padding:2px 6px; }
            QListWidget::item:selected { background:#2a4a6e; color:#ccc; }
        """)
        for i in range(self.total_frames):
            item = QListWidgetItem(str(i))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.frame_list.addItem(item)
        self.frame_list.currentRowChanged.connect(self._on_list_select)

        # Scene / View
        self.scene = QGraphicsScene()
        self.view  = FrameView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor("#111111")))
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setFrameShape(QFrame.NoFrame)

        # 播放控件行
        ctrl = QWidget(); ctrl.setStyleSheet("background:#252526;")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(8, 4, 8, 4); ctrl_lay.setSpacing(6)

        btn_style = """
            QPushButton       { background:#3e3e42; color:#d4d4d4; border:none;
                                padding:4px 8px; font:11pt Menlo; }
            QPushButton:hover { background:#5a5a5e; }
        """
        prev_btn = QPushButton("◀"); prev_btn.setFixedWidth(32)
        prev_btn.setStyleSheet(btn_style)
        prev_btn.clicked.connect(lambda: self._step(-1))

        self.play_btn = QPushButton("▶"); self.play_btn.setFixedWidth(32)
        self.play_btn.setStyleSheet(btn_style)
        self.play_btn.clicked.connect(self._toggle_play)

        next_btn = QPushButton("▶|"); next_btn.setFixedWidth(36)
        next_btn.setStyleSheet(btn_style)
        next_btn.clicked.connect(lambda: self._step(1))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height:4px; background:#3e3e42; }
            QSlider::handle:horizontal { background:#4ec9b0; width:12px;
                                         margin:-4px 0; border-radius:6px; }
            QSlider::sub-page:horizontal { background:#4ec9b0; }
        """)
        self.slider.valueChanged.connect(self._on_slider)

        self.frame_lbl = QLabel("0 / 0", styleSheet="color:#d4d4d4; font:11pt Menlo;")
        self.frame_lbl.setFixedWidth(130)

        ctrl_lay.addWidget(prev_btn)
        ctrl_lay.addWidget(self.play_btn)
        ctrl_lay.addWidget(next_btn)
        ctrl_lay.addWidget(self.slider, 1)
        ctrl_lay.addWidget(self.frame_lbl)

        hint = QLabel(
            "←/→ 逐帧  ·  P 播放/暂停  ·  ⌘+/- 缩放  ·  ⌘0 复位  ·  空格+拖拽 平移",
            styleSheet="background:#1a1a1a;color:#555;font:10pt Menlo;padding:3px 12px;",
        )

        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        rl.addWidget(self.view, 1)
        rl.addWidget(ctrl)
        rl.addWidget(hint)

        # 右侧信息面板
        self.info_browser = QTextBrowser()
        self.info_browser.setFixedWidth(210)
        self.info_browser.setReadOnly(True)
        self.info_browser.setOpenLinks(False)
        self.info_browser.setStyleSheet(
            "QTextBrowser { background:#141414; border:none;"
            " border-left:1px solid #2d2d2d; }"
        )

        # 分割器：左=帧号列表，中=主视图，右=信息面板
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.frame_list)
        splitter.addWidget(right)
        splitter.addWidget(self.info_browser)
        splitter.setSizes([_LIST_W, 1060, 210])
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background:#333; }")

        central = QWidget()
        cl = QVBoxLayout(central); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        cl.addWidget(toolbar)
        cl.addWidget(splitter, 1)
        self.setCentralWidget(central)

    # ── 帧导航 ────────────────────────────────────────────────────────────────

    def _goto_frame(self, idx: int):
        idx = max(0, min(idx, self.total_frames - 1))

        if idx != self.current_frame:
            # 顺序前进时跳过 seek，显著提升播放速度
            if idx != self.current_frame + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = self.cap.read()
            if not ok:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._img_w, self._img_h = w, h
            self.current_frame = idx

        self._redraw()

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.frame_lbl.setText(f"{self.current_frame} / {self.total_frames - 1}")
        self._update_status()

        # 同步帧号列表（blockSignals 防止回调 _on_list_select 形成循环）
        self.frame_list.blockSignals(True)
        self.frame_list.setCurrentRow(self.current_frame)
        self.frame_list.blockSignals(False)
        self.frame_list.scrollToItem(
            self.frame_list.item(self.current_frame),
            QAbstractItemView.PositionAtCenter,
        )
        self._update_info_panel()

    def _redraw(self):
        self.scene.clear()
        if self._current_pixmap:
            self.scene.addPixmap(self._current_pixmap)
            self.scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._render_annotations()

    def _step(self, delta: int):
        self._goto_frame(self.current_frame + delta)

    def _on_slider(self, val: int):
        self._goto_frame(val)

    def _on_list_select(self, row: int):
        if row >= 0 and row != self.current_frame:
            self._goto_frame(row)

    # ── 播放 ──────────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._play_timer.stop()
            self._playing = False
            self.play_btn.setText("▶")
        else:
            self._play_timer.start(max(1, int(1000 / self.fps)))
            self._playing = True
            self.play_btn.setText("⏸")

    def _play_next(self):
        if self.current_frame >= self.total_frames - 1:
            self._toggle_play()
            return
        self._step(1)

    # ── 标注渲染 ──────────────────────────────────────────────────────────────

    def _render_annotations(self):
        anns = self.frame_anns.get(self.current_frame, [])
        font_size = max(12, self._img_h // 80)
        font = QFont("Menlo", font_size)

        # 计算每条轨迹的首个正向确认帧（非 backward 点的最小帧号），
        # 用于 backward_found 检测框的因果性判断：确认前按未追踪样式显示
        forward_start: dict[int, int] = {}
        if self._ball_traj:
            for tid, pts in self._ball_traj.items():
                non_bwd = [p[0] for p in pts if not p[4]]
                if non_bwd:
                    forward_start[tid] = min(non_bwd)
        cur = self.current_frame

        anns_sorted = sorted(
            [a for a in anns if a.get("category_id", 0) in self.visible_cats],
            key=lambda a: a["bbox"][2] * a["bbox"][3],
            reverse=True,
        )
        for z, ann in enumerate(anns_sorted):
            cid   = ann["category_id"]
            x, y, w, h = ann["bbox"]
            valid         = ann.get("valid", True)
            track_id      = ann.get("track_id")
            interpolated  = ann.get("interpolated", False)
            backward_found = ann.get("backward_found", False)
            rescue        = ann.get("rescue", False)
            is_ball       = cid in self.ball_cids

            # ── 颜色 / 样式判断 ───────────────────────────────────────────────
            # parse 阶段 invalid：
            #   球轨迹点（有 track_id）→ 轨迹色 + X，保留 conf/标签显示
            #   非球 / 无 track_id   → 暗色 + X（原逻辑）
            if not valid:
                if is_ball and track_id is not None:
                    # 因果性检查：backward+interpolated 合成帧在确认前仍须隐藏
                    if backward_found and interpolated:
                        fwd_start = forward_start.get(track_id)
                        if fwd_start is not None and cur < fwd_start:
                            continue
                    color   = self._ball_traj_color(track_id)
                    special = "invalid_interp" if interpolated else "invalid"
                else:
                    color   = self.category_dark_colors.get(cid, QColor("white"))
                    special = "invalid"
            # track 阶段未追踪（track_id=None）→ 暗橙 + X（仅在 tracked JSON 中生效）
            elif is_ball and track_id is None and self._has_tracked:
                color   = _BALL_UNTRACKED_COLOR
                special = "untracked"
            # 反向搜索找回的检测 / 反向插值帧
            # 因果性原则：
            #   确认前（cur < fwd_start）：
            #     真实检测 → 按原来未追踪样式显示（暗色 + X）
            #     合成插值帧 → 完全不显示（原本不存在）
            #   确认后（cur >= fwd_start）：全部补全显示
            #     真实检测 → 轨迹色 + X（backward 样式）
            #     插值帧   → 轨迹色 + 虚线（interpolated 样式）
            elif is_ball and backward_found:
                fwd_start = forward_start.get(track_id)
                if fwd_start is not None and cur < fwd_start:
                    if interpolated:
                        continue   # 合成帧确认前不存在，直接跳过
                    color   = _BALL_UNTRACKED_COLOR
                    special = "untracked"
                elif interpolated:
                    color   = self._ball_traj_color(track_id)
                    special = "interpolated"
                else:
                    color   = self._ball_traj_color(track_id)
                    special = "backward"
            # 正向插值帧 → 同 track_id 颜色 + 虚线
            elif is_ball and interpolated:
                color   = self._ball_traj_color(track_id)
                special = "interpolated"
            # 已追踪的球 → 按 track_id 着色
            elif is_ball and track_id is not None:
                color   = self._ball_traj_color(track_id)
                special = None
            else:
                color   = self.category_colors.get(cid, QColor("white"))
                special = None

            pen = QPen(color, 1 if special in ("invalid", "invalid_interp", "untracked", "backward") else 2)
            pen.setCosmetic(True)
            if special in ("interpolated", "invalid_interp"):
                pen.setStyle(Qt.DashLine)

            box = self.scene.addRect(QRectF(x, y, w, h), pen, QBrush(Qt.NoBrush))
            box.setZValue(z + 1)

            if special in ("invalid", "invalid_interp", "untracked", "backward"):
                self.scene.addLine(x, y, x + w, y + h, pen).setZValue(z + 1)
                self.scene.addLine(x + w, y, x, y + h, pen).setZValue(z + 1)
                if special == "untracked":
                    score = ann.get("score")
                    if score is not None:
                        lbl = self.scene.addSimpleText(f"{score:.2f}", font)
                        lbl.setBrush(QBrush(color))
                        lbl.setPos(x, y - font_size * 1.4 if y >= font_size * 1.4 else y + h + 2)
                        lbl.setZValue(len(anns_sorted) + z + 1)
                if special not in ("invalid", "invalid_interp"):
                    continue

            if cid in self.player_cids and track_id is not None:
                label = f"P{track_id}"
            elif cid in self.racket_cids and track_id is not None:
                label = f"R{track_id}"
            else:
                label = self.category_labels.get(cid, "?")
            score = ann.get("score")
            if score is not None and score != 1.0 and special not in ("interpolated", "invalid_interp"):
                label = f"{label} {score:.2f}"
            if special in ("interpolated", "invalid_interp"):
                label = f"{label} ~"
            if rescue:
                label = f"{label} [R]"

            txt = self.scene.addSimpleText(label, font)
            txt.setBrush(QBrush(color))
            txt.setPos(x, y - font_size * 1.4 if y >= font_size * 1.4 else y + h + 2)
            txt.setZValue(len(anns_sorted) + z + 1)

        if self.show_ball_traj and self._ball_traj:
            self._render_ball_trajectories()
            if not self._has_parsed:
                self._render_predictions()

        if self.show_player_traj and self._player_traj:
            self._render_player_trajectories()

        if self.show_racket_traj and self._racket_traj:
            self._render_racket_trajectories()

        if self.show_pose:
            self._render_skeletons(anns)

        if self.show_court and self.court:
            self._render_court(font_size)

    def _render_court(self, font_size: int):
        court_pen    = QPen(QColor(_COURT_COLOR),    1); court_pen.setCosmetic(True)
        volume_pen   = QPen(QColor(_VOLUME_COLOR),   1); volume_pen.setCosmetic(True)
        sideline_pen = QPen(QColor(_SIDELINE_COLOR), 1); sideline_pen.setCosmetic(True)

        # 球场线条 + 网线 + 关键点
        keypoints = self.court.get("keypoints", [])
        if keypoints:
            H = compute_H_from_kps(np.array(keypoints, dtype=np.float32).flatten())
            for (p1, p2, _lw) in COURT_LINES:
                pt1, pt2 = _project_line(H, p1[0], p1[1], p2[0], p2[1])
                self.scene.addLine(pt1[0], pt1[1], pt2[0], pt2[1], court_pen).setZValue(0)
            pt1, pt2 = _project_line(H, 0, _NET_Y, _COURT_W, _NET_Y)
            self.scene.addLine(pt1[0], pt1[1], pt2[0], pt2[1], court_pen).setZValue(0)

            kp_radius = max(6, font_size * 0.4)
            for kp in keypoints:
                x, y = kp
                self.scene.addEllipse(
                    x - kp_radius, y - kp_radius, kp_radius * 2, kp_radius * 2,
                    court_pen, QBrush(QColor(_COURT_COLOR)),
                ).setZValue(0)

        # 地面缓冲区轮廓
        ground_hull = self.court["ground_hull"]
        poly = QPolygonF([QPointF(p[0], p[1]) for p in ground_hull])
        self.scene.addPolygon(poly, court_pen, QBrush(Qt.NoBrush)).setZValue(0)

        # 双打侧线立方体（球员过滤边界）：遮罩 + 线框，与缓冲区显示方式一致
        # 先画内墙，再画外墙，遮罩从内到外叠加
        c_bot = self.court["court_bottom_pts"]
        c_top = self.court["court_top_pts"]
        self._render_outside_masks(c_bot, c_top)
        for i in range(4):
            j = (i + 1) % 4
            self.scene.addLine(c_bot[i][0], c_bot[i][1], c_bot[j][0], c_bot[j][1], sideline_pen).setZValue(1)
            self.scene.addLine(c_top[i][0], c_top[i][1], c_top[j][0], c_top[j][1], sideline_pen).setZValue(1)
            self.scene.addLine(c_bot[i][0], c_bot[i][1], c_top[i][0], c_top[i][1], sideline_pen).setZValue(1)

        # 缓冲区立方体线框 + 侧边外部遮罩
        vol_bot = self.court["vol_bottom_pts"]
        vol_top = self.court["vol_top_pts"]
        self._render_outside_masks(vol_bot, vol_top)
        for i in range(4):
            j = (i + 1) % 4
            self.scene.addLine(vol_bot[i][0], vol_bot[i][1], vol_bot[j][0], vol_bot[j][1], volume_pen).setZValue(1)
            self.scene.addLine(vol_top[i][0], vol_top[i][1], vol_top[j][0], vol_top[j][1], volume_pen).setZValue(1)
            self.scene.addLine(vol_bot[i][0], vol_bot[i][1], vol_top[i][0], vol_top[i][1], volume_pen).setZValue(1)

    def _render_outside_masks(self, vol_bot, vol_top):
        """在缓冲区侧边外部绘制半透明遮罩。
        侧边沿体积盒竖边方向（fl_bot→fl_top）向天空延伸，像一堵墙。
        vol_bot/vol_top 点顺序: [0]=远左, [1]=远右, [2]=近右, [3]=近左
        """
        def wall_to_sky(p_bot, p_top, sky_y):
            """沿 p_bot→p_top 方向从 p_top 继续延伸到 sky_y，返回 (x, sky_y)。"""
            bx, by = float(p_bot[0]), float(p_bot[1])
            tx, ty = float(p_top[0]), float(p_top[1])
            dy = ty - by
            if abs(dy) < 1e-6:
                return tx, sky_y
            t = (sky_y - ty) / dy
            return tx + (tx - bx) * t, sky_y

        def x_at_y(p1, p2, y):
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            dy = y2 - y1
            if abs(dy) < 1e-6:
                return x1
            return x1 + (x2 - x1) / dy * (y - y1)

        sky_y   = -self._img_h
        floor_y =  self._img_h * 2

        # vol_bot/vol_top: [0]=远左, [1]=远右, [2]=近右, [3]=近左
        nl_bot, fl_bot = vol_bot[3], vol_bot[0]
        nr_bot, fr_bot = vol_bot[2], vol_bot[1]
        nl_top, fl_top = vol_top[3], vol_top[0]
        nr_top, fr_top = vol_top[2], vol_top[1]

        # 沿墙竖边方向延伸到天空
        fl_sky_x, _ = wall_to_sky(fl_bot, fl_top, sky_y)
        fr_sky_x, _ = wall_to_sky(fr_bot, fr_top, sky_y)

        # 地面边界向近端延伸至图像底部以下
        x_floor_l = x_at_y(fl_bot, nl_bot, floor_y)
        x_floor_r = x_at_y(fr_bot, nr_bot, floor_y)

        mask_brush = QBrush(QColor(0, 0, 0, 70))
        no_pen = QPen(Qt.NoPen)

        # 左侧遮罩：以远左竖边延伸到天空为界，覆盖其左侧区域
        left_poly = QPolygonF([
            QPointF(-self._img_w, sky_y),
            QPointF(fl_sky_x,     sky_y),
            QPointF(fl_top[0],    fl_top[1]),
            QPointF(fl_bot[0],    fl_bot[1]),
            QPointF(nl_bot[0],    nl_bot[1]),
            QPointF(x_floor_l,    floor_y),
            QPointF(-self._img_w, floor_y),
        ])
        self.scene.addPolygon(left_poly, no_pen, mask_brush).setZValue(0.5)

        # 右侧遮罩：以远右竖边延伸到天空为界，覆盖其右侧区域
        right_poly = QPolygonF([
            QPointF(self._img_w * 2, sky_y),
            QPointF(fr_sky_x,        sky_y),
            QPointF(fr_top[0],       fr_top[1]),
            QPointF(fr_bot[0],       fr_bot[1]),
            QPointF(nr_bot[0],       nr_bot[1]),
            QPointF(x_floor_r,       floor_y),
            QPointF(self._img_w * 2, floor_y),
        ])
        self.scene.addPolygon(right_poly, no_pen, mask_brush).setZValue(0.5)

    def _build_ball_trajectories(self) -> dict[int, list[tuple]]:
        """返回 {track_id: [(frame_idx, cx, cy, ball_d_px, backward_found), ...]}。

        仅含已追踪（track_id != None）的网球标注。
        ball_d_px 为检测 bbox 均值宽高，用于计算透视自适应搜索半径。
        backward_found 标记该点是否由反向搜索找回（渲染时用虚线区分）。
        """
        traj: dict[int, list] = {}
        for frame_idx, anns in self.frame_anns.items():
            for ann in anns:
                tid = ann.get("track_id")
                if tid is None or ann.get("category_id") not in self.ball_cids:
                    continue
                x, y, w, h = ann["bbox"]
                cx, cy = x + w / 2, y + h / 2
                ball_d_px = (w + h) / 2.0
                backward = ann.get("backward_found", False)
                traj.setdefault(tid, []).append((frame_idx, cx, cy, ball_d_px, backward))
        for pts in traj.values():
            pts.sort(key=lambda t: t[0])
        return traj

    def _build_player_trajectories(self) -> dict[int, list[tuple[int, float, float]]]:
        """返回 {track_id: [(frame_idx, foot_x, foot_y), ...]}，仅含有效的、已追踪球员标注。
        使用检测框底部中心（脚步位置）作为轨迹锚点。
        """
        traj: dict[int, list] = {}
        for frame_idx, anns in self.frame_anns.items():
            for ann in anns:
                tid = ann.get("track_id")
                if tid is None or ann.get("category_id") not in self.player_cids:
                    continue
                if not ann.get("valid", True):
                    continue
                if 'foot' in ann:
                    fx, fy = ann['foot']
                else:
                    x, y, w, h = ann["bbox"]
                    fx, fy = x + w / 2, y + h
                traj.setdefault(tid, []).append((frame_idx, fx, fy))
        for pts in traj.values():
            pts.sort(key=lambda t: t[0])
        return traj

    def _player_traj_color(self, track_id: int) -> QColor:
        return _PLAYER_TRAJ_COLORS[track_id % len(_PLAYER_TRAJ_COLORS)]

    def _build_racket_trajectories(self) -> dict[int, list[tuple[int, float, float]]]:
        """返回 {track_id: [(frame_idx, cx, cy), ...]}，仅含有效的、已追踪球拍标注。"""
        traj: dict[int, list] = {}
        for frame_idx, anns in self.frame_anns.items():
            for ann in anns:
                tid = ann.get("track_id")
                if tid is None or ann.get("category_id") not in self.racket_cids:
                    continue
                if not ann.get("valid", True):
                    continue
                if 'center' in ann:
                    cx, cy = ann['center']
                else:
                    x, y, w, h = ann["bbox"]
                    cx, cy = x + w / 2, y + h / 2
                traj.setdefault(tid, []).append((frame_idx, cx, cy))
        for pts in traj.values():
            pts.sort(key=lambda t: t[0])
        return traj

    def _racket_traj_color(self, track_id: int) -> QColor:
        return _RACKET_TRAJ_COLORS[track_id % len(_RACKET_TRAJ_COLORS)]

    def _compute_max_dist(self) -> float | None:
        """从 court keypoints + fps 推算单帧最大球速像素位移（与 tracker.py effective_gate 的 max_dist 一致）。"""
        if not self.court:
            return None
        kps = self.court.get("keypoints", [])
        if len(kps) < 4:
            return None
        k = np.array(kps, dtype=np.float32).reshape(-1, 2)
        far_ppm  = float(np.linalg.norm(k[1] - k[0])) / _COURT_W
        near_ppm = float(np.linalg.norm(k[3] - k[2])) / _COURT_W
        px_per_meter = (far_ppm + near_ppm) / 2.0
        return _MAX_SPEED_MS / self.fps * px_per_meter * _RADIUS_MARGIN

    def _ball_traj_color(self, track_id: int) -> QColor:
        return _BALL_TRAJ_COLORS[track_id % len(_BALL_TRAJ_COLORS)]

    def _render_ball_trajectories(self):
        """绘制当前帧及之前的网球轨迹线段。

        反向搜索找回的线段（backward_found）用虚线绘制，正常追踪线段用实线。
        时序规则：backward 段仅在当前帧 >= 轨迹首个正向确认帧时才显示，
        模拟"反向补齐"在轨迹被确认后才回溯出现的效果。
        """
        cur = self.current_frame

        # 计算每条轨迹的首个正向确认帧（非 backward 的最小帧号）
        forward_start: dict[int, int] = {}
        for tid, pts in self._ball_traj.items():
            non_bwd = [p[0] for p in pts if not p[4]]
            if non_bwd:
                forward_start[tid] = min(non_bwd)

        for tid, pts in self._ball_traj.items():
            color = self._ball_traj_color(tid)
            fwd_start = forward_start.get(tid, 0)
            prev = None
            for frame_idx, cx, cy, _, backward in pts:
                if frame_idx > cur:
                    break
                # backward 点：仅在轨迹已被正向确认后才显示
                if backward and cur < fwd_start:
                    prev = None   # 不与后续点相连
                    continue
                if prev is not None:
                    pf, px, py, prev_backward = prev
                    # 仅连接相邻帧，避免跨越长间隙时画长线
                    if frame_idx - pf <= self._ball_traj_max_age:
                        alpha = int(255 * max(0.2, 1.0 - (cur - frame_idx) / _BALL_TRAJ_FADE_FRAMES))
                        c = QColor(color)
                        c.setAlpha(alpha)
                        p = QPen(c, 2)
                        p.setCosmetic(True)
                        # 两端点任一为反向找回则用虚线
                        if backward or prev_backward:
                            p.setStyle(Qt.DashLine)
                        _add_arrowed_line(self.scene, px, py, cx, cy, p)
                prev = (frame_idx, cx, cy, backward)

    def _render_predictions(self):
        """为每条活动轨迹绘制预测曲线及下一帧搜索圆。

        预测：最多取最近 _LINEAR_WINDOW 个点线性外推（与 tracker.py 一致）。

        搜索圆半径（与 tracker.py effective_gate 保持一致）：
          hist == 1 → self._max_dist（单帧最大球速，物理兜底）
          hist  > 1 → _SEARCH_DIAMETERS × 最近一帧检测球径（透视自适应）
        """
        cur = self.current_frame

        for tid, pts in self._ball_traj.items():
            # 取当前帧及之前的检测点（5-tuple: frame_idx, cx, cy, ball_d_px, backward_found）
            past = [p for p in pts if p[0] <= cur]
            if not past:
                continue
            last_fi = past[-1][0]
            # 超过 max_age 帧未检测到，tracker 已删除该轨迹，不再显示预测
            if cur - last_fi > self._ball_traj_max_age:
                continue

            color = self._ball_traj_color(tid)
            ts = np.array([p[0] for p in past], dtype=float)
            xs = np.array([p[1] for p in past], dtype=float)
            ys = np.array([p[2] for p in past], dtype=float)
            t0    = ts[-1]
            tn    = ts - t0
            t_end = float(cur + 1 + self._ball_traj_max_age - t0)

            # 最多取最近 _LINEAR_WINDOW 个点（不足时取全部）线性外推
            w = _LINEAR_WINDOW
            deg = min(1, len(past) - 1)
            px_coef = np.polyfit(tn[-w:], xs[-w:], deg)
            py_coef = np.polyfit(tn[-w:], ys[-w:], deg)

            # 预测曲线：从最后已知点到 cur+1+traj_max_age
            n_pts  = max(8, min(120, int(t_end * 3)))
            sample = np.linspace(0.0, t_end, n_pts)
            pred_x = np.polyval(px_coef, sample)
            pred_y = np.polyval(py_coef, sample)

            pred_color = QColor(color); pred_color.setAlpha(110)
            pred_pen   = QPen(pred_color, 1); pred_pen.setCosmetic(True)
            pred_pen.setStyle(Qt.DotLine)
            for i in range(n_pts - 1):
                self.scene.addLine(
                    float(pred_x[i]),   float(pred_y[i]),
                    float(pred_x[i+1]), float(pred_y[i+1]),
                    pred_pen,
                ).setZValue(11)

            # 搜索圆半径：与 tracker.py effective_gate 逻辑一致
            if len(past) == 1 and self._max_dist is not None:
                sr = self._max_dist
            else:
                sr = _SEARCH_DIAMETERS * past[-1][3]   # past[-1][3] = ball_d_px

            # 下一帧预测位置的搜索圆
            t_next  = float(cur + 1 - t0)
            next_cx = float(np.polyval(px_coef, t_next))
            next_cy = float(np.polyval(py_coef, t_next))
            c_color = QColor(color); c_color.setAlpha(180)
            c_pen   = QPen(c_color, 1); c_pen.setCosmetic(True)
            self.scene.addEllipse(
                next_cx - sr, next_cy - sr, sr * 2, sr * 2,
                c_pen, QBrush(Qt.NoBrush),
            ).setZValue(11)

    def _render_player_trajectories(self):
        """绘制球员脚步全量历史轨迹（淡出效果，不含将来帧）。"""
        cur = self.current_frame
        for tid, pts in self._player_traj.items():
            color = self._player_traj_color(tid)
            prev = None
            for frame_idx, fx, fy in pts:
                if frame_idx > cur:
                    break
                if prev is not None:
                    pf, px, py = prev
                    if frame_idx - pf <= self._ball_traj_max_age:
                        alpha = int(255 * max(0.15, 1.0 - (cur - frame_idx) / _PLAYER_TRAJ_FADE_FRAMES))
                        c = QColor(color)
                        c.setAlpha(alpha)
                        p = QPen(c, 2); p.setCosmetic(True)
                        _add_arrowed_line(self.scene, px, py, fx, fy, p)
                prev = (frame_idx, fx, fy)

    def _toggle_ball_traj(self):
        self.show_ball_traj = not self.show_ball_traj
        self._redraw()

    def _render_racket_trajectories(self):
        """绘制当前帧及之前的球拍中心轨迹线段（淡出效果）。"""
        cur = self.current_frame
        for tid, pts in self._racket_traj.items():
            color = self._racket_traj_color(tid)
            prev = None
            for frame_idx, cx, cy in pts:
                if frame_idx > cur:
                    break
                if prev is not None:
                    pf, px, py = prev
                    if frame_idx - pf <= self._ball_traj_max_age:
                        alpha = int(255 * max(0.15, 1.0 - (cur - frame_idx) / _RACKET_TRAJ_FADE_FRAMES))
                        c = QColor(color)
                        c.setAlpha(alpha)
                        p = QPen(c, 2); p.setCosmetic(True)
                        _add_arrowed_line(self.scene, px, py, cx, cy, p)
                prev = (frame_idx, cx, cy)

    def _toggle_player_traj(self):
        self.show_player_traj = not self.show_player_traj
        self._redraw()

    def _render_skeletons(self, anns):
        """绘制当前帧所有有效球员的姿态骨架。"""
        for ann in anns:
            if ann.get("category_id") not in self.player_cids:
                continue
            if not ann.get("valid", True):
                continue
            if "keypoints" not in ann:
                continue
            kps = ann["keypoints"]  # list of 17 [x, y, conf]
            track_id = ann.get("track_id")
            if track_id is not None:
                color = _PLAYER_TRAJ_COLORS[track_id % len(_PLAYER_TRAJ_COLORS)]
            else:
                color = QColor("#00ff80")
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            for (i, j) in _SKELETON:
                if i >= len(kps) or j >= len(kps):
                    continue
                xi, yi, ci = kps[i]
                xj, yj, cj = kps[j]
                if ci >= _KP_CONF_THRESH and cj >= _KP_CONF_THRESH:
                    self.scene.addLine(float(xi), float(yi), float(xj), float(yj),
                                       pen).setZValue(20)
            for kp in kps:
                x, y, c = kp
                if c >= _KP_CONF_THRESH:
                    r = 4
                    self.scene.addEllipse(float(x) - r, float(y) - r, r * 2, r * 2,
                                          pen, QBrush(color)).setZValue(20)

    def _toggle_racket_traj(self):
        self.show_racket_traj = not self.show_racket_traj
        self._redraw()

    def _toggle_pose(self):
        self.show_pose = not self.show_pose
        self._redraw()

    def _toggle_category(self, cat_id: int):
        if cat_id in self.visible_cats:
            self.visible_cats.discard(cat_id)
        else:
            self.visible_cats.add(cat_id)
        self._redraw()
        self._update_status()

    def _toggle_court(self):
        self.show_court = not self.show_court
        self._redraw()

    def _update_status(self):
        anns = self.frame_anns.get(self.current_frame, [])
        n_visible = sum(1 for a in anns if a.get("category_id", 0) in self.visible_cats)
        self.status_lbl.setText(
            f"帧 {self.current_frame}  ·  {n_visible} 个标注  ·  {self.fps:.1f} fps")

    def _update_info_panel(self):
        fi   = self.current_frame
        anns = self.frame_anns.get(fi, [])
        time_s = fi / self.fps if self.fps else 0.0

        # ── 按类别 + valid 分类 ────────────────────────────────────────────────
        def _split(cids):
            v = [a for a in anns if a.get('category_id') in cids and     a.get('valid', True)]
            i = [a for a in anns if a.get('category_id') in cids and not a.get('valid', True)]
            return v, i

        v_balls,   inv_balls   = _split(self.ball_cids)
        v_persons, inv_persons = _split(self.player_cids)
        v_rackets, inv_rackets = _split(self.racket_cids)

        # ── 合理性判断（仅 parse 后有意义）────────────────────────────────────
        def _status(count, valid_counts):
            """返回 (icon, color_hex)"""
            if count == 0:
                return '—', '#555555'
            if count in valid_counts:
                return '✓', '#4ec9b0'
            return '✗', '#ff4444'

        ball_icon,   ball_color   = _status(len(v_balls),   {1})
        person_icon, person_color = _status(len(v_persons), {2, 4})
        racket_icon, racket_color = _status(len(v_rackets), {2, 4})

        # ── track_id 列表 ──────────────────────────────────────────────────────
        def _tids(dets):
            ids = sorted({a['track_id'] for a in dets if a.get('track_id') is not None})
            return ' '.join(f'#{t}' for t in ids) if ids else '—'

        S  = 'font-family:Menlo,monospace; font-size:10pt;'
        DIM = '#555555'
        HDR = '#777777'

        def row(label, count, icon, color, tids=''):
            tid_html = (f'<br><span style="color:{DIM}; font-size:9pt;">'
                        f'&nbsp;&nbsp;&nbsp;{tids}</span>') if tids and tids != '—' else ''
            return (f'<tr>'
                    f'<td style="color:{HDR}; padding:1px 4px 1px 0;">{label}</td>'
                    f'<td style="color:#cccccc; padding:1px 6px; text-align:right;">{count}</td>'
                    f'<td style="color:{color}; padding:1px 4px;">{icon}</td>'
                    f'</tr>'
                    + (f'<tr><td colspan="3" style="color:{DIM}; font-size:9pt; padding:0 4px 3px 4px;">'
                       f'{tids}</td></tr>' if tids and tids != '—' else ''))

        # ── 无效行（只在 parsed 后显示）───────────────────────────────────────
        def inv_row(label, count):
            c = '#666666' if count == 0 else '#ff8844'
            return (f'<tr>'
                    f'<td style="color:{DIM}; padding:1px 4px 1px 0;">{label}</td>'
                    f'<td style="color:{c}; padding:1px 6px; text-align:right;">{count}</td>'
                    f'<td></td></tr>')

        sep = f'<tr><td colspan="3"><hr style="border:none; border-top:1px solid #2d2d2d; margin:4px 0;"></td></tr>'

        html_parts = [
            f'<html><body style="{S} background:#141414; color:#aaa; padding:8px;">',
            f'<p style="color:#4ec9b0; margin:0 0 2px 0; font-size:10pt;">'
            f'帧 {fi:>5}</p>',
            f'<p style="color:{DIM}; margin:0 0 6px 0; font-size:9pt;">'
            f'{time_s:.2f}s</p>',
            f'<table cellspacing="0" cellpadding="0" width="100%">',
            sep,
        ]

        if self._has_parsed:
            html_parts.append(
                f'<tr><td colspan="3" style="color:{HDR}; font-size:9pt; padding:2px 0 3px 0;">'
                f'有效检测</td></tr>')
            html_parts.append(row('网球', len(v_balls),   ball_icon,   ball_color,   _tids(v_balls)))
            html_parts.append(row('球员', len(v_persons), person_icon, person_color, _tids(v_persons)))
            html_parts.append(row('球拍', len(v_rackets), racket_icon, racket_color, _tids(v_rackets)))
            html_parts.append(sep)
            html_parts.append(
                f'<tr><td colspan="3" style="color:{HDR}; font-size:9pt; padding:2px 0 3px 0;">'
                f'无效检测</td></tr>')
            html_parts.append(inv_row('网球', len(inv_balls)))
            html_parts.append(inv_row('球员', len(inv_persons)))
            html_parts.append(inv_row('球拍', len(inv_rackets)))
        else:
            html_parts.append(
                f'<tr><td colspan="3" style="color:{HDR}; font-size:9pt; padding:2px 0 3px 0;">'
                f'检测</td></tr>')
            html_parts.append(row('网球', len(v_balls),   '', '#aaaaaa', _tids(v_balls)))
            html_parts.append(row('球员', len(v_persons), '', '#aaaaaa', _tids(v_persons)))
            html_parts.append(row('球拍', len(v_rackets), '', '#aaaaaa', _tids(v_rackets)))

        html_parts.append('</table></body></html>')
        self.info_browser.setHtml(''.join(html_parts))

    # ── 键盘 / 窗口事件 ───────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k   = event.key()
        mod = event.modifiers()

        if k == Qt.Key_Space and not event.isAutoRepeat():
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            return
        if k == Qt.Key_P:
            self._toggle_play()
        elif k in (Qt.Key_Right, Qt.Key_Down):
            self._step(1)
        elif k in (Qt.Key_Left, Qt.Key_Up):
            self._step(-1)
        elif k in (Qt.Key_Equal, Qt.Key_Plus) and mod & Qt.ControlModifier:
            self.view.scale(1.3, 1.3)
        elif k == Qt.Key_Minus and mod & Qt.ControlModifier:
            self.view.scale(1 / 1.3, 1 / 1.3)
        elif k == Qt.Key_0 and mod & Qt.ControlModifier:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.view.setDragMode(QGraphicsView.NoDrag)
        else:
            super().keyReleaseEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._view_fitted and self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._view_fitted = True

    def closeEvent(self, event):
        self._play_timer.stop()
        self.cap.release()
        super().closeEvent(event)


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video annotation browser — view frame-by-frame annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python browse.py detections.json\n"
        ),
    )
    parser.add_argument("json", metavar="JSON",
                        help="annotation JSON (COCO format)，视频路径从 video 字段读取")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        print(f"错误: JSON 文件不存在: {json_path}", file=sys.stderr)
        sys.exit(1)

    frame_anns, categories, court, video_path = load_annotations(json_path)

    if not video_path:
        print("错误: JSON 中未包含 video 字段，无法定位视频文件", file=sys.stderr)
        sys.exit(1)

    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_anns = sum(len(v) for v in frame_anns.values())
    print(f"已加载: {len(frame_anns)} 帧有标注，共 {total_anns} 个标注，{len(categories)} 个类别"
          + ("，含球场数据" if court else ""))

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = BrowseApp(video_path, frame_anns, categories, court)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
