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
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QVBoxLayout, QWidget,
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


# ── JSON 加载 ─────────────────────────────────────────────────────────────────

def load_annotations(json_path: Path) -> tuple[dict, dict, dict | None]:
    """
    读取 COCO 格式 JSON，返回 (frame_anns, categories, court)
      frame_anns : {frame_idx: [{"bbox":[x,y,w,h], "category_id":int, "score":float}, ...]}
      categories : {cat_id: name}
      court      : {"keypoints":[[x,y],...], "valid_hull":[[x,y],...]} 或 None
    image.id 直接作为帧号。
    """
    with open(json_path) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data.get("categories", [])}

    frame_anns: dict[int, list] = {}
    for ann in data.get("annotations", []):
        frame_anns.setdefault(ann["image_id"], []).append({
            "bbox":        ann["bbox"],
            "category_id": ann["category_id"],
            "score":       ann.get("score", 1.0),
        })

    court = data.get("court")   # {"keypoints": [...], "valid_hull": [...]} 或 None
    return frame_anns, cats, court


# ── View（缩放/平移，与 annotate.py 保持一致） ────────────────────────────────

class View(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0 + delta / 1200.0
        factor = max(0.5, min(factor, 2.0))
        self.scale(factor, factor)


# ── 主窗口 ────────────────────────────────────────────────────────────────────

_COURT_COLOR = "#f0c040"   # 球场标注颜色（金黄）

class BrowseApp(QMainWindow):
    def __init__(self, video_path: Path, frame_anns: dict, categories: dict, court: dict | None):
        super().__init__()
        self.video_path  = video_path
        self.frame_anns  = frame_anns
        self.categories  = categories
        self.court       = court   # {"keypoints":[[x,y],...], "valid_hull":[[x,y],...]}

        cat_ids = sorted(categories.keys())
        self.cat_colors:  dict[int, QColor] = {
            cid: QColor(_PALETTE[i % len(_PALETTE)][0])
            for i, cid in enumerate(cat_ids)
        }
        # 标签缩写：取最后一个词首字母大写（与 annotate.py 一致）
        self.cat_labels: dict[int, str] = {
            cid: (words[-1][0].upper() if (words := categories[cid].split()) else "?")
            for cid in cat_ids
        }
        self.visible_cats: set = set(cat_ids)
        self.show_court: bool = court is not None

        # OpenCV 视频
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}", file=sys.stderr)
            sys.exit(1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.current_frame: int = -1   # -1 表示尚未加载

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
        self.resize(1280, 820)
        self.setStyleSheet("QMainWindow { background:#1e1e1e; }")

        # 工具栏（类别可见性切换）
        toolbar = QWidget(); toolbar.setStyleSheet("background:#2d2d2d;")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(12, 6, 12, 6); tl.setSpacing(6)
        tl.addWidget(QLabel("显示:", styleSheet="color:#858585; font:11pt Menlo;"))

        self.vis_btns: dict[int, QPushButton] = {}
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
            btn.clicked.connect(lambda _, c=cid: self._toggle_vis(c))
            tl.addWidget(btn)
            self.vis_btns[cid] = btn

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

        tl.addStretch()
        self.status_lbl = QLabel("", styleSheet="color:#4ec9b0; font:11pt Menlo;")
        tl.addWidget(self.status_lbl)

        # Scene / View
        self.scene = QGraphicsScene()
        self.view  = View(self.scene)
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
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        rl.addWidget(self.view, 1)
        rl.addWidget(ctrl)
        rl.addWidget(hint)

        central = QWidget()
        cl = QVBoxLayout(central); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        cl.addWidget(toolbar)
        cl.addWidget(right, 1)
        self.setCentralWidget(central)

    # ── 帧导航 ────────────────────────────────────────────────────────────────

    def _goto_frame(self, idx: int):
        idx = max(0, min(idx, self.total_frames - 1))

        # 只有帧号变化时才重新读取视频
        if idx != self.current_frame:
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

    def _redraw(self):
        """清空场景，重新绘制当前帧图像 + 标注。"""
        self.scene.clear()
        if self._current_pixmap:
            self.scene.addPixmap(self._current_pixmap)
            self.scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._render_annotations()

    def _step(self, delta: int):
        self._goto_frame(self.current_frame + delta)

    def _on_slider(self, val: int):
        self._goto_frame(val)

    # ── 播放 ──────────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._playing:
            self._play_timer.stop()
            self._playing = False
            self.play_btn.setText("▶")
        else:
            interval = max(1, int(1000 / self.fps))
            self._play_timer.start(interval)
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

        # 按面积从大到小排序，小框显示在上层
        anns_sorted = sorted(
            [a for a in anns if a.get("category_id", 0) in self.visible_cats],
            key=lambda a: a["bbox"][2] * a["bbox"][3],
            reverse=True,
        )
        for z, ann in enumerate(anns_sorted):
            cid = ann["category_id"]
            x, y, w, h = ann["bbox"]
            color = self.cat_colors.get(cid, QColor("white"))

            pen = QPen(color, 2)
            pen.setCosmetic(True)
            box = self.scene.addRect(QRectF(x, y, w, h), pen, QBrush(Qt.NoBrush))
            box.setZValue(z + 1)

            label = self.cat_labels.get(cid, "?")
            score = ann.get("score")
            if score is not None and score != 1.0:
                label = f"{label} {score:.2f}"

            txt = self.scene.addSimpleText(label, font)
            txt.setBrush(QBrush(color))
            txt.setPos(x, y - font_size * 1.4 if y >= font_size * 1.4 else y + h + 2)
            txt.setZValue(len(anns_sorted) + z + 1)

        # 球场
        if self.show_court and self.court:
            self._render_court(font_size)

    def _render_court(self, font_size: int):
        court_color = QColor(_COURT_COLOR)
        pen = QPen(court_color, 2)
        pen.setCosmetic(True)

        # valid_hull 轮廓多边形
        hull = self.court.get("valid_hull", [])
        if len(hull) >= 2:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in hull])
            item = self.scene.addPolygon(poly, pen, QBrush(Qt.NoBrush))
            item.setZValue(0)

        # 关键点
        kp_radius = max(6, font_size * 0.4)
        for kp in self.court.get("keypoints", []):
            x, y = kp
            item = self.scene.addEllipse(
                x - kp_radius, y - kp_radius, kp_radius * 2, kp_radius * 2,
                pen, QBrush(court_color),
            )
            item.setZValue(0)

    def _toggle_vis(self, cat_id: int):
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

    # ── 键盘 / 窗口事件 ───────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k   = event.key()
        mod = event.modifiers()

        if k == Qt.Key_Space and not event.isAutoRepeat():
            # 空格进入平移模式（与 annotate.py 一致）
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
            "  python browse.py -v match.mp4 -j detections.json\n"
        ),
    )
    parser.add_argument("-v", "--video", required=True, metavar="VIDEO",
                        help="input video file")
    parser.add_argument("-j", "--json", required=True, metavar="JSON",
                        help="annotation JSON (COCO format or frame-indexed dict)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    json_path  = Path(args.json).resolve()

    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not json_path.exists():
        print(f"错误: JSON 文件不存在: {json_path}", file=sys.stderr)
        sys.exit(1)

    frame_anns, categories, court = load_annotations(json_path)
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
