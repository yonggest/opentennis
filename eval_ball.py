#!/usr/bin/env python3
"""
网球检测模型可视化评测：左侧文件列表 + 右侧分页网格，列表与网格联动高亮。

用法：
    python eval_ball.py --data datasets/xxx/data.yaml
    python eval_ball.py --data datasets/xxx/data.yaml --model models/yolo26n-ball.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

_CELL_W    = 160    # 正方形缩略图边长（px）
_GRID_COLS = 5
_GRID_ROWS = 4
_PAGE_SIZE = _GRID_COLS * _GRID_ROWS   # 每页 20 张
_LIST_W    = 220    # 左侧列表初始宽度（px）

_GT_COLOR   = (0, 200,   0)   # BGR 绿色：GT 标注框
_DET_HIT    = (0, 100, 255)   # BGR 橙红色：检出 conf ≥ 阈值
_DET_MISS   = (80,  80, 220)  # BGR 蓝紫色：检出 conf < 阈值
_CONF_LOW    = 0.01
_CONF_ACCEPT = 0.5


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _resolve_images(yaml_path: Path):
    """解析 data.yaml，返回 (train_images, val_images)，各自按文件名排序。"""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get('path', yaml_path.parent)).resolve()
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    def collect(key):
        rel = cfg.get(key)
        if not rel:
            return []
        d = (root / rel).resolve()
        return sorted(p for p in d.rglob('*') if p.suffix.lower() in exts) if d.exists() else []

    return collect('train'), collect('val')


def _load_gt_boxes(image_path: Path, img_w: int, img_h: int) -> list:
    """从 YOLO .txt 标注文件读取 GT 框（像素坐标）。"""
    label_path = Path(
        str(image_path).replace('/images/', '/labels/')
    ).with_suffix('.txt')
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((
                int((cx - w / 2) * img_w), int((cy - h / 2) * img_h),
                int((cx + w / 2) * img_w), int((cy + h / 2) * img_h),
            ))
    return boxes


# ── 缩略图渲染 ────────────────────────────────────────────────────────────────

def _render(image_path: Path, best_det) -> QPixmap:
    """渲染正方形缩略图（letterbox），叠加 GT 框和最佳检测框。"""
    img = cv2.imread(str(image_path))
    if img is None:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
    h, w = img.shape[:2]

    # GT 框（绿色，细线）
    for x1, y1, x2, y2 in _load_gt_boxes(image_path, w, h):
        cv2.rectangle(img, (x1, y1), (x2, y2), _GT_COLOR, 1)

    # 最佳检测框
    if best_det is not None:
        conf, bx1, by1, bx2, by2 = best_det
        color = _DET_HIT if conf >= _CONF_ACCEPT else _DET_MISS
        cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 1)
        cv2.putText(img, f"{conf:.2f}",
                    (int(bx1), max(int(by1) - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Letterbox → 正方形
    size   = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    x_off, y_off = (size - w) // 2, (size - h) // 2
    square[y_off:y_off + h, x_off:x_off + w] = img

    thumb = cv2.resize(square, (_CELL_W, _CELL_W), interpolation=cv2.INTER_AREA)
    rgb   = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    qimg  = QImage(rgb.data, _CELL_W, _CELL_W, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ── 后台推断线程 ──────────────────────────────────────────────────────────────

class _InferWorker(QThread):
    """对一批图片推断，命中缓存则跳过推断直接渲染。"""

    result_ready = Signal(int, QPixmap, object)   # (local_idx, pixmap, best_det)

    def __init__(self, paths: list, model, imgsz: int, cache: dict, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._model = model
        self._imgsz = imgsz
        self._cache = cache

    def run(self):
        # 批量推断所有未缓存的图片（一次 predict() 调用，避免 per-image 开销）
        uncached = [(i, p) for i, p in enumerate(self._paths) if p not in self._cache]
        if uncached and not self.isInterruptionRequested():
            idxs, paths = zip(*uncached)
            try:
                results = self._model.predict(
                    [str(p) for p in paths],
                    conf=_CONF_LOW, imgsz=self._imgsz, verbose=False,
                )
                for local_idx, r in zip(idxs, results):
                    best_det = None
                    best_conf = 0.0
                    for box in r.boxes:
                        c = float(box.conf)
                        if c > best_conf:
                            best_conf = c
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            best_det = (c, x1, y1, x2, y2)
                    self._cache[self._paths[local_idx]] = best_det
            except Exception:
                for local_idx, path in uncached:
                    if path not in self._cache:
                        self._cache[path] = None

        # 逐一渲染并发射结果（缓存 + 新推断）
        for local_idx, path in enumerate(self._paths):
            if self.isInterruptionRequested():
                break
            best_det = self._cache.get(path)
            pixmap = _render(path, best_det)
            self.result_ready.emit(local_idx, pixmap, best_det)


# ── 单元格 ────────────────────────────────────────────────────────────────────

_STYLE_NORMAL   = "QFrame { border: 1px solid #444; background: #1e1e1e; }"
_STYLE_HIT      = "QFrame { border: 2px solid #00cc44; background: #1e1e1e; }"
_STYLE_MISS     = "QFrame { border: 2px solid #ff8800; background: #1e1e1e; }"
_STYLE_SELECTED = "QFrame { border: 3px solid #ffee00; background: #2a2a10; }"

_PLACEHOLDER = None

def _placeholder_pixmap() -> QPixmap:
    global _PLACEHOLDER
    if _PLACEHOLDER is None:
        img  = np.full((_CELL_W, _CELL_W, 3), 30, dtype=np.uint8)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, _CELL_W, _CELL_W, rgb.strides[0], QImage.Format_RGB888)
        _PLACEHOLDER = QPixmap.fromImage(qimg.copy())
    return _PLACEHOLDER


class _ImageCell(QFrame):
    """网格中的单个图片格，可点击，支持高亮选中状态。"""

    clicked = Signal(int)   # local_idx

    def __init__(self, local_idx: int, parent=None):
        super().__init__(parent)
        self._local_idx = local_idx
        self._best_det  = None
        self._selected  = False
        self.setFixedSize(_CELL_W + 6, _CELL_W + 20)
        self.setStyleSheet(_STYLE_NORMAL)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setFixedSize(_CELL_W, _CELL_W)
        self._img_lbl.setStyleSheet("border: none;")
        layout.addWidget(self._img_lbl)

        self._name_lbl = QLabel()
        self._name_lbl.setAlignment(Qt.AlignCenter)
        font = QFont(); font.setPointSize(7)
        self._name_lbl.setFont(font)
        self._name_lbl.setMaximumWidth(_CELL_W)
        self._name_lbl.setStyleSheet("border: none;")
        layout.addWidget(self._name_lbl)

        self.reset()

    def reset(self):
        self._best_det = None
        self._selected = False
        self._img_lbl.setPixmap(_placeholder_pixmap())
        self._name_lbl.setText("")
        self.setStyleSheet(_STYLE_NORMAL)
        self.setVisible(False)

    def set_loading(self, name: str):
        self._best_det = None
        self._selected = False
        self._img_lbl.setPixmap(_placeholder_pixmap())
        self._name_lbl.setText(name)
        self.setStyleSheet(_STYLE_NORMAL)
        self.setVisible(True)

    def set_result(self, pixmap: QPixmap, best_det):
        self._best_det = best_det
        self._img_lbl.setPixmap(pixmap)
        self._refresh_style()

    def set_selected(self, selected: bool):
        self._selected = selected
        self._refresh_style()

    def _refresh_style(self):
        if self._selected:
            self.setStyleSheet(_STYLE_SELECTED)
        elif self._best_det and self._best_det[0] >= _CONF_ACCEPT:
            self.setStyleSheet(_STYLE_HIT)
        elif self._best_det:
            self.setStyleSheet(_STYLE_MISS)
        else:
            self.setStyleSheet(_STYLE_NORMAL)

    def mousePressEvent(self, event):
        self.clicked.emit(self._local_idx)
        super().mousePressEvent(event)


# ── 数据集 Tab ────────────────────────────────────────────────────────────────

class _EvalTab(QWidget):
    """单个数据集的完整视图：左侧文件列表 + 右侧分页网格，两者联动。"""

    status_msg = Signal(str)

    def __init__(self, paths: list, model, imgsz: int, parent=None):
        super().__init__(parent)
        self._paths    = paths
        self._model    = model
        self._imgsz    = imgsz
        self._cache: dict = {}
        self._page     = 0
        self._selected = -1
        self._worker   = None
        self._n_pages  = max(1, (len(paths) + _PAGE_SIZE - 1) // _PAGE_SIZE)

        self._started = False
        self._build_ui()

    # ── UI 构建 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # ── 左侧：文件列表 ────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        font = QFont(); font.setPointSize(9)
        self._list.setFont(font)
        for path in self._paths:
            self._list.addItem(QListWidgetItem(path.name))
        self._list.currentRowChanged.connect(self._on_list_select)
        splitter.addWidget(self._list)

        # ── 右侧：网格 + 翻页 ─────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(4)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        self._cells: list[_ImageCell] = []
        for i in range(_PAGE_SIZE):
            cell = _ImageCell(i)
            cell.clicked.connect(self._on_cell_click)
            self._cells.append(cell)
            grid_layout.addWidget(cell, i // _GRID_COLS, i % _GRID_COLS)
        right_layout.addWidget(grid_widget)
        right_layout.addStretch()

        # 翻页控件
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("◀  上一页")
        self._btn_next = QPushButton("下一页  ▶")
        self._page_lbl = QLabel()
        self._page_lbl.setAlignment(Qt.AlignCenter)
        self._btn_prev.clicked.connect(self._prev_page)
        self._btn_next.clicked.connect(self._next_page)
        nav.addWidget(self._btn_prev)
        nav.addStretch()
        nav.addWidget(self._page_lbl)
        nav.addStretch()
        nav.addWidget(self._btn_next)
        right_layout.addLayout(nav)

        splitter.addWidget(right)
        splitter.setSizes([_LIST_W, 900])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ── 页面加载 ─────────────────────────────────────────────────────────────

    def _load_page(self, page: int):
        self._stop_worker()
        self._page = page
        start = page * _PAGE_SIZE
        batch = self._paths[start: start + _PAGE_SIZE]

        for i, cell in enumerate(self._cells):
            if i < len(batch):
                cell.set_loading(batch[i].name)
                if self._page * _PAGE_SIZE + i == self._selected:
                    cell.set_selected(True)
            else:
                cell.reset()

        self._update_nav()

        self._worker = _InferWorker(batch, self._model, self._imgsz, self._cache)
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished.connect(self._on_page_done)
        self._worker.start()

        cached = sum(1 for p in batch if p in self._cache)
        self.status_msg.emit(
            f"第 {page+1}/{self._n_pages} 页  ({cached}/{len(batch)} 已缓存，推断中…)")

    def _on_result(self, local_idx: int, pixmap: QPixmap, best_det):
        if local_idx >= len(self._cells):
            return
        global_idx   = self._page * _PAGE_SIZE + local_idx
        start        = self._page * _PAGE_SIZE
        batch        = self._paths[start: start + _PAGE_SIZE]
        if local_idx >= len(batch):
            return

        cell = self._cells[local_idx]
        cell.set_result(pixmap, best_det)
        if global_idx == self._selected:
            cell.set_selected(True)

        # 更新列表项颜色
        item = self._list.item(global_idx)
        if item:
            if best_det and best_det[0] >= _CONF_ACCEPT:
                item.setForeground(QColor('#00cc44'))
            elif best_det:
                item.setForeground(QColor('#ff8800'))
            else:
                item.setForeground(QColor('#cc4444'))

    def _on_page_done(self):
        start = self._page * _PAGE_SIZE
        batch = self._paths[start: start + _PAGE_SIZE]
        hit   = sum(1 for p in batch
                    if self._cache.get(p) and self._cache[p][0] >= _CONF_ACCEPT)
        self.status_msg.emit(
            f"第 {self._page+1}/{self._n_pages} 页  "
            f"命中 {hit}/{len(batch)}  (conf≥{_CONF_ACCEPT})")

    # ── 选中联动 ─────────────────────────────────────────────────────────────

    def _select(self, global_idx: int):
        """统一选中入口：更新列表、导航页面、高亮单元格。"""
        if global_idx < 0 or global_idx >= len(self._paths):
            return

        prev_selected = self._selected
        self._selected = global_idx

        target_page = global_idx // _PAGE_SIZE
        if target_page != self._page:
            # 取消旧页高亮（页面会被重建，无需操作）
            self._load_page(target_page)
        else:
            # 同页内：取消旧高亮，设置新高亮
            if prev_selected >= 0:
                old_local = prev_selected - self._page * _PAGE_SIZE
                if 0 <= old_local < len(self._cells):
                    self._cells[old_local].set_selected(False)
            new_local = global_idx - self._page * _PAGE_SIZE
            if 0 <= new_local < len(self._cells):
                self._cells[new_local].set_selected(True)

        # 同步列表（阻断信号避免递归）
        self._list.blockSignals(True)
        self._list.setCurrentRow(global_idx)
        self._list.scrollToItem(self._list.item(global_idx),
                                QAbstractItemView.EnsureVisible)
        self._list.blockSignals(False)

        self.status_msg.emit(
            f"第 {self._page+1}/{self._n_pages} 页  "
            f"选中: {self._paths[global_idx].name}")

    def _on_list_select(self, row: int):
        if row >= 0:
            self._select(row)

    def _on_cell_click(self, local_idx: int):
        global_idx = self._page * _PAGE_SIZE + local_idx
        start      = self._page * _PAGE_SIZE
        if local_idx < len(self._paths[start: start + _PAGE_SIZE]):
            self._select(global_idx)

    # ── 翻页 ─────────────────────────────────────────────────────────────────

    def _prev_page(self):
        if self._page > 0:
            self._load_page(self._page - 1)

    def _next_page(self):
        if self._page < self._n_pages - 1:
            self._load_page(self._page + 1)

    def _update_nav(self):
        self._page_lbl.setText(f"第 {self._page + 1} / {self._n_pages} 页")
        self._btn_prev.setEnabled(self._page > 0)
        self._btn_next.setEnabled(self._page < self._n_pages - 1)

    def start(self):
        """延迟启动：Tab 首次显示时调用，触发第 0 页推断。"""
        if not self._started:
            self._started = True
            self._load_page(0)

    def _stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait()


# ── 主窗口 ────────────────────────────────────────────────────────────────────

class EvalApp(QMainWindow):
    def __init__(self, train_paths: list, val_paths: list, model, imgsz: int):
        super().__init__()
        self.setWindowTitle("eval_ball — 网球检测评测")
        self.resize(1440, 920)

        self._eval_tabs: list[_EvalTab] = []
        tabs = QTabWidget()

        for paths, label in ((train_paths, "训练集"), (val_paths, "验证集")):
            if not paths:
                continue
            tab = _EvalTab(paths, model, imgsz)
            tab.status_msg.connect(
                lambda msg, lbl=label: self._status.showMessage(
                    f"[{lbl}]  {msg}"
                    f"   ▪ 绿框/边=GT/命中  ▪ 橙=低置信  ▪ 黄边=选中"))
            self._eval_tabs.append(tab)
            tabs.addTab(tab, f"{label}  ({len(paths)})")

        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)
        self._tabs_widget = tabs
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            f"训练集 {len(train_paths)} 张 | 验证集 {len(val_paths)} 张"
            f"   ▪ 绿框/边=GT/命中  ▪ 橙=低置信  ▪ 黄边=选中")

    def _on_tab_changed(self, index: int):
        if 0 <= index < len(self._eval_tabs):
            self._eval_tabs[index].start()

    def showEvent(self, event):
        super().showEvent(event)
        # 启动第一个 Tab 的推断
        self._on_tab_changed(self._tabs_widget.currentIndex())

    def closeEvent(self, event):
        for tab in self._eval_tabs:
            tab._stop_worker()
        super().closeEvent(event)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',  required=True,  help='data.yaml 路径')
    p.add_argument('--model', default=str(Path(__file__).parent / 'models/yolo26n-ball.pt'),
                   help='YOLO 模型路径')
    p.add_argument('--imgsz', type=int, default=96, help='推断图片尺寸')
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    yaml_path = Path(args.data)
    if not yaml_path.exists():
        print(f"Error: 找不到 {yaml_path}", file=sys.stderr); sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: 找不到模型 {model_path}", file=sys.stderr); sys.exit(1)

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER as _ul_logger
    _prev = _ul_logger.level
    _ul_logger.setLevel(logging.WARNING)
    model = YOLO(str(model_path), verbose=False)
    _ul_logger.setLevel(_prev)

    train_paths, val_paths = _resolve_images(yaml_path)
    print(f"训练集: {len(train_paths)} 张  验证集: {len(val_paths)} 张")
    if not train_paths and not val_paths:
        print("Error: 未找到任何图片", file=sys.stderr); sys.exit(1)

    app = QApplication(sys.argv)
    win = EvalApp(train_paths, val_paths, model, args.imgsz)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
