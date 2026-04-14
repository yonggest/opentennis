# opentennis

网球比赛视频分析流水线：检测球员、球拍、网球，检测球场关键点，过滤无效检测，输出标注视频。

## 环境

GTX 1080 Ti (sm_61) 不支持 PyTorch 2.x，需用 Python 3.10 + torch 1.13.1+cu117：

```bash
python3.10 -m venv .venv
.venv/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
.venv/bin/pip install ultralytics==8.4.35 opencv-python numpy==1.26.4 pandas scipy tqdm PySide6
```

---

## 分析流水线

### 第一阶段：检测（球场 + 物体）

```bash
.venv/bin/python detect.py \
  -i <video> \
  -m models/yolo26x.pt \
  -s models/court_seg.pt \
  -z 1920
```

输出：`<video>.json`（COCO 格式，含球场关键点和缓冲区凸包）

### 第二阶段：解析（球追踪 + 缓冲区过滤）

```bash
.venv/bin/python parse.py -i <video>.json
```

输出：`<video>_parsed.json`（同 COCO 格式，每个 annotation 加 `track_id` / `valid` 字段）

### 第三阶段：渲染

```bash
.venv/bin/python render.py -i <video> -j <video>_parsed.json -o <output>.mp4
```

`-j` 可传 detect.py 或 parse.py 的输出。输出：H.264 标注视频（crf=18）。

---

## 辅助工具

### 逐帧浏览（browse.py）

```bash
.venv/bin/python browse.py -v <video> -j <video>.json
```

可视化任意阶段的 JSON，支持缩放/平移/播放，无效标注用暗色 + X 显示。

### 提取训练帧（build_coco.py）

```bash
.venv/bin/python build_coco.py -i <video> -o frames/ -j <video>_parsed.json
```

从视频提取 JPEG 帧，将有效标注（`valid=True`）迁移为标准 COCO JSON（剔除 `track_id`/`valid`/`score` 等运行时字段），输出 `_annotations.coco.json`。

### 调试球场检测（debug_court.py）

```bash
.venv/bin/python debug_court.py -i <video>
```

输出：`<video>_debug/` 目录，包含各阶段中间结果（0_original ～ 9_step3_optimized）。

---

## JSON 格式

两个阶段共用同一 COCO 格式，向后兼容：

```
images       : [{id, width, height, frame_id}, ...]
annotations  : [{id, image_id, category_id, bbox [x,y,w,h], area, iscrowd, score,
                 track_id?,  ← parse.py 追加
                 valid?},    ← parse.py 追加（缺省视为 true）
                ...]
categories   : [{id, name, supercategory}, ...]
fps          : float
court        : {keypoints, ground_hull, volume_hull, vol_bottom_pts, vol_top_pts}
```

---

## 球场检测算法

球场检测的目标是求一个 3×3 单应矩阵 H，将球场米坐标映射到图像像素坐标，进而投影出 14 个关键点。

### 代价函数

将球场线条模板的所有白线像素通过 H 投影到图像，取落点处 `dist_map` 值的**透视加权平均**作为代价。`dist_map` 是图像中实际白线像素的距离变换（每个像素 = 到最近白线的距离，上限 cap）。

权重正比于每个模板点在图像中的局部面积放大倍数（单应矩阵在该点的 Jacobian 行列式绝对值）：近端线条在图像中占更大面积，权重更高，与肉眼感知一致。权重在优化开始前由初始 H 计算一次，整个优化过程固定不变。

### 三阶段检测流程

**步骤1 — YOLO seg 初始化**

用 `court_seg.pt` 对第一帧做实例分割，得到球场区域多边形。对凸包做近似，提取四个角点，用 `getPerspectiveTransform` 计算初始 H。这一步给出足够准确的起点，使后续优化不会陷入错误局部极小。

YOLO seg 的多边形经填充后膨胀（~5% 图像高度）作为 court_mask，限定后续白线检测范围。

**步骤2 — 粗 dist_map + Nelder-Mead 精调**

在 court_mask 内检测白线像素（HSV V>180, S<50），做距离变换得到 dist_map。以 dist_map 为目标函数，用 Nelder-Mead 优化 4 个角点的图像坐标（8 个参数），从步骤1的 H 出发精调。此阶段 mask 范围宽松，dist_map 可能含场地表面的假白点，但足以把 H 收敛到正确区域。

**步骤3 — 精 dist_map + 再次精调**

用步骤2的 H 将所有球场线条（含网线）投影回图像，按自然线宽绘制后膨胀（LINE_W = 5 cm），得到仅覆盖线条附近的带状 line_mask。在 line_mask 内重建 dist_map2，从步骤2的 H 出发再次 Nelder-Mead 精调。line_mask 排除了大面积场地表面的假白点，代价函数对 H 的偏移更敏感，最终结果更准确。

---

## 模型

| 文件 | 用途 |
|---|---|
| `models/yolo26x.pt` | 检测球员 / 球拍 / 网球（YOLOv8x fine-tuned） |
| `models/court_seg.pt` | 球场区域分割，用于初始化单应矩阵（YOLOv8n-seg） |

---

## 训练

### 物体检测模型（yolo26x）

```bash
.venv/bin/python train_detect.py \
  --model models/yolo26x.pt \
  --data <dataset>/data.yaml \
  --epochs 50 --batch 2 --imgsz 1920 \
  --name finetune
```

### 球场分割模型（court_seg）

数据准备（COCO → YOLO seg 格式）：

```bash
cd ../annotation
python coco2yolo.py \
  -i ../datasets/court-26 \
  -o ../opentennis/runs/datasets/court-26-yolo
```

> `coco2yolo.py` 只做格式转换，不拆分 train/val。train/val 需手动准备为独立目录后分别转换，或直接编辑生成的 `data.yaml` 指定路径。

训练：

```bash
.venv/bin/python train_detect.py \
  --model models/yolov8n-seg.pt \
  --data <yolo_dir>/data.yaml \
  --epochs 100 --batch 4 --imgsz 640 \
  --lr0 0.001 --freeze 0 \
  --name court-seg-v1
```

> `--freeze 0`：训练数据量小（~15张）且与 COCO 域差异大，需放开全部层充分学习。

复制最佳权重：

```bash
cp runs/segment/exp/court-seg-v1/weights/best.pt models/court_seg.pt
```

### 评估

```bash
# 物体检测
.venv/bin/python eval_detect.py --model models/yolo26x.pt --data <dataset>/data.yaml

# 球场分割
.venv/bin/python eval_detect.py \
  --model runs/segment/exp/court-seg-v1/weights/best.pt \
  --data runs/datasets/court-26-yolo/data.yaml \
  --imgsz 640
```
