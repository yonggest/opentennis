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
# 评估原始模型基线
.venv/bin/python eval_detect.py --data <dataset>/data.yaml

# 微调
.venv/bin/python train_detect.py --data <dataset>/data.yaml

# 评估微调后模型
.venv/bin/python eval_detect.py \
  --model runs/detect/<run_name>/weights/best.pt \
  --data <dataset>/data.yaml
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
  --epochs 100 --lr0 0.001
```

复制最佳权重：

```bash
cp runs/segment/<run_name>/weights/best.pt models/court_seg.pt
```

---

## YOLO 训练参数说明

"本项目默认值" 指 `train_detect.py` 中硬编码或命令行默认值；"ultralytics 默认值" 指不传该参数时 ultralytics 自动使用的值。`—` 表示本项目未设置（使用 ultralytics 默认值）。

### 基础参数

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `model` | models/yolo26x.pt | — | 预训练权重路径，支持 .pt / .yaml |
| `data` | —（必填） | — | 数据集配置文件路径（data.yaml），定义 train/val 路径和类别名 |
| `epochs` | 100 | 100 | 训练总轮数，每轮遍历一次全部训练数据 |
| `time` | — | None | 最大训练时长（小时），设置后覆盖 `epochs` |
| `batch` | 1 | 16 | 每次梯度更新用的图片数，越大梯度越稳定但显存占用越高；`-1` 自动推断最大可用 batch |
| `imgsz` | 1920 | 640 | 训练图片长边尺寸，图片等比缩放到此尺寸 |
| `device` | auto | None（自动） | 训练设备：`0`=第一块 GPU，`0,1`=多 GPU，`mps`=Apple 芯片，`cpu`=CPU |
| `workers` | — | 8 | 数据加载子进程数，Windows 上若出错可设为 0 |
| `project` | runs/{task} | runs/detect | 输出根目录 |
| `name` | {dataset}-{timestamp} | train | 运行名称，输出目录 = `project/name/` |
| `exist_ok` | False | False | 是否允许覆盖已有的同名运行目录 |
| `pretrained` | True | True | 是否加载预训练权重；微调时必须为 True |
| `seed` | — | 0 | 随机种子，用于复现实验 |
| `deterministic` | — | True | 启用确定性模式（固定 cuDNN 算法），牺牲少量速度换取可复现性 |
| `verbose` | — | False | 是否打印每个 epoch 的详细日志 |
| `single_cls` | — | False | 把所有类别当作一个类训练，用于只关心"有没有目标"的场景 |
| `classes` | — | None | 只训练指定类别 ID 列表，如 `[0, 2]`；None 表示所有类 |
| `rect` | — | False | 矩形训练：按图片宽高比排序后组 batch，减少 padding，但不能与 mosaic 同用 |
| `multi_scale` | — | False | 多尺度训练：每批随机在 `imgsz ± 50%` 范围内变换尺寸，增强尺度鲁棒性 |
| `cos_lr` | — | False | 余弦学习率调度：学习率按余弦曲线从 lr0 衰减到 lr0×lrf，比线性衰减更平滑 |
| `close_mosaic` | — | 10 | 最后 N 个 epoch 关闭 mosaic 增强，让模型在真实分布上收敛 |
| `resume` | — | False | 从上次中断的 checkpoint 恢复训练，传入 last.pt 路径或 True |
| `amp` | — | True | 自动混合精度（FP16）：显著减少显存占用并加速训练，极少数情况下可能导致不稳定 |
| `fraction` | — | 1.0 | 实际使用数据集的比例（0–1），用于快速实验 |
| `profile` | — | False | 记录 ONNX / TensorRT 速度 benchmark 供架构对比 |
| `freeze` | 23 | None（0） | 冻结前 N 层不更新权重。YOLO26x 共 24 层（0–23）：`freeze=23` 只训练 Detect head，`freeze=11` 训练 neck + head，`freeze=0` 全部训练 |
| `max_det` | — | 300 | 每张图最多输出的检测框数 |
| `val` | — | True | 每个 epoch 结束后在验证集上评估；False 只训练不验证 |
| `cache` | False | False | 缓存图片到 RAM（`True`）或磁盘（`"disk"`），加速训练但占用额外内存 |

### 学习率

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `optimizer` | AdamW | auto（SGD） | 优化器：`AdamW` 适合微调，`SGD` 适合从头训练；`auto` 根据模型自动选 |
| `lr0` | 0.001 | 0.01 | 初始学习率，微调时应比从头训练小 10–100 倍 |
| `lrf` | 0.1 | 0.01 | 最终学习率系数，最终 lr = lr0 × lrf，控制衰减终点 |
| `momentum` | — | 0.937 | SGD 动量 / Adam beta1，控制梯度历史的影响权重 |
| `weight_decay` | — | 0.0005 | L2 正则化系数，防止过拟合，对 bias/BN 参数无效 |
| `warmup_epochs` | 1 | 3.0 | 前 N 轮从 0 线性升温到 lr0，防止训练初期梯度爆炸；freeze > 0 时建议改为 1 |
| `warmup_momentum` | — | 0.8 | warmup 阶段的初始动量，逐渐升至 `momentum` |
| `warmup_bias_lr` | — | 0.1 | warmup 阶段偏置参数的初始学习率，逐渐降至 lr0 |

### 损失函数权重

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `box` | — | 7.5 | 边界框回归损失（CIoU）权重，增大使定位更精准 |
| `cls` | — | 0.5 | 分类损失（BCE）权重，增大使分类更准确 |
| `dfl` | — | 1.5 | Distribution Focal Loss 权重，用于精细化边界框分布估计 |

### 正则化 / 训练策略

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `patience` | 20 | 50 | Early stopping：val mAP 连续 N 个 epoch 不提升则停止；0 表示禁用 |
| `dropout` | — | 0.0 | 分类头 dropout 率，防止过拟合，仅对分类任务有效 |

### 保存

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `save` | True | True | 是否保存 checkpoint（best.pt / last.pt 始终保存） |
| `save_period` | 5 | -1（禁用） | 每 N 个 epoch 额外保存一次；-1 表示只保存 best / last |
| `plots` | False | True | 是否生成训练曲线、混淆矩阵等图表；服务器无 GUI 时建议关闭 |

### 数据增强

| 参数 | 本项目默认值 | ultralytics 默认值 | 说明 |
|---|---|---|---|
| `hsv_h` | 0.0 | 0.015 | 色调随机扰动幅度（比例），增强颜色鲁棒性 |
| `hsv_s` | 0.1 | 0.7 | 饱和度随机扰动幅度 |
| `hsv_v` | 0.2 | 0.4 | 亮度随机扰动幅度，增强光照鲁棒性 |
| `degrees` | 0.0 | 0.0 | 随机旋转角度范围（±degrees），0 表示禁用 |
| `translate` | 0.1 | 0.1 | 随机平移幅度（图片尺寸的比例） |
| `scale` | 0.0 | 0.5 | 随机缩放幅度（增益范围 1±scale） |
| `shear` | — | 0.0 | 随机剪切角度（±degrees），模拟倾斜视角 |
| `perspective` | — | 0.0 | 随机透视变换幅度（0–0.001），模拟相机角度变化 |
| `flipud` | — | 0.0 | 垂直翻转概率，在重力对称场景（如卫星图）中有用 |
| `fliplr` | 0.0 | 0.5 | 水平翻转概率；球场左右对称时可设为 0.5 |
| `bgr` | — | 0.0 | BGR 通道翻转概率，模拟 BGR/RGB 输入误配 |
| `mosaic` | 0.0 | 1.0 | Mosaic 增强概率：把 4 张图拼成 1 张，丰富背景和尺度；小数据集效果显著 |
| `mixup` | 0.0 | 0.0 | MixUp 增强概率：两张图像素级叠加，一般只在大数据集使用 |
| `copy_paste` | 0.0 | 0.0 | 实例复制粘贴概率：把一张图的物体随机贴到另一张，需要分割标注 |
| `copy_paste_mode` | — | "flip" | 复制粘贴的来源方式：`"flip"`=水平翻转本图，`"mixup"`=从其他图取 |
| `auto_augment` | — | "randaugment" | 自动增强策略（分类任务）：`"randaugment"` / `"autoaugment"` / `"augmix"` |
| `erasing` | — | 0.4 | 随机擦除概率（分类任务）：随机遮盖图片中的矩形区域，增强遮挡鲁棒性 |
| `crop_fraction` | — | 1.0 | 分类任务中裁剪比例（0–1），1.0 表示不裁剪 |

### 微调参数选择指南

| 场景 | freeze | lr0 | epochs | 增强建议 |
|---|---|---|---|---|
| 小数据集域适应（< 200 张/类） | 23 | 1e-3 | 30–50 | fliplr=0.5, scale=0.3 |
| 中等数据集（200–500 张/类） | 11 | 1e-4 | 50–100 | + hsv_v=0.3, mosaic=0.5 |
| 大数据集 / 新类别（500+ 张/类） | 0 | 1e-5 | 100–200 | 全部开启 |
