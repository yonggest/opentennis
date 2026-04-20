#!/usr/bin/env python3
"""
训练专项网球检测器（单类 ball，小尺寸 patch 输入）。

训练样本由 build_ball_dataset.py 生成，每张图像为 patch_size×patch_size（默认 96）
的裁图，与 rescue 推断时的输入完全一致。

用法：
    python train_ball.py --data datasets/ball-96/data.yaml
    python train_ball.py --data datasets/ball-96/data.yaml --imgsz 96 --epochs 100
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from utils import pick_free_gpu


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',    required=True,               help='data.yaml 路径（build_ball_dataset.py 生成）')
    p.add_argument('--model',   default=str(Path(__file__).parent / 'models/yolo26n.pt'),
                                                             help='预训练权重（建议用小模型 yolo26n.pt）')
    p.add_argument('--imgsz',   type=int,   default=96,      help='训练图像尺寸（应与 patch_size 一致）')
    p.add_argument('--epochs',  type=int,   default=100,     help='训练轮数')
    p.add_argument('--lr0',     type=float, default=0.001,   help='初始学习率')
    p.add_argument('--device',  default='',                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help(); sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model      {args.model}")
    print(f"  data       {args.data}")
    print(f"  imgsz      {args.imgsz}")
    print(f"  epochs     {args.epochs}")
    print(f"  lr0        {args.lr0}")
    _gpu = pick_free_gpu() or 'framework default'
    print(f"  device     {args.device or f'auto → {_gpu}'}")
    print(f"  freeze     0  (全网络训练)")
    print(f"  augment    hsv_s=0.2 hsv_v=0.3  translate=0.1")
    print(f"             mosaic=0  scale=0  fliplr=0  mixup=0")
    print("─" * 60, flush=True)

    if not Path(args.model).exists():
        raise FileNotFoundError(f"找不到模型: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"找不到 data.yaml: {args.data}")

    device = args.device or pick_free_gpu()

    dataset_name = Path(args.data).parent.name
    run_name     = f"{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model        = YOLO(args.model)
    project_dir  = str(Path(__file__).parent / 'runs' / model.task)

    results = model.train(
        data        = args.data,
        epochs      = args.epochs,
        batch       = 16,           # patch 很小，batch 可以开大
        imgsz       = args.imgsz,
        lr0         = args.lr0,
        lrf         = 0.05,
        freeze      = 0,            # 全网络训练（单类小目标，需要充分适应）
        warmup_epochs = 3,
        patience    = 30,
        save        = True,
        save_period = 10,
        cache       = True,         # patch 小，可全量缓存到内存
        device      = device,
        project     = project_dir,
        name        = run_name,
        exist_ok    = False,
        pretrained  = True,
        plots       = False,
        optimizer   = 'AdamW',
        # 数据增强：保守设置，避免小目标 bbox 被破坏
        hsv_h       = 0.0,
        hsv_s       = 0.2,
        hsv_v       = 0.3,
        degrees     = 0.0,
        translate   = 0.1,
        scale       = 0.0,          # 禁用缩放（球已经很小）
        fliplr      = 0.0,          # 禁用翻转（场地有方向性）
        mosaic      = 0.0,          # 禁用（patch 已是小图，mosaic 无意义）
        mixup       = 0.0,
        copy_paste  = 0.0,
    )

    print("\n训练完成！")
    print(f"最佳权重: {project_dir}/{run_name}/weights/best.pt")
    for key, label in [('metrics/mAP50(B)', 'mAP50'),
                        ('metrics/mAP50-95(B)', 'mAP50-95'),
                        ('metrics/recall(B)', 'recall'),
                        ('metrics/precision(B)', 'precision')]:
        val = results.results_dict.get(key)
        print(f"  {label}: {val:.4f}" if val is not None else f"  {label}: N/A")


if __name__ == '__main__':
    main()
