"""
训练目标检测模型（球员 / 球拍 / 网球）。

用法:
    python train_detect.py --data <data.yaml>
    python train_detect.py --data <data.yaml> --epochs 50 --device 0
"""

import sys
import argparse

from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default=str(Path(__file__).parent / "models/yolo26x.pt"), help="预训练权重路径")
    p.add_argument("--data",    required=True,               help="数据集配置文件（如 datasets/xxx-yolo/data.yaml）")
    p.add_argument("--lr0",     type=float, default=0.001,   help="初始学习率（微调时比默认小）")
    p.add_argument("--epochs",  type=int,   default=50,     help="训练轮数")
    p.add_argument("--device",  default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model          {args.model}")
    print(f"  data           {args.data}")
    print(f"  device         {args.device or 'auto'}")
    print(f"  epochs         {args.epochs}")
    print(f"  batch          1")
    print(f"  imgsz          1920")
    print(f"  lr0            {args.lr0}")
    print(f"  lrf            0.01")
    print(f"  optimizer      AdamW")
    print(f"  freeze         23")
    print(f"  warmup_epochs  1")
    print(f"  patience       20")
    print(f"  save_period    5")
    print(f"  pretrained     True")
    print(f"  plots          False")
    print(f"  cache          False")
    print(f"  augmentation   hsv_h=0  hsv_s=0.1  hsv_v=0.2  degrees=0")
    print(f"                 translate=0.1  scale=0.25  fliplr=0.25")
    print(f"                 mosaic=0  mixup=0  copy_paste=0")
    print("─" * 60, flush=True)

    if not Path(args.model).exists():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"找不到数据配置: {args.data}  请先运行 coco2yolo.py 和 split_dataset.py")

    dataset_name = Path(args.data).parent.name or Path(args.data).stem  # e.g. huanglong-yolo/data.yaml → huanglong-yolo
    run_name = f"{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model = YOLO(args.model)
    project_dir = str(Path(__file__).parent / "runs" / model.task)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=1,
        lr0=args.lr0,
        imgsz=1920,
        freeze=23,           # 冻结 backbone + neck，只微调 Detect head
        lrf=0.01,            # 最终学习率 = lr0 * lrf
        warmup_epochs=1,     # freeze > 0 只训练 head 时，建议改为 1
        patience=20,         # Early stopping
        save=True,
        save_period=5,
        cache=False,         # 内存不足时保持 False；充裕时改 True 加速
        device=args.device or None,
        project=project_dir,
        name=run_name,
        exist_ok=False,
        pretrained=True,
        plots=False,
        optimizer="AdamW",
        # 数据增强（微调时适度降低）
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.1,
        fliplr=0.25,
        mosaic=0.0,
        scale=0.25,
        mixup=0.0,
        copy_paste=0.0,
    )

    print("\n训练完成！")
    print(f"最佳权重: {project_dir}/{run_name}/weights/best.pt")
    for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
        val = results.results_dict.get(key)
        print(f"{label}:    {val:.4f}" if val is not None else f"{label}:    N/A")


if __name__ == "__main__":
    main()
