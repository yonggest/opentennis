"""
用 tennis 数据集微调 YOLO26x 模型。

用法:
    python train_yolo.py --data datasets/xxx-yolo/data.yaml
    python train_yolo.py --data datasets/xxx-yolo/data.yaml --epochs 50 --batch 4 --device mps
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default="models/yolo26x.pt", help="预训练权重路径")
    p.add_argument("--data",    required=True,                help="数据集配置文件（如 datasets/xxx-yolo/data.yaml）")
    p.add_argument("--epochs",  type=int,   default=100,     help="训练轮数")
    p.add_argument("--batch",   type=int,   default=2,       help="batch size，1920px 时建议 2-4")
    p.add_argument("--imgsz",   type=int,   default=1920,    help="输入图片尺寸")
    p.add_argument("--lr0",     type=float, default=0.001,   help="初始学习率（微调时比默认小）")
    p.add_argument("--freeze",  type=int,   default=10,      help="冻结前 N 层 backbone，0=不冻结")
    p.add_argument("--project", default="finetune",          help="输出根目录")
    p.add_argument("--name",    default="tennis-01",         help="本次训练子目录名")
    p.add_argument("--device",  default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"找不到数据配置: {args.data}  请先运行 coco2yolo.py 和 split_dataset.py")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=0.01,            # 最终学习率 = lr0 * lrf
        warmup_epochs=3,
        freeze=args.freeze,  # 冻结 backbone 前 N 层，只微调 head
        patience=20,         # Early stopping
        save=True,
        save_period=10,
        cache=False,         # 内存不足时保持 False；充裕时改 True 加速
        device=args.device or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        # 数据增强（微调时适度降低）
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.1,
        fliplr=0.0,
        mosaic=0.0,
        scale=0.0,
        mixup=0.1,
        copy_paste=0.0,
    )

    print("\n✅ 训练完成！")
    print(f"最佳权重: {args.project}/{args.name}/weights/best.pt")
    for key, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95")]:
        val = results.results_dict.get(key)
        print(f"{label}:    {val:.4f}" if val is not None else f"{label}:    N/A")


if __name__ == "__main__":
    main()
