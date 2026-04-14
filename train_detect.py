"""
训练目标检测模型（球员 / 球拍 / 网球）。

用法:
    python train_detect.py --data <data.yaml>
    python train_detect.py --data <data.yaml> --name finetune --epochs 50 --batch 4 --device 0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",   default=str(Path(__file__).parent / "models/yolo26x.pt"), help="预训练权重路径")
    p.add_argument("--data",    required=True,                help="数据集配置文件（如 datasets/xxx-yolo/data.yaml）")
    p.add_argument("--epochs",  type=int,   default=100,     help="训练轮数")
    p.add_argument("--batch",   type=int,   default=2,       help="batch size，1920px 时建议 2-4")
    p.add_argument("--imgsz",   type=int,   default=1920,    help="输入图片尺寸")
    p.add_argument("--lr0",     type=float, default=0.001,   help="初始学习率（微调时比默认小）")
    p.add_argument("--freeze",  type=int,   default=10,      help="冻结前 N 层 backbone，0=不冻结")
    p.add_argument("--name",    default=None,                help="本次运行名，输出到 runs/{task}/exp/{name}/（默认：{数据集名}-{时间戳}）")
    p.add_argument("--device",  default="",                  help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}")
    print(f"  epochs   {args.epochs:<10}  batch    {args.batch:<10}  imgsz   {args.imgsz}")
    print(f"  lr0      {args.lr0:<10}  freeze   {args.freeze:<10}  device  {args.device or 'auto'}")
    print(f"  name     {args.name or '(auto)'}")
    print("─" * 60, flush=True)

    if not Path(args.model).exists():
        raise FileNotFoundError(f"找不到模型文件: {args.model}")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"找不到数据配置: {args.data}  请先运行 coco2yolo.py 和 split_dataset.py")

    dataset_name = Path(args.data).parent.name or Path(args.data).stem  # e.g. huanglong-yolo/data.yaml → huanglong-yolo
    run_name = args.name or f"{dataset_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model = YOLO(args.model)
    project_dir = str(Path(__file__).parent / "runs" / model.task / "exp")

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
        project=project_dir,
        name=run_name,
        exist_ok=False,
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
