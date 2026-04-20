"""
评测目标检测模型（使用 YOLO 内置 val，与训练过程一致）。

用法:
  python eval_detect.py --data <data.yaml>
  python eval_detect.py --data <data.yaml> --model best.pt
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",  default=str(Path(__file__).parent / "models/yolo26x.pt"), help="模型权重路径")
    p.add_argument("--data",   required=True,              help="数据集配置文件（data.yaml）")
    p.add_argument("--imgsz",  type=int, default=1920,     help="推理图片尺寸")
    p.add_argument("--device", default="",                 help="'mps'/'cpu'/'0'(CUDA)，留空自动选择")
    if len(sys.argv) == 1:
        p.print_help()
        sys.exit(0)
    return p.parse_args()


def _print_class_metrics(validator):
    try:
        m    = validator.metrics
        nt   = m.nt_per_class          # shape (nc,), GT instances per class
        seen = validator.seen
        if len(m.ap_class_index) == 0:
            return

        hf = "%22s%11s%11s%11s%11s%11s%11s"
        pf = "%22s%11i%11i%11.3g%11.3g%11.3g%11.3g"
        print(hf % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"))
        print(pf % ("all", seen, int(nt.sum()), m.box.mp, m.box.mr, m.box.map50, m.box.map))
        for i, c in enumerate(m.ap_class_index):
            name = m.names.get(int(c), str(c))
            print(pf % (name, seen, int(nt[int(c)]),
                        m.box.p[i], m.box.r[i], m.box.ap50[i], m.box.ap[i]))
    except Exception as e:
        print(f"  [per-class metrics] 打印失败: {e}")


def main():
    args = parse_args()

    print("─" * 60)
    print(f"  model    {args.model}")
    print(f"  data     {args.data}")
    print(f"  imgsz    {args.imgsz}")
    print(f"  conf     0.001  (ultralytics default, sweeps thresholds for AP)")
    print(f"  iou      0.6    (ultralytics default NMS IoU threshold)")
    print(f"  device   {args.device or 'auto'}")
    print("─" * 60, flush=True)

    model = YOLO(args.model)

    _captured = []
    def _capture(v): _captured.append(v)
    model.add_callback("on_val_end", _capture)
    model.val(data=args.data, imgsz=args.imgsz, batch=1, plots=False, device=args.device or None, verbose=False)

    print()
    if _captured:
        _print_class_metrics(_captured[0])
    else:
        print("  验证未返回结果")


if __name__ == "__main__":
    main()
