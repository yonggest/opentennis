#!/usr/bin/env bash
# 评估目标检测模型（球员 / 球拍 / 网球）
#
# 用法:  bash scripts/eval_detect.sh <data.yaml> <模型权重路径>
# 示例:  bash scripts/eval_detect.sh runs/in-out/tennis-track-26.yolo/data.yaml runs/detect/exp/detect/weights/best.pt

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: bash scripts/eval_detect.sh <data.yaml> <模型权重路径>" >&2
  echo "示例: bash scripts/eval_detect.sh runs/in-out/tennis-track-26.yolo/data.yaml runs/detect/exp/detect/weights/best.pt" >&2
  exit 1
fi

DATA="$1"
MODEL="$2"
IMGSZ=1920
CONF=0.5

cd "$(dirname "$0")/.."

CMD=".venv/bin/python eval_yolo.py \
  --model $MODEL \
  --data  $DATA \
  --imgsz $IMGSZ \
  --conf  $CONF"

echo "════════════════════════════════════════════════════════════"
echo "\$ $CMD"
echo "════════════════════════════════════════════════════════════"
echo ""

eval "$CMD"
