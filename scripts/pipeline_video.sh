#!/usr/bin/env bash
# 对单个视频运行完整三阶段流水线：detect.py → parse.py → render.py
#
# 用法:  bash scripts/pipeline_video.sh <视频文件名>
# 示例:  bash scripts/pipeline_video.sh 20260328_CB_M01_S01_G01_R01.mp4
#
# 脚本会在 ../datasets/tennis-video-26 下递归查找该文件名，
# 输出镜像到 runs/in-out/tennis-video-26.out/ 下相同子目录。

set -euo pipefail

DATASET_DIR="../datasets/tennis-video-26"
OUTPUT_ROOT="runs/in-out/tennis-video-26.out"
OBJECT_MODEL="models/yolo26x-finetuned.pt"
COURT_MODEL="models/yolov8n-seg-finetuned.pt"

if [[ $# -eq 0 ]]; then
  echo "用法: bash scripts/pipeline_video.sh <视频文件名>" >&2
  echo "示例: bash scripts/pipeline_video.sh 20260328_CB_M01_S01_G01_R01.mp4" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

target_name="$1"
video="$(find "$DATASET_DIR" -name "$target_name" | head -1)"

if [[ -z "$video" ]]; then
  echo "错误: 在 $DATASET_DIR 下未找到 $target_name" >&2
  exit 1
fi

rel="${video#$DATASET_DIR/}"
stem="${rel%.*}"
detect_json="$OUTPUT_ROOT/${stem}.detect.json"
parse_json="$OUTPUT_ROOT/${stem}.parse.json"
render_mp4="$OUTPUT_ROOT/${stem}.mp4"

echo "════════════════════════════════════════════════════════════"
echo "视频:   $video"
echo "detect: $detect_json"
echo "parse:  $parse_json"
echo "render: $render_mp4"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$(dirname "$detect_json")"

echo ""
echo "[1/3] detect"
echo "\$ .venv/bin/python detect.py -i $video -o $detect_json -m $OBJECT_MODEL -s $COURT_MODEL"
.venv/bin/python detect.py \
  -i "$video" -o "$detect_json" \
  -m "$OBJECT_MODEL" -s "$COURT_MODEL"

echo ""
echo "[2/3] parse"
echo "\$ .venv/bin/python parse.py -i $detect_json -o $parse_json"
.venv/bin/python parse.py -i "$detect_json" -o "$parse_json"

echo ""
echo "[3/3] render"
echo "\$ .venv/bin/python render.py -i $video -j $parse_json -o $render_mp4"
.venv/bin/python render.py -i "$video" -j "$parse_json" -o "$render_mp4"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "完成: $render_mp4"
