#!/usr/bin/env bash
# 对单个视频运行完整四阶段流水线：detect.py → track.py → parse.py → render.py
#
# 用法:  bash scripts/pipeline_video.sh [物体模型] <视频文件名>
# 示例:  bash scripts/pipeline_video.sh 20260328_CB_M01_S01_G01_R01.mp4
#        bash scripts/pipeline_video.sh models/yolo26x.pt 20260328_CB_M01_S01_G01_R01.mp4
#
# 脚本会在 ../datasets/tennis-video-26 下递归查找该文件名，
# 输出镜像到 runs/in-out/tennis-video-26.out/ 下相同子目录。

set -euo pipefail

DATASET_DIR="../datasets/tennis-video-26"
OUTPUT_ROOT="runs/in-out/tennis-video-26.out"
COURT_MODEL="models/yolov8n-seg-finetuned.pt"
CONF_HIGH=0.5   # 高置信度阈值：>= 此值的检测可新建轨迹
CONF_LOW=0.1    # 低置信度下限：detect.py 的 -c，[low,high) 的检测仅续接已有轨迹

if [[ $# -eq 0 ]]; then
  echo "用法: bash scripts/pipeline_video.sh [物体模型] <视频文件名>" >&2
  echo "示例: bash scripts/pipeline_video.sh 20260328_CB_M01_S01_G01_R01.mp4" >&2
  echo "      bash scripts/pipeline_video.sh models/yolo26x.pt 20260328_CB_M01_S01_G01_R01.mp4" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

if [[ $# -eq 2 ]]; then
  OBJECT_MODEL="$1"
  target_name="$2"
else
  OBJECT_MODEL="models/yolo26x.pt"
  target_name="$1"
fi
video="$(find "$DATASET_DIR" -name "$target_name" | head -1)"

if [[ -z "$video" ]]; then
  echo "错误: 在 $DATASET_DIR 下未找到 $target_name" >&2
  exit 1
fi

rel="${video#$DATASET_DIR/}"
stem="${rel%.*}"
detected_json="$OUTPUT_ROOT/${stem}_detected.json"
tracked_json="$OUTPUT_ROOT/${stem}_tracked.json"
parsed_json="$OUTPUT_ROOT/${stem}_parsed.json"
render_mp4="$OUTPUT_ROOT/${stem}.mp4"

echo "════════════════════════════════════════════════════════════"
echo "视频:     $video"
echo "detected: $detected_json"
echo "tracked:  $tracked_json"
echo "parsed:   $parsed_json"
echo "render:   $render_mp4"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$(dirname "$detected_json")"

echo ""
echo "[1/4] detect"
echo "\$ .venv/bin/python detect.py -i $video -o $detected_json -m $OBJECT_MODEL -s $COURT_MODEL -c $CONF_LOW"
.venv/bin/python detect.py \
  -i "$video" -o "$detected_json" \
  -m "$OBJECT_MODEL" -s "$COURT_MODEL" -c "$CONF_LOW"

echo ""
echo "[2/4] track"
echo "\$ .venv/bin/python track.py -i $detected_json -o $tracked_json --conf-high $CONF_HIGH --conf-low $CONF_LOW"
.venv/bin/python track.py -i "$detected_json" -o "$tracked_json" \
  --conf-high "$CONF_HIGH" --conf-low "$CONF_LOW"

echo ""
echo "[3/4] parse"
echo "\$ .venv/bin/python parse.py -i $tracked_json -o $parsed_json"
.venv/bin/python parse.py -i "$tracked_json" -o "$parsed_json"

echo ""
echo "[4/4] render"
echo "\$ .venv/bin/python render.py -i $video -j $parsed_json -o $render_mp4"
.venv/bin/python render.py -i "$video" -j "$parsed_json" -o "$render_mp4"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "完成: $render_mp4"
