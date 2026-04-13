#!/usr/bin/env bash
# 批量对 tennis-video-26 所有视频运行完整三阶段流水线：
#   detect.py → parse.py → render.py
#
# 输入:  ../datasets/tennis-video-26/**/*.mp4
# 输出:  runs/in-out/tennis-video-26.out/**/<stem>.detect.json
#        runs/in-out/tennis-video-26.out/**/<stem>.parse.json
#        runs/in-out/tennis-video-26.out/**/<stem>.mp4
# 用法:  bash scripts/pipeline_tennis_video_26.sh [--dry-run]

set -uo pipefail

DATASET_DIR="../datasets/tennis-video-26"
OUTPUT_ROOT="runs/in-out/tennis-video-26.out"
OUTPUT_ROOT="runs/in-out/tennis-video-26.out"
OBJECT_MODEL="models/yolo26x-finetuned.pt"
COURT_MODEL="models/yolov8n-seg-finetuned.pt"

DRY_RUN=false

for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

cd "$(dirname "$0")/.."

mapfile -t videos < <(find "$DATASET_DIR" -name "*.mp4" | sort)

if [[ ${#videos[@]} -eq 0 ]]; then
  echo "未找到视频文件: $DATASET_DIR" >&2
  exit 1
fi

echo "共 ${#videos[@]} 个视频"
echo "════════════════════════════════════════════════════════════"

failed=()

for video in "${videos[@]}"; do
  rel="${video#$DATASET_DIR/}"
  stem="${rel%.*}"
  detect_json="$OUTPUT_ROOT/${stem}.detect.json"
  parse_json="$OUTPUT_ROOT/${stem}.parse.json"
  render_mp4="$OUTPUT_ROOT/${stem}.mp4"

  echo ""
  echo "── $rel"

  if $DRY_RUN; then
    echo "  [1] .venv/bin/python detect.py -i $video -o $detect_json -m $OBJECT_MODEL -s $COURT_MODEL"
    echo "  [2] .venv/bin/python parse.py -i $detect_json -o $parse_json"
    echo "  [3] .venv/bin/python render.py -i $video -j $parse_json -o $render_mp4"
    continue
  fi

  mkdir -p "$(dirname "$detect_json")"
  ok=true

  echo ""
  echo "  \$ .venv/bin/python detect.py -i $video -o $detect_json -m $OBJECT_MODEL -s $COURT_MODEL"
  if ! .venv/bin/python detect.py \
      -i "$video" -o "$detect_json" \
      -m "$OBJECT_MODEL" -s "$COURT_MODEL"; then
    echo "  ✗ detect 失败: $rel" >&2
    failed+=("$rel (detect)")
    ok=false
  fi

  if $ok; then
    echo ""
    echo "  \$ .venv/bin/python parse.py -i $detect_json -o $parse_json"
    if ! .venv/bin/python parse.py -i "$detect_json" -o "$parse_json"; then
      echo "  ✗ parse 失败: $rel" >&2
      failed+=("$rel (parse)")
      ok=false
    fi
  fi

  if $ok; then
    echo ""
    echo "  \$ .venv/bin/python render.py -i $video -j $parse_json -o $render_mp4"
    if ! .venv/bin/python render.py \
        -i "$video" -j "$parse_json" -o "$render_mp4"; then
      echo "  ✗ render 失败: $rel" >&2
      failed+=("$rel (render)")
      ok=false
    fi
  fi

  echo ""
  $ok && echo "  ✓ done"
done

echo ""
echo "════════════════════════════════════════════════════════════"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "完成，${#failed[@]} 个失败："
  for f in "${failed[@]}"; do
    echo "  ✗ $f"
  done
  exit 1
else
  echo "全部完成"
fi
