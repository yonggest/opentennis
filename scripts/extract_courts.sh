#!/usr/bin/env bash
# extract_court.py 的批量包装。
#
# 对输入目录中的每个视频分别运行 extract_court.py，
# 每个视频的输出放在独立子目录中。
#
# 用法:
#   bash extract_courts.sh <视频目录> <输出根目录> [--pos N] [--neg N] [--seg-model PT]
#
# 示例:
#   bash extract_courts.sh input_videos/ datasets/mycourt
#   bash extract_courts.sh input_videos/ datasets/mycourt --pos 100 --neg 10

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "" >&2
  echo "用法: extract_courts.sh <视频目录> <输出根目录> [--pos N] [--neg N] [--seg-model PT]" >&2
  echo "" >&2
  exit 1
fi

VIDEO_DIR="$1"
OUT_ROOT="$2"
shift 2
EXTRA_ARGS=("$@")   # 剩余参数原样传给 extract_court.py

if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "错误: 目录不存在: $VIDEO_DIR" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# 收集所有视频
mapfile -t VIDEOS < <(find "$VIDEO_DIR" -maxdepth 1 \( -name "*.mp4" -o -name "*.MP4" -o -name "*.avi" -o -name "*.AVI" \) | sort)

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
  echo "错误: 未找到 .mp4/.avi 文件: $VIDEO_DIR" >&2
  exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  视频目录  $VIDEO_DIR"
echo "  输出根目录 $OUT_ROOT"
echo "  视频数量  ${#VIDEOS[@]}"
echo "════════════════════════════════════════════════════════════"

ok=0; fail=0
for video in "${VIDEOS[@]}"; do
  stem="$(basename "$video")"
  stem="${stem%.*}"
  out_dir="$OUT_ROOT/$stem"

  echo ""
  echo "── $stem"
  echo "   → $out_dir"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python extract_court.py\033[0m -i \"$video\" -o \"$out_dir\" --pos 100 ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"

  if .venv/bin/python extract_court.py -i "$video" -o "$out_dir" --pos 100 "${EXTRA_ARGS[@]}"; then
    ok=$((ok + 1))
  else
    echo "  [FAILED] $video" >&2
    fail=$((fail + 1))
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  完成: $ok 成功  $fail 失败"
echo "════════════════════════════════════════════════════════════"
