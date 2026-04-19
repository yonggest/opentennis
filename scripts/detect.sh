#!/usr/bin/env bash
# detect.py 的便捷包装，内置模型路径和参数。
#
# 用法:
#   bash detect.sh <视频文件> <输出.json>       # 单文件
#   bash detect.sh <视频目录> <输出目录>         # 批量
#
# 示例:
#   bash detect.sh ../datasets/tennis-video-26/trials/footprint.mp4 out/footprint.detected.json
#   bash detect.sh ../datasets/tennis-video-26/games/ out/games/

set -euo pipefail

VIDEO_EXTS="mp4 MP4 mov MOV avi AVI mkv MKV"

# ── 参数检查 ──────────────────────────────────────────────────────────────────
if [[ $# -ne 2 ]]; then
  echo "用法: bash detect.sh <视频文件|视频目录> <输出.json|输出目录>" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

cd "$(dirname "$0")/.."

# ── 模型选择（macOS 用 CoreML，其他用 .pt）────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
  OBJECT_MODEL="models/yolo26x.mlpackage"
else
  OBJECT_MODEL="models/yolo26x.pt"
fi
COURT_MODEL="models/yolo26n-seg-tuned.pt"
CONF=0.1

echo "════════════════════════════════════════════════════════════"
echo "  object model  $OBJECT_MODEL"
echo "  court model   $COURT_MODEL"
echo "  conf          $CONF"
echo "════════════════════════════════════════════════════════════"

# ── 收集 (视频, 输出json) 列表 ────────────────────────────────────────────────
declare -a video_list=()
declare -a json_list=()

if [[ -f "$INPUT" ]]; then
  # 单文件模式：output 是 json 文件路径
  video_list+=("$INPUT")
  json_list+=("$OUTPUT")
elif [[ -d "$INPUT" ]]; then
  # 批量模式：output 是目录
  find_args=("$INPUT" -type f \()
  first=1
  for ext in $VIDEO_EXTS; do
    if [[ $first -eq 1 ]]; then
      find_args+=(-name "*.$ext"); first=0
    else
      find_args+=(-o -name "*.$ext")
    fi
  done
  find_args+=(\))
  while IFS= read -r f; do
    stem="$(basename "${f%.*}")"
    video_list+=("$f")
    json_list+=("$OUTPUT/${stem}.detected.json")
  done < <(find "${find_args[@]}" | sort)
else
  echo "错误: 输入路径不存在: $INPUT" >&2
  exit 1
fi

if [[ ${#video_list[@]} -eq 0 ]]; then
  echo "错误: 未找到任何视频文件" >&2
  exit 1
fi

echo "  找到 ${#video_list[@]} 个视频文件"
echo "════════════════════════════════════════════════════════════"

# ── 逐个处理 ──────────────────────────────────────────────────────────────────
ok=0
fail=0
for i in "${!video_list[@]}"; do
  video="${video_list[$i]}"
  out_json="${json_list[$i]}"
  mkdir -p "$(dirname "$out_json")"

  echo ""
  echo "── $video"
  echo "   → $out_json"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python detect.py\033[0m -i \"$video\" -o \"$out_json\" -m \"$OBJECT_MODEL\" -s \"$COURT_MODEL\" -c $CONF"

  if .venv/bin/python detect.py \
      -i "$video" \
      -o "$out_json" \
      -m "$OBJECT_MODEL" \
      -s "$COURT_MODEL" \
      -c "$CONF"; then
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
