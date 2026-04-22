#!/usr/bin/env bash
# pose.py 的便捷包装。视频路径从 JSON 的 video 字段自动读取。
#
# 用法:
#   bash pose.sh <输入.parsed.json> <输出.posed.json>   # 单文件
#   bash pose.sh <输入目录> <输出目录>                   # 批量

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "" >&2
  echo "% pose.sh input.parsed.json  out.posed.json" >&2
  echo "% pose.sh parsed/            posed/" >&2
  echo "" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

cd "$(dirname "$0")/.."

POSE_MODEL="models/yolo26x-pose.pt"

declare -a in_list=()
declare -a out_list=()

if [[ -f "$INPUT" ]]; then
  in_list+=("$INPUT")
  out_list+=("$OUTPUT")
elif [[ -d "$INPUT" ]]; then
  while IFS= read -r f; do
    stem="$(basename "$f" .parsed.json)"
    in_list+=("$f")
    out_list+=("$OUTPUT/${stem}.posed.json")
  done < <(find "$INPUT" -maxdepth 1 -name "*.parsed.json" | sort)
else
  echo "错误: 输入路径不存在: $INPUT" >&2; exit 1
fi

if [[ ${#in_list[@]} -eq 0 ]]; then
  echo "错误: 未找到 *.parsed.json 文件" >&2; exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  pose model    $POSE_MODEL"
echo "  找到 ${#in_list[@]} 个文件"
echo "════════════════════════════════════════════════════════════"

ok=0; fail=0
for i in "${!in_list[@]}"; do
  in_json="${in_list[$i]}"
  out_json="${out_list[$i]}"
  mkdir -p "$(dirname "$out_json")"

  echo ""
  echo "── $(basename "$in_json")"
  echo "   → $out_json"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python pose.py\033[0m -i \"$in_json\" -m \"$POSE_MODEL\" -o \"$out_json\""

  if .venv/bin/python pose.py -i "$in_json" -m "$POSE_MODEL" -o "$out_json"; then
    ok=$((ok + 1))
  else
    echo "  [FAILED] $in_json" >&2
    fail=$((fail + 1))
  fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  完成: $ok 成功  $fail 失败"
echo "════════════════════════════════════════════════════════════"
