#!/usr/bin/env bash
# 从 tracked JSON 中提取验证器训练数据（四类样本）。
#
# 输入：.tracked.json 文件
# 输出：四个子目录，与输入文件同级
#   <stem>_正常样本   — 所有位置、所有类别、高置信度正常样本
#   <stem>_插值背景   — 背景板附近的插值网球
#   <stem>_插值网带   — 网带附近的插值网球
#   <stem>_插值球拍   — 球拍附近的插值网球

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "" >&2
  echo "% extract_balls.sh game.tracked.json" >&2
  echo "" >&2
  exit 1
fi

JSON="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

if [[ ! -f "$JSON" ]]; then
  echo "错误: 找不到文件: $JSON" >&2
  exit 1
fi

STEM="$(basename "$JSON" .tracked.json)"
DIR="$(dirname "$JSON")"

cd "$(dirname "$0")/.."

echo "════════════════════════════════════════════════════════════"
echo "  JSON   $JSON"
echo "════════════════════════════════════════════════════════════"

run() {
  local label="$1"; shift
  local out_dir="${DIR}/${STEM}_${label}"
  echo ""
  echo "── $label"
  echo "   → $out_dir"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python extract_dataset.py\033[0m $* -o \"$out_dir\""
  .venv/bin/python extract_dataset.py "$@" -o "$out_dir"
}

run "正常样本" -i "$JSON" -p all    --sample high-conf
run "插值背景" -i "$JSON" -p backdrop --sample interpolated --category "sports ball"
run "插值网带" -i "$JSON" -p net      --sample interpolated --category "sports ball"
run "插值球拍" -i "$JSON" -p racket   --sample interpolated --category "sports ball"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  完成"
echo "════════════════════════════════════════════════════════════"
