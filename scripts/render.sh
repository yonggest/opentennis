#!/usr/bin/env bash
# render.py 的便捷包装。视频路径从 JSON 的 video 字段自动读取。
# 可接受任意阶段的 JSON（detected / tracked / parsed / posed），JSON 里有多少信息就渲染多少。
#
# 用法:
#   bash render.sh <输入.json> <输出.mp4>   # 单文件（任意阶段 JSON）
#   bash render.sh <输入目录> <输出目录>     # 批量（自动选每个 stem 最高阶段的 JSON）

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "" >&2
  echo "% render.sh input.parsed.json  out.mp4" >&2
  echo "% render.sh jsons/             videos/" >&2
  echo "" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

cd "$(dirname "$0")/.."

# 阶段优先级（高到低）
STAGES=(posed parsed tracked detected)

declare -a in_list=()
declare -a out_list=()

if [[ -f "$INPUT" ]]; then
  in_list+=("$INPUT")
  out_list+=("$OUTPUT")
elif [[ -d "$INPUT" ]]; then
  # 收集目录下所有 stem，每个 stem 选最高阶段
  declare -A best_json   # stem → json path
  declare -A best_rank   # stem → stage rank (lower = better)
  for rank in "${!STAGES[@]}"; do
    stage="${STAGES[$rank]}"
    while IFS= read -r f; do
      stem="$(basename "$f" ".${stage}.json")"
      if [[ -z "${best_rank[$stem]+x}" ]] || (( rank < best_rank[$stem] )); then
        best_json[$stem]="$f"
        best_rank[$stem]=$rank
      fi
    done < <(find "$INPUT" -maxdepth 1 -name "*.${stage}.json" 2>/dev/null | sort)
  done
  for stem in $(echo "${!best_json[@]}" | tr ' ' '\n' | sort); do
    in_list+=("${best_json[$stem]}")
    out_list+=("$OUTPUT/${stem}.mp4")
  done
else
  echo "错误: 输入路径不存在: $INPUT" >&2; exit 1
fi

if [[ ${#in_list[@]} -eq 0 ]]; then
  echo "错误: 未找到可渲染的 JSON 文件" >&2; exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  找到 ${#in_list[@]} 个文件"
echo "════════════════════════════════════════════════════════════"

ok=0; fail=0
for i in "${!in_list[@]}"; do
  in_json="${in_list[$i]}"
  out_mp4="${out_list[$i]}"
  mkdir -p "$(dirname "$out_mp4")"

  echo ""
  echo "── $(basename "$in_json")"
  echo "   → $out_mp4"
  echo -e "   \033[1;32m$\033[0m \033[1;33m.venv/bin/python render.py\033[0m -j \"$in_json\" -o \"$out_mp4\""

  if .venv/bin/python render.py -j "$in_json" -o "$out_mp4"; then
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
