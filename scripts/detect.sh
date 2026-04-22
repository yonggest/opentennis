#!/usr/bin/env bash
# detect.py 的便捷包装，内置模型路径和参数。
# 批量模式下自动检测可用 GPU 并并行处理（每块 GPU 一个进程）。
#
# 用法:
#   bash detect.sh <视频文件> <输出.json>       # 单文件
#   bash detect.sh <视频目录> <输出目录>         # 批量
#
# 批量模式说明:
#   自动用 nvidia-smi 查询所有可用 GPU。
#   若检测到多块 GPU，则为每块 GPU 启动一个后台进程并行处理，
#   视频按 round-robin 顺序分配（GPU0→视频0,2,4...  GPU1→视频1,3,5...）。
#   每个进程通过 CUDA_VISIBLE_DEVICES 绑定到指定 GPU。
#   macOS（CoreML）或只有一块 GPU 时退化为串行。
#
# 示例:
#   bash detect.sh ../datasets/tennis-video-26/trials/footprint.mp4 out/footprint.detected.json
#   bash detect.sh ../datasets/tennis-video-26/games/ out/games/

set -euo pipefail

VIDEO_EXTS="mp4 MP4 mov MOV avi AVI mkv MKV"

# ── 参数检查 ──────────────────────────────────────────────────────────────────
if [[ $# -ne 2 ]]; then
  echo "" >&2
  echo "% detect.sh video.mp4  out.detected.json" >&2
  echo "% detect.sh videos/    out/" >&2
  echo "" >&2
  echo "批量模式自动检测可用 GPU，每块 GPU 启动一个进程并行处理。" >&2
  echo "macOS（CoreML）或只有一块 GPU 时退化为串行。" >&2
  echo "" >&2
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

# ── GPU 检测 ──────────────────────────────────────────────────────────────────
GPUS=()
if command -v nvidia-smi &>/dev/null; then
  while IFS= read -r idx; do
    [[ -n "$idx" ]] && GPUS+=("$idx")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
NUM_GPUS=${#GPUS[@]}

echo "════════════════════════════════════════════════════════════"
echo "  object model  $OBJECT_MODEL"
echo "  court model   $COURT_MODEL"
if [[ $NUM_GPUS -gt 0 ]]; then
  echo "  GPUs          ${GPUS[*]}  (并行 $NUM_GPUS 进程)"
else
  echo "  GPUs          无（串行）"
fi
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

# ── 单视频处理函数 ─────────────────────────────────────────────────────────────
# 用法: run_one <gpu_label> <video> <out_json>
# gpu_label 为空字符串时不设置 CUDA_VISIBLE_DEVICES
run_one() {
  local gpu_label="$1"
  local video="$2"
  local out_json="$3"
  mkdir -p "$(dirname "$out_json")"
  local prefix=""
  [[ -n "$gpu_label" ]] && prefix="[GPU $gpu_label] "
  echo ""
  echo "${prefix}── $(basename "$video")"
  echo "   → $out_json"
  if [[ -n "$gpu_label" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_label" .venv/bin/python detect.py \
      -i "$video" -o "$out_json" -m "$OBJECT_MODEL" -s "$COURT_MODEL"
  else
    .venv/bin/python detect.py \
      -i "$video" -o "$out_json" -m "$OBJECT_MODEL" -s "$COURT_MODEL"
  fi
}

# ── 处理逻辑 ──────────────────────────────────────────────────────────────────
ok=0
fail=0

if [[ ${#video_list[@]} -eq 1 || $NUM_GPUS -le 1 ]]; then
  # ── 串行 ──────────────────────────────────────────────────────────────────
  gpu_label=""
  [[ $NUM_GPUS -eq 1 ]] && gpu_label="${GPUS[0]}"
  for i in "${!video_list[@]}"; do
    if run_one "$gpu_label" "${video_list[$i]}" "${json_list[$i]}"; then
      ok=$((ok + 1))
    else
      echo "  [FAILED] ${video_list[$i]}" >&2
      fail=$((fail + 1))
    fi
  done

else
  # ── 并行：每块 GPU 一个后台子进程，round-robin 分配视频 ───────────────────
  tmpdir=$(mktemp -d)
  trap 'rm -rf "$tmpdir"' EXIT

  for g in "${!GPUS[@]}"; do
    gpu="${GPUS[$g]}"
    (
      w_ok=0; w_fail=0
      for i in "${!video_list[@]}"; do
        [[ $((i % NUM_GPUS)) -ne $g ]] && continue
        if run_one "$gpu" "${video_list[$i]}" "${json_list[$i]}"; then
          w_ok=$((w_ok + 1))
        else
          echo "  [FAILED] ${video_list[$i]}" >&2
          w_fail=$((w_fail + 1))
        fi
      done
      echo "$w_ok $w_fail" > "$tmpdir/result_$g"
    ) &
  done

  wait  # 等待所有 GPU 工作进程结束

  # 汇总各进程的结果
  for g in "${!GPUS[@]}"; do
    if [[ -f "$tmpdir/result_$g" ]]; then
      read -r r_ok r_fail < "$tmpdir/result_$g"
      ok=$((ok + r_ok))
      fail=$((fail + r_fail))
    fi
  done
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  完成: $ok 成功  $fail 失败"
echo "════════════════════════════════════════════════════════════"
