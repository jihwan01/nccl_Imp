#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVCC_BIN="${NVCC_BIN:-nvcc}"
ARCH="${ARCH:-sm_90a}"
SRC_FILE="$ROOT_DIR/simple_tma_transfer_bench.cu"
BIN_FILE="$ROOT_DIR/simple_tma_transfer_bench"

TOTAL_MB="${TOTAL_MB:-128}"
GRID="${GRID:-32}"
BLOCK="${BLOCK:-544}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"
SRC_DEV="${SRC_DEV:-0}"
DST_DEV="${DST_DEV:-1}"
PROTO="${PROTO:-0}"              # 0=both, 1=tma only, 2=simple only
PIPE="${PIPE:-2}"
SMEM_KB="${SMEM_KB:-64}"
TMA_ISSUE_WARP="${TMA_ISSUE_WARP:-1}"
SIMPLE_SLICE_KB="${SIMPLE_SLICE_KB:-1024}"
TMA_SLICE_KB="${TMA_SLICE_KB:-1024}"
TMA_TILE_KB="${TMA_TILE_KB:-32}"
VERIFY="${VERIFY:-1}"
EXTRA_NVCC_FLAGS="${EXTRA_NVCC_FLAGS:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Compile and run simple_tma_transfer_bench.cu, emitting profile logs consumable by process_profile_tma.py.

Options:
  --nvcc-bin PATH           nvcc binary path (default: $NVCC_BIN)
  --arch ARCH               CUDA arch, e.g. sm_90a (default: $ARCH)
  --total-mb N              total transfer size in MB (default: $TOTAL_MB)
  --grid N                  CTAs (default: $GRID)
  --block N                 threads per CTA (default: $BLOCK)
  --iters N                 timed iterations (default: $ITERS)
  --warmup N                warmup iterations (default: $WARMUP)
  --src N                   source GPU id (default: $SRC_DEV)
  --dst N                   destination GPU id (default: $DST_DEV)
  --proto N                 0=both, 1=tma only, 2=simple only (default: $PROTO)
  --pipe N                  TMA pipe depth (default: $PIPE)
  --smem-kb N               dynamic smem per CTA (default: $SMEM_KB)
  --tma-issue-warp N        0 or 1 (default: $TMA_ISSUE_WARP)
  --simple-slice-kb N       simple slice size in KB (default: $SIMPLE_SLICE_KB)
  --tma-slice-kb N          TMA slice size in KB (default: $TMA_SLICE_KB)
  --tma-tile-kb N           TMA tile size in KB (default: $TMA_TILE_KB)
  --verify N                0 or 1 (default: $VERIFY)
  --extra-nvcc-flags STR    appended to nvcc command
  --help                    show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcc-bin) NVCC_BIN="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --total-mb) TOTAL_MB="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --block) BLOCK="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --src) SRC_DEV="$2"; shift 2 ;;
    --dst) DST_DEV="$2"; shift 2 ;;
    --proto) PROTO="$2"; shift 2 ;;
    --pipe) PIPE="$2"; shift 2 ;;
    --smem-kb) SMEM_KB="$2"; shift 2 ;;
    --tma-issue-warp) TMA_ISSUE_WARP="$2"; shift 2 ;;
    --simple-slice-kb) SIMPLE_SLICE_KB="$2"; shift 2 ;;
    --tma-slice-kb) TMA_SLICE_KB="$2"; shift 2 ;;
    --tma-tile-kb) TMA_TILE_KB="$2"; shift 2 ;;
    --verify) VERIFY="$2"; shift 2 ;;
    --extra-nvcc-flags) EXTRA_NVCC_FLAGS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$SRC_FILE" ]]; then
  echo "[error] source file not found: $SRC_FILE" >&2
  exit 2
fi

proto_name="BOTH"
if [[ "$PROTO" == "1" ]]; then
  proto_name="TMA"
elif [[ "$PROTO" == "2" ]]; then
  proto_name="SIMPLE"
fi

echo "========================================"
echo "Simple TMA Transfer Configuration"
echo "========================================"
echo "Protocol    : $proto_name"
echo "GPUs        : 2"
echo "Data Size   : ${TOTAL_MB} MB"
echo "Algorithm   : MICRO_TRANSFER"
echo "Grid        : $GRID"
echo "Block       : $BLOCK"
echo "Pipe        : $PIPE"
echo "SMEM        : ${SMEM_KB} KB"
echo "Issue Warp  : $TMA_ISSUE_WARP"
echo "Iterations  : $ITERS"
echo "Warmup      : $WARMUP"
echo "Src/Dst     : $SRC_DEV -> $DST_DEV"
echo "========================================"

compile_cmd=(
  "$NVCC_BIN"
  "-arch=$ARCH"
  "-O3"
  "-std=c++20"
  "$SRC_FILE"
  "-o" "$BIN_FILE"
)

if [[ -n "$EXTRA_NVCC_FLAGS" ]]; then
  # shellcheck disable=SC2206
  extra_flags=( $EXTRA_NVCC_FLAGS )
  compile_cmd+=("${extra_flags[@]}")
fi

echo "[compile] ${compile_cmd[*]}"
"${compile_cmd[@]}"

run_cmd=(
  "$BIN_FILE"
  "--total_mb" "$TOTAL_MB"
  "--grid" "$GRID"
  "--block" "$BLOCK"
  "--iters" "$ITERS"
  "--warmup" "$WARMUP"
  "--src" "$SRC_DEV"
  "--dst" "$DST_DEV"
  "--proto" "$PROTO"
  "--pipe" "$PIPE"
  "--smem_kb" "$SMEM_KB"
  "--tma_issue_warp" "$TMA_ISSUE_WARP"
  "--simple_slice_kb" "$SIMPLE_SLICE_KB"
  "--tma_slice_kb" "$TMA_SLICE_KB"
  "--tma_tile_kb" "$TMA_TILE_KB"
  "--verify" "$VERIFY"
)

echo "[run] ${run_cmd[*]}"
"${run_cmd[@]}"
