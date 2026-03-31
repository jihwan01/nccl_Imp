#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NVCC_BIN="${NVCC_BIN:-nvcc}"
ARCH="${ARCH:-sm_90a}"

NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-$ROOT_DIR/build/include}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-$ROOT_DIR/build/lib}"

SRC_FILE="$ROOT_DIR/allgather_micro_nccl_bench.cu"
BIN_FILE="$ROOT_DIR/allgather_micro_nccl_bench"

DEVICES="${DEVICES:-0,1,2,3}"
SEND_MB="${SEND_MB:-128}"
GRID="${GRID:-32}"
BLOCK="${BLOCK:-544}"
PIPE="${PIPE:-2}"
SMEM_KB="${SMEM_KB:-64}"
TMA_ISSUE_WARP="${TMA_ISSUE_WARP:-1}"
SIMPLE_SLICE_KB="${SIMPLE_SLICE_KB:-1024}"
TMA_SLICE_KB="${TMA_SLICE_KB:-1024}"
TMA_TILE_KB="${TMA_TILE_KB:-32}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-3}"
VERIFY="${VERIFY:-1}"
BENCH="${BENCH:-micro_tma,nccl_tma}"
EXTRA_NVCC_FLAGS="${EXTRA_NVCC_FLAGS:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Compile and run allgather_micro_nccl_bench.cu assuming NCCL is already built.

Options:
  --nvcc-bin PATH           nvcc binary path (default: $NVCC_BIN)
  --arch ARCH               CUDA arch, e.g. sm_90a (default: $ARCH)
  --nccl-include-dir DIR    NCCL include dir (default: $NCCL_INCLUDE_DIR)
  --nccl-lib-dir DIR        NCCL lib dir (default: $NCCL_LIB_DIR)
  --devices LIST            ring-ordered GPU list, e.g. 0,1,2,3 (default: $DEVICES)
  --send-mb N               per-rank send size in MB (default: $SEND_MB)
  --grid N                  CTAs per transfer kernel (default: $GRID)
  --block N                 threads per CTA (default: $BLOCK)
  --pipe N                  TMA pipe depth (default: $PIPE)
  --smem-kb N               dynamic smem per CTA for TMA (default: $SMEM_KB)
  --tma-issue-warp N        0 or 1 (default: $TMA_ISSUE_WARP)
  --simple-slice-kb N       Simple slice size in KB (default: $SIMPLE_SLICE_KB)
  --tma-slice-kb N          TMA slice size in KB (default: $TMA_SLICE_KB)
  --tma-tile-kb N           TMA tile size in KB (default: $TMA_TILE_KB)
  --iters N                 timed iterations (default: $ITERS)
  --warmup N                warmup iterations (default: $WARMUP)
  --verify N                0 or 1 (default: $VERIFY)
  --bench LIST              e.g. micro_tma,nccl_tma or all (default: $BENCH)
  --extra-nvcc-flags STR    appended to nvcc command
  --help                    show this message

Examples:
  $(basename "$0")

  $(basename "$0") \\
    --devices 0,1,2,3 \\
    --send-mb 128 \\
    --pipe 2 \\
    --smem-kb 64 \\
    --tma-slice-kb 1024 \\
    --tma-tile-kb 32 \\
    --bench micro_tma,nccl_tma
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvcc-bin) NVCC_BIN="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --nccl-include-dir) NCCL_INCLUDE_DIR="$2"; shift 2 ;;
    --nccl-lib-dir) NCCL_LIB_DIR="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --send-mb) SEND_MB="$2"; shift 2 ;;
    --grid) GRID="$2"; shift 2 ;;
    --block) BLOCK="$2"; shift 2 ;;
    --pipe) PIPE="$2"; shift 2 ;;
    --smem-kb) SMEM_KB="$2"; shift 2 ;;
    --tma-issue-warp) TMA_ISSUE_WARP="$2"; shift 2 ;;
    --simple-slice-kb) SIMPLE_SLICE_KB="$2"; shift 2 ;;
    --tma-slice-kb) TMA_SLICE_KB="$2"; shift 2 ;;
    --tma-tile-kb) TMA_TILE_KB="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --verify) VERIFY="$2"; shift 2 ;;
    --bench) BENCH="$2"; shift 2 ;;
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

if [[ ! -d "$NCCL_INCLUDE_DIR" ]]; then
  echo "[error] NCCL include dir not found: $NCCL_INCLUDE_DIR" >&2
  exit 2
fi

if [[ ! -d "$NCCL_LIB_DIR" ]]; then
  echo "[error] NCCL lib dir not found: $NCCL_LIB_DIR" >&2
  echo "        set --nccl-lib-dir or NCCL_LIB_DIR to your built NCCL library path" >&2
  exit 2
fi

if [[ ! -f "$NCCL_INCLUDE_DIR/nccl.h" ]]; then
  echo "[error] nccl.h not found under: $NCCL_INCLUDE_DIR" >&2
  exit 2
fi

if [[ ! -f "$NCCL_LIB_DIR/libnccl.so" && ! -f "$NCCL_LIB_DIR/libnccl_static.a" ]]; then
  echo "[error] neither libnccl.so nor libnccl_static.a found under: $NCCL_LIB_DIR" >&2
  exit 2
fi

compile_cmd=(
  "$NVCC_BIN"
  "-arch=$ARCH"
  "-O3"
  "-std=c++20"
  "$SRC_FILE"
  "-I$NCCL_INCLUDE_DIR"
  "-L$NCCL_LIB_DIR"
  "-lnccl"
  "-Wl,-rpath,$NCCL_LIB_DIR"
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
  "--devices" "$DEVICES"
  "--send_mb" "$SEND_MB"
  "--grid" "$GRID"
  "--block" "$BLOCK"
  "--pipe" "$PIPE"
  "--smem_kb" "$SMEM_KB"
  "--tma_issue_warp" "$TMA_ISSUE_WARP"
  "--simple_slice_kb" "$SIMPLE_SLICE_KB"
  "--tma_slice_kb" "$TMA_SLICE_KB"
  "--tma_tile_kb" "$TMA_TILE_KB"
  "--iters" "$ITERS"
  "--warmup" "$WARMUP"
  "--verify" "$VERIFY"
  "--bench" "$BENCH"
)

echo "[run] ${run_cmd[*]}"
"${run_cmd[@]}"
