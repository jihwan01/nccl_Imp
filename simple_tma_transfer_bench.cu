#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/ptx>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using bf16 = __nv_bfloat16;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#ifndef ENABLE_PROFILING
#define ENABLE_PROFILING 1
#endif
#ifndef ENABLE_TILE_PROFILING
#define ENABLE_TILE_PROFILING ENABLE_PROFILING
#endif
#ifndef ENABLE_SLICE_PROFILING
#define ENABLE_SLICE_PROFILING ENABLE_PROFILING
#endif

#define CUDA_CHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

static constexpr int kMaxPipeDepth = 8;

enum class TrafficMode {
  None = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3,
};

enum class TrafficTarget {
  Src = 0,
  Dst = 1,
  Both = 2,
};

struct Args {
  size_t total_mb = 128;
  int grid = 32;
  int block = 544;
  int iters = 20;
  int warmup = 3;
  int src_dev = 0;
  int dst_dev = 1;

  // 0=both, 1=tma only, 2=simple only
  int proto = 0;

  // pipeline depth / dynamic smem control for TMA
  int pipe = 2;
  size_t smem_kb = 64;
  // explicit slice/tile controls requested for comparison
  size_t simple_slice_kb = 1024; // 1MB default
  size_t tma_slice_kb = 1024;    // 1MB default
  size_t tma_tile_kb = 32;       // 32KB default

  TrafficMode traffic_mode = TrafficMode::None;
  TrafficTarget traffic_target = TrafficTarget::Src;
  size_t traffic_mb = 256;
  int traffic_grid = 32;
  int traffic_block = 256;
  int traffic_rounds = 64;

  int verify = 1;
};

struct BenchResult {
  float avg_ms = 0.0f;
  float gbps = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float stddev_ms = 0.0f;
  float p50_ms = 0.0f;
  float p95_ms = 0.0f;
  std::vector<float> iter_ms;
};

struct TrafficInstance {
  bool active = false;
  int dev = -1;
  cudaStream_t stream = nullptr;
  int* h_stop = nullptr;
  int* d_stop = nullptr;
  uint4* d_src = nullptr;
  uint4* d_dst = nullptr;
  unsigned int* d_sink = nullptr;
  size_t traffic_bytes = 0;
};

struct TrafficContext {
  TrafficInstance src;
  TrafficInstance dst;
};

__global__ void init_src_bf16(bf16* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) src[i] = __float2bfloat16(float(i % 4096));
}

__global__ void kernel_mem_traffic(
    uint4* __restrict__ dst,
    const uint4* __restrict__ src,
    size_t vec_count,
    int mode,
    int rounds,
    volatile int* stop_flag,
    unsigned int* sink) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  unsigned int acc = 0x9e3779b9u ^ unsigned(tid);

  for (int round = 0; (rounds <= 0 || round < rounds) && *stop_flag == 0; ++round) {
    for (size_t i = tid; i < vec_count; i += stride) {
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (mode != int(TrafficMode::Write)) {
        x = src[i];
        acc ^= x.x + 0x9e3779b9u + (acc << 6) + (acc >> 2);
        acc ^= x.y + 0x85ebca6bu + (acc << 6) + (acc >> 2);
        acc ^= x.z + 0xc2b2ae35u + (acc << 6) + (acc >> 2);
        acc ^= x.w + 0x27d4eb2fu + (acc << 6) + (acc >> 2);
      }
      if (mode != int(TrafficMode::Read)) {
        const unsigned int i32 = unsigned(i);
        uint4 y = make_uint4(acc ^ i32, acc + i32 * 17u, acc ^ i32 * 31u, acc + i32 * 63u);
        if (mode == int(TrafficMode::ReadWrite)) {
          y.x ^= x.x;
          y.y += x.y;
          y.z ^= x.z;
          y.w += x.w;
        }
        dst[i] = y;
      }
    }
  }

  if (sink != nullptr) sink[tid] = acc;
}

__global__ void kernel_simple_transfer(
    bf16* __restrict__ dst_peer,
    const bf16* __restrict__ src,
    size_t chunk_bytes,
    size_t simple_slice_bytes) {
  const size_t tid = threadIdx.x;
  const int nworkers = blockDim.x;
  constexpr int kWarpSize = 32;
  const int simple_compute_workers =
      (nworkers >= 2 * kWarpSize) ? (nworkers - kWarpSize) : nworkers;
  const size_t cta_start = blockIdx.x * chunk_bytes;

  const uint4* src_v = reinterpret_cast<const uint4*>(src);
  uint4* dst_v = reinterpret_cast<uint4*>(dst_peer);

  for (size_t slice_off = 0; slice_off < chunk_bytes; slice_off += simple_slice_bytes) {
    const size_t curr_slice_bytes = min(simple_slice_bytes, chunk_bytes - slice_off);
    const size_t curr_vecs = curr_slice_bytes / 16;
    const size_t vec_base = (cta_start + slice_off) / 16;

    if (int(tid) < simple_compute_workers) {
      const size_t stride = size_t(simple_compute_workers);
      #pragma unroll 8
      for (size_t v = tid; v < curr_vecs; v += stride) {
        uint4 x = src_v[vec_base + v];
        dst_v[vec_base + v] = x;
      }
    }
  }
}

struct TmaIssuePlan {
  bool should_issue;
  size_t idx;
  int slot;
  size_t off;
  size_t bytes;
};

__device__ __forceinline__ TmaIssuePlan make_tma_issue_plan(
    size_t t,
    size_t tile_count,
    int pipe_depth,
    size_t tma_tile_bytes,
    size_t curr_slice_bytes) {
  const size_t idx = t + size_t(pipe_depth - 1);
  const bool should_issue = t > 0 && idx < tile_count;
  const size_t off = idx * tma_tile_bytes;
  return {
      should_issue,
      idx,
      int(idx % size_t(pipe_depth)),
      off,
      should_issue ? min(tma_tile_bytes, curr_slice_bytes - off) : size_t(0),
  };
}

// Choose how many compute threads consume the current TMA tile from shared
// memory and store it to the destination. This mirrors the NCCL
// reduceCopyShared<NCCL_TMA_UNROLL=8> shape: each active warp covers roughly
// 8 unrolled uint4 values per lane, i.e. 8 * 32 lanes * 16B = 4KB per pass.
// Small tiles therefore use fewer warps, while larger tiles are capped by the
// compute workers left after reserving the final warp for TMA issue.
__device__ __forceinline__ int tma_active_compute_threads(
    size_t copy_bytes,
    int tma_compute_workers) {
  constexpr int kWarpSize = 32;
  constexpr int kCopyUnroll = 8;
  constexpr size_t kCopyBytesPerWarp = size_t(kCopyUnroll) * size_t(kWarpSize) * 16ULL;
  int max_compute_warps = tma_compute_workers / kWarpSize;
  if (max_compute_warps < 1) max_compute_warps = 1;
  int active_compute_warps = int((copy_bytes + kCopyBytesPerWarp - 1) / kCopyBytesPerWarp);
  if (active_compute_warps > max_compute_warps) active_compute_warps = max_compute_warps;
  int active_compute_threads = active_compute_warps * kWarpSize;
  return active_compute_threads > tma_compute_workers ? tma_compute_workers : active_compute_threads;
}

/*
 * TMA-style transfer micro-kernel
 *
 * Data path (per CTA):
 *   global(src on src_dev) -> shared-memory slot -> global(dst on dst_dev peer)
 *
 * Hierarchy:
 *   chunk (per CTA) -> slice (tma_slice_bytes) -> tile (tma_tile_bytes)
 *
 * Pipelining:
 *   - `pipe_depth` shared-memory slots are used as a ring buffer.
 *   - Always reserve the final warp and use one lane as issuer.
 *   - Prologue preloads up to `pipe_depth` tiles.
 *   - Main loop consumes one tile and issues one future tile into the freed slot.
 *
 */
__global__ void kernel_tma_transfer(
    bf16* __restrict__ dst_peer,
    const bf16* __restrict__ src,
    size_t chunk_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes,
    size_t tma_slot_bytes,
    int pipe_depth,
    int emit_profile) {
  extern __shared__ __align__(16) unsigned char smem_all[];

  // One barrier/token per pipeline slot.
  __shared__ barrier_t bars[kMaxPipeDepth];
  barrier_t::arrival_token tokens[kMaxPipeDepth];

  // Shared-memory ring-buffer slots: slot i starts at i * tma_slot_bytes.
  unsigned char* smem_slot[kMaxPipeDepth];
  #pragma unroll
  for (int i = 0; i < kMaxPipeDepth; ++i) {
    smem_slot[i] = smem_all + size_t(i) * tma_slot_bytes;
  }

  const size_t tid = threadIdx.x;
  const size_t cta_start = blockIdx.x * chunk_bytes;
  const int nworkers = blockDim.x;
  const bool profile_block = emit_profile && blockIdx.x == 0;

  // NCCL-like warp specialization: with the benchmark's block>=512 assumption,
  // reserve the final warp and use its first lane as the async TMA issuer.
  constexpr int kWarpSize = 32;
  const int tma_compute_workers = nworkers - kWarpSize;
  const int tma_issue_tid = tma_compute_workers;
  const bool is_tma_compute_thread = int(tid) < tma_compute_workers;
  const bool is_tma_issuer_thread = int(tid) == tma_issue_tid;

  uint4* dst_v = reinterpret_cast<uint4*>(dst_peer);

  if (threadIdx.x == 0) {
    for (int i = 0; i < pipe_depth; ++i) init(&bars[i], blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);
  }
  __syncthreads();

  // Process the CTA chunk slice by slice.
  int slice_idx = 0;
  for (size_t slice_off = 0; slice_off < chunk_bytes; slice_off += tma_slice_bytes, ++slice_idx) {
    const size_t curr_slice_bytes = min(tma_slice_bytes, chunk_bytes - slice_off);
    const size_t tile_count = (curr_slice_bytes + tma_tile_bytes - 1) / tma_tile_bytes;

    #if ENABLE_SLICE_PROFILING
    unsigned long long tSliceCopyStart = 0;
    if (profile_block && tid == 0) tSliceCopyStart = clock64();
    #endif

    // 1) Prologue: preload up to pipe_depth tiles.
    int in_flight = 0;
    for (; in_flight < pipe_depth && in_flight < (int)tile_count; ++in_flight) {
      const size_t tile_idx = size_t(in_flight);
      const size_t tile_off = tile_idx * tma_tile_bytes;
      const size_t copy_bytes = min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const int slot = in_flight % pipe_depth;

      #if ENABLE_TILE_PROFILING
      unsigned long long tIssueStart = 0;
      if (profile_block && is_tma_issuer_thread) tIssueStart = clock64();
      #endif
      if (is_tma_issuer_thread) {
        cuda::device::memcpy_async_tx(
            reinterpret_cast<char*>(smem_slot[slot]),
            reinterpret_cast<const char*>(src) + cta_start + slice_off + tile_off,
            cuda::aligned_size_t<16>(copy_bytes),
            bars[slot]);
        tokens[slot] = cuda::device::barrier_arrive_tx(bars[slot], 1, copy_bytes);
      } else {
        tokens[slot] = bars[slot].arrive();
      }
      #if ENABLE_TILE_PROFILING
      unsigned long long tIssueEnd = clock64();
      if (profile_block && is_tma_issuer_thread) {
        printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,SEND_ISSUE_ENQ_PRELOAD,%llu\n", 0, 0, slice_idx, int(tile_idx), tIssueEnd - tIssueStart);
      }
      #endif
    }

    #if ENABLE_SLICE_PROFILING
    if (profile_block && tid == 0) {
      tSliceCopyStart = clock64();
    }
    #endif

    // 2) Steady state: wait/consume current tile, then issue next tile.
    for (size_t t = 0; t < tile_count; ++t) {
      const int slot = int(t % size_t(pipe_depth));
      const size_t tile_off = t * tma_tile_bytes;
      const size_t copy_bytes = min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const size_t vec_count = copy_bytes / 16;
      const TmaIssuePlan issue = make_tma_issue_plan(
          t, tile_count, pipe_depth, tma_tile_bytes, curr_slice_bytes);

      // ===================== Measure TMA Load Wait Time =======================
      #if ENABLE_TILE_PROFILING
      unsigned long long tWaitStart = 0;
      if (profile_block && tid == 0) tWaitStart = clock64();
      #endif

      bars[slot].wait(std::move(tokens[slot]));

      #if ENABLE_TILE_PROFILING
      unsigned long long tWaitEnd = 0;
      if (profile_block && tid == 0) {
        tWaitEnd = clock64();
        printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,SEND_WAIT,%llu\n", 0, 0, slice_idx, int(t), tWaitEnd - tWaitStart);
      }
      // =========================================================================

      unsigned long long tCopyCoreStart = 0;
      if (profile_block && tid == 0) tCopyCoreStart = clock64();
      #endif

      const uint4* smem_v = reinterpret_cast<const uint4*>(smem_slot[slot]);
      const size_t dst_vec_base = (cta_start + slice_off + tile_off) / 16;

      if (issue.should_issue && !is_tma_issuer_thread) {
        tokens[issue.slot] = bars[issue.slot].arrive();
      }

      const int active_compute_threads = tma_active_compute_threads(copy_bytes, tma_compute_workers);

      if (is_tma_compute_thread && int(tid) < active_compute_threads) {
        const size_t stride = size_t(active_compute_threads);
        #pragma unroll 8
        for (size_t v = tid; v < vec_count; v += stride) {
          uint4 x = smem_v[v];
          dst_v[dst_vec_base + v] = x;
        }
      }

      // ======= Overlapped Issue (Executed only by issue warp) =======
      if (issue.should_issue && is_tma_issuer_thread) {

        #if ENABLE_TILE_PROFILING
        unsigned long long tIssueOverlapStart = 0;
        if (profile_block) tIssueOverlapStart = clock64();
        #endif
        cuda::device::memcpy_async_tx(
            reinterpret_cast<char*>(smem_slot[issue.slot]),
            reinterpret_cast<const char*>(src) + cta_start + slice_off + issue.off,
            cuda::aligned_size_t<16>(issue.bytes),
            bars[issue.slot]);
        tokens[issue.slot] = cuda::device::barrier_arrive_tx(bars[issue.slot], 1, issue.bytes);

        #if ENABLE_TILE_PROFILING
        if (profile_block) {
          unsigned long long tIssueOverlapEnd = clock64();
          printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,SEND_ISSUE_ENQ_OVERLAP,%llu\n", 0, 0, slice_idx, int(issue.idx), tIssueOverlapEnd - tIssueOverlapStart);
        }
        #endif
      }
      // ======= Overlapped Issue End =======


      #if ENABLE_TILE_PROFILING
      unsigned long long tCopyCoreEnd = 0;
      if (profile_block && tid == 0) {
        tCopyCoreEnd = clock64();
        printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,SEND_COPY_CORE,%llu\n", 0, 0, slice_idx, int(t), tCopyCoreEnd - tCopyCoreStart);
      }
      unsigned long long tTailSyncStart = 0;
      if (profile_block && tid == 0) tTailSyncStart = clock64();
      #endif
      __syncthreads();
      #if ENABLE_TILE_PROFILING
      if (profile_block && tid == 0) {
        unsigned long long tTailSyncEnd = clock64();
        printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,SEND_TAIL_SYNC,%llu\n", 0, 0, slice_idx, int(t), tTailSyncEnd - tTailSyncStart);
      }
      #endif
    }
    #if ENABLE_SLICE_PROFILING
    if (profile_block && tid == 0) {
      unsigned long long tSliceCopyMainEnd = clock64();
      printf("NCCL_PROFILE_SLICE,%d,%d,%d,SEND_COPY_MAIN,%llu\n", 0, 0, slice_idx, tSliceCopyMainEnd - tSliceCopyStart);
    }
    #endif
  }
}

static const char* traffic_mode_str(TrafficMode mode) {
  switch (mode) {
    case TrafficMode::None: return "none";
    case TrafficMode::Read: return "read";
    case TrafficMode::Write: return "write";
    case TrafficMode::ReadWrite: return "readwrite";
  }
  return "unknown";
}

static const char* traffic_target_str(TrafficTarget target) {
  switch (target) {
    case TrafficTarget::Src: return "src";
    case TrafficTarget::Dst: return "dst";
    case TrafficTarget::Both: return "both";
  }
  return "unknown";
}

static TrafficMode parse_traffic_mode(const std::string& value) {
  if (value == "none") return TrafficMode::None;
  if (value == "read") return TrafficMode::Read;
  if (value == "write") return TrafficMode::Write;
  if (value == "readwrite") return TrafficMode::ReadWrite;
  fprintf(stderr, "Invalid --traffic_mode value: %s (expected none/read/write/readwrite)\n", value.c_str());
  std::exit(2);
}

static TrafficTarget parse_traffic_target(const std::string& value) {
  if (value == "src") return TrafficTarget::Src;
  if (value == "dst") return TrafficTarget::Dst;
  if (value == "both") return TrafficTarget::Both;
  fprintf(stderr, "Invalid --traffic_target value: %s (expected src/dst/both)\n", value.c_str());
  std::exit(2);
}

static inline void parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--help" || k == "-h") {
      printf("Usage: ./simple_tma_transfer_bench [options]\\n"
             "  --total_mb N          Total transfer size in MB (default: 128)\\n"
             "  --grid N              Number of CTAs (default: 32)\\n"
             "  --block N             Threads per CTA (default: 544)\\n"
             "  --iters N             Timed iterations (default: 20)\\n"
             "  --warmup N            Warmup iterations (default: 3)\\n"
             "  --src N               Source GPU id (default: 0)\\n"
             "  --dst N               Destination GPU id (default: 1)\\n"
             "  --proto N             0=both, 1=tma only, 2=simple only (default: 0)\\n"
             "  --smem_kb N           Dynamic shared memory per block for TMA (default: 64)\\n"
             "  --pipe N              TMA pipeline depth, must be >=2 (default: 2)\\n"
             "  --tma_issue_warp N    Ignored; issue warp is always enabled\\n"
             "  --simple_slice_kb N   Simple slice size in KB (default: 1024)\\n"
             "  --tma_slice_kb N      TMA slice size in KB (default: 1024)\\n"
             "  --tma_tile_kb N       TMA tile size in KB (default: 32)\\n"
             "                        NOTE: TMA enforces slot_bytes == tile_bytes, so smem_kb must satisfy\\n"
             "                              smem_kb*1024 == tma_tile_kb*1024*pipe\\n"
             "  --traffic_mode STR    none/read/write/readwrite (default: none)\\n"
             "  --traffic_target STR  src/dst/both for background traffic (default: src)\\n"
             "  --traffic_mb N        Background traffic buffer size in MB per target GPU (default: 256)\\n"
             "  --traffic_grid N      Background traffic CTAs (default: 32)\\n"
             "  --traffic_block N     Background traffic threads per CTA (default: 256)\\n"
             "  --traffic_rounds N    Finite sweeps over traffic buffer; 0=infinite (default: 64)\\n"
             "  --detail N            Ignored (kept for script compatibility)\\n"
             "  --verify N            0=off, 1=on (default: 1)\\n");
      std::exit(0);
    }

    auto need = [&](int n) {
      if (i + n >= argc) {
        fprintf(stderr, "Missing value for %s\\n", k.c_str());
        std::exit(2);
      }
    };

    if (k == "--total_mb") { need(1); a.total_mb = std::stoull(argv[++i]); }
    else if (k == "--grid") { need(1); a.grid = std::stoi(argv[++i]); }
    else if (k == "--block") { need(1); a.block = std::stoi(argv[++i]); }
    else if (k == "--iters") { need(1); a.iters = std::stoi(argv[++i]); }
    else if (k == "--warmup") { need(1); a.warmup = std::stoi(argv[++i]); }
    else if (k == "--src") { need(1); a.src_dev = std::stoi(argv[++i]); }
    else if (k == "--dst") { need(1); a.dst_dev = std::stoi(argv[++i]); }
    else if (k == "--proto") { need(1); a.proto = std::stoi(argv[++i]); }
    else if (k == "--pipe") { need(1); a.pipe = std::stoi(argv[++i]); }
    else if (k == "--tma_issue_warp") { need(1); ++i; }
    else if (k == "--smem_kb") { need(1); a.smem_kb = std::stoull(argv[++i]); }
    else if (k == "--simple_slice_kb") { need(1); a.simple_slice_kb = std::stoull(argv[++i]); }
    else if (k == "--tma_slice_kb") { need(1); a.tma_slice_kb = std::stoull(argv[++i]); }
    else if (k == "--tma_tile_kb") { need(1); a.tma_tile_kb = std::stoull(argv[++i]); }
    else if (k == "--traffic_mode") { need(1); a.traffic_mode = parse_traffic_mode(argv[++i]); }
    else if (k == "--traffic_target") { need(1); a.traffic_target = parse_traffic_target(argv[++i]); }
    else if (k == "--traffic_mb") { need(1); a.traffic_mb = std::stoull(argv[++i]); }
    else if (k == "--traffic_grid") { need(1); a.traffic_grid = std::stoi(argv[++i]); }
    else if (k == "--traffic_block") { need(1); a.traffic_block = std::stoi(argv[++i]); }
    else if (k == "--traffic_rounds") { need(1); a.traffic_rounds = std::stoi(argv[++i]); }
    else if (k == "--detail") { need(1); ++i; }
    else if (k == "--verify") { need(1); a.verify = std::stoi(argv[++i]); }
    else {
      fprintf(stderr, "Unknown arg: %s\\n", k.c_str());
      std::exit(2);
    }
  }
}

static inline void validate_or_die(
    const Args& a,
    size_t total_bytes,
    size_t chunk_bytes,
    size_t smem_bytes,
    size_t tma_slot_bytes,
    size_t simple_slice_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes) {
  if (a.grid <= 0 || a.block <= 0 || a.iters <= 0 || a.warmup < 0) {
    fprintf(stderr, "Invalid grid/block/iters/warmup\\n");
    std::exit(2);
  }
  if (a.pipe < 2 || a.pipe > kMaxPipeDepth) {
    fprintf(stderr, "--pipe must be in [2, %d] for the TMA overlap pipeline\\n", kMaxPipeDepth);
    std::exit(2);
  }
  if (!(a.proto == 0 || a.proto == 1 || a.proto == 2)) {
    fprintf(stderr, "--proto must be 0(both), 1(tma), or 2(simple)\\n");
    std::exit(2);
  }
  if ((a.proto == 0 || a.proto == 1) && a.block < 512) {
    fprintf(stderr, "TMA path assumes --block >= 512 because the final warp is always reserved for issue\\n");
    std::exit(2);
  }
  if (a.src_dev == a.dst_dev) {
    fprintf(stderr, "--src and --dst must be different\\n");
    std::exit(2);
  }
  if (a.traffic_mode != TrafficMode::None) {
    if (a.traffic_mb == 0) {
      fprintf(stderr, "--traffic_mb must be > 0 when traffic is enabled\\n");
      std::exit(2);
    }
    if (a.traffic_grid <= 0 || a.traffic_block <= 0) {
      fprintf(stderr, "--traffic_grid and --traffic_block must be > 0 when traffic is enabled\\n");
      std::exit(2);
    }
    if (a.traffic_rounds < 0) {
      fprintf(stderr, "--traffic_rounds must be >= 0\\n");
      std::exit(2);
    }
  }
  if (total_bytes == 0) {
    fprintf(stderr, "total bytes must be > 0\\n");
    std::exit(2);
  }
  if ((total_bytes % size_t(a.grid)) != 0) {
    fprintf(stderr, "total bytes must be multiple of grid\\n");
    std::exit(2);
  }
  if (chunk_bytes == 0 || (chunk_bytes % 16) != 0) {
    fprintf(stderr, "chunk bytes must be >0 and 16-byte aligned\\n");
    std::exit(2);
  }
  if (smem_bytes == 0 || (smem_bytes % size_t(a.pipe)) != 0) {
    fprintf(stderr, "smem bytes must be >0 and divisible by pipe\\n");
    std::exit(2);
  }
  if ((tma_slot_bytes % 16) != 0) {
    fprintf(stderr, "tma_slot_bytes (smem/pipe=%zu) must be 16-byte aligned\\n", tma_slot_bytes);
    std::exit(2);
  }

  if (simple_slice_bytes == 0 || (simple_slice_bytes % 16) != 0) {
    fprintf(stderr, "simple_slice_bytes must be >0 and 16-byte aligned\\n");
    std::exit(2);
  }
  if (tma_slice_bytes == 0 || (tma_slice_bytes % 16) != 0) {
    fprintf(stderr, "tma_slice_bytes must be >0 and 16-byte aligned\\n");
    std::exit(2);
  }
  if (tma_tile_bytes == 0 || (tma_tile_bytes % 16) != 0) {
    fprintf(stderr, "tma_tile_bytes must be >0 and 16-byte aligned\\n");
    std::exit(2);
  }
  if (tma_slot_bytes != tma_tile_bytes) {
    fprintf(stderr,
            "TMA requires slot_bytes == tile_bytes, but got slot=%zu (smem/pipe) and tile=%zu. "
            "Set smem_bytes = tile_bytes * pipe.\\n",
            tma_slot_bytes, tma_tile_bytes);
    std::exit(2);
  }
}

static inline size_t verify_copy(int src_dev, int dst_dev, const bf16* d_src, const bf16* d_dst, size_t n_elem, size_t max_report = 8) {
  std::vector<uint16_t> h_src(n_elem), h_dst(n_elem);

  CUDA_CHECK(cudaSetDevice(src_dev));
  CUDA_CHECK(cudaMemcpy(h_src.data(), d_src, n_elem * sizeof(bf16), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, n_elem * sizeof(bf16), cudaMemcpyDeviceToHost));

  size_t mismatch = 0;
  for (size_t i = 0; i < n_elem; ++i) {
    if (h_src[i] != h_dst[i]) {
      if (mismatch < max_report) {
        bf16 src_v = *reinterpret_cast<const bf16*>(&h_src[i]);
        bf16 dst_v = *reinterpret_cast<const bf16*>(&h_dst[i]);
        float a = __bfloat162float(src_v);
        float b = __bfloat162float(dst_v);
        fprintf(stderr, "mismatch[%zu] src=0x%04x dst=0x%04x (%.3f vs %.3f)\\n", i, h_src[i], h_dst[i], a, b);
      }
      ++mismatch;
    }
  }
  return mismatch;
}

static inline float bytes_to_gbps(size_t bytes, float ms) {
  if (ms <= 0.0f) return 0.0f;
  double gb = double(bytes) / 1.0e9;
  return float(gb / (double(ms) * 1.0e-3));
}

static inline size_t align_up(size_t x, size_t align) {
  return ((x + align - 1) / align) * align;
}

static inline float percentile_ms(const std::vector<float>& values, float q) {
  if (values.empty()) return 0.0f;
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.size() == 1) return sorted[0];
  const float pos = q * float(sorted.size() - 1);
  const size_t lo = size_t(pos);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const float frac = pos - float(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

static inline void finalize_result(BenchResult& out, size_t total_bytes) {
  if (out.iter_ms.empty()) return;
  out.min_ms = *std::min_element(out.iter_ms.begin(), out.iter_ms.end());
  out.max_ms = *std::max_element(out.iter_ms.begin(), out.iter_ms.end());
  double sum = 0.0;
  for (float ms : out.iter_ms) sum += ms;
  out.avg_ms = float(sum / double(out.iter_ms.size()));
  double sq = 0.0;
  for (float ms : out.iter_ms) {
    const double diff = double(ms) - double(out.avg_ms);
    sq += diff * diff;
  }
  out.stddev_ms = float(std::sqrt(sq / double(out.iter_ms.size())));
  out.p50_ms = percentile_ms(out.iter_ms, 0.50f);
  out.p95_ms = percentile_ms(out.iter_ms, 0.95f);
  out.gbps = bytes_to_gbps(total_bytes, out.avg_ms);
}

static inline bool traffic_targets_src(const Args& a) {
  return a.traffic_mode != TrafficMode::None &&
         (a.traffic_target == TrafficTarget::Src || a.traffic_target == TrafficTarget::Both);
}

static inline bool traffic_targets_dst(const Args& a) {
  return a.traffic_mode != TrafficMode::None &&
         (a.traffic_target == TrafficTarget::Dst || a.traffic_target == TrafficTarget::Both);
}

static inline void start_traffic_instance(const Args& a, int dev, TrafficInstance& inst) {
  inst.active = true;
  inst.dev = dev;
  inst.traffic_bytes = align_up(a.traffic_mb * 1024ULL * 1024ULL, sizeof(uint4));

  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaStreamCreateWithFlags(&inst.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaHostAlloc(&inst.h_stop, sizeof(*inst.h_stop), cudaHostAllocMapped));
  *inst.h_stop = 0;
  CUDA_CHECK(cudaHostGetDevicePointer(&inst.d_stop, inst.h_stop, 0));

  const size_t thread_count = size_t(a.traffic_grid) * size_t(a.traffic_block);
  const size_t vec_count = inst.traffic_bytes / sizeof(uint4);

  if (a.traffic_mode == TrafficMode::Read || a.traffic_mode == TrafficMode::ReadWrite) {
    CUDA_CHECK(cudaMalloc(&inst.d_src, inst.traffic_bytes));
    CUDA_CHECK(cudaMemset(inst.d_src, 0x5a, inst.traffic_bytes));
  }
  if (a.traffic_mode == TrafficMode::Write || a.traffic_mode == TrafficMode::ReadWrite) {
    CUDA_CHECK(cudaMalloc(&inst.d_dst, inst.traffic_bytes));
    CUDA_CHECK(cudaMemset(inst.d_dst, 0x00, inst.traffic_bytes));
  }

  CUDA_CHECK(cudaMalloc(&inst.d_sink, thread_count * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(inst.d_sink, 0, thread_count * sizeof(unsigned int)));

  kernel_mem_traffic<<<a.traffic_grid, a.traffic_block, 0, inst.stream>>>(
      inst.d_dst,
      inst.d_src,
      vec_count,
      int(a.traffic_mode),
      a.traffic_rounds,
      inst.d_stop,
      inst.d_sink);
  CUDA_CHECK(cudaGetLastError());
}

static inline void stop_traffic_instance(TrafficInstance& inst) {
  if (!inst.active) return;

  *inst.h_stop = 1;
  CUDA_CHECK(cudaSetDevice(inst.dev));
  CUDA_CHECK(cudaStreamSynchronize(inst.stream));

  if (inst.d_sink) CUDA_CHECK(cudaFree(inst.d_sink));
  if (inst.d_dst) CUDA_CHECK(cudaFree(inst.d_dst));
  if (inst.d_src) CUDA_CHECK(cudaFree(inst.d_src));
  if (inst.stream) CUDA_CHECK(cudaStreamDestroy(inst.stream));
  if (inst.h_stop) CUDA_CHECK(cudaFreeHost(inst.h_stop));

  inst = TrafficInstance{};
}

static inline void start_traffic(const Args& a, TrafficContext& ctx) {
  int prev_dev = -1;
  CUDA_CHECK(cudaGetDevice(&prev_dev));
  if (traffic_targets_src(a)) start_traffic_instance(a, a.src_dev, ctx.src);
  if (traffic_targets_dst(a)) start_traffic_instance(a, a.dst_dev, ctx.dst);
  if (prev_dev >= 0) CUDA_CHECK(cudaSetDevice(prev_dev));
}

static inline void stop_traffic(TrafficContext& ctx) {
  int prev_dev = -1;
  CUDA_CHECK(cudaGetDevice(&prev_dev));
  stop_traffic_instance(ctx.src);
  stop_traffic_instance(ctx.dst);
  if (prev_dev >= 0) CUDA_CHECK(cudaSetDevice(prev_dev));
}

static inline void print_result(const char* tag, const BenchResult& r, size_t wrong) {
  printf("[%s] avg=%.3f us  min=%.3f us  p50=%.3f us  p95=%.3f us  max=%.3f us  std=%.3f us  BW=%.2f GB/s  wrong=%zu\n",
         tag,
         r.avg_ms * 1000.0f,
         r.min_ms * 1000.0f,
         r.p50_ms * 1000.0f,
         r.p95_ms * 1000.0f,
         r.max_ms * 1000.0f,
         r.stddev_ms * 1000.0f,
         r.gbps,
         wrong);
}

static BenchResult run_simple(
    const Args& a,
    bf16* d_dst,
    const bf16* d_src,
    size_t total_bytes,
  size_t chunk_bytes,
  size_t simple_slice_bytes) {
  BenchResult out;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  for (int i = 0; i < a.warmup; ++i) {
    kernel_simple_transfer<<<a.grid, a.block, 0, stream>>>(d_dst, d_src, chunk_bytes, simple_slice_bytes);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<cudaEvent_t> starts(a.iters), stops(a.iters);
  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventCreate(&starts[it]));
    CUDA_CHECK(cudaEventCreate(&stops[it]));
  }

  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventRecord(starts[it], stream));
    kernel_simple_transfer<<<a.grid, a.block, 0, stream>>>(d_dst, d_src, chunk_bytes, simple_slice_bytes);
    CUDA_CHECK(cudaEventRecord(stops[it], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  out.iter_ms.resize(a.iters);
  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventElapsedTime(&out.iter_ms[it], starts[it], stops[it]));
    CUDA_CHECK(cudaEventDestroy(starts[it]));
    CUDA_CHECK(cudaEventDestroy(stops[it]));
  }
  finalize_result(out, total_bytes);
  CUDA_CHECK(cudaStreamDestroy(stream));

  return out;
}

static BenchResult run_tma(
    const Args& a,
    bf16* d_dst,
    const bf16* d_src,
    size_t total_bytes,
    size_t chunk_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes,
    size_t smem_bytes,
    size_t tma_slot_bytes) {
  BenchResult out;
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaFuncSetAttribute(
      kernel_tma_transfer,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      int(smem_bytes)));

  for (int i = 0; i < a.warmup; ++i) {
    kernel_tma_transfer<<<a.grid, a.block, smem_bytes, stream>>>(
        d_dst, d_src, chunk_bytes, tma_slice_bytes, tma_tile_bytes, tma_slot_bytes, a.pipe, 0);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<cudaEvent_t> starts(a.iters), stops(a.iters);
  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventCreate(&starts[it]));
    CUDA_CHECK(cudaEventCreate(&stops[it]));
  }

  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventRecord(starts[it], stream));
    kernel_tma_transfer<<<a.grid, a.block, smem_bytes, stream>>>(
        d_dst, d_src, chunk_bytes, tma_slice_bytes, tma_tile_bytes, tma_slot_bytes, a.pipe, 1);
    CUDA_CHECK(cudaEventRecord(stops[it], stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  out.iter_ms.resize(a.iters);
  for (int it = 0; it < a.iters; ++it) {
    CUDA_CHECK(cudaEventElapsedTime(&out.iter_ms[it], starts[it], stops[it]));
    CUDA_CHECK(cudaEventDestroy(starts[it]));
    CUDA_CHECK(cudaEventDestroy(stops[it]));
  }
  finalize_result(out, total_bytes);
  CUDA_CHECK(cudaStreamDestroy(stream));

  return out;
}

int main(int argc, char** argv) {
  Args a;
  parse_args(argc, argv, a);

  const size_t total_bytes = a.total_mb * 1024ULL * 1024ULL;
  const size_t smem_bytes = a.smem_kb * 1024ULL;
  const size_t simple_slice_bytes = a.simple_slice_kb * 1024ULL;
  const size_t tma_slice_bytes = a.tma_slice_kb * 1024ULL;
  const size_t tma_tile_bytes = a.tma_tile_kb * 1024ULL;

  const size_t chunk_bytes = total_bytes / size_t(a.grid);
  const size_t tma_slot_bytes = smem_bytes / size_t(a.pipe);
  const size_t n_elem = total_bytes / sizeof(bf16);

  validate_or_die(a, total_bytes, chunk_bytes, smem_bytes, tma_slot_bytes,
                  simple_slice_bytes, tma_slice_bytes, tma_tile_bytes);

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    fprintf(stderr, "Need at least 2 GPUs\\n");
    return 2;
  }
  if (a.src_dev < 0 || a.src_dev >= ndev || a.dst_dev < 0 || a.dst_dev >= ndev) {
    fprintf(stderr, "Invalid src/dst device index\\n");
    return 2;
  }

  int can01 = 0, can10 = 0;
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, a.src_dev, a.dst_dev));
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, a.dst_dev, a.src_dev));
  if (!can01 || !can10) {
    fprintf(stderr, "Peer access not available between %d and %d\\n", a.src_dev, a.dst_dev);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(a.src_dev));
  cudaError_t pe0 = cudaDeviceEnablePeerAccess(a.dst_dev, 0);
  if (pe0 != cudaSuccess && pe0 != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(pe0);

  CUDA_CHECK(cudaSetDevice(a.dst_dev));
  cudaError_t pe1 = cudaDeviceEnablePeerAccess(a.src_dev, 0);
  if (pe1 != cudaSuccess && pe1 != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(pe1);

  int max_dyn_smem = 0;
  CUDA_CHECK(cudaSetDevice(a.src_dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_dyn_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, a.src_dev));
  if (int(smem_bytes) > max_dyn_smem) {
    fprintf(stderr, "Requested smem=%zu exceeds max dynamic shared memory=%d on src GPU\\n", smem_bytes, max_dyn_smem);
    return 2;
  }

  bf16* d_src = nullptr;
  bf16* d_dst = nullptr;

  CUDA_CHECK(cudaSetDevice(a.src_dev));
  CUDA_CHECK(cudaMalloc(&d_src, total_bytes));
  {
    int bs = 256;
    int gs = int((n_elem + bs - 1) / bs);
    init_src_bf16<<<gs, bs>>>(d_src, n_elem);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaSetDevice(a.dst_dev));
  CUDA_CHECK(cudaMalloc(&d_dst, total_bytes));

  printf("[CONFIG] total=%zu MB grid=%d block=%d iters=%d warmup=%d src=%d dst=%d\\n"
         "         proto=%d pipe=%d issue_warp=always smem=%zu KB (slot=%zu B)\\n"
         "         simple_slice=%zu KB | tma_slice=%zu KB tma_tile=%zu KB\\n"
         "         traffic=%s target=%s traffic_mb=%zu traffic_grid=%d traffic_block=%d traffic_rounds=%d\\n",
         a.total_mb, a.grid, a.block, a.iters, a.warmup, a.src_dev, a.dst_dev,
         a.proto, a.pipe, a.smem_kb, tma_slot_bytes,
         a.simple_slice_kb, a.tma_slice_kb, a.tma_tile_kb,
         traffic_mode_str(a.traffic_mode), traffic_target_str(a.traffic_target),
         a.traffic_mb, a.traffic_grid, a.traffic_block, a.traffic_rounds);

  size_t wrong_tma = 0;
  size_t wrong_simple = 0;

  if (a.proto == 0 || a.proto == 1) {
    CUDA_CHECK(cudaSetDevice(a.dst_dev));
    CUDA_CHECK(cudaMemset(d_dst, 0, total_bytes));

    CUDA_CHECK(cudaSetDevice(a.src_dev));
    TrafficContext traffic;
    start_traffic(a, traffic);
    BenchResult r = run_tma(a, d_dst, d_src, total_bytes, chunk_bytes,
                            tma_slice_bytes, tma_tile_bytes, smem_bytes, tma_slot_bytes);
    stop_traffic(traffic);

    if (a.verify) wrong_tma = verify_copy(a.src_dev, a.dst_dev, d_src, d_dst, n_elem);
    print_result("TMA", r, wrong_tma);
  }

  if (a.proto == 0 || a.proto == 2) {
    CUDA_CHECK(cudaSetDevice(a.dst_dev));
    CUDA_CHECK(cudaMemset(d_dst, 0, total_bytes));

    CUDA_CHECK(cudaSetDevice(a.src_dev));
    TrafficContext traffic;
    start_traffic(a, traffic);
    BenchResult r = run_simple(a, d_dst, d_src, total_bytes, chunk_bytes, simple_slice_bytes);
    stop_traffic(traffic);

    if (a.verify) wrong_simple = verify_copy(a.src_dev, a.dst_dev, d_src, d_dst, n_elem);
    print_result("SIMPLE", r, wrong_simple);
  }

  CUDA_CHECK(cudaSetDevice(a.src_dev));
  if (d_src) CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaSetDevice(a.dst_dev));
  if (d_dst) CUDA_CHECK(cudaFree(d_dst));

  const bool ok = (!a.verify) || (wrong_tma == 0 && wrong_simple == 0);
  return ok ? 0 : 2;
}
