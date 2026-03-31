#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/ptx>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

/*
 * This benchmark compares two different levels of ring all-gather cost.
 *
 * 1. MICRO paths
 *    - Reuse the custom peer-transfer kernels built for the transfer microbenchmark.
 *    - Each GPU explicitly writes its payload into the peer GPU's recv buffer.
 *    - This measures a stripped-down "data movement only" version of ring all-gather.
 *
 * 2. NCCL paths
 *    - Call real ncclAllGather() on the same input size and same GPUs.
 *    - This includes NCCL's runtime / enqueue / channel / protocol machinery.
 *
 * Important scope:
 *   - The MICRO path explicitly constructs a ring all-gather using peer writes.
 *   - The intended comparison point is a 4-GPU ring, but the code supports any
 *     number of ranks >= 2 passed in --devices.
 *   - Each rank owns one chunk in sendbuf and must end with all chunks laid out in
 *     rank order inside recvbuf.
 *   - The NCCL path is still the full library all-gather, so this file compares:
 *       custom ring data path vs NCCL full-stack ring all-gather
 */

using bf16 = __nv_bfloat16;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#define CUDA_CHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

#define NCCL_CHECK(cmd) do { \
  ncclResult_t r = (cmd); \
  if (r != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    std::exit(2); \
  } \
} while (0)

static constexpr int kMaxPipeDepth = 8;
static constexpr int kMinNranks = 2;

// User-facing configuration.
// The key distinction is:
//   - send_mb          : bytes contributed by each rank
//   - grid/block       : launch shape for the custom MICRO peer-transfer kernels
//   - pipe/smem/tile   : only affect the MICRO_TMA path
//   - run_* flags      : select which benchmark variants to execute
struct Args {
  size_t send_mb = 128;
  int iters = 20;
  int warmup = 3;
  std::vector<int> devices = {0, 1, 2, 3};

  int grid = 32;
  int block = 544;

  int pipe = 2;
  size_t smem_kb = 64;
  int tma_issue_warp = 1;

  size_t simple_slice_kb = 1024;
  size_t tma_slice_kb = 1024;
  size_t tma_tile_kb = 32;

  int verify = 1;

  bool run_micro_simple = true;
  bool run_micro_tma = true;
  bool run_nccl_simple = true;
  bool run_nccl_tma = true;
};

// Common result container used by MICRO and NCCL runs.
// avg_ms is the per-iteration end-to-end time seen by the slowest rank.
struct BenchResult {
  float avg_ms = 0.0f;
  float gbps = 0.0f;
  size_t wrong = 0;
};

static inline int nranks_of(const Args& a) {
  return int(a.devices.size());
}

static inline int prev_rank(int rank, int nranks) {
  return (rank + nranks - 1) % nranks;
}

static inline int next_rank(int rank, int nranks) {
  return (rank + 1) % nranks;
}

static inline std::string format_devices(const std::vector<int>& devices) {
  std::string out;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(devices[i]);
  }
  return out;
}

__device__ __forceinline__ size_t div_up(size_t a, size_t b) {
  return (a + b - 1) / b;
}

// Fill each rank's input buffer with a rank-specific pattern so that verification
// can later distinguish:
//   - "this chunk came from rank 0"
//   - "this chunk came from rank 1"
//
// We do not need random values here. A deterministic pattern is better because:
//   - verification is reproducible
//   - debugging mismatches is easier
__global__ void init_rank_pattern(bf16* dst, size_t n, int rank) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = float((rank + 1) * 1000) + float(i % 997);
    dst[i] = __float2bfloat16(v);
  }
}

/*
 * Simple peer-transfer kernel
 *
 * Data path:
 *   src (local GPU global memory) -> dst_peer (peer GPU global memory)
 *
 * Mapping:
 *   Each CTA owns a contiguous chunk of the transfer.
 *   Inside a chunk, data is visited in "simple slices".
 *   Inside a slice, worker threads move 16B vectors (`uint4`) with unroll 8.
 *
 * Design choice:
 *   We intentionally reserve one warp when blockDim is large enough, matching the
 *   thread budgeting style used in the earlier microbenchmark and keeping worker
 *   count aligned with the TMA path.
 */
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
    const size_t curr_slice_bytes = std::min(simple_slice_bytes, chunk_bytes - slice_off);
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

/*
 * TMA-style peer-transfer kernel
 *
 * Data path:
 *   src (local GPU global memory)
 *     -> shared-memory slot (filled by cuda::memcpy_async)
 *     -> dst_peer (peer GPU global memory)
 *
 * Hierarchy:
 *   CTA chunk -> TMA slice -> TMA tile
 *
 * Core idea:
 *   - One "issuer" thread is responsible for posting async copies into shared memory.
 *   - Compute workers drain the ready shared-memory tile into the peer GPU buffer.
 *   - Multiple slots form a ring buffer so copy(issue) and copy-out(consume) can overlap.
 *
 * This is still a MICRO kernel:
 *   it models the data-transfer mechanism of the TMA path, not the full NCCL runtime.
 */
__global__ void kernel_tma_transfer(
    bf16* __restrict__ dst_peer,
    const bf16* __restrict__ src,
    size_t chunk_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes,
    size_t tma_slot_bytes,
    int pipe_depth,
    int enable_issue_warp) {
  extern __shared__ __align__(16) unsigned char smem_all[];

  // One barrier per slot. Each slot is reused over time, so we need per-slot
  // synchronization to know when:
  //   1) the async global->smem copy is complete
  //   2) it is safe to refill that slot with a later tile
  __shared__ barrier_t bars[kMaxPipeDepth];
  barrier_t::arrival_token tokens[kMaxPipeDepth];

  // Shared memory is carved into pipe_depth equal-sized slots.
  // The caller guarantees:
  //   tma_slot_bytes == tma_tile_bytes
  // so each slot holds exactly one tile.
  unsigned char* smem_slot[kMaxPipeDepth];
  #pragma unroll
  for (int i = 0; i < kMaxPipeDepth; ++i) {
    smem_slot[i] = smem_all + size_t(i) * tma_slot_bytes;
  }

  const size_t tid = threadIdx.x;
  const size_t cta_start = blockIdx.x * chunk_bytes;
  const int nworkers = blockDim.x;

  // kCopyBytesPerWarp captures how much data one warp can cover in one pass of
  // the unrolled vector loop:
  //   unroll(8) * 32 threads * 16 B = 4096 B per warp
  //
  // For small tiles, we intentionally activate only the number of worker warps
  // needed to cover the tile, mirroring the idea behind NCCL's reduceCopyShared.
  constexpr int kWarpSize = 32;
  constexpr int kCopyUnroll = 8;
  constexpr size_t kCopyBytesPerWarp = size_t(kCopyUnroll) * size_t(kWarpSize) * 16ULL;
  const bool allow_issue_warp = (enable_issue_warp != 0) && (pipe_depth >= 2);
  const int tma_compute_workers =
      (allow_issue_warp && nworkers >= 2 * kWarpSize) ? (nworkers - kWarpSize) : nworkers;
  const bool tma_issue_warp_enabled = (tma_compute_workers < nworkers);
  const int tma_issue_tid = tma_compute_workers;
  const bool is_tma_compute_thread = int(tid) < tma_compute_workers;
  const bool is_tma_issuer_thread = tma_issue_warp_enabled ? (int(tid) == tma_issue_tid) : (tid == 0);

  uint4* dst_v = reinterpret_cast<uint4*>(dst_peer);

  // Initialize the block-local barriers once.
  // fence_proxy_async is required so async proxy operations and shared memory
  // accesses observe the same ordering rules.
  if (threadIdx.x == 0) {
    for (int i = 0; i < pipe_depth; ++i) init(&bars[i], blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);
  }
  __syncthreads();

  // The CTA's chunk is processed slice-by-slice.
  // Each slice is further split into fixed-size tiles, which are what the TMA
  // pipeline actually moves through shared memory.
  for (size_t slice_off = 0; slice_off < chunk_bytes; slice_off += tma_slice_bytes) {
    const size_t curr_slice_bytes = std::min(tma_slice_bytes, chunk_bytes - slice_off);
    const size_t tile_count = div_up(curr_slice_bytes, tma_tile_bytes);

    // Prologue:
    // Fill up to `pipe_depth` slots before consumption begins.
    int in_flight = 0;
    for (; in_flight < pipe_depth && in_flight < (int)tile_count; ++in_flight) {
      const size_t tile_idx = size_t(in_flight);
      const size_t tile_off = tile_idx * tma_tile_bytes;
      const size_t copy_bytes = std::min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const int slot = in_flight % pipe_depth;

      if (is_tma_issuer_thread) {
        cuda::memcpy_async(
            smem_slot[slot],
            reinterpret_cast<const unsigned char*>(src) + cta_start + slice_off + tile_off,
            cuda::aligned_size_t<16>(copy_bytes),
            bars[slot]);
      }

      // All threads participate in the barrier arrival protocol even though only
      // the issuer thread posts the async copy. Later wait() will block until the
      // async copy into this slot is complete.
      tokens[slot] = bars[slot].arrive();
    }

    // Steady-state pipeline:
    //   wait current slot
    //   consume current tile (smem -> peer global)
    //   refill the freed slot with a future tile
    for (size_t t = 0; t < tile_count; ++t) {
      const int slot = int(t % size_t(pipe_depth));
      const size_t tile_off = t * tma_tile_bytes;
      const size_t copy_bytes = std::min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const size_t vec_count = copy_bytes / 16;

      // This wait is the point where the copy workers are guaranteed to see a
      // fully populated shared-memory tile.
      bars[slot].wait(std::move(tokens[slot]));

      const uint4* smem_v = reinterpret_cast<const uint4*>(smem_slot[slot]);
      const size_t dst_vec_base = (cta_start + slice_off + tile_off) / 16;

      int max_compute_warps = tma_compute_workers / kWarpSize;
      if (max_compute_warps < 1) max_compute_warps = 1;
      int active_compute_warps = int((copy_bytes + kCopyBytesPerWarp - 1) / kCopyBytesPerWarp);
      if (active_compute_warps > max_compute_warps) active_compute_warps = max_compute_warps;
      int active_compute_threads = active_compute_warps * kWarpSize;
      if (active_compute_threads > tma_compute_workers) active_compute_threads = tma_compute_workers;

      // Only the active subset of compute threads participates for this tile.
      // Example:
      //   32 KB tile / 4 KB per warp = 8 active warps = 256 active threads.
      if (is_tma_compute_thread && int(tid) < active_compute_threads) {
        const size_t stride = size_t(active_compute_threads);
        #pragma unroll 8
        for (size_t v = tid; v < vec_count; v += stride) {
          uint4 x = smem_v[v];
          dst_v[dst_vec_base + v] = x;
        }
      }

      // This block-wide sync is required before the same slot can be reused.
      // Without it, the issuer could overwrite smem_slot[slot] while another
      // worker thread is still draining that slot to the peer buffer.
      __syncthreads();

      // Refill the slot we just consumed.
      // The slot index is reused in strict ring-buffer order.
      const size_t issue_idx = t + size_t(pipe_depth);
      if (issue_idx < tile_count) {
        const size_t issue_off = issue_idx * tma_tile_bytes;
        const size_t issue_copy_bytes = std::min(tma_tile_bytes, curr_slice_bytes - issue_off);

        if (is_tma_issuer_thread) {
          cuda::memcpy_async(
              smem_slot[slot],
              reinterpret_cast<const unsigned char*>(src) + cta_start + slice_off + issue_off,
              cuda::aligned_size_t<16>(issue_copy_bytes),
              bars[slot]);
        }
        tokens[slot] = bars[slot].arrive();
      }
    }
  }
}

// CLI help text. This benchmark is intended to be driven from the shell or from
// a sweep script, so it keeps all tuning knobs explicit.
static inline void usage_and_exit() {
  printf("Usage: ./allgather_micro_nccl_bench [options]\n"
         "  --send_mb N            Per-rank input size in MB (default: 128)\n"
         "  --iters N              Timed iterations (default: 20)\n"
         "  --warmup N             Warmup iterations (default: 3)\n"
         "  --devices A,B,C,D      GPU ids in ring order (default: 0,1,2,3)\n"
         "  --grid N               CTAs per remote transfer kernel (default: 32)\n"
         "  --block N              Threads per CTA (default: 544)\n"
         "  --pipe N               TMA pipe depth (default: 2)\n"
         "  --smem_kb N            TMA dynamic shared memory per CTA (default: 64)\n"
         "  --tma_issue_warp N     0=off, 1=on (default: 1)\n"
         "  --simple_slice_kb N    Simple slice size in KB (default: 1024)\n"
         "  --tma_slice_kb N       TMA slice size in KB (default: 1024)\n"
         "  --tma_tile_kb N        TMA tile size in KB (default: 32)\n"
         "  --verify N             0=off, 1=on (default: 1)\n"
         "  --bench LIST           Comma list from: micro_simple,micro_tma,nccl_simple,nccl_tma,all\n");
  std::exit(0);
}

// Parse "0,1,2,3" style device lists.
static inline std::vector<int> parse_devices(const std::string& spec) {
  std::vector<int> out;
  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find(',', start);
    std::string token = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!token.empty()) out.push_back(std::stoi(token));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return out;
}

// Decode which benchmark variants should be run in this invocation.
// Example:
//   --bench micro_tma,nccl_tma
// runs only those two lines and skips the others.
static inline void parse_bench_list(const std::string& spec, Args& a) {
  a.run_micro_simple = false;
  a.run_micro_tma = false;
  a.run_nccl_simple = false;
  a.run_nccl_tma = false;

  size_t start = 0;
  while (start < spec.size()) {
    size_t end = spec.find(',', start);
    std::string token = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (token == "all") {
      a.run_micro_simple = true;
      a.run_micro_tma = true;
      a.run_nccl_simple = true;
      a.run_nccl_tma = true;
    } else if (token == "micro_simple") {
      a.run_micro_simple = true;
    } else if (token == "micro_tma") {
      a.run_micro_tma = true;
    } else if (token == "nccl_simple") {
      a.run_nccl_simple = true;
    } else if (token == "nccl_tma") {
      a.run_nccl_tma = true;
    } else if (!token.empty()) {
      fprintf(stderr, "Unknown --bench token: %s\n", token.c_str());
      std::exit(2);
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
}

// Straightforward command-line parser. We intentionally keep the parser simple
// because the benchmark is expected to be launched from scripts with known args.
static inline void parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](int n) {
      if (i + n >= argc) {
        fprintf(stderr, "Missing value for %s\n", k.c_str());
        std::exit(2);
      }
    };

    if (k == "--help" || k == "-h") {
      usage_and_exit();
    } else if (k == "--send_mb") {
      need(1); a.send_mb = std::stoull(argv[++i]);
    } else if (k == "--iters") {
      need(1); a.iters = std::stoi(argv[++i]);
    } else if (k == "--warmup") {
      need(1); a.warmup = std::stoi(argv[++i]);
    } else if (k == "--devices") {
      need(1); a.devices = parse_devices(argv[++i]);
    } else if (k == "--grid") {
      need(1); a.grid = std::stoi(argv[++i]);
    } else if (k == "--block") {
      need(1); a.block = std::stoi(argv[++i]);
    } else if (k == "--pipe") {
      need(1); a.pipe = std::stoi(argv[++i]);
    } else if (k == "--smem_kb") {
      need(1); a.smem_kb = std::stoull(argv[++i]);
    } else if (k == "--tma_issue_warp") {
      need(1); a.tma_issue_warp = std::stoi(argv[++i]);
    } else if (k == "--simple_slice_kb") {
      need(1); a.simple_slice_kb = std::stoull(argv[++i]);
    } else if (k == "--tma_slice_kb") {
      need(1); a.tma_slice_kb = std::stoull(argv[++i]);
    } else if (k == "--tma_tile_kb") {
      need(1); a.tma_tile_kb = std::stoull(argv[++i]);
    } else if (k == "--verify") {
      need(1); a.verify = std::stoi(argv[++i]);
    } else if (k == "--bench") {
      need(1); parse_bench_list(argv[++i], a);
    } else {
      fprintf(stderr, "Unknown arg: %s\n", k.c_str());
      std::exit(2);
    }
  }
}

// Validate all size relationships once up front.
// The most important TMA-specific constraint is:
//   smem_bytes / pipe_depth == tma_tile_bytes
// so each pipeline slot holds exactly one tile.
static inline void validate_or_die(
    const Args& a,
    size_t send_bytes,
    size_t chunk_bytes,
    size_t smem_bytes,
    size_t tma_slot_bytes,
    size_t simple_slice_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes) {
  if (a.devices.size() < kMinNranks) {
    fprintf(stderr, "This benchmark requires at least %d devices\n", kMinNranks);
    std::exit(2);
  }
  if (a.send_mb == 0 || a.grid <= 0 || a.block <= 0 || a.iters <= 0 || a.warmup < 0) {
    fprintf(stderr, "Invalid send/grid/block/iters/warmup\n");
    std::exit(2);
  }
  if (a.pipe < 1 || a.pipe > kMaxPipeDepth) {
    fprintf(stderr, "--pipe must be in [1, %d]\n", kMaxPipeDepth);
    std::exit(2);
  }
  if (!(a.tma_issue_warp == 0 || a.tma_issue_warp == 1)) {
    fprintf(stderr, "--tma_issue_warp must be 0 or 1\n");
    std::exit(2);
  }
  if (send_bytes == 0 || (send_bytes % sizeof(bf16)) != 0) {
    fprintf(stderr, "send_bytes must be >0 and divisible by sizeof(bf16)\n");
    std::exit(2);
  }
  if ((send_bytes % size_t(a.grid)) != 0) {
    fprintf(stderr, "send_bytes must be divisible by grid\n");
    std::exit(2);
  }
  if (chunk_bytes == 0 || (chunk_bytes % 16) != 0) {
    fprintf(stderr, "chunk_bytes must be >0 and 16B aligned\n");
    std::exit(2);
  }
  if (simple_slice_bytes == 0 || (simple_slice_bytes % 16) != 0) {
    fprintf(stderr, "simple_slice_bytes must be >0 and 16B aligned\n");
    std::exit(2);
  }
  if (tma_slice_bytes == 0 || (tma_slice_bytes % 16) != 0) {
    fprintf(stderr, "tma_slice_bytes must be >0 and 16B aligned\n");
    std::exit(2);
  }
  if (tma_tile_bytes == 0 || (tma_tile_bytes % 16) != 0) {
    fprintf(stderr, "tma_tile_bytes must be >0 and 16B aligned\n");
    std::exit(2);
  }
  if (smem_bytes == 0 || (smem_bytes % size_t(a.pipe)) != 0) {
    fprintf(stderr, "smem_bytes must be >0 and divisible by pipe\n");
    std::exit(2);
  }
  if ((tma_slot_bytes % 16) != 0) {
    fprintf(stderr, "tma_slot_bytes must be 16B aligned\n");
    std::exit(2);
  }
  if (tma_slot_bytes != tma_tile_bytes) {
    fprintf(stderr, "Require smem_bytes/pipe == tma_tile_bytes for TMA path\n");
    std::exit(2);
  }
}

// Convert timed bytes / iteration into GB/s.
// For 2-rank all-gather, we report output bytes per rank:
//   recv_bytes = 2 * send_bytes
// This makes all variants directly comparable using the same denominator.
static inline float bytes_to_gbps(size_t bytes, float ms) {
  if (ms <= 0.0f) return 0.0f;
  return float((double(bytes) / 1.0e9) / (double(ms) * 1.0e-3));
}

// Each rank has its own stream and event pair.
// We take the slowest rank's elapsed time because the collective iteration is
// only complete once every rank finishes.
static inline float elapsed_max_ms(const Args& a,
                                   const std::vector<cudaEvent_t>& starts,
                                   const std::vector<cudaEvent_t>& stops) {
  float max_ms = 0.0f;
  for (int r = 0; r < nranks_of(a); ++r) {
    float ms = 0.0f;
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventElapsedTime(&ms, starts[r], stops[r]));
    if (ms > max_ms) max_ms = ms;
  }
  return max_ms;
}

// Force NCCL into a specific protocol for the next communicator initialization.
// We intentionally pin the algorithm to Ring because that is the path we want
// to compare against the custom ring micro implementation.
static inline void set_nccl_env(const char* proto) {
  setenv("NCCL_ALGO", "Ring", 1);
  setenv("NCCL_PROTO", proto, 1);
}

// Copy the results back to the host and verify the exact all-gather layout.
//
// Expected recv layout on every rank:
//   recv = [ chunk_from_rank0 | chunk_from_rank1 | ... | chunk_from_rank{nranks-1} ]
static inline size_t verify_allgather(const Args& a,
                                      const std::vector<bf16*>& d_send,
                                      const std::vector<bf16*>& d_recv,
                                      size_t send_elems,
                                      size_t max_report = 8) {
  size_t wrong = 0;
  const int nranks = nranks_of(a);
  std::vector<std::vector<uint16_t>> h_send(nranks);
  std::vector<std::vector<uint16_t>> h_recv(nranks);

  for (int r = 0; r < nranks; ++r) {
    h_send[r].resize(send_elems);
    h_recv[r].resize(send_elems * nranks);

    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaMemcpy(h_send[r].data(), d_send[r], send_elems * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_recv[r].data(), d_recv[r], send_elems * nranks * sizeof(bf16), cudaMemcpyDeviceToHost));
  }

  for (int dst_rank = 0; dst_rank < nranks; ++dst_rank) {
    for (int src_rank = 0; src_rank < nranks; ++src_rank) {
      const size_t recv_base = size_t(src_rank) * send_elems;
      for (size_t i = 0; i < send_elems; ++i) {
        if (h_recv[dst_rank][recv_base + i] != h_send[src_rank][i]) {
          if (wrong < max_report) {
            bf16 a_v = *reinterpret_cast<const bf16*>(&h_send[src_rank][i]);
            bf16 b_v = *reinterpret_cast<const bf16*>(&h_recv[dst_rank][recv_base + i]);
            fprintf(stderr,
                    "mismatch dst_rank=%d src_rank=%d elem=%zu src=0x%04x recv=0x%04x (%.3f vs %.3f)\n",
                    dst_rank, src_rank, i, h_send[src_rank][i], h_recv[dst_rank][recv_base + i],
                    __bfloat162float(a_v), __bfloat162float(b_v));
          }
          ++wrong;
        }
      }
    }
  }

  return wrong;
}

// One MICRO_SIMPLE ring all-gather iteration.
//
// Initialization:
//   Each rank copies its own local input into recv[rank].
//
// Ring step s:
//   rank r forwards chunk (r - s) to next(r).
//   The next rank stores that chunk in the slot identified by the source rank.
//
// After nranks-1 steps, every rank has received every other rank's chunk.
static inline void micro_simple_iter(const Args& a,
                                     const std::vector<bf16*>& d_send,
                                     const std::vector<bf16*>& d_recv,
                                     const std::vector<cudaStream_t>& streams,
                                     const std::vector<std::vector<cudaEvent_t>>& step_done,
                                     const std::vector<cudaEvent_t>& iter_done,
                                     size_t send_bytes,
                                     size_t send_elems,
                                     size_t chunk_bytes,
                                     size_t simple_slice_bytes) {
  const int nranks = nranks_of(a);
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaMemcpyAsync(d_recv[r] + size_t(r) * send_elems, d_send[r], send_bytes,
                               cudaMemcpyDeviceToDevice, streams[r]));
  }

  for (int step = 0; step < nranks - 1; ++step) {
    for (int r = 0; r < nranks; ++r) {
      const int prev = prev_rank(r, nranks);
      const int next = next_rank(r, nranks);
      const int chunk_rank = (r - step + nranks) % nranks;

      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      if (step > 0) CUDA_CHECK(cudaStreamWaitEvent(streams[r], step_done[prev][step - 1], 0));

      kernel_simple_transfer<<<a.grid, a.block, 0, streams[r]>>>(
          d_recv[next] + size_t(chunk_rank) * send_elems,
          d_recv[r] + size_t(chunk_rank) * send_elems,
          chunk_bytes, simple_slice_bytes);
      CUDA_CHECK(cudaEventRecord(step_done[r][step], streams[r]));
    }
  }

  // End-of-iteration barrier:
  // prevent rank r from starting the next iteration and overwriting next(r)'s
  // recv buffer before next(r) has finished consuming the current iteration.
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(iter_done[r], streams[r]));
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamWaitEvent(streams[r], iter_done[next_rank(r, nranks)], 0));
  }
}

// Same ring schedule as micro_simple_iter(), but each ring step uses the
// TMA-style peer-transfer kernel.
static inline void micro_tma_iter(const Args& a,
                                  const std::vector<bf16*>& d_send,
                                  const std::vector<bf16*>& d_recv,
                                  const std::vector<cudaStream_t>& streams,
                                  const std::vector<std::vector<cudaEvent_t>>& step_done,
                                  const std::vector<cudaEvent_t>& iter_done,
                                  size_t send_bytes,
                                  size_t send_elems,
                                  size_t chunk_bytes,
                                  size_t tma_slice_bytes,
                                  size_t tma_tile_bytes,
                                  size_t tma_slot_bytes,
                                  size_t smem_bytes) {
  const int nranks = nranks_of(a);
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaMemcpyAsync(d_recv[r] + size_t(r) * send_elems, d_send[r], send_bytes,
                               cudaMemcpyDeviceToDevice, streams[r]));
  }

  for (int step = 0; step < nranks - 1; ++step) {
    for (int r = 0; r < nranks; ++r) {
      const int prev = prev_rank(r, nranks);
      const int next = next_rank(r, nranks);
      const int chunk_rank = (r - step + nranks) % nranks;

      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      if (step > 0) CUDA_CHECK(cudaStreamWaitEvent(streams[r], step_done[prev][step - 1], 0));

      kernel_tma_transfer<<<a.grid, a.block, smem_bytes, streams[r]>>>(
          d_recv[next] + size_t(chunk_rank) * send_elems,
          d_recv[r] + size_t(chunk_rank) * send_elems,
          chunk_bytes, tma_slice_bytes, tma_tile_bytes, tma_slot_bytes,
          a.pipe, a.tma_issue_warp);
      CUDA_CHECK(cudaEventRecord(step_done[r][step], streams[r]));
    }
  }

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(iter_done[r], streams[r]));
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamWaitEvent(streams[r], iter_done[next_rank(r, nranks)], 0));
  }
}

// Time the MICRO_SIMPLE path.
//
// Measurement policy:
//   - warmup iterations are excluded
//   - one start/stop event pair per GPU encloses the entire repeated loop
//   - average time is total_time / iters
//
// This matches the style used in the earlier transfer benchmark:
// we care about end-to-end kernel time, not per-slice internal cycle accounting.
static BenchResult run_micro_simple(const Args& a,
                                    const std::vector<bf16*>& d_send,
                                    const std::vector<bf16*>& d_recv,
                                    const std::vector<cudaStream_t>& streams,
                                    size_t send_bytes,
                                    size_t send_elems,
                                    size_t out_bytes_per_rank,
                                    size_t chunk_bytes,
                                    size_t simple_slice_bytes) {
  BenchResult out;
  const int nranks = nranks_of(a);
  std::vector<cudaEvent_t> start(nranks), stop(nranks), iter_done(nranks);
  std::vector<std::vector<cudaEvent_t>> step_done(nranks, std::vector<cudaEvent_t>(nranks - 1));
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventCreate(&start[r]));
    CUDA_CHECK(cudaEventCreate(&stop[r]));
    CUDA_CHECK(cudaEventCreateWithFlags(&iter_done[r], cudaEventDisableTiming));
    for (int step = 0; step < nranks - 1; ++step) {
      CUDA_CHECK(cudaEventCreateWithFlags(&step_done[r][step], cudaEventDisableTiming));
    }
  }

  for (int it = 0; it < a.warmup; ++it) {
    micro_simple_iter(a, d_send, d_recv, streams, step_done, iter_done,
                      send_bytes, send_elems, chunk_bytes, simple_slice_bytes);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamSynchronize(streams[r]));
  }

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(start[r], streams[r]));
  }
  for (int it = 0; it < a.iters; ++it) {
    micro_simple_iter(a, d_send, d_recv, streams, step_done, iter_done,
                      send_bytes, send_elems, chunk_bytes, simple_slice_bytes);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(stop[r], streams[r]));
    CUDA_CHECK(cudaEventSynchronize(stop[r]));
  }

  out.avg_ms = elapsed_max_ms(a, start, stop) / float(a.iters);
  out.gbps = bytes_to_gbps(out_bytes_per_rank, out.avg_ms);
  if (a.verify) out.wrong = verify_allgather(a, d_send, d_recv, send_elems);

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventDestroy(start[r]));
    CUDA_CHECK(cudaEventDestroy(stop[r]));
    CUDA_CHECK(cudaEventDestroy(iter_done[r]));
    for (int step = 0; step < nranks - 1; ++step) CUDA_CHECK(cudaEventDestroy(step_done[r][step]));
  }
  return out;
}

// Time the MICRO_TMA path.
// The only extra setup relative to MICRO_SIMPLE is opting the kernel into the
// requested dynamic shared memory size on each GPU before launch.
static BenchResult run_micro_tma(const Args& a,
                                 const std::vector<bf16*>& d_send,
                                 const std::vector<bf16*>& d_recv,
                                 const std::vector<cudaStream_t>& streams,
                                 size_t send_bytes,
                                 size_t send_elems,
                                 size_t out_bytes_per_rank,
                                 size_t chunk_bytes,
                                 size_t tma_slice_bytes,
                                 size_t tma_tile_bytes,
                                 size_t tma_slot_bytes,
                                 size_t smem_bytes) {
  BenchResult out;
  const int nranks = nranks_of(a);
  std::vector<cudaEvent_t> start(nranks), stop(nranks), iter_done(nranks);
  std::vector<std::vector<cudaEvent_t>> step_done(nranks, std::vector<cudaEvent_t>(nranks - 1));

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_tma_transfer,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    int(smem_bytes)));
    CUDA_CHECK(cudaEventCreate(&start[r]));
    CUDA_CHECK(cudaEventCreate(&stop[r]));
    CUDA_CHECK(cudaEventCreateWithFlags(&iter_done[r], cudaEventDisableTiming));
    for (int step = 0; step < nranks - 1; ++step) {
      CUDA_CHECK(cudaEventCreateWithFlags(&step_done[r][step], cudaEventDisableTiming));
    }
  }

  for (int it = 0; it < a.warmup; ++it) {
    micro_tma_iter(a, d_send, d_recv, streams, step_done, iter_done,
                   send_bytes, send_elems, chunk_bytes, tma_slice_bytes,
                   tma_tile_bytes, tma_slot_bytes, smem_bytes);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamSynchronize(streams[r]));
  }

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(start[r], streams[r]));
  }
  for (int it = 0; it < a.iters; ++it) {
    micro_tma_iter(a, d_send, d_recv, streams, step_done, iter_done,
                   send_bytes, send_elems, chunk_bytes, tma_slice_bytes,
                   tma_tile_bytes, tma_slot_bytes, smem_bytes);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(stop[r], streams[r]));
    CUDA_CHECK(cudaEventSynchronize(stop[r]));
  }

  out.avg_ms = elapsed_max_ms(a, start, stop) / float(a.iters);
  out.gbps = bytes_to_gbps(out_bytes_per_rank, out.avg_ms);
  if (a.verify) out.wrong = verify_allgather(a, d_send, d_recv, send_elems);

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventDestroy(start[r]));
    CUDA_CHECK(cudaEventDestroy(stop[r]));
    CUDA_CHECK(cudaEventDestroy(iter_done[r]));
    for (int step = 0; step < nranks - 1; ++step) CUDA_CHECK(cudaEventDestroy(step_done[r][step]));
  }
  return out;
}

// Time the real NCCL all-gather path under a fixed protocol.
//
// We recreate communicators per protocol choice so that:
//   setenv("NCCL_PROTO", ...)
// takes effect during communicator initialization.
//
// This function is the "full stack" comparison point against the MICRO runs.
static BenchResult run_nccl_allgather(const Args& a,
                                      const std::vector<bf16*>& d_send,
                                      const std::vector<bf16*>& d_recv,
                                      size_t send_elems,
                                      size_t out_bytes_per_rank,
                                      const char* proto_name) {
  BenchResult out;
  const int nranks = nranks_of(a);
  std::vector<ncclComm_t> comms(nranks);
  std::vector<cudaStream_t> streams(nranks);
  std::vector<cudaEvent_t> start(nranks), stop(nranks);

  set_nccl_env(proto_name);
  NCCL_CHECK(ncclCommInitAll(comms.data(), nranks, a.devices.data()));

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start[r]));
    CUDA_CHECK(cudaEventCreate(&stop[r]));
  }

  for (int it = 0; it < a.warmup; ++it) {
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < nranks; ++r) {
      NCCL_CHECK(ncclAllGather(d_send[r], d_recv[r], send_elems, ncclBfloat16, comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamSynchronize(streams[r]));
  }

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(start[r], streams[r]));
  }
  for (int it = 0; it < a.iters; ++it) {
    NCCL_CHECK(ncclGroupStart());
    for (int r = 0; r < nranks; ++r) {
      NCCL_CHECK(ncclAllGather(d_send[r], d_recv[r], send_elems, ncclBfloat16, comms[r], streams[r]));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventRecord(stop[r], streams[r]));
    CUDA_CHECK(cudaEventSynchronize(stop[r]));
  }

  out.avg_ms = elapsed_max_ms(a, start, stop) / float(a.iters);
  out.gbps = bytes_to_gbps(out_bytes_per_rank, out.avg_ms);
  if (a.verify) out.wrong = verify_allgather(a, d_send, d_recv, send_elems);

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaEventDestroy(start[r]));
    CUDA_CHECK(cudaEventDestroy(stop[r]));
    CUDA_CHECK(cudaStreamDestroy(streams[r]));
    NCCL_CHECK(ncclCommFinalize(comms[r]));
    NCCL_CHECK(ncclCommDestroy(comms[r]));
  }

  return out;
}

int main(int argc, char** argv) {
  Args a;
  parse_args(argc, argv, a);
  const int nranks = nranks_of(a);

  // send_bytes is the per-rank input payload size.
  // In ring all-gather, each rank's output holds all nranks chunks.
  const size_t send_bytes = a.send_mb * 1024ULL * 1024ULL;
  const size_t send_elems = send_bytes / sizeof(bf16);
  const size_t recv_elems = send_elems * nranks;
  const size_t recv_bytes = recv_elems * sizeof(bf16);
  const size_t out_bytes_per_rank = recv_bytes;

  // The MICRO kernels use send_bytes as the transferred remote payload.
  // Each CTA owns send_bytes / grid bytes of that payload.
  const size_t simple_slice_bytes = a.simple_slice_kb * 1024ULL;
  const size_t tma_slice_bytes = a.tma_slice_kb * 1024ULL;
  const size_t tma_tile_bytes = a.tma_tile_kb * 1024ULL;
  const size_t smem_bytes = a.smem_kb * 1024ULL;
  const size_t chunk_bytes = send_bytes / size_t(a.grid);
  const size_t tma_slot_bytes = smem_bytes / size_t(a.pipe);

  validate_or_die(a, send_bytes, chunk_bytes, smem_bytes, tma_slot_bytes,
                  simple_slice_bytes, tma_slice_bytes, tma_tile_bytes);

  // Sanity-check the selected device ids and peer accessibility.
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  for (int dev : a.devices) {
    if (dev < 0 || dev >= ndev) {
      fprintf(stderr, "Invalid device id %d\n", dev);
      return 2;
    }
  }

  for (int r = 0; r < nranks; ++r) {
    int can_access = 0;
    const int next = next_rank(r, nranks);
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, a.devices[r], a.devices[next]));
    if (!can_access) {
      fprintf(stderr, "Peer access not available from %d to %d for ring step\n",
              a.devices[r], a.devices[next]);
      return 2;
    }
  }

  // Enable peer access along the ring because each rank writes directly into the
  // next rank's recv buffer in the MICRO ring path.
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    int peer = a.devices[next_rank(r, nranks)];
    cudaError_t pe = cudaDeviceEnablePeerAccess(peer, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(pe);
  }

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    int max_dyn_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_dyn_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, a.devices[r]));
    if (int(smem_bytes) > max_dyn_smem) {
      fprintf(stderr, "Requested smem=%zu exceeds max dynamic shared memory=%d on device %d\n",
              smem_bytes, max_dyn_smem, a.devices[r]);
      return 2;
    }
  }

  // One send buffer + one recv buffer per GPU.
  // recv is sized for the full all-gather result on that rank.
  std::vector<bf16*> d_send(nranks, nullptr);
  std::vector<bf16*> d_recv(nranks, nullptr);
  std::vector<cudaStream_t> micro_streams(nranks);

  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamCreateWithFlags(&micro_streams[r], cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&d_send[r], send_bytes));
    CUDA_CHECK(cudaMalloc(&d_recv[r], recv_bytes));
    CUDA_CHECK(cudaMemset(d_recv[r], 0, recv_bytes));
    int bs = 256;
    int gs = int((send_elems + size_t(bs) - 1) / size_t(bs));
    init_rank_pattern<<<gs, bs, 0, micro_streams[r]>>>(d_send[r], send_elems, r);
  }
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    CUDA_CHECK(cudaStreamSynchronize(micro_streams[r]));
  }

  // Print the effective configuration once so each result line can be read in context.
  const std::string device_list = format_devices(a.devices);
  printf("[CONFIG] send=%zu MB per rank ranks=%d devices=%s iters=%d warmup=%d\n"
         "         grid=%d block=%d pipe=%d smem=%zu KB slot=%zu B issue_warp=%d\n"
         "         simple_slice=%zu KB tma_slice=%zu KB tma_tile=%zu KB\n",
         a.send_mb, nranks, device_list.c_str(), a.iters, a.warmup,
         a.grid, a.block, a.pipe, a.smem_kb, tma_slot_bytes, a.tma_issue_warp,
         a.simple_slice_kb, a.tma_slice_kb, a.tma_tile_kb);

  // Each path is run independently with recv buffers cleared beforehand so that:
  //   - verification always checks fresh output
  //   - stale data from a previous variant cannot hide bugs
  if (a.run_micro_simple) {
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaMemsetAsync(d_recv[r], 0, recv_bytes, micro_streams[r]));
    }
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaStreamSynchronize(micro_streams[r]));
    }
    BenchResult res = run_micro_simple(a, d_send, d_recv, micro_streams,
                                       send_bytes, send_elems, out_bytes_per_rank,
                                       chunk_bytes, simple_slice_bytes);
    printf("[MICRO_SIMPLE] avg=%.3f us  BW=%.2f GB/s  wrong=%zu\n",
           res.avg_ms * 1000.0f, res.gbps, res.wrong);
  }

  // MICRO_TMA uses the same logical all-gather layout, only the remote transfer
  // mechanism is different.
  if (a.run_micro_tma) {
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaMemsetAsync(d_recv[r], 0, recv_bytes, micro_streams[r]));
    }
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaStreamSynchronize(micro_streams[r]));
    }
    BenchResult res = run_micro_tma(a, d_send, d_recv, micro_streams,
                                    send_bytes, send_elems, out_bytes_per_rank,
                                    chunk_bytes, tma_slice_bytes, tma_tile_bytes,
                                    tma_slot_bytes, smem_bytes);
    printf("[MICRO_TMA]    avg=%.3f us  BW=%.2f GB/s  wrong=%zu\n",
           res.avg_ms * 1000.0f, res.gbps, res.wrong);
  }

  // NCCL_SIMPLE is the "real library" comparison point for the Simple protocol.
  if (a.run_nccl_simple) {
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaMemset(d_recv[r], 0, recv_bytes));
    }
    BenchResult res = run_nccl_allgather(a, d_send, d_recv, send_elems, out_bytes_per_rank, "SIMPLE");
    printf("[NCCL_SIMPLE]  avg=%.3f us  BW=%.2f GB/s  wrong=%zu\n",
           res.avg_ms * 1000.0f, res.gbps, res.wrong);
  }

  // NCCL_TMA is the "real library" comparison point for the TMA protocol.
  if (a.run_nccl_tma) {
    for (int r = 0; r < nranks; ++r) {
      CUDA_CHECK(cudaSetDevice(a.devices[r]));
      CUDA_CHECK(cudaMemset(d_recv[r], 0, recv_bytes));
    }
    BenchResult res = run_nccl_allgather(a, d_send, d_recv, send_elems, out_bytes_per_rank, "TMA");
    printf("[NCCL_TMA]     avg=%.3f us  BW=%.2f GB/s  wrong=%zu\n",
           res.avg_ms * 1000.0f, res.gbps, res.wrong);
  }

  // Cleanup is explicit per device because send/recv buffers and streams belong
  // to the CUDA context of each selected GPU.
  for (int r = 0; r < nranks; ++r) {
    CUDA_CHECK(cudaSetDevice(a.devices[r]));
    if (d_send[r]) CUDA_CHECK(cudaFree(d_send[r]));
    if (d_recv[r]) CUDA_CHECK(cudaFree(d_recv[r]));
    CUDA_CHECK(cudaStreamDestroy(micro_streams[r]));
  }

  return 0;
}
