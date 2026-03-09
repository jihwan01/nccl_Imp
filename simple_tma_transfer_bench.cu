#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda/ptx>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using bf16 = __nv_bfloat16;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

#define CUDA_CHECK(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

static constexpr int kMaxPipeDepth = 8;

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
  int tma_issue_warp = 1;

  // explicit slice/tile controls requested for comparison
  size_t simple_slice_kb = 1024; // 1MB default
  size_t tma_slice_kb = 1024;    // 1MB default
  size_t tma_tile_kb = 32;       // 32KB default

  int detail = 1;
  int verify = 1;
};

struct TmaKernelStats {
  unsigned long long wait_cycles = 0ULL;
  unsigned long long issue_cycles = 0ULL;
  unsigned long long store_cycles = 0ULL;
  unsigned long long wait_samples = 0ULL;
  unsigned long long issue_samples = 0ULL;
  unsigned long long store_samples = 0ULL;
  unsigned long long blocks = 0ULL;
};

struct SimpleKernelStats {
  unsigned long long io_cycles = 0ULL;
  unsigned long long io_samples = 0ULL;
  unsigned long long blocks = 0ULL;
};

struct CommDetail {
  double wait_ns = 0.0;
  double issue_ns = 0.0;
  double store_ns = 0.0;
  double io_ns = 0.0;

  unsigned long long wait_samples = 0ULL;
  unsigned long long issue_samples = 0ULL;
  unsigned long long store_samples = 0ULL;
  unsigned long long io_samples = 0ULL;
  unsigned long long blocks = 0ULL;
};

struct BenchResult {
  float avg_ms = 0.0f;
  float gbps = 0.0f;
  CommDetail detail{};
};

__device__ __forceinline__ size_t div_up(size_t a, size_t b) {
  return (a + b - 1) / b;
}

__global__ void init_src_bf16(bf16* src, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) src[i] = __float2bfloat16(float(i % 4096));
}

__global__ void kernel_simple_transfer(
    bf16* __restrict__ dst_peer,
    const bf16* __restrict__ src,
    size_t chunk_bytes,
    size_t simple_slice_bytes,
    SimpleKernelStats* __restrict__ prof) {
  const size_t tid = threadIdx.x;
  const size_t cta_start = blockIdx.x * chunk_bytes;

  const uint4* src_v = reinterpret_cast<const uint4*>(src);
  uint4* dst_v = reinterpret_cast<uint4*>(dst_peer);

  unsigned long long io_cycles_local = 0ULL;
  unsigned long long io_samples_local = 0ULL;

  for (size_t slice_off = 0; slice_off < chunk_bytes; slice_off += simple_slice_bytes) {
    const size_t curr_slice_bytes = min(simple_slice_bytes, chunk_bytes - slice_off);
    const size_t curr_vecs = curr_slice_bytes / 16;
    const size_t vec_base = (cta_start + slice_off) / 16;

    if (prof != nullptr) __syncthreads();
    unsigned long long t0 = 0ULL;
    if (prof != nullptr && tid == 0) t0 = clock64();

    const size_t stride = size_t(blockDim.x);
    size_t v = tid;
    for (; v + 7 * stride < curr_vecs; v += 8 * stride) {
      uint4 x0 = src_v[vec_base + v + 0 * stride];
      uint4 x1 = src_v[vec_base + v + 1 * stride];
      uint4 x2 = src_v[vec_base + v + 2 * stride];
      uint4 x3 = src_v[vec_base + v + 3 * stride];
      uint4 x4 = src_v[vec_base + v + 4 * stride];
      uint4 x5 = src_v[vec_base + v + 5 * stride];
      uint4 x6 = src_v[vec_base + v + 6 * stride];
      uint4 x7 = src_v[vec_base + v + 7 * stride];
      dst_v[vec_base + v + 0 * stride] = x0;
      dst_v[vec_base + v + 1 * stride] = x1;
      dst_v[vec_base + v + 2 * stride] = x2;
      dst_v[vec_base + v + 3 * stride] = x3;
      dst_v[vec_base + v + 4 * stride] = x4;
      dst_v[vec_base + v + 5 * stride] = x5;
      dst_v[vec_base + v + 6 * stride] = x6;
      dst_v[vec_base + v + 7 * stride] = x7;
    }
    for (; v < curr_vecs; v += stride) {
      uint4 x = src_v[vec_base + v];
      dst_v[vec_base + v] = x;
    }

    __syncthreads();
    if (prof != nullptr && tid == 0) {
      io_cycles_local += (clock64() - t0);
      io_samples_local += 1ULL;
    }
  }

  if (prof != nullptr && tid == 0) {
    atomicAdd(&prof->io_cycles, io_cycles_local);
    atomicAdd(&prof->io_samples, io_samples_local);
    atomicAdd(&prof->blocks, 1ULL);
  }
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
 *   - If `enable_issue_warp=1` and enough threads exist, last warp becomes issuer.
 *   - Prologue preloads up to `pipe_depth` tiles.
 *   - Main loop consumes one tile and issues one future tile into the freed slot.
 *
 * Timing (optional, prof != nullptr):
 *   - issue: enqueue memcpy_async into a slot
 *   - wait : barrier wait for tile readiness
 *   - store: shared -> peer global vector stores
 */
__global__ void kernel_tma_transfer(
    bf16* __restrict__ dst_peer,
    const bf16* __restrict__ src,
    size_t chunk_bytes,
    size_t tma_slice_bytes,
    size_t tma_tile_bytes,
    size_t tma_slot_bytes,
    int pipe_depth,
    int enable_issue_warp,
    TmaKernelStats* __restrict__ prof) {
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

  // NCCL-like warp specialization:
  // reserve one warp for async issue when block has at least 2 warps.
  constexpr int kWarpSize = 32;
  const bool allow_issue_warp = (enable_issue_warp != 0) && (pipe_depth >= 2);
  const int tma_compute_workers =
      (allow_issue_warp && nworkers >= 2 * kWarpSize) ? (nworkers - kWarpSize) : nworkers;
  const bool tma_issue_warp_enabled = (tma_compute_workers < nworkers);
  const int tma_issue_tid = tma_compute_workers;
  const bool is_tma_compute_thread = int(tid) < tma_compute_workers;
  const bool is_tma_issuer_thread = tma_issue_warp_enabled ? (int(tid) == tma_issue_tid) : (tid == 0);

  uint4* dst_v = reinterpret_cast<uint4*>(dst_peer);

  if (threadIdx.x == 0) {
    for (int i = 0; i < pipe_depth; ++i) init(&bars[i], blockDim.x);
    ptx::fence_proxy_async(ptx::space_shared);
  }
  __syncthreads();

  unsigned long long wait_cycles_local = 0ULL;
  unsigned long long issue_cycles_local = 0ULL;
  unsigned long long store_cycles_local = 0ULL;
  unsigned long long wait_samples_local = 0ULL;
  unsigned long long issue_samples_local = 0ULL;
  unsigned long long store_samples_local = 0ULL;

  // Process the CTA chunk slice by slice.
  for (size_t slice_off = 0; slice_off < chunk_bytes; slice_off += tma_slice_bytes) {
    const size_t curr_slice_bytes = min(tma_slice_bytes, chunk_bytes - slice_off);
    const size_t tile_count = div_up(curr_slice_bytes, tma_tile_bytes);

    // 1) Prologue: preload up to pipe_depth tiles.
    int in_flight = 0;
    for (; in_flight < pipe_depth && in_flight < (int)tile_count; ++in_flight) {
      const size_t tile_idx = size_t(in_flight);
      const size_t tile_off = tile_idx * tma_tile_bytes;
      const size_t copy_bytes = min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const int slot = in_flight % pipe_depth;

      if (is_tma_issuer_thread) {
        unsigned long long t0 = 0ULL;
        if (prof != nullptr) t0 = clock64();
        cuda::memcpy_async(
            smem_slot[slot],
            reinterpret_cast<const unsigned char*>(src) + cta_start + slice_off + tile_off,
            cuda::aligned_size_t<16>(copy_bytes),
            bars[slot]);
        if (prof != nullptr) {
          issue_cycles_local += (clock64() - t0);
          issue_samples_local += 1ULL;
        }
      }
      tokens[slot] = bars[slot].arrive();
    }

    // 2) Steady state: wait/consume current tile, then issue next tile.
    for (size_t t = 0; t < tile_count; ++t) {
      const int slot = int(t % size_t(pipe_depth));
      const size_t tile_off = t * tma_tile_bytes;
      const size_t copy_bytes = min(tma_tile_bytes, curr_slice_bytes - tile_off);
      const size_t vec_count = copy_bytes / 16;

      if (prof != nullptr && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        bars[slot].wait(std::move(tokens[slot]));
        wait_cycles_local += (clock64() - t0);
        wait_samples_local += 1ULL;
      } else {
        bars[slot].wait(std::move(tokens[slot]));
      }

      if (prof != nullptr) __syncthreads();
      unsigned long long t0_store = 0ULL;
      if (prof != nullptr && threadIdx.x == 0) t0_store = clock64();

      const uint4* smem_v = reinterpret_cast<const uint4*>(smem_slot[slot]);
      const size_t dst_vec_base = (cta_start + slice_off + tile_off) / 16;

      if (is_tma_compute_thread) {
        const size_t stride = size_t(tma_compute_workers);
        size_t v = tid;
        for (; v + 7 * stride < vec_count; v += 8 * stride) {
          uint4 x0 = smem_v[v + 0 * stride];
          uint4 x1 = smem_v[v + 1 * stride];
          uint4 x2 = smem_v[v + 2 * stride];
          uint4 x3 = smem_v[v + 3 * stride];
          uint4 x4 = smem_v[v + 4 * stride];
          uint4 x5 = smem_v[v + 5 * stride];
          uint4 x6 = smem_v[v + 6 * stride];
          uint4 x7 = smem_v[v + 7 * stride];
          dst_v[dst_vec_base + v + 0 * stride] = x0;
          dst_v[dst_vec_base + v + 1 * stride] = x1;
          dst_v[dst_vec_base + v + 2 * stride] = x2;
          dst_v[dst_vec_base + v + 3 * stride] = x3;
          dst_v[dst_vec_base + v + 4 * stride] = x4;
          dst_v[dst_vec_base + v + 5 * stride] = x5;
          dst_v[dst_vec_base + v + 6 * stride] = x6;
          dst_v[dst_vec_base + v + 7 * stride] = x7;
        }
        for (; v < vec_count; v += stride) {
          uint4 x = smem_v[v];
          dst_v[dst_vec_base + v] = x;
        }
      }

      __syncthreads();
      if (prof != nullptr && threadIdx.x == 0) {
        store_cycles_local += (clock64() - t0_store);
        store_samples_local += 1ULL;
      }

      // 3) Refill:
      // Always keep ring-buffer semantics correct: refill slot (t % pipe_depth)
      // with tile (t + pipe_depth) after consuming tile t.
      const size_t issue_idx = t + size_t(pipe_depth);
      if (issue_idx < tile_count) {
        const size_t issue_off = issue_idx * tma_tile_bytes;
        const size_t issue_copy_bytes = min(tma_tile_bytes, curr_slice_bytes - issue_off);

        if (is_tma_issuer_thread) {
          unsigned long long t0 = 0ULL;
          if (prof != nullptr) t0 = clock64();
          cuda::memcpy_async(
              smem_slot[slot],
              reinterpret_cast<const unsigned char*>(src) + cta_start + slice_off + issue_off,
              cuda::aligned_size_t<16>(issue_copy_bytes),
              bars[slot]);
          if (prof != nullptr) {
            issue_cycles_local += (clock64() - t0);
            issue_samples_local += 1ULL;
          }
        }
        tokens[slot] = bars[slot].arrive();
      }
    }
  }

  if (prof != nullptr && threadIdx.x == 0) {
    atomicAdd(&prof->wait_cycles, wait_cycles_local);
    atomicAdd(&prof->issue_cycles, issue_cycles_local);
    atomicAdd(&prof->store_cycles, store_cycles_local);
    atomicAdd(&prof->wait_samples, wait_samples_local);
    atomicAdd(&prof->issue_samples, issue_samples_local);
    atomicAdd(&prof->store_samples, store_samples_local);
    atomicAdd(&prof->blocks, 1ULL);
  }
}

static inline int device_clock_khz(int dev) {
  int clock_khz = 0;
  cudaError_t st = cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
  if (st != cudaSuccess) return 0;
  return clock_khz;
}

static inline double cycles_to_ns(unsigned long long cycles, int clock_khz) {
  if (clock_khz <= 0) return 0.0;
  return double(cycles) * (1.0e6 / double(clock_khz));
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
             "  --pipe N              TMA pipeline depth (default: 2)\\n"
             "  --tma_issue_warp N    0=disable, 1=enable issue-warp specialization (default: 1)\\n"
             "  --simple_slice_kb N   Simple slice size in KB (default: 1024)\\n"
             "  --tma_slice_kb N      TMA slice size in KB (default: 1024)\\n"
             "  --tma_tile_kb N       TMA tile size in KB (default: 32)\\n"
             "                        NOTE: TMA enforces slot_bytes == tile_bytes, so smem_kb must satisfy\\n"
             "                              smem_kb*1024 == tma_tile_kb*1024*pipe\\n"
             "  --detail N            0=off, 1=on (default: 1)\\n"
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
    else if (k == "--tma_issue_warp") { need(1); a.tma_issue_warp = std::stoi(argv[++i]); }
    else if (k == "--smem_kb") { need(1); a.smem_kb = std::stoull(argv[++i]); }
    else if (k == "--simple_slice_kb") { need(1); a.simple_slice_kb = std::stoull(argv[++i]); }
    else if (k == "--tma_slice_kb") { need(1); a.tma_slice_kb = std::stoull(argv[++i]); }
    else if (k == "--tma_tile_kb") { need(1); a.tma_tile_kb = std::stoull(argv[++i]); }
    else if (k == "--detail") { need(1); a.detail = std::stoi(argv[++i]); }
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
  if (a.pipe < 1 || a.pipe > kMaxPipeDepth) {
    fprintf(stderr, "--pipe must be in [1, %d]\\n", kMaxPipeDepth);
    std::exit(2);
  }
  if (!(a.proto == 0 || a.proto == 1 || a.proto == 2)) {
    fprintf(stderr, "--proto must be 0(both), 1(tma), or 2(simple)\\n");
    std::exit(2);
  }
  if (!(a.tma_issue_warp == 0 || a.tma_issue_warp == 1)) {
    fprintf(stderr, "--tma_issue_warp must be 0 or 1\\n");
    std::exit(2);
  }
  if (a.src_dev == a.dst_dev) {
    fprintf(stderr, "--src and --dst must be different\\n");
    std::exit(2);
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

static BenchResult run_simple(
    const Args& a,
    bf16* d_dst,
    const bf16* d_src,
    size_t total_bytes,
    size_t chunk_bytes,
    size_t simple_slice_bytes) {
  BenchResult out;

  SimpleKernelStats* d_prof = nullptr;
  SimpleKernelStats h_prof{};
  if (a.detail) CUDA_CHECK(cudaMalloc(&d_prof, sizeof(SimpleKernelStats)));

  for (int i = 0; i < a.warmup; ++i) {
    kernel_simple_transfer<<<a.grid, a.block, 0>>>(d_dst, d_src, chunk_bytes, simple_slice_bytes, nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  if (a.detail) CUDA_CHECK(cudaMemset(d_prof, 0, sizeof(SimpleKernelStats)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < a.iters; ++it) {
    kernel_simple_transfer<<<a.grid, a.block, 0>>>(d_dst, d_src, chunk_bytes, simple_slice_bytes, d_prof);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaEventElapsedTime(&out.avg_ms, start, stop));
  out.avg_ms /= float(a.iters);
  out.gbps = bytes_to_gbps(total_bytes, out.avg_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (a.detail) {
    CUDA_CHECK(cudaMemcpy(&h_prof, d_prof, sizeof(SimpleKernelStats), cudaMemcpyDeviceToHost));
    int clk = device_clock_khz(a.src_dev);
    out.detail.blocks = h_prof.blocks;
    out.detail.io_samples = h_prof.io_samples;
    if (h_prof.io_samples > 0ULL) {
      out.detail.io_ns = cycles_to_ns(h_prof.io_cycles, clk) / double(h_prof.io_samples);
    }
    CUDA_CHECK(cudaFree(d_prof));
  }

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
    size_t tma_slot_bytes,
    int tma_issue_warp) {
  BenchResult out;

  CUDA_CHECK(cudaFuncSetAttribute(
      kernel_tma_transfer,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      int(smem_bytes)));

  TmaKernelStats* d_prof = nullptr;
  TmaKernelStats h_prof{};
  if (a.detail) CUDA_CHECK(cudaMalloc(&d_prof, sizeof(TmaKernelStats)));

  for (int i = 0; i < a.warmup; ++i) {
    kernel_tma_transfer<<<a.grid, a.block, smem_bytes>>>(
        d_dst, d_src, chunk_bytes, tma_slice_bytes, tma_tile_bytes, tma_slot_bytes, a.pipe, tma_issue_warp, nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  if (a.detail) CUDA_CHECK(cudaMemset(d_prof, 0, sizeof(TmaKernelStats)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int it = 0; it < a.iters; ++it) {
    kernel_tma_transfer<<<a.grid, a.block, smem_bytes>>>(
        d_dst, d_src, chunk_bytes, tma_slice_bytes, tma_tile_bytes, tma_slot_bytes, a.pipe, tma_issue_warp, d_prof);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  CUDA_CHECK(cudaEventElapsedTime(&out.avg_ms, start, stop));
  out.avg_ms /= float(a.iters);
  out.gbps = bytes_to_gbps(total_bytes, out.avg_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (a.detail) {
    CUDA_CHECK(cudaMemcpy(&h_prof, d_prof, sizeof(TmaKernelStats), cudaMemcpyDeviceToHost));
    int clk = device_clock_khz(a.src_dev);
    out.detail.blocks = h_prof.blocks;
    out.detail.wait_samples = h_prof.wait_samples;
    out.detail.issue_samples = h_prof.issue_samples;
    out.detail.store_samples = h_prof.store_samples;

    if (h_prof.wait_samples > 0ULL) {
      out.detail.wait_ns = cycles_to_ns(h_prof.wait_cycles, clk) / double(h_prof.wait_samples);
    }
    if (h_prof.issue_samples > 0ULL) {
      out.detail.issue_ns = cycles_to_ns(h_prof.issue_cycles, clk) / double(h_prof.issue_samples);
    }
    if (h_prof.store_samples > 0ULL) {
      out.detail.store_ns = cycles_to_ns(h_prof.store_cycles, clk) / double(h_prof.store_samples);
    }
    CUDA_CHECK(cudaFree(d_prof));
  }

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
         "         proto=%d pipe=%d issue_warp=%d smem=%zu KB (slot=%zu B)\\n"
         "         simple_slice=%zu KB | tma_slice=%zu KB tma_tile=%zu KB\\n",
         a.total_mb, a.grid, a.block, a.iters, a.warmup, a.src_dev, a.dst_dev,
         a.proto, a.pipe, a.tma_issue_warp, a.smem_kb, tma_slot_bytes,
         a.simple_slice_kb, a.tma_slice_kb, a.tma_tile_kb);

  size_t wrong_tma = 0;
  size_t wrong_simple = 0;

  if (a.proto == 0 || a.proto == 1) {
    CUDA_CHECK(cudaSetDevice(a.dst_dev));
    CUDA_CHECK(cudaMemset(d_dst, 0, total_bytes));

    CUDA_CHECK(cudaSetDevice(a.src_dev));
    BenchResult r = run_tma(a, d_dst, d_src, total_bytes, chunk_bytes,
                            tma_slice_bytes, tma_tile_bytes, smem_bytes, tma_slot_bytes, a.tma_issue_warp);

    if (a.verify) wrong_tma = verify_copy(a.src_dev, a.dst_dev, d_src, d_dst, n_elem);

    const float avg_us = r.avg_ms * 1000.0f;
    printf("[TMA]    avg=%.3f us  BW=%.2f GB/s  wrong=%zu\\n", avg_us, r.gbps, wrong_tma);
    if (a.detail) {
      printf("         detail(wait/issue/store) ns = %.3f / %.3f / %.3f  samples = %llu / %llu / %llu  blocks=%llu\\n",
             r.detail.wait_ns, r.detail.issue_ns, r.detail.store_ns,
             r.detail.wait_samples, r.detail.issue_samples, r.detail.store_samples, r.detail.blocks);
    }
  }

  if (a.proto == 0 || a.proto == 2) {
    CUDA_CHECK(cudaSetDevice(a.dst_dev));
    CUDA_CHECK(cudaMemset(d_dst, 0, total_bytes));

    CUDA_CHECK(cudaSetDevice(a.src_dev));
    BenchResult r = run_simple(a, d_dst, d_src, total_bytes, chunk_bytes, simple_slice_bytes);

    if (a.verify) wrong_simple = verify_copy(a.src_dev, a.dst_dev, d_src, d_dst, n_elem);

    const float avg_us = r.avg_ms * 1000.0f;
    printf("[SIMPLE] avg=%.3f us  BW=%.2f GB/s  wrong=%zu\\n", avg_us, r.gbps, wrong_simple);
    if (a.detail) {
      printf("         detail(io) ns = %.3f  samples = %llu  blocks=%llu\\n",
             r.detail.io_ns, r.detail.io_samples, r.detail.blocks);
    }
  }

  CUDA_CHECK(cudaSetDevice(a.src_dev));
  if (d_src) CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaSetDevice(a.dst_dev));
  if (d_dst) CUDA_CHECK(cudaFree(d_dst));

  const bool ok = (!a.verify) || (wrong_tma == 0 && wrong_simple == 0);
  return ok ? 0 : 2;
}
