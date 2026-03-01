#include "network/unpack/unpack.h"
#include <cassert>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <new>

#define NCCL_TMA_PIPE_DEPTH 2
#define NCCL_TMA_SLOT_SIZE (32 * 1024)

#ifndef ENABLE_PROFILING
#define ENABLE_PROFILING 1
#endif
#ifndef ENABLE_TILE_PROFILING
#define ENABLE_TILE_PROFILING ENABLE_PROFILING
#endif
#ifndef ENABLE_SLICE_PROFILING
#define ENABLE_SLICE_PROFILING ENABLE_PROFILING
#endif

// enum primsMode {
//   primsModeDefault = 0,
//   primsModePatRs = 1,
//   primsModePatAg = 2
// };

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p, int MultimemSrcs, int MultimemDsts, bool isNetOffload>
class Primitives<
    T, RedOp, Fan, Direct, ProtoTMA<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, isNetOffload
  > {
public:
  // Profiling context
  int profileChunkId = -1;

private:
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       NetRegMode = 0x80,
                       ConnFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       PatMode = 0x800,
                       NvlsMinPolling = 0x1000,
                       NetDeviceUnpack = 0x2000,
                       AnyNetDeviceUnpack = 0x4000;
  const int tid, tidInBlock;
  const int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  struct ncclConnInfo* conn = NULL;
  struct ncclConnFifo* connFifo = NULL;
  T* connEltsFifo;
  T* directBuff = NULL;
  uint64_t *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)
  int      connStepSize; // Connection step size
  void*    netDeviceHandle;
  uint64_t accSize;

  // Don't use barrier 0 as it's used by the final sync
  __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group;
      barrier_sync(bar, nthreads);
    }
  }
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group - (nworkers!=nthreads ? 1 : 0);
      barrier_sync(bar, nworkers);
    }
  }

  // PAT uses a single barrier across all groups
  __device__ void patBarrier() {
    barrier_sync(15, NCCL_PAT_NWORKERS);
  }

  __device__ bool barrierAny(int vote) {
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group;
      return barrier_red_or(vote, name, nthreads);
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group - (nworkers!=nthreads ? 1 : 0);
      return barrier_red_or(vote, name, nworkers);
    }
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
    #if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    if (flags & NvlsMinPolling) {
      uint64_t ans;
      asm volatile("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
      return ans;
    }
    #endif
    // volatile is faster than acquire but not as correct. Make sure reduceCopy
    // loads data using volatile so it doesn't see stale data in L1.
    return ld_volatile_global(ptr);
  }

  // TMA Support: Wait for peer data availability for a specific slice offset without advancing the global step
  // Renamed from waitPeerForTmaLoad to waitPeerForPreload per user request
  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeerForPreload(intptr_t srcIx, intptr_t dstIx, int offset, int nelts, int sliceOffset) {
     
    if (!(flags & (Recv * RoleWaitRecv))) return;

    // [Optimization] stepIncrement is always StepPerSlice due to the check above
    uint64_t absStep = step + sliceOffset * StepPerSlice;
    int buffSlot = absStep % NCCL_STEPS;
     
    // For Recv: Wait until peer has produced data for the specified slice step
    int spins = 0;
    while (connStepCache < absStep + StepPerSlice) { 
      connStepCache = loadStepValue(connStepPtr); // head
      if (checkAbort(flags, Aborted, spins)) break;
    }

    // Set up source pointers to peer's FIFO buffer at the specified step
    void **ptrs = ncclShmem.groups[group].srcs + Src;
    const int index = 0;
    
    if ((flags & ConnFifoEnabled) && connFifo[buffSlot].mode == NCCL_MODE_OFFSET) {
      ptrs[index] = connEltsFifo + loadInt(&connFifo[buffSlot].offset)/sizeof(T);
    } else if (DirectRecv) {
      if (flags & DirectRead) {
        ptrs[index] = directBuff + srcIx + offset;
      } else {
        ptrs[index] = connEltsFifo + buffSlot*connStepSize;
      }
    } else {
      ptrs[index] = connEltsFifo + buffSlot*connStepSize;
    }
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    if ((flags & (Recv * RoleWaitRecv)) || (flags & (Send * RoleWaitSend))) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
        connFifo[step%NCCL_STEPS].size = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if ((flags & NetRegMode) && ((!isSendNotRecv && DirectRecv) || (isSendNotRecv && DirectSend))) {
        if (P2p) {
          ptrs[index] = NULL;
        } else {
          if (isSendNotRecv) {
            if (!Recv)
              ptrs[index] = NULL;
            else
              ptrs[index] = (T*)ncclShmem.groups[group].userOutput + dstIx + offset;
          } else {
            ptrs[index] = (T*)ncclShmem.groups[group].userOutput + srcIx + offset;
          }
        }
      } else if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
        ptrs[index] = connEltsFifo + loadInt(&connFifo[step%NCCL_STEPS].offset)/sizeof(T);
      } else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;
        } else if (flags & DirectRead) {  // empty send
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
        }
      } else if (!isSendNotRecv && DirectRecv) {
        if (flags & DirectRead) {
          ptrs[index] = directBuff + srcIx + offset;
        } else if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
        }
      }
      else {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
      }
      if (flags & NetDeviceUnpack) {
        ncclNetDeviceIncrementHead(group, index);
      }
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored||(flags&ConnFifoEnabled))) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;
    int tmaTileSize = sliceSize;
    int tmaTileSizeBytes = tmaTileSize * (int)sizeof(T);
    if (tmaTileSizeBytes > NCCL_TMA_SLOT_SIZE) {
      tmaTileSize = NCCL_TMA_SLOT_SIZE / (int)sizeof(T);
      tmaTileSizeBytes = tmaTileSize * (int)sizeof(T);
    }

    // [TMA] Determine if TMA should be used based on test mode
    // 1: Send Only, 2: RecvSend Only, 3: Recv Only, 4: All, 5: Send,Recv, 6: Send,RecvSend
    const int tmaMode = 4; 
    bool useTma = false;

    // Check availability (Src present for Send, or Recv role)
    // Note: Recv role implies pulling from Peer (Network) to SMEM
    bool isTmaEligible = (Send && !Recv && Src) || (Recv);

    // [TMA Safety] Check alignments. If using TMA, we must ensure 16-byte alignment
    // for both Address and Size to avoid single-thread software copy fallback which is slow.
    if (isTmaEligible) {
        if (Src) {
            T* ptr = (SrcBuf == Input ? (T*)ncclShmem.groups[group].userInput : (T*)ncclShmem.groups[group].userOutput) + srcIx;
             // Check Base Alignment
            if ((uintptr_t)ptr % 16 != 0) {
                // if (tid == 0) printf("[TMA Fallback] Source Address Misaligned: %p\n", ptr);
                isTmaEligible = false;
            }
        }
        // Check Size Alignment (tmaTileSize is in elements, so * sizeof(T))
        if ((tmaTileSize * sizeof(T)) % 16 != 0) {
          // if (tid == 0) printf("[TMA Fallback] Copy Size Misaligned: %d bytes\n", tmaTileSize * (int)sizeof(T));
            isTmaEligible = false;
        }
    }

    if (isTmaEligible) {
        switch(tmaMode) {
            case 1: useTma = (Send && !Recv); break;
            case 2: useTma = (Send && Recv); break;
            case 3: useTma = (!Send && Recv); break;
            case 4: useTma = true; break;
            case 5: useTma = (Send && !Recv) || (!Send && Recv); break;
            case 6: useTma = (Send); break;
            case 7: useTma = false; break;
        }
    }

    using tma_barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(alignof(tma_barrier_t)) char barrierMem[sizeof(tma_barrier_t) * NCCL_TMA_PIPE_DEPTH];
    tma_barrier_t* barriers = reinterpret_cast<tma_barrier_t*>(barrierMem);
    typename tma_barrier_t::arrival_token tmaTokens[NCCL_TMA_PIPE_DEPTH];
    // Warp specialization for TMA issue:
    // Only enable when fan-in/out is 1 (ring-like path), then reserve the last
    // worker warp for issuing TMA copies while earlier worker warps keep computing.
    constexpr bool TmaIssueWarpEligible = (MaxRecv <= 1 && MaxSend <= 1);
    const int tmaComputeWorkers = (TmaIssueWarpEligible && nworkers >= 2*WARP_SIZE) ? (nworkers - WARP_SIZE) : nworkers;
    const bool tmaIssueWarpEnabled = (tmaComputeWorkers < nworkers);
    const int tmaIssueTid = tmaComputeWorkers;
    const bool isTmaComputeThread = tid < tmaComputeWorkers;
    const bool isTmaIssuerThread = tmaIssueWarpEnabled ? (tid == tmaIssueTid) : (tid == 0);

    if (tid < nworkers && offset < nelem && !isNetOffload) {
      if (useTma) {
        if (tid == 0) {
          #pragma unroll
          for (int i=0; i<NCCL_TMA_PIPE_DEPTH; ++i) new (&barriers[i]) tma_barrier_t(nworkers);
        }
        subBarrier();

        // Slice-level waitPeer, tile-level TMA pipeline
        T* tmaSliceBaseSrc = nullptr;
        T* tmaSliceBaseDst = nullptr;
        T* peerFifoBase = nullptr;  // Base pointer for dsts[1] (peer FIFO in Send)
        T* peerSrcBase = nullptr;   // Base pointer for srcs[1] (peer FIFO in Recv)

        while (slice < SlicePerChunk && offset < nelem) {
          int currSliceSize = (sliceSize < nelem-offset) ? sliceSize : nelem-offset;
          
          #if ENABLE_TILE_PROFILING || ENABLE_SLICE_PROFILING
          long long t0 = 0;
          #endif
          #if ENABLE_SLICE_PROFILING
          long long t0_slice_copy = 0;
          t0 = clock64();
          #endif

          if (tid == 0) {
            T* userInput = (T*)ncclShmem.groups[group].userInput;
            T* userOutput = (T*)ncclShmem.groups[group].userOutput;
            if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
            if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
          }
          waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, currSliceSize);
          subBarrier();

          #if ENABLE_SLICE_PROFILING
          long long t1_wait = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
              long long dt = t1_wait - t0;
              const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
              printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_WAIT,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          // Reset t0_slice_copy for actual copy measurement (includes preload phase)
          t0_slice_copy = clock64();
          #endif

          // ================================================================
          // =================== ReduceCopy or Copy Phase ===================
          // ================================================================

          // Store slice base pointers once waitPeer/setup is complete.
          tmaSliceBaseSrc = (T*)ncclShmem.groups[group].srcs[0];
          tmaSliceBaseDst = (T*)ncclShmem.groups[group].dsts[0];
          // Save base pointer for dsts[1] (peer FIFO) if Send path
          if (Send && Dst && ncclShmem.groups[group].dsts[1] != nullptr) {
            peerFifoBase = (T*)ncclShmem.groups[group].dsts[1];
          }
          // Save base pointer for srcs[1] (peer FIFO) if Recv path
          if (Recv && Src && ncclShmem.groups[group].srcs[1] != nullptr) { // For operation involving reduce operation
            peerSrcBase = (T*)ncclShmem.groups[group].srcs[1];
          }

          int sliceTiles = (currSliceSize + tmaTileSize - 1) / tmaTileSize;
          int preloadCount = min(NCCL_TMA_PIPE_DEPTH, sliceTiles);
          int tileOffset = 0;

          // ================================================================
          // =================== Preload Phase ===================
          // ================================================================

          if (isTmaIssuerThread) {
            for (int i=0; i<preloadCount; ++i) {
              int currTileSize = (tmaTileSize < currSliceSize - tileOffset) ? tmaTileSize : currSliceSize - tileOffset;
              int tmaSlot = i % NCCL_TMA_PIPE_DEPTH;
              void* globalSrc = (void*)(tmaSliceBaseSrc + tileOffset);
              void* shmemDst = (char*)ncclTmaShmemPtr() + tmaSlot * NCCL_TMA_SLOT_SIZE;
              size_t copySize = currTileSize * sizeof(T);

              #if ENABLE_TILE_PROFILING
              t0 = clock64();
              #endif

              #if __CUDA_ARCH__ >= 900
                if (copySize % 16 == 0 && (uintptr_t)globalSrc % 16 == 0) {
                  cuda::device::memcpy_async_tx(
                      (char*)shmemDst,
                      (char*)globalSrc,
                      cuda::aligned_size_t<16>(copySize),
                      barriers[tmaSlot]
                  );
                  tmaTokens[tmaSlot] = cuda::device::barrier_arrive_tx(barriers[tmaSlot], 1, copySize);
                } else {
                  cuda::memcpy_async(
                      shmemDst,
                      globalSrc,
                      copySize,
                      barriers[tmaSlot]
                  );
                  tmaTokens[tmaSlot] = barriers[tmaSlot].arrive();
                }
              #else
              cuda::memcpy_async(
                  shmemDst,
                  globalSrc,
                  copySize,
                  barriers[tmaSlot]
              );
              tmaTokens[tmaSlot] = barriers[tmaSlot].arrive();
              #endif
              #if ENABLE_TILE_PROFILING
              long long t1_issue = clock64();
              if (ncclShmem.channelId == 0 && profileChunkId == 0) {
                long long dt = t1_issue - t0;
                const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
                printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,%s_ISSUE,%lld\n", ncclShmem.channelId, profileChunkId, slice, i, opName, dt);
              }
              #endif
              tileOffset += currTileSize;
            }
          } else {
            // Non-issuer threads still contribute to slot handoff counts.
            for (int i=0; i<preloadCount; ++i) {
              int tmaSlot = i % NCCL_TMA_PIPE_DEPTH;
              tmaTokens[tmaSlot] = barriers[tmaSlot].arrive();
            }
          }

          // [TMA] Preload sync is covered by per-slot barrier wait in the tile loop.

          #if ENABLE_SLICE_PROFILING
          long long t1_copy_pre = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
              long long dt = t1_copy_pre - t0_slice_copy;
              const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
              printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_COPY_PRE,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          // Start timer for main loop copy.
          t0_slice_copy = clock64();
          #endif

          // ================================================================
          // =================== Main Loop Phase ===================
          // ================================================================


          tileOffset = 0;
          for (int t=0; t<sliceTiles; ++t) {
            int currTileSize = (tmaTileSize < currSliceSize - tileOffset) ? tmaTileSize : currSliceSize - tileOffset;
            int tmaSlot = t % NCCL_TMA_PIPE_DEPTH;

            // Wait TMA
            #if ENABLE_TILE_PROFILING
            t0 = clock64();
            #endif
            barriers[tmaSlot].wait(std::move(tmaTokens[tmaSlot]));
            #if ENABLE_TILE_PROFILING
            long long t1 = clock64();
            if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
              long long dt = t1 - t0;
              const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
              printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,%s_WAIT,%lld\n", ncclShmem.channelId, profileChunkId, slice, t, opName, dt);
            }
            #endif

            // Pointer swap for consumer and tile dst adjustment
            if (tid == 0) {
              // Update srcs[0] to point to SMEM for this tile (Send and Recv paths)
              ncclShmem.groups[group].srcs[0] = (char*)ncclTmaShmemPtr() + tmaSlot * NCCL_TMA_SLOT_SIZE;
              
              // Update dsts[0] to output buffer for this tile
              if (Dst && tmaSliceBaseDst != nullptr) {
                ncclShmem.groups[group].dsts[0] = tmaSliceBaseDst + tileOffset;
              }
              
              // Update dsts[1] (peer FIFO) for Send path
              if (Send && peerFifoBase != nullptr) {
                ncclShmem.groups[group].dsts[1] = peerFifoBase + tileOffset;
              }
            }
            subBarrier();

            // In specialized mode, issue(t+depth-1) to overlap with current consume.
            // Without specialization, keep the original refill schedule issue(t+depth).
            int issueTileIdx = t + (tmaIssueWarpEnabled ? (NCCL_TMA_PIPE_DEPTH - 1) : NCCL_TMA_PIPE_DEPTH);
            bool overlapIssue = issueTileIdx < sliceTiles;
            if (tmaIssueWarpEnabled) overlapIssue &= issueTileIdx >= preloadCount;
            int issueOffset = 0;
            int issueTileSize = 0;
            int issueSlot = 0;
            if (overlapIssue) {
              issueOffset = issueTileIdx * tmaTileSize;
              issueTileSize = (tmaTileSize < currSliceSize - issueOffset) ? tmaTileSize : currSliceSize - issueOffset;
              issueSlot = issueTileIdx % NCCL_TMA_PIPE_DEPTH;
              if (!isTmaIssuerThread) {
                tmaTokens[issueSlot] = barriers[issueSlot].arrive(); //
              }
            }

            int workSize = ncclShmem.aborted ? 0 : currTileSize;
            
            // --- DUPLICATED REDUCE COPY LOGIC START ---
            #if ENABLE_TILE_PROFILING
            t0 = clock64();
            #endif

            if (flags & AnyNetDeviceUnpack) {
              if (isTmaComputeThread) {
                ncclNetDeviceUnpack<Recv>(tid, tidInBlock, tmaComputeWorkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
              }
              subBarrier();
            }

            if (isTmaComputeThread) {
              if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
                  && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
                if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
                  reduceCopyShared<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
                    (tid, tmaComputeWorkers, /*redArg*/0, /*postOp*/false,
                    1, ncclShmem.groups[group].srcs,
                    fan.nsend(), ncclShmem.groups[group].dsts+1,
                    workSize);
                }
              } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
                reduceCopyShared<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
                  (tid, tmaComputeWorkers, ncclShmem.groups[group].redOpArgs, postOp,
                  Recv, ncclShmem.groups[group].srcs,
                  Dst, ncclShmem.groups[group].dsts,
                  workSize);
              } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
                constexpr int PreOpSrcs = SrcBuf != Input ? 0 : 1;
                if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
                  reduceCopyShared<Unroll, RedOp, T,
                    0, Recv + Src, Recv * MaxRecv + Src,
                    0, 1, 1, PreOpSrcs>
                    (tid, tmaComputeWorkers, ncclShmem.groups[group].redOpArgs, postOp,
                      Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                      1, ncclShmem.groups[group].dsts,
                      workSize);
                } else {
                  reduceCopyShared<Unroll, RedOp, T,
                    MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
                    MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
                    (tid, tmaComputeWorkers, ncclShmem.groups[group].redOpArgs, postOp,
                      Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                      Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
                      workSize);
                }
              } else {
                workSize = 0;
              }
            }
            // --- DUPLICATED REDUCE COPY LOGIC END ---
            
            #if ENABLE_TILE_PROFILING
            long long t1_copy = clock64();
            if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
                long long dt = t1_copy - t0;
                const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
                printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,%s_COPY,%lld\n", ncclShmem.channelId, profileChunkId, slice, t, opName, dt);
            }
            #endif

            if (overlapIssue && isTmaIssuerThread) {
              void* globalSrc = (void*)(tmaSliceBaseSrc + issueOffset);
              void* shmemDst = (char*)ncclTmaShmemPtr() + issueSlot * NCCL_TMA_SLOT_SIZE;
              size_t copySize = issueTileSize * sizeof(T);
              #if ENABLE_TILE_PROFILING
              t0 = clock64();
              #endif
              #if __CUDA_ARCH__ >= 900
                if (copySize % 16 == 0 && (uintptr_t)globalSrc % 16 == 0) {
                  cuda::device::memcpy_async_tx(
                      (char*)shmemDst,
                      (char*)globalSrc,
                      cuda::aligned_size_t<16>(copySize),
                      barriers[issueSlot]
                  );
                  tmaTokens[issueSlot] = cuda::device::barrier_arrive_tx(barriers[issueSlot], 1, copySize);
                } else {
                  cuda::memcpy_async(
                      shmemDst,
                      globalSrc,
                      copySize,
                      barriers[issueSlot]
                  );
                  tmaTokens[issueSlot] = barriers[issueSlot].arrive();
                }
              #else
              cuda::memcpy_async(
                  shmemDst,
                  globalSrc,
                  copySize,
                  barriers[issueSlot]
              );
              tmaTokens[issueSlot] = barriers[issueSlot].arrive();
              #endif
              #if ENABLE_TILE_PROFILING
              long long t1_issue_overlap = clock64();
              if (ncclShmem.channelId == 0 && profileChunkId == 0) {
                long long dt = t1_issue_overlap - t0;
                const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
                printf("NCCL_PROFILE_TILE,%d,%d,%d,%d,%s_ISSUE,%lld\n", ncclShmem.channelId, profileChunkId, slice, issueTileIdx, opName, dt);
              }
              #endif
            }
            
            tileOffset += currTileSize;
            // Ensure all workers have finished consuming this tile before tid0
            // updates shared srcs/dsts pointers for the next tile.
            subBarrier();
          }

          barrier();
          int sliceWorkSize = ncclShmem.aborted ? 0 : currSliceSize;

          #if ENABLE_SLICE_PROFILING
          long long t1_copy_main = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
              long long dt = t1_copy_main - t0_slice_copy;
              const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
              printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_COPY_MAIN,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          #endif
          
          #if ENABLE_SLICE_PROFILING
          t0 = clock64();
          #endif
          postPeer<Recv, Send>(0 < sliceWorkSize);
        
          #if ENABLE_SLICE_PROFILING
          long long t1_post = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
              long long dt = t1_post - t0;
              const char* opName = (Send && Recv) ? "RECVSEND" : (Send ? "SEND" : "RECV");
              printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_POST,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          #endif

          offset += currSliceSize;
          slice += 1;
        }
      } else {
        // [Simple] Existing Code Branch
        #pragma unroll 1
        do {
          sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
          if (tid == 0) {
            T* userInput = (T*)ncclShmem.groups[group].userInput;
            T* userOutput = (T*)ncclShmem.groups[group].userOutput;
            if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
            if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
          }

#if ENABLE_SLICE_PROFILING
          long long t0 = clock64();
#endif

          waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
          subBarrier();

#if ENABLE_SLICE_PROFILING
          long long t1_wait = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
            long long dt = t1_wait - t0;
            const char* opName = (Send && Recv) ? "COPY" : (Send ? "SEND" : "RECV");
            printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_WAIT,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          t0 = clock64();
#endif
        
          int workSize = ncclShmem.aborted ? 0 : sliceSize;
          if (flags & AnyNetDeviceUnpack) {
            ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
            subBarrier();
          }

          if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
              && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
            if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
              reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
                (tid, nworkers, /*redArg*/0, /*postOp*/false,
                 1, ncclShmem.groups[group].srcs,
                 fan.nsend(), ncclShmem.groups[group].dsts+1,
                 workSize);
            }
          } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
              (tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp,
               Recv, ncclShmem.groups[group].srcs,
               Dst, ncclShmem.groups[group].dsts,
               workSize);
          } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
            constexpr int PreOpSrcs = SrcBuf != Input ? 0 : 1;
            if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
              reduceCopy<Unroll, RedOp, T,
                0, Recv + Src, Recv * MaxRecv + Src,
                0, 1, 1, PreOpSrcs>
                (tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp,
                  Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                  1, ncclShmem.groups[group].dsts,
                  workSize);
            } else {
              reduceCopy<Unroll, RedOp, T,
                MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
                MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
                (tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp,
                  Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                  Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
                  workSize);
            }
          } else {
            workSize = 0;
          }
          barrier();

#if ENABLE_SLICE_PROFILING
          long long t1_copy = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
            long long dt = t1_copy - t0;
            const char* opName = (Send && Recv) ? "COPY" : (Send ? "SEND" : "RECV");
            printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_COPY,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
          t0 = clock64();
#endif

          postPeer<Recv, Send>(0 < workSize);

#if ENABLE_SLICE_PROFILING
          long long t1_post = clock64();
          if (tid == 0 && ncclShmem.channelId == 0 && profileChunkId == 0) {
            long long dt = t1_post - t0;
            const char* opName = (Send && Recv) ? "COPY" : (Send ? "SEND" : "RECV");
            printf("NCCL_PROFILE_SLICE,%d,%d,%d,%s_POST,%lld\n", ncclShmem.channelId, profileChunkId, slice, opName, dt);
          }
#endif

          offset += sliceSize;
          slice += 1;
        } while (slice < SlicePerChunk && offset < nelem);
      }
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, sliceSize);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      int workSize = ncclShmem.aborted ? 0 : sliceSize;
      postPeer<Recv, Send>(0 < workSize);
      offset += sliceSize;
      slice += 1;
    }
  }

public:
  static inline __device__ void sendPeerNotify(int peer, int connIndex, int steps) {
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->send[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->send[connIndex].tail, peerPtr->send[connIndex].step);
  }

  static inline __device__ void recvPeerNotify(int peer, int connIndex, int steps) {
    int spins = 0;
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->recv[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->recv[connIndex].head, peerPtr->recv[connIndex].step);
    while (ld_volatile_global(peerPtr->recv[connIndex].tail) < peerPtr->recv[connIndex].step) {
      int abort = 0;
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  template<int Recv, int Send, typename Fn>
  __device__ __forceinline__ void process(Fn &&fn, uint32_t sendDirectFlag = 0, uint32_t recvDirectFlag = 0) {
    #pragma unroll 1
    for (int slice=0; slice < SlicePerChunk; slice++) {
      if (tid < nworkers) {
        int nsend, nrecv;
        if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
          const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
          int spins = 0;
          while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
            connStepCache = loadStepValue(connStepPtr);
            if (checkAbort(flags, Aborted, spins)) break;
          }
          void **ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts
                                      : ncclShmem.groups[group].srcs;
          if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
            int offset = loadInt(&connFifo[step%NCCL_STEPS].offset);
            ptrs[index] = connEltsFifo + offset/sizeof(T);
          } else if (Direct && fn.work->regUsed) {
            if (isSendNotRecv) {
              if (flags & DirectWrite) {
                ptrs[index] = directBuff;
              } else if (flags & DirectRead) {  // empty send
                ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            } else {
              if (flags & DirectRead) {
                ptrs[index] = directBuff;
              } else if (flags & DirectWrite) {
                if (Send)
                  ptrs[index] = directBuff;  // send to next from my output buffer
                else
                  ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            }
          } else {
            ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
          }
        }
        subBarrier();
        if (Recv == 0 || ncclShmem.groups[group].srcs[0] == nullptr) {
          nrecv = 0;
        } else {
          nrecv = fan.nrecv();
        }

        if (Send == 0 || ncclShmem.groups[group].dsts[0] == nullptr) {
          nsend = 0;
        } else {
          nsend = fan.nsend();
        }
        fn.template operator()<SlicePerChunk, 0, Recv*MaxRecv, 0, Send*MaxSend, MultimemSrcs, MultimemDsts>
          (tid, nworkers, slice, stepSize * StepPerSlice,
            nrecv, ncclShmem.groups[group].srcs,
            nsend, ncclShmem.groups[group].dsts, ncclShmem.groups[group].dstSizes, sendDirectFlag, recvDirectFlag);
      }
      barrier();
      int32_t dstSize = 0;
      if (flags & Send*RolePostSend) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_begin]
        dstSize = ncclShmem.groups[group].dstSizes[index];
        ncclShmem.groups[group].dstSizes[index] = 0;
        if (flags & ConnFifoEnabled) connFifo[step%NCCL_STEPS].size = dstSize*sizeof(T);
      }
      barrier();
      if (flags & (Recv*(RoleWaitRecv|RolePostRecv) | Send*(RoleWaitSend|RolePostSend))) {
        step += StepPerSlice;
      }
      if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
        if (Send && (!Recv || (flags & RolePostSend)) && (dstSize!=0 || (flags&ConnFifoEnabled))) {
          fence_acq_rel_sys();
        }
        st_relaxed_sys_global(connStepPtr, step);
      }
    }
  }

private:
  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      ssize_t realSize = max(0, min(dataSize, peerElem-offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (tid==0) ncclShmem.groups[group].srcs[0] = (T*)ncclShmem.groups[group].userInput + inpIx + offset;
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            ssize_t pOffset = i*peerOffset;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, PreOpSrcs>(tid, nworkers, ncclShmem.groups[group].redOpArgs, false, 1, &src0, 1, ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (tid==0) ncclShmem.groups[group].dsts[0] = (T*)ncclShmem.groups[group].userOutput + outIx + offset;
          ssize_t pOffset = index*peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerOffset;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx+pOffset, outIx+pOffset, offset, realSize);
          subBarrier();
          #pragma unroll
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            pOffset = i*peerOffset;
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (DirectRecv && ncclShmem.groups[group].srcs[i] == dst0) realPeerSize = 0;
            if (realPeerSize > 0) reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(tid, nworkers, ncclShmem.groups[group].redOpArgs, postOp, 1, ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->recv[connIndex];
    if (conn->netDeviceHandle.netDeviceType == NCCL_NET_DEVICE_UNPACK) {
      // handle must be a device ptr
      netDeviceHandle = conn->netDeviceHandle.handle;
      // Cache the handle
      ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
      flags |= NetDeviceUnpack;
    }
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);
    if (flags & RolePostRecv) {
      connStepPtr = conn->head;
      *connStepPtr = step; // Return credits in case we rounded up.
    }
    if (flags & RoleWaitRecv) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->tail;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
      if (conn->connFifo != nullptr) {
        flags |= ConnFifoEnabled;
        connFifo = conn->connFifo;
      }
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectRead;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
            connFifo[step % NCCL_STEPS].size = 0;
          }
        }
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->send[connIndex];
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);

    connFifo = conn->connFifo;
    if (connFifo != nullptr) flags |= ConnFifoEnabled;

    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
    }
    if (flags & RoleWaitSend) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_TMA];
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectWrite;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
          }
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, int mode = primsModeDefault
    ):
    tid(tid), nthreads(nthreads), tidInBlock(threadIdx.x), group(group),
    stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_TMA]/NCCL_STEPS/sizeof(T) : stepSize_) {

    // [jihwan] Shared memory test print  
    // if (tid == 0 && group == 0) {
    //   void* tmaPtr = ncclTmaShmemPtr();
    //   void* basePtr = ncclShmemPerWarp;
    //   long offset = (char*)tmaPtr - (char*)basePtr;
    //   // [Simple Verification]
    //   printf("[TMA_DEBUG] Rank %d Block %d: Allocated Dynamic SMEM Size = %d bytes\n", ncclShmem.comm.rank, blockIdx.x, ncclShmemDynamicSize());
    //   printf("[TMA_DEBUG] Rank %d Block %d: TMA Buffer Start = %p (Offset from base = %ld bytes)\n", ncclShmem.comm.rank, blockIdx.x, tmaPtr, offset);
    //   printf("[TMA_DEBUG] Rank %d Block %d: Targeted TMA Size = %d bytes\n", ncclShmem.comm.rank, blockIdx.x, NCCL_TMA_TOTAL_SMEM_SIZE);
      
    //   // Simple write check (only if enough space)
    //   if (offset + 128 < ncclShmemDynamicSize() + NCCL_TMA_TOTAL_SMEM_SIZE) {
    //      *(int*)tmaPtr = 0xDEADBEEF;
    //   }
    // }

    int peer = -1;
    flags = 0;
    index = -1;
    if (mode == primsModeDefault) { // Connect to ranks in sendPeers/recvPeers
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

      int nrecv=0, nsend=0;
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
      // coverity[dead_error_line]
      while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        MaxSend >= 16 || MaxRecv >= 16 ? 32 : // NVLS may have an arity > 8. In that case increase the size of the groups
        MaxSend >= 8 || MaxRecv >= 8 ? 16 :
        8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
      static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

      assert(2*(nrecv+nsend) <= nthreads); // Ensure no thread is assigned more than one role.
      // Coverity assumes that index will equal tid based on the line below, but it doesn't consider the setting
      // of flags.  This results in multiple false positive overruns being reported here and in all_reduce.h.
      // Unfortunately, we've been unsuccessful in trying to silence them with a single directive here so
      // instead it's being done at the callers.
      // coverity[assignment:FALSE]
      if      (tid < nrecv)                 { flags |= RoleWaitRecv; index = tid; }
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      else if (tid < nrecv+nsend)           { flags |= RoleWaitSend; index = tid-nrecv; }
      else if (nthreads-nsend <= tid)       { flags |= RolePostSend; index = tid-(nthreads-nsend); }
      else if (nthreads-nrecv-nsend <= tid) { flags |= RolePostRecv; index = tid-(nthreads-nrecv-nsend); }

      if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
      if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

      // Coverity thinks that index could be -1 here but that's not actually the case.
      // coverity[negative_returns:FALSE]
      int sendIpcReg;
      int recvIpcReg;
      int sendNetReg;
      int recvNetReg;
      if (P2p) {
        sendIpcReg = p2pWork ? p2pWork->sendIpcReg : 0;
        recvIpcReg = p2pWork ? p2pWork->recvIpcReg : 0;
        sendNetReg = p2pWork ? p2pWork->sendNetReg : 0;
        recvNetReg = p2pWork ? p2pWork->recvNetReg : 0;
      } else {
        recvIpcReg = sendIpcReg = collWork ? collWork->regUsed : 0;
        recvNetReg = sendNetReg = collWork ? collWork->netRegUsed : 0;
      }

      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(ncclShmem.channel.peers[peer], connIndexRecv, collWork ? collWork->direct : 0, recvIpcReg, recvNetReg);
      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(ncclShmem.channel.peers[peer], connIndexSend, collWork ? collWork->direct : 0, sendIpcReg, sendNetReg);

      // coverity[negative_returns:FALSE] => coverity thinks that index could be -1 but that's not actually the case
      // coverity[var_deref_model] => coverity thinks work can dereferenced if NULL but this is not the case
      setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclDevWorkCollReg*)collWork, sendIpcReg || recvIpcReg, peer);
      // coverity[uninit_member] => coverity thinks fan.n is not initialized

      if (barrierAny(flags & NetDeviceUnpack)) {
        flags |= AnyNetDeviceUnpack;
        // RoleWaitRecv starts at tid=0, so this creates the bitmask of which recv peers
        // have NetDeviceUnpack.
        uint32_t mask = __ballot_sync(~0u, ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
        if (tid == 0) {
          ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
        }
      }
    } else if (mode == primsModePatRs || mode == primsModePatAg) { // Connect to all ranks +/- 2^n
      flags |= PatMode;
      const int roles[5] = { RoleWaitRecv, RolePostRecv, RoleWaitSend, RolePostSend, RoleInput | RoleOutput };
      if (tid < 5) flags |= roles[tid];

      int nranks = ncclShmem.comm.nRanks;
      if (tid < 32 && ((1UL<<tid) < nranks)) {
        int rank = ncclShmem.comm.rank;
        uint32_t delta = 1 << tid;
        // Load recv peer
        int recvPeer = mode == primsModePatRs ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        struct ncclPatPeer* peer = ((struct ncclPatPeer*)recvPeers)+tid;
        struct ncclConnInfo* conn = peer->conn = ncclShmem.channel.peers[recvPeer]->recv+connIndexRecv;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_TMA];
        peer->stepCache = loadStepValue(peer->tailPtr = conn->tail);
        peer->headPtr = conn->head;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
        // Load send peer
        int sendPeer = mode == primsModePatAg ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        peer = ((struct ncclPatPeer*)sendPeers)+tid;
        conn = peer->conn = ncclShmem.channel.peers[sendPeer]->send+connIndexSend;
        peer->step = conn->step;
        peer->connFifo = conn->connFifo;
        peer->buff = conn->buffs[NCCL_PROTO_TMA];
        peer->stepCache = loadStepValue(peer->headPtr = conn->head);
        peer->tailPtr = conn->tail;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSizes[NCCL_PROTO_TMA]/sizeof(T);
      }
      if (tid==0) {
        ncclShmem.groups[group].userInput = (void*)inputBuf;
        ncclShmem.groups[group].userOutput = (void*)outputBuf;
        ncclShmem.groups[group].redOpArgs = redOpArg;  // scaler for local input
      }
      patBarrier();
    }
  }

  __device__ ~Primitives() {
    if (flags&PatMode) return;
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) conn->step = step;
    if ((flags & NetRegMode) && (flags & RoleWaitSend)) {
      // Make sure we wait until the proxy has sent data before we return.
      // We don't want the next CUDA kernel to overwrite the send buffer which
      // was accessed directly.
      uint64_t prevStep = step - StepPerSlice;
      volatile ssize_t* ptr = &(connFifo[prevStep%NCCL_STEPS].size);
      int spins = 0;
      while (*ptr != -1) if (checkAbort(flags, Aborted, spins)) break;
    }

    if (flags & NetDeviceUnpack) {
      ncclNetDeviceSaveHead(netDeviceHandle, group, index);
    }

    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();

    if ((flags & DirectRead) && (flags & RoleWaitSend) && P2p) {
      // For sendrecv DirectRead, sender needs to wait for receiver reading data from src.
      // This has to be done after barrier() since post thread might have contention with
      // this check.
      int spins = 0;
      volatile uint64_t* tail = conn->tail;
      volatile uint64_t* head = conn->head;
      while (*tail > *head) if (checkAbort(flags, Aborted, spins)) break;
    }
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclDevWorkCollReg* work, uint8_t ipcReg, int peer) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.groups[group].redOpArgs = redOpArg;  // scaler for local input
    }

    if (Direct && ipcReg) {
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite);
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead); // sender provides direct buffer (to be fetched)
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead); // receiver accepts direct buffer
      if (recvProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.
        if (slot) {
          T* exchgPtr;
          directBuff = (T*)outputBuf;
          while (*slot != nullptr && !checkAbort(flags, Aborted, spins));
          if (P2p) {
            exchgPtr = (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            // coverity[deref_parm:FALSE] => work cannot be NULL if ipcReg != NULL
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
      if (sendAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // coverity[var_deref_op]
          directBuff = (T*)work->dnOutputs[index];
        }
      }
      if (sendProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.
        if (slot) {
          T* exchgPtr;
          while ((*slot != nullptr) && !checkAbort(flags, Aborted, spins));
          // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
          // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
          directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          if (P2p) {
            exchgPtr = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            if (MaxRecv == 0)
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.sendbuffOffset + work->coll.sendbuffRmtAddrs[localPeer]);
            else
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }

          // Exchange pre-scalers for use in direct pull
          *slot = reinterpret_cast<T*>(exchgPtr);
        }
      }
      if (recvAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // Coverity complains about work being possibly NULL below.  However, slot
          // being NULL means that the NVLS buffer is registered (regUsed == 1)
          // so work can't be NULL in this code path.
          // coverity[var_deref_op]
          directBuff = (T*)work->dnInputs[index];
        }
      }
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (T*)ncclShmem.groups[group].userInput + delta;
      ncclShmem.groups[group].userOutput = (T*)ncclShmem.groups[group].userOutput + delta;
    }
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopy(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, -1>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, -1, -1>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void patReduce(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
      if (peer->accSize < ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 && (flags & RoleOutput)) { // Destination is our own local buffer
      ncclShmem.groups[group].dsts[0] = userOutput + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    patBarrier();
    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;
    if (ps->recvDim < 0) { srcs++; nSrcs--; } // No peer to receive from, remove one source

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.groups[group].redOpArgs, /*postOp=*/false,
       nSrcs, srcs, 1, ncclShmem.groups[group].dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->sendDim < 0 && (flags & RoleOutput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & RoleWaitSend)) atomicMax(&peer->accSize, ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

  __device__ __forceinline__ void patCopy(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      if (peer->accSize < ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->recvDim < 0 && (flags & RoleInput)) { // Source is our own local buffer
      ncclShmem.groups[group].srcs[0] = userInput + ps->inpIx;
      if (localAccSize < ps->inpIx + nelem) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
        localAccSize = ps->inpIx + nelem;
      } else {
        // Already done
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0];
      }
    }
    patBarrier();
    int nDsts = 2;
    void** dsts = ncclShmem.groups[group].dsts;
    if (ps->sendDim < 0) { dsts++; nDsts--; } // No peer to send to, remove one dest
    if (ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[1]) nDsts--; // In-place or already done.

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 2, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.groups[group].redOpArgs, /*postOp=*/false,
       1, ncclShmem.groups[group].srcs, nDsts, dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->recvDim < 0 && (flags & RoleInput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->recvDim >= 0 && (flags & RoleWaitRecv)) atomicMax(&peer->accSize, ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

};
