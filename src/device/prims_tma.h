/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMS_TMA_H_
#define NCCL_PRIMS_TMA_H_

#include <cuda/barrier>
#include <cuda/pipeline>
#include <new>
#include <utility>

// Requested fixed TMA staging parameters.
#define NCCL_TMA_PIPE_DEPTH 2
#define NCCL_TMA_SLOT_SIZE (32 * 1024)

// TMA protocol on top of LL128 transport semantics.
template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p,
         int MultimemSrcs, int MultimemDsts, bool isNetOffload>
class Primitives<
    T, RedOp, Fan, Direct,
    ProtoTMA<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>,
    P2p, isNetOffload>
  : public Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload> {
  using Base = Primitives<T, RedOp, Fan, Direct, ProtoLL128, P2p, isNetOffload>;

  const int tid;
  const int nthreads;
  const int group;
  const bool tmaEnabled;
  T* userInput;
  T* userOutput;

  __device__ __forceinline__ void groupBarrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else barrier_sync(15-group, nthreads);
  }

  template<typename SrcPtrFn, typename TileFn>
  __device__ __forceinline__ void runTmaTiles(intptr_t srcIx, int eltN, SrcPtrFn srcPtrFn, TileFn tileFn) {
    if (eltN <= 0) return;

    constexpr int Depth = NCCL_TMA_PIPE_DEPTH;
    const int tileElems = max(1, NCCL_TMA_SLOT_SIZE / (int)sizeof(T));
    const int nTiles = divUp(eltN, tileElems);

    using tma_barrier_t = cuda::barrier<cuda::thread_scope_block>;
    __shared__ alignas(alignof(tma_barrier_t)) char barrierMem[sizeof(tma_barrier_t) * Depth];
    tma_barrier_t* barriers = reinterpret_cast<tma_barrier_t*>(barrierMem);
    typename tma_barrier_t::arrival_token tokens[Depth];

    if (tid == 0) {
      #pragma unroll
      for (int i = 0; i < Depth; ++i) new (&barriers[i]) tma_barrier_t(nthreads);
    }
    groupBarrier();

    auto issueTile = [&](int tileIdx) {
      const int slot = tileIdx % Depth;
      const int tileOffset = tileIdx * tileElems;
      const int count = min(tileElems, eltN - tileOffset);
      const size_t copySize = (size_t)count * sizeof(T);
      const T* globalSrc = srcPtrFn() + srcIx + tileOffset;
      void* shmemDst = (char*)ncclTmaShmemPtr() + slot * NCCL_TMA_SLOT_SIZE;

      if (tid == 0) {
#if __CUDA_ARCH__ >= 900
        if (copySize % 16 == 0 && (uintptr_t)globalSrc % 16 == 0) {
          cuda::device::memcpy_async_tx(
              (char*)shmemDst,
              (char const*)globalSrc,
              cuda::aligned_size_t<16>(copySize),
              barriers[slot]);
          tokens[slot] = cuda::device::barrier_arrive_tx(barriers[slot], 1, copySize);
        } else {
          cuda::memcpy_async(shmemDst, globalSrc, copySize, barriers[slot]);
          tokens[slot] = barriers[slot].arrive();
        }
#else
        cuda::memcpy_async(shmemDst, globalSrc, copySize, barriers[slot]);
        tokens[slot] = barriers[slot].arrive();
#endif
      } else {
        tokens[slot] = barriers[slot].arrive();
      }
    };

    int preload = min(Depth, nTiles);
    for (int i = 0; i < preload; ++i) issueTile(i);

    for (int t = 0; t < nTiles; ++t) {
      const int slot = t % Depth;
      barriers[slot].wait(std::move(tokens[slot]));

      const int tileOffset = t * tileElems;
      const int count = min(tileElems, eltN - tileOffset);
      T* stagedSrc = (T*)((char*)ncclTmaShmemPtr() + slot * NCCL_TMA_SLOT_SIZE);
      tileFn(tileOffset, count, stagedSrc);

      int next = t + Depth;
      if (next < nTiles) issueTile(next);
    }

    groupBarrier();
  }

 public:
  using Base::profileChunkId;

  __device__ Primitives(
      int tid, int nthreads, int const* recvPeers, int const* sendPeers,
      void const* inputBuf, void* outputBuf, uint64_t redOpArg, uint8_t group = 0,
      uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0,
      struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork = nullptr,
      int stepSize_ = 0, int mode = primsModeDefault)
      : Base(tid, nthreads, recvPeers, sendPeers,
             inputBuf, outputBuf, redOpArg, group,
             connIndexRecv, connIndexSend,
             collWork,
             P2p ? (p2pWork ? (p2pWork->sendIpcReg || p2pWork->recvIpcReg) : false)
                 : (collWork ? collWork->regUsed : false),
             P2p ? (p2pWork ? (p2pWork->sendNetReg || p2pWork->recvNetReg) : false)
                 : (collWork ? collWork->netRegUsed : false),
             stepSize_),
        tid(tid), nthreads(nthreads), group(group),
        tmaEnabled(
          !isNetOffload &&
          !(P2p ? (p2pWork ? (p2pWork->sendNetReg || p2pWork->recvNetReg) : false)
                : (collWork ? collWork->netRegUsed : false))
        ),
        userInput((T*)inputBuf), userOutput((T*)outputBuf) {
    // PAT mode is implemented only for Simple primitives.
    (void)mode;
  }

  __device__ __forceinline__ void setDataPtrs(void const* inputBuf, void* outputBuf) {
    userInput = (T*)inputBuf;
    userOutput = (T*)outputBuf;
    Base::setDataPtrs(inputBuf, outputBuf);
  }

  __device__ __forceinline__ void setDataPtrs(
      void const* inputBuf, void* outputBuf, uint64_t redOpArg,
      struct ncclDevWorkCollReg* work, uint8_t ipcReg, int peer) {
    (void)redOpArg;
    (void)work;
    (void)ipcReg;
    (void)peer;
    setDataPtrs(inputBuf, outputBuf);
  }

  __device__ __forceinline__ void moveDataPtrs(intptr_t delta) {
    userInput += delta;
    userOutput += delta;
    Base::moveDataPtrs(delta);
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    if (!tmaEnabled) {
      Base::send(inpIx, eltN);
      return;
    }
    runTmaTiles(inpIx, eltN,
      [&] __device__ () { return (T const*)userInput; },
      [&] __device__ (int /*tileOffset*/, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(stagedSrc, userOutput);
        Base::send(0, tileCount);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    if (!tmaEnabled) {
      Base::sendFromOutput(outIx, eltN);
      return;
    }
    runTmaTiles(outIx, eltN,
      [&] __device__ () { return (T const*)userOutput; },
      [&] __device__ (int /*tileOffset*/, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(userInput, stagedSrc);
        Base::sendFromOutput(0, tileCount);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    Base::recv(outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN) {
    if (!tmaEnabled) {
      Base::recvReduceSend(inpIx, eltN);
      return;
    }
    runTmaTiles(inpIx, eltN,
      [&] __device__ () { return (T const*)userInput; },
      [&] __device__ (int /*tileOffset*/, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(stagedSrc, userOutput);
        Base::recvReduceSend(0, tileCount);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    if (!tmaEnabled) {
      Base::recvReduceCopy(inpIx, outIx, eltN, postOp);
      return;
    }
    runTmaTiles(inpIx, eltN,
      [&] __device__ () { return (T const*)userInput; },
      [&] __device__ (int tileOffset, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(stagedSrc, userOutput);
        Base::recvReduceCopy(0, outIx + tileOffset, tileCount, postOp);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    if (!tmaEnabled) {
      Base::copySend(inpIx, outIx, eltN, postOp);
      return;
    }
    runTmaTiles(inpIx, eltN,
      [&] __device__ () { return (T const*)userInput; },
      [&] __device__ (int tileOffset, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(stagedSrc, userOutput);
        Base::copySend(0, outIx + tileOffset, tileCount, postOp);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    Base::recvCopySend(outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    if (!tmaEnabled) {
      Base::recvReduceCopySend(inpIx, outIx, eltN, postOp);
      return;
    }
    runTmaTiles(inpIx, eltN,
      [&] __device__ () { return (T const*)userInput; },
      [&] __device__ (int tileOffset, int tileCount, T* stagedSrc) {
        Base::setDataPtrs(stagedSrc, userOutput);
        Base::recvReduceCopySend(0, outIx + tileOffset, tileCount, postOp);
      });
    Base::setDataPtrs(userInput, userOutput);
  }

  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    (void)outIx;
    send(inpIx, eltN);
  }

  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    sendFromOutput(outIx, eltN);
  }

  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN) {
    recv(outIx, eltN, false);
  }

  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    copySend(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    (void)inpIx;
    recvCopySend(outIx, eltN, postOp);
  }

  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    (void)inpIx;
    (void)outIx;
    (void)eltN;
    (void)postOp;
  }

  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    (void)outIx;
    (void)postOp;
    recvReduceSend(inpIx, eltN);
  }

  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }

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
};

#endif
