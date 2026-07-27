#pragma once

// Compile-time configuration knobs for the container kernels. The container-set /
// chunk-strategy flags (kUseLazySort, kUseChunkHashMap, kUseLazyTombstoning,
// kUseSingletonChunkMap) are per-bitmap policy and live as static members of
// frsr::roaring::bitmap in bitmap.hpp; the kernel-level knobs below are free.

#include <cstdint>

namespace frsr::roaring::detail {

// Inline storage size for small array containers: covers the typical sparse
// chunk size without heap allocation. Arrays beyond this size spill to heap.
// Keep SBO intentionally small: CRoaring keeps array/run payloads heap-backed
// with near-zero default capacity; shrinking SBO reduces container_variant size
// and chunk-vector move/copy cost in sparse binary-op hot paths.
inline constexpr std::uint32_t array_sbo_size{ 8 };

// Route the in-place bitset combine through the Harley-Seal carry-save kernel
// (fused_combine_inplace_popcount) instead of the naive per-word loop in
// combine_bitset_bitset_inplace (bitset_ops.hpp).
//
// Harley-Seal is what CRoaring uses to compute the result cardinality in one pass:
// a carry-save tree folds 16 SIMD registers' worth of op-results into weighted
// counters so one vector popcount runs per 16 registers instead of one per register.
// In a standalone microbench (clang, AVX2, LTO) the kernel measures faster
// than the naive loop (whose popcount clang already auto-vectorizes via Mula). But
// wired into the live library it regressed large in-place set-ops,
// consistently, inline and [[gnu::noinline]], across repeated measurement
// sessions — the in-context cause is unresolved (whole-module LTO register
// allocation / scheduling, or the cold freshly-copied destination being more
// memory-bound than the hot microbench). Left off by default; the kernel is
// retained, gated, to retry under a different workload shape, a different
// microarchitecture, or GCC, where the in-context codegen may differ. Flip to
// true and A/B the *Inplace + UnionMany set-op bench. Retried on AArch64
// (NEON native per-byte count) at larger scale — neutral-to-worse there too,
// across repeated runs; the in-context loss is not the AVX2 popcount
// emulation. Stays off everywhere.
inline constexpr bool kFusedHarleySealPopcount{ false };

// Route array ∩ array through the SSE4.2 cmpestrm vectorized kernel
// (intersect_array_array_sse42, array_ops.hpp) instead of the scalar two-pointer merge.
//
// The kernel is a faithful port of CRoaring's intersect_vector16 — PCMPESTRM/PCMPISTRM
// match-finding + the portable subscript-shuffle compaction (clang lowers it to a single
// pshufb; no _mm_shuffle_epi8 needed). It is correct (covered by the crosscheck) but
// DEFAULT OFF: on a small-operand apples-to-apples bench it produced zero gain,
// because small array set-ops are per-call-overhead-bound, not kernel-bound — at
// that size CRoaring wins via a leaner per-call path (ra + inline result write),
// not its SIMD kernel, so a faster merge is masked by frsr's chunk_entry/variant/
// get_if overhead. x86-only; a NEON twin would be a separate kernel.
//
// An earlier measurement pass found a net loss on a very-large, heavily-sparse
// synthetic set-ops shape and the flag shipped off. A later differential perf
// pass on a downstream workload proved that shape unrepresentative: the real
// workload issues a large volume of array-intersect operations with a mid-size,
// moderately-skewed operand profile (below the binary-search gate, so it rides
// this path), and the scalar merge loop alone was a large fraction of that
// workload's self time. Enabling the kernel gave a large win on x86, closing a
// meaningful chunk of the gap to CRoaring on that path. The MidSkewIntersect
// bench family (bitmap_bench.cpp) pins that shape; the sparse-overlap loss
// remains real but that shape resolves via the skewed binary-search path in
// practice. x86-only gate (__SSE4_2__); AArch64 keeps the scalar merge.
inline constexpr bool kSimdArrayIntersect{ true };

// AArch64 twin of kSimdArrayIntersect: route array ∩ array through the NEON
// block-merge kernel (intersect_array_array_neon, array_ops.hpp) instead of the
// scalar two-pointer merge. CRoaring has NO NEON counterpart (its ARM builds run
// the scalar merge), so unlike the x86 kernel this is not parity-chasing — it is
// a straight attempt to beat the scalar merge on the same downstream-dominant
// mid-skew shape. Match-finding costs more than PCMPESTRM (8x CMEQ
// broadcast-compare + ADDV mask fold per block), so the win margin is
// narrower than on x86; gated separately to allow an Apple-Silicon-measured
// verdict.
inline constexpr bool kSimdArrayIntersectNeon{ true };

} // namespace frsr::roaring::detail
