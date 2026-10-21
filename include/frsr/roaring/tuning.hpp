#pragma once

// Compile-time configuration knobs for the container kernels.

#include <cstddef>
#include <cstdint>

namespace frsr::roaring::detail {

// Inline storage size for small array containers: covers the typical sparse
// chunk size without heap allocation. Arrays beyond this size spill to heap.
// Keep SBO intentionally small: CRoaring keeps array/run payloads heap-backed
// with near-zero default capacity; shrinking SBO reduces container_variant size
// and chunk-vector move/copy cost in sparse binary-op hot paths.
// Equal-terms benchmarking knob (default OFF - never ship it on). The reference
// implementation has no small-buffer optimization at any level, so with this set
// the library gives up its own: container payloads always spill and the combine
// scratch becomes a plain heap vector. Used to measure whether this library beats
// the reference on the reference's own terms, before any SBO advantage is added
// back on top.
#ifndef FRSR_ROARING_NO_SBO
#   define FRSR_ROARING_NO_SBO 0
#endif

inline constexpr std::uint32_t array_sbo_size{ 8 };

// Lazy in-place K-way union (bulk_or_inplace): an array∪array fold whose summed
// cardinality is at or below this bound merges in place; above it the accumulator
// becomes a bitset and every later operand is scattered into it. The reference's
// in-place lazy union (with bitsetconversion, the way this library's consumer
// always called it) converts on the FIRST fold whatever the size. That is the
// right form as soon as the accumulator actually grows — 16 operands of 32
// elements measured 2.4x the reference when merged as arrays — but it turns a
// chunk whose union stays tiny into an 8 KB bitset for nothing, and a 64-way
// union of 4-element chunks measured 4x slower converted than merged (the
// reference pays the same). The bound keeps the merge only where a fold cannot
// have grown the accumulator past a few cache lines.
#ifndef FRSR_ROARING_LAZY_UNION_ARRAY_BOUND
#   define FRSR_ROARING_LAZY_UNION_ARRAY_BOUND 64
#endif
inline constexpr std::size_t lazy_union_array_bound{ FRSR_ROARING_LAZY_UNION_ARRAY_BOUND };

// Lazy in-place K-way exclusive-or (bulk_xor_intermediate): an array against an array whose summed cardinality is
// at or below this bound is merged as arrays; above it the accumulator becomes a bitset, which every later
// operand flips bits in.
// [croaring-ref] deps/croaring/src/containers/perfparameters.h:ARRAY_LAZY_LOWERBOUND
inline constexpr std::size_t lazy_xor_array_bound{ 1024 };

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

// A materializing bitset AND (`a & b`) whose result holds fewer values than the array limit: when true it
// counts the common bits first and extracts them straight from the operands' words into an array container,
// as the reference does (intersect_bitset_bitset_to_array, bitset_ops.hpp); when false it writes the
// intermediate bitset in one fused AND + store + popcount pass and leaves the result a bitset.
//
// DEFAULT OFF. The extra read-only counting pass makes every dense result ~7% slower, and the sparse result
// the extraction is for is the slow side of the trade: extracting 1024 or 2048 values costs ~1.3 us per chunk
// pair against 0.22 us for the fused pass that leaves them in a bitset, so the paired bands with one or two
// bits per word go from ~7x faster than the reference to parity. Only an operand-clustered result (a few
// full words) gains, and by 8%. The in-place AND walk never demotes either way, its accumulators staying
// bitsets. Flip it, and A/B the BitsetAndBitset bands, to match the reference's result container kind.
#ifndef FRSR_ROARING_AND_DEMOTES_SPARSE_BITSETS
#   define FRSR_ROARING_AND_DEMOTES_SPARSE_BITSETS 0
#endif
inline constexpr bool kAndDemotesSparseBitsets{ FRSR_ROARING_AND_DEMOTES_SPARSE_BITSETS != 0 };

// The chunk table of a bitmap with wide (64-bit) keys is a B+ tree; once it holds this many chunks and
// the chunk keys are spread evenly enough, a flat directory indexed by the high bits of the key takes
// the place of the inner levels (chunk_tree.hpp, "the leaf directory"). Measured against the tree from
// a few hundred to a few thousand chunks: lookups of ascending keys are 40% faster, scattered inserts
// and erases 20-30% faster from about two thousand chunks on, lookups of random keys within 15% either
// way and building by appends 4-8% slower; below about a thousand chunks the tree is as fast to read
// and faster to build. Overridable so a test can put every table of a few
// dozen chunks on the directory.
#ifndef FRSR_ROARING_CHUNK_DIRECTORY_MIN_ENTRIES
#   define FRSR_ROARING_CHUNK_DIRECTORY_MIN_ENTRIES 1024
#endif
inline constexpr std::uint64_t chunk_directory_min_entries{ FRSR_ROARING_CHUNK_DIRECTORY_MIN_ENTRIES };

// The directory is given up (the tree is rebuilt over the same leaves) when more than this many leaves
// end in one of its buckets, which is what bounds the leaf chain a lookup walks from its bucket. On
// evenly spread keys a limit of 3, 6 or 12 measures the same; the limit only decides how much local
// crowding a directory tolerates before the tree is the better index.
#ifndef FRSR_ROARING_CHUNK_DIRECTORY_PILE_LIMIT
#   define FRSR_ROARING_CHUNK_DIRECTORY_PILE_LIMIT 6
#endif
inline constexpr unsigned chunk_directory_pile_limit{ FRSR_ROARING_CHUNK_DIRECTORY_PILE_LIMIT };

} // namespace frsr::roaring::detail
