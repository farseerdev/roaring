#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined( __ARM_NEON ) && defined( __aarch64__ )
#include <arm_neon.h>
#endif

namespace frsr::roaring::detail {

#if defined( __SSE4_2__ ) || ( defined( __ARM_NEON ) && defined( __aarch64__ ) )
// Lane-compaction table for the SIMD array intersection: table[mask] holds the
// byte-index shuffle control (PSHUFB on x86, TBL on AArch64) compacting the uint16
// lanes named by the set bits of an 8-bit match mask to the front (remaining lanes
// zeroed via the 0x80 high bit — out-of-range for TBL, MSB-set for PSHUFB, both
// yield zero).
// The CRoaring shuffle_mask16 table verbatim — applied with an explicit
// _mm_shuffle_epi8: the earlier portable subscript-shuffle formulation
// (p[k] = va[idx[k]] over a uint16 index table) was NOT pattern-matched to pshufb
// by clang 22 in the LTO build; it lowered to 8 scalar movzwl+vpinsrw per vector
// and dominated the kernel's cycles (measured via profiler annotation on a downstream workload).
inline constexpr std::array<std::array<std::uint8_t, 16>, 256> array_intersect_compact_table{ []{
    std::array<std::array<std::uint8_t, 16>, 256> table{};
    for ( unsigned mask{ 0 }; mask < 256U; ++mask ) {
        unsigned pos{ 0 };
        for ( unsigned lane{ 0 }; lane < 8U; ++lane ) {
            if ( mask & ( 1U << lane ) ) {
                table[ mask ][ 2 * pos     ] = static_cast<std::uint8_t>( 2 * lane );
                table[ mask ][ 2 * pos + 1 ] = static_cast<std::uint8_t>( 2 * lane + 1 );
                ++pos;
            }
        }
        for ( ; pos < 8U; ++pos ) {
            table[ mask ][ 2 * pos     ] = 0x80;
            table[ mask ][ 2 * pos + 1 ] = 0x80;
        }
    }
    return table;
}() };
#endif  // __SSE4_2__ || AArch64 NEON

#if defined( __SSE4_2__ )
// SSE4.2 sorted-array intersection — a port of CRoaring's intersect_vector16. Match
// finding uses PCMPESTRM/PCMPISTRM ("equal any" over 8 uint16 vs 8 uint16); the matched
// lanes of A are compacted to the front with the portable subscript-shuffle and stored,
// then the block with the smaller maximum is advanced (a SIMD merge). Each store writes
// a full 8-lane vector, so `out` must have at least min(sa, sb) + 8 slots of capacity.
// [croaring-ref] deps/croaring/src/array_util.c:intersect_vector16
[[ gnu::hot ]] inline std::size_t intersect_array_array_sse42(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t vl{ 8 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };

    // Compact the matched lanes of `va_raw` (per the 8-bit mask `r`) to the front and
    // store them; advance the running count by the match population.
    auto const compact_store{ [ & ]( __m128i const va_raw, int const r ) {
        __m128i const ctrl{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( array_intersect_compact_table[ static_cast<unsigned>( r ) ].data() ) ) };
        __m128i const p{ _mm_shuffle_epi8( va_raw, ctrl ) };
        _mm_storeu_si128( reinterpret_cast<__m128i *>( out + count ), p );
        count += static_cast<std::size_t>( std::popcount( static_cast<unsigned>( r ) ) );
    } };

    if ( ia < sta && ib < stb ) {
        __m128i va{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) ) };
        __m128i vb{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) ) };
        // PCMPISTRM treats a zero element as a string terminator, so while either current
        // block contains a 0 use the explicit-length PCMPESTRM.
        while ( a[ ia ] == 0 || b[ ib ] == 0 ) {
            __m128i const res{ _mm_cmpestrm( vb, vl, va, vl, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
            compact_store( va, _mm_extract_epi32( res, 0 ) );
            std::uint16_t const amax{ a[ ia + vl - 1 ] };
            std::uint16_t const bmax{ b[ ib + vl - 1 ] };
            if ( amax <= bmax ) {
                ia += vl;
                if ( ia == sta ) { break; }
                va = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) );
            }
            if ( bmax <= amax ) {
                ib += vl;
                if ( ib == stb ) { break; }
                vb = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) );
            }
        }
        // No zeros remain in either current block — use the faster PCMPISTRM.
        if ( ia < sta && ib < stb ) {
            while ( true ) {
                __m128i const res{ _mm_cmpistrm( vb, va, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
                compact_store( va, _mm_extract_epi32( res, 0 ) );
                std::uint16_t const amax{ a[ ia + vl - 1 ] };
                std::uint16_t const bmax{ b[ ib + vl - 1 ] };
                if ( amax <= bmax ) {
                    ia += vl;
                    if ( ia == sta ) { break; }
                    va = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) );
                }
                if ( bmax <= amax ) {
                    ib += vl;
                    if ( ib == stb ) { break; }
                    vb = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) );
                }
            }
        }
    }
    // Scalar tail for the sub-vector remainders of each array.
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        if ( av < bv ) {
            ++ia;
        } else if ( bv < av ) {
            ++ib;
        } else {
            out[ count++ ] = av;
            ++ia;
            ++ib;
        }
    }
    return count;
}

// In-place variant of the SSE4.2 intersection: the result is compacted into the
// front of `a` itself, so no separate (over-allocated) result buffer is needed —
// this is the zero-allocation kernel behind the in-place AND fold, where the
// materializing kernel's +1-vector result buffer forced a payload reallocation
// per pair whenever the recycled payload's capacity was exact. The overwrite
// hazard (a full-vector store landing on lanes of `a` not yet consumed) is
// handled CRoaring-style: matches are staged in a 2-vector on-stack buffer and
// flushed to a[count] only when the read cursor has advanced past the flush
// window. [croaring-ref] deps/croaring/src/array_util.c:intersect_vector16_inplace
[[ gnu::hot ]] inline std::size_t intersect_array_array_inplace_sse42(
    std::uint16_t       * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb
) noexcept {
    constexpr std::size_t vl{ 8 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };

    if ( ia < sta && ib < stb ) {
        alignas( 16 ) std::uint16_t tmp[ 2 * vl ]{};
        std::size_t tmp_count{ 0 };

        // Stage the matched lanes of `va_raw` (per the 8-bit mask `r`) into the tmp buffer.
        auto const compact_stage{ [ & ]( __m128i const va_raw, int const r ) {
            __m128i const ctrl{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( array_intersect_compact_table[ static_cast<unsigned>( r ) ].data() ) ) };
            __m128i const p{ _mm_shuffle_epi8( va_raw, ctrl ) };
            _mm_storeu_si128( reinterpret_cast<__m128i *>( tmp + tmp_count ), p );
            tmp_count += static_cast<std::size_t>( std::popcount( static_cast<unsigned>( r ) ) );
        } };
        // Flush the first vector of staged matches to a[count] — only called right
        // after `ia` advanced past a[count + vl), so the store cannot clobber
        // unconsumed input.
        auto const flush{ [ & ] {
            _mm_storeu_si128( reinterpret_cast<__m128i *>( a + count ), _mm_lddqu_si128( reinterpret_cast<__m128i const *>( tmp ) ) );
            _mm_storeu_si128( reinterpret_cast<__m128i *>( tmp ), _mm_setzero_si128() );
            count += tmp_count;
            tmp_count = 0;
        } };

        __m128i va{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) ) };
        __m128i vb{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) ) };
        // PCMPISTRM treats a zero element as a string terminator, so while either current
        // block contains a 0 use the explicit-length PCMPESTRM.
        while ( a[ ia ] == 0 || b[ ib ] == 0 ) {
            __m128i const res{ _mm_cmpestrm( vb, vl, va, vl, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
            compact_stage( va, _mm_extract_epi32( res, 0 ) );
            std::uint16_t const amax{ a[ ia + vl - 1 ] };
            std::uint16_t const bmax{ b[ ib + vl - 1 ] };
            if ( amax <= bmax ) {
                ia += vl;
                flush();
                if ( ia == sta ) { break; }
                va = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) );
            }
            if ( bmax <= amax ) {
                ib += vl;
                if ( ib == stb ) { break; }
                vb = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) );
            }
        }
        // No zeros remain in either current block — use the faster PCMPISTRM.
        if ( ia < sta && ib < stb ) {
            while ( true ) {
                __m128i const res{ _mm_cmpistrm( vb, va, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
                compact_stage( va, _mm_extract_epi32( res, 0 ) );
                std::uint16_t const amax{ a[ ia + vl - 1 ] };
                std::uint16_t const bmax{ b[ ib + vl - 1 ] };
                if ( amax <= bmax ) {
                    ia += vl;
                    flush();
                    if ( ia == sta ) { break; }
                    va = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) );
                }
                if ( bmax <= amax ) {
                    ib += vl;
                    if ( ib == stb ) { break; }
                    vb = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) );
                }
            }
        }
        // Drain any still-staged matches scalar-wise (tmp_count <= vl) and skip the
        // consumed prefix of the current block: those lanes were already matched
        // against the exhausted region of b. [croaring-ref] same tail handling.
        for ( std::size_t i{ 0 }; i < tmp_count; ++i ) {
            a[ count++ ] = tmp[ i ];
        }
        ia += tmp_count;
    }
    // Scalar tail for the sub-vector remainders of each array.
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        if ( av < bv ) {
            ++ia;
        } else if ( bv < av ) {
            ++ib;
        } else {
            a[ count++ ] = av;
            ++ia;
            ++ib;
        }
    }
    return count;
}
#endif  // __SSE4_2__

#if defined( __ARM_NEON ) && defined( __aarch64__ )
// AArch64 NEON sorted-array intersection — the same block-merge structure as the
// SSE4.2 kernel above, with the two ISA-specific pieces reformulated (CRoaring has
// no NEON twin of intersect_vector16; ARM falls back to scalar there):
//  - match finding: NEON has no PCMPESTRM "equal any", so each block of A is
//    compared against all 8 broadcast lanes of B (8x CMEQ + OR tree) and the
//    per-lane 0xFFFF results are folded to an 8-bit scalar mask by ANDing with a
//    {1,2,4,...,128} weight vector and horizontally adding (ADDV);
//  - compaction: TBL (vqtbl1q_u8) with the shared shuffle-control table — an
//    out-of-range 0x80 index yields zero, matching PSHUFB's MSB convention.
// No zero-terminator special-casing is needed (that is a PCMPISTRM artifact), so a
// single loop suffices. Same contract: each store writes a full 8-lane vector, so
// `out` needs min(sa, sb) + 8 slots of capacity.
[[ gnu::hot ]] inline std::size_t intersect_array_array_neon(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t vl{ 8 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };

    if ( ia < sta && ib < stb ) {
        uint16x8_t const lane_bits{ 1, 2, 4, 8, 16, 32, 64, 128 };
        uint16x8_t va{ vld1q_u16( a + ia ) };
        uint16x8_t vb{ vld1q_u16( b + ib ) };
        while ( true ) {
            uint16x8_t m{                vceqq_u16( va, vdupq_laneq_u16( vb, 0 ) )   };
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 1 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 2 ) ) ) );
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 3 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 4 ) ) ) );
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 5 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 6 ) ) ) );
            m = vorrq_u16( m,            vceqq_u16( va, vdupq_laneq_u16( vb, 7 ) )   );
            unsigned const r{ vaddvq_u16( vandq_u16( m, lane_bits ) ) };
            uint8x16_t const ctrl{ vld1q_u8( array_intersect_compact_table[ r ].data() ) };
            uint8x16_t const p{ vqtbl1q_u8( vreinterpretq_u8_u16( va ), ctrl ) };
            vst1q_u8( reinterpret_cast<std::uint8_t *>( out + count ), p );
            count += static_cast<std::size_t>( std::popcount( r ) );
            std::uint16_t const amax{ a[ ia + vl - 1 ] };
            std::uint16_t const bmax{ b[ ib + vl - 1 ] };
            if ( amax <= bmax ) {
                ia += vl;
                if ( ia == sta ) { break; }
                va = vld1q_u16( a + ia );
            }
            if ( bmax <= amax ) {
                ib += vl;
                if ( ib == stb ) { break; }
                vb = vld1q_u16( b + ib );
            }
        }
    }
    // Scalar tail for the sub-vector remainders of each array.
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        if ( av < bv ) {
            ++ia;
        } else if ( bv < av ) {
            ++ib;
        } else {
            out[ count++ ] = av;
            ++ia;
            ++ib;
        }
    }
    return count;
}

// In-place NEON intersection — same staged-flush discipline as the SSE4.2
// in-place kernel above (see intersect_array_array_inplace_sse42): matches are
// compacted into a 2-vector on-stack buffer and flushed to a[count] only after
// the read cursor has advanced past the flush window, so the result overwrites
// `a` itself with no separate result buffer.
[[ gnu::hot ]] inline std::size_t intersect_array_array_inplace_neon(
    std::uint16_t       * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb
) noexcept {
    constexpr std::size_t vl{ 8 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };

    if ( ia < sta && ib < stb ) {
        alignas( 16 ) std::uint16_t tmp[ 2 * vl ]{};
        std::size_t tmp_count{ 0 };
        uint16x8_t const lane_bits{ 1, 2, 4, 8, 16, 32, 64, 128 };
        uint16x8_t va{ vld1q_u16( a + ia ) };
        uint16x8_t vb{ vld1q_u16( b + ib ) };
        while ( true ) {
            uint16x8_t m{                vceqq_u16( va, vdupq_laneq_u16( vb, 0 ) )   };
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 1 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 2 ) ) ) );
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 3 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 4 ) ) ) );
            m = vorrq_u16( m, vorrq_u16( vceqq_u16( va, vdupq_laneq_u16( vb, 5 ) ),
                                         vceqq_u16( va, vdupq_laneq_u16( vb, 6 ) ) ) );
            m = vorrq_u16( m,            vceqq_u16( va, vdupq_laneq_u16( vb, 7 ) )   );
            unsigned const r{ vaddvq_u16( vandq_u16( m, lane_bits ) ) };
            uint8x16_t const ctrl{ vld1q_u8( array_intersect_compact_table[ r ].data() ) };
            uint8x16_t const p{ vqtbl1q_u8( vreinterpretq_u8_u16( va ), ctrl ) };
            vst1q_u8( reinterpret_cast<std::uint8_t *>( tmp + tmp_count ), p );
            tmp_count += static_cast<std::size_t>( std::popcount( r ) );
            std::uint16_t const amax{ a[ ia + vl - 1 ] };
            std::uint16_t const bmax{ b[ ib + vl - 1 ] };
            if ( amax <= bmax ) {
                ia += vl;
                vst1q_u16( a + count, vld1q_u16( tmp ) );
                vst1q_u16( tmp, vdupq_n_u16( 0 ) );
                count += tmp_count;
                tmp_count = 0;
                if ( ia == sta ) { break; }
                va = vld1q_u16( a + ia );
            }
            if ( bmax <= amax ) {
                ib += vl;
                if ( ib == stb ) { break; }
                vb = vld1q_u16( b + ib );
            }
        }
        for ( std::size_t i{ 0 }; i < tmp_count; ++i ) {
            a[ count++ ] = tmp[ i ];
        }
        ia += tmp_count;
    }
    // Scalar tail for the sub-vector remainders of each array.
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        if ( av < bv ) {
            ++ia;
        } else if ( bv < av ) {
            ++ib;
        } else {
            a[ count++ ] = av;
            ++ia;
            ++ib;
        }
    }
    return count;
}
#endif  // AArch64 NEON

// Skewed sorted-array intersection: when one array is far smaller than the other
// (CRoaring's exact gate — small_size * 64 < large_size, see
// array_container_intersection in croaring/src/containers/array.c), binary-search
// each element of the smaller array in the (monotonically shrinking) remaining
// window of the larger array instead of running the linear two-pointer merge to
// completion. That merge is O(small + large) — every element of `large` gets at
// least one comparison even when almost none of it can possibly match — whereas
// this is O(small * log(large)): decisive when `small` is a handful of elements
// and `large` is a near-array_to_bitset_threshold-sized array container, which is
// common in real (non-uniform-random) roaring workloads with wide range-filter /
// narrow-selection intersections.
// [croaring-ref] deps/croaring/src/array_util.c:intersect_skewed_uint16
//
// The probes run four keys per iteration through an interleaved branch-free
// binary search (four independent halving chains in flight, hiding the load
// latency of each other's cache misses) instead of one serial
// std::lower_bound dependence chain per key.
// [croaring-ref] deps/croaring/src/array_util.c:binarySearch4
template <typename T>
[[ gnu::hot ]] inline void binary_search_4way(
    T const * const array, std::size_t const size,
    T const k0, T const k1, T const k2, T const k3,
    std::size_t & r0, std::size_t & r1, std::size_t & r2, std::size_t & r3
) noexcept {
    if ( size == 0 ) {
        r0 = r1 = r2 = r3 = 0;
        return;
    }
    auto const * base0{ array };
    auto const * base1{ array };
    auto const * base2{ array };
    auto const * base3{ array };
    auto n{ size };
    while ( n > 1 ) {
        auto const half{ n >> 1 };
        base0 = ( base0[ half ] < k0 ) ? base0 + half : base0;
        base1 = ( base1[ half ] < k1 ) ? base1 + half : base1;
        base2 = ( base2[ half ] < k2 ) ? base2 + half : base2;
        base3 = ( base3[ half ] < k3 ) ? base3 + half : base3;
        n -= half;
    }
    r0 = static_cast<std::size_t>( base0 - array ) + ( *base0 < k0 );
    r1 = static_cast<std::size_t>( base1 - array ) + ( *base1 < k1 );
    r2 = static_cast<std::size_t>( base2 - array ) + ( *base2 < k2 );
    r3 = static_cast<std::size_t>( base3 - array ) + ( *base3 < k3 );
}

template <typename T, typename OutVector>
[[ gnu::hot ]] inline std::size_t intersect_array_array_skewed_into(
    T const * const small, std::size_t const size_small,
    T const * const large, std::size_t const size_large,
    OutVector & result
) {
    resize_uninitialized( result, size_small );
    auto * out{ result.data() };
    std::size_t idx_l{ 0 };
    std::size_t idx_s{ 0 };
    while ( idx_s + 4 <= size_small && idx_l < size_large ) {
        auto const k0{ small[ idx_s + 0 ] };
        auto const k1{ small[ idx_s + 1 ] };
        auto const k2{ small[ idx_s + 2 ] };
        auto const k3{ small[ idx_s + 3 ] };
        std::size_t r0, r1, r2, r3;
        auto const * const window{ large + idx_l };
        auto const window_size{ size_large - idx_l };
        binary_search_4way( window, window_size, k0, k1, k2, k3, r0, r1, r2, r3 );
        if ( r0 < window_size && window[ r0 ] == k0 ) { *out++ = k0; }
        if ( r1 < window_size && window[ r1 ] == k1 ) { *out++ = k1; }
        if ( r2 < window_size && window[ r2 ] == k2 ) { *out++ = k2; }
        if ( r3 < window_size && window[ r3 ] == k3 ) { *out++ = k3; }
        idx_l += r3 + ( r3 < window_size && window[ r3 ] == k3 );
        idx_s += 4;
    }
    for ( ; idx_s < size_small && idx_l < size_large; ++idx_s ) {
        auto const value{ small[ idx_s ] };
        auto const * const it{ std::lower_bound( large + idx_l, large + size_large, value ) };
        idx_l = static_cast<std::size_t>( it - large );
        if ( it != large + size_large && *it == value ) {
            *out++ = value;
            ++idx_l;
        }
    }
    auto const count{ static_cast<std::size_t>( out - result.data() ) };
    result.resize( static_cast<std::uint32_t>( count ) );
    return count;
}

// Below this ratio, the linear two-pointer / SSE4.2 merge is used instead of the
// skewed binary-search path — matches CRoaring's threshold exactly (`const int
// threshold = 64;` in array_container_intersection).
inline constexpr std::size_t kSkewedIntersectRatioThreshold{ 64 };

// OutVector: a small_array_values scratch or a handle's payload_vector (direct write).
template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void combine_array_array_into(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    set_operation const op,
    OutVector & result
) {
    auto const * li{ lhs.values.data() };
    auto const * ri{ rhs.values.data() };
    auto const * const li_end{ li + lhs.values.size() };
    auto const * const ri_end{ ri + rhs.values.size() };

    if ( op == set_operation::bit_or ) {
        resize_uninitialized( result, lhs.values.size() + rhs.values.size() );
        auto * out{ result.data() };
        while ( li != li_end && ri != ri_end ) {
            if ( *li < *ri ) {
                *out++ = *li++;
            } else if ( *ri < *li ) {
                *out++ = *ri++;
            } else {
                *out++ = *li;
                ++li;
                ++ri;
            }
        }
        out = std::copy( li, li_end, out );
        out = std::copy( ri, ri_end, out );
        result.resize( static_cast<std::uint32_t>( out - result.data() ) );
        return;
    }

    if ( op == set_operation::bit_and ) {
        {
            auto const sa{ lhs.values.size() };
            auto const sb{ rhs.values.size() };
            auto const lo{ std::min( sa, sb ) };
            auto const hi{ std::max( sa, sb ) };
            if ( lo * kSkewedIntersectRatioThreshold < hi ) {
                if ( sa <= sb ) {
                    intersect_array_array_skewed_into( li, sa, ri, sb, result );
                } else {
                    intersect_array_array_skewed_into( ri, sb, li, sa, result );
                }
                return;
            }
        }
#if defined( __SSE4_2__ )
        // SSE4.2 vectorized intersection for the 16-bit container element type. The
        // store writes a full vector past `count`, so over-allocate by one vector and
        // shrink to the true cardinality afterwards.
        if constexpr ( kSimdArrayIntersect && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
            auto const sa{ lhs.values.size() };
            auto const sb{ rhs.values.size() };
            resize_uninitialized( result, std::min( sa, sb ) + 8U );
            auto const count{ intersect_array_array_sse42( lhs.values.data(), sa, rhs.values.data(), sb, result.data() ) };
            result.resize( static_cast<std::uint32_t>( count ) );
            return;
        }
#elif defined( __ARM_NEON ) && defined( __aarch64__ )
        // NEON vectorized intersection — same over-allocate-by-one-vector contract.
        if constexpr ( kSimdArrayIntersectNeon && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
            auto const sa{ lhs.values.size() };
            auto const sb{ rhs.values.size() };
            resize_uninitialized( result, std::min( sa, sb ) + 8U );
            auto const count{ intersect_array_array_neon( lhs.values.data(), sa, rhs.values.data(), sb, result.data() ) };
            result.resize( static_cast<std::uint32_t>( count ) );
            return;
        }
#endif
        resize_uninitialized( result, std::min( lhs.values.size(), rhs.values.size() ) );
        auto * out{ result.data() };
        while ( li != li_end && ri != ri_end ) {
            if ( *li < *ri ) {
                ++li;
            } else if ( *ri < *li ) {
                ++ri;
            } else {
                *out++ = *li;
                ++li;
                ++ri;
            }
        }
        result.resize( static_cast<std::uint32_t>( out - result.data() ) );
        return;
    }

    resize_uninitialized( result, lhs.values.size() );
    auto * out{ result.data() };
    while ( li != li_end && ri != ri_end ) {
        if ( *li < *ri ) {
            *out++ = *li++;
        } else if ( *ri < *li ) {
            ++ri;
        } else {
            ++li;
            ++ri;
        }
    }
    out = std::copy( li, li_end, out );
    result.resize( static_cast<std::uint32_t>( out - result.data() ) );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] [[ gnu::hot ]] inline small_array_values<typename Layout::low_type> combine_array_array(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    set_operation const op
) {
    small_array_values<typename Layout::low_type> result;
    combine_array_array_into<Layout>( lhs, rhs, op, result );
    return result;
}

// Lazy-union form gate: an array∪array whose worst-case result exceeds this many
// elements is accumulated as a BITSET instead of as an array. Rationale (CRoaring's,
// arrived at by its own benchmarking): every later operand then costs an O(|operand|)
// scatter rather than a full rewrite of the accumulator, and the consumers of an
// accumulated union (intersections against many small operands) are cheaper against a
// bitset; the one-time form repair at finish time pays for itself. Below the bound the
// array stays an array — the rewrite is short, and an 8 KB bitset would cost more in
// cache footprint than the merge saves.
// [croaring-ref] deps/croaring/include/roaring/containers/perfparameters.h:ARRAY_LAZY_LOWERBOUND
//               deps/croaring/src/containers/mixed_union.c:array_array_container_lazy_union
inline constexpr std::size_t kLazyUnionArrayLowerBound{ 1024 };

// True in-place union: grow lhs's payload to the worst-case size and merge
// BACKWARDS (largest values first, writing down from slot la+lb-1). The write
// cursor k never catches the read cursor i (k == i + j + dedup_slack, j >= 1
// inside the loop), so no scratch buffer and no copy-back are needed — the
// former merge-into-scratch + buffer-swap pays a full extra pass.
// [croaring-ref] deps/croaring/src/array_util.c:union_uint16_inplace
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void union_array_array_inplace(
    array_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    auto const la{ lhs.values.size() };
    auto const lb{ rhs.values.size() };
    auto const total{ la + lb };
    lhs.values.resize_uninitialized( total );
    auto       * const a{ lhs.values.data() };
    auto const * const b{ rhs.values.data() };

    auto i{ la };
    auto j{ lb };
    auto k{ total };
    while ( i > 0 && j > 0 ) {
        if ( a[ i - 1 ] > b[ j - 1 ] ) {
            a[ --k ] = a[ --i ];
        } else if ( b[ j - 1 ] > a[ i - 1 ] ) {
            a[ --k ] = b[ --j ];
        } else {
            a[ --k ] = a[ --i ];
            --j;
        }
    }
    while ( j > 0 ) {
        a[ --k ] = b[ --j ];
    }
    // The merged tail lives at [k, total); the untouched prefix a[0, i) is
    // already in place. Close the dedup gap between them (no-op when k == i).
    if ( k != i ) {
        std::memmove( a + i, a + k, std::size_t{ total - k } * sizeof( *a ) );
    }
    lhs.values.resize_uninitialized( i + ( total - k ) );
    lhs.sync_header();
}

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void union_array_array_to_vector(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    OutVector & out
) {
    resize_uninitialized( out, lhs.values.size() + rhs.values.size() );

    auto const * li{ lhs.values.data() };
    auto const * ri{ rhs.values.data() };
    auto const * const li_end{ li + lhs.values.size() };
    auto const * const ri_end{ ri + rhs.values.size() };
    auto * out_it{ out.data() };

    while ( li != li_end && ri != ri_end ) {
        if ( *li < *ri ) {
            *out_it++ = *li++;
        } else if ( *ri < *li ) {
            *out_it++ = *ri++;
        } else {
            *out_it++ = *li;
            ++li;
            ++ri;
        }
    }
    out_it = std::copy( li, li_end, out_it );
    out_it = std::copy( ri, ri_end, out_it );
    out.resize( static_cast<std::uint32_t>( out_it - out.data() ) );
}

// Above this result size the in-place intersection stops paying for itself: its
// per-block staging overhead grows with the operand while the allocation it saves
// is a fixed cost. Measured on the in-place array-intersection benchmarks (~30%
// win at 64-element operands, ~30% loss at ~1000), so the bound sits an octave
// inside the win region.
inline constexpr std::uint32_t kInplaceIntersectMaxElements{ 256 };

// Minimal OutVector shim over an existing payload the caller guarantees is big
// enough (in-place use: the result is a subset of that payload's contents, so
// capacity can never be exceeded). resize_uninitialized is a capacity request —
// a no-op here by that guarantee.
template <typename T>
struct inplace_array_out {
    T *         data_;
    std::size_t size_;
    [[nodiscard]] T * data() noexcept { return data_; }
    void resize( std::size_t const n ) noexcept { size_ = n; }
};
template <typename T>
inline void resize_uninitialized( inplace_array_out<T> &, std::size_t ) noexcept {}

// True in-place intersection: the result is compacted into lhs's own payload —
// the in-place AND fold's counterpart to difference_array_array_inplace. The
// skewed and scalar arms write forward with a cursor that can never pass the
// read cursor (the result is a subset of the merge progress); the SIMD arms use
// the staged-flush in-place kernels. Zero allocation on every arm — this is
// what lets the in-place combine walk drop the per-pair result-buffer
// allocation that dominated the AND fold.
// [croaring-ref] deps/croaring/src/containers/array.c:array_container_intersection_inplace
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void intersect_array_array_inplace(
    array_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    auto       * const a { lhs.values.data() };
    auto const * const b { rhs.values.data() };
    std::size_t const sa{ lhs.values.size() };
    std::size_t const sb{ rhs.values.size() };

    auto const finish{ [ & ]( std::size_t const count ) {
        lhs.values.resize( static_cast<std::uint32_t>( count ) );
        lhs.sync_header();
    } };

    {
        auto const lo{ std::min( sa, sb ) };
        auto const hi{ std::max( sa, sb ) };
        if ( lo * kSkewedIntersectRatioThreshold < hi ) {
            // Both roles are in-place safe: the forward write cursor trails both the
            // small-side read (small == lhs) and the matched-window progress
            // (large == lhs), so writing into lhs's own payload never clobbers
            // unread input.
            inplace_array_out<typename Layout::low_type> out{ a, 0 };
            auto const count{
                ( sa <= sb )
                    ? intersect_array_array_skewed_into( a, sa, b, sb, out )
                    : intersect_array_array_skewed_into( b, sb, a, sa, out )
            };
            finish( count );
            return;
        }
    }
#if defined( __SSE4_2__ )
    if constexpr ( kSimdArrayIntersect && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        finish( intersect_array_array_inplace_sse42( a, sa, b, sb ) );
        return;
    }
#elif defined( __ARM_NEON ) && defined( __aarch64__ )
    if constexpr ( kSimdArrayIntersectNeon && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        finish( intersect_array_array_inplace_neon( a, sa, b, sb ) );
        return;
    }
#endif
    // Scalar in-place merge (out <= li always).
    auto const * li{ a };
    auto const * ri{ b };
    auto const * const li_end{ li + sa };
    auto const * const ri_end{ ri + sb };
    auto * out{ a };
    while ( li != li_end && ri != ri_end ) {
        if ( *li < *ri ) {
            ++li;
        } else if ( *ri < *li ) {
            ++ri;
        } else {
            *out++ = *li;
            ++li;
            ++ri;
        }
    }
    finish( static_cast<std::size_t>( out - a ) );
}

// True in-place difference: the write cursor can never pass the read cursor
// (the result is a subset of lhs), so the filter runs forward directly on
// lhs's payload — no scratch buffer, no copy-back.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void difference_array_array_inplace(
    array_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    auto       * const a{ lhs.values.data() };
    auto const * li{ a };
    auto const * ri{ rhs.values.data() };
    auto const * const li_end{ li + lhs.values.size() };
    auto const * const ri_end{ ri + rhs.values.size() };
    auto * out{ a };

    while ( li != li_end && ri != ri_end ) {
        if ( *li < *ri ) {
            *out++ = *li++;
        } else if ( *ri < *li ) {
            ++ri;
        } else {
            ++li;
            ++ri;
        }
    }
    if ( out != li ) {
        std::memmove( out, li, static_cast<std::size_t>( li_end - li ) * sizeof( *a ) );
    }
    out += li_end - li;
    lhs.values.resize_uninitialized( static_cast<std::uint32_t>( out - a ) );
    lhs.sync_header();
}

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void difference_array_array_to_vector(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    OutVector & out
) {
    resize_uninitialized( out, lhs.values.size() );

    auto const * li{ lhs.values.data() };
    auto const * ri{ rhs.values.data() };
    auto const * const li_end{ li + lhs.values.size() };
    auto const * const ri_end{ ri + rhs.values.size() };
    auto * out_it{ out.data() };

    while ( li != li_end && ri != ri_end ) {
        if ( *li < *ri ) {
            *out_it++ = *li++;
        } else if ( *ri < *li ) {
            ++ri;
        } else {
            ++li;
            ++ri;
        }
    }
    out_it = std::copy( li, li_end, out_it );
    out.resize( static_cast<std::uint32_t>( out_it - out.data() ) );
}

// array∩run written directly into caller-supplied output (same OutVector
// contract as filter_array_bitset_into: reusable scratch or, on the hot combine
// path, the result container's own payload). Per-RUN slice copies over a
// monotone cursor: for each run, scan the (sorted) array forward past values
// below the run, then past the values it covers, and block-copy the covered
// slice in one memcpy. Same total element walk as a per-value merge, but the
// match span is emitted as a bulk copy instead of one compare+append per value,
// and both scans are perfectly predictable forward loops (a binary-search-per-
// run variant measured SLOWER on the ArrayRunAndFold bench — mispredicted
// probes beat its O(runs·log n) on real shapes). [croaring-ref: the block-copy
// shape of a downstream engine's run∩array kernel.]
// Exponential (galloping) forward search: first index in [lo, hi) whose key is
// >= limit (Greater == false) or > limit (Greater == true). Two probes settle a
// skip of 0-1 elements (the tiny-skip shapes the linear walk was chosen for),
// while a long skip costs O(log skip) — HW instruction counters on the
// production fold showed the pure linear walk retiring ~7x the instructions of
// a downstream engine's binary-search-per-run kernel on real model shapes,
// whose skew the ArrayRunAndFold bench (which settled the linear walk) does
// not reproduce.
template <bool Greater, typename E>
[[nodiscard]] [[gnu::always_inline]] inline std::size_t gallop_forward(
    E const * const keys, std::size_t lo, std::size_t const hi, E const limit
) noexcept {
    auto const below{ []( E const key, E const lim ) noexcept {
        return Greater ? key <= lim : key < lim;
    } };
    if ( lo == hi || !below( keys[ lo ], limit ) ) {
        return lo;
    }
    std::size_t step{ 1 };
    while ( lo + step < hi && below( keys[ lo + step ], limit ) ) {
        lo += step;
        step <<= 1U;
    }
    auto const window_end{ std::min( lo + step, hi ) };
    // keys[lo] fails the bound; the first passing key lies in (lo, window_end].
    ++lo;
    auto const * const found{ Greater
        ? std::upper_bound( keys + lo, keys + window_end, limit )
        : std::lower_bound( keys + lo, keys + window_end, limit ) };
    return static_cast<std::size_t>( found - keys );
}

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void filter_array_run_into(
    array_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    OutVector & result
) {
    using low_type = typename Layout::low_type;
    auto const * const keys{ lhs.values.data() };
    auto const card{ static_cast<std::size_t>( lhs.values.size() ) };
    resize_uninitialized( result, lhs.values.size() );
    auto * const out{ result.data() };
    std::size_t written{ 0 };
    std::size_t ap{ 0 };
    if ( card < rhs.runs.size() ) {
        // Array-driven variant: the run-driven loop below pays two gallops and a
        // memcpy per run regardless of how few array values are in play, so a
        // small probe against many runs is dominated by that fixed per-run cost.
        // Drive by array element with lazy run advance instead (run.end is
        // inclusive, matching difference_array_run).
        auto       run_it { rhs.runs.begin() };
        auto const run_end{ rhs.runs.end  () };
        while ( ap < card ) {
            auto const value{ keys[ ap ] };
            while ( static_cast<low_type>( run_it->end ) < value ) {
                if ( ++run_it == run_end ) {
                    resize_uninitialized( result, static_cast<std::uint32_t>( written ) );
                    return;
                }
            }
            if ( value >= static_cast<low_type>( run_it->begin ) ) {
                out[ written++ ] = value;
                ++ap;
            } else {
                ap = gallop_forward<false>( keys, ap, card, static_cast<low_type>( run_it->begin ) );
            }
        }
        resize_uninitialized( result, static_cast<std::uint32_t>( written ) );
        return;
    }
    for ( auto const & run : rhs.runs ) {
        ap = gallop_forward<false>( keys, ap, card, static_cast<low_type>( run.begin ) );
        auto const span_begin{ ap };
        ap = gallop_forward<true >( keys, ap, card, static_cast<low_type>( run.end   ) );
        std::memcpy( out + written, keys + span_begin, ( ap - span_begin ) * sizeof( low_type ) );
        written += ap - span_begin;
        if ( ap >= card ) {
            break;
        }
    }
    resize_uninitialized( result, static_cast<std::uint32_t>( written ) );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline small_array_values<typename Layout::low_type> intersect_array_run(
    array_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    small_array_values<typename Layout::low_type> result;
    filter_array_run_into<Layout>( lhs, rhs, result );
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline small_array_values<typename Layout::low_type> difference_array_run(
    array_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    small_array_values<typename Layout::low_type> result;
    result.reserve( static_cast<std::uint32_t>( lhs.values.size() ) );

    auto run_it{ rhs.runs.begin() };
    for ( auto const value : lhs.values ) {
        while ( run_it != rhs.runs.end() && run_it->end < value ) {
            ++run_it;
        }
        if ( run_it == rhs.runs.end() || value < run_it->begin ) {
            result.push_back( value );
        }
    }
    return result;
}

// Membership filter of a sorted array against a bitset — array∩bitset for
// keep_matches == true, array\bitset for keep_matches == false — written directly
// into a caller-supplied output. OutVector is the same contract as
// combine_array_array_into: a reusable small_array_values scratch, or (on the hot
// combine path) the result container's own payload_vector, which lets the caller
// skip the intermediate scratch allocation and the scratch→payload copy that
// make_container_from_sorted_vector otherwise pays per call.
template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void filter_array_bitset_into(
    array_cref<Layout, CowPolicy> const lhs,
    bitset_cref<Layout, CowPolicy> const rhs,
    bool const keep_matches,
    OutVector & result
) {
    resize_uninitialized( result, lhs.values.size() );
    auto const * const keys{ lhs.values.data() };
    auto const card{ static_cast<std::size_t>( lhs.values.size() ) };
    auto * const out{ result.data() };
    if ( card == 0 ) [[ unlikely ]] {
        return;
    }
    // Arch-split kernels, each side settled by binary-alternating A/B on the
    // full witness set:
    //  - aarch64 (Apple M1): intersection uses the word-cached span walk —
    //    sorted keys usually share the bitset word (key >> 6), so load it once
    //    per span and short-circuit whole spans on an all-zero (nothing
    //    survives) or all-one (everything survives) word. Best measured variant
    //    on M1 across downstream workloads.
    //  - x86-64: flat branchless per-key probe for BOTH polarities (CRoaring's
    //    mixed_intersection.c rationale: an unpredictable match pattern costs
    //    more in mispredicted branches than an unconditional store + arithmetic
    //    advance). The span walk measurably regressed downstream workloads on
    //    x86 while gaining nothing elsewhere.
    // ANDNOT stays flat on both arches (matches a downstream engine, whose
    // word-cached kernel serves only the AND side of its merge walk).
#if defined( __aarch64__ ) || defined( _M_ARM64 )
    if ( keep_matches ) {
        std::size_t i{ 0 }, written{ 0 };
        while ( i < card ) {
            auto const word_index{ static_cast<std::size_t>( keys[ i ] ) >> 6U };
            auto const word{ rhs.words[ word_index ] };
            auto const next_base{ static_cast<std::uint32_t>( word_index + 1 ) << 6U };
            if ( word == 0 ) {
                while ( ++i < card && keys[ i ] < next_base ) {}
            } else if ( word == ~std::uint64_t{ 0 } ) {
                do {
                    out[ written++ ] = keys[ i ];
                } while ( ++i < card && keys[ i ] < next_base );
            } else {
                do {
                    auto const value{ keys[ i ] };
                    out[ written ] = value;
                    written += static_cast<std::size_t>( ( word >> ( value & 63U ) ) & 1U );
                } while ( ++i < card && keys[ i ] < next_base );
            }
        }
        resize_uninitialized( result, written );
        return;
    }
#endif
    // Hoist the word base once (matches CRoaring's direct words[] access) and
    // specialize polarity so the inner loop is `+= bit` / `+= 1-bit` with no
    // boolean compare against keep_matches.
    auto const * const words{ rhs.words.begin() };
    auto * outp{ out };
    if ( keep_matches ) {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ keys[ i ] };
            *outp = value;
            outp += static_cast<std::ptrdiff_t>(
                ( words[ static_cast<std::size_t>( value ) >> 6U ] >> ( value & 63U ) ) & 1U
            );
        }
    } else {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ keys[ i ] };
            *outp = value;
            outp += static_cast<std::ptrdiff_t>(
                1U - ( ( words[ static_cast<std::size_t>( value ) >> 6U ] >> ( value & 63U ) ) & 1U )
            );
        }
    }
    resize_uninitialized( result, static_cast<std::size_t>( outp - out ) );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline small_array_values<typename Layout::low_type> filter_array_bitset(
    array_cref<Layout, CowPolicy> const lhs,
    bitset_cref<Layout, CowPolicy> const rhs,
    bool const keep_matches
) {
    small_array_values<typename Layout::low_type> result;
    filter_array_bitset_into<Layout>( lhs, rhs, keep_matches, result );
    return result;
}

// In-place membership filter of `arr` against `bitset` — array∩bitset for
// keep_matches == true, array\bitset for keep_matches == false. The result is a
// subset of the input, so it compacts within arr's own payload — the write cursor
// never overtakes the read cursor (out advances 0 or 1 per element, in advances 1,
// so out <= in throughout), no allocation, no representation change
// (|result| <= |arr| < the bitset threshold ⇒ still an array). The array∩array
// sibling merges into a caller payload via combine_array_array_into; this is the
// array-vs-bitset in-place form used by the sole-referent fast paths of
// operator&= (keep) and operator-= (drop).
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void filter_array_bitset_inplace(
    array_ref<Layout, CowPolicy> arr,
    bitset_cref<Layout, CowPolicy> const bitset,
    bool const keep_matches = true
) {
    auto * const base{ arr.values.data() };
    auto const card{ static_cast<std::size_t>( arr.values.size() ) };
    // Flat branchless probe for both polarities. A downstream engine's prior
    // in-place array-x-bitset compaction attempt with a word-spanning walk
    // found that reading keys shortly after writing survivors a few slots
    // behind in the same cache line stalls the store buffer, and the flat
    // probe keeps that window minimal.
    // Polarity specialized + words base hoisted (same as filter_array_bitset_into).
    auto const * const words{ bitset.words.begin() };
    auto * out{ base };
    if ( keep_matches ) {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ base[ i ] };
            *out = value;
            out += static_cast<std::ptrdiff_t>(
                ( words[ static_cast<std::size_t>( value ) >> 6U ] >> ( value & 63U ) ) & 1U
            );
        }
    } else {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ base[ i ] };
            *out = value;
            out += static_cast<std::ptrdiff_t>(
                1U - ( ( words[ static_cast<std::size_t>( value ) >> 6U ] >> ( value & 63U ) ) & 1U )
            );
        }
    }
    arr.values.resize_uninitialized( static_cast<std::uint32_t>( out - base ) );
}

// In-place membership filter of `arr` against a run list — array∩run for
// keep_matches == true, array\run for keep_matches == false. Same contract as
// filter_array_bitset_inplace above (result ⊆ input, forward compaction within
// arr's own payload, no allocation, stays an array); the run list drives the
// walk over a monotone cursor, so each kept span moves as one memmove block
// instead of one compare+append per element.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void filter_array_run_inplace(
    array_ref<Layout, CowPolicy> arr,
    run_cref<Layout, CowPolicy> const rhs,
    bool const keep_matches = true
) {
    auto * const base{ arr.values.data() };
    auto const card{ static_cast<std::size_t>( arr.values.size() ) };
    std::size_t out{ 0 };
    std::size_t in { 0 };
    auto const emit_span{ [ & ]( std::size_t const span_begin, std::size_t const span_end ) {
        std::memmove( base + out, base + span_begin, ( span_end - span_begin ) * sizeof( *base ) );
        out += span_end - span_begin;
    } };
    for ( auto const & run : rhs.runs ) {
        auto const miss_begin{ in };
        in = gallop_forward<false>( base, in, card, static_cast<typename Layout::low_type>( run.begin ) );
        if ( !keep_matches ) {
            emit_span( miss_begin, in );
        }
        auto const match_begin{ in };
        in = gallop_forward<true >( base, in, card, static_cast<typename Layout::low_type>( run.end ) );
        if ( keep_matches ) {
            emit_span( match_begin, in );
        }
        if ( in == card ) {
            break;
        }
    }
    if ( !keep_matches ) {
        emit_span( in, card );  // tail past the last run: all misses
    }
    arr.values.resize_uninitialized( static_cast<std::uint32_t>( out ) );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline word_array<Layout> combine_bitset_array_words(
    bitset_cref<Layout, CowPolicy> const bitset,
    array_cref<Layout, CowPolicy> const array,
    set_operation const op
) {
    auto words{ bitset.words.as_array() };
    if ( op == set_operation::bit_and ) {
        words.fill( 0 );
    }

    for ( auto const value : array.values ) {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        auto const mask{ std::uint64_t{ 1 } << bit_index };
        switch ( op ) {
            case set_operation::bit_or:
                words[ word_index ] |= mask;
                break;
            case set_operation::bit_and:
                words[ word_index ] |= bitset.words[ word_index ] & mask;
                break;
            case set_operation::bit_andnot:
                words[ word_index ] &= ~mask;
                break;
        }
    }
    return words;
}

} // namespace frsr::roaring::detail
