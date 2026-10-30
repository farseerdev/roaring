#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/hw_info.hpp>
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
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
// Both sides must fill at least one 8-lane block or the kernel's vector loop is
// skipped entirely (sta = (sa/vl)*vl == 0) and only its scalar tail runs — while
// the caller has already paid the over-allocation that tail does not need. See
// the dispatch in combine_array_array_into.
inline constexpr std::size_t sse42_intersect_min_side{ 8 };

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
// SSE4.2 sorted-array difference a \ b — a port of CRoaring's difference_vector16 in the
// shape of the intersection kernel above. PCMPISTRM ("equal any", 8 uint16 vs 8 uint16)
// marks the lanes of the current A block that occur in the current B block; the marks
// accumulate while B advances under that block, and once no further B value can match it
// (a_max <= b_max) the UNMARKED lanes are compacted to the front and stored. Each store
// writes a full 8-lane vector, so `out` must have sa + 8 slots.
// Zeros: PCMPISTRM treats a zero lane as a terminator, and sorted input can hold a zero
// only at position 0 — so, as the reference does, that one element is settled up front
// and the loop runs the implicit-length compare throughout. B's sub-vector remainder is
// applied to the last pending A block through a zero-padded, explicit-length compare.
// [croaring-ref] deps/croaring/src/array_util.c:difference_vector16
[[ gnu::hot ]] inline std::size_t difference_array_array_sse42(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t vl{ 8 };
    constexpr int flags{ _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK };
    if ( sa == 0 ) { return 0; }
    if ( sb == 0 ) {
        std::memcpy( out, a, sa * sizeof( *a ) );
        return sa;
    }
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };
    if ( a[ 0 ] == 0 || b[ 0 ] == 0 ) {
        if      ( a[ 0 ] == 0 && b[ 0 ] == 0 ) { ++ia; ++ib; }
        else if ( a[ 0 ] == 0 )                { out[ count++ ] = 0; ++ia; }
        else                                   { ++ib; }
    }
    std::size_t const sta{ ia + ( ( sa - ia ) / vl ) * vl };
    std::size_t const stb{ ib + ( ( sb - ib ) / vl ) * vl };

    // Keep (compact + store) the lanes of `va_raw` whose bit in `keep` is set.
    auto const compact_store{ [ & ]( __m128i const va_raw, unsigned const keep ) {
        __m128i const ctrl{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( array_intersect_compact_table[ keep ].data() ) ) };
        _mm_storeu_si128( reinterpret_cast<__m128i *>( out + count ), _mm_shuffle_epi8( va_raw, ctrl ) );
        count += static_cast<std::size_t>( std::popcount( keep ) );
    } };
    auto const unmarked{ []( __m128i const found ) {
        return static_cast<unsigned>( _mm_extract_epi32( found, 0 ) ^ 0xFF ) & 0xFFU;
    } };

    if ( ia < sta && ib < stb ) {
        __m128i va{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) ) };
        __m128i vb{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) ) };
        __m128i found{ _mm_setzero_si128() };
        while ( true ) {
            found = _mm_or_si128( found, _mm_cmpistrm( vb, va, flags ) );
            std::uint16_t const amax{ a[ ia + vl - 1 ] };
            std::uint16_t const bmax{ b[ ib + vl - 1 ] };
            if ( amax <= bmax ) {
                compact_store( va, unmarked( found ) );
                ia += vl;
                if ( ia == sta ) { break; }
                found = _mm_setzero_si128();
                va = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) );
            }
            if ( bmax <= amax ) {
                ib += vl;
                if ( ib == stb ) { break; }
                vb = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) );
            }
        }
        if ( ia < sta ) {
            // B's vector blocks ran out under the current A block: finish that block
            // against B's remainder, zero-padded, with the true length made explicit.
            alignas( 16 ) std::uint16_t tail[ vl ]{};
            std::memcpy( tail, b + ib, ( sb - ib ) * sizeof( *b ) );
            vb = _mm_load_si128( reinterpret_cast<__m128i const *>( tail ) );
            found = _mm_or_si128( found, _mm_cmpestrm( vb, static_cast<int>( sb - ib ), va, vl, flags ) );
            compact_store( va, unmarked( found ) );
            ia += vl;
        }
    }
    // Scalar tail for the sub-vector remainders, then whatever is left of A survives.
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        if ( av < bv ) {
            out[ count++ ] = av;
            ++ia;
        } else if ( bv < av ) {
            ++ib;
        } else {
            ++ia;
            ++ib;
        }
    }
    if ( ia < sa ) {
        std::memcpy( out + count, a + ia, ( sa - ia ) * sizeof( *a ) );
        count += sa - ia;
    }
    return count;
}

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

#if FRSR_ROARING_VP2
// Sorted-array intersection through AVX-512 VP2INTERSECT. Same merge skeleton as
// intersect_array_array_sse42 — compare the two current blocks, emit the matches,
// advance whichever block has the smaller maximum — with two differences:
//
//   * the block is 16 lanes instead of 8, and
//   * the all-pairs compare is ONE instruction (vp2intersectd) instead of PCMPISTRM,
//     which is what makes the wider block pay. uint16 has no vp2intersect form, so
//     each block is zero-extended to 16 uint32, intersected, compressed, and narrowed
//     back — three cheap shuffles around the one compare.
//
// Store contract, as in the SSE kernel but wider: the compress-store writes a FULL
// 16-lane (32-byte) vector, so `out` needs min(sa, sb) + 16 slots of capacity.
//
// No zero-terminator special case: unlike PCMPISTRM, vp2intersect gives a value 0 no
// special meaning, so the SSE kernel's two-phase PCMPESTRM/PCMPISTRM dance is simply
// absent here.
//
// Measured on Zen 5 against the SSE4.2 kernel it replaces: 2.2-2.5x from 16 elements
// upward, and 2.8x SLOWER at 8 — hence intersect_min_side, mirroring
// x86_v4::union_min_side.
namespace x86_vp2 {

// Both operands must fill at least one 16-lane block, or the kernel is all setup and
// the scalar tail does the work anyway. Measured crossover, not a guess.
inline constexpr std::size_t intersect_min_side{ 16 };

FRSR_ROARING_VP2_KERNEL
inline std::size_t intersect_sorted_uint16(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t vl{ 16 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };

    while ( ia < sta && ib < stb ) {
        __m512i const va{ _mm512_cvtepu16_epi32( _mm256_loadu_si256( reinterpret_cast<__m256i const *>( a + ia ) ) ) };
        __m512i const vb{ _mm512_cvtepu16_epi32( _mm256_loadu_si256( reinterpret_cast<__m256i const *>( b + ib ) ) ) };
        __mmask16 ma, mb;
        _mm512_2intersect_epi32( va, vb, &ma, &mb );
        _mm256_storeu_si256(
            reinterpret_cast<__m256i *>( out + count ),
            _mm512_cvtepi32_epi16( _mm512_maskz_compress_epi32( ma, va ) )
        );
        count += static_cast<std::size_t>( std::popcount( static_cast<unsigned>( ma ) ) );
        std::uint16_t const amax{ a[ ia + vl - 1 ] };
        std::uint16_t const bmax{ b[ ib + vl - 1 ] };
        if ( amax <= bmax ) { ia += vl; }
        if ( bmax <= amax ) { ib += vl; }
    }
    // Scalar tail for the sub-block remainders. Safe to resume at (ia, ib) with no
    // carried state: an intersection emits its matches as it finds them, so unlike a
    // difference there is never a pending block whose marks would be lost here.
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

} // namespace x86_vp2
#endif // FRSR_ROARING_VP2

// Below this ratio, the linear two-pointer / SSE4.2 merge is used instead of the
// skewed binary-search path — matches CRoaring's threshold exactly (`const int
// threshold = 64;` in array_container_intersection).
inline constexpr std::size_t kSkewedIntersectRatioThreshold{ 64 };

// Hybrid branchless scalar merges of two sorted arrays. A branch per value mispredicts about once per value
// on sets that interleave unpredictably, so while four values remain on each side the merge first looks for a
// block of four to handle at once (all below the other side's current value, or equal to the other side's next
// four), which is common and well predicted on clustered data; otherwise it takes four branchless steps.
// [croaring-ref] deps/croaring/src/array_util.c:intersect_uint16, union_uint16, difference_uint16
// A tail copy: both pointers may be null for an empty tail, which memmove does not allow.
template <typename T>
[[ gnu::always_inline ]] inline void move_tail( T * const to, T const * const from, std::size_t const count ) noexcept {
    if ( count != 0 ) { std::memmove( to, from, count * sizeof( T ) ); }
}

template <typename T>
[[ gnu::always_inline ]] inline bool equal_blocks4( T const * const a, T const * const b ) noexcept {
    return std::memcmp( a, b, 4 * sizeof( T ) ) == 0;
}

// Every write goes to out[ k.. ] with k <= i, so `out` may be `a`, but not `b`.
template <typename T>
[[ gnu::hot ]] inline std::size_t intersect_sorted_scalar(
    T const * const a, std::size_t const sa,
    T const * const b, std::size_t const sb,
    T * const out
) noexcept {
    assert( out != b || a == b );
    std::size_t i{ 0 }, j{ 0 }, k{ 0 };
    while ( i + 4 <= sa && j + 4 <= sb ) {
        if ( a[ i + 3 ] < b[ j ] ) {
            i += 4;
        } else if ( b[ j + 3 ] < a[ i ] ) {
            j += 4;
        } else if ( equal_blocks4( a + i, b + j ) ) {
            std::memmove( out + k, a + i, 4 * sizeof( T ) );
            k += 4;
            i += 4;
            j += 4;
        } else {
            for ( int step{ 0 }; step < 4; ++step ) {
                T const x{ a[ i ] }, y{ b[ j ] };
                out[ k ] = x;
                k += ( x == y );
                i += ( x <= y );
                j += ( x >= y );
            }
        }
    }
    while ( i < sa && j < sb ) {
        T const x{ a[ i ] }, y{ b[ j ] };
        out[ k ] = x;
        k += ( x == y );
        i += ( x <= y );
        j += ( x >= y );
    }
    return k;
}

// `out` has sa + sb slots; it may start up to sb values before `a` (the in-place union layout): every write
// lands before out + i + sb.
template <typename T>
[[ gnu::hot ]] inline std::size_t union_sorted_scalar(
    T const * const a, std::size_t const sa,
    T const * const b, std::size_t const sb,
    T * const out
) noexcept {
    std::size_t i{ 0 }, j{ 0 }, k{ 0 };
    while ( i + 4 <= sa && j + 4 <= sb ) {
        if ( a[ i + 3 ] < b[ j ] ) {
            std::memmove( out + k, a + i, 4 * sizeof( T ) );
            k += 4;
            i += 4;
        } else if ( b[ j + 3 ] < a[ i ] ) {
            std::memmove( out + k, b + j, 4 * sizeof( T ) );
            k += 4;
            j += 4;
        } else if ( equal_blocks4( a + i, b + j ) ) {
            std::memmove( out + k, a + i, 4 * sizeof( T ) );
            k += 4;
            i += 4;
            j += 4;
        } else {
            for ( int step{ 0 }; step < 4; ++step ) {
                T const x{ a[ i ] }, y{ b[ j ] };
                out[ k++ ] = x < y ? x : y;
                i += ( x <= y );
                j += ( x >= y );
            }
        }
    }
    while ( i < sa && j < sb ) {
        T const x{ a[ i ] }, y{ b[ j ] };
        out[ k++ ] = x < y ? x : y;
        i += ( x <= y );
        j += ( x >= y );
    }
    move_tail( out + k, a + i, sa - i );
    k += sa - i;
    move_tail( out + k, b + j, sb - j );
    return k + ( sb - j );
}

// Every write goes to out[ k.. ] with k <= i, so `out` may be `a`.
template <typename T>
[[ gnu::hot ]] inline std::size_t difference_sorted_scalar(
    T const * const a, std::size_t const sa,
    T const * const b, std::size_t const sb,
    T * const out
) noexcept {
    std::size_t i{ 0 }, j{ 0 }, k{ 0 };
    while ( i + 4 <= sa && j + 4 <= sb ) {
        if ( a[ i + 3 ] < b[ j ] ) {
            std::memmove( out + k, a + i, 4 * sizeof( T ) );
            k += 4;
            i += 4;
        } else if ( b[ j + 3 ] < a[ i ] ) {
            j += 4;
        } else if ( equal_blocks4( a + i, b + j ) ) {
            i += 4;
            j += 4;
        } else {
            for ( int step{ 0 }; step < 4; ++step ) {
                T const x{ a[ i ] }, y{ b[ j ] };
                out[ k ] = x;
                k += ( x < y );
                i += ( x <= y );
                j += ( x >= y );
            }
        }
    }
    while ( i < sa && j < sb ) {
        T const x{ a[ i ] }, y{ b[ j ] };
        out[ k ] = x;
        k += ( x < y );
        i += ( x <= y );
        j += ( x >= y );
    }
    move_tail( out + k, a + i, sa - i );
    return k + ( sa - i );
}

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

    if ( op == set_operation::bit_or ) {
        resize_uninitialized( result, lhs.values.size() + rhs.values.size() );
        result.resize( static_cast<std::uint32_t>( union_sorted_scalar( li, lhs.values.size(), ri, rhs.values.size(), result.data() ) ) );
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
#if FRSR_ROARING_VP2
        // VP2INTERSECT ahead of SSE4.2 where the CPU has it and both sides fill a
        // 16-lane block: one instruction for the all-pairs compare against PCMPISTRM's
        // 8-lane one. Below the floor, or on a CPU without it, this falls through to
        // the SSE4.2 arm — the kernel is a pure addition, never a replacement.
        if constexpr ( kSimdArrayIntersect && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
            auto const sa{ lhs.values.size() };
            auto const sb{ rhs.values.size() };
            if ( sa >= x86_vp2::intersect_min_side && sb >= x86_vp2::intersect_min_side && have_vp2intersect() ) {
                // +16: the compress-store writes a full 16-lane vector past `count`.
                resize_uninitialized( result, std::min( sa, sb ) + 16U );
                auto const count{ x86_vp2::intersect_sorted_uint16( lhs.values.data(), sa, rhs.values.data(), sb, result.data() ) };
                result.resize( static_cast<std::uint32_t>( count ) );
                return;
            }
        }
#endif
#if defined( __SSE4_2__ )
        // SSE4.2 vectorized intersection for the 16-bit container element type. The
        // store writes a full vector past `count`, so over-allocate by one vector and
        // shrink to the true cardinality afterwards.
        if constexpr ( kSimdArrayIntersect && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
            auto const sa{ lhs.values.size() };
            auto const sb{ rhs.values.size() };
            // Only when the kernel's vector loop can actually run. Below one block
            // per side it degenerates to its own scalar tail, and taking it anyway
            // costs twice over: the +8 store slack it needs is dead, and it pushes
            // the result past the handle's inline payload capacity — so a result
            // that would have lived inline is forced onto the heap. The scalar
            // merge below sizes the result exactly and keeps small ones inline.
            if ( std::min( sa, sb ) >= sse42_intersect_min_side ) {
                resize_uninitialized( result, std::min( sa, sb ) + 8U );
                auto const count{ intersect_array_array_sse42( lhs.values.data(), sa, rhs.values.data(), sb, result.data() ) };
                result.resize( static_cast<std::uint32_t>( count ) );
                return;
            }
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
        result.resize( static_cast<std::uint32_t>( intersect_sorted_scalar( li, lhs.values.size(), ri, rhs.values.size(), result.data() ) ) );
        return;
    }

    assert( op == set_operation::bit_andnot && "an exclusive-or goes through xor_array_array_to_vector" );
    resize_uninitialized( result, lhs.values.size() );
    result.resize( static_cast<std::uint32_t>( difference_sorted_scalar( li, lhs.values.size(), ri, rhs.values.size(), result.data() ) ) );
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

// Scalar forward merge of two sorted uint16 arrays into `out` (which must not
// alias the inputs' unread parts and has sa + sb slots). Returns the count.
// [croaring-ref] deps/croaring/src/array_util.c:union_uint16
[[ gnu::hot ]] inline std::size_t union_sorted_uint16_scalar(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t * const out
) noexcept {
    return union_sorted_scalar( a, sa, b, sb, out );
}

#if FRSR_ROARING_X86_V4
// AVX-512 one-pass union of two sorted uint16 arrays: a Batcher bitonic merge of
// 32-lane blocks (one reverse permute + 5 compare-exchange stages per half), each
// merged low half deduplicated against its predecessor with a compare + register
// compress and stored as a FULL vector — so `out` needs sa + sb + 32 slots. The
// register-compress + plain store is deliberate: the compress-to-memory form is
// microcoded and slow on Zen 4 class parts, which the shipped workload runs on.
//
// Aliasing: an in-place caller may lay lhs out as `out + sb` and pass rhs
// separately (CRoaring's array_container_union_inplace shape). After consuming
// p1 + p2 whole blocks this routine has emitted at most 32·(p1 + p2) − 32 values —
// one merged block is always held back in vmax — so even a full 32-lane store
// ends at out + 32·(p1 + p2) ≤ out + sb + 32·p1, the first unread lhs value. The
// bound is exactly tight; emitting vmax sooner or shrinking `pending` breaks
// aliased callers.
// [croaring-ref] deps/croaring/src/array_util.c:avx512_union_uint16
namespace x86_v4 {

FRSR_ROARING_X86_V4_HELPER
inline __m512i bitonic_cx16( __m512i const v, __m512i const t, __mmask32 const hi ) noexcept {
    return _mm512_mask_mov_epi16( _mm512_min_epu16( v, t ), hi, _mm512_max_epu16( v, t ) );
}

// Sort a bitonic 32-lane sequence ascending; every distance has a cheap in-lane shuffle.
FRSR_ROARING_X86_V4_HELPER
inline __m512i bitonic_sort32( __m512i v ) noexcept {
    v = bitonic_cx16( v, _mm512_shuffle_i64x2( v, v, static_cast<_MM_PERM_ENUM>( 0x4E ) ), 0xFFFF0000U ); // d=16
    v = bitonic_cx16( v, _mm512_shuffle_i64x2( v, v, static_cast<_MM_PERM_ENUM>( 0xB1 ) ), 0xFF00FF00U ); // d=8
    v = bitonic_cx16( v, _mm512_shuffle_epi32( v, static_cast<_MM_PERM_ENUM>( 0x4E ) ),    0xF0F0F0F0U ); // d=4
    v = bitonic_cx16( v, _mm512_shuffle_epi32( v, static_cast<_MM_PERM_ENUM>( 0xB1 ) ),    0xCCCCCCCCU ); // d=2
    v = bitonic_cx16( v, _mm512_rol_epi32( v, 16 ),          0xAAAAAAAAU ); // d=1
    return v;
}

// Merge two ascending 32-lane vectors: lo ← the 32 smallest, hi ← the 32 largest.
FRSR_ROARING_X86_V4_HELPER
inline void bitonic_merge32( __m512i const a, __m512i const b, __m512i & lo, __m512i & hi ) noexcept {
    alignas( 64 ) static constexpr std::uint16_t revtab[ 32 ]{
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
    };
    __m512i const br{ _mm512_permutexvar_epi16( _mm512_load_si512( revtab ), b ) };
    lo = bitonic_sort32( _mm512_min_epu16( a, br ) );
    hi = bitonic_sort32( _mm512_max_epu16( a, br ) );
}

// Store the values of the ascending vector `v` that differ from their predecessor
// (lane 0's predecessor is `last`) as a full 32-lane vector at `out`; updates
// `last` to the vector's maximum and returns how many of the stored lanes count.
FRSR_ROARING_X86_V4_HELPER
inline std::size_t emit_unique16( __m512i const v, std::uint16_t * const out, std::uint16_t & last ) noexcept {
    alignas( 64 ) static constexpr std::uint16_t shift1[ 32 ]{
        32,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    };
    __m512i const prev{ _mm512_permutex2var_epi16( v, _mm512_load_si512( shift1 ), _mm512_set1_epi16( static_cast<short>( last ) ) ) };
    __mmask32 const keep{ _mm512_cmpneq_epi16_mask( v, prev ) };
    _mm512_storeu_si512( out, _mm512_maskz_compress_epi16( keep, v ) );
    last = static_cast<std::uint16_t>( _mm_extract_epi16( _mm512_extracti32x4_epi32( v, 3 ), 7 ) );
    return static_cast<std::size_t>( std::popcount( static_cast<std::uint32_t>( keep ) ) );
}

FRSR_ROARING_X86_V4_KERNEL
inline std::size_t union_sorted_uint16(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t W{ 32 };
    // Both sides need a whole block to start the machine; the caller keeps
    // shorter inputs on the scalar merge.
    std::size_t const blocks_a{ sa / W };
    std::size_t const blocks_b{ sb / W };
    std::size_t pa{ 1 };
    std::size_t pb{ 1 };
    auto * o{ out };
    __m512i vmin, vmax;
    bitonic_merge32( _mm512_loadu_si512( a ), _mm512_loadu_si512( b ), vmin, vmax );
    // Lane 0 of the first block has no predecessor: seed `last` with a value that
    // differs from it so the first value is kept.
    std::uint16_t last{ static_cast<std::uint16_t>( _mm_extract_epi16( _mm512_castsi512_si128( vmin ), 0 ) ^ 1 ) };
    o += emit_unique16( vmin, o, last );

    while ( pa < blocks_a && pb < blocks_b ) {
        // Which side advances is essentially unpredictable: select without branching.
        auto const * const na{ a + W * pa };
        auto const * const nb{ b + W * pb };
        bool const take_a{ *na <= *nb };
        __m512i const v{ _mm512_loadu_si512( take_a ? na : nb ) };
        pa += take_a;
        pb += !take_a;
        bitonic_merge32( v, vmax, vmin, vmax );
        o += emit_unique16( vmin, o, last );
    }

    // The ragged tail: the block still held in vmax (which may hold a value twice,
    // once from each input, so it goes through the dedup emit), at most W−1
    // leftovers of the exhausted side, and the rest of the other side. Two two-way
    // merges branch-predict far better than one three-way merge. Every remaining
    // value is >= last and sorted, so only the first of each scalar input can still
    // duplicate `last`.
    alignas( 64 ) std::uint16_t pending[ W ];
    std::uint16_t pending_last{ last };
    std::size_t const npending{ emit_unique16( vmax, pending, pending_last ) };

    auto const * rest_short{ a + W * pa };   std::size_t nshort{ sa - W * pa };
    auto const * rest_long { b + W * pb };   std::size_t nlong { sb - W * pb };
    if ( pa != blocks_a ) { std::swap( rest_short, rest_long ); std::swap( nshort, nlong ); }
    if ( nshort > 0 && *rest_short == last ) { ++rest_short; --nshort; }
    if ( nlong  > 0 && *rest_long  == last ) { ++rest_long;  --nlong;  }

    std::uint16_t merged[ 2 * W ];  // <= W pending + (W−1) leftovers
    std::size_t const nmerged{ union_sorted_uint16_scalar( pending, npending, rest_short, nshort, merged ) };
    o += union_sorted_uint16_scalar( merged, nmerged, rest_long, nlong, o );
    return static_cast<std::size_t>( o - out );
}

// Extra output slots the kernel's full-vector stores need past sa + sb.
inline constexpr std::size_t union_store_slack{ 32 };
// Below this per-side size the scalar merge is used (no whole 32-lane block).
inline constexpr std::size_t union_min_side{ 32 };

// Widening decode of an array container into 32-bit values, sixteen per store:
// a 256-bit load of keys is widened to a full 512-bit block, the chunk base is
// added, and the block is stored whole; the tail takes one masked load/store
// pair instead of a scalar loop. The baseline loop in container_decode_into
// auto-vectorizes only to 256-bit stores, half the width this tier moves per
// instruction, and the decode is store-bound.
// [croaring-ref] deps/croaring/src/array_util.c:avx512_array_container_to_uint32_array
FRSR_ROARING_X86_V4_KERNEL
inline std::uint32_t * decode_array_uint32(
    std::uint16_t const * const keys, std::size_t const count, std::uint32_t * const out, std::uint32_t const base
) noexcept {
    __m512i const base_lanes{ _mm512_set1_epi32( static_cast<int>( base ) ) };
    std::size_t i{ 0 };
    for ( ; i + 16 <= count; i += 16 ) {
        __m256i const packed{ _mm256_loadu_si256( reinterpret_cast<__m256i const *>( keys + i ) ) };
        _mm512_storeu_si512( out + i, _mm512_add_epi32( _mm512_cvtepu16_epi32( packed ), base_lanes ) );
    }
    if ( i < count ) {
        auto const tail{ static_cast<__mmask16>( ( 1U << ( count - i ) ) - 1U ) };
        __m256i const packed{ _mm256_maskz_loadu_epi16( tail, keys + i ) };
        _mm512_mask_storeu_epi32( out + i, tail, _mm512_add_epi32( _mm512_cvtepu16_epi32( packed ), base_lanes ) );
    }
    return out + count;
}
// Below this count the dispatched call is not worth its out-of-line hop.
inline constexpr std::size_t decode_array_min_count{ 64 };

} // namespace x86_v4
#endif


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
    if ( lhs.values.data() == rhs.values.data() ) [[unlikely]] { return; }  // self-union
#if FRSR_ROARING_X86_V4
    // The AVX-512 merge cannot run backwards, so it takes CRoaring's in-place shape:
    // shift lhs up by lb and merge forwards into the vacated front — the kernel's
    // held-back block keeps the write cursor behind the shifted lhs (see its note).
    if constexpr ( std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        if ( la >= x86_v4::union_min_side && lb >= x86_v4::union_min_side && have_x86_v4() ) {
            lhs.values.resize_uninitialized( total );
            auto * const a{ lhs.values.data() };
            std::memmove( a + lb, a, la * sizeof( *a ) );
            auto const n{ x86_v4::union_sorted_uint16( a + lb, la, rhs.values.data(), lb, a ) };
            lhs.values.resize_uninitialized( static_cast<std::uint32_t>( n ) );
            lhs.sync_header();
            return;
        }
    }
#endif
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
#if FRSR_ROARING_X86_V4
    if constexpr ( std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        auto const sa{ lhs.values.size() };
        auto const sb{ rhs.values.size() };
        if ( sa >= x86_v4::union_min_side && sb >= x86_v4::union_min_side && have_x86_v4() ) {
            resize_uninitialized( out, sa + sb + x86_v4::union_store_slack );
            auto const n{ x86_v4::union_sorted_uint16( lhs.values.data(), sa, rhs.values.data(), sb, out.data() ) };
            out.resize( static_cast<std::uint32_t>( n ) );
            return;
        }
    }
#endif
    resize_uninitialized( out, lhs.values.size() + rhs.values.size() );
    out.resize( static_cast<std::uint32_t>( union_sorted_scalar(
        lhs.values.data(), lhs.values.size(), rhs.values.data(), rhs.values.size(), out.data()
    ) ) );
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
    finish( intersect_sorted_scalar( static_cast<typename Layout::low_type const *>( a ), sa, b, sb, a ) );
}

// True in-place difference: the write cursor can never pass the read cursor
// (the result is a subset of lhs), so the filter runs forward directly on
// lhs's payload — no scratch buffer, no copy-back.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void difference_array_array_inplace(
    array_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    auto * const a{ lhs.values.data() };
    lhs.values.resize_uninitialized( static_cast<std::uint32_t>(
        difference_sorted_scalar( static_cast<typename Layout::low_type const *>( a ), lhs.values.size(), rhs.values.data(), rhs.values.size(), a )
    ) );
    lhs.sync_header();
}

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void difference_array_array_to_vector(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    OutVector & out
) {
#if defined( __SSE4_2__ )
    // Same over-allocate-by-one-vector contract as the intersection: the kernel's
    // stores write a full vector past the true count.
    // Below one vector on either side the kernel is all setup (and the +8 slots are
    // an over-allocation of the whole result), so those pairs take the scalar merge.
    if constexpr ( kSimdArrayIntersect && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        auto const sa{ lhs.values.size() };
        auto const sb{ rhs.values.size() };
        if ( sa >= 8U && sb >= 8U ) {
            resize_uninitialized( out, sa + 8U );
            auto const count{ difference_array_array_sse42( lhs.values.data(), sa, rhs.values.data(), sb, out.data() ) };
            out.resize( static_cast<std::uint32_t>( count ) );
            return;
        }
    }
#endif
    resize_uninitialized( out, lhs.values.size() );
    out.resize( static_cast<std::uint32_t>( difference_sorted_scalar(
        lhs.values.data(), lhs.values.size(), rhs.values.data(), rhs.values.size(), out.data()
    ) ) );
}

// The values in exactly one of two sorted arrays. Same hybrid merge as the others: while four values remain on
// each side it looks for a block of four to handle at once (all below the other side's current value, or equal
// to the other side's next four, which cancel), otherwise four branchless steps. `out` has sa + sb slots and
// may not alias either input.
// [croaring-ref] deps/croaring/src/array_util.c:xor_uint16
template <typename T>
[[ gnu::hot ]] inline std::size_t xor_sorted_scalar(
    T const * const a, std::size_t const sa,
    T const * const b, std::size_t const sb,
    T * const out
) noexcept {
    std::size_t i{ 0 }, j{ 0 }, k{ 0 };
    while ( i + 4 <= sa && j + 4 <= sb ) {
        if ( a[ i + 3 ] < b[ j ] ) {
            std::memcpy( out + k, a + i, 4 * sizeof( T ) );
            k += 4;
            i += 4;
        } else if ( b[ j + 3 ] < a[ i ] ) {
            std::memcpy( out + k, b + j, 4 * sizeof( T ) );
            k += 4;
            j += 4;
        } else if ( equal_blocks4( a + i, b + j ) ) {
            i += 4;
            j += 4;
        } else {
            for ( int step{ 0 }; step < 4; ++step ) {
                T const x{ a[ i ] }, y{ b[ j ] };
                out[ k ] = x < y ? x : y;
                k += ( x != y );
                i += ( x <= y );
                j += ( x >= y );
            }
        }
    }
    while ( i < sa && j < sb ) {
        T const x{ a[ i ] }, y{ b[ j ] };
        out[ k ] = x < y ? x : y;
        k += ( x != y );
        i += ( x <= y );
        j += ( x >= y );
    }
    move_tail( out + k, a + i, sa - i );
    k += sa - i;
    move_tail( out + k, b + j, sb - j );
    return k + ( sb - j );
}

#if defined( __SSE4_2__ )
// SSE4.2 sorted-array symmetric difference — a port of CRoaring's xor_vector16. Two sorted 8-lane blocks are
// merged by a min/max network (the smaller block stays in `vec_max` as the running tail, the other is the next
// block of the array whose head is smaller); each merged low half is stored with the values that occur twice (a
// value in both arrays) dropped, comparing each lane with its neighbours across the vector boundary. What is left
// when one array's blocks run out — the running tail and the sub-vector remainders — is finished by a scalar
// exclusive-or. Each store writes a full 8-lane vector, so `out` must have sa + sb + 8 slots.
// [croaring-ref] deps/croaring/src/array_util.c:xor_vector16
inline constexpr std::size_t sse42_xor_min_side{ 8 };

// The lanes of `a` and `b` (both ascending, strictly) sorted ascending: low half in `lo`, high half in `hi`.
// [croaring-ref] deps/croaring/src/array_util.c:sse_merge
[[ gnu::always_inline ]] inline void merge_sorted_blocks_sse( __m128i const a, __m128i const b, __m128i & lo, __m128i & hi ) noexcept {
    __m128i const reverse{ _mm_setr_epi8( 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1 ) };
    __m128i x{ a };
    __m128i y{ _mm_shuffle_epi8( b, reverse ) };
    for ( int stage{ 0 }; stage < 4; ++stage ) {
        __m128i const mn{ _mm_min_epu16( x, y ) };
        __m128i const mx{ _mm_max_epu16( x, y ) };
        x = _mm_unpacklo_epi16( mn, mx );
        y = _mm_unpackhi_epi16( mn, mx );
    }
    lo = x;
    hi = y;
}

// Stores, from the merged stream `old` ++ `fresh`, the last lane of `old` and the first seven of `fresh` with
// every value that occurs twice in a row removed (the last lane of `fresh` is decided by the next block), and
// returns how many values were stored. `old` is all ones on the first call: that lane never survives, since
// its neighbour on the left is itself.
[[ gnu::always_inline ]] inline int store_unique_xor_sse( __m128i const old, __m128i const fresh, std::uint16_t * const out ) noexcept {
    __m128i const two_back{ _mm_alignr_epi8( fresh, old, 16 - 4 ) };
    __m128i const one_back{ _mm_alignr_epi8( fresh, old, 16 - 2 ) };
    __m128i const equal{ _mm_or_si128( _mm_cmpeq_epi16( one_back, two_back ), _mm_cmpeq_epi16( one_back, fresh ) ) };
    unsigned const doubled{ static_cast<unsigned>( _mm_movemask_epi8( _mm_packs_epi16( equal, _mm_setzero_si128() ) ) ) };
    unsigned const keep{ ~doubled & 0xFFU };
    __m128i const ctrl{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( array_intersect_compact_table[ keep ].data() ) ) };
    _mm_storeu_si128( reinterpret_cast<__m128i *>( out ), _mm_shuffle_epi8( one_back, ctrl ) );
    return std::popcount( keep );
}

[[ gnu::hot ]] inline std::size_t xor_sorted_uint16_sse(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb,
    std::uint16_t       * const out
) noexcept {
    constexpr std::size_t vl{ 8 };
    assert( sa >= vl && sb >= vl );
    std::size_t const blocks_a{ sa / vl };
    std::size_t const blocks_b{ sb / vl };
    std::size_t pos_a{ 1 }, pos_b{ 1 };   // blocks taken
    std::uint16_t * cursor{ out };

    __m128i low, high;
    merge_sorted_blocks_sse( _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a ) ), _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b ) ), low, high );
    cursor += store_unique_xor_sse( _mm_set1_epi16( -1 ), low, cursor );
    __m128i last{ low };

    if ( pos_a < blocks_a && pos_b < blocks_b ) {
        std::uint16_t head_a{ a[ vl * pos_a ] };
        std::uint16_t head_b{ b[ vl * pos_b ] };
        __m128i next;
        while ( true ) {
            if ( head_a <= head_b ) {
                next = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + vl * pos_a ) );
                if ( ++pos_a == blocks_a ) { break; }
                head_a = a[ vl * pos_a ];
            } else {
                next = _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + vl * pos_b ) );
                if ( ++pos_b == blocks_b ) { break; }
                head_b = b[ vl * pos_b ];
            }
            merge_sorted_blocks_sse( high, next, low, high );
            cursor += store_unique_xor_sse( last, low, cursor );
            last = low;
        }
        merge_sorted_blocks_sse( high, next, low, high );
        cursor += store_unique_xor_sse( last, low, cursor );
        last = low;
    }
    std::size_t count{ static_cast<std::size_t>( cursor - out ) };

    // The running tail, deduplicated, plus the remainder of the array whose blocks ran out, sorted: a handful of
    // values, which the scalar exclusive-or then merges with the blocks of the other array that were not taken.
    std::uint16_t pending[ 2 * vl + 1 ];
    std::size_t pending_count{ static_cast<std::size_t>( store_unique_xor_sse( last, high, pending ) ) };
    std::uint16_t const tail_top{ static_cast<std::uint16_t>( _mm_extract_epi16( high, 7 ) ) };
    if ( tail_top != static_cast<std::uint16_t>( _mm_extract_epi16( high, 6 ) ) ) { pending[ pending_count++ ] = tail_top; }
    bool const a_ran_out{ pos_a == blocks_a };
    std::uint16_t const * const rest      { a_ran_out ? b + vl * pos_b : a + vl * pos_a };
    std::size_t           const rest_count{ a_ran_out ? sb - vl * pos_b : sa - vl * pos_a };
    std::uint16_t const * const sub_vector{ a_ran_out ? a + vl * blocks_a : b + vl * blocks_b };
    std::size_t           const sub_count { a_ran_out ? sa - vl * blocks_a : sb - vl * blocks_b };
    std::memcpy( pending + pending_count, sub_vector, sub_count * sizeof( std::uint16_t ) );
    pending_count += sub_count;
    if ( pending_count == 0 ) {
        std::memcpy( out + count, rest, rest_count * sizeof( std::uint16_t ) );
        return count + rest_count;
    }
    // insertion sort of at most 16 values, then the pairs in it (equal neighbours) cancel
    for ( std::size_t i{ 1 }; i < pending_count; ++i ) {
        std::uint16_t const v{ pending[ i ] };
        std::size_t j{ i };
        for ( ; j > 0 && pending[ j - 1 ] > v; --j ) { pending[ j ] = pending[ j - 1 ]; }
        pending[ j ] = v;
    }
    std::size_t kept{ 0 };
    for ( std::size_t i{ 0 }; i < pending_count; ++i ) {
        if ( kept != 0 && pending[ kept - 1 ] == pending[ i ] ) { --kept; } else { pending[ kept++ ] = pending[ i ]; }
    }
    return count + xor_sorted_scalar( static_cast<std::uint16_t const *>( pending ), kept, rest, rest_count, out + count );
}

// How many values two sorted arrays share: the intersection kernel above with the stores replaced by a count of the
// matches. Both sides must fill at least one block.
// [croaring-ref] deps/croaring/src/array_util.c:intersect_vector16_cardinality
[[ gnu::hot ]] inline std::size_t intersect_cardinality_array_array_sse42(
    std::uint16_t const * const a, std::size_t const sa,
    std::uint16_t const * const b, std::size_t const sb
) noexcept {
    constexpr std::size_t vl{ 8 };
    std::size_t const sta{ ( sa / vl ) * vl };
    std::size_t const stb{ ( sb / vl ) * vl };
    std::size_t ia{ 0 }, ib{ 0 }, count{ 0 };
    if ( ia < sta && ib < stb ) {
        __m128i va{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( a + ia ) ) };
        __m128i vb{ _mm_lddqu_si128( reinterpret_cast<__m128i const *>( b + ib ) ) };
        // PCMPISTRM treats a zero element as a terminator: while either block holds one, the explicit-length form.
        while ( a[ ia ] == 0 || b[ ib ] == 0 ) {
            __m128i const res{ _mm_cmpestrm( vb, vl, va, vl, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
            count += static_cast<std::size_t>( std::popcount( static_cast<unsigned>( _mm_extract_epi32( res, 0 ) ) ) );
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
        if ( ia < sta && ib < stb ) {
            while ( true ) {
                __m128i const res{ _mm_cmpistrm( vb, va, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK ) };
                count += static_cast<std::size_t>( std::popcount( static_cast<unsigned>( _mm_extract_epi32( res, 0 ) ) ) );
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
    while ( ia < sa && ib < sb ) {
        std::uint16_t const av{ a[ ia ] };
        std::uint16_t const bv{ b[ ib ] };
        count += av == bv;
        ia += av <= bv;
        ib += bv <= av;
    }
    return count;
}
#endif  // __SSE4_2__

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void xor_array_array_to_vector(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    OutVector & out
) {
#if defined( __SSE4_2__ )
    // The vector merge stores whole blocks, so the result buffer has a block of slack; below one block on either
    // side it is all setup, and the scalar merge sizes the result exactly.
    if constexpr ( std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
        auto const sa{ lhs.values.size() };
        auto const sb{ rhs.values.size() };
        if ( sa >= sse42_xor_min_side && sb >= sse42_xor_min_side ) {
            resize_uninitialized( out, sa + sb + 8U );
            out.resize( static_cast<std::uint32_t>( xor_sorted_uint16_sse( lhs.values.data(), sa, rhs.values.data(), sb, out.data() ) ) );
            return;
        }
    }
#endif
    resize_uninitialized( out, lhs.values.size() + rhs.values.size() );
    out.resize( static_cast<std::uint32_t>( xor_sorted_scalar(
        lhs.values.data(), lhs.values.size(), rhs.values.data(), rhs.values.size(), out.data()
    ) ) );
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
//
// The elements need not be bare keys: `key_of` projects the compared key out of an element (the run
// lists are searched by their end), so the run kernels share this search rather than carry their own.
struct identity_key {
    template <typename T>
    [[nodiscard]] constexpr T const & operator()( T const & value ) const noexcept { return value; }
};

// The key of a run in a sorted run list: its inclusive end, as `Key`.
template <typename Key>
struct run_end_key {
    template <typename Run>
    [[nodiscard]] constexpr Key operator()( Run const & run ) const noexcept { return static_cast<Key>( run.end ); }
};

template <bool Greater, typename Element, typename Key, typename KeyOf = identity_key>
[[nodiscard]] [[gnu::always_inline]] inline std::size_t gallop_forward(
    Element const * const keys, std::size_t lo, std::size_t const hi, Key const limit, KeyOf const key_of = {}
) noexcept {
    auto const below{ [ key_of ]( Element const & element, Key const lim ) noexcept {
        auto const key{ key_of( element ) };
        return Greater ? key <= lim : key < lim;
    } };
    if ( lo == hi || !below( keys[ lo ], limit ) ) {
        return lo;
    }
    // A skip of up to four elements settles with predictable compares and no search: the dominant
    // outcome when the key sequence and the limits interleave closely (short runs against an array).
    // [croaring-ref] deps/croaring/include/roaring/array_util.h:advanceUntil
    for ( std::size_t offset{ 1 }; offset <= 3; ++offset ) {
        if ( lo + offset >= hi || !below( keys[ lo + offset ], limit ) ) {
            return lo + offset;
        }
    }
    lo += 3;
    std::size_t step{ 1 };
    while ( lo + step < hi && below( keys[ lo + step ], limit ) ) {
        lo += step;
        step <<= 1U;
    }
    auto const window_end{ std::min( lo + step, hi ) };
    // keys[lo] fails the bound; the first passing key lies in (lo, window_end].
    ++lo;
    auto const * const found{ Greater
        ? std::upper_bound( keys + lo, keys + window_end, limit, [ key_of ]( Key const lim, Element const & element ) noexcept { return lim < key_of( element ); } )
        : std::lower_bound( keys + lo, keys + window_end, limit, [ key_of ]( Element const & element, Key const lim ) noexcept { return key_of( element ) < lim; } ) };
    return static_cast<std::size_t>( found - keys );
}

// The first run at or after `index` (< `size`) whose end is not below `value` (`size` if none). The run the
// previous value landed in usually still reaches this one: one compare. A value a few runs ahead is reached by
// predictable linear steps (a gallop over such a skip mispredicts more than it saves); a value further ahead
// pays the gallop, kept out of line so it does not weigh on the loops that call this per array value.
inline constexpr std::size_t linear_run_steps{ 16 };

// The array values per run from which the values that follow a kept one in its run are moved by one copy loop
// (see filter_array_run_into) instead of being tested one by one.
inline constexpr std::size_t values_per_run_for_stretch_copy{ 16 };

template <typename Run, typename Value>
[[gnu::noinline]] inline std::size_t gallop_to_run_reaching( Run const * const runs, std::size_t const index, std::size_t const size, Value const value ) noexcept {
    return gallop_forward<false>( runs, index, size, value, run_end_key<Value>{} );
}

template <typename Run, typename Value>
[[nodiscard]] [[gnu::always_inline]] inline std::size_t run_reaching( Run const * const runs, std::size_t index, std::size_t const size, Value const value ) noexcept {
    assert( index < size );
    std::size_t steps{ 0 };
    while ( static_cast<Value>( runs[ index ].end ) < value ) {
        if ( ++index == size ) {
            return size;
        }
        if ( ++steps == linear_run_steps ) {
            return gallop_to_run_reaching( runs, index, size, value );
        }
    }
    return index;
}

// With a few array values to a run the kept values come one by one (for the difference, the values in the gap before
// a run), and the loop or copy that moves the stretch of values at once only adds an exit test or a call to each (a
// mispredicted one when the stretches vary): the stretch form is the instantiation for lists with at least
// values_per_run_for_stretch_copy values to a run. The in-place filter below is built the same way.
template <typename Layout, typename OutVector, typename CowPolicy, typename Keep, bool LongStretches>
[[ gnu::hot, gnu::noinline ]] inline void filter_array_run_into_with(
    array_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    OutVector & result
) {
    using low_type = typename Layout::low_type;
    // true_type: the intersection, false_type: the difference.
    constexpr bool keep_matches{ Keep::value };
    auto const * const keys{ lhs.values.data() };
    auto const card{ static_cast<std::size_t>( lhs.values.size() ) };
    resize_uninitialized( result, lhs.values.size() );
    auto * const out{ result.data() };
    std::size_t written{ 0 };
    std::size_t ap{ 0 };
    // Array-driven with a lazy run advance, the reference's shape: each array
    // value is tested against the current run, the run cursor moves only when a
    // value has passed it, and a gap before the next run is crossed by a
    // gallop. A run-driven form (gallop to each run's start and end, memcpy the
    // span) was measured against it: it loses on short runs — two gallops and a
    // libc call per run for a 20-element span put array∩run 12-19 % behind the
    // reference — and an inline copy in its place lost 1.7x on long runs, while
    // this loop is at parity on both. (run.end is inclusive.)
    // The difference (keep_matches == false) keeps what the run does not cover,
    // and the stretch it keeps before a run is the long one when the runs are few,
    // so that stretch is bounded by a gallop and moved in one copy.
    // [croaring-ref] deps/croaring/src/containers/mixed_intersection.c:array_run_container_intersection
    // [croaring-ref] deps/croaring/src/containers/mixed_andnot.c:array_run_container_andnot
    // The run cursor moves by a gallop on the run ends: one compare when the current run still reaches
    // the value (the adjacent-run shape), and a few array values against thousands of short runs cost
    // O(values * log runs) instead of O(runs).
    auto const * const runs   { rhs.runs.data() };
    auto const         run_end{ static_cast<std::size_t>( rhs.runs.size() ) };
    std::size_t        run_ix { 0 };
    auto const finish{ [ & ] {
        if constexpr ( !keep_matches ) {
            // past the last run: every remaining value misses
            std::copy_n( keys + ap, card - ap, out + written );
            written += card - ap;
        }
        resize_uninitialized( result, static_cast<std::uint32_t>( written ) );
    } };
    if ( run_end == 0 ) {
        // a run chunk emptied by removals and kept as a lazily erased slot holds no runs
        finish();
        return;
    }
    while ( ap < card ) {
        auto const value{ keys[ ap ] };
        run_ix = run_reaching( runs, run_ix, run_end, value );
        if ( run_ix == run_end ) {
            finish();
            return;
        }
        auto const & run{ runs[ run_ix ] };
        if ( value >= static_cast<low_type>( run.begin ) ) {
            if constexpr ( keep_matches ) {
                if constexpr ( LongStretches ) {
                    // the values up to the end of the run
                    auto const stretch_end{ static_cast<std::size_t>( run.end ) + 1U };
                    do {
                        out[ written++ ] = keys[ ap++ ];
                    } while ( ap < card && keys[ ap ] < stretch_end );
                } else {
                    out[ written++ ] = value;
                    ++ap;
                }
            } else {
                ap = gallop_forward<true>( keys, ap, card, static_cast<low_type>( run.end ) );
            }
        } else if constexpr ( keep_matches ) {
            ap = gallop_forward<false>( keys, ap, card, static_cast<low_type>( run.begin ) );
        } else if constexpr ( LongStretches ) {
            auto const stretch_end{ gallop_forward<false>( keys, ap, card, static_cast<low_type>( run.begin ) ) };
            std::copy_n( keys + ap, stretch_end - ap, out + written );
            written += stretch_end - ap;
            ap = stretch_end;
        } else {
            out[ written++ ] = value;
            ++ap;
        }
    }
    resize_uninitialized( result, static_cast<std::uint32_t>( written ) );
}

template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics, typename Keep = std::true_type>
[[ gnu::hot ]] inline void filter_array_run_into(
    array_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    OutVector & result,
    Keep = {}
) {
    if ( lhs.values.size() >= rhs.runs.size() * values_per_run_for_stretch_copy ) {
        filter_array_run_into_with<Layout, OutVector, CowPolicy, Keep, true>( lhs, rhs, result );
    } else {
        filter_array_run_into_with<Layout, OutVector, CowPolicy, Keep, false>( lhs, rhs, result );
    }
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
    filter_array_run_into<Layout>( lhs, rhs, result, std::false_type{} );
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
    // Indexed survivor cursor, as in filter_array_bitset_inplace — see the note
    // there for the paired measurement that settled the form.
    std::size_t kept{ 0 };
    if ( keep_matches ) {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ keys[ i ] };
            out[ kept ] = value;
            kept += ( words[ static_cast<std::size_t>( value ) >> 6U ] &
                      ( std::uint64_t{ 1 } << ( value & 63U ) ) ) != 0;
        }
    } else {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ keys[ i ] };
            out[ kept ] = value;
            kept += ( words[ static_cast<std::size_t>( value ) >> 6U ] &
                      ( std::uint64_t{ 1 } << ( value & 63U ) ) ) == 0;
        }
    }
    resize_uninitialized( result, kept );
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
    // Indexed survivor cursor (not an advancing output POINTER) plus a mask
    // membership test (not shift-and-1). Both are load-bearing and they split by
    // polarity — isolated against each other in a dedicated build, paired and
    // interleaved with the reference implementation on a pinned core, min-of-runs,
    // 64-element probe against a dense block (us/op):
    //   intersect: 0.1061 -> 0.0956 (cursor) -> 0.0939 (both)
    //   subtract : 0.1143 -> 0.1093 (cursor) -> 0.0942 (both)
    // The cursor keeps the store address a plain base+index instead of extending
    // the loop-carried pointer chain, which is what the intersect side was paying;
    // the mask test drops the subtract side's `1 - bit` negation (bt+setnc against
    // shift+and+sub), which is what dominates there.
    // ⛔ Do not re-attribute this to the cursor alone.
    // [croaring-ref] src/containers/mixed_andnot.c:array_bitset_container_andnot
    // uses both forms.
    std::size_t kept{ 0 };
    if ( keep_matches ) {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ base[ i ] };
            base[ kept ] = value;
            kept += ( words[ static_cast<std::size_t>( value ) >> 6U ] &
                      ( std::uint64_t{ 1 } << ( value & 63U ) ) ) != 0;
        }
    } else {
        for ( std::size_t i{ 0 }; i < card; ++i ) {
            auto const value{ base[ i ] };
            base[ kept ] = value;
            kept += ( words[ static_cast<std::size_t>( value ) >> 6U ] &
                      ( std::uint64_t{ 1 } << ( value & 63U ) ) ) == 0;
        }
    }
    arr.values.resize_uninitialized( static_cast<std::uint32_t>( kept ) );
}

// In-place membership filter of `arr` against a run list — array∩run for
// keep_matches == true, array\run for keep_matches == false. Same contract as
// filter_array_bitset_inplace above (result ⊆ input, forward compaction within
// arr's own payload, no allocation, stays an array). The walk is
// filter_array_run_into's array-driven shape, the reference's: each array value
// is tested against the current run, the run cursor moves only when a value has
// passed it, a kept value is written on its own, and a stretch the filter drops
// (the gap before a run for ∩, the run's cover for \) is crossed by a gallop. A
// value is never written ahead of where it was read, so the compaction needs no
// scratch.
// [croaring-ref] deps/croaring/src/containers/mixed_intersection.c:array_run_container_intersection
// [croaring-ref] deps/croaring/src/containers/mixed_andnot.c:array_run_container_andnot
template <typename Layout, typename CowPolicy, bool LongStretches>
[[ gnu::hot, gnu::noinline ]] inline void filter_array_run_inplace_with(
    array_ref<Layout, CowPolicy> arr,
    run_cref<Layout, CowPolicy> const rhs,
    bool const keep_matches
) {
    using low_type = typename Layout::low_type;
    auto * const base{ arr.values.data() };
    auto const card{ static_cast<std::size_t>( arr.values.size() ) };
    std::size_t out{ 0 };
    std::size_t in { 0 };
    auto const finish{ [ & ] {
        if ( !keep_matches ) {
            // past the last run: every remaining value misses
            std::memmove( base + out, base + in, ( card - in ) * sizeof( *base ) );
            out += card - in;
        }
        arr.values.resize_uninitialized( static_cast<std::uint32_t>( out ) );
    } };
    auto const * const runs   { rhs.runs.data() };
    auto const         run_end{ static_cast<std::size_t>( rhs.runs.size() ) };
    std::size_t        run_ix { 0 };
    if ( run_end == 0 ) {
        // a run chunk emptied by removals and kept as a lazily erased slot holds no runs
        finish();
        return;
    }
    while ( in < card ) {
        auto const value{ base[ in ] };
        run_ix = run_reaching( runs, run_ix, run_end, value );
        if ( run_ix == run_end ) {
            finish();
            return;
        }
        auto const & run{ runs[ run_ix ] };
        bool const in_run{ value >= static_cast<low_type>( run.begin ) };
        if ( in_run == keep_matches ) {
            // the kept stretch: the values up to the end of the run (intersection) or to the start of the next run (difference)
            if constexpr ( LongStretches ) {
                auto const stretch_end{ keep_matches ? static_cast<std::size_t>( run.end ) + 1U : static_cast<std::size_t>( run.begin ) };
                do {
                    base[ out++ ] = base[ in++ ];
                } while ( in < card && base[ in ] < stretch_end );
            } else {
                base[ out++ ] = value;
                ++in;
            }
        } else if ( in_run ) {
            in = gallop_forward<true >( base, in, card, static_cast<low_type>( run.end ) );
        } else {
            in = gallop_forward<false>( base, in, card, static_cast<low_type>( run.begin ) );
        }
    }
    finish();
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void filter_array_run_inplace(
    array_ref<Layout, CowPolicy> arr,
    run_cref<Layout, CowPolicy> const rhs,
    bool const keep_matches = true
) {
    if ( arr.values.size() >= rhs.runs.size() * values_per_run_for_stretch_copy ) {
        filter_array_run_inplace_with<Layout, CowPolicy, true>( arr, rhs, keep_matches );
    } else {
        filter_array_run_inplace_with<Layout, CowPolicy, false>( arr, rhs, keep_matches );
    }
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
            case set_operation::bit_xor:
                words[ word_index ] ^= mask;
                break;
        }
    }
    return words;
}

} // namespace frsr::roaring::detail
