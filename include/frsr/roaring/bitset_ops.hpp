#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/hw_info.hpp>

#if defined( __x86_64__ ) || defined( _M_X64 )
#include <immintrin.h>
#endif
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

namespace frsr::roaring::detail {

#if FRSR_ROARING_X86_V4

FRSR_ROARING_X86_V4_KERNEL
[[nodiscard]] inline std::size_t combine_words_into_popcount_v4(
    std::uint64_t * const out, std::uint64_t const * const a, std::uint64_t const * const b,
    std::size_t const n, set_operation const op
) noexcept {
    // Fused single pass: with VPOPCNTDQ the popcount no longer serializes the
    // combine (the reason the baseline template keeps two passes), and a second
    // 8 KB read through a separate noinline call measurably loses.
    std::size_t cardinality{ 0 };
    switch ( op ) {
        case set_operation::bit_or:
            for ( std::size_t i{ 0 }; i < n; ++i ) { out[ i ] = a[ i ] |  b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) ); }
            break;
        case set_operation::bit_and:
            for ( std::size_t i{ 0 }; i < n; ++i ) { out[ i ] = a[ i ] &  b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) ); }
            break;
        case set_operation::bit_andnot:
            for ( std::size_t i{ 0 }; i < n; ++i ) { out[ i ] = a[ i ] & ~b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) ); }
            break;
    }
    return cardinality;
}

// In-place form: a[i] = a[i] OP b[i]. Not the materializing kernel with out == a —
// clang versions that loop on an out/a overlap check, and the aliasing call
// takes the scalar fallback (measured 1.6x slower than the baseline tile loop).
FRSR_ROARING_X86_V4_KERNEL
[[nodiscard]] inline std::size_t combine_words_inplace_popcount_v4(
    std::uint64_t * const a, std::uint64_t const * const b, std::size_t const n, set_operation const op
) noexcept {
    std::size_t cardinality{ 0 };
    switch ( op ) {
        case set_operation::bit_or:
            for ( std::size_t i{ 0 }; i < n; ++i ) { a[ i ] |=  b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) ); }
            break;
        case set_operation::bit_and:
            for ( std::size_t i{ 0 }; i < n; ++i ) { a[ i ] &=  b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) ); }
            break;
        case set_operation::bit_andnot:
            for ( std::size_t i{ 0 }; i < n; ++i ) { a[ i ] &= ~b[ i ]; cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) ); }
            break;
    }
    return cardinality;
}

FRSR_ROARING_X86_V4_KERNEL
inline void or_words_inplace_v4( std::uint64_t * const a, std::uint64_t const * const b, std::size_t const n ) noexcept {
    for ( std::size_t i{ 0 }; i < n; ++i ) { a[ i ] |= b[ i ]; }
}

#endif // FRSR_ROARING_X86_V4

#if defined( __x86_64__ ) || defined( _M_X64 )
// Set-bit positions of one byte, eight 16-bit lanes, zero padded past the count.
// [croaring-ref] deps/croaring/src/bitset_util.c:vecDecodeTable_uint16 (0-based here)
inline constexpr auto setbit_decode_table_uint16{ []() {
    std::array<std::array<std::uint16_t, 8>, 256> table{};
    for ( unsigned value{ 0 }; value < 256U; ++value ) {
        unsigned pos{ 0 };
        for ( unsigned bit{ 0 }; bit < 8U; ++bit ) {
            if ( value & ( 1U << bit ) ) { table[ value ][ pos++ ] = static_cast<std::uint16_t>( bit ); }
        }
    }
    return table;
}() };

// Decode the set bits of `n` words into ascending 16-bit values (bit index from the
// first word). One table lookup + vector add + vector store per byte; a zero word
// costs one add. Every byte writes a full 8-lane vector past its true count, so `out`
// must have popcount + 16 slots — the caller over-allocates and shrinks, as for the
// array kernels. [croaring-ref] deps/croaring/src/bitset_util.c:bitset_extract_setbits_sse_uint16
[[ gnu::hot ]] inline std::size_t extract_setbits_sse_uint16(
    std::uint64_t const * const words, std::size_t const n, std::uint16_t * out
) noexcept {
    auto const * const start{ out };
    __m128i       base { _mm_setzero_si128() };
    __m128i const inc8 { _mm_set1_epi16( 8  ) };
    __m128i const inc64{ _mm_set1_epi16( 64 ) };
    for ( std::size_t i{ 0 }; i < n; ++i ) {
        std::uint64_t const w{ words[ i ] };
        if ( w == 0 ) {
            base = _mm_add_epi16( base, inc64 );
            continue;
        }
        for ( unsigned k{ 0 }; k < 8U; ++k ) {
            auto const byte{ static_cast<std::uint8_t>( w >> ( 8U * k ) ) };
            __m128i const positions{ _mm_loadu_si128( reinterpret_cast<__m128i const *>( setbit_decode_table_uint16[ byte ].data() ) ) };
            _mm_storeu_si128( reinterpret_cast<__m128i *>( out ), _mm_add_epi16( base, positions ) );
            out  += std::popcount( static_cast<unsigned>( byte ) );
            base  = _mm_add_epi16( base, inc8 );
        }
    }
    return static_cast<std::size_t>( out - start );
}
#endif

#if FRSR_ROARING_X86_V4
namespace x86_v4 {

// Density above which the AVX-512 decoder below repays its per-word cost over the
// byte-table SSE decoder: measured by CRoaring at ~1024 set bits per 8 KB block
// (1.2x there, 1.4x at the array/bitset boundary).
inline constexpr std::uint32_t extract_setbits_min_cardinality{ 1024 };

// AVX-512 VBMI2 set-bit decoder for one 8 KB block into 16-bit values: per word,
// VPCOMPRESSB compacts the byte indices 0..63 named by the word's bits into a
// vector, which is widened to 16 bits, offset by the word base and masked-stored
// (two halves when more than 32 bits are set). A zero word costs one add. Masked
// stores write exactly the decoded count, so `out` needs only popcount slots.
// [croaring-ref] deps/croaring/src/bitset_util.c:bitset_extract_setbits_avx512_uint16
FRSR_ROARING_X86_V4_KERNEL
inline std::size_t extract_setbits_uint16(
    std::uint64_t const * const words, std::size_t const n, std::uint16_t * out
) noexcept {
    alignas( 64 ) static constexpr std::uint8_t index_table[ 64 ]{
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };
    auto const * const start{ out };
    __m512i       base    { _mm512_setzero_si512() };
    __m512i const inc64   { _mm512_set1_epi16( 64 ) };
    __m512i const indices { _mm512_load_si512( index_table ) };
    for ( std::size_t i{ 0 }; i < n; ++i ) {
        std::uint64_t const w{ words[ i ] };
        if ( w != 0 ) {
            auto const count{ static_cast<unsigned>( std::popcount( w ) ) };
            __m512i const packed{ _mm512_maskz_compress_epi8( static_cast<__mmask64>( w ), indices ) };
            std::uint64_t const lanes{ ~std::uint64_t{ 0 } >> ( 64U - count ) };
            _mm512_mask_storeu_epi16( out, static_cast<__mmask32>( lanes ),
                _mm512_add_epi16( base, _mm512_cvtepu8_epi16( _mm512_castsi512_si256( packed ) ) ) );
            if ( count > 32U ) {
                _mm512_mask_storeu_epi16( out + 32, static_cast<__mmask32>( lanes >> 32U ),
                    _mm512_add_epi16( base, _mm512_cvtepu8_epi16( _mm512_extracti64x4_epi64( packed, 1 ) ) ) );
            }
            out += count;
        }
        base = _mm512_add_epi16( base, inc64 );
    }
    return static_cast<std::size_t>( out - start );
}

// AVX-512 VBMI2 set-bit decoder into 32-bit values: the same byte-index compress
// as the uint16 form, widened to 32 bits in blocks of 16 and masked-stored, so a
// store writes exactly the values it produced (an exactly-sized output buffer
// needs no padding) and only the ceil(popcount/16) blocks that carry a value are
// computed — a bitset container is usually well under half full.
// [croaring-ref] deps/croaring/src/bitset_util.c:bitset_extract_setbits_avx512
FRSR_ROARING_X86_V4_KERNEL
inline std::uint32_t * extract_setbits_uint32(
    std::uint64_t const * const words, std::size_t const n,
    std::uint32_t * out, std::uint32_t * const out_end, std::uint32_t const base
) noexcept {
    alignas( 64 ) static constexpr std::uint8_t index_table[ 64 ]{
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };
    __m512i       base_lanes{ _mm512_set1_epi32( static_cast<int>( base ) ) };
    __m512i const inc64     { _mm512_set1_epi32( 64 ) };
    __m512i const indices   { _mm512_load_si512( index_table ) };
    std::size_t index{ 0 };
    for ( ; index < n; ++index ) {
        auto const word{ words[ index ] };
        if ( word != 0 ) {
            auto const count{ static_cast<std::size_t>( std::popcount( word ) ) };
            if ( out + count > out_end ) { break; }
            __m512i const packed{ _mm512_maskz_compress_epi8( static_cast<__mmask64>( word ), indices ) };
            std::uint64_t const lanes{ ~std::uint64_t{ 0 } >> ( 64U - count ) };  // count is 1..64 here
            _mm512_mask_storeu_epi32( out, static_cast<__mmask16>( lanes ),
                _mm512_add_epi32( base_lanes, _mm512_cvtepu8_epi32( _mm512_castsi512_si128( packed ) ) ) );
            if ( count > 16 ) {
                _mm512_mask_storeu_epi32( out + 16, static_cast<__mmask16>( lanes >> 16 ),
                    _mm512_add_epi32( base_lanes, _mm512_cvtepu8_epi32( _mm512_extracti32x4_epi32( packed, 1 ) ) ) );
                if ( count > 32 ) {
                    _mm512_mask_storeu_epi32( out + 32, static_cast<__mmask16>( lanes >> 32 ),
                        _mm512_add_epi32( base_lanes, _mm512_cvtepu8_epi32( _mm512_extracti32x4_epi32( packed, 2 ) ) ) );
                    if ( count > 48 ) {
                        _mm512_mask_storeu_epi32( out + 48, static_cast<__mmask16>( lanes >> 48 ),
                            _mm512_add_epi32( base_lanes, _mm512_cvtepu8_epi32( _mm512_extracti32x4_epi32( packed, 3 ) ) ) );
                    }
                }
            }
            out += count;
        }
        base_lanes = _mm512_add_epi32( base_lanes, inc64 );
    }
    // Tail: whatever the capacity guard above stopped short of.
    auto value_base{ base + static_cast<std::uint32_t>( index * 64 ) };
    for ( ; index < n && out < out_end; ++index, value_base += 64 ) {
        auto word{ words[ index ] };
        while ( word != 0 && out < out_end ) {
            *out++ = value_base + static_cast<std::uint32_t>( std::countr_zero( word ) );
            word &= word - 1;
        }
    }
    return out;
}

} // namespace x86_v4
#endif // FRSR_ROARING_X86_V4

// Set-bit positions of one byte as eight 32-bit lanes, zero padded past the count —
// the uint32 twin of setbit_decode_table_uint16, kept 0-based (CRoaring's is 1-based
// with a base-1 accumulator).
// [croaring-ref] deps/croaring/src/bitset_util.c:vecDecodeTable
inline constexpr auto setbit_decode_table_uint32{ []() {
    std::array<std::array<std::uint32_t, 8>, 256> table{};
    for ( unsigned value{ 0 }; value < 256U; ++value ) {
        unsigned pos{ 0 };
        for ( unsigned bit{ 0 }; bit < 8U; ++bit ) {
            if ( value & ( 1U << bit ) ) { table[ value ][ pos++ ] = bit; }
        }
    }
    return table;
}() };

// Eight uint32 lanes as one wide gnu vector: clang lowers a load/add/store triple to
// one AVX2 instruction each, two SSE ones, two NEON ones — no per-ISA duplication.
typedef std::uint32_t setbit_decode_lane_block
    __attribute__(( vector_size( 8 * sizeof( std::uint32_t ) ), aligned( 1 ), may_alias ));

// Writes the positions of the set bits of `words`, offset by `base`, as uint32 values
// into [out, out_end) and returns the end pointer. Byte-table driven: one vector load,
// add and store per byte, advanced by that byte's popcount, so a set bit costs a
// fraction of the tzcnt+store chain a scalar decode pays per value. Every byte stores
// a full 8-lane vector, so the vector loop stops 64 values short of `out_end` and a
// scalar tail finishes — which keeps an exactly-sized output buffer safe.
// [croaring-ref] deps/croaring/src/bitset_util.c:bitset_extract_setbits_avx2
[[ gnu::hot ]] inline std::uint32_t * extract_setbits_uint32(
    std::uint64_t const * const words, std::size_t const n,
    std::uint32_t * out, std::uint32_t * const out_end, std::uint32_t const base
) noexcept {
#if FRSR_ROARING_X86_V4
    if ( have_x86_v4() ) [[likely]] {
        return x86_v4::extract_setbits_uint32( words, n, out, out_end, base );
    }
#endif
    setbit_decode_lane_block base_lanes;
    for ( unsigned lane{ 0 }; lane < 8U; ++lane ) { base_lanes[ lane ] = base; }
    setbit_decode_lane_block const eight{ 8, 8, 8, 8, 8, 8, 8, 8 };
    std::size_t index{ 0 };
    for ( ; index < n && out + 64 <= out_end; ++index ) {
        auto word{ words[ index ] };
        if ( word == 0 ) {
            for ( unsigned lane{ 0 }; lane < 8U; ++lane ) { base_lanes[ lane ] += 64; }
            continue;
        }
        for ( unsigned byte_index{ 0 }; byte_index < 8U; ++byte_index ) {
            auto const byte{ static_cast<std::uint8_t>( word >> ( 8U * byte_index ) ) };
            auto const positions{ *reinterpret_cast<setbit_decode_lane_block const *>( setbit_decode_table_uint32[ byte ].data() ) };
            *reinterpret_cast<setbit_decode_lane_block *>( out ) = base_lanes + positions;
            out       += std::popcount( static_cast<unsigned>( byte ) );
            base_lanes = base_lanes + eight;
        }
    }
    // Scalar tail: the last words, and any layout whose word count is small.
    auto value_base{ base + static_cast<std::uint32_t>( index * 64 ) };
    for ( ; index < n && out < out_end; ++index, value_base += 64 ) {
        auto word{ words[ index ] };
        while ( word != 0 && out < out_end ) {
            *out++ = value_base + static_cast<std::uint32_t>( std::countr_zero( word ) );
            word &= word - 1;
        }
    }
    return out;
}

// Extract a low-cardinality bitset result into a fresh array handle.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> array_from_bitset(
    bitset_cref<Layout, CowPolicy> const bitset,
    std::uint32_t const cardinality
) {
    container_handle<Layout, CowPolicy> result;
    auto array{ result.as_array() };
    auto const & words{ bitset.words.as_array() };
#if defined( __x86_64__ ) || defined( _M_X64 )
    if constexpr ( std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
#if FRSR_ROARING_X86_V4
        if ( cardinality >= x86_v4::extract_setbits_min_cardinality && have_x86_v4() ) {
            resize_uninitialized( array.values, cardinality );
            auto const decoded{ x86_v4::extract_setbits_uint16( words.data(), words.size(), array.values.data() ) };
            assert( decoded == cardinality );
            array.values.resize( static_cast<std::uint32_t>( decoded ) );
            array.sync_header();
            return result;
        }
#endif
        resize_uninitialized( array.values, cardinality + 16U );
        auto const decoded{ extract_setbits_sse_uint16( words.data(), words.size(), array.values.data() ) };
        assert( decoded == cardinality );
        array.values.resize( static_cast<std::uint32_t>( decoded ) );
        array.sync_header();
        return result;
    }
#endif
    resize_uninitialized( array.values, cardinality );
    auto * out{ array.values.data() };
    for ( std::size_t word_index{ 0 }; word_index < words.size(); ++word_index ) {
        auto word{ words[ word_index ] };
        auto const base{ static_cast<std::uint32_t>( word_index ) << 6U };
        while ( word != 0 ) {
            *out++ = static_cast<typename Layout::low_type>( base + static_cast<std::uint32_t>( std::countr_zero( word ) ) );
            word &= word - 1;
        }
    }
    array.sync_header();
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_variant<Layout, CowPolicy> combine_bitset_bitset(
    bitset_cref<Layout, CowPolicy> const lhs,
    bitset_cref<Layout, CowPolicy> const rhs,
    set_operation const op,
    container_handle<Layout, CowPolicy> && reuse = {}
) noexcept {
    // Every word of the result is written below, so skip the zero-init. A
    // retired scratch bitset (chunk_store::take_retired) is adopted instead of
    // allocating a fresh 8 KB block — the scratch-reuse hot path.
    auto result{ reuse.holds_bitset()
        ? std::move( reuse )
        : container_handle<Layout, CowPolicy>::make_bitset_uninitialized() };
    auto result_bitset{ result.as_bitset() };
    auto const & a{ lhs.words.as_array() };
    auto const & b{ rhs.words.as_array() };
    auto       & out{ result_bitset.words.as_array() };
    auto const n{ a.size() };
    // Hoist the operation out of the loop so each variant is a branchless,
    // auto-vectorizable word kernel, with the popcount FUSED into each loop
    // (clang vectorizes the OP store + Mula popcount reduction together —
    // same single-pass shape that won on the in-place combine below).
    // [croaring-ref] deps/croaring/src/containers/bitset.c:bitset_container_{or,and,andnot}
#if FRSR_ROARING_X86_V4
    if ( have_x86_v4() ) [[likely]] {
        auto const v4_cardinality{ combine_words_into_popcount_v4( out.data(), a.data(), b.data(), n, op ) };
        if ( v4_cardinality == 0 ) {
            return container_handle<Layout, CowPolicy>{};
        }
        result.set_cardinality( static_cast<std::uint32_t>( v4_cardinality ) );
        return result;
    }
#endif
    std::size_t cardinality{ 0 };
    switch ( op ) {
        case set_operation::bit_or:
            for ( auto i{ 0U }; i < n; ++i ) {
                out[ i ] = a[ i ] | b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) );
            }
            break;
        case set_operation::bit_and:
            for ( auto i{ 0U }; i < n; ++i ) {
                out[ i ] = a[ i ] & b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) );
            }
            break;
        case set_operation::bit_andnot:
            for ( auto i{ 0U }; i < n; ++i ) {
                out[ i ] = a[ i ] & ~b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( out[ i ] ) );
            }
            break;
    }
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    return result;
}

// run∩bitset written directly into a (retired or fresh) bitset handle — the
// dedicated combine-spine arm's analog of combine_bitset_bitset's reuse
// contract. The generic path builds a zero-initialized stack word_array, ORs
// masked source words into it per run, then pays make_bitset_container_from_words
// (fresh 8 KB alloc + copy + separate popcount pass + visit dispatch upstream);
// here the masked words land in the result payload directly with the popcount
// fused into the fill. Runs are disjoint, so per-(run,word) popcounts of the
// masked increments sum exactly even when adjacent runs share a boundary word.
// The result stays a bitset whatever its cardinality: demotion is the combine
// spine's decision (bitmap.hpp), which owns the chunk_store and can retire the
// vacated payload for scratch reuse — demoting inside a kernel destroys the
// adopted retired bitset and measured up to +14% on the fold-heavy workload.
// The spine demotes ANDNOT results and deliberately does not demote AND ones
// (an AND fold's accumulator loses its dense arms for every later pair: ~12%).
// [croaring-ref] deps/croaring/src/containers/mixed_intersection.c:
// run_bitset_container_intersection (dense-run branch)
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> intersect_run_bitset(
    run_cref<Layout, CowPolicy> const runs,
    bitset_cref<Layout, CowPolicy> const bitset,
    container_handle<Layout, CowPolicy> && reuse = {}
) noexcept {
    auto result{ reuse.holds_bitset()
        ? std::move( reuse )
        : container_handle<Layout, CowPolicy>::make_bitset_uninitialized() };
    auto result_bitset{ result.as_bitset() };
    auto const & src{ bitset.words.as_array() };
    auto       & out{ result_bitset.words.as_array() };
    // Zero only the inter-run gaps instead of a full upfront 8 KB fill: runs are
    // sorted and disjoint, so each word is either covered by a run (written with
    // the masked source word — OR-merged only at a boundary word shared with the
    // previous run) or lies in a gap (zeroed here or in the tail fill below).
    // With many short runs the per-gap fills lose to one bulk memset (measured:
    // 64 scattered 32-value runs regress ~40% under pure gap-zeroing while the
    // few-long-runs shape wins ~17%), so fall back to the upfront fill there —
    // the per-run write logic below is correct under either pre-state.
    bool const many_short_runs{ runs.runs.size() > 32U };
    if ( many_short_runs ) {
        out.fill( 0 );
    }
    std::size_t cardinality{ 0 };
    std::size_t next_unwritten{ 0 }; // first word index not yet written
    for ( auto const & current : runs.runs ) {
        auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
        auto const last_word { static_cast<std::size_t>( current.end   ) >> 6U };
        auto const first_bit { static_cast<unsigned>( current.begin ) & 63U };
        auto const last_bit  { static_cast<unsigned>( current.end   ) & 63U };
        if ( !many_short_runs && first_word > next_unwritten ) {
            std::fill( &out[ next_unwritten ], &out[ first_word ], std::uint64_t{ 0 } );
        }
        auto const first_mask{ std::numeric_limits<std::uint64_t>::max() << first_bit };
        auto const last_mask { ( last_bit == 63U )
            ? std::numeric_limits<std::uint64_t>::max()
            : ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U };
        if ( first_word == last_word ) {
            auto const masked{ src[ first_word ] & first_mask & last_mask };
            if ( first_word < next_unwritten ) {
                out[ first_word ] |= masked; // boundary word shared with the previous run
            } else {
                out[ first_word ] = masked;
            }
            cardinality += static_cast<std::size_t>( std::popcount( masked ) );
        } else {
            auto const first_masked{ src[ first_word ] & first_mask };
            if ( first_word < next_unwritten ) {
                out[ first_word ] |= first_masked;
            } else {
                out[ first_word ] = first_masked;
            }
            cardinality += static_cast<std::size_t>( std::popcount( first_masked ) );
            // Interior words are fully covered by the run — a straight copy with the
            // popcount fused (branch- and mask-free, so the loop auto-vectorizes).
            for ( auto word_index{ first_word + 1U }; word_index < last_word; ++word_index ) {
                auto const word{ src[ word_index ] };
                out[ word_index ] = word;
                cardinality += static_cast<std::size_t>( std::popcount( word ) );
            }
            auto const last_masked{ src[ last_word ] & last_mask };
            out[ last_word ] = last_masked;
            cardinality += static_cast<std::size_t>( std::popcount( last_masked ) );
        }
        next_unwritten = last_word + 1U;
    }
    if ( !many_short_runs && next_unwritten < out.size() ) {
        std::fill( &out[ next_unwritten ], out.data() + out.size(), std::uint64_t{ 0 } );
    }
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    return result;
}

// Near-full run∩bitset: clone the bitset payload wholesale and clear only the
// inter-run gaps, subtracting each cleared word's popcount from the bitset's
// known cardinality — no per-run masked rewrite and no full-block popcount.
// Wins over the masked-fill sibling above when the runs cover almost the whole
// domain (the fill degenerates to copying every word anyway, but word-by-word
// with a fused scalar popcount, where this variant pays one straight memcpy plus
// work proportional to the gap volume only). Callers gate on run coverage.
// [croaring-ref] deps/croaring/src/containers/mixed_intersection.c:
// run_bitset_container_intersection (clone + bitset_reset_range per gap, then a
// full recount — the cardinality delta here replaces that recount)
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> intersect_run_bitset_dense_runs(
    run_cref<Layout, CowPolicy> const runs,
    bitset_cref<Layout, CowPolicy> const bitset,
    std::size_t const bitset_cardinality,
    container_handle<Layout, CowPolicy> && reuse = {}
) noexcept {
    auto result{ reuse.holds_bitset()
        ? std::move( reuse )
        : container_handle<Layout, CowPolicy>::make_bitset_uninitialized() };
    auto result_bitset{ result.as_bitset() };
    auto const & src{ bitset.words.as_array() };
    auto       & out{ result_bitset.words.as_array() };
    std::copy( src.data(), src.data() + src.size(), out.data() );
    std::size_t removed{ 0 };
    auto const clear_bits{ [ & ]( std::size_t const begin, std::size_t const end ) { // inclusive
        auto const first_word{ begin >> 6U };
        auto const last_word { end   >> 6U };
        auto const first_mask{ std::numeric_limits<std::uint64_t>::max() << ( begin & 63U ) };
        auto const last_bit  { static_cast<unsigned>( end & 63U ) };
        auto const last_mask { ( last_bit == 63U )
            ? std::numeric_limits<std::uint64_t>::max()
            : ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U };
        if ( first_word == last_word ) {
            auto const mask{ first_mask & last_mask };
            removed += static_cast<std::size_t>( std::popcount( out[ first_word ] & mask ) );
            out[ first_word ] &= ~mask;
        } else {
            removed += static_cast<std::size_t>( std::popcount( out[ first_word ] & first_mask ) );
            out[ first_word ] &= ~first_mask;
            for ( auto word_index{ first_word + 1U }; word_index < last_word; ++word_index ) {
                removed += static_cast<std::size_t>( std::popcount( out[ word_index ] ) );
                out[ word_index ] = 0;
            }
            removed += static_cast<std::size_t>( std::popcount( out[ last_word ] & last_mask ) );
            out[ last_word ] &= ~last_mask;
        }
    } };
    std::size_t next{ 0 }; // first value not yet covered by a processed run
    for ( auto const & current : runs.runs ) {
        auto const begin{ static_cast<std::size_t>( current.begin ) };
        if ( begin > next ) {
            clear_bits( next, begin - 1U );
        }
        next = static_cast<std::size_t>( current.end ) + 1U;
    }
    if ( next < Layout::low_domain_size ) {
        clear_bits( next, Layout::low_domain_size - 1U );
    }
    auto const cardinality{ bitset_cardinality - removed };
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    return result;
}

// Sparse run∩bitset extracted directly into an array handle: only the words the
// runs cover are read (masked at run boundaries) and their set bits emitted in
// order — the bitset-materializing sibling above touches all Layout::word_count
// words (gap zeroing / tail fill) and pays a full-block popcount even when the
// runs cover a handful of words. Callers gate on the run cardinality (result ⊆
// the runs, so |result| <= run cardinality < the array threshold ⇒ the array
// form is always legal). Runs are disjoint and sorted, so emission order is
// ascending even when adjacent runs share a boundary word (their masks don't
// overlap). Ported from a downstream engine's sparse run∩bitset kernel, whose
// HW instruction counts on the production fold ran ~an order of magnitude
// below the full-block fill on sparse-run shapes.
// [croaring-ref] deps/croaring/src/containers/mixed_intersection.c:
// run_bitset_container_intersection (sparse-run branch)
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> intersect_run_bitset_sparse(
    run_cref<Layout, CowPolicy> const runs,
    bitset_cref<Layout, CowPolicy> const bitset,
    std::uint32_t const max_cardinality,   // min(run, bitset cardinality) — result upper bound
    container_handle<Layout, CowPolicy> && reuse = {}
) {
    using low_type = typename Layout::low_type;
    auto result{ reuse.holds_array() ? std::move( reuse ) : container_handle<Layout, CowPolicy>{} };
    auto result_array{ result.as_array() };
    resize_uninitialized( result_array.values, max_cardinality );
    auto * const out{ result_array.values.data() };
    std::size_t written{ 0 };
    auto const & src{ bitset.words.as_array() };
    auto const emit_word{ [ & ]( std::uint64_t word, std::uint32_t const base ) {
        while ( word != 0 ) {
            out[ written++ ] = static_cast<low_type>( base + static_cast<std::uint32_t>( std::countr_zero( word ) ) );
            word &= word - 1;
        }
    } };
    for ( auto const & current : runs.runs ) {
        auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
        auto const last_word { static_cast<std::size_t>( current.end   ) >> 6U };
        auto const first_bit { static_cast<unsigned>( current.begin ) & 63U };
        auto const last_bit  { static_cast<unsigned>( current.end   ) & 63U };
        auto const first_mask{ std::numeric_limits<std::uint64_t>::max() << first_bit };
        auto const last_mask { ( last_bit == 63U )
            ? std::numeric_limits<std::uint64_t>::max()
            : ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U };
        if ( first_word == last_word ) {
            emit_word( src[ first_word ] & first_mask & last_mask, static_cast<std::uint32_t>( first_word ) << 6U );
        } else {
            emit_word( src[ first_word ] & first_mask, static_cast<std::uint32_t>( first_word ) << 6U );
            for ( auto word_index{ first_word + 1U }; word_index < last_word; ++word_index ) {
                emit_word( src[ word_index ], static_cast<std::uint32_t>( word_index ) << 6U );
            }
            emit_word( src[ last_word ] & last_mask, static_cast<std::uint32_t>( last_word ) << 6U );
        }
    }
    if ( written == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    resize_uninitialized( result_array.values, static_cast<std::uint32_t>( written ) );
    result_array.sync_header();
    return result;
}

// In-place bitset\array: clear each array value's bit from the LHS bitset's own
// 8 KB block, tracking the removed count for an exact cardinality update — no
// allocation, no result construction, result stays a bitset (representation
// re-decision deferred to optimize(), as elsewhere in the spine).
// [croaring-ref] deps/croaring/src/containers/mixed_andnot.c:
// bitset_array_container_iandnot
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::hot ]] inline void difference_bitset_array_inplace(
    bitset_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const values
) noexcept {
    auto & words{ lhs.words.as_array() };
    std::size_t removed{ 0 };
    for ( auto const value : values.values ) {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const mask{ std::uint64_t{ 1 } << ( static_cast<unsigned>( value ) & 63U ) };
        removed += static_cast<std::size_t>( ( words[ word_index ] & mask ) != 0 );
        words[ word_index ] &= ~mask;
    }
    lhs.cardinality -= static_cast<std::uint32_t>( removed );
    lhs.mark_endpoints_stale();
}

// In-place bitset×bitset combine: lhs.words OP= rhs.words, cardinality recomputed.
// Mirrors CRoaring's roaring_bitmap_{or,and,andnot}_inplace bitset×bitset case —
// mutates the LHS's existing 8 KB block (no allocation, no result construction),
// which is the dominant win for the *_inplace operators and pairwise N-way unions.
// The popcount is FUSED into each op loop so the 8 KB block is read+written once
// instead of re-read by a separate cardinality pass (CRoaring computes cardinality
// in the same pass via harley-seal; here clang vectorizes the OP store + the
// popcount reduction together). The op switch stays outside the loop so each
// variant is still a branchless, auto-vectorizable word kernel.
// [croaring-ref] deps/croaring/src/containers/bitset.c:bitset_container_{or,and,andnot}

// Defined below (uses bitset_reg / the Harley-Seal kernel from hw_info.hpp);
// forward-declared here for combine_bitset_bitset_inplace's dispatch.
template <typename Layout, set_operation Op>
std::size_t fused_combine_inplace_popcount(
    std::uint64_t * lhs_words, std::uint64_t const * rhs_words
) noexcept;

template <typename Layout, typename CowPolicy = cow_value_semantics>
void combine_bitset_bitset_inplace(
    bitset_ref<Layout, CowPolicy> lhs,
    bitset_cref<Layout, CowPolicy> const rhs,
    set_operation const op
) noexcept {
    // [croaring-ref] roaring_bitmap_or_inplace skips the combine when the LHS
    // container is already saturated — OR-ing into a full bitset is a no-op and its
    // cardinality is unchanged. The cardinality is exact on the in-place operators,
    // so this is an O(1) test that elides the 8 KB read+write+popcount entirely. A
    // fully populated chunk dominates the cost on dense operands; this shortcut is
    // also why CRoaring's in-place OR beats its in-place AND (which has no analog).
    if ( op == set_operation::bit_or && lhs.cardinality == Layout::word_count * 64U ) {
        return;
    }
    if constexpr ( kFusedHarleySealPopcount ) {
        auto       * const a{ lhs.words.as_array().data() };
        auto const * const b{ rhs.words.as_array().data() };
        switch ( op ) {
            case set_operation::bit_or:
                lhs.cardinality = static_cast<std::uint32_t>( fused_combine_inplace_popcount<Layout, set_operation::bit_or    >( a, b ) );
                break;
            case set_operation::bit_and:
                lhs.cardinality = static_cast<std::uint32_t>( fused_combine_inplace_popcount<Layout, set_operation::bit_and   >( a, b ) );
                break;
            case set_operation::bit_andnot:
                lhs.cardinality = static_cast<std::uint32_t>( fused_combine_inplace_popcount<Layout, set_operation::bit_andnot >( a, b ) );
                break;
        }
        lhs.mark_endpoints_stale();
        return;
    }
    auto       & a{ lhs.words.as_array() };
    auto const & b{ rhs.words.as_array() };
    auto const n{ a.size() };
#if FRSR_ROARING_X86_V4
    // Dispatched like the materializing combine. An earlier attempt (2026-07-19)
    // measured a ~4% in-context loss on the bitset-heavy workload from outlining
    // this spine-inlined step; re-measured now that the workload runs on AVX-512
    // silicon, where the baseline tile loop is the 2x-slower half of every
    // in-place bitset union.
    if ( have_x86_v4() ) [[likely]] {
        lhs.cardinality = static_cast<std::uint32_t>( combine_words_inplace_popcount_v4( a.data(), b.data(), n, op ) );
        lhs.mark_endpoints_stale();
        return;
    }
#endif
    std::size_t cardinality{ 0 };
    switch ( op ) {
        case set_operation::bit_or:
            for ( auto i{ 0U }; i < n; ++i ) {
                a[ i ] |= b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) );
            }
            break;
        case set_operation::bit_and:
            for ( auto i{ 0U }; i < n; ++i ) {
                a[ i ] &= b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) );
            }
            break;
        case set_operation::bit_andnot:
            for ( auto i{ 0U }; i < n; ++i ) {
                a[ i ] &= ~b[ i ];
                cardinality += static_cast<std::size_t>( std::popcount( a[ i ] ) );
            }
            break;
    }
    lhs.cardinality = static_cast<std::uint32_t>( cardinality );
    lhs.mark_endpoints_stale();
}

// Fused in-place bitwise-combine + Harley-Seal population count: lhs[i] OP= rhs[i] for
// every word, returning the result cardinality in ONE pass over the 8 KB block. The
// carry-save tree folds 16 SIMD registers' worth of op-results into weighted counters
// (ones/twos/.../sixteens) so a single vector popcount runs per 16 registers — the
// instruction-count reduction CRoaring uses. Gated behind kFusedHarleySealPopcount
// (off): faster in isolation, slower wired into the library on this target — see the
// flag's comment. rhs is a distinct container from lhs, so it is restrict-qualified.
// [croaring-ref] deps/croaring/include/roaring/bitset_util.h:avx2_harley_seal_popcount256andstore
template <typename Layout, set_operation Op>
[[ gnu::noinline, gnu::hot ]] std::size_t fused_combine_inplace_popcount(
    std::uint64_t * lhs_words, std::uint64_t const * rhs_words
) noexcept {
    constexpr std::size_t lanes{ sizeof( bitset_reg ) / sizeof( std::uint64_t ) };
    constexpr std::size_t n    { Layout::word_count };
    constexpr std::size_t regs { n / lanes };
    bitset_reg       *            const a{ reinterpret_cast<bitset_reg       *>( lhs_words ) };
    bitset_reg const * __restrict const b{ reinterpret_cast<bitset_reg const *>( rhs_words ) };

    bitset_reg total{}, ones{}, twos{}, fours{}, eights{}, sixteens{};
    bitset_reg twosA, twosB, foursA, foursB, eightsA, eightsB;

    // Apply the op to register i, store it back in place, and yield it for the tree.
    auto const combine{ [ & ]( std::size_t const i ) -> bitset_reg {
        bitset_reg const r{ apply_bitwise_op<Op>( a[ i ], b[ i ] ) };
        a[ i ] = r;
        return r;
    } };

    constexpr std::size_t limit{ regs - regs % 16 };
    std::size_t i{ 0 };
    for ( ; i < limit; i += 16 ) {
        bitset_reg A1, A2;
        A1 = combine( i +  0 ); A2 = combine( i +  1 ); carry_save_add( twosA, ones, ones, A1, A2 );
        A1 = combine( i +  2 ); A2 = combine( i +  3 ); carry_save_add( twosB, ones, ones, A1, A2 );
        carry_save_add( foursA, twos, twos, twosA, twosB );
        A1 = combine( i +  4 ); A2 = combine( i +  5 ); carry_save_add( twosA, ones, ones, A1, A2 );
        A1 = combine( i +  6 ); A2 = combine( i +  7 ); carry_save_add( twosB, ones, ones, A1, A2 );
        carry_save_add( foursB,  twos,   twos,   twosA,  twosB  );
        carry_save_add( eightsA, fours,  fours,  foursA, foursB );
        A1 = combine( i +  8 ); A2 = combine( i +  9 ); carry_save_add( twosA, ones, ones, A1, A2 );
        A1 = combine( i + 10 ); A2 = combine( i + 11 ); carry_save_add( twosB, ones, ones, A1, A2 );
        carry_save_add( foursA, twos, twos, twosA, twosB );
        A1 = combine( i + 12 ); A2 = combine( i + 13 ); carry_save_add( twosA, ones, ones, A1, A2 );
        A1 = combine( i + 14 ); A2 = combine( i + 15 ); carry_save_add( twosB, ones, ones, A1, A2 );
        carry_save_add( foursB,   twos,   twos,   twosA,   twosB   );
        carry_save_add( eightsB,  fours,  fours,  foursA,  foursB  );
        carry_save_add( sixteens, eights, eights, eightsA, eightsB );
        total += __builtin_elementwise_popcount( sixteens );
    }
    total <<= 4;
    total += __builtin_elementwise_popcount( eights ) << 3;
    total += __builtin_elementwise_popcount( fours  ) << 2;
    total += __builtin_elementwise_popcount( twos   ) << 1;
    total += __builtin_elementwise_popcount( ones   );
    for ( ; i < regs; ++i ) {
        total += __builtin_elementwise_popcount( combine( i ) );
    }
    std::size_t cardinality{ static_cast<std::size_t>( __builtin_reduce_add( total ) ) };
    // Scalar tail for layouts whose word_count is not a multiple of the register width.
    for ( std::size_t w{ regs * lanes }; w < n; ++w ) {
        std::uint64_t const r{ apply_bitwise_op<Op>( lhs_words[ w ], rhs_words[ w ] ) };
        lhs_words[ w ] = r;
        cardinality += static_cast<std::size_t>( std::popcount( r ) );
    }
    return cardinality;
}

// Lazy in-place bitset OR: lhs.words |= rhs.words, cardinality left STALE. Used by
// the N-way bulk-union path (or_many_in_place / bulk_or_*) so the popcount runs
// once at the end (bitmap::repair_cardinality) instead of K times, mirroring
// CRoaring's lazy-OR + repair. Standalone callers must call repair afterwards.
// The wide-vector tile lowers to kernel_tile_vectors native OR instructions per
// iteration (CRoaring hand-unrolls 8×; clang auto-vectorized only 4×). A manual
// wide loop also drops clang's runtime aliasing check + scalar fallback, and stays
// correct under self-union (a==b) since the OR is elementwise.
// [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_lazy_or_inplace
template <typename Layout, typename CowPolicy = cow_value_semantics>
void or_bitset_bitset_inplace_lazy(
    bitset_ref<Layout, CowPolicy> lhs,
    bitset_cref<Layout, CowPolicy> const rhs
) noexcept {
    auto       & a{ lhs.words.as_array() };
    auto const & b{ rhs.words.as_array() };
    constexpr std::size_t n{ Layout::word_count };
    constexpr std::size_t lanes{ hw_info::bitset_tile_words };
    constexpr std::size_t tiles{ n / lanes };
#if FRSR_ROARING_X86_V4
    if ( have_x86_v4() ) [[likely]] {
        or_words_inplace_v4( a.data(), b.data(), n );
        lhs.mark_cardinality_stale();
        lhs.mark_endpoints_stale();
        return;
    }
#endif
    auto       * wa{ reinterpret_cast<bitset_word_tile       *>( a.data() ) };
    auto const * wb{ reinterpret_cast<bitset_word_tile const *>( b.data() ) };
    for ( std::size_t i{ 0 }; i < tiles; ++i ) {
        wa[ i ] |= wb[ i ];
    }
    // Tail for layouts whose word_count is not a multiple of the tile (small bitsets).
    for ( std::size_t i{ tiles * lanes }; i < n; ++i ) {
        a[ i ] |= b[ i ];
    }
    lhs.mark_cardinality_stale();
    lhs.mark_endpoints_stale();
}

// Bulk-union bitset×bitset step with live cardinality (CRoaring's
// bitset_container_or under LAZY_OR_BITSET_CONVERSION_TO_FULL, minus the
// bitset→full-run conversion). Keeping cardinality exact lets container_is_known_full
// fire on later folds once a chunk saturates. noinline: inlining this into the
// bulk_or spine previously MidSkew-regressed via i-cache (measured 1.14).
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::noinline ]] void or_bitset_bitset_inplace_bulk(
    bitset_ref<Layout, CowPolicy> lhs,
    bitset_cref<Layout, CowPolicy> const rhs
) noexcept {
#if FRSR_ROARING_X86_V4
    // Already out of line, so — unlike the spine-inlined in-place combine — the
    // dispatched kernel costs no extra call here. Same saturated-LHS shortcut as
    // combine_bitset_bitset_inplace.
    if ( have_x86_v4() ) [[likely]] {
        if ( lhs.cardinality == Layout::word_count * 64U ) {
            return;
        }
        auto       * const a{ lhs.words.as_array().data() };
        auto const * const b{ rhs.words.as_array().data() };
        lhs.cardinality = static_cast<std::uint32_t>( combine_words_inplace_popcount_v4( a, b, Layout::word_count, set_operation::bit_or ) );
        lhs.mark_endpoints_stale();
        return;
    }
#endif
    combine_bitset_bitset_inplace<Layout, CowPolicy>( lhs, rhs, set_operation::bit_or );
}

// Span-to-count ratio at or below which the scatter switches to the word-grouped
// loop. The per-value read-modify-write costs a store-to-load forwarding round
// trip whenever consecutive values land in the SAME word — worst case a
// contiguous run, which hits one word 64 times in a row. Grouping them into a
// register-held mask makes that one store per word instead of 64, at the price of
// a data-dependent inner exit that mispredicts once the groups get short.
// Measured crossover (2048 values, x86-64-v3): grouped wins from 64 values/word
// (2.6x) down to ~4 values/word, and loses below 2. 16 keeps the switch inside
// the region where it clearly wins.
inline constexpr std::size_t kScatterGroupingSpanRatio{ 16 };

// Lazy in-place scatter of an array into a bitset: set each array value's bit in
// lhs.words directly, cardinality left STALE. The mixed-container analog of
// or_bitset_bitset_inplace_lazy for the N-way bulk-union path — it avoids the 8 KB
// copy + fresh allocation + popcount a materialized bitset×array combine would
// incur. Caller must repair cardinality afterwards.
// [croaring-ref] deps/croaring/src/containers/mixed_union.c:array_bitset_container_union
//
// noinline+cold: the word-grouped body grew this function enough that its
// presence in the shared header-only set-op spine shifted MidSkewIntersect
// (array∩array, never calls this) from ~0.31 to ~0.38 us — a pure
// i-cache/layout effect confirmed by fine-bisect (b88dd79 still 0.95×,
// 1ddde6e alone 1.30×). Keep the scatter out of line and in .text.unlikely
// so it does not sit next to the materializing AND merge.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[ gnu::noinline, gnu::cold ]] void or_array_into_bitset_inplace_lazy(
    bitset_ref<Layout, CowPolicy> lhs,
    array_cref<Layout, CowPolicy> const rhs
) noexcept {
    auto       & words { lhs.words.as_array() };
    auto const & values{ rhs.values };
    auto const   count { values.size() };
    if ( count == 0 ) [[unlikely]] {
        return;
    }
    // Array payloads are sorted, so the endpoints bound the span in O(1) — the
    // cheapest way to ask "does this array cluster?" without touching the middle.
    auto const span{ static_cast<std::size_t>( values[ count - 1 ] ) - values[ 0 ] + 1U };
    if ( span <= kScatterGroupingSpanRatio * count ) {
        std::size_t index{ 0 };
        do {
            auto const    word_index{ static_cast<std::size_t>( values[ index ] ) >> 6U };
            std::uint64_t mask{ 0 };
            do {
                mask |= std::uint64_t{ 1 } << ( static_cast<unsigned>( values[ index ] ) & 63U );
                ++index;
            } while ( ( index < count ) && ( ( static_cast<std::size_t>( values[ index ] ) >> 6U ) == word_index ) );
            words[ word_index ] |= mask;
        } while ( index < count );
        lhs.mark_cardinality_stale();
        lhs.mark_endpoints_stale();
        return;
    }
    for ( auto const value : values ) {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index { static_cast<unsigned>( value ) & 63U };
        words[ word_index ] |= ( std::uint64_t{ 1 } << bit_index );
    }
    lhs.mark_cardinality_stale();
    lhs.mark_endpoints_stale();
}

} // namespace frsr::roaring::detail
