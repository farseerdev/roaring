#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/hw_info.hpp>
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace frsr::roaring::detail {

// ---- x86 runtime dispatch (function multi-versioning) ----------------------
// The shipped baselines are AVX-512-less (-march=skylake class) while current
// server silicon has AVX-512 incl. VPOPCNTDQ, which roughly halves the 8 KB
// bitset word loops (512-bit ops + native vector popcount; measured: it erases
// the whole remaining frsr-vs-CRoaring gap on the bitset-heavy workload, and
// CRoaring itself runtime-dispatches its AVX-512 kernels). clang does not allow
// target attributes on templates, so the dispatchable kernels are plain
// functions over raw word spans; the templated entry points funnel into them
// when the CPU qualifies. Compiled out when the whole TU already targets
// AVX-512 (-march=native builds) or off x86.
//
// ⚠ Site selection is MEASURED, not uniform (bisected on an AVX-512 x86 server, 2026-07-19):
// only the MATERIALIZING combine (fused with its popcount) and the lazy bulk-OR
// carry dispatch. Dispatching the in-place combine — which inlines into the
// merge-walk spine — regressed the bitset-heavy workload ~4% in context despite
// winning its microbench (outlining cost inside the spine), and the
// repair_cardinality popcount dispatch measured neutral-to-negative. Don't add
// sites without an in-context full-scale A/B.
#if ( defined( __x86_64__ ) || defined( _M_X64 ) ) && defined( __clang__ ) && !defined( __AVX512VPOPCNTDQ__ )
#   define FRSR_ROARING_X86_V4_DISPATCH 1
#else
#   define FRSR_ROARING_X86_V4_DISPATCH 0
#endif

#if FRSR_ROARING_X86_V4_DISPATCH

// The dispatched kernels ask for AVX-512 F/BW/DQ/CD/VL (the x86-64-v4 set) PLUS
// VPOPCNTDQ — the vector popcount is where most of the win is, and every AVX-512
// CPU still in service that matters (Ice Lake+, Zen 4+) has it. Detection is
// hand-rolled cpuid/xgetbv (GNU inline asm — works under clang-cl too, where
// __builtin_cpu_supports lacks its libgcc-style runtime).
#define FRSR_ROARING_X86_V4_TARGET "avx512f,avx512bw,avx512dq,avx512cd,avx512vl,avx512vpopcntdq"

[[nodiscard]] inline bool have_x86_v4() noexcept {
    static bool const value{ []() noexcept {
        std::uint32_t eax, ebx, ecx, edx;
        __asm__ volatile ( "cpuid" : "=a"( eax ), "=b"( ebx ), "=c"( ecx ), "=d"( edx ) : "a"( 1U ), "c"( 0U ) );
        if ( !( ecx & ( 1U << 27 ) ) ) { return false; }  // OSXSAVE
        std::uint32_t xlo, xhi;
        __asm__ volatile ( "xgetbv" : "=a"( xlo ), "=d"( xhi ) : "c"( 0U ) );
        if ( ( xlo & 0xE6U ) != 0xE6U ) { return false; }  // XMM+YMM+opmask+ZMM state enabled by the OS
        __asm__ volatile ( "cpuid" : "=a"( eax ), "=b"( ebx ), "=c"( ecx ), "=d"( edx ) : "a"( 7U ), "c"( 0U ) );
        constexpr std::uint32_t ebx_need{ ( 1U << 16 ) | ( 1U << 17 ) | ( 1U << 28 ) | ( 1U << 30 ) | ( 1U << 31 ) };  // F, DQ, CD, BW, VL
        if ( ( ebx & ebx_need ) != ebx_need ) { return false; }
        return ( ecx & ( 1U << 14 ) ) != 0;  // VPOPCNTDQ
    }() };
    return value;
}

[[ using gnu: target( FRSR_ROARING_X86_V4_TARGET ), noinline, hot ]]
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

[[ using gnu: target( FRSR_ROARING_X86_V4_TARGET ), noinline, hot ]]
inline void or_words_inplace_v4( std::uint64_t * const a, std::uint64_t const * const b, std::size_t const n ) noexcept {
    for ( std::size_t i{ 0 }; i < n; ++i ) { a[ i ] |= b[ i ]; }
}

#endif // FRSR_ROARING_X86_V4_DISPATCH

// Cardinality below which a freshly-built bitset result is demoted to an array
// container. CRoaring demotes at its array/bitset boundary (DEFAULT_MAX_SIZE,
// 4096) so chained folds run array kernels instead of full 8 KB word loops on
// near-empty bitsets — deferring the re-decision to optimize() cost the
// fold-heavy witness ~4% (popcount passes on near-empty bitsets). But demoting
// right at the boundary ping-pongs on workloads whose fold results hover just
// under it unless the vacated 8 KB payload is retired for scratch reuse —
// demoting INSIDE the kernel destroyed the adopted retired bitset (breaking the
// chunk_store scratch cycle) and cost the fold-heavy witness up to +14%.
// Demotion therefore happens at the combine-spine call sites (bitmap.hpp),
// which own the chunk_store and retire the vacated bitset payload.
template <typename Layout>
inline constexpr std::uint32_t bitset_demote_threshold{
    Layout::low_domain_size >= 1024 ? Layout::low_domain_size / 16 : 64
};

// Extract a low-cardinality bitset result into a fresh array handle.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> array_from_bitset(
    bitset_cref<Layout, CowPolicy> const bitset,
    std::uint32_t const cardinality
) {
    container_handle<Layout, CowPolicy> result;
    auto array{ result.as_array() };
    resize_uninitialized( array.values, cardinality );
    auto * out{ array.values.data() };
    auto const & words{ bitset.words.as_array() };
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
#if FRSR_ROARING_X86_V4_DISPATCH
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
// A low-cardinality result is demoted to array (bitset_demote_threshold),
// matching CRoaring's mixed-intersection conversion.
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
    // NO v4 dispatch here (unlike the materializing combine below/above): this
    // in-place combine inlines into the merge-walk spine, and outlining it into
    // a dispatched call measurably regressed the bitset-heavy workload (~+4%)
    // while the microbench showed a win — in-context loss, bisected 2026-07-19.
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
#if FRSR_ROARING_X86_V4_DISPATCH
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
