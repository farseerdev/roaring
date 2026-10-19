#pragma once

#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/container_layout.hpp>
#include <frsr/roaring/run_container.hpp>
#include <frsr/roaring/operations.hpp>
#include <frsr/roaring/run_selection_policy.hpp>
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#ifndef FRSR_ROARING_HAS_PSI_VM
#   include <vector>
#else
#   include <psi/vm/containers/small_vector.hpp>
#endif

namespace frsr::roaring::detail {

#ifdef FRSR_ROARING_HAS_PSI_VM
// SBO-backed scratch storage mirroring the handle's inline capacity: value
// staging below array_sbo_size elements stays on the stack, larger scratch
// spills to the CRT heap.
template <typename T>
using small_array_values =
#if FRSR_ROARING_NO_SBO
    // Equal terms with the reference: no inline staging either.
    psi::vm::heap_vector<T, std::uint32_t>;
#else
    psi::vm::small_vector<T, array_sbo_size, std::uint32_t>;
#endif
#else
template <typename T>
using small_array_values = std::vector<T>;

template <typename T>
using heap_vector = std::vector<T>;
#endif

template <typename T>
inline void resize_uninitialized( small_array_values<T> & values, std::size_t const new_size ) {
#ifdef FRSR_ROARING_HAS_PSI_VM
    values.resize( static_cast<std::uint32_t>( new_size ), psi::vm::no_init );
#else
    values.resize( new_size );
#endif
}

#if defined( FRSR_ROARING_HAS_PSI_VM ) && !FRSR_ROARING_NO_SBO
// Distinct type from small_array_values only under psi::vm; without it both are
// std::vector and the overload above already covers it. Under FRSR_ROARING_NO_SBO
// small_array_values IS heap_vector, so the overload above covers it there too and
// defining this one would redeclare the same function.
template <typename T>
inline void resize_uninitialized( heap_vector<T> & values, std::size_t const new_size ) {
    values.resize( static_cast<std::uint32_t>( new_size ), psi::vm::no_init );
}
#endif

template <typename Layout, typename E, typename CowPolicy>
inline void resize_uninitialized( payload_vector<Layout, E, CowPolicy> & values, std::size_t const new_size ) {
    values.resize_uninitialized( static_cast<std::uint32_t>( new_size ) );
}

template <typename Layout>
using run_container = ::frsr::roaring::run_container<typename Layout::key_type>;

// The concrete container handle replaced std::variant<array, run, bitset>;
// the old name is kept as an alias so downstream spellings still read.
template <typename Layout, typename CowPolicy = cow_value_semantics>
using container_variant = container_handle<Layout, CowPolicy>;

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::size_t container_size( container_handle<Layout, CowPolicy> const & container ) {
    return container.cardinality();
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline bool container_contains(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::low_type const value
) noexcept {
    switch ( container.kind() ) {
        case container_kind::array : return container.as_array ().contains( value );
        case container_kind::run   : return container.as_run   ().contains( value );
        case container_kind::bitset: return container.as_bitset().contains( value );
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline bool container_add(
    container_handle<Layout, CowPolicy> & container,
    typename Layout::low_type const value
) {
    switch ( container.kind() ) {
        case container_kind::array : return container.as_array ().add( value );
        case container_kind::run   : return container.as_run   ().add( value );
        case container_kind::bitset: return container.as_bitset().add( value );
    }
    std::unreachable();
}

// container_add() for a container the caller has already made private in this
// write sequence: the kind switch stays, the copy-on-write barrier does not run.
// See as_array_already_private() for the precondition - a bulk_context that has
// added to this very slot since it resolved it establishes it.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline bool container_add_already_private(
    container_handle<Layout, CowPolicy> & container,
    typename Layout::low_type const value
) {
    switch ( container.kind() ) {
        case container_kind::array : return as_array_already_private ( container ).add( value );
        case container_kind::run   : return as_run_already_private   ( container ).add( value );
        case container_kind::bitset: return as_bitset_already_private( container ).add( value );
    }
    std::unreachable();
}

// The group form of container_add(): the kind switch and the copy-on-write
// barrier behind the mutable view are paid once for the whole sorted group,
// where a value-at-a-time loop pays both per value. `sorted_values` must be
// sorted and duplicate-free - bitmap::add_many_sorted's grouping collapses
// repeats as it builds a group, which is where such input comes from (add_many
// sorts its input but does not unique it). Returns how many values were added.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::size_t container_add_sorted(
    container_handle<Layout, CowPolicy> & container,
    std::span<typename Layout::low_type const> const sorted_values
) {
    switch ( container.kind() ) {
        case container_kind::array : return container.as_array ().add_sorted( sorted_values );
        case container_kind::run   : return container.as_run   ().add_sorted( sorted_values );
        case container_kind::bitset: return container.as_bitset().add_sorted( sorted_values );
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline bool container_remove(
    container_handle<Layout, CowPolicy> & container,
    typename Layout::low_type const value
) {
    switch ( container.kind() ) {
        case container_kind::array : return container.as_array ().remove( value );
        case container_kind::run   : return container.as_run   ().remove( value );
        case container_kind::bitset: return container.as_bitset().remove( value );
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::optional<typename Layout::low_type> container_first(
    container_handle<Layout, CowPolicy> const & container
) noexcept {
    return container.visit( []( auto const & current ) noexcept { return current.first(); } );
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::optional<typename Layout::low_type> container_next_after(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::low_type const value
) noexcept {
    return container.visit( [value]( auto const & current ) noexcept { return current.next_after( value ); } );
}

template <typename Layout, typename CowPolicy, typename F>
inline void container_for_each( container_handle<Layout, CowPolicy> const & container, F && f ) {
    container.visit( [&]( auto const & current ) { current.for_each( std::forward<F>( f ) ); } );
}

// Gives the container's payload slack back to the allocator, returning the bytes
// freed. A bitset payload is exactly one word_count block, so it has no slack.
// [croaring-ref] deps/croaring/include/roaring/containers/containers.h:container_shrink_to_fit
template <typename Layout, typename CowPolicy>
inline std::size_t container_shrink_to_fit( container_handle<Layout, CowPolicy> & container ) {
    switch ( container.kind() ) {
        case container_kind::array : return container.template shrink_payload_to_fit<typename Layout::low_type>();
        case container_kind::run   : return container.template shrink_payload_to_fit<::frsr::roaring::run<typename Layout::low_type>>();
        case container_kind::bitset: return 0;
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::size_t container_byte_size( container_handle<Layout, CowPolicy> const & container ) noexcept {
    switch ( container.kind() ) {
        case container_kind::array : return container.count() * sizeof( typename Layout::low_type );
        case container_kind::run   : return container.count() * sizeof( ::frsr::roaring::run<typename Layout::low_type> );
        case container_kind::bitset: return Layout::word_count * sizeof( std::uint64_t );
    }
    std::unreachable();
}

// Extracts every value of an arbitrary container into a fresh sorted array
// handle (the array analog of the former array_from_bitset / array_from_run).
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> array_handle_from_container( container_handle<Layout, CowPolicy> const & container ) {
    container_handle<Layout, CowPolicy> result;
    auto array{ result.as_array() };
    resize_uninitialized( array.values, container_size<Layout>( container ) );
    auto * out{ array.values.data() };
    container_for_each<Layout>( container, [&]( typename Layout::low_type const value ) { *out++ = value; } );
    array.sync_header();
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline heap_vector<typename Layout::low_type> sorted_values_from_container( container_handle<Layout, CowPolicy> const & container ) {
    heap_vector<typename Layout::low_type> result;
    result.reserve( static_cast<std::uint32_t>( container_size<Layout>( container ) ) );
    container_for_each<Layout>( container, [&]( typename Layout::low_type const value ) { result.push_back( value ); } );
    return result;
}

template <typename Layout>
[[nodiscard]] inline std::optional<run_container<Layout>> try_make_run_from_sorted_values_capped(
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const byte_limit
) {
    if ( sorted_values.empty() ) {
        return run_container<Layout>{};
    }

    run_container<Layout> result;
    auto current_begin{ sorted_values.front() };
    auto current_end{ current_begin };
    auto const push_run = [&]( typename Layout::low_type const begin, typename Layout::low_type const end ) -> bool {
        result.runs.push_back( run<typename Layout::low_type>{ begin, end } );
        if ( result.runs.size() * sizeof( typename run_container<Layout>::run ) >= byte_limit ) {
            return false;
        }
        result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( begin ) + 1U;
        return true;
    };

    for ( auto index{ std::size_t{ 1 } }; index < sorted_values.size(); ++index ) {
        auto const value{ sorted_values[ index ] };
        if (
            value == current_end ||
            ( current_end != std::numeric_limits<typename Layout::low_type>::max() &&
              value == static_cast<typename Layout::low_type>( current_end + 1U ) )
        ) {
            current_end = value;
            continue;
        }
        if ( !push_run( current_begin, current_end ) ) {
            return std::nullopt;
        }
        current_begin = value;
        current_end = value;
    }
    if ( !push_run( current_begin, current_end ) ) {
        return std::nullopt;
    }
    return result;
}

// Counts maximal runs, stopping early once `cap` is reached (the count is then
// reported as exactly `cap`). Callers that only compare the run count against a
// threshold pass that threshold as the cap: on run-poor data — the common case,
// where the count grows by one per element — the scan then stops around the
// midpoint instead of walking the whole container. `cap` == npos counts exactly.
template <typename Layout>
[[nodiscard]] inline std::size_t run_count_from_sorted_values(
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const cap = std::numeric_limits<std::size_t>::max()
) noexcept {
    if ( sorted_values.empty() ) {
        return 0;
    }

    auto run_count{ std::size_t{ 1 } };
    auto previous{ sorted_values.front() };
    for ( auto index{ std::size_t{ 1 } }; index < sorted_values.size(); ++index ) {
        auto const value{ sorted_values[ index ] };
        if (
            previous == std::numeric_limits<typename Layout::low_type>::max() ||
            value != static_cast<typename Layout::low_type>( previous + 1U )
        ) {
            if ( ++run_count >= cap ) { return cap; }
        }
        previous = value;
    }
    return run_count;
}

// Counts maximal runs directly from a bitset's words — a run starts at every set
// bit whose predecessor (across the word boundary too) is clear — so the form
// decision for a bitset container needs no value materialization at all. Same
// early-exit contract as the sorted-values twin: the count saturates at `cap`.
template <typename Layout>
[[nodiscard]] inline std::size_t run_count_from_bitset_words(
    std::span<std::uint64_t const> const words,
    std::size_t const cap = std::numeric_limits<std::size_t>::max()
) noexcept {
    std::size_t   run_count   { 0 };
    std::uint64_t previous_top{ 0 };  // the previous word's MSB, as bit 0
    for ( auto const word : words ) {
        auto const starts{ word & ~( ( word << 1U ) | previous_top ) };
        run_count += static_cast<std::size_t>( std::popcount( starts ) );
        if ( run_count >= cap ) { return cap; }
        previous_top = word >> 63U;
    }
    return run_count;
}

// Builds a bitset handle by scattering the given sorted values.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> bitset_handle_from_sorted_values(
    std::span<typename Layout::low_type const> const sorted_values,
    container_handle<Layout, CowPolicy> && reuse = {}
) {
    // A retired scratch bitset (chunk_store::take_retired) is rezeroed in place
    // instead of allocating a fresh 8 KB block (scratch-reuse path).
    auto result{ [&]() {
        if ( reuse.holds_bitset() ) {
            std::memset( reuse.payload_data_raw(), 0, Layout::word_count * sizeof( std::uint64_t ) );
            return std::move( reuse );
        }
        return container_handle<Layout, CowPolicy>::make_bitset_zeroed();
    }() };
    auto bitset{ result.as_bitset() };
    auto & words{ bitset.words.as_array() };
    for ( auto const value : sorted_values ) {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        words[ word_index ] |= std::uint64_t{ 1 } << bit_index;
    }
    result.set_cardinality( static_cast<std::uint32_t>( sorted_values.size() ) );
    if ( !sorted_values.empty() ) {
        result.set_endpoints( sorted_values.front(), sorted_values.back() );
    }
    return result;
}

// Wraps a standalone run_container (the run kernels' result-builder type) into
// a run handle. Cardinality is taken as tracked by the kernel; endpoints come
// from the first/last run.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> handle_from_run_container( run_container<Layout> const & run ) {
    auto result{ container_handle<Layout, CowPolicy>::make_run() };
    auto ref{ result.as_run() };
    ref.runs.assign( { run.runs.data(), run.runs.size() } );
    result.set_cardinality( static_cast<std::uint32_t>( run.cardinality ) );
    if ( !run.runs.empty() ) {
        result.set_endpoints( run.runs.front().begin, run.runs.back().end );
    }
    return result;
}

// Exact CRoaring container_is_full: true only when the header still knows the
// container is saturated. Lazy bitsets with a stale/unknown card return false
// even if every bit happens to be set.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline bool container_is_known_full(
    container_handle<Layout, CowPolicy> const & handle
) noexcept {
    if ( handle.cardinality() != Layout::low_domain_size ) {
        return false;
    }
    if ( handle.holds_run() ) {
        auto const run{ handle.as_run() };
        return run.runs.size() == 1U &&
            run.runs.front().begin == 0U &&
            run.runs.front().end == std::numeric_limits<typename Layout::low_type>::max();
    }
    // Bitset (or a rare full-size array): cardinality alone is authoritative
    // when it equals the domain — matching CRoaring's bitset/array branches.
    return handle.holds_bitset() || handle.holds_array();
}

// Builds a run handle from the given sorted values.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> run_handle_from_sorted_values(
    std::span<typename Layout::low_type const> const sorted_values
) {
    auto result{ container_handle<Layout, CowPolicy>::make_run() };
    result.as_run().from_sorted_values( sorted_values );
    return result;
}

// RunSelectionPolicy gates whether the run-encoding branch is even considered:
// under run_selection_lazy this degrades to the CRoaring-parity array/bitset-only
// decision (see run_selection_policy.hpp). Defaults to run_selection_eager so
// every EXISTING caller of this function (direct or via the unparameterized
// bitmap<> instantiations) keeps today's behavior unless it opts into laziness.
// The array-vs-bitset-vs-run size heuristic shared by make_container_from_sorted_values
// (which copies the values into the chosen representation) and
// make_container_from_filled_array (which can keep an already-materialized array
// handle without a second allocation). Assumes sorted_values is non-empty.
//
// bitset_threshold is the cardinality from which the caller wants a bitset: the
// bitmap's default (array_to_bitset_threshold) sits exactly at the byte-parity
// point (a bitset is never larger than the array from there on), so the default
// decision is the pure smallest-form rule. A caller passing a LOWER threshold is
// asking for bitsets before they pay off in bytes — dense chunks whose consumers
// are intersection kernels (array∩bitset probes and word-parallel bitset∩bitset
// are far cheaper than merging two multi-thousand-element arrays) — so the
// threshold is honored as given rather than being re-gated on byte parity, which
// would silently reduce it to the default. A run still wins over the bitset when it
// is the smaller of the two (an explicit threshold trades against arrays only).
template <typename Layout, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_kind choose_sorted_container_kind(
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const bitset_threshold
) {
    auto const can_consider_run{ RunSelectionPolicy::eager && sorted_values.size() >= 64U };
    auto const array_bytes{ sorted_values.size() * sizeof( typename Layout::low_type ) };
    auto const bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    // Only the ORDER of run_bytes against array_bytes/bitset_bytes matters below, so
    // the run scan may stop at the first count that can no longer win — decision-
    // equivalent, because a capped (hence >= the cap's bound) run_bytes fails both
    // the `run_bytes < array_bytes` and the `bitset_bytes <= run_bytes` tests
    // exactly as the true count would.
    constexpr auto run_bytes_per_run{ sizeof( typename run_container<Layout>::run ) };
    auto const run_count_cap{ std::max( array_bytes, bitset_bytes ) / run_bytes_per_run + 1U };
    auto const run_bytes{
        can_consider_run
            ? run_count_from_sorted_values<Layout>( sorted_values, run_count_cap ) * run_bytes_per_run
            : std::numeric_limits<std::size_t>::max()
    };

    if ( sorted_values.size() >= bitset_threshold && bitset_bytes <= run_bytes ) {
        return container_kind::bitset;
    }
    if ( can_consider_run && run_bytes < array_bytes ) {
        return container_kind::run;
    }
    return container_kind::array;
}

template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_container_from_sorted_values(
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const bitset_threshold
) {
    if ( sorted_values.empty() ) {
        return container_handle<Layout, CowPolicy>{};
    }
    switch ( choose_sorted_container_kind<Layout, RunSelectionPolicy>( sorted_values, bitset_threshold ) ) {
        case container_kind::bitset: return bitset_handle_from_sorted_values<Layout, CowPolicy>( sorted_values );
        case container_kind::run   : return run_handle_from_sorted_values   <Layout, CowPolicy>( sorted_values );
        case container_kind::array : break;
    }
    return container_handle<Layout, CowPolicy>::make_array_from_sorted( sorted_values );
}

template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_container_from_sorted_vector(
    small_array_values<typename Layout::low_type> && sorted_values,
    std::size_t const bitset_threshold
) {
    return make_container_from_sorted_values<Layout, CowPolicy, RunSelectionPolicy>( { sorted_values.data(), sorted_values.size() }, bitset_threshold );
}

// Companion to make_container_from_sorted_values for callers that have ALREADY
// materialized the sorted result into an owned array handle (e.g. the array∩bitset /
// array\bitset combine arms, which run filter_array_bitset_into straight into the
// result payload). The common array outcome returns that handle as-is — no second
// allocation, no scratch→payload copy, the whole point on the hot combine path —
// while the rare bitset/run outcomes rebuild from the handle's values exactly as
// make_container_from_sorted_values would.
template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_container_from_filled_array(
    container_handle<Layout, CowPolicy> && filled_array,
    std::size_t const bitset_threshold
) {
    auto const & const_handle{ filled_array };
    auto const view{ const_handle.as_array().values };
    if ( view.empty() ) {
        return container_handle<Layout, CowPolicy>{};
    }
    std::span<typename Layout::low_type const> const sorted_values{ view.data(), view.size() };
    switch ( choose_sorted_container_kind<Layout, RunSelectionPolicy>( sorted_values, bitset_threshold ) ) {
        case container_kind::bitset: return bitset_handle_from_sorted_values<Layout, CowPolicy>( sorted_values );
        case container_kind::run   : return run_handle_from_sorted_values   <Layout, CowPolicy>( sorted_values );
        case container_kind::array : break;
    }
    return std::move( filled_array );
}

// Explicit, caller-requested compaction (bitmap::optimize()/optimize_for_storage(),
// the CRoaring run_optimize() analog): callers that want the eager heuristic
// unconditionally pass RunSelectionPolicy = run_selection_eager explicitly here
// regardless of the bitmap's own ambient policy.
template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> optimize_container_for_storage(
    container_handle<Layout, CowPolicy> container,
    std::size_t const bitset_threshold
) {
    if ( container.holds_bitset() ) {
        // A bitset's form decision needs only its cardinality (in the header) and its
        // run count (countable from the words), so the by-far-common "stays a bitset"
        // outcome costs a word scan and returns the very handle — where the generic
        // path below materializes every value and re-scatters them into a freshly
        // allocated, bit-identical bitset.
        auto const cardinality{ static_cast<std::size_t>( container.cardinality() ) };
        if ( cardinality == 0 ) { return container_handle<Layout, CowPolicy>{}; }
        constexpr auto run_bytes_per_run{ sizeof( typename run_container<Layout>::run ) };
        constexpr auto bitset_bytes     { Layout::word_count * sizeof( std::uint64_t ) };
        auto const array_bytes{ cardinality * sizeof( typename Layout::low_type ) };
        auto const can_consider_run{ RunSelectionPolicy::eager && cardinality >= 64U };
        auto const run_bytes{
            can_consider_run
                ? run_count_from_bitset_words<Layout>(
                      std::span<std::uint64_t const>{ std::as_const( container ).as_bitset().words.as_array() },
                      std::max( array_bytes, bitset_bytes ) / run_bytes_per_run + 1U
                  ) * run_bytes_per_run
                : std::numeric_limits<std::size_t>::max()
        };
        if ( cardinality >= bitset_threshold && bitset_bytes <= array_bytes && bitset_bytes <= run_bytes ) {
            return container;  // identity — no extraction, no allocation
        }
        auto values{ sorted_values_from_container<Layout>( container ) };
        if ( can_consider_run && run_bytes < array_bytes ) {
            return run_handle_from_sorted_values<Layout, CowPolicy>( { values.data(), values.size() } );
        }
        return container_handle<Layout, CowPolicy>::make_array_from_sorted( { values.data(), values.size() } );
    }
    if ( container.holds_array() ) {
        // An array container IS its sorted-values span already: decide on it in
        // place and keep the very handle when the decision is "stay an array" —
        // the generic path below would extract a copy of the values and rebuild
        // an identical array from them (two allocations and two copies per
        // container for the by-far-common no-op outcome; CRoaring's
        // repair_after_lazy/run_optimize likewise never rebuild an array that
        // stays an array).
        return make_container_from_filled_array<Layout, CowPolicy, RunSelectionPolicy>( std::move( container ), bitset_threshold );
    }
    auto values{ sorted_values_from_container<Layout>( container ) };
    if ( values.empty() ) {
        return container_handle<Layout, CowPolicy>{};
    }
    return make_container_from_sorted_values<Layout, CowPolicy, RunSelectionPolicy>( { values.data(), values.size() }, bitset_threshold );
}

// run_optimize()'s bitset arm: reconsider ONLY run encoding for a bitset container,
// never the bitset→array down-conversion. Decided from the words (no value
// materialization) and returns the very handle when the bitset stays — so the
// caller needs no defensive copy of the container it is about to replace.
template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> optimize_bitset_run_only(
    container_handle<Layout, CowPolicy> container,
    std::size_t const bitset_threshold
) {
    auto const cardinality{ static_cast<std::size_t>( container.cardinality() ) };
    auto const can_consider_run{ RunSelectionPolicy::eager && cardinality >= 64U };
    if ( !can_consider_run ) { return container; }
    constexpr auto run_bytes_per_run{ sizeof( typename run_container<Layout>::run ) };
    constexpr auto bitset_bytes     { Layout::word_count * sizeof( std::uint64_t ) };
    auto const array_bytes{ cardinality * sizeof( typename Layout::low_type ) };
    auto const run_bytes{
        run_count_from_bitset_words<Layout>(
            std::span<std::uint64_t const>{ std::as_const( container ).as_bitset().words.as_array() },
            std::max( array_bytes, bitset_bytes ) / run_bytes_per_run + 1U
        ) * run_bytes_per_run
    };
    // choose_sorted_container_kind's decision, evaluated without materializing the
    // values: a "bitset" outcome is the identity here and an "array" outcome is the
    // one this call exists to suppress, so only a "run" outcome does any work.
    auto const stays_bitset{ cardinality >= bitset_threshold && bitset_bytes <= array_bytes && bitset_bytes <= run_bytes };
    if ( stays_bitset || run_bytes >= array_bytes ) { return container; }
    auto values{ sorted_values_from_container<Layout>( container ) };
    return run_handle_from_sorted_values<Layout, CowPolicy>( { values.data(), values.size() } );
}

template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> optimize_container_keep_bitsets(
    container_handle<Layout, CowPolicy> container,
    std::size_t const bitset_threshold
) {
    if ( container.holds_bitset() ) {
        return container;
    }
    return optimize_container_for_storage<Layout, CowPolicy, RunSelectionPolicy>( std::move( container ), bitset_threshold );
}

template <typename Layout>
using word_array = std::array<std::uint64_t, Layout::word_count>;

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline word_array<Layout> words_from_container( container_handle<Layout, CowPolicy> const & container ) {
    if ( container.holds_bitset() ) {
        return container.as_bitset().words.as_array();
    }
    word_array<Layout> words{};
    container_for_each<Layout>( container, [&]( typename Layout::low_type const value ) {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        words[ word_index ] |= std::uint64_t{ 1 } << bit_index;
    } );
    return words;
}

template <typename Layout>
inline void apply_range_to_words(
    word_array<Layout> & words,
    typename Layout::low_type const begin,
    typename Layout::low_type const end,
    range_operation const op
) noexcept {
    auto const first_word{ static_cast<std::size_t>( begin ) >> 6U };
    auto const last_word{ static_cast<std::size_t>( end ) >> 6U };
    auto const first_bit{ static_cast<unsigned>( begin ) & 63U };
    auto const last_bit{ static_cast<unsigned>( end ) & 63U };

    auto mask_for{ []( unsigned const start_bit, unsigned const end_bit ) noexcept {
        auto const leading{ std::numeric_limits<std::uint64_t>::max() << start_bit };
        auto const trailing{
            end_bit == 63U
                ? std::numeric_limits<std::uint64_t>::max()
                : ( std::uint64_t{ 1 } << ( end_bit + 1U ) ) - 1U
        };
        return leading & trailing;
    } };

    for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
        auto const start_bit{ word_index == first_word ? first_bit : 0U };
        auto const end_bit{ word_index == last_word ? last_bit : 63U };
        auto const mask{ mask_for( start_bit, end_bit ) };
        switch ( op ) {
            case range_operation::add:
                words[ word_index ] |= mask;
                break;
            case range_operation::remove:
                words[ word_index ] &= ~mask;
                break;
            case range_operation::flip:
                words[ word_index ] ^= mask;
                break;
        }
    }
}

template <typename Layout>
[[nodiscard]] inline small_array_values<typename Layout::low_type> values_from_words( word_array<Layout> const & words ) {
    small_array_values<typename Layout::low_type> result;
    for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
        auto word{ words[ index ] };
        while ( word != 0 ) {
            auto const bit{ static_cast<std::size_t>( std::countr_zero( word ) ) };
            result.push_back( static_cast<typename Layout::low_type>( ( index << 6U ) + bit ) );
            word &= word - 1U;
        }
    }
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_bitset_container_from_words( word_array<Layout> const & words ) {
    auto const cardinality{ std::ranges::fold_left(
        words,
        std::size_t{ 0 },
        []( std::size_t const acc, std::uint64_t const word ) noexcept {
            return acc + static_cast<std::size_t>( std::popcount( word ) );
        }
    ) };
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    auto result{ container_handle<Layout, CowPolicy>::make_bitset_uninitialized() };
    std::memcpy( result.payload_data_raw(), words.data(), sizeof( word_array<Layout> ) );
    result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_container_from_words(
    word_array<Layout> const & words,
    std::size_t const bitset_threshold
) {
    auto const cardinality{ std::ranges::fold_left(
        words,
        std::size_t{ 0 },
        []( std::size_t const acc, std::uint64_t const word ) noexcept {
            return acc + static_cast<std::size_t>( std::popcount( word ) );
        }
    ) };
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }

    auto const array_bytes{ cardinality * sizeof( typename Layout::low_type ) };
    auto const bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    if ( cardinality >= bitset_threshold && bitset_bytes <= array_bytes ) {
        auto result{ container_handle<Layout, CowPolicy>::make_bitset_uninitialized() };
        std::memcpy( result.payload_data_raw(), words.data(), sizeof( word_array<Layout> ) );
        result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
        return result;
    }

    auto values{ values_from_words<Layout>( words ) };
    return make_container_from_sorted_values<Layout, CowPolicy, RunSelectionPolicy>( { values.data(), values.size() }, bitset_threshold );
}

template <typename Layout>
[[nodiscard]] inline word_array<Layout> combine_words(
    word_array<Layout> lhs,
    word_array<Layout> const & rhs,
    set_operation const op
) noexcept {
    for ( std::size_t index{ 0 }; index < lhs.size(); ++index ) {
        switch ( op ) {
            case set_operation::bit_or:
                lhs[ index ] |= rhs[ index ];
                break;
            case set_operation::bit_and:
                lhs[ index ] &= rhs[ index ];
                break;
            case set_operation::bit_andnot:
                lhs[ index ] &= ~rhs[ index ];
                break;
        }
    }
    return lhs;
}

// ---- range flip ---------------------------------------------------------------
//
// The per-form negation over a closed range, as the reference's container_not_range
// does it: every form has a kernel that works on its own representation and decides
// its own result form, instead of expanding to words and re-deriving the form.

// The form decision's bitset arm: a result of this cardinality is held as a bitset.
template <typename Layout>
[[nodiscard]] constexpr bool is_bitset_cardinality( std::size_t const cardinality, std::size_t const bitset_threshold ) noexcept {
    return cardinality >= bitset_threshold && Layout::word_count * sizeof( std::uint64_t ) <= cardinality * sizeof( typename Layout::low_type );
}

// A new run [begin, end] appended after the n runs, merged into the last one when
// they touch. Returns the new run count.
template <typename Run>
[[nodiscard]] inline std::size_t append_run( Run * const runs, std::size_t const n, std::size_t const begin, std::size_t const end ) noexcept {
    using low_type = decltype( Run::begin );
    if ( n != 0 && static_cast<std::size_t>( runs[ n - 1 ].end ) + 1U == begin ) {
        runs[ n - 1 ].end = static_cast<low_type>( end );
        return n;
    }
    runs[ n ] = Run{ static_cast<low_type>( begin ), static_cast<low_type>( end ) };
    return n + 1U;
}

// Exclusive-or of the run [begin, end] into the runs built so far, where begin is
// not below the last run's begin (run_container_smart_append_exclusive).
template <typename Run>
[[nodiscard]] inline std::size_t append_run_exclusive( Run * const runs, std::size_t n, std::size_t const begin, std::size_t const end ) noexcept {
    using low_type = decltype( Run::begin );
    if ( n == 0 || begin > static_cast<std::size_t>( runs[ n - 1 ].end ) + 1U ) {
        runs[ n ] = Run{ static_cast<low_type>( begin ), static_cast<low_type>( end ) };
        return n + 1U;
    }
    auto & last{ runs[ n - 1 ] };
    auto const old_end{ static_cast<std::size_t>( last.end ) };
    if ( old_end + 1U == begin ) {
        last.end = static_cast<low_type>( end );
        return n;
    }
    if ( begin == last.begin ) {  // wipes out the previous run
        if ( end < old_end ) {
            last = Run{ static_cast<low_type>( end + 1U ), static_cast<low_type>( old_end ) };
        } else if ( end > old_end ) {
            last = Run{ static_cast<low_type>( old_end + 1U ), static_cast<low_type>( end ) };
        } else {
            --n;
        }
        return n;
    }
    last.end = static_cast<low_type>( begin - 1U );
    if ( end < old_end ) {
        runs[ n ] = Run{ static_cast<low_type>( end + 1U ), static_cast<low_type>( old_end ) };
        return n + 1U;
    }
    if ( end > old_end ) {
        runs[ n ] = Run{ static_cast<low_type>( old_end + 1U ), static_cast<low_type>( end ) };
        return n + 1U;
    }
    return n;
}

// Flip of [begin, end] in an array. A result past the bitset threshold with fewer
// runs than a bitset would take is built as runs straight from the values (the
// gaps inside the range, the values outside it): array_container_negation_range_to_any.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> flip_array_range(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::low_type const begin,
    typename Layout::low_type const end,
    std::size_t const bitset_threshold
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    using low_type    = typename Layout::low_type;
    using run_type    = ::frsr::roaring::run<low_type>;
    auto const view{ container.as_array().values };
    std::span<low_type const> const values{ view.data(), view.size() };
    auto const cardinality{ values.size() };
    auto const first{ std::lower_bound( values.begin(), values.end(), begin ) };
    auto const last { std::upper_bound( first         , values.end(), end   ) };
    auto const in_range{ static_cast<std::size_t>( last - first ) };
    auto const span    { static_cast<std::size_t>( end ) - begin + 1U };
    auto const new_cardinality{ cardinality - in_range + ( span - in_range ) };

    constexpr auto bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    if ( span > cardinality && is_bitset_cardinality<Layout>( span - cardinality, bitset_threshold ) &&
         ( cardinality + 1U ) * sizeof( run_type ) + sizeof( low_type ) < bitset_bytes )
    {
        auto result{ handle_type::make_run() };
        auto ref{ result.as_run() };
        ref.runs.resize_uninitialized( static_cast<std::uint32_t>( cardinality + 1U ) );
        auto * const runs{ ref.runs.data() };
        std::size_t n{ 0 };
        auto in{ values.begin() };
        for ( ; in != first; ++in ) { n = append_run( runs, n, *in, *in ); }
        std::size_t next{ begin };  // first value of the range not yet decided
        for ( ; in != last; ++in ) {
            if ( *in > next ) { n = append_run( runs, n, next, std::size_t{ *in } - 1U ); }
            next = std::size_t{ *in } + 1U;
        }
        if ( next <= end ) { n = append_run( runs, n, next, end ); }
        for ( ; in != values.end(); ++in ) { n = append_run( runs, n, *in, *in ); }
        assert( n != 0 );
        ref.runs.resize_uninitialized( static_cast<std::uint32_t>( n ) );
        ref.sync_header();
        return result;
    }

    if ( is_bitset_cardinality<Layout>( new_cardinality, bitset_threshold ) ) {
        if ( begin == 0 && end == Layout::low_domain_size - 1U ) {
            // The negation of a whole array: all bits set, then the values cleared.
            auto result{ handle_type::make_bitset_uninitialized() };
            auto & words{ result.as_bitset().words.as_array() };
            words.fill( std::numeric_limits<std::uint64_t>::max() );
            for ( auto const value : values ) {
                words[ static_cast<std::size_t>( value ) >> 6U ] &= ~( std::uint64_t{ 1 } << ( value & 63U ) );
            }
            result.set_cardinality( static_cast<std::uint32_t>( new_cardinality ) );
            result.mark_endpoints_stale();
            return result;
        }
        auto result{ bitset_handle_from_sorted_values<Layout, CowPolicy>( values ) };
        apply_range_to_words<Layout>( result.as_bitset().words.as_array(), begin, end, range_operation::flip );
        result.set_cardinality( static_cast<std::uint32_t>( new_cardinality ) );
        result.mark_endpoints_stale();
        return result;
    }

    if ( new_cardinality == 0 ) { return handle_type{}; }
    auto result{ handle_type::make_array() };
    auto ref{ result.as_array() };
    ref.values.resize_uninitialized( static_cast<std::uint32_t>( new_cardinality ) );
    auto * out{ ref.values.data() };
    out = std::copy( values.begin(), first, out );
    auto in{ first };
    std::size_t value{ begin };
    for ( ; value <= end && in != last; ++value ) {
        if ( value != *in ) { *out++ = static_cast<low_type>( value ); } else { ++in; }
    }
    for ( ; value <= end; ++value ) { *out++ = static_cast<low_type>( value ); }
    std::copy( last, values.end(), out );
    ref.sync_header();
    return result;
}

// Flip of [begin, end] in a run container: the range joins the runs as one more
// exclusive-or append (run_container_negation_range), then the result takes the
// smallest of the three forms.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> flip_run_range(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::low_type const begin,
    typename Layout::low_type const end,
    std::size_t const bitset_threshold
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    using low_type    = typename Layout::low_type;
    using run_type    = ::frsr::roaring::run<low_type>;
    auto const view{ container.as_run().runs };
    std::span<run_type const> const source{ view.data(), view.size() };
    auto result{ handle_type::make_run() };
    auto ref{ result.as_run() };
    // An exclusive-or with one interval adds at most one run.
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( source.size() + 1U ) );
    auto * const runs{ ref.runs.data() };
    std::size_t n{ 0 };
    auto in{ source.begin() };
    for ( ; in != source.end() && in->begin < begin; ++in ) { runs[ n++ ] = *in; }
    n = append_run_exclusive( runs, n, begin, end );
    for ( ; in != source.end(); ++in ) { n = append_run_exclusive( runs, n, in->begin, in->end ); }
    assert( n <= source.size() + 1U );
    if ( n == 0 ) { return handle_type{}; }
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( n ) );
    ref.sync_header();

    constexpr auto bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    auto const cardinality{ static_cast<std::size_t>( result.cardinality() ) };
    auto const run_bytes  { n * sizeof( run_type ) + sizeof( low_type ) };
    auto const array_bytes{ cardinality * sizeof( low_type ) };
    if ( run_bytes <= std::min( bitset_bytes, array_bytes ) ) { return result; }
    if ( !is_bitset_cardinality<Layout>( cardinality, bitset_threshold ) ) {
        return array_handle_from_container<Layout, CowPolicy>( result );
    }
    word_array<Layout> words{};
    for ( auto const & current : std::span<run_type const>{ ref.runs.data(), n } ) {
        apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::add );
    }
    return make_bitset_container_from_words<Layout, CowPolicy>( words );
}

// Flip of [begin, end] in a bitset: the words flip in place, and the result is an
// array again once it falls below the bitset threshold.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> flip_bitset_range(
    container_handle<Layout, CowPolicy> container,
    typename Layout::low_type const begin,
    typename Layout::low_type const end,
    std::size_t const bitset_threshold
) {
    auto & words{ container.as_bitset().words.as_array() };
    apply_range_to_words<Layout>( words, begin, end, range_operation::flip );
    auto const cardinality{ std::ranges::fold_left(
        words, std::size_t{ 0 }, []( std::size_t const acc, std::uint64_t const word ) noexcept { return acc + static_cast<std::size_t>( std::popcount( word ) ); }
    ) };
    if ( cardinality == 0 ) { return container_handle<Layout, CowPolicy>{}; }
    if ( !is_bitset_cardinality<Layout>( cardinality, bitset_threshold ) ) {
        auto const values{ values_from_words<Layout>( words ) };
        return container_handle<Layout, CowPolicy>::make_array_from_sorted( { values.data(), values.size() } );
    }
    container.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    container.mark_endpoints_stale();
    return container;
}

// Applies add/remove/flip of the closed range [begin, end] to one chunk's
// container (an empty handle for an absent chunk).
template <typename Layout, typename CowPolicy, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_handle<Layout, CowPolicy> apply_range_to_container(
    container_handle<Layout, CowPolicy> container,
    typename Layout::low_type const begin,
    typename Layout::low_type const end,
    range_operation const op,
    std::size_t const bitset_threshold
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    using low_type    = typename Layout::low_type;
    if ( op != range_operation::flip ) {
        auto words{ words_from_container<Layout>( container ) };
        apply_range_to_words<Layout>( words, begin, end, op );
        return make_container_from_words<Layout, CowPolicy, RunSelectionPolicy>( words, bitset_threshold );
    }
    if ( container.cardinality() == 0 ) {
        // container_range_of_ones: a run, but for the two values an array holds.
        if ( static_cast<std::size_t>( end ) - begin < 2U ) {
            std::array<low_type, 2> const values{ begin, end };
            return handle_type::make_array_from_sorted( { values.data(), begin == end ? 1U : 2U } );
        }
        auto result{ handle_type::make_run() };
        auto ref{ result.as_run() };
        ref.runs.push_back( ::frsr::roaring::run<low_type>{ begin, end } );
        ref.sync_header();
        return result;
    }
    switch ( container.kind() ) {
        case container_kind::array : return flip_array_range <Layout, CowPolicy>( container, begin, end, bitset_threshold );
        case container_kind::run   : return flip_run_range   <Layout, CowPolicy>( container, begin, end, bitset_threshold );
        case container_kind::bitset: return flip_bitset_range<Layout, CowPolicy>( std::move( container ), begin, end, bitset_threshold );
    }
    std::unreachable();
}

} // namespace frsr::roaring::detail
