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
using small_array_values = psi::vm::small_vector<T, array_sbo_size, std::uint32_t>;
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
template <typename Layout, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_kind choose_sorted_container_kind(
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const bitset_threshold
) {
    auto const can_consider_run{ RunSelectionPolicy::eager && sorted_values.size() >= 64U };
    auto const array_bytes{ sorted_values.size() * sizeof( typename Layout::low_type ) };
    // Only the ORDER of run_bytes against array_bytes/bitset_bytes matters below, so
    // the run scan may stop at the first count that can no longer win — decision-
    // equivalent, because a capped (hence >= array_bytes) run_bytes fails the run
    // branch outright, and the bitset branch's own `bitset_bytes <= array_bytes`
    // conjunct then implies `bitset_bytes <= run_bytes` regardless of the true count.
    constexpr auto run_bytes_per_run{ sizeof( typename run_container<Layout>::run ) };
    auto const run_count_cap{ array_bytes / run_bytes_per_run + 1U };
    auto const run_bytes{
        can_consider_run
            ? run_count_from_sorted_values<Layout>( sorted_values, run_count_cap ) * run_bytes_per_run
            : std::numeric_limits<std::size_t>::max()
    };
    auto const bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };

    if ( sorted_values.size() >= bitset_threshold && bitset_bytes <= array_bytes && bitset_bytes <= run_bytes ) {
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

} // namespace frsr::roaring::detail
