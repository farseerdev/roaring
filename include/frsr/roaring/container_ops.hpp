#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/array_ops.hpp>
#include <frsr/roaring/run_ops.hpp>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>


namespace frsr::roaring::detail {

// RunSelectionPolicy gates only the array/bitset-vs-run size-estimate decisions
// this function forwards into make_container_from_sorted_vector (the array∩array,
// array∩run, array-andnot-run, and array∩bitset/array-andnot-bitset arms below).
// The run∩run / run∪run / run-andnot-run / run-andnot-array / run-andnot-bitset /
// union(run,array) arms feed an already-run-shaped kernel result into
// make_container_from_runs (run-andnot-bitset counts its survivors' runs and
// chooses the form from that count, by the same rule), a materially
// different decision (keep-vs-downgrade an existing run result, not "should sorted
// values become a run") that is intentionally left unconditional: under
// run_selection_lazy those kernels are only reachable at all if an operand was
// already run-encoded (i.e. produced by an explicit optimize()/deserialization),
// so it is not an implicit-run-creation site in the sense this policy targets.
template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_variant<Layout, CowPolicy> combine_containers(
    container_variant<Layout, CowPolicy> const & lhs,
    container_variant<Layout, CowPolicy> const & rhs,
    std::size_t const bitset_threshold,
    set_operation const op
) {
    assert( op != set_operation::bit_xor && "an exclusive-or goes through xor_containers" );
    auto combine_run_bitset_words = []( run_cref<Layout, CowPolicy> const run, bitset_cref<Layout, CowPolicy> const bitset, set_operation const run_op ) {
        if ( run_op == set_operation::bit_or ) {
            auto words{ bitset.words.as_array() };
            for ( auto const & current : run.runs ) {
                apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::add );
            }
            return words;
        }

        // Hoist the and-vs-andnot choice out of the innermost per-word loop (run_op is
        // bit_and or bit_andnot here — bit_or already returned above), matching the
        // sibling combine_bitset_bitset kernel's "hoist the operation out of the loop"
        // principle instead of re-branching on every word of every run.
        word_array<Layout> words{};
        auto const combine_run{ [ & ]( ::frsr::roaring::run<typename Layout::low_type> const & current, auto const combine_word ) {
            auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
            auto const last_word{ static_cast<std::size_t>( current.end ) >> 6U };
            auto const first_bit{ static_cast<unsigned>( current.begin ) & 63U };
            auto const last_bit{ static_cast<unsigned>( current.end ) & 63U };
            for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                auto mask{ std::numeric_limits<std::uint64_t>::max() };
                if ( word_index == first_word ) {
                    mask &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                }
                if ( word_index == last_word && last_bit != 63U ) {
                    mask &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                }
                words[ word_index ] |= combine_word( bitset.words[ word_index ], mask );
            }
        } };
        if ( run_op == set_operation::bit_and ) {
            for ( auto const & current : run.runs ) {
                combine_run( current, []( std::uint64_t const bits, std::uint64_t const mask ) { return bits & mask; } );
            }
        } else {
            for ( auto const & current : run.runs ) {
                combine_run( current, []( std::uint64_t const bits, std::uint64_t const mask ) { return ( ~bits ) & mask; } );
            }
        }
        return words;
    };

    // array∩bitset / array\bitset: run the membership filter straight into the
    // result container's own array payload (no intermediate small_array_values
    // scratch, no scratch→payload copy). |result| ≤ |array operand| < the bitset
    // threshold on the common path, so make_container_from_filled_array keeps this
    // very handle — matching CRoaring's write-into-preallocated-result shape.
    auto filter_into_container{ [ bitset_threshold ](
        array_cref<Layout, CowPolicy> const array,
        bitset_cref<Layout, CowPolicy> const bitset,
        bool const keep_matches
    ) -> container_handle<Layout, CowPolicy> {
        container_handle<Layout, CowPolicy> result_handle;
        auto result_array{ result_handle.as_array() };
        filter_array_bitset_into<Layout>( array, bitset, keep_matches, result_array.values );
        if ( result_array.values.empty() ) {
            return container_handle<Layout, CowPolicy>{};
        }
        result_array.sync_header();
        return make_container_from_filled_array<Layout, CowPolicy, RunSelectionPolicy>( std::move( result_handle ), bitset_threshold );
    } };

    return visit_container_pair<Layout, CowPolicy>( [&]( auto const & left, auto const & right ) -> container_handle<Layout, CowPolicy> {
        using left_type = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;

        if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( combine_array_array<Layout, CowPolicy>( left, right, op ), bitset_threshold );
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_array<Layout, CowPolicy>( right, left ), bitset_threshold );
            }
            if ( op == set_operation::bit_and ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( intersect_array_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( difference_array_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_array<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_and ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( intersect_array_run<Layout, CowPolicy>( right, left ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return difference_run_array<Layout, CowPolicy>( left, right, bitset_threshold );
            }
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_bitset_container_from_words<Layout, CowPolicy>( combine_bitset_array_words<Layout, CowPolicy>( right, left, op ) );
            }
            if ( op == set_operation::bit_and ) {
                return filter_into_container( left, right, true );
            }
            if ( op == set_operation::bit_andnot ) {
                return filter_into_container( left, right, false );
            }
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or || op == set_operation::bit_andnot ) {
                return make_bitset_container_from_words<Layout, CowPolicy>( combine_bitset_array_words<Layout, CowPolicy>( left, right, op ) );
            }
            return filter_into_container( right, left, true );
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_andnot ) {
                // A subtraction only thins the left runs: they stay runs while that is the smaller form (like the
                // run-array and run-run arms), instead of being scattered into a bitset.
                return difference_run_bitset<Layout, CowPolicy>( left, right, bitset_threshold );
            }
            return make_bitset_container_from_words<Layout, CowPolicy>( combine_run_bitset_words( left, right, op ) );
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_andnot ) {
                auto words{ left.words.as_array() };
                for ( auto const & current : right.runs ) {
                    apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::remove );
                }
                return make_bitset_container_from_words<Layout, CowPolicy>( words );
            }
            return make_bitset_container_from_words<Layout, CowPolicy>( combine_run_bitset_words( right, left, op ) );
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            return combine_bitset_bitset<Layout, CowPolicy>( left, right, op );
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_and ) {
                return make_container_from_runs<Layout, CowPolicy>( intersect_run_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return difference_run_run<Layout, CowPolicy>( left, right, bitset_threshold );
            }
        }

        auto words{ combine_words<Layout>( words_from_container<Layout>( lhs ), words_from_container<Layout>( rhs ), op ) };
        return make_container_from_words<Layout, CowPolicy>( words, bitset_threshold );
    }, lhs, rhs );
}

// The group form of container_remove(), one pass over the container for the whole sorted, duplicate-free group.
// An array compacts in place, a bitset clears its bits, and a run container is replaced by the runs minus the
// values, merged in one walk and settled to the smallest form (the container it replaces may be shared, so it is
// never written). Returns how many values were removed.
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline std::size_t container_remove_sorted(
    container_handle<Layout, CowPolicy> & container,
    std::span<typename Layout::low_type const> const sorted_values,
    std::size_t const bitset_threshold
) {
    assert( !sorted_values.empty() );
    assert( std::ranges::adjacent_find( sorted_values, std::greater_equal<>{} ) == sorted_values.end() );
    switch ( container.kind() ) {
        case container_kind::array : return container.as_array ().remove_sorted( sorted_values );
        case container_kind::bitset: return container.as_bitset().remove_sorted( sorted_values );
        case container_kind::run   : {
            auto const runs{ std::as_const( container ).as_run() };
            // Nothing to take out when the group lies wholly outside the covered span.
            if ( runs.runs.empty() || sorted_values.back() < runs.runs.front().begin || runs.runs.back().end < sorted_values.front() ) {
                return 0;
            }
            auto const before{ container_size<Layout>( container ) };
            container = difference_run_values<Layout, CowPolicy>( runs, sorted_values, bitset_threshold );
            return before - container_size<Layout>( container );
        }
    }
    std::unreachable();
}

// ---- symmetric difference ---------------------------------------------------------------------------------
//
// The result of an exclusive-or lands in the form the reference gives it: an array when it holds no more values
// than `array_limit`, a bitset above, and runs only where an operand was run-encoded and they are the smallest of
// the three. The rules the union and difference kernels use for minting a run encoding do not apply: an
// exclusive-or of two arrays never produces one.

// A bitset result in its final form: empty, an array when it fell to the limit or below, else itself.
template <typename Layout, typename CowPolicy, typename Retire>
[[nodiscard]] inline container_handle<Layout, CowPolicy> settle_xor_bitset(
    container_handle<Layout, CowPolicy> && result,
    std::size_t const array_limit,
    Retire && retire
) {
    auto const cardinality{ result.cardinality() };
    if ( cardinality == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    if ( cardinality <= array_limit ) {
        auto array{ array_from_bitset<Layout, CowPolicy>( std::as_const( result ).as_bitset(), cardinality ) };
        retire( std::move( result ) );
        return array;
    }
    return std::move( result );
}

template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> settle_xor_bitset(
    container_handle<Layout, CowPolicy> && result,
    std::size_t const array_limit
) {
    return settle_xor_bitset<Layout, CowPolicy>( std::move( result ), array_limit, []( auto && ) noexcept {} );
}

// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:array_array_container_xor
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_array_array(
    array_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    std::size_t const array_limit
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    if ( lhs.values.size() + rhs.values.size() <= array_limit ) {
        handle_type result;
        auto result_array{ result.as_array() };
        xor_array_array_to_vector<Layout>( lhs, rhs, result_array.values );
        if ( result_array.values.empty() ) {
            return handle_type{};
        }
        result_array.sync_header();
        return result;
    }
    // The result may be larger than an array can be: one side scattered into a bitset, the other flipped in.
    auto result{ bitset_handle_from_sorted_values<Layout, CowPolicy>( { lhs.values.data(), lhs.values.size() } ) };
    xor_array_into_bitset_inplace<false, Layout, CowPolicy>( result.as_bitset(), rhs );
    return settle_xor_bitset<Layout, CowPolicy>( std::move( result ), array_limit );
}

// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:array_bitset_container_xor
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_array_bitset(
    array_cref<Layout, CowPolicy> const array,
    bitset_cref<Layout, CowPolicy> const bitset,
    std::size_t const array_limit
) {
    auto result{ clone_bitset_words<Layout, CowPolicy>( bitset ) };
    xor_array_into_bitset_inplace<false, Layout, CowPolicy>( result.as_bitset(), array );
    return settle_xor_bitset<Layout, CowPolicy>( std::move( result ), array_limit );
}

// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:run_bitset_container_xor
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_run_bitset(
    run_cref<Layout, CowPolicy> const runs,
    bitset_cref<Layout, CowPolicy> const bitset,
    std::size_t const array_limit
) {
    auto result{ clone_bitset_words<Layout, CowPolicy>( bitset ) };
    xor_runs_into_bitset_inplace<false, Layout, CowPolicy>( result.as_bitset(), runs );
    return settle_xor_bitset<Layout, CowPolicy>( std::move( result ), array_limit );
}

// A run container of `capacity` runs filled by `fill` (which returns how many it wrote), in its smallest form.
template <typename Layout, typename CowPolicy, typename Fill>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_into_run_result(
    std::size_t const capacity,
    std::size_t const array_limit,
    Fill && fill
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    auto result{ handle_type::make_run() };
    auto ref{ result.as_run() };
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( capacity ) );
    auto const n{ fill( ref.runs.data() ) };
    if ( n == 0 ) {
        return handle_type{};
    }
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( n ) );
    ref.sync_header();
    return settle_run_result<Layout, CowPolicy>( std::move( result ), array_limit + 1U );
}

// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:run_run_container_xor
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    std::size_t const array_limit
) {
    return xor_into_run_result<Layout, CowPolicy>( lhs.runs.size() + rhs.runs.size(), array_limit, [ & ]( auto * const out ) noexcept {
        return xor_runs<Layout>( lhs.runs.data(), lhs.runs.size(), rhs.runs.data(), rhs.runs.size(), out );
    } );
}

// A few values against runs stay runs; many against a run container that fits an array go through the array
// merge; many against a larger one are flipped into the run container's bitset.
// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:array_run_container_xor
template <typename Layout, typename CowPolicy>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_array_run(
    array_cref<Layout, CowPolicy> const array,
    container_handle<Layout, CowPolicy> const & run_handle,
    std::size_t const array_limit
) {
    constexpr std::size_t few_values{ 32 };
    auto const runs{ run_handle.as_run() };
    if ( array.values.size() < few_values ) {
        return xor_into_run_result<Layout, CowPolicy>( array.values.size() + runs.runs.size(), array_limit, [ & ]( auto * const out ) noexcept {
            return xor_values_runs<Layout>( array.values.data(), array.values.size(), runs.runs.data(), runs.runs.size(), out );
        } );
    }
    if ( run_handle.cardinality() <= array_limit ) {
        auto const expanded{ array_handle_from_container<Layout, CowPolicy>( run_handle ) };
        return xor_array_array<Layout, CowPolicy>( expanded.as_array(), array, array_limit );
    }
    auto result{ container_handle<Layout, CowPolicy>::make_bitset_zeroed() };
    {
        auto bitset{ result.as_bitset() };
        auto & words{ bitset.words.as_array() };
        for ( auto const & current : runs.runs ) {
            apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::add );
        }
    }
    result.set_cardinality( run_handle.cardinality() );
    xor_array_into_bitset_inplace<false, Layout, CowPolicy>( result.as_bitset(), array );
    return settle_xor_bitset<Layout, CowPolicy>( std::move( result ), array_limit );
}

// lhs ^ rhs as a legal container. `array_limit` is the largest cardinality an array may have.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> xor_containers(
    container_handle<Layout, CowPolicy> const & lhs,
    container_handle<Layout, CowPolicy> const & rhs,
    std::size_t const array_limit
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    if ( lhs.cardinality() == 0 ) { return handle_type{ rhs }; }
    if ( rhs.cardinality() == 0 ) { return handle_type{ lhs }; }
    return visit_container_pair<Layout, CowPolicy>( [ & ]( auto const & left, auto const & right ) -> handle_type {
        using left_type  = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;
        constexpr bool left_array  { std::same_as<left_type , array_cref <Layout, CowPolicy>> };
        constexpr bool right_array { std::same_as<right_type, array_cref <Layout, CowPolicy>> };
        constexpr bool left_run    { std::same_as<left_type , run_cref   <Layout, CowPolicy>> };
        constexpr bool right_run   { std::same_as<right_type, run_cref   <Layout, CowPolicy>> };
        constexpr bool left_bitset { std::same_as<left_type , bitset_cref<Layout, CowPolicy>> };
        constexpr bool right_bitset{ std::same_as<right_type, bitset_cref<Layout, CowPolicy>> };
        if constexpr ( left_array && right_array ) {
            return xor_array_array<Layout, CowPolicy>( left, right, array_limit );
        } else if constexpr ( left_array && right_bitset ) {
            return xor_array_bitset<Layout, CowPolicy>( left, right, array_limit );
        } else if constexpr ( left_bitset && right_array ) {
            return xor_array_bitset<Layout, CowPolicy>( right, left, array_limit );
        } else if constexpr ( left_run && right_bitset ) {
            return xor_run_bitset<Layout, CowPolicy>( left, right, array_limit );
        } else if constexpr ( left_bitset && right_run ) {
            return xor_run_bitset<Layout, CowPolicy>( right, left, array_limit );
        } else if constexpr ( left_array && right_run ) {
            return xor_array_run<Layout, CowPolicy>( left, rhs, array_limit );
        } else if constexpr ( left_run && right_array ) {
            return xor_array_run<Layout, CowPolicy>( right, lhs, array_limit );
        } else if constexpr ( left_run && right_run ) {
            return xor_run_run<Layout, CowPolicy>( left, right, array_limit );
        } else {
            static_assert( left_bitset && right_bitset );
            return settle_xor_bitset<Layout, CowPolicy>(
                combine_bitset_bitset<Layout, CowPolicy>( left, right, set_operation::bit_xor ), array_limit );
        }
    }, lhs, rhs );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline bool container_intersects(
    container_variant<Layout, CowPolicy> const & lhs,
    container_variant<Layout, CowPolicy> const & rhs
) noexcept {
    return visit_container_pair<Layout, CowPolicy>( []( auto const & left, auto const & right ) noexcept {
        using left_type = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;

        if constexpr (
            std::same_as<left_type, array_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, array_cref<Layout, CowPolicy>>
        ) {
            auto li{ left.values.begin() };
            auto ri{ right.values.begin() };
            while ( li != left.values.end() && ri != right.values.end() ) {
                if ( *li < *ri ) {
                    ++li;
                } else if ( *ri < *li ) {
                    ++ri;
                } else {
                    return true;
                }
            }
            return false;
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> ) {
            for ( auto const value : left.values ) {
                if ( right.contains( value ) ) {
                    return true;
                }
            }
            return false;
        } else if constexpr ( std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            for ( auto const value : right.values ) {
                if ( left.contains( value ) ) {
                    return true;
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, run_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, run_cref<Layout, CowPolicy>>
        ) {
            auto li{ left.runs.begin() };
            auto ri{ right.runs.begin() };
            while ( li != left.runs.end() && ri != right.runs.end() ) {
                if ( li->end < ri->begin ) {
                    ++li;
                } else if ( ri->end < li->begin ) {
                    ++ri;
                } else {
                    return true;
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, run_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, bitset_cref<Layout, CowPolicy>>
        ) {
            for ( auto const & run : left.runs ) {
                auto const first_word{ static_cast<std::size_t>( run.begin ) >> 6U };
                auto const last_word{ static_cast<std::size_t>( run.end ) >> 6U };
                auto const first_bit{ static_cast<unsigned>( run.begin ) & 63U };
                auto const last_bit{ static_cast<unsigned>( run.end ) & 63U };
                for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                    auto word{ right.words[ word_index ] };
                    if ( word == 0 ) {
                        continue;
                    }
                    if ( word_index == first_word ) {
                        word &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                    }
                    if ( word_index == last_word && last_bit != 63U ) {
                        word &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                    }
                    if ( word != 0 ) {
                        return true;
                    }
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, bitset_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, run_cref<Layout, CowPolicy>>
        ) {
            for ( auto const & run : right.runs ) {
                auto const first_word{ static_cast<std::size_t>( run.begin ) >> 6U };
                auto const last_word{ static_cast<std::size_t>( run.end ) >> 6U };
                auto const first_bit{ static_cast<unsigned>( run.begin ) & 63U };
                auto const last_bit{ static_cast<unsigned>( run.end ) & 63U };
                for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                    auto word{ left.words[ word_index ] };
                    if ( word == 0 ) {
                        continue;
                    }
                    if ( word_index == first_word ) {
                        word &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                    }
                    if ( word_index == last_word && last_bit != 63U ) {
                        word &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                    }
                    if ( word != 0 ) {
                        return true;
                    }
                }
            }
            return false;
        } else {
            for ( std::size_t index{ 0 }; index < left.words.size(); ++index ) {
                if ( ( left.words[ index ] & right.words[ index ] ) != 0 ) {
                    return true;
                }
            }
            return false;
        }
    }, lhs, rhs );
}

// How many values the two containers have in common, without building the intersection. Not for two
// arrays: the array kernels are the materializing ones, so a caller that meets that pair intersects it.
// [croaring-ref] deps/croaring/src/containers/mixed_intersection.c (the *_intersection_cardinality family)
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline std::size_t container_intersect_cardinality(
    container_variant<Layout, CowPolicy> const & lhs,
    container_variant<Layout, CowPolicy> const & rhs
) noexcept {
    using low_type = typename Layout::low_type;
    // set bits of one word block inside [first, last], both inclusive
    auto const bits_in_range{ []( std::uint64_t const * const words, std::size_t const first, std::size_t const last ) noexcept {
        auto const first_word{ first >> 6U };
        auto const last_word{ last >> 6U };
        auto const first_mask{ std::numeric_limits<std::uint64_t>::max() << ( first & 63U ) };
        auto const last_mask{ std::numeric_limits<std::uint64_t>::max() >> ( 63U - ( last & 63U ) ) };
        if ( first_word == last_word ) {
            return static_cast<std::size_t>( std::popcount( words[ first_word ] & first_mask & last_mask ) );
        }
        auto count{ static_cast<std::size_t>( std::popcount( words[ first_word ] & first_mask ) ) };
        count += popcount_words( words + first_word + 1U, last_word - first_word - 1U );
        return count + static_cast<std::size_t>( std::popcount( words[ last_word ] & last_mask ) );
    } };
    return visit_container_pair<Layout, CowPolicy>( [ & ]( auto const & left, auto const & right ) noexcept -> std::size_t {
        using left_type = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;
        constexpr bool left_array { std::same_as<left_type , array_cref<Layout, CowPolicy>> };
        constexpr bool right_array{ std::same_as<right_type, array_cref<Layout, CowPolicy>> };
        constexpr bool left_run   { std::same_as<left_type , run_cref  <Layout, CowPolicy>> };
        constexpr bool right_run  { std::same_as<right_type, run_cref  <Layout, CowPolicy>> };
        if constexpr ( left_array && right_array ) {
            return 0;  // not a counted pair, see above
        } else if constexpr ( ( left_array && right_run ) || ( left_run && right_array ) ) {
            auto const & values{ [ & ]() -> auto const & { if constexpr ( left_array ) { return left.values; } else { return right.values; } }() };
            auto const & runs  { [ & ]() -> auto const & { if constexpr ( left_run   ) { return left.runs  ; } else { return right.runs  ; } }() };
            auto const * const keys{ values.data() };
            auto const card{ static_cast<std::size_t>( values.size() ) };
            auto const * const run_list{ runs.data() };
            auto const         run_end { static_cast<std::size_t>( runs.size() ) };
            std::size_t        run_ix  { 0 };
            std::size_t count{ 0 };
            std::size_t ap{ 0 };
            if ( run_end == 0 ) { return 0; }
            if ( runs.size() * 16U <= card ) {
                // few runs over many values: two gallops per run count the values it covers
                for ( auto const & run : runs ) {
                    ap = gallop_forward<false>( keys, ap, card, static_cast<low_type>( run.begin ) );
                    if ( ap == card ) { break; }
                    auto const past{ gallop_forward<true>( keys, ap, card, static_cast<low_type>( run.end ) ) };
                    count += past - ap;
                    ap = past;
                }
                return count;
            }
            // the array-driven walk of filter_array_run_into, counting instead of storing
            while ( ap < card ) {
                auto const value{ keys[ ap ] };
                run_ix = run_reaching( run_list, run_ix, run_end, value );
                if ( run_ix == run_end ) { return count; }
                if ( value >= static_cast<low_type>( run_list[ run_ix ].begin ) ) {
                    ++count;
                    ++ap;
                } else {
                    ap = gallop_forward<false>( keys, ap, card, static_cast<low_type>( run_list[ run_ix ].begin ) );
                }
            }
            return count;
        } else if constexpr ( left_array || right_array ) {
            // array against a bitset: one bit test per value
            auto const & values{ [ & ]() -> auto const & { if constexpr ( left_array ) { return left.values; } else { return right.values; } }() };
            auto const * const words{ [ & ]() { if constexpr ( left_array ) { return right.words.begin(); } else { return left.words.begin(); } }() };
            std::size_t count{ 0 };
            for ( auto const value : values ) {
                count += ( words[ value >> 6U ] >> ( value & 63U ) ) & 1U;
            }
            return count;
        } else if constexpr ( left_run && right_run ) {
            return intersect_runs_cardinality<Layout>( left.runs.data(), left.runs.size(), right.runs.data(), right.runs.size() );
        } else if constexpr ( left_run || right_run ) {
            auto const & runs { [ & ]() -> auto const & { if constexpr ( left_run ) { return left.runs; } else { return right.runs; } }() };
            auto const * const words{ [ & ]() { if constexpr ( left_run ) { return right.words.begin(); } else { return left.words.begin(); } }() };
            std::size_t count{ 0 };
            for ( auto const & run : runs ) {
                count += bits_in_range( words, run.begin, run.end );
            }
            return count;
        } else {
            return and_popcount_words( left.words.begin(), right.words.begin(), left.words.size() );
        }
    }, lhs, rhs );
}

// Writes every value of `container`, composed with `chunk_key` into the full key
// space, straight into `out` (which MUST hold container_size() entries — the
// contract is the reference's, and `out_end` is a slack guard for the vector
// stores, not a bounds check) and returns the end pointer. The equivalent of a for_each with a push_back per value
// minus the per-value capacity check and size increment: each kind writes through a
// plain pointer, which is what lets the array kind's widening loop vectorise.
// [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_to_uint32_array
// Inlined into its per-chunk callers (to_array_into, to_vector): outlined, the
// decode loops lose the callers' knowledge of the output span and the run and
// array kinds decode at half speed.
template <typename Layout, typename CowPolicy>
[[ using gnu: hot, always_inline ]] inline typename Layout::key_type * container_decode_into(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::chunk_type const chunk_key,
    typename Layout::key_type * out,
    typename Layout::key_type * const out_end
) noexcept {
    using key_type = typename Layout::key_type;
    auto const base{ Layout::compose( chunk_key, 0 ) };
    switch ( container.kind() ) {
        case container_kind::array: {
            auto const & values{ container.as_array().values };
            auto const   count { values.size() };
            auto const * const source{ values.data() };
#if FRSR_ROARING_X86_V4
            if constexpr ( std::is_same_v<key_type, std::uint32_t> && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
                if ( count >= x86_v4::decode_array_min_count && have_x86_v4() ) {
                    return x86_v4::decode_array_uint32( source, count, out, base );
                }
            }
#endif
            // Widening add over a contiguous run: clang lowers this to unpack +
            // add + store (vpmovzxwd/vpaddd on x86, ushll/add on AArch64).
            for ( std::uint32_t i{ 0 }; i < count; ++i ) {
                out[ i ] = base + static_cast<key_type>( source[ i ] );
            }
            return out + count;
        }
        case container_kind::run: {
            for ( auto const & run : container.as_run().runs ) {
                auto const first{ base + static_cast<key_type>( run.begin ) };
                using index_type = std::common_type_t<key_type, std::uint32_t>;
                auto const count{ static_cast<index_type>( run.end - run.begin ) + 1U };  // end is inclusive
                // Counted, index-addressed loop (not a pointer-incrementing one) whose index has the
                // width of the values it is added to (never narrower than 32 bits, as a run of 65536
                // values does not fit a 16-bit index): the shape both compilers vectorise into a
                // broadcast + index-vector add + store. A pointer-sized index makes GCC count in 64-bit
                // lanes and narrow every vector.
                // [croaring-ref] deps/croaring/src/containers/run.c:run_container_to_uint32_array
                for ( index_type i{ 0 }; i < count; ++i ) {
                    out[ i ] = first + static_cast<key_type>( i );
                }
                out += count;
            }
            return out;
        }
        case container_kind::bitset: {
            auto const & words{ container.as_bitset().words.as_array() };
            if constexpr ( std::is_same_v<key_type, std::uint32_t> ) {
                return extract_setbits_uint32( words.data(), words.size(), out, out_end, base );
            } else {
                for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
                    auto word{ words[ index ] };
                    auto const word_base{ base + static_cast<key_type>( index << 6U ) };
                    while ( word != 0 ) {
                        *out++ = word_base + static_cast<key_type>( std::countr_zero( word ) );
                        word &= word - 1;
                    }
                }
                return out;
            }
        }
    }
    std::unreachable();
}

} // namespace frsr::roaring::detail
