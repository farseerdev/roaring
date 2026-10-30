#pragma once

#include <frsr/roaring/array_ops.hpp>
#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/hw_info.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace frsr::roaring::detail {

// Copies a run handle's payload out into the standalone result-builder type.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> run_container_from( run_cref<Layout, CowPolicy> const source ) {
    run_container<Layout> result;
    result.runs.reserve( source.runs.size() );
    for ( auto const & current : source.runs ) {
        result.runs.push_back( current );
    }
    result.cardinality = static_cast<typename run_container<Layout>::cardinality_type>( source.size() );
    return result;
}

// Whether the runs are the whole domain in one run.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline bool run_is_full( run_cref<Layout, CowPolicy> const run ) noexcept {
    return run.runs.size() == 1U &&
        run.runs.front().begin == 0U &&
        run.runs.front().end == std::numeric_limits<typename Layout::low_type>::max();
}

// The first run after `index` that ends at or beyond `limit` (`size` if none): the runs wholly below the other
// list's current run are crossed at once, so a list with few runs against one with many costs the few.
template <typename Run>
[[nodiscard]] [[gnu::always_inline]] inline std::size_t next_run_reaching( Run const * const runs, std::size_t const index, std::size_t const size, std::size_t const limit ) noexcept {
    return gallop_forward<false>(
        runs, index + 1U, size, limit,
        run_end_key<std::size_t>{}
    );
}

struct run_intersection { std::size_t runs; std::size_t cardinality; };

// A run list this many times longer than the other is crossed by gallops and copied in pieces (see
// copy_runs_inside); against lists of comparable length the plain merge is faster, the pieces being too short
// to pay for the gallop and the copy call.
inline constexpr std::size_t skewed_run_lists_ratio{ 16 };

// The runs from `index` on that lie wholly inside the run of the other list that ends at `limit` (exclusive):
// they are the intersection's runs unchanged, so they are found by one gallop and copied in one piece instead
// of being merged one by one - a list of many short runs against a list of a few long ones costs the few.
// Returns the index of the first run that is not inside; `index` must be inside.
template <typename Layout>
[[ gnu::always_inline ]] inline std::size_t copy_runs_inside(
    run<typename Layout::low_type> const * const runs, std::size_t const index, std::size_t const size, std::size_t const limit,
    run<typename Layout::low_type> * const out, run_intersection & result
) noexcept {
    auto const stop{ gallop_forward<false>( runs, index + 1U, size, limit, run_end_key<std::size_t>{} ) };
    std::memcpy( out + result.runs, runs + index, ( stop - index ) * sizeof( *runs ) );
    result.runs += stop - index;
    std::size_t covered{ 0 };
    for ( auto at{ index }; at != stop; ++at ) {
        covered += static_cast<std::size_t>( runs[ at ].end ) - runs[ at ].begin + 1U;
    }
    result.cardinality += covered;
    return stop;
}

template <typename Layout, bool SkewedLists>
[[ gnu::hot ]] inline run_intersection intersect_runs_merge(
    run<typename Layout::low_type> const * const lhs_runs, std::size_t const lhs_size,
    run<typename Layout::low_type> const * const rhs_runs, std::size_t const rhs_size,
    run<typename Layout::low_type> * const out
) noexcept {
    using low_type = typename Layout::low_type;
    run_intersection result{ 0, 0 };
    if ( lhs_size == 0 || rhs_size == 0 ) {
        return result;
    }
    auto li{ std::size_t{ 0 } };
    auto ri{ std::size_t{ 0 } };
    auto start{ static_cast<std::size_t>( lhs_runs[ li ].begin ) };
    auto end{ static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U };
    auto xstart{ static_cast<std::size_t>( rhs_runs[ ri ].begin ) };
    auto xend{ static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U };

    FRSR_ROARING_LOOP_ALIGN while ( li < lhs_size && ri < rhs_size ) {
        if ( end <= xstart ) {
            li = next_run_reaching( lhs_runs, li, lhs_size, xstart );
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
            continue;
        }
        if ( xend <= start ) {
            ri = next_run_reaching( rhs_runs, ri, rhs_size, start );
            if ( ri < rhs_size ) {
                xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
            }
            continue;
        }

        auto const begin{ std::max( start, xstart ) };
        auto earliest_end{ end };
        if ( end == xend ) {
            ++li;
            ++ri;
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
            if ( ri < rhs_size ) {
                xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
            }
        } else if ( end < xend ) {
            ++li;
            out[ result.runs++ ] = run<low_type>{ static_cast<low_type>( begin ), static_cast<low_type>( earliest_end - 1U ) };
            result.cardinality += earliest_end - begin;
            if constexpr ( SkewedLists ) {
                if ( li < lhs_size && static_cast<std::size_t>( lhs_runs[ li ].end ) < xend ) {
                    li = copy_runs_inside<Layout>( lhs_runs, li, lhs_size, xend, out, result );
                }
            }
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
            continue;
        } else {
            earliest_end = xend;
            ++ri;
            out[ result.runs++ ] = run<low_type>{ static_cast<low_type>( begin ), static_cast<low_type>( earliest_end - 1U ) };
            result.cardinality += earliest_end - begin;
            if constexpr ( SkewedLists ) {
                if ( ri < rhs_size && static_cast<std::size_t>( rhs_runs[ ri ].end ) < end ) {
                    ri = copy_runs_inside<Layout>( rhs_runs, ri, rhs_size, end, out, result );
                }
            }
            if ( ri < rhs_size ) {
                xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
            }
            continue;
        }

        out[ result.runs++ ] = run<low_type>{ static_cast<low_type>( begin ), static_cast<low_type>( earliest_end - 1U ) };
        result.cardinality += earliest_end - begin;
    }
    return result;
}

// Run∩run merge into `out`, which MUST hold lhs.size + rhs.size runs (the most an intersection can
// produce, one more than it does). Returns the run count and the cardinality they cover.
// [croaring-ref] deps/croaring/src/containers/run.c:run_container_intersection
template <typename Layout>
[[ gnu::hot ]] inline run_intersection intersect_runs(
    run<typename Layout::low_type> const * const lhs_runs, std::size_t const lhs_size,
    run<typename Layout::low_type> const * const rhs_runs, std::size_t const rhs_size,
    run<typename Layout::low_type> * const out
) noexcept {
    if ( lhs_size >= skewed_run_lists_ratio * rhs_size || rhs_size >= skewed_run_lists_ratio * lhs_size ) {
        return intersect_runs_merge<Layout, true>( lhs_runs, lhs_size, rhs_runs, rhs_size, out );
    }
    return intersect_runs_merge<Layout, false>( lhs_runs, lhs_size, rhs_runs, rhs_size, out );
}

// How many values two run lists have in common: the merge of intersect_runs with the stores dropped.
// [croaring-ref] deps/croaring/src/containers/run.c:run_container_intersection_cardinality
template <typename Layout>
[[nodiscard]] [[ gnu::hot ]] inline std::size_t intersect_runs_cardinality(
    run<typename Layout::low_type> const * const lhs_runs, std::size_t const lhs_size,
    run<typename Layout::low_type> const * const rhs_runs, std::size_t const rhs_size
) noexcept {
    std::size_t cardinality{ 0 };
    if ( lhs_size == 0 || rhs_size == 0 ) {
        return cardinality;
    }
    auto li{ std::size_t{ 0 } };
    auto ri{ std::size_t{ 0 } };
    auto start{ static_cast<std::size_t>( lhs_runs[ li ].begin ) };
    auto end{ static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U };
    auto xstart{ static_cast<std::size_t>( rhs_runs[ ri ].begin ) };
    auto xend{ static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U };
    FRSR_ROARING_LOOP_ALIGN while ( li < lhs_size && ri < rhs_size ) {
        if ( end <= xstart ) {
            li = next_run_reaching( lhs_runs, li, lhs_size, xstart );
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
        } else if ( xend <= start ) {
            ri = next_run_reaching( rhs_runs, ri, rhs_size, start );
            if ( ri < rhs_size ) {
                xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
            }
        } else {
            cardinality += std::min( end, xend ) - std::max( start, xstart );
            // both decisions are taken on the runs just compared: the left run advancing reloads `end`
            auto const left_done { end <= xend };
            auto const right_done{ xend <= end };
            if ( left_done ) {
                ++li;
                if ( li < lhs_size ) {
                    start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                    end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
                }
            }
            if ( right_done ) {
                ++ri;
                if ( ri < rhs_size ) {
                    xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                    xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
                }
            }
        }
    }
    return cardinality;
}

// Run∩run written straight into a caller-supplied run payload (the result container's own, on the
// hot combine path, so no standalone result is built and copied). The caller sets the header.
template <typename Layout, typename OutVector, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline std::size_t intersect_run_run_into(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    OutVector & result
) {
    // Neither side is mutated while the merge runs, so the raw pointers are loop-invariant (indexing
    // the views re-derives them, through a launder, per access).
    auto const lhs_size{ lhs.runs.size() };
    auto const rhs_size{ rhs.runs.size() };
    resize_uninitialized( result, lhs_size + rhs_size );
    auto const merged{ intersect_runs<Layout>( lhs.runs.data(), lhs_size, rhs.runs.data(), rhs_size, result.data() ) };
    resize_uninitialized( result, merged.runs );
    return merged.cardinality;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> intersect_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    if ( lhs.runs.empty() || rhs.runs.empty() ) {
        return result;
    }
    if ( run_is_full<Layout, CowPolicy>( lhs ) ) {
        return run_container_from<Layout, CowPolicy>( rhs );
    }
    if ( run_is_full<Layout, CowPolicy>( rhs ) ) {
        return run_container_from<Layout, CowPolicy>( lhs );
    }
    result.cardinality = static_cast<typename run_container<Layout>::cardinality_type>(
        intersect_run_run_into<Layout, decltype( result.runs ), CowPolicy>( lhs, rhs, result.runs ) );
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> union_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    result.runs.reserve( lhs.runs.size() + rhs.runs.size() );

    auto append_run = [&]( typename run_container<Layout>::run const & next ) {
        if ( result.runs.empty() ) {
            result.runs.push_back( next );
            result.cardinality += static_cast<std::size_t>( next.end ) - static_cast<std::size_t>( next.begin ) + 1U;
            return;
        }

        auto & current{ result.runs.back() };
        if ( static_cast<std::size_t>( current.end ) + 1U < static_cast<std::size_t>( next.begin ) ) {
            result.runs.push_back( next );
            result.cardinality += static_cast<std::size_t>( next.end ) - static_cast<std::size_t>( next.begin ) + 1U;
            return;
        }

        if ( next.end > current.end ) {
            result.cardinality += static_cast<std::size_t>( next.end ) - static_cast<std::size_t>( current.end );
            current.end = next.end;
        }
    };

    auto li{ lhs.runs.begin() };
    auto ri{ rhs.runs.begin() };
    while ( li != lhs.runs.end() && ri != rhs.runs.end() ) {
        if ( li->begin < ri->begin ) {
            append_run( *li++ );
        } else {
            append_run( *ri++ );
        }
    }
    while ( li != lhs.runs.end() ) {
        append_run( *li++ );
    }
    while ( ri != rhs.runs.end() ) {
        append_run( *ri++ );
    }
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> union_run_array(
    run_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    result.runs.reserve( lhs.runs.size() + rhs.values.size() );

    auto append_run = [&]( typename Layout::low_type const begin, typename Layout::low_type const end ) {
        if ( result.runs.empty() ) {
            result.runs.push_back( run<typename Layout::low_type>{ begin, end } );
            result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( begin ) + 1U;
            return;
        }

        auto & current{ result.runs.back() };
        if ( static_cast<std::size_t>( current.end ) + 1U < static_cast<std::size_t>( begin ) ) {
            result.runs.push_back( run<typename Layout::low_type>{ begin, end } );
            result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( begin ) + 1U;
            return;
        }

        if ( end > current.end ) {
            result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( current.end );
            current.end = end;
        }
    };

    auto run_it{ lhs.runs.begin() };
    auto value_it{ rhs.values.begin() };
    while ( run_it != lhs.runs.end() && value_it != rhs.values.end() ) {
        if ( run_it->begin <= *value_it ) {
            append_run( run_it->begin, run_it->end );
            ++run_it;
        } else {
            append_run( *value_it, *value_it );
            ++value_it;
        }
    }
    while ( run_it != lhs.runs.end() ) {
        append_run( run_it->begin, run_it->end );
        ++run_it;
    }
    while ( value_it != rhs.values.end() ) {
        append_run( *value_it, *value_it );
        ++value_it;
    }
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> difference_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs,
    std::size_t const bitset_threshold
) {
    auto result{ container_handle<Layout, CowPolicy>::make_run() };
    auto ref{ result.as_run() };
    // Cutting runs out of runs leaves at most one more run per run taken out.
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( lhs.runs.size() + rhs.runs.size() ) );
    auto * const out{ ref.runs.data() };
    std::size_t n{ 0 };

    auto ri{ rhs.runs.begin() };
    for ( auto const & left : lhs.runs ) {
        while ( ri != rhs.runs.end() && ri->end < left.begin ) {
            ++ri;
        }

        auto current_begin{ left.begin };
        auto exhausted{ false };
        auto scan{ ri };
        while ( scan != rhs.runs.end() && scan->begin <= left.end ) {
            if ( current_begin < scan->begin ) {
                auto const end{ static_cast<typename Layout::low_type>( scan->begin - 1U ) };
                out[ n++ ] = ::frsr::roaring::run<typename Layout::low_type>{ current_begin, end };
            }

            if ( scan->end == std::numeric_limits<typename Layout::low_type>::max() || scan->end >= left.end ) {
                exhausted = true;
                break;
            }

            current_begin = static_cast<typename Layout::low_type>( scan->end + 1U );
            ++scan;
        }

        if ( !exhausted && current_begin <= left.end ) {
            out[ n++ ] = ::frsr::roaring::run<typename Layout::low_type>{ current_begin, left.end };
        }
    }
    if ( n == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( n ) );
    ref.sync_header();
    return settle_run_result<Layout, CowPolicy>( std::move( result ), bitset_threshold );
}

// The runs minus a sorted, duplicate-free list of values: one merge pass over both, O(runs + values), written
// straight into the result's run payload and settled to the smallest form. Used by the run-array difference and by
// the batch removal of sorted values from a run chunk.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> difference_run_values(
    run_cref<Layout, CowPolicy> const lhs,
    std::span<typename Layout::low_type const> const values,
    std::size_t const bitset_threshold
) {
    auto result{ container_handle<Layout, CowPolicy>::make_run() };
    auto ref{ result.as_run() };
    // Every value taken out splits at most one run in two, and disjoint runs that do not touch cannot be more than
    // half the domain.
    constexpr std::size_t most_runs{ Layout::low_domain_size / 2U };
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( std::min<std::size_t>( lhs.runs.size() + values.size(), most_runs ) ) );
    auto * const out{ ref.runs.data() };
    std::size_t n{ 0 };

    auto value_it{ values.begin() };
    for ( auto const & run : lhs.runs ) {
        while ( value_it != values.end() && *value_it < run.begin ) {
            ++value_it;
        }

        auto current_begin{ run.begin };
        auto exhausted{ false };
        auto scan{ value_it };
        while ( scan != values.end() && *scan <= run.end ) {
            if ( current_begin < *scan ) {
                auto const end{ static_cast<typename Layout::low_type>( *scan - 1U ) };
                out[ n++ ] = ::frsr::roaring::run<typename Layout::low_type>{ current_begin, end };
            }
            if ( *scan == std::numeric_limits<typename Layout::low_type>::max() ) {
                exhausted = true;
                break;
            }
            current_begin = static_cast<typename Layout::low_type>( *scan + 1U );
            ++scan;
        }

        value_it = scan;
        if ( !exhausted && current_begin <= run.end ) {
            out[ n++ ] = ::frsr::roaring::run<typename Layout::low_type>{ current_begin, run.end };
        }
    }
    if ( n == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }
    ref.runs.resize_uninitialized( static_cast<std::uint32_t>( n ) );
    ref.sync_header();
    return settle_run_result<Layout, CowPolicy>( std::move( result ), bitset_threshold );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> difference_run_array(
    run_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs,
    std::size_t const bitset_threshold
) {
    return difference_run_values<Layout, CowPolicy>( lhs, { rhs.values.data(), rhs.values.size() }, bitset_threshold );
}

// The symmetric difference of two run lists, merged by begin with every run exclusive-or-appended to what is
// built so far. `out` must hold lhs_size + rhs_size runs (the most it can come to). Returns the run count.
// [croaring-ref] deps/croaring/src/containers/run.c:run_container_xor
template <typename Layout>
[[ gnu::hot ]] inline std::size_t xor_runs(
    run<typename Layout::low_type> const * const lhs, std::size_t const lhs_size,
    run<typename Layout::low_type> const * const rhs, std::size_t const rhs_size,
    run<typename Layout::low_type> * const out
) noexcept {
    std::size_t li{ 0 }, ri{ 0 }, n{ 0 };
    while ( li < lhs_size && ri < rhs_size ) {
        if ( lhs[ li ].begin <= rhs[ ri ].begin ) {
            n = append_run_exclusive( out, n, lhs[ li ].begin, lhs[ li ].end );
            ++li;
        } else {
            n = append_run_exclusive( out, n, rhs[ ri ].begin, rhs[ ri ].end );
            ++ri;
        }
    }
    for ( ; li < lhs_size; ++li ) { n = append_run_exclusive( out, n, lhs[ li ].begin, lhs[ li ].end ); }
    for ( ; ri < rhs_size; ++ri ) { n = append_run_exclusive( out, n, rhs[ ri ].begin, rhs[ ri ].end ); }
    return n;
}

// The same for a sorted array of values (each a run of one) against a run list. `out` must hold
// values_size + runs_size runs.
// [croaring-ref] deps/croaring/src/containers/mixed_xor.c:array_run_container_lazy_xor
template <typename Layout>
[[ gnu::hot ]] inline std::size_t xor_values_runs(
    typename Layout::low_type const * const values, std::size_t const values_size,
    run<typename Layout::low_type> const * const runs, std::size_t const runs_size,
    run<typename Layout::low_type> * const out
) noexcept {
    std::size_t vi{ 0 }, ri{ 0 }, n{ 0 };
    while ( vi < values_size && ri < runs_size ) {
        if ( runs[ ri ].begin <= values[ vi ] ) {
            n = append_run_exclusive( out, n, runs[ ri ].begin, runs[ ri ].end );
            ++ri;
        } else {
            n = append_run_exclusive( out, n, values[ vi ], values[ vi ] );
            ++vi;
        }
    }
    for ( ; vi < values_size; ++vi ) { n = append_run_exclusive( out, n, values[ vi ], values[ vi ] ); }
    for ( ; ri < runs_size  ; ++ri ) { n = append_run_exclusive( out, n, runs[ ri ].begin, runs[ ri ].end ); }
    return n;
}

// ---- run minus bitset ---------------------------------------------------------------------------------
//
// The survivors of a run container after a bitset is taken out of it are the zero bits of the bitset under the
// runs. They are read as a stream of words (a word where the bitset is all zero or all one under the runs is
// skipped four at a time) and consumed twice: once to count them and their runs, which decides the form the result
// takes before anything is built, and once to write that form. A result with more runs than a bitset holds is
// written as words.
//
// [croaring-ref] deps/croaring/src/containers/mixed_andnot.c:run_bitset_container_andnot

// Calls sink.word( index, survivors ) for every word with survivors that is not part of a longer stretch and
// sink.ones( first index, count ) for the stretches of words that are all survivors, in index order. A sink call
// returns false to stop the scan, which then returns false too. The words that are not reported have no survivors.
template <typename Layout, typename CowPolicy, typename Sink>
[[ gnu::always_inline ]] inline bool scan_run_bitset_difference(
    run_cref<Layout, CowPolicy> const lhs,
    word_array<Layout> const & bits,
    Sink & sink
) {
    constexpr auto none{ ~std::size_t{ 0 } };
    constexpr auto all { ~std::uint64_t{ 0 } };
    auto pending_index{ none };
    auto pending{ std::uint64_t{ 0 } };
    auto const flush{ [ & ]() {
        auto const index{ std::exchange( pending_index, none ) };
        return index == none || sink.word( index, pending );
    } };
    auto const put{ [ & ]( std::size_t const index, std::uint64_t const survivors ) {
        if ( index == pending_index ) {
            pending |= survivors;
            return true;
        }
        if ( !flush() ) {
            return false;
        }
        pending_index = index;
        pending       = survivors;
        return true;
    } };
    for ( auto const & current : lhs.runs ) {
        auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
        auto const last_word { static_cast<std::size_t>( current.end   ) >> 6U };
        auto const first_mask{ all << ( current.begin & 63U ) };
        auto const last_mask { all >> ( 63U - ( current.end & 63U ) ) };
        if ( first_word == last_word ) {
            if ( !put( first_word, first_mask & last_mask & ~bits[ first_word ] ) ) {
                return false;
            }
            continue;
        }
        if ( !put( first_word, first_mask & ~bits[ first_word ] ) ) {
            return false;
        }
        auto index{ first_word + 1U };
        while ( index < last_word ) {
            auto const word{ bits[ index ] };
            if ( word != 0 && word != all ) {
                if ( !flush() || !sink.word( index, ~word ) ) {
                    return false;
                }
                ++index;
                continue;
            }
            auto end{ index + 1U };
            if ( word == 0 ) {
                while ( end + 4U <= last_word && ( bits[ end ] | bits[ end + 1U ] | bits[ end + 2U ] | bits[ end + 3U ] ) == 0 ) {
                    end += 4U;
                }
                while ( end < last_word && bits[ end ] == 0 ) {
                    ++end;
                }
                if ( !flush() || !sink.ones( index, end - index ) ) {
                    return false;
                }
            } else {
                while ( end + 4U <= last_word && ( bits[ end ] & bits[ end + 1U ] & bits[ end + 2U ] & bits[ end + 3U ] ) == all ) {
                    end += 4U;
                }
                while ( end < last_word && bits[ end ] == all ) {
                    ++end;
                }
            }
            index = end;
        }
        if ( !put( last_word, last_mask & ~bits[ last_word ] ) ) {
            return false;
        }
    }
    return flush();
}

// Counts the survivors and their runs, giving up once the runs are more than `run_limit`.
struct run_survivor_counter {
    std::size_t   run_limit;
    std::size_t   cardinality{ 0 };
    std::size_t   runs{ 0 };
    std::size_t   last_index{ ~std::size_t{ 0 } };
    std::uint64_t last_top{ 0 };

    [[nodiscard]] std::uint64_t carry_into( std::size_t const index ) const noexcept {
        return index == last_index + 1U ? last_top : 0U;
    }
    [[nodiscard]] bool word( std::size_t const index, std::uint64_t const survivors ) noexcept {
        auto const starts{ survivors & ~( ( survivors << 1U ) | carry_into( index ) ) };
        runs        += static_cast<std::size_t>( std::popcount( starts ) );
        cardinality += static_cast<std::size_t>( std::popcount( survivors ) );
        last_index = index;
        last_top   = survivors >> 63U;
        return runs <= run_limit;
    }
    [[nodiscard]] bool ones( std::size_t const first_index, std::size_t const count ) noexcept {
        runs        += carry_into( first_index ) == 0 ? 1U : 0U;
        cardinality += count * 64U;
        last_index = first_index + count - 1U;
        last_top   = 1U;
        return runs <= run_limit;
    }
};

// Writes the survivors as runs, `out` holding exactly as many as the counter found.
template <typename Layout>
struct run_survivor_writer {
    using low_type = typename Layout::low_type;
    using run_type = ::frsr::roaring::run<low_type>;

    run_type    * out;
    std::size_t   written{ 0 };
    bool          open{ false };
    std::size_t   open_begin{ 0 };
    std::size_t   last_index{ ~std::size_t{ 0 } };

    void close_at( std::size_t const end ) noexcept {
        out[ written++ ] = run_type{ static_cast<low_type>( open_begin ), static_cast<low_type>( end ) };
        open = false;
    }
    void leave_open_run_if_not_adjacent( std::size_t const index ) noexcept {
        if ( open && index != last_index + 1U ) {
            close_at( last_index * 64U + 63U );
        }
    }
    [[nodiscard]] bool word( std::size_t const index, std::uint64_t survivors ) noexcept {
        leave_open_run_if_not_adjacent( index );
        auto const base{ index * 64U };
        if ( open ) {
            auto const lead{ static_cast<unsigned>( std::countr_one( survivors ) ) };
            if ( lead == 64U ) {
                last_index = index;
                return true;
            }
            close_at( base + lead - 1U );
            survivors &= all_bits_from( lead );
        }
        while ( survivors != 0 ) {
            auto const start { static_cast<unsigned>( std::countr_zero( survivors ) ) };
            auto const length{ static_cast<unsigned>( std::countr_one( survivors >> start ) ) };
            if ( start + length == 64U ) {
                open       = true;
                open_begin = base + start;
                break;
            }
            out[ written++ ] = run_type{ static_cast<low_type>( base + start ), static_cast<low_type>( base + start + length - 1U ) };
            survivors &= all_bits_from( start + length );
        }
        last_index = index;
        return true;
    }
    [[nodiscard]] bool ones( std::size_t const first_index, std::size_t const count ) noexcept {
        leave_open_run_if_not_adjacent( first_index );
        if ( !open ) {
            open       = true;
            open_begin = first_index * 64U;
        }
        last_index = first_index + count - 1U;
        return true;
    }
    void finish() noexcept {
        if ( open ) {
            close_at( last_index * 64U + 63U );
        }
    }

private:
    // The bits at and above `position` (below 64).
    [[nodiscard]] static constexpr std::uint64_t all_bits_from( unsigned const position ) noexcept {
        return ~std::uint64_t{ 0 } << position;
    }
};

// Writes the survivors as array values, `out` holding exactly as many as the counter found.
template <typename Layout>
struct array_survivor_writer {
    using low_type = typename Layout::low_type;

    low_type * out;

    [[nodiscard]] bool word( std::size_t const index, std::uint64_t survivors ) noexcept {
        auto const base{ index * 64U };
        while ( survivors != 0 ) {
            *out++ = static_cast<low_type>( base + static_cast<unsigned>( std::countr_zero( survivors ) ) );
            survivors &= survivors - 1U;
        }
        return true;
    }
    [[nodiscard]] bool ones( std::size_t const first_index, std::size_t const count ) noexcept {
        auto const end{ ( first_index + count ) * 64U };
        for ( auto value{ first_index * 64U }; value != end; ++value ) {
            *out++ = static_cast<low_type>( value );
        }
        return true;
    }
};

// Run minus bitset, in the smallest of its three forms.
template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> difference_run_bitset(
    run_cref<Layout, CowPolicy> const lhs,
    bitset_cref<Layout, CowPolicy> const rhs,
    std::size_t const bitset_threshold
) {
    using handle_type = container_handle<Layout, CowPolicy>;
    using run_type    = ::frsr::roaring::run<typename Layout::low_type>;
    constexpr auto bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    // More runs than this take more bytes than a bitset does.
    constexpr auto run_limit{ bitset_bytes / sizeof( run_type ) };
    auto const & bits{ rhs.words.as_array() };

    run_survivor_counter counter{ .run_limit = run_limit };
    auto const counted{ scan_run_bitset_difference<Layout, CowPolicy>( lhs, bits, counter ) };
    if ( counted && counter.cardinality == 0 ) {
        return handle_type{};
    }
    auto const kind{ counted
        ? choose_run_container_kind<Layout>( counter.cardinality, counter.runs, bitset_threshold )
        : container_kind::bitset
    };

    if ( kind == container_kind::array ) {
        handle_type result;
        auto array{ result.as_array() };
        resize_uninitialized( array.values, counter.cardinality );
        array_survivor_writer<Layout> writer{ .out = array.values.data() };
        std::ignore = scan_run_bitset_difference<Layout, CowPolicy>( lhs, bits, writer );
        assert( writer.out == array.values.data() + counter.cardinality );
        array.sync_header();
        return result;
    }

    if ( kind == container_kind::run ) {
        auto result{ handle_type::make_run() };
        auto ref{ result.as_run() };
        ref.runs.resize_uninitialized( static_cast<std::uint32_t>( counter.runs ) );
        run_survivor_writer<Layout> writer{ .out = ref.runs.data() };
        std::ignore = scan_run_bitset_difference<Layout, CowPolicy>( lhs, bits, writer );
        writer.finish();
        assert( writer.written == counter.runs );
        ref.sync_header();
        return result;
    }

    // The survivors as words: the bitset's zero bits under the runs, nothing elsewhere.
    auto result{ handle_type::make_bitset_uninitialized() };
    auto * const out{ static_cast<std::uint64_t *>( result.payload_data_raw() ) };
    auto next_word{ std::size_t{ 0 } };
    for ( auto const & current : lhs.runs ) {
        auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
        auto const last_word { static_cast<std::size_t>( current.end   ) >> 6U };
        auto const first_mask{ ~std::uint64_t{ 0 } << ( current.begin & 63U ) };
        auto const last_mask { ~std::uint64_t{ 0 } >> ( 63U - ( current.end & 63U ) ) };
        if ( first_word < next_word ) {  // shares its first word with the previous run
            out[ first_word ] |= first_mask & ( first_word == last_word ? last_mask : ~std::uint64_t{ 0 } ) & ~bits[ first_word ];
        } else {
            std::fill( out + next_word, out + first_word, std::uint64_t{ 0 } );
            out[ first_word ] = first_mask & ( first_word == last_word ? last_mask : ~std::uint64_t{ 0 } ) & ~bits[ first_word ];
        }
        if ( first_word != last_word ) {
            for ( auto index{ first_word + 1U }; index < last_word; ++index ) {
                out[ index ] = ~bits[ index ];
            }
            out[ last_word ] = last_mask & ~bits[ last_word ];
        }
        next_word = last_word + 1U;
    }
    std::fill( out + next_word, out + Layout::word_count, std::uint64_t{ 0 } );
    auto const cardinality{ counted ? counter.cardinality : popcount_words( out, Layout::word_count ) };
    if ( cardinality == 0 ) {
        return handle_type{};
    }
    if ( !is_bitset_cardinality<Layout>( cardinality, bitset_threshold ) ) {
        return array_from_bitset<Layout, CowPolicy>( std::as_const( result ).as_bitset(), static_cast<std::uint32_t>( cardinality ) );
    }
    result.set_cardinality( static_cast<std::uint32_t>( cardinality ) );
    result.mark_endpoints_stale();
    return result;
}

} // namespace frsr::roaring::detail
