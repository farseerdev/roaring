#pragma once

#include <frsr/roaring/containers.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

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

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> intersect_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    if ( lhs.runs.empty() || rhs.runs.empty() ) {
        return result;
    }

    auto const is_full = []( run_cref<Layout, CowPolicy> const run ) noexcept {
        return run.runs.size() == 1U &&
            run.runs.front().begin == 0U &&
            run.runs.front().end == std::numeric_limits<typename Layout::low_type>::max();
    };
    if ( is_full( lhs ) ) {
        return run_container_from<Layout, CowPolicy>( rhs );
    }
    if ( is_full( rhs ) ) {
        return run_container_from<Layout, CowPolicy>( lhs );
    }

    result.runs.reserve( lhs.runs.size() + rhs.runs.size() );

    // Cache the payload pointer + count ONCE: `.runs[i]` on the const-view goes
    // through container_handle::payload_data_raw(), which re-derives the
    // spilled-payload pointer via std::launder on every call (the launder is a
    // hard optimizer barrier — it forbids hoisting/CSE-ing that load across
    // calls, so indexing through the view per-iteration re-pays it every time).
    // Neither lhs nor rhs is mutated for the lifetime of this loop, so the raw
    // pointer and size are loop-invariant; index those locals instead.
    auto const * const lhs_runs{ lhs.runs.data() };
    auto const lhs_size{ lhs.runs.size() };
    auto const * const rhs_runs{ rhs.runs.data() };
    auto const rhs_size{ rhs.runs.size() };

    auto li{ std::size_t{ 0 } };
    auto ri{ std::size_t{ 0 } };
    auto start{ static_cast<std::size_t>( lhs_runs[ li ].begin ) };
    auto end{ static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U };
    auto xstart{ static_cast<std::size_t>( rhs_runs[ ri ].begin ) };
    auto xend{ static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U };

    while ( li < lhs_size && ri < rhs_size ) {
        if ( end <= xstart ) {
            ++li;
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
            continue;
        }
        if ( xend <= start ) {
            ++ri;
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
            if ( li < lhs_size ) {
                start = static_cast<std::size_t>( lhs_runs[ li ].begin );
                end = static_cast<std::size_t>( lhs_runs[ li ].end ) + 1U;
            }
        } else {
            earliest_end = xend;
            ++ri;
            if ( ri < rhs_size ) {
                xstart = static_cast<std::size_t>( rhs_runs[ ri ].begin );
                xend = static_cast<std::size_t>( rhs_runs[ ri ].end ) + 1U;
            }
        }

        result.runs.push_back( run<typename Layout::low_type>{
            static_cast<typename Layout::low_type>( begin ),
            static_cast<typename Layout::low_type>( earliest_end - 1U )
        } );
        result.cardinality += earliest_end - begin;
    }
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
[[nodiscard]] inline run_container<Layout> difference_run_run(
    run_cref<Layout, CowPolicy> const lhs,
    run_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    result.runs.reserve( lhs.runs.size() );

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
                result.runs.push_back( run<typename Layout::low_type>{ current_begin, end } );
                result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( current_begin ) + 1U;
            }

            if ( scan->end == std::numeric_limits<typename Layout::low_type>::max() || scan->end >= left.end ) {
                exhausted = true;
                break;
            }

            current_begin = static_cast<typename Layout::low_type>( scan->end + 1U );
            ++scan;
        }

        if ( !exhausted && current_begin <= left.end ) {
            result.runs.push_back( run<typename Layout::low_type>{ current_begin, left.end } );
            result.cardinality += static_cast<std::size_t>( left.end ) - static_cast<std::size_t>( current_begin ) + 1U;
        }
    }
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline run_container<Layout> difference_run_array(
    run_cref<Layout, CowPolicy> const lhs,
    array_cref<Layout, CowPolicy> const rhs
) {
    run_container<Layout> result;
    result.runs.reserve( lhs.runs.size() + rhs.values.size() );

    auto value_it{ rhs.values.begin() };
    for ( auto const & run : lhs.runs ) {
        while ( value_it != rhs.values.end() && *value_it < run.begin ) {
            ++value_it;
        }

        auto current_begin{ run.begin };
        auto exhausted{ false };
        auto scan{ value_it };
        while ( scan != rhs.values.end() && *scan <= run.end ) {
            if ( current_begin < *scan ) {
                auto const end{ static_cast<typename Layout::low_type>( *scan - 1U ) };
                result.runs.push_back( frsr::roaring::run<typename Layout::low_type>{ current_begin, end } );
                result.cardinality += static_cast<std::size_t>( end ) - static_cast<std::size_t>( current_begin ) + 1U;
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
            result.runs.push_back( frsr::roaring::run<typename Layout::low_type>{ current_begin, run.end } );
            result.cardinality += static_cast<std::size_t>( run.end ) - static_cast<std::size_t>( current_begin ) + 1U;
        }
    }
    return result;
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline container_handle<Layout, CowPolicy> make_container_from_runs(
    run_container<Layout> run,
    std::size_t const bitset_threshold
) {
    if ( run.size() == 0 ) {
        return container_handle<Layout, CowPolicy>{};
    }

    auto const array_bytes{ run.size() * sizeof( typename Layout::low_type ) };
    auto const run_bytes{ run.runs.size() * sizeof( typename run_container<Layout>::run ) };
    auto const bitset_bytes{ Layout::word_count * sizeof( std::uint64_t ) };
    if ( run.size() >= bitset_threshold && bitset_bytes <= array_bytes && bitset_bytes <= run_bytes ) {
        word_array<Layout> words{};
        for ( auto const & current : run.runs ) {
            apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::add );
        }
        return make_container_from_words<Layout, CowPolicy>( words, bitset_threshold );
    }
    if ( run_bytes <= array_bytes ) {
        return handle_from_run_container<Layout, CowPolicy>( run );
    }

    // Expand the runs into a sorted array handle.
    container_handle<Layout, CowPolicy> result;
    auto array{ result.as_array() };
    resize_uninitialized( array.values, run.size() );
    auto * out{ array.values.data() };
    run.for_each( [&]( typename Layout::low_type const value ) { *out++ = value; } );
    array.sync_header();
    return result;
}

} // namespace frsr::roaring::detail
