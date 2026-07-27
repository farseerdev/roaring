#pragma once

#include <frsr/roaring/container_layout.hpp>
#include <frsr/roaring/run.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>

#ifdef FRSR_ROARING_HAS_PSI_VM
#   include <psi/vm/containers/heap_vector.hpp>
#else
#   include <vector>
#endif

namespace frsr::roaring {

namespace detail {
#ifdef FRSR_ROARING_HAS_PSI_VM
    template <typename T>
    using heap_vector = psi::vm::heap_vector<T, std::uint32_t>;
#else
    template <typename T>
    using heap_vector = std::vector<T>;
#endif
} // namespace detail

template <std::unsigned_integral LowType>
[[nodiscard]] inline int32_t interleaved_binary_search(
    run<LowType> const * array,
    int32_t lenarray,
    LowType const ikey
);

template <std::unsigned_integral Key>
struct run_container {
    using layout_type = container_layout<Key>;
    using low_type = typename layout_type::low_type;
    using run_type = run<low_type>;
    using run = run_type;
    using cardinality_type = std::conditional_t<
        ( layout_type::low_domain_size <= std::numeric_limits<std::uint16_t>::max() ),
        std::uint16_t,
        std::uint32_t
    >;

    [[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] bool contains( low_type const value ) const noexcept {
        // interleaved_binary_search returns the run index when value exactly equals a run's
        // begin, else -(insertion_point + 1). A run covers the closed interval [begin, end],
        // so on an inexact match we must still test the run immediately before the insertion
        // point (the last run with begin < value) against its end.
        // [croaring-ref] deps/croaring/src/containers/run.c:run_container_contains
        auto index{ interleaved_binary_search( runs.data(), static_cast<int32_t>( runs.size() ), value ) };
        if ( index >= 0 ) {
            return true;
        }
        index = -index - 2;
        if ( index < 0 ) {
            return false;
        }
        return value <= runs[ static_cast<std::uint32_t>( index ) ].end;
    }

    [[nodiscard]] bool add( low_type const value ) {
        auto const previous_size{ cardinality };
        add_closed_range( value, value );
        return cardinality != previous_size;
    }

    [[nodiscard]] bool remove( low_type const value ) {
        auto const previous_size{ cardinality };
        remove_closed_range( value, value );
        return cardinality != previous_size;
    }

    void add_closed_range( low_type const begin, low_type const end ) {
        if ( begin > end ) {
            return;
        }

        detail::heap_vector<run_type> merged;
        merged.reserve( static_cast<std::uint32_t>( runs.size() + 1U ) );

        auto new_begin{ begin };
        auto new_end{ end };
        auto inserted{ false };
        auto const separated_before{ []( run_type const & current, low_type const next_begin ) noexcept {
            return static_cast<std::size_t>( current.end ) + 1U < static_cast<std::size_t>( next_begin );
        } };
        auto const separated_after{ []( low_type const previous_end, run_type const & current ) noexcept {
            return static_cast<std::size_t>( previous_end ) + 1U < static_cast<std::size_t>( current.begin );
        } };

        for ( auto const & current : runs ) {
            if ( separated_before( current, begin ) ) {
                merged.push_back( current );
                continue;
            }
            if ( separated_after( end, current ) ) {
                if ( !inserted ) {
                    merged.push_back( run_type{ new_begin, new_end } );
                    inserted = true;
                }
                merged.push_back( current );
                continue;
            }
            new_begin = std::min( new_begin, current.begin );
            new_end = std::max( new_end, current.end );
        }

        if ( !inserted ) {
            merged.push_back( run_type{ new_begin, new_end } );
        }

        runs.swap( merged );
        recalculate_cardinality();
    }

    void remove_closed_range( low_type const begin, low_type const end ) {
        if ( begin > end || runs.empty() ) {
            return;
        }

        detail::heap_vector<run_type> updated;
        updated.reserve( static_cast<std::uint32_t>( runs.size() + 1U ) );
        for ( auto const & current : runs ) {
            if ( current.end < begin || end < current.begin ) {
                updated.push_back( current );
                continue;
            }

            if ( current.begin < begin ) {
                updated.push_back( run_type{ current.begin, static_cast<low_type>( begin - 1U ) } );
            }
            if ( end < current.end && end != std::numeric_limits<low_type>::max() ) {
                updated.push_back( run_type{ static_cast<low_type>( end + 1U ), current.end } );
            }
        }

        runs.swap( updated );
        recalculate_cardinality();
    }

    [[nodiscard]] std::size_t size() const noexcept { return static_cast<std::size_t>( cardinality ); }

    [[nodiscard]] std::optional<low_type> first() const noexcept {
        if ( runs.empty() ) {
            return std::nullopt;
        }
        return runs.front().begin;
    }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        auto const it{ std::ranges::lower_bound( runs, value, {}, &run_type::end ) };
        if ( it == runs.end() ) {
            return std::nullopt;
        }
        if ( value < it->begin ) {
            return it->begin;
        }
        if ( value < it->end && value != std::numeric_limits<low_type>::max() ) {
            return static_cast<low_type>( value + 1U );
        }
        auto next{ it + 1 };
        if ( next == runs.end() ) {
            return std::nullopt;
        }
        return next->begin;
    }

    template <typename F>
    void for_each( F && f ) const {
        for ( auto const & current : runs ) {
            for ( auto value{ current.begin }; ; ++value ) {
                f( value );
                if ( value == current.end ) {
                    break;
                }
            }
        }
    }

    [[nodiscard]] static run_container from_sorted_values( std::span<low_type const> const sorted_values ) {
        run_container result;
        if ( sorted_values.empty() ) {
            return result;
        }

        auto current_begin{ sorted_values.front() };
        auto current_end{ current_begin };
        for ( auto index{ std::size_t{ 1 } }; index < sorted_values.size(); ++index ) {
            auto const value{ sorted_values[ index ] };
            if ( value == current_end || ( current_end != std::numeric_limits<low_type>::max() && value == static_cast<low_type>( current_end + 1U ) ) ) {
                current_end = value;
                continue;
            }
            result.runs.push_back( run_type{ current_begin, current_end } );
            current_begin = value;
            current_end = value;
        }
        result.runs.push_back( run_type{ current_begin, current_end } );
        result.recalculate_cardinality();
        return result;
    }

    void recalculate_cardinality() noexcept {
        cardinality = 0;
        for ( auto const & current : runs ) {
            cardinality += static_cast<cardinality_type>(
                static_cast<std::size_t>( current.end ) - static_cast<std::size_t>( current.begin ) + 1U
            );
        }
    }

    detail::heap_vector<run_type> runs;
    cardinality_type cardinality{};
};

template <std::unsigned_integral LowType>
[[nodiscard]] inline int32_t interleaved_binary_search(
    run<LowType> const * array,
    int32_t lenarray,
    LowType const ikey
) {
    int32_t low = 0;
    int32_t high = lenarray - 1;
    while ( low <= high ) {
        int32_t middleIndex = ( low + high ) >> 1;
        auto const middleValue{ array[ middleIndex ].begin };
        if ( middleValue < ikey ) {
            low = middleIndex + 1;
        } else if ( middleValue > ikey ) {
            high = middleIndex - 1;
        } else {
            return middleIndex;
        }
    }
    return -( low + 1 );
}

} // namespace frsr::roaring
