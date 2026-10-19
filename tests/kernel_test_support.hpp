#pragma once

// Shared generators and comparison helpers for the container-kernel tests
// (kernel_coverage.cpp, kernel_crosscheck.cpp) and the contains tests.
//
// Everything here produces plain sorted vectors: the oracle for every test is a
// std:: algorithm over those vectors, never a second bitmap implementation.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace frsr::roaring::kernel_test {

inline constexpr std::size_t chunk_span{ 65'536 };

enum class pattern : std::uint8_t { random, blocks64, one_run, every_third, empty, tiny };

inline constexpr std::array all_patterns{
    pattern::random, pattern::blocks64, pattern::one_run, pattern::every_third, pattern::empty, pattern::tiny
};

[[nodiscard]] inline char const * name_of( pattern const p ) noexcept {
    switch ( p ) {
        case pattern::random:      return "random";
        case pattern::blocks64:    return "blocks64";
        case pattern::one_run:     return "one_run";
        case pattern::every_third: return "every_third";
        case pattern::empty:       return "empty";
        case pattern::tiny:        return "tiny";
    }
    return "?";
}

// `n` distinct values of [0, 65536), sorted. Requires n <= 4096 for the structured
// patterns (the array-container regime they model).
[[nodiscard]] inline std::vector<std::uint16_t> make_pattern( pattern const p, std::size_t n, std::mt19937_64 & rng ) {
    std::vector<std::uint16_t> values;
    switch ( p ) {
        case pattern::empty:
            break;
        case pattern::tiny:
            n = 1 + rng() % 3;
            [[fallthrough]];
        case pattern::random: {
            std::vector<bool> taken( chunk_span );
            while ( values.size() < n ) {
                auto const v{ static_cast<std::uint16_t>( rng() ) };
                if ( !taken[ v ] ) { taken[ v ] = true; values.push_back( v ); }
            }
            std::ranges::sort( values );
            break;
        }
        case pattern::blocks64: {
            // Whole aligned 64-value blocks, i.e. the word-shaped input a bitset
            // kernel and an array merge kernel both see as dense.
            auto const blocks{ ( n + 63 ) / 64 };
            std::vector<bool> taken( chunk_span / 64 );
            std::size_t picked{ 0 };
            while ( picked < blocks ) {
                auto const b{ static_cast<std::size_t>( rng() % taken.size() ) };
                if ( !taken[ b ] ) { taken[ b ] = true; ++picked; }
            }
            for ( std::size_t b{ 0 }; b < taken.size(); ++b ) {
                if ( taken[ b ] ) {
                    for ( std::size_t i{ 0 }; i < 64; ++i ) { values.push_back( static_cast<std::uint16_t>( b * 64 + i ) ); }
                }
            }
            break;
        }
        case pattern::one_run: {
            auto const start{ n == 0 ? std::size_t{ 0 } : static_cast<std::size_t>( rng() % ( chunk_span - n + 1 ) ) };
            for ( std::size_t i{ 0 }; i < n; ++i ) { values.push_back( static_cast<std::uint16_t>( start + i ) ); }
            break;
        }
        case pattern::every_third: {
            auto const start{ static_cast<std::size_t>( rng() % ( chunk_span - 3 * n ) ) };
            for ( std::size_t i{ 0 }; i < n; ++i ) { values.push_back( static_cast<std::uint16_t>( start + 3 * i ) ); }
            break;
        }
    }
    return values;
}

// Distinct sorted values of one chunk, composed with the chunk's base key.
template <typename Key>
[[nodiscard]] std::vector<Key> in_chunk( std::vector<std::uint16_t> const & lows, Key const base ) {
    std::vector<Key> values;
    values.reserve( lows.size() );
    for ( auto const low : lows ) { values.push_back( static_cast<Key>( base + low ) ); }
    return values;
}

// `n` distinct values of [0, 65536) that are NOT in `exclude` (sorted), sorted.
[[nodiscard]] inline std::vector<std::uint16_t> random_outside(
    std::size_t const n, std::vector<std::uint16_t> const & exclude, std::mt19937_64 & rng
) {
    std::vector<bool> taken( chunk_span );
    for ( auto const v : exclude ) { taken[ v ] = true; }
    std::vector<std::uint16_t> values;
    while ( values.size() < n ) {
        auto const v{ static_cast<std::uint16_t>( rng() ) };
        if ( !taken[ v ] ) { taken[ v ] = true; values.push_back( v ); }
    }
    std::ranges::sort( values );
    return values;
}

template <typename T>
[[nodiscard]] std::vector<T> set_union_of( std::vector<T> const & a, std::vector<T> const & b ) {
    std::vector<T> r;
    std::ranges::set_union( a, b, std::back_inserter( r ) );
    return r;
}
template <typename T>
[[nodiscard]] std::vector<T> set_intersection_of( std::vector<T> const & a, std::vector<T> const & b ) {
    std::vector<T> r;
    std::ranges::set_intersection( a, b, std::back_inserter( r ) );
    return r;
}
template <typename T>
[[nodiscard]] std::vector<T> set_difference_of( std::vector<T> const & a, std::vector<T> const & b ) {
    std::vector<T> r;
    std::ranges::set_difference( a, b, std::back_inserter( r ) );
    return r;
}

template <typename Key>
[[nodiscard]] std::vector<Key> sorted_unique( std::vector<Key> values ) {
    std::ranges::sort( values );
    values.erase( std::unique( values.begin(), values.end() ), values.end() );
    return values;
}

// The bitmap holds exactly `expected`, in order, and its own invariants hold.
// `canonical` additionally demands the canonical container kinds (a function of the
// cardinality alone), which the result of a set operation is documented to settle in.
template <typename Bitmap>
[[nodiscard]] ::testing::AssertionResult matches(
    Bitmap const & bm, std::vector<typename Bitmap::key_type> const & expected, bool const canonical = false
) {
    if ( auto const * const bad{ bm.validate( canonical ) } ) {
        return ::testing::AssertionFailure() << "invariant violated: " << bad;
    }
    if ( bm.size() != expected.size() ) {
        return ::testing::AssertionFailure() << "size " << bm.size() << " != expected " << expected.size();
    }
    auto const got{ bm.to_vector() };
    auto const [g, e]{ std::ranges::mismatch( got, expected ) };
    if ( g != got.end() || e != expected.end() ) {
        return ::testing::AssertionFailure()
            << "contents differ at index " << ( g - got.begin() ) << ": got "
            << ( g != got.end() ? static_cast<unsigned long long>( *g ) : 0ULL ) << ", expected "
            << ( e != expected.end() ? static_cast<unsigned long long>( *e ) : 0ULL );
    }
    return ::testing::AssertionSuccess();
}

// How the lhs of an in-place operation came to hold its values; the payload's spare
// capacity (and, for the shuffled adds, the container's growth history) differs.
enum class layout : std::uint8_t { bulk, spare_capacity, shuffled_adds };

inline constexpr std::array all_layouts{ layout::bulk, layout::spare_capacity, layout::shuffled_adds };

// Builds a bitmap holding exactly `values` (sorted, distinct, one chunk or several).
// spare_capacity: grown past `values` with filler from the same chunk(s) and shrunk
// back by removal, so the arrays keep capacity beyond their cardinality.
template <typename Bitmap>
[[nodiscard]] Bitmap build_with_layout(
    std::vector<typename Bitmap::key_type> const & values, layout const how, std::mt19937_64 & rng
) {
    using key_type = typename Bitmap::key_type;
    Bitmap bm;
    switch ( how ) {
        case layout::bulk:
            bm.add_many( values );
            break;
        case layout::spare_capacity: {
            std::vector<key_type> filler;
            if ( !values.empty() ) {
                auto const chunk_base{ static_cast<key_type>( values.front() & ~static_cast<key_type>( 0xFFFF ) ) };
                for ( std::size_t i{ 0 }; i < 200; ++i ) {
                    auto const v{ static_cast<key_type>( chunk_base + static_cast<std::uint16_t>( rng() ) ) };
                    if ( !std::ranges::binary_search( values, v ) ) { filler.push_back( v ); }
                }
                filler = sorted_unique( std::move( filler ) );
            }
            bm.add_many( set_union_of( values, filler ) );
            bm.remove_many( filler );
            break;
        }
        case layout::shuffled_adds: {
            auto shuffled{ values };
            std::ranges::shuffle( shuffled, rng );
            for ( auto const v : shuffled ) { (void)bm.add( v ); }
            break;
        }
    }
    return bm;
}

} // namespace frsr::roaring::kernel_test
