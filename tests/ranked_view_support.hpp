// Shared by the tests of the ranked frozen image: the id sets it is exercised with, how they are built and frozen, and
// the naive oracle (a sorted vector) every answer is compared with.
#pragma once

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/ranked_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace frsr::roaring::ranked_test {

using table_bitmap = bitmap<std::uint32_t>;
using table_view   = ranked_view<std::uint32_t>;

struct shape {
    std::string                name;
    std::vector<std::uint32_t> ids;   // sorted, distinct
};

// The ids of [first, last] that survive a coin of probability 1 / range_per_entry: a table whose range holds that
// many ids' worth of space per entry.
[[nodiscard]] inline std::vector<std::uint32_t> thinned( std::uint32_t const first, std::uint64_t const last, double const range_per_entry, std::mt19937_64 & rng ) {
    std::bernoulli_distribution keep{ 1.0 / range_per_entry };
    std::vector<std::uint32_t> ids;
    for ( std::uint64_t id{ first }; id <= last; ++id ) {
        if ( keep( rng ) ) {
            ids.push_back( static_cast<std::uint32_t>( id ) );
        }
    }
    return ids;
}

[[nodiscard]] inline std::vector<std::uint32_t> span_of( std::uint64_t const first, std::uint64_t const last ) {
    std::vector<std::uint32_t> ids;
    for ( auto id{ first }; id <= last; ++id ) {
        ids.push_back( static_cast<std::uint32_t>( id ) );
    }
    return ids;
}

[[nodiscard]] inline std::vector<std::uint32_t> merged( std::vector<std::vector<std::uint32_t>> const & parts ) {
    std::vector<std::uint32_t> ids;
    for ( auto const & part : parts ) {
        ids.insert( ids.end(), part.begin(), part.end() );
    }
    std::ranges::sort( ids );
    ids.erase( std::ranges::unique( ids ).begin(), ids.end() );
    return ids;
}

[[nodiscard]] inline std::vector<shape> shapes() {
    std::mt19937_64 rng{ 20240611 };
    constexpr std::uint32_t chunk{ 65'536 };
    std::vector<shape> result;
    result.push_back( { "empty", {} } );
    result.push_back( { "single_zero", { 0U } } );
    result.push_back( { "single_mid", { 123'456U } } );
    result.push_back( { "single_max", { 0xFFFF'FFFFU } } );
    result.push_back( { "both_ends", { 0U, 0xFFFF'FFFFU } } );
    result.push_back( { "full_chunk", span_of( 3ULL * chunk, 4ULL * chunk - 1 ) } );
    result.push_back( { "full_chunk_at_the_top", span_of( 0xFFFF'0000ULL, 0xFFFF'FFFFULL ) } );
    result.push_back( { "sparse", thinned( 0, 6ULL * chunk, 590.0, rng ) } );
    result.push_back( { "random_one_in_16", thinned( 70'000, 70'000 + 8ULL * chunk, 16.0, rng ) } );
    result.push_back( { "random_dense", thinned( 0, 5ULL * chunk, 2.0, rng ) } );
    for ( auto const ratio : { 1.001, 1.01, 1.03, 1.1 } ) {
        result.push_back( { "near_contiguous_" + std::to_string( ratio ), thinned( 1'000'000, 1'000'000 + 8ULL * chunk, ratio, rng ) } );
    }
    {
        // many short runs of random lengths and gaps, ending exactly on a chunk end
        std::vector<std::uint32_t> ids;
        std::uint32_t at{ 100 };
        while ( at < 4 * chunk - 400 ) {
            auto const length{ static_cast<std::uint32_t>( 1 + rng() % 300 ) };
            for ( std::uint32_t i{ 0 }; i < length; ++i ) {
                ids.push_back( at + i );
            }
            at += length + 1 + static_cast<std::uint32_t>( rng() % 200 );
        }
        auto tail{ span_of( 4ULL * chunk - 300, 4ULL * chunk - 1 ) };
        result.push_back( { "runs", merged( { ids, tail } ) } );
    }
    {
        // array chunks whose values straddle and exceed 0x8000, which a signed 16-bit compare mishandles
        std::vector<std::uint32_t> ids;
        for ( std::uint32_t c{ 0 }; c < 3; ++c ) {
            for ( std::size_t i{ 0 }; i < 20 + 900 * c; ++i ) {
                ids.push_back( c * chunk + 0x7F00U + static_cast<std::uint32_t>( rng() % 0x8100U ) );
            }
            ids.push_back( c * chunk + 0xFFFFU );
            ids.push_back( c * chunk + 0x7FFFU );
            ids.push_back( c * chunk + 0x8000U );
        }
        std::ranges::sort( ids );
        ids.erase( std::ranges::unique( ids ).begin(), ids.end() );
        result.push_back( { "array_high_values", std::move( ids ) } );
    }
    {
        // chunks that are not consecutive: a gap between every pair, and the first and the last of the key space
        std::vector<std::vector<std::uint32_t>> parts;
        for ( std::uint32_t const c : { 0U, 5U, 6U, 9U, 200U, 40'000U, 65'535U } ) {
            parts.push_back( thinned( c * chunk, std::uint64_t{ c } * chunk + chunk - 1, 3.0, rng ) );
        }
        result.push_back( { "gapped_chunks", merged( parts ) } );
    }
    return result;
}

enum class build : std::uint8_t { plain, optimized, keep_bitsets };

[[nodiscard]] inline char const * name_of( build const how ) noexcept {
    switch ( how ) {
        case build::plain:        return "plain";
        case build::optimized:    return "optimized";
        case build::keep_bitsets: return "keep_bitsets";
    }
    return "?";
}

[[nodiscard]] inline table_bitmap make_bitmap( std::span<std::uint32_t const> const ids, build const how ) {
    table_bitmap table;
    table.add_sorted_many( ids );
    switch ( how ) {
        case build::plain:        break;
        case build::optimized:    table.optimize(); break;
        case build::keep_bitsets: table.optimize_keep_bitsets(); break;
    }
    return table;
}

// An 8-byte aligned image: the storage is words.
struct image {
    std::vector<std::uint64_t> words;
    std::size_t                size{};

    [[nodiscard]] std::span<std::byte> writable() noexcept { return { reinterpret_cast<std::byte *>( words.data() ), size }; }
    [[nodiscard]] std::span<std::byte const> bytes() const noexcept { return { reinterpret_cast<std::byte const *>( words.data() ), size }; }
};

[[nodiscard]] inline image freeze( table_bitmap const & table, frozen_options const options ) {
    image result;
    result.size = table.frozen_size_bytes( options );
    result.words.resize( ( result.size + 7 ) / 8 );
    EXPECT_EQ( table.frozen_serialize( result.writable(), options ), result.size );
    return result;
}

// What to ask a table: every id and both neighbours, the edges of the key space and of every chunk it touches, and random keys.
[[nodiscard]] inline std::vector<std::uint32_t> probes( std::span<std::uint32_t const> const ids ) {
    std::mt19937_64 rng{ 77 };
    std::vector<std::uint32_t> result{ 0U, 1U, 0xFFFF'FFFEU, 0xFFFF'FFFFU, 0x7FFF'FFFFU, 0x8000'0000U };
    for ( auto const id : ids ) {
        result.push_back( id );
        result.push_back( id - 1U );   // wraps for 0, which is just another probe
        result.push_back( id + 1U );
        auto const chunk_start{ id & 0xFFFF'0000U };
        result.push_back( chunk_start );
        result.push_back( chunk_start - 1U );
        result.push_back( chunk_start | 0xFFFFU );
        result.push_back( ( chunk_start | 0xFFFFU ) + 1U );
    }
    for ( int i{ 0 }; i < 20'000; ++i ) {
        result.push_back( ids.empty() || i % 4 == 0 ? static_cast<std::uint32_t>( rng() ) : ids[ rng() % ids.size() ] + static_cast<std::uint32_t>( rng() % 140'000 ) - 70'000U );
    }
    return result;
}

} // namespace frsr::roaring::ranked_test
