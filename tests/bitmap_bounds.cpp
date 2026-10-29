// front(), back() and the key-interval intersection of a bitmap against a sorted vector, over every container kind
// a chunk can hold: array, bitset, run, and a bitset that holds few values, all of them in its first word (what a
// lazy union of small bitmaps leaves behind). The last value of a bitset is found by a scan from the top word
// down, which has to read word 0 as well; the reference library's bitset maximum never reads it, so the answers
// here are those of the sorted vector, not of that library.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring;

template <typename Key, typename Cow>
using bitmap_of = bitmap<Key, default_container_set<Key>, Cow>;

template <typename Bitmap>
class BitmapBounds : public ::testing::Test {};

using bounds_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint64_t, detail::cow_value_semantics>,
    bitmap_of<std::uint64_t, detail::cow_atomic_refcount>
>;
TYPED_TEST_SUITE( BitmapBounds, bounds_bitmaps );

using values_type = std::vector<std::uint64_t>;

constexpr std::uint64_t chunk_span{ 65'536 };

template <typename Bitmap>
[[nodiscard]] Bitmap make_bitmap( values_type const & values ) {
    Bitmap result;
    for ( auto const value : values ) {
        std::ignore = result.add( static_cast<typename Bitmap::key_type>( value ) );
    }
    return result;
}

void sort_unique( values_type & values ) {
    std::ranges::sort( values );
    values.erase( std::ranges::unique( values ).begin(), values.end() );
}

// The ends and the neighbours of the held values, and the edges of the words and the chunks around them.
[[nodiscard]] values_type interval_points( values_type const & values ) {
    values_type points{ 0, 1, 63, 64, 65 };
    auto const around{ [ & ]( std::uint64_t const value ) {
        for ( std::uint64_t delta{ 0 }; delta < 5; ++delta ) {
            if ( value + delta >= 2 ) {
                points.push_back( value + delta - 2 );
            }
        }
    } };
    around( values.front() );
    around( values.back() );
    around( values[ values.size() / 2 ] );
    around( values.back() & ~std::uint64_t{ 63 } );
    around( values.back() & ~( chunk_span - 1 ) );
    sort_unique( points );
    return points;
}

// Everything the bounds queries answer, against the sorted vector `values` (not empty).
template <typename Bitmap>
void expect_bounds( Bitmap const & bitmap, values_type const & values, std::string const & what ) {
    SCOPED_TRACE( what );
    using key_type = typename Bitmap::key_type;
    ASSERT_FALSE( values.empty() );
    EXPECT_EQ( bitmap.front(), static_cast<key_type>( values.front() ) );
    EXPECT_EQ( bitmap.back (), static_cast<key_type>( values.back () ) );
    auto const points{ interval_points( values ) };
    for ( auto const lo : points ) {
        for ( auto const hi : points ) {
            auto const first{ std::ranges::lower_bound( values, lo ) };
            auto const half_open{ lo < hi && first != values.end() && *first < hi };
            auto const closed   { lo <= hi && first != values.end() && *first <= hi };
            EXPECT_EQ( bitmap.intersects( key_range::half_open( lo, hi ) ), half_open ) << '[' << lo << ", " << hi << ')';
            EXPECT_EQ( bitmap.intersects_range( static_cast<key_type>( lo ), static_cast<key_type>( hi ) ), half_open ) << '[' << lo << ", " << hi << ')';
            EXPECT_EQ( bitmap.intersects( key_range::closed( lo, hi ) ), closed ) << '[' << lo << ", " << hi << ']';
        }
    }
}

template <typename Bitmap>
void expect_bounds_of_copies( Bitmap const & bitmap, values_type const & values, std::string const & what ) {
    expect_bounds( bitmap, values, what );
    Bitmap const copy{ bitmap };
    expect_bounds( copy, values, what + " (copy)" );
    EXPECT_EQ( bitmap.validate(), nullptr );
}

[[nodiscard]] values_type random_values( std::mt19937_64 & rng, std::uint64_t const base, std::size_t const count ) {
    values_type values;
    while ( values.size() < count ) {
        values.push_back( base + rng() % chunk_span );
        if ( values.size() % 512 == 0 ) {
            sort_unique( values );
        }
    }
    sort_unique( values );
    while ( values.size() > count ) {
        values.erase( values.begin() + static_cast<std::ptrdiff_t>( rng() % values.size() ) );
    }
    return values;
}

[[nodiscard]] values_type span_of( std::uint64_t const first, std::uint64_t const last ) {
    values_type values;
    for ( auto value{ first }; value <= last; ++value ) {
        values.push_back( value );
    }
    return values;
}

// A bitset holding only `values` (below 5000): the chunk is filled past the array limit, which makes it a bitset, and
// everything else is removed again, which leaves it one.
template <typename Bitmap>
[[nodiscard]] Bitmap emptied_bitset( values_type const & values ) {
    using key_type = typename Bitmap::key_type;
    Bitmap result;
    for ( key_type value{ 0 }; value < 5000; ++value ) {
        std::ignore = result.add( value );
    }
    for ( key_type value{ 0 }; value < 5000; ++value ) {
        if ( !std::ranges::binary_search( values, static_cast<std::uint64_t>( value ) ) ) {
            std::ignore = result.remove( value );
        }
    }
    return result;
}

TYPED_TEST( BitmapBounds, ArrayContainers ) {
    using Bitmap = TypeParam;
    for ( values_type const & values : std::vector<values_type>{
        { 0 }, { 0, 1 }, { 3, 5 }, { 65'535 }, { 7, 65'535 }, { 65'536 }, { 5, 70'000, 200'000 }, { 0, 63, 64, 4000 }
    } ) {
        auto const bitmap{ make_bitmap<Bitmap>( values ) };
        expect_bounds_of_copies( bitmap, values, "array of " + std::to_string( values.size() ) + " values from " + std::to_string( values.front() ) );
    }
    std::mt19937_64 rng{ 17 };
    auto const values{ random_values( rng, 3 * chunk_span, 4000 ) };
    auto const bitmap{ make_bitmap<Bitmap>( values ) };
    EXPECT_EQ( bitmap.statistics().array_containers, 1U );
    expect_bounds_of_copies( bitmap, values, "array of 4000" );
}

TYPED_TEST( BitmapBounds, BitsetContainers ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 29 };
    for ( auto const count : { std::size_t{ 4097 }, std::size_t{ 20'000 }, std::size_t{ 60'000 } } ) {
        auto const values{ random_values( rng, 2 * chunk_span, count ) };
        auto const bitmap{ make_bitmap<Bitmap>( values ) };
        EXPECT_EQ( bitmap.statistics().bitset_containers, 1U );
        expect_bounds_of_copies( bitmap, values, "bitset of " + std::to_string( count ) );
    }
    // the last value in the first words, in a middle one and in the last one of the chunk
    for ( auto const last : { std::uint64_t{ 4096 }, std::uint64_t{ 4100 }, std::uint64_t{ 30'000 }, std::uint64_t{ 65'535 - 63 }, std::uint64_t{ 65'535 } } ) {
        auto const values{ span_of( last - 4096, last ) };
        auto const bitmap{ make_bitmap<Bitmap>( values ) };
        EXPECT_EQ( bitmap.statistics().bitset_containers, 1U );
        expect_bounds_of_copies( bitmap, values, "bitset ending at " + std::to_string( last ) );
    }
}

TYPED_TEST( BitmapBounds, RunContainers ) {
    using Bitmap = TypeParam;
    for ( auto const & [ first, last ] : std::vector<std::pair<std::uint64_t, std::uint64_t>>{ { 0, 99 }, { 100, 5000 }, { 60'000, 65'535 }, { 65'536 + 7, 65'536 + 107 } } ) {
        auto const values{ span_of( first, last ) };
        auto bitmap{ make_bitmap<Bitmap>( values ) };
        bitmap.optimize();
        EXPECT_EQ( bitmap.statistics().run_containers, 1U ) << first << ' ' << last;
        expect_bounds_of_copies( bitmap, values, "run " + std::to_string( first ) + ".." + std::to_string( last ) );
    }
    // several runs, the last one short
    values_type values;
    for ( auto const base : { std::uint64_t{ 10 }, std::uint64_t{ 1000 }, std::uint64_t{ 30'000 }, std::uint64_t{ 65'000 } } ) {
        auto const run{ span_of( base, base + 299 ) };
        values.insert( values.end(), run.begin(), run.end() );
    }
    values.push_back( 65'535 );
    auto bitmap{ make_bitmap<Bitmap>( values ) };
    bitmap.optimize();
    EXPECT_EQ( bitmap.statistics().run_containers, 1U );
    expect_bounds_of_copies( bitmap, values, "runs" );
}

TYPED_TEST( BitmapBounds, SmallBitsetsOfTheFirstWord ) {
    using Bitmap = TypeParam;
    for ( values_type const & values : std::vector<values_type>{
        { 0 }, { 0, 1 }, { 3 }, { 3, 5 }, { 63 }, { 62, 63 }, { 0, 63 }, { 7, 9, 1000 }, { 4999 }, { 64, 4000 }
    } ) {
        auto const bitmap{ emptied_bitset<Bitmap>( values ) };
        EXPECT_EQ( bitmap.statistics().bitset_containers, 1U ) << "the scenario holds a bitset";
        expect_bounds_of_copies( bitmap, values, "small bitset from " + std::to_string( values.front() ) + " to " + std::to_string( values.back() ) );
    }
}

TYPED_TEST( BitmapBounds, ManyChunksTheLastOfEachKind ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 41 };
    auto const dense { random_values( rng, 0, 9000 ) };
    auto const sparse{ random_values( rng, 5 * chunk_span, 20 ) };
    auto const run   { span_of( 9 * chunk_span + 100, 9 * chunk_span + 4000 ) };
    auto const small { values_type{ 12 * chunk_span + 3, 12 * chunk_span + 5 } };
    for ( auto const & last_chunk : { sparse, run, small } ) {
        auto values{ dense };
        values.insert( values.end(), last_chunk.begin(), last_chunk.end() );
        auto bitmap{ make_bitmap<Bitmap>( values ) };
        bitmap.optimize();
        expect_bounds_of_copies( bitmap, values, "dense chunk followed by a chunk of " + std::to_string( last_chunk.size() ) );
    }
    // removing the values of the last chunk moves the answer to the chunk before it
    auto bitmap{ make_bitmap<Bitmap>( small ) };
    std::ignore = bitmap.add( 4 );
    EXPECT_EQ( bitmap.back(), static_cast<typename Bitmap::key_type>( small.back() ) );
    for ( auto const value : small ) {
        EXPECT_TRUE( bitmap.remove( static_cast<typename Bitmap::key_type>( value ) ) );
    }
    EXPECT_EQ( bitmap.back(), 4U );
    EXPECT_EQ( bitmap.front(), 4U );
}

TYPED_TEST( BitmapBounds, SharedContainersKeepTheirOwnBounds ) {
    using Bitmap = TypeParam;
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 53 };
    auto values{ random_values( rng, chunk_span, 6000 ) };
    std::erase( values, 2 * chunk_span - 1 );
    auto original{ make_bitmap<Bitmap>( values ) };
    Bitmap const copy{ original };
    // growing and shrinking the original must not change what the copy answers, nor the reverse
    std::ignore = original.add( static_cast<key_type>( 2 * chunk_span - 1 ) );
    auto grown{ values };
    grown.push_back( 2 * chunk_span - 1 );
    sort_unique( grown );
    expect_bounds( original, grown, "original after the add" );
    expect_bounds( copy, values, "copy after the add to the original" );
    EXPECT_TRUE( original.remove( static_cast<key_type>( 2 * chunk_span - 1 ) ) );
    EXPECT_EQ( original.back(), static_cast<key_type>( values.back() ) );
    EXPECT_TRUE( original.remove( static_cast<key_type>( values.back() ) ) );
    EXPECT_EQ( original.back(), static_cast<key_type>( values[ values.size() - 2 ] ) );
    expect_bounds( copy, values, "copy after the removals from the original" );
}

// tests/toplevel_unit.c test_maximum_minimum: one bitmap grown from an array to a bitset to a run container
TYPED_TEST( BitmapBounds, UpstreamMaximumMinimum ) {
    using Bitmap = TypeParam;
    using key_type = typename Bitmap::key_type;
    for ( std::uint32_t first{ 123 }; first < 1'000'000; first *= 2 ) {
        Bitmap bitmap;
        std::uint32_t x{ first };
        for ( ; x < 1000 + first; x += 100 ) {
            std::ignore = bitmap.add( static_cast<key_type>( x ) );
        }
        EXPECT_EQ( bitmap.front(), static_cast<key_type>( first ) );
        EXPECT_EQ( bitmap.back (), static_cast<key_type>( x - 100 ) );
        x = first;
        for ( ; x < 64'000 + first; x += 2 ) {
            std::ignore = bitmap.add( static_cast<key_type>( x ) );
        }
        EXPECT_EQ( bitmap.front(), static_cast<key_type>( first ) );
        EXPECT_EQ( bitmap.back (), static_cast<key_type>( x - 2 ) );
        x = first;
        for ( ; x < 64'000 + first; ++x ) {
            std::ignore = bitmap.add( static_cast<key_type>( x ) );
        }
        bitmap.optimize();
        EXPECT_EQ( bitmap.front(), static_cast<key_type>( first ) );
        EXPECT_EQ( bitmap.back (), static_cast<key_type>( x - 1 ) );
    }
}

TYPED_TEST( BitmapBounds, EmptyBitmapHasNoIntersection ) {
    using Bitmap = TypeParam;
    Bitmap const empty;
    EXPECT_FALSE( empty.intersects( key_range::all() ) );
    EXPECT_FALSE( empty.intersects_range( 0, 100 ) );
    EXPECT_TRUE( empty.empty() );
}

} // namespace
