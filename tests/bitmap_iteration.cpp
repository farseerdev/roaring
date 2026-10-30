// The value iterator and for_each() of a bitmap, against the sorted vector of the values it holds, over every container
// kind a chunk can hold (array, bitset, run) and their neighbours: single-value chunks, a full chunk, runs that end at the
// last value of a chunk, and a bitset whose values all sit in one word. The iterator walks a container with a cursor of
// its own, so the steps at the ends of the containers (the last value of a run, of an array, of a word) are what is checked.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace frsr::roaring;

template <typename Key, typename Cow>
using bitmap_of = bitmap<Key, default_container_set<Key>, Cow>;

template <typename Bitmap>
class BitmapIteration : public ::testing::Test {};

using iteration_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint64_t, detail::cow_value_semantics>,
    bitmap_of<std::uint64_t, detail::cow_atomic_refcount>
>;
TYPED_TEST_SUITE( BitmapIteration, iteration_bitmaps );

using values_type = std::vector<std::uint64_t>;

constexpr std::uint64_t chunk_span{ 65'536 };

void append_span( values_type & values, std::uint64_t const first, std::uint64_t const last ) {
    for ( auto value{ first }; value <= last; ++value ) {
        values.push_back( value );
    }
}

template <typename Bitmap>
void expect_iteration( Bitmap & bitmap, values_type const & values, std::string const & what ) {
    SCOPED_TRACE( what );
    using key_type = typename Bitmap::key_type;
    auto const check{ [ & ]( Bitmap const & held, std::string const & which ) {
        SCOPED_TRACE( which );
        std::vector<std::uint64_t> walked;
        for ( auto const value : held ) {
            walked.push_back( static_cast<std::uint64_t>( value ) );
        }
        EXPECT_EQ( walked, values );

        std::vector<std::uint64_t> visited;
        held.for_each( [ & ]( key_type const value ) { visited.push_back( static_cast<std::uint64_t>( value ) ); } );
        EXPECT_EQ( visited, values );

        EXPECT_EQ( static_cast<std::size_t>( std::distance( held.begin(), held.end() ) ), values.size() );
    } };
    check( bitmap, "as built" );
    bitmap.optimize();
    check( bitmap, "optimized" );
    Bitmap const copy{ bitmap };
    check( copy, "copy" );
}

template <typename Bitmap>
[[nodiscard]] Bitmap make_bitmap( values_type const & values ) {
    Bitmap result;
    for ( auto const value : values ) {
        std::ignore = result.add( static_cast<typename Bitmap::key_type>( value ) );
    }
    return result;
}

} // namespace

TYPED_TEST( BitmapIteration, Empty ) {
    TypeParam const bitmap;
    EXPECT_TRUE( bitmap.begin() == bitmap.end() );
}

TYPED_TEST( BitmapIteration, SingleValueChunks ) {
    values_type const values{ 0, chunk_span - 1, chunk_span, 3 * chunk_span + 7, 5 * chunk_span + chunk_span - 1 };
    auto bitmap{ make_bitmap<TypeParam>( values ) };
    expect_iteration( bitmap, values, "single values" );
}

TYPED_TEST( BitmapIteration, StridedArrayChunks ) {
    values_type values;
    for ( std::uint64_t chunk{ 0 }; chunk < 3; ++chunk ) {
        for ( std::uint64_t index{ 0 }; index < 1'000; ++index ) {
            values.push_back( chunk * chunk_span + index * 61 + chunk );
        }
    }
    auto bitmap{ make_bitmap<TypeParam>( values ) };
    expect_iteration( bitmap, values, "strided array chunks" );
}

TYPED_TEST( BitmapIteration, DenseBitsetChunks ) {
    values_type values;
    for ( std::uint64_t chunk{ 0 }; chunk < 2; ++chunk ) {
        for ( std::uint64_t index{ 0 }; index < chunk_span; index += 2 ) {
            values.push_back( chunk * chunk_span + index + chunk );
        }
    }
    values.push_back( 2 * chunk_span - 1 );
    std::ranges::sort( values );
    values.erase( std::ranges::unique( values ).begin(), values.end() );
    auto bitmap{ make_bitmap<TypeParam>( values ) };
    expect_iteration( bitmap, values, "dense bitset chunks" );
}

TYPED_TEST( BitmapIteration, BitsetWithValuesInOneWord ) {
    values_type values;
    for ( std::uint64_t value{ 0 }; value < 64; value += 3 ) {
        values.push_back( value );
    }
    // enough values elsewhere to make the chunk a bitset, then removed again: the words past the first are empty
    TypeParam bitmap;
    for ( std::uint64_t value{ 0 }; value < chunk_span; value += 2 ) {
        std::ignore = bitmap.add( static_cast<typename TypeParam::key_type>( value ) );
    }
    for ( std::uint64_t value{ 64 }; value < chunk_span; value += 2 ) {
        std::ignore = bitmap.remove( static_cast<typename TypeParam::key_type>( value ) );
    }
    for ( std::uint64_t value{ 0 }; value < 64; value += 2 ) {
        std::ignore = bitmap.remove( static_cast<typename TypeParam::key_type>( value ) );
    }
    for ( std::uint64_t value{ 1 }; value < 64; value += 2 ) {
        std::ignore = bitmap.add( static_cast<typename TypeParam::key_type>( value ) );
    }
    values.clear();
    for ( std::uint64_t value{ 1 }; value < 64; value += 2 ) {
        values.push_back( value );
    }
    expect_iteration( bitmap, values, "bitset with values in one word" );
}

TYPED_TEST( BitmapIteration, RunsAndFullChunk ) {
    values_type values;
    append_span( values, 10, 20 );
    append_span( values, 22, 22 );
    append_span( values, 30'000, 40'000 );
    append_span( values, chunk_span - 3, chunk_span - 1 );
    append_span( values, chunk_span, chunk_span + 4 );
    append_span( values, 2 * chunk_span, 3 * chunk_span - 1 );
    append_span( values, 4 * chunk_span + 100, 4 * chunk_span + 100 );
    auto bitmap{ make_bitmap<TypeParam>( values ) };
    expect_iteration( bitmap, values, "runs and a full chunk" );
}

TYPED_TEST( BitmapIteration, RandomMixOfKinds ) {
    std::mt19937_64 rng{ 0x17E4A7U };
    for ( int round{ 0 }; round < 6; ++round ) {
        values_type values;
        for ( std::uint64_t chunk{ 0 }; chunk < 12; ++chunk ) {
            switch ( rng() % 4 ) {
                case 0:
                case 1: {
                    auto const count{ rng() % 2 == 0 ? 1 + rng() % 3'000 : 20'000 + rng() % 40'000 };
                    for ( std::size_t added{ 0 }; added < count; ++added ) {
                        values.push_back( chunk * chunk_span + rng() % chunk_span );
                    }
                    break;
                }
                case 2: {
                    auto const first{ rng() % ( chunk_span - 1 ) };
                    append_span( values, chunk * chunk_span + first, chunk * chunk_span + first + rng() % ( chunk_span - first ) );
                    break;
                }
                default:
                    break;
            }
        }
        std::ranges::sort( values );
        values.erase( std::ranges::unique( values ).begin(), values.end() );
        auto bitmap{ make_bitmap<TypeParam>( values ) };
        expect_iteration( bitmap, values, "round " + std::to_string( round ) );
    }
}
