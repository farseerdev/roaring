// The size queries and the strict subset test the reference ships with (roaring_bitmap_or_cardinality,
// roaring_bitmap_andnot_cardinality, roaring_bitmap_jaccard_index, roaring_bitmap_is_strict_subset), replayed against
// bitmap, a view of a bitmap and a key interval, for the value-semantics policy and the atomic-refcount one (the
// reference runs the same tests with copy-on-write off and on). The tests carry the names of the upstream ones
// (tests/toplevel_unit.c and tests/cpp_unit.cpp) and their data.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace frsr::roaring;

template <typename Key, typename Cow>
using bitmap_of = bitmap<Key, default_container_set<Key>, Cow>;

template <typename Bitmap>
class UpstreamParity : public ::testing::Test {};

using parity_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint64_t, detail::cow_value_semantics>,
    bitmap_of<std::uint64_t, detail::cow_atomic_refcount>
>;
TYPED_TEST_SUITE( UpstreamParity, parity_bitmaps );

// tests/toplevel_unit.c: a value of the bitmap's key space that the bitmap does not hold
template <typename Bitmap>
typename Bitmap::key_type choose_missing_value( Bitmap const & bitmap, std::uint64_t const bound, std::mt19937_64 & rng ) {
    for ( ;; ) {
        auto const value{ static_cast<typename Bitmap::key_type>( rng() % bound ) };
        if ( !bitmap.contains( value ) ) { return value; }
    }
}

// every query of a bitmap against `operand`, which holds the same keys as the bitmap `same`
template <typename Bitmap, typename Operand>
void expect_sizes( Bitmap const & lhs, Operand const & operand, Bitmap const & same ) {
    auto const common{ lhs.and_cardinality( same ) };
    EXPECT_EQ( lhs.or_cardinality( operand ), ( lhs | same ).size() );
    EXPECT_EQ( lhs.or_cardinality( operand ), lhs.size() + same.size() - common );
    EXPECT_EQ( lhs.andnot_cardinality( operand ), ( lhs - same ).size() );
    EXPECT_EQ( lhs.andnot_cardinality( operand ), lhs.size() - common );
    EXPECT_EQ( lhs.andnot_cardinality( complement_of( operand ) ), common );
    EXPECT_EQ( lhs.xor_cardinality( operand ), ( lhs ^ same ).size() );
    EXPECT_EQ( lhs.is_strict_subset_of( operand ), lhs.size() < same.size() && lhs.is_subset_of( same ) );
    auto const jaccard{ lhs.jaccard_index( operand ) };
    if ( lhs.size() + same.size() == 0 ) {
        EXPECT_TRUE( std::isnan( jaccard ) );
    } else {
        EXPECT_DOUBLE_EQ( jaccard, static_cast<double>( common ) / static_cast<double>( lhs.size() + same.size() - common ) );
    }
}

template <typename Bitmap>
void expect_sizes_everywhere( Bitmap const & lhs, Bitmap const & rhs ) {
    expect_sizes( lhs, rhs, rhs );
    expect_sizes( lhs, view_of( rhs ), rhs );
}

// tests/toplevel_unit.c: test_example / test_union: r1 = {2 i}, r2 = {3 i} for i < 100, 166 keys in the union
TYPED_TEST( UpstreamParity, test_example_or_cardinality ) {
    using Bitmap = TypeParam;
    Bitmap r1, r2;
    for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
        r1.add( 2 * i );
        r2.add( 3 * i );
        ASSERT_EQ( r2.size(), i + 1 );
        ASSERT_EQ( r1.size(), i + 1 );
    }
    Bitmap const r1_or_r2{ r1 | r2 };
    EXPECT_EQ( r1_or_r2.size(), r1.or_cardinality( r2 ) );
    EXPECT_EQ( r1.or_cardinality( view_of( r2 ) ), r1_or_r2.size() );
    EXPECT_EQ( r1_or_r2.size(), 166U );
    expect_sizes_everywhere( r1, r2 );
    expect_sizes_everywhere( r2, r1 );
}

// tests/toplevel_unit.c: the bitmaps of test_union: or_cardinality is the size of the union, and_cardinality of the intersection
TYPED_TEST( UpstreamParity, test_union ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 20261022 };
    for ( int round{ 0 }; round < 6; ++round ) {
        Bitmap r1, r2;
        for ( int i{ 0 }; i < 2000; ++i ) { r1.add( static_cast<typename Bitmap::key_type>( rng() % ( 3U << 16 ) ) ); }
        for ( int i{ 0 }; i < 2000 + round * 20000; ++i ) { r2.add( static_cast<typename Bitmap::key_type>( rng() % ( 3U << 16 ) ) ); }
        if ( round % 2 == 1 ) { r1.optimize(); r2.optimize(); }
        EXPECT_EQ( ( r1 | r2 ).size(), r1.or_cardinality( r2 ) );
        EXPECT_EQ( ( r1 & r2 ).size(), r1.and_cardinality( r2 ) );
        EXPECT_EQ( ( r1 - r2 ).size(), r1.andnot_cardinality( r2 ) );
        expect_sizes_everywhere( r1, r2 );
        expect_sizes_everywhere( r2, r1 );
        expect_sizes_everywhere( r1, r1 );
    }
}

// tests/toplevel_unit.c: test_subset
TYPED_TEST( UpstreamParity, test_subset ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 4008 };
    Bitmap rb1, rb2;
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    // Sparse values
    for ( int i{ 0 }; i < 1000; i++ ) { rb2.add( choose_missing_value( rb2, std::uint64_t{ 1 } << 31, rng ) ); }
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    rb1 |= rb2;
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    auto value{ choose_missing_value( rb1, std::uint64_t{ 1 } << 31, rng ) };
    rb1.add( value );
    rb2.add( choose_missing_value( rb1, std::uint64_t{ 1 } << 31, rng ) );
    EXPECT_FALSE( rb1.is_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    rb2.add( value );
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    // Dense values
    for ( int i{ 0 }; i < 50000; i++ ) {
        value = choose_missing_value( rb2, 1 << 17, rng );
        rb1.add( value );
        rb2.add( value );
    }
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    value = choose_missing_value( rb2, 1 << 16, rng );
    rb1.add( value );
    rb2.add( choose_missing_value( rb1, 1 << 16, rng ) );
    EXPECT_FALSE( rb1.is_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_FALSE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    rb2.add( value );
    EXPECT_TRUE( rb1.is_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( rb2 ) );
    EXPECT_TRUE( rb1.is_strict_subset_of( view_of( rb2 ) ) );
    // an interval, and a range of keys: the operands whose size is not known are tested by walking them
    Bitmap range_holder;
    range_holder.add_closed_range( 10, 20 );
    EXPECT_TRUE( range_holder.is_strict_subset_of( key_range::closed( 5, 25 ) ) );
    EXPECT_FALSE( range_holder.is_strict_subset_of( key_range::closed( 10, 20 ) ) );
    EXPECT_FALSE( range_holder.is_strict_subset_of( key_range::closed( 11, 25 ) ) );
    std::vector<typename Bitmap::key_type> const keys{ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    EXPECT_TRUE( range_holder.is_strict_subset_of( keys ) );
    EXPECT_FALSE( range_holder.is_strict_subset_of( std::vector<typename Bitmap::key_type>{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 } ) );
}

// tests/cpp_unit.cpp: test_cpp_or_cardinality_64, with the 64-bit keys of Roaring64Map
TEST( UpstreamParity64, test_cpp_or_cardinality_64 ) {
    using Bitmap = bitmap<std::uint64_t>;
    Bitmap r1, r2, empty;
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
        r1.add( i );
        r1.add( ( std::uint64_t{ 2 } << 32 ) + i );
    }
    for ( std::uint64_t i{ 50 }; i < 150; ++i ) {
        r2.add( i );
        r2.add( ( std::uint64_t{ 8 } << 32 ) + i );
    }
    EXPECT_EQ( r1.or_cardinality( r2 ), ( r1 | r2 ).size() );
    EXPECT_EQ( r2.or_cardinality( r1 ), ( r1 | r2 ).size() );
    EXPECT_EQ( r1.or_cardinality( empty ), r1.size() );
    EXPECT_EQ( empty.or_cardinality( r1 ), r1.size() );
    EXPECT_EQ( r1.or_cardinality( r1 ), r1.size() );
    EXPECT_EQ( r1.or_cardinality( view_of( r2 ) ), ( r1 | r2 ).size() );
    EXPECT_EQ( r1.or_cardinality( view_of( empty ) ), r1.size() );
    EXPECT_EQ( empty.or_cardinality( view_of( r1 ) ), r1.size() );
    EXPECT_EQ( r1.or_cardinality( view_of( r1 ) ), r1.size() );
}

// tests/cpp_unit.cpp: test_cpp_xor_cardinality_64
TEST( UpstreamParity64, test_cpp_xor_cardinality_64 ) {
    using Bitmap = bitmap<std::uint64_t>;
    Bitmap r1, r2, empty;
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
        r1.add( i );
        r1.add( ( std::uint64_t{ 4 } << 32 ) + i );
    }
    for ( std::uint64_t i{ 50 }; i < 150; ++i ) {
        r2.add( i );
        r2.add( ( std::uint64_t{ 6 } << 32 ) + i );
    }
    EXPECT_EQ( r1.xor_cardinality( view_of( r2 ) ), ( r1 ^ r2 ).size() );
    EXPECT_EQ( r2.xor_cardinality( view_of( r1 ) ), ( r1 ^ r2 ).size() );
    EXPECT_EQ( r1.xor_cardinality( view_of( empty ) ), r1.size() );
    EXPECT_EQ( r1.xor_cardinality( view_of( r1 ) ), 0U );
}

// tests/cpp_unit.cpp: test_cpp_andnot_cardinality_64
TEST( UpstreamParity64, test_cpp_andnot_cardinality_64 ) {
    using Bitmap = bitmap<std::uint64_t>;
    Bitmap r1, r2, empty;
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
        r1.add( i );
        r1.add( ( std::uint64_t{ 3 } << 32 ) + i );
    }
    for ( std::uint64_t i{ 50 }; i < 150; ++i ) {
        r2.add( i );
        r2.add( ( std::uint64_t{ 5 } << 32 ) + i );
    }
    // A key held only by r1, so the two differences have distinct sizes and the asymmetry check below is meaningful.
    r1.add( ( std::uint64_t{ 9 } << 32 ) + 1 );
    EXPECT_EQ( r1.andnot_cardinality( r2 ), ( r1 - r2 ).size() );
    EXPECT_EQ( r2.andnot_cardinality( r1 ), ( r2 - r1 ).size() );
    EXPECT_NE( r1.andnot_cardinality( r2 ), r2.andnot_cardinality( r1 ) );
    EXPECT_EQ( r1.andnot_cardinality( empty ), r1.size() );
    EXPECT_EQ( empty.andnot_cardinality( r1 ), 0U );
    EXPECT_EQ( r1.andnot_cardinality( r1 ), 0U );
    EXPECT_EQ( r1.andnot_cardinality( view_of( r2 ) ), ( r1 - r2 ).size() );
    EXPECT_EQ( r2.andnot_cardinality( view_of( r1 ) ), ( r2 - r1 ).size() );
    EXPECT_NE( r1.andnot_cardinality( view_of( r2 ) ), r2.andnot_cardinality( view_of( r1 ) ) );
    EXPECT_EQ( r1.andnot_cardinality( view_of( empty ) ), r1.size() );
    EXPECT_EQ( empty.andnot_cardinality( view_of( r1 ) ), 0U );
    EXPECT_EQ( r1.andnot_cardinality( view_of( r1 ) ), 0U );
    EXPECT_EQ( r1.andnot_cardinality( !view_of( r2 ) ), r1.and_cardinality( r2 ) );
}

// the interval operand: sizes, clipped to the keys the bitmap's key type has
TYPED_TEST( UpstreamParity, IntervalOperand ) {
    using Bitmap = TypeParam;
    Bitmap bitmap;
    bitmap.add_closed_range( 100, 199 );
    EXPECT_EQ( bitmap.or_cardinality( key_range::closed( 150, 249 ) ), 150U );
    EXPECT_EQ( bitmap.or_cardinality( key_range{} ), 100U );
    EXPECT_EQ( bitmap.or_cardinality( key_range::closed( 300, 399 ) ), 200U );
    EXPECT_EQ( bitmap.andnot_cardinality( key_range::closed( 150, 249 ) ), 50U );
    EXPECT_DOUBLE_EQ( bitmap.jaccard_index( key_range::closed( 150, 249 ) ), 50.0 / 150.0 );
    EXPECT_DOUBLE_EQ( bitmap.jaccard_index( key_range::closed( 100, 199 ) ), 1.0 );
    EXPECT_DOUBLE_EQ( bitmap.jaccard_index( key_range::closed( 300, 399 ) ), 0.0 );
    EXPECT_TRUE( std::isnan( Bitmap{}.jaccard_index( key_range{} ) ) );
    EXPECT_TRUE( std::isnan( Bitmap{}.jaccard_index( Bitmap{} ) ) );
    EXPECT_EQ( Bitmap{}.or_cardinality( key_range::closed( 5, 9 ) ), 5U );
    if constexpr ( std::numeric_limits<typename Bitmap::key_type>::digits == 32 ) {
        // an interval reaching past the key type counts only the keys the type has
        EXPECT_EQ( Bitmap{}.or_cardinality( key_range::closed( 0xFFFFFFF0ULL, 0x1'0000'0020ULL ) ), 16U );
        EXPECT_EQ( Bitmap{}.or_cardinality( key_range::all() ), std::uint64_t{ 1 } << 32 );
    }
}

} // namespace
