// The interop operations against CRoaring: the contents an operation leaves, and the answers of the queries,
// must be those CRoaring gives for the same operands (built into a bitmap of its own, or through its native
// range functions).

#include "interop_support.hpp"

#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <bitset>
#include <cstring>
#include <utility>

namespace {

using namespace frsr::roaring::interop_test;
using frsr::roaring::bit_words;
using frsr::roaring::bits_at;
using frsr::roaring::key_range;
using frsr::roaring::sorted_keys;

// ---- CRoaring, 32- and 64-bit keys behind one shape ------------------------------------------------------------

template <typename Key>
struct reference;

template <>
struct reference<std::uint32_t> {
    using handle = roaring_bitmap_t;
    struct owner {
        explicit owner( handle * h ) : h{ h } {}
        owner( owner && other ) noexcept : h{ std::exchange( other.h, nullptr ) } {}
        ~owner() { if ( h ) { roaring_bitmap_free( h ); } }
        handle * h;
    };
    static owner make( oracle_set const & keys ) {
        std::vector<std::uint32_t> values( keys.begin(), keys.end() );
        return owner{ roaring_bitmap_of_ptr( values.size(), values.data() ) };
    }
    static void and_inplace   ( handle * a, handle const * b ) { roaring_bitmap_and_inplace( a, b ); }
    static void or_inplace    ( handle * a, handle const * b ) { roaring_bitmap_or_inplace( a, b ); }
    static void andnot_inplace( handle * a, handle const * b ) { roaring_bitmap_andnot_inplace( a, b ); }
    static void xor_inplace   ( handle * a, handle const * b ) { roaring_bitmap_xor_inplace( a, b ); }
    static std::uint64_t xor_card( handle const * a, handle const * b ) { return roaring_bitmap_xor_cardinality( a, b ); }
    static bool intersect     ( handle const * a, handle const * b ) { return roaring_bitmap_intersect( a, b ); }
    static std::uint64_t and_card( handle const * a, handle const * b ) { return roaring_bitmap_and_cardinality( a, b ); }
    static bool subset ( handle const * a, handle const * b ) { return roaring_bitmap_is_subset( a, b ); }
    static bool equals ( handle const * a, handle const * b ) { return roaring_bitmap_equals( a, b ); }
    static std::uint64_t cardinality( handle const * a ) { return roaring_bitmap_get_cardinality( a ); }
    static oracle_set contents( handle const * a ) {
        std::vector<std::uint32_t> values( roaring_bitmap_get_cardinality( a ) );
        roaring_bitmap_to_uint32_array( a, values.data() );
        return { values.begin(), values.end() };
    }
    // closed range [first, last]
    static bool intersects_range( handle const * a, std::uint64_t first, std::uint64_t last ) { return roaring_bitmap_intersect_with_range( a, first, last + 1 ); }
    static bool contains_range  ( handle const * a, std::uint64_t first, std::uint64_t last ) { return roaring_bitmap_contains_range( a, first, last + 1 ); }
    static std::uint64_t range_cardinality( handle const * a, std::uint64_t first, std::uint64_t last ) { return roaring_bitmap_range_cardinality( a, first, last + 1 ); }
    static owner from_range( std::uint64_t first, std::uint64_t last ) { return owner{ roaring_bitmap_from_range( first, last + 1, 1 ) }; }
    static void keep_range  ( handle * a, std::uint64_t first, std::uint64_t last ) {
        if ( first > 0 ) { roaring_bitmap_remove_range_closed( a, 0, static_cast<std::uint32_t>( first - 1 ) ); }
        if ( last < std::numeric_limits<std::uint32_t>::max() ) { roaring_bitmap_remove_range_closed( a, static_cast<std::uint32_t>( last + 1 ), std::numeric_limits<std::uint32_t>::max() ); }
    }
    static void remove_range( handle * a, std::uint64_t first, std::uint64_t last ) { roaring_bitmap_remove_range_closed( a, static_cast<std::uint32_t>( first ), static_cast<std::uint32_t>( last ) ); }
    static void add_range   ( handle * a, std::uint64_t first, std::uint64_t last ) { roaring_bitmap_add_range_closed( a, static_cast<std::uint32_t>( first ), static_cast<std::uint32_t>( last ) ); }
    static void flip_range  ( handle * a, std::uint64_t first, std::uint64_t last ) { roaring_bitmap_flip_inplace( a, first, last + 1 ); }
};

template <>
struct reference<std::uint64_t> {
    using handle = roaring64_bitmap_t;
    struct owner {
        explicit owner( handle * h ) : h{ h } {}
        owner( owner && other ) noexcept : h{ std::exchange( other.h, nullptr ) } {}
        ~owner() { if ( h ) { roaring64_bitmap_free( h ); } }
        handle * h;
    };
    static owner make( oracle_set const & keys ) {
        auto * const h{ roaring64_bitmap_create() };
        roaring64_bitmap_add_many( h, keys.size(), keys.data() );
        return owner{ h };
    }
    static void and_inplace   ( handle * a, handle const * b ) { roaring64_bitmap_and_inplace( a, b ); }
    static void or_inplace    ( handle * a, handle const * b ) { roaring64_bitmap_or_inplace( a, b ); }
    static void andnot_inplace( handle * a, handle const * b ) { roaring64_bitmap_andnot_inplace( a, b ); }
    static void xor_inplace   ( handle * a, handle const * b ) { roaring64_bitmap_xor_inplace( a, b ); }
    static std::uint64_t xor_card( handle const * a, handle const * b ) { return roaring64_bitmap_xor_cardinality( a, b ); }
    static bool intersect     ( handle const * a, handle const * b ) { return roaring64_bitmap_intersect( a, b ); }
    static std::uint64_t and_card( handle const * a, handle const * b ) { return roaring64_bitmap_and_cardinality( a, b ); }
    static bool subset ( handle const * a, handle const * b ) { return roaring64_bitmap_is_subset( a, b ); }
    static bool equals ( handle const * a, handle const * b ) { return roaring64_bitmap_equals( a, b ); }
    static std::uint64_t cardinality( handle const * a ) { return roaring64_bitmap_get_cardinality( a ); }
    static oracle_set contents( handle const * a ) {
        oracle_set values( roaring64_bitmap_get_cardinality( a ) );
        roaring64_bitmap_to_uint64_array( a, values.data() );
        return values;
    }
    static bool intersects_range( handle const * a, std::uint64_t first, std::uint64_t last ) {
        return last == std::numeric_limits<std::uint64_t>::max()
            ? roaring64_bitmap_intersect_with_range( a, first, last ) || roaring64_bitmap_contains( a, last )
            : roaring64_bitmap_intersect_with_range( a, first, last + 1 );
    }
    static bool contains_range  ( handle const * a, std::uint64_t first, std::uint64_t last ) { return roaring64_bitmap_contains_range_closed( a, first, last ); }
    static std::uint64_t range_cardinality( handle const * a, std::uint64_t first, std::uint64_t last ) { return roaring64_bitmap_range_closed_cardinality( a, first, last ); }
    static owner from_range( std::uint64_t first, std::uint64_t last ) {
        auto * const h{ roaring64_bitmap_create() };
        roaring64_bitmap_add_range_closed( h, first, last );
        return owner{ h };
    }
    static void keep_range  ( handle * a, std::uint64_t first, std::uint64_t last ) {
        if ( first > 0 ) { roaring64_bitmap_remove_range_closed( a, 0, first - 1 ); }
        if ( last < std::numeric_limits<std::uint64_t>::max() ) { roaring64_bitmap_remove_range_closed( a, last + 1, std::numeric_limits<std::uint64_t>::max() ); }
    }
    static void remove_range( handle * a, std::uint64_t first, std::uint64_t last ) { roaring64_bitmap_remove_range_closed( a, first, last ); }
    static void add_range   ( handle * a, std::uint64_t first, std::uint64_t last ) { roaring64_bitmap_add_range_closed( a, first, last ); }
    static void flip_range  ( handle * a, std::uint64_t first, std::uint64_t last ) { roaring64_bitmap_flip_closed_inplace( a, first, last ); }
};

template <typename Key>
class InteropCrosscheck : public ::testing::Test {};
using key_types = ::testing::Types<std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE( InteropCrosscheck, key_types );

// the operations and queries of `bitmap` against `operand`, whose keys are `members`
template <typename Key, typename Operand>
void compare( frsr::roaring::bitmap<Key> const & bitmap, oracle_set const & content, Operand const & operand, oracle_set const & members, std::string const & what ) {
    using ref = reference<Key>;
    using bitmap_type = frsr::roaring::bitmap<Key>;
    auto const reference_content{ ref::make( content ) };
    auto const reference_operand{ ref::make( members ) };
    auto const expect_same{ [ & ]( bitmap_type const & result, auto const & make_expected, char const * op ) {
        auto const owner{ ref::make( content ) };
        make_expected( owner.h, reference_operand.h );
        auto const expected{ ref::contents( owner.h ) };
        ASSERT_EQ( result.validate(), nullptr ) << what << " " << op;
        ASSERT_EQ( result.size(), expected.size() ) << what << " " << op;
        std::vector<Key> narrow( expected.begin(), expected.end() );
        ASSERT_TRUE( result.to_vector() == narrow ) << what << " " << op << " differs from CRoaring";
    } };
    {
        bitmap_type b{ bitmap };
        b.and_with( operand );
        expect_same( b, ref::and_inplace, "and" );
    }
    {
        bitmap_type b{ bitmap };
        b.or_with( operand );
        expect_same( b, ref::or_inplace, "or" );
    }
    {
        bitmap_type b{ bitmap };
        b.andnot_with( operand );
        expect_same( b, ref::andnot_inplace, "andnot" );
    }
    {
        bitmap_type b{ bitmap };
        b.xor_with( operand );
        expect_same( b, ref::xor_inplace, "xor" );
    }
    EXPECT_EQ( bitmap.xor_cardinality( operand ), ref::xor_card( reference_content.h, reference_operand.h ) ) << what;
    EXPECT_EQ( bitmap.intersects( operand ), ref::intersect( reference_content.h, reference_operand.h ) ) << what;
    EXPECT_EQ( bitmap.and_cardinality( operand ), ref::and_card( reference_content.h, reference_operand.h ) ) << what;
    EXPECT_EQ( bitmap.is_subset_of( operand ), ref::subset( reference_content.h, reference_operand.h ) ) << what;
    EXPECT_EQ( bitmap.is_superset_of( operand ), ref::subset( reference_operand.h, reference_content.h ) ) << what;
    EXPECT_EQ( bitmap.equals( operand ), ref::equals( reference_content.h, reference_operand.h ) ) << what;
}

TYPED_TEST( InteropCrosscheck, KeysBitsAndWords ) {
    using Key = TypeParam;
    using Bitmap = frsr::roaring::bitmap<Key>;
    std::mt19937_64 rng{ 1701 };
    for ( int round{ 0 }; round < 6; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        auto const bitmap{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        auto const related{ random_related<Bitmap>( rng, content, false ) };
        std::vector<Key> keys( related.begin(), related.end() );
        compare( bitmap, content, keys, related, "sorted keys" );
        std::shuffle( keys.begin(), keys.end(), rng );
        compare( bitmap, content, std::span<Key const>{ keys }, related, "unsorted keys" );
        compare( bitmap, content, sorted_keys{ related }, related, "declared sorted keys" );
        compare( bitmap, content, Bitmap{ keys }, related, "bitmap" );

        for ( std::uint64_t const base : { std::uint64_t{ 0 }, std::uint64_t{ 65536 }, std::uint64_t{ 131072 + 37 }, std::uint64_t{ 5 } << 32 } ) {
            if ( base > domain<Bitmap>::max ) { continue; }
            auto const string{ random_bit_string( rng, base, 150000, domain<Bitmap>::max ) };
            auto const in_range{ in_domain( string.members, domain<Bitmap>::max ) };
            if ( in_range.size() != string.members.size() ) { continue; }
            compare( bitmap, content, bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ string.words }, base, 150000 }, string.members, "words base " + std::to_string( base ) );
        }
        std::bitset<65536> bits;
        oracle_set members;
        for ( std::size_t i{ 0 }; i < 65536; ++i ) { if ( rng() % 7 == 0 ) { bits.set( i ); members.push_back( 65536 + i ); } }
        compare( bitmap, content, bits_at( bits, 65536 ), members, "std::bitset<65536> at a chunk" );
        members.clear();
        for ( std::size_t i{ 0 }; i < 65536; ++i ) { if ( bits.test( i ) ) { members.push_back( 100 + i ); } }
        compare( bitmap, content, bits_at( bits, 100 ), members, "std::bitset<65536> off a chunk" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( InteropCrosscheck, Ranges ) {
    using Key = TypeParam;
    using Bitmap = frsr::roaring::bitmap<Key>;
    using ref = reference<Key>;
    constexpr auto max{ domain<Bitmap>::max };
    std::mt19937_64 rng{ 1812 };
    for ( int round{ 0 }; round < 8; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        auto const bitmap{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        auto const reference_content{ ref::make( content ) };
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{ { 0, max }, { 0, 0 }, { max, max }, { 1, max - 1 }, { 65535, 65536 }, { 70000, 70000 } };
        for ( int i{ 0 }; i < 24; ++i ) {
            std::uint64_t first{ rng() % ( max / 2 ) };
            std::uint64_t last{ first + rng() % ( max / 4 ) };
            if ( !content.empty() && rng() % 2 ) {
                first = content[ rng() % content.size() ];
                last = rng() % 3 == 0 ? first : first + std::min( max - first, rng() % 300000 );
            }
            ranges.emplace_back( first, last );
        }
        // ranges that end inside what the bitmap holds, and exactly at its keys
        for ( int i{ 0 }; i < 12 && !content.empty(); ++i ) {
            auto const first{ content[ rng() % content.size() ] };
            auto const last{ content[ rng() % content.size() ] };
            if ( first <= last ) { ranges.emplace_back( first, last ); }
        }
        for ( auto const & [ first, last ] : ranges ) {
            if ( last < first ) { continue; }
            auto const what{ "range [" + std::to_string( first ) + ", " + std::to_string( last ) + "]" };
            {   // the oracle is the arbiter where CRoaring and the library differ
                auto const inside{ static_cast<std::uint64_t>( std::upper_bound( content.begin(), content.end(), last ) - std::lower_bound( content.begin(), content.end(), first ) ) };
                EXPECT_EQ( bitmap.and_cardinality( key_range::closed( first, last ) ), inside ) << what;
                EXPECT_EQ( bitmap.is_superset_of( key_range::closed( first, last ) ), inside == last - first + 1 ) << what << " (oracle)";
                // roaring64_bitmap_contains_range_closed answers true for some intervals that cross chunks the bitmap
                // lacks, so for 64-bit keys CRoaring is not compared on this query: the oracle above decides.
                if constexpr ( sizeof( Key ) == 4 ) {
                    EXPECT_EQ( ref::contains_range( reference_content.h, first, last ), inside == last - first + 1 ) << what << " (CRoaring vs oracle)";
                }
            }
            EXPECT_EQ( bitmap.intersects( key_range::closed( first, last ) ), ref::intersects_range( reference_content.h, first, last ) ) << what;
            EXPECT_EQ( bitmap.and_cardinality( key_range::closed( first, last ) ), ref::range_cardinality( reference_content.h, first, last ) ) << what;
            if ( last < max ) {
                EXPECT_EQ( bitmap.intersects_range( static_cast<Key>( first ), static_cast<Key>( last + 1 ) ), ref::intersects_range( reference_content.h, first, last ) ) << what;
                EXPECT_EQ( bitmap.contains_range( static_cast<Key>( first ), static_cast<Key>( last + 1 ) ), bitmap.is_superset_of( key_range::closed( first, last ) ) ) << what;
                EXPECT_EQ( bitmap.cardinality_in_range( static_cast<Key>( first ), static_cast<Key>( last + 1 ) ), ref::range_cardinality( reference_content.h, first, last ) ) << what;
            }
            // the in-place forms against CRoaring's own range operations
            for ( int which{ 0 }; which < 4; ++which ) {
                if ( ( which == 1 || which == 3 ) && last - first >= ( std::uint64_t{ 1 } << 32 ) ) { continue; }   // a union or flip that wide is 2^16 chunks per 2^32 keys
                Bitmap b{ bitmap };
                auto const owner{ ref::make( content ) };
                if ( which == 0 ) { b.and_with( key_range::closed( first, last ) ); ref::keep_range( owner.h, first, last ); }
                if ( which == 1 ) { b.or_with( key_range::closed( first, last ) ); ref::add_range( owner.h, first, last ); }
                if ( which == 2 ) { b.andnot_with( key_range::closed( first, last ) ); ref::remove_range( owner.h, first, last ); }
                if ( which == 3 ) { b.xor_with( key_range::closed( first, last ) ); ref::flip_range( owner.h, first, last ); }
                ASSERT_EQ( b.validate(), nullptr ) << what << " op " << which;
                ASSERT_EQ( b.size(), ref::cardinality( owner.h ) ) << what << " op " << which;
                if ( ref::cardinality( owner.h ) > 20'000'000 ) { continue; }   // too wide to list; the size and validate() stand
                auto const expected{ ref::contents( owner.h ) };
                std::vector<Key> narrow( expected.begin(), expected.end() );
                ASSERT_TRUE( b.to_vector() == narrow ) << what << " op " << which;
            }
        }
    }
}

} // namespace
