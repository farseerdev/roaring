// Operations of a bitmap against operands that are not bitmaps (keys in a span or any range, std::bitset,
// dynamic bit-sets, raw bit words, key intervals), checked against a std::set oracle.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <list>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <string>
#include <variant>
#include <vector>

#if __has_include( <boost/dynamic_bitset.hpp> )
#   include <boost/dynamic_bitset.hpp>
#   define FRSR_TEST_HAS_DYNAMIC_BITSET 1
#endif

#include "interop_support.hpp"

namespace {

using namespace frsr::roaring::interop_test;

using frsr::roaring::bit_words;
using frsr::roaring::bits_at;
using frsr::roaring::key_range;
using frsr::roaring::sorted_keys;

#if defined( __SANITIZE_ADDRESS__ ) || defined( __SANITIZE_THREAD__ )
constexpr int scale{ 1 };
#elif defined( __has_feature )
#   if __has_feature( address_sanitizer ) || __has_feature( undefined_behavior_sanitizer )
constexpr int scale{ 1 };
#   else
constexpr int scale{ 3 };
#   endif
#else
constexpr int scale{ 3 };
#endif

static_assert( frsr::roaring::interop_operand<std::bitset<100>> );
static_assert( frsr::roaring::interop_operand<block_bits> );
static_assert( frsr::roaring::interop_operand<find_only_bits> );
static_assert( frsr::roaring::interop_operand<key_range> );
static_assert( frsr::roaring::interop_operand<std::span<std::uint32_t const>> );
static_assert( frsr::roaring::interop_operand<std::vector<std::uint64_t>> );
static_assert( frsr::roaring::interop_operand<std::set<std::uint16_t>> );
static_assert( frsr::roaring::interop_operand<bit_words<std::uint64_t>> );
static_assert( !frsr::roaring::interop_operand<std::vector<int>> );
static_assert( !frsr::roaring::interop_operand<std::vector<double>> );
static_assert( !frsr::roaring::interop_operand<frsr::roaring::bitmap<std::uint32_t>> );
static_assert( frsr::roaring::interop_operand<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );
// a view is no sequence of keys (it would otherwise be taken for one), no bitmap (it would otherwise keep a bitmap's overloads away
// from the bitmaps) and no bit-set
static_assert( !std::ranges::input_range<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );
static_assert( !frsr::roaring::detail::key_sequence<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );
static_assert( !frsr::roaring::detail::bitmap_like<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );
static_assert( !frsr::roaring::detail::bit_container<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );
static_assert( sizeof( frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>> ) == sizeof( void * ) );
static_assert( std::is_trivially_copyable_v<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>> );

// the read-only operations cannot throw against operands they never copy out
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().intersects( std::declval<key_range>() ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().is_superset_of( std::declval<bit_words<std::uint64_t>>() ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint64_t> const &>().and_cardinality( std::declval<std::bitset<65536>>() ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().contains_range( 1, 2 ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().xor_cardinality( std::declval<key_range>() ) ) );
static_assert( !noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().xor_cardinality( std::declval<std::vector<std::uint32_t>>() ) ) );
static_assert( !noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().intersects( std::declval<std::vector<std::uint32_t>>() ) ) );
static_assert( !noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().equals( std::declval<std::bitset<1000000>>() ) ) );
// ... and so are the ones against a view, which allocates nothing
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> const &>().intersects( std::declval<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>>() ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint64_t> const &>().and_cardinality( std::declval<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint64_t>>>() ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> &>().and_with( std::declval<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>>() ) )
               == frsr::roaring::detail::allocation_failure_terminates );

template <typename Bitmap>
void expect_holds( Bitmap const & bitmap, oracle_set const & expected, std::string const & what ) {
    using key_type = typename Bitmap::key_type;
    ASSERT_EQ( bitmap.validate(), nullptr ) << what << ": " << ( bitmap.validate() ? bitmap.validate() : "" );
    ASSERT_EQ( bitmap.size(), expected.size() ) << what;
    std::vector<key_type> expected_vector;
    expected_vector.reserve( expected.size() );
    for ( auto const key : expected ) { expected_vector.push_back( static_cast<key_type>( key ) ); }
    ASSERT_TRUE( bitmap.to_vector() == expected_vector ) << what << ": contents differ";
}

// Every operation with `operand`, whose members are `members` (which may reach beyond the key type), on
// copies of `bitmap`, whose members are `content`.
template <typename Bitmap, typename Operand>
void check_operand( Bitmap const & bitmap, oracle_set const & content, Operand const & operand, oracle_set const & members, std::string const & what ) {
    auto const max{ domain<Bitmap>::max };
    auto const representable{ in_domain( members, max ) };
    {
        Bitmap b{ bitmap };
        b.and_with( operand );
        expect_holds( b, intersection( content, members ), what + " and_with" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    {
        Bitmap b{ bitmap };
        b.or_with( operand );
        expect_holds( b, unite( content, representable ), what + " or_with" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    {
        Bitmap b{ bitmap };
        b.andnot_with( operand );
        expect_holds( b, difference( content, members ), what + " andnot_with" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    {
        Bitmap b{ bitmap };
        b.xor_with( operand );
        expect_holds( b, symmetric_difference( content, representable ), what + " xor_with" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    {
        Bitmap b{ bitmap };
        b &= operand;
        expect_holds( b, intersection( content, members ), what + " &=" );
        b = bitmap;
        b |= operand;
        expect_holds( b, unite( content, representable ), what + " |=" );
        b = bitmap;
        b -= operand;
        expect_holds( b, difference( content, members ), what + " -=" );
        b = bitmap;
        b ^= operand;
        expect_holds( b, symmetric_difference( content, representable ), what + " ^=" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    auto const common{ intersection( content, members ) };
    EXPECT_EQ( bitmap.intersects( operand ), !common.empty() ) << what << " intersects";
    EXPECT_EQ( bitmap.and_cardinality( operand ), common.size() ) << what << " and_cardinality";
    EXPECT_EQ( bitmap.xor_cardinality( operand ), content.size() + representable.size() - 2 * common.size() ) << what << " xor_cardinality";
    EXPECT_EQ( bitmap.is_subset_of( operand ), common.size() == content.size() ) << what << " is_subset_of";
    EXPECT_EQ( bitmap.is_superset_of( operand ), common.size() == members.size() ) << what << " is_superset_of";
    EXPECT_EQ( bitmap.equals( operand ), content == members ) << what << " equals";
}

// ---- operand forms ----------------------------------------------------------------------------------------

// `members` as the key operands: spans of the key type (when they fit), unsorted and with duplicates, a
// declared-sorted range, a std::set and a lazily generated range
template <typename Bitmap, typename Check>
void with_key_operands( oracle_set const & members, std::mt19937_64 & rng, Check && check ) {
    using key_type = typename Bitmap::key_type;
    std::vector<std::uint64_t> wide( members.begin(), members.end() );
    check( wide, "vector<uint64_t> sorted" );
    check( sorted_keys{ wide }, "sorted_keys<vector<uint64_t>>" );
    auto shuffled{ wide };
    std::shuffle( shuffled.begin(), shuffled.end(), rng );
    check( shuffled, "vector<uint64_t> shuffled" );
    auto repeated{ shuffled };
    for ( std::size_t i{ 0 }; i < wide.size() && i < 50; ++i ) { repeated.push_back( wide[ rng() % wide.size() ] ); }
    std::shuffle( repeated.begin(), repeated.end(), rng );
    check( repeated, "vector<uint64_t> with duplicates" );
    auto sorted_repeated{ wide };
    for ( std::size_t i{ 0 }; i < wide.size() && i < 50; ++i ) { sorted_repeated.push_back( wide[ rng() % wide.size() ] ); }
    std::ranges::sort( sorted_repeated );
    check( sorted_repeated, "vector<uint64_t> sorted with duplicates" );
    if ( members.size() < 20000 ) { check( std::set<std::uint64_t>{ members.begin(), members.end() }, "std::set<uint64_t>" ); }
    if ( in_domain( members, domain<Bitmap>::max ).size() == members.size() ) {
        std::vector<key_type> narrow( members.begin(), members.end() );
        check( narrow, "vector<key_type> sorted" );
        check( std::span<key_type const>{ narrow }, "span<key_type> sorted" );
        check( sorted_keys{ narrow }, "sorted_keys<vector<key_type>>" );
        auto repeated_narrow{ narrow };
        for ( std::size_t i{ 0 }; i < narrow.size() && i < 40; ++i ) { repeated_narrow.push_back( narrow[ rng() % narrow.size() ] ); }
        std::ranges::sort( repeated_narrow );
        check( std::span<key_type const>{ repeated_narrow }, "span<key_type> sorted with duplicates" );
        check( sorted_keys{ repeated_narrow }, "sorted_keys<vector<key_type>> with duplicates" );
        std::shuffle( narrow.begin(), narrow.end(), rng );
        check( std::span<key_type const>{ narrow }, "span<key_type> shuffled" );
        std::list<key_type> list( narrow.begin(), narrow.end() );
        list.sort();
        check( list, "std::list sorted" );
        std::vector<std::uint8_t> bytes;   // a narrower element type: only the keys that fit
        if ( members.empty() || members.back() <= 255 ) {
            for ( auto const key : members ) { bytes.push_back( static_cast<std::uint8_t>( key ) ); }
            check( bytes, "vector<uint8_t>" );
        }
    }
}

template <typename Bitmap, typename Check>
void with_word_operands( std::mt19937_64 & rng, Check && check ) {
    auto const max{ domain<Bitmap>::max };
    std::vector<std::uint64_t> bases{ 0, 1, 63, 64, 65, 65535, 65536, 65537, 70000, 131072 + 17 };
    if ( max > 200000 ) {
        bases.push_back( max - 200000 );     // the string reaches past the key type or ends at its maximum
        bases.push_back( max - 100000 );
        bases.push_back( ( max >> 16 << 16 ) - 5 );
    }
    constexpr std::uint64_t lengths[]{ 1, 63, 64, 65, 1000, 4096, 65535, 65536, 65537, 200000 };
    for ( int round{ 0 }; round < 6 * scale; ++round ) {
        auto const base{ bases[ rng() % bases.size() ] };
        auto length{ lengths[ rng() % std::size( lengths ) ] };
        // a string of the widest keys ends at the maximum key rather than wrapping around
        if ( base > std::numeric_limits<std::uint64_t>::max() - ( length - 1 ) ) { length = std::numeric_limits<std::uint64_t>::max() - base + 1; }
        auto const string{ random_bit_string( rng, base, length, max ) };
        auto const tag{ "words base=" + std::to_string( base ) + " bits=" + std::to_string( length ) };
        check( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ string.words }, base, length }, string.members, tag + " u64" );
        // the same string as narrower words (the byte order of the words is the bit order)
        std::vector<std::uint32_t> words32( ( string.words.size() * 2 ) );
        std::memcpy( words32.data(), string.words.data(), string.words.size() * 8 );
        check( bit_words<std::uint32_t>{ std::span<std::uint32_t const>{ words32 }, base, length }, string.members, tag + " u32" );
        std::vector<std::uint8_t> bytes( string.words.size() * 8 );
        std::memcpy( bytes.data(), string.words.data(), bytes.size() );
        check( bit_words<std::uint8_t>{ std::span<std::uint8_t const>{ bytes }, base, length }, string.members, tag + " u8" );
        // unaligned storage: the words one byte into a buffer
        std::vector<std::uint8_t> shifted( bytes.size() + 8 );
        std::memcpy( shifted.data() + 1, bytes.data(), bytes.size() );
        check( bit_words<std::uint8_t>{ std::span<std::uint8_t const>{ shifted }.subspan( 1, bytes.size() ), base, length }, string.members, tag + " u8 unaligned" );
        // dynamic bit-set doubles
        block_bits blocks{ static_cast<std::size_t>( length ) };
        find_only_bits finder{ static_cast<std::size_t>( length ) };
        for ( auto const key : string.members ) { blocks.set( key - base ); finder.set( key - base ); }
        check( frsr::roaring::bits_at( blocks, base ), string.members, tag + " blocks" );
        check( frsr::roaring::bits_at( finder, base ), string.members, tag + " find_first/find_next" );
#ifdef FRSR_TEST_HAS_DYNAMIC_BITSET
        boost::dynamic_bitset<> dynamic_default( static_cast<std::size_t>( length ) );
        boost::dynamic_bitset<std::uint64_t> dynamic_wide( static_cast<std::size_t>( length ) );
        boost::dynamic_bitset<std::uint8_t> dynamic_narrow( static_cast<std::size_t>( length ) );
        for ( auto const key : string.members ) { dynamic_default.set( key - base ); dynamic_wide.set( key - base ); dynamic_narrow.set( key - base ); }
        check( frsr::roaring::bits_at( dynamic_default, base ), string.members, tag + " boost::dynamic_bitset<>" );
        check( frsr::roaring::bits_at( dynamic_wide, base ), string.members, tag + " boost::dynamic_bitset<u64>" );
        check( frsr::roaring::bits_at( dynamic_narrow, base ), string.members, tag + " boost::dynamic_bitset<u8>" );
        if ( base == 0 ) { check( dynamic_default, string.members, tag + " boost::dynamic_bitset<> unbased" ); }
#endif
    }
}

template <std::size_t N, typename Bitmap, typename Check>
void with_std_bitset( std::mt19937_64 & rng, Check && check ) {
    auto const max{ domain<Bitmap>::max };
    for ( std::uint64_t const base : { std::uint64_t{ 0 }, std::uint64_t{ 65536 }, std::uint64_t{ 70001 },
        max <= N ? std::uint64_t{ 0 } : max == std::numeric_limits<std::uint64_t>::max() ? max - ( N - 1 ) : max - N / 2 } ) {
        std::bitset<N> bits;
        oracle_set members;
        constexpr double densities[]{ 0.0, 0.001, 0.1, 0.5, 1.0 };
        auto const density{ densities[ rng() % std::size( densities ) ] };
        std::uniform_real_distribution<double> unit{ 0.0, 1.0 };
        for ( std::size_t i{ 0 }; i < N; ++i ) {
            if ( unit( rng ) < density ) { bits.set( i ); members.push_back( base + i ); }
        }
        auto const tag{ "std::bitset<" + std::to_string( N ) + "> base=" + std::to_string( base ) };
        check( bits_at( bits, base ), members, tag );
        if ( base == 0 ) { check( bits, members, tag + " unbased" ); }
    }
}

template <typename Bitmap, typename Check>
void with_range_operands( std::mt19937_64 & rng, Check && check ) {
    auto const max{ domain<Bitmap>::max };
    std::vector<std::uint64_t> edges{ 0, 1, 255, 256, 4095, 4096, 65534, 65535, 65536, 65537, 70000, 131071, 131072 };
    for ( auto const chunk : domain<Bitmap>::chunks() ) {
        edges.push_back( chunk << 16 ); edges.push_back( ( chunk << 16 ) + 1 ); edges.push_back( ( chunk << 16 ) + 65535 );
        if ( ( chunk << 16 ) > 0 ) { edges.push_back( ( chunk << 16 ) - 1 ); }
    }
    edges.erase( std::remove_if( edges.begin(), edges.end(), [ & ]( auto const e ) { return e > max; } ), edges.end() );
    for ( int round{ 0 }; round < 10 * scale; ++round ) {
        auto const first{ edges[ rng() % edges.size() ] };
        std::uint64_t last{ first + ( rng() % 3 == 0 ? rng() % 200 : rng() % 200000 ) };
        if ( last > max || last < first ) { last = max; }
        // only ranges small enough to enumerate are compared here; wide ones have their own test
        if ( last - first >= 400000 ) { continue; }
        oracle_set members;
        for ( auto k{ first }; ; ++k ) { members.push_back( k ); if ( k == last ) { break; } }
        check( key_range::closed( first, last ), members, "closed " + std::to_string( first ) + ".." + std::to_string( last ) );
        if ( last < max ) { check( key_range::half_open( first, last + 1 ), members, "half_open" ); }
    }
}

// ---- tests ------------------------------------------------------------------------------------------------

template <typename Bitmap>
class InteropTest : public ::testing::Test {};

using bitmap_types = ::testing::Types<
    frsr::roaring::bitmap<std::uint32_t>,
    frsr::roaring::bitmap<std::uint64_t>,
    frsr::roaring::bitmap<std::uint16_t>,
    frsr::roaring::bitmap<std::uint8_t>,
    frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>,
    frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount, frsr::roaring::detail::run_selection_lazy>,
    frsr::roaring::bitmap<std::uint32_t, std::variant<frsr::roaring::array_container<std::uint32_t>, frsr::roaring::bitset_container<std::uint32_t>>>
>;
TYPED_TEST_SUITE( InteropTest, bitmap_types );

TYPED_TEST( InteropTest, KeyOperands ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 20261020 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        for ( bool const optimize : { false, true } ) {
            auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
            for ( int variant{ 0 }; variant < 2; ++variant ) {
                auto const members{ random_related<Bitmap>( rng, content, true ) };
                with_key_operands<Bitmap>( members, rng, [ & ]( auto const & operand, std::string const & what ) {
                    check_operand( bitmap, content, operand, members, what + ( optimize ? " (optimized)" : "" ) );
                } );
                if ( ::testing::Test::HasFatalFailure() ) { return; }
            }
        }
    }
}

// unsorted keys on both sides of the size where probing key by key gives way to sorting a copy and merging
TYPED_TEST( InteropTest, UnsortedKeysAroundTheProbeLimit ) {
    using Bitmap = TypeParam;
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 616 };
    auto const content{ random_content<Bitmap>( rng ) };
    if ( content.size() < 10 ) { return; }
    auto const bitmap{ make_bitmap<Bitmap>( content, false ) };
    for ( std::size_t const count : { std::size_t{ 1 }, std::size_t{ 5 }, std::size_t{ 100 }, std::size_t{ 511 }, std::size_t{ 512 }, std::size_t{ 2047 }, std::size_t{ 2048 }, std::size_t{ 5000 } } ) {
        oracle_set members;
        std::vector<key_type> keys;
        while ( keys.size() < count ) {
            // half from the bitmap, half anywhere; repeats happen
            std::uint64_t const key{ rng() % 2 ? content[ rng() % content.size() ] : rng() % domain<Bitmap>::max };
            keys.push_back( static_cast<key_type>( key ) );
        }
        for ( auto const key : keys ) { members.push_back( key ); }
        finalize( members );
        std::ranges::shuffle( keys, rng );
        if ( std::ranges::is_sorted( keys ) && keys.size() > 1 ) { std::swap( keys.front(), keys.back() ); }
        check_operand( bitmap, content, std::span<key_type const>{ keys }, members, "unsorted n=" + std::to_string( count ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

// the bitmap overloads of the same names (a bitmap is an operand too), including a bitmap that is equal to,
// larger than and smaller than this one
TYPED_TEST( InteropTest, BitmapOperands ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 808 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        for ( bool const optimize : { false, true } ) {
            auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
            for ( int variant{ 0 }; variant < 3; ++variant ) {
                auto const members{ random_related<Bitmap>( rng, content, false ) };
                check_operand( bitmap, content, make_bitmap<Bitmap>( members, variant == 1 ), members, "bitmap operand" );
            }
            check_operand( bitmap, content, bitmap, content, "an equal bitmap" );
            auto larger{ content };
            larger.push_back( domain<Bitmap>::max );
            finalize( larger );
            check_operand( bitmap, content, make_bitmap<Bitmap>( larger, false ), larger, "a larger bitmap" );
            oracle_set smaller;
            for ( std::size_t i{ 0 }; i < content.size(); i += 3 ) { smaller.push_back( content[ i ] ); }
            check_operand( bitmap, content, make_bitmap<Bitmap>( smaller, false ), smaller, "a smaller bitmap" );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

// a view of a bitmap is an operand too, with the same answers as the bitmap itself, whatever either side holds
TYPED_TEST( InteropTest, ViewOperands ) {
    using Bitmap = TypeParam;
    using frsr::roaring::view_of;
    std::mt19937_64 rng{ 909 };
    auto const run{ [ & ]( oracle_set const & content, oracle_set const & members, bool const optimize_receiver, bool const optimize_viewed, std::string const & what ) {
        auto const bitmap{ make_bitmap<Bitmap>( content, optimize_receiver ) };
        auto const viewed{ make_bitmap<Bitmap>( members, optimize_viewed ) };
        check_operand( bitmap, content, view_of( viewed ), members, "view " + what );
    } };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        for ( bool const optimize : { false, true } ) {
            for ( int variant{ 0 }; variant < 3; ++variant ) {
                run( content, random_related<Bitmap>( rng, content, false ), optimize, variant == 1, "related" );
            }
            run( content, content, optimize, !optimize, "equal" );
            auto larger{ content };
            larger.push_back( domain<Bitmap>::max );
            finalize( larger );
            run( content, larger, optimize, false, "larger" );
            oracle_set smaller;
            for ( std::size_t i{ 0 }; i < content.size(); i += 3 ) { smaller.push_back( content[ i ] ); }
            run( content, smaller, optimize, false, "smaller" );
            run( content, random_content<Bitmap>( rng ), optimize, optimize, "unrelated" );
            run( content, {}, optimize, false, "empty" );
            run( {}, content, optimize, optimize, "into an empty bitmap" );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
    // chunks of one kind against chunks of every kind
    constexpr shape shapes[]{ shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::ends, shape::single, shape::mixed };
    for ( auto const left : shapes ) {
        for ( auto const right : shapes ) {
            oracle_set content, members;
            fill_chunk( content, 0, left, rng, domain<Bitmap>::max );
            fill_chunk( members, 0, right, rng, domain<Bitmap>::max );
            if ( !domain<Bitmap>::single_chunk ) {
                fill_chunk( content, 1, right, rng, domain<Bitmap>::max );
                fill_chunk( members, 1, left, rng, domain<Bitmap>::max );
            }
            finalize( content );
            finalize( members );
            for ( bool const optimize : { false, true } ) {
                run( content, members, optimize, !optimize, "shape " + std::to_string( static_cast<int>( left ) ) + " against " + std::to_string( static_cast<int>( right ) ) );
                if ( ::testing::Test::HasFatalFailure() ) { return; }
            }
        }
    }
}

// The complement of an operand: the operations it is rewritten to, on every operand form, against the oracle
// (a bitmap holds only keys, so a complement is read through what the bitmap holds)
template <typename Bitmap, typename Complement>
void check_complement( Bitmap const & bitmap, oracle_set const & content, Complement const & complement, oracle_set const & members, std::string const & what ) {
    auto const without{ difference( content, members ) };
    auto const common{ intersection( content, members ) };
    {
        Bitmap b{ bitmap };
        b &= complement;
        expect_holds( b, without, what + " &= !x" );
        b = bitmap;
        b.and_with( complement );
        expect_holds( b, without, what + " and_with( !x )" );
        b = bitmap;
        b -= complement;
        expect_holds( b, common, what + " -= !x" );
        b = bitmap;
        b.andnot_with( complement );
        expect_holds( b, common, what + " andnot_with( !x )" );
        expect_holds( bitmap & complement, without, what + " bitmap & !x" );
        expect_holds( complement & bitmap, without, what + " !x & bitmap" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    EXPECT_EQ( bitmap.intersects( complement ), !without.empty() ) << what << " intersects";
    EXPECT_EQ( bitmap.and_cardinality( complement ), without.size() ) << what << " and_cardinality";
    EXPECT_EQ( bitmap.is_subset_of( complement ), common.empty() ) << what << " is_subset_of";
}

TYPED_TEST( InteropTest, ComplementOperands ) {
    using Bitmap = TypeParam;
    using frsr::roaring::complement_of;
    using frsr::roaring::view_of;
    std::mt19937_64 rng{ 1111 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        for ( bool const optimize : { false, true } ) {
            auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
            for ( int variant{ 0 }; variant < 3; ++variant ) {
                auto const members{ random_related<Bitmap>( rng, content, false ) };
                auto const viewed{ make_bitmap<Bitmap>( members, variant == 1 ) };
                auto const tag{ std::string{ optimize ? " (optimized) " : " " } + std::to_string( variant ) };
                check_complement( bitmap, content, !view_of( viewed ), members, "!view" + tag );
                check_complement( bitmap, content, complement_of( view_of( viewed ) ), members, "complement_of( view )" + tag );
                check_complement( bitmap, content, !viewed, members, "!bitmap" + tag );
                if ( ::testing::Test::HasFatalFailure() ) { return; }
                with_key_operands<Bitmap>( members, rng, [ & ]( auto const & operand, std::string const & what ) {
                    check_complement( bitmap, content, complement_of( operand ), members, "complement_of( " + what + " )" + tag );
                } );
                // !!x is x
                auto const twice{ !!view_of( viewed ) };
                static_assert( std::same_as<std::remove_cvref_t<decltype( twice )>, frsr::roaring::typed_view<Bitmap>> );
                EXPECT_TRUE( twice == view_of( viewed ) );
                if ( ::testing::Test::HasFatalFailure() ) { return; }
            }
            with_word_operands<Bitmap>( rng, [ & ]( auto const & operand, oracle_set const & members, std::string const & what ) {
                check_complement( bitmap, content, complement_of( operand ), members, "complement_of( " + what + " )" );
            } );
            with_range_operands<Bitmap>( rng, [ & ]( auto const & operand, oracle_set const & members, std::string const & what ) {
                check_complement( bitmap, content, !operand, members, "!" + what );
            } );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

// a complement does not copy a container operand, and the complement of a complement is the operand
TEST( ComplementOperand, StorageAndInvolution ) {
    using frsr::roaring::complement;
    using frsr::roaring::complement_of;
    std::vector<std::uint32_t> keys{ 1, 2, 3 };
    auto const by_reference{ complement_of( keys ) };
    EXPECT_EQ( &by_reference.operand, &keys );   // held, not copied: a std::vector would allocate
    std::bitset<100> bits;
    auto const bits_complement{ complement_of( bits ) };
    EXPECT_EQ( &bits_complement.operand, &bits );
    static_assert( sizeof( complement<frsr::roaring::key_range> ) == sizeof( frsr::roaring::key_range ) );
    static_assert( std::is_trivially_copyable_v<complement<frsr::roaring::key_range>> );
    static_assert( std::is_trivially_copyable_v<complement<frsr::roaring::typed_view<frsr::roaring::bitmap<std::uint32_t>>>> );
    static_assert( std::is_trivially_copyable_v<complement<std::span<std::uint32_t const>>> );
    static_assert( noexcept( complement_of( keys ) ) );
    auto const range{ frsr::roaring::key_range::closed( 5, 9 ) };
    auto const back{ !!range };
    static_assert( std::same_as<std::remove_cvref_t<decltype( back )>, frsr::roaring::key_range> );
    EXPECT_EQ( back.first, 5U );
    EXPECT_EQ( back.last, 9U );
    EXPECT_EQ( &complement_of( complement_of( keys ) ), &keys );
}

// the combinations a complement does not offer do not compile: a union with the rest of the key space is wanted for no operand but an interval
template <typename Bitmap>
constexpr bool complement_forms_hold() {
    using frsr::roaring::complement;
    using frsr::roaring::key_range;
    using frsr::roaring::typed_view;
    using view = typed_view<Bitmap>;
    using bits = std::bitset<64>;
    return
        requires ( Bitmap & b, complement<key_range> c ) { b |= c; b ^= c; b &= c; b -= c; b.or_with( c ); b.xor_with( c ); } &&
        requires ( Bitmap & b, complement<view> c ) { b &= c; b -= c; b.and_with( c ); b.andnot_with( c ); } &&
        requires ( Bitmap & b, complement<Bitmap> c ) { b &= c; b -= c; } &&
        requires ( Bitmap const & b, complement<view> c ) { b.intersects( c ); b.and_cardinality( c ); b.is_subset_of( c ); } &&
        requires ( Bitmap const & b, complement<Bitmap> c ) { b.intersects( c ); b.and_cardinality( c ); b.is_subset_of( c ); } &&
        !requires ( Bitmap & b, complement<view> c ) { b |= c; } &&
        !requires ( Bitmap & b, complement<view> c ) { b ^= c; } &&
        !requires ( Bitmap & b, complement<view> c ) { b.or_with( c ); } &&
        !requires ( Bitmap & b, complement<bits> c ) { b |= c; } &&
        !requires ( Bitmap & b, complement<Bitmap> c ) { b |= c; } &&
        !requires ( Bitmap & b, complement<Bitmap> c ) { b ^= c; } &&
        !requires ( Bitmap & b, complement<std::span<std::uint32_t const>> c ) { b.xor_with( c ); } &&
        !requires ( Bitmap const & b, complement<view> c ) { b.xor_cardinality( c ); } &&
        !requires ( Bitmap const & b, complement<key_range> c ) { b.xor_cardinality( c ); } &&
        !requires ( Bitmap const & b, complement<view> c ) { b.is_superset_of( c ); } &&
        !requires ( Bitmap const & b, complement<view> c ) { b.equals( c ); } &&
        !requires ( Bitmap const & b, complement<key_range> c ) { b.equals( c ); };
}
static_assert( complement_forms_hold<frsr::roaring::bitmap<std::uint32_t>>() );
static_assert( complement_forms_hold<frsr::roaring::bitmap<std::uint64_t, frsr::roaring::default_container_set<std::uint64_t>, frsr::roaring::detail::cow_atomic_refcount>>() );
static_assert( complement_forms_hold<frsr::roaring::bitmap<std::uint16_t>>() );

// the union and exclusive-or with the complement of an interval, the one complement that has them: the two intervals
// outside it, including the ends of the key space and the intervals with nothing outside or everything
TYPED_TEST( InteropTest, ComplementOfAnInterval ) {
    using Bitmap = TypeParam;
    using key_type = typename Bitmap::key_type;
    using frsr::roaring::key_range;
    constexpr std::uint64_t max{ domain<Bitmap>::max };
    if constexpr ( std::numeric_limits<key_type>::digits > 32 || ( max > 0xFFFF && std::variant_size_v<typename Bitmap::container_set_type> < 3 ) ) {
        return;   // the rest of a 64-bit key space cannot be materialized, nor that of 32-bit keys where a full chunk takes 8 KB (no run containers)
    } else {
        // a domain that can be listed whole is compared whole; a wider one by its size and by probing keys around the interval
        constexpr bool listable{ max <= 0xFFFF };
        std::mt19937_64 rng{ 1212 };
        std::vector<key_range> ranges{
            key_range{}, key_range::all(), key_range::closed( 0, 0 ), key_range::closed( max, max ),
            key_range::closed( 0, max ), key_range::closed( 0, max / 2 ), key_range::closed( max / 2, max ),
            key_range::closed( 1, max - 1 ), key_range::closed( max / 4, max / 4 + 70000 ), key_range::from( max ), key_range::from( 1 ), key_range::closed( max / 3, max / 3 ),
            key_range::closed( max / 2, max + 5 ),      // reaches past the key type
            key_range::closed( max + 1, max + 9 )       // entirely past it: nothing of it is a key
        };
        for ( int round{ 0 }; round < scale; ++round ) {
            auto const content{ random_content<Bitmap>( rng ) };
            for ( bool const optimize : { false, true } ) {
                auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
                for ( auto const & range : ranges ) {
                    for ( bool const exclusive : { false, true } ) {
                        Bitmap b{ bitmap };
                        if ( exclusive ) { b ^= !range; } else { b |= !range; }
                        ASSERT_EQ( b.validate(), nullptr );
                        auto const inside{ [ & ]( std::uint64_t const key ) { return !range.empty() && range.first <= key && key <= range.last; } };
                        auto const was{ [ & ]( std::uint64_t const key ) { return std::binary_search( content.begin(), content.end(), key ); } };
                        auto const expected{ [ & ]( std::uint64_t const key ) { return inside( key ) ? was( key ) : ( exclusive ? !was( key ) : true ); } };
                        auto const what{ std::string{ exclusive ? "^=" : "|=" } + " !range [" + std::to_string( range.first ) + ", " + std::to_string( range.last ) + "]" };
                        if constexpr ( listable ) {
                            oracle_set want;
                            for ( std::uint64_t key{ 0 }; key <= max; ++key ) { if ( expected( key ) ) { want.push_back( key ); } }
                            expect_holds( b, want, what );
                        } else {
                            std::uint64_t inside_count{ 0 };
                            if ( !range.empty() && range.first <= max ) { inside_count = std::min( range.last, max ) - range.first + 1; }
                            std::uint64_t members_inside{ 0 };
                            for ( auto const key : content ) { members_inside += inside( key ); }
                            auto const members_outside{ content.size() - members_inside };
                            auto const outside_count{ max - inside_count + 1 };
                            auto const want_size{ exclusive ? members_inside + ( outside_count - members_outside ) : members_inside + outside_count };
                            ASSERT_EQ( b.size(), want_size ) << what;
                            std::vector<std::uint64_t> probes{ 0, 1, max, max - 1, max / 2 };
                            if ( !range.empty() ) { for ( auto const edge : { range.first, range.last } ) { for ( std::int64_t d{ -2 }; d <= 2; ++d ) { probes.push_back( edge + static_cast<std::uint64_t>( d ) ); } } }
                            for ( int i{ 0 }; i < 20 && !content.empty(); ++i ) { probes.push_back( content[ rng() % content.size() ] ); }
                            for ( int i{ 0 }; i < 20; ++i ) { probes.push_back( rng() % ( max + 1 ) ); }
                            for ( auto const key : probes ) {
                                if ( key > max ) { continue; }
                                ASSERT_EQ( b.contains( static_cast<key_type>( key ) ), expected( key ) ) << "key " << key << " " << what;
                            }
                        }
                    }
                }
                if ( ::testing::Test::HasFatalFailure() ) { return; }
            }
        }
    }
}

// the complement of an interval that leaves only the ends of the key space (the overflow cases of the two-interval split), for
// every key width: the union and the exclusive-or add or flip exactly those keys
TYPED_TEST( InteropTest, ComplementOfAnIntervalAtTheEnds ) {
    using Bitmap = TypeParam;
    using frsr::roaring::key_range;
    constexpr std::uint64_t max{ domain<Bitmap>::max };
    struct ends { key_range range; oracle_set outside; };
    std::vector<ends> cases{
        { key_range::closed( 1, max - 1 ), { 0, max } },
        { key_range::from( 1 ), { 0 } },
        { key_range::closed( 0, max - 1 ), { max } },
        { key_range::closed( 0, max ), {} }
    };
    if constexpr ( max < std::numeric_limits<std::uint64_t>::max() ) {
        cases.push_back( { key_range::closed( 0, max + 7 ), {} } );   // reaches past the key type: still all of it
    }
    std::mt19937_64 rng{ 1313 };
    for ( int round{ 0 }; round < scale; ++round ) {
        auto content{ random_content<Bitmap>( rng ) };
        content.push_back( 0 );
        content.push_back( max );
        finalize( content );
        for ( bool const optimize : { false, true } ) {
            auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
            for ( auto const & [ range, outside ] : cases ) {
                Bitmap united{ bitmap };
                united |= !range;
                expect_holds( united, unite( content, outside ), "|= !range at the ends" );
                Bitmap flipped{ bitmap };
                flipped ^= !range;
                expect_holds( flipped, symmetric_difference( content, outside ), "^= !range at the ends" );
                Bitmap intersected{ bitmap };
                intersected &= !range;
                expect_holds( intersected, intersection( content, outside ), "&= !range at the ends" );
                Bitmap removed{ bitmap };
                removed -= !range;
                expect_holds( removed, difference( content, outside ), "-= !range at the ends" );
                EXPECT_EQ( bitmap.and_cardinality( !range ), intersection( content, outside ).size() );
                if ( ::testing::Test::HasFatalFailure() ) { return; }
            }
        }
    }
}

// a bitmap updated by a view of itself
TYPED_TEST( InteropTest, ViewOfTheBitmapItself ) {
    using Bitmap = TypeParam;
    using frsr::roaring::view_of;
    std::mt19937_64 rng{ 1010 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        for ( bool const optimize : { false, true } ) {
            auto const original{ make_bitmap<Bitmap>( content, optimize ) };
            Bitmap b{ original };
            b &= view_of( b );
            expect_holds( b, content, "&= itself" );
            b |= view_of( b );
            expect_holds( b, content, "|= itself" );
            EXPECT_TRUE( b.intersects( view_of( b ) ) == !content.empty() );
            EXPECT_EQ( b.and_cardinality( view_of( b ) ), content.size() );
            EXPECT_EQ( b.xor_cardinality( view_of( b ) ), 0U );
            EXPECT_TRUE( b.is_subset_of( view_of( b ) ) );
            EXPECT_TRUE( b.is_superset_of( view_of( b ) ) );
            EXPECT_TRUE( b.equals( view_of( b ) ) );
            Bitmap c{ original };
            c ^= view_of( c );
            expect_holds( c, {}, "^= itself" );
            Bitmap d{ original };
            d -= view_of( d );
            expect_holds( d, {}, "-= itself" );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

TYPED_TEST( InteropTest, BitStringOperands ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 77 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        auto const bitmap{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        with_word_operands<Bitmap>( rng, [ & ]( auto const & operand, oracle_set const & members, std::string const & what ) {
            check_operand( bitmap, content, operand, members, what );
        } );
        auto const check{ [ & ]( auto const & operand, oracle_set const & members, std::string const & what ) {
            check_operand( bitmap, content, operand, members, what );
        } };
        with_std_bitset<1, Bitmap>( rng, check );
        with_std_bitset<63, Bitmap>( rng, check );
        with_std_bitset<64, Bitmap>( rng, check );
        with_std_bitset<65, Bitmap>( rng, check );
        with_std_bitset<1000, Bitmap>( rng, check );
        with_std_bitset<65536, Bitmap>( rng, check );
        with_std_bitset<70001, Bitmap>( rng, check );
        with_std_bitset<300000, Bitmap>( rng, check );   // too large to copy to the stack
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( InteropTest, KeyRangeOperands ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 5150 };
    for ( int round{ 0 }; round < 2 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        auto const bitmap{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        with_range_operands<Bitmap>( rng, [ & ]( auto const & operand, oracle_set const & members, std::string const & what ) {
            check_operand( bitmap, content, operand, members, what );
        } );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}


// intervals against a single chunk of each container kind, with ends on and around the chunk's edges, word
// edges and run edges
TYPED_TEST( InteropTest, IntervalsAgainstEachContainerKind ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 41 };
    for ( auto const kind : { shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::mixed, shape::ends } ) {
        oracle_set content;
        fill_chunk( content, 0, kind, rng, domain<Bitmap>::max );
        if ( kind != shape::full ) { fill_chunk( content, 1, shape::array, rng, domain<Bitmap>::max ); }
        finalize( content );
        for ( bool const optimize : { false, true } ) {
            auto const bitmap{ make_bitmap<Bitmap>( content, optimize ) };
            SCOPED_TRACE( "shape " + std::to_string( static_cast<int>( kind ) ) + ( optimize ? " optimized" : " plain" ) + " arrays/runs/bitsets " + std::to_string( bitmap.statistics().array_containers ) + "/" + std::to_string( bitmap.statistics().run_containers ) + "/" + std::to_string( bitmap.statistics().bitset_containers ) );
            std::vector<std::uint64_t> points{ 0, 1, 63, 64, 4096, 65535, 65536, 65537 };
            for ( int i{ 0 }; i < 3; ++i ) { points.push_back( rng() % 131072 ); }
            if ( !content.empty() ) { for ( int i{ 0 }; i < 2; ++i ) { points.push_back( *std::next( content.begin(), static_cast<std::ptrdiff_t>( rng() % content.size() ) ) ); } }
            for ( auto const first : points ) {
                for ( auto const last : points ) {
                    if ( last < first || first > domain<Bitmap>::max ) { continue; }
                    auto const clipped_last{ std::min( last, domain<Bitmap>::max ) };
                    oracle_set range;
                    for ( auto k{ first }; k <= clipped_last; ++k ) { range.push_back( k ); }
                    check_operand( bitmap, content, key_range::closed( first, last ), range,
                        "interval " + std::to_string( first ) + ".." + std::to_string( last ) + ( optimize ? " optimized" : "" ) );
                    if ( ::testing::Test::HasFatalFailure() || ::testing::Test::HasNonfatalFailure() ) { return; }
                }
            }
        }
    }
}

// Ranges too wide to enumerate: the oracle is the content clipped to the range.
TYPED_TEST( InteropTest, WideKeyRanges ) {
    using Bitmap = TypeParam;
    constexpr auto max{ domain<Bitmap>::max };
    std::mt19937_64 rng{ 3 };
    for ( int round{ 0 }; round < 3 * scale; ++round ) {
        auto const content{ random_content<Bitmap>( rng ) };
        auto const bitmap{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{ { 0, max }, { 1, max }, { 0, max - 1 }, { 1, max - 1 }, { max, max }, { 0, 0 } };
        for ( int i{ 0 }; i < 8; ++i ) {
            auto a{ rng() % ( max / 2 + 1 ) };
            auto b{ a + rng() % ( max / 2 + 1 ) };
            if ( !content.empty() && rng() % 2 ) { a = *std::next( content.begin(), static_cast<std::ptrdiff_t>( rng() % content.size() ) ); b = std::max( a, b ); }
            ranges.emplace_back( a, b );
        }
        for ( auto const & [ first, last ] : ranges ) {
            oracle_set inside;
            for ( auto it{ std::lower_bound( content.begin(), content.end(), first ) }; it != content.end() && *it <= last; ++it ) { inside.push_back( *it ); }
            auto const range{ key_range::closed( first, last ) };
            auto const what{ "range [" + std::to_string( first ) + ", " + std::to_string( last ) + "]" };
            Bitmap and_result{ bitmap };
            and_result.and_with( range );
            expect_holds( and_result, inside, what + " and_with" );
            Bitmap andnot_result{ bitmap };
            andnot_result.andnot_with( range );
            expect_holds( andnot_result, difference( content, inside ), what + " andnot_with" );
            if ( last - first < ( std::uint64_t{ 1 } << 22 ) ) {   // the oracle lists the interval key by key; wider ones are compared with CRoaring
                Bitmap xor_result{ bitmap };
                xor_result.xor_with( range );
                oracle_set flipped;
                for ( auto key{ first }; ; ++key ) {
                    if ( !std::binary_search( inside.begin(), inside.end(), key ) ) { flipped.push_back( key ); }
                    if ( key == last ) { break; }
                }
                expect_holds( xor_result, unite( difference( content, inside ), flipped ), what + " xor_with" );
            }
            EXPECT_EQ( bitmap.intersects( range ), !inside.empty() ) << what;
            EXPECT_EQ( bitmap.and_cardinality( range ), inside.size() ) << what;
            EXPECT_EQ( bitmap.is_subset_of( range ), inside.size() == content.size() ) << what;
            // the range is a subset of the bitmap iff it has as many keys as the content has inside it
            auto const span{ last - first };   // keys - 1
            auto const full{ span < std::numeric_limits<std::uint64_t>::max() && span + 1 == inside.size() };
            EXPECT_EQ( bitmap.is_superset_of( range ), full ) << what;
            EXPECT_EQ( bitmap.equals( range ), full && inside.size() == content.size() ) << what;
        }
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( InteropTest, OrWithWideRangeFillsTheKeySpaceOfSmallKeys ) {
    using Bitmap = TypeParam;
    if constexpr ( std::numeric_limits<typename Bitmap::key_type>::digits <= 32 ) {
        Bitmap bitmap;
        static_cast<void>( bitmap.add( 5 ) );
        bitmap.or_with( key_range::all() );
        EXPECT_EQ( bitmap.size(), domain<Bitmap>::max + 1 );
        EXPECT_EQ( bitmap.validate(), nullptr );
        EXPECT_TRUE( bitmap.contains_range( 0, domain<Bitmap>::max ) );
        EXPECT_TRUE( bitmap.contains( domain<Bitmap>::max ) );
        EXPECT_TRUE( bitmap.is_superset_of( key_range::all() ) );
        EXPECT_TRUE( bitmap.equals( key_range::all() ) );
        bitmap.andnot_with( key_range::closed( 100, domain<Bitmap>::max - 100 ) );
        EXPECT_EQ( bitmap.size(), 200 );
        EXPECT_EQ( bitmap.validate(), nullptr );
    }
}

TYPED_TEST( InteropTest, NamedRangeOperations ) {
    using Bitmap = TypeParam;
    using key_type = typename Bitmap::key_type;
    if constexpr ( domain<Bitmap>::max < 65535 ) { return; }   // the keys below do not exist in a narrower key type
    else {
    Bitmap bitmap;
    bitmap.add_closed_range( 100, 60000 );
    static_cast<void>( bitmap.add( 62000 ) );
    EXPECT_TRUE( bitmap.intersects_range( 0, 101 ) );
    EXPECT_FALSE( bitmap.intersects_range( 0, 100 ) );
    EXPECT_FALSE( bitmap.intersects_range( 100, 100 ) );
    EXPECT_FALSE( bitmap.intersects_range( 150, 100 ) );
    EXPECT_TRUE( bitmap.intersects_range( 61000, 63000 ) );
    EXPECT_FALSE( bitmap.intersects_range( 60001, 62000 ) );
    EXPECT_TRUE( bitmap.contains_range( 100, 60001 ) );
    EXPECT_FALSE( bitmap.contains_range( 99, 60001 ) );
    EXPECT_FALSE( bitmap.contains_range( 100, 60002 ) );
    EXPECT_TRUE( bitmap.contains_range( 5, 5 ) );
    EXPECT_EQ( bitmap.cardinality_in_range( 100, 60001 ), 59901 );
    EXPECT_EQ( bitmap.cardinality_in_range( 0, 0 ), 0 );
    EXPECT_EQ( bitmap.cardinality_in_range( 0, 63000 ), 59902 );
    auto copy{ bitmap };
    copy.and_range( 1000, 2000 );
    EXPECT_EQ( copy.size(), 1000 );
    EXPECT_EQ( copy.front(), key_type{ 1000 } );
    EXPECT_EQ( copy.back(), key_type{ 1999 } );
    EXPECT_EQ( copy.validate(), nullptr );
    }
}

TYPED_TEST( InteropTest, EmptyOperandsAndBitmaps ) {
    using Bitmap = TypeParam;
    Bitmap empty;
    Bitmap some{ std::vector<typename Bitmap::key_type>{ 1, 2, 3, static_cast<typename Bitmap::key_type>( 70000 > domain<Bitmap>::max ? 4 : 70000 ) } };
    std::vector<std::uint32_t> none;
    for ( auto const * what : { "empty bitmap, empty operand", "bitmap, empty operand" } ) {
        auto const & b{ what[ 0 ] == 'e' ? empty : some };
        auto copy{ b };
        EXPECT_FALSE( b.intersects( none ) ) << what;
        EXPECT_FALSE( b.intersects( key_range{} ) ) << what;
        EXPECT_EQ( b.and_cardinality( none ), 0 ) << what;
        EXPECT_EQ( b.is_subset_of( none ), b.empty() ) << what;
        EXPECT_TRUE( b.is_superset_of( none ) ) << what;
        EXPECT_TRUE( b.is_superset_of( key_range{} ) ) << what;
        EXPECT_EQ( b.equals( none ), b.empty() ) << what;
        copy.or_with( none );
        EXPECT_TRUE( copy == b ) << what;
        copy.andnot_with( none );
        EXPECT_TRUE( copy == b ) << what;
        copy.andnot_with( key_range{} );
        EXPECT_TRUE( copy == b ) << what;
        copy.or_with( key_range{} );
        EXPECT_TRUE( copy == b ) << what;
        copy.and_with( none );
        EXPECT_TRUE( copy.empty() ) << what;
        EXPECT_EQ( copy.validate(), nullptr ) << what;
    }
    auto copy{ some };
    copy.and_with( std::bitset<0>{} );
    EXPECT_TRUE( copy.empty() );
    empty.or_with( std::bitset<16>{ 0xA5A5 } );
    EXPECT_EQ( empty.size(), 8 );
    EXPECT_EQ( empty.validate(), nullptr );
}

// the result must not point into the operand: read it after the operand is gone
TYPED_TEST( InteropTest, ResultsDoNotAliasTheOperand ) {
    using Bitmap = TypeParam;
    constexpr auto max{ domain<Bitmap>::max };
    std::mt19937_64 rng{ 99 };
    auto const content{ random_content<Bitmap>( rng ) };
    auto const reference{ make_bitmap<Bitmap>( content, false ) };
    Bitmap or_bitmap;
    Bitmap and_bitmap{ reference };
    Bitmap or_into_content{ reference };
    oracle_set string_members;
    oracle_set all_members;
    {
        auto const string{ random_bit_string( rng, 0, 200000, max ) };
        string_members = string.members;
        auto words{ std::make_unique<std::vector<std::uint64_t>>( string.words ) };
        or_bitmap.or_with( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ *words }, 0, 200000 } );
        or_into_content.or_with( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ *words }, 0, 200000 } );
        and_bitmap.and_with( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ *words }, 0, 200000 } );
        std::fill( words->begin(), words->end(), 0xDEADBEEFDEADBEEFULL );
        auto bits{ std::make_unique<std::bitset<70000>>() };
        bits->set( 3 ); bits->set( 69999 ); bits->set( 40000 );
        or_bitmap.or_with( *bits );
        or_into_content.or_with( *bits );
        bits->flip();
        std::vector<std::uint32_t> keys{ 5, 9, 70001, 70002 };
        or_bitmap.or_with( keys );
        or_into_content.or_with( keys );
        keys.assign( keys.size(), 0xFFFFFFFFU );
        all_members = string_members;
        for ( auto const key : { 3, 69999, 40000, 5, 9, 70001, 70002 } ) { all_members.push_back( key ); }
        finalize( all_members );
    }
    expect_holds( or_bitmap, in_domain( all_members, max ), "or into empty" );
    expect_holds( or_into_content, unite( content, in_domain( all_members, max ) ), "or into content" );
    expect_holds( and_bitmap, intersection( content, string_members ), "and" );
}

// copies that share storage (a refcounted policy) are not disturbed by an operation on one of them
TEST( InteropShared, OperationsDoNotDisturbCopies ) {
    using Bitmap = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>;
    std::mt19937_64 rng{ 12 };
    auto const content{ random_content<Bitmap>( rng ) };
    auto const original{ make_bitmap<Bitmap>( content, true ) };
    auto const related{ random_related<Bitmap>( rng, content, false ) };
    std::vector<std::uint32_t> keys( related.begin(), related.end() );
    for ( int which{ 0 }; which < 4; ++which ) {
        auto copy{ original };
        auto another{ original };
        if ( which == 0 ) { copy.and_with( keys ); } else if ( which == 1 ) { copy.or_with( keys ); } else if ( which == 2 ) { copy.andnot_with( keys ); } else { copy.xor_with( keys ); }
        expect_holds( another, content, "the copy" );
        expect_holds( original, content, "the original" );
    }
}

TEST( InteropKeyRange, Construction ) {
    EXPECT_TRUE( key_range{}.empty() );
    EXPECT_TRUE( key_range::closed( 5, 4 ).empty() );
    EXPECT_FALSE( key_range::closed( 5, 5 ).empty() );
    EXPECT_TRUE( key_range::half_open( 5, 5 ).empty() );
    EXPECT_TRUE( key_range::half_open( 6, 5 ).empty() );
    EXPECT_TRUE( key_range::half_open( 0, 0 ).empty() );
    EXPECT_EQ( key_range::half_open( 5, 6 ).first, 5 );
    EXPECT_EQ( key_range::half_open( 5, 6 ).last, 5 );
    EXPECT_FALSE( key_range::all().empty() );
    EXPECT_EQ( key_range::from( 9 ).last, std::numeric_limits<std::uint64_t>::max() );
}

// keys past the bitmap's key type are not members: they match nothing, and an operand that holds one is
// not a subset of the bitmap
TEST( InteropDomain, WiderKeysAreNotMembers ) {
    using Bitmap = frsr::roaring::bitmap<std::uint32_t>;
    Bitmap bitmap{ std::vector<std::uint32_t>{ 1, 2, 0xFFFFFFFFU } };
    std::vector<std::uint64_t> beyond{ 1, 0x1'0000'0001ULL };
    EXPECT_TRUE( bitmap.intersects( beyond ) );
    EXPECT_EQ( bitmap.and_cardinality( beyond ), 1 );
    EXPECT_FALSE( bitmap.is_superset_of( beyond ) );
    EXPECT_FALSE( bitmap.is_subset_of( beyond ) );
    EXPECT_FALSE( bitmap.equals( beyond ) );
    EXPECT_FALSE( bitmap.is_superset_of( sorted_keys{ beyond } ) );
    auto copy{ bitmap };
    copy.or_with( beyond );
    EXPECT_EQ( copy.size(), 3 );
    EXPECT_EQ( copy.validate(), nullptr );
    copy = bitmap;
    copy.and_with( beyond );
    EXPECT_EQ( copy.to_vector(), ( std::vector<std::uint32_t>{ 1 } ) );
    // an interval is clipped to the key type, unlike keys listed one by one
    EXPECT_TRUE( bitmap.is_superset_of( key_range::closed( 0xFFFFFFFFULL, 0x1'0000'0000ULL ) ) );
    EXPECT_TRUE( bitmap.contains_range( 0xFFFFFFFFU, 0xFFFFFFFFU ) );
    std::bitset<64> high;
    high.set( 63 );
    EXPECT_FALSE( bitmap.is_superset_of( bits_at( high, 0xFFFFFFFFULL - 10 ) ) );
    EXPECT_TRUE( bitmap.is_superset_of( bits_at( std::bitset<64>{ 1 }, 0xFFFFFFFFULL ) ) );
}

TEST( InteropDomain, SixtyFourBitStringEndingAtTheTypeMaximum ) {
    using Bitmap = frsr::roaring::bitmap<std::uint64_t>;
    constexpr auto max{ std::numeric_limits<std::uint64_t>::max() };
    Bitmap bitmap;
    static_cast<void>( bitmap.add( max ) );
    static_cast<void>( bitmap.add( max - 70 ) );
    static_cast<void>( bitmap.add( 5 ) );
    std::bitset<128> bits;
    bits.set( 70 );
    bits.set( 127 );
    EXPECT_EQ( bitmap.and_cardinality( bits_at( bits, max - 127 ) ), 1 );   // max - 127 + 127 = max; + 70 = max - 57
    bits.set( 57 );   // the key max - 70
    EXPECT_EQ( bitmap.and_cardinality( bits_at( bits, max - 127 ) ), 2 );
    auto copy{ bitmap };
    copy.or_with( bits_at( bits, max - 127 ) );
    EXPECT_EQ( copy.size(), 4 );
    EXPECT_TRUE( copy.contains( max - 57 ) );
    EXPECT_EQ( copy.validate(), nullptr );
    // a string that runs past the type: the part beyond max stands for keys that do not exist
    std::bitset<128> past;
    past.set( 100 );
    EXPECT_TRUE( bitmap.is_superset_of( bits_at( std::bitset<128>{}, max - 10 ) ) );
    EXPECT_FALSE( bitmap.is_superset_of( bits_at( past, max - 60 ) ) );
    EXPECT_EQ( bitmap.and_cardinality( key_range::from( max - 100 ) ), 2 );
    EXPECT_TRUE( bitmap.contains_range( max, max ) );
    EXPECT_FALSE( bitmap.contains_range( max - 1, max ) );
    EXPECT_EQ( bitmap.cardinality_in_range( 0, max ), 2 );   // [0, max): everything but max itself
}

TEST( InteropDeclaredSorted, SortedKeysAreTrustedAndGroupedByChunk ) {
    using Bitmap = frsr::roaring::bitmap<std::uint32_t>;
    std::vector<std::uint32_t> keys;
    for ( std::uint32_t chunk{ 0 }; chunk < 40; ++chunk ) {
        for ( std::uint32_t i{ 0 }; i < 5000; i += 1 + chunk ) { keys.push_back( ( chunk << 16 ) + i ); }
    }
    Bitmap bitmap;
    bitmap.add_closed_range( 5, 3'000'000 );
    auto copy{ bitmap };
    copy.and_with( sorted_keys{ keys } );
    Bitmap expected{ keys };
    expected.and_with( key_range::closed( 5, 3'000'000 ) );
    EXPECT_TRUE( copy == expected );
    EXPECT_EQ( copy.validate(), nullptr );
    EXPECT_EQ( bitmap.and_cardinality( sorted_keys{ keys } ), expected.size() );
}

} // namespace

