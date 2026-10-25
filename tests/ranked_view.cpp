// The ranked frozen image: count_below / index_of / select / iteration against a sorted vector, over every container
// kind and chunk layout; the format's validation; version 1 images still open; the file-mapped round trip.

#include "ranked_view_support.hpp"

#include <frsr/roaring/persistent_bitmap.hpp>

#include <cstdio>
#include <cstring>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::ranked_test;

constexpr std::size_t header_bytes{ 24 };
constexpr std::size_t entry_bytes{ 32 };

template <typename T>
void poke( image & target, std::size_t const offset, T const value ) {
    std::memcpy( target.writable().data() + offset, &value, sizeof( value ) );
}

void expect_matches_oracle( table_view const & view, std::vector<std::uint32_t> const & ids, std::string const & context ) {
    SCOPED_TRACE( context );
    ASSERT_TRUE( view );
    ASSERT_EQ( view.size(), ids.size() );
    EXPECT_EQ( view.empty(), ids.empty() );
    if ( !ids.empty() ) {
        EXPECT_EQ( view.front(), ids.front() );
        EXPECT_EQ( view.back(), ids.back() );
    }
    for ( auto const x : probes( ids ) ) {
        auto const below{ static_cast<std::uint32_t>( std::ranges::lower_bound( ids, x ) - ids.begin() ) };
        auto const present{ below < ids.size() && ids[ below ] == x };
        ASSERT_EQ( view.count_below( x ), below ) << x;
        ASSERT_EQ( view.contains( x ), present ) << x;
        ASSERT_EQ( view.index_of( x ), present ? below : table_view::npos ) << x;
    }
    for ( std::uint32_t i{ 0 }; i < ids.size(); ++i ) {
        ASSERT_EQ( view.select( i ), ids[ i ] ) << i;
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> seen;
    view.for_each_indexed( [ & ]( std::uint32_t const id, std::uint32_t const index ) { seen.emplace_back( id, index ); } );
    ASSERT_EQ( seen.size(), ids.size() );
    for ( std::uint32_t i{ 0 }; i < ids.size(); ++i ) {
        ASSERT_EQ( seen[ i ], std::make_pair( ids[ i ], i ) );
    }
    for ( auto const from : { 0U, 1U, 65'535U, 65'536U, 1'000'000U, 0xFFFF'FFFFU } ) {
        auto const first{ static_cast<std::uint32_t>( std::ranges::lower_bound( ids, from ) - ids.begin() ) };
        std::uint32_t expected{ first };
        view.for_each_indexed_from( from, [ & ]( std::uint32_t const id, std::uint32_t const index ) {
            ASSERT_LT( expected, ids.size() );
            ASSERT_EQ( id, ids[ expected ] );
            ASSERT_EQ( index, expected );
            ++expected;
        } );
        ASSERT_EQ( expected, ids.size() ) << from;
    }
    EXPECT_EQ( view.frozen().materialize().to_vector().size(), ids.size() );
}

TEST( FrsrRoaringRanked, AnswersLikeASortedVectorOnEveryShapeAndBuild ) {
    for ( auto const & [ name, ids ] : shapes() ) {
        for ( auto const how : { build::plain, build::optimized, build::keep_bitsets } ) {
            auto const table{ make_bitmap( ids, how ) };
            auto const frozen{ freeze( table, { .ranked = true } ) };
            expect_matches_oracle( table_view::from( frozen.bytes() ), ids, name + "/" + name_of( how ) );
        }
    }
}

TEST( FrsrRoaringRanked, TheShapesCoverEveryContainerKind ) {
    bool seen[ 4 ]{};
    for ( auto const & [ name, ids ] : shapes() ) {
        for ( auto const how : { build::plain, build::optimized, build::keep_bitsets } ) {
            auto const frozen{ freeze( make_bitmap( ids, how ), { .ranked = true } ) };
            std::uint32_t chunks{};
            std::memcpy( &chunks, frozen.bytes().data() + 8, sizeof( chunks ) );
            for ( std::uint32_t c{ 0 }; c < chunks; ++c ) {
                seen[ static_cast<std::uint8_t>( frozen.bytes()[ header_bytes + c * entry_bytes + 24 ] ) ] = true;
            }
        }
    }
    EXPECT_TRUE( seen[ 1 ] && seen[ 2 ] && seen[ 3 ] );
}

TEST( FrsrRoaringRanked, ByteSizeIsTheImage ) {
    auto const ids{ shapes()[ 9 ].ids };
    auto const frozen{ freeze( make_bitmap( ids, build::optimized ), { .ranked = true } ) };
    EXPECT_EQ( table_view::from( frozen.bytes() ).byte_size(), frozen.size );
}

TEST( FrsrRoaringRanked, AnUnrankedImageIsByteIdenticalVersionOne ) {
    auto const table{ make_bitmap( shapes()[ 11 ].ids, build::optimized ) };
    auto const plain{ freeze( table, {} ) };
    EXPECT_EQ( plain.size, table.frozen_size_bytes() );
    std::uint16_t version{};
    std::memcpy( &version, plain.bytes().data() + 6, sizeof( version ) );
    EXPECT_EQ( version, 1U );
    std::uint32_t chunks{};
    std::memcpy( &chunks, plain.bytes().data() + 8, sizeof( chunks ) );
    for ( std::uint32_t c{ 0 }; c < chunks; ++c ) {
        std::uint32_t base{ 1 };
        std::memcpy( &base, plain.bytes().data() + header_bytes + c * entry_bytes + 28, sizeof( base ) );
        EXPECT_EQ( base, 0U );
    }
    // a version-1 image still opens and reads, and is not a ranked view
    auto const view{ table_bitmap::frozen_view_from( plain.bytes() ) };
    ASSERT_TRUE( view );
    EXPECT_FALSE( view.ranked() );
    EXPECT_EQ( view.materialize().to_vector(), table.to_vector() );
    EXPECT_FALSE( table_view::from( plain.bytes() ) );
    // the rank data costs bytes and changes nothing else
    auto const ranked{ freeze( table, { .ranked = true } ) };
    EXPECT_GT( ranked.size, plain.size );
    EXPECT_EQ( table_bitmap::frozen_view_from( ranked.bytes() ).materialize().to_vector(), table.to_vector() );
}

// a multi-chunk image with an array, a run and a bitset chunk, so each kind of rank block is in play
[[nodiscard]] table_bitmap mixed_table() {
    std::mt19937_64 rng{ 5 };
    auto ids{ merged( {
        thinned( 0, 65'535, 400.0, rng ),
        span_of( 65'536 + 100, 65'536 + 30'000 ),
        thinned( 2 * 65'536, 3 * 65'536 - 1, 2.0, rng ),
    } ) };
    return make_bitmap( ids, build::keep_bitsets );
}

TEST( FrsrRoaringRanked, RejectsAnImageWhoseRankDataDoesNotAddUp ) {
    auto const table{ mixed_table() };
    auto const good{ freeze( table, { .ranked = true } ) };
    ASSERT_TRUE( table_view::from( good.bytes() ) );

    auto const rejects{ [ & ]( auto && damage ) {
        auto bad{ good };
        damage( bad );
        return !table_view::from( bad.bytes() ) && !table_bitmap::frozen_view_from( bad.bytes() );
    } };
    EXPECT_TRUE( rejects( []( image & i ) { poke<std::uint32_t>( i, header_bytes + entry_bytes + 28, 7U ); } ) );   // base of the second chunk
    EXPECT_TRUE( rejects( []( image & i ) { poke<std::uint32_t>( i, header_bytes + 28, 1U ); } ) );              // base of the first chunk
    EXPECT_TRUE( rejects( []( image & i ) { std::uint32_t first{}; std::memcpy( &first, i.writable().data() + header_bytes + 8, 4 ); poke<std::uint32_t>( i, header_bytes + entry_bytes + 8, first ); } ) );   // payloads out of order
    EXPECT_TRUE( rejects( []( image & i ) { i.size -= 1; } ) );                                                  // the last rank block runs past the image
    EXPECT_TRUE( rejects( []( image & i ) { poke<std::uint16_t>( i, 6, 3 ); } ) );                               // an unknown version
    EXPECT_TRUE( rejects( []( image & i ) { poke<std::uint64_t>( i, 16, std::uint64_t{ 1 } << 32 ); } ) );       // 2^32 values
}

TEST( FrsrRoaringRanked, ARankBlockOverlappingTheNextPayloadIsRejected ) {
    // chunk 0 is a bitset (its rank block is 2048 bytes): a second payload starting inside that block must not be accepted
    auto const table{ mixed_table() };
    auto bad{ freeze( table, { .ranked = true } ) };
    std::uint32_t offset{};
    std::memcpy( &offset, bad.writable().data() + header_bytes + 2 * entry_bytes + 8, 4 );
    poke<std::uint32_t>( bad, header_bytes + 2 * entry_bytes + 8, offset - 8U );
    EXPECT_FALSE( table_view::from( bad.bytes() ) );
}

TEST( FrsrRoaringRanked, TheNullViewIsEmptyAndFalse ) {
    table_view const none;
    EXPECT_FALSE( none );
    EXPECT_EQ( none.size(), 0U );
    EXPECT_TRUE( none.empty() );
    EXPECT_FALSE( none.contains( 5 ) );
    EXPECT_EQ( none.count_below( 5 ), 0U );
    EXPECT_EQ( none.index_of( 5 ), table_view::npos );
}

#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)

TEST( FrsrRoaringRanked, PersistentBitmapKeepsTheRanks ) {
    auto const table{ mixed_table() };
    auto const ids{ table.to_vector() };
    auto const file_name{ std::string{ ::testing::TempDir() } + "frsr_ranked.frozen" };
    ASSERT_TRUE( ( persistent_bitmap<std::uint32_t>::store( file_name.c_str(), table, { .ranked = true } ) ) );
    {
        auto opened{ persistent_bitmap<std::uint32_t>::open( file_name.c_str() ) };
        ASSERT_TRUE( opened.has_value() );
        EXPECT_EQ( ( *opened )->to_vector(), std::vector<std::uint32_t>( ids.begin(), ids.end() ) );
        expect_matches_oracle( table_view::from( opened->frozen() ), std::vector<std::uint32_t>( ids.begin(), ids.end() ), "mapped" );
    }
    ASSERT_TRUE( ( persistent_bitmap<std::uint32_t>::store( file_name.c_str(), table ) ) );
    {
        auto opened{ persistent_bitmap<std::uint32_t>::open( file_name.c_str() ) };
        ASSERT_TRUE( opened.has_value() );
        EXPECT_FALSE( table_view::from( opened->frozen() ) );
        EXPECT_EQ( ( *opened )->size(), table.size() );
    }
    std::remove( file_name.c_str() );
}

#endif

} // namespace
