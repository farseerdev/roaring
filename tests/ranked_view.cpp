// The ranked frozen image: count_below / index_of / select / iteration against a sorted vector, over every container
// kind and chunk layout; the format's validation; version 1 images still open; the file-mapped round trip.

#include "ranked_view_support.hpp"

#include <frsr/roaring/persistent_bitmap.hpp>

#include <cstdio>
#include <cstring>
#include <random>
#include <string_view>

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

// The same image in storage that is `extra` bytes longer: the view must still report where the image ends.
[[nodiscard]] image with_trailing_bytes( image const & source, std::size_t const extra ) {
    image result;
    result.words.assign( ( source.size + extra + 7 ) / 8, 0xA5A5'A5A5'A5A5'A5A5ULL );
    result.size = source.size + extra;
    std::memcpy( result.writable().data(), source.bytes().data(), source.size );
    return result;
}

TEST( FrsrRoaringRanked, ByteSizeIsTheImageForBothVersions ) {
    auto const table{ make_bitmap( shapes()[ 9 ].ids, build::optimized ) };
    for ( auto const ranked : { false, true } ) {
        auto const frozen{ freeze( table, { .ranked = ranked } ) };
        EXPECT_EQ( table_bitmap::frozen_view_from( frozen.bytes() ).byte_size(), frozen.size ) << ranked;
        auto const padded{ with_trailing_bytes( frozen, 24 ) };
        auto const view{ table_bitmap::frozen_view_from( padded.bytes() ) };
        ASSERT_TRUE( view ) << ranked;
        EXPECT_EQ( view.byte_size(), frozen.size ) << ranked;
        EXPECT_EQ( view.materialize().to_vector(), table.to_vector() ) << ranked;
    }
    auto const ranked{ freeze( table, { .ranked = true } ) };
    EXPECT_EQ( table_view::from( ranked.bytes() ).byte_size(), ranked.size );
    EXPECT_EQ( table_bitmap{}.frozen_size_bytes(), header_bytes );
    auto const empty{ freeze( table_bitmap{}, {} ) };
    EXPECT_EQ( table_bitmap::frozen_view_from( empty.bytes() ).byte_size(), header_bytes );
}

TEST( FrsrRoaringRanked, TheFourHeaderBytesAtOffsetTwelveAreNotRead ) {
    auto const table{ mixed_table() };
    for ( auto const ranked : { false, true } ) {
        auto image{ freeze( table, { .ranked = ranked } ) };
        poke<std::uint32_t>( image, 12, 0xDEAD'BEEFU );
        auto const view{ table_bitmap::frozen_view_from( image.bytes() ) };
        ASSERT_TRUE( view ) << ranked;
        EXPECT_EQ( view.materialize().to_vector(), table.to_vector() ) << ranked;
        if ( ranked ) {
            auto const all{ table.to_vector() };
            expect_matches_oracle( table_view::from( image.bytes() ), std::vector<std::uint32_t>( all.begin(), all.end() ), "garbage in the header" );
        }
    }
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

// ---- the image of the plain writer ----

[[nodiscard]] image from_hex( std::string_view const hex ) {
    image result;
    result.size = hex.size() / 2;
    result.words.resize( ( result.size + 7 ) / 8 );
    for ( std::size_t i{ 0 }; i < result.size; ++i ) {
        result.writable()[ i ] = static_cast<std::byte>( std::stoul( std::string{ hex.substr( 2 * i, 2 ) }, nullptr, 16 ) );
    }
    return result;
}

[[nodiscard]] std::uint64_t fnv1a( std::span<std::byte const> const bytes ) {
    std::uint64_t hash{ 14695981039346656037ULL };
    for ( auto const b : bytes ) {
        hash ^= static_cast<std::uint8_t>( b );
        hash *= 1099511628211ULL;
    }
    return hash;
}

TEST( FrsrRoaringRanked, AnUnrankedImageIsByteForByteTheImageOfTheWriterBeforeRankData ) {
    // Written by the plain writer as it was before the ranked option existed: an array, a run and an array chunk.
    std::vector<std::uint32_t> small{ 3, 70, 4000, 65'535, 2 * 65'536 + 1, 2 * 65'536 + 5, 2 * 65'536 + 9 };
    for ( std::uint32_t i{ 65'536 + 100 }; i < 65'536 + 4100; ++i ) {
        small.push_back( i );
    }
    std::ranges::sort( small );
    auto const golden{ from_hex(
        "52424631200001000300000000000000a70f00000000000000000000000000007800000008000000040000000400000001000000000000000100000000000000"
        "800000000400000001000000a00f00000200000000000000020000000000000084000000060000000300000003000000010000000000000003004600a00f"
        "ffff64000310010005000900"
    ) };
    auto const written{ freeze( make_bitmap( small, build::keep_bitsets ), {} ) };
    ASSERT_EQ( written.size, golden.size );
    EXPECT_EQ( std::memcmp( written.bytes().data(), golden.bytes().data(), golden.size ), 0 );

    // The same for a bitset chunk (its 8 KiB are compared by size and hash), an array chunk and a run chunk.
    std::vector<std::uint32_t> large;
    for ( std::uint32_t i{ 0 }; i < 65'536; i += 3 ) {
        large.push_back( i );
    }
    for ( std::uint32_t i{ 0 }; i < 50; ++i ) {
        large.push_back( 65'536 + i * 7 );
    }
    for ( std::uint32_t i{ 100 }; i < 60'000; ++i ) {
        large.push_back( 2 * 65'536 + i );
    }
    auto const bulky{ freeze( make_bitmap( large, build::keep_bitsets ), {} ) };
    EXPECT_EQ( bulky.size, 8416U );
    EXPECT_EQ( fnv1a( bulky.bytes() ), 0x7c5ce03bfc8f6462ULL );
}

// ---- hostile images ----

// An image of `chunks` full chunks, each one run [0, 65535] (so 65536 values and a rank block of one count): the shape
// that adds up to 2^32 values in the fewest bytes.
[[nodiscard]] image full_chunk_runs( std::uint32_t const chunks ) {
    constexpr std::size_t payload_bytes{ 4 };
    constexpr std::size_t rank_bytes{ 2 };
    auto const payloads{ header_bytes + std::size_t{ chunks } * entry_bytes };
    image result;
    result.size = payloads + std::size_t{ chunks } * ( payload_bytes + rank_bytes );
    result.words.resize( ( result.size + 7 ) / 8 );
    auto const good{ freeze( make_bitmap( std::vector<std::uint32_t>{ 7 }, build::plain ), { .ranked = true } ) };
    std::memcpy( result.writable().data(), good.bytes().data(), header_bytes );   // magic, key bits
    poke<std::uint16_t>( result, 6, 2 );
    poke<std::uint32_t>( result, 8, chunks );
    poke<std::uint64_t>( result, 16, std::uint64_t{ chunks } * 65'536 );
    for ( std::uint32_t c{ 0 }; c < chunks; ++c ) {
        auto const entry{ header_bytes + std::size_t{ c } * entry_bytes };
        auto const offset{ payloads + std::size_t{ c } * ( payload_bytes + rank_bytes ) };
        poke<std::uint64_t>( result, entry, c );
        poke<std::uint32_t>( result, entry + 8, static_cast<std::uint32_t>( offset ) );
        poke<std::uint32_t>( result, entry + 12, payload_bytes );
        poke<std::uint32_t>( result, entry + 16, 1U );
        poke<std::uint32_t>( result, entry + 20, 65'536U );
        poke<std::uint8_t>( result, entry + 24, 2 );
        poke<std::uint32_t>( result, entry + 28, c * 65'536U );
        poke<std::uint16_t>( result, offset + 2, 65'535 );   // the run's end
    }
    return result;
}

TEST( FrsrRoaringRanked, TwoToTheThirtyTwoValuesAreRejectedAlone ) {
    // everything else adds up: the bases, the cardinalities, their sum, the offsets
    auto const below{ full_chunk_runs( 65'535 ) };
    auto const view{ table_view::from( below.bytes() ) };
    ASSERT_TRUE( view );
    EXPECT_EQ( view.size(), 0xFFFF'0000U );
    EXPECT_EQ( view.select( 0xFFFE'FFFFU ), 0xFFFE'FFFFU );
    EXPECT_EQ( view.count_below( 0xFFFE'FFFFU ), 0xFFFE'FFFFU );
    auto const at_the_limit{ full_chunk_runs( 65'536 ) };
    EXPECT_FALSE( table_view::from( at_the_limit.bytes() ) );
    EXPECT_FALSE( table_bitmap::frozen_view_from( at_the_limit.bytes() ) );
}

TEST( FrsrRoaringRanked, RejectsCountsThatDoNotDescribeThePayload ) {
    auto const table{ mixed_table() };
    auto const good{ freeze( table, { .ranked = true } ) };
    ASSERT_TRUE( table_view::from( good.bytes() ) );
    std::vector<std::uint8_t> kinds;
    for ( std::size_t c{ 0 }; c < 3; ++c ) {
        kinds.push_back( static_cast<std::uint8_t>( good.bytes()[ header_bytes + c * entry_bytes + 24 ] ) );
    }
    ASSERT_EQ( kinds, ( std::vector<std::uint8_t>{ 1, 2, 3 } ) );   // array, run, bitset

    auto const rejects{ [ & ]( auto && damage ) {
        auto bad{ good };
        damage( bad );
        return !table_view::from( bad.bytes() ) && !table_bitmap::frozen_view_from( bad.bytes() );
    } };
    auto const field{ []( std::size_t const chunk, std::size_t const offset ) { return header_bytes + chunk * entry_bytes + offset; } };
    // fewer array values than the cardinality (the payload size is kept: only the count is wrong)
    EXPECT_TRUE( rejects( [ & ]( image & i ) {
        std::uint32_t count{};
        std::memcpy( &count, i.writable().data() + field( 0, 16 ), 4 );
        poke<std::uint32_t>( i, field( 0, 16 ), count - 1 );
    } ) );
    // a run chunk without runs, or with more runs than values
    EXPECT_TRUE( rejects( [ & ]( image & i ) { poke<std::uint32_t>( i, field( 1, 16 ), 0U ); } ) );
    EXPECT_TRUE( rejects( [ & ]( image & i ) { poke<std::uint32_t>( i, field( 1, 20 ), 1U ); } ) );
    // a bitset that says it has fewer words than its rank block has
    EXPECT_TRUE( rejects( [ & ]( image & i ) { poke<std::uint32_t>( i, field( 2, 16 ), 1U ); } ) );
    // a cardinality beyond the chunk
    EXPECT_TRUE( rejects( [ & ]( image & i ) { poke<std::uint32_t>( i, field( 2, 20 ), 65'537U ); } ) );
}

// The image with every count of every rank block replaced by fill( running index ).
template <typename Fill>
[[nodiscard]] image with_rank_blocks( image const & good, Fill && fill ) {
    auto result{ good };
    std::uint32_t chunks{};
    std::memcpy( &chunks, good.bytes().data() + 8, sizeof( chunks ) );
    std::size_t at{};
    for ( std::uint32_t c{ 0 }; c < chunks; ++c ) {
        auto const entry{ header_bytes + std::size_t{ c } * entry_bytes };
        std::uint32_t offset{};
        std::uint32_t bytes{};
        std::uint32_t count{};
        std::memcpy( &offset, good.bytes().data() + entry + 8, 4 );
        std::memcpy( &bytes, good.bytes().data() + entry + 12, 4 );
        std::memcpy( &count, good.bytes().data() + entry + 16, 4 );
        auto const kind{ static_cast<std::uint8_t>( good.bytes()[ entry + 24 ] ) };
        auto const ranks{ kind == 2 ? count : kind == 3 ? 1024U : 0U };
        for ( std::uint32_t r{ 0 }; r < ranks; ++r ) {
            poke<std::uint16_t>( result, offset + bytes + 2 * r, fill( at++ ) );
        }
    }
    return result;
}

TEST( FrsrRoaringRanked, ReadsStayInTheImageWhateverTheRankBlocksSay ) {
    // Meant for a sanitizer: the answers are wrong, but every read stays inside the payload and rank block of a chunk.
    auto const table{ mixed_table() };
    auto const good{ freeze( table, { .ranked = true } ) };
    auto const all{ table.to_vector() };
    std::vector<std::uint32_t> const ids( all.begin(), all.end() );
    std::mt19937 rng{ 11 };
    std::vector<std::pair<std::string, image>> damaged;
    damaged.emplace_back( "all_zero", with_rank_blocks( good, []( std::size_t ) -> std::uint16_t { return 0; } ) );
    damaged.emplace_back( "all_max", with_rank_blocks( good, []( std::size_t ) -> std::uint16_t { return 0xFFFF; } ) );
    damaged.emplace_back( "descending", with_rank_blocks( good, []( std::size_t const i ) { return static_cast<std::uint16_t>( 0xFFFF - i ); } ) );
    damaged.emplace_back( "random", with_rank_blocks( good, [ & ]( std::size_t ) { return static_cast<std::uint16_t>( rng() ); } ) );
    for ( auto const & [ name, bad ] : damaged ) {
        SCOPED_TRACE( name );
        auto const view{ table_view::from( bad.bytes() ) };
        ASSERT_TRUE( view );   // the contents of the rank blocks are not read when the image is opened
        std::uint64_t sink{};
        for ( std::uint32_t i{ 0 }; i < view.size(); ++i ) {
            sink += view.select( i );
        }
        for ( auto const x : probes( ids ) ) {
            sink += view.count_below( x ) + view.index_of( x ) + view.contains( x );
        }
        view.for_each_indexed( [ & ]( std::uint32_t const id, std::uint32_t const index ) { sink += id + index; } );
        EXPECT_NE( sink, 0U );
    }
}

TEST( FrsrRoaringRanked, SelectInAWordToleratesARankTheWordDoesNotHave ) {
    EXPECT_EQ( detail::select_in_word( 0b1011, 0 ), 0U );
    EXPECT_EQ( detail::select_in_word( 0b1011, 2 ), 3U );
    EXPECT_LE( detail::select_in_word( 0b1011, 3 ), 64U );
    EXPECT_LE( detail::select_in_word( 0b1011, 64 ), 64U );
    EXPECT_LE( detail::select_in_word( 1, 0xFFFF'FFFFU ), 64U );
    EXPECT_LE( detail::select_in_word( 0, 7 ), 64U );
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
