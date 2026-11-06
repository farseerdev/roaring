// The robustness and portability cases of the reference's portable-format and frozen-view tests
// (tests/robust_deserialization_unit.c, tests/format_portability_unit.c), some of them ported there from the Java
// and Go libraries, replayed against bitmap's portable reader and writer and its own frozen image. The tests carry
// the names of the upstream ones and their data.
//
// Where the contracts differ the test pins ours: the reader rejects an array whose values are not strictly
// ascending (the reference accepts it and leaves the bitmap to fail validation), there is no separate size probe
// (the size a bitmap occupied is its portable_size_in_bytes()), and the frozen image is this library's own format
// (docs/frozen-format.md), so the reference's frozen bytes and cookies have no counterpart.

#include "ranked_view_support.hpp"

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring;
using frsr::roaring::ranked_test::freeze;
using frsr::roaring::ranked_test::image;

using Bitmap = bitmap<std::uint32_t>;
using Bytes  = std::vector<std::byte>;

constexpr std::uint32_t chunk{ 1U << 16 };

[[nodiscard]] Bytes read_fixture( std::string const & name ) {
    std::ifstream in{ std::string{ FRSR_ROARING_TEST_DATA_DIR } + "/" + name, std::ios::binary };
    EXPECT_TRUE( in.is_open() ) << name;
    std::string const text{ std::istreambuf_iterator<char>{ in }, std::istreambuf_iterator<char>{} };
    Bytes bytes( text.size() );
    std::ranges::transform( text, bytes.begin(), []( char const c ) { return static_cast<std::byte>( c ); } );
    return bytes;
}

[[nodiscard]] Bytes bytes_of( std::initializer_list<std::uint8_t> const values ) {
    Bytes bytes( values.size() );
    std::ranges::transform( values, bytes.begin(), []( std::uint8_t const v ) { return static_cast<std::byte>( v ); } );
    return bytes;
}

template <typename T>
void poke( std::span<std::byte> const bytes, std::size_t const offset, T const value ) {
    ASSERT_LE( offset + sizeof( T ), bytes.size() );
    std::memcpy( bytes.data() + offset, &value, sizeof( T ) );
}

template <typename T>
[[nodiscard]] T peek( std::span<std::byte const> const bytes, std::size_t const offset ) {
    T value{};
    std::memcpy( &value, bytes.data() + offset, sizeof( T ) );
    return value;
}

// A dirty destination: every output byte has to be written by the serializer.
[[nodiscard]] Bytes serialized( Bitmap const & bitmap ) {
    Bytes bytes( bitmap.portable_size_in_bytes(), std::byte{ 0xA5 } );
    EXPECT_EQ( bitmap.portable_serialize( bytes ), bytes.size() );
    return bytes;
}

[[nodiscard]] bool rejected( std::span<std::byte const> const bytes ) {
    return !Bitmap::portable_deserialize( bytes ).has_value();
}

// check_portable_round_trip
void expect_round_trip( Bitmap const & bitmap ) {
    auto const bytes{ serialized( bitmap ) };
    auto const loaded{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( loaded.has_value() );
    EXPECT_EQ( loaded->validate(), nullptr );
    EXPECT_TRUE( *loaded == bitmap );
    EXPECT_EQ( loaded->portable_size_in_bytes(), bytes.size() );
}

// valid_deserialize_test: the bytes decode to a valid bitmap of exactly their size, and a byte less is refused.
[[nodiscard]] Bitmap expect_valid( std::span<std::byte const> const bytes ) {
    EXPECT_TRUE( rejected( bytes.first( bytes.size() - 1 ) ) );
    auto loaded{ Bitmap::portable_deserialize( bytes ) };
    EXPECT_TRUE( loaded.has_value() );
    if ( !loaded ) {
        return {};
    }
    EXPECT_EQ( loaded->validate(), nullptr );
    EXPECT_EQ( loaded->portable_size_in_bytes(), bytes.size() );
    return *std::move( loaded );
}

// ---- tests/robust_deserialization_unit.c ----

// test_robust_deserialize8, test_robust_deserialize_invalid_cookie ("0000" in ASCII),
// test_robust_deserialize_without_cookie (an empty file): inputs the Go library found crashing readers.
TEST( UpstreamSerialization, test_robust_deserialize_go_inputs ) {
    for ( auto const * const name : { "crashproneinput8.bin", "crashwithinvalidcookie.bin", "crashwithoutcookie.bin" } ) {
        EXPECT_TRUE( rejected( read_fixture( name ) ) ) << name;
    }
}

// test_robust_deserialize_bad_bitmap: one array container of 1518 values, mostly zero (from the Java library). The
// reference accepts it and fails validation afterwards; here the reader refuses it. The same bytes with the values
// made ascending are accepted, so the refusal is for the order of the values and not for their length.
TEST( UpstreamSerialization, test_robust_deserialize_bad_bitmap ) {
    auto const bytes{ read_fixture( "bad-bitmap.bin" ) };
    constexpr std::size_t header{ 16 };
    constexpr std::uint16_t cardinality{ 1518 };
    ASSERT_EQ( bytes.size(), header + 2U * cardinality );
    EXPECT_TRUE( rejected( bytes ) );

    auto ascending{ bytes };
    for ( std::uint16_t i{ 0 }; i < cardinality; ++i ) {
        poke<std::uint16_t>( ascending, header + 2U * i, i );
    }
    auto const repaired{ Bitmap::portable_deserialize( ascending ) };
    ASSERT_TRUE( repaired.has_value() );
    EXPECT_EQ( repaired->size(), cardinality );
    EXPECT_EQ( repaired->validate(), nullptr );
    EXPECT_TRUE( serialized( *repaired ) == ascending );
}

// deserialize_run_container_end_boundary: a run may end at 65535, the last value of its container, and not past it.
TEST( UpstreamSerialization, deserialize_run_container_end_boundary ) {
    auto data{ bytes_of( {
        0x3B, 0x30, // cookie with runs
        0, 0,       // container count - 1
        0x01,       // run flags: the container is a run container
        0, 0,       // key
        0x17, 0x02, // cardinality - 1 (535)
        1, 0,       // number of runs
        0xE8, 0xFD, // run start (65000)
        0x17, 0x02, // run length - 1 (535): ends at 65535
    } ) };
    auto const bitmap{ expect_valid( data ) };
    EXPECT_EQ( bitmap.size(), 536U );
    EXPECT_TRUE( bitmap.contains( 65535U ) );
    EXPECT_FALSE( bitmap.contains( 65536U ) );
    EXPECT_TRUE( serialized( bitmap ) == data );

    poke<std::uint16_t>( data, 13, 536 );   // would end at 65536
    EXPECT_TRUE( rejected( data ) );
    poke<std::uint16_t>( data, 7, 536 );    // the cardinality agrees with the run: only the end is wrong
    EXPECT_TRUE( rejected( data ) );
}

// deserialize_offsets_are_ignored: the containers are read back to back; the offset header is not consulted.
TEST( UpstreamSerialization, deserialize_offsets_are_ignored ) {
    auto const aliasing{ bytes_of( {
        0x3A, 0x30, 0, 0,   // cookie without runs
        2, 0, 0, 0,         // container count
        0, 0, 0, 0,         // key 0, cardinality - 1 = 0
        1, 0, 0, 0,         // key 1, cardinality - 1 = 0
        24, 0, 0, 0,        // offset of container 0 (correct)
        24, 0, 0, 0,        // offset of container 1 (wrong: 26)
        7, 0,               // value of container 0
        9, 0,               // value of container 1
    } ) };
    auto const both{ expect_valid( aliasing ) };
    EXPECT_EQ( both.to_vector(), ( std::vector<std::uint32_t>{ 7U, chunk + 9U } ) );

    auto const out_of_range{ bytes_of( {
        0x3A, 0x30, 0, 0,   // cookie without runs
        1, 0, 0, 0,         // container count
        5, 0, 0, 0,         // key 5, cardinality - 1 = 0
        0xFF, 0, 0, 0,      // offset (past the end)
        7, 0,               // value
    } ) };
    auto const one{ expect_valid( out_of_range ) };
    EXPECT_EQ( one.to_vector(), ( std::vector<std::uint32_t>{ 5U * chunk + 7U } ) );
}

// deserialize_trailing_bytes: bytes after a bitmap are not part of it.
TEST( UpstreamSerialization, deserialize_trailing_bytes ) {
    auto const data{ bytes_of( {
        0x3A, 0x30, 0, 0,   // cookie without runs
        1, 0, 0, 0,         // container count
        5, 0, 0, 0,         // key 5, cardinality - 1 = 0
        16, 0, 0, 0,        // offset
        7, 0,               // value
        0xAA, 0xBB,         // not part of the bitmap
    } ) };
    auto const loaded{ Bitmap::portable_deserialize( data ) };
    ASSERT_TRUE( loaded.has_value() );
    EXPECT_EQ( loaded->validate(), nullptr );
    EXPECT_EQ( loaded->to_vector(), ( std::vector<std::uint32_t>{ 5U * chunk + 7U } ) );
    EXPECT_EQ( loaded->portable_size_in_bytes(), 18U );
}

// deserialize_non_minimal_run_containers: a run container is valid whether or not another kind would be smaller.
TEST( UpstreamSerialization, deserialize_non_minimal_run_containers ) {
    auto const small{ bytes_of( {
        0x3B, 0x30, // cookie with runs
        0, 0,       // container count - 1
        0x01,       // run flags
        0, 0,       // key
        1, 0,       // cardinality - 1 (2 values)
        1, 0,       // number of runs
        0, 0,       // run start
        1, 0,       // run length - 1 (an array would be smaller)
    } ) };
    auto const pair{ expect_valid( small ) };
    EXPECT_EQ( pair.to_vector(), ( std::vector<std::uint32_t>{ 0U, 1U } ) );
    EXPECT_EQ( pair.statistics().run_containers, 1U );

    // 3000 runs of one value at 4 i: larger than a bitset
    constexpr std::size_t runs{ 3000 };
    Bytes large( 11 + 4 * runs );
    poke<std::uint32_t>( large, 0, 0x303BU );
    poke<std::uint8_t >( large, 4, 0x01 );
    poke<std::uint16_t>( large, 5, 0 );
    poke<std::uint16_t>( large, 7, static_cast<std::uint16_t>( runs - 1 ) );
    poke<std::uint16_t>( large, 9, static_cast<std::uint16_t>( runs ) );
    for ( std::size_t i{ 0 }; i < runs; ++i ) {
        poke<std::uint16_t>( large, 11 + 4 * i, static_cast<std::uint16_t>( 4 * i ) );
        poke<std::uint16_t>( large, 13 + 4 * i, 0 );
    }
    ASSERT_EQ( large.size(), 12011U );
    auto const spread{ expect_valid( large ) };
    EXPECT_EQ( spread.size(), runs );
    EXPECT_TRUE( spread.contains( 4U * 2999U ) );
    EXPECT_FALSE( spread.contains( 1U ) );
    EXPECT_EQ( spread.statistics().run_containers, 1U );
}

// serialize_empty_golden_bytes: the empty bitmap is eight bytes, and anything shorter is refused. (The empty frozen
// image, the header alone, is pinned by FrsrRoaringRanked.ByteSizeIsTheImageForBothVersions.)
TEST( UpstreamSerialization, serialize_empty_golden_bytes ) {
    Bitmap const empty;
    auto const bytes{ serialized( empty ) };
    EXPECT_TRUE( bytes == bytes_of( { 0x3A, 0x30, 0, 0, 0, 0, 0, 0 } ) );
    expect_round_trip( empty );
    for ( std::size_t const length : { 0U, 4U, 7U } ) {
        EXPECT_TRUE( rejected( std::span{ bytes }.first( length ) ) ) << length;
    }
}

// serialize_no_offset_threshold_golden_bytes: k containers each holding [j 2^16, j 2^16 + 10); with runs and fewer than
// four containers there is no offset header.
TEST( UpstreamSerialization, serialize_no_offset_threshold_golden_bytes ) {
    std::vector<Bytes> const golden{
        bytes_of( {
            0x3b,0x30,0x00,0x00,0x01,0x00,0x00,0x09,0x00,0x01,0x00,0x00,0x00,
            0x09,0x00 } ),
        bytes_of( {
            0x3b,0x30,0x01,0x00,0x03,0x00,0x00,0x09,0x00,0x01,0x00,0x09,0x00,
            0x01,0x00,0x00,0x00,0x09,0x00,0x01,0x00,0x00,0x00,0x09,0x00 } ),
        bytes_of( {
            0x3b,0x30,0x02,0x00,0x07,0x00,0x00,0x09,0x00,0x01,0x00,0x09,0x00,
            0x02,0x00,0x09,0x00,0x01,0x00,0x00,0x00,0x09,0x00,0x01,0x00,0x00,
            0x00,0x09,0x00,0x01,0x00,0x00,0x00,0x09,0x00 } ),
        bytes_of( {
            0x3b,0x30,0x03,0x00,0x0f,0x00,0x00,0x09,0x00,0x01,0x00,0x09,0x00,
            0x02,0x00,0x09,0x00,0x03,0x00,0x09,0x00,0x25,0x00,0x00,0x00,0x2b,
            0x00,0x00,0x00,0x31,0x00,0x00,0x00,0x37,0x00,0x00,0x00,0x01,0x00,
            0x00,0x00,0x09,0x00,0x01,0x00,0x00,0x00,0x09,0x00,0x01,0x00,0x00,
            0x00,0x09,0x00,0x01,0x00,0x00,0x00,0x09,0x00 } ),
    };
    constexpr std::size_t run_sizes[]{ 15, 25, 35, 61, 75 };
    for ( std::uint32_t k{ 1 }; k <= 5; ++k ) {
        SCOPED_TRACE( k );
        Bitmap runs;
        Bitmap plain;
        for ( std::uint32_t j{ 0 }; j < k; ++j ) {
            runs.add_closed_range( j * chunk, j * chunk + 9U );
            std::ignore = plain.add( j * chunk + 7U );
        }
        runs.optimize_keep_bitsets();
        auto const bytes{ serialized( runs ) };
        EXPECT_EQ( bytes.size(), run_sizes[ k - 1 ] );
        if ( k <= 4 ) {
            EXPECT_TRUE( bytes == golden[ k - 1 ] );
        }
        expect_round_trip( runs );
        EXPECT_EQ( plain.portable_size_in_bytes(), 8U + 10U * k );
        expect_round_trip( plain );
    }
}

// serialize_array_bitset_boundary_golden_bytes: 4096 values are an array, 4097 a bitset, and both take 8208 bytes.
TEST( UpstreamSerialization, serialize_array_bitset_boundary_golden_bytes ) {
    for ( std::uint32_t const cardinality : { 4096U, 4097U } ) {
        SCOPED_TRACE( cardinality );
        Bitmap bitmap;
        for ( std::uint32_t i{ 0 }; i < cardinality; ++i ) {
            std::ignore = bitmap.add( 2U * i );
        }
        auto const bytes{ serialized( bitmap ) };
        ASSERT_EQ( bytes.size(), 8208U );
        auto const header{ bytes_of( {
            0x3A, 0x30, 0, 0,
            1, 0, 0, 0,
            0, 0, static_cast<std::uint8_t>( ( cardinality - 1 ) & 0xFF ), static_cast<std::uint8_t>( ( cardinality - 1 ) >> 8 ),
            16, 0, 0, 0,
        } ) };
        EXPECT_TRUE( std::ranges::equal( std::span{ bytes }.first( header.size() ), header ) );
        auto const loaded{ Bitmap::portable_deserialize( bytes ) };
        ASSERT_TRUE( loaded.has_value() );
        EXPECT_EQ( loaded->validate(), nullptr );
        EXPECT_EQ( loaded->size(), cardinality );
        EXPECT_EQ( loaded->statistics().array_containers, cardinality == 4096U ? 1U : 0U );
        EXPECT_EQ( loaded->statistics().bitset_containers, cardinality == 4097U ? 1U : 0U );
        EXPECT_TRUE( serialized( *loaded ) == bytes );
    }
}

// serialize_small_container_counts (Java TestSerializationViaByteBuffer): every container count from 1 to 15, with
// and without runs.
TEST( UpstreamSerialization, serialize_small_container_counts ) {
    for ( std::uint32_t keys{ 1 }; keys <= 15; ++keys ) {
        for ( bool const optimize : { false, true } ) {
            SCOPED_TRACE( ::testing::Message() << keys << " containers, optimized " << optimize );
            Bitmap bitmap;
            for ( std::uint32_t j{ 0 }; j < keys; ++j ) {
                auto const base{ ( 3U * j + 1U ) * chunk };
                switch ( j % 3 ) {
                    case 0:
                        for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
                            std::ignore = bitmap.add( base + 7U * i );
                        }
                        break;
                    case 1:
                        for ( std::uint32_t i{ 0 }; i < 10000; ++i ) {
                            std::ignore = bitmap.add( base + 2U * i );
                        }
                        break;
                    default:
                        bitmap.add_closed_range( base + 100U, base + 1099U );
                        break;
                }
            }
            if ( optimize ) {
                bitmap.optimize_keep_bitsets();
            }
            ASSERT_EQ( bitmap.validate(), nullptr );
            expect_round_trip( bitmap );
        }
    }
}

// deserialize_back_to_back: two bitmaps written one after the other into one buffer.
TEST( UpstreamSerialization, deserialize_back_to_back ) {
    Bitmap const empty;
    Bitmap second;
    for ( std::uint32_t value{ 0 }; value < 200000U; value += 3 ) {
        std::ignore = second.add( value );
    }
    second.add_closed_range( 500000U, 599999U );
    second.optimize_keep_bitsets();

    auto buffer{ serialized( empty ) };
    auto const first_size{ buffer.size() };
    auto const tail{ serialized( second ) };
    buffer.insert( buffer.end(), tail.begin(), tail.end() );

    auto const first{ Bitmap::portable_deserialize( buffer ) };
    ASSERT_TRUE( first.has_value() );
    EXPECT_TRUE( first->empty() );
    ASSERT_EQ( first->portable_size_in_bytes(), first_size );
    auto const rest{ Bitmap::portable_deserialize( std::span{ buffer }.subspan( first_size ) ) };
    ASSERT_TRUE( rest.has_value() );
    EXPECT_TRUE( *rest == second );
    EXPECT_EQ( rest->portable_size_in_bytes(), tail.size() );
}

// deserialize_unaligned_source: the portable reader and writer work at any byte offset.
TEST( UpstreamSerialization, deserialize_unaligned_source ) {
    Bitmap bitmap;
    for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
        std::ignore = bitmap.add( 3U * i );
    }
    for ( std::uint32_t i{ 0 }; i < 10000; ++i ) {
        std::ignore = bitmap.add( chunk + 2U * i );
    }
    bitmap.add_closed_range( 3U * chunk, 3U * chunk + 4999U );
    for ( std::uint32_t i{ 0 }; i < 5; ++i ) {
        std::ignore = bitmap.add( ( 10U + i ) * chunk );
    }
    bitmap.optimize_keep_bitsets();
    auto const size{ bitmap.portable_size_in_bytes() };
    std::vector<std::uint64_t> storage( ( size + 8 + 7 ) / 8 );
    std::span<std::byte> const raw{ reinterpret_cast<std::byte *>( storage.data() ), size + 8 };
    for ( std::size_t offset{ 0 }; offset < 8; ++offset ) {
        SCOPED_TRACE( offset );
        auto const at{ raw.subspan( offset, size ) };
        ASSERT_EQ( bitmap.portable_serialize( at ), size );
        auto const loaded{ Bitmap::portable_deserialize( at ) };
        ASSERT_TRUE( loaded.has_value() );
        EXPECT_EQ( loaded->validate(), nullptr );
        EXPECT_TRUE( *loaded == bitmap );
    }
}

// deserialize_truncation_sweep: every strict prefix of a valid file is refused; on the larger files a sample of the
// prefixes and the last 64.
TEST( UpstreamSerialization, deserialize_truncation_sweep ) {
    for ( auto const * const name : {
        "bitmapwithruns.bin",
        "bitmapwithoutruns.bin",
        "testfrozendata/mixed.portable",
        "testfrozendata/runs_only.portable",
        "testfrozendata/arrays_only.portable",
        "testfrozendata/bitmaps_only.portable",
    } ) {
        SCOPED_TRACE( name );
        auto const bytes{ read_fixture( name ) };
        ASSERT_FALSE( rejected( bytes ) );
        auto const stride{ bytes.size() > 20000 ? std::size_t{ 97 } : std::size_t{ 1 } };
        for ( std::size_t length{ 0 }; length < bytes.size(); ++length ) {
            if ( length % stride == 0 || length + 64 >= bytes.size() ) {
                EXPECT_TRUE( rejected( std::span{ bytes }.first( length ) ) ) << length;
            }
        }
    }
}

// ---- the frozen view (this library's own image) ----

constexpr std::size_t frozen_header_bytes{ 24 };
constexpr std::size_t frozen_entry_bytes { 32 };
constexpr std::size_t frozen_entry_chunk { 0 };
constexpr std::size_t frozen_entry_offset{ 8 };
constexpr std::size_t frozen_entry_kind  { 24 };

[[nodiscard]] std::size_t entry_at( std::size_t const index ) { return frozen_header_bytes + index * frozen_entry_bytes; }

[[nodiscard]] std::size_t payload_of( image const & frozen, std::size_t const index ) {
    return peek<std::uint32_t>( frozen.bytes(), entry_at( index ) + frozen_entry_offset );
}

// `frozen` with one more index entry, a copy of the last one (sharing its payload) for chunk `key`.
[[nodiscard]] image with_extra_entry( image const & frozen, std::uint64_t const key ) {
    auto const source{ frozen.bytes() };
    auto const count{ peek<std::uint32_t>( source, 8 ) };
    auto const payloads{ entry_at( count ) };
    image result;
    result.size = frozen.size + frozen_entry_bytes;
    result.words.resize( ( result.size + 7 ) / 8 );
    auto const target{ result.writable() };
    std::memcpy( target.data(), source.data(), payloads );
    std::memcpy( target.data() + payloads + frozen_entry_bytes, source.data() + payloads, frozen.size - payloads );
    std::memcpy( target.data() + payloads, source.data() + entry_at( count - 1 ), frozen_entry_bytes );
    for ( std::uint32_t i{ 0 }; i <= count; ++i ) {
        auto const offset{ peek<std::uint32_t>( target, entry_at( i ) + frozen_entry_offset ) };
        poke<std::uint32_t>( target, entry_at( i ) + frozen_entry_offset, offset + static_cast<std::uint32_t>( frozen_entry_bytes ) );
    }
    poke<std::uint64_t>( target, payloads + frozen_entry_chunk, key );
    poke<std::uint32_t>( target, 8, count + 1 );
    poke<std::uint64_t>( target, 16, peek<std::uint64_t>( source, 16 ) + 1 );   // the extra entry holds one value
    return result;
}

// frozen_view_too_many_containers (Go ErrFrozenBitmapOverpopulated): an image of a 32-bit bitmap holds at most
// 65536 containers, one per chunk key; one more is refused.
TEST( UpstreamSerialization, frozen_view_too_many_containers ) {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t key{ 0 }; key < chunk; ++key ) {
        values.emplace_back( key * chunk + 7U );
    }
    Bitmap all_but_last;
    all_but_last.add_sorted_many( std::span{ values }.first( chunk - 1 ) );
    Bitmap const full{ values };

    auto const full_image{ freeze( full, {} ) };
    auto const view{ Bitmap::frozen_view_from( full_image.bytes() ) };
    ASSERT_TRUE( view );
    EXPECT_EQ( view.size(), std::size_t{ chunk } );
    EXPECT_TRUE( view.contains( 0xFFFF0007U ) );
    EXPECT_EQ( view.materialize().validate(), nullptr );

    // The extra entry is built the same way in both: it completes the 65535 chunks to the last key, and it is
    // refused past it.
    auto const completed{ with_extra_entry( freeze( all_but_last, {} ), chunk - 1 ) };
    auto const completed_view{ Bitmap::frozen_view_from( completed.bytes() ) };
    ASSERT_TRUE( completed_view );
    EXPECT_TRUE( completed_view.materialize() == full );
    EXPECT_FALSE( Bitmap::frozen_view_from( with_extra_entry( full_image, chunk ).bytes() ) );
}

// frozen_view_negative_cases. A frozen view opened on bytes beyond the image is not one of them here: the span may be
// longer than the image (FrsrRoaringRanked.ByteSizeIsTheImageForBothVersions). Unknown container kinds on the last
// entry reach the refusal after the earlier entries were taken, the path that freed the wrong pointer in the reference
// (#926); a view owns nothing, so there is nothing to release on any path.
TEST( UpstreamSerialization, frozen_view_negative_cases ) {
    Bitmap bitmap;
    for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
        std::ignore = bitmap.add( 5U * i );
    }
    for ( std::uint32_t i{ 0 }; i < 10000; ++i ) {
        std::ignore = bitmap.add( chunk + 2U * i );
    }
    bitmap.add_closed_range( 2U * chunk, 2U * chunk + 999U );
    bitmap.optimize_keep_bitsets();
    ASSERT_EQ( bitmap.statistics().array_containers, 1U );
    ASSERT_EQ( bitmap.statistics().bitset_containers, 1U );
    ASSERT_EQ( bitmap.statistics().run_containers, 1U );

    auto const good{ freeze( bitmap, {} ) };
    ASSERT_TRUE( Bitmap::frozen_view_from( good.bytes() ).materialize() == bitmap );

    EXPECT_FALSE( Bitmap::frozen_view_from( good.bytes().first( good.size - 1 ) ) );   // the last payload is cut
    EXPECT_FALSE( Bitmap::frozen_view_from( good.bytes().first( 3 ) ) );
    EXPECT_FALSE( Bitmap::frozen_view_from( good.bytes().first( frozen_header_bytes - 1 ) ) );
    {
        // aligned padding before a valid image
        image padded;
        padded.size = good.size + 32;
        padded.words.resize( ( padded.size + 7 ) / 8 );
        std::memcpy( padded.writable().data() + 32, good.bytes().data(), good.size );
        EXPECT_FALSE( Bitmap::frozen_view_from( padded.bytes() ) );
    }
    {
        // the magic in the other byte order
        auto swapped{ good };
        poke<std::uint32_t>( swapped.writable(), 0, std::byteswap( peek<std::uint32_t>( good.bytes(), 0 ) ) );
        EXPECT_FALSE( Bitmap::frozen_view_from( swapped.bytes() ) );
    }
    {
        // the portable bytes of the empty bitmap are not a frozen image
        image portable;
        portable.size = frozen_header_bytes;
        portable.words.resize( frozen_header_bytes / 8 );
        auto const empty{ bytes_of( { 0x3A, 0x30, 0, 0, 0, 0, 0, 0 } ) };
        std::ranges::copy( empty, portable.writable().begin() );
        EXPECT_FALSE( Bitmap::frozen_view_from( portable.bytes() ) );
    }
    for ( std::size_t const index : { 0U, 2U } ) {
        for ( std::uint8_t const kind : std::initializer_list<std::uint8_t>{ 0, 4, 255 } ) {
            auto bad{ good };
            poke<std::uint8_t>( bad.writable(), entry_at( index ) + frozen_entry_kind, kind );
            EXPECT_FALSE( Bitmap::frozen_view_from( bad.bytes() ) ) << "kind " << int{ kind } << " at entry " << index;
        }
    }
}

// frozen_view_corrupted_payload (Go TestBitMapValidationFromFrozen): opening a view checks the image's structure, not
// the payloads, so a view over damaged payloads opens; the bitmap it holds then fails validation.
void expect_corruption_reported( Bitmap const & bitmap, std::size_t const offset, std::uint16_t const value ) {
    auto frozen{ freeze( bitmap, {} ) };
    ASSERT_EQ( Bitmap::frozen_view_from( frozen.bytes() ).materialize().validate(), nullptr );
    poke<std::uint16_t>( frozen.writable(), payload_of( frozen, 0 ) + offset, value );
    auto const view{ Bitmap::frozen_view_from( frozen.bytes() ) };
    ASSERT_TRUE( view );
    EXPECT_NE( view.materialize().validate(), nullptr );
}

TEST( UpstreamSerialization, frozen_view_corrupted_payload ) {
    // A run payload is (first, last) pairs of 16-bit values.
    {
        // runs [0,1], [4,5], [8,99]: the last one ends at 65535, more values than the chunk entry says
        Bitmap runs;
        runs.add_closed_range( 0U, 1U );
        runs.add_closed_range( 4U, 5U );
        runs.add_closed_range( 8U, 99U );
        runs.optimize_keep_bitsets();
        ASSERT_EQ( runs.statistics().run_containers, 1U );
        expect_corruption_reported( runs, 10, 0xFFFF );
    }
    {
        // the run [100,109] ends before it begins
        Bitmap run;
        run.add_closed_range( 100U, 109U );
        run.optimize_keep_bitsets();
        ASSERT_EQ( run.statistics().run_containers, 1U );
        expect_corruption_reported( run, 2, 99 );
    }
    {
        // runs [100,109] and [115,124]: the second one moved to start at 108, inside the first
        Bitmap runs;
        runs.add_closed_range( 100U, 109U );
        runs.add_closed_range( 115U, 124U );
        runs.optimize_keep_bitsets();
        ASSERT_EQ( runs.statistics().run_containers, 1U );
        expect_corruption_reported( runs, 4, 108 );
    }
    {
        // the array 0..9 with its third value set to 0
        Bitmap array;
        for ( std::uint32_t i{ 0 }; i < 10; ++i ) {
            std::ignore = array.add( i );
        }
        ASSERT_EQ( array.statistics().array_containers, 1U );
        expect_corruption_reported( array, 4, 0 );
    }
}

// go_fuzzer_reproducers (Go roaring_test.go TestFuzzerRepro_*): decode, operate, and the result is still valid.
TEST( UpstreamSerialization, go_fuzzer_reproducers ) {
    auto const decode{ []( Bytes const & bytes ) { return expect_valid( bytes ); } };
    // {9 2^16 | 0xE055, 15 2^16 | 0xFFFC}
    auto const two_keys{ bytes_of( { 0x3a, 0x30, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x09, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
        0x1a, 0x00, 0x00, 0x00, 0x55, 0xe0, 0xfc, 0xff } ) };
    // {9 2^16 | 0xE055}
    auto const first_key{ bytes_of( { 0x3a, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x09, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x55, 0xe0 } ) };
    // the run [3,6]
    auto const run3to6{ bytes_of( { 0x3b, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03,
        0x00, 0x01, 0x00, 0x03, 0x00, 0x03, 0x00 } ) };
    // {1}
    auto const one{ bytes_of( { 0x3a, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00 } ) };
    // {0, 15 2^16 | 0xFFFF}
    auto const zero_and_top{ bytes_of( { 0x3a, 0x30, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00,
        0x00, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff } ) };
    // the run [15 2^16 | 0xFFF5, +3]
    auto const run_fff5{ bytes_of( { 0x3b, 0x30, 0x00, 0x00, 0x01, 0x0f, 0x00, 0x03,
        0x00, 0x01, 0x00, 0xf5, 0xff, 0x03, 0x00 } ) };
    // {15 2^16 | 0xFFF6, 15 2^16 | 0xFFF7}
    auto const two_inside{ bytes_of( { 0x3a, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x0f, 0x00, 0x01, 0x00, 0x10, 0x00, 0x00, 0x00, 0xf6, 0xff, 0xf7,
        0xff } ) };
    // the run [15 2^16 | 0xFFE8, +3]
    auto const run_ffe8{ bytes_of( { 0x3b, 0x30, 0x00, 0x00, 0x01, 0x0f, 0x00, 0x03,
        0x00, 0x01, 0x00, 0xe8, 0xff, 0x03, 0x00 } ) };
    // the run [2,5]
    auto const run2to5{ bytes_of( { 0x3b, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03,
        0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00 } ) };
    // the run [54,57]
    auto const run54to57{ bytes_of( { 0x3b, 0x30, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03,
        0x00, 0x01, 0x00, 0x36, 0x00, 0x03, 0x00 } ) };
    constexpr std::uint32_t top{ 15U * chunk };
    {
        // andnot empties the first container
        auto a{ decode( two_keys ) };
        a -= decode( first_key );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ top | 0xFFFCU } ) );
    }
    {
        // or of a run with a value before it
        auto a{ decode( run3to6 ) };
        a |= decode( one );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ 1U, 3U, 4U, 5U, 6U } ) );
    }
    {
        // or with an equal copy leaves the bitmap as it was, xor empties it
        auto a{ decode( zero_and_top ) };
        auto const b{ decode( zero_and_top ) };
        a |= b;
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_TRUE( a == b );
        a ^= b;
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_TRUE( a.empty() );
    }
    {
        // andnot splits a run into two single values
        auto a{ decode( run_fff5 ) };
        a -= decode( two_inside );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ top | 0xFFF5U, top | 0xFFF8U } ) );
    }
    {
        // flip the last value of a run
        auto a{ decode( run_ffe8 ) };
        a.flip_closed_range( 1048555U, 1048555U );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ 1048552U, 1048553U, 1048554U } ) );
    }
    {
        // remove inside a short run
        auto a{ decode( run2to5 ) };
        EXPECT_TRUE( a.remove( 3U ) );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ 2U, 4U, 5U } ) );
    }
    {
        // add after a short run
        auto a{ decode( run54to57 ) };
        EXPECT_TRUE( a.add( 112U ) );
        EXPECT_EQ( a.validate(), nullptr );
        EXPECT_EQ( a.to_vector(), ( std::vector<std::uint32_t>{ 54U, 55U, 56U, 57U, 112U } ) );
    }
}

// ---- tests/format_portability_unit.c ----

// test_shared_files_content: bitmapwithruns.bin and bitmapwithoutruns.bin (Java SerializeToDiskExample) hold
// {k : k % 1000 == 0, k < 100000} U {3 k : 100000 <= k < 200000} U [700000, 800000); run-optimizing that set gives
// the file with runs byte for byte.
TEST( UpstreamSerialization, test_shared_files_content ) {
    Bitmap expected;
    for ( std::uint32_t k{ 0 }; k < 100000; k += 1000 ) {
        std::ignore = expected.add( k );
    }
    for ( std::uint32_t k{ 100000 }; k < 200000; ++k ) {
        std::ignore = expected.add( 3U * k );
    }
    expected.add_closed_range( 700000U, 799999U );
    for ( auto const & [ name, runs ] : { std::pair{ "bitmapwithoutruns.bin", 0U }, std::pair{ "bitmapwithruns.bin", 3U } } ) {
        SCOPED_TRACE( name );
        auto const bytes{ read_fixture( name ) };
        auto const loaded{ Bitmap::portable_deserialize( bytes ) };
        ASSERT_TRUE( loaded.has_value() );
        EXPECT_EQ( loaded->validate(), nullptr );
        EXPECT_EQ( loaded->size(), 200100U );
        EXPECT_TRUE( *loaded == expected );
        EXPECT_EQ( loaded->statistics().container_count(), 11U );
        EXPECT_EQ( loaded->statistics().run_containers, runs );
        if ( runs > 0 ) {
            auto optimized{ expected };
            optimized.optimize_keep_bitsets();
            EXPECT_TRUE( serialized( optimized ) == bytes );
        }
    }
}

// test_frozen_golden_files: the portable files made with gocroaring hold the bitmaps their .stats files describe and
// are rewritten byte for byte. The reference's frozen files are in its own frozen format; this library's frozen image
// of the same bitmap stands in for them.
TEST( UpstreamSerialization, test_frozen_golden_files ) {
    struct golden {
        char const * name;
        std::size_t  cardinality;
        std::size_t  bitsets;
        std::size_t  arrays;
        std::size_t  runs;
    };
    for ( auto const & file : {
        golden{ "bitmaps_only", 65536, 2, 0, 0 },
        golden{ "arrays_only", 132, 0, 2, 0 },
        golden{ "runs_only", 122878, 0, 0, 2 },
        golden{ "mixed", 188546, 2, 2, 2 },
    } ) {
        SCOPED_TRACE( file.name );
        auto const bytes{ read_fixture( std::string{ "testfrozendata/" } + file.name + ".portable" ) };
        auto const loaded{ Bitmap::portable_deserialize( bytes ) };
        ASSERT_TRUE( loaded.has_value() );
        EXPECT_EQ( loaded->validate(), nullptr );
        EXPECT_EQ( loaded->size(), file.cardinality );
        EXPECT_EQ( loaded->statistics().bitset_containers, file.bitsets );
        EXPECT_EQ( loaded->statistics().array_containers, file.arrays );
        EXPECT_EQ( loaded->statistics().run_containers, file.runs );
        EXPECT_TRUE( serialized( *loaded ) == bytes );

        auto const frozen{ freeze( *loaded, {} ) };
        auto const view{ Bitmap::frozen_view_from( frozen.bytes() ) };
        ASSERT_TRUE( view );
        EXPECT_EQ( view.size(), file.cardinality );
        EXPECT_TRUE( view.materialize() == *loaded );
    }
}

} // namespace
