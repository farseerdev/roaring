// The 64-bit cases of the reference's suite (tests/roaring64_unit.cpp, tests/cpp_roaring64_unit.cpp,
// tests/roaring64_serialization.cpp) run on the reference itself, CRoaring's roaring64 API, next to bitmap<std::uint64_t>:
// the bytes both write, the input both accept or reject, and their answers on the same data, with std::set as the
// judge where they could differ. run_optimize() is optimize_keep_bitsets() here, rank( x ) the cardinality of [0, x].

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>
#include <roaring/roaring64.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace frsr::roaring;

using Bitmap = bitmap<std::uint64_t>;
using Bytes  = std::vector<std::byte>;

constexpr std::uint64_t K32{ std::uint64_t{ 1 } << 32 };
constexpr std::uint64_t H63{ std::uint64_t{ 1 } << 63 };
constexpr std::uint64_t M64{ std::numeric_limits<std::uint64_t>::max() };

struct reference_free {
    void operator()( roaring64_bitmap_t * const r ) const noexcept { roaring64_bitmap_free( r ); }
};
using Reference = std::unique_ptr<roaring64_bitmap_t, reference_free>;

Reference reference() {
    return Reference{ roaring64_bitmap_create() };
}

std::vector<std::uint64_t> values_of( roaring64_bitmap_t const * const r ) {
    std::vector<std::uint64_t> out( roaring64_bitmap_get_cardinality( r ) );
    roaring64_bitmap_to_uint64_array( r, out.data() );
    return out;
}

std::vector<std::uint64_t> values_of( Bitmap const & bitmap ) {
    std::vector<std::uint64_t> out;
    for ( auto const value : bitmap ) {
        out.emplace_back( value );
    }
    return out;
}

Bytes portable_bytes( Bitmap const & bitmap ) {
    Bytes out( bitmap.portable_size_in_bytes() );
    EXPECT_EQ( bitmap.portable_serialize( out ), out.size() );
    return out;
}

Bytes portable_bytes( roaring64_bitmap_t const * const r ) {
    Bytes out( roaring64_bitmap_portable_size_in_bytes( r ) );
    EXPECT_EQ( roaring64_bitmap_portable_serialize( r, reinterpret_cast<char *>( out.data() ) ), out.size() );
    return out;
}

Reference reference_from( std::span<std::byte const> const bytes ) {
    return Reference{ roaring64_bitmap_portable_deserialize_safe( reinterpret_cast<char const *>( bytes.data() ), bytes.size() ) };
}

Bytes read_data_file( char const * const name ) {
    std::ifstream in( std::string{ FRSR_ROARING_TEST_DATA_DIR } + "/" + name, std::ios::binary );
    std::vector<char> const raw( std::istreambuf_iterator<char>( in ), {} );
    auto const bytes{ std::as_bytes( std::span{ raw } ) };
    return { bytes.begin(), bytes.end() };
}

// tests/roaring64_unit.cpp: test_portable_serialize, a bucket with a run container between buckets without one: both
// write the same bytes, and each reads the other's
TEST( Upstream64Reference, test_portable_serialize_run_bucket_between_plain_buckets ) {
    Bitmap mine;
    std::ignore = mine.add( 5 );
    mine.or_with( key_range::half_open( K32, K32 + 1000 ) );
    std::ignore = mine.add( K32 + ( 1 << 16 ) );
    std::ignore = mine.add( 2 * K32 );
    mine.optimize_keep_bitsets();

    auto const theirs{ reference() };
    roaring64_bitmap_add( theirs.get(), 5 );
    roaring64_bitmap_add_range( theirs.get(), K32, K32 + 1000 );
    roaring64_bitmap_add( theirs.get(), K32 + ( 1 << 16 ) );
    roaring64_bitmap_add( theirs.get(), 2 * K32 );
    roaring64_bitmap_run_optimize( theirs.get() );

    auto const bytes{ portable_bytes( theirs.get() ) };
    EXPECT_TRUE( portable_bytes( mine ) == bytes );
    auto const loaded{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( loaded.has_value() );
    EXPECT_TRUE( *loaded == mine );
    auto const back{ reference_from( portable_bytes( mine ) ) };
    ASSERT_NE( back, nullptr );
    EXPECT_TRUE( roaring64_bitmap_equals( back.get(), theirs.get() ) );
}

// tests/roaring64_serialization.cpp: test_64maprunbitsetvals: the file of the Java library reads the same in both,
// and both write it again byte for byte from the run-optimized values
TEST( Upstream64Reference, test_64maprunbitsetvals ) {
    auto const bytes{ read_data_file( "64maprunbitsetvals.bin" ) };
    ASSERT_EQ( bytes.size(), 8370U );
    auto const theirs{ reference_from( bytes ) };
    ASSERT_NE( theirs, nullptr );
    auto const mine{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( mine.has_value() );
    EXPECT_EQ( values_of( *mine ), values_of( theirs.get() ) );
    EXPECT_EQ( roaring64_bitmap_get_cardinality( theirs.get() ), 6108U );

    Bitmap rebuilt{ values_of( theirs.get() ) };
    rebuilt.optimize_keep_bitsets();
    auto const again{ reference() };
    for ( auto const value : values_of( theirs.get() ) ) {
        roaring64_bitmap_add( again.get(), value );
    }
    roaring64_bitmap_run_optimize( again.get() );
    EXPECT_TRUE( portable_bytes( again.get() ) == bytes );
    EXPECT_TRUE( portable_bytes( rebuilt ) == bytes );
}

// tests/roaring64_serialization.cpp: test_64truncated_every_prefix: both reject every strict prefix
TEST( Upstream64Reference, test_64truncated_every_prefix ) {
    auto const source{ reference() };
    for ( auto const value : { std::uint64_t{ 1 }, K32, M64 } ) {
        roaring64_bitmap_add( source.get(), value );
    }
    roaring64_bitmap_add_range( source.get(), std::uint64_t{ 1 } << 40, ( std::uint64_t{ 1 } << 40 ) + 5000 );
    roaring64_bitmap_run_optimize( source.get() );
    auto const bytes{ portable_bytes( source.get() ) };
    for ( std::size_t k{ 0 }; k < bytes.size(); k++ ) {
        Bytes const prefix( bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>( k ) );
        EXPECT_EQ( reference_from( prefix ), nullptr ) << k;
        EXPECT_FALSE( Bitmap::portable_deserialize( prefix ).has_value() ) << k;
    }
    auto const mine{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( mine.has_value() );
    EXPECT_EQ( values_of( *mine ), values_of( source.get() ) );
    EXPECT_TRUE( portable_bytes( *mine ) == bytes );
}

// tests/roaring64_serialization.cpp: test_64run_past_container_end: a run ending at 65536 is rejected by this library;
// the reference rejects it or returns a bitmap its own validation refuses
TEST( Upstream64Reference, test_64run_past_container_end ) {
    auto const source{ reference() };
    roaring64_bitmap_add_range( source.get(), 65000, 65536 );
    ASSERT_TRUE( roaring64_bitmap_run_optimize( source.get() ) );
    auto bytes{ portable_bytes( source.get() ) };
    auto const n{ bytes.size() };
    auto const mine{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( mine.has_value() );
    EXPECT_EQ( values_of( *mine ), values_of( source.get() ) );

    bytes[ n - 2 ] = std::byte{ 536 & 0xFF };
    bytes[ n - 1 ] = std::byte{ 536 >> 8 };
    EXPECT_FALSE( Bitmap::portable_deserialize( bytes ).has_value() );
    if ( auto const theirs{ reference_from( bytes ) } ) {
        char const * reason{ nullptr };
        EXPECT_FALSE( roaring64_bitmap_internal_validate( theirs.get(), &reason ) );
    }
}

// tests/roaring64_serialization.cpp: test_64deserdecreasingkeys and test_64deseremptybucket: the same buffers are
// accepted or rejected by both, and what is accepted reads and writes the same
TEST( Upstream64Reference, test_64deserdecreasingkeys_and_emptybucket ) {
    std::array<std::uint8_t, 52> decreasing{
        2, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0,
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 16, 0, 0, 0,
        3, 0,
        1, 0, 0, 0,
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 16, 0, 0, 0,
        4, 0,
    };
    EXPECT_EQ( reference_from( std::as_bytes( std::span{ decreasing } ) ), nullptr );
    EXPECT_FALSE( Bitmap::portable_deserialize( std::as_bytes( std::span{ decreasing } ) ).has_value() );
    decreasing[ 8 ] = 1;
    decreasing[ 8 + 4 + 18 ] = 2;
    std::array<std::uint8_t, 42> const empty_bucket{
        2, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
        0x3A, 0x30, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0,
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 16, 0, 0, 0,
        7, 0,
    };
    for ( auto const bytes : std::initializer_list<std::span<std::byte const>>{ std::as_bytes( std::span{ decreasing } ), std::as_bytes( std::span{ empty_bucket } ) } ) {
        auto const theirs{ reference_from( bytes ) };
        ASSERT_NE( theirs, nullptr );
        auto const mine{ Bitmap::portable_deserialize( bytes ) };
        ASSERT_TRUE( mine.has_value() );
        EXPECT_EQ( values_of( *mine ), values_of( theirs.get() ) );
        EXPECT_TRUE( portable_bytes( *mine ) == portable_bytes( theirs.get() ) );
    }
}

// tests/roaring64_unit.cpp: test_portable_deserialize_frozen_unaligned: the reference's portable frozen view over the
// bytes this library writes at every alignment, and this library reading the reference's bytes at every alignment
TEST( Upstream64Reference, test_portable_deserialize_frozen_unaligned ) {
    Bitmap mine;
    for ( std::uint64_t i{ 1 }; i < 3000; i += 3 ) {
        std::ignore = mine.add( i );
    }
    for ( std::uint64_t i{ 65536 }; i < 65536 + 20000; i += 2 ) {
        std::ignore = mine.add( i );
    }
    mine.or_with( key_range::half_open( 140000, 150000 ) );
    for ( std::uint64_t k{ 3 }; k < 12; k++ ) {
        std::ignore = mine.add( ( k << 16 ) + 7 * k );
        std::ignore = mine.add( ( k << 32 ) + 7 * k );
    }
    mine.optimize_keep_bitsets();
    auto const theirs{ reference_from( portable_bytes( mine ) ) };
    ASSERT_NE( theirs, nullptr );
    auto const size{ mine.portable_size_in_bytes() };
    for ( std::size_t offset{ 0 }; offset < 8; offset++ ) {
        Bytes storage( size + offset );
        std::span const at{ storage.data() + offset, size };
        EXPECT_EQ( mine.portable_serialize( at ), size );
        Reference const view{ roaring64_bitmap_portable_deserialize_frozen( reinterpret_cast<char const *>( at.data() ), size ) };
        ASSERT_NE( view, nullptr ) << offset;
        EXPECT_TRUE( roaring64_bitmap_equals( view.get(), theirs.get() ) ) << offset;
        EXPECT_EQ( roaring64_bitmap_get_cardinality( view.get() ), mine.size() ) << offset;

        Bytes their_storage( size + offset );
        std::span const their_at{ their_storage.data() + offset, size };
        EXPECT_EQ( roaring64_bitmap_portable_serialize( theirs.get(), reinterpret_cast<char *>( their_at.data() ) ), size );
        auto const loaded{ Bitmap::portable_deserialize( their_at ) };
        ASSERT_TRUE( loaded.has_value() ) << offset;
        EXPECT_TRUE( *loaded == mine ) << offset;
    }
}

// tests/roaring64_unit.cpp: test_to_uint64_array, the containers under several high keys up to the top of the 64-bit
// range: the same values in the same order
TEST( Upstream64Reference, test_to_uint64_array ) {
    Bitmap mine;
    auto const theirs{ reference() };
    for ( std::uint64_t const high : { std::uint64_t{ 0 }, std::uint64_t{ 1 } << 20, std::uint64_t{ 1 } << 40, M64 - 0xFFFF } ) {
        for ( std::uint64_t i{ 0 }; i < 65536; i += 3 ) {
            std::ignore = mine.add( high + i );
            roaring64_bitmap_add( theirs.get(), high + i );
        }
        auto const array_base{ high + ( std::uint64_t{ 1 } << 16 ) };
        for ( std::uint64_t i{ 0 }; i < 65536; i += 1000 ) {
            std::ignore = mine.add( array_base + i );
            roaring64_bitmap_add( theirs.get(), array_base + i );
        }
        auto const run_base{ high + ( std::uint64_t{ 2 } << 16 ) };
        mine.add_closed_range( run_base + 10, run_base + 500 );
        mine.add_closed_range( run_base + 60000, run_base + 65535 );
        roaring64_bitmap_add_range_closed( theirs.get(), run_base + 10, run_base + 500 );
        roaring64_bitmap_add_range_closed( theirs.get(), run_base + 60000, run_base + 65535 );
        auto const full_base{ high + ( std::uint64_t{ 3 } << 16 ) };
        mine.add_closed_range( full_base, full_base + 65535 );
        roaring64_bitmap_add_range_closed( theirs.get(), full_base, full_base + 65535 );
        if ( high == M64 - 0xFFFF ) {
            break;
        }
    }
    mine.optimize_keep_bitsets();
    roaring64_bitmap_run_optimize( theirs.get() );
    std::vector<std::uint64_t> array( mine.size() );
    EXPECT_EQ( mine.to_array_into( array ), array.size() );
    EXPECT_EQ( array, values_of( theirs.get() ) );
    EXPECT_TRUE( portable_bytes( mine ) == portable_bytes( theirs.get() ) );
}

// tests/roaring64_unit.cpp: test_rank_select_extremes, test_rank_in_missing_bucket: rank and select at the extremes and
// between buckets agree with the reference's
TEST( Upstream64Reference, test_rank_select_extremes ) {
    std::vector<std::vector<std::uint64_t>> const sets{
        { 0, 1, H63 - 1, H63 },
        { M64 },
        { 1, M64 },
        { 123, H63 - 1 },
        { 5, 3 * K32 + 7 },
    };
    std::uint64_t const probes[]{ 0, 1, 2, 5, 6, 122, 123, 124, 2 * K32, 3 * K32 + 6, 3 * K32 + 7, ( H63 - 1 ) / 2, H63 - 2, H63 - 1, H63, H63 + 1, M64 - 1, M64 };
    for ( auto const & values : sets ) {
        Bitmap const mine{ values };
        auto const theirs{ reference() };
        roaring64_bitmap_add_many( theirs.get(), values.size(), values.data() );
        for ( auto const probe : probes ) {
            EXPECT_EQ( mine.and_cardinality( key_range::closed( 0, probe ) ), roaring64_bitmap_rank( theirs.get(), probe ) ) << probe;
        }
        auto const ordered{ values_of( mine ) };
        for ( std::uint64_t i{ 0 }; i <= ordered.size(); ++i ) {
            std::uint64_t element{ 0 };
            EXPECT_EQ( roaring64_bitmap_select( theirs.get(), i, &element ), i < ordered.size() ) << i;
            if ( i < ordered.size() ) {
                EXPECT_EQ( element, ordered[ i ] ) << i;
            }
        }
    }
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_random_range_ops_vs_set: the stream of range operations on both, with
// std::set as the judge (the reference's doublechecked::Roaring64, tests/roaring64_checked.hh)
TEST( Upstream64Reference, test_cpp_r64_random_range_ops_vs_set ) {
    Bitmap mine;
    auto const theirs{ reference() };
    std::set<std::uint64_t> check;
    auto const for_each_in = []( std::uint64_t const min, std::uint64_t const max, auto && f ) {
        if ( min > max ) {
            return;
        }
        for ( auto v{ min };; ++v ) {
            f( v );
            if ( v == max ) {
                break;
            }
        }
    };
    auto const flip_in_check = [ & ]( std::uint64_t const v ) {
        if ( check.erase( v ) == 0 ) {
            check.insert( v );
        }
    };
    auto const check_contains_closed = [ & ]( std::uint64_t const min, std::uint64_t const max ) {
        auto const count{ static_cast<std::uint64_t>( std::distance( check.lower_bound( min ), check.upper_bound( max ) ) ) };
        return count > 0 && count - 1 == max - min;
    };

    std::uint64_t seed{ 123456789 };
    auto const next = [ &seed ] {
        seed = 6364136223846793005ULL * seed + 1442695040888963407ULL;
        return seed;
    };
    std::uint64_t const bases[]{ 0, ( std::uint64_t{ 1 } << 16 ) - 512, ( std::uint64_t{ 1 } << 32 ) - 512, ( std::uint64_t{ 1 } << 48 ) - 512, M64 - 1024 };

    for ( std::size_t i{ 0 }; i < 1000; ++i ) {
        auto const base{ bases[ next() % std::size( bases ) ] };
        auto const min{ base + next() % 1024 };
        auto const width{ next() % 8 == 0 ? next() % 10000 : next() % 300 };
        auto const max{ M64 - min < width ? M64 : min + width };

        switch ( next() % 8 ) {
            case 0:
                mine.or_with( key_range::half_open( min, max ) );
                roaring64_bitmap_add_range( theirs.get(), min, max );
                for_each_in( min, max - 1, [ & ]( std::uint64_t const v ) { check.insert( v ); } );
                break;
            case 1:
                mine.add_closed_range( min, max );
                roaring64_bitmap_add_range_closed( theirs.get(), min, max );
                for_each_in( min, max, [ & ]( std::uint64_t const v ) { check.insert( v ); } );
                break;
            case 2:
                mine.andnot_with( key_range::half_open( min, max ) );
                roaring64_bitmap_remove_range( theirs.get(), min, max );
                check.erase( check.lower_bound( min ), check.lower_bound( max ) );
                break;
            case 3:
                mine.remove_closed_range( min, max );
                roaring64_bitmap_remove_range_closed( theirs.get(), min, max );
                check.erase( check.lower_bound( min ), check.upper_bound( max ) );
                break;
            case 4:
                mine.xor_with( key_range::half_open( min, max ) );
                roaring64_bitmap_flip_inplace( theirs.get(), min, max );
                for_each_in( min, max - 1, flip_in_check );
                break;
            case 5:
                mine.flip_closed_range( min, max );
                roaring64_bitmap_flip_closed_inplace( theirs.get(), min, max );
                for_each_in( min, max, flip_in_check );
                break;
            case 6: {
                auto const added{ check.insert( min ).second };
                EXPECT_EQ( mine.add( min ), added ) << i;
                EXPECT_EQ( roaring64_bitmap_add_checked( theirs.get(), min ), added ) << i;
                break;
            }
            default: {
                auto const removed{ check.erase( min ) == 1 };
                EXPECT_EQ( mine.remove( min ), removed ) << i;
                EXPECT_EQ( roaring64_bitmap_remove_checked( theirs.get(), min ), removed ) << i;
                break;
            }
        }
        if ( next() % 10 == 0 ) {
            mine.optimize_keep_bitsets();
            roaring64_bitmap_run_optimize( theirs.get() );
        }

        auto const half_open_held{ min >= max || check_contains_closed( min, max - 1 ) };
        auto const closed_held{ check_contains_closed( min, max ) };
        EXPECT_EQ( mine.contains_range( min, max ), half_open_held ) << i;
        EXPECT_EQ( roaring64_bitmap_contains_range( theirs.get(), min, max ), half_open_held ) << i;
        EXPECT_EQ( mine.is_superset_of( key_range::closed( min, max ) ), closed_held ) << i;
        EXPECT_EQ( roaring64_bitmap_contains_range_closed( theirs.get(), min, max ), closed_held ) << i;
        if ( i % 16 == 0 ) {
            ASSERT_TRUE( std::ranges::equal( values_of( mine ), check ) ) << i;
            ASSERT_TRUE( std::ranges::equal( values_of( theirs.get() ), check ) ) << i;
        }
    }
    EXPECT_TRUE( std::ranges::equal( values_of( mine ), check ) );
    EXPECT_TRUE( std::ranges::equal( values_of( theirs.get() ), check ) );
}

} // namespace
