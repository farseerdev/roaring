// The standard Roaring portable format, in both directions, against CRoaring:
// frsr writes what CRoaring reads and the other way round, byte for byte where
// the container kinds agree, and malformed input is rejected rather than trusted.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>
#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using Bitmap32 = frsr::roaring::bitmap<std::uint32_t>;
using Bitmap64 = frsr::roaring::bitmap<std::uint64_t>;
using Bytes    = std::vector<std::byte>;

// The reference side, one trait per key width.
struct Reference32 {
    using bitmap = Bitmap32;
    using handle = std::unique_ptr<roaring_bitmap_t, decltype( &roaring_bitmap_free )>;

    static handle build( std::span<std::uint64_t const> values, bool const optimize ) {
        std::vector<std::uint32_t> narrow( values.begin(), values.end() );
        handle result{ roaring_bitmap_create(), &roaring_bitmap_free };
        roaring_bitmap_add_many( result.get(), narrow.size(), narrow.data() );
        if ( optimize ) {
            roaring_bitmap_run_optimize( result.get() );
        }
        return result;
    }
    static Bytes serialize( handle const & r ) {
        Bytes bytes( roaring_bitmap_portable_size_in_bytes( r.get() ) );
        EXPECT_EQ( roaring_bitmap_portable_serialize( r.get(), reinterpret_cast<char *>( bytes.data() ) ), bytes.size() );
        return bytes;
    }
    static handle deserialize( Bytes const & bytes ) {
        return handle{ roaring_bitmap_portable_deserialize_safe( reinterpret_cast<char const *>( bytes.data() ), bytes.size() ), &roaring_bitmap_free };
    }
    static std::vector<std::uint64_t> values( handle const & r ) {
        std::vector<std::uint32_t> raw( roaring_bitmap_get_cardinality( r.get() ) );
        roaring_bitmap_to_uint32_array( r.get(), raw.data() );
        return { raw.begin(), raw.end() };
    }
};

struct Reference64 {
    using bitmap = Bitmap64;
    using handle = std::unique_ptr<roaring64_bitmap_t, decltype( &roaring64_bitmap_free )>;

    static handle build( std::span<std::uint64_t const> values, bool const optimize ) {
        handle result{ roaring64_bitmap_create(), &roaring64_bitmap_free };
        roaring64_bitmap_add_many( result.get(), values.size(), values.data() );
        if ( optimize ) {
            roaring64_bitmap_run_optimize( result.get() );
        }
        return result;
    }
    static Bytes serialize( handle const & r ) {
        Bytes bytes( roaring64_bitmap_portable_size_in_bytes( r.get() ) );
        EXPECT_EQ( roaring64_bitmap_portable_serialize( r.get(), reinterpret_cast<char *>( bytes.data() ) ), bytes.size() );
        return bytes;
    }
    static handle deserialize( Bytes const & bytes ) {
        return handle{ roaring64_bitmap_portable_deserialize_safe( reinterpret_cast<char const *>( bytes.data() ), bytes.size() ), &roaring64_bitmap_free };
    }
    static std::vector<std::uint64_t> values( handle const & r ) {
        std::vector<std::uint64_t> raw( roaring64_bitmap_get_cardinality( r.get() ) );
        roaring64_bitmap_to_uint64_array( r.get(), raw.data() );
        return raw;
    }
};

struct Fixture {
    std::string                name;
    std::vector<std::uint64_t> values;   // sorted, unique
};

void append_range( std::vector<std::uint64_t> & out, std::uint64_t const first, std::uint64_t const last ) {
    for ( auto value{ first }; value <= last; ++value ) {
        out.push_back( value );
    }
}

void append_random( std::vector<std::uint64_t> & out, std::mt19937_64 & rng, std::uint64_t const base, std::size_t const count ) {
    for ( std::size_t i{ 0 }; i < count; ++i ) {
        out.push_back( base + ( rng() & 0xFFFFU ) );
    }
}

void finish( Fixture & fixture ) {
    std::ranges::sort( fixture.values );
    auto const duplicates{ std::ranges::unique( fixture.values ) };
    fixture.values.erase( duplicates.begin(), duplicates.end() );
}

// Container shapes of every kind, empty/single/full, the array/bitset boundary and
// bitmaps with fewer and more than four containers (with and without runs, which is
// where the offset header comes and goes). `high` shifts the whole set into another
// 32-bit bucket of a 64-bit key space.
std::vector<Fixture> make_fixtures( std::uint64_t const high ) {
    std::mt19937_64 rng{ 20261019U + high };
    constexpr std::uint64_t chunk{ 65536U };
    std::vector<Fixture> result;
    auto const add{ [&]( std::string name, auto && fill ) {
        Fixture fixture{ std::move( name ), {} };
        fill( fixture.values );
        for ( auto & value : fixture.values ) {
            value += high;
        }
        finish( fixture );
        result.push_back( std::move( fixture ) );
    } };

    add( "empty", []( auto & ) {} );
    add( "single", []( auto & v ) { v.push_back( 123456U ); } );
    add( "single_last", []( auto & v ) { v.push_back( 0xFFFFFFFFULL ); } );
    add( "array", [&]( auto & v ) { append_random( v, rng, 3 * chunk, 300 ); } );
    add( "array_4096", [&]( auto & v ) { for ( std::uint64_t i{ 0 }; i < 4096; ++i ) v.push_back( chunk + i * 16 ); } );
    add( "bitset_4097", [&]( auto & v ) { for ( std::uint64_t i{ 0 }; i < 4097; ++i ) v.push_back( chunk + i * 15 ); } );
    add( "bitset", [&]( auto & v ) { append_random( v, rng, chunk, 30000 ); } );
    add( "run", []( auto & v ) { append_range( v, 2 * chunk + 100, 2 * chunk + 30000 ); } );
    add( "run_many", [&]( auto & v ) { for ( std::uint64_t i{ 0 }; i < 200; ++i ) append_range( v, 5 * chunk + i * 300, 5 * chunk + i * 300 + 120 ); } );
    add( "full_chunk", []( auto & v ) { append_range( v, 7 * chunk, 8 * chunk - 1 ); } );
    add( "full_three", []( auto & v ) { append_range( v, 0, 3 * chunk - 1 ); } );
    add( "mixed", [&]( auto & v ) {
        append_random( v, rng, 0, 100 );
        append_random( v, rng, chunk, 20000 );
        append_range( v, 2 * chunk + 10, 2 * chunk + 40000 );
        append_random( v, rng, 40000 * chunk, 50 );
        append_range( v, 65535 * chunk, 65535 * chunk + 65535 );
    } );
    add( "three_containers_with_run", [&]( auto & v ) {
        append_random( v, rng, 0, 50 );
        append_range( v, chunk, chunk + 9000 );
        append_random( v, rng, 2 * chunk, 5000 );
    } );
    add( "three_containers_no_run", [&]( auto & v ) {
        for ( std::uint64_t c{ 0 }; c < 3; ++c ) append_random( v, rng, c * chunk, 40 );
    } );
    add( "many_containers", [&]( auto & v ) {
        for ( std::uint64_t c{ 0 }; c < 400; ++c ) {
            if ( c % 7 == 0 ) append_range( v, c * chunk + 5, c * chunk + 5 + c * 20 );
            else if ( c % 5 == 0 ) append_random( v, rng, c * chunk, 6000 );
            else append_random( v, rng, c * chunk, 1 + c % 90 );
        }
    } );
    return result;
}

std::vector<Fixture> make_all_fixtures_64() {
    std::vector<Fixture> all;
    for ( std::uint64_t const high : { 0ULL, 1ULL << 32, 7ULL << 32, 0xFFFFFFFEULL << 32 } ) {
        for ( auto & fixture : make_fixtures( high ) ) {
            if ( high != 0 && fixture.name == "single_last" ) {
                continue;   // would overflow the 64-bit space when shifted
            }
            if ( high != 0 ) fixture.name += "@" + std::to_string( high >> 32 );
            all.push_back( std::move( fixture ) );
        }
    }
    // Several high-32 buckets in ONE bitmap, including the extremes.
    Fixture spanning{ "buckets", {} };
    for ( auto const & fixture : make_fixtures( 0 ) ) {
        if ( fixture.name != "mixed" && fixture.name != "run_many" ) continue;
        for ( std::uint64_t const high : { 0ULL, 3ULL << 32, 9ULL << 32, 0xFFFFFFFFULL << 32 } ) {
            for ( auto const value : fixture.values ) spanning.values.push_back( value + high );
        }
    }
    spanning.values.push_back( ~std::uint64_t{ 0 } );
    finish( spanning );
    all.push_back( std::move( spanning ) );
    // One value per bucket: as many buckets as values.
    Fixture sparse{ "one_per_bucket", {} };
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) sparse.values.push_back( ( i * 40000001ULL ) << 32 | ( i * 977 ) );
    finish( sparse );
    all.push_back( std::move( sparse ) );
    return all;
}

std::vector<Fixture> make_all_fixtures_32() {
    return make_fixtures( 0 );
}

template <typename Ref>
typename Ref::bitmap build_frsr( std::span<std::uint64_t const> values, bool const optimize ) {
    using key_type = typename Ref::bitmap::key_type;
    std::vector<key_type> keys( values.begin(), values.end() );
    typename Ref::bitmap result{ std::span<key_type const>{ keys } };
    if ( optimize ) {
        result.optimize_for_storage();
    }
    return result;
}

template <typename Ref>
Bytes frsr_bytes( typename Ref::bitmap const & bitmap ) {
    Bytes bytes( bitmap.portable_size_in_bytes() );
    EXPECT_EQ( bitmap.portable_serialize( bytes ), bytes.size() );
    return bytes;
}

template <typename Ref>
std::vector<std::uint64_t> frsr_values( typename Ref::bitmap const & bitmap ) {
    auto const raw{ bitmap.to_vector() };
    return { raw.begin(), raw.end() };
}

template <typename Ref>
void check_interop( std::vector<Fixture> const & fixtures ) {
    for ( auto const & fixture : fixtures ) {
        for ( bool const optimize : { false, true } ) {
            SCOPED_TRACE( fixture.name + ( optimize ? " optimized" : " plain" ) );
            auto const mine{ build_frsr<Ref>( fixture.values, optimize ) };
            auto const theirs{ Ref::build( fixture.values, optimize ) };

            // frsr -> bytes -> CRoaring (the safe reader) == original.
            auto const written{ frsr_bytes<Ref>( mine ) };
            auto const read_back{ Ref::deserialize( written ) };
            ASSERT_NE( read_back, nullptr );
            EXPECT_EQ( Ref::values( read_back ), fixture.values );

            // CRoaring -> bytes -> frsr == original.
            auto const reference_bytes{ Ref::serialize( theirs ) };
            auto const loaded{ Ref::bitmap::portable_deserialize( reference_bytes ) };
            ASSERT_TRUE( loaded.has_value() );
            EXPECT_EQ( loaded->size(), fixture.values.size() );
            EXPECT_EQ( frsr_values<Ref>( *loaded ), fixture.values );
            EXPECT_TRUE( *loaded == mine );

            // The same bytes also when the writer is frsr.
            auto const reloaded{ Ref::bitmap::portable_deserialize( written ) };
            ASSERT_TRUE( reloaded.has_value() );
            EXPECT_EQ( frsr_values<Ref>( *reloaded ), fixture.values );

            // Same logical set and same container kinds => same bytes. Run-optimizing
            // both sides makes the kinds a function of the set.
            if ( optimize ) {
                EXPECT_EQ( written.size(), reference_bytes.size() );
                EXPECT_TRUE( written == reference_bytes );
            }
        }
    }
}

TEST( FrsrRoaringPortable, Interop32 ) {
    check_interop<Reference32>( make_all_fixtures_32() );
}

TEST( FrsrRoaringPortable, Interop64 ) {
    check_interop<Reference64>( make_all_fixtures_64() );
}

// A deserialized bitmap is a first-class bitmap: it mutates and combines like any other.
TEST( FrsrRoaringPortable, LoadedBitmapStaysUsable ) {
    auto fixtures{ make_all_fixtures_32() };
    for ( auto const & fixture : fixtures ) {
        auto const theirs{ Reference32::build( fixture.values, true ) };
        auto loaded{ *Bitmap32::portable_deserialize( Reference32::serialize( theirs ) ) };
        auto expected{ build_frsr<Reference32>( fixture.values, false ) };
        for ( std::uint32_t const value : { 0U, 77U, 65536U + 5U, 300000U, 0xFFFFFFFFU } ) {
            loaded.add( value );
            expected.add( value );
        }
        loaded.remove( 123456U );
        expected.remove( 123456U );
        EXPECT_TRUE( loaded == expected ) << fixture.name;
        EXPECT_TRUE( ( loaded & expected ) == expected ) << fixture.name;
    }
}

TEST( FrsrRoaringPortable, SizeQueryMatchesReference ) {
    for ( auto const & fixture : make_all_fixtures_64() ) {
        auto const mine{ build_frsr<Reference64>( fixture.values, true ) };
        auto const theirs{ Reference64::build( fixture.values, true ) };
        EXPECT_EQ( mine.portable_size_in_bytes(), roaring64_bitmap_portable_size_in_bytes( theirs.get() ) ) << fixture.name;
    }
}

// ---- malformed input ---------------------------------------------------------------

template <typename Bitmap>
void expect_rejected( Bytes const & bytes, char const * what ) {
    EXPECT_FALSE( Bitmap::portable_deserialize( bytes ).has_value() ) << what;
}

template <typename T>
void poke( Bytes & bytes, std::size_t const offset, T const value ) {
    ASSERT_LE( offset + sizeof( T ), bytes.size() );
    std::memcpy( bytes.data() + offset, &value, sizeof( T ) );
}

TEST( FrsrRoaringPortable, RejectsEveryTruncation ) {
    for ( auto const & fixture : make_all_fixtures_32() ) {
        if ( fixture.values.size() > 70000 ) continue;   // keep the quadratic sweep small
        auto const bytes{ frsr_bytes<Reference32>( build_frsr<Reference32>( fixture.values, true ) ) };
        for ( std::size_t length{ 0 }; length < bytes.size(); length += 1 + length / 64 ) {
            Bytes const prefix( bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>( length ) );
            EXPECT_FALSE( Bitmap32::portable_deserialize( prefix ).has_value() ) << fixture.name << " prefix " << length;
        }
    }
    for ( auto const & fixture : make_all_fixtures_64() ) {
        if ( fixture.values.size() > 70000 ) continue;
        auto const bytes{ frsr_bytes<Reference64>( build_frsr<Reference64>( fixture.values, true ) ) };
        for ( std::size_t length{ 0 }; length < bytes.size(); length += 1 + length / 64 ) {
            Bytes const prefix( bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>( length ) );
            EXPECT_FALSE( Bitmap64::portable_deserialize( prefix ).has_value() ) << fixture.name << " prefix " << length;
        }
    }
}

TEST( FrsrRoaringPortable, RejectsStructurallyInvalidInput ) {
    // Cookie 12346: [cookie][count] then (key, card-1) pairs, offsets, payloads.
    std::vector<std::uint64_t> two_arrays{ 1, 2, 3, 65536 + 9 };
    auto const valid{ frsr_bytes<Reference32>( build_frsr<Reference32>( two_arrays, false ) ) };
    ASSERT_TRUE( Bitmap32::portable_deserialize( valid ).has_value() );
    constexpr std::size_t descriptive{ 8 };
    constexpr std::size_t payload{ descriptive + 2 * 4 + 2 * 4 };

    { auto b{ valid }; poke<std::uint32_t>( b, 0, 0xDEADBEEF ); expect_rejected<Bitmap32>( b, "bad cookie" ); }
    { auto b{ valid }; poke<std::uint32_t>( b, 4, 65537 ); expect_rejected<Bitmap32>( b, "container count above 65536" ); }
    { auto b{ valid }; poke<std::uint32_t>( b, 4, 0xFFFFFFFF ); expect_rejected<Bitmap32>( b, "huge container count" ); }
    { auto b{ valid }; poke<std::uint16_t>( b, descriptive + 4, 0 ); expect_rejected<Bitmap32>( b, "keys not increasing" ); }
    { auto b{ valid }; poke<std::uint16_t>( b, descriptive + 2, 5 ); expect_rejected<Bitmap32>( b, "cardinality exceeding the data" ); }
    { auto b{ valid }; poke<std::uint16_t>( b, payload, 3 ); poke<std::uint16_t>( b, payload + 2, 2 ); expect_rejected<Bitmap32>( b, "unsorted array" ); }
    { auto b{ valid }; poke<std::uint16_t>( b, payload + 2, 1 ); expect_rejected<Bitmap32>( b, "duplicate array value" ); }

    // A bitset whose popcount disagrees with the descriptive header.
    std::vector<std::uint64_t> dense;
    append_range( dense, 0, 20000 );
    auto const bitset_bytes{ frsr_bytes<Reference32>( build_frsr<Reference32>( dense, false ) ) };
    { auto b{ bitset_bytes }; b[ b.size() - 1 ] ^= std::byte{ 1 }; expect_rejected<Bitmap32>( b, "bitset popcount" ); }

    // Run containers: out of range, overlapping, and a cardinality mismatch.
    std::vector<std::uint64_t> runs;
    append_range( runs, 10, 20 );
    append_range( runs, 40, 60 );
    auto const run_bytes{ frsr_bytes<Reference32>( build_frsr<Reference32>( runs, true ) ) };
    auto const run_payload{ run_bytes.size() - ( 2 + 8 ) };
    ASSERT_TRUE( Bitmap32::portable_deserialize( run_bytes ).has_value() );
    { auto b{ run_bytes }; poke<std::uint16_t>( b, run_payload + 2 + 4, 65530 ); expect_rejected<Bitmap32>( b, "run beyond 65535" ); }
    { auto b{ run_bytes }; poke<std::uint16_t>( b, run_payload + 2 + 4, 15 ); expect_rejected<Bitmap32>( b, "overlapping runs" ); }
    { auto b{ run_bytes }; poke<std::uint16_t>( b, run_payload + 2 + 6, 3 ); expect_rejected<Bitmap32>( b, "run cardinality mismatch" ); }
    { auto b{ run_bytes }; poke<std::uint16_t>( b, run_payload, 0 ); expect_rejected<Bitmap32>( b, "zero runs" ); }

    // 64-bit: bucket count and key order.
    std::vector<std::uint64_t> buckets{ 5, 6, ( 3ULL << 32 ) + 1, ( 9ULL << 32 ) + 2 };
    auto const wide{ frsr_bytes<Reference64>( build_frsr<Reference64>( buckets, false ) ) };
    ASSERT_TRUE( Bitmap64::portable_deserialize( wide ).has_value() );
    { auto b{ wide }; poke<std::uint64_t>( b, 0, 1ULL << 40 ); expect_rejected<Bitmap64>( b, "bucket count above 32 bits" ); }
    { auto b{ wide }; poke<std::uint64_t>( b, 0, 0xFFFFFFFFULL ); expect_rejected<Bitmap64>( b, "more buckets than bytes" ); }
    { auto b{ wide }; poke<std::uint32_t>( b, 8, 9 ); expect_rejected<Bitmap64>( b, "bucket keys not increasing" ); }
}

// Whatever a damaged buffer decodes to must be a valid bitmap; mostly it must be refused.
template <typename Ref>
void fuzz( std::vector<Fixture> const & fixtures, std::size_t const rounds ) {
    std::mt19937_64 rng{ 77U };
    std::size_t accepted{ 0 }, refused{ 0 };
    for ( auto const & fixture : fixtures ) {
        if ( fixture.values.empty() || fixture.values.size() > 70000 ) continue;
        auto const original{ frsr_bytes<Ref>( build_frsr<Ref>( fixture.values, true ) ) };
        for ( std::size_t round{ 0 }; round < rounds; ++round ) {
            auto bytes{ original };
            auto const mutations{ 1 + rng() % 4 };
            for ( std::size_t m{ 0 }; m < mutations; ++m ) {
                switch ( rng() % 4 ) {
                    case 0: bytes[ rng() % bytes.size() ] = static_cast<std::byte>( rng() ); break;
                    case 1: bytes[ rng() % bytes.size() ] ^= static_cast<std::byte>( 1U << ( rng() % 8 ) ); break;
                    case 2: {   // overwrite a field-sized window, biased toward the header
                        auto const at{ rng() % std::min<std::size_t>( bytes.size(), 64 ) };
                        auto const value{ static_cast<std::uint32_t>( rng() ) };
                        std::memcpy( bytes.data() + at, &value, std::min<std::size_t>( 4, bytes.size() - at ) );
                        break;
                    }
                    default: bytes.resize( rng() % ( bytes.size() + 1 ) ); if ( bytes.empty() ) bytes.push_back( std::byte{} ); break;
                }
            }
            auto const result{ Ref::bitmap::portable_deserialize( bytes ) };
            if ( !result ) {
                ++refused;
                continue;
            }
            ++accepted;
            auto const values{ frsr_values<Ref>( *result ) };
            EXPECT_EQ( values.size(), result->size() );
            EXPECT_TRUE( std::ranges::adjacent_find( values, std::greater_equal<>{} ) == values.end() );
            auto const again{ Ref::bitmap::portable_deserialize( frsr_bytes<Ref>( *result ) ) };
            ASSERT_TRUE( again.has_value() );
            EXPECT_TRUE( *again == *result );
        }
    }
    EXPECT_GT( refused + accepted, std::size_t{ 0 } );
}

TEST( FrsrRoaringPortable, MutatedInputNeverMisbehaves32 ) { fuzz<Reference32>( make_all_fixtures_32(), 60 ); }
TEST( FrsrRoaringPortable, MutatedInputNeverMisbehaves64 ) { fuzz<Reference64>( make_all_fixtures_64(), 12 ); }

} // namespace
