// The 64-bit cases of the reference's suite (tests/roaring64_unit.cpp, tests/cpp_roaring64_unit.cpp,
// tests/roaring64_serialization.cpp and tests/art_unit.cpp), replayed against bitmap<std::uint64_t> under the
// value-semantics and the atomic-refcount policies. The tests carry the names of the upstream ones and their data.
// Where the reference API has no counterpart of the same shape, the semantic is reached in the shape this library
// offers it:
//
//   - rank( x ) is the cardinality of [0, x]; a select( i ) is the i-th value of the ascending sequence
//   - run_optimize() is optimize_keep_bitsets()
//   - the checked add / remove are the bools add() and remove() return
//   - the half-open range operations are the key_range operand lane (or_with, andnot_with, xor_with of
//     key_range::half_open), the closed ones add_closed_range, remove_closed_range, flip_closed_range
//   - the iterator's move_equalorlarger is a frozen view's for_each_from
//   - the reference's memory hooks are byte_size() and the fault injection of allocation_fault.hpp

#include "allocation_fault.hpp"

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace frsr::roaring;

template <typename Cow>
using bitmap64 = bitmap<std::uint64_t, default_container_set<std::uint64_t>, Cow>;

template <typename Cow>
class Upstream64 : public ::testing::Test {};

using upstream64_policies = ::testing::Types<detail::cow_value_semantics, detail::cow_atomic_refcount>;
TYPED_TEST_SUITE( Upstream64, upstream64_policies );

constexpr std::uint64_t K32{ std::uint64_t{ 1 } << 32 };
constexpr std::uint64_t H63{ std::uint64_t{ 1 } << 63 };
constexpr std::uint64_t M64{ std::numeric_limits<std::uint64_t>::max() };

using Bytes = std::vector<std::byte>;

template <typename Bitmap>
Bitmap of( std::initializer_list<std::uint64_t> const values ) {
    return Bitmap{ values };
}

template <typename Bitmap>
std::vector<std::uint64_t> values_of( Bitmap const & bitmap ) {
    std::vector<std::uint64_t> out;
    for ( auto const value : bitmap ) {
        out.emplace_back( value );
    }
    return out;
}

// The reference's rank( x ): how many values are at most x.
template <typename Bitmap>
std::uint64_t rank( Bitmap const & bitmap, std::uint64_t const x ) {
    return bitmap.and_cardinality( key_range::closed( 0, x ) );
}

template <typename Bitmap>
std::size_t containers( Bitmap const & bitmap ) {
    return bitmap.statistics().container_count();
}

template <typename Bitmap>
Bytes portable_bytes( Bitmap const & bitmap ) {
    Bytes out( bitmap.portable_size_in_bytes() );
    EXPECT_EQ( bitmap.portable_serialize( out ), out.size() );
    return out;
}

Bytes read_data_file( char const * const name ) {
    std::ifstream in( std::string{ FRSR_ROARING_TEST_DATA_DIR } + "/" + name, std::ios::binary );
    std::vector<char> const raw( std::istreambuf_iterator<char>( in ), {} );
    auto const bytes{ std::as_bytes( std::span{ raw } ) };
    return { bytes.begin(), bytes.end() };
}

// The frozen image of a bitmap in 8-byte aligned storage, and the view on it.
template <typename Bitmap>
struct frozen_image {
    std::vector<std::uint64_t>   storage;
    typename Bitmap::frozen_view view;

    explicit frozen_image( Bitmap const & bitmap ) : storage( ( bitmap.frozen_size_bytes() + 7 ) / 8 ) {
        std::span const bytes{ reinterpret_cast<std::byte *>( storage.data() ), bitmap.frozen_size_bytes() };
        EXPECT_EQ( bitmap.frozen_serialize( bytes ), bytes.size() );
        view = Bitmap::frozen_view_from( bytes );
    }
    frozen_image( frozen_image const & ) = delete;
};

// The values of a frozen view from the first one >= x on: the reference iterator's move_equalorlarger( x ) is
// the front, and false when it is empty.
template <typename View>
std::vector<std::uint64_t> values_from( View const & view, std::uint64_t const x ) {
    std::vector<std::uint64_t> out;
    view.for_each_from( x, [ & ]( std::uint64_t const value ) { out.emplace_back( value ); } );
    return out;
}

// tests/roaring64_unit.cpp: test_move_from_roaring32_copy_on_write. The 64-bit bitmap is built from the 32-bit one
// (the reference adopts its containers; here they are copied); the source and its copy keep their values.
TYPED_TEST( Upstream64, test_move_from_roaring32_copy_on_write ) {
    using Bitmap   = bitmap64<TypeParam>;
    using Bitmap32 = bitmap<std::uint32_t, default_container_set<std::uint32_t>, TypeParam>;
    // array, bitset and run containers
    Bitmap32 values;
    for ( std::uint32_t i{ 0 }; i < 100; i++ ) {
        std::ignore = values.add( 2 * i );
    }
    for ( std::uint32_t i{ 0 }; i < 5000; i++ ) {
        std::ignore = values.add( ( 1U << 16 ) + 2 * i );
    }
    values.or_with( key_range::half_open( 2U << 16, ( 2U << 16 ) + 1000 ) );
    std::uint64_t const absent[]{ 1, ( 1U << 16 ) + 1, ( 2U << 16 ) + 1001 };

    for ( bool const source_alive : { true, false } ) {
        auto source{ std::make_unique<Bitmap32>( values ) };
        Bitmap32 const copy{ *source };
        if ( !source_alive ) {
            source.reset();
        }
        Bitmap moved{ copy };
        EXPECT_STREQ( moved.validate(), nullptr );
        for ( auto const value : absent ) {
            std::ignore = moved.add( value );
        }
        EXPECT_STREQ( moved.validate(), nullptr );
        EXPECT_EQ( moved.size(), values.size() + 3 );
        EXPECT_TRUE( values.iterate( [ & ]( std::uint32_t const value ) { return moved.contains( value ); } ) );
        for ( auto const value : absent ) {
            EXPECT_TRUE( moved.contains( value ) );
        }
        EXPECT_TRUE( copy == values );
        if ( source_alive ) {
            EXPECT_STREQ( source->validate(), nullptr );
            EXPECT_TRUE( *source == values );
        }
    }
}

// tests/roaring64_unit.cpp: test_contains_range and test_contains_range_closed, "containers missing from the end of
// the range, followed by a container past the range"
TYPED_TEST( Upstream64, test_contains_range_missing_tail_containers ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        Bitmap r;
        r.or_with( key_range::half_open( 0, 1 << 16 ) );
        std::ignore = r.add( 4 << 16 );
        EXPECT_TRUE ( r.contains_range( 0, 1 << 16 ) );
        EXPECT_FALSE( r.contains_range( 0, ( 1 << 16 ) + 1 ) );
        EXPECT_FALSE( r.contains_range( 0, ( 3 << 16 ) + 1 ) );
    }
    {
        Bitmap r;
        r.add_closed_range( 0, ( 1 << 16 ) - 1 );
        std::ignore = r.add( 4 << 16 );
        EXPECT_TRUE ( r.is_superset_of( key_range::closed( 0, ( 1 << 16 ) - 1 ) ) );
        EXPECT_FALSE( r.is_superset_of( key_range::closed( 0, 1 << 16 ) ) );
        EXPECT_FALSE( r.is_superset_of( key_range::closed( 0, 3 << 16 ) ) );
    }
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_contains_range
TYPED_TEST( Upstream64, test_cpp_r64_contains_range ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    r.add_closed_range( K32 - 10, K32 + 10 );
    EXPECT_TRUE ( r.contains_range( K32 - 10, K32 + 11 ) );
    EXPECT_FALSE( r.contains_range( K32 - 11, K32 + 11 ) );
    EXPECT_FALSE( r.contains_range( K32 - 10, K32 + 12 ) );
    EXPECT_TRUE ( r.is_superset_of( key_range::closed( K32 - 10, K32 + 10 ) ) );
    EXPECT_FALSE( r.is_superset_of( key_range::closed( K32 - 10, K32 + 11 ) ) );

    // empty ranges are always contained
    EXPECT_TRUE( r.contains_range( 7, 7 ) );
    EXPECT_TRUE( r.is_superset_of( key_range::closed( 8, 7 ) ) );

    std::ignore = r.remove( K32 );
    EXPECT_FALSE( r.is_superset_of( key_range::closed( K32 - 10, K32 + 10 ) ) );
    EXPECT_TRUE ( r.contains_range( K32 + 1, K32 + 11 ) );

    r.add_closed_range( M64 - 4, M64 );
    EXPECT_TRUE ( r.is_superset_of( key_range::closed( M64 - 4, M64 ) ) );
    EXPECT_FALSE( r.is_superset_of( key_range::closed( M64 - 5, M64 ) ) );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_add_remove_checked
TYPED_TEST( Upstream64, test_cpp_r64_add_remove_checked ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    std::uint64_t const v{ std::uint64_t{ 1 } << 50 };
    EXPECT_TRUE ( r.add( v ) );
    EXPECT_FALSE( r.add( v ) );
    EXPECT_TRUE ( r.contains( v ) );
    EXPECT_TRUE ( r.remove( v ) );
    EXPECT_FALSE( r.remove( v ) );
    EXPECT_TRUE ( r.empty() );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_add_remove_range
TYPED_TEST( Upstream64, test_cpp_r64_add_remove_range ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    r.or_with( key_range::half_open( K32 - 10, K32 + 10 ) );   // spans two containers
    EXPECT_EQ( r.size(), 20U );
    EXPECT_EQ( r.front(), K32 - 10 );
    EXPECT_EQ( r.back(), K32 + 9 );

    r.add_closed_range( K32 + 10, K32 + 19 );
    EXPECT_EQ( r.size(), 30U );
    EXPECT_EQ( r.back(), K32 + 19 );

    r.andnot_with( key_range::half_open( K32 - 10, K32 ) );
    EXPECT_EQ( r.size(), 20U );
    EXPECT_EQ( r.front(), K32 );

    r.remove_closed_range( K32, K32 + 18 );
    EXPECT_TRUE( r == of<Bitmap>( { K32 + 19 } ) );

    // empty ranges are no-ops
    r.or_with( key_range::half_open( 5, 5 ) );
    r.add_closed_range( 6, 5 );
    r.andnot_with( key_range::half_open( K32 + 19, K32 + 19 ) );
    r.remove_closed_range( K32 + 20, K32 + 19 );
    EXPECT_TRUE( r == of<Bitmap>( { K32 + 19 } ) );

    // closed ranges can reach UINT64_MAX
    r.add_closed_range( M64 - 9, M64 );
    EXPECT_EQ( r.size(), 11U );
    EXPECT_TRUE( r.contains( M64 ) );
    r.remove_closed_range( M64 - 9, M64 );
    EXPECT_TRUE( r == of<Bitmap>( { K32 + 19 } ) );

    // full containers
    Bitmap big;
    big.or_with( key_range::half_open( 0, std::uint64_t{ 1 } << 18 ) );
    EXPECT_EQ( big.size(), std::uint64_t{ 1 } << 18 );
    big.andnot_with( key_range::half_open( 1000, ( std::uint64_t{ 1 } << 18 ) - 1000 ) );
    EXPECT_EQ( big.size(), 2000U );
    EXPECT_TRUE ( big.contains( 999 ) );
    EXPECT_FALSE( big.contains( 1000 ) );
    EXPECT_TRUE ( big.contains( ( std::uint64_t{ 1 } << 18 ) - 1000 ) );
    EXPECT_STREQ( big.validate(), nullptr );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_is_subset
TYPED_TEST( Upstream64, test_cpp_r64_is_subset ) {
    using Bitmap = bitmap64<TypeParam>;
    auto a{ of<Bitmap>( { 1, std::uint64_t{ 1 } << 40 } ) };
    auto const b{ of<Bitmap>( { 1, 7, std::uint64_t{ 1 } << 40 } ) };
    Bitmap const empty;

    EXPECT_TRUE ( a.is_subset_of( b ) );
    EXPECT_TRUE ( a.is_strict_subset_of( b ) );
    EXPECT_FALSE( b.is_subset_of( a ) );
    EXPECT_FALSE( b.is_strict_subset_of( a ) );

    EXPECT_TRUE ( a.is_subset_of( a ) );
    EXPECT_FALSE( a.is_strict_subset_of( a ) );

    EXPECT_TRUE ( empty.is_subset_of( a ) );
    EXPECT_TRUE ( empty.is_strict_subset_of( a ) );
    EXPECT_TRUE ( empty.is_subset_of( empty ) );
    EXPECT_FALSE( empty.is_strict_subset_of( empty ) );

    std::ignore = a.add( std::uint64_t{ 1 } << 50 );
    EXPECT_FALSE( a.is_subset_of( b ) );
    EXPECT_FALSE( a.is_strict_subset_of( b ) );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_flip
TYPED_TEST( Upstream64, test_cpp_r64_flip ) {
    using Bitmap = bitmap64<TypeParam>;
    std::uint64_t const far{ std::uint64_t{ 1 } << 50 };

    auto r{ of<Bitmap>( { K32 - 2, K32, K32 + 2, far } ) };
    r.xor_with( key_range::half_open( K32 - 2, K32 + 2 ) );
    EXPECT_TRUE( r == of<Bitmap>( { K32 - 1, K32 + 1, K32 + 2, far } ) );
    r.flip_closed_range( K32 - 2, K32 + 2 );
    EXPECT_TRUE( r == of<Bitmap>( { K32 - 2, K32, far } ) );

    // empty ranges are no-ops
    r.xor_with( key_range::half_open( 9, 9 ) );
    r.flip_closed_range( 10, 9 );
    EXPECT_TRUE( r == of<Bitmap>( { K32 - 2, K32, far } ) );

    // closed ranges can reach UINT64_MAX
    Bitmap e;
    e.flip_closed_range( M64 - 9, M64 );
    EXPECT_EQ( e.size(), 10U );
    EXPECT_TRUE( e.contains( M64 ) );
    e.flip_closed_range( M64 - 9, M64 );
    EXPECT_TRUE( e.empty() );
}

// tests/roaring64_unit.cpp: test_self_aliasing_ops, and tests/cpp_roaring64_unit.cpp: test_cpp_r64_self_aliasing (the
// same data). The reference does not allow `r ^= r` and `r -= r`; here they are allowed and empty the bitmap.
TYPED_TEST( Upstream64, test_self_aliasing_ops ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap b;
    b.or_with( key_range::half_open( 1, 11 ) );
    b.optimize_keep_bitsets();
    std::ignore = b.add( K32 + 3 );
    std::ignore = b.add( M64 );
    Bitmap const orig{ b };
    auto const card{ b.size() };

    b |= b;
    EXPECT_STREQ( b.validate(), nullptr );
    EXPECT_TRUE( b == orig );
    b &= b;
    EXPECT_STREQ( b.validate(), nullptr );
    EXPECT_TRUE( b == orig );

    EXPECT_TRUE( ( b | b ) == orig );
    EXPECT_TRUE( ( b & b ) == orig );
    EXPECT_TRUE( ( b ^ b ).empty() );
    EXPECT_TRUE( ( b - b ).empty() );
    EXPECT_STREQ( ( b ^ b ).validate(), nullptr );
    EXPECT_STREQ( ( b - b ).validate(), nullptr );

    EXPECT_EQ( b.and_cardinality( b ), card );
    EXPECT_EQ( b.or_cardinality( b ), card );
    EXPECT_EQ( b.xor_cardinality( b ), 0U );
    EXPECT_EQ( b.andnot_cardinality( b ), 0U );
    EXPECT_TRUE ( b.intersects( b ) );
    EXPECT_TRUE ( b.is_subset_of( b ) );
    EXPECT_FALSE( b.is_strict_subset_of( b ) );
    EXPECT_TRUE ( b == orig );

    Bitmap x{ orig };
    x ^= x;
    EXPECT_STREQ( x.validate(), nullptr );
    EXPECT_TRUE( x.empty() );
    Bitmap d{ orig };
    d -= d;
    EXPECT_STREQ( d.validate(), nullptr );
    EXPECT_TRUE( d.empty() );
}

// tests/roaring64_unit.cpp: test_run_optimize_recovers_space, and tests/cpp_roaring64_unit.cpp:
// test_cpp_r64_run_optimize_recovers_space (the same data): a run container fragmented into runs of one value goes
// back to a compact container, with the same values.
TYPED_TEST( Upstream64, test_run_optimize_recovers_space ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    r.or_with( key_range::half_open( 0, 100 ) );
    r.optimize_keep_bitsets();
    EXPECT_EQ( r.statistics().run_containers, 1U );
    for ( std::uint64_t v{ 200 }; v < 65536; v += 2 ) {
        std::ignore = r.add( v );
    }
    Bitmap const expected{ r };
    auto const size_before{ r.portable_size_in_bytes() };
    r.optimize_keep_bitsets();
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.statistics().run_containers, 0U );
    auto const size_after{ r.portable_size_in_bytes() };
    EXPECT_LT( size_after, size_before / 10 ) << size_before;
    EXPECT_TRUE( r == expected );
    EXPECT_TRUE( expected == r );
    EXPECT_EQ( r.size(), 100U + ( 65536U - 200U ) / 2 );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_deserialize_run_bitset_vals, and tests/roaring64_serialization.cpp:
// test_64maprunbitsetvals. The file was written by the Java library: run, bitset and array containers, and a bucket
// with more than four containers, whose 32-bit bitmap carries the offset header.
TYPED_TEST( Upstream64, test_64maprunbitsetvals ) {
    using Bitmap = bitmap64<TypeParam>;
    auto const bytes{ read_data_file( "64maprunbitsetvals.bin" ) };
    ASSERT_EQ( bytes.size(), 8370U );
    auto const loaded{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( loaded.has_value() );
    EXPECT_STREQ( loaded->validate(), nullptr );

    Bitmap expected;
    std::ignore = expected.add( 5 );
    expected.or_with( key_range::half_open( K32, K32 + 1000 ) );   // a run
    std::ignore = expected.add( K32 + 65536 );
    for ( std::uint64_t i{ 0 }; i < 5000; i++ ) {
        std::ignore = expected.add( 2 * K32 + 2 * i );   // a bitset
    }
    for ( std::uint64_t i{ 1 }; i <= 4; i++ ) {
        std::ignore = expected.add( 2 * K32 + i * 65536 );
    }
    std::ignore = expected.add( H63 - 1 );
    std::ignore = expected.add( H63 );
    expected.add_closed_range( M64 - 99, M64 );

    EXPECT_EQ( loaded->size(), 6108U );
    EXPECT_TRUE( loaded->contains( M64 ) );
    EXPECT_TRUE( *loaded == expected );
    EXPECT_EQ( loaded->portable_size_in_bytes(), bytes.size() );
    EXPECT_TRUE( portable_bytes( *loaded ) == bytes );

    // run-optimized, the expected bitmap serializes to the same bytes
    expected.optimize_keep_bitsets();
    EXPECT_TRUE( portable_bytes( expected ) == bytes );
}

// tests/roaring64_unit.cpp: test_portable_serialize, "a bucket with a run container (and fewer than
// NO_OFFSET_THRESHOLD containers) between buckets without one, so each needs its own header", with the check
// check_portable_serialization makes of the deserialized bitmap's size
TYPED_TEST( Upstream64, test_portable_serialize_run_bucket_between_plain_buckets ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    std::ignore = r.add( 5 );
    r.or_with( key_range::half_open( K32, K32 + 1000 ) );
    std::ignore = r.add( K32 + ( 1 << 16 ) );
    std::ignore = r.add( 2 * K32 );
    r.optimize_keep_bitsets();
    auto const bytes{ portable_bytes( r ) };
    auto const loaded{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( loaded.has_value() );
    EXPECT_STREQ( loaded->validate(), nullptr );
    EXPECT_TRUE( *loaded == r );
    EXPECT_EQ( loaded->portable_size_in_bytes(), bytes.size() );
    EXPECT_TRUE( portable_bytes( *loaded ) == bytes );
}

// tests/roaring64_unit.cpp: check_frozen_view_queries: the read-only API over `v`, which holds what `r` holds and
// was loaded from `bytes`
template <typename Bitmap>
void expect_read_paths( Bitmap const & v, Bitmap const & r, Bytes const & bytes ) {
    EXPECT_STREQ( v.validate(), nullptr );
    EXPECT_TRUE( v == r );
    EXPECT_TRUE( v.is_subset_of( r ) );
    EXPECT_TRUE( r.is_subset_of( v ) );
    EXPECT_TRUE( v.intersects( r ) );

    auto const card{ r.size() };
    EXPECT_EQ( v.size(), card );
    EXPECT_EQ( v.front(), r.front() );
    EXPECT_EQ( v.back(), r.back() );
    EXPECT_EQ( v.cardinality_in_range( 100, 200000 ), r.cardinality_in_range( 100, 200000 ) );

    std::vector<std::uint64_t> values( card );
    EXPECT_EQ( v.to_array_into( values ), card );
    for ( std::size_t i{ 0 }; i < card; i += 97 ) {
        EXPECT_TRUE( v.contains( values[ i ] ) );
        EXPECT_EQ( v.contains( values[ i ] + 1 ), r.contains( values[ i ] + 1 ) );
        EXPECT_EQ( rank( v, values[ i ] ), i + 1 );
    }

    std::uint64_t visited{ 0 };
    EXPECT_TRUE( v.iterate( [ & ]( std::uint64_t ) { ++visited; return true; } ) );
    EXPECT_EQ( visited, card );
    EXPECT_EQ( values_of( v ), values );

    Bitmap const results[]{ v & r, r & v, v | r, r | v, Bitmap{ v }, v ^ r, v - r, r - v };
    for ( std::size_t i{ 0 }; i < 5; i++ ) {
        EXPECT_TRUE( results[ i ] == r ) << i;
    }
    for ( std::size_t i{ 5 }; i < 8; i++ ) {
        EXPECT_TRUE( results[ i ].empty() ) << i;
    }
    Bitmap and_in_place{ r };
    and_in_place &= v;
    EXPECT_TRUE( and_in_place == r );
    Bitmap or_in_place{ r };
    or_in_place |= v;
    EXPECT_TRUE( or_in_place == r );
    Bitmap xor_in_place{ r };
    xor_in_place ^= v;
    EXPECT_TRUE( xor_in_place.empty() );
    Bitmap andnot_in_place{ r };
    andnot_in_place -= v;
    EXPECT_TRUE( andnot_in_place.empty() );
    EXPECT_EQ( v.and_cardinality( r ), card );
    EXPECT_EQ( v.or_cardinality( r ), card );
    EXPECT_EQ( v.xor_cardinality( r ), 0U );
    EXPECT_EQ( v.andnot_cardinality( r ), 0U );

    EXPECT_TRUE( portable_bytes( v ) == bytes );
}

// The portable bytes of `r` written and read at every alignment, and the frozen image borrowed in place: every read
// path of the loaded and of the borrowing bitmap.
template <typename Bitmap>
void expect_unaligned_and_borrowed_reads( Bitmap const & r ) {
    auto const bytes{ portable_bytes( r ) };
    for ( std::size_t offset{ 0 }; offset < 8; offset++ ) {
        Bytes storage( bytes.size() + offset );
        std::span const at{ storage.data() + offset, bytes.size() };
        EXPECT_EQ( r.portable_serialize( at ), bytes.size() );
        auto const loaded{ Bitmap::portable_deserialize( at ) };
        ASSERT_TRUE( loaded.has_value() ) << offset;
        EXPECT_EQ( loaded->portable_size_in_bytes(), bytes.size() );
        expect_read_paths( *loaded, r, bytes );
    }
    frozen_image<Bitmap> const image{ r };
    ASSERT_TRUE( image.view );
    Bitmap borrowed;
    image.view.borrow_into( borrowed );
    expect_read_paths( borrowed, r, bytes );
}

// tests/roaring64_unit.cpp: test_portable_deserialize_frozen_unaligned. The reference reads container payloads where
// they sit in the buffer; here portable input and output at any alignment, and a bitmap that borrows the payloads of a
// frozen image, go through the same read paths.
TYPED_TEST( Upstream64, test_portable_deserialize_frozen_unaligned ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    for ( std::uint64_t i{ 1 }; i < 3000; i += 3 ) {
        std::ignore = r.add( i );
    }
    for ( std::uint64_t i{ 65536 }; i < 65536 + 20000; i += 2 ) {
        std::ignore = r.add( i );
    }
    r.or_with( key_range::half_open( 140000, 150000 ) );
    r.optimize_keep_bitsets();
    expect_unaligned_and_borrowed_reads( r );
    // several 32-bit buckets, some with enough containers for offset headers
    for ( std::uint64_t k{ 3 }; k < 12; k++ ) {
        std::ignore = r.add( ( k << 16 ) + 7 * k );
        std::ignore = r.add( ( k << 32 ) + 7 * k );
    }
    expect_unaligned_and_borrowed_reads( r );
    // the bitset container first, for the minimum
    r.andnot_with( key_range::half_open( 0, 65536 ) );
    expect_unaligned_and_borrowed_reads( r );
}

// tests/roaring64_unit.cpp: test_to_uint64_array, the empty bitmap and the containers under several high keys
TYPED_TEST( Upstream64, test_to_uint64_array_writes_exactly_the_values ) {
    using Bitmap = bitmap64<TypeParam>;
    // empty bitmap: nothing is written
    Bitmap const empty;
    std::uint64_t sentinel{ 42 };
    EXPECT_EQ( empty.to_array_into( std::span{ &sentinel, 1 } ), 0U );
    EXPECT_EQ( sentinel, 42U );

    // bitset, array and run containers under several high keys, including the top of the 64-bit range, checked
    // against the iterator
    Bitmap r;
    for ( std::uint64_t const high : { std::uint64_t{ 0 }, std::uint64_t{ 1 } << 20, std::uint64_t{ 1 } << 40, M64 - 0xFFFF } ) {
        for ( std::uint64_t i{ 0 }; i < 65536; i += 3 ) {   // bitset
            std::ignore = r.add( high + i );
        }
        auto const array_base{ high + ( std::uint64_t{ 1 } << 16 ) };
        for ( std::uint64_t i{ 0 }; i < 65536; i += 1000 ) {   // array
            std::ignore = r.add( array_base + i );
        }
        auto const run_base{ high + ( std::uint64_t{ 2 } << 16 ) };
        r.add_closed_range( run_base + 10, run_base + 500 );
        r.add_closed_range( run_base + 60000, run_base + 65535 );
        auto const full_base{ high + ( std::uint64_t{ 3 } << 16 ) };
        r.add_closed_range( full_base, full_base + 65535 );
        if ( high == M64 - 0xFFFF ) {
            break;   // the other containers would wrap around
        }
    }
    r.optimize_keep_bitsets();
    auto const card{ r.size() };
    std::vector<std::uint64_t> got( card + 1, 7 );
    EXPECT_EQ( r.to_array_into( got ), card );
    EXPECT_EQ( got[ card ], 7U );   // nothing written past the end
    got.pop_back();
    EXPECT_EQ( got, values_of( r ) );
    EXPECT_TRUE( std::ranges::is_sorted( got ) );
    EXPECT_TRUE( r.contains( M64 ) );
}

// tests/roaring64_unit.cpp: test_container_slot_churn. Containers come and go: removing the only value of a container
// drops it, adding a value to a new chunk makes one. The structure (validate) and the contents against std::set.
TYPED_TEST( Upstream64, test_container_slot_churn ) {
    using Bitmap = bitmap64<TypeParam>;
    std::mt19937_64 gen{ 1234 };
    for ( std::uint64_t const universe : { 8U, 64U, 1000U } ) {
        Bitmap r;
        std::set<std::uint64_t> ref;
        for ( int step{ 0 }; step < 6000; ++step ) {
            auto const high{ gen() % universe };
            auto const v{ high << 16 | gen() % 3 };
            switch ( gen() % 5 ) {
                case 0:
                case 1:
                    std::ignore = r.add( v );
                    ref.insert( v );
                    break;
                case 2:
                case 3:
                    std::ignore = r.remove( v );
                    ref.erase( v );
                    break;
                default:
                    if ( step % 97 == 0 ) {
                        std::ignore = r.shrink_to_fit();
                    } else {
                        // remove whole containers at once
                        auto const lo{ ( gen() % universe ) << 16 };
                        auto const hi{ lo + ( ( gen() % 4 ) << 16 ) };
                        r.remove_closed_range( lo, hi );
                        ref.erase( ref.lower_bound( lo ), ref.upper_bound( hi ) );
                    }
            }
            if ( step % 50 == 0 ) {
                ASSERT_STREQ( r.validate(), nullptr ) << step;
                ASSERT_EQ( r.size(), ref.size() ) << step;
            }
        }
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_TRUE( std::ranges::equal( values_of( r ), ref ) );
        // a copy and a shrunken bitmap keep a consistent structure too
        Bitmap const c{ r };
        EXPECT_STREQ( c.validate(), nullptr );
        std::ignore = r.shrink_to_fit();
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_TRUE( r == c );
        std::ignore = r.add( universe << 16 );
        std::ignore = r.remove( *ref.begin() );
        EXPECT_STREQ( r.validate(), nullptr );
    }
}

// tests/roaring64_unit.cpp: test_shrink_to_fit_emptied_holds_no_more_than_new, with byte_size() for the reference's
// memory hook: a bitmap that lost every container holds, once shrunk, no more than a new one, and shrink_to_fit()
// returns what it gave back
TYPED_TEST( Upstream64, test_shrink_to_fit_emptied_holds_no_more_than_new ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    auto const new_bytes{ r.byte_size() };
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
        std::ignore = r.add( i << 16 );
    }
    for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
        std::ignore = r.remove( i << 16 );
    }
    auto const held{ r.byte_size() };

    auto const saved{ r.shrink_to_fit() };

    EXPECT_EQ( saved, held - r.byte_size() );
    EXPECT_LE( r.byte_size(), new_bytes );
    std::ignore = r.add( 42 );
    EXPECT_TRUE( r.contains( 42 ) );
}

// tests/roaring64_unit.cpp: test_shrink_to_fit_failed_allocation_keeps_values. Each allocation of the shrink fails
// alone: the bitmap keeps its values and takes new keys, and the next shrink fits it for the frozen image. The kept
// chunks carry payloads with spare capacity, so the shrink has payloads to reallocate whatever the inline storage.
TYPED_TEST( Upstream64, test_shrink_to_fit_failed_allocation_keeps_values ) {
    if constexpr ( detail::allocation_failure_terminates ) {
        GTEST_SKIP() << "an allocation failure terminates in this build";
    }
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    namespace fault = frsr::roaring::kernel_test::alloc_fault;
    using Bitmap = bitmap64<TypeParam>;
    std::uint32_t failed{ 0 };
    for ( std::uint32_t failing{ 1 };; ++failing ) {
        Bitmap r;
        std::vector<std::uint64_t> kept;
        for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
            std::ignore = r.add( i << 16 );
        }
        for ( std::uint64_t i{ 0 }; i < 100; ++i ) {
            if ( i % 3 != 0 ) {
                std::ignore = r.remove( i << 16 );
                continue;
            }
            kept.emplace_back( i << 16 );
            if ( i % 9 == 0 ) {
                for ( std::uint64_t j{ 1 }; j < 1000; ++j ) {
                    std::ignore = r.add( ( i << 16 ) + 2 * j );
                }
                for ( std::uint64_t j{ 10 }; j < 1000; ++j ) {
                    std::ignore = r.remove( ( i << 16 ) + 2 * j );
                }
                for ( std::uint64_t j{ 1 }; j < 10; ++j ) {
                    kept.emplace_back( ( i << 16 ) + 2 * j );
                }
            }
        }

        bool threw{ false };
        try {
            fault::counter = 0;
            fault::armed const armed{ failing };
            std::ignore = r.shrink_to_fit();
        } catch ( std::bad_alloc const & ) {
            threw = true;
            ++failed;
        }

        ASSERT_STREQ( r.validate(), nullptr ) << failing;
        ASSERT_EQ( values_of( r ), kept ) << failing;
        // new keys need new containers and chunk table entries
        std::vector<std::uint64_t> added;
        for ( std::uint64_t i{ 1 }; i < 100; i += 3 ) {
            added.emplace_back( i << 16 );
        }
        added.emplace_back( std::uint64_t{ 1 } << 40 );
        for ( auto const v : added ) {
            std::ignore = r.add( v );
        }
        EXPECT_STREQ( r.validate(), nullptr );
        for ( auto const v : added ) {
            EXPECT_TRUE( r.contains( v ) );
        }
        EXPECT_EQ( r.size(), kept.size() + added.size() );

        // the next shrink fits it for the frozen image, and leaves nothing for another one
        std::ignore = r.shrink_to_fit();
        EXPECT_EQ( r.shrink_to_fit(), 0U );
        frozen_image<Bitmap> const image{ r };
        ASSERT_TRUE( image.view );
        EXPECT_TRUE( image.view.materialize() == r );
        if ( !threw ) {
            break;   // each allocation of the shrink has failed alone
        }
    }
    EXPECT_GT( failed, 0U );
#endif
}

// tests/roaring64_unit.cpp: test_add_remove_range_across_2_32. A range whose exclusive end falls on 2^32 makes no
// container in the next 32-bit bucket.
TYPED_TEST( Upstream64, test_add_remove_range_across_2_32 ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        Bitmap r;
        r.or_with( key_range::half_open( K32 - 2, K32 ) );
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_EQ( r.size(), 2U );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ K32 - 2, K32 - 1 } ) );
        EXPECT_EQ( containers( r ), 1U );
        EXPECT_FALSE( r.contains( K32 ) );
    }
    {
        Bitmap r;
        r.add_closed_range( K32 - 2, K32 - 1 );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ K32 - 2, K32 - 1 } ) );
        EXPECT_EQ( containers( r ), 1U );
    }
    Bitmap r;
    r.or_with( key_range::half_open( K32 - 10, K32 + 10 ) );
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.size(), 20U );
    EXPECT_EQ( r.front(), K32 - 10 );
    EXPECT_EQ( r.back(), K32 + 9 );
    r.andnot_with( key_range::half_open( K32 - 5, K32 + 5 ) );
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.size(), 10U );
    EXPECT_TRUE ( r.contains( K32 - 6 ) );
    EXPECT_FALSE( r.contains( K32 - 5 ) );
    EXPECT_FALSE( r.contains( K32 + 4 ) );
    EXPECT_TRUE ( r.contains( K32 + 5 ) );
    r.andnot_with( key_range::half_open( K32 - 10, K32 ) );
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.front(), K32 + 5 );
    EXPECT_EQ( containers( r ), 1U );
}

// tests/roaring64_unit.cpp: test_add_range_across_2_63. Ranges crossing 2^63 are unsigned.
TYPED_TEST( Upstream64, test_add_range_across_2_63 ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    r.or_with( key_range::half_open( H63 - 2, H63 + 2 ) );
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.size(), 4U );
    EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ H63 - 2, H63 - 1, H63, H63 + 1 } ) );
    EXPECT_EQ( rank( r, H63 - 1 ), 2U );
    EXPECT_EQ( values_of( r )[ 2 ], H63 );
}

// tests/roaring64_unit.cpp: test_flip_across_2_63
TYPED_TEST( Upstream64, test_flip_across_2_63 ) {
    using Bitmap = bitmap64<TypeParam>;
    auto r{ of<Bitmap>( { 0 } ) };
    Bitmap flipped{ r };
    flipped.xor_with( key_range::half_open( H63 - 1, H63 + 1 ) );
    r.flip_closed_range( H63 - 1, H63 );
    for ( Bitmap const * const b : { &r, &flipped } ) {
        EXPECT_STREQ( b->validate(), nullptr );
        EXPECT_EQ( values_of( *b ), ( std::vector<std::uint64_t>{ 0, H63 - 1, H63 } ) );
        EXPECT_EQ( rank( *b, H63 - 1 ), 2U );
    }
}

// tests/roaring64_unit.cpp: test_flip_xor_empties_first_or_last_container. An exclusive-or or a flip that empties the
// first or the last container keeps the minimum, the maximum and the container count right.
TYPED_TEST( Upstream64, test_flip_xor_empties_first_or_last_container ) {
    using Bitmap = bitmap64<TypeParam>;
    auto const expect_only = []( Bitmap const & b, std::uint64_t const value ) {
        EXPECT_STREQ( b.validate(), nullptr );
        EXPECT_EQ( b.size(), 1U );
        EXPECT_EQ( b.front(), value );
        EXPECT_EQ( b.back(), value );
        EXPECT_EQ( containers( b ), 1U );
    };
    for ( std::uint64_t const second : { std::uint64_t{ 65536 }, K32, M64 } ) {
        auto r{ of<Bitmap>( { 0, second } ) };
        auto const zero{ of<Bitmap>( { 0 } ) };
        auto const last{ of<Bitmap>( { second } ) };

        Bitmap xi{ r };
        xi ^= zero;
        expect_only( r ^ zero, second );
        expect_only( xi, second );

        Bitmap yi{ r };
        yi ^= last;
        expect_only( r ^ last, 0 );
        expect_only( yi, 0 );

        Bitmap f{ r };
        f.flip_closed_range( 0, 0 );
        r.xor_with( key_range::half_open( 0, 1 ) );
        expect_only( f, second );
        expect_only( r, second );
    }
}

// tests/roaring64_unit.cpp: test_boundary_values. Add, contains, remove and the values at the boundaries of the
// 16/32/48/64-bit key spaces, inserted in a shuffled order.
TYPED_TEST( Upstream64, test_boundary_values ) {
    using Bitmap = bitmap64<TypeParam>;
    std::vector<std::uint64_t> const sorted{
        0, 65535, 65536, K32 - 1, K32, ( std::uint64_t{ 1 } << 48 ) - 1, std::uint64_t{ 1 } << 48, H63 - 1, H63, M64 - 1, M64
    };
    auto shuffled{ sorted };
    std::mt19937_64 gen{ 1234 };
    std::ranges::shuffle( shuffled, gen );
    Bitmap r;
    for ( auto const v : shuffled ) {
        EXPECT_TRUE( r.add( v ) );
    }
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_EQ( r.size(), sorted.size() );
    for ( auto const v : sorted ) {
        EXPECT_TRUE( r.contains( v ) );
        if ( v != M64 && !std::ranges::binary_search( sorted, v + 1 ) ) {
            EXPECT_FALSE( r.contains( v + 1 ) );
        }
        if ( v != 0 && !std::ranges::binary_search( sorted, v - 1 ) ) {
            EXPECT_FALSE( r.contains( v - 1 ) );
        }
    }
    EXPECT_EQ( values_of( r ), sorted );
    EXPECT_EQ( r.front(), 0U );
    EXPECT_EQ( r.back(), M64 );

    auto expected_card{ sorted.size() };
    for ( auto const v : shuffled ) {
        EXPECT_TRUE( r.remove( v ) );
        expected_card--;
        EXPECT_FALSE( r.contains( v ) );
        EXPECT_EQ( r.size(), expected_card );
        EXPECT_STREQ( r.validate(), nullptr );
    }
    EXPECT_TRUE( r.empty() );
}

// tests/roaring64_unit.cpp: test_rank_select_extremes. Rank, select, minimum and maximum with values at the
// extremes, 2^63 and UINT64_MAX included (unsigned order).
TYPED_TEST( Upstream64, test_rank_select_extremes ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        auto const r{ of<Bitmap>( { 0, 1, H63 - 1, H63 } ) };
        std::vector<std::uint64_t> const expected{ 0, 1, H63 - 1, H63 };
        EXPECT_EQ( values_of( r ), expected );
        for ( std::uint64_t i{ 0 }; i < 4; ++i ) {
            EXPECT_EQ( rank( r, expected[ i ] ) - 1, i );   // get_index
        }
        EXPECT_EQ( rank( r, 0 ), 1U );
        EXPECT_EQ( rank( r, 1 ), 2U );
        EXPECT_EQ( rank( r, 2 ), 2U );
        EXPECT_EQ( rank( r, H63 - 2 ), 2U );
        EXPECT_EQ( rank( r, H63 - 1 ), 3U );
        EXPECT_EQ( rank( r, H63 ), 4U );
        EXPECT_EQ( rank( r, M64 ), 4U );
        EXPECT_EQ( r.front(), 0U );
        EXPECT_EQ( r.back(), H63 );
    }
    {
        auto const r{ of<Bitmap>( { M64 } ) };
        EXPECT_EQ( rank( r, M64 - 1 ), 0U );
        EXPECT_EQ( rank( r, M64 ), 1U );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ M64 } ) );
        EXPECT_EQ( r.front(), M64 );
        EXPECT_EQ( r.back(), M64 );
    }
    {
        auto const r{ of<Bitmap>( { 1, M64 } ) };
        EXPECT_EQ( values_of( r )[ 1 ], M64 );
        EXPECT_EQ( rank( r, M64 ), 2U );
    }
}

// tests/roaring64_unit.cpp: test_rank_select_empty_and_out_of_range
TYPED_TEST( Upstream64, test_rank_select_empty_and_out_of_range ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    for ( std::uint64_t const p : { std::uint64_t{ 0 }, std::uint64_t{ 1 }, H63 - 1, H63, M64 } ) {
        EXPECT_EQ( rank( r, p ), 0U );
    }
    EXPECT_TRUE( values_of( r ).empty() );
    std::ignore = r.add( 123 );
    EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ 123 } ) );
}

// tests/roaring64_unit.cpp: test_rank_in_missing_bucket: the rank of a value whose 32-bit bucket is absent, between
// two buckets
TYPED_TEST( Upstream64, test_rank_in_missing_bucket ) {
    using Bitmap = bitmap64<TypeParam>;
    auto const r{ of<Bitmap>( { 123, H63 - 1 } ) };
    EXPECT_EQ( rank( r, ( H63 - 1 ) / 2 ), 1U );
    EXPECT_EQ( rank( r, H63 - 2 ), 1U );
    EXPECT_EQ( rank( r, H63 - 1 ), 2U );

    auto const s{ of<Bitmap>( { 5, 3 * K32 + 7 } ) };
    EXPECT_EQ( rank( s, 2 * K32 ), 1U );
    EXPECT_EQ( rank( s, 3 * K32 + 6 ), 1U );
    EXPECT_EQ( rank( s, 3 * K32 + 7 ), 2U );
}

// tests/roaring64_unit.cpp: test_remove_last_value_of_bucket: the only value of a 32-bit bucket removed, then the
// values and an add again
TYPED_TEST( Upstream64, test_remove_last_value_of_bucket ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        auto r{ of<Bitmap>( { 123, H63 - 1 } ) };
        std::ignore = r.remove( H63 - 1 );
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_EQ( r.size(), 1U );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ 123 } ) );
        std::ignore = r.add( H63 - 1 );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ 123, H63 - 1 } ) );
    }
    {
        auto r{ of<Bitmap>( { 123, H63 - 1 } ) };
        std::ignore = r.remove( H63 - 1 );
        std::ignore = r.add( ( H63 - 1 ) / 2 );
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ 123, ( H63 - 1 ) / 2 } ) );
    }
    {
        auto r{ of<Bitmap>( { 123, ( H63 - 1 ) / 2 } ) };
        std::ignore = r.remove( ( H63 - 1 ) / 2 );
        std::ignore = r.add( H63 - 1 );
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_EQ( r.size(), 2U );
        EXPECT_EQ( values_of( r ), ( std::vector<std::uint64_t>{ 123, H63 - 1 } ) );
    }
}

// tests/roaring64_unit.cpp: test_java_art_regressions, the regression vectors of Java's TestRoaring64Bitmap
TYPED_TEST( Upstream64, test_java_art_regressions ) {
    using Bitmap = bitmap64<TypeParam>;
    {   // testIssue428
        auto const in{ of<Bitmap>( { 1353768194141061120U } ) };
        auto const compare{ of<Bitmap>( {
            5192650370358181888U, 5193776270265024512U, 5194532734264934400U, 5194544828892839936U,
            5194545653526560768U, 5194545688960040960U, 5194545692181266432U, 5194545705066168320U,
            5194545722246037504U, 5194545928404467712U, 5194550326450978816U, 5194620695195156480U,
            5206161169240293376U
        } ) };
        Bitmap c{ compare };
        c &= in;
        EXPECT_STREQ( c.validate(), nullptr );
        EXPECT_EQ( c.size(), 0U );
        c = compare;
        c |= in;
        EXPECT_STREQ( c.validate(), nullptr );
        EXPECT_EQ( c.size(), 14U );
        c = compare;
        c -= in;
        EXPECT_STREQ( c.validate(), nullptr );
        EXPECT_EQ( c.size(), 13U );
    }
    {   // shouldNotThrowNPE / shouldNotThrowAIOOB / shouldNotThrowIAE
        auto const ref{ of<Bitmap>( { 5183829215128059904U } ) };
        std::vector<std::vector<std::uint64_t>> const crossers{
            { 4413527634823086080U, 4418031234450456576U, 4421408934170984448U, 4421690409147695104U,
              4421479302915162112U, 4421426526357028864U, 4421413332217495552U, 4421416630752378880U,
              4421416905630285824U, 4421417111788716032U, 4421417128968585216U, 4421417133263552512U,
              4421417134337294336U },
            { 4413527634823086080U, 4418031234450456576U, 4421408934170984448U, 4421127459194273792U,
              4420916352961740800U, 4420863576403607552U, 4420850382264074240U, 4420847083729190912U,
              4420847358607097856U, 4420847564765528064U, 4420847616305135616U, 4420847620600102912U,
              4420847623821328384U },
            { 4421416447812311717U, 4420658333523655893U, 4420658332008999025U }
        };
        for ( auto const & cr : crossers ) {
            Bitmap c{ cr };
            EXPECT_STREQ( c.validate(), nullptr );
            EXPECT_EQ( c.size(), cr.size() );
            c &= ref;
            EXPECT_STREQ( c.validate(), nullptr );
            EXPECT_EQ( c.size(), 0U );
        }
    }
    {   // testIssue537
        std::uint64_t const v{ 275846320 };
        auto const a{ of<Bitmap>( { v } ) };
        auto const b{ of<Bitmap>( { v } ) };
        auto c{ of<Bitmap>( {
            275845652, 275845746, 275846148, 275847372, 275847380, 275847388, 275847459, 275847528, 275847586,
            275847588, 275847600, 275847607, 275847610, 275847613, 275847631, 275847664, 275847672, 275847677,
            275847680, 275847742, 275847808, 275847811, 275847824, 275847830, 275847856, 275847861, 275847863,
            275847872, 275847896, 275847923, 275847924, 275847975, 275847990, 275847995, 275848003, 275848080,
            275848081, 275848084, 275848095, 275848100, 275848120, 275848129, 275848134, 275848163, 275848174,
            275848206, 275848218, 275848231, 275848272, 275848281, 275848308, 275848344, 275848376, 275848382,
            275848395, 275848400, 275848411, 275848426, 275848445, 275848449, 275848451, 275848454, 275848469
        } ) };
        c &= b;
        EXPECT_FALSE( c.contains( v ) );
        c &= a;
        EXPECT_FALSE( c.contains( v ) );
        EXPECT_STREQ( c.validate(), nullptr );
        EXPECT_TRUE( c.empty() );
    }
    {   // testIssue580
        auto const r{ of<Bitmap>( {
            3242766498713841665U, 3492544636360507394U, 3418218112527884289U, 3220956490660966402U,
            3495344165583036418U, 3495023214002368514U, 3485108231289675778U
        } ) };
        auto const values{ values_of( r ) };
        EXPECT_EQ( values.size(), 7U );
        EXPECT_TRUE( std::ranges::is_sorted( values ) );
        EXPECT_EQ( std::ranges::adjacent_find( values ), values.end() );
    }
    {   // testIssue619
        std::uint64_t const cleaner_value{ 140664568792144U };
        std::uint64_t const address_value{ 140662937752432U };
        Bitmap address_space;
        auto const cleaner{ of<Bitmap>( { cleaner_value } ) };
        for ( int iteration{ 0 };; iteration++ ) {
            std::ignore = address_space.add( address_value );
            std::ignore = address_space.add( cleaner_value );
            if ( iteration == 33 ) {
                break;
            }
            address_space -= cleaner;
            EXPECT_STREQ( address_space.validate(), nullptr );
        }
        EXPECT_EQ( address_space.size(), 2U );
    }
    {   // testToArrayAfterAndOptHasEmptyContainer
        auto r0{ of<Bitmap>( { 0 } ) };
        auto const r1{ of<Bitmap>( { 1 } ) };
        auto const a{ r0 & r1 };
        r0 &= r1;
        for ( Bitmap const * const b : std::initializer_list<Bitmap const *>{ &r0, &a } ) {
            EXPECT_STREQ( b->validate(), nullptr );
            EXPECT_TRUE( b->empty() );
            EXPECT_EQ( containers( *b ), 0U );
            EXPECT_TRUE( values_of( *b ).empty() );
        }
    }
    {   // testAndDisjoint
        auto left{ of<Bitmap>( { 1076595327100U, 1074755534972U, 5060192403580U, 5060308664444U } ) };
        auto const right{ of<Bitmap>( { 3470563844U } ) };
        auto const a{ left & right };
        auto const swapped{ right & left };
        Bitmap right_copy{ right };
        auto const left_again{ left };
        left &= right;
        right_copy &= left_again;
        for ( Bitmap const * const b : std::initializer_list<Bitmap const *>{ &a, &swapped, &left, &right_copy } ) {
            EXPECT_STREQ( b->validate(), nullptr );
            EXPECT_EQ( b->size(), 0U );
            EXPECT_EQ( containers( *b ), 0U );
        }
    }
}

// tests/roaring64_unit.cpp: test_op_cardinalities_disjoint_and_empty: the cardinality forms and intersects with
// disjoint high keys, several buckets, and an empty operand
TYPED_TEST( Upstream64, test_op_cardinalities_disjoint_and_empty ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        auto const r1{ of<Bitmap>( { 1, 2 * K32 + 5 } ) };
        auto const r2{ of<Bitmap>( { 7 * K32 + 1, 9 * K32 + 5 } ) };
        EXPECT_EQ( r1.and_cardinality( r2 ), 0U );
        EXPECT_FALSE( r1.intersects( r2 ) );
        EXPECT_EQ( r1.or_cardinality( r2 ), 4U );
        EXPECT_EQ( r1.xor_cardinality( r2 ), 4U );
        EXPECT_EQ( r1.andnot_cardinality( r2 ), 2U );
    }
    {
        Bitmap r1;
        r1.or_with( key_range::half_open( 0, 100 ) );
        r1.or_with( key_range::half_open( 2 * K32, 2 * K32 + 100 ) );
        Bitmap r2;
        r2.or_with( key_range::half_open( 50, 150 ) );
        r2.or_with( key_range::half_open( 8 * K32 + 50, 8 * K32 + 150 ) );
        std::ignore = r1.add( 9 * K32 + 1 );
        auto const o{ r1 | r2 };
        auto const x{ r1 ^ r2 };
        auto const d12{ r1 - r2 };
        auto const d21{ r2 - r1 };
        auto const a{ r1 & r2 };
        EXPECT_EQ( o.size(), 351U );
        EXPECT_EQ( r1.or_cardinality( r2 ), o.size() );
        EXPECT_EQ( r1.xor_cardinality( r2 ), x.size() );
        EXPECT_EQ( r1.and_cardinality( r2 ), a.size() );
        EXPECT_EQ( a.size(), 50U );
        EXPECT_EQ( r1.andnot_cardinality( r2 ), d12.size() );
        EXPECT_EQ( r2.andnot_cardinality( r1 ), d21.size() );
        EXPECT_EQ( d12.size(), 151U );
        EXPECT_EQ( d21.size(), 150U );
        EXPECT_TRUE( r1.intersects( r2 ) );
        for ( Bitmap const * const b : { &o, &x, &d12, &d21, &a } ) {
            EXPECT_STREQ( b->validate(), nullptr );
        }
    }
    {
        auto const r{ of<Bitmap>( { 1, K32 + 2, M64 } ) };
        Bitmap const e;
        std::uint64_t const card{ 3 };
        EXPECT_EQ( r.and_cardinality( e ), 0U );
        EXPECT_EQ( e.and_cardinality( r ), 0U );
        EXPECT_EQ( r.or_cardinality( e ), card );
        EXPECT_EQ( e.or_cardinality( r ), card );
        EXPECT_EQ( r.xor_cardinality( e ), card );
        EXPECT_EQ( e.xor_cardinality( r ), card );
        EXPECT_EQ( r.andnot_cardinality( e ), card );
        EXPECT_EQ( e.andnot_cardinality( r ), 0U );
        EXPECT_FALSE( r.intersects( e ) );
        EXPECT_FALSE( e.intersects( r ) );
        EXPECT_FALSE( e.intersects( e ) );
        EXPECT_TRUE ( e.is_subset_of( r ) );
        EXPECT_FALSE( r.is_subset_of( e ) );
    }
}

// tests/roaring64_unit.cpp: test_iterator_move_across_2_63: move_equalorlarger across 2^63 (unsigned order)
TYPED_TEST( Upstream64, test_iterator_move_across_2_63 ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
    r.or_with( key_range::half_open( H63 - 1, H63 + 3 ) );
    frozen_image<Bitmap> const image{ r };
    ASSERT_TRUE( image.view );
    EXPECT_EQ( values_from( image.view, H63 - 1 ), ( std::vector<std::uint64_t>{ H63 - 1, H63, H63 + 1, H63 + 2 } ) );
    EXPECT_EQ( values_from( image.view, H63 + 1 ), ( std::vector<std::uint64_t>{ H63 + 1, H63 + 2 } ) );
    EXPECT_EQ( values_from( image.view, 0 ).front(), H63 - 1 );
    EXPECT_TRUE( values_from( image.view, H63 + 3 ).empty() );
}

// tests/roaring64_unit.cpp: test_iterator_move_to_uint64_max: move_equalorlarger to UINT64_MAX, present or absent
TYPED_TEST( Upstream64, test_iterator_move_to_uint64_max ) {
    using Bitmap = bitmap64<TypeParam>;
    {
        frozen_image<Bitmap> const image{ of<Bitmap>( { 5, M64 - 1, M64 } ) };
        ASSERT_TRUE( image.view );
        EXPECT_EQ( values_from( image.view, M64 ), ( std::vector<std::uint64_t>{ M64 } ) );
        EXPECT_EQ( values_from( image.view, M64 - 1 ), ( std::vector<std::uint64_t>{ M64 - 1, M64 } ) );
    }
    {
        frozen_image<Bitmap> const image{ of<Bitmap>( { 5, M64 - 1 } ) };
        ASSERT_TRUE( image.view );
        EXPECT_TRUE( values_from( image.view, M64 ).empty() );
        EXPECT_EQ( values_from( image.view, M64 - 1 ), ( std::vector<std::uint64_t>{ M64 - 1 } ) );
    }
}

// tests/roaring64_unit.cpp: test_iteration_order_unsigned: the iterator, to_array, iterate and a frozen view all use
// unsigned order across 2^63
TYPED_TEST( Upstream64, test_iteration_order_unsigned ) {
    using Bitmap = bitmap64<TypeParam>;
    auto const r{ of<Bitmap>( { H63, 0, M64, 1, H63 - 1 } ) };
    std::vector<std::uint64_t> const expected{ 0, 1, H63 - 1, H63, M64 };
    EXPECT_EQ( values_of( r ), expected );

    std::vector<std::uint64_t> array( r.size() );
    EXPECT_EQ( r.to_array_into( array ), expected.size() );
    EXPECT_EQ( array, expected );

    std::vector<std::uint64_t> iterated;
    EXPECT_TRUE( r.iterate( [ & ]( std::uint64_t const value ) { iterated.emplace_back( value ); return true; } ) );
    EXPECT_EQ( iterated, expected );

    std::vector<std::uint64_t> visited;
    r.for_each( [ & ]( std::uint64_t const value ) { visited.emplace_back( value ); } );
    EXPECT_EQ( visited, expected );

    frozen_image<Bitmap> const image{ r };
    ASSERT_TRUE( image.view );
    EXPECT_EQ( values_from( image.view, 0 ), expected );
}

// tests/cpp_roaring64_unit.cpp: test_cpp_r64_random_range_ops_vs_set, with std::set as the judge (the reference's
// doublechecked::Roaring64, tests/roaring64_checked.hh): ranges start near container and key boundaries and near the
// top of the 64-bit space, so that they overlap and span containers.
TYPED_TEST( Upstream64, test_cpp_r64_random_range_ops_vs_set ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap r;
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
    auto const is_subset = []( std::set<std::uint64_t> const & a, std::set<std::uint64_t> const & b ) {
        return std::ranges::includes( b, a );
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
        // mostly short ranges, some long enough to fill bitset containers
        auto const width{ next() % 8 == 0 ? next() % 10000 : next() % 300 };
        auto const max{ M64 - min < width ? M64 : min + width };

        bool const check_subset{ i % 16 == 0 };
        Bitmap const before{ check_subset ? r : Bitmap{} };
        auto const check_before{ check };

        switch ( next() % 8 ) {
            case 0:
                r.or_with( key_range::half_open( min, max ) );
                for_each_in( min, max - 1, [ & ]( std::uint64_t const v ) { check.insert( v ); } );
                break;
            case 1:
                r.add_closed_range( min, max );
                for_each_in( min, max, [ & ]( std::uint64_t const v ) { check.insert( v ); } );
                break;
            case 2:
                r.andnot_with( key_range::half_open( min, max ) );
                check.erase( check.lower_bound( min ), check.lower_bound( max ) );
                break;
            case 3:
                r.remove_closed_range( min, max );
                check.erase( check.lower_bound( min ), check.upper_bound( max ) );
                break;
            case 4:
                r.xor_with( key_range::half_open( min, max ) );
                for_each_in( min, max - 1, flip_in_check );
                break;
            case 5:
                r.flip_closed_range( min, max );
                for_each_in( min, max, flip_in_check );
                break;
            case 6:
                EXPECT_EQ( r.add( min ), check.insert( min ).second ) << i;
                break;
            default:
                EXPECT_EQ( r.remove( min ), check.erase( min ) == 1 ) << i;
                break;
        }
        if ( next() % 10 == 0 ) {
            r.optimize_keep_bitsets();
        }

        EXPECT_EQ( r.contains_range( min, max ), min >= max || check_contains_closed( min, max - 1 ) ) << i;
        EXPECT_EQ( r.is_superset_of( key_range::closed( min, max ) ), check_contains_closed( min, max ) ) << i;
        ASSERT_EQ( r.size(), check.size() ) << i;
        if ( check_subset ) {
            EXPECT_EQ( before.is_subset_of( r ), is_subset( check_before, check ) ) << i;
            EXPECT_EQ( r.is_subset_of( before ), is_subset( check, check_before ) ) << i;
            EXPECT_EQ( before.is_strict_subset_of( r ), check_before.size() < check.size() && is_subset( check_before, check ) ) << i;
            EXPECT_EQ( r.is_strict_subset_of( before ), check.size() < check_before.size() && is_subset( check, check_before ) ) << i;
            ASSERT_STREQ( r.validate(), nullptr ) << i;
            ASSERT_TRUE( std::ranges::equal( values_of( r ), check ) ) << i;
        }
    }
    EXPECT_STREQ( r.validate(), nullptr );
    EXPECT_TRUE( std::ranges::equal( values_of( r ), check ) );
}

// tests/roaring64_serialization.cpp: test_64truncated_every_prefix: every strict prefix of a valid serialization is
// rejected. Each prefix is a copy, so that a sanitizer catches a read past it.
TYPED_TEST( Upstream64, test_64truncated_every_prefix ) {
    using Bitmap = bitmap64<TypeParam>;
    auto source{ of<Bitmap>( { 1, K32, M64 } ) };
    source.or_with( key_range::half_open( std::uint64_t{ 1 } << 40, ( std::uint64_t{ 1 } << 40 ) + 5000 ) );
    source.optimize_keep_bitsets();
    auto const bytes{ portable_bytes( source ) };
    for ( std::size_t k{ 0 }; k < bytes.size(); k++ ) {
        Bytes const prefix( bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>( k ) );
        EXPECT_FALSE( Bitmap::portable_deserialize( prefix ).has_value() ) << "prefix of length " << k << " was accepted";
    }
    auto const r{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( r.has_value() );
    EXPECT_TRUE( *r == source );
}

// tests/roaring64_serialization.cpp: test_64run_past_container_end: a run ending at 65535 is valid, one ending at
// 65536 is not
TYPED_TEST( Upstream64, test_64run_past_container_end ) {
    using Bitmap = bitmap64<TypeParam>;
    Bitmap source;
    source.or_with( key_range::half_open( 65000, 65536 ) );
    source.optimize_keep_bitsets();
    ASSERT_EQ( source.statistics().run_containers, 1U );
    auto bytes{ portable_bytes( source ) };
    // the last two bytes hold the length - 1 of the only run (little endian)
    auto const n{ bytes.size() };
    EXPECT_EQ( std::to_integer<unsigned>( bytes[ n - 2 ] ) | std::to_integer<unsigned>( bytes[ n - 1 ] ) << 8, 535U );
    auto const r{ Bitmap::portable_deserialize( bytes ) };
    ASSERT_TRUE( r.has_value() );
    EXPECT_STREQ( r->validate(), nullptr );
    EXPECT_EQ( r->size(), 536U );
    EXPECT_TRUE( r->contains( 65535 ) );

    bytes[ n - 2 ] = std::byte{ 536 & 0xFF };
    bytes[ n - 1 ] = std::byte{ 536 >> 8 };
    EXPECT_FALSE( Bitmap::portable_deserialize( bytes ).has_value() ) << "a run past the end of the container must be rejected";
}

// tests/roaring64_serialization.cpp: test_64deserdecreasingkeys: the keys of the 32-bit buckets must be strictly
// increasing
TYPED_TEST( Upstream64, test_64deserdecreasingkeys ) {
    using Bitmap = bitmap64<TypeParam>;
    std::array<std::uint8_t, 52> data{
        // number of 32-bit bitmaps
        2, 0, 0, 0, 0, 0, 0, 0,
        // top 32 bits of the first bitmap: 2
        2, 0, 0, 0,
        // no-run cookie, 1 container
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        // key 0, cardinality - 1 0, offset 16
        0, 0, 0, 0, 16, 0, 0, 0,
        // value 3
        3, 0,
        // top 32 bits of the second bitmap: 1 (decreasing)
        1, 0, 0, 0,
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 16, 0, 0, 0,
        // value 4
        4, 0,
    };
    EXPECT_FALSE( Bitmap::portable_deserialize( std::as_bytes( std::span{ data } ) ).has_value() );
    // with increasing keys the same buffer is accepted
    data[ 8 ] = 1;
    data[ 8 + 4 + 18 ] = 2;
    auto const r{ Bitmap::portable_deserialize( std::as_bytes( std::span{ data } ) ) };
    ASSERT_TRUE( r.has_value() );
    EXPECT_STREQ( r->validate(), nullptr );
    EXPECT_EQ( r->size(), 2U );
    EXPECT_TRUE( r->contains( K32 + 3 ) );
    EXPECT_TRUE( r->contains( 2 * K32 + 4 ) );
}

// tests/roaring64_serialization.cpp: test_64deseremptybucket: an empty 32-bit bucket is accepted and dropped
TYPED_TEST( Upstream64, test_64deseremptybucket ) {
    using Bitmap = bitmap64<TypeParam>;
    std::array<std::uint8_t, 42> const data{
        // number of 32-bit bitmaps
        2, 0, 0, 0, 0, 0, 0, 0,
        // top 32 bits: 0, then an empty bitmap
        0, 0, 0, 0,
        0x3A, 0x30, 0, 0, 0, 0, 0, 0,
        // top 32 bits: 1, then {7}
        1, 0, 0, 0,
        0x3A, 0x30, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 16, 0, 0, 0,
        7, 0,
    };
    auto const r{ Bitmap::portable_deserialize( std::as_bytes( std::span{ data } ) ) };
    ASSERT_TRUE( r.has_value() );
    EXPECT_STREQ( r->validate(), nullptr );
    EXPECT_EQ( r->size(), 1U );
    EXPECT_TRUE( r->contains( K32 + 7 ) );
    auto const again{ portable_bytes( *r ) };
    // the empty bucket (a 4 byte key and an 8 byte bitmap) is not written again
    EXPECT_EQ( again.size(), data.size() - 12 );
    EXPECT_EQ( again[ 0 ], std::byte{ 1 } );
}

// tests/roaring64_serialization.cpp: test_64roundtrip_after_remove_changes_type: a round trip after a remove has
// turned a bitset container into an array
TYPED_TEST( Upstream64, test_64roundtrip_after_remove_changes_type ) {
    using Bitmap = bitmap64<TypeParam>;
    for ( std::uint64_t const base : { std::uint64_t{ 0 }, std::uint64_t{ 1 } << 40, H63 } ) {
        Bitmap r;
        for ( std::uint64_t i{ 0 }; i < 4097; i++ ) {
            std::ignore = r.add( base + 2 * i );
        }
        std::ignore = r.remove( base );
        EXPECT_EQ( r.size(), 4096U );
        auto const back{ Bitmap::portable_deserialize( portable_bytes( r ) ) };
        ASSERT_TRUE( back.has_value() );
        EXPECT_STREQ( back->validate(), nullptr );
        EXPECT_TRUE( r == *back );
        EXPECT_FALSE( back->contains( base ) );
        EXPECT_TRUE ( back->contains( base + 2 ) );
    }
}

// tests/art_unit.cpp: test_art_find_prefixes. The six-byte ART keys are the 48-bit chunk keys of a 64-bit value: chunk
// tables with keys of long shared prefixes (several fan-outs, the deepest, a single key). Every key is found, and a
// key that differs from a present one in any single byte is found only if it is present too.
TYPED_TEST( Upstream64, test_art_find_prefixes ) {
    using Bitmap = bitmap64<TypeParam>;
    using Key6   = std::array<std::uint8_t, 6>;
    std::vector<std::vector<Key6>> key_sets;
    {
        // several fan-outs under long shared prefixes
        std::vector<Key6> keys;
        std::size_t const sizes[]{ 4, 16, 48, 256 };
        for ( std::size_t i{ 0 }; i < std::size( sizes ); i++ ) {
            for ( std::size_t j{ 0 }; j < sizes[ i ]; j++ ) {
                keys.emplace_back( Key6{ 0, 0, 0, static_cast<std::uint8_t>( i ), static_cast<std::uint8_t>( j ), 7 } );
            }
        }
        key_sets.emplace_back( std::move( keys ) );
    }
    // keys that differ in every byte position from the first one
    key_sets.emplace_back( std::vector<Key6>{
        { 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 1 },
        { 0, 0, 0, 0, 1, 0 },
        { 0, 0, 0, 1, 0, 0 },
        { 0, 0, 1, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0 },
    } );
    // a single key
    key_sets.emplace_back( std::vector<Key6>{ { 1, 2, 3, 4, 5, 6 } } );

    // the value of chunk `key`, low bits i
    auto const value_of = []( Key6 const & key, std::size_t const i ) {
        std::uint64_t chunk{ 0 };
        for ( auto const byte : key ) {
            chunk = chunk << 8 | byte;
        }
        return chunk << 16 | ( i & 0xFFFF );
    };
    for ( auto const & keys : key_sets ) {
        Bitmap r;
        for ( std::size_t i{ 0 }; i < keys.size(); ++i ) {
            std::ignore = r.add( value_of( keys[ i ], i ) );
        }
        EXPECT_STREQ( r.validate(), nullptr );
        EXPECT_EQ( containers( r ), keys.size() );
        for ( std::size_t i{ 0 }; i < keys.size(); ++i ) {
            EXPECT_TRUE( r.contains( value_of( keys[ i ], i ) ) );
            for ( std::size_t b{ 0 }; b < 6; ++b ) {
                auto other{ keys[ i ] };
                other[ b ] ^= 0x80;
                auto const at{ std::ranges::find( keys, other ) };
                auto const present{ at != keys.end() };
                auto const probe{ value_of( other, present ? static_cast<std::size_t>( at - keys.begin() ) : i ) };
                EXPECT_EQ( r.contains( probe ), present );
                EXPECT_EQ( r.intersects_range( probe & ~std::uint64_t{ 0xFFFF }, ( probe | 0xFFFF ) ), present );
            }
        }
    }
}

} // namespace
