// Symmetric difference against CRoaring: roaring_bitmap_xor / xor_inplace / xor_cardinality / xor_many and the
// roaring64 counterparts, on bitmaps fed the same values. Contents are compared; the form a result settles in
// is checked against the library's own rules in symmetric_difference.cpp.

#include "kernel_test_support.hpp"

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

#include <numeric>
#include <utility>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

using Bitmap32 = bitmap<std::uint32_t>;
using Bitmap64 = bitmap<std::uint64_t>;

template <typename Key>
struct reference;

template <>
struct reference<std::uint32_t> {
    struct owner {
        explicit owner( roaring_bitmap_t * raw ) noexcept : h{ raw } {}
        owner( owner && other ) noexcept : h{ std::exchange( other.h, nullptr ) } {}
        ~owner() { if ( h != nullptr ) { roaring_bitmap_free( h ); } }
        roaring_bitmap_t * h;
    };
    static owner make( std::vector<std::uint32_t> const & values, bool const run_optimize ) {
        owner r{ roaring_bitmap_of_ptr( values.size(), values.data() ) };
        if ( run_optimize ) { roaring_bitmap_run_optimize( r.h ); }
        return r;
    }
    static owner copy( owner const & a ) { return owner{ roaring_bitmap_copy( a.h ) }; }
    static owner xor_of( owner const & a, owner const & b ) { return owner{ roaring_bitmap_xor( a.h, b.h ) }; }
    static void xor_inplace( owner & a, owner const & b ) { roaring_bitmap_xor_inplace( a.h, b.h ); }
    static std::uint64_t xor_cardinality( owner const & a, owner const & b ) { return roaring_bitmap_xor_cardinality( a.h, b.h ); }
    static owner xor_many( std::vector<owner> const & operands ) {
        std::vector<roaring_bitmap_t const *> raw;
        for ( auto const & o : operands ) { raw.push_back( o.h ); }
        return owner{ roaring_bitmap_xor_many( raw.size(), raw.data() ) };
    }
    static std::vector<std::uint32_t> contents( owner const & a ) {
        std::vector<std::uint32_t> values( roaring_bitmap_get_cardinality( a.h ) );
        roaring_bitmap_to_uint32_array( a.h, values.data() );
        return values;
    }
    static void flip_closed( owner & a, std::uint64_t const first, std::uint64_t const last ) { roaring_bitmap_flip_inplace( a.h, first, last + 1 ); }
};

template <>
struct reference<std::uint64_t> {
    struct owner {
        explicit owner( roaring64_bitmap_t * raw ) noexcept : h{ raw } {}
        owner( owner && other ) noexcept : h{ std::exchange( other.h, nullptr ) } {}
        ~owner() { if ( h != nullptr ) { roaring64_bitmap_free( h ); } }
        roaring64_bitmap_t * h;
    };
    static owner make( std::vector<std::uint64_t> const & values, bool const run_optimize ) {
        owner r{ roaring64_bitmap_create() };
        roaring64_bitmap_add_many( r.h, values.size(), values.data() );
        if ( run_optimize ) { roaring64_bitmap_run_optimize( r.h ); }
        return r;
    }
    static owner copy( owner const & a ) { return owner{ roaring64_bitmap_copy( a.h ) }; }
    static owner xor_of( owner const & a, owner const & b ) { return owner{ roaring64_bitmap_xor( a.h, b.h ) }; }
    static void xor_inplace( owner & a, owner const & b ) { roaring64_bitmap_xor_inplace( a.h, b.h ); }
    static std::uint64_t xor_cardinality( owner const & a, owner const & b ) { return roaring64_bitmap_xor_cardinality( a.h, b.h ); }
    static std::vector<std::uint64_t> contents( owner const & a ) {
        std::vector<std::uint64_t> values( roaring64_bitmap_get_cardinality( a.h ) );
        roaring64_bitmap_to_uint64_array( a.h, values.data() );
        return values;
    }
    static void flip_closed( owner & a, std::uint64_t const first, std::uint64_t const last ) { roaring64_bitmap_flip_closed_inplace( a.h, first, last ); }
};

// `count` chunks, spread as the key type allows, in the container forms the sizes and shapes give.
template <typename Key>
[[nodiscard]] std::vector<Key> random_values( std::mt19937_64 & rng, std::size_t const count, std::uint64_t const key_span ) {
    std::vector<Key> values;
    for ( std::size_t c{ 0 }; c < count; ++c ) {
        auto const base{ static_cast<Key>( ( rng() % key_span ) << 16 ) };
        std::size_t const sizes[]{ 0, 1, 20, 300, 2000, 4095, 5000, 12'000, 40'000 };
        auto const n{ sizes[ rng() % std::size( sizes ) ] };
        std::vector<std::uint16_t> lows;
        if ( n > 4096 ) {
            lows.resize( chunk_span );
            std::iota( lows.begin(), lows.end(), std::uint16_t{ 0 } );
            std::ranges::shuffle( lows, rng );
            lows.resize( n );
            std::ranges::sort( lows );
        } else if ( rng() % 3 == 0 ) {
            lows = make_pattern( pattern::one_run, n, rng );
        } else if ( rng() % 3 == 0 ) {
            lows = make_pattern( pattern::blocks64, n, rng );
        } else {
            lows = make_pattern( pattern::random, n, rng );
        }
        auto const piece{ in_chunk<Key>( lows, base ) };
        values.insert( values.end(), piece.begin(), piece.end() );
    }
    return sorted_unique( std::move( values ) );
}

template <typename Key>
void same_as_reference( bitmap<Key> const & ours, std::vector<Key> const & theirs, char const * what ) {
    ASSERT_EQ( ours.validate(), nullptr ) << what << ": " << ours.validate();
    ASSERT_EQ( ours.size(), theirs.size() ) << what;
    auto const mine{ ours.to_vector() };
    ASSERT_TRUE( std::equal( mine.begin(), mine.end(), theirs.begin(), theirs.end() ) ) << what << " differs from CRoaring";
}

template <typename Key>
void crosscheck( std::uint64_t const seed, std::uint64_t const key_span ) {
    using Bitmap = bitmap<Key>;
    using ref = reference<Key>;
    std::mt19937_64 rng{ seed };
    for ( int trial{ 0 }; trial < 150; ++trial ) {
        auto const va{ random_values<Key>( rng, 1 + rng() % 12, key_span ) };
        auto const vb{ random_values<Key>( rng, 1 + rng() % 12, key_span ) };
        bool const optimize_a{ rng() % 2 == 0 };
        bool const optimize_b{ rng() % 2 == 0 };
        Bitmap a;
        Bitmap b;
        a.add_many( va );
        b.add_many( vb );
        if ( optimize_a ) { a.optimize(); }
        if ( optimize_b ) { b.optimize(); }
        auto ra{ ref::make( va, optimize_a ) };
        auto rb{ ref::make( vb, optimize_b ) };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " " << va.size() << " ^ " << vb.size() << " optimized " << optimize_a << optimize_b );

        auto const expected{ ref::contents( ref::xor_of( ra, rb ) ) };
        std::vector<Key> narrow( expected.begin(), expected.end() );
        same_as_reference( a ^ b, narrow, "operator^" );
        same_as_reference( b ^ a, narrow, "operator^ reversed" );
        {
            auto c{ a };
            c ^= b;
            same_as_reference( c, narrow, "operator^=" );
            auto rc{ ref::copy( ra ) };
            ref::xor_inplace( rc, rb );
            auto const in_place{ ref::contents( rc ) };
            ASSERT_TRUE( std::equal( in_place.begin(), in_place.end(), expected.begin(), expected.end() ) );
        }
        EXPECT_EQ( a.xor_cardinality( b ), ref::xor_cardinality( ra, rb ) );
        EXPECT_EQ( b.xor_cardinality( a ), ref::xor_cardinality( rb, ra ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TEST( SymmetricDifferenceCrosscheck, ThirtyTwoBitKeysFewChunks )  { crosscheck<std::uint32_t>( 1, 6 ); }
TEST( SymmetricDifferenceCrosscheck, ThirtyTwoBitKeysManyChunks ) { crosscheck<std::uint32_t>( 2, 4000 ); }
TEST( SymmetricDifferenceCrosscheck, SixtyFourBitKeysFewChunks )  { crosscheck<std::uint64_t>( 3, 6 ); }
TEST( SymmetricDifferenceCrosscheck, SixtyFourBitKeysSparseChunks ) { crosscheck<std::uint64_t>( 4, std::uint64_t{ 1 } << 40 ); }

// The container-pair matrix: both libraries are given the same pair of forms and the result sets are compared.
TEST( SymmetricDifferenceCrosscheck, EveryPairOfContainerForms ) {
    std::mt19937_64 rng{ 17 };
    std::vector<std::pair<std::string, std::vector<std::uint32_t>>> forms;
    auto const add_form{ [ & ]( std::string name, std::vector<std::uint16_t> const & lows ) { forms.emplace_back( std::move( name ), in_chunk<std::uint32_t>( lows, 0x20000 ) ); } };
    add_form( "empty", {} );
    for ( std::size_t const n : { 1, 31, 32, 33, 2048, 4094, 4095 } ) { add_form( "array/" + std::to_string( n ), make_pattern( pattern::random, n, rng ) ); }
    for ( std::size_t const n : { 4097, 8000, 30'000, 65'535 } ) {
        std::vector<std::uint16_t> all( chunk_span );
        std::iota( all.begin(), all.end(), std::uint16_t{ 0 } );
        std::ranges::shuffle( all, rng );
        all.resize( n );
        std::ranges::sort( all );
        add_form( "bitset/" + std::to_string( n ), all );
    }
    add_form( "run/wide", values_of_spans( { { 100, 65'000 } } ) );
    add_form( "run/full", values_of_spans( { { 0, 65'535 } } ) );
    add_form( "run/few", values_of_spans( { { 10, 20 }, { 500, 900 }, { 4096, 4200 }, { 30'000, 40'000 } } ) );
    add_form( "run/singletons", values_of_spans( { { 7, 7 }, { 9, 9 }, { 11, 11 }, { 13, 13 }, { 15, 20 } } ) );
    for ( auto const & [name_a, va] : forms ) {
        for ( auto const & [name_b, vb] : forms ) {
            SCOPED_TRACE( name_a + " ^ " + name_b );
            Bitmap32 a;
            Bitmap32 b;
            a.add_many( va );
            b.add_many( vb );
            a.optimize();
            b.optimize();
            auto ra{ reference<std::uint32_t>::make( va, true ) };
            auto rb{ reference<std::uint32_t>::make( vb, true ) };
            auto const expected{ reference<std::uint32_t>::contents( reference<std::uint32_t>::xor_of( ra, rb ) ) };
            same_as_reference( a ^ b, expected, "operator^" );
            auto c{ a };
            c ^= b;
            same_as_reference( c, expected, "operator^=" );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

TEST( SymmetricDifferenceCrosscheck, FoldMatchesXorMany ) {
    using ref = reference<std::uint32_t>;
    std::mt19937_64 rng{ 29 };
    for ( int trial{ 0 }; trial < 40; ++trial ) {
        std::vector<Bitmap32> ours;
        std::vector<ref::owner> theirs;
        for ( std::size_t i{ 0 }; i < 1 + rng() % 8; ++i ) {
            auto const values{ random_values<std::uint32_t>( rng, 1 + rng() % 6, 5 ) };
            ours.emplace_back();
            ours.back().add_many( values );
            theirs.push_back( ref::make( values, false ) );
        }
        std::vector<Bitmap32 const *> pointers;
        for ( auto const & bm : ours ) { pointers.push_back( &bm ); }
        auto const expected{ ref::contents( ref::xor_many( theirs ) ) };
        same_as_reference( Bitmap32::xor_many( pointers ), expected, "xor_many" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TEST( SymmetricDifferenceCrosscheck, FlipOfAnIntervalMatchesTheReferenceFlip ) {
    using ref = reference<std::uint32_t>;
    std::mt19937_64 rng{ 41 };
    for ( int trial{ 0 }; trial < 120; ++trial ) {
        auto const values{ random_values<std::uint32_t>( rng, 1 + rng() % 6, 8 ) };
        Bitmap32 a;
        a.add_many( values );
        auto const optimized{ rng() % 2 == 0 };
        if ( optimized ) { a.optimize(); }
        auto r{ ref::make( values, optimized ) };
        auto const first{ rng() % ( 9ULL << 16 ) };
        auto const last{ first + rng() % ( 3ULL << 16 ) };
        a.xor_with( key_range::closed( first, last ) );
        ref::flip_closed( r, first, last );
        auto const expected{ ref::contents( r ) };
        same_as_reference( a, expected, "flip" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

} // namespace
