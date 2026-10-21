// Symmetric difference (xor): every pair of container forms in both operand orders, the in-place, aliased and
// shared-payload forms, the form a result settles in, chunk tables that gain and lose chunks, the N-way fold and
// the cardinality-only form, against std:: oracles. The CRoaring comparison is in symmetric_difference_crosscheck.cpp.

#include "kernel_test_support.hpp"
#include "allocation_fault.hpp"

#include <new>
#include <numeric>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

namespace alloc_fault = frsr::roaring::kernel_test::alloc_fault;

using Bitmap16 = bitmap<std::uint16_t>;
using Bitmap32 = bitmap<std::uint32_t>;
using Bitmap64 = bitmap<std::uint64_t>;
using CowBitmap32 = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using LazyCowBitmap32 = bitmap<
    std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy
>;

// ---- the container forms an operand can be in ------------------------------------------------------------------

enum class kind : std::uint8_t { absent, array, run, bitset };

struct form {
    std::string                 name;
    kind                        expected;
    std::vector<std::uint16_t>  lows;
};

[[nodiscard]] std::vector<std::uint16_t> random_lows( std::size_t const n, std::mt19937_64 & rng ) {
    return make_pattern( pattern::random, n, rng );
}

[[nodiscard]] std::vector<std::uint16_t> dense_lows( std::size_t const n, std::mt19937_64 & rng ) {
    std::vector<std::uint16_t> all( chunk_span );
    std::iota( all.begin(), all.end(), std::uint16_t{ 0 } );
    std::ranges::shuffle( all, rng );
    all.resize( n );
    std::ranges::sort( all );
    return all;
}

[[nodiscard]] std::vector<form> forms( std::mt19937_64 & rng ) {
    std::vector<form> result;
    result.push_back( { "absent", kind::absent, {} } );
    for ( std::size_t const n : { 1, 2, 31, 32, 33, 300, 2048, 4094, 4095 } ) {
        result.push_back( { "array/" + std::to_string( n ), kind::array, random_lows( n, rng ) } );
    }
    for ( std::size_t const n : { 4097, 4100, 20'000, 40'000, 65'535 } ) {
        result.push_back( { "bitset/" + std::to_string( n ), kind::bitset, dense_lows( n, rng ) } );
    }
    result.push_back( { "run/one-wide",   kind::run, values_of_spans( { { 100, 65'000 } } ) } );
    result.push_back( { "run/full",       kind::run, values_of_spans( { { 0, 65'535 } } ) } );
    result.push_back( { "run/both-ends",  kind::run, values_of_spans( { { 0, 40 }, { 5000, 5100 }, { 65'400, 65'535 } } ) } );
    result.push_back( { "run/few",        kind::run, values_of_spans( { { 10, 20 }, { 500, 900 }, { 4096, 4200 }, { 30'000, 40'000 } } ) } );
    result.push_back( { "run/short",       kind::run, values_of_spans( { { 7, 18 }, { 30, 41 }, { 60, 71 }, { 5000, 5011 }, { 40'000, 40'011 }, { 65'524, 65'535 } } ) } );
    {
        std::vector<std::uint16_t> values;
        for ( std::uint32_t start{ 7 }; start + 40 < chunk_span; start += 1000 ) {
            for ( std::uint32_t v{ start }; v < start + 40; ++v ) { values.push_back( static_cast<std::uint16_t>( v ) ); }
        }
        result.push_back( { "run/many-short", kind::run, std::move( values ) } );
    }
    {   // the same runs shifted by a few places: their exclusive-or is many one-value runs
        std::vector<std::uint16_t> values;
        for ( std::uint32_t start{ 10 }; start + 40 < chunk_span; start += 1000 ) {
            for ( std::uint32_t v{ start }; v < start + 40; ++v ) { values.push_back( static_cast<std::uint16_t>( v ) ); }
        }
        result.push_back( { "run/many-short-shifted", kind::run, std::move( values ) } );
    }
    return result;
}

template <typename Bitmap>
[[nodiscard]] Bitmap make_form( form const & f, typename Bitmap::key_type const base ) {
    using key_type = typename Bitmap::key_type;
    Bitmap bm;
    auto const values{ in_chunk<key_type>( f.lows, base ) };
    if ( f.expected == kind::array ) {
        // add_many settles a full-limit array as a bitset: the array form of that size is reached one key at a time
        for ( auto const v : values ) { (void)bm.add( v ); }
    } else {
        bm.add_many( values );
    }
    if ( f.expected == kind::run ) { bm.optimize(); }
    return bm;
}

template <typename Bitmap>
[[nodiscard]] kind kind_of( Bitmap const & bm ) {
    auto const s{ bm.statistics() };
    if ( s.container_count() == 0 ) { return kind::absent; }
    EXPECT_EQ( s.container_count(), 1U );
    return s.run_containers != 0 ? kind::run : s.bitset_containers != 0 ? kind::bitset : kind::array;
}

// What an exclusive-or leaves in a chunk: an array up to the limit, a bitset above, a run only when an operand was one.
template <typename Bitmap>
void expect_settled_form( Bitmap const & result, form const & a, form const & b ) {
    auto const s{ result.statistics() };
    if ( result.empty() ) {
        EXPECT_EQ( s.container_count(), 0U );
        return;
    }
    if ( a.expected != kind::run && b.expected != kind::run ) {
        EXPECT_EQ( s.run_containers, 0U ) << "an exclusive-or of non-run operands minted a run";
    }
    if ( s.run_containers == 0 ) {
        EXPECT_EQ( s.array_containers != 0, result.size() <= Bitmap::array_to_bitset_threshold ) << "size " << result.size();
    }
}

// ---- every pair of forms ----------------------------------------------------------------------------------------

// `exact_forms`: the operands are in the container form each case names. (Under the lazy run policy the form an
// operand settles in is the policy's to choose, so there the cases only name the contents.)
template <typename Bitmap, bool exact_forms = true>
void every_pair_of_forms() {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0x5E7 };
    auto const all_forms{ forms( rng ) };
    for ( auto const & fa : all_forms ) {
        for ( auto const & fb : all_forms ) {
            SCOPED_TRACE( fa.name + " ^ " + fb.name );
            auto const base{ static_cast<key_type>( sizeof( key_type ) > 2 ? 0x30000 : 0 ) };
            auto const a{ make_form<Bitmap>( fa, base ) };
            auto const b{ make_form<Bitmap>( fb, base ) };
            if constexpr ( exact_forms ) {
                ASSERT_EQ( kind_of( a ), fa.expected );
                ASSERT_EQ( kind_of( b ), fb.expected );
            }
            auto const va{ in_chunk<key_type>( fa.lows, base ) };
            auto const vb{ in_chunk<key_type>( fb.lows, base ) };
            auto const expected{ set_symmetric_difference_of( va, vb ) };

            auto const ab{ a ^ b };
            ASSERT_TRUE( matches( ab, expected, true ) );
            if constexpr ( exact_forms ) { expect_settled_form( ab, fa, fb ); }
            ASSERT_TRUE( matches( b ^ a, expected, true ) );
            // neither operand was touched
            ASSERT_TRUE( matches( a, va ) );
            ASSERT_TRUE( matches( b, vb ) );

            {
                auto c{ a };
                c ^= b;
                ASSERT_TRUE( matches( c, expected, true ) );
                if constexpr ( exact_forms ) { expect_settled_form( c, fa, fb ); }
                ASSERT_TRUE( matches( b, vb ) );
            }
            {
                auto c{ b };
                c ^= a;
                ASSERT_TRUE( matches( c, expected, true ) );
            }
            {
                auto c{ a };
                c.xor_with( b );
                ASSERT_TRUE( matches( c, expected, true ) );
            }
            EXPECT_EQ( a.xor_cardinality( b ), expected.size() );
            EXPECT_EQ( b.xor_cardinality( a ), expected.size() );
            // an exclusive-or twice is the identity
            auto twice{ ab };
            twice ^= b;
            ASSERT_TRUE( matches( twice, va ) );
            // x ^ x is empty, however it is spelled
            auto self{ a };
            self ^= self;
            EXPECT_TRUE( self.empty() );
            ASSERT_EQ( self.validate(), nullptr );
            EXPECT_TRUE( ( a ^ a ).empty() );
            EXPECT_EQ( a.xor_cardinality( a ), 0U );
        }
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TEST( SymmetricDifference, EveryPairOfContainerFormsSixteenBitKeys ) { every_pair_of_forms<Bitmap16>(); }
TEST( SymmetricDifference, EveryPairOfContainerFormsThirtyTwoBitKeys ) { every_pair_of_forms<Bitmap32>(); }
TEST( SymmetricDifference, EveryPairOfContainerFormsSixtyFourBitKeys ) { every_pair_of_forms<Bitmap64>(); }
TEST( SymmetricDifference, EveryPairOfContainerFormsOverSharedPayloads ) { every_pair_of_forms<CowBitmap32>(); }
TEST( SymmetricDifference, EveryPairOfContainerFormsUnderTheLazyRunPolicy ) { every_pair_of_forms<LazyCowBitmap32, false>(); }

// The left operand of an in-place exclusive-or came to hold its values in different ways, which leaves its
// payloads with different spare capacity.
TEST( SymmetricDifference, InPlaceOnPayloadsOfEveryHistory ) {
    std::mt19937_64 rng{ 0xFA11 };
    for ( int trial{ 0 }; trial < 600; ++trial ) {
        auto const pa{ all_patterns[ rng() % all_patterns.size() ] };
        auto const pb{ all_patterns[ rng() % all_patterns.size() ] };
        std::size_t const sizes[]{ 0, 1, 31, 64, 300, 2048, 4095, 5000, 30'000 };
        auto const na{ sizes[ rng() % std::size( sizes ) ] };
        auto const nb{ sizes[ rng() % std::size( sizes ) ] };
        auto const la{ na > 4096 ? dense_lows( na, rng ) : make_pattern( pa, na, rng ) };
        auto const lb{ nb > 4096 ? dense_lows( nb, rng ) : make_pattern( pb, nb, rng ) };
        auto const va{ in_chunk<std::uint32_t>( la, 0x10000 ) };
        auto const vb{ in_chunk<std::uint32_t>( lb, 0x10000 ) };
        auto a{ build_with_layout<Bitmap32>( va, all_layouts[ rng() % all_layouts.size() ], rng ) };
        auto const b{ build_with_layout<Bitmap32>( vb, all_layouts[ rng() % all_layouts.size() ], rng ) };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " " << va.size() << " ^ " << vb.size() );
        if ( rng() % 3 == 0 ) { a.optimize(); }
        a ^= b;
        ASSERT_TRUE( matches( a, set_symmetric_difference_of( va, vb ) ) );
    }
}

// Arrays of every small size against every other, on value sets that overlap heavily and that reach both ends of the
// chunk: the block boundaries of the vector merge, its partial tails and the values 0 and 65535.
TEST( SymmetricDifference, ArraysOfEverySmallSizeAndOverlap ) {
    std::mt19937_64 rng{ 0xB10C };
    for ( std::uint32_t const domain : { 24U, 40U, 100U, 400U, 65'536U } ) {
        for ( std::size_t na{ 0 }; na <= 41; ++na ) {
            for ( std::size_t nb{ 0 }; nb <= 41; ++nb ) {
                if ( na > domain || nb > domain ) { continue; }
                auto draw{ [ & ]( std::size_t const n ) {
                    std::vector<std::uint32_t> values;
                    std::vector<bool> taken( domain );
                    auto const map{ [ & ]( std::uint32_t const v ) { return domain == 65'536U ? v : ( v * 2654435761U ) % 65'536U; } };
                    while ( values.size() < n ) {
                        auto const v{ static_cast<std::uint32_t>( rng() % domain ) };
                        if ( !taken[ v ] ) { taken[ v ] = true; values.push_back( domain == 65'536U && rng() % 4 == 0 ? ( rng() % 2 ? 0U : 65'535U ) : map( v ) ); }
                    }
                    return sorted_unique( std::move( values ) );
                } };
                auto const va{ draw( na ) };
                auto const vb{ draw( nb ) };
                Bitmap32 a;
                Bitmap32 b;
                for ( auto const v : va ) { (void)a.add( v ); }
                for ( auto const v : vb ) { (void)b.add( v ); }
                auto const expected{ set_symmetric_difference_of( va, vb ) };
                SCOPED_TRACE( ::testing::Message() << va.size() << " ^ " << vb.size() << " over " << domain );
                ASSERT_TRUE( matches( a ^ b, expected, true ) );
                ASSERT_TRUE( matches( b ^ a, expected, true ) );
                ASSERT_EQ( a.xor_cardinality( b ), expected.size() );
                ASSERT_EQ( a.and_cardinality( b ), set_intersection_of( va, vb ).size() );
            }
        }
    }
}

// The result of an exclusive-or of arrays is an array up to the limit and a bitset above it.
TEST( SymmetricDifference, ResultFormAtTheArrayLimit ) {
    for ( std::size_t const left : { std::size_t{ 2048 }, std::size_t{ 2049 }, std::size_t{ 3000 } } ) {
        for ( std::size_t const right : { std::size_t{ 2048 }, std::size_t{ 2047 }, std::size_t{ 1000 } } ) {
            Bitmap32 a;
            Bitmap32 b;
            for ( std::uint32_t i{ 0 }; i < left; ++i ) { (void)a.add( 2 * i ); }
            for ( std::uint32_t i{ 0 }; i < right; ++i ) { (void)b.add( 2 * i + 1 ); }   // disjoint: nothing cancels
            auto const result{ a ^ b };
            auto c{ a };
            c ^= b;
            for ( Bitmap32 const * const r : std::array<Bitmap32 const *, 2>{ &result, &c } ) {
                ASSERT_EQ( r->size(), left + right );
                EXPECT_EQ( r->statistics().array_containers, left + right <= 4096 ? 1U : 0U ) << left << " ^ " << right;
                EXPECT_EQ( r->statistics().bitset_containers, left + right <= 4096 ? 0U : 1U ) << left << " ^ " << right;
                EXPECT_EQ( r->validate( true ), nullptr );
            }
        }
    }
}

// ---- shared payloads ---------------------------------------------------------------------------------------------

TEST( SymmetricDifference, SharedPayloadsAreNotDisturbed ) {
    std::mt19937_64 rng{ 0xC0DE };
    auto const all_forms{ forms( rng ) };
    for ( auto const & fa : all_forms ) {
        for ( auto const & fb : all_forms ) {
            SCOPED_TRACE( fa.name + " ^ " + fb.name );
            auto const original{ make_form<CowBitmap32>( fa, 0x10000 ) };
            auto const operand{ make_form<CowBitmap32>( fb, 0x10000 ) };
            auto const va{ in_chunk<std::uint32_t>( fa.lows, 0x10000 ) };
            auto const vb{ in_chunk<std::uint32_t>( fb.lows, 0x10000 ) };
            auto copy{ original };
            auto operand_copy{ operand };
            copy ^= operand_copy;
            ASSERT_TRUE( matches( copy, set_symmetric_difference_of( va, vb ), true ) );
            ASSERT_TRUE( matches( original, va ) );
            ASSERT_TRUE( matches( operand_copy, vb ) );
            ASSERT_TRUE( matches( operand, vb ) );
            // the result of the materializing form does not alias either operand
            auto result{ original ^ operand };
            result ^= result;
            ASSERT_TRUE( result.empty() );
            ASSERT_TRUE( matches( original, va ) );
            // an operand that shares its payloads with the left side
            auto shared{ original };
            shared ^= original;
            ASSERT_TRUE( shared.empty() );
            ASSERT_TRUE( matches( original, va ) );
        }
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

// ---- edges ---------------------------------------------------------------------------------------------------------

template <typename Key>
[[nodiscard]] std::vector<Key> to_values( std::uint64_t const first, std::uint64_t const last ) {
    std::vector<Key> values;
    for ( auto v{ first }; v <= last; ++v ) { values.push_back( static_cast<Key>( v ) ); }
    return values;
}

template <typename Bitmap>
void edge_cases() {
    using key_type = typename Bitmap::key_type;
    constexpr auto max{ std::numeric_limits<key_type>::max() };
    Bitmap const empty;
    Bitmap one;
    (void)one.add( 7 );

    EXPECT_TRUE( ( empty ^ empty ).empty() );
    EXPECT_TRUE( matches( empty ^ one, { 7 } ) );
    EXPECT_TRUE( matches( one ^ empty, { 7 } ) );
    {
        auto c{ empty };
        c ^= one;
        EXPECT_TRUE( matches( c, { 7 } ) );
        c ^= empty;
        EXPECT_TRUE( matches( c, { 7 } ) );
        c ^= one;
        EXPECT_TRUE( c.empty() );
        EXPECT_EQ( c.validate(), nullptr );
    }
    EXPECT_EQ( empty.xor_cardinality( one ), 1U );
    EXPECT_EQ( one.xor_cardinality( empty ), 1U );
    EXPECT_EQ( empty.xor_cardinality( empty ), 0U );

    // the extreme keys, and the keys on both sides of a chunk boundary
    Bitmap ends;
    (void)ends.add( 0 );
    (void)ends.add( max );
    Bitmap other;
    (void)other.add( 0 );
    (void)other.add( 1 );
    EXPECT_TRUE( matches( ends ^ other, { 1, max } ) );
    if constexpr ( sizeof( key_type ) > 2 ) {
        Bitmap lower;
        Bitmap upper;
        lower.add_closed_range( 65'000, 65'535 );
        upper.add_closed_range( 65'536, 66'000 );
        auto both{ lower ^ upper };
        EXPECT_EQ( both.size(), 536U + 465U );
        EXPECT_EQ( both.container_count(), 2U );
        EXPECT_EQ( both.validate(), nullptr );
        both ^= upper;
        EXPECT_TRUE( matches( both, to_values<key_type>( 65'000, 65'535 ) ) );
        EXPECT_EQ( both.container_count(), 1U );
    }

    // two full chunks cancel, and a full chunk with a hole leaves the hole
    Bitmap full_a;
    Bitmap full_b;
    full_a.add_closed_range( 0, 65'535 );
    full_b.add_closed_range( 0, 65'535 );
    EXPECT_TRUE( ( full_a ^ full_b ).empty() );
    full_b.remove( 4242 );
    EXPECT_TRUE( matches( full_a ^ full_b, { 4242 } ) );
    {
        auto c{ full_a };
        c ^= full_b;
        EXPECT_TRUE( matches( c, { 4242 } ) );
        EXPECT_EQ( c.container_count(), 1U );
    }
    // the complement inside a chunk
    Bitmap evens;
    Bitmap odds;
    for ( std::uint32_t v{ 0 }; v < 65'535; v += 2 ) { (void)evens.add( static_cast<key_type>( v ) ); (void)odds.add( static_cast<key_type>( v + 1 ) ); }
    EXPECT_EQ( ( evens ^ odds ).size(), 65'536U );
    EXPECT_EQ( ( evens ^ odds ).validate(), nullptr );
}

TEST( SymmetricDifference, EdgeCases16 ) { edge_cases<Bitmap16>(); }
TEST( SymmetricDifference, EdgeCases32 ) { edge_cases<Bitmap32>(); }
TEST( SymmetricDifference, EdgeCases64 ) { edge_cases<Bitmap64>(); }
TEST( SymmetricDifference, EdgeCasesCow32 ) { edge_cases<CowBitmap32>(); }

// ---- chunk tables ---------------------------------------------------------------------------------------------------

// Bitmaps over many chunks, some shared, some only on one side, some equal (and so cancelling); the result is
// compared with the oracle, and its chunk table must not hold an empty chunk.
template <typename Bitmap>
void many_chunks( std::uint64_t const seed, std::size_t const chunks_per_side, std::uint64_t const key_span ) {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ seed };
    for ( int trial{ 0 }; trial < 40; ++trial ) {
        std::vector<key_type> va;
        std::vector<key_type> vb;
        for ( std::size_t c{ 0 }; c < chunks_per_side; ++c ) {
            auto const chunk{ static_cast<key_type>( ( rng() % key_span ) << 16 ) };
            auto const lows{ make_pattern( all_patterns[ rng() % all_patterns.size() ], std::size_t{ 1 } + rng() % 600, rng ) };
            auto const values{ in_chunk<key_type>( lows, chunk ) };
            auto const where{ rng() % 4 };
            if ( where != 1 ) { va.insert( va.end(), values.begin(), values.end() ); }
            if ( where != 0 ) { vb.insert( vb.end(), values.begin(), values.end() ); }   // 2 and 3: equal chunks
            if ( where == 3 && !vb.empty() ) {   // and a near-equal one
                vb.pop_back();
            }
        }
        va = sorted_unique( std::move( va ) );
        vb = sorted_unique( std::move( vb ) );
        Bitmap a;
        Bitmap b;
        for ( auto const v : std::views::reverse( va ) ) { (void)a.add( v ); }   // out of order: a deferred sort in a wide table
        b.add_many( vb );
        if ( trial % 3 == 0 ) { a.optimize(); }
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " " << va.size() << " ^ " << vb.size() );
        auto const expected{ set_symmetric_difference_of( va, vb ) };
        auto const result{ a ^ b };
        ASSERT_TRUE( matches( result, expected, true ) );
        ASSERT_TRUE( matches( b ^ a, expected, true ) );
        EXPECT_EQ( result.container_count(), result.statistics().container_count() );
        auto c{ a };
        c ^= b;
        ASSERT_TRUE( matches( c, expected, true ) );
        auto d{ b };
        d ^= a;
        ASSERT_TRUE( matches( d, expected, true ) );
        EXPECT_EQ( a.xor_cardinality( b ), expected.size() );
    }
}

TEST( SymmetricDifference, ManyChunks32Dense )  { many_chunks<Bitmap32>( 1, 60, 80 ); }
TEST( SymmetricDifference, ManyChunks32Sparse ) { many_chunks<Bitmap32>( 2, 30, 60'000 ); }
TEST( SymmetricDifference, ManyChunks64Dense )  { many_chunks<Bitmap64>( 3, 60, 80 ); }
TEST( SymmetricDifference, ManyChunks64Sparse ) { many_chunks<Bitmap64>( 4, 30, 1ULL << 40 ); }
TEST( SymmetricDifference, ManyChunks64LargeTables ) { many_chunks<Bitmap64>( 5, 2500, 4000 ); }
TEST( SymmetricDifference, ManyChunksOverSharedPayloads ) { many_chunks<CowBitmap32>( 6, 60, 80 ); }

// A few chunks of one side against a large table of the other, in both orders: the walks that find the other's
// keys by searching rather than by stepping.
TEST( SymmetricDifference, FewChunksAgainstMany ) {
    std::mt19937_64 rng{ 77 };
    for ( int trial{ 0 }; trial < 30; ++trial ) {
        std::vector<std::uint64_t> big;
        std::vector<std::uint64_t> small;
        for ( std::uint64_t c{ 0 }; c < 1500; ++c ) {
            auto const lows{ in_chunk<std::uint64_t>( random_lows( 1 + rng() % 40, rng ), c << 16 ) };
            big.insert( big.end(), lows.begin(), lows.end() );
        }
        for ( int i{ 0 }; i < 1 + trial % 5; ++i ) {
            auto const chunk{ rng() % 1700 };
            auto const lows{ in_chunk<std::uint64_t>( random_lows( 1 + rng() % 60, rng ), chunk << 16 ) };
            small.insert( small.end(), lows.begin(), lows.end() );
            if ( rng() % 2 ) { small.insert( small.end(), big.begin() + static_cast<std::ptrdiff_t>( chunk % 1000 ), big.begin() + static_cast<std::ptrdiff_t>( chunk % 1000 + 30 ) ); }
        }
        big = sorted_unique( std::move( big ) );
        small = sorted_unique( std::move( small ) );
        Bitmap64 a;
        a.add_many( big );
        Bitmap64 b;
        b.add_many( small );
        auto const expected{ set_symmetric_difference_of( big, small ) };
        auto c{ a };
        c ^= b;
        ASSERT_TRUE( matches( c, expected, true ) );
        auto d{ b };
        d ^= a;
        ASSERT_TRUE( matches( d, expected, true ) );
        ASSERT_TRUE( matches( a ^ b, expected, true ) );
    }
}

// ---- the N-way fold -------------------------------------------------------------------------------------------------

template <typename Bitmap>
void xor_fold() {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0xF01D };
    for ( int trial{ 0 }; trial < 60; ++trial ) {
        auto const count{ 1 + rng() % 9 };
        std::vector<Bitmap> operands;
        std::vector<std::vector<key_type>> values;
        for ( std::size_t i{ 0 }; i < count; ++i ) {
            std::vector<key_type> v;
            for ( std::uint32_t chunk{ 0 }; chunk < 1 + rng() % 4; ++chunk ) {
                std::size_t const sizes[]{ 5, 40, 700, 3000, 9000 };
                auto const n{ sizes[ rng() % std::size( sizes ) ] };
                auto const lows{ n > 4096 ? dense_lows( n, rng ) : random_lows( n, rng ) };
                auto const piece{ in_chunk<key_type>( lows, static_cast<key_type>( ( rng() % 5 ) << 16 ) ) };
                v.insert( v.end(), piece.begin(), piece.end() );
            }
            v = sorted_unique( std::move( v ) );
            Bitmap bm;
            bm.add_many( v );
            if ( rng() % 4 == 0 ) { bm.optimize(); }
            values.push_back( std::move( v ) );
            operands.push_back( std::move( bm ) );
        }
        std::vector<key_type> expected;
        for ( auto const & v : values ) { expected = set_symmetric_difference_of( expected, v ); }
        std::vector<Bitmap const *> pointers;
        for ( auto const & bm : operands ) { pointers.push_back( &bm ); }
        pointers.push_back( nullptr );   // skipped
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " of " << count );

        auto folded{ Bitmap::xor_many( pointers ) };
        ASSERT_TRUE( matches( folded, expected, true ) );
        Bitmap accumulator;
        accumulator.add_many( values.front() );
        accumulator.xor_many_in_place( std::span<Bitmap const * const>{ pointers }.subspan( 1 ) );
        ASSERT_TRUE( matches( accumulator, expected, true ) );
        // the fold of a bitmap with itself
        auto twice{ operands.front() };
        std::array<Bitmap const *, 2> same{ &operands.front(), &operands.front() };
        twice.xor_many_in_place( same );
        ASSERT_TRUE( matches( twice, values.front() ) );
    }
}

TEST( SymmetricDifference, FoldOfManyBitmaps32 ) { xor_fold<Bitmap32>(); }
TEST( SymmetricDifference, FoldOfManyBitmaps64 ) { xor_fold<Bitmap64>(); }
TEST( SymmetricDifference, FoldOfManyBitmapsOverSharedPayloads ) { xor_fold<CowBitmap32>(); }

// ---- the operations against operands that are not bitmaps -----------------------------------------------------------------

TEST( SymmetricDifference, FlipOfAnIntervalMatchesFlipClosedRange ) {
    std::mt19937_64 rng{ 99 };
    auto const sources{ flip_sources( rng ) };
    for ( auto const & source : sources ) {
        for ( auto const [first, last] : flip_ranges<std::uint32_t>( 0x20000 ) ) {
            SCOPED_TRACE( source.name + " flipped over [" + std::to_string( first ) + ", " + std::to_string( last ) + "]" );
            Bitmap32 a;
            a.add_many( in_chunk<std::uint32_t>( source.lows, 0x20000 ) );
            if ( source.name.starts_with( "runs" ) ) { a.optimize(); }
            auto expected{ a };
            expected.flip_closed_range( first, last );
            a.xor_with( key_range::closed( first, last ) );
            ASSERT_EQ( a.validate( true ), nullptr );
            ASSERT_TRUE( a.to_vector() == expected.to_vector() );
            ASSERT_EQ( a.size(), expected.size() );
        }
    }
}

TEST( SymmetricDifference, IntervalsOverEmptyAndFullTables ) {
    Bitmap32 bm;
    bm.xor_with( key_range::closed( 100, 70'000 ) );
    EXPECT_EQ( bm.size(), 69'901U );
    EXPECT_EQ( bm.container_count(), 2U );
    EXPECT_EQ( bm.validate( true ), nullptr );
    bm.xor_with( key_range::closed( 100, 70'000 ) );
    EXPECT_TRUE( bm.empty() );
    EXPECT_EQ( bm.container_count(), 0U );
    bm.xor_with( key_range::all() );
    EXPECT_EQ( bm.size(), std::uint64_t{ 1 } << 32 );
    EXPECT_EQ( bm.validate( true ), nullptr );
    bm.xor_with( key_range::from( 5 ) );
    EXPECT_TRUE( matches( bm, { 0, 1, 2, 3, 4 } ) );
    bm.xor_with( key_range{} );
    EXPECT_EQ( bm.size(), 5U );
    // a key beyond the key type's range is not a member
    Bitmap16 small;
    small.xor_with( key_range::closed( 65'000, 70'000 ) );
    EXPECT_EQ( small.size(), 536U );
    EXPECT_EQ( small.xor_cardinality( key_range::all() ), 65'536U - 536U );
}

TEST( SymmetricDifference, AgainstKeysAndBitStrings ) {
    std::mt19937_64 rng{ 5150 };
    for ( int trial{ 0 }; trial < 40; ++trial ) {
        std::vector<std::uint32_t> content;
        for ( std::uint32_t chunk{ 0 }; chunk < 4; ++chunk ) {
            auto const lows{ in_chunk<std::uint32_t>( rng() % 2 ? random_lows( 3000, rng ) : dense_lows( 30'000, rng ), chunk << 16 ) };
            content.insert( content.end(), lows.begin(), lows.end() );
        }
        Bitmap32 a;
        a.add_many( content );
        std::vector<std::uint32_t> keys;
        for ( int i{ 0 }; i < 5000; ++i ) { keys.push_back( static_cast<std::uint32_t>( rng() % 300'000 ) ); }
        auto sorted{ sorted_unique( keys ) };
        auto const expected{ set_symmetric_difference_of( content, sorted ) };
        {
            auto c{ a };
            c.xor_with( std::span<std::uint32_t const>{ keys } );   // unsorted, with repeats
            ASSERT_TRUE( matches( c, expected, true ) );
        }
        {
            auto c{ a };
            c ^= sorted_keys{ sorted };
            ASSERT_TRUE( matches( c, expected, true ) );
        }
        EXPECT_EQ( a.xor_cardinality( std::span<std::uint32_t const>{ keys } ), expected.size() );
        std::vector<std::uint64_t> words( 2000 );
        std::vector<std::uint32_t> in_words;
        for ( std::uint32_t bit{ 0 }; bit < 128'000; ++bit ) {
            if ( rng() % 3 == 0 ) { words[ bit / 64 ] |= std::uint64_t{ 1 } << ( bit % 64 ); in_words.push_back( 70'000 + bit ); }
        }
        auto c{ a };
        c.xor_with( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ words }, 70'000 } );
        ASSERT_TRUE( matches( c, set_symmetric_difference_of( content, in_words ), true ) );
    }
}

// ---- allocation failure ------------------------------------------------------------------------------------------------

// A set operation that runs out of memory leaves the bitmaps valid: an operand is never changed, and the left side
// of an in-place exclusive-or is a legal bitmap (its contents are only defined when the call returned).
TEST( SymmetricDifference, AnAllocationFailureLeavesValidBitmaps ) {
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    std::mt19937_64 rng{ 31 };
    std::vector<Bitmap32> operands;
    std::vector<std::vector<std::uint32_t>> values;
    for ( int i{ 0 }; i < 6; ++i ) {
        std::vector<std::uint32_t> v;
        for ( std::uint32_t chunk{ 0 }; chunk < 40; ++chunk ) {
            std::size_t const sizes[]{ 20, 500, 3000, 9000 };
            auto const n{ sizes[ rng() % std::size( sizes ) ] };
            auto const piece{ in_chunk<std::uint32_t>( n > 4096 ? dense_lows( n, rng ) : random_lows( n, rng ), ( chunk * 2 + rng() % 2 ) << 16 ) };
            v.insert( v.end(), piece.begin(), piece.end() );
        }
        v = sorted_unique( std::move( v ) );
        Bitmap32 bm;
        bm.add_many( v );
        operands.push_back( std::move( bm ) );
        values.push_back( std::move( v ) );
    }
    std::uint32_t failed{ 0 };
    for ( std::uint32_t const period : { 2U, 3U, 7U, 19U, 101U } ) {
        for ( std::size_t i{ 0 }; i < operands.size(); ++i ) {
            for ( std::size_t j{ 0 }; j < operands.size(); ++j ) {
                bool ok{ true };
                auto copy{ operands[ i ] };
                try {
                    alloc_fault::armed const armed{ period };
                    copy ^= operands[ j ];
                } catch ( std::bad_alloc const & ) {
                    ok = false;
                    ++failed;
                }
                ASSERT_EQ( copy.validate(), nullptr ) << copy.validate();
                if ( ok ) { ASSERT_TRUE( matches( copy, set_symmetric_difference_of( values[ i ], values[ j ] ) ) ); }
                ASSERT_TRUE( matches( operands[ j ], values[ j ] ) );
                try {
                    alloc_fault::armed const armed{ period };
                    auto result{ operands[ i ] ^ operands[ j ] };
                    alloc_fault::period = 0;
                    ASSERT_TRUE( matches( result, set_symmetric_difference_of( values[ i ], values[ j ] ) ) );
                } catch ( std::bad_alloc const & ) {
                    ++failed;
                }
                ASSERT_TRUE( matches( operands[ i ], values[ i ] ) );
            }
        }
        std::vector<Bitmap32 const *> pointers;
        for ( auto const & bm : operands ) { pointers.push_back( &bm ); }
        Bitmap32 folded;
        try {
            alloc_fault::armed const armed{ period };
            folded.xor_many_in_place( pointers );
        } catch ( std::bad_alloc const & ) {
            ++failed;
        }
        ASSERT_EQ( folded.validate(), nullptr ) << folded.validate();
    }
    EXPECT_GT( failed, 0U );
#endif
}

} // namespace
