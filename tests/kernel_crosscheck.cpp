// The container kernels cross-checked against CRoaring by contents: both libraries
// are fed the same values, the same operation is applied, and the resulting value
// sets must be equal. Container kinds are not compared (each library picks its
// own); the std:: oracle tests live in kernel_coverage.cpp.

#include "kernel_test_support.hpp"

#include <roaring/roaring.h>

#include <bit>
#include <set>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

using Bitmap32 = bitmap<std::uint32_t>;
using Values32 = std::vector<std::uint32_t>;

struct croaring {
    croaring() : r{ roaring_bitmap_create() } {}
    explicit croaring( Values32 const & values ) : r{ roaring_bitmap_of_ptr( values.size(), values.data() ) } {}
    explicit croaring( roaring_bitmap_t * const raw ) noexcept : r{ raw } {}
    croaring( croaring const & other ) : r{ roaring_bitmap_copy( other.r ) } {}
    croaring & operator=( croaring const & ) = delete;
    ~croaring() { roaring_bitmap_free( r ); }

    [[nodiscard]] Values32 values() const {
        Values32 out( roaring_bitmap_get_cardinality( r ) );
        roaring_bitmap_to_uint32_array( r, out.data() );
        return out;
    }

    roaring_bitmap_t * r;
};

// `ours` holds exactly the set CRoaring holds.
[[nodiscard]] ::testing::AssertionResult same_contents( Bitmap32 const & ours, croaring const & theirs ) {
    if ( auto const * const bad{ ours.validate() } ) { return ::testing::AssertionFailure() << "invariant violated: " << bad; }
    auto const mine{ ours.to_vector() };
    auto const other{ theirs.values() };
    if ( mine.size() != other.size() || !std::equal( mine.begin(), mine.end(), other.begin() ) ) {
        return ::testing::AssertionFailure() << "contents differ: ours " << mine.size() << " values, CRoaring " << other.size();
    }
    return ::testing::AssertionSuccess();
}

[[nodiscard]] Bitmap32 make_ours( Values32 const & values, bool const run_optimize ) {
    Bitmap32 bm;
    bm.add_many( values );
    if ( run_optimize ) { bm.optimize(); }
    return bm;
}

[[nodiscard]] croaring make_theirs( Values32 const & values, bool const run_optimize ) {
    croaring r{ values };
    if ( run_optimize ) { roaring_bitmap_run_optimize( r.r ); }
    return r;
}

constexpr std::size_t array_sizes[]{ 0, 1, 2, 7, 16, 31, 32, 33, 64, 100, 255, 256, 257, 1000, 2047, 2048, 3000, 4096 };

TEST( KernelCrosscheck, ArrayArrayOperations ) {
    std::mt19937_64 rng{ 0xC0FFEE };
    for ( int trial{ 0 }; trial < 1500; ++trial ) {
        auto const pa{ all_patterns[ rng() % all_patterns.size() ] };
        auto const pb{ all_patterns[ rng() % all_patterns.size() ] };
        auto const va{ in_chunk<std::uint32_t>( make_pattern( pa, array_sizes[ rng() % std::size( array_sizes ) ], rng ), 0x30000U ) };
        auto const vb{ in_chunk<std::uint32_t>( make_pattern( pb, array_sizes[ rng() % std::size( array_sizes ) ], rng ), 0x30000U ) };
        bool const run_optimize{ rng() % 4 == 0 };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " a=" << name_of( pa ) << '/' << va.size() << " b=" << name_of( pb ) << '/' << vb.size() << " runs " << run_optimize );
        auto const a{ make_ours( va, run_optimize ) };
        auto const b{ make_ours( vb, run_optimize ) };
        auto const ca{ make_theirs( va, run_optimize ) };
        auto const cb{ make_theirs( vb, run_optimize ) };

        EXPECT_TRUE( same_contents( a | b, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or";
        { auto t{ a }; t |= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or inplace"; }
        EXPECT_TRUE( same_contents( a & b, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ) << "and";
        EXPECT_EQ( a.intersects( b ), roaring_bitmap_intersect( ca.r, cb.r ) );
        EXPECT_TRUE( same_contents( a - b, croaring{ roaring_bitmap_andnot( ca.r, cb.r ) } ) ) << "andnot";
        { auto t{ a }; t -= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_andnot( ca.r, cb.r ) } ) ) << "andnot inplace"; }
    }
}

TEST( KernelCrosscheck, FlipSparseArray ) {
    std::mt19937_64 rng{ 0xF11B };
    constexpr std::uint32_t base{ 3U << 16 };
    struct range_t { std::uint32_t first, last; };
    constexpr range_t ranges[]{
        { base, base + 65535 }, { base + 5, base + 65530 }, { base + 1000, base + 20000 }, { base + 777, base + 777 },
        { base - 10, base + 65535 + 10 }, { base + 100, base + 2 * 65536 + 500 }, { base - 65536, base + 65535 + 65536 },
    };
    for ( std::size_t const cardinality : { 1, 2, 100, 2045, 2046, 2047, 3000, 4096 } ) {
        for ( auto const pat : { pattern::random, pattern::one_run, pattern::every_third } ) {
            auto const values{ in_chunk<std::uint32_t>( make_pattern( pat, cardinality, rng ), base ) };
            for ( auto const & r : ranges ) {
                for ( bool const run_optimize : { false, true } ) {
                    SCOPED_TRACE( ::testing::Message() << "card " << cardinality << ' ' << name_of( pat ) << " range " << r.first << ".." << r.last << " runs " << run_optimize );
                    auto ours{ make_ours( values, run_optimize ) };
                    auto theirs{ make_theirs( values, run_optimize ) };
                    ours.flip_closed_range( r.first, r.last );
                    roaring_bitmap_flip_inplace_closed( theirs.r, r.first, r.last );
                    ASSERT_TRUE( same_contents( ours, theirs ) );
                }
            }
        }
    }
}

TEST( KernelCrosscheck, FlipAcrossForms ) {
    std::mt19937_64 rng{ 0xF1195 };
    constexpr std::uint32_t base{ 3U << 16 };
    auto const ranges{ flip_ranges<std::uint32_t>( base ) };
    for ( auto const & source : flip_sources( rng ) ) {
        auto const values{ in_chunk<std::uint32_t>( source.lows, base ) };
        for ( bool const run_optimize : { false, true } ) {
            for ( auto const & [first, last] : ranges ) {
                SCOPED_TRACE( ::testing::Message() << source.name << " range " << first - base << ".." << last - base << " runs " << run_optimize );
                auto ours{ make_ours( values, run_optimize ) };
                auto theirs{ make_theirs( values, run_optimize ) };
                ours.flip_closed_range( first, last );
                roaring_bitmap_flip_inplace_closed( theirs.r, first, last );
                ASSERT_TRUE( same_contents( ours, theirs ) );
                ASSERT_EQ( ours.validate( false ), nullptr );
            }
        }
    }
}

TEST( KernelCrosscheck, BitsetAndBitsetAcrossDensities ) {
    std::mt19937_64 rng{ 0xB175E7 };
    auto const expand{ []( auto const & words ) {
        Values32 v;
        for ( std::size_t i{ 0 }; i < words.size(); ++i ) {
            for ( auto w{ words[ i ] }; w != 0; w &= w - 1 ) { v.push_back( 0x20000U + static_cast<std::uint32_t>( i * 64 + std::countr_zero( w ) ) ); }
        }
        return v;
    } };
    for ( bool const clustered : { false, true } ) {
        for ( unsigned bits{ 0 }; bits <= 64; ++bits ) {
            SCOPED_TRACE( ::testing::Message() << ( clustered ? "clustered" : "spread" ) << " bits/word " << bits );
            std::array<std::uint64_t, 1024> wa{}, wb{};
            auto const cluster_words{ ( bits * 1024U + 32U ) / 64U };
            for ( std::size_t i{ 0 }; i < 1024; ++i ) {
                std::uint64_t shared{ 0 };
                if ( clustered ) {
                    shared = i < cluster_words ? ~std::uint64_t{ 0 } : 0;
                } else {
                    while ( static_cast<unsigned>( std::popcount( shared ) ) < bits ) { shared |= std::uint64_t{ 1 } << ( rng() % 64 ); }
                }
                auto const split{ rng() };
                wa[ i ] = shared | ( ~shared & split );
                wb[ i ] = shared | ( ~shared & ~split & rng() );
            }
            auto const va{ expand( wa ) }, vb{ expand( wb ) };
            auto const a{ make_ours( va, false ) }, b{ make_ours( vb, false ) };
            auto const ca{ make_theirs( va, false ) }, cb{ make_theirs( vb, false ) };
            EXPECT_TRUE( same_contents( a & b, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) );
            { auto t{ a }; t &= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ); }
            EXPECT_EQ( a.intersects( b ), roaring_bitmap_intersect( ca.r, cb.r ) );
        }
    }
}

// Interleaved, blocky and nested value sets, whose merges exercise the block shortcuts of the scalar kernels,
// and unions whose total crosses the array limit.
TEST( KernelCrosscheck, ArrayArrayOperationsInterleavedAndBlocky ) {
    std::mt19937_64 rng{ 0x1417E8 };
    constexpr std::uint32_t base{ 9U << 16 };
    auto const blocky{ [ & ]( std::size_t const n, unsigned const stride ) {
        std::set<std::uint32_t> chosen;
        while ( chosen.size() < n ) {
            auto const start{ static_cast<std::uint32_t>( rng() % ( 65536 - 16 * stride ) ) };
            auto const length{ 1 + rng() % 9 };
            for ( std::uint32_t i{ 0 }; i < length && chosen.size() < n; ++i ) { chosen.insert( base + start + i * stride ); }
        }
        return Values32{ chosen.begin(), chosen.end() };
    } };
    constexpr std::size_t sizes[]{ 4, 9, 33, 100, 500, 2000, 3000, 4000 };
    for ( int trial{ 0 }; trial < 600; ++trial ) {
        auto const na{ sizes[ rng() % std::size( sizes ) ] };
        auto const nb{ sizes[ rng() % std::size( sizes ) ] };
        auto const stride_a{ 1U + static_cast<unsigned>( rng() % 3 ) };
        auto const stride_b{ 1U + static_cast<unsigned>( rng() % 3 ) };
        auto va{ blocky( na, stride_a ) };
        auto vb{ blocky( nb, stride_b ) };
        if ( rng() % 3 == 0 ) { vb.insert( vb.end(), va.begin(), va.begin() + static_cast<std::ptrdiff_t>( va.size() / 2 ) ); vb = sorted_unique( std::move( vb ) ); }
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " a " << va.size() << " b " << vb.size() );
        auto const a{ make_ours( va, false ) };
        auto const b{ make_ours( vb, false ) };
        auto const ca{ make_theirs( va, false ) };
        auto const cb{ make_theirs( vb, false ) };

        EXPECT_TRUE( same_contents( a | b, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or";
        { auto t{ a }; t |= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or inplace"; }
        { auto t{ b }; t |= a; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or inplace swapped"; }
        EXPECT_TRUE( same_contents( a & b, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ) << "and";
        { auto t{ a }; t &= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ) << "and inplace"; }
        EXPECT_TRUE( same_contents( a - b, croaring{ roaring_bitmap_andnot( ca.r, cb.r ) } ) ) << "andnot";
        { auto t{ a }; t -= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_andnot( ca.r, cb.r ) } ) ) << "andnot inplace"; }
        EXPECT_EQ( a.intersects( b ), roaring_bitmap_intersect( ca.r, cb.r ) );
    }
}

// Bitset operands whose words differ wildly in density: empty stretches, full words, sparse and dense words in
// one block, so the operands settle as bitsets or arrays and the AND's result runs from nothing to a bitset.
TEST( KernelCrosscheck, BitsetAndBitsetMixedWordDensities ) {
    std::mt19937_64 rng{ 0x31DD };
    for ( int trial{ 0 }; trial < 200; ++trial ) {
        std::array<std::uint64_t, 1024> wa{}, wb{};
        auto const thinning{ static_cast<unsigned>( rng() % 9 ) };
        auto const empty_one_in{ 1U + static_cast<unsigned>( rng() % 12 ) };
        for ( std::size_t i{ 0 }; i < 1024; ++i ) {
            if ( rng() % empty_one_in == 0 ) { continue; }
            wa[ i ] = rng() | rng() | rng();
            wb[ i ] = rng();
            for ( unsigned k{ 0 }; k < thinning; ++k ) { wb[ i ] &= rng(); }
        }
        auto const expand{ [ & ]( auto const & words ) {
            Values32 v;
            for ( std::size_t i{ 0 }; i < words.size(); ++i ) {
                for ( auto w{ words[ i ] }; w != 0; w &= w - 1 ) { v.push_back( 0x40000U + static_cast<std::uint32_t>( i * 64 + std::countr_zero( w ) ) ); }
            }
            return v;
        } };
        auto const va{ expand( wa ) }, vb{ expand( wb ) };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " thinning " << thinning << " empty 1/" << empty_one_in );
        auto const a{ make_ours( va, false ) }, b{ make_ours( vb, false ) };
        auto const ca{ make_theirs( va, false ) }, cb{ make_theirs( vb, false ) };
        EXPECT_TRUE( same_contents( a & b, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ) << "and";
        { auto t{ a }; t &= b; EXPECT_TRUE( same_contents( t, croaring{ roaring_bitmap_and( ca.r, cb.r ) } ) ) << "and inplace"; }
        EXPECT_TRUE( same_contents( a - b, croaring{ roaring_bitmap_andnot( ca.r, cb.r ) } ) ) << "andnot";
        EXPECT_TRUE( same_contents( a | b, croaring{ roaring_bitmap_or( ca.r, cb.r ) } ) ) << "or";
        EXPECT_EQ( a.intersects( b ), roaring_bitmap_intersect( ca.r, cb.r ) );
    }
}

TEST( KernelCrosscheck, SmallArrayExport ) {
    std::mt19937_64 rng{ 0xE5B027 };
    for ( std::uint32_t const base : { 0U, 0xFFFF0000U, 0x00010000U } ) {
        for ( std::size_t n{ 0 }; n <= 80; ++n ) {
            auto const values{ in_chunk<std::uint32_t>( make_pattern( pattern::random, n, rng ), base ) };
            auto const ours{ make_ours( values, false ) };
            auto const theirs{ make_theirs( values, false ) };
            std::vector<std::uint32_t> out( n + 2, 0xDEADBEEFU );
            ASSERT_EQ( ours.to_array_into( std::span{ out.data() + 1, n } ), n );
            auto const reference{ theirs.values() };
            ASSERT_TRUE( std::equal( reference.begin(), reference.end(), out.begin() + 1 ) ) << "base " << base << " n " << n;
            ASSERT_EQ( out.front(), 0xDEADBEEFU );
            ASSERT_EQ( out[ n + 1 ], 0xDEADBEEFU );
        }
    }
}

// A big target |= a few keys, against CRoaring's in-place union.
TEST( KernelCrosscheck, BigByFewUnion ) {
    std::mt19937_64 rng{ 0xB16B };
    Values32 target_values;
    for ( std::uint32_t i{ 0 }; i < 1000; ++i ) {
        auto const base{ ( 10U + 2 * i ) << 16 };
        switch ( i % 4 ) {
            case 0: for ( auto const low : make_pattern( pattern::random, 100, rng ) ) { target_values.push_back( base + low ); } break;
            case 1: for ( auto const low : make_pattern( pattern::random, 6000, rng ) ) { target_values.push_back( base + low ); } break;
            case 2: for ( std::uint32_t r{ 0 }; r < 5; ++r ) { for ( std::uint32_t k{ 0 }; k < 150; ++k ) { target_values.push_back( base + r * 5000 + k ); } } break;
            default: for ( std::uint32_t k{ 0 }; k < ( i % 40 == 3 ? 65536U : 2000U ); ++k ) { target_values.push_back( base + ( i % 40 == 3 ? 0U : 1000U ) + k ); } break;
        }
    }
    for ( std::size_t const keys : { 1, 4, 16, 20 } ) {
        for ( int const where : { 0, 1, 2, 3 } ) { // existing keys, between, before, after
            Values32 few;
            for ( std::size_t k{ 0 }; k < keys; ++k ) {
                std::uint32_t key{};
                switch ( where ) {
                    case 0: key = 10U + 2 * static_cast<std::uint32_t>( rng() % 1000 ); break;
                    case 1: key = 11U + 2 * static_cast<std::uint32_t>( rng() % 1000 ); break;
                    case 2: key = static_cast<std::uint32_t>( rng() % 10 ); break;
                    default: key = 2010U + static_cast<std::uint32_t>( rng() % 50 ); break;
                }
                for ( auto const low : make_pattern( k % 2 ? pattern::random : pattern::one_run, k % 2 ? 3000 : 70, rng ) ) { few.push_back( ( key << 16 ) + low ); }
            }
            few = sorted_unique( std::move( few ) );
            for ( bool const run_optimize : { false, true } ) {
                SCOPED_TRACE( ::testing::Message() << keys << " keys, where " << where << " runs " << run_optimize );
                auto ours{ make_ours( target_values, run_optimize ) };
                auto theirs{ make_theirs( target_values, run_optimize ) };
                ours |= make_ours( few, false );
                roaring_bitmap_or_inplace( theirs.r, make_theirs( few, false ).r );
                ASSERT_TRUE( same_contents( ours, theirs ) );
            }
        }
    }
}

// ---- a bitmap with few chunks |= one with many ---------------------------------------------------

// The values of one chunk, in one of several container shapes (array, bitset, a few runs, a full chunk).
template <typename Key>
[[nodiscard]] std::vector<Key> chunk_values( Key const chunk, std::size_t const shape, std::mt19937_64 & rng ) {
    std::vector<Key> out;
    auto const base{ static_cast<Key>( chunk << 16 ) };
    switch ( shape % 4 ) {
        case 0: for ( auto const low : make_pattern( pattern::random, 40, rng ) )   { out.push_back( base + low ); } break;
        case 1: for ( auto const low : make_pattern( pattern::random, 5000, rng ) ) { out.push_back( base + low ); } break;
        case 2: for ( std::uint32_t r{ 0 }; r < 5; ++r ) { for ( std::uint32_t k{ 0 }; k < 150; ++k ) { out.push_back( base + r * 5000 + k ); } } break;
        default: for ( std::uint32_t k{ 0 }; k < 65536; ++k ) { out.push_back( base + k ); } break;
    }
    return out;
}

template <typename Key>
void check_few_union() {
    using Bitmap = bitmap<Key>;
    // Wide keys sit far above 32 bits so that the tree's separators are exercised on high bits.
    constexpr Key origin{ sizeof( Key ) > 4 ? Key{ 0x5A5A } << 24 : Key{ 100 } };
    constexpr std::int64_t chunks{ 200 };
    std::mt19937_64 rng{ 0xFE77 + sizeof( Key ) };

    // Chunk key of slot s: origin + 10 + s. The target holds the even slots 0 .. 2 * chunks - 2.
    auto const key_of = [ & ]( std::int64_t const slot ) { return static_cast<Key>( origin + static_cast<Key>( 10 + slot ) ); };
    std::vector<Key> target_values;
    for ( std::int64_t i{ 0 }; i < chunks; ++i ) {
        for ( auto const v : chunk_values<Key>( key_of( 2 * i ), static_cast<std::size_t>( i ), rng ) ) { target_values.push_back( v ); }
    }
    Bitmap const target{ std::span<Key const>{ target_values } };
    ASSERT_EQ( target.size(), target_values.size() );

    struct operand_case { char const * name; std::vector<std::int64_t> slots; };
    std::vector<operand_case> const cases{
        { "one held",                  { 100 } },
        { "one between",               { 101 } },
        { "one below the range",       { -3 } },
        { "one above the range",       { 2 * chunks + 5 } },
        { "all held",                  { 20, 100, 398 } },
        { "all new, interleaved",      { 21, 201, 397 } },
        { "held and new interleaved",  { 20, 21, 22, 200, 201 } },
        { "first and last held",       { 0, 398 } },
        { "new at both ends",          { -4, -2, 399, 401 } },
        { "held, new, below, above",   { -6, 0, 1, 250, 251, 398, 400, 403 } },
        { "adjacent new keys",         { 101, 103, 105 } },
        { "new key in the first gap",  { 1 } },
    };
    for ( auto const & c : cases ) {
        for ( std::size_t const dense : { 0u, 1u } ) {
            SCOPED_TRACE( ::testing::Message() << c.name << ( dense ? ", dense operand" : ", sparse operand" ) );
            std::vector<Key> few_values;
            std::size_t shape{ dense ? 1u : 0u };
            for ( auto const slot : c.slots ) {
                for ( auto const v : chunk_values<Key>( key_of( slot ), shape, rng ) ) { few_values.push_back( v ); }
                shape += dense ? 2 : 1;
            }
            few_values = sorted_unique( std::move( few_values ) );
            Bitmap const few{ std::span<Key const>{ few_values } };

            std::set<Key> oracle{ target_values.begin(), target_values.end() };
            oracle.insert( few_values.begin(), few_values.end() );
            std::vector<Key> const expected{ oracle.begin(), oracle.end() };

            auto united{ target };
            united |= few;
            ASSERT_TRUE( matches( united, expected ) );
            EXPECT_TRUE( matches( few, few_values ) ) << "the operand is left alone";

            united |= few;   // repeating it changes nothing
            EXPECT_TRUE( matches( united, expected ) );

            auto flipped{ few };   // the other way round holds the same set
            flipped |= target;
            EXPECT_TRUE( matches( flipped, expected ) );
        }
    }

    // an empty operand, an empty target, a self union
    auto unchanged{ target };
    unchanged |= Bitmap{};
    EXPECT_TRUE( matches( unchanged, target_values ) );
    Bitmap empty;
    empty |= target;
    EXPECT_TRUE( matches( empty, target_values ) );
    auto self{ target };
    self |= self;
    EXPECT_TRUE( matches( self, target_values ) );
}

TEST( KernelCrosscheck, FewChunksUnionedIntoManyContents32 ) { check_few_union<std::uint32_t>(); }
TEST( KernelCrosscheck, FewChunksUnionedIntoManyContents64 ) { check_few_union<std::uint64_t>(); }

// The same against CRoaring: a target that grows over several rounds of few-chunk operands that land
// on held keys, between them, below and above.
TEST( KernelCrosscheck, FewChunksUnionedIntoManyRoundsMatchCRoaring ) {
    std::mt19937_64 rng{ 0xF3A5 };
    Values32 target_values;
    for ( std::uint32_t i{ 0 }; i < 300; ++i ) {
        for ( auto const v : chunk_values<std::uint32_t>( 5U + 2 * i, i, rng ) ) { target_values.push_back( v ); }
    }
    for ( bool const run_optimize : { false, true } ) {
        auto ours{ make_ours( target_values, run_optimize ) };
        auto theirs{ make_theirs( target_values, run_optimize ) };
        for ( int round{ 0 }; round < 40; ++round ) {
            Values32 few;
            auto const keys{ 1 + rng() % 8 };
            for ( std::uint64_t k{ 0 }; k < keys; ++k ) {
                auto const key{ static_cast<std::uint32_t>( rng() % 700 ) };
                for ( auto const v : chunk_values<std::uint32_t>( key, rng() % 4, rng ) ) { few.push_back( v ); }
            }
            few = sorted_unique( std::move( few ) );
            SCOPED_TRACE( ::testing::Message() << "round " << round << ", runs " << run_optimize );
            ours |= make_ours( few, run_optimize && round % 2 );
            roaring_bitmap_or_inplace( theirs.r, make_theirs( few, run_optimize && round % 2 ).r );
            ASSERT_TRUE( same_contents( ours, theirs ) );
        }
    }
}

// ---- a bitmap |= one of comparable size ----------------------------------------------------------

// Operands with the same chunk count or a few times fewer, so the in-place union walks both tables:
// every key held (no table change), the first key missing late in the walk, keys only below, only
// above, and only interleaved with the target's.
template <typename Key>
void check_comparable_union() {
    using Bitmap = bitmap<Key>;
    constexpr Key origin{ sizeof( Key ) > 4 ? Key{ 0x3C3C } << 24 : Key{ 7 } };
    constexpr std::int64_t chunks{ 64 };
    std::mt19937_64 rng{ 0xC0DA + sizeof( Key ) };
    auto const key_of = [ & ]( std::int64_t const slot ) { return static_cast<Key>( origin + static_cast<Key>( 10 + slot ) ); };

    std::vector<Key> target_values;
    for ( std::int64_t i{ 0 }; i < chunks; ++i ) {
        for ( auto const v : chunk_values<Key>( key_of( 2 * i ), static_cast<std::size_t>( i ), rng ) ) { target_values.push_back( v ); }
    }
    Bitmap const target{ std::span<Key const>{ target_values } };

    struct operand_case { char const * name; std::vector<std::int64_t> slots; };
    std::vector<operand_case> cases{
        { "every key held",              {} },
        { "every other key held",        {} },
        { "last key held",               { 2 * chunks - 2 } },
        { "held then new late",          {} },
        { "new key first",               { 1, 10, 20 } },
        { "all below",                   { -9, -8, -7, -5 } },
        { "all above",                   { 2 * chunks + 1, 2 * chunks + 4, 2 * chunks + 9 } },
        { "new keys only, interleaved",  { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33 } },
        { "held, then new tail",         { 0, 2, 4, 2 * chunks - 2, 2 * chunks, 2 * chunks + 2 } },
    };
    for ( std::int64_t i{ 0 }; i < chunks; ++i )      { cases[ 0 ].slots.push_back( 2 * i ); }
    for ( std::int64_t i{ 0 }; i < chunks; i += 2 )   { cases[ 1 ].slots.push_back( 2 * i ); }
    for ( std::int64_t i{ 0 }; i < chunks - 4; ++i )  { cases[ 3 ].slots.push_back( 2 * i ); }
    cases[ 3 ].slots.push_back( 2 * chunks - 7 );
    cases[ 3 ].slots.push_back( 2 * chunks + 3 );
    for ( auto const & c : cases ) {
        for ( std::size_t const dense : { 0u, 1u } ) {
            SCOPED_TRACE( ::testing::Message() << c.name << ( dense ? ", dense operand" : ", sparse operand" ) );
            std::vector<Key> other_values;
            std::size_t shape{ dense ? 1u : 0u };
            for ( auto const slot : c.slots ) {
                for ( auto const v : chunk_values<Key>( key_of( slot ), shape, rng ) ) { other_values.push_back( v ); }
                shape += dense ? 2 : 1;
            }
            other_values = sorted_unique( std::move( other_values ) );
            Bitmap const other{ std::span<Key const>{ other_values } };

            std::set<Key> oracle{ target_values.begin(), target_values.end() };
            oracle.insert( other_values.begin(), other_values.end() );
            std::vector<Key> const expected{ oracle.begin(), oracle.end() };

            auto united{ target };
            united |= other;
            ASSERT_TRUE( matches( united, expected ) );
            EXPECT_TRUE( matches( other, other_values ) ) << "the operand is left alone";
            EXPECT_EQ( united.size(), expected.size() );

            auto flipped{ other };
            flipped |= target;
            EXPECT_TRUE( matches( flipped, expected ) );
        }
    }
}

TEST( KernelCrosscheck, ComparableSizeUnionContents32 ) { check_comparable_union<std::uint32_t>(); }
TEST( KernelCrosscheck, ComparableSizeUnionContents64 ) { check_comparable_union<std::uint64_t>(); }

// The same against CRoaring over several rounds of operands that hold, miss and interleave with the
// target's keys.
TEST( KernelCrosscheck, ComparableSizeUnionRoundsMatchCRoaring ) {
    std::mt19937_64 rng{ 0xC0DB };
    Values32 target_values;
    for ( std::uint32_t i{ 0 }; i < 48; ++i ) {
        for ( auto const v : chunk_values<std::uint32_t>( 5U + 2 * i, i, rng ) ) { target_values.push_back( v ); }
    }
    for ( bool const run_optimize : { false, true } ) {
        auto ours{ make_ours( target_values, run_optimize ) };
        auto theirs{ make_theirs( target_values, run_optimize ) };
        for ( int round{ 0 }; round < 40; ++round ) {
            Values32 other;
            auto const keys{ 2 + rng() % 40 };
            std::set<std::uint32_t> chosen;
            for ( std::uint64_t k{ 0 }; k < keys; ++k ) { chosen.insert( static_cast<std::uint32_t>( rng() % 120 ) ); }
            for ( auto const key : chosen ) {
                for ( auto const v : chunk_values<std::uint32_t>( key, rng() % 4, rng ) ) { other.push_back( v ); }
            }
            other = sorted_unique( std::move( other ) );
            SCOPED_TRACE( ::testing::Message() << "round " << round << ", runs " << run_optimize );
            ours |= make_ours( other, run_optimize && round % 2 );
            roaring_bitmap_or_inplace( theirs.r, make_theirs( other, run_optimize && round % 2 ).r );
            ASSERT_TRUE( same_contents( ours, theirs ) );
        }
    }
}

// ---- run-encoded chunks of very different run counts ---------------------------------------------------
//
// Intersecting two run lists crosses the stretch of one list that lies below the other's current run in one
// step: lists of one run against thousands, of equal length, interleaved closely and far apart, in both
// orders and through the cardinality-only form.
TEST( KernelCrosscheck, RunListsOfDifferentLengthsIntersectLikeCRoaring ) {
    std::mt19937_64 rng{ 0x7E11 };
    auto const runs_of = [ & ]( std::uint32_t const count, std::uint32_t const max_length, std::uint32_t const max_gap ) {
        Values32 out;
        std::uint32_t at{ static_cast<std::uint32_t>( rng() % 200 ) };
        for ( std::uint32_t r{ 0 }; r < count; ++r ) {
            auto const length{ 1U + static_cast<std::uint32_t>( rng() % max_length ) };
            if ( at + length > 65536U ) { break; }
            for ( std::uint32_t k{ 0 }; k < length; ++k ) { out.push_back( ( 3U << 16 ) + at + k ); }
            at += length + 1U + static_cast<std::uint32_t>( rng() % max_gap );
        }
        return out;
    };
    for ( int trial{ 0 }; trial < 800; ++trial ) {
        static constexpr std::uint32_t counts[]{ 1, 2, 3, 40, 500, 3000 };
        auto const left_count { counts[ rng() % std::size( counts ) ] };
        auto const right_count{ counts[ rng() % std::size( counts ) ] };
        auto const left_values { runs_of( left_count,  1U + static_cast<std::uint32_t>( rng() % 30'000 / left_count ),  1U + static_cast<std::uint32_t>( rng() % 20 ) ) };
        auto const right_values{ runs_of( right_count, 1U + static_cast<std::uint32_t>( rng() % 30'000 / right_count ), 1U + static_cast<std::uint32_t>( rng() % 20 ) ) };
        if ( left_values.empty() || right_values.empty() ) { continue; }
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << ": " << left_count << " and " << right_count << " runs" );
        auto const left { make_ours( left_values, true ) };
        auto const right{ make_ours( right_values, true ) };
        auto const theirs_left { make_theirs( left_values, true ) };
        auto const theirs_right{ make_theirs( right_values, true ) };
        croaring const expected{ roaring_bitmap_and( theirs_left.r, theirs_right.r ) };
        EXPECT_TRUE( same_contents( left & right, expected ) );
        EXPECT_TRUE( same_contents( right & left, expected ) );
        EXPECT_EQ( left.and_cardinality( right ), roaring_bitmap_and_cardinality( theirs_left.r, theirs_right.r ) );
        auto copy{ left };
        copy &= right;
        EXPECT_TRUE( same_contents( copy, expected ) );
    }
}

// ---- an array chunk against a run chunk of very many short runs -----------------------------------------
//
// The run list is crossed by a gallop on the run ends, so a handful of array keys against thousands of runs
// reach the right run from far away: keys before the first run, on a run's first and last value, inside it,
// in the gap between runs, past the last run, and a run ending on the chunk's last value. Every operation that
// pairs the two forms is compared with CRoaring: and (both operand orders), the cardinality, the difference
// both ways, and the in-place forms.
struct run_shape {
    char const * name;
    std::uint32_t first;     // first value of the first run
    std::uint32_t length;    // values per run
    std::uint32_t stride;    // distance between run starts
    std::uint32_t count;     // runs
};

constexpr std::uint32_t run_chunk_base{ 3U << 16 };

// The operand held as one run chunk by this library (run-encoded whatever the size: optimize() would turn
// thousands of short runs into a bitset) and as the reference holds the same values.
struct run_operand {
    explicit run_operand( Values32 values_ ) : values{ std::move( values_ ) }, theirs{ values } {
        ours.add_many( values );
        ours.optimize_run_shaped( 1 );
        roaring_bitmap_run_optimize( theirs.r );
    }

    Values32 values;
    Bitmap32 ours;
    croaring theirs;
};

[[nodiscard]] run_operand make_run_operand( std::vector<std::pair<std::uint32_t, std::uint32_t>> const & ranges ) {
    Values32 values;
    for ( auto const [first, last] : ranges ) {
        for ( auto v{ first }; v <= last; ++v ) { values.push_back( run_chunk_base + v ); }
    }
    return run_operand{ std::move( values ) };
}

[[nodiscard]] run_operand make_run_operand( run_shape const & s ) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
    for ( std::uint32_t r{ 0 }; r < s.count; ++r ) {
        auto const first{ s.first + r * s.stride };
        ranges.emplace_back( first, std::min( first + s.length - 1U, 65535U ) );
    }
    return make_run_operand( ranges );
}

// Every operation on the pair (array side, run side), against CRoaring.
void check_array_run_pair( Values32 const & array_values, run_operand const & run, char const * const trace, bool const expect_array = true ) {
    SCOPED_TRACE( ::testing::Message() << trace << ": " << array_values.size() << " array values" );
    auto const array{ make_ours( array_values, false ) };
    auto const theirs_array{ make_theirs( array_values, false ) };
    if ( expect_array ) {
        ASSERT_EQ( array.statistics().array_containers, array.statistics().container_count() ) << "the array side must stay an array";
    }
    croaring const and_expected{ roaring_bitmap_and( theirs_array.r, run.theirs.r ) };
    croaring const array_minus_run{ roaring_bitmap_andnot( theirs_array.r, run.theirs.r ) };
    croaring const run_minus_array{ roaring_bitmap_andnot( run.theirs.r, theirs_array.r ) };
    EXPECT_TRUE( same_contents( array & run.ours, and_expected ) ) << "array & run";
    EXPECT_TRUE( same_contents( run.ours & array, and_expected ) ) << "run & array";
    EXPECT_EQ( array.and_cardinality( run.ours ), roaring_bitmap_and_cardinality( theirs_array.r, run.theirs.r ) );
    EXPECT_EQ( run.ours.and_cardinality( array ), roaring_bitmap_and_cardinality( theirs_array.r, run.theirs.r ) );
    EXPECT_EQ( array.intersects( run.ours ), roaring_bitmap_intersect( theirs_array.r, run.theirs.r ) );
    EXPECT_TRUE( same_contents( array - run.ours, array_minus_run ) ) << "array - run";
    EXPECT_TRUE( same_contents( run.ours - array, run_minus_array ) ) << "run - array";
    { auto t{ array }; t &= run.ours; EXPECT_TRUE( same_contents( t, and_expected ) ) << "array &= run"; }
    { auto t{ array }; t -= run.ours; EXPECT_TRUE( same_contents( t, array_minus_run ) ) << "array -= run"; }
    { auto t{ run.ours }; t &= array; EXPECT_TRUE( same_contents( t, and_expected ) ) << "run &= array"; }
    { auto t{ run.ours }; t -= array; EXPECT_TRUE( same_contents( t, run_minus_array ) ) << "run -= array"; }
}

TEST( KernelCrosscheck, ArrayKeysAgainstManyShortRunsLikeCRoaring ) {
    std::mt19937_64 rng{ 0xA77A };
    static constexpr run_shape shapes[]{
        { "1k runs of 20",              100, 20, 60, 1000 },
        { "1k runs of 20, to the end",  15, 20, 65, 1008 },
        { "10k runs of 3",              2,   3,  6, 10000 },
        { "2k runs of 1",               0,   1, 32, 2048 },
        { "3 long runs",                1000, 5000, 20000, 3 },
        { "one run",                    40000, 100, 1, 1 },
    };
    for ( auto const & shape : shapes ) {
        auto const run{ make_run_operand( shape ) };
        ASSERT_EQ( run.ours.statistics().run_containers, 1U ) << shape.name;
        for ( std::size_t const keys : { 1, 4, 16, 64 } ) {
            // keys before, inside, between and after the runs, at the run edges and beyond the last run
            for ( int trial{ 0 }; trial < 40; ++trial ) {
                std::set<std::uint32_t> lows;
                while ( lows.size() < keys ) {
                    auto const pick{ rng() % 8 };
                    auto const r{ static_cast<std::uint32_t>( rng() % shape.count ) };
                    auto const start{ shape.first + r * shape.stride };
                    switch ( pick ) {
                        case 0: lows.insert( static_cast<std::uint32_t>( rng() % std::max( shape.first, 1U ) ) ); break;         // before the first run
                        case 1: lows.insert( start ); break;                                                                      // a run's first value
                        case 2: lows.insert( start + shape.length - 1U ); break;                                                  // its last
                        case 3: lows.insert( start + static_cast<std::uint32_t>( rng() % shape.length ) ); break;                 // inside
                        case 4: lows.insert( start + shape.length + static_cast<std::uint32_t>( rng() % std::max( shape.stride - shape.length, 1U ) ) ); break;   // the gap after it
                        case 5: lows.insert( 65535U - static_cast<std::uint32_t>( rng() % 64 ) ); break;                          // the chunk's last values
                        case 6: lows.insert( shape.first + shape.count * shape.stride + static_cast<std::uint32_t>( rng() % 100 ) ); break;   // past the last run
                        default: lows.insert( static_cast<std::uint32_t>( rng() % 65536 ) ); break;
                    }
                    for ( auto it{ lows.begin() }; it != lows.end(); ) { it = *it > 65535U ? lows.erase( it ) : std::next( it ); }
                }
                Values32 array_values;
                for ( auto const low : lows ) { array_values.push_back( run_chunk_base + low ); }
                check_array_run_pair( array_values, run, shape.name );
            }
        }
    }
}

TEST( KernelCrosscheck, LongArrayAgainstFewRunsLikeCRoaring ) {
    std::mt19937_64 rng{ 0xB16A };
    auto const run{ make_run_operand( { { 1000, 1999 }, { 30000, 30000 }, { 50000, 65535 } } ) };
    ASSERT_EQ( run.ours.statistics().run_containers, 1U );
    for ( std::size_t const count : { 100, 1000, 2000, 4000 } ) {
        for ( int trial{ 0 }; trial < 20; ++trial ) {
            std::set<std::uint32_t> lows;
            while ( lows.size() < count ) { lows.insert( static_cast<std::uint32_t>( rng() % 65536 ) ); }
            Values32 array_values;
            for ( auto const low : lows ) { array_values.push_back( run_chunk_base + low ); }
            check_array_run_pair( array_values, run, "long array, few runs" );
        }
    }
    // consecutive keys across the run edges
    Values32 edge;
    for ( std::uint32_t low{ 990 }; low < 1010; ++low ) { edge.push_back( run_chunk_base + low ); }
    for ( std::uint32_t low{ 1990 }; low < 2010; ++low ) { edge.push_back( run_chunk_base + low ); }
    for ( std::uint32_t low{ 29995 }; low < 30005; ++low ) { edge.push_back( run_chunk_base + low ); }
    for ( std::uint32_t low{ 49990 }; low < 50010; ++low ) { edge.push_back( run_chunk_base + low ); }
    check_array_run_pair( edge, run, "run edges", false );   // consecutive values: the receiver may hold them as runs
}

} // namespace
