// Container-kernel coverage against std:: oracles: array/array merges, bitset AND
// bitset extraction, small-array export, big-by-few union, in-place array union
// with spare capacity, run counting, and shrink_to_fit followed by mutation.
//
// Contents are compared, never container kinds: the kind a result settles in is a
// property of the implementation (validate() checks it is a legal one).

#include "kernel_test_support.hpp"

#include <bit>
#include <map>
#include <set>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

using Bitmap32 = bitmap<std::uint32_t>;
using Bitmap64 = bitmap<std::uint64_t>;
using CowBitmap32 = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using LazyCowBitmap32 = bitmap<
    std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy
>;
using Values32 = std::vector<std::uint32_t>;

constexpr std::size_t array_sizes[]{ 0, 1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 2047, 2048, 2049, 3000, 4095, 4096 };

// ---------------------------------------------------------------------------
// Array-array merge kernels, through the real dispatch (materialized and in-place).
// The operation set is the library's: union, intersection, difference.
// ---------------------------------------------------------------------------

TEST( KernelCoverage, ArrayArrayMergesMatchReference ) {
    std::mt19937_64 rng{ 0xA77A7 };
    for ( int trial{ 0 }; trial < 2000; ++trial ) {
        auto const pa{ all_patterns[ rng() % all_patterns.size() ] };
        auto const pb{ all_patterns[ rng() % all_patterns.size() ] };
        auto const na{ array_sizes[ rng() % std::size(array_sizes) ] };
        auto const nb{ array_sizes[ rng() % std::size(array_sizes) ] };
        constexpr std::uint32_t bases[]{ 0U, 0x10000U, 0xFFFF0000U };
        auto const base{ bases[ rng() % std::size(bases) ] };
        auto const va{ in_chunk( make_pattern( pa, na, rng ), base ) };
        auto const vb{ in_chunk( make_pattern( pb, nb, rng ), base ) };
        auto const how_a{ all_layouts[ rng() % all_layouts.size() ] };
        auto const how_b{ all_layouts[ rng() % all_layouts.size() ] };
        SCOPED_TRACE( ::testing::Message()
            << "trial " << trial << " a=" << name_of( pa ) << '/' << va.size() << " b=" << name_of( pb ) << '/' << vb.size()
            << " layouts " << int( how_a ) << ',' << int( how_b ) << " base " << base );

        auto const a{ build_with_layout<Bitmap32>( va, how_a, rng ) };
        auto const b{ build_with_layout<Bitmap32>( vb, how_b, rng ) };
        ASSERT_TRUE( matches( a, va ) );
        ASSERT_TRUE( matches( b, vb ) );

        auto const expected_or { set_union_of       ( va, vb ) };
        auto const expected_and{ set_intersection_of( va, vb ) };
        auto const expected_sub{ set_difference_of  ( va, vb ) };

        EXPECT_TRUE( matches( a | b, expected_or  ) ) << "operator|";
        EXPECT_TRUE( matches( a & b, expected_and ) ) << "operator&";
        EXPECT_TRUE( matches( a - b, expected_sub ) ) << "operator-";
        EXPECT_EQ( a.intersects( b ), !expected_and.empty() );

        { auto t{ a }; t |= b; EXPECT_TRUE( matches( t, expected_or  ) ) << "operator|="; }
        { auto t{ a }; t &= b; EXPECT_TRUE( matches( t, expected_and ) ) << "operator&="; }
        { auto t{ a }; t -= b; EXPECT_TRUE( matches( t, expected_sub ) ) << "operator-="; }
        // The operands are unchanged by the materializing forms.
        EXPECT_TRUE( matches( a, va ) );
        EXPECT_TRUE( matches( b, vb ) );
    }
}

// The kernels themselves, driven directly across sizes that straddle the vector
// widths, with the output guarded on both sides.
TEST( KernelCoverage, SortedUnionKernelsMatchReference ) {
    std::mt19937_64 rng{ 0x4E10 };
    constexpr std::size_t guard{ 64 };
    constexpr std::uint16_t canary{ 0xCAFE };
    for ( int trial{ 0 }; trial < 500; ++trial ) {
        auto const na{ array_sizes[ 1 + rng() % ( std::size(array_sizes) - 1 ) ] };
        auto const nb{ array_sizes[ 1 + rng() % ( std::size(array_sizes) - 1 ) ] };
        auto const pa{ all_patterns[ rng() % 4 ] };
        auto const pb{ all_patterns[ rng() % 4 ] };
        auto const a{ make_pattern( pa, na, rng ) };
        auto const b{ make_pattern( pb, nb, rng ) };
        std::vector<std::uint16_t> expected;
        std::ranges::set_union( a, b, std::back_inserter( expected ) );
        SCOPED_TRACE( ::testing::Message() << "a=" << name_of( pa ) << '/' << a.size() << " b=" << name_of( pb ) << '/' << b.size() );

        auto const run{ [&]( auto && kernel, std::size_t const slack, char const * const what ) {
            std::vector<std::uint16_t> out( guard + a.size() + b.size() + slack + guard, canary );
            auto const n{ kernel( a.data(), a.size(), b.data(), b.size(), out.data() + guard ) };
            ASSERT_EQ( n, expected.size() ) << what;
            EXPECT_TRUE( std::equal( expected.begin(), expected.end(), out.begin() + guard ) ) << what;
            EXPECT_TRUE( std::all_of( out.begin(), out.begin() + guard, [&]( auto v ) { return v == canary; } ) ) << what << ": wrote before the output";
            EXPECT_TRUE( std::all_of( out.begin() + guard + a.size() + b.size() + slack, out.end(), [&]( auto v ) { return v == canary; } ) ) << what << ": wrote past the output slack";
        } };
        run( detail::union_sorted_uint16_scalar, 0, "scalar" );
#if FRSR_ROARING_X86_V4
        if ( detail::have_x86_v4() && a.size() >= detail::x86_v4::union_min_side && b.size() >= detail::x86_v4::union_min_side ) {
            run( detail::x86_v4::union_sorted_uint16, detail::x86_v4::union_store_slack, "avx512" );
        }
#endif
    }
}

// ---------------------------------------------------------------------------
// bitset AND bitset -> (usually) array, swept over the result's bits per 64-bit word.
// ---------------------------------------------------------------------------

struct word_pair {
    std::array<std::uint64_t, 1024> a, b, both;
};

// `overlap_words` words are fully shared (clustered) or every word shares exactly
// `bits` positions (spread); outside the shared bits the operands never overlap, so
// the intersection is exactly the shared set.
[[nodiscard]] word_pair make_word_pair( unsigned const bits, bool const clustered, std::mt19937_64 & rng ) {
    word_pair w{};
    auto const cluster_words{ ( bits * 1024U + 32U ) / 64U };
    for ( std::size_t i{ 0 }; i < 1024; ++i ) {
        std::uint64_t shared{ 0 };
        if ( clustered ) {
            shared = i < cluster_words ? ~std::uint64_t{ 0 } : 0;
        } else {
            while ( static_cast<unsigned>( std::popcount( shared ) ) < bits ) { shared |= std::uint64_t{ 1 } << ( rng() % 64 ); }
        }
        auto const rest{ ~shared };
        auto const split{ rng() };
        w.a[ i ] = shared | ( rest & split );
        w.b[ i ] = shared | ( rest & ~split & rng() );
        w.both[ i ] = shared;
    }
    return w;
}

[[nodiscard]] Values32 values_of_words( std::array<std::uint64_t, 1024> const & words, std::uint32_t const base ) {
    Values32 values;
    for ( std::size_t i{ 0 }; i < words.size(); ++i ) {
        for ( auto w{ words[ i ] }; w != 0; w &= w - 1 ) {
            values.push_back( base + static_cast<std::uint32_t>( i * 64 + std::countr_zero( w ) ) );
        }
    }
    return values;
}

TEST( KernelCoverage, BitsetAndBitsetAcrossDensities ) {
    std::mt19937_64 rng{ 0xB175E7 };
    for ( bool const clustered : { false, true } ) {
        for ( unsigned bits{ 0 }; bits <= 64; ++bits ) {
            SCOPED_TRACE( ::testing::Message() << ( clustered ? "clustered" : "spread" ) << " bits/word " << bits );
            constexpr std::uint32_t base{ 0x20000U };
            auto const w{ make_word_pair( bits, clustered, rng ) };
            auto const va{ values_of_words( w.a, base ) };
            auto const vb{ values_of_words( w.b, base ) };
            auto const expected{ values_of_words( w.both, base ) };
            ASSERT_GT( va.size(), 4096U );
            ASSERT_GT( vb.size(), 4096U );
            Bitmap32 a, b;
            a.add_many( va );
            b.add_many( vb );
            EXPECT_TRUE( matches( a & b, expected ) ) << "operator&";
            EXPECT_TRUE( matches( b & a, expected ) ) << "operator& swapped";
            { auto t{ a }; t &= b; EXPECT_TRUE( matches( t, expected ) ) << "operator&="; }
            EXPECT_EQ( a.intersects( b ), !expected.empty() );
        }
    }
}

// ---------------------------------------------------------------------------
// Small-array export into a caller's buffer: every cardinality around the vector
// widths, misaligned destinations, a guarded buffer, and the top chunk.
// ---------------------------------------------------------------------------

template <typename Bitmap>
void expect_export_exact( Bitmap const & bm, std::vector<typename Bitmap::key_type> const & expected ) {
    using key_type = typename Bitmap::key_type;
    constexpr std::size_t guard{ 32 };
    constexpr key_type canary{ static_cast<key_type>( 0xDEADBEEFDEADBEEFULL ) };
    for ( std::size_t const misalign : { 0, 1, 2, 3, 5, 7, 15 } ) {
        std::vector<key_type> buffer( guard + misalign + expected.size() + guard, canary );
        std::span const out{ buffer.data() + guard + misalign, expected.size() };
        auto const written{ bm.to_array_into( out ) };
        ASSERT_EQ( written, expected.size() ) << "misalign " << misalign;
        EXPECT_TRUE( std::equal( expected.begin(), expected.end(), out.begin() ) ) << "misalign " << misalign;
        EXPECT_TRUE( std::all_of( buffer.begin(), out.data() ? buffer.begin() + guard + misalign : buffer.begin(), [&]( auto v ) { return v == canary; } ) )
            << "wrote before the output, misalign " << misalign;
        EXPECT_TRUE( std::all_of( buffer.begin() + guard + misalign + expected.size(), buffer.end(), [&]( auto v ) { return v == canary; } ) )
            << "wrote past the output, misalign " << misalign;
    }
}

template <typename Bitmap>
void export_small_arrays_case() {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0xE5B027 };
    constexpr key_type bases[]{
        0, static_cast<key_type>( 0xFFFF0000ULL ), static_cast<key_type>( 0x00010000ULL ), static_cast<key_type>( ~key_type{ 0 } & ~key_type{ 0xFFFF } )
    };
    for ( auto const base : bases ) {
        for ( std::size_t n{ 0 }; n <= 80; ++n ) {
            SCOPED_TRACE( ::testing::Message() << "base " << static_cast<unsigned long long>( base ) << " n " << n );
            for ( auto const p : { pattern::random, pattern::one_run, pattern::every_third } ) {
                auto const values{ in_chunk<key_type>( make_pattern( p, n, rng ), base ) };
                Bitmap bm;
                bm.add_many( values );
                expect_export_exact( bm, values );
            }
        }
    }
    // Many tiny containers in one export, the shape the per-container decode setup
    // dominates: 4096 chunks of 1..16 values.
    std::vector<key_type> values;
    for ( std::size_t chunk{ 0 }; chunk < 4096; ++chunk ) {
        auto const base{ static_cast<key_type>( chunk << 16 ) };
        auto const n{ 1 + rng() % 16 };
        for ( auto const low : make_pattern( pattern::random, n, rng ) ) { values.push_back( static_cast<key_type>( base + low ) ); }
    }
    Bitmap bm;
    bm.add_many( values );
    expect_export_exact( bm, values );
}

TEST( KernelCoverage, SmallArrayExport32 ) { export_small_arrays_case<Bitmap32>(); }
TEST( KernelCoverage, SmallArrayExport64 ) { export_small_arrays_case<Bitmap64>(); }

// ---------------------------------------------------------------------------
// Big target |= few keys: the target holds 1000 containers of every kind, the
// operand 1..20 keys that hit existing keys, fall between them, or lie before/after.
// ---------------------------------------------------------------------------

enum class key_mode : std::uint8_t { existing, between, before, after, mixed };

constexpr std::uint32_t target_first_key{ 10 };
constexpr std::size_t   target_chunks{ 1000 };

[[nodiscard]] Values32 make_target_values( std::mt19937_64 & rng, Values32 & fulls ) {
    Values32 values;
    for ( std::size_t i{ 0 }; i < target_chunks; ++i ) {
        auto const base{ ( target_first_key + 2 * static_cast<std::uint32_t>( i ) ) << 16 };
        switch ( i % 4 ) {
            case 0: // array
                for ( auto const low : make_pattern( pattern::random, 100, rng ) ) { values.push_back( base + low ); }
                break;
            case 1: // bitset
                for ( auto const low : make_pattern( pattern::random, 6000, rng ) ) { values.push_back( base + low ); }
                break;
            case 2: // runs
                for ( std::uint32_t r{ 0 }; r < 5; ++r ) {
                    for ( std::uint32_t k{ 0 }; k < 150; ++k ) { values.push_back( base + r * 5000 + k ); }
                }
                break;
            case 3: // full, on every tenth such chunk (a few dozen of them) the rest runs
                if ( i % 40 == 3 ) {
                    for ( std::uint32_t k{ 0 }; k < 65536; ++k ) { values.push_back( base + k ); }
                    fulls.push_back( base );
                } else {
                    for ( std::uint32_t k{ 1000 }; k < 3000; ++k ) { values.push_back( base + k ); }
                }
                break;
        }
    }
    return values;
}

template <typename Bitmap>
[[nodiscard]] Bitmap make_big_target( Values32 const & values ) {
    Bitmap target;
    target.add_many( values );
    target.optimize(); // arrays/runs/full chunks settle in their run form where that is smaller
    return target;
}

// Every key the target does not hold, or does, in the places `mode` names: before the
// first key (10 of them), between the existing ones, after the last (50), or the
// existing ones themselves.
[[nodiscard]] std::vector<std::uint32_t> candidate_keys( key_mode const mode ) {
    std::vector<std::uint32_t> keys;
    switch ( mode ) {
        case key_mode::before:
            for ( std::uint32_t k{ 0 }; k < target_first_key; ++k ) { keys.push_back( k ); }
            break;
        case key_mode::after:
            for ( std::uint32_t k{ 0 }; k < 50; ++k ) { keys.push_back( target_first_key + 2 * static_cast<std::uint32_t>( target_chunks ) + k ); }
            break;
        default:
            for ( std::uint32_t i{ 0 }; i < target_chunks; ++i ) {
                keys.push_back( target_first_key + 2 * i + ( mode == key_mode::between ? 1U : 0U ) );
            }
            break;
    }
    return keys;
}

[[nodiscard]] Values32 make_few_values( std::size_t const keys, key_mode const mode, std::mt19937_64 & rng ) {
    Values32 values;
    std::map<key_mode, std::vector<std::uint32_t>> pools;
    for ( auto const m : { key_mode::existing, key_mode::between, key_mode::before, key_mode::after } ) {
        pools[ m ] = candidate_keys( m );
        std::ranges::shuffle( pools[ m ], rng );
    }
    for ( std::size_t k{ 0 }; k < keys; ++k ) {
        auto const which{ mode == key_mode::mixed ? static_cast<key_mode>( k % 4 ) : mode };
        auto & pool{ pools[ which ] };
        if ( pool.empty() ) { continue; } // a mode with fewer candidate keys than asked for
        auto const key{ pool.back() };
        pool.pop_back();
        auto const base{ key << 16 };
        switch ( k % 3 ) {
            case 0: for ( auto const low : make_pattern( pattern::random, 50, rng ) ) { values.push_back( base + low ); } break;
            case 1: for ( auto const low : make_pattern( pattern::random, 5000, rng ) ) { values.push_back( base + low ); } break;
            default: for ( std::uint32_t i{ 0 }; i < 4000; ++i ) { values.push_back( base + 100 + i ); } break;
        }
    }
    return sorted_unique( std::move( values ) );
}

template <typename Bitmap>
void big_by_few_case() {
    std::mt19937_64 rng{ 0xB16B };
    Values32 fulls;
    auto const target_values{ sorted_unique( make_target_values( rng, fulls ) ) };
    auto const target{ make_big_target<Bitmap>( target_values ) };
    ASSERT_TRUE( matches( target, target_values ) );
    ASSERT_GE( target.container_count(), target_chunks );

    for ( std::size_t const keys : { 1, 2, 4, 8, 16, 20 } ) {
        for ( auto const mode : { key_mode::existing, key_mode::between, key_mode::before, key_mode::after, key_mode::mixed } ) {
            SCOPED_TRACE( ::testing::Message() << keys << " keys, mode " << int( mode ) );
            auto const few_values{ make_few_values( keys, mode, rng ) };
            Bitmap few;
            few.add_many( few_values );
            auto const expected{ set_union_of( target_values, few_values ) };

            // Unshared: the target is a private copy of the prebuilt one (a value copy
            // under value semantics, a payload-sharing copy under CoW).
            {
                auto t{ target };
                t |= few;
                EXPECT_TRUE( matches( t, expected ) );
                // Mutating the result must never reach the original or the operand.
                (void)t.add( 0xFFFFFFFFU );
                t.remove_closed_range( target_first_key << 16, ( target_first_key + 40 ) << 16 );
                EXPECT_TRUE( matches( target, target_values ) ) << "the source of the copy changed";
                EXPECT_TRUE( matches( few, few_values ) ) << "the operand changed";
            }
            // Shared: another copy stays alive across the union and is checked after.
            {
                auto t{ target };
                auto const sibling{ t };
                t |= few;
                EXPECT_TRUE( matches( t, expected ) );
                EXPECT_TRUE( matches( sibling, target_values ) ) << "a sharing copy changed under |=";
                auto u{ sibling };
                u |= few;
                (void)t.add( 123456789U );
                EXPECT_TRUE( matches( u, expected ) ) << "a later copy was affected by another copy's write";
            }
            // The few-by-big direction (operand larger than the receiver).
            {
                auto t{ few };
                t |= target;
                EXPECT_TRUE( matches( t, expected ) );
            }
            // Self-aliasing through a copy.
            {
                auto t{ target };
                auto const same{ t };
                t |= same;
                EXPECT_TRUE( matches( t, target_values ) );
            }
        }
    }
}

TEST( KernelCoverage, BigByFewUnion )          { big_by_few_case<Bitmap32>(); }
TEST( KernelCoverage, BigByFewUnionCow )       { big_by_few_case<CowBitmap32>(); }
TEST( KernelCoverage, BigByFewUnionLazyCow )   { big_by_few_case<LazyCowBitmap32>(); }

// ---------------------------------------------------------------------------
// Array |= array with spare capacity in the receiver, including unions that
// overflow the array limit and must settle as a bitset.
// ---------------------------------------------------------------------------

TEST( KernelCoverage, ArrayInplaceUnionWithSpareCapacity ) {
    std::mt19937_64 rng{ 0x5ACE };
    constexpr std::size_t sizes[]{ 32, 33, 100, 1000, 2000, 2048, 3000, 4000, 4096 };
    constexpr std::uint32_t base{ 5U << 16 };
    for ( auto const la : sizes ) {
        for ( auto const lb : sizes ) {
            for ( unsigned const overlap_percent : { 0U, 50U, 100U } ) {
                SCOPED_TRACE( ::testing::Message() << "la " << la << " lb " << lb << " overlap " << overlap_percent << '%' );
                auto const lows_a{ make_pattern( pattern::random, la, rng ) };
                // The shared part of b comes from a, the rest from outside a.
                auto const shared_count{ std::min( lb * overlap_percent / 100U, la ) };
                std::vector<std::uint16_t> lows_b;
                {
                    auto pool{ lows_a };
                    std::ranges::shuffle( pool, rng );
                    lows_b.assign( pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>( shared_count ) );
                    auto const rest{ random_outside( lb - shared_count, lows_a, rng ) };
                    lows_b.insert( lows_b.end(), rest.begin(), rest.end() );
                    std::ranges::sort( lows_b );
                }
                auto const va{ in_chunk( lows_a, base ) };
                auto const vb{ in_chunk( lows_b, base ) };
                auto const expected{ set_union_of( va, vb ) };

                for ( auto const how : all_layouts ) {
                    auto a{ build_with_layout<Bitmap32>( va, how, rng ) };
                    auto const b{ build_with_layout<Bitmap32>( vb, layout::bulk, rng ) };
                    a |= b;
                    ASSERT_TRUE( matches( a, expected ) ) << "layout " << int( how );
                    ASSERT_TRUE( matches( b, vb ) );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Run counting over sorted values, with and without the early-exit cap.
// ---------------------------------------------------------------------------

[[nodiscard]] std::size_t naive_run_count( std::vector<std::uint16_t> const & v ) {
    std::size_t runs{ 0 };
    for ( std::size_t i{ 0 }; i < v.size(); ++i ) {
        if ( i == 0 || static_cast<unsigned>( v[ i ] ) != static_cast<unsigned>( v[ i - 1 ] ) + 1U ) { ++runs; }
    }
    return runs;
}

TEST( KernelCoverage, RunCountMatchesNaive ) {
    using layout_t = detail::default_layout<std::uint32_t>;
    std::mt19937_64 rng{ 0x2U };
    constexpr auto npos{ std::numeric_limits<std::size_t>::max() };
    EXPECT_EQ( detail::run_count_from_sorted_values<layout_t>( {} ), 0U );
    for ( int trial{ 0 }; trial < 1000; ++trial ) {
        auto const p{ all_patterns[ rng() % all_patterns.size() ] };
        auto const n{ array_sizes[ rng() % std::size(array_sizes) ] };
        auto values{ make_pattern( p, n, rng ) };
        if ( trial % 5 == 0 ) { // exercise the top of the 16-bit range, where value + 1 wraps
            for ( std::uint16_t v{ 0xFFF0 }; v != 0; ++v ) { values.push_back( v ); }
            values = sorted_unique( std::move( values ) );
        }
        auto const exact{ naive_run_count( values ) };
        EXPECT_EQ( detail::run_count_from_sorted_values<layout_t>( values ), exact ) << name_of( p ) << '/' << n;
        for ( std::size_t const cap : { std::size_t{ 1 }, std::size_t{ 2 }, exact, exact + 1, exact / 2 + 1 } ) {
            if ( cap == 0 ) { continue; }
            EXPECT_EQ( detail::run_count_from_sorted_values<layout_t>( values, cap ), std::min( exact, cap ) )
                << name_of( p ) << '/' << n << " cap " << cap;
        }
        EXPECT_EQ( detail::run_count_from_sorted_values<layout_t>( values, npos ), exact );
    }
}

// ---------------------------------------------------------------------------
// shrink_to_fit after mass removal, then every kind of mutation.
// ---------------------------------------------------------------------------

template <typename Bitmap>
void shrink_then_mutate_case() {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0x5121 };
    constexpr std::size_t chunks{ 600 };
    std::vector<key_type> all;
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        auto const base{ static_cast<key_type>( c * 3 ) << 16 };
        auto const n{ c % 3 == 0 ? 6000U : ( c % 3 == 1 ? 900U : 20U ) };
        for ( auto const low : make_pattern( pattern::random, n, rng ) ) { all.push_back( static_cast<key_type>( base + low ) ); }
    }
    all = sorted_unique( std::move( all ) );

    Bitmap bm;
    bm.add_many( all );
    // Mass removal: keep roughly one value in 32, and empty a third of the chunks.
    std::vector<key_type> kept, doomed;
    for ( auto const v : all ) {
        ( ( rng() % 32 == 0 && ( ( v >> 16 ) / 3 ) % 3 != 2 ) ? kept : doomed ).push_back( v );
    }
    bm.remove_many( doomed );
    ASSERT_TRUE( matches( bm, kept ) );

    auto const before_bytes{ bm.byte_size() };
    (void)bm.shrink_to_fit();
    EXPECT_LE( bm.byte_size(), before_bytes );
    ASSERT_TRUE( matches( bm, kept ) );
    for ( std::size_t i{ 0 }; i < kept.size(); i += 7 ) { EXPECT_TRUE( bm.contains( kept[ i ] ) ); }

    // Mutate after the shrink: point adds into shrunk arrays (forcing regrowth), a
    // range add, point removes, a union, an intersection and a difference.
    auto expected{ kept };
    for ( int i{ 0 }; i < 3000; ++i ) {
        auto const v{ static_cast<key_type>( all[ rng() % all.size() ] ) };
        if ( bm.add( v ) ) { expected.insert( std::ranges::lower_bound( expected, v ), v ); }
    }
    ASSERT_TRUE( matches( bm, expected ) );
    for ( int i{ 0 }; i < 500; ++i ) {
        auto const v{ expected[ rng() % expected.size() ] };
        if ( bm.remove( v ) ) { expected.erase( std::ranges::lower_bound( expected, v ) ); }
    }
    ASSERT_TRUE( matches( bm, expected ) );

    constexpr key_type range_begin{ static_cast<key_type>( 7U ) << 16 };
    constexpr key_type range_end{ ( static_cast<key_type>( 9U ) << 16 ) + 123 };
    bm.add_closed_range( range_begin, range_end );
    std::vector<key_type> range;
    for ( key_type v{ range_begin }; v <= range_end; ++v ) { range.push_back( v ); }
    expected = set_union_of( expected, range );
    ASSERT_TRUE( matches( bm, expected ) );

    Bitmap other;
    other.add_many( sorted_unique( [&] { std::vector<key_type> v; for ( std::size_t i{ 0 }; i < all.size(); i += 5 ) { v.push_back( all[ i ] ); } return v; }() ) );
    auto const other_values{ other.to_vector() };
    std::vector<key_type> const other_copy( other_values.begin(), other_values.end() );
    { auto t{ bm }; t |= other; EXPECT_TRUE( matches( t, set_union_of( expected, other_copy ) ) ); }
    { auto t{ bm }; t &= other; EXPECT_TRUE( matches( t, set_intersection_of( expected, other_copy ) ) ); }
    { auto t{ bm }; t -= other; EXPECT_TRUE( matches( t, set_difference_of( expected, other_copy ) ) ); }

    // Shrinking an emptied bitmap, and mutating it afterwards.
    Bitmap emptied;
    emptied.add_many( kept );
    emptied.remove_many( kept );
    (void)emptied.shrink_to_fit();
    EXPECT_TRUE( emptied.empty() );
    ASSERT_TRUE( emptied.add( 42 ) );
    EXPECT_TRUE( emptied.contains( 42 ) );
    EXPECT_EQ( emptied.size(), 1U );
}

TEST( KernelCoverage, ShrinkToFitThenMutate32 ) { shrink_then_mutate_case<Bitmap32>(); }
TEST( KernelCoverage, ShrinkToFitThenMutate64 ) { shrink_then_mutate_case<Bitmap64>(); }

// ---------------------------------------------------------------------------
// Flipping a closed range over sparse arrays: cardinalities around the array
// limit and around half of a chunk, ranges inside a chunk, over a whole chunk and
// across several. Only the contents are checked (the container kind the result
// settles in is the implementation's choice).
// ---------------------------------------------------------------------------

template <typename Bitmap>
void flip_sparse_case() {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0xF11B };
    constexpr key_type chunk{ 3 };
    constexpr key_type base{ chunk << 16 };
    struct range_t { key_type first, last; char const * name; };
    constexpr range_t ranges[]{
        { base,           base + 65535,                 "whole chunk" },
        { base + 5,       base + 65530,                 "inside, almost whole" },
        { base + 1000,    base + 20000,                 "inside, middle" },
        { base + 777,     base + 777,                   "single value" },
        { base + 65535,   base + 65535,                 "last value" },
        { base,           base,                         "first value" },
        { base - 10,      base + 65535 + 10,            "chunk and a margin each side" },
        { base + 100,     base + 2 * 65536 + 500,       "multi-chunk" },
        { base - 65536,   base + 65535 + 65536,         "three whole chunks" },
        { base + 65000,   base + 65536 + 600,           "chunk tail into the next" },
    };
    for ( std::size_t const cardinality : { 1, 2, 100, 2045, 2046, 2047, 3000, 4096 } ) {
        for ( auto const pat : { pattern::random, pattern::one_run, pattern::every_third } ) {
            auto const values{ in_chunk<key_type>( make_pattern( pat, cardinality, rng ), base ) };
            // Neighbouring chunks hold data too, so a multi-chunk flip mixes
            // existing and absent chunks.
            auto const neighbour{ in_chunk<key_type>( make_pattern( pattern::random, 100, rng ), static_cast<key_type>( base + 65536 ) ) };
            auto const before  { in_chunk<key_type>( make_pattern( pattern::random, 5000, rng ), static_cast<key_type>( base - 65536 ) ) };
            for ( bool const with_neighbours : { false, true } ) {
                auto all{ values };
                if ( with_neighbours ) { all = sorted_unique( [&] { auto v{ before }; v.insert( v.end(), values.begin(), values.end() ); v.insert( v.end(), neighbour.begin(), neighbour.end() ); return v; }() ); }
                for ( auto const & r : ranges ) {
                    SCOPED_TRACE( ::testing::Message() << "card " << cardinality << " " << name_of( pat ) << " neighbours " << with_neighbours << " range " << r.name );
                    std::vector<key_type> range;
                    for ( key_type v{ r.first }; v <= r.last; ++v ) { range.push_back( v ); }
                    auto const expected{ set_union_of( set_difference_of( all, range ), set_difference_of( range, all ) ) };
                    for ( bool const run_optimized : { false, true } ) {
                        Bitmap bm;
                        bm.add_many( all );
                        if ( run_optimized ) { bm.optimize(); }
                        bm.flip_closed_range( r.first, r.last );
                        ASSERT_TRUE( matches( bm, expected ) ) << "run_optimized " << run_optimized;
                        bm.flip_closed_range( r.first, r.last );
                        ASSERT_TRUE( matches( bm, all ) ) << "the second flip must restore the input";
                    }
                }
            }
        }
    }
}

TEST( KernelCoverage, FlipSparseArray32 ) { flip_sparse_case<Bitmap32>(); }
TEST( KernelCoverage, FlipSparseArray64 ) { flip_sparse_case<Bitmap64>(); }

// A 64-bit bitmap whose high key bytes are spread widely (so the chunk index has
// every fan-out), shrunk after most of its chunks were removed.
TEST( KernelCoverage, ShrinkToFitSpreadHighKeys64 ) {
    constexpr std::uint64_t highs[]{ 0, 1, 3, 7, 20, 100, 1000 };
    for ( std::uint64_t const step : { 1ULL, 2ULL, 3ULL, 10ULL, 100ULL, 1'000'000ULL } ) {
        SCOPED_TRACE( ::testing::Message() << "step " << step );
        std::vector<std::uint64_t> all, kept;
        for ( auto const high : highs ) {
            for ( std::uint64_t i{ 0 }, n{ 4 * ( high + 1 ) + 300 }; i < n; ++i ) { all.push_back( ( high << 40 ) | ( i << 16 ) | ( i & 7 ) ); }
        }
        for ( std::size_t k{ 0 }; k < all.size(); ++k ) { if ( k % step == 0 ) { kept.push_back( all[ k ] ); } }
        Bitmap64 bm;
        bm.add_many( all );
        bm.remove_many( set_difference_of( all, kept ) );
        (void)bm.shrink_to_fit();
        ASSERT_TRUE( matches( bm, kept ) );
        auto expected{ kept };
        for ( auto const v : kept ) { if ( bm.add( v + 1 ) ) { expected.push_back( v + 1 ); } }
        (void)bm.remove( kept[ 0 ] );
        expected = sorted_unique( std::move( expected ) );
        expected.erase( std::ranges::lower_bound( expected, kept[ 0 ] ) );
        ASSERT_TRUE( matches( bm, expected ) );
    }
    // Every chunk removed, then reuse.
    Bitmap64 bm;
    for ( std::uint64_t i{ 0 }; i < 1000; ++i ) { (void)bm.add( i << 16 ); }
    for ( std::uint64_t i{ 0 }; i < 1000; ++i ) { (void)bm.remove( i << 16 ); }
    (void)bm.shrink_to_fit();
    EXPECT_TRUE( bm.empty() );
    (void)bm.add( 42 );
    EXPECT_TRUE( bm.contains( 42 ) );
}

// remove_many naming the same value twice, the second time after the first removal
// emptied its chunk.
TEST( KernelCoverage, RemoveManyDuplicateAfterEmptyingChunk ) {
    for ( bool const wide : { false, true } ) {
        Bitmap64 bm;
        bm.add_many( std::vector<std::uint64_t>{ 0, 1ULL << ( wide ? 40 : 20 ) } );
        std::array<std::uint64_t, 2> const doomed{ 0, 0 };
        bm.remove_many( doomed );
        EXPECT_TRUE( matches( bm, { 1ULL << ( wide ? 40 : 20 ) } ) );
        EXPECT_TRUE( bm.contains( 1ULL << ( wide ? 40 : 20 ) ) );
        EXPECT_FALSE( bm.contains( 0 ) );
    }
}

} // namespace
