// contains() / contains_bulk() answer correctly in every state a bitmap can reach:
// built sorted or out of order, grown by add_many in reverse batches, shrunk by
// remove (including emptying whole chunks), run-optimized, copied and moved - over
// array, bitset, run and mixed containers, with few chunks and with many (more than
// the chunk-lookup index threshold), at chunk strides from adjacent to 2^40 apart.
//
// The oracle is a sorted std::vector; probes are members, neighbours of members,
// values in absent chunks, and the key-domain edges.

#include "kernel_test_support.hpp"

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

using CowBitmap32 = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using LazyCowBitmap32 = bitmap<
    std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy
>;

enum class kinds : std::uint8_t { arrays, bitsets, runs, mixed };

[[nodiscard]] constexpr char const * name_of( kinds const k ) noexcept {
    switch ( k ) {
        case kinds::arrays:  return "arrays";
        case kinds::bitsets: return "bitsets";
        case kinds::runs:    return "runs";
        case kinds::mixed:   return "mixed";
    }
    return "?";
}

// The low 16 bits of one chunk, in the container kind asked for (before any
// optimize(): an array is built below the array limit, a bitset above it, a run as
// one long run).
[[nodiscard]] std::vector<std::uint16_t> chunk_lows( kinds const k, std::size_t const index, std::mt19937_64 & rng ) {
    auto const which{ k == kinds::mixed ? static_cast<kinds>( index % 3 ) : k };
    switch ( which ) {
        case kinds::arrays:  return make_pattern( pattern::random, 50 + rng() % 1500, rng );
        case kinds::bitsets: return make_pattern( pattern::random, 5000 + rng() % 20000, rng );
        default: { // runs: a few long ones
            std::vector<std::uint16_t> lows;
            for ( std::uint32_t r{ 0 }; r < 3; ++r ) {
                auto const first{ r * 20000U + static_cast<std::uint32_t>( rng() % 500 ) };
                auto const length{ 300U + static_cast<std::uint32_t>( rng() % 3000 ) };
                for ( std::uint32_t i{ 0 }; i < length; ++i ) { lows.push_back( static_cast<std::uint16_t>( first + i ) ); }
            }
            return lows;
        }
    }
}

template <typename Key>
[[nodiscard]] std::vector<Key> make_values( kinds const k, std::size_t const chunks, Key const stride, std::mt19937_64 & rng ) {
    std::vector<Key> values;
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        auto const base{ static_cast<Key>( ( static_cast<Key>( c ) + 1 ) * stride ) };
        for ( auto const low : chunk_lows( k, c, rng ) ) { values.push_back( static_cast<Key>( base + low ) ); }
    }
    return sorted_unique( std::move( values ) );
}

template <typename Key>
struct probes {
    std::vector<Key> hits, misses;
};

// Members (every one when few, a stride otherwise), neighbours that are not members,
// values in chunks nobody holds, and the domain edges.
template <typename Key>
[[nodiscard]] probes<Key> make_probes( std::vector<Key> const & values, Key const stride, std::mt19937_64 & rng ) {
    probes<Key> p;
    auto const step{ std::max<std::size_t>( 1, values.size() / 20000 ) };
    for ( std::size_t i{ 0 }; i < values.size(); i += step ) { p.hits.push_back( values[ i ] ); }
    auto const consider{ [&]( Key const v ) {
        if ( !std::ranges::binary_search( values, v ) ) { p.misses.push_back( v ); }
    } };
    for ( std::size_t i{ 0 }; i < values.size(); i += step ) {
        consider( static_cast<Key>( values[ i ] + 1 ) );
        consider( static_cast<Key>( values[ i ] - 1 ) );
        consider( static_cast<Key>( values[ i ] ^ 0x8000 ) );
    }
    for ( std::size_t i{ 0 }; i < 4000; ++i ) {
        // A chunk key nobody holds: between the held ones, below, or above.
        auto const chunk{ static_cast<Key>( rng() % ( values.size() / 20 + 3 ) ) };
        consider( static_cast<Key>( chunk * stride + ( stride > 65536 ? stride / 2 : 0 ) + ( rng() & 0xFFFF ) ) );
    }
    consider( 0 );
    consider( static_cast<Key>( ~Key{ 0 } ) );
    consider( static_cast<Key>( ~Key{ 0 } - 65535 ) );
    return p;
}

template <typename Bitmap>
void expect_contains_everything( Bitmap const & bm, probes<typename Bitmap::key_type> const & p, std::mt19937_64 & rng ) {
    std::size_t wrong{ 0 };
    for ( auto const v : p.hits ) { wrong += bm.contains( v ) ? 0 : 1; }
    EXPECT_EQ( wrong, 0U ) << "contains() missed members";
    wrong = 0;
    for ( auto const v : p.misses ) { wrong += bm.contains( v ) ? 1 : 0; }
    EXPECT_EQ( wrong, 0U ) << "contains() reported non-members";

    // The bulk form, ascending (the cached-chunk lane) and in a permuted order.
    auto const bulk{ [&]( auto probes_in_order, bool const expected, char const * const what ) {
        typename Bitmap::bulk_context ctx{};
        std::size_t bad{ 0 };
        for ( auto const v : probes_in_order ) { bad += bm.contains_bulk( ctx, v ) == expected ? 0 : 1; }
        EXPECT_EQ( bad, 0U ) << what;
    } };
    auto hits{ p.hits };
    auto misses{ p.misses };
    std::ranges::sort( misses );
    bulk( hits, true, "contains_bulk ascending hits" );
    bulk( misses, false, "contains_bulk ascending misses" );
    std::ranges::shuffle( hits, rng );
    std::ranges::shuffle( misses, rng );
    bulk( hits, true, "contains_bulk permuted hits" );
    bulk( misses, false, "contains_bulk permuted misses" );
    // Interleaved hits and misses through one context.
    typename Bitmap::bulk_context ctx{};
    std::size_t bad{ 0 };
    for ( std::size_t i{ 0 }; i < std::min( hits.size(), misses.size() ); ++i ) {
        bad += bm.contains_bulk( ctx, hits[ i ] ) ? 0 : 1;
        bad += bm.contains_bulk( ctx, misses[ i ] ) ? 1 : 0;
    }
    EXPECT_EQ( bad, 0U ) << "contains_bulk interleaved";
}

template <typename Bitmap>
void contains_in_all_states_case( kinds const k, std::size_t const chunks, typename Bitmap::key_type const stride ) {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ 0xC0DE + chunks };
    auto const values{ make_values<key_type>( k, chunks, stride, rng ) };
    auto const p{ make_probes<key_type>( values, stride, rng ) };
    SCOPED_TRACE( ::testing::Message() << name_of( k ) << ", " << chunks << " chunks, stride " << static_cast<unsigned long long>( stride ) << ", " << values.size() << " values" );

    // Extra values in the same chunks and in chunks of their own, to be removed again.
    auto const extra{ [&] {
        std::vector<key_type> v;
        for ( std::size_t i{ 0 }; i < values.size() / 8; ++i ) { v.push_back( static_cast<key_type>( values[ rng() % values.size() ] ^ 0x4001 ) ); }
        for ( std::size_t c{ 0 }; c < 40; ++c ) { v.push_back( static_cast<key_type>( ( static_cast<key_type>( chunks + 2 + c ) ) * stride + ( rng() & 0xFFFF ) ) ); }
        v = sorted_unique( std::move( v ) );
        return set_difference_of( v, values );
    }() };

    {
        SCOPED_TRACE( "sorted add_many" );
        Bitmap bm;
        bm.add_many( values );
        ASSERT_TRUE( matches( bm, values ) );
        expect_contains_everything( bm, p, rng );
        {
            SCOPED_TRACE( "copy" );
            Bitmap const copy{ bm };
            expect_contains_everything( copy, p, rng );
        }
        {
            SCOPED_TRACE( "moved-to" );
            Bitmap moved{ std::move( bm ) };
            expect_contains_everything( moved, p, rng );
        }
    }
    {
        SCOPED_TRACE( "run-optimized" );
        Bitmap bm;
        bm.add_many( values );
        bm.optimize();
        ASSERT_TRUE( matches( bm, values ) );
        expect_contains_everything( bm, p, rng );
    }
    {
        SCOPED_TRACE( "added out of order, one by one" );
        auto shuffled{ values };
        std::ranges::shuffle( shuffled, rng );
        Bitmap bm;
        for ( auto const v : shuffled ) { (void)bm.add( v ); }
        ASSERT_TRUE( matches( bm, values ) );
        expect_contains_everything( bm, p, rng );
        {
            SCOPED_TRACE( "copy" );
            Bitmap const copy{ bm };
            expect_contains_everything( copy, p, rng );
        }
        {
            SCOPED_TRACE( "move-assigned" );
            Bitmap moved;
            moved = std::move( bm );
            expect_contains_everything( moved, p, rng );
        }
    }
    {
        SCOPED_TRACE( "add_many in reverse batches" );
        Bitmap bm;
        auto const batch{ std::max<std::size_t>( 1, values.size() / 7 ) };
        for ( std::size_t end{ values.size() }; end > 0; ) {
            auto const begin{ end > batch ? end - batch : 0 };
            bm.add_many( std::span{ values.data() + begin, end - begin } );
            end = begin;
        }
        // EXPECT: the contains checks below are still meaningful when the content check fails.
        EXPECT_TRUE( matches( bm, values ) );
        expect_contains_everything( bm, p, rng );
    }
    {
        SCOPED_TRACE( "after remove" );
        Bitmap bm;
        bm.add_many( set_union_of( values, extra ) );
        bm.remove_many( extra );
        ASSERT_TRUE( matches( bm, values ) );
        expect_contains_everything( bm, p, rng );
        {
            SCOPED_TRACE( "copy" );
            Bitmap const copy{ bm };
            expect_contains_everything( copy, p, rng );
        }
    }
    {
        // Whole chunks emptied by remove: their members turn into misses.
        SCOPED_TRACE( "after emptying chunks" );
        Bitmap bm;
        bm.add_many( values );
        std::vector<key_type> doomed, kept;
        for ( auto const v : values ) { ( ( ( v / stride ) % 3 == 1 ) ? doomed : kept ).push_back( v ); }
        for ( auto const v : doomed ) { (void)bm.remove( v ); }
        ASSERT_TRUE( matches( bm, kept ) );
        auto const after{ make_probes<key_type>( kept, stride, rng ) };
        expect_contains_everything( bm, after, rng );
        std::size_t wrong{ 0 };
        for ( auto const v : doomed ) { wrong += bm.contains( v ) ? 1 : 0; }
        EXPECT_EQ( wrong, 0U ) << "a removed member is still reported";
    }
}

template <typename Bitmap>
void contains_in_all_states_sweep( std::initializer_list<typename Bitmap::key_type> strides ) {
    for ( auto const stride : strides ) {
        for ( auto const k : { kinds::arrays, kinds::bitsets, kinds::runs, kinds::mixed } ) {
            // Below and above the 512 chunk lookup-index threshold and the 64 chunk limit of the unprefetched search.
            for ( std::size_t const chunks : { 3, 40, 64, 65, 700 } ) {
                if ( static_cast<std::uint64_t>( stride ) == ( std::uint64_t{ 1 } << 40 ) && chunks > 40 && k != kinds::mixed ) { continue; }
                contains_in_all_states_case<Bitmap>( k, chunks, stride );
            }
        }
    }
}

TEST( ContainsStates, Bitmap32 ) {
    contains_in_all_states_sweep<bitmap<std::uint32_t>>( { 1U << 16 } );
}
TEST( ContainsStates, CowBitmap32 ) {
    contains_in_all_states_sweep<CowBitmap32>( { 1U << 16 } );
}
TEST( ContainsStates, LazyCowBitmap32 ) {
    contains_in_all_states_sweep<LazyCowBitmap32>( { 1U << 16 } );
}
TEST( ContainsStates, Bitmap64Adjacent ) {
    contains_in_all_states_sweep<bitmap<std::uint64_t>>( { std::uint64_t{ 1 } << 16 } );
}
TEST( ContainsStates, Bitmap64Wide ) {
    contains_in_all_states_sweep<bitmap<std::uint64_t>>( { std::uint64_t{ 1 } << 32, std::uint64_t{ 1 } << 40 } );
}

} // namespace
