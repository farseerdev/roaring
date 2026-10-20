// The wide-key chunk table against a std::map, through every structural transition: leaf growth,
// splits at the end and in the middle, merges, emptied leaves, root collapse, the rewriter and the
// appenders. Entries are identified by the single value their handle holds.
#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/chunk_tree.hpp>
#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/container_ops.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace frsr::roaring;
using layout = detail::default_layout<std::uint64_t>;
using handle = detail::container_handle<layout>;
using tree   = detail::chunk_tree<handle>;

// A handle that remembers `tag`.
handle make( std::uint16_t const tag ) {
    return handle::make_array_from_sorted( std::span<std::uint16_t const>{ &tag, 1 } );
}
std::uint16_t tag_of( handle const & h ) { return *detail::container_first( h ); }
std::uint16_t tag_for( std::uint64_t const key ) { return static_cast<std::uint16_t>( key * 40503U >> 3 ); }

using model = std::map<std::uint64_t, std::uint16_t>;

void expect_same( tree const & t, model const & m ) {
    ASSERT_EQ( t.validate(), nullptr ) << t.validate();
    ASSERT_EQ( t.size(), m.size() );
    auto at{ t.begin() };
    for ( auto const & [key, tag] : m ) {
        ASSERT_FALSE( at == t.end() );
        ASSERT_EQ( t.key( at ), key );
        ASSERT_EQ( tag_of( t.slot( at ) ), tag );
        ++at;
    }
    ASSERT_TRUE( at == t.end() );
    // and backwards
    auto back{ t.end() };
    for ( auto it{ m.rbegin() }; it != m.rend(); ++it ) {
        back = t.prev( back );
        ASSERT_EQ( t.key( back ), it->first );
    }
    std::size_t counted{ 0 };
    for ( auto const & slot : t.slots() ) { counted += slot.cardinality(); }
    ASSERT_EQ( counted, m.size() );
}

void check_lower_bound( tree const & t, model const & m, std::uint64_t const key ) {
    auto const at{ t.lower_bound( key ) };
    auto const it{ m.lower_bound( key ) };
    if ( it == m.end() ) {
        ASSERT_TRUE( at == t.end() ) << key;
    } else {
        ASSERT_FALSE( at == t.end() ) << key;
        ASSERT_EQ( t.key( at ), it->first ) << key;
    }
}

void put( tree & t, model & m, std::uint64_t const key ) {
    auto const at{ t.lower_bound( key ) };
    if ( at == t.end() || t.key( at ) != key ) {
        auto const placed{ t.insert( at, key, make( tag_for( key ) ) ) };
        ASSERT_EQ( t.key( placed ), key );
        ASSERT_EQ( tag_of( t.slot( placed ) ), tag_for( key ) );
        m[ key ] = tag_for( key );
    }
}

void take( tree & t, model & m, std::uint64_t const key ) {
    auto const at{ t.lower_bound( key ) };
    if ( at != t.end() && t.key( at ) == key ) {
        auto const following{ t.erase( at ) };
        auto const it{ m.upper_bound( key ) };
        if ( it == m.end() ) {
            ASSERT_TRUE( following == t.end() );
        } else {
            ASSERT_FALSE( following == t.end() );
            ASSERT_EQ( t.key( following ), it->first );
        }
        m.erase( key );
    }
}

TEST(ChunkTree, EveryGrowthStepFromEmptyUp) {
    tree t;
    model m;
    for ( std::uint64_t k{ 0 }; k < 400; ++k ) {
        put( t, m, 1000 + 7 * k );
        expect_same( t, m );
    }
    for ( std::uint64_t k{ 0 }; k < 400; ++k ) {
        take( t, m, 1000 + 7 * k );
        expect_same( t, m );
    }
    EXPECT_TRUE( t.empty() );
}

TEST(ChunkTree, DescendingAndRandomInsertionOrders) {
    std::mt19937_64 rng{ 5 };
    for ( int round{ 0 }; round < 3; ++round ) {
        tree t;
        model m;
        std::vector<std::uint64_t> keys;
        for ( std::uint64_t k{ 0 }; k < 5000; ++k ) { keys.push_back( 3 * k + 1 ); }
        if ( round == 0 ) { std::ranges::reverse( keys ); }
        if ( round == 2 ) { std::ranges::shuffle( keys, rng ); }
        for ( auto const k : keys ) { put( t, m, k ); }
        expect_same( t, m );
        std::ranges::shuffle( keys, rng );
        for ( std::size_t i{ 0 }; i < keys.size(); ++i ) {
            take( t, m, keys[ i ] );
            if ( i % 509 == 0 ) { expect_same( t, m ); }
        }
        expect_same( t, m );
        EXPECT_TRUE( t.empty() );
    }
}

TEST(ChunkTree, RandomOperationsAgainstAModel) {
    for ( std::uint64_t const domain : { 40ULL, 700ULL, 30'000ULL, 1ULL << 46 } ) {
        std::mt19937_64 rng{ 11 + domain };
        tree t;
        model m;
        for ( int step{ 0 }; step < 60'000; ++step ) {
            auto const key{ rng() % domain };
            switch ( rng() % 6 ) {
                case 0: case 1: case 2: put( t, m, key ); break;
                case 3: case 4: take( t, m, key ); break;
                default: {
                    check_lower_bound( t, m, key );
                    if ( !m.empty() ) {
                        // the forward form, from a position at or before the answer
                        auto const from{ t.lower_bound( rng() % domain ) };
                        if ( !( from == t.end() ) && t.key( from ) <= key ) {
                            auto const at{ t.lower_bound( key, from ) };
                            auto const it{ m.lower_bound( key ) };
                            ASSERT_EQ( at == t.end(), it == m.end() ) << key;
                            if ( it != m.end() ) { ASSERT_EQ( t.key( at ), it->first ); }
                        }
                    }
                }
            }
            if ( step % 4999 == 0 ) { expect_same( t, m ); }
        }
        expect_same( t, m );
        tree copy;
        copy.copy_from( t );
        expect_same( copy, m );
        tree moved{ std::move( copy ) };
        expect_same( moved, m );
        moved.swap( t );
        expect_same( moved, m );
    }
}

TEST(ChunkTree, LookupsAgreeWithTheModelAtEveryHeight) {
    // sizes that put the tree at one, two, three and four levels of nodes, built by appending and by scattered insertion
    for ( std::uint64_t const count : { 14ULL, 15ULL, 16ULL, 240ULL, 700ULL, 4'000ULL, 12'000ULL, 40'000ULL, 130'000ULL } ) {
        for ( bool const scattered : { false, true } ) {
            std::mt19937_64 rng{ count };
            tree t;
            model m;
            std::vector<std::uint64_t> keys( count );
            for ( std::uint64_t k{ 0 }; k < count; ++k ) { keys[ k ] = 4 * k + 2; }
            if ( scattered ) {
                std::ranges::shuffle( keys, rng );
                for ( auto const k : keys ) { t.insert( t.lower_bound( k ), k, make( tag_for( k ) ) ); m[ k ] = tag_for( k ); }
            } else {
                for ( auto const k : keys ) { t.push_back( k, make( tag_for( k ) ) ); m[ k ] = tag_for( k ); }
            }
            ASSERT_EQ( t.validate(), nullptr ) << count;
            auto const top{ 4 * count + 6 };
            for ( std::uint64_t probe{ 0 }; probe < top; probe += 1 + rng() % 5 ) { check_lower_bound( t, m, probe ); }
            for ( int i{ 0 }; i < 20'000; ++i ) {
                auto const probe{ rng() % ( top + 8 ) };
                check_lower_bound( t, m, probe );
                if ( HasFatalFailure() ) { return; }
            }
            check_lower_bound( t, m, std::uint64_t{ 1 } << 47 );
            // erase a scattered third and look again: separators outlive their keys
            std::ranges::shuffle( keys, rng );
            for ( std::size_t i{ 0 }; i < keys.size() / 3; ++i ) { take( t, m, keys[ i ] ); }
            ASSERT_EQ( t.validate(), nullptr ) << count;
            for ( int i{ 0 }; i < 20'000; ++i ) { check_lower_bound( t, m, rng() % ( top + 8 ) ); }
        }
    }
}

TEST(ChunkTree, AppendersBuildFullLeavesAndAbandonedOpensLeaveNothing) {
    for ( std::uint32_t const count : { 1U, 2U, 4U, 5U, 8U, 9U, 15U, 16U, 30U, 31U, 255U, 256U, 257U, 4000U } ) {
        tree t;
        model m;
        bool moved{ false };
        for ( std::uint64_t k{ 0 }; k < count; ++k ) {
            auto const key{ 10 * k + 3 };
            if ( k % 3 == 0 ) {
                t.push_back( key, make( tag_for( key ) ) );
            } else {
                // an opened slot that is abandoned once, then opened again and committed
                if ( k % 2 == 0 ) { (void)t.open_back( moved ); }
                auto * const slot{ t.open_back( moved ) };
                std::construct_at( slot, make( tag_for( key ) ) );
                t.commit_back( key );
            }
            m[ key ] = tag_for( key );
        }
        expect_same( t, m );
        t.shrink_to_fit();
        expect_same( t, m );
    }
}

// An open_back() that is never committed may leave an empty last leaf attached; every operation
// has to see through it.
TEST(ChunkTree, AnAbandonedOpenLeavesAConsistentTree) {
    for ( std::uint32_t const count : { 0U, 1U, 4U, 8U, 15U, 30U, 45U, 300U } ) {
        tree t;
        model m;
        bool moved{ false };
        for ( std::uint64_t k{ 0 }; k < count; ++k ) {
            auto const key{ 10 * k + 3 };
            t.push_back( key, make( tag_for( key ) ) );
            m[ key ] = tag_for( key );
        }
        (void)t.open_back( moved );                               // abandoned: nothing is committed
        expect_same( t, m );
        check_lower_bound( t, m, 0 );
        check_lower_bound( t, m, 10 * count + 100 );
        put( t, m, 10 * count + 5 );                              // lands in the empty last leaf, or beyond it
        expect_same( t, m );
        (void)t.open_back( moved );
        tree copy;
        copy.copy_from( t );
        expect_same( copy, m );
        t.shrink_to_fit();
        expect_same( t, m );
        (void)t.open_back( moved );
        for ( std::uint64_t k{ 0 }; k < count; k += 2 ) { take( t, m, 10 * k + 3 ); }
        expect_same( t, m );
        {
            auto retire{ []( handle && ) {} };
            (void)t.open_back( moved );
            tree::rewriter<decltype( retire )> rewrite{ t, retire };
            for ( auto at{ t.begin() }; !( at == t.end() ); ++at ) { rewrite.keep( at ); }
            rewrite.finish();
        }
        expect_same( t, m );
    }
}

// The empty last leaf of an abandoned open_back() is reachable from the real leaf before it when
// the entries that made the gap are gone: a search for a key in that gap must still end at end().
TEST(ChunkTree, ASearchPastTheLastEntryIgnoresAnAbandonedLeaf) {
    tree t;
    model m;
    bool moved{ false };
    for ( std::uint64_t k{ 0 }; k < 30; ++k ) {                     // two full leaves
        auto const key{ 10 * k + 3 };
        t.push_back( key, make( tag_for( key ) ) );
        m[ key ] = tag_for( key );
    }
    (void)t.open_back( moved );                                     // abandoned: a third, empty leaf
    for ( std::uint64_t k{ 28 }; k < 30; ++k ) { take( t, m, 10 * k + 3 ); }
    expect_same( t, m );
    for ( std::uint64_t probe{ 10 * 27 + 4 }; probe < 10 * 30 + 20; ++probe ) {
        check_lower_bound( t, m, probe );
        auto const from{ t.lower_bound( 3 ) };
        auto const at{ t.lower_bound( probe, from ) };
        ASSERT_TRUE( at == t.end() ) << probe;
    }
    ASSERT_TRUE( t.lower_bound( 10 * 29 + 3 ) == t.end() );
}

TEST(ChunkTree, RewriterKeepsReplacesAndDropsInKeyOrder) {
    std::mt19937_64 rng{ 21 };
    for ( std::uint32_t const count : { 1U, 3U, 4U, 9U, 15U, 16U, 40U, 300U, 5000U } ) {
        for ( unsigned const drop_per_mille : { 0U, 30U, 500U, 950U, 1000U } ) {
            tree t;
            model m;
            for ( std::uint64_t k{ 0 }; k < count; ++k ) {
                auto const key{ 5 * k + 2 };
                t.push_back( key, make( tag_for( key ) ) );
                m[ key ] = tag_for( key );
            }
            model expected;
            std::size_t retired{ 0 };
            {
                auto retire{ [ & ]( handle && h ) { ++retired; (void)h; } };
                tree::rewriter<decltype( retire )> rewrite{ t, retire };
                for ( auto at{ t.begin() }; !( at == t.end() ); ++at ) {
                    auto const key{ t.key( at ) };
                    auto const roll{ rng() % 1000 };
                    if ( roll < drop_per_mille ) {
                        continue;
                    }
                    if ( roll % 2 == 0 ) {
                        rewrite.keep( at );
                        expected[ key ] = m[ key ];
                    } else {
                        rewrite.put( key, make( static_cast<std::uint16_t>( ~m[ key ] ) ) );
                        expected[ key ] = static_cast<std::uint16_t>( ~m[ key ] );
                    }
                }
                rewrite.finish();
            }
            expect_same( t, expected );
            // the tree is a working tree afterwards
            for ( std::uint64_t k{ 0 }; k < 200; ++k ) { put( t, expected, 3 * k + 1 ); }
            expect_same( t, expected );
        }
    }
}

// Handles that share their payload by reference count: copies of the tree share, and a rewrite or an
// erase of one tree must leave the payloads the other still holds.
TEST(ChunkTree, SharedPayloadsSurviveCopiesRewritesAndErases) {
    using shared_handle = detail::container_handle<layout, detail::cow_atomic_refcount>;
    using shared_tree   = detail::chunk_tree<shared_handle>;
    std::vector<std::uint16_t> values( 48 );
    for ( std::uint16_t i{ 0 }; i < values.size(); ++i ) { values[ i ] = static_cast<std::uint16_t>( 3 * i ); }
    shared_tree original;
    for ( std::uint64_t k{ 0 }; k < 200; ++k ) { original.push_back( 2 * k, shared_handle::make_array_from_sorted( values ) ); }
    shared_tree copy;
    copy.copy_from( original );
    ASSERT_EQ( copy.validate(), nullptr );
    {
        auto retire{ []( shared_handle && ) {} };
        shared_tree::rewriter<decltype( retire )> rewrite{ copy, retire };
        std::uint64_t seen{ 0 };
        for ( auto at{ copy.begin() }; !( at == copy.end() ); ++at, ++seen ) {
            if ( seen % 3 == 0 ) { continue; }
            if ( seen % 3 == 1 ) { rewrite.keep( at ); } else { rewrite.put( copy.key( at ), shared_handle::make_array_from_sorted( values ) ); }
        }
        rewrite.finish();
    }
    ASSERT_EQ( copy.validate(), nullptr );
    for ( std::uint64_t k{ 0 }; k < 200; k += 2 ) { copy.erase( copy.lower_bound( 2 * k ) ); }
    ASSERT_EQ( copy.validate(), nullptr );
    ASSERT_EQ( original.validate(), nullptr );
    ASSERT_EQ( original.size(), 200U );
    for ( auto const & slot : original.slots() ) {
        ASSERT_EQ( slot.cardinality(), values.size() );
        ASSERT_EQ( *detail::container_first( slot ), 0 );
    }
}

TEST(ChunkTree, NodeBytesPerChunk) {
    std::printf( "[ bytes   ] sizeof( bitmap<uint64_t> ) = %zu, sizeof( bitmap<uint32_t> ) = %zu\n", sizeof( frsr::roaring::bitmap<std::uint64_t> ), sizeof( frsr::roaring::bitmap<std::uint32_t> ) );
    for ( std::uint64_t const count : { 1ULL, 4ULL, 15ULL, 1000ULL, 100'000ULL } ) {
        tree ascending;
        for ( std::uint64_t k{ 0 }; k < count; ++k ) { ascending.push_back( 2 * k, make( 1 ) ); }
        std::mt19937_64 rng{ 3 };
        std::vector<std::uint64_t> keys( count );
        for ( std::uint64_t k{ 0 }; k < count; ++k ) { keys[ k ] = 2 * k; }
        std::ranges::shuffle( keys, rng );
        tree scattered;
        for ( auto const k : keys ) { scattered.insert( scattered.lower_bound( k ), k, make( 1 ) ); }
        ASSERT_EQ( scattered.validate(), nullptr );
        std::printf( "[ bytes   ] %8llu chunks: ascending %6.1f B/chunk, scattered %6.1f B/chunk\n", static_cast<unsigned long long>( count ),
            static_cast<double>( ascending.node_bytes() ) / static_cast<double>( count ),
            static_cast<double>( scattered.node_bytes() ) / static_cast<double>( count ) );
    }
}

} // namespace
