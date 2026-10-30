// The wide-key chunk table against a std::map, through every structural transition: leaf growth,
// splits at the end and in the middle, merges, emptied leaves, root collapse, the rewriter and the
// appenders. Entries are identified by the single value their handle holds.
#include "allocation_fault.hpp"

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/chunk_tree.hpp>
#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/container_ops.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <new>
#include <random>
#include <span>
#include <thread>
#include <vector>


namespace {

using namespace frsr::roaring;
namespace alloc_fault = frsr::roaring::kernel_test::alloc_fault;
using layout = detail::default_layout<std::uint64_t>;
using handle = detail::container_handle<layout>;
using tree   = detail::chunk_tree<handle>;
// The same table with the flat directory from 16 chunks on, so that the small tables of these tests run on it too.
using eager_tree = detail::chunk_tree<handle, 16>;

template <typename Tree>
class ChunkTreeTyped : public ::testing::Test {};
using tree_types = ::testing::Types<tree, eager_tree>;
TYPED_TEST_SUITE( ChunkTreeTyped, tree_types );

// A handle that remembers `tag`.
handle make( std::uint16_t const tag ) {
    return handle::make_array_from_sorted( std::span<std::uint16_t const>{ &tag, 1 } );
}
std::uint16_t tag_of( handle const & h ) { return detail::container_first( h ); }
std::uint16_t tag_for( std::uint64_t const key ) { return static_cast<std::uint16_t>( key * 40503U >> 3 ); }

using model = std::map<std::uint64_t, std::uint16_t>;

void expect_same( auto const & t, model const & m ) {
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

void check_lower_bound( auto const & t, model const & m, std::uint64_t const key ) {
    auto const at{ t.lower_bound( key ) };
    auto const it{ m.lower_bound( key ) };
    if ( it == m.end() ) {
        ASSERT_TRUE( at == t.end() ) << key;
    } else {
        ASSERT_FALSE( at == t.end() ) << key;
        ASSERT_EQ( t.key( at ), it->first ) << key;
    }
}

void put( auto & t, model & m, std::uint64_t const key ) {
    auto const at{ t.lower_bound( key ) };
    if ( at == t.end() || t.key( at ) != key ) {
        auto const placed{ t.insert( at, key, make( tag_for( key ) ) ) };
        ASSERT_EQ( t.key( placed ), key );
        ASSERT_EQ( tag_of( t.slot( placed ) ), tag_for( key ) );
        m[ key ] = tag_for( key );
    }
}

void take( auto & t, model & m, std::uint64_t const key ) {
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

TYPED_TEST(ChunkTreeTyped, EveryGrowthStepFromEmptyUp) {
    using tree = TypeParam;
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

TYPED_TEST(ChunkTreeTyped, DescendingAndRandomInsertionOrders) {
    using tree = TypeParam;
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

TYPED_TEST(ChunkTreeTyped, RandomOperationsAgainstAModel) {
    using tree = TypeParam;
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

TYPED_TEST(ChunkTreeTyped, LookupsAgreeWithTheModelAtEveryHeight) {
    using tree = TypeParam;
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
                if ( ::testing::Test::HasFatalFailure() ) { return; }
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

TYPED_TEST(ChunkTreeTyped, AppendersBuildFullLeavesAndAbandonedOpensLeaveNothing) {
    using tree = TypeParam;
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
TYPED_TEST(ChunkTreeTyped, AnAbandonedOpenLeavesAConsistentTree) {
    using tree = TypeParam;
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
            typename tree::template rewriter<decltype( retire )> rewrite{ t, retire };
            for ( auto at{ t.begin() }; !( at == t.end() ); ++at ) { rewrite.keep( at ); }
            rewrite.finish();
        }
        expect_same( t, m );
    }
}

// The empty last leaf of an abandoned open_back() is reachable from the real leaf before it when
// the entries that made the gap are gone: a search for a key in that gap must still end at end().
TYPED_TEST(ChunkTreeTyped, ASearchPastTheLastEntryIgnoresAnAbandonedLeaf) {
    using tree = TypeParam;
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

TYPED_TEST(ChunkTreeTyped, RewriterKeepsReplacesAndDropsInKeyOrder) {
    using tree = TypeParam;
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
                typename tree::template rewriter<decltype( retire )> rewrite{ t, retire };
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
        ASSERT_EQ( detail::container_first( slot ), 0 );
    }
}

// Key patterns that spread unevenly: the lookup structure the table picks for evenly spread keys has to
// stay correct when the spread changes under it, whatever it is at that moment.
TYPED_TEST(ChunkTreeTyped, UnevenKeyPatternsKeepLookupsAndOrderCorrect) {
    using tree = TypeParam;
    std::mt19937_64 rng{ 77 };
    auto const probe_all = [&] ( auto const & t, model const & m, std::uint64_t const top ) {
        for ( int i{ 0 }; i < 3000; ++i ) { check_lower_bound( t, m, rng() % ( top + 16 ) ); }
        check_lower_bound( t, m, 0 );
        check_lower_bound( t, m, std::uint64_t{ 1 } << 47 );
    };
    // evenly spread keys, then a dense cluster inside one gap, then a sparse run far above
    {
        tree t;
        model m;
        for ( std::uint64_t k{ 0 }; k < 20'000; ++k ) { t.push_back( 4096 * k, make( tag_for( 4096 * k ) ) ); m[ 4096 * k ] = tag_for( 4096 * k ); }
        expect_same( t, m );
        EXPECT_TRUE( t.indexed() );
        for ( std::uint64_t k{ 1 }; k < 4000; ++k ) { put( t, m, 4096 * 7000 + k ); }
        expect_same( t, m );
        probe_all( t, m, 4096 * 20'000 );
        for ( std::uint64_t k{ 0 }; k < 3000; ++k ) { put( t, m, ( std::uint64_t{ 1 } << 40 ) + k * ( std::uint64_t{ 1 } << 22 ) ); }
        expect_same( t, m );
        probe_all( t, m, std::uint64_t{ 1 } << 41 );
        // the front goes first, then every other entry of the rest
        std::size_t dropped{ 0 };
        for ( auto it{ m.begin() }; it != m.end() && dropped < 9000; ) {
            auto const key{ it->first };
            ++it;
            take( t, m, key );
            ++dropped;
        }
        expect_same( t, m );
        bool every_other{ false };
        for ( auto it{ m.begin() }; it != m.end(); every_other = !every_other ) {
            auto const key{ it->first };
            ++it;
            if ( every_other ) { take( t, m, key ); }
        }
        expect_same( t, m );
        probe_all( t, m, std::uint64_t{ 1 } << 41 );
    }
    // descending insertion below everything held, over a base built evenly
    {
        tree t;
        model m;
        for ( std::uint64_t k{ 0 }; k < 6000; ++k ) { t.push_back( 1'000'000 + 8 * k, make( tag_for( 1'000'000 + 8 * k ) ) ); m[ 1'000'000 + 8 * k ] = tag_for( 1'000'000 + 8 * k ); }
        for ( std::uint64_t k{ 999'999 }; k > 600'000; k -= 3 ) { put( t, m, k ); }
        expect_same( t, m );
        probe_all( t, m, 1'100'000 );
    }
    // a mix of everything against a model
    {
        tree t;
        model m;
        for ( int step{ 0 }; step < 120'000; ++step ) {
            std::uint64_t key;
            switch ( rng() % 4 ) {
                case 0: key = 5 * ( rng() % 40'000 ); break;                                     // even spread
                case 1: key = 90'000 + rng() % 600; break;                                      // dense cluster
                case 2: key = ( std::uint64_t{ 1 } << 30 ) + ( rng() % 5000 ) * 77'777; break;  // far, sparse
                default: key = rng() % 300'000; break;
            }
            if ( rng() % 5 < 3 ) { put( t, m, key ); } else { take( t, m, key ); }
            if ( step % 9973 == 0 ) { expect_same( t, m ); }
        }
        expect_same( t, m );
        probe_all( t, m, std::uint64_t{ 1 } << 31 );
    }
}

// Several dense runs far apart are not one even spread: the table stays on the tree, however it is
// built and edited, and does not move between the two indexes.
TYPED_TEST(ChunkTreeTyped, SeveralDenseClustersStayOnTheTree) {
    using tree = TypeParam;
    constexpr std::uint64_t bases[]{ 0, std::uint64_t{ 1 } << 34, std::uint64_t{ 1 } << 40, ( std::uint64_t{ 1 } << 44 ) + 12345 };
    std::mt19937_64 rng{ 3 };
    tree t;
    model m;
    for ( auto const base : bases ) {
        for ( std::uint64_t k{ 0 }; k < 3000; ++k ) { t.push_back( base + 2 * k, make( tag_for( base + 2 * k ) ) ); m[ base + 2 * k ] = tag_for( base + 2 * k ); }
        expect_same( t, m );
        if ( base != bases[ 0 ] ) { ASSERT_FALSE( t.indexed() ); }   // (the first run alone is an even spread)
    }
    for ( int i{ 0 }; i < 20'000; ++i ) {
        auto const key{ bases[ rng() % 4 ] + rng() % 6000 };
        if ( rng() % 3 != 0 ) { put( t, m, key ); } else { take( t, m, key ); }
        if ( i % 4001 == 0 ) {
            expect_same( t, m );
            ASSERT_FALSE( t.indexed() );
        }
    }
    expect_same( t, m );
    ASSERT_FALSE( t.indexed() );
    tree copy;
    copy.copy_from( t );
    expect_same( copy, m );
    ASSERT_FALSE( copy.indexed() );
    for ( int i{ 0 }; i < 20'000; ++i ) { check_lower_bound( t, m, bases[ rng() % 4 ] + rng() % 8000 ); }
}

// A table that was on the directory and has shrunk gives the table back: the directory is rebuilt
// narrower while it still suits the keys, and dropped once the table is small.
TYPED_TEST(ChunkTreeTyped, AShrinkingTableDoesNotKeepAnOversizedDirectory) {
    using tree = TypeParam;
    std::mt19937_64 rng{ 41 };
    tree t;
    model m;
    for ( std::uint64_t k{ 0 }; k < 100'000; ++k ) { t.push_back( 4 * k, make( tag_for( 4 * k ) ) ); m[ 4 * k ] = tag_for( 4 * k ); }
    ASSERT_TRUE( t.indexed() );
    auto const grown{ t.directory_bytes() };
    std::vector<std::uint64_t> keys;
    for ( auto const & [key, tag] : m ) { keys.push_back( key ); }
    std::ranges::shuffle( keys, rng );
    for ( std::size_t i{ 0 }; i < keys.size() - 1000; ++i ) {
        take( t, m, keys[ i ] );
        if ( i % 7919 == 0 ) {
            ASSERT_EQ( t.validate(), nullptr ) << t.validate();
            ASSERT_LE( t.directory_bytes(), ( t.size() + 40 ) * sizeof( void * ) ) << t.size();
        }
    }
    expect_same( t, m );
    ASSERT_EQ( t.size(), 1000U );
    EXPECT_LE( t.directory_bytes(), ( t.size() + 40 ) * sizeof( void * ) );
    EXPECT_LT( t.directory_bytes(), grown / 20 );
    for ( int i{ 0 }; i < 5000; ++i ) { check_lower_bound( t, m, rng() % 400'000 ); }
    // and it takes the table's growth again
    for ( std::uint64_t k{ 0 }; k < 60'000; ++k ) { put( t, m, 4 * k + 1 ); }
    expect_same( t, m );
}

// A table on the directory is read through its table and the leaf chain only, so any number of
// threads may read it at once (the test is meant to run under ThreadSanitizer as well).
TYPED_TEST(ChunkTreeTyped, ConcurrentReadersOfADirectoryTable) {
    using tree = TypeParam;
    tree t;
    std::vector<std::uint64_t> keys;
    for ( std::uint64_t k{ 0 }; k < 60'000; ++k ) { keys.push_back( 6 * k + 2 ); t.push_back( keys.back(), make( tag_for( keys.back() ) ) ); }
    ASSERT_TRUE( t.indexed() );
    constexpr unsigned readers{ 8 };
    std::atomic<bool> go{ false };
    std::atomic<unsigned> bad{ 0 };
    std::vector<std::thread> threads;
    for ( unsigned id{ 0 }; id < readers; ++id ) {
        threads.emplace_back( [&, id] {
            std::mt19937_64 rng{ id + 1 };
            while ( !go.load( std::memory_order_acquire ) ) {}
            for ( int i{ 0 }; i < 150'000; ++i ) {
                auto const probe{ rng() % ( 6 * 60'000 + 40 ) };
                auto const expected{ std::ranges::lower_bound( keys, probe ) };
                auto const at{ t.lower_bound( probe ) };
                if ( expected == keys.end() ? !( at == t.end() ) : ( at == t.end() || t.key( at ) != *expected || tag_of( t.slot( at ) ) != tag_for( *expected ) ) ) { ++bad; }
                if ( i % 1000 == 0 ) {
                    std::size_t seen{ 0 };
                    for ( auto it{ t.lower_bound( probe ) }; !( it == t.end() ) && seen < 50; ++it ) { ++seen; }
                    if ( seen > 50 ) { ++bad; }
                }
            }
        } );
    }
    go.store( true, std::memory_order_release );
    for ( auto & thread : threads ) { thread.join(); }
    EXPECT_EQ( bad.load(), 0U );
}

// An allocation that fails leaves the table valid and exactly as it was before the call that failed,
// through adoption, growth, giving the directory up, shrinking and the bulk builds.
TYPED_TEST(ChunkTreeTyped, AnAllocationFailureLeavesTheTableAsItWas) {
    using tree = TypeParam;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations of the table cannot be made to fail in this build";
#else
    for ( std::uint32_t const period : { 2U, 3U, 5U, 11U, 41U } ) {
        std::mt19937_64 rng{ period };
        tree t;
        model m;
        std::uint32_t failed{ 0 };
        auto const attempt = [&] ( auto && operation ) -> bool {
            alloc_fault::armed const armed{ period };
            try {
                operation();
                return true;
            } catch ( std::bad_alloc const & ) {
                ++failed;
                return false;
            }
        };
        auto const insert_key = [&] ( std::uint64_t const key ) {
            auto const at{ t.lower_bound( key ) };
            if ( !( at == t.end() ) && t.key( at ) == key ) { return; }
            auto h{ make( tag_for( key ) ) };
            if ( attempt( [&] { (void)t.insert( at, key, std::move( h ) ); } ) ) { m[ key ] = tag_for( key ); }
        };
        auto const erase_key = [&] ( std::uint64_t const key ) {
            auto const at{ t.lower_bound( key ) };
            if ( at == t.end() || t.key( at ) != key ) { return; }
            if ( attempt( [&] { (void)t.erase( at ); } ) ) { m.erase( key ); }
        };
        auto const append_key = [&] ( std::uint64_t const key, bool const open ) {
            auto h{ make( tag_for( key ) ) };
            bool committed{ false };
            if ( open ) {
                handle * slot{ nullptr };
                bool moved{ false };
                if ( !attempt( [&] { slot = t.open_back( moved ); } ) ) { return; }
                std::construct_at( slot, std::move( h ) );
                committed = attempt( [&] { t.commit_back( key ); } );
                if ( !committed ) { *slot = handle{}; }
            } else {
                committed = attempt( [&] { t.push_back( key, std::move( h ) ); } );
            }
            if ( committed ) { m[ key ] = tag_for( key ); }
        };
        auto const verify = [&] {
            ASSERT_EQ( t.validate(), nullptr ) << t.validate();
            expect_same( t, m );
        };
        // an evenly spread table, built by appends across the point where it moves to the directory
        for ( std::uint64_t k{ 0 }; k < 5000; ++k ) { append_key( 8 * k, k % 3 == 0 ); }
        verify();
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        // a dense run inside one gap, which overfills buckets and forces the directory to be given up
        for ( std::uint64_t k{ 1 }; k < 1500; ++k ) { insert_key( 8 * 2500 + k % 8 + 8 * 20'000 + 8 * ( k / 8 ) ); }
        verify();
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        for ( std::uint64_t k{ 0 }; k < 5000; ++k ) { insert_key( 8 * k + 1 + 2 * ( rng() % 3 ) ); }
        // keys far above the rest, past what the table can cover
        for ( std::uint64_t k{ 0 }; k < 400; ++k ) { append_key( ( std::uint64_t{ 1 } << 40 ) + 977 * k, false ); }
        verify();
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        // erase most of it, scattered
        std::vector<std::uint64_t> keys;
        for ( auto const & [key, tag] : m ) { keys.push_back( key ); }
        std::ranges::shuffle( keys, rng );
        for ( std::size_t i{ 0 }; i < keys.size() * 9 / 10; ++i ) { erase_key( keys[ i ] ); }
        verify();
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        // a copy either is one or is nothing
        {
            tree copy;
            bool made{ false };
            for ( int tries{ 0 }; tries < 50 && !made; ++tries ) { made = attempt( [&] { copy.copy_from( t ); } ); }
            if ( made ) { expect_same( copy, m ); } else { ASSERT_TRUE( copy.empty() ); }
            ASSERT_EQ( copy.validate(), nullptr ) << copy.validate();
        }
        // an abandoned open is dropped, or not, by shrink_to_fit()
        { bool moved{ false }; (void)t.open_back( moved ); }
        while ( !attempt( [&] { t.shrink_to_fit(); } ) ) {}
        verify();
        EXPECT_GT( failed, 0U ) << "no allocation failed at period " << period;
    }
#endif
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
        tree descending;
        for ( std::uint64_t k{ count }; k-- > 0; ) { descending.insert( descending.lower_bound( 2 * k ), 2 * k, make( 1 ) ); }
        ASSERT_EQ( descending.validate(), nullptr );
        if ( count >= 1000 ) {
            // a table built from the largest key down fills its leaves like one built upward does
            EXPECT_LE( descending.node_bytes(), ascending.node_bytes() + ascending.node_bytes() / 10 ) << count;
        }
        std::printf( "[ bytes   ] %8llu chunks: ascending %6.1f B/chunk, descending %6.1f B/chunk, scattered %6.1f B/chunk\n", static_cast<unsigned long long>( count ),
            static_cast<double>( ascending.node_bytes() ) / static_cast<double>( count ),
            static_cast<double>( descending.node_bytes() ) / static_cast<double>( count ),
            static_cast<double>( scattered.node_bytes() ) / static_cast<double>( count ) );
    }
}

} // namespace
