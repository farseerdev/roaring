// run_ref::add_sorted - the group insert of a sorted, duplicate-free batch into a run container -
// against a std::set oracle, under every copy-on-write policy (and with or without the small-payload
// optimization, per the build). Contents, cardinality, endpoints and the run normal form (sorted,
// non-adjacent, non-overlapping) are all checked after each batch.

#include <frsr/roaring/container_handle.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring::detail;

using layout   = default_layout<std::uint32_t>;
using low_type = layout::low_type;
using values_t = std::vector<low_type>;
using ranges_t = std::vector<std::pair<low_type, low_type>>;

template <typename CowPolicy>
class RunAddSorted : public ::testing::Test {
protected:
    using handle_t = container_handle<layout, CowPolicy>;

    // A run container over the given closed ranges.
    static handle_t make( ranges_t const & ranges ) {
        auto handle{ handle_t::make_run() };
        auto run{ handle.as_run() };
        for ( auto const & [ begin, end ] : ranges ) {
            run.add_closed_range( begin, end );
        }
        return handle;
    }

    static std::set<low_type> contents_of( handle_t const & handle ) {
        std::set<low_type> result;
        handle.as_run().for_each( [ & ]( low_type const value ) { result.insert( value ); } );
        return result;
    }

    // Adds the batch and checks everything against the oracle; returns the added count.
    static std::size_t add_and_check( handle_t & handle, values_t const & batch ) {
        auto oracle{ contents_of( handle ) };
        auto const expected_added{ static_cast<std::size_t>( std::ranges::count_if( batch, [ & ]( low_type const v ) { return !oracle.contains( v ); } ) ) };
        oracle.insert( batch.begin(), batch.end() );

        auto const added{ handle.as_run().add_sorted( { batch.data(), batch.size() } ) };
        EXPECT_EQ( added, expected_added );
        check( handle, oracle );
        return added;
    }

    static void check( handle_t const & handle, std::set<low_type> const & oracle ) {
        EXPECT_EQ( contents_of( handle ), oracle );
        EXPECT_EQ( handle.cardinality(), oracle.size() );
        auto const runs{ handle.as_run().runs };
        std::uint32_t total{ 0 };
        for ( std::uint32_t i{ 0 }; i < runs.size(); ++i ) {
            EXPECT_LE( runs[ i ].begin, runs[ i ].end );
            if ( i != 0 ) {
                EXPECT_LT( std::size_t{ runs[ i - 1 ].end } + 1U, std::size_t{ runs[ i ].begin } ) << "runs " << i - 1 << '/' << i << " not normalized";
            }
            total += static_cast<std::uint32_t>( runs[ i ].end - runs[ i ].begin + 1U );
        }
        EXPECT_EQ( total, oracle.size() );
        if ( !oracle.empty() ) {
            EXPECT_EQ( handle.min_value(), *oracle.begin() );
            EXPECT_EQ( handle.max_value(), *oracle.rbegin() );
        }
    }
};

using all_policies = ::testing::Types<cow_value_semantics, cow_atomic_refcount, cow_unsynchronized_refcount>;
TYPED_TEST_SUITE( RunAddSorted, all_policies );

TYPED_TEST( RunAddSorted, EmptyBatchChangesNothing ) {
    auto handle{ TestFixture::make( { { 10, 20 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, {} ), 0U );
}

TYPED_TEST( RunAddSorted, IntoAnEmptyRunContainer ) {
    auto handle{ TestFixture::make( {} ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, { 5, 6, 7, 20, 40, 41 } ), 6U );
}

TYPED_TEST( RunAddSorted, FillsGapsWithoutMergingNeighbours ) {
    auto handle{ TestFixture::make( { { 10, 20 }, { 40, 50 }, { 100, 110 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, { 25, 30, 70, 120 } ), 4U );
}

TYPED_TEST( RunAddSorted, BatchBridgesRuns ) {
    auto handle{ TestFixture::make( { { 10, 20 }, { 24, 30 }, { 34, 40 }, { 60, 70 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, { 21, 22, 23, 31, 32, 33 } ), 6U );
    EXPECT_EQ( handle.count(), 2U );
}

TYPED_TEST( RunAddSorted, ExtendsTheFirstAndLastRun ) {
    auto handle{ TestFixture::make( { { 100, 110 }, { 200, 210 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, { 98, 99 } ), 2U );
    EXPECT_EQ( TestFixture::add_and_check( handle, { 211, 212, 213 } ), 3U );
    EXPECT_EQ( TestFixture::add_and_check( handle, { 0, 65'535 } ), 2U );
    EXPECT_EQ( handle.count(), 4U );
}

TYPED_TEST( RunAddSorted, ValuesAlreadyPresentAddNothing ) {
    auto handle{ TestFixture::make( { { 10, 20 }, { 40, 50 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( handle, { 10, 15, 20 } ), 0U );
    EXPECT_EQ( TestFixture::add_and_check( handle, { 12, 18, 40, 50 } ), 0U );
    EXPECT_EQ( TestFixture::add_and_check( handle, { 20, 21, 40, 41, 45, 51, 52 } ), 3U );
}

TYPED_TEST( RunAddSorted, FullRange ) {
    values_t all( 65'536 );
    for ( std::uint32_t i{ 0 }; i < all.size(); ++i ) { all[ i ] = static_cast<low_type>( i ); }

    auto empty{ TestFixture::make( {} ) };
    EXPECT_EQ( TestFixture::add_and_check( empty, all ), 65'536U );
    EXPECT_EQ( empty.count(), 1U );

    auto sparse{ TestFixture::make( { { 3, 3 }, { 500, 600 }, { 65'535, 65'535 } } ) };
    EXPECT_EQ( TestFixture::add_and_check( sparse, all ), 65'536U - 103U );
    EXPECT_EQ( sparse.count(), 1U );

    EXPECT_EQ( TestFixture::add_and_check( sparse, all ), 0U );
}

TYPED_TEST( RunAddSorted, WritesThroughASharedCopyOnlyOnTheWrittenHandle ) {
    auto original{ TestFixture::make( { { 10, 20 }, { 40, 50 }, { 70, 80 }, { 100, 110 }, { 130, 140 }, { 160, 170 } } ) };
    auto const before{ TestFixture::contents_of( original ) };
    auto copy{ original };
    EXPECT_EQ( TestFixture::add_and_check( copy, { 21, 60, 61, 150 } ), 4U );
    TestFixture::check( original, before );
}

// A batch whose every value is already inside some run opens no run: nothing is reserved, however many
// groups the batch has.
TYPED_TEST( RunAddSorted, AllPresentBatchAllocatesNothing ) {
    auto handle{ TestFixture::make( { { 0, 30'000 }, { 30'002, 65'535 } } ) };
    values_t evens;
    for ( std::uint32_t value{ 0 }; value < 65'536U; value += 2 ) { evens.push_back( static_cast<low_type>( value ) ); }

    auto const runs{ handle.as_run().runs };
    auto const capacity{ runs.capacity() };
    auto const data    { runs.data() };
    EXPECT_EQ( TestFixture::add_and_check( handle, evens ), 0U );
    EXPECT_EQ( handle.count(), 2U );
    EXPECT_EQ( handle.as_run().runs.capacity(), capacity );
    EXPECT_EQ( handle.as_run().runs.data(), data );
}

// Only groups outside the existing runs are reserved for.
TYPED_TEST( RunAddSorted, ReservesOnlyForGroupsOutsideTheRuns ) {
    auto handle{ TestFixture::make( { { 0, 3'000 }, { 5'000, 9'000 } } ) };
    values_t every_other;
    for ( std::uint32_t value{ 0 }; value <= 9'000U; value += 2 ) { every_other.push_back( static_cast<low_type>( value ) ); }
    auto const capacity{ handle.as_run().runs.capacity() };
    auto const required{ 2U + 999U };
    TestFixture::add_and_check( handle, every_other );
    // one growth step, sized by the groups that open runs and not by all of the batch's groups
    EXPECT_EQ( handle.as_run().runs.capacity(), std::max( required, 2U * capacity ) );
}

// A result that fits the payload's capacity is written in place; one that does not grows it exactly once.
TYPED_TEST( RunAddSorted, GrowsAtMostOncePerBatch ) {
    auto handle{ TestFixture::make( { { 10, 20 }, { 40, 50 } } ) };
    handle.as_run().runs.reserve( 8 );
    auto const capacity{ handle.as_run().runs.capacity() };
    auto const data    { handle.as_run().runs.data() };
    ASSERT_GE( capacity, 8U );

    values_t const fits( { 1, 3, 5, 25, 60, 62 } );
    TestFixture::add_and_check( handle, fits );
    EXPECT_EQ( handle.as_run().runs.capacity(), capacity );
    EXPECT_EQ( handle.as_run().runs.data(), data );

    values_t exceeds;
    for ( std::uint32_t value{ 100 }; value < 100U + 2U * ( capacity + 5U ); value += 2 ) { exceeds.push_back( static_cast<low_type>( value ) ); }
    auto const required{ handle.count() + static_cast<std::uint32_t>( exceeds.size() ) };
    TestFixture::add_and_check( handle, exceeds );
    EXPECT_EQ( handle.as_run().runs.capacity(), std::max( required, 2U * capacity ) );
}

// Random batches against random run lists: sparse, clustered and dense shapes, so batches land inside,
// between, around and beyond runs, over both the inline and the spilled payload.
TYPED_TEST( RunAddSorted, MatchesTheOracleOnRandomBatches ) {
    std::mt19937 rng{ 0x5EED };
    for ( int trial{ 0 }; trial < 400; ++trial ) {
        auto const universe{ std::array<std::uint32_t, 4>{ 64U, 400U, 5'000U, 65'536U }[ rng() % 4 ] };
        auto const offset  { rng() % ( 65'536U - universe + 1U ) };
        auto draw{ [ & ]() { return static_cast<low_type>( offset + rng() % universe ); } };

        ranges_t ranges;
        for ( auto n{ rng() % 40 }; n != 0; --n ) {
            auto const begin{ draw() };
            ranges.emplace_back( begin, static_cast<low_type>( std::min<std::uint32_t>( 65'535U, begin + rng() % 6 ) ) );
        }
        auto handle{ TestFixture::make( ranges ) };

        for ( int round{ 0 }; round < 3; ++round ) {
            std::set<low_type> batch_set;
            for ( auto n{ rng() % ( 1U + universe / 2U ) }; n != 0; --n ) { batch_set.insert( draw() ); }
            for ( auto n{ rng() % 4 }; n != 0; --n ) {
                auto const begin{ draw() };
                for ( auto i{ 0U }, length{ static_cast<std::uint32_t>( rng() % 20 ) }; i < length && begin + i <= 65'535U; ++i ) {
                    batch_set.insert( static_cast<low_type>( begin + i ) );
                }
            }
            values_t const batch{ batch_set.begin(), batch_set.end() };
            SCOPED_TRACE( ::testing::Message() << "trial " << trial << " round " << round );
            TestFixture::add_and_check( handle, batch );
            if ( ::testing::Test::HasFailure() ) { return; }
        }
    }
}

} // namespace
