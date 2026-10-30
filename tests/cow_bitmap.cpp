// bitmap-level CowPolicy threading: a refcounted (CoW Model 2) bitmap
// instantiation whose copy constructor is O(#chunks) atomic refcount
// increments instead of a deep copy.
//
// container_handle's own born-shared / write-barrier semantics are already
// exercised directly in cow_semantics.cpp; this file proves the same
// guarantees survive being threaded all the way up through bitmap<>: copies
// share chunk payloads, mutating one copy through the bitmap's public API
// clones only that copy's touched chunk, and the resulting values are
// bitwise identical to the Model-1 (value-semantics) bitmap running the
// same operation sequence.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace {

using CowBitmap   = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>;
using ValueBitmap = frsr::roaring::bitmap<std::uint32_t>;

// Enough values, split across chunk keys, to force at least one chunk's
// container off the container_handle inline body.
[[nodiscard]] std::vector<std::uint32_t> make_values() {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t i{ 0 }; i < 128U; ++i ) {
        values.push_back( i * 3U );          // chunk 0
        values.push_back( 65'536U + i * 5U ); // chunk 1
    }
    return values;
}

TEST( CowBitmap, CopyIsCloneOnWriteNotSharedMutation ) {
    CowBitmap original{ make_values() };
    auto const original_snapshot{ original.to_vector() };

    auto copy{ original };
    ASSERT_TRUE( copy.add( 999'999U ) );

    // The mutation is visible in the copy but the original is untouched —
    // proof that the write barrier cloned the touched chunk rather than the
    // two bitmaps sharing (and one call mutating) the same payload.
    EXPECT_TRUE( copy.contains( 999'999U ) );
    EXPECT_FALSE( original.contains( 999'999U ) );
    EXPECT_EQ( original.to_vector(), original_snapshot );
}

TEST( CowBitmap, RemovalOnCopyLeavesOriginalIntact ) {
    CowBitmap original{ make_values() };
    auto copy{ original };

    ASSERT_TRUE( copy.remove( 0U ) );
    EXPECT_FALSE( copy.contains( 0U ) );
    EXPECT_TRUE( original.contains( 0U ) );
}

TEST( CowBitmap, ChainedCopiesMutateIndependently ) {
    CowBitmap const a{ make_values() };
    auto b{ a };
    auto c{ b };

    ASSERT_TRUE( b.add( 111'111U ) );
    ASSERT_TRUE( c.add( 222'222U ) );

    EXPECT_TRUE ( b.contains( 111'111U ) );
    EXPECT_FALSE( b.contains( 222'222U ) );

    EXPECT_TRUE ( c.contains( 222'222U ) );
    EXPECT_FALSE( c.contains( 111'111U ) );

    EXPECT_FALSE( a.contains( 111'111U ) );
    EXPECT_FALSE( a.contains( 222'222U ) );

    ASSERT_TRUE( b.remove( 3U ) );
    EXPECT_FALSE( b.contains( 3U ) );
    EXPECT_TRUE( c.contains( 3U ) );   // c's own copy is untouched by b's remove
    EXPECT_TRUE( a.contains( 3U ) );
}

TEST( CowBitmap, MixedOpsMatchValueSemanticsBitwiseIdentically ) {
    auto const lhs_values{ make_values() };
    std::vector<std::uint32_t> rhs_values;
    for ( std::uint32_t i{ 0 }; i < 96U; ++i ) {
        rhs_values.push_back( i * 2U );
        rhs_values.push_back( 65'536U + i * 7U );
    }

    CowBitmap cow_lhs{ lhs_values };
    CowBitmap const cow_rhs{ rhs_values };
    ValueBitmap value_lhs{ lhs_values };
    ValueBitmap const value_rhs{ rhs_values };

    // Share cow_lhs before mutating it in place, mirroring the scenario the
    // O(1)-copy instantiation exists for: a reader still holds `shared_copy`.
    auto const shared_copy{ cow_lhs };

    cow_lhs |= cow_rhs;
    value_lhs |= value_rhs;
    EXPECT_EQ( cow_lhs.to_vector(), value_lhs.to_vector() );

    cow_lhs &= cow_rhs;
    value_lhs &= value_rhs;
    EXPECT_EQ( cow_lhs.to_vector(), value_lhs.to_vector() );

    ASSERT_TRUE( cow_lhs.add( 500'000U ) );
    ASSERT_TRUE( value_lhs.add( 500'000U ) );
    EXPECT_EQ( cow_lhs.to_vector(), value_lhs.to_vector() );

    ASSERT_TRUE( cow_lhs.remove( 500'000U ) );
    ASSERT_TRUE( value_lhs.remove( 500'000U ) );
    EXPECT_EQ( cow_lhs.to_vector(), value_lhs.to_vector() );

    cow_lhs -= cow_rhs;
    value_lhs -= value_rhs;
    EXPECT_EQ( cow_lhs.to_vector(), value_lhs.to_vector() );

    // The shared copy taken before the in-place ops above must be unaffected.
    CowBitmap const original_lhs{ lhs_values };
    EXPECT_EQ( shared_copy.to_vector(), original_lhs.to_vector() );
}

// A sorted batch added to a copy: the copy's run containers take it (a batch inside the runs changes nothing, a batch
// outside them is merged into a payload of the copy's own) and the original keeps what it had. Every copy-on-write
// policy with either run policy.
template <typename Bitmap>
class CowBitmapBatch : public ::testing::Test {};

template <typename CowPolicy, typename RunPolicy>
using batch_bitmap = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, CowPolicy, RunPolicy>;
using frsr::roaring::detail::cow_atomic_refcount;
using frsr::roaring::detail::cow_unsynchronized_refcount;
using frsr::roaring::detail::cow_value_semantics;
using frsr::roaring::detail::run_selection_eager;
using frsr::roaring::detail::run_selection_lazy;

using batch_bitmaps = ::testing::Types<
    batch_bitmap<cow_value_semantics        , run_selection_eager>,
    batch_bitmap<cow_value_semantics        , run_selection_lazy >,
    batch_bitmap<cow_atomic_refcount        , run_selection_eager>,
    batch_bitmap<cow_atomic_refcount        , run_selection_lazy >,
    batch_bitmap<cow_unsynchronized_refcount, run_selection_eager>,
    batch_bitmap<cow_unsynchronized_refcount, run_selection_lazy >
>;
TYPED_TEST_SUITE( CowBitmapBatch, batch_bitmaps );

TYPED_TEST( CowBitmapBatch, SortedBatchIntoCopyLeavesOriginalIntact ) {
    TypeParam original;
    for ( std::uint32_t i{ 0 }; i < 60U; ++i ) {
        original.add_closed_range( 100U + i * 50U, 120U + i * 50U );            // chunk 0: sixty runs
        original.add_closed_range( 65'536U + i * 50U, 65'536U + i * 50U + 9U ); // chunk 1: sixty runs
    }
    original.optimize();
    auto const original_snapshot{ original.to_vector() };

    auto copy{ original };
    auto expected{ original_snapshot };

    // Wholly inside existing runs: nothing is added.
    std::vector<std::uint32_t> const inside{ 105U, 110U, 155U, 65'541U };
    copy.add_many_sorted( { inside.data(), inside.size() } );
    EXPECT_EQ( copy.to_vector(), expected );
    EXPECT_EQ( copy.size(), original.size() );

    // Outside them, in both chunks, and in chunks the copy did not have.
    std::vector<std::uint32_t> const outside{ 30U, 125U, 126U, 127U, 3'000U, 65'600U, 65'601U, 70'000U, 1'000'000U };
    expected.insert( expected.end(), outside.begin(), outside.end() );
    std::ranges::sort( expected );
    expected.erase( std::ranges::unique( expected ).begin(), expected.end() );
    copy.add_many_sorted( { outside.data(), outside.size() } );
    EXPECT_EQ( copy.to_vector(), expected );
    EXPECT_EQ( copy.size(), expected.size() );
    EXPECT_EQ( original.to_vector(), original_snapshot );
    EXPECT_EQ( original.size(), original_snapshot.size() );

    // A copy of the changed one, and a batch that fills a new chunk past the array threshold.
    auto const grand{ copy };
    std::vector<std::uint32_t> dense;
    for ( std::uint32_t value{ 2U * 65'536U }; value < 2U * 65'536U + 5'000U; ++value ) {
        dense.push_back( value );
    }
    copy.add_many_sorted( { dense.data(), dense.size() } );
    EXPECT_EQ( copy.size(), expected.size() + dense.size() );
    EXPECT_EQ( grand.to_vector(), expected );
}

} // namespace
