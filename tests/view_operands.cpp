// A view of a finished bitmap as the operand of a set operation (typed_view, interop_operands.hpp).
//
// What a view is for is reading a bitmap that other threads read too, so these tests pin what an operation reading
// through one WRITES: nothing the viewed bitmap owns (no reference count, no container header, no payload, no
// chunk table), and what a result KEEPS of it: nothing, so a result outlives the viewed bitmap. The answers
// themselves are checked against a sorted-vector oracle, as the other interop tests do (interop.cpp has the
// per-operation matrix; this file has the properties that matrix cannot see).

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/bitmap_view.hpp>

#include <gtest/gtest.h>

#include "interop_support.hpp"
#include "view_support.hpp"

#include <array>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::interop_test;
using frsr::roaring::detail::bitmap_test_access;

template <typename Key, typename Cow, typename Run = detail::run_selection_eager>
using bitmap_of = bitmap<Key, default_container_set<Key>, Cow, Run>;

// The policies under which a copy of a bitmap shares its payloads by reference count: where a view's promise
// (it writes no count) means something.
using counted_bitmaps = ::testing::Types<
    bitmap_of<std::uint16_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount, detail::run_selection_lazy>,
    bitmap_of<std::uint32_t, detail::cow_unsynchronized_refcount>,
    bitmap_of<std::uint64_t, detail::cow_atomic_refcount>
>;

template <typename Bitmap>
class ViewWitness : public ::testing::Test {};
TYPED_TEST_SUITE( ViewWitness, counted_bitmaps );

template <typename Bitmap>
oracle_set contents_of( Bitmap const & bitmap ) {
    auto const values{ bitmap.to_vector() };
    return { values.begin(), values.end() };
}

template <typename Bitmap>
void expect_equals( Bitmap const & bitmap, oracle_set const & expected, std::string const & what ) {
    ASSERT_EQ( bitmap.validate(), nullptr ) << what;
    ASSERT_EQ( contents_of( bitmap ), expected ) << what;
}

// A bitmap as a consumer publishes one: the union of several, folded lazily and finished once, which leaves bitset
// endpoints stale and (as the other-only chunks are spliced in) shares payloads with the parts until they are gone.
template <typename Bitmap>
Bitmap publish( std::mt19937_64 & rng, std::size_t const parts_count = 4 ) {
    std::vector<Bitmap> parts;
    for ( std::size_t i{ 0 }; i < parts_count; ++i ) { parts.push_back( make_bitmap<Bitmap>( random_content<Bitmap>( rng ), i % 2 == 1 ) ); }
    Bitmap published;
    for ( auto const & part : parts ) { published.bulk_or_intermediate( part ); }
    published.bulk_or_finish();
    return published;   // the parts die here: the payloads they shared with it are the bitmap's alone again
}

template <typename Bitmap>
std::size_t stale_endpoint_containers( Bitmap const & bitmap ) {
    std::size_t count{ 0 };
    for ( auto const & state : bitmap_test_access<Bitmap>::snapshot( bitmap ) ) { count += !state.endpoints_valid; }
    return count;
}

// Every operation that takes a view, on receivers private to the test, with the viewed bitmap's state captured
// before and compared after. The results are kept, the viewed bitmap is destroyed, and the results are read.
template <typename Bitmap>
void exercise_views_of( std::unique_ptr<Bitmap> viewed, std::mt19937_64 & rng, std::string const & what ) {
    using access = bitmap_test_access<Bitmap>;
    auto const members{ contents_of( *viewed ) };
    auto const before{ access::snapshot( *viewed ) };
    for ( auto const & state : before ) { ASSERT_LE( state.refcount, 1U ) << what << ": the bitmap to publish shares a payload"; }

    std::vector<std::pair<Bitmap, oracle_set>> kept;
    auto const keep{ [ & ]( Bitmap && result, oracle_set expected, char const * op ) {
        expect_equals( result, expected, what + " " + op );
        kept.emplace_back( std::move( result ), std::move( expected ) );
    } };

    for ( int round{ 0 }; round < 4; ++round ) {
        auto const content{ round == 0 ? oracle_set{} : random_content<Bitmap>( rng ) };
        auto const receiver{ make_bitmap<Bitmap>( content, round % 2 == 1 ) };
        auto const view{ view_of( *viewed ) };
        { Bitmap b{ make_bitmap<Bitmap>( content, round % 2 == 1 ) }; b &= view; keep( std::move( b ), intersection( content, members ), "&=" ); }
        { Bitmap b{ make_bitmap<Bitmap>( content, round % 2 == 1 ) }; b |= view; keep( std::move( b ), unite( content, members ), "|=" ); }
        { Bitmap b{ make_bitmap<Bitmap>( content, round % 2 == 1 ) }; b -= view; keep( std::move( b ), difference( content, members ), "-=" ); }
        { Bitmap b{ make_bitmap<Bitmap>( content, round % 2 == 1 ) }; b ^= view; keep( std::move( b ), symmetric_difference( content, members ), "^=" ); }
        keep( receiver & view, intersection( content, members ), "bitmap & view" );
        keep( view & receiver, intersection( content, members ), "view & bitmap" );
        keep( receiver | view, unite( content, members ), "bitmap | view" );
        keep( receiver - view, difference( content, members ), "bitmap - view" );
        keep( view - receiver, difference( members, content ), "view - bitmap" );
        keep( receiver ^ view, symmetric_difference( content, members ), "bitmap ^ view" );
        keep( view & view, members, "view & view" );
        keep( view | view, members, "view | view" );
        keep( view - view, {}, "view - view" );
        keep( view ^ view, {}, "view ^ view" );
        {
            Bitmap into;
            receiver.intersect_into( view, into );
            keep( std::move( into ), intersection( content, members ), "intersect_into" );
            into = Bitmap{};
            std::array<typed_view<Bitmap>, 2> const both{ view_of( receiver ), view };
            Bitmap::and_many_into( both, into );
            keep( std::move( into ), intersection( content, members ), "and_many_into" );
            keep( Bitmap::and_many( both ), intersection( content, members ), "and_many of views" );
            std::array<typed_view<Bitmap>, 1> const one{ view };
            keep( Bitmap::and_many( one ), members, "and_many of one view" );
            std::array<Bitmap const *, 1> const one_bitmap{ viewed.get() };
            keep( Bitmap::and_many( one_bitmap ), members, "and_many of one bitmap" );
        }

        auto const common{ intersection( content, members ) };
        EXPECT_EQ( receiver.intersects( view ), !common.empty() ) << what;
        EXPECT_EQ( receiver.and_cardinality( view ), common.size() ) << what;
        EXPECT_EQ( receiver.xor_cardinality( view ), content.size() + members.size() - 2 * common.size() ) << what;
        EXPECT_EQ( receiver.is_subset_of( view ), common.size() == content.size() ) << what;
        EXPECT_EQ( receiver.is_superset_of( view ), common.size() == members.size() ) << what;
        EXPECT_EQ( receiver.equals( view ), content == members ) << what;
        std::array<typed_view<Bitmap>, 2> const both{ view_of( receiver ), view };
        EXPECT_EQ( Bitmap::and_many_cardinality( both ), common.size() ) << what;
        EXPECT_EQ( Bitmap::and_many_intersects( both ), !common.empty() ) << what;
        oracle_set seen;
        Bitmap::and_many_for_each( both, [ & ]( auto const key ) { seen.push_back( key ); } );
        EXPECT_EQ( seen, common ) << what;
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }

    // nothing the viewed bitmap owns was written
    auto const after{ access::snapshot( *viewed ) };
    EXPECT_TRUE( before == after ) << what << ": an operation reading through a view wrote to the viewed bitmap";

    // ... and no result keeps any of it
    viewed.reset();
    for ( auto const & [ result, expected ] : kept ) {
        ASSERT_EQ( result.validate(), nullptr ) << what;
        ASSERT_EQ( contents_of( result ), expected ) << what << ": a result outlived the viewed bitmap with different contents";
    }
}

// The n-way AND of a single bitmap copies its containers where it used to take another reference to each: the bitmap
// is left byte-identical and the result shares nothing with it.
TYPED_TEST( ViewWitness, TheAndOfOneBitmapCopiesItsContainersAndCountsNothing ) {
    using Bitmap = TypeParam;
    using access = bitmap_test_access<Bitmap>;
    std::mt19937_64 rng{ 9 };
    for ( int round{ 0 }; round < 6; ++round ) {
        auto const published{ make_bitmap<Bitmap>( random_content<Bitmap>( rng ), round % 2 == 1 ) };
        auto const before{ access::snapshot( published ) };
        for ( auto const & state : before ) { ASSERT_LE( state.refcount, 1U ); }
        std::array<Bitmap const *, 1> const one{ &published };
        auto const result{ Bitmap::and_many( one ) };
        EXPECT_TRUE( before == access::snapshot( published ) ) << "the n-way AND of one bitmap wrote to it";
        ASSERT_EQ( result.validate(), nullptr );
        EXPECT_TRUE( result == published );
        for ( auto const & state : access::snapshot( result ) ) { EXPECT_LE( state.refcount, 1U ); }
        EXPECT_EQ( Bitmap::and_many_cardinality( one ), published.size() );
    }
}

TYPED_TEST( ViewWitness, ReadingThroughAViewWritesNothingAndKeepsNothing ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 20261022 };
    for ( int round{ 0 }; round < 4; ++round ) {
        exercise_views_of<Bitmap>( std::make_unique<Bitmap>( publish<Bitmap>( rng ) ), rng, "published " + std::to_string( round ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    // a bitmap built the ordinary way, optimized into runs
    exercise_views_of<Bitmap>( std::make_unique<Bitmap>( make_bitmap<Bitmap>( random_content<Bitmap>( rng ), true ) ), rng, "optimized" );
    // every container kind and the full chunk
    for ( auto const kind : { shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::ends, shape::mixed } ) {
        oracle_set content;
        for ( auto const chunk : domain<Bitmap>::chunks() ) { fill_chunk( content, chunk, kind, rng, domain<Bitmap>::max ); }
        finalize( content );
        for ( bool const optimize : { false, true } ) {
            exercise_views_of<Bitmap>( std::make_unique<Bitmap>( make_bitmap<Bitmap>( content, optimize ) ), rng, "shape " + std::to_string( static_cast<int>( kind ) ) );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

// The bitmaps a lazy fold publishes keep stale bitset endpoints; reading through a view must not repair them (that
// is a write to a container header another thread may be reading).
TYPED_TEST( ViewWitness, StaleEndpointsOfAPublishedBitmapStayStale ) {
    using Bitmap = TypeParam;
    if constexpr ( std::numeric_limits<typename Bitmap::key_type>::digits >= 32 ) {
        std::mt19937_64 rng{ 77 };
        std::size_t stale{ 0 };
        for ( int attempt{ 0 }; attempt < 8 && stale == 0; ++attempt ) {
            auto published{ std::make_unique<Bitmap>( publish<Bitmap>( rng, 6 ) ) };
            stale = stale_endpoint_containers( *published );
            if ( stale != 0 ) { exercise_views_of<Bitmap>( std::move( published ), rng, "stale endpoints" ); }
        }
        EXPECT_NE( stale, 0U ) << "the publishing sequence no longer leaves a stale endpoint: this test no longer tests what it says";
    }
}

} // namespace
