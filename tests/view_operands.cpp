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

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <variant>
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

// the same properties under the value-semantics policy, where a copy of a bitmap copies every payload
using value_bitmaps = ::testing::Types<
    bitmap_of<std::uint16_t, detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, detail::cow_value_semantics, detail::run_selection_lazy>,
    bitmap_of<std::uint64_t, detail::cow_value_semantics>
>;
template <typename Bitmap>
class ViewValueSemantics : public ::testing::Test {};
TYPED_TEST_SUITE( ViewValueSemantics, value_bitmaps );

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

// ---- intersect_into / and_many_into ------------------------------------------------------------------------------

// The scratch of intersect_into( view, scratch ) is reused: its table keeps its capacity and the payloads of the previous
// result are the ones the next result is built in. Whatever the previous result left in them must not show: each round
// intersects a different pair, alternating between large and small results, and the scratch is checked against the oracle
// every time. (How many allocations the reuse saves is measured by the contention benchmark.)
template <typename Bitmap>
void check_scratch_reuse() {
    std::mt19937_64 rng{ 31 };
    Bitmap scratch;
    for ( int round{ 0 }; round < 24; ++round ) {
        auto const content_a{ round % 3 == 2 ? oracle_set{} : random_content<Bitmap>( rng ) };
        auto const content_b{ round % 4 == 3 ? content_a : random_related<Bitmap>( rng, content_a, false ) };
        auto const a{ make_bitmap<Bitmap>( content_a, round % 2 == 1 ) };
        auto const b{ make_bitmap<Bitmap>( content_b, round % 5 == 4 ) };
        a.intersect_into( view_of( b ), scratch );
        expect_equals( scratch, intersection( content_a, content_b ), "intersect_into, round " + std::to_string( round ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        scratch.clear_keep_capacity();   // as a caller does between uses
        a.intersect_into( view_of( b ), scratch );
        expect_equals( scratch, intersection( content_a, content_b ), "intersect_into after clear_keep_capacity, round " + std::to_string( round ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( ViewWitness, IntersectIntoAReusedScratch ) { check_scratch_reuse<TypeParam>(); }
TYPED_TEST( ViewValueSemantics, IntersectIntoAReusedScratch ) { check_scratch_reuse<TypeParam>(); }

// ---- borrowed_copy ----------------------------------------------------------------------------------------------

// A borrowed copy shares the payloads of the viewed bitmap without writing anything of it, is as mutable as any bitmap,
// and never changes the viewed one.
template <typename Bitmap>
void check_borrowed_copy( std::unique_ptr<Bitmap> master, std::mt19937_64 & rng, std::string const & what ) {
    using access = bitmap_test_access<Bitmap>;
    using key_type = typename Bitmap::key_type;
    auto const members{ contents_of( *master ) };
    auto const before{ access::snapshot( *master ) };
    {
        auto copy{ Bitmap::borrowed_copy( view_of( *master ) ) };
        ASSERT_EQ( copy.validate(), nullptr ) << what;
        ASSERT_EQ( copy.size(), members.size() ) << what;
        ASSERT_EQ( contents_of( copy ), members ) << what;
        EXPECT_EQ( copy.statistics().container_count(), master->statistics().container_count() ) << what;
        EXPECT_TRUE( copy == *master ) << what;

        // every spilled container is the master's own payload, borrowed: nothing was copied
        auto const copied{ access::snapshot( copy ) };
        ASSERT_EQ( copied.size(), before.size() ) << what;
        for ( std::size_t i{ 0 }; i < copied.size(); ++i ) {
            EXPECT_EQ( copied[ i ].key, before[ i ].key ) << what;
            EXPECT_EQ( copied[ i ].payload, before[ i ].payload ) << what << ": a payload was copied";
            EXPECT_EQ( copied[ i ].content, before[ i ].content ) << what;
        }
        // a copy of the borrowed copy aliases the same payloads, and counts nothing
        Bitmap second{ copy };
        EXPECT_TRUE( before == access::snapshot( *master ) ) << what << ": copying the borrowed copy wrote to the master";
        EXPECT_EQ( contents_of( second ), members ) << what;

        // the copy is mutable: every kind of update, with the master and the second copy as the witnesses
        auto expected{ members };
        auto const forget{ [ & ]( std::uint64_t const key ) { expected.erase( std::remove( expected.begin(), expected.end(), key ), expected.end() ); } };
        for ( int i{ 0 }; i < 300 && !expected.empty(); ++i ) {
            auto const key{ expected[ rng() % expected.size() ] };
            ASSERT_TRUE( copy.remove( static_cast<key_type>( key ) ) ) << what;
            forget( key );
        }
        for ( int i{ 0 }; i < 300; ++i ) {
            std::uint64_t const key{ domain<Bitmap>::max == std::numeric_limits<std::uint64_t>::max() ? rng() : rng() % ( domain<Bitmap>::max + 1 ) };
            copy.add( static_cast<key_type>( key ) );
            expected.push_back( key );
        }
        finalize( expected );
        expect_equals( copy, expected, what + " after single updates" );
        auto const other{ make_bitmap<Bitmap>( random_content<Bitmap>( rng ), false ) };
        copy |= other;
        expected = unite( expected, contents_of( other ) );
        expect_equals( copy, expected, what + " after |=" );
        copy &= view_of( other );
        expected = intersection( expected, contents_of( other ) );
        expect_equals( copy, expected, what + " after &= view" );
        copy.optimize();
        expect_equals( copy, expected, what + " after optimize" );

        EXPECT_TRUE( before == access::snapshot( *master ) ) << what << ": updating the borrowed copy changed the master";
        EXPECT_EQ( contents_of( second ), members ) << what << ": updating the borrowed copy changed another copy of it";
    }
    EXPECT_TRUE( before == access::snapshot( *master ) ) << what << ": dropping the borrowed copies changed the master";
    ASSERT_EQ( master->validate(), nullptr ) << what;
    ASSERT_EQ( contents_of( *master ), members ) << what;
}

template <typename Bitmap>
void check_borrowed_copies() {
    std::mt19937_64 rng{ 5005 };
    for ( int round{ 0 }; round < 3; ++round ) {
        check_borrowed_copy<Bitmap>( std::make_unique<Bitmap>( publish<Bitmap>( rng ) ), rng, "published " + std::to_string( round ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
    check_borrowed_copy<Bitmap>( std::make_unique<Bitmap>(), rng, "empty" );
    for ( auto const kind : { shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::ends, shape::single, shape::mixed } ) {
        oracle_set content;
        for ( auto const chunk : domain<Bitmap>::chunks() ) { fill_chunk( content, chunk, kind, rng, domain<Bitmap>::max ); }
        finalize( content );
        for ( bool const optimize : { false, true } ) {
            check_borrowed_copy<Bitmap>( std::make_unique<Bitmap>( make_bitmap<Bitmap>( content, optimize ) ), rng, "shape " + std::to_string( static_cast<int>( kind ) ) );
            if ( ::testing::Test::HasFatalFailure() ) { return; }
        }
    }
}

TYPED_TEST( ViewWitness, BorrowedCopyWritesNothingAndIsolatesTheMaster ) { check_borrowed_copies<TypeParam>(); }

// ... and the same under the value-semantics policy, where a copy is a deep copy and the borrowed one is not
TYPED_TEST( ViewValueSemantics, BorrowedCopyAliasesAndIsolatesTheMaster ) { check_borrowed_copies<TypeParam>(); }
TYPED_TEST( ViewValueSemantics, ReadingThroughAViewKeepsNothing ) {
    using Bitmap = TypeParam;
    std::mt19937_64 rng{ 20261023 };
    for ( int round{ 0 }; round < 3; ++round ) { exercise_views_of<Bitmap>( std::make_unique<Bitmap>( publish<Bitmap>( rng ) ), rng, "published " + std::to_string( round ) ); }
}

// ---- and_many_into with a result that is reused -----------------------------------------------------------------

// The result of the previous call is parked in the table of `out` when the next call clears it, and the walk carries on with
// those payloads: after a dense result a small one, after a small one a dense one, and an empty one in between, with the pointer
// and the view forms taking turns. Whatever the previous result left in a payload must not show.
template <typename Bitmap>
void check_and_many_into_reuse() {
    std::mt19937_64 rng{ 4321 };
    Bitmap out;
    for ( int round{ 0 }; round < 18; ++round ) {
        auto const kind{ round % 3 == 0 ? shape::dense_bitset : round % 3 == 1 ? shape::sparse : shape::runs };
        std::array<oracle_set, 3> contents;
        for ( auto & content : contents ) {
            for ( auto const chunk : domain<Bitmap>::chunks() ) { fill_chunk( content, chunk, round % 6 == 5 ? shape::single : kind, rng, domain<Bitmap>::max ); }
            finalize( content );
        }
        // operands that share keys, so that the answers are not all empty
        for ( auto const key : contents[ 0 ] ) {
            if ( rng() % 3 == 0 ) { contents[ 1 ].push_back( key ); }
            if ( rng() % 2 == 0 ) { contents[ 2 ].push_back( key ); }
        }
        finalize( contents[ 1 ] );
        finalize( contents[ 2 ] );
        if ( round % 7 == 6 ) { contents[ 2 ].clear(); }
        std::array<Bitmap, 3> const bitmaps{ make_bitmap<Bitmap>( contents[ 0 ], round % 2 == 1 ), make_bitmap<Bitmap>( contents[ 1 ], false ), make_bitmap<Bitmap>( contents[ 2 ], round % 4 == 3 ) };
        auto const operand_count{ 2 + static_cast<std::size_t>( round % 2 ) };
        auto expected{ contents[ 0 ] };
        for ( std::size_t i{ 1 }; i < operand_count; ++i ) { expected = intersection( expected, contents[ i ] ); }
        if ( round % 2 == 0 ) {
            std::array<typed_view<Bitmap>, 3> const views{ view_of( bitmaps[ 0 ] ), view_of( bitmaps[ 1 ] ), view_of( bitmaps[ 2 ] ) };
            Bitmap::and_many_into( std::span<typed_view<Bitmap> const>{ views.data(), operand_count }, out );
        } else {
            std::array<Bitmap const *, 3> const pointers{ &bitmaps[ 0 ], &bitmaps[ 1 ], &bitmaps[ 2 ] };
            Bitmap::and_many_into( std::span<Bitmap const * const>{ pointers.data(), operand_count }, out );
        }
        expect_equals( out, expected, "and_many_into, round " + std::to_string( round ) );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
        EXPECT_EQ( out.size(), expected.size() );
    }
}

TYPED_TEST( ViewWitness, AndManyIntoAReusedResult ) { check_and_many_into_reuse<TypeParam>(); }
TYPED_TEST( ViewValueSemantics, AndManyIntoAReusedResult ) { check_and_many_into_reuse<TypeParam>(); }

// ---- a copy-on-write copy of the viewed bitmap, updated through its view -------------------------------------------

// The slots of the copy share their payloads with the pieces of the view (contract 6): releasing a share is the copy's own
// reference to drop, the viewed bitmap keeps its contents, and the copy ends up with the answer.
template <typename Bitmap>
void check_a_copy_updated_through_the_view_of_its_source() {
    std::mt19937_64 rng{ 606 };
    for ( int round{ 0 }; round < 4; ++round ) {
        Bitmap const original{ publish<Bitmap>( rng ) };
        auto const members{ contents_of( original ) };
        auto const check{ [ & ]( auto && update, oracle_set const & expected, char const * what ) {
            Bitmap copy{ original };
            update( copy, view_of( original ) );
            expect_equals( copy, expected, what );
            expect_equals( original, members, std::string{ what } + ": the viewed bitmap" );
        } };
        check( []( Bitmap & b, auto const view ) { b &= view; }, members, "&=" );
        check( []( Bitmap & b, auto const view ) { b |= view; }, members, "|=" );
        check( []( Bitmap & b, auto const view ) { b -= view; }, {}, "-=" );
        check( []( Bitmap & b, auto const view ) { b ^= view; }, {}, "^=" );
        check( []( Bitmap & b, auto const view ) { b &= !view; }, {}, "&= !" );
        check( []( Bitmap & b, auto const view ) { b.andnot_with( !view ); }, members, "andnot_with !" );
        // the copy updated first, then combined with the view of its source
        Bitmap copy{ original };
        copy.add( static_cast<typename Bitmap::key_type>( domain<Bitmap>::max == std::numeric_limits<std::uint64_t>::max() ? rng() : rng() % ( domain<Bitmap>::max + 1 ) ) );
        static_cast<void>( copy.remove( static_cast<typename Bitmap::key_type>( members.empty() ? 0 : members[ rng() % members.size() ] ) ) );
        auto const changed{ contents_of( copy ) };
        copy |= view_of( original );
        expect_equals( copy, unite( changed, members ), "updated copy |=" );
        copy -= view_of( original );
        expect_equals( copy, difference( unite( changed, members ), members ), "updated copy -=" );
        expect_equals( original, members, "the viewed bitmap after the updated copy" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( ViewWitness, ACopyUpdatedThroughTheViewOfItsSource ) { check_a_copy_updated_through_the_view_of_its_source<TypeParam>(); }
TYPED_TEST( ViewValueSemantics, ACopyUpdatedThroughTheViewOfItsSource ) { check_a_copy_updated_through_the_view_of_its_source<TypeParam>(); }

// ---- what holds on to a borrowed copy ----------------------------------------------------------------------------

// A borrowed copy refers to its master, and so does every bitmap that took a container of it by copying a handle: a copy of it,
// the receiver of `|= copy` and `^= copy` (the containers it did not have), `&= copy` (a full chunk), an assignment. A view of
// the borrowed copy as the operand adopts nothing: the receiver outlives the master. The second half is run under the address
// sanitizer, which is where reading what a destroyed master owned shows.
template <typename Bitmap>
void check_what_holds_on_to_a_borrowed_copy() {
    using access = bitmap_test_access<Bitmap>;
    std::mt19937_64 rng{ 8080 };
    for ( int round{ 0 }; round < 3; ++round ) {
        auto master{ std::make_unique<Bitmap>( publish<Bitmap>( rng ) ) };
        auto const members{ contents_of( *master ) };
        std::vector<void const *> master_payloads;
        for ( auto const & state : access::snapshot( *master ) ) { if ( state.payload != nullptr ) { master_payloads.push_back( state.payload ); } }
        auto const refers_to_master{ [ & ]( Bitmap const & bitmap ) {
            for ( auto const & state : access::snapshot( bitmap ) ) {
                if ( state.payload != nullptr && std::ranges::find( master_payloads, state.payload ) != master_payloads.end() ) { return true; }
            }
            return false;
        } };

        auto borrowed{ Bitmap::borrowed_copy( view_of( *master ) ) };
        Bitmap by_copy{ borrowed };
        Bitmap by_assignment;  by_assignment = borrowed;
        Bitmap by_union;       by_union |= borrowed;
        Bitmap by_exclusive;   by_exclusive ^= borrowed;
        Bitmap by_and{ borrowed }; by_and &= borrowed;
        if ( !master_payloads.empty() ) {
            EXPECT_TRUE( refers_to_master( by_copy ) );
            EXPECT_TRUE( refers_to_master( by_assignment ) );
            EXPECT_TRUE( refers_to_master( by_union ) ) << "a union adopts the containers the receiver lacks";
        }
        // while the master lives they all read as the master does
        for ( auto const * bitmap : { &by_copy, &by_assignment, &by_union, &by_exclusive, &by_and } ) { expect_equals( *bitmap, members, "while the master is alive" ); }

        // through a view of the borrowed copy nothing is adopted
        Bitmap by_view_union;     by_view_union |= view_of( borrowed );
        Bitmap by_view_exclusive; by_view_exclusive ^= view_of( borrowed );
        Bitmap by_view_and;       by_view_and |= view_of( borrowed ); by_view_and &= view_of( borrowed );
        Bitmap by_many = Bitmap::and_many( std::array<typed_view<Bitmap>, 1>{ view_of( borrowed ) } );
        Bitmap by_difference = view_of( borrowed ) - Bitmap{};
        for ( auto const * bitmap : { &by_view_union, &by_view_exclusive, &by_many, &by_difference } ) {
            EXPECT_FALSE( refers_to_master( *bitmap ) ) << "a view operand adopted a container of the borrowed copy";
        }

        // dropping the master: the ones that did not adopt keep their answer
        borrowed = Bitmap{};
        by_copy = Bitmap{};  by_assignment = Bitmap{};  by_union = Bitmap{};  by_exclusive = Bitmap{};  by_and = Bitmap{};
        master.reset();
        for ( auto const * bitmap : { &by_view_union, &by_view_exclusive, &by_many, &by_difference } ) { expect_equals( *bitmap, members, "after the master is gone" ); }
        expect_equals( by_view_and, members, "after the master is gone" );
        if ( ::testing::Test::HasFatalFailure() ) { return; }
    }
}

TYPED_TEST( ViewWitness, WhatHoldsOnToABorrowedCopy ) { check_what_holds_on_to_a_borrowed_copy<TypeParam>(); }
TYPED_TEST( ViewValueSemantics, WhatHoldsOnToABorrowedCopy ) { check_what_holds_on_to_a_borrowed_copy<TypeParam>(); }

// ---- the kind of the container an intersection leaves --------------------------------------------------------------

// A bitset that an intersection with a bit string, a key range or a list of keys leaves below the array limit is stored as an
// array, as the reference does. Against a bitmap, or a view of one, it stays a bitset: that is the rule of the walk of bitmap
// against bitmap (an intersection is the accumulator of a fold), and a view takes part in it so that `a &= view` is `a &= bitmap`.
TEST( ViewOperandsKinds, TheIntersectionOfABitsetWithASparseOperand ) {
    using Bitmap = bitmap<std::uint32_t>;
    Bitmap evens;
    for ( std::uint32_t key{ 0 }; key < 65536; key += 2 ) { evens.add( key ); }
    ASSERT_EQ( evens.statistics().bitset_containers, 1U );

    // the odd keys, and a hundred even ones
    std::vector<std::uint64_t> words( 1024, 0xAAAA'AAAA'AAAA'AAAAULL );
    for ( std::size_t i{ 0 }; i < 100; ++i ) { words[ i ] |= 1; }
    Bitmap operand;
    for ( std::uint32_t key{ 1 }; key < 65536; key += 2 ) { operand.add( key ); }
    for ( std::uint32_t key{ 0 }; key < 6400; key += 64 ) { operand.add( key ); }

    Bitmap by_words{ evens };
    by_words &= bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ words } };
    EXPECT_EQ( by_words.size(), 100U );
    EXPECT_EQ( by_words.statistics().array_containers, 1U );
    EXPECT_EQ( by_words.statistics().bitset_containers, 0U );

    Bitmap by_range{ evens };
    by_range &= key_range::closed( 0, 198 );
    EXPECT_EQ( by_range.size(), 100U );
    EXPECT_EQ( by_range.statistics().array_containers, 1U );

    Bitmap by_bitmap{ evens };
    by_bitmap &= operand;
    Bitmap by_view{ evens };
    by_view &= view_of( operand );
    EXPECT_EQ( by_view.size(), 100U );
    EXPECT_TRUE( by_view == by_bitmap );
    EXPECT_EQ( by_view.statistics().array_containers, by_bitmap.statistics().array_containers );
    EXPECT_EQ( by_view.statistics().bitset_containers, by_bitmap.statistics().bitset_containers );
    EXPECT_EQ( by_view.statistics().bitset_containers, detail::kAndDemotesSparseBitsets ? 0U : 1U );
}

// a view of a bitmap of a container set the library does not hold, and of 8-bit keys, which only the header serves
TEST( ViewOperandsHeaderOnly, UncompiledInstantiations ) {
    using narrow = bitmap<std::uint8_t>;
    using other_set = bitmap<std::uint32_t, std::variant<array_container<std::uint32_t>, bitset_container<std::uint32_t>>>;
    std::mt19937_64 rng{ 77 };
    {
        auto const content{ random_content<narrow>( rng ) };
        auto const viewed{ make_bitmap<narrow>( content, false ) };
        narrow b{ viewed };
        b |= view_of( viewed );
        expect_equals( b, content, "u8" );
        b ^= view_of( viewed );
        expect_equals( b, {}, "u8 xor" );
        EXPECT_EQ( narrow::borrowed_copy( view_of( viewed ) ).size(), content.size() );
    }
    {
        auto const content{ random_content<other_set>( rng ) };
        auto const viewed{ make_bitmap<other_set>( content, false ) };
        other_set b;
        b |= view_of( viewed );
        expect_equals( b, content, "variant set" );
        EXPECT_EQ( b.and_cardinality( view_of( viewed ) ), content.size() );
        EXPECT_EQ( other_set::and_many( std::array<typed_view<other_set>, 2>{ view_of( b ), view_of( viewed ) } ).size(), content.size() );
        EXPECT_EQ( contents_of( other_set::borrowed_copy( view_of( viewed ) ) ), content );
    }
}

} // namespace
