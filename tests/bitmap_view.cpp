// Phase-4 gate: type-erased bitmap_view / bitmap_ref over distinct bitmap
// instantiations, exercised through non-template consumers, with query parity
// against the concrete calls.

#include <frsr/roaring/bitmap_view.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace {

using frsr::roaring::bitmap;
using frsr::roaring::bitmap_ref;
using frsr::roaring::bitmap_view;

// Non-template consumers — the compiled-boundary use case the view exists for.

std::size_t count_contained( bitmap_view const view, std::vector<std::uint64_t> const & probes ) {
    std::size_t hits{ 0 };
    for ( auto const probe : probes ) {
        hits += view.contains( probe );
    }
    return hits;
}

std::vector<std::uint64_t> collect( bitmap_view const view ) {
    std::vector<std::uint64_t> values;
    values.reserve( view.size() );
    view.for_each( [&]( std::uint64_t const value ) { values.push_back( value ); } );
    return values;
}

std::size_t add_all( bitmap_ref const ref, std::vector<std::uint64_t> const & values ) {
    std::size_t added{ 0 };
    for ( auto const value : values ) {
        added += ref.add( value );
    }
    return added;
}

template <typename Bitmap>
Bitmap make_populated() {
    Bitmap bm;
    // Spread across chunks: small array chunks, a dense run-ish region, singles.
    for ( std::uint32_t v{ 0 }; v < 1000; ++v ) {
        std::ignore = bm.add( static_cast<typename Bitmap::key_type>( v * 7 ) );
    }
    for ( std::uint32_t v{ 40000 }; v < 40200; ++v ) {
        std::ignore = bm.add( static_cast<typename Bitmap::key_type>( v ) );
    }
    return bm;
}

template <typename Bitmap>
void expect_query_parity() {
    auto const bm{ make_populated<Bitmap>() };
    bitmap_view const view{ bm };

    EXPECT_EQ( view.size(), bm.size() );
    EXPECT_EQ( view.byte_size(), bm.byte_size() );
    EXPECT_EQ( view.empty(), bm.empty() );
    EXPECT_EQ( view.front(), std::uint64_t{ bm.front() } );
    EXPECT_EQ( view.back (), std::uint64_t{ bm.back () } );

    std::vector<std::uint64_t> const probes{ 0, 7, 8, 6993, 6994, 39999, 40000, 40199, 40200 };
    std::size_t concrete_hits{ 0 };
    for ( auto const probe : probes ) {
        concrete_hits += bm.contains( static_cast<typename Bitmap::key_type>( probe ) );
    }
    EXPECT_EQ( count_contained( view, probes ), concrete_hits );

    auto const via_view{ collect( view ) };
    auto const via_concrete{ bm.to_vector() };
    ASSERT_EQ( via_view.size(), via_concrete.size() );
    for ( std::size_t i{ 0 }; i < via_view.size(); ++i ) {
        EXPECT_EQ( via_view[ i ], std::uint64_t{ via_concrete[ i ] } );
    }
}

TEST( BitmapView, QueryParityU16 ) { expect_query_parity<bitmap<std::uint16_t>>(); }
TEST( BitmapView, QueryParityU32 ) { expect_query_parity<bitmap<std::uint32_t>>(); }
TEST( BitmapView, QueryParityU64 ) { expect_query_parity<bitmap<std::uint64_t>>(); }

TEST( BitmapView, OutOfDomainValuesAreNotMembers ) {
    bitmap<std::uint16_t> bm;
    ASSERT_TRUE( bm.add( 65535 ) );
    bitmap_view const view{ bm };
    EXPECT_TRUE ( view.contains( 65535 ) );
    EXPECT_FALSE( view.contains( 65536 ) );
    EXPECT_FALSE( view.contains( std::uint64_t{ 1 } << 40 ) );
}

TEST( BitmapView, IterateEarlyExit ) {
    auto const bm{ make_populated<bitmap<std::uint32_t>>() };
    bitmap_view const view{ bm };
    std::size_t visited{ 0 };
    bool const completed{ view.iterate( [&]( std::uint64_t ) { return ++visited < 10; } ) };
    EXPECT_FALSE( completed );
    EXPECT_EQ( visited, 10u );
}

TEST( BitmapView, EqualitySameInstantiation ) {
    auto const a{ make_populated<bitmap<std::uint32_t>>() };
    auto const b{ make_populated<bitmap<std::uint32_t>>() };
    bitmap<std::uint32_t> c{ b };
    ASSERT_TRUE( c.remove( 7 ) );

    EXPECT_TRUE ( bitmap_view{ a } == bitmap_view{ b } );
    EXPECT_FALSE( bitmap_view{ a } == bitmap_view{ c } );
}

TEST( BitmapView, EqualityAcrossInstantiations ) {
    auto const narrow{ make_populated<bitmap<std::uint16_t>>() };
    auto const wide  { make_populated<bitmap<std::uint64_t>>() };
    EXPECT_TRUE( bitmap_view{ narrow } == bitmap_view{ wide } );

    bitmap<std::uint64_t> different{ wide };
    ASSERT_TRUE( different.add( std::uint64_t{ 1 } << 40 ) );
    EXPECT_FALSE( bitmap_view{ narrow } == bitmap_view{ different } );
    EXPECT_FALSE( bitmap_view{ different } == bitmap_view{ narrow } );
}

TEST( BitmapView, TryAsRecoversConcreteType ) {
    auto const bm{ make_populated<bitmap<std::uint32_t>>() };
    bitmap_view const view{ bm };
    EXPECT_EQ( view.try_as<bitmap<std::uint32_t>>(), &bm );
    EXPECT_EQ( view.try_as<bitmap<std::uint64_t>>(), nullptr );
}

TEST( BitmapRef, MutationThroughErasedBoundary ) {
    bitmap<std::uint32_t> bm;
    bitmap_ref const ref{ bm };

    EXPECT_EQ( add_all( ref, { 1, 2, 3, 100000, 2 } ), 4u );
    EXPECT_EQ( bm.size(), 4u );
    EXPECT_TRUE( bm.contains( 100000 ) );

    EXPECT_TRUE ( ref.remove( 2 ) );
    EXPECT_FALSE( ref.remove( 2 ) );
    EXPECT_EQ( bm.size(), 3u );

    ref.clear();
    EXPECT_TRUE( bm.empty() );
    EXPECT_TRUE( ref.empty() );
}

TEST( BitmapRef, OutOfDomainRemoveIsFalse ) {
    bitmap<std::uint16_t> bm;
    ASSERT_TRUE( bm.add( 10 ) );
    bitmap_ref const ref{ bm };
    EXPECT_FALSE( ref.remove( 65536 ) );
    EXPECT_EQ( bm.size(), 1u );
}

TEST( BitmapRef, ConvertsToView ) {
    auto bm{ make_populated<bitmap<std::uint32_t>>() };
    bitmap_ref const ref{ bm };
    bitmap_view const view{ ref };
    EXPECT_EQ( view.size(), bm.size() );
    EXPECT_EQ( count_contained( ref, { 0, 7, 8 } ), count_contained( view, { 0, 7, 8 } ) );
    EXPECT_TRUE( view == bitmap_view{ bm } );
    EXPECT_EQ( view.try_as<bitmap<std::uint32_t>>(), &bm );
}

} // anonymous namespace
