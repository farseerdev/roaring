#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <random>
#include <span>
#include <thread>
#include <vector>

// Concurrent readers of an unmodified bitmap must be safe (CRoaring's guarantee).
// Every thread runs only const operations on shared bitmaps; under
// -fsanitize=thread any store reached from a const path is reported as a race.
// The set covers the public const entry points: point queries (with a bulk
// context private to the thread), cardinality, iteration, conversion, the set
// operators, the *_into forms with a per-thread scratch bitmap, intersects and
// the n-way AND family over shared operands.

namespace {

using frsr::roaring::default_container_set;

using wide_bitmap      = frsr::roaring::bitmap<std::uint64_t>;
using narrow_bitmap    = frsr::roaring::bitmap<std::uint32_t>;
using narrow_cow_bitmap = frsr::roaring::bitmap<std::uint32_t, default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>;
using wide_cow_bitmap  = frsr::roaring::bitmap<std::uint64_t, default_container_set<std::uint64_t>, frsr::roaring::detail::cow_atomic_refcount>;

// Chunk keys spread over the key space, inserted and removed in random order,
// so the chunk array has been through every mutation path before it is shared.
template <typename Bitmap>
Bitmap make_scattered( std::uint32_t const seed, std::size_t const chunk_count, std::vector<typename Bitmap::key_type> & present ) {
    using key_type = typename Bitmap::key_type;
    constexpr key_type key_mask{ ~key_type{ 0 } };
    std::mt19937_64 rng{ seed };
    Bitmap result;
    std::vector<key_type> added;
    for ( std::size_t i{ 0 }; i != chunk_count * 2; ++i ) {
        auto const value{ static_cast<key_type>( ( rng() & ~std::uint64_t{ 0xFFFF } ) | ( rng() & 0xFFFF ) ) & key_mask };
        result.add( value );
        added.push_back( value );
    }
    for ( std::size_t i{ 0 }; i < added.size(); i += 3 ) {
        result.remove( added[ i ] );
    }
    for ( auto const value : added ) {
        if ( result.contains( value ) ) { present.push_back( value ); }
    }
    return result;
}

void run_readers( unsigned const thread_count, auto const & body ) {
    std::atomic<bool> go{ false };
    std::atomic<unsigned> failures{ 0 };
    std::vector<std::jthread> threads;
    for ( unsigned t{ 0 }; t != thread_count; ++t ) {
        threads.emplace_back( [&, t] {
            while ( !go.load( std::memory_order_acquire ) ) { std::this_thread::yield(); }
            if ( !body( t ) ) { failures.fetch_add( 1, std::memory_order_relaxed ); }
        } );
    }
    go.store( true, std::memory_order_release );
    threads.clear(); // joins
    EXPECT_EQ( failures.load(), 0U );
}

template <typename Bitmap>
void point_queries_from_many_threads() {
    using key_type = typename Bitmap::key_type;
    for ( auto const chunk_count : { std::size_t{ 100 }, std::size_t{ 3000 } } ) {
        std::vector<key_type> present;
        Bitmap const bitmap{ make_scattered<Bitmap>( 1U, chunk_count, present ) };
        run_readers( 8, [&]( unsigned const t ) {
            typename Bitmap::bulk_context ctx;
            std::mt19937_64 rng{ 100U + t };
            for ( auto const value : present ) {
                if ( !bitmap.contains( value ) ) { return false; }
                if ( !bitmap.contains_bulk( ctx, value ) ) { return false; }
            }
            for ( int i{ 0 }; i != 20000; ++i ) {
                auto const probe{ static_cast<key_type>( rng() ) };
                (void)bitmap.contains( probe );
                (void)bitmap.contains_bulk( ctx, probe );
            }
            return true;
        } );
    }
}

template <typename Bitmap>
void whole_bitmap_queries_from_many_threads() {
    using key_type = typename Bitmap::key_type;
    std::vector<key_type> present_a, present_b, present_c;
    Bitmap const a{ make_scattered<Bitmap>( 2U, 1500, present_a ) };
    Bitmap const b{ make_scattered<Bitmap>( 3U, 1500, present_b ) };
    Bitmap const c{ make_scattered<Bitmap>( 4U, 1500, present_c ) };
    // operands that overlap, so the intersections are not empty
    Bitmap const ab{ a | b };
    Bitmap const ac{ a | c };
    auto const expected_cardinality{ a.size() };
    auto const expected_values{ a.to_vector() };
    auto const expected_and{ ( a & b ).size() };
    auto const expected_and3{ ( ab & ac ).size() };
    run_readers( 8, [&]( unsigned const t ) {
        if ( a.size() != expected_cardinality ) { return false; }
        if ( a.to_vector() != expected_values ) { return false; }
        if ( ( a & b ).size() != expected_and ) { return false; }
        if ( ( a | b ).size() < expected_cardinality ) { return false; }
        if ( ( a - b ).size() > expected_cardinality ) { return false; }
        if ( ( a & !b ).size() > expected_cardinality ) { return false; }
        if ( a.intersects( b ) != ( expected_and != 0 ) ) { return false; }
        if ( ( t & 1U ) && !a.contains( expected_values.front() ) ) { return false; }

        // the scratch forms: the operands are shared, the scratch is this thread's own
        Bitmap scratch;
        a.intersect_into( b, scratch );
        if ( scratch.size() != expected_and ) { return false; }
        a.union_into( b, scratch );
        if ( scratch.size() < expected_cardinality ) { return false; }
        a.difference_into( b, scratch );
        if ( scratch.size() > expected_cardinality ) { return false; }

        // the n-way AND over shared operands
        std::array<Bitmap const *, 3> const operands{ &a, &ab, &ac };
        std::span<Bitmap const * const> const span{ operands };
        if ( Bitmap::and_many( span ).size() != Bitmap::and_many_cardinality( span ) ) { return false; }
        if ( Bitmap::and_many_intersects( span ) != ( Bitmap::and_many_cardinality( span ) != 0 ) ) { return false; }
        if ( Bitmap::and_many_intersects( std::span<Bitmap const * const>{ operands.data(), 2 }, ac ) != ( ( ab & a & ac ).size() != 0 ) ) { return false; }
        if ( ( ab & ac ).size() != expected_and3 ) { return false; }

        std::size_t visited{ 0 };
        a.for_each( [&]( key_type ) { ++visited; } );
        return visited == expected_cardinality;
    } );
}

// A bitmap as a consumer publishes one: dense, sparse and run chunks folded in lazily from several parts and finished
// once, which leaves bitset endpoints stale (the one place a read-only path could be tempted to repair them).
template <typename Bitmap>
Bitmap make_published( std::uint32_t const seed ) {
    using key_type = typename Bitmap::key_type;
    std::mt19937_64 rng{ seed };
    std::vector<std::uint64_t> chunks{ 0, 1, 2, 5, 9, 40 };
    if constexpr ( std::numeric_limits<key_type>::digits == 64 ) { for ( auto & chunk : chunks ) { chunk = chunk * 1000003U + ( chunk << 20 ); } }
    if constexpr ( std::numeric_limits<key_type>::digits == 16 ) { chunks = { 0 }; }
    std::vector<Bitmap> parts( 4 );
    for ( std::size_t p{ 0 }; p != parts.size(); ++p ) {
        for ( auto const chunk : chunks ) {
            auto const base{ chunk << 16 };
            switch ( ( rng() + p ) % 4 ) {
                case 0: for ( int i{ 0 }; i != 20000; ++i ) { parts[ p ].add( static_cast<key_type>( base + ( rng() & 0xFFFF ) ) ); } break;
                case 1: for ( int i{ 0 }; i != 200; ++i )   { parts[ p ].add( static_cast<key_type>( base + ( rng() & 0xFFFF ) ) ); } break;
                case 2: { auto const first{ rng() & 0x7FFF }; parts[ p ].add_closed_range( static_cast<key_type>( base + first ), static_cast<key_type>( base + first + 9000 ) ); break; }
                default: for ( int i{ 0 }; i != 60000; ++i ) { parts[ p ].add( static_cast<key_type>( base + ( rng() & 0xFFFF ) ) ); } break;
            }
        }
        if ( p % 2 == 1 ) { parts[ p ].optimize(); }
    }
    Bitmap published;
    for ( auto const & part : parts ) { published.bulk_or_intermediate( part ); }
    published.bulk_or_finish();
    return published;
}

// Views of published bitmaps are what many readers share: every query and every update that takes a view goes through
// the viewed bitmaps without writing to them (under -fsanitize=thread a store is a race), and each thread's result is
// its own.
template <typename Bitmap>
void view_reads_from_many_threads() {
    using key_type = typename Bitmap::key_type;
    using frsr::roaring::view_of;
    Bitmap const a{ make_published<Bitmap>( 11U ) };
    Bitmap const b{ make_published<Bitmap>( 12U ) };
    Bitmap const c{ make_published<Bitmap>( 13U ) };
    auto const expected_and { ( a & b ).to_vector() };
    auto const expected_or  { ( a | b ).to_vector() };
    auto const expected_diff{ ( a - b ).to_vector() };
    auto const expected_xor { ( a ^ b ).to_vector() };
    auto const expected_and3{ ( a & b & c ).to_vector() };
    auto const to_vector{ []( Bitmap const & bitmap ) { return bitmap.to_vector(); } };
    run_readers( 8, [&]( unsigned const t ) {
        auto const va{ view_of( a ) };
        auto const vb{ view_of( b ) };
        auto const vc{ view_of( c ) };
        // updates of a private receiver through a view of a shared bitmap
        for ( int round{ 0 }; round != 3; ++round ) {
            Bitmap r;
            if ( t % 2 == 0 ) { r |= va; } else { r = Bitmap{ a.to_vector() }; }   // a receiver filled through the view, and one built on its own
            r &= vb;
            if ( to_vector( r ) != expected_and ) { return false; }
            r |= vb;
            r ^= va;
            r -= vb;
            if ( to_vector( r ) != expected_diff ) { return false; }
        }
        { Bitmap r; r |= va; r &= vb; if ( to_vector( r ) != expected_and ) { return false; } }
        { Bitmap r; r |= va; r |= vb; if ( to_vector( r ) != expected_or ) { return false; } }
        { Bitmap r; r |= va; r -= vb; if ( to_vector( r ) != expected_diff ) { return false; } }
        { Bitmap r; r |= va; r ^= vb; if ( to_vector( r ) != expected_xor ) { return false; } }
        if ( to_vector( va & vb ) != expected_and || to_vector( va | vb ) != expected_or ) { return false; }
        if ( to_vector( va - vb ) != expected_diff || to_vector( va ^ vb ) != expected_xor ) { return false; }
        // the queries
        if ( a.and_cardinality( vb ) != expected_and.size() ) { return false; }
        if ( a.intersects( vb ) != !expected_and.empty() ) { return false; }
        if ( a.xor_cardinality( vb ) != expected_xor.size() ) { return false; }
        if ( a.is_subset_of( va ) != true || a.equals( va ) != true || a.is_superset_of( va ) != true ) { return false; }
        if ( a.is_subset_of( vb ) != ( expected_and.size() == a.size() ) ) { return false; }
        // the n-way forms and the scratch form
        std::array<frsr::roaring::typed_view<Bitmap>, 3> const views{ va, vb, vc };
        if ( to_vector( Bitmap::and_many( views ) ) != expected_and3 ) { return false; }
        if ( Bitmap::and_many_cardinality( views ) != expected_and3.size() ) { return false; }
        if ( Bitmap::and_many_intersects( views ) != !expected_and3.empty() ) { return false; }
        Bitmap scratch;
        a.intersect_into( vb, scratch );
        if ( to_vector( scratch ) != expected_and ) { return false; }
        // a borrowed copy of a shared bitmap, updated in this thread: the shared one is neither written nor changed
        {
            auto copy{ Bitmap::borrowed_copy( va ) };
            if ( copy.size() != a.size() ) { return false; }
            copy &= vb;
            if ( to_vector( copy ) != expected_and ) { return false; }
            auto again{ Bitmap::borrowed_copy( vb ) };
            again.add( static_cast<key_type>( t ) );
            again |= va;
            again -= vc;
            again ^= va;
            again.optimize();
            if ( again.validate() != nullptr ) { return false; }
        }
        std::size_t visited{ 0 };
        Bitmap::and_many_for_each( views, [&]( key_type ) { ++visited; } );
        return visited == expected_and3.size();
    } );
}

} // namespace

TEST( FrsrRoaringConcurrentReads, ViewsOfPublishedBitmapsFromManyThreads ) {
    view_reads_from_many_threads<narrow_bitmap>();
    view_reads_from_many_threads<wide_bitmap>();
    view_reads_from_many_threads<narrow_cow_bitmap>();
    view_reads_from_many_threads<wide_cow_bitmap>();
}

// tests/threads_unit.cpp: copy-on-write bitmaps whose copies share payloads, copied and combined in place from several threads
// at once (there the check is for data races); here the answers are checked too, and the same sequence runs through views.
TEST( FrsrRoaringConcurrentReads, CopiesThatSharePayloadsCombinedFromManyThreads ) {
    using frsr::roaring::view_of;
    narrow_cow_bitmap r1;
    for ( std::uint32_t i{ 0 }; i < 50000; ++i ) { if ( i != 300 ) { r1.add( 65536 + i ); } }
    for ( std::uint32_t i{ 50000 }; i < 150000; ++i ) { if ( i % 500 == 0 ) { r1.add( i ); } }
    for ( std::uint32_t i{ 150000 }; i < 200000; ++i ) { if ( i % 2 == 0 ) { r1.add( i ); } }
    r1.optimize();
    narrow_cow_bitmap const r2{ std::vector<std::uint32_t>{ 10010, 10020, 10030, 10040, 10050, 65537, 65900, 150002, 150003, 150500 } };
    narrow_cow_bitmap const r3{ r1 };
    auto const expected{ ( ( r1 & r2 ) - r3 ).to_vector() };
    auto const expected_through_views{ ( r1 & r2 ).to_vector() };
    EXPECT_TRUE( expected.empty() );
    EXPECT_FALSE( expected_through_views.empty() );

    narrow_cow_bitmap const sets[ 2 ][ 3 ]{ { r1, r2, r3 }, { r1, r2, r3 } };
    run_readers( 2, [&]( unsigned const t ) {
        auto const & set{ sets[ t ] };
        for ( int i{ 0 }; i != 100; ++i ) {
            narrow_cow_bitmap c1{ set[ 0 ] };
            narrow_cow_bitmap const c2{ set[ 1 ] };
            narrow_cow_bitmap const c3{ set[ 2 ] };
            if ( i % 2 == 0 ) { c1 &= c2; c1 -= c3; if ( c1.to_vector() != expected ) { return false; } }
            else              { c1 &= view_of( c2 ); if ( c1.to_vector() != expected_through_views ) { return false; } c1 -= view_of( c3 ); if ( !c1.empty() ) { return false; } }
            if ( ( view_of( set[ 0 ] ) & view_of( c2 ) ).to_vector() != expected_through_views ) { return false; }
        }
        return true;
    } );
}

TEST( FrsrRoaringConcurrentReads, WideKeyPointQueriesFromManyThreads ) { point_queries_from_many_threads<wide_bitmap>(); }
TEST( FrsrRoaringConcurrentReads, NarrowKeyPointQueriesFromManyThreads ) { point_queries_from_many_threads<narrow_bitmap>(); }

TEST( FrsrRoaringConcurrentReads, WideKeyWholeBitmapQueriesFromManyThreads ) { whole_bitmap_queries_from_many_threads<wide_bitmap>(); }
TEST( FrsrRoaringConcurrentReads, NarrowKeyWholeBitmapQueriesFromManyThreads ) { whole_bitmap_queries_from_many_threads<narrow_bitmap>(); }

// Set operations on bitmaps that share chunk payloads count references; the
// atomic-refcount policy is what makes that safe between readers.
TEST( FrsrRoaringConcurrentReads, SharedPayloadsWithAtomicRefcountFromManyThreads ) {
    point_queries_from_many_threads<wide_cow_bitmap>();
    whole_bitmap_queries_from_many_threads<wide_cow_bitmap>();
}
