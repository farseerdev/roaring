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

} // namespace

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
