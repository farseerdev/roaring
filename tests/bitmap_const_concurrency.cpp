// Concurrent readers of a shared const bitmap.
//
// A const bitmap is documented as safe for any number of simultaneous readers. Reads
// can still touch lazily maintained state (a deferred chunk sort, a chunk lookup
// index, a cached hot chunk), so the bitmaps here are built through every mutation
// path that can leave such state pending, and then read - without any prior read that
// would settle it - by several threads released at the same instant.
//
// Besides the value checks below, the test is meant to be run under
// ThreadSanitizer: a race on lazily built state shows up there even when every
// answer happens to be right.

#include "kernel_test_support.hpp"

#include <atomic>
#include <thread>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::kernel_test;

using CowBitmap32 = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;

constexpr unsigned reader_count{ 8 };

enum class build_path : std::uint8_t {
    out_of_order_adds, reverse_add_many, after_remove, optimized, copied, moved, bulk_sorted
};

constexpr build_path all_paths[]{
    build_path::out_of_order_adds, build_path::reverse_add_many, build_path::after_remove, build_path::optimized,
    build_path::copied, build_path::moved, build_path::bulk_sorted
};

[[nodiscard]] constexpr char const * name_of( build_path const p ) noexcept {
    switch ( p ) {
        case build_path::out_of_order_adds: return "out-of-order adds";
        case build_path::reverse_add_many:  return "reverse add_many";
        case build_path::after_remove:      return "after remove";
        case build_path::optimized:         return "optimized";
        case build_path::copied:            return "copied";
        case build_path::moved:             return "moved";
        case build_path::bulk_sorted:       return "bulk sorted";
    }
    return "?";
}

// Many chunks (past the lookup-index threshold), of every container kind, spread
// over the key domain as far as the key type allows.
template <typename Key>
[[nodiscard]] std::vector<Key> make_values( std::uint64_t const seed, std::size_t const chunks ) {
    std::mt19937_64 rng{ seed };
    constexpr Key stride{ sizeof( Key ) > 4 ? Key{ 1 } << 34 : Key{ 1 } << 16 };
    std::vector<Key> values;
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        auto const base{ static_cast<Key>( ( static_cast<Key>( c ) * 3 + ( rng() % 3 ) + 1 ) * stride ) };
        switch ( c % 3 ) {
            case 0: for ( auto const low : make_pattern( pattern::random, 40 + rng() % 1000, rng ) ) { values.push_back( static_cast<Key>( base + low ) ); } break;
            case 1: for ( auto const low : make_pattern( pattern::random, 5000 + rng() % 9000, rng ) ) { values.push_back( static_cast<Key>( base + low ) ); } break;
            default: for ( std::uint32_t i{ 0 }; i < 4000; ++i ) { values.push_back( static_cast<Key>( base + 1000 + i ) ); } break;
        }
    }
    return sorted_unique( std::move( values ) );
}

// A bitmap holding exactly `values`, reached by the given mutation path.
template <typename Bitmap>
[[nodiscard]] Bitmap build( build_path const path, std::vector<typename Bitmap::key_type> const & values, std::mt19937_64 & rng ) {
    using key_type = typename Bitmap::key_type;
    switch ( path ) {
        case build_path::out_of_order_adds: {
            auto shuffled{ values };
            std::ranges::shuffle( shuffled, rng );
            Bitmap bm;
            for ( auto const v : shuffled ) { (void)bm.add( v ); }
            return bm;
        }
        case build_path::reverse_add_many: {
            Bitmap bm;
            auto const batch{ values.size() / 5 + 1 };
            for ( std::size_t end{ values.size() }; end > 0; ) {
                auto const begin{ end > batch ? end - batch : 0 };
                bm.add_many( std::span{ values.data() + begin, end - begin } );
                end = begin;
            }
            return bm;
        }
        case build_path::after_remove: {
            std::vector<key_type> extra;
            for ( std::size_t i{ 0 }; i < values.size() / 6; ++i ) { extra.push_back( static_cast<key_type>( values[ rng() % values.size() ] ^ 0x2001 ) ); }
            extra = set_difference_of( sorted_unique( std::move( extra ) ), values );
            Bitmap bm;
            bm.add_many( set_union_of( values, extra ) );
            bm.remove_many( extra );
            return bm;
        }
        case build_path::optimized: {
            Bitmap bm;
            bm.add_many( values );
            bm.optimize();
            return bm;
        }
        case build_path::copied: {
            auto source{ build<Bitmap>( build_path::out_of_order_adds, values, rng ) };
            return Bitmap{ source };
        }
        case build_path::moved: {
            auto source{ build<Bitmap>( build_path::reverse_add_many, values, rng ) };
            return Bitmap{ std::move( source ) };
        }
        case build_path::bulk_sorted:
            break;
    }
    Bitmap bm;
    bm.add_many( values );
    return bm;
}

struct start_gate {
    std::atomic<unsigned> arrived{ 0 };
    std::atomic<bool> go{ false };

    void wait( unsigned const total ) noexcept {
        if ( arrived.fetch_add( 1 ) + 1 == total ) { go.store( true ); }
        while ( !go.load() ) { std::this_thread::yield(); }
    }
};

// Runs `body( thread_index )` on `reader_count` threads released together; returns
// the total number of wrong answers the bodies reported.
template <typename Body>
[[nodiscard]] std::size_t run_readers( Body && body ) {
    start_gate gate;
    std::atomic<std::size_t> wrong{ 0 };
    std::vector<std::thread> threads;
    for ( unsigned t{ 0 }; t < reader_count; ++t ) {
        threads.emplace_back( [&, t] {
            gate.wait( reader_count );
            wrong += body( t );
        } );
    }
    for ( auto & thread : threads ) { thread.join(); }
    return wrong;
}

template <typename Bitmap>
void concurrent_readers_case() {
    using key_type = typename Bitmap::key_type;
    constexpr std::size_t chunk_count{ 700 }; // past the chunk-lookup index threshold
    auto const values{ make_values<key_type>( 1, chunk_count ) };
    auto const other_values{ make_values<key_type>( 2, chunk_count ) };

    std::vector<key_type> probes;
    std::mt19937_64 prng{ 99 };
    for ( std::size_t i{ 0 }; i < values.size(); i += 29 ) {
        probes.push_back( values[ i ] );
        probes.push_back( static_cast<key_type>( values[ i ] ^ 0x8000 ) );
    }
    for ( std::size_t i{ 0 }; i < 3000; ++i ) { probes.push_back( static_cast<key_type>( prng() ) ); }
    auto const expected_union{ set_union_of( values, other_values ) };
    auto const expected_inter{ set_intersection_of( values, other_values ) };
    auto const expected_diff{ set_difference_of( values, other_values ) };
    key_type expected_sum{ 0 };
    for ( auto const v : values ) { expected_sum = static_cast<key_type>( expected_sum + v ); }

    for ( auto const path : all_paths ) {
        // Fresh bitmaps per path: the first read of each must happen inside the race.
        std::mt19937_64 rng{ 7 };
        Bitmap const shared{ build<Bitmap>( path, values, rng ) };
        Bitmap const other{ build<Bitmap>( build_path::out_of_order_adds, other_values, rng ) };
        SCOPED_TRACE( name_of( path ) );
        ASSERT_EQ( shared.size(), values.size() );

        auto const wrong{ run_readers( [&]( unsigned const t ) -> std::size_t {
            std::size_t bad{ 0 };
            std::mt19937_64 local{ 1000 + t };
            auto mine{ probes };
            std::ranges::shuffle( mine, local );
            // The kind of read each thread starts with differs, so every lazy path
            // is entered first by some thread while the others are mid-flight.
            auto const phases{ std::array{ 0, 1, 2, 3, 4, 5 } };
            for ( unsigned round{ 0 }; round < 6; ++round ) {
                switch ( phases[ ( t + round ) % phases.size() ] ) {
                    case 0:
                        for ( auto const v : mine ) { bad += shared.contains( v ) == std::ranges::binary_search( values, v ) ? 0 : 1; }
                        break;
                    case 1: {
                        typename Bitmap::bulk_context ctx{};
                        for ( auto const v : mine ) { bad += shared.contains_bulk( ctx, v ) == std::ranges::binary_search( values, v ) ? 0 : 1; }
                        break;
                    }
                    case 2: {
                        key_type sum{ 0 };
                        std::size_t n{ 0 };
                        for ( auto const v : shared ) { sum = static_cast<key_type>( sum + v ); ++n; }
                        bad += ( sum != expected_sum || n != values.size() ) ? 1 : 0;
                        break;
                    }
                    case 3:
                        bad += shared.size() != values.size() ? 1 : 0;
                        bad += shared.intersects( other ) != !expected_inter.empty() ? 1 : 0;
                        bad += other.intersects( shared ) != !expected_inter.empty() ? 1 : 0;
                        break;
                    case 4:
                        bad += ( shared | other ).size() != expected_union.size() ? 1 : 0;
                        bad += ( shared & other ).size() != expected_inter.size() ? 1 : 0;
                        bad += ( shared - other ).size() != expected_diff.size() ? 1 : 0;
                        break;
                    default: {
                        std::size_t n{ 0 };
                        key_type sum{ 0 };
                        shared.for_each( [&]( key_type const v ) { ++n; sum = static_cast<key_type>( sum + v ); } );
                        bad += ( sum != expected_sum || n != values.size() ) ? 1 : 0;
                        break;
                    }
                }
            }
            return bad;
        } ) };
        EXPECT_EQ( wrong, 0U ) << "readers saw wrong answers on a shared const bitmap";
        // The readers must not have changed what the bitmap holds.
        EXPECT_TRUE( matches( shared, values ) );
    }
}

TEST( BitmapConstConcurrency, Readers32 )    { concurrent_readers_case<bitmap<std::uint32_t>>(); }
TEST( BitmapConstConcurrency, ReadersCow32 ) { concurrent_readers_case<CowBitmap32>(); }
TEST( BitmapConstConcurrency, Readers64 )    { concurrent_readers_case<bitmap<std::uint64_t>>(); }

// Readers that share a bitmap with a copy of it must also leave the copy alone.
TEST( BitmapConstConcurrency, ReadersOfCopiesShareNothingMutable ) {
    using Bitmap = bitmap<std::uint64_t>;
    auto const values{ make_values<std::uint64_t>( 3, 700 ) };
    std::mt19937_64 rng{ 5 };
    Bitmap const original{ build<Bitmap>( build_path::out_of_order_adds, values, rng ) };
    Bitmap const copy{ original };
    auto const wrong{ run_readers( [&]( unsigned const t ) -> std::size_t {
        std::size_t bad{ 0 };
        Bitmap const & mine{ t % 2 == 0 ? original : copy };
        std::mt19937_64 local{ t };
        for ( std::size_t i{ 0 }; i < 20000; ++i ) {
            auto const v{ values[ local() % values.size() ] };
            bad += mine.contains( v ) ? 0 : 1;
            bad += mine.contains( static_cast<std::uint64_t>( v ^ 0x8000 ) ) == std::ranges::binary_search( values, v ^ 0x8000 ) ? 0 : 1;
        }
        return bad;
    } ) };
    EXPECT_EQ( wrong, 0U );
}

} // namespace
