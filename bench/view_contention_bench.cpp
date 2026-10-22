// Readers of published bitmaps: what it costs, as the number of threads grows, to combine bitmaps that other threads
// combine at the same time, with the bitmap itself as the operand (a copy of a counted bitmap writes one reference
// count per container, to lines the other readers hold) against a view of it (writes nothing it owns).
//
//   bands "and3" / "and5": every thread repeatedly ANDs three / five of the published bitmaps (64 chunks of dense, sparse and
//     run containers); "and3-small" / "and5-small" the same over bitmaps of thousands of small array containers, where the
//     per-container reference count of a copy is most of the work
//     cow-copy      a copy of the smallest, then `&=` of each of the others (the counted copy, what a reader without views does)
//     and-many-ptr  Bitmap::and_many over pointers to the bitmaps (the n-way AND walk, copies nothing)
//     views         Bitmap::and_many over views (the walk of and-many-ptr, over views)
//     many-into     Bitmap::and_many_into over views, with a result bitmap reused by the thread
//     views-into    intersect_into( view, scratch ), then `&= view` of the rest, with a scratch reused by the thread
//     views-card    and_many_cardinality over views
//     cr-and        CRoaring roaring_bitmap_and, then roaring_bitmap_and_inplace of the rest (copy-on-write off)
//     cr-and-cow    the same on bitmaps with copy-on-write on
//     cr-copy-cow   roaring_bitmap_copy of the smallest with copy-on-write on, then roaring_bitmap_and_inplace of the rest
//     cr-frozen     roaring_bitmap_and over frozen views (read-only operands), then roaring_bitmap_and_inplace
//   bands "copy": a copy of one published bitmap, then a small update of the copy, dropped again
//     cow-copy      the counted copy
//     borrowed      borrowed_copy( view )
//     deep-copy     a copy under the value-semantics policy, which copies every payload
//
// Each arm runs for --ms milliseconds on every thread, --rounds times in rotating order; the best round of each arm
// is reported (operations per second over all threads, and per thread). Threads are pinned one per CPU of --cpus
// when given (distinct physical cores are the caller's to choose).
//
//   frsr_roaring_view_contention_bench [--filter text] [--rounds n] [--ms ms] [--threads 1,2,4,8] [--cpus 0,1,2,3] [--main-cpu n] [--chunks n] [--small-chunks n]
//
// (--main-cpu keeps the thread that starts and stops the workers off their CPUs.)

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined( __linux__ )
#   include <pthread.h>
#   include <sched.h>
#endif
#if FRSR_ROARING_HAS_CROARING
#   include <roaring/roaring.h>
#endif

namespace {

using counted_bitmap = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>;
using value_bitmap   = frsr::roaring::bitmap<std::uint32_t>;
#ifndef FRSR_BENCH_BASELINE   // built against a tree that has no views: only the arms that do not need them
using frsr::roaring::view_of;
using frsr::roaring::typed_view;
#endif

struct options {
    std::string           filter;
    int                   rounds{ 5 };
    int                   ms{ 250 };
    std::vector<unsigned> threads{ 1, 2, 4, 8 };
    std::vector<int>      cpus;
    int                   main_cpu{ -1 };
    std::size_t           chunks{ 64 };
    std::size_t           small_chunks{ 4000 };
};

std::vector<int> parse_list( std::string const & text ) {
    std::vector<int> values;
    std::stringstream stream{ text };
    for ( std::string item; std::getline( stream, item, ',' ); ) { values.push_back( std::atoi( item.c_str() ) ); }
    return values;
}

void pin_to( int const cpu ) {
#if defined( __linux__ )
    if ( cpu < 0 ) { return; }
    cpu_set_t set;
    CPU_ZERO( &set );
    CPU_SET( cpu, &set );
    pthread_setaffinity_np( pthread_self(), sizeof( set ), &set );
#else
    (void)cpu;
#endif
}

// ---- the published bitmaps --------------------------------------------------------------------------------

enum class shape { dense, sparse, runs };

// `count` bitmaps of `chunks` chunks each, a mix of dense, sparse and run chunks, built as a consumer publishes them:
// parts folded in lazily and finished once. Every chunk of a bitmap is of the same chunks, so the multi-way AND meets
// in every chunk; dense chunks overlap heavily, so the results are not empty.
template <typename Bitmap>
std::vector<Bitmap> make_published( std::size_t const count, std::size_t const chunks ) {
    std::vector<Bitmap> result;
    for ( std::size_t j{ 0 }; j < count; ++j ) {
        std::mt19937_64 rng{ 1000 + j };
        std::vector<Bitmap> parts( 3 );
        for ( std::size_t c{ 0 }; c < chunks; ++c ) {
            auto const base{ static_cast<std::uint32_t>( ( 8 + c ) << 16 ) };
            auto const kind{ static_cast<shape>( ( c * 7 + j ) % 5 < 3 ? 0 : ( ( c + j ) % 2 ? 1 : 2 ) ) };   // mostly dense, some sparse, some runs
            auto & part{ parts[ c % parts.size() ] };
            switch ( kind ) {
                case shape::dense:  for ( int i{ 0 }; i < 36000; ++i ) { part.add( base + static_cast<std::uint32_t>( rng() & 0xFFFF ) ); } break;
                case shape::sparse: for ( int i{ 0 }; i < 900; ++i )   { part.add( base + static_cast<std::uint32_t>( rng() & 0xFFFF ) ); } break;
                case shape::runs: {
                    for ( int r{ 0 }; r < 8; ++r ) {
                        auto const begin{ static_cast<std::uint32_t>( rng() % 56000 ) };
                        part.add_closed_range( base + begin, base + begin + 3000 + static_cast<std::uint32_t>( rng() % 4000 ) );
                    }
                    break;
                }
            }
        }
        parts[ 1 ].optimize();
        Bitmap published;
        for ( auto const & part : parts ) { published.bulk_or_intermediate( part ); }
        published.bulk_or_finish();
        result.push_back( std::move( published ) );
    }
    return result;
}

// `count` bitmaps of `chunks` chunks of one small array each (a few hundred keys of the chunk's 65536), the chunks shared by all of
// them so that the n-way AND meets at every chunk. Finished by a lazy fold, as above.
template <typename Bitmap>
std::vector<Bitmap> make_published_small( std::size_t const count, std::size_t const chunks ) {
    std::vector<Bitmap> result;
    for ( std::size_t j{ 0 }; j < count; ++j ) {
        std::mt19937_64 rng{ 5000 + j };
        std::vector<Bitmap> parts( 2 );
        for ( std::size_t c{ 0 }; c < chunks; ++c ) {
            auto const base{ static_cast<std::uint32_t>( ( 8 + c ) << 16 ) };
            for ( int i{ 0 }; i != 300; ++i ) { parts[ c % parts.size() ].add( base + static_cast<std::uint32_t>( rng() % 3000 ) ); }
        }
        Bitmap published;
        for ( auto const & part : parts ) { published.bulk_or_intermediate( part ); }
        published.bulk_or_finish();
        result.push_back( std::move( published ) );
    }
    return result;
}

// ---- running an arm on several threads ----------------------------------------------------------------------

std::atomic<std::uint64_t> g_sink{};

double run_threads( unsigned const thread_count, options const & opts, std::function<std::function<std::uint64_t()>( unsigned const )> const & make_body ) {
    std::atomic<bool> go{ false }, stop{ false };
    std::atomic<unsigned> ready{ 0 };
    std::vector<std::uint64_t> counts( thread_count, 0 );
    std::vector<std::thread> threads;
    for ( unsigned t{ 0 }; t < thread_count; ++t ) {
        threads.emplace_back( [ & ]( unsigned const index ) {
            if ( !opts.cpus.empty() ) { pin_to( opts.cpus[ index % opts.cpus.size() ] ); }
            auto body{ make_body( index ) };
            std::uint64_t sink{ 0 };
            for ( int warm{ 0 }; warm < 3; ++warm ) { sink += body(); }
            ready.fetch_add( 1 );
            while ( !go.load( std::memory_order_acquire ) ) { std::this_thread::yield(); }
            std::uint64_t operations{ 0 };
            while ( !stop.load( std::memory_order_relaxed ) ) {
                for ( int batch{ 0 }; batch < 4; ++batch ) { sink += body(); ++operations; }
            }
            counts[ index ] = operations;
            g_sink.fetch_add( sink );
        }, t );
    }
    while ( ready.load() != thread_count ) { std::this_thread::yield(); }
    auto const begin{ std::chrono::steady_clock::now() };
    go.store( true, std::memory_order_release );
    std::this_thread::sleep_for( std::chrono::milliseconds( opts.ms ) );
    stop.store( true );
    for ( auto & thread : threads ) { thread.join(); }
    auto const seconds{ std::chrono::duration<double>( std::chrono::steady_clock::now() - begin ).count() };
    std::uint64_t total{ 0 };
    for ( auto const count : counts ) { total += count; }
    return static_cast<double>( total ) / seconds;
}

using body_factory = std::function<std::function<std::uint64_t()>( unsigned const )>;

void run_band( std::string const & name, std::vector<std::pair<std::string, body_factory>> arms, options const & opts ) {
    if ( !opts.filter.empty() && name.find( opts.filter ) == std::string::npos ) { return; }
    for ( auto const threads : opts.threads ) {
        std::vector<double> best( arms.size(), 0.0 );
        for ( int round{ 0 }; round < opts.rounds; ++round ) {
            for ( std::size_t j{ 0 }; j < arms.size(); ++j ) {
                auto const k{ ( j + static_cast<std::size_t>( round ) ) % arms.size() };
                best[ k ] = std::max( best[ k ], run_threads( threads, opts, arms[ k ].second ) );
            }
        }
        std::printf( "%-26s threads=%-2u", name.c_str(), threads );
        for ( std::size_t k{ 0 }; k < arms.size(); ++k ) {
            std::printf( " | %-11s %9.0f/s %8.0f/thr x%-5.2f", arms[ k ].first.c_str(), best[ k ], best[ k ] / threads, best[ k ] / best[ 0 ] );
        }
        std::printf( "\n" );
        std::fflush( stdout );
    }
}

// ---- the AND bands --------------------------------------------------------------------------------------------

#if FRSR_ROARING_HAS_CROARING
struct roaring_set {
    std::vector<roaring_bitmap_t *> plain;      // copy-on-write off
    std::vector<roaring_bitmap_t *> cow;        // copy-on-write on, shared once so that later copies only count
    std::vector<roaring_bitmap_t const *> frozen;
    std::vector<void *> frozen_buffers;
};

// 32-byte aligned, as the frozen format requires
void * aligned_buffer( std::size_t const bytes ) {
    auto const rounded{ ( bytes + 31 ) / 32 * 32 };
#   if defined( _WIN32 )
    return _aligned_malloc( rounded, 32 );
#   else
    return std::aligned_alloc( 32, rounded );
#   endif
}

roaring_set make_roaring_set( std::vector<counted_bitmap> const & published ) {
    roaring_set set;
    for ( auto const & bitmap : published ) {
        auto const keys{ bitmap.to_vector() };
        auto * const plain{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
        roaring_bitmap_run_optimize( plain );
        set.plain.push_back( plain );
        auto * const cow{ roaring_bitmap_copy( plain ) };
        roaring_bitmap_set_copy_on_write( cow, true );
        roaring_bitmap_free( roaring_bitmap_copy( cow ) );    // the first copy converts the containers to shared ones: done once, before the threads
        set.cow.push_back( cow );
        auto const bytes{ roaring_bitmap_frozen_size_in_bytes( plain ) };
        auto * const buffer{ aligned_buffer( bytes ) };
        roaring_bitmap_frozen_serialize( plain, static_cast<char *>( buffer ) );
        set.frozen_buffers.push_back( buffer );
        set.frozen.push_back( roaring_bitmap_frozen_view( static_cast<char const *>( buffer ), bytes ) );
    }
    return set;
}
#endif

void and_bands( options const & opts, bool const small ) {
    auto const published{ std::make_shared<std::vector<counted_bitmap>>( small ? make_published_small<counted_bitmap>( 6, opts.small_chunks )
                                                                               : make_published<counted_bitmap>( 6, opts.chunks ) ) };
#if FRSR_ROARING_HAS_CROARING
    auto const roaring{ std::make_shared<roaring_set>( make_roaring_set( *published ) ) };
#endif
    // the operands of iteration i: `operands` of the published bitmaps, starting at a rotating one; the smallest first
    auto const pick{ [ published ]( std::size_t const operands, std::size_t const iteration ) {
        std::vector<std::size_t> indices;
        for ( std::size_t k{ 0 }; k < operands; ++k ) { indices.push_back( ( iteration + k ) % published->size() ); }
        std::ranges::sort( indices, [ & ]( std::size_t const a, std::size_t const b ) { return ( *published )[ a ].size() < ( *published )[ b ].size(); } );
        return indices;
    } };
    for ( std::size_t const operands : { std::size_t{ 3 }, std::size_t{ 5 } } ) {
        std::vector<std::pair<std::string, body_factory>> arms;
        arms.emplace_back( "cow-copy", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t } ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                counted_bitmap acc{ ( *published )[ indices[ 0 ] ] };
                for ( std::size_t k{ 1 }; k < indices.size(); ++k ) { acc &= ( *published )[ indices[ k ] ]; }
                return acc.size();
            };
        } );
        arms.emplace_back( "and-many-ptr", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t} ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                std::vector<counted_bitmap const *> pointers;
                for ( auto const k : indices ) { pointers.push_back( &( *published )[ k ] ); }
                return counted_bitmap::and_many( pointers ).size();
            };
        } );
#ifndef FRSR_BENCH_BASELINE
        arms.emplace_back( "views", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t } ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                std::vector<typed_view<counted_bitmap>> views;
                for ( auto const k : indices ) { views.push_back( view_of( ( *published )[ k ] ) ); }
                return counted_bitmap::and_many( views ).size();
            };
        } );
        arms.emplace_back( "many-into", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t }, out = counted_bitmap{} ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                std::vector<typed_view<counted_bitmap>> views;
                for ( auto const k : indices ) { views.push_back( view_of( ( *published )[ k ] ) ); }
                counted_bitmap::and_many_into( views, out );
                return out.size();
            };
        } );
        arms.emplace_back( "views-into", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t }, scratch = counted_bitmap{} ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                ( *published )[ indices[ 0 ] ].intersect_into( view_of( ( *published )[ indices[ 1 ] ] ), scratch );
                for ( std::size_t k{ 2 }; k < indices.size(); ++k ) { scratch &= view_of( ( *published )[ indices[ k ] ] ); }
                return scratch.size();
            };
        } );
        arms.emplace_back( "views-card", [ published, operands, pick ]( unsigned const t ) {
            return [ published, operands, pick, i = std::size_t{ t } ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                std::vector<typed_view<counted_bitmap>> views;
                for ( auto const k : indices ) { views.push_back( view_of( ( *published )[ k ] ) ); }
                return counted_bitmap::and_many_cardinality( views );
            };
        } );
#endif
#if FRSR_ROARING_HAS_CROARING
        auto const croaring_and{ [ roaring, operands, pick ]( std::vector<roaring_bitmap_t const *> const & pool ) -> body_factory {
            return [ roaring, operands, pick, &pool ]( unsigned const t ) {
                return [ operands, pick, &pool, i = std::size_t{ t } ]() mutable -> std::uint64_t {
                    auto const indices{ pick( operands, i++ ) };
                    auto * const acc{ roaring_bitmap_and( pool[ indices[ 0 ] ], pool[ indices[ 1 ] ] ) };
                    for ( std::size_t k{ 2 }; k < indices.size(); ++k ) { roaring_bitmap_and_inplace( acc, pool[ indices[ k ] ] ); }
                    auto const n{ roaring_bitmap_get_cardinality( acc ) };
                    roaring_bitmap_free( acc );
                    return n;
                };
            };
        } };
        static std::vector<roaring_bitmap_t const *> plain_pool, cow_pool;
        plain_pool.assign( roaring->plain.begin(), roaring->plain.end() );
        cow_pool.assign( roaring->cow.begin(), roaring->cow.end() );
        arms.emplace_back( "cr-and", croaring_and( plain_pool ) );
        arms.emplace_back( "cr-and-cow", croaring_and( cow_pool ) );
        arms.emplace_back( "cr-frozen", croaring_and( roaring->frozen ) );
        arms.emplace_back( "cr-copy-cow", [ roaring, operands, pick ]( unsigned const t ) {
            return [ roaring, operands, pick, i = std::size_t{ t } ]() mutable -> std::uint64_t {
                auto const indices{ pick( operands, i++ ) };
                auto * const acc{ roaring_bitmap_copy( roaring->cow[ indices[ 0 ] ] ) };
                for ( std::size_t k{ 1 }; k < indices.size(); ++k ) { roaring_bitmap_and_inplace( acc, roaring->cow[ indices[ k ] ] ); }
                auto const n{ roaring_bitmap_get_cardinality( acc ) };
                roaring_bitmap_free( acc );
                return n;
            };
        } );
#endif
        run_band( "and" + std::to_string( operands ) + ( small ? "-small" : "" ), std::move( arms ), opts );
    }
}

// ---- the copy band ----------------------------------------------------------------------------------------------

void copy_band( options const & opts ) {
    auto const counted{ std::make_shared<counted_bitmap>( std::move( make_published<counted_bitmap>( 1, opts.chunks ).front() ) ) };
    auto const value{ std::make_shared<value_bitmap>( std::move( make_published<value_bitmap>( 1, opts.chunks ).front() ) ) };
    // the update of a copy: a few keys into 1% of the containers (at least one), which clones those containers
    auto const touched{ std::max<std::size_t>( 1, opts.chunks / 100 ) };
    auto const update{ [ touched ]( auto & copy, std::size_t const step ) {
        for ( std::size_t c{ 0 }; c < touched; ++c ) {
            auto const base{ static_cast<std::uint32_t>( ( 8 + ( step * 7 + c * 13 ) % 61 ) << 16 ) };
            for ( std::uint32_t k{ 0 }; k < 64; ++k ) { copy.add( base + static_cast<std::uint32_t>( ( step * 977 + k * 1009 ) & 0xFFFF ) ); }
        }
    } };
    for ( bool const mutate : { false, true } ) {
        std::vector<std::pair<std::string, body_factory>> arms;
        arms.emplace_back( "cow-copy", [ counted, mutate, update ]( unsigned const ) {
            return [ counted, mutate, update, step = std::size_t{ 1 } ]() mutable -> std::uint64_t {
                counted_bitmap copy{ *counted };
                if ( mutate ) { update( copy, step++ ); }
                return copy.container_count();
            };
        } );
#ifndef FRSR_BENCH_BASELINE
        arms.emplace_back( "borrowed", [ counted, mutate, update ]( unsigned const ) {
            return [ counted, mutate, update, step = std::size_t{ 1 } ]() mutable -> std::uint64_t {
                auto copy{ counted_bitmap::borrowed_copy( view_of( *counted ) ) };
                if ( mutate ) { update( copy, step++ ); }
                return copy.container_count();
            };
        } );
#endif
        arms.emplace_back( "deep-copy", [ value, mutate, update ]( unsigned const ) {
            return [ value, mutate, update, step = std::size_t{ 1 } ]() mutable -> std::uint64_t {
                value_bitmap copy{ *value };
                if ( mutate ) { update( copy, step++ ); }
                return copy.container_count();
            };
        } );
        run_band( std::string{ "copy" } + ( mutate ? "+update" : "" ), std::move( arms ), opts );
    }
}

} // namespace

int main( int argc, char ** argv ) {
    options opts;
    for ( int i{ 1 }; i < argc; ++i ) {
        std::string const arg{ argv[ i ] };
        if ( arg == "--filter" && i + 1 < argc ) { opts.filter = argv[ ++i ]; }
        else if ( arg == "--rounds" && i + 1 < argc ) { opts.rounds = std::atoi( argv[ ++i ] ); }
        else if ( arg == "--ms" && i + 1 < argc ) { opts.ms = std::atoi( argv[ ++i ] ); }
        else if ( arg == "--chunks" && i + 1 < argc ) { opts.chunks = static_cast<std::size_t>( std::atoi( argv[ ++i ] ) ); }
        else if ( arg == "--small-chunks" && i + 1 < argc ) { opts.small_chunks = static_cast<std::size_t>( std::atoi( argv[ ++i ] ) ); }
        else if ( arg == "--threads" && i + 1 < argc ) { opts.threads.clear(); for ( auto const t : parse_list( argv[ ++i ] ) ) { opts.threads.push_back( static_cast<unsigned>( t ) ); } }
        else if ( arg == "--cpus" && i + 1 < argc ) { opts.cpus = parse_list( argv[ ++i ] ); }
        else if ( arg == "--main-cpu" && i + 1 < argc ) { opts.main_cpu = std::atoi( argv[ ++i ] ); }
        else { std::fprintf( stderr, "unknown argument: %s\n", arg.c_str() ); return 2; }
    }
    pin_to( opts.main_cpu );
    and_bands( opts, false );
    and_bands( opts, true );
    copy_band( opts );
    std::fprintf( stderr, "# sink %llu\n", static_cast<unsigned long long>( g_sink.load() ) );
    return 0;
}
