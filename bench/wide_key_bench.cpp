// Wide-key (64-bit) chunk-index bands: bitmap<std::uint64_t> against CRoaring's
// roaring64_bitmap_t on the operations whose cost is dominated by locating or
// inserting a chunk. Each band is timed as a tight loop in one process (no
// per-op harness), best of several repeats, ns per operation.
//
// usage: frsr_roaring_wide_key_bench [--count N]... [--pattern dense|sparse|random|clustered]...
//                                    [--band substring] [--reps R] [--arm frsr|r64]
//                                    [--scatter-cap N] [--lookups-only|--mutations-only]
//                                    [--narrow]   (32-bit lookup bands only: bitmap<uint32_t> vs roaring_bitmap_t)
//
// Chunk keys are even (2*i, scaled by the pattern) so that the odd chunk keys in
// between are absent inside the stored range: a miss probe has to search, not just
// compare against the end.
#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if FRSR_ROARING_HAS_CROARING
#include <roaring/roaring.h>
#include <roaring/roaring64.h>
#endif

namespace {

using u64 = std::uint64_t;
using frsr_bitmap = frsr::roaring::bitmap<u64>;
using Clock = std::chrono::steady_clock;

u64 g_sink;

template <class F>
double best_ns_per_op( int const reps, std::size_t const ops, F && f ) {
    double best{ 1e30 };
    for ( int r{ 0 }; r < reps; ++r ) {
        auto const t0{ Clock::now() };
        g_sink += f();
        auto const t1{ Clock::now() };
        best = std::min( best, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( ops ) );
    }
    return best;
}

// Times only `destroy( make() )`'s second half: the object is built outside the clock.
template <class Make, class Destroy>
double best_destroy_ns_per_op( int const reps, std::size_t const ops, Make && make, Destroy && destroy ) {
    double best{ 1e30 };
    for ( int r{ 0 }; r < reps; ++r ) {
        auto * const object{ make() };
        auto const t0{ Clock::now() };
        destroy( object );
        auto const t1{ Clock::now() };
        best = std::min( best, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( ops ) );
    }
    return best;
}

enum class pattern { dense, sparse, random, clustered };

char const * name( pattern const p ) { switch ( p ) {
        case pattern::dense: return "dense";
        case pattern::sparse: return "sparse";
        case pattern::random: return "random";
        case pattern::clustered: return "cluster";
    }
    return "?";
}

// Sorted, distinct, EVEN chunk keys; the value stored in a chunk is chunk<<16 | 7.
std::vector<u64> make_chunks( pattern const p, std::size_t const n ) {
    std::vector<u64> chunks;
    chunks.reserve( n );
    std::mt19937_64 rng{ 12345 };
    switch ( p ) {
        case pattern::dense : for ( std::size_t i{ 0 }; i < n; ++i ) { chunks.push_back( 2 * i ); } break;
        case pattern::sparse: for ( std::size_t i{ 0 }; i < n; ++i ) { chunks.push_back( ( 2 * i ) << 16 ); } break;
        case pattern::clustered:   // eight dense runs, far apart: not one spread a flat directory could cover
            for ( std::size_t i{ 0 }; i < n; ++i ) { chunks.push_back( ( ( i % 8 ) << 38 ) + 2 * ( i / 8 ) ); }
            std::sort( chunks.begin(), chunks.end() );
            break;
        case pattern::random: {
            for ( std::size_t i{ 0 }; i < n; ++i ) { chunks.push_back( ( rng() >> 17 ) << 1 ); }
            std::sort( chunks.begin(), chunks.end() );
            chunks.erase( std::unique( chunks.begin(), chunks.end() ), chunks.end() );
            break;
        }
    }
    return chunks;
}

u64 value_of( u64 const chunk ) { return ( chunk << 16 ) | 7; }

struct config {
    int reps{ 3 };
    std::string band_filter;
    std::string arm_filter;
    std::size_t scatter_cap{ 200000 }; // largest chunk count for the per-chunk insert/erase bands
};

bool selected( config const & c, char const * band, char const * arm ) {
    return ( c.band_filter.empty() || std::strstr( band, c.band_filter.c_str() ) ) &&
           ( c.arm_filter.empty() || c.arm_filter == arm );
}

void report( char const * band, char const * arm, pattern const p, std::size_t const n, double const ns ) {
    std::printf( "%-18s %-5s %-6s n=%-8zu %9.2f ns/op\n", band, arm, name( p ), n, ns );
    std::fflush( stdout );
}

struct probes {
    std::vector<u64> asc_hit, rnd_hit, rnd_miss;
};

probes make_probes( std::vector<u64> const & chunks ) {
    constexpr std::size_t kProbes{ 200000 };
    probes pr;
    std::mt19937_64 rng{ 99 };
    auto const n{ chunks.size() };
    auto const start{ rng() % n };
    for ( std::size_t i{ 0 }; i < std::min( kProbes, n ); ++i ) { pr.asc_hit.push_back( value_of( chunks[ ( start + i ) % n ] ) ); }
    for ( std::size_t i{ 0 }; i < kProbes; ++i ) {
        auto const c{ chunks[ rng() % n ] };
        pr.rnd_hit .push_back( value_of( c ) );
        pr.rnd_miss.push_back( value_of( c + 1 ) ); // odd chunk: absent
    }
    return pr;
}

void run_lookup_bands( config const & cfg, pattern const p, std::size_t const requested ) {
    auto const chunks{ make_chunks( p, requested ) };
    auto const n{ chunks.size() };
    auto const pr{ make_probes( chunks ) };

    auto const frsr_bands = [&]() {
        frsr_bitmap b;
        for ( auto const c : chunks ) { (void)b.add( value_of( c ) ); }
        struct { char const * band; std::vector<u64> const * v; bool expect; } const bands[]{
            { "containsAscHit", &pr.asc_hit, true }, { "containsRndHit", &pr.rnd_hit, true }, { "containsRndMiss", &pr.rnd_miss, false } };
        for ( auto const & bd : bands ) {
            if ( !selected( cfg, bd.band, "frsr" ) ) { continue; }
            auto const ns{ best_ns_per_op( cfg.reps, bd.v->size() * 5, [&] {
                u64 hits{};
                for ( int rep{ 0 }; rep < 5; ++rep ) { for ( auto const v : *bd.v ) { hits += b.contains( v ); } }
                return hits;
            } ) };
            report( bd.band, "frsr", p, n, ns );
        }
        if ( selected( cfg, "containsBulkAsc", "frsr" ) ) {
            auto const ns{ best_ns_per_op( cfg.reps, pr.asc_hit.size() * 5, [&] {
                u64 hits{};
                for ( int rep{ 0 }; rep < 5; ++rep ) { frsr_bitmap::bulk_context ctx; for ( auto const v : pr.asc_hit ) { hits += b.contains_bulk( ctx, v ); } }
                return hits;
            } ) };
            report( "containsBulkAsc", "frsr", p, n, ns );
        }
    };
    frsr_bands();
#if FRSR_ROARING_HAS_CROARING
    {
        auto * r{ roaring64_bitmap_create() };
        for ( auto const c : chunks ) { roaring64_bitmap_add( r, value_of( c ) ); }
        struct { char const * band; std::vector<u64> const * v; } const bands[]{
            { "containsAscHit", &pr.asc_hit }, { "containsRndHit", &pr.rnd_hit }, { "containsRndMiss", &pr.rnd_miss } };
        for ( auto const & bd : bands ) {
            if ( !selected( cfg, bd.band, "r64" ) ) { continue; }
            auto const ns{ best_ns_per_op( cfg.reps, bd.v->size() * 5, [&] {
                u64 hits{};
                for ( int rep{ 0 }; rep < 5; ++rep ) { for ( auto const v : *bd.v ) { hits += roaring64_bitmap_contains( r, v ); } }
                return hits;
            } ) };
            report( bd.band, "r64", p, n, ns );
        }
        if ( selected( cfg, "containsBulkAsc", "r64" ) ) {
            auto const ns{ best_ns_per_op( cfg.reps, pr.asc_hit.size() * 5, [&] {
                u64 hits{};
                for ( int rep{ 0 }; rep < 5; ++rep ) { roaring64_bulk_context_t ctx{}; for ( auto const v : pr.asc_hit ) { hits += roaring64_bitmap_contains_bulk( r, &ctx, v ); } }
                return hits;
            } ) };
            report( "containsBulkAsc", "r64", p, n, ns );
        }
        roaring64_bitmap_free( r );
    }
#endif
}

void run_mutation_bands( config const & cfg, pattern const p, std::size_t const requested ) {
    auto const chunks{ make_chunks( p, requested ) };
    auto const n{ chunks.size() };
    std::vector<u64> values; values.reserve( n );
    for ( auto const c : chunks ) { values.push_back( value_of( c ) ); }
    auto shuffled{ values };
    std::shuffle( shuffled.begin(), shuffled.end(), std::mt19937_64{ 7 } );

    // Per-chunk insert/erase on the flat chunk array shifts its tail (quadratic over n), so those
    // bands are only run up to a bounded chunk count.
    bool const frsr_scattered_ok{ n <= cfg.scatter_cap };

    if ( selected( cfg, "addScattered", "frsr" ) && frsr_scattered_ok ) {
        report( "addScattered", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { frsr_bitmap b; for ( auto const v : shuffled ) { (void)b.add( v ); } return b.size(); } ) );
    }
    if ( selected( cfg, "addAscending", "frsr" ) ) {
        report( "addAscending", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { frsr_bitmap b; for ( auto const v : values ) { (void)b.add( v ); } return b.size(); } ) );
    }
    if ( selected( cfg, "addMany", "frsr" ) ) {
        report( "addMany", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { frsr_bitmap b; b.add_many( values ); return b.size(); } ) );
    }
    if ( selected( cfg, "removeScattered", "frsr" ) && frsr_scattered_ok ) {
        frsr_bitmap base; base.add_many( values );
        report( "removeScattered", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { auto b{ base }; u64 r{}; for ( auto const v : shuffled ) { r += b.remove( v ); } return r; } ) );
    }
    if ( selected( cfg, "removeAscending", "frsr" ) && frsr_scattered_ok ) {
        frsr_bitmap base; base.add_many( values );
        report( "removeAscending", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { auto b{ base }; u64 r{}; for ( auto const v : values ) { r += b.remove( v ); } return r; } ) );
    }
    if ( selected( cfg, "iterate", "frsr" ) ) {
        frsr_bitmap base; base.add_many( values );
        report( "iterate", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { u64 s{}; base.for_each( [&]( u64 const v ) { s += v; } ); return s; } ) );
    }
    if ( selected( cfg, "copy", "frsr" ) ) {
        frsr_bitmap base; base.add_many( values );
        report( "copy", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { frsr_bitmap c{ base }; return c.size(); } ) );
    }
    if ( selected( cfg, "destroy", "frsr" ) ) {
        frsr_bitmap base; base.add_many( values );
        report( "destroy", "frsr", p, n, best_destroy_ns_per_op( cfg.reps, n, [&] { return new frsr_bitmap{ base }; }, []( frsr_bitmap * b ) { delete b; } ) );
    }
#if FRSR_ROARING_HAS_CROARING
    if ( selected( cfg, "destroy", "r64" ) ) {
        auto * base{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( base, values.size(), values.data() );
        report( "destroy", "r64", p, n, best_destroy_ns_per_op( cfg.reps, n, [&] { return roaring64_bitmap_copy( base ); }, []( roaring64_bitmap_t * b ) { roaring64_bitmap_free( b ); } ) );
        roaring64_bitmap_free( base );
    }
#endif
    {
        // two interleaved halves: union / intersection / difference walk the chunk tables together
        std::vector<u64> a, b;
        for ( std::size_t i{ 0 }; i < n; ++i ) { ( ( i % 3 ) != 0 ? a : b ).push_back( values[ i ] ); }
        for ( std::size_t i{ 0 }; i < n; i += 2 ) { b.push_back( values[ i ] ); }
        std::sort( b.begin(), b.end() ); b.erase( std::unique( b.begin(), b.end() ), b.end() );
        frsr_bitmap fa, fb; fa.add_many( a ); fb.add_many( b );
        if ( selected( cfg, "or", "frsr" ) ) { report( "or", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { return ( fa | fb ).size(); } ) ); }
        if ( selected( cfg, "and", "frsr" ) ) { report( "and", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { return ( fa & fb ).size(); } ) ); }
        if ( selected( cfg, "andnot", "frsr" ) ) { report( "andnot", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { return ( fa - fb ).size(); } ) ); }
        if ( selected( cfg, "orInplace", "frsr" ) ) { report( "orInplace", "frsr", p, n, best_ns_per_op( cfg.reps, n, [&] { auto c{ fa }; c |= fb; return c.size(); } ) ); }
#if FRSR_ROARING_HAS_CROARING
        auto * ra{ roaring64_bitmap_create() }; auto * rb{ roaring64_bitmap_create() };
        roaring64_bitmap_add_many( ra, a.size(), a.data() ); roaring64_bitmap_add_many( rb, b.size(), b.data() );
        if ( selected( cfg, "or", "r64" ) ) { report( "or", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_or( ra, rb ) }; auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) ); }
        if ( selected( cfg, "and", "r64" ) ) { report( "and", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_and( ra, rb ) }; auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) ); }
        if ( selected( cfg, "andnot", "r64" ) ) { report( "andnot", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_andnot( ra, rb ) }; auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) ); }
        if ( selected( cfg, "orInplace", "r64" ) ) { report( "orInplace", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * c{ roaring64_bitmap_copy( ra ) }; roaring64_bitmap_or_inplace( c, rb ); auto k{ roaring64_bitmap_get_cardinality( c ) }; roaring64_bitmap_free( c ); return k; } ) ); }
        roaring64_bitmap_free( ra ); roaring64_bitmap_free( rb );
#endif
    }

#if FRSR_ROARING_HAS_CROARING
    if ( selected( cfg, "addScattered", "r64" ) && frsr_scattered_ok ) {
        report( "addScattered", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_create() }; for ( auto const v : shuffled ) { roaring64_bitmap_add( r, v ); } auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) );
    }
    if ( selected( cfg, "addAscending", "r64" ) ) {
        report( "addAscending", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_create() }; for ( auto const v : values ) { roaring64_bitmap_add( r, v ); } auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) );
    }
    if ( selected( cfg, "addMany", "r64" ) ) {
        report( "addMany", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * r{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( r, values.size(), values.data() ); auto c{ roaring64_bitmap_get_cardinality( r ) }; roaring64_bitmap_free( r ); return c; } ) );
    }
    if ( selected( cfg, "removeScattered", "r64" ) && frsr_scattered_ok ) {
        auto * base{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( base, values.size(), values.data() );
        report( "removeScattered", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * c{ roaring64_bitmap_copy( base ) }; for ( auto const v : shuffled ) { roaring64_bitmap_remove( c, v ); } auto k{ roaring64_bitmap_get_cardinality( c ) }; roaring64_bitmap_free( c ); return k; } ) );
        roaring64_bitmap_free( base );
    }
    if ( selected( cfg, "removeAscending", "r64" ) && frsr_scattered_ok ) {
        auto * base{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( base, values.size(), values.data() );
        report( "removeAscending", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * c{ roaring64_bitmap_copy( base ) }; for ( auto const v : values ) { roaring64_bitmap_remove( c, v ); } auto k{ roaring64_bitmap_get_cardinality( c ) }; roaring64_bitmap_free( c ); return k; } ) );
        roaring64_bitmap_free( base );
    }
    if ( selected( cfg, "iterate", "r64" ) ) {
        auto * base{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( base, values.size(), values.data() );
        report( "iterate", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { u64 s{}; roaring64_bitmap_iterate( base, []( u64 const v, void * ptr ) { *static_cast<u64 *>( ptr ) += v; return true; }, &s ); return s; } ) );
        roaring64_bitmap_free( base );
    }
    if ( selected( cfg, "copy", "r64" ) ) {
        auto * base{ roaring64_bitmap_create() }; roaring64_bitmap_add_many( base, values.size(), values.data() );
        report( "copy", "r64", p, n, best_ns_per_op( cfg.reps, n, [&] { auto * c{ roaring64_bitmap_copy( base ) }; auto k{ roaring64_bitmap_get_cardinality( c ) }; roaring64_bitmap_free( c ); return k; } ) );
        roaring64_bitmap_free( base );
    }
#endif
}

// The 32-bit counterpart of the lookup bands: bitmap<uint32_t> against roaring_bitmap_t, whose
// chunk table holds at most 65536 sorted 16-bit keys. Chunk keys are even; value = chunk << 16 | 7.
void run_narrow_lookup_bands( config const & cfg, std::size_t const requested ) {
    auto const n{ std::min<std::size_t>( requested, 32768 ) };
    std::vector<std::uint32_t> present, absent, shuffled;
    for ( std::size_t i{ 0 }; i < n; ++i ) { present.push_back( static_cast<std::uint32_t>( ( 2 * i << 16 ) | 7 ) ); }
    shuffled = present;
    std::shuffle( shuffled.begin(), shuffled.end(), std::mt19937_64{ 5 } );
    for ( auto const v : shuffled ) { absent.push_back( v + ( 1u << 16 ) ); }
    struct { char const * band; std::vector<std::uint32_t> const * v; } const bands[]{
        { "containsAscHit", &present }, { "containsRndHit", &shuffled }, { "containsRndMiss", &absent } };
    auto const report32{ [&]( char const * band, char const * arm, double ns ) {
        std::printf( "%-18s %-5s %-6s n=%-8zu %9.2f ns/op\n", band, arm, "narrow", n, ns ); std::fflush( stdout ); } };
    {
        frsr::roaring::bitmap<std::uint32_t> b;
        for ( auto const v : present ) { (void)b.add( v ); }
        for ( auto const & bd : bands ) {
            if ( !selected( cfg, bd.band, "frsr32" ) ) { continue; }
            report32( bd.band, "frsr32", best_ns_per_op( cfg.reps, bd.v->size() * 20, [&] {
                u64 hits{}; for ( int rep{ 0 }; rep < 20; ++rep ) { for ( auto const v : *bd.v ) { hits += b.contains( v ); } } return hits; } ) );
        }
    }
#if FRSR_ROARING_HAS_CROARING
    {
        auto * r{ roaring_bitmap_create() };
        for ( auto const v : present ) { roaring_bitmap_add( r, v ); }
        for ( auto const & bd : bands ) {
            if ( !selected( cfg, bd.band, "r32" ) ) { continue; }
            report32( bd.band, "r32", best_ns_per_op( cfg.reps, bd.v->size() * 20, [&] {
                u64 hits{}; for ( int rep{ 0 }; rep < 20; ++rep ) { for ( auto const v : *bd.v ) { hits += roaring_bitmap_contains( r, v ); } } return hits; } ) );
        }
        roaring_bitmap_free( r );
    }
#endif
}

} // namespace

int main( int argc, char ** argv ) {
    config cfg;
    std::vector<std::size_t> counts;
    std::vector<pattern> patterns;
    bool mutations{ true }, lookups{ true }, narrow{ false };
    for ( int i{ 1 }; i < argc; ++i ) {
        std::string const a{ argv[ i ] };
        auto const next{ [&] { return i + 1 < argc ? std::string{ argv[ ++i ] } : std::string{}; } };
        if      ( a == "--count"   ) { counts.push_back( std::strtoull( next().c_str(), nullptr, 10 ) ); }
        else if ( a == "--pattern" ) { auto const s{ next() }; patterns.push_back( s == "dense" ? pattern::dense : s == "sparse" ? pattern::sparse : s == "clustered" ? pattern::clustered : pattern::random ); }
        else if ( a == "--band"    ) { cfg.band_filter = next(); }
        else if ( a == "--arm"     ) { cfg.arm_filter = next(); }
        else if ( a == "--scatter-cap" ) { cfg.scatter_cap = std::strtoull( next().c_str(), nullptr, 10 ); }
        else if ( a == "--reps"    ) { cfg.reps = std::atoi( next().c_str() ); }
        else if ( a == "--narrow"  ) { narrow = true; }
        else if ( a == "--lookups-only"   ) { mutations = false; }
        else if ( a == "--mutations-only" ) { lookups = false; }
    }
    if ( counts.empty() ) { counts = { 1000, 100000, 1000000 }; }
    if ( patterns.empty() ) { patterns = { pattern::dense, pattern::sparse, pattern::random, pattern::clustered }; }
    if ( narrow ) { for ( auto const n : counts ) { run_narrow_lookup_bands( cfg, n ); } return 0; }
    for ( auto const p : patterns ) {
        for ( auto const n : counts ) {
            if ( lookups   ) { run_lookup_bands  ( cfg, p, n ); }
            if ( mutations ) { run_mutation_bands( cfg, p, n ); }
        }
    }
    std::fprintf( stderr, "%llu\n", static_cast<unsigned long long>( g_sink ) );
}
