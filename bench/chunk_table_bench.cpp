// Chunk-table bands: the cost of locating, inserting and removing a chunk in detail::chunk_store,
// the structure bitmap<Key> keeps its containers in, measured on the store itself so that no
// container work is in the number. The same code runs for every key width; which structure sits
// behind the store (a sorted array or a tree) is the store's choice.
//
// usage: frsr_roaring_chunk_table_bench [--width 32|64] [--count N] [--pattern dense|random] [--band substring] [--reps R]
//                                       [--scatter-cap N]   (skip the scattered-insert bands above N chunks: they are quadratic for a flat array)
//
// Keys are even, so that key + 1 is absent inside the stored range. A lookup is a lower_bound,
// a key compare and a touch of the handle, which is what bitmap::contains does before it
// reaches into the container.
#include <frsr/roaring/chunk_store.hpp>
#include <frsr/roaring/container_layout.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#if defined( __GLIBC__ )
#   include <malloc.h>
#endif

namespace {

using namespace frsr::roaring;
using Clock = std::chrono::steady_clock;

std::uint64_t g_sink;

struct config {
    int         reps{ 3 };
    int         width{ 64 };
    std::size_t count{ 100000 };
    std::string pattern{ "random" };
    std::string band;
    std::size_t scatter_cap{ 200000 };   // above this many chunks the bands that build or edit by scattered inserts are skipped
};
config C;

bool want( char const * band ) { return C.band.empty() || std::strstr( band, C.band.c_str() ); }

void report( char const * band, double const value, char const * unit = "ns/op" ) {
    std::printf( "%-12s w%-3d %-6s n=%-8zu %10.2f %s\n", band, C.width, C.pattern.c_str(), C.count, value, unit );
    std::fflush( stdout );
}

template <typename F>
double best( double const ops, F && f ) {
    double b{ 1e30 };
    for ( int r{ 0 }; r < C.reps; ++r ) {
        auto const t0{ Clock::now() };
        g_sink += f();
        auto const t1{ Clock::now() };
        b = std::min( b, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / ops );
    }
    return b;
}

std::size_t heap_in_use() {
#if defined( __GLIBC__ )
    return mallinfo2().uordblks;
#else
    return 0;
#endif
}

template <std::unsigned_integral Key>
void run_bands() {
    using layout = detail::default_layout<Key>;
    using store  = detail::chunk_store<layout>;
    using handle = typename store::handle_type;
    using key_t  = typename layout::chunk_type;

    std::mt19937_64 rng{ 12345 };
    std::vector<key_t> keys;
    keys.reserve( C.count );
    auto const domain{ std::numeric_limits<key_t>::digits >= 48 ? 46 : std::numeric_limits<key_t>::digits - 2 };
    if ( C.pattern == "dense" ) {
        for ( std::size_t i{ 0 }; i < C.count; ++i ) { keys.push_back( static_cast<key_t>( 2 * i ) ); }
    } else {
        for ( std::size_t i{ 0 }; i < C.count; ++i ) { keys.push_back( static_cast<key_t>( ( rng() >> ( 64 - domain ) ) << 1 ) ); }
        std::sort( keys.begin(), keys.end() );
        keys.erase( std::unique( keys.begin(), keys.end() ), keys.end() );
    }
    auto const n{ keys.size() };

    auto const add = [] ( store & s, key_t const k ) {
        auto const at{ s.lower_bound( k ) };
        if ( at == s.end() || s.key( at ) != k ) { (void)s.insert( at, k, handle{} ); }
    };
    auto const del = [] ( store & s, key_t const k ) -> std::uint64_t {
        auto const at{ s.lower_bound( k ) };
        if ( at == s.end() || s.key( at ) != k ) { return 0; }
        (void)s.erase( at );
        return 1;
    };
    auto const build = [&] ( store & s ) { for ( auto const k : keys ) { s.push_back( k, handle{} ); } };
    auto const has = [] ( store const & s, key_t const k ) -> std::uint64_t {
        auto const at{ s.lower_bound( k ) };
        if ( at == s.end() || s.key( at ) != k ) { return 0; }
        return s.slot( at ).cardinality() + 1;
    };

    {
        store s;
        build( s );
        constexpr std::size_t P{ 200000 };
        std::vector<key_t> hit, miss, asc;
        for ( std::size_t i{ 0 }; i < P; ++i ) { auto const k{ keys[ rng() % n ] }; hit.push_back( k ); miss.push_back( k + 1 ); }
        auto const start{ rng() % n };
        for ( std::size_t i{ 0 }; i < P; ++i ) { asc.push_back( keys[ ( start + i ) % n ] ); }
        auto const lookups = [&] ( char const * name, std::vector<key_t> const & v ) {
            if ( !want( name ) ) { return; }
            report( name, best( static_cast<double>( v.size() ) * 5, [&] { std::uint64_t h{}; for ( int r{ 0 }; r < 5; ++r ) { for ( auto const k : v ) { h += has( s, k ); } } return h; } ) );
        };
        lookups( "hitRnd", hit );
        lookups( "missRnd", miss );
        lookups( "hitAsc", asc );
        if ( want( "lowerBound" ) ) {
            report( "lowerBound", best( static_cast<double>( miss.size() ) * 5, [&] {
                std::uint64_t h{};
                for ( int r{ 0 }; r < 5; ++r ) {
                    for ( auto const k : miss ) {
                        auto const at{ s.lower_bound( k ) };
                        if ( at != s.end() ) { h += s.key( at ) + s.slot( at ).cardinality(); }
                    }
                }
                return h;
            } ) );
        }
        if ( want( "iterate" ) ) {
            report( "iterate", best( static_cast<double>( n ), [&] {
                std::uint64_t h{};
                for ( auto at{ s.begin() }; at != s.end(); ++at ) { h += s.key( at ) + s.slot( at ).cardinality(); }
                return h;
            } ) );
        }
    }

    bool const scattered_ok{ n <= C.scatter_cap };
    if ( want( "mem" ) && scattered_ok ) {
        // heap bytes per chunk (node slack included), for an ascending and a scattered build
        auto const h0{ heap_in_use() };
        {
            store s;
            build( s );
            report( "memAsc", static_cast<double>( heap_in_use() - h0 ) / static_cast<double>( n ), "B/chunk" );
        }
        auto shuffled{ keys };
        std::shuffle( shuffled.begin(), shuffled.end(), std::mt19937_64{ 3 } );
        auto const h1{ heap_in_use() };
        store s;
        for ( auto const k : shuffled ) { add( s, k ); }
        report( "memScat", static_cast<double>( heap_in_use() - h1 ) / static_cast<double>( n ), "B/chunk" );
    }

    // mutations at steady state around n chunks: K absent keys go in, then the same K come out
    auto const K{ std::min<std::size_t>( n / 2, n >= 1000000 ? 5000 : 20000 ) };
    std::vector<key_t> odd;
    {
        std::vector<std::size_t> order( n );
        std::iota( order.begin(), order.end(), std::size_t{ 0 } );
        std::shuffle( order.begin(), order.end(), rng );
        for ( std::size_t i{ 0 }; i < K; ++i ) { odd.push_back( keys[ order[ i ] ] + 1 ); }
    }
    auto odd_out{ odd };
    std::shuffle( odd_out.begin(), odd_out.end(), rng );
    for ( int scattered_base{ 0 }; scattered_base < 2; ++scattered_base ) {
        char const * const ni{ scattered_base ? "insScatS" : "insScat" };
        char const * const nr{ scattered_base ? "remScatS" : "remScat" };
        if ( !( want( ni ) || want( nr ) ) || !scattered_ok ) { continue; }
        auto shuffled{ keys };
        std::shuffle( shuffled.begin(), shuffled.end(), std::mt19937_64{ 5 } );
        double bi{ 1e30 }, br{ 1e30 };
        for ( int r{ 0 }; r < C.reps; ++r ) {
            store s;
            if ( scattered_base ) { for ( auto const k : shuffled ) { add( s, k ); } } else { build( s ); }
            auto const t0{ Clock::now() };
            for ( auto const k : odd ) { add( s, k ); }
            auto const t1{ Clock::now() };
            for ( auto const k : odd_out ) { g_sink += del( s, k ); }
            auto const t2{ Clock::now() };
            bi = std::min( bi, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( K ) );
            br = std::min( br, std::chrono::duration<double, std::nano>( t2 - t1 ).count() / static_cast<double>( K ) );
        }
        if ( want( ni ) ) { report( ni, bi ); }
        if ( want( nr ) ) { report( nr, br ); }
    }
    if ( want( "tbl" ) ) {
        // two interleaved tables walked together, as the set operations walk the chunk tables of
        // their operands: a copy, and a union / intersection materialized by appends
        store left, right;
        for ( std::size_t i{ 0 }; i < n; ++i ) {
            if ( i % 3 != 0 ) { left.push_back( keys[ i ], handle{} ); }
            if ( i % 3 == 0 || i % 2 == 0 ) { right.push_back( keys[ i ], handle{} ); }
        }
        auto const walk = [&] ( bool const keep_unmatched ) {
            store result;
            result.reserve( left.size() + right.size() );
            auto l{ left.begin() };
            auto r{ right.begin() };
            while ( l != left.end() && r != right.end() ) {
                auto const lk{ left.key( l ) };
                auto const rk{ right.key( r ) };
                if ( lk < rk ) {
                    if ( keep_unmatched ) { result.push_back( lk, left.slot( l ) ); }
                    ++l;
                } else if ( rk < lk ) {
                    if ( keep_unmatched ) { result.push_back( rk, right.slot( r ) ); }
                    ++r;
                } else {
                    result.push_back( lk, left.slot( l ) );
                    ++l;
                    ++r;
                }
            }
            for ( ; keep_unmatched && l != left.end(); ++l ) { result.push_back( left.key( l ), left.slot( l ) ); }
            for ( ; keep_unmatched && r != right.end(); ++r ) { result.push_back( right.key( r ), right.slot( r ) ); }
            return static_cast<std::uint64_t>( result.size() );
        };
        report( "tblCopy", best( static_cast<double>( n ), [&] { store copy{ left }; return std::uint64_t{ copy.size() }; } ) );
        report( "tblOr"  , best( static_cast<double>( n ), [&] { return walk( true  ); } ) );
        report( "tblAnd" , best( static_cast<double>( n ), [&] { return walk( false ); } ) );
    }
    if ( want( "insAsc" ) ) {
        report( "insAsc", best( static_cast<double>( n ), [&] { store s; build( s ); return std::uint64_t{ s.size() }; } ) );
    }
    if ( want( "remFront" ) ) {
        auto const F{ std::min<std::size_t>( n, 1000 ) };
        double b{ 1e30 };
        for ( int r{ 0 }; r < C.reps; ++r ) {
            store s;
            build( s );
            auto const t0{ Clock::now() };
            for ( std::size_t i{ 0 }; i < F; ++i ) { g_sink += del( s, keys[ i ] ); }
            auto const t1{ Clock::now() };
            b = std::min( b, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( F ) );
        }
        report( "remFront", b );
    }
}

} // namespace

int main( int argc, char ** argv ) {
    for ( int i{ 1 }; i + 1 < argc; i += 2 ) {
        std::string const a{ argv[ i ] };
        std::string const v{ argv[ i + 1 ] };
        if      ( a == "--width"   ) { C.width   = std::stoi( v ); }
        else if ( a == "--count"   ) { C.count   = std::stoull( v ); }
        else if ( a == "--pattern" ) { C.pattern = v; }
        else if ( a == "--band"    ) { C.band    = v; }
        else if ( a == "--reps"    ) { C.reps    = std::stoi( v ); }
        else if ( a == "--scatter-cap" ) { C.scatter_cap = std::stoull( v ); }
    }
    if ( C.width == 32 ) { run_bands<std::uint32_t>(); } else { run_bands<std::uint64_t>(); }
    std::fprintf( stderr, "%llu\n", static_cast<unsigned long long>( g_sink ) );
}
