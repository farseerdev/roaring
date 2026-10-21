// Symmetric difference (xor) of bitmaps, against CRoaring doing the same.
//
// The bands mirror the container-pair and chunk-table shapes of the other set operations: array, bitset and run
// containers against each other (one chunk, and many), tables of many small chunks with 32- and 64-bit keys, the
// in-place and the cardinality-only forms, the N-way fold, and the operations against plain operands (a key
// interval, which is the flip of that interval, and a sorted key range).
//
// Every band runs its arms interleaved, round after round, and reports the fastest of the rounds per arm
// (nanoseconds per operation). A ratio is "CRoaring time / frsr time": above 1, frsr is faster. An operation that
// changes its bitmap starts every iteration with a copy of the target, which both arms pay; a "copy" column
// times that copy alone, and the "net" ratio subtracts it.
//
// Build both sides the same way for a fair comparison: -DFRSR_ROARING_NO_SBO=1 takes away the small-buffer
// storage the reference does not have.
//
//   frsr_roaring_symmetric_difference_bench [--filter text] [--rounds n] [--min-time-ms ms]

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <roaring/roaring.h>
#include <roaring/roaring64.h>

namespace {

using bitmap32 = frsr::roaring::bitmap<std::uint32_t>;
using bitmap64 = frsr::roaring::bitmap<std::uint64_t>;
using frsr::roaring::key_range;

using clock_type = std::chrono::steady_clock;

struct arm {
    std::string                    name;
    std::function<std::uint64_t()> run;
    double                         best_ns{ 1e300 };
};

struct band {
    std::string                    name;
    bool                           mutates{ false };
    std::function<std::uint64_t()> copy_frsr;      // the copy alone, per library, when the band mutates
    std::function<std::uint64_t()> copy_croaring;
    std::vector<arm>               arms;           // arms[ 0 ] is frsr, arms[ 1 ] is CRoaring
};

struct options {
    std::string filter;
    int         rounds{ 7 };
    double      min_time_ms{ 3.0 };
};

std::uint64_t g_sink{};

double time_arm( arm & a, std::int64_t const reps ) {
    auto const begin{ clock_type::now() };
    std::uint64_t sink{ 0 };
    for ( std::int64_t i{ 0 }; i < reps; ++i ) { sink += a.run(); }
    auto const end{ clock_type::now() };
    g_sink += sink;
    return std::chrono::duration<double, std::nano>( end - begin ).count() / static_cast<double>( reps );
}

void run_band( band & b, options const & opts ) {
    arm copy_frsr{ "copy", b.copy_frsr, 1e300 };
    arm copy_croaring{ "copy", b.copy_croaring, 1e300 };
    std::vector<arm *> all;
    for ( auto & a : b.arms ) { all.push_back( &a ); }
    if ( b.mutates ) { all.push_back( &copy_frsr ); all.push_back( &copy_croaring ); }
    std::vector<std::int64_t> reps( all.size(), 1 );
    for ( std::size_t k{ 0 }; k < all.size(); ++k ) {
        all[ k ]->run();
        auto const once{ time_arm( *all[ k ], 1 ) };
        reps[ k ] = std::clamp<std::int64_t>( static_cast<std::int64_t>( opts.min_time_ms * 1e6 / std::max( once, 1.0 ) ), 1, 200000 );
    }
    for ( int round{ 0 }; round < opts.rounds; ++round ) {
        for ( std::size_t j{ 0 }; j < all.size(); ++j ) {
            auto const k{ ( j + static_cast<std::size_t>( round ) ) % all.size() };
            all[ k ]->best_ns = std::min( all[ k ]->best_ns, time_arm( *all[ k ], reps[ k ] ) );
        }
    }
    auto const frsr_ns{ b.arms[ 0 ].best_ns };
    auto const croaring_ns{ b.arms[ 1 ].best_ns };
    auto const frsr_copy{ b.mutates ? copy_frsr.best_ns : 0.0 };
    auto const croaring_copy{ b.mutates ? copy_croaring.best_ns : 0.0 };
    auto const net_frsr{ frsr_ns - frsr_copy };
    auto const net_croaring{ croaring_ns - croaring_copy };
    std::printf( "%-58s frsr %10.0f  croaring %10.0f  x%-6.2f", b.name.c_str(), frsr_ns, croaring_ns, croaring_ns / frsr_ns );
    if ( b.mutates && net_frsr > 1.0 && net_croaring > 1.0 ) {
        std::printf( "  net %9.0f %9.0f x%-6.2f", net_frsr, net_croaring, net_croaring / net_frsr );
    }
    std::printf( "\n" );
    std::fflush( stdout );
}

std::vector<band> g_bands;

// ---- operands in every container form ---------------------------------------------------------------------------

enum class shape : std::uint8_t { array, bitset, runs_few, runs_many, runs_many_shifted };

// The low 16 bits of a chunk in the given shape (sorted, distinct); `n` is the number of values for the array and
// bitset shapes. `salt` lets two operands of one shape differ.
std::vector<std::uint16_t> lows_of( shape const kind, std::size_t const n, std::uint64_t const salt, std::mt19937_64 & rng ) {
    std::vector<std::uint16_t> lows;
    switch ( kind ) {
        case shape::array:
        case shape::bitset: {
            std::vector<bool> taken( 65536 );
            while ( lows.size() < n ) {
                auto const v{ static_cast<std::uint16_t>( rng() ) };
                if ( !taken[ v ] ) { taken[ v ] = true; lows.push_back( v ); }
            }
            std::ranges::sort( lows );
            break;
        }
        case shape::runs_few:
            for ( std::uint32_t r{ 0 }; r < 6; ++r ) {
                auto const begin{ static_cast<std::uint32_t>( 1000 * r + 13 * ( salt % 7 ) ) };
                for ( std::uint32_t v{ begin }; v < begin + 400 + 50 * r; ++v ) { lows.push_back( static_cast<std::uint16_t>( v ) ); }
            }
            break;
        case shape::runs_many:
        case shape::runs_many_shifted:
            for ( std::uint32_t start{ kind == shape::runs_many ? 7U : 10U }; start + 40 < 65536; start += 400 ) {
                for ( std::uint32_t v{ start }; v < start + 40; ++v ) { lows.push_back( static_cast<std::uint16_t>( v ) ); }
            }
            break;
    }
    return lows;
}

struct operand_pair {
    std::vector<std::uint64_t> a;
    std::vector<std::uint64_t> b;
    bool run_optimized{ false };
};

// `chunks` chunks at `stride` apart (chunk numbers, so a stride of 1 is adjacent chunks), each in shape ka / kb
operand_pair make_operands( shape const ka, std::size_t const na, shape const kb, std::size_t const nb, std::size_t const chunks, std::uint64_t const stride, std::uint64_t const first_chunk, std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    operand_pair p;
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        auto const base{ ( first_chunk + c * stride ) << 16 };
        for ( auto const low : lows_of( ka, na, c, rng ) ) { p.a.push_back( base + low ); }
        for ( auto const low : lows_of( kb, nb, c + 1, rng ) ) { p.b.push_back( base + low ); }
    }
    p.run_optimized = ka == shape::runs_few || ka == shape::runs_many || ka == shape::runs_many_shifted || kb == shape::runs_few || kb == shape::runs_many || kb == shape::runs_many_shifted;
    return p;
}

// ---- 32-bit bands --------------------------------------------------------------------------------------------------

struct pair32 {
    bitmap32           fa, fb;
    roaring_bitmap_t * ca{};
    roaring_bitmap_t * cb{};
    ~pair32() { roaring_bitmap_free( ca ); roaring_bitmap_free( cb ); }
    pair32() = default;
    pair32( pair32 const & ) = delete;
};

std::shared_ptr<pair32> build32( operand_pair const & p ) {
    auto s{ std::make_shared<pair32>() };
    std::vector<std::uint32_t> a( p.a.begin(), p.a.end() );
    std::vector<std::uint32_t> b( p.b.begin(), p.b.end() );
    s->fa.add_many( a );
    s->fb.add_many( b );
    s->ca = roaring_bitmap_of_ptr( a.size(), a.data() );
    s->cb = roaring_bitmap_of_ptr( b.size(), b.data() );
    // run encoding is chosen the same way on both sides, where the shapes are runs
    if ( p.run_optimized ) {
        s->fa.optimize();
        s->fb.optimize();
        roaring_bitmap_run_optimize( s->ca );
        roaring_bitmap_run_optimize( s->cb );
    }
    return s;
}

void bands32( std::string const & name, operand_pair const & p ) {
    auto const s{ build32( p ) };
    {
        band b;
        b.name = "xor " + name;
        b.arms.push_back( { "frsr", [ s ] { return std::uint64_t{ ( s->fa ^ s->fb ).size() }; } } );
        b.arms.push_back( { "croaring", [ s ] {
            auto * const r{ roaring_bitmap_xor( s->ca, s->cb ) };
            auto const n{ roaring_bitmap_get_cardinality( r ) };
            roaring_bitmap_free( r );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
    {
        band b;
        b.name = "xor in place " + name;
        b.mutates = true;
        b.copy_frsr = [ s ] { bitmap32 c{ s->fa }; return std::uint64_t{ c.size() }; };
        b.copy_croaring = [ s ] { auto * const c{ roaring_bitmap_copy( s->ca ) }; auto const n{ roaring_bitmap_get_cardinality( c ) }; roaring_bitmap_free( c ); return n; };
        b.arms.push_back( { "frsr", [ s ] { bitmap32 c{ s->fa }; c ^= s->fb; return std::uint64_t{ c.size() }; } } );
        b.arms.push_back( { "croaring", [ s ] {
            auto * const c{ roaring_bitmap_copy( s->ca ) };
            roaring_bitmap_xor_inplace( c, s->cb );
            auto const n{ roaring_bitmap_get_cardinality( c ) };
            roaring_bitmap_free( c );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
    {
        band b;
        b.name = "xor cardinality " + name;
        b.arms.push_back( { "frsr", [ s ] { return std::uint64_t{ s->fa.xor_cardinality( s->fb ) }; } } );
        b.arms.push_back( { "croaring", [ s ] { return roaring_bitmap_xor_cardinality( s->ca, s->cb ); } } );
        g_bands.push_back( std::move( b ) );
    }
}

struct pair64 {
    bitmap64             fa, fb;
    roaring64_bitmap_t * ca{};
    roaring64_bitmap_t * cb{};
    ~pair64() { roaring64_bitmap_free( ca ); roaring64_bitmap_free( cb ); }
    pair64() = default;
    pair64( pair64 const & ) = delete;
};

void bands64( std::string const & name, operand_pair const & p ) {
    auto s{ std::make_shared<pair64>() };
    s->fa.add_many( p.a );
    s->fb.add_many( p.b );
    s->ca = roaring64_bitmap_create();
    s->cb = roaring64_bitmap_create();
    roaring64_bitmap_add_many( s->ca, p.a.size(), p.a.data() );
    roaring64_bitmap_add_many( s->cb, p.b.size(), p.b.data() );
    if ( p.run_optimized ) {
        s->fa.optimize();
        s->fb.optimize();
        roaring64_bitmap_run_optimize( s->ca );
        roaring64_bitmap_run_optimize( s->cb );
    }
    {
        band b;
        b.name = "xor " + name;
        b.arms.push_back( { "frsr", [ s ] { return std::uint64_t{ ( s->fa ^ s->fb ).size() }; } } );
        b.arms.push_back( { "croaring", [ s ] {
            auto * const r{ roaring64_bitmap_xor( s->ca, s->cb ) };
            auto const n{ roaring64_bitmap_get_cardinality( r ) };
            roaring64_bitmap_free( r );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
    {
        band b;
        b.name = "xor in place " + name;
        b.mutates = true;
        b.copy_frsr = [ s ] { bitmap64 c{ s->fa }; return std::uint64_t{ c.size() }; };
        b.copy_croaring = [ s ] { auto * const c{ roaring64_bitmap_copy( s->ca ) }; auto const n{ roaring64_bitmap_get_cardinality( c ) }; roaring64_bitmap_free( c ); return n; };
        b.arms.push_back( { "frsr", [ s ] { bitmap64 c{ s->fa }; c ^= s->fb; return std::uint64_t{ c.size() }; } } );
        b.arms.push_back( { "croaring", [ s ] {
            auto * const c{ roaring64_bitmap_copy( s->ca ) };
            roaring64_bitmap_xor_inplace( c, s->cb );
            auto const n{ roaring64_bitmap_get_cardinality( c ) };
            roaring64_bitmap_free( c );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
    {
        band b;
        b.name = "xor cardinality " + name;
        b.arms.push_back( { "frsr", [ s ] { return std::uint64_t{ s->fa.xor_cardinality( s->fb ) }; } } );
        b.arms.push_back( { "croaring", [ s ] { return roaring64_bitmap_xor_cardinality( s->ca, s->cb ); } } );
        g_bands.push_back( std::move( b ) );
    }
}

struct pair_form {
    char const * name;
    shape        a;
    std::size_t  na;
    shape        b;
    std::size_t  nb;
};

constexpr pair_form pair_forms[]{
    { "array 100 x array 100",             shape::array,  100,    shape::array,  100 },
    { "array 1500 x array 1500",           shape::array,  1500,   shape::array,  1500 },
    { "array 3000 x array 3000 (to bitset)", shape::array,  3000,   shape::array,  3000 },
    { "array 4000 x array 4000",           shape::array,  4000,   shape::array,  4000 },
    { "bitset 8000 x bitset 8000",         shape::bitset, 8000,   shape::bitset, 8000 },
    { "bitset 30000 x bitset 30000",       shape::bitset, 30000,  shape::bitset, 30000 },
    { "array 1000 x bitset 20000",         shape::array,  1000,   shape::bitset, 20000 },
    { "array 4000 x bitset 9000",          shape::array,  4000,   shape::bitset, 9000 },
    { "run few x run few",                 shape::runs_few, 0,    shape::runs_few, 0 },
    { "run many x run many shifted",       shape::runs_many, 0,   shape::runs_many_shifted, 0 },
    { "array 20 x run few",                shape::array,  20,     shape::runs_few, 0 },
    { "array 2000 x run few",              shape::array,  2000,   shape::runs_few, 0 },
    { "array 2000 x run many",             shape::array,  2000,   shape::runs_many, 0 },
    { "run few x bitset 20000",            shape::runs_few, 0,    shape::bitset, 20000 },
    { "run many x bitset 20000",           shape::runs_many, 0,   shape::bitset, 20000 },
    { "run many x array 100",              shape::runs_many, 0,   shape::array,  100 },
};

void register_all() {
    for ( auto const & f : pair_forms ) {
        bands32( std::string{ "32-bit 1 chunk: " } + f.name, make_operands( f.a, f.na, f.b, f.nb, 1, 1, 5, 11 ) );
        bands32( std::string{ "32-bit 256 chunks: " } + f.name, make_operands( f.a, f.na, f.b, f.nb, 256, 1, 5, 12 ) );
    }
    // tables of many small chunks: the chunk walk dominates
    bands32( "32-bit 4000 chunks of 30: array x array", make_operands( shape::array, 30, shape::array, 30, 4000, 1, 0, 21 ) );
    bands64( "64-bit 4000 chunks of 30: array x array", make_operands( shape::array, 30, shape::array, 30, 4000, 1, 0, 22 ) );
    bands64( "64-bit 4000 sparse chunks of 30: array x array", make_operands( shape::array, 30, shape::array, 30, 4000, 977, 1ULL << 30, 23 ) );
    bands64( "64-bit 256 chunks: array 3000 x bitset 20000", make_operands( shape::array, 3000, shape::bitset, 20000, 256, 1, 1ULL << 33, 24 ) );
    bands64( "64-bit 256 chunks: bitset x bitset", make_operands( shape::bitset, 30000, shape::bitset, 30000, 256, 1, 1ULL << 33, 25 ) );
    bands64( "64-bit 256 chunks: run few x run few", make_operands( shape::runs_few, 0, shape::runs_few, 0, 256, 1, 1ULL << 33, 26 ) );
    // tables that only partly overlap, and a few chunks against many
    {
        auto p{ make_operands( shape::array, 100, shape::array, 100, 2000, 1, 0, 31 ) };
        std::vector<std::uint64_t> shifted;
        for ( auto const v : p.b ) { shifted.push_back( v + ( std::uint64_t{ 700 } << 16 ) ); }
        p.b = std::move( shifted );
        bands32( "32-bit 2000 chunks, tables overlap by 1300 chunks", p );
        std::vector<std::uint64_t> head( p.b.begin(), p.b.begin() + 4 * 100 );
        p.b = head;
        bands32( "32-bit 2000 chunks x 4 chunks", p );
    }
    // N-way fold
    for ( auto const chunks : { std::size_t{ 8 }, std::size_t{ 300 } } ) {
        constexpr std::size_t operands{ 16 };
        auto frsr{ std::make_shared<std::vector<bitmap32>>() };
        auto croaring{ std::make_shared<std::vector<roaring_bitmap_t *>>() };
        for ( std::size_t i{ 0 }; i < operands; ++i ) {
            auto const p{ make_operands( i % 3 == 0 ? shape::bitset : shape::array, i % 3 == 0 ? 12000 : 700, shape::array, 1, chunks, 1, 3, 40 + i ) };
            std::vector<std::uint32_t> v( p.a.begin(), p.a.end() );
            frsr->emplace_back();
            frsr->back().add_many( v );
            croaring->push_back( roaring_bitmap_of_ptr( v.size(), v.data() ) );
        }
        band b;
        b.name = "xor many (16 bitmaps) " + std::to_string( chunks ) + " chunks";
        b.arms.push_back( { "frsr", [ frsr ] {
            std::vector<bitmap32 const *> ptr;
            for ( auto const & x : *frsr ) { ptr.push_back( &x ); }
            return std::uint64_t{ bitmap32::xor_many( ptr ).size() };
        } } );
        b.arms.push_back( { "croaring", [ croaring ] {
            std::vector<roaring_bitmap_t const *> ptr( croaring->begin(), croaring->end() );
            auto * const r{ roaring_bitmap_xor_many( ptr.size(), ptr.data() ) };
            auto const n{ roaring_bitmap_get_cardinality( r ) };
            roaring_bitmap_free( r );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
    // against plain operands
    {
        auto const p{ make_operands( shape::array, 3000, shape::array, 1, 256, 1, 8, 51 ) };
        auto const s{ build32( p ) };
        for ( auto const [first, last] : { std::pair<std::uint64_t, std::uint64_t>{ 1000, 200'000 }, std::pair<std::uint64_t, std::uint64_t>{ 8U << 16, 264U << 16 }, std::pair<std::uint64_t, std::uint64_t>{ 100, 4000 } } ) {
            band b;
            b.name = "xor_with interval [" + std::to_string( first ) + ", " + std::to_string( last ) + "] (flip) over 256 chunks";
            b.mutates = true;
            b.copy_frsr = [ s ] { bitmap32 c{ s->fa }; return std::uint64_t{ c.size() }; };
            b.copy_croaring = [ s ] { auto * const c{ roaring_bitmap_copy( s->ca ) }; auto const n{ roaring_bitmap_get_cardinality( c ) }; roaring_bitmap_free( c ); return n; };
            b.arms.push_back( { "frsr", [ s, first, last ] { bitmap32 c{ s->fa }; c.xor_with( key_range::closed( first, last ) ); return std::uint64_t{ c.size() }; } } );
            b.arms.push_back( { "croaring", [ s, first, last ] {
                auto * const c{ roaring_bitmap_copy( s->ca ) };
                roaring_bitmap_flip_inplace( c, first, last + 1 );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
            } } );
            g_bands.push_back( std::move( b ) );
        }
        auto keys{ std::make_shared<std::vector<std::uint32_t>>( p.b.begin(), p.b.end() ) };
        std::mt19937_64 rng{ 52 };
        for ( int i{ 0 }; i < 20000; ++i ) { keys->push_back( static_cast<std::uint32_t>( ( 8U << 16 ) + rng() % ( 256U << 16 ) ) ); }
        std::ranges::sort( *keys );
        keys->erase( std::unique( keys->begin(), keys->end() ), keys->end() );
        band b;
        b.name = "xor_with sorted keys (20000) over 256 chunks";
        b.mutates = true;
        b.copy_frsr = [ s ] { bitmap32 c{ s->fa }; return std::uint64_t{ c.size() }; };
        b.copy_croaring = [ s ] { auto * const c{ roaring_bitmap_copy( s->ca ) }; auto const n{ roaring_bitmap_get_cardinality( c ) }; roaring_bitmap_free( c ); return n; };
        b.arms.push_back( { "frsr", [ s, keys ] { bitmap32 c{ s->fa }; c.xor_with( std::span<std::uint32_t const>{ *keys } ); return std::uint64_t{ c.size() }; } } );
        b.arms.push_back( { "croaring", [ s, keys ] {
            auto * const c{ roaring_bitmap_copy( s->ca ) };
            auto * const other{ roaring_bitmap_of_ptr( keys->size(), keys->data() ) };
            roaring_bitmap_xor_inplace( c, other );
            auto const n{ roaring_bitmap_get_cardinality( c ) };
            roaring_bitmap_free( other );
            roaring_bitmap_free( c );
            return n;
        } } );
        g_bands.push_back( std::move( b ) );
    }
}

} // namespace

int main( int argc, char ** argv ) {
    options opts;
    for ( int i{ 1 }; i < argc; ++i ) {
        std::string const arg{ argv[ i ] };
        if ( arg == "--filter" && i + 1 < argc ) { opts.filter = argv[ ++i ]; }
        else if ( arg == "--rounds" && i + 1 < argc ) { opts.rounds = std::atoi( argv[ ++i ] ); }
        else if ( arg == "--min-time-ms" && i + 1 < argc ) { opts.min_time_ms = std::atof( argv[ ++i ] ); }
        else { std::fprintf( stderr, "usage: %s [--filter text] [--rounds n] [--min-time-ms ms]\n", argv[ 0 ] ); return 2; }
    }
    register_all();
    for ( auto & b : g_bands ) {
        if ( !opts.filter.empty() && b.name.find( opts.filter ) == std::string::npos ) { continue; }
        run_band( b, opts );
    }
    return g_sink == 0xFFFFFFFFFFFFFFFFULL ? 1 : 0;
}
