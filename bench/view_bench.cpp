// A view of a finished bitmap, and the complement of an operand, against the bitmap itself as the operand of the same operations:
// the operations of interop_bench against what a reader of a published bitmap has without views. Every band runs its arms
// interleaved, round after round, and reports the fastest of the rounds per arm (nanoseconds per operation).
//
// The "view" bands put view_of( b ) against b for every operation (and CRoaring for the value-semantics policy), for both counting
// policies; an operation that changes its bitmap starts every iteration with a copy of the target, which every arm pays: a "copy"
// column times that copy alone. The "complement" bands put `!view` against the operation it is rewritten to: any difference
// between the two arms means the rewrite did not happen. The arms of every band are checked to compute the same thing before
// anything is timed.
//
//   frsr_roaring_view_bench [--filter text] [--rounds n] [--min-time-ms ms]

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if FRSR_ROARING_HAS_CROARING
#   include <roaring/roaring.h>
#endif

namespace {

using bitmap = frsr::roaring::bitmap<std::uint32_t>;

using clock_type = std::chrono::steady_clock;

struct arm {
    std::string                  name;
    std::function<std::uint64_t()> run;     // one operation; returns a value so that it is not optimized away
    double                       best_ns{ 1e300 };
};

struct band {
    std::string       name;
    bool              mutates{ false };   // the arms start from a copy of the target
    std::function<std::uint64_t()> copy;  // the copy alone, when the band mutates
    std::vector<arm>  arms;               // arms[ 0 ] is the view (or the rewritten form)
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
    std::vector<arm *> all;
    arm copy_arm{ "copy", b.copy, 1e300 };
    for ( auto & a : b.arms ) { all.push_back( &a ); }
    if ( b.mutates ) { all.push_back( &copy_arm ); }
    // each arm repeats its operation so that one sample lasts at least min_time_ms; the repetitions do not matter
    // to the result, which is per operation
    std::vector<std::int64_t> reps( all.size(), 1 );
    for ( std::size_t k{ 0 }; k < all.size(); ++k ) {
        all[ k ]->run();
        auto const once{ time_arm( *all[ k ], 1 ) };
        reps[ k ] = std::clamp<std::int64_t>( static_cast<std::int64_t>( opts.min_time_ms * 1e6 / std::max( once, 1.0 ) ), 1, 200000 );
    }
    for ( int round{ 0 }; round < opts.rounds; ++round ) {
        for ( std::size_t j{ 0 }; j < all.size(); ++j ) {
            auto const k{ ( j + static_cast<std::size_t>( round ) ) % all.size() };   // rotate the order
            all[ k ]->best_ns = std::min( all[ k ]->best_ns, time_arm( *all[ k ], reps[ k ] ) );
        }
    }
    auto const copy_ns{ b.mutates ? copy_arm.best_ns : 0.0 };
    std::printf( "%-46s", b.name.c_str() );
    std::printf( " %-8s %10.0f", b.arms[ 0 ].name.c_str(), b.arms[ 0 ].best_ns );
    for ( std::size_t i{ 1 }; i < b.arms.size(); ++i ) {
        auto const & a{ b.arms[ i ] };
        auto const ratio{ a.best_ns / b.arms[ 0 ].best_ns };
        if ( b.mutates && b.arms[ 0 ].best_ns - copy_ns > 1.0 ) {
            std::printf( " | %-8s %10.0f x%-6.2f net x%-6.2f", a.name.c_str(), a.best_ns, ratio, ( a.best_ns - copy_ns ) / ( b.arms[ 0 ].best_ns - copy_ns ) );
        } else {
            std::printf( " | %-8s %10.0f x%-6.2f", a.name.c_str(), a.best_ns, ratio );
        }
    }
    if ( b.mutates ) { std::printf( " | copy %.0f", copy_ns ); }
    std::printf( "\n" );
    std::fflush( stdout );
}

// ---- data ---------------------------------------------------------------------------------------------------

struct target {
    std::string                 name;
    std::vector<std::uint32_t>  keys;
    bitmap                      frsr;
#if FRSR_ROARING_HAS_CROARING
    roaring_bitmap_t *          croaring{};
#endif
};

// `chunks` chunks starting at chunk 8, each holding `per_chunk` random keys
target make_target( std::string name, std::size_t const chunks, std::size_t const per_chunk, std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    target t;
    t.name = std::move( name );
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        std::vector<std::uint32_t> lows;
        while ( lows.size() < per_chunk ) { lows.push_back( static_cast<std::uint32_t>( rng() & 0xFFFF ) ); }
        std::ranges::sort( lows );
        lows.erase( std::unique( lows.begin(), lows.end() ), lows.end() );
        for ( auto const low : lows ) { t.keys.push_back( ( static_cast<std::uint32_t>( 8 + c ) << 16 ) + low ); }
    }
    t.frsr = bitmap{ t.keys };
#if FRSR_ROARING_HAS_CROARING
    t.croaring = roaring_bitmap_of_ptr( t.keys.size(), t.keys.data() );
#endif
    return t;
}

// `count` keys, about half of them in the target
std::vector<std::uint32_t> make_keys( target const & t, std::size_t const count, bool const sorted, std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    std::vector<std::uint32_t> keys;
    auto const first_chunk{ 8U };
    auto const chunk_span{ static_cast<std::uint32_t>( t.keys.back() >> 16 ) - first_chunk + 1 };
    while ( keys.size() < count ) {
        if ( rng() & 1 ) { keys.push_back( t.keys[ rng() % t.keys.size() ] ); }
        else             { keys.push_back( ( ( first_chunk + static_cast<std::uint32_t>( rng() % chunk_span ) ) << 16 ) + static_cast<std::uint32_t>( rng() & 0xFFFF ) ); }
    }
    if ( sorted ) { std::ranges::sort( keys ); }
    return keys;
}

// ---- bands --------------------------------------------------------------------------------------------------

std::vector<band> g_bands;

void add( band b ) { g_bands.push_back( std::move( b ) ); }

// ---- views and complements ----------------------------------------------------------------------------------------

using cow_bitmap = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>, frsr::roaring::detail::cow_atomic_refcount>;
using frsr::roaring::view_of;

#if FRSR_ROARING_HAS_CROARING
#   define FRSR_BENCH_CROARING_INPLACE( FN ) []( roaring_bitmap_t * r, roaring_bitmap_t const * o ) { FN( r, o ); }
#else
#   define FRSR_BENCH_CROARING_INPLACE( FN ) 0
#endif

#if FRSR_ROARING_HAS_CROARING
#   define FRSR_BENCH_CROARING_QUERY( FN ) []( roaring_bitmap_t const * r, roaring_bitmap_t const * o ) { return FN( r, o ); }
#   define FRSR_BENCH_CROARING_JACCARD() []( roaring_bitmap_t const * r, roaring_bitmap_t const * o ) { return static_cast<std::uint64_t>( roaring_bitmap_jaccard_index( r, o ) * 1e6 ); }
#else
#   define FRSR_BENCH_CROARING_QUERY( FN ) 0
#   define FRSR_BENCH_CROARING_JACCARD() 0
#endif

enum class fill { dense, sparse, runs };

// `chunks` chunks from chunk `first_chunk`, each holding `per_chunk` random keys (dense and sparse) or a few runs of that many keys
template <typename Bitmap>
Bitmap make_filled( fill const kind, std::size_t const chunks, std::size_t const per_chunk, std::uint32_t const first_chunk, std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    Bitmap result;
    for ( std::size_t c{ 0 }; c < chunks; ++c ) {
        auto const base{ static_cast<std::uint32_t>( ( first_chunk + c ) << 16 ) };
        if ( kind == fill::runs ) {
            for ( std::size_t r{ 0 }; r < 6; ++r ) {
                auto const begin{ static_cast<std::uint32_t>( rng() % 50000 ) };
                result.add_closed_range( base + begin, base + begin + static_cast<std::uint32_t>( per_chunk / 6 ) );
            }
        } else {
            std::vector<std::uint32_t> lows;
            while ( lows.size() < per_chunk ) { lows.push_back( static_cast<std::uint32_t>( rng() & 0xFFFF ) ); }
            std::ranges::sort( lows );
            lows.erase( std::unique( lows.begin(), lows.end() ), lows.end() );
            for ( auto const low : lows ) { result.add( base + low ); }
        }
    }
    if ( kind == fill::runs ) { result.optimize(); }
    return result;
}

template <typename Bitmap>
struct view_case {
    std::string name;
    Bitmap      receiver;
    Bitmap      operand;
#if FRSR_ROARING_HAS_CROARING
    roaring_bitmap_t * croaring_receiver{};
    roaring_bitmap_t * croaring_operand{};
#endif
};

template <typename Bitmap>
std::shared_ptr<view_case<Bitmap>> make_view_case( std::string name, Bitmap receiver, Bitmap operand ) {
    auto result{ std::make_shared<view_case<Bitmap>>() };
    result->name     = std::move( name );
    result->receiver = std::move( receiver );
    result->operand  = std::move( operand );
#if FRSR_ROARING_HAS_CROARING
    auto const keys_of{ []( Bitmap const & bitmap ) {
        auto const keys{ bitmap.to_vector() };
        return roaring_bitmap_of_ptr( keys.size(), keys.data() );
    } };
    result->croaring_receiver = keys_of( result->receiver );
    result->croaring_operand  = keys_of( result->operand );
#endif
    return result;
}

template <typename Bitmap>
void view_bands( std::string const & policy ) {
    std::vector<std::shared_ptr<view_case<Bitmap>>> cases;
    cases.push_back( make_view_case( "dense/dense", make_filled<Bitmap>( fill::dense, 64, 20000, 8, 1 ), make_filled<Bitmap>( fill::dense, 64, 20000, 8, 2 ) ) );
    cases.push_back( make_view_case( "sparse/sparse", make_filled<Bitmap>( fill::sparse, 256, 600, 8, 3 ), make_filled<Bitmap>( fill::sparse, 256, 600, 8, 4 ) ) );
    cases.push_back( make_view_case( "dense/runs", make_filled<Bitmap>( fill::dense, 64, 20000, 8, 5 ), make_filled<Bitmap>( fill::runs, 64, 12000, 8, 6 ) ) );
    cases.push_back( make_view_case( "sparse/half-overlap", make_filled<Bitmap>( fill::sparse, 256, 600, 8, 7 ), make_filled<Bitmap>( fill::sparse, 256, 600, 136, 8 ) ) );
    cases.push_back( make_view_case( "sparse/disjoint", make_filled<Bitmap>( fill::sparse, 256, 600, 8, 9 ), make_filled<Bitmap>( fill::sparse, 256, 600, 400, 10 ) ) );

    for ( auto const & c : cases ) {
        auto const label{ "[" + policy + "] " + c->name };
        auto const cp{ c };   // the shared case: the bands outlive this function
        auto const mutating{ [ & ]( std::string const & op, auto const view_op, auto const bitmap_op, auto const complement_op ) {
            band b;
            b.name = "view " + op + " " + label;
            b.mutates = true;
            b.copy = [ cp ] { Bitmap r{ cp->receiver }; return std::uint64_t{ r.size() }; };
            b.arms.push_back( { "view", [ cp, view_op ] { Bitmap r{ cp->receiver }; view_op( r, view_of( cp->operand ) ); return std::uint64_t{ r.size() }; } } );
            b.arms.push_back( { "bitmap", [ cp, bitmap_op ] { Bitmap r{ cp->receiver }; bitmap_op( r, cp->operand ); return std::uint64_t{ r.size() }; } } );
#if FRSR_ROARING_HAS_CROARING
            if constexpr ( std::is_same_v<Bitmap, bitmap> ) {
                b.arms.push_back( { "croaring", [ cp, complement_op ] {
                    auto * const r{ roaring_bitmap_copy( cp->croaring_receiver ) };
                    complement_op( r, cp->croaring_operand );
                    auto const n{ roaring_bitmap_get_cardinality( r ) };
                    roaring_bitmap_free( r );
                    return n;
                } } );
            }
#else
            (void)complement_op;
#endif
            add( std::move( b ) );
        } };
        mutating( "and_with", []( Bitmap & r, auto const v ) { r &= v; }, []( Bitmap & r, Bitmap const & o ) { r &= o; },
            FRSR_BENCH_CROARING_INPLACE( roaring_bitmap_and_inplace ) );
        mutating( "or_with", []( Bitmap & r, auto const v ) { r |= v; }, []( Bitmap & r, Bitmap const & o ) { r |= o; },
            FRSR_BENCH_CROARING_INPLACE( roaring_bitmap_or_inplace ) );
        mutating( "andnot_with", []( Bitmap & r, auto const v ) { r -= v; }, []( Bitmap & r, Bitmap const & o ) { r -= o; },
            FRSR_BENCH_CROARING_INPLACE( roaring_bitmap_andnot_inplace ) );
        mutating( "xor_with", []( Bitmap & r, auto const v ) { r ^= v; }, []( Bitmap & r, Bitmap const & o ) { r ^= o; },
            FRSR_BENCH_CROARING_INPLACE( roaring_bitmap_xor_inplace ) );
        auto const query{ [ & ]( std::string const & op, auto const view_op, auto const bitmap_op ) {
            band b;
            b.name = "view " + op + " " + label;
            b.arms.push_back( { "view", [ cp, view_op ] { return std::uint64_t{ view_op( cp->receiver, view_of( cp->operand ) ) }; } } );
            b.arms.push_back( { "bitmap", [ cp, bitmap_op ] { return std::uint64_t{ bitmap_op( cp->receiver, cp->operand ) }; } } );
            add( std::move( b ) );
        } };
        query( "and_cardinality", []( Bitmap const & r, auto const v ) { return r.and_cardinality( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.and_cardinality( o ); } );
        query( "intersects", []( Bitmap const & r, auto const v ) { return r.intersects( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.intersects( o ); } );
        query( "is_subset_of", []( Bitmap const & r, auto const v ) { return r.is_subset_of( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.is_subset_of( o ); } );
        // the sizes of combinations, against the reference's own
        {
            auto const sizes{ [ & ]( std::string const & op, auto const view_op, auto const bitmap_op, auto const croaring_op ) {
                band b;
                b.name = "view " + op + " " + label;
                b.arms.push_back( { "view", [ cp, view_op ] { return std::uint64_t( view_op( cp->receiver, view_of( cp->operand ) ) ); } } );
                b.arms.push_back( { "bitmap", [ cp, bitmap_op ] { return std::uint64_t( bitmap_op( cp->receiver, cp->operand ) ); } } );
#if FRSR_ROARING_HAS_CROARING
                if constexpr ( std::is_same_v<Bitmap, bitmap> ) {
                    b.arms.push_back( { "croaring", [ cp, croaring_op ] { return std::uint64_t( croaring_op( cp->croaring_receiver, cp->croaring_operand ) ); } } );
                }
#else
                (void)croaring_op;
#endif
                add( std::move( b ) );
            } };
            sizes( "or_cardinality", []( Bitmap const & r, auto const v ) { return r.or_cardinality( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.or_cardinality( o ); },
                FRSR_BENCH_CROARING_QUERY( roaring_bitmap_or_cardinality ) );
            sizes( "andnot_cardinality", []( Bitmap const & r, auto const v ) { return r.andnot_cardinality( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.andnot_cardinality( o ); },
                FRSR_BENCH_CROARING_QUERY( roaring_bitmap_andnot_cardinality ) );
            sizes( "jaccard_index", []( Bitmap const & r, auto const v ) { return static_cast<std::uint64_t>( r.jaccard_index( v ) * 1e6 ); }, []( Bitmap const & r, Bitmap const & o ) { return static_cast<std::uint64_t>( r.jaccard_index( o ) * 1e6 ); },
                FRSR_BENCH_CROARING_JACCARD() );
            sizes( "is_strict_subset_of", []( Bitmap const & r, auto const v ) { return r.is_strict_subset_of( v ); }, []( Bitmap const & r, Bitmap const & o ) { return r.is_strict_subset_of( o ); },
                FRSR_BENCH_CROARING_QUERY( roaring_bitmap_is_strict_subset ) );
        }
        // the binary forms, which build a result: a view on the right against the bitmap on the right
        {
            band b;
            b.name = "view operator& " + label;
            b.arms.push_back( { "view", [ cp ] { return std::uint64_t{ ( cp->receiver & view_of( cp->operand ) ).size() }; } } );
            b.arms.push_back( { "bitmap", [ cp ] { return std::uint64_t{ ( cp->receiver & cp->operand ).size() }; } } );
            add( std::move( b ) );
        }
        // the complement bands: `!view` against what it is rewritten to
        {
            band b;
            b.name = "complement &= !view vs -= view " + label;
            b.mutates = true;
            b.copy = [ cp ] { Bitmap r{ cp->receiver }; return std::uint64_t{ r.size() }; };
            b.arms.push_back( { "!view", [ cp ] { Bitmap r{ cp->receiver }; r &= !view_of( cp->operand ); return std::uint64_t{ r.size() }; } } );
            b.arms.push_back( { "rewrite", [ cp ] { Bitmap r{ cp->receiver }; r -= view_of( cp->operand ); return std::uint64_t{ r.size() }; } } );
            add( std::move( b ) );
        }
        {
            band b;
            b.name = "complement and_cardinality( !view ) vs size - and_cardinality " + label;
            b.arms.push_back( { "!view", [ cp ] { return std::uint64_t{ cp->receiver.and_cardinality( !view_of( cp->operand ) ) }; } } );
            b.arms.push_back( { "rewrite", [ cp ] { return std::uint64_t{ cp->receiver.size() - cp->receiver.and_cardinality( view_of( cp->operand ) ) }; } } );
            add( std::move( b ) );
        }
        {
            band b;
            b.name = "complement intersects( !view ) vs !is_subset_of " + label;
            b.arms.push_back( { "!view", [ cp ] { return std::uint64_t{ cp->receiver.intersects( !view_of( cp->operand ) ) }; } } );
            b.arms.push_back( { "rewrite", [ cp ] { return std::uint64_t{ !cp->receiver.is_subset_of( view_of( cp->operand ) ) }; } } );
            add( std::move( b ) );
        }
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
        else { std::fprintf( stderr, "unknown argument: %s\n", arg.c_str() ); return 2; }
    }
    view_bands<bitmap>( "value" );
    view_bands<cow_bitmap>( "atomic" );
    // the arms of a band compute the same thing: refuse to time them otherwise
    for ( auto & b : g_bands ) {
        auto const expected{ b.arms.front().run() };
        for ( auto & a : b.arms ) {
            if ( a.run() != expected ) {
                std::fprintf( stderr, "arms of \"%s\" disagree: %s gives %llu, %s gives %llu\n", b.name.c_str(),
                    b.arms.front().name.c_str(), static_cast<unsigned long long>( expected ), a.name.c_str(), static_cast<unsigned long long>( a.run() ) );
                return 3;
            }
        }
    }
    for ( auto & b : g_bands ) {
        if ( !opts.filter.empty() && b.name.find( opts.filter ) == std::string::npos ) { continue; }
        run_band( b, opts );
    }
    std::fprintf( stderr, "# sink %llu\n", static_cast<unsigned long long>( g_sink ) );
    return 0;
}
