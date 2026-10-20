// Bitmap against plain operands (key spans, bit-sets and bit words, key intervals): the interop operations
// against what a caller without them does, which is to build a temporary bitmap from the operand and combine
// two bitmaps, and against CRoaring doing the same through a temporary roaring_bitmap_t (and, for intervals,
// through its native range functions).
//
// Every band runs its arms interleaved, round after round, and reports the fastest of the rounds per arm
// (nanoseconds per operation). A ratio is "arm time / interop time": above 1, the interop operation is faster.
// An operation that changes its bitmap starts every iteration with a copy of the target, which every arm
// pays; a "copy" column times that copy alone and the "net" ratios subtract it.
//
//   frsr_roaring_interop_bench [--filter text] [--rounds n] [--min-time-ms ms]

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <span>
#include <random>
#include <string>
#include <vector>

#if FRSR_ROARING_HAS_CROARING
#   include <roaring/roaring.h>
#endif
#if __has_include( <boost/dynamic_bitset.hpp> )
#   include <boost/dynamic_bitset.hpp>
#   define FRSR_BENCH_HAS_DYNAMIC_BITSET 1
#endif

namespace {

using bitmap = frsr::roaring::bitmap<std::uint32_t>;
using frsr::roaring::bit_words;
using frsr::roaring::bits_at;
using frsr::roaring::key_range;

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
    std::vector<arm>  arms;               // arms[ 0 ] is the interop operation
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

#if FRSR_ROARING_HAS_CROARING
struct roaring_owner {
    explicit roaring_owner( roaring_bitmap_t * r ) : r{ r } {}
    ~roaring_owner() { roaring_bitmap_free( r ); }
    roaring_owner( roaring_owner const & ) = delete;
    roaring_bitmap_t * r;
};
#endif

void span_bands( target & t, std::size_t const count, bool const sorted ) {
    auto const keys{ std::make_shared<std::vector<std::uint32_t>>( make_keys( t, count, sorted, 7 + count ) ) };
    auto const label{ t.name + "/" + std::to_string( count ) + " keys " + ( sorted ? "sorted" : "unsorted" ) };
    auto * const tp{ &t };
    {
        band b;
        b.name = "and_with span: " + label;
        b.mutates = true;
        b.copy = [ tp ] { bitmap c{ tp->frsr }; return c.size(); };
        b.arms.push_back( { "interop", [ tp, keys ] { bitmap c{ tp->frsr }; c.and_with( std::span<std::uint32_t const>{ *keys } ); return c.size(); } } );
        b.arms.push_back( { "temp", [ tp, keys ] { bitmap c{ tp->frsr }; bitmap const other{ *keys }; c &= other; return c.size(); } } );
#if FRSR_ROARING_HAS_CROARING
        b.arms.push_back( { "croaring", [ tp, keys ] {
            auto * const c{ roaring_bitmap_copy( tp->croaring ) };
            roaring_owner const other{ roaring_bitmap_of_ptr( keys->size(), keys->data() ) };
            roaring_bitmap_and_inplace( c, other.r );
            auto const n{ roaring_bitmap_get_cardinality( c ) };
            roaring_bitmap_free( c );
            return n;
        } } );
#endif
        add( std::move( b ) );
    }
    {
        band b;
        b.name = "intersects span: " + label;
        b.arms.push_back( { "interop", [ tp, keys ] { return std::uint64_t{ tp->frsr.intersects( std::span<std::uint32_t const>{ *keys } ) }; } } );
        b.arms.push_back( { "temp", [ tp, keys ] { bitmap const other{ *keys }; return std::uint64_t{ tp->frsr.intersects( other ) }; } } );
#if FRSR_ROARING_HAS_CROARING
        b.arms.push_back( { "croaring", [ tp, keys ] {
            roaring_owner const other{ roaring_bitmap_of_ptr( keys->size(), keys->data() ) };
            return std::uint64_t{ roaring_bitmap_intersect( tp->croaring, other.r ) };
        } } );
#endif
        add( std::move( b ) );
    }
    {
        band b;
        b.name = "and_cardinality span: " + label;
        b.arms.push_back( { "interop", [ tp, keys ] { return std::uint64_t{ tp->frsr.and_cardinality( std::span<std::uint32_t const>{ *keys } ) }; } } );
        b.arms.push_back( { "temp", [ tp, keys ] { bitmap const other{ *keys }; return std::uint64_t{ tp->frsr.and_cardinality( other ) }; } } );
#if FRSR_ROARING_HAS_CROARING
        b.arms.push_back( { "croaring", [ tp, keys ] {
            roaring_owner const other{ roaring_bitmap_of_ptr( keys->size(), keys->data() ) };
            return roaring_bitmap_and_cardinality( tp->croaring, other.r );
        } } );
#endif
        add( std::move( b ) );
    }
    if ( sorted ) {
        band b;
        b.name = "or_with span: " + label;
        b.mutates = true;
        b.copy = [ tp ] { bitmap c{ tp->frsr }; return c.size(); };
        b.arms.push_back( { "interop", [ tp, keys ] { bitmap c{ tp->frsr }; c.or_with( frsr::roaring::sorted_keys{ *keys } ); return c.size(); } } );
        b.arms.push_back( { "temp", [ tp, keys ] { bitmap c{ tp->frsr }; bitmap const other{ *keys }; c |= other; return c.size(); } } );
#if FRSR_ROARING_HAS_CROARING
        b.arms.push_back( { "croaring", [ tp, keys ] {
            auto * const c{ roaring_bitmap_copy( tp->croaring ) };
            roaring_owner const other{ roaring_bitmap_of_ptr( keys->size(), keys->data() ) };
            roaring_bitmap_or_inplace( c, other.r );
            auto const n{ roaring_bitmap_get_cardinality( c ) };
            roaring_bitmap_free( c );
            return n;
        } } );
#endif
        add( std::move( b ) );
    }
}

// the keys of the bit string `words`, as the temporary-bitmap approach has to extract them
std::vector<std::uint32_t> keys_of_words( std::span<std::uint64_t const> words, std::uint32_t const base ) {
    std::vector<std::uint32_t> keys;
    for ( std::size_t w{ 0 }; w < words.size(); ++w ) {
        for ( auto bits{ words[ w ] }; bits != 0; bits &= bits - 1 ) {
            keys.push_back( base + static_cast<std::uint32_t>( w * 64 + static_cast<std::size_t>( std::countr_zero( bits ) ) ) );
        }
    }
    return keys;
}

void bitset_bands( target & t ) {
    auto * const tp{ &t };
    // a chunk-aligned bitset of one chunk, and a 16-chunk string of words not aligned to a chunk
    std::mt19937_64 rng{ 99 };
    auto const single{ std::make_shared<std::bitset<65536>>() };
    for ( std::size_t i{ 0 }; i < 65536; ++i ) { if ( rng() % 3 == 0 ) { single->set( i ); } }
    auto const words{ std::make_shared<std::vector<std::uint64_t>>( 16 * 1024 ) };
    for ( auto & w : *words ) { w = rng() & rng(); }
    constexpr std::uint32_t chunk_base{ 8U << 16 };
    constexpr std::uint32_t odd_base{ ( 8U << 16 ) + 1000 };
    auto const label{ t.name };

    auto const add_bitset_band{ [ & ]( std::string const & op, bool const mutates, auto const interop, auto const temp, auto const croaring_arm ) {
        band b;
        b.name = op + ": " + label;
        b.mutates = mutates;
        if ( mutates ) { b.copy = [ tp ] { bitmap c{ tp->frsr }; return c.size(); }; }
        b.arms.push_back( { "interop", [ tp, interop ] { return interop( tp ); } } );
        b.arms.push_back( { "temp", [ tp, temp ] { return temp( tp ); } } );
#if FRSR_ROARING_HAS_CROARING
        b.arms.push_back( { "croaring", [ tp, croaring_arm ] { return croaring_arm( tp ); } } );
#else
        (void)croaring_arm;
#endif
        add( std::move( b ) );
    } };

    // std::bitset<65536> at a chunk
    {
        auto const keys_of_single{ [ single ]( std::uint32_t base ) {
            std::vector<std::uint32_t> keys;
            for ( std::size_t i{ 0 }; i < 65536; ++i ) { if ( single->test( i ) ) { keys.push_back( base + static_cast<std::uint32_t>( i ) ); } }
            return keys;
        } };
        add_bitset_band( "and_with std::bitset<65536>", true,
            [ single ]( target * t ) { bitmap c{ t->frsr }; c.and_with( bits_at( *single, chunk_base ) ); return std::uint64_t{ c.size() }; },
            [ keys_of_single ]( target * t ) { bitmap c{ t->frsr }; bitmap const other{ keys_of_single( chunk_base ) }; c &= other; return std::uint64_t{ c.size() }; },
            [ keys_of_single ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto * const c{ roaring_bitmap_copy( t->croaring ) };
                auto const keys{ keys_of_single( chunk_base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                roaring_bitmap_and_inplace( c, other.r );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
        add_bitset_band( "intersects std::bitset<65536>", false,
            [ single ]( target * t ) { return std::uint64_t{ t->frsr.intersects( bits_at( *single, chunk_base ) ) }; },
            [ keys_of_single ]( target * t ) { bitmap const other{ keys_of_single( chunk_base ) }; return std::uint64_t{ t->frsr.intersects( other ) }; },
            [ keys_of_single ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto const keys{ keys_of_single( chunk_base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                return std::uint64_t{ roaring_bitmap_intersect( t->croaring, other.r ) };
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
        add_bitset_band( "or_with std::bitset<65536>", true,
            [ single ]( target * t ) { bitmap c{ t->frsr }; c.or_with( bits_at( *single, chunk_base ) ); return std::uint64_t{ c.size() }; },
            [ keys_of_single ]( target * t ) { bitmap c{ t->frsr }; bitmap const other{ keys_of_single( chunk_base ) }; c |= other; return std::uint64_t{ c.size() }; },
            [ keys_of_single ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto * const c{ roaring_bitmap_copy( t->croaring ) };
                auto const keys{ keys_of_single( chunk_base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                roaring_bitmap_or_inplace( c, other.r );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
    }
#ifdef FRSR_BENCH_HAS_DYNAMIC_BITSET
    // boost::dynamic_bitset<> (the default block type) holding the same bits as the std::bitset above
    {
        auto const dynamic{ std::make_shared<boost::dynamic_bitset<>>( 65536 ) };
        for ( std::size_t i{ 0 }; i < 65536; ++i ) { if ( single->test( i ) ) { dynamic->set( i ); } }
        auto const keys_of_dynamic{ [ dynamic ]( std::uint32_t const base ) {
            std::vector<std::uint32_t> keys;
            for ( auto i{ dynamic->find_first() }; i != boost::dynamic_bitset<>::npos; i = dynamic->find_next( i ) ) { keys.push_back( base + static_cast<std::uint32_t>( i ) ); }
            return keys;
        } };
        add_bitset_band( "and_with boost::dynamic_bitset<65536>", true,
            [ dynamic ]( target * t ) { bitmap c{ t->frsr }; c.and_with( bits_at( *dynamic, chunk_base ) ); return std::uint64_t{ c.size() }; },
            [ keys_of_dynamic ]( target * t ) { bitmap c{ t->frsr }; bitmap const other{ keys_of_dynamic( chunk_base ) }; c &= other; return std::uint64_t{ c.size() }; },
            [ keys_of_dynamic ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto * const c{ roaring_bitmap_copy( t->croaring ) };
                auto const keys{ keys_of_dynamic( chunk_base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                roaring_bitmap_and_inplace( c, other.r );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
        add_bitset_band( "intersects boost::dynamic_bitset<65536>", false,
            [ dynamic ]( target * t ) { return std::uint64_t{ t->frsr.intersects( bits_at( *dynamic, chunk_base ) ) }; },
            [ keys_of_dynamic ]( target * t ) { bitmap const other{ keys_of_dynamic( chunk_base ) }; return std::uint64_t{ t->frsr.intersects( other ) }; },
            [ keys_of_dynamic ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto const keys{ keys_of_dynamic( chunk_base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                return std::uint64_t{ roaring_bitmap_intersect( t->croaring, other.r ) };
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
    }
#endif
    // 16 chunks of words, at a chunk and off it
    for ( auto const base : { chunk_base, odd_base } ) {
        auto const tag{ std::string( base == chunk_base ? " aligned" : " off a chunk" ) };
        add_bitset_band( "and_with 1M-bit words" + tag, true,
            [ words, base ]( target * t ) { bitmap c{ t->frsr }; c.and_with( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ *words }, base } ); return std::uint64_t{ c.size() }; },
            [ words, base ]( target * t ) { bitmap c{ t->frsr }; bitmap const other{ keys_of_words( *words, base ) }; c &= other; return std::uint64_t{ c.size() }; },
            [ words, base ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto * const c{ roaring_bitmap_copy( t->croaring ) };
                auto const keys{ keys_of_words( *words, base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                roaring_bitmap_and_inplace( c, other.r );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
        add_bitset_band( "and_cardinality 1M-bit words" + tag, false,
            [ words, base ]( target * t ) { return std::uint64_t{ t->frsr.and_cardinality( bit_words<std::uint64_t>{ std::span<std::uint64_t const>{ *words }, base } ) }; },
            [ words, base ]( target * t ) { bitmap const other{ keys_of_words( *words, base ) }; return std::uint64_t{ t->frsr.and_cardinality( other ) }; },
            [ words, base ]( target * t ) {
#if FRSR_ROARING_HAS_CROARING
                auto const keys{ keys_of_words( *words, base ) };
                roaring_owner const other{ roaring_bitmap_of_ptr( keys.size(), keys.data() ) };
                return roaring_bitmap_and_cardinality( t->croaring, other.r );
#else
                (void)t; return std::uint64_t{ 0 };
#endif
            } );
    }
}

void range_bands( target & t ) {
    auto * const tp{ &t };
    auto const first_chunk{ 8U };
    auto const last_chunk{ static_cast<std::uint32_t>( t.keys.back() >> 16 ) };
    struct width { char const * name; std::uint32_t lo; std::uint32_t hi; };   // closed interval
    // a range that lies in one chunk, one that spans a few, one that spans most of the target, and one that misses it
    std::array<width, 4> const widths{ {
        { "narrow (1000 keys)", ( first_chunk << 16 ) + 5000, ( first_chunk << 16 ) + 5999 },
        { "few chunks", ( first_chunk << 16 ) + 30000, ( ( first_chunk + 3 ) << 16 ) + 20000 },
        { "wide", ( first_chunk << 16 ) + 1, ( last_chunk << 16 ) - 1 },
        { "outside", 100, ( first_chunk << 16 ) - 100 },
    } };
    for ( auto const & w : widths ) {
        auto const label{ t.name + "/" + w.name };
        auto const lo{ w.lo };
        auto const hi{ w.hi };
        {
            band b;
            b.name = "intersects_range: " + label;
            b.arms.push_back( { "interop", [ tp, lo, hi ] { return std::uint64_t{ tp->frsr.intersects( key_range::closed( lo, hi ) ) }; } } );
            b.arms.push_back( { "temp", [ tp, lo, hi ] { bitmap range; range.add_closed_range( lo, hi ); return std::uint64_t{ tp->frsr.intersects( range ) }; } } );
#if FRSR_ROARING_HAS_CROARING
            b.arms.push_back( { "croaring", [ tp, lo, hi ] { return std::uint64_t{ roaring_bitmap_intersect_with_range( tp->croaring, lo, static_cast<std::uint64_t>( hi ) + 1 ) }; } } );
            b.arms.push_back( { "croa-tmp", [ tp, lo, hi ] {
                roaring_owner const range{ roaring_bitmap_from_range( lo, static_cast<std::uint64_t>( hi ) + 1, 1 ) };
                return std::uint64_t{ roaring_bitmap_intersect( tp->croaring, range.r ) };
            } } );
#endif
            add( std::move( b ) );
        }
        {
            band b;
            b.name = "cardinality_in_range: " + label;
            b.arms.push_back( { "interop", [ tp, lo, hi ] { return std::uint64_t{ tp->frsr.and_cardinality( key_range::closed( lo, hi ) ) }; } } );
            b.arms.push_back( { "temp", [ tp, lo, hi ] { bitmap range; range.add_closed_range( lo, hi ); return std::uint64_t{ tp->frsr.and_cardinality( range ) }; } } );
#if FRSR_ROARING_HAS_CROARING
            b.arms.push_back( { "croaring", [ tp, lo, hi ] { return roaring_bitmap_range_cardinality( tp->croaring, lo, static_cast<std::uint64_t>( hi ) + 1 ); } } );
#endif
            add( std::move( b ) );
        }
        {
            band b;
            b.name = "contains_range: " + label;
            b.arms.push_back( { "interop", [ tp, lo, hi ] { return std::uint64_t{ tp->frsr.is_superset_of( key_range::closed( lo, hi ) ) }; } } );
            b.arms.push_back( { "temp", [ tp, lo, hi ] { bitmap range; range.add_closed_range( lo, hi ); return std::uint64_t{ range.is_subset_of( tp->frsr ) }; } } );
#if FRSR_ROARING_HAS_CROARING
            b.arms.push_back( { "croaring", [ tp, lo, hi ] { return std::uint64_t{ roaring_bitmap_contains_range( tp->croaring, lo, static_cast<std::uint64_t>( hi ) + 1 ) }; } } );
#endif
            add( std::move( b ) );
        }
        {
            band b;
            b.name = "and_with range: " + label;
            b.mutates = true;
            b.copy = [ tp ] { bitmap c{ tp->frsr }; return c.size(); };
            b.arms.push_back( { "interop", [ tp, lo, hi ] { bitmap c{ tp->frsr }; c.and_with( key_range::closed( lo, hi ) ); return std::uint64_t{ c.size() }; } } );
            b.arms.push_back( { "temp", [ tp, lo, hi ] { bitmap c{ tp->frsr }; bitmap range; range.add_closed_range( lo, hi ); c &= range; return std::uint64_t{ c.size() }; } } );
#if FRSR_ROARING_HAS_CROARING
            b.arms.push_back( { "croa-tmp", [ tp, lo, hi ] {
                auto * const c{ roaring_bitmap_copy( tp->croaring ) };
                roaring_owner const range{ roaring_bitmap_from_range( lo, static_cast<std::uint64_t>( hi ) + 1, 1 ) };
                roaring_bitmap_and_inplace( c, range.r );
                auto const n{ roaring_bitmap_get_cardinality( c ) };
                roaring_bitmap_free( c );
                return n;
            } } );
#endif
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
    // targets: dense bitset containers, and small array containers
    static target dense{ make_target( "dense", 64, 20000, 1 ) };
    static target sparse{ make_target( "sparse", 256, 600, 2 ) };
    for ( auto * t : { &dense, &sparse } ) {
        for ( std::size_t const count : { std::size_t{ 16 }, std::size_t{ 1000 }, std::size_t{ 64000 } } ) {
            span_bands( *t, count, true );
            span_bands( *t, count, false );
        }
        bitset_bands( *t );
        range_bands( *t );
    }
    for ( auto & b : g_bands ) {
        if ( !opts.filter.empty() && b.name.find( opts.filter ) == std::string::npos ) { continue; }
        run_band( b, opts );
    }
    std::fprintf( stderr, "# sink %llu\n", static_cast<unsigned long long>( g_sink ) );
    return 0;
}
