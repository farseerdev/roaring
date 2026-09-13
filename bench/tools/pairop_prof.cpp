// Profiling driver for the binary set_ops bands (Union / Difference /
// DifferenceInplace on the [0,count) vs [offset,offset+count) pair fixture),
// ToArray, and the small mixed-form pairs (MixedArrayBitsetAndnot, RunVsBitset,
// SaturatedBitsetAndnot): the fixture is built once and the operation loops long
// enough for a sampling profiler to attribute the operation itself. The
// benchmark's bands run a fixed number of inner reps, so their profiles carry
// too few samples.
//
// Not part of the build (nothing globs this directory). Compile with the
// benchmark target's own flags (see bulkadd_prof.cpp), then:
//
//   pairop_prof <frsr|cpp> <union|difference|diffinplace|toarray|mixedandnot|runbitset|satandnot|envelope|coldcard|coldcardnot|lazyfold|subuniq|unioninplace> <count> <high|mid|low> [passes]
//
// mixedandnot ignores count/overlap (the band's fixture: 64 strided values \ a
// 32768-value even-number bitset — an all-hit probe, empty result); runbitset
// takes the run operand's cardinality as count (64 runs at stride 1024 ∩ the
// bitset [0,32768)); satandnot takes the bitset's fill count as count (64
// values at stride 2048 \ [0,count)).
#include <frsr/roaring/bitmap.hpp>
#include <roaring/roaring.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
using Bitmap = frsr::roaring::bitmap<
    std::uint32_t,
    frsr::roaring::default_container_set<std::uint32_t>,
    frsr::roaring::detail::cow_atomic_refcount,
    frsr::roaring::detail::run_selection_lazy
>;

// The operands as value lists, so both arms build the identical fixture.
struct Fixture { std::vector<std::uint32_t> a, b; std::vector<std::pair<std::uint32_t, std::uint32_t>> a_runs; bool optimize_a{ false }; };
static Fixture make_fixture( std::string const & op, std::size_t const count, std::size_t const offset ) {
    Fixture f;
    if ( op == "toarray" ) {
        for ( std::size_t c = 0; c < 16; ++c ) for ( std::size_t i = 0; i < 2000; ++i ) { f.a.push_back( std::uint32_t( c * 65536 + i * 16 ) ); }
    } else if ( op == "mixedandnot" ) {
        for ( std::size_t i = 0; i < 64; ++i ) { f.a.push_back( std::uint32_t( i * 1024 ) ); }
        for ( std::size_t i = 0; i < 32768; ++i ) { f.b.push_back( std::uint32_t( 2 * i ) ); }
    } else if ( op == "runbitset" ) {
        auto const run_length{ count / 64 };
        for ( std::size_t r = 0; r < 64; ++r ) { f.a_runs.emplace_back( std::uint32_t( r * 1024 ), std::uint32_t( r * 1024 + run_length - 1 ) ); }
        f.optimize_a = true;
        for ( std::size_t i = 0; i < 32768; ++i ) { f.b.push_back( std::uint32_t( i ) ); }
    } else if ( op == "envelope" ) {
        // EnvelopePerChunk/chunks=<count>: both sides hold count chunks of 4 values, one of which matches.
        for ( std::size_t c = 0; c < count; ++c ) { auto const base{ std::uint32_t( c * 65536 ) }; f.a.insert( f.a.end(), { base + 1, base + 3, base + 5, base + 7 } ); f.b.insert( f.b.end(), { base + 1, base + 2, base + 4, base + 6 } ); }
    } else if ( op == "satandnot" ) {
        for ( std::size_t i = 0; i < 64; ++i ) { f.a.push_back( std::uint32_t( i * 2048 ) ); }
        for ( std::size_t i = 0; i < count; ++i ) { f.b.push_back( std::uint32_t( i ) ); }
    } else {
        for ( std::size_t i = 0; i < count; ++i ) { f.a.push_back( std::uint32_t( i ) ); f.b.push_back( std::uint32_t( i + offset ) ); }
    }
    return f;
}

// cold_card/ColdCard{Intersect,Andnot}/acard=<count>: 102 pairs (a 1 MB nominal working set), each an array of
// count values at stride 16384/count against the bitset [0,8192), touched in one shuffled order.
template <typename Bitmap, typename Add>
static std::vector<std::pair<Bitmap, Bitmap>> cold_pairs( std::size_t const acard, Add add ) {
    std::vector<std::pair<Bitmap, Bitmap>> pairs( 102 );
    std::size_t const stride{ std::max<std::size_t>( 1, 16384 / std::max<std::size_t>( acard, 1 ) ) };
    for ( auto & [ arr, bmp ] : pairs ) {
        for ( std::size_t v = 0; v < 8192; ++v ) { add( bmp, std::uint32_t( v ) ); }
        for ( std::size_t v = 0; v < acard; ++v ) { add( arr, std::uint32_t( v * stride ) ); }
    }
    return pairs;
}
// PAIROP_PAD=<bytes>: one held allocation of that size between the construction of the two operands, which moves
// the second operand (and every later block) by a controlled offset relative to the first: sweeping it through a
// page exposes address-dependent stalls such as store-to-load aliasing on the low 12 address bits.
static void hold_padding() {
    if ( char const * const pad{ std::getenv( "PAIROP_PAD" ) } ) {
        if ( auto const bytes{ std::strtoull( pad, nullptr, 10 ) } ) { static_cast<char *>( std::malloc( bytes ) )[ 0 ] = 1; }
    }
}
static std::vector<std::size_t> shuffled_order( std::size_t const n ) {
    std::vector<std::size_t> order( n ); for ( std::size_t i = 0; i < n; ++i ) { order[ i ] = i; }
    std::uint64_t x{ 0x9e3779b97f4a7c15ULL };
    for ( std::size_t i = n; i > 1; --i ) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; std::swap( order[ i - 1 ], order[ x % i ] ); }
    return order;
}

int main( int argc, char * argv[] ) {
    std::string const arm    { argc > 1 ? argv[ 1 ] : "frsr"  };
    std::string const op     { argc > 2 ? argv[ 2 ] : "union" };
    std::size_t const count  { argc > 3 ? std::strtoull( argv[ 3 ], nullptr, 10 ) : 100'000 };
    std::string const overlap{ argc > 4 ? argv[ 4 ] : "high" };
    std::size_t const passes { argc > 5 ? std::strtoull( argv[ 5 ], nullptr, 10 ) : 20'000 };
    std::size_t const offset { overlap == "high" ? count / 4 : overlap == "mid" ? count / 2 : 9 * count / 10 };
    // PAIROP_CHURN_MB=<n>: before any fixture exists, allocate n MB of mixed-size blocks (16 B..16 KB, log-uniform),
    // then free them in shuffled order keeping one in PAIROP_CHURN_KEEP (default 64) alive — the fragmented heap a
    // band inherits inside the full benchmark process, where earlier bands built and tore down large working sets.
    std::vector<void *> churn_survivors;
    if ( char const * const churn_mb{ std::getenv( "PAIROP_CHURN_MB" ) } ) {
        std::size_t const budget{ std::size_t( std::strtoull( churn_mb, nullptr, 10 ) ) << 20 };
        char const * const keep_env{ std::getenv( "PAIROP_CHURN_KEEP" ) };
        std::size_t const keep{ keep_env ? std::max<std::size_t>( 1, std::strtoull( keep_env, nullptr, 10 ) ) : 64 };
        std::vector<void *> blocks;
        std::uint64_t x{ 0x2545f4914f6cdd1dULL };
        for ( std::size_t total{ 0 }; total < budget; ) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            std::size_t const size{ ( std::size_t{ 16 } << ( x % 11 ) ) + ( ( x >> 32 ) % 16 ) };
            auto * const block{ static_cast<char *>( std::malloc( size ) ) };
            block[ 0 ] = block[ size - 1 ] = 1;
            blocks.push_back( block ); total += size;
        }
        for ( std::size_t i = blocks.size(); i > 1; --i ) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; std::swap( blocks[ i - 1 ], blocks[ x % i ] ); }
        for ( std::size_t i = 0; i < blocks.size(); ++i ) { if ( i % keep == 0 ) { churn_survivors.push_back( blocks[ i ] ); } else { std::free( blocks[ i ] ); } }
        std::fprintf( stderr, "churn: %zu blocks, %zu kept\n", blocks.size(), churn_survivors.size() );
    }
    if ( op == "lazyfold" ) {
        // LazyFoldOnly/band=<count>: 32 probes (64 chunks x 64 values) each copied and AND-folded in place against a
        // 64-chunk bitset target holding <count> values per chunk.
        std::size_t const band{ count }, chunks{ 64 }, nprobes{ 32 }, probe_card{ 64 };
        auto const band_value{ [ band ]( std::size_t c, std::size_t i ) { return std::uint32_t( c * 65536 + i * ( 65536 / band ) ); } };
        std::size_t const step{ std::max<std::size_t>( 1, band / probe_card ) };
        std::int64_t sink{ 0 };
        if ( arm == "frsr" ) {
            Bitmap target; std::vector<Bitmap> probes( nprobes );
            for ( std::size_t c = 0; c < chunks; ++c ) for ( std::size_t i = 0; i < band; ++i ) { (void)target.add( band_value( c, i ) ); }
            for ( std::size_t p = 0; p < nprobes; ++p ) for ( std::size_t c = 0; c < chunks; ++c ) for ( std::size_t j = 0; j < probe_card; ++j ) { (void)probes[ p ].add( band_value( c, ( j * step + p ) % band ) ); }
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { Bitmap probe{ probes[ p % nprobes ] }; probe &= target; sink += std::int64_t( probe.size() ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "frsr %s band=%zu passes=%zu us/op=%.4f (per chunk %.4f) sink=%lld\n", op.c_str(), band, passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes * chunks ), (long long)sink );
        } else {
            auto * target{ roaring_bitmap_create() }; roaring_bitmap_set_copy_on_write( target, true ); std::vector<roaring_bitmap_t *> probes( nprobes );
            for ( std::size_t c = 0; c < chunks; ++c ) for ( std::size_t i = 0; i < band; ++i ) { roaring_bitmap_add( target, band_value( c, i ) ); }
            for ( std::size_t p = 0; p < nprobes; ++p ) { probes[ p ] = roaring_bitmap_create(); roaring_bitmap_set_copy_on_write( probes[ p ], true ); for ( std::size_t c = 0; c < chunks; ++c ) for ( std::size_t j = 0; j < probe_card; ++j ) { roaring_bitmap_add( probes[ p ], band_value( c, ( j * step + p ) % band ) ); } }
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { auto * probe{ roaring_bitmap_copy( probes[ p % nprobes ] ) }; roaring_bitmap_and_inplace( probe, target ); sink += std::int64_t( roaring_bitmap_get_cardinality( probe ) ); roaring_bitmap_free( probe ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "cpp  %s band=%zu passes=%zu us/op=%.4f (per chunk %.4f) sink=%lld\n", op.c_str(), band, passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes * chunks ), (long long)sink );
            for ( auto * b : probes ) { roaring_bitmap_free( b ); } roaring_bitmap_free( target );
        }
        return 0;
    }
    if ( op == "subuniq" ) {
        // MixedArrayBitsetSubtractUniqueInplace: 64 values i*1024+1 (one sole-owned array) -= the 32768 even values
        // (one bitset), in place on the same bitmap every pass; nothing matches, so the array never changes.
        std::int64_t sink{ 0 };
        if ( arm == "frsr" ) {
            Bitmap sparse, dense;
            for ( std::size_t i = 0; i < 32768; ++i ) { (void)dense.add( std::uint32_t( 2 * i ) ); }
            hold_padding();
            for ( std::size_t i = 0; i < 64; ++i ) { (void)sparse.add( std::uint32_t( i * 1024 + 1 ) ); }
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { sparse -= dense; sink += std::int64_t( sparse.size() ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "frsr %s passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
        } else {
            auto * sparse{ roaring_bitmap_create() }; auto * dense{ roaring_bitmap_create() };
            roaring_bitmap_set_copy_on_write( sparse, true ); roaring_bitmap_set_copy_on_write( dense, true );
            for ( std::size_t i = 0; i < 32768; ++i ) { roaring_bitmap_add( dense, std::uint32_t( 2 * i ) ); }
            roaring_bitmap_run_optimize( dense );
            hold_padding();
            for ( std::size_t i = 0; i < 64; ++i ) { roaring_bitmap_add( sparse, std::uint32_t( i * 1024 + 1 ) ); }
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { roaring_bitmap_andnot_inplace( sparse, dense ); sink += std::int64_t( roaring_bitmap_get_cardinality( sparse ) ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "cpp  %s passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
            roaring_bitmap_free( sparse ); roaring_bitmap_free( dense );
        }
        return 0;
    }
    if ( op == "coldcard" || op == "coldcardnot" ) {
        bool const andnot{ op == "coldcardnot" };
        auto const order{ shuffled_order( 102 ) };
        std::int64_t sink{ 0 };
        if ( arm == "frsr" ) {
            auto pairs{ cold_pairs<Bitmap>( count, []( Bitmap & b, std::uint32_t v ) { (void)b.add( v ); } ) };
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { auto & [ arr, bmp ]{ pairs[ order[ p % 102 ] ] }; Bitmap r{ andnot ? arr - bmp : arr & bmp }; sink += std::int64_t( r.size() ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "frsr %s acard=%zu passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), count, passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
        } else {
            auto pairs{ cold_pairs<roaring_bitmap_t *>( count, []( roaring_bitmap_t * & b, std::uint32_t v ) { if ( !b ) { b = roaring_bitmap_create(); roaring_bitmap_set_copy_on_write( b, true ); } roaring_bitmap_add( b, v ); } ) };
            auto const t0{ std::chrono::steady_clock::now() };
            for ( std::size_t p = 0; p < passes; ++p ) { auto & [ arr, bmp ]{ pairs[ order[ p % 102 ] ] }; auto * r{ andnot ? roaring_bitmap_andnot( arr, bmp ) : roaring_bitmap_and( arr, bmp ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            auto const t1{ std::chrono::steady_clock::now() };
            std::printf( "cpp  %s acard=%zu passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), count, passes, std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
            for ( auto & [ arr, bmp ] : pairs ) { roaring_bitmap_free( arr ); roaring_bitmap_free( bmp ); }
        }
        return 0;
    }
    auto const fx{ make_fixture( op, count, offset ) };
    bool const is_andnot{ op == "difference" || op == "mixedandnot" || op == "satandnot" };
    bool const is_and   { op == "runbitset" || op == "envelope" };
    std::int64_t sink{ 0 };
    if ( arm == "frsr" ) {
        Bitmap a, b;
        for ( auto const v : fx.a ) { (void)a.add( v ); }
        for ( auto const [ lo, hi ] : fx.a_runs ) { a.add_closed_range( lo, hi ); }
        if ( fx.optimize_a ) { a.optimize(); }
        hold_padding();
        for ( auto const v : fx.b ) { (void)b.add( v ); }
        std::vector<std::uint32_t> out( a.size() );
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            if      ( op == "union"      ) { Bitmap r{ a | b }; sink += std::int64_t( r.size() ); }
            else if ( is_andnot          ) { Bitmap r{ a - b }; sink += std::int64_t( r.size() ); }
            else if ( is_and             ) { Bitmap r{ a & b }; sink += std::int64_t( r.size() ); }
            else if ( op == "diffinplace") { Bitmap tmp{ a }; tmp -= b; sink += std::int64_t( tmp.size() ); }
            else if ( op == "unioninplace") { Bitmap tmp{ a }; tmp |= b; sink += std::int64_t( tmp.size() ); }
            else if ( op == "toarray"    ) { sink += std::int64_t( a.to_array_into( { out.data(), out.size() } ) ); }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "frsr %s count=%zu %s passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), count, overlap.c_str(), passes,
            std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
    } else {
        auto * a{ roaring_bitmap_create() }; auto * b{ roaring_bitmap_create() };
        roaring_bitmap_set_copy_on_write( a, true ); roaring_bitmap_set_copy_on_write( b, true );
        for ( auto const v : fx.a ) { roaring_bitmap_add( a, v ); }
        for ( auto const [ lo, hi ] : fx.a_runs ) { roaring_bitmap_add_range_closed( a, lo, hi ); }
        if ( fx.optimize_a ) { roaring_bitmap_run_optimize( a ); }
        hold_padding();
        for ( auto const v : fx.b ) { roaring_bitmap_add( b, v ); }
        if ( op == "mixedandnot" ) { roaring_bitmap_run_optimize( b ); }   // as the band does (a no-op form-wise: alternating bits stay a bitset)
        std::vector<std::uint32_t> out( roaring_bitmap_get_cardinality( a ) );
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            if      ( op == "union"      ) { auto * r{ roaring_bitmap_or( a, b ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            else if ( is_andnot          ) { auto * r{ roaring_bitmap_andnot( a, b ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            else if ( is_and             ) { auto * r{ roaring_bitmap_and( a, b ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            else if ( op == "diffinplace") { auto * tmp{ roaring_bitmap_copy( a ) }; roaring_bitmap_andnot_inplace( tmp, b ); sink += std::int64_t( roaring_bitmap_get_cardinality( tmp ) ); roaring_bitmap_free( tmp ); }
            else if ( op == "unioninplace") { auto * tmp{ roaring_bitmap_copy( a ) }; roaring_bitmap_or_inplace( tmp, b ); sink += std::int64_t( roaring_bitmap_get_cardinality( tmp ) ); roaring_bitmap_free( tmp ); }
            else if ( op == "toarray"    ) { roaring_bitmap_to_uint32_array( a, out.data() ); sink += std::int64_t( out.size() ); }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "cpp  %s count=%zu %s passes=%zu us/op=%.4f sink=%lld\n", op.c_str(), count, overlap.c_str(), passes,
            std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
        roaring_bitmap_free( a ); roaring_bitmap_free( b );
    }
    return 0;
}
