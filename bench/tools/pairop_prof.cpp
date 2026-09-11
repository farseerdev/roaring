// Profiling driver for the binary set_ops bands (Union / Difference /
// DifferenceInplace on the [0,count) vs [offset,offset+count) pair fixture) and
// ToArray: the fixture is built once and the operation loops long enough for a
// sampling profiler to attribute the operation itself. The benchmark's bands
// run a fixed number of inner reps, so their profiles carry too few samples.
//
// Not part of the build (nothing globs this directory). Compile with the
// benchmark target's own flags (see bulkadd_prof.cpp), then:
//
//   pairop_prof <frsr|cpp> <union|difference|diffinplace|toarray> <count> <high|mid|low> [passes]
#include <frsr/roaring/bitmap.hpp>
#include <roaring/roaring.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Bitmap = frsr::roaring::bitmap<
    std::uint32_t,
    frsr::roaring::default_container_set<std::uint32_t>,
    frsr::roaring::detail::cow_atomic_refcount,
    frsr::roaring::detail::run_selection_lazy
>;

int main( int argc, char * argv[] ) {
    std::string const arm    { argc > 1 ? argv[ 1 ] : "frsr"  };
    std::string const op     { argc > 2 ? argv[ 2 ] : "union" };
    std::size_t const count  { argc > 3 ? std::strtoull( argv[ 3 ], nullptr, 10 ) : 100'000 };
    std::string const overlap{ argc > 4 ? argv[ 4 ] : "high" };
    std::size_t const passes { argc > 5 ? std::strtoull( argv[ 5 ], nullptr, 10 ) : 20'000 };
    std::size_t const offset { overlap == "high" ? count / 4 : overlap == "mid" ? count / 2 : 9 * count / 10 };
    // ToArray/chunks=16/shape=array: the band's array shape (16 chunks, 2000 values at stride 16).
    std::size_t const ta_chunks{ 16 }, ta_per_chunk{ 2000 }, ta_stride{ 16 };

    std::int64_t sink{ 0 };
    if ( arm == "frsr" ) {
        Bitmap a, b;
        if ( op == "toarray" ) {
            for ( std::size_t c = 0; c < ta_chunks; ++c ) for ( std::size_t i = 0; i < ta_per_chunk; ++i ) { (void)a.add( std::uint32_t( c * 65536 + i * ta_stride ) ); }
        } else {
            for ( std::size_t i = 0; i < count; ++i ) { (void)a.add( std::uint32_t( i ) ); (void)b.add( std::uint32_t( i + offset ) ); }
        }
        std::vector<std::uint32_t> out( a.size() );
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            if      ( op == "union"      ) { Bitmap r{ a | b }; sink += std::int64_t( r.size() ); }
            else if ( op == "difference" ) { Bitmap r{ a - b }; sink += std::int64_t( r.size() ); }
            else if ( op == "diffinplace") { Bitmap tmp{ a }; tmp -= b; sink += std::int64_t( tmp.size() ); }
            else if ( op == "toarray"    ) { sink += std::int64_t( a.to_array_into( { out.data(), out.size() } ) ); }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "frsr %s count=%zu %s passes=%zu us/op=%.3f sink=%lld\n", op.c_str(), count, overlap.c_str(), passes,
            std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
    } else {
        auto * a{ roaring_bitmap_create() }; auto * b{ roaring_bitmap_create() };
        roaring_bitmap_set_copy_on_write( a, true ); roaring_bitmap_set_copy_on_write( b, true );
        if ( op == "toarray" ) {
            for ( std::size_t c = 0; c < ta_chunks; ++c ) for ( std::size_t i = 0; i < ta_per_chunk; ++i ) { roaring_bitmap_add( a, std::uint32_t( c * 65536 + i * ta_stride ) ); }
        } else {
            for ( std::size_t i = 0; i < count; ++i ) { roaring_bitmap_add( a, std::uint32_t( i ) ); roaring_bitmap_add( b, std::uint32_t( i + offset ) ); }
        }
        std::vector<std::uint32_t> out( roaring_bitmap_get_cardinality( a ) );
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            if      ( op == "union"      ) { auto * r{ roaring_bitmap_or( a, b ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            else if ( op == "difference" ) { auto * r{ roaring_bitmap_andnot( a, b ) }; sink += std::int64_t( roaring_bitmap_get_cardinality( r ) ); roaring_bitmap_free( r ); }
            else if ( op == "diffinplace") { auto * tmp{ roaring_bitmap_copy( a ) }; roaring_bitmap_andnot_inplace( tmp, b ); sink += std::int64_t( roaring_bitmap_get_cardinality( tmp ) ); roaring_bitmap_free( tmp ); }
            else if ( op == "toarray"    ) { roaring_bitmap_to_uint32_array( a, out.data() ); sink += std::int64_t( out.size() ); }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "cpp  %s count=%zu %s passes=%zu us/op=%.3f sink=%lld\n", op.c_str(), count, overlap.c_str(), passes,
            std::chrono::duration<double, std::micro>( t1 - t0 ).count() / double( passes ), (long long)sink );
        roaring_bitmap_free( a ); roaring_bitmap_free( b );
    }
    return sink != 0 ? 0 : 1;
}
