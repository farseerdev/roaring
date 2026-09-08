// Profiling driver for the BulkAddRows shape: the fixture is built ONCE and the
// timed insert loop then runs long enough that a sampling profiler attributes
// the insert path itself instead of the fixture construction. The benchmark's
// own band cannot do that - it rebuilds its fixture per repetition, so at any
// useful sample count the profile is dominated by the setup.
//
// Not part of the build (nothing globs this directory). Compile it with the
// benchmark target's own flags, e.g.
//
//   clang++ -std=c++2c -O3 -march=native -DFRSR_ROARING_HAS_CROARING=1 //           -I../../include -I<psi-vm>/include -I<croaring>/include //           bulkadd_prof.cpp -lroaring -o bulkadd_prof
//
// then: bulkadd_prof <frsr|frsr-batch|cpp|cpp-batch> <narrow|wide> [passes] [group]
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
    std::string const arm  { argc > 1 ? argv[ 1 ] : "frsr"   };
    std::string const shape{ argc > 2 ? argv[ 2 ] : "narrow" };
    std::size_t const passes{ argc > 3 ? std::strtoull( argv[ 3 ], nullptr, 10 ) : 50 };

    std::size_t const values  { shape == "wide" ?   256u :      8u };
    std::size_t const existing{ shape == "wide" ? 4'000u : 100'000u };
    std::size_t const fresh   { 200'000 };

    std::uint32_t row{ 0 };
    std::int64_t  added{ 0 };
    auto const t_start{ std::chrono::steady_clock::now() };

    // Batch modes: the same row ids, but handed over in per-bitmap sorted groups —
    // the shape a batched insert API can exploit (resolve the container once, append
    // the whole group) instead of paying the per-value bookkeeping.
    std::size_t const group{ argc > 4 ? std::strtoull( argv[ 4 ], nullptr, 10 ) : 256 };

    if ( arm == "frsr-batch" ) {
        std::vector<Bitmap> index( values );
        for ( std::size_t i = 0; i < existing; ++i ) {
            for ( std::size_t v = 0; v < values; ++v ) { (void)index[ v ].add( row++ ); }
        }
        std::vector<std::vector<std::uint32_t>> buf( values );
        for ( auto & b : buf ) { b.reserve( group ); }
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            for ( std::size_t i = 0; i < fresh; ++i ) {
                auto const v{ i % values };
                buf[ v ].push_back( row++ );
                if ( buf[ v ].size() == group ) {
                    index[ v ].add_many_sorted( { buf[ v ].data(), buf[ v ].size() } );
                    added += (std::int64_t)buf[ v ].size();
                    buf[ v ].clear();
                }
            }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "frsr-batch %s group=%zu timed=%.3fs adds=%lld ns/add=%.2f\n",
            shape.c_str(), group, std::chrono::duration<double>( t1 - t0 ).count(), (long long)added,
            std::chrono::duration<double, std::nano>( t1 - t0 ).count() / double( added ) );
        return added != 0 ? 0 : 1;
    }
    if ( arm == "cpp-batch" ) {
        std::vector<roaring_bitmap_t *> index( values );
        for ( auto & bm : index ) { bm = roaring_bitmap_create(); roaring_bitmap_set_copy_on_write( bm, true ); }
        for ( std::size_t i = 0; i < existing; ++i ) {
            for ( std::size_t v = 0; v < values; ++v ) { roaring_bitmap_add( index[ v ], row++ ); }
        }
        std::vector<std::vector<std::uint32_t>> buf( values );
        for ( auto & b : buf ) { b.reserve( group ); }
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            for ( std::size_t i = 0; i < fresh; ++i ) {
                auto const v{ i % values };
                buf[ v ].push_back( row++ );
                if ( buf[ v ].size() == group ) {
                    roaring_bitmap_add_many( index[ v ], buf[ v ].size(), buf[ v ].data() );
                    added += (std::int64_t)buf[ v ].size();
                    buf[ v ].clear();
                }
            }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "cpp-batch  %s group=%zu timed=%.3fs adds=%lld ns/add=%.2f\n",
            shape.c_str(), group, std::chrono::duration<double>( t1 - t0 ).count(), (long long)added,
            std::chrono::duration<double, std::nano>( t1 - t0 ).count() / double( added ) );
        for ( auto * bm : index ) { roaring_bitmap_free( bm ); }
        return added != 0 ? 0 : 1;
    }

    if ( arm == "frsr" ) {
        std::vector<Bitmap> index( values );
        for ( std::size_t i = 0; i < existing; ++i ) {
            for ( std::size_t v = 0; v < values; ++v ) { (void)index[ v ].add( row++ ); }
        }
        std::vector<Bitmap::bulk_context> ctx( values );
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            for ( std::size_t i = 0; i < fresh; ++i ) {
                auto const v{ i % values };
                added += index[ v ].add_bulk( ctx[ v ], row++ );
            }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        std::printf( "frsr   %s setup=%.2fs timed=%.3fs adds=%zu ns/add=%.2f added=%lld\n",
            shape.c_str(),
            std::chrono::duration<double>( t0 - t_start ).count(),
            std::chrono::duration<double>( t1 - t0 ).count(),
            passes * fresh,
            std::chrono::duration<double, std::nano>( t1 - t0 ).count() / double( passes * fresh ),
            (long long)added );
    } else {
        std::vector<roaring_bitmap_t *> index( values );
        for ( auto & bm : index ) { bm = roaring_bitmap_create(); roaring_bitmap_set_copy_on_write( bm, true ); }
        for ( std::size_t i = 0; i < existing; ++i ) {
            for ( std::size_t v = 0; v < values; ++v ) { roaring_bitmap_add( index[ v ], row++ ); }
        }
        std::vector<roaring_bulk_context_t> ctx( values );
        for ( auto & c : ctx ) { c = roaring_bulk_context_t{}; }
        auto const t0{ std::chrono::steady_clock::now() };
        for ( std::size_t p = 0; p < passes; ++p ) {
            for ( std::size_t i = 0; i < fresh; ++i ) {
                auto const v{ i % values };
                roaring_bitmap_add_bulk( index[ v ], &ctx[ v ], row++ );
            }
        }
        auto const t1{ std::chrono::steady_clock::now() };
        added = (std::int64_t)( passes * fresh );
        std::printf( "cpp    %s setup=%.2fs timed=%.3fs adds=%zu ns/add=%.2f added=%lld\n",
            shape.c_str(),
            std::chrono::duration<double>( t0 - t_start ).count(),
            std::chrono::duration<double>( t1 - t0 ).count(),
            passes * fresh,
            std::chrono::duration<double, std::nano>( t1 - t0 ).count() / double( passes * fresh ),
            (long long)added );
        for ( auto * bm : index ) { roaring_bitmap_free( bm ); }
    }
    return added != 0 ? 0 : 1;
}
