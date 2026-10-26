// index_of (the ordinal of an id among the ids, or "absent") on a table of ids, in the arms a consumer can pick from:
//   baseline          one bit per id in 64-bit words and a u32 count of the ids before each word
//   flat_narrow       the same words with a u16 count per word, relative to a u32 base per 1024 words
//   sorted_array      std::vector<u32> and lower_bound
//   ranked_default    ranked frozen image of the default bitmap (which picks run containers by itself) after optimize()
//   ranked_bitsets    ranked frozen image of a bitmap that never picks runs by itself (arrays and bitsets only)
//   ranked_runs_64    the same, then runs only for chunks whose mean run length is at least 64
//   ranked_runs_1024  the same with a bar of 1024
//   ranked_runs_4096  the same with a bar of 4096 (at most 16 runs in a chunk)
//   croaring          roaring_bitmap_get_index (when built with CRoaring)
// Tables hold about `entries` ids spread over entries * range/entries values. Four probe patterns run on every table:
//   hit_indep   random present ids, every lookup independent of the previous ones (throughput)
//   miss_indep  random absent ids inside the range, independent
//   hit_dep     present ids, the next probe is chosen from the previous answer (a dependent chain: latency)
//   miss_dep    absent ids, dependent
// Arms run interleaved round after round and the fastest round per arm is reported, in nanoseconds per lookup.
// Every arm's answers are compared with the baseline's before anything is timed.
//
//   frsr_roaring_rank_bench [--rounds n] [--sizes a,b,c] [--ratios x,y,z]

#include <frsr/roaring/ranked_view.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#if FRSR_ROARING_HAS_CROARING
#   include <roaring/roaring.h>
#endif

namespace {

using namespace frsr::roaring;
using table      = bitmap<std::uint32_t>;
using lazy_table = rank_table_bitmap<std::uint32_t>;

constexpr std::uint32_t absent{ ranked_view<std::uint32_t>::npos };

using runner = std::function<std::uint64_t( std::span<std::uint32_t const>, bool )>;

// The lookup is a template argument of the loop so that it inlines: the cost measured is the lookup's, not a call's.
template <typename Lookup>
runner make_runner( Lookup lookup ) {
    return [ lookup ]( std::span<std::uint32_t const> const keys, bool const dependent ) {
        std::uint64_t sum{};
        if ( !dependent ) {
            for ( auto const key : keys ) {
                sum += lookup( key );
            }
            return sum;
        }
        auto const mask{ static_cast<std::uint32_t>( keys.size() - 1 ) };
        std::uint32_t at{};
        for ( std::size_t i{ 0 }; i < keys.size(); ++i ) {
            auto const answer{ lookup( keys[ at ] ) };
            sum += answer;
            at = ( at + answer + 1U + keys[ at ] ) & mask;
        }
        return sum;
    };
}

struct arm {
    std::string name;
    runner      run;
    std::size_t bytes{};
    double      best_ns{ 1e300 };
    int         passes{ 1 };
};

// The structure this work is measured against: a bit string and a running count at each 64-bit word.
struct flat_table {
    std::vector<std::uint64_t> words;
    std::vector<std::uint32_t> prefix;

    explicit flat_table( std::vector<std::uint32_t> const & ids ) {
        words.assign( ( std::size_t{ ids.back() } >> 6U ) + 1U, 0 );
        for ( auto const id : ids ) {
            words[ id >> 6U ] |= std::uint64_t{ 1 } << ( id & 63U );
        }
        prefix.resize( words.size() );
        std::uint32_t running{};
        for ( std::size_t w{ 0 }; w < words.size(); ++w ) {
            prefix[ w ] = running;
            running += static_cast<std::uint32_t>( std::popcount( words[ w ] ) );
        }
    }
    [[nodiscard]] bool test( std::uint32_t const x ) const noexcept {
        return ( ( x >> 6U ) < words.size() ) && ( ( words[ x >> 6U ] >> ( x & 63U ) ) & 1U );
    }
    [[nodiscard]] std::uint32_t index_of( std::uint32_t const x ) const noexcept {
        auto const w{ x >> 6U };
        auto const below{ prefix[ w ] + static_cast<std::uint32_t>( std::popcount( words[ w ] & ( ( std::uint64_t{ 1 } << ( x & 63U ) ) - 1U ) ) ) };
        return ( ( words[ w ] >> ( x & 63U ) ) & 1U ) ? below : absent;
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return words.size() * 8 + prefix.size() * 4; }
};

// A u16 count per word, relative to the first word of its block of 1024 words (65536 ids), and a u32 base per block.
struct flat_narrow_table {
    std::vector<std::uint64_t> words;
    std::vector<std::uint16_t> count;
    std::vector<std::uint32_t> base;

    explicit flat_narrow_table( flat_table const & flat ) : words{ flat.words }, count( flat.words.size() ), base( ( flat.words.size() >> 10U ) + 1U ) {
        std::uint32_t before_block{};
        std::uint32_t within{};
        for ( std::size_t w{ 0 }; w < words.size(); ++w ) {
            if ( ( w & 1023U ) == 0 ) {
                before_block += within;
                within = 0;
                base[ w >> 10U ] = before_block;
            }
            count[ w ] = static_cast<std::uint16_t>( within );
            within += static_cast<std::uint32_t>( std::popcount( words[ w ] ) );
        }
    }
    [[nodiscard]] std::uint32_t index_of( std::uint32_t const x ) const noexcept {
        auto const w{ x >> 6U };
        auto const below{ base[ w >> 10U ] + count[ w ] + static_cast<std::uint32_t>( std::popcount( words[ w ] & ( ( std::uint64_t{ 1 } << ( x & 63U ) ) - 1U ) ) ) };
        return ( ( words[ w ] >> ( x & 63U ) ) & 1U ) ? below : absent;
    }
    [[nodiscard]] std::size_t bytes() const noexcept { return words.size() * 8 + count.size() * 2 + base.size() * 4; }
};

struct frozen_arm {
    std::vector<std::uint64_t> storage;
    ranked_view<std::uint32_t> view;
    std::size_t                size{};
};

template <typename Table>
std::unique_ptr<frozen_arm> freeze_ranked( Table const & t ) {
    auto result{ std::make_unique<frozen_arm>() };
    result->size = t.frozen_size_bytes( { .ranked = true } );
    result->storage.resize( ( result->size + 7 ) / 8 );
    t.frozen_serialize( { reinterpret_cast<std::byte *>( result->storage.data() ), result->size }, { .ranked = true } );
    result->view = ranked_view<std::uint32_t>::from( std::span<std::byte const>{ reinterpret_cast<std::byte const *>( result->storage.data() ), result->size } );
    return result;
}

std::vector<double> parse_list( char const * text ) {
    std::vector<double> values;
    std::stringstream   in{ text };
    for ( std::string item; std::getline( in, item, ',' ); ) {
        values.push_back( std::atof( item.c_str() ) );
    }
    return values;
}

} // namespace

int main( int argc, char ** argv ) {
    int rounds{ 7 };
    std::vector<double> sizes{ 50, 10000, 1000000 };
    std::vector<double> ratios{ 1.001, 1.01, 1.03, 1.05, 1.1, 2.0, 16.0, 68.0, 590.0 };
    for ( int i{ 1 }; i + 1 < argc; i += 2 ) {
        if ( std::strcmp( argv[ i ], "--rounds" ) == 0 ) {
            rounds = std::atoi( argv[ i + 1 ] );
        } else if ( std::strcmp( argv[ i ], "--sizes" ) == 0 ) {
            sizes = parse_list( argv[ i + 1 ] );
        } else if ( std::strcmp( argv[ i ], "--ratios" ) == 0 ) {
            ratios = parse_list( argv[ i + 1 ] );
        }
    }
    constexpr std::size_t batch{ 1U << 14 };
    std::printf( "entries\trange/n\tprobes\tarm\tns/lookup\tB/entry\tvs_baseline\n" );
    std::uint64_t sink{};
    for ( auto const entries : sizes ) {
        for ( auto const ratio : ratios ) {
            std::mt19937_64 rng{ 1 };
            // The gaps between neighbouring ids are geometric: every value is present with probability 1 / ratio.
            std::geometric_distribution<std::uint32_t> gap{ 1.0 / ratio };
            std::vector<std::uint32_t> ids;
            std::uint64_t at{ gap( rng ) };
            auto const limit{ static_cast<std::uint64_t>( entries * ratio ) };
            while ( at < limit ) {
                ids.push_back( static_cast<std::uint32_t>( at ) );
                at += 1U + gap( rng );
            }
            if ( ids.size() < 2 ) {
                continue;
            }
            flat_table const        flat{ ids };
            flat_narrow_table const narrow{ flat };

            std::vector<std::uint32_t> hits( batch );
            for ( auto & p : hits ) {
                p = ids[ rng() % ids.size() ];
            }
            std::vector<std::uint32_t> misses;
            for ( std::uint64_t tries{ 0 }; misses.size() < batch && tries < ( std::uint64_t{ 1 } << 26 ); ++tries ) {
                auto const x{ static_cast<std::uint32_t>( rng() % ( std::uint64_t{ ids.back() } + 1 ) ) };
                if ( !flat.test( x ) ) {
                    misses.push_back( x );
                }
            }
            if ( !misses.empty() ) {
                auto const found{ misses.size() };
                misses.resize( batch );
                for ( std::size_t i{ found }; i < batch; ++i ) {
                    misses[ i ] = misses[ rng() % found ];
                }
            }

            table      optimized;
            lazy_table lazy;
            optimized.add_sorted_many( ids );
            optimized.optimize();
            lazy.add_sorted_many( ids );
            lazy_table shaped_64{ lazy };
            lazy_table shaped_1024{ lazy };
            lazy_table shaped_4096{ lazy };
            shaped_64.optimize_run_shaped( 64 );
            shaped_1024.optimize_run_shaped( 1024 );
            shaped_4096.optimize_run_shaped( 4096 );
            auto const frozen_default{ freeze_ranked( optimized ) };
            auto const frozen_bitsets{ freeze_ranked( lazy ) };
            auto const frozen_64{ freeze_ranked( shaped_64 ) };
            auto const frozen_1024{ freeze_ranked( shaped_1024 ) };
            auto const frozen_4096{ freeze_ranked( shaped_4096 ) };

            std::vector<arm> arms;
            arms.push_back( { "baseline", make_runner( [ &flat ]( std::uint32_t const k ) { return flat.index_of( k ); } ), flat.bytes() } );
            arms.push_back( { "flat_narrow", make_runner( [ &narrow ]( std::uint32_t const k ) { return narrow.index_of( k ); } ), narrow.bytes() } );
            arms.push_back( { "sorted_array", make_runner( [ &ids ]( std::uint32_t const k ) {
                auto const it{ std::ranges::lower_bound( ids, k ) };
                return ( it != ids.end() && *it == k ) ? static_cast<std::uint32_t>( it - ids.begin() ) : absent;
            } ), ids.size() * 4 } );
            auto const add_ranked{ [ & ]( char const * name, frozen_arm const & f ) {
                arms.push_back( { name, make_runner( [ view = &f.view ]( std::uint32_t const k ) { return view->index_of( k ); } ), f.size } );
            } };
            add_ranked( "ranked_default", *frozen_default );
            add_ranked( "ranked_bitsets", *frozen_bitsets );
            add_ranked( "ranked_runs_64", *frozen_64 );
            add_ranked( "ranked_runs_1024", *frozen_1024 );
            add_ranked( "ranked_runs_4096", *frozen_4096 );
#if FRSR_ROARING_HAS_CROARING
            auto * const reference{ roaring_bitmap_of_ptr( ids.size(), ids.data() ) };
            roaring_bitmap_run_optimize( reference );
            arms.push_back( { "croaring", make_runner( [ reference ]( std::uint32_t const k ) { return static_cast<std::uint32_t>( roaring_bitmap_get_index( reference, k ) ); } ), roaring_bitmap_size_in_bytes( reference ) } );
#endif
            struct pattern {
                char const *                       name;
                std::vector<std::uint32_t> const * keys;
                bool                               dependent;
            };
            std::vector<pattern> patterns{ { "hit_indep", &hits, false }, { "hit_dep", &hits, true } };
            if ( !misses.empty() ) {
                patterns.push_back( { "miss_indep", &misses, false } );
                patterns.push_back( { "miss_dep", &misses, true } );
            }
            for ( auto const & pat : patterns ) {
                auto const expected{ arms[ 0 ].run( *pat.keys, pat.dependent ) };
                for ( auto & a : arms ) {
                    auto const begin{ std::chrono::steady_clock::now() };
                    auto const answer{ a.run( *pat.keys, pat.dependent ) };
                    auto const one{ std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - begin ).count() };
                    if ( answer != expected ) {
                        std::fprintf( stderr, "arm %s disagrees (%s, %zu ids, range/n %g)\n", a.name.c_str(), pat.name, ids.size(), ratio );
                        return 1;
                    }
                    a.passes  = static_cast<int>( std::clamp( 4e6 / std::max( one, 1.0 ), 1.0, 512.0 ) );
                    a.best_ns = 1e300;
                }
                for ( int round{ 0 }; round < rounds; ++round ) {
                    for ( auto & a : arms ) {
                        auto const begin{ std::chrono::steady_clock::now() };
                        for ( int pass{ 0 }; pass < a.passes; ++pass ) {
                            sink += a.run( *pat.keys, pat.dependent );
                        }
                        auto const ns{ std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - begin ).count() / static_cast<double>( a.passes ) / static_cast<double>( batch ) };
                        a.best_ns = std::min( a.best_ns, ns );
                    }
                }
                for ( auto const & a : arms ) {
                    std::printf( "%zu\t%g\t%s\t%s\t%.2f\t%.3f\t%.2f\n", ids.size(), ratio, pat.name, a.name.c_str(), a.best_ns,
                        static_cast<double>( a.bytes ) / static_cast<double>( ids.size() ), a.best_ns / arms[ 0 ].best_ns );
                }
                std::fflush( stdout );
            }
#if FRSR_ROARING_HAS_CROARING
            roaring_bitmap_free( reference );
#endif
        }
    }
    if ( sink == 1 ) {
        std::puts( "" );
    }
    return 0;
}
