// optimize_keep_bitsets() and optimize() cross-checked against CRoaring's roaring_bitmap_run_optimize by their
// portable serialization, which spells out every chunk's kind: both libraries start from the same bytes (so every
// chunk starts as the same kind), and after the call the bytes must be equal again, and stay so on a second call.
// The chunks are scattered and run-shaped arrays, dense and run-shaped bitsets, run containers that values were
// added to after they became runs, and the shapes at each rule's size boundary.
//
// One difference is left out by construction: at exactly 4096 values CRoaring holds an array (its array limit is
// inclusive) where this library holds a bitset (the two serialize to the same 8192 bytes), so no chunk of the
// generated bitmaps holds exactly 4096 values.

#include <frsr/roaring/bitmap.hpp>

#include <roaring/roaring.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace frsr::roaring;

using eager_bitmap       = bitmap<std::uint32_t>;
using atomic_lazy_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>;
using bytes_t            = std::vector<std::byte>;

constexpr std::uint32_t chunk{ 65'536U };

struct croaring {
    croaring() : r{ roaring_bitmap_create() } {}
    explicit croaring( roaring_bitmap_t * const raw ) noexcept : r{ raw } {}
    croaring( croaring const & ) = delete;
    ~croaring() { roaring_bitmap_free( r ); }

    [[nodiscard]] bytes_t bytes() const {
        bytes_t out( roaring_bitmap_portable_size_in_bytes( r ) );
        roaring_bitmap_portable_serialize( r, reinterpret_cast<char *>( out.data() ) );
        return out;
    }

    roaring_bitmap_t * r;
};

template <typename Bitmap>
[[nodiscard]] bytes_t bytes_of( Bitmap const & bitmap ) {
    bytes_t out( bitmap.portable_size_in_bytes() );
    auto const written{ bitmap.portable_serialize( out ) };
    EXPECT_EQ( written, out.size() );
    return out;
}

void add_range( croaring & target, std::uint32_t const first, std::uint32_t const last ) {
    roaring_bitmap_add_range_closed( target.r, first, last );
}

// `runs` runs of `length` values each, `gap` absent values apart, from `first` on, added one value at a time.
void add_runs( croaring & target, std::uint32_t const first, std::uint32_t const runs, std::uint32_t const length, std::uint32_t const gap ) {
    for ( std::uint32_t run{ 0U }; run < runs; ++run ) {
        auto const begin{ first + run * ( length + gap ) };
        for ( std::uint32_t offset{ 0U }; offset < length; ++offset ) {
            roaring_bitmap_add( target.r, begin + offset );
        }
    }
}

void add_scattered( croaring & target, std::mt19937_64 & rng, std::uint32_t const base, std::uint32_t const count ) {
    std::uniform_int_distribution<std::uint32_t> low{ 0U, chunk - 1U };
    for ( std::uint32_t i{ 0U }; i < count; ++i ) {
        roaring_bitmap_add( target.r, base + low( rng ) );
    }
}

// Random runs, each 1 to `max_length` values long and 1 to `max_gap` apart, from `base` to the chunk's end at most.
void add_random_runs( croaring & target, std::mt19937_64 & rng, std::uint32_t const base, std::uint32_t const max_length, std::uint32_t const max_gap ) {
    std::uniform_int_distribution<std::uint32_t> length{ 1U, max_length };
    std::uniform_int_distribution<std::uint32_t> gap{ 1U, max_gap };
    std::uint32_t at{ gap( rng ) };
    while ( at < chunk ) {
        auto const last{ std::min( at + length( rng ) - 1U, chunk - 1U ) };
        for ( auto value{ at }; value <= last; ++value ) {
            roaring_bitmap_add( target.r, base + value );
        }
        at = last + 1U + gap( rng );
    }
}

// Run containers first, made by ranges and run_optimize, then given scattered values and holes one at a time:
// CRoaring keeps a run container a run container under single adds and removes. Then every other shape, built by
// single adds into fresh chunks, as arrays (to 4096 values) and bitsets (beyond).
[[nodiscard]] bytes_t random_source( std::mt19937_64 & rng, unsigned const chunks ) {
    croaring source;
    std::uniform_int_distribution<int> shape_of{ 0, 8 };
    std::vector<int> shapes( chunks );
    for ( auto & shape : shapes ) {
        shape = shape_of( rng );
    }
    std::uniform_int_distribution<std::uint32_t> range_length{ 1U, 20'000U };
    for ( unsigned c{ 0U }; c < chunks; ++c ) {
        if ( shapes[ c ] != 0 ) {
            continue;
        }
        auto const base{ c * chunk };
        for ( std::uint32_t at{ 0U }; at + 1U < chunk; ) {
            auto const last{ std::min( at + range_length( rng ), chunk - 1U ) };
            add_range( source, base + at, base + last );
            at = last + 1U + range_length( rng ) / 4U;
        }
    }
    roaring_bitmap_run_optimize( source.r );
    std::uniform_int_distribution<std::uint32_t> singles{ 0U, 3'000U };
    for ( unsigned c{ 0U }; c < chunks; ++c ) {
        if ( shapes[ c ] != 0 ) {
            continue;
        }
        auto const base{ c * chunk };
        add_scattered( source, rng, base, singles( rng ) );
        std::uniform_int_distribution<std::uint32_t> low{ 0U, chunk - 1U };
        for ( auto holes{ singles( rng ) }; holes != 0U; --holes ) {
            roaring_bitmap_remove( source.r, base + low( rng ) );
        }
    }
    std::uniform_int_distribution<std::uint32_t> array_count{ 1U, 4'000U };
    std::uniform_int_distribution<std::uint32_t> bitset_count{ 4'200U, 40'000U };
    for ( unsigned c{ 0U }; c < chunks; ++c ) {
        auto const base{ c * chunk };
        switch ( shapes[ c ] ) {
            case 1: add_scattered( source, rng, base, array_count( rng ) ); break;
            case 2: add_random_runs( source, rng, base, 40U, 3'000U ); break;
            case 3: add_scattered( source, rng, base, bitset_count( rng ) ); break;
            case 4: add_random_runs( source, rng, base, 200U, 6U ); break;
            case 5: add_random_runs( source, rng, base, 4U, 3U ); break;
            // the boundaries: an array tied with its runs and one value past it, a bitset just below and at the
            // run count where runs stop being smaller
            case 6: add_runs( source, base, 99U, 2U, 1U ); add_runs( source, base + 297U, 1U, 3U + ( rng() & 1U ), 0U ); break;
            case 7: add_runs( source, base, 2'047U + ( rng() & 1U ), 3U, 1U ); break;
            case 8: add_runs( source, base, 1U, 1U + static_cast<std::uint32_t>( rng() % 70U ), 0U ); break;
            default: break;
        }
        if ( roaring_bitmap_range_cardinality( source.r, base, std::uint64_t{ base } + chunk ) == 4'096U ) {
            auto const before{ base == 0U ? std::uint64_t{ 0 } : roaring_bitmap_rank( source.r, base - 1U ) };
            std::uint32_t first{};
            EXPECT_TRUE( roaring_bitmap_select( source.r, static_cast<std::uint32_t>( before ), &first ) );
            roaring_bitmap_remove( source.r, first );
        }
    }
    return source.bytes();
}

template <typename Bitmap>
void optimize_trials( std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    for ( int trial{ 0 }; trial < 40; ++trial ) {
        auto const source{ random_source( rng, 24U ) };
        croaring const reference{ roaring_bitmap_portable_deserialize_safe( reinterpret_cast<char const *>( source.data() ), source.size() ) };
        ASSERT_NE( reference.r, nullptr );
        roaring_bitmap_run_optimize( reference.r );
        auto const expected{ reference.bytes() };

        auto const loaded{ Bitmap::portable_deserialize( source ) };
        ASSERT_TRUE( loaded.has_value() );
        ASSERT_EQ( bytes_of( *loaded ), source ) << "trial " << trial << ": the source does not load as the same kinds";
        for ( auto const keep_bitsets : { true, false } ) {
            auto ours{ *loaded };
            auto const optimize{ [ & ] {
                if ( keep_bitsets ) {
                    ours.optimize_keep_bitsets();
                } else {
                    ours.optimize();
                }
            } };
            optimize();
            EXPECT_EQ( ours.validate(), nullptr );
            EXPECT_EQ( bytes_of( ours ), expected ) << "trial " << trial << " keep_bitsets=" << keep_bitsets;
            optimize();
            EXPECT_EQ( bytes_of( ours ), expected ) << "trial " << trial << " keep_bitsets=" << keep_bitsets << ", second call";
            EXPECT_EQ( bytes_of( *loaded ), source ) << "trial " << trial << ": the copy's source changed";
        }
    }
}

TEST( OptimizeCrosscheck, eager ) { optimize_trials<eager_bitmap>( 0x0B71'0001 ); }
TEST( OptimizeCrosscheck, atomic_lazy ) { optimize_trials<atomic_lazy_bitmap>( 0x0B71'0002 ); }

} // namespace
