// The reference's tests whose expectations come from the reference itself (an interval query checked against another
// interval query, an in-place operation against the out-of-place one), replayed here against the reference directly:
// both libraries are given the same values and every answer of this library is compared with CRoaring's. The interval
// ends are 64-bit on both sides (key_range here, the uint64_t range functions there), so ends past 2^32 are among them:
// both must clip them to 2^32 (CRoaring 5.5.5 does since its fix for wrapped range ends).

#include <frsr/roaring/bitmap.hpp>

#include <roaring/roaring.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace {

using namespace frsr::roaring;

using values = std::vector<std::uint32_t>;

template <typename Cow, typename Runs>
using bitmap_of = bitmap<std::uint32_t, default_container_set<std::uint32_t>, Cow, Runs>;

template <typename Bitmap>
class UpstreamReference : public ::testing::Test {};

using reference_bitmaps = ::testing::Types<
    bitmap_of<detail::cow_value_semantics, detail::run_selection_eager>,
    bitmap_of<detail::cow_atomic_refcount, detail::run_selection_lazy>
>;
TYPED_TEST_SUITE( UpstreamReference, reference_bitmaps );

constexpr std::uint32_t chunk{ 65'536U };
constexpr std::uint64_t universe{ std::uint64_t{ 1 } << 32 };
constexpr std::uint32_t max32{ std::numeric_limits<std::uint32_t>::max() };
constexpr std::uint64_t max64{ std::numeric_limits<std::uint64_t>::max() };

struct croaring {
    croaring() : r{ roaring_bitmap_create() } {}
    explicit croaring( roaring_bitmap_t * const raw ) noexcept : r{ raw } {}
    croaring( croaring const & ) = delete;
    ~croaring() { roaring_bitmap_free( r ); }

    [[nodiscard]] values to_values() const {
        values out( roaring_bitmap_get_cardinality( r ) );
        roaring_bitmap_to_uint32_array( r, out.data() );
        return out;
    }

    roaring_bitmap_t * r;
};

template <typename Bitmap>
[[nodiscard]] values values_of( Bitmap const & bitmap ) {
    auto const held{ bitmap.to_vector() };
    return values( held.begin(), held.end() );
}

// The same values in both libraries, both optimized as the reference's run_optimize does.
template <typename Bitmap>
struct in_both {
    explicit in_both( values const & held ) {
        ours.add_many( held );
        roaring_bitmap_add_many( theirs.r, held.size(), held.data() );
        ours.optimize_keep_bitsets();
        roaring_bitmap_run_optimize( theirs.r );
    }

    Bitmap ours;
    croaring theirs;
};

// The values of the reference's interval tests: arrays, bitsets and runs at the given chunks, and a few single values.
[[nodiscard]] values mixed_containers( std::initializer_list<std::uint32_t> const array_keys, std::initializer_list<std::uint32_t> const bitset_keys, std::initializer_list<std::uint32_t> const run_keys ) {
    values out;
    for ( auto const key : array_keys ) {
        for ( std::uint32_t i{ 0 }; i < 1000; i += 17 ) {
            out.emplace_back( ( key << 16 ) + i );
        }
    }
    for ( auto const key : bitset_keys ) {
        for ( std::uint32_t i{ 0 }; i < chunk; i += 2 ) {
            out.emplace_back( ( key << 16 ) + i );
        }
    }
    for ( auto const key : run_keys ) {
        for ( std::uint32_t i{ 0 }; i < 50'000; ++i ) {
            out.emplace_back( ( key << 16 ) + i );
        }
    }
    std::ranges::sort( out );
    return out;
}

[[nodiscard]] std::vector<values> interval_values() {
    std::vector<values> all;
    all.emplace_back( values{ 1, 2, 3 } );
    all.emplace_back( values{ 0xC000'0000U } );
    all.emplace_back( values{ 1, chunk, 0xC000'0000U } );
    all.emplace_back( mixed_containers( { 10 }, { 20 }, { 30 } ) );
    all.emplace_back( mixed_containers( { 0 }, { 1 }, { 20 } ) );
    all.emplace_back( mixed_containers( { 0, 1, 2 }, { 200 }, { 205 } ) );
    auto & top{ all.emplace_back( mixed_containers( { 2 }, { 0 }, { 1 } ) ) };
    for ( std::uint32_t i{ 0xFFFE'0000U }; i != 0; ++i ) {   // the top two chunks, full
        top.emplace_back( i );
    }
    all.emplace_back( values{ 0, 1 << 16, 3 << 16, 0xFFFF'FFFEU, 0xFFFF'FFFFU } );
    return all;
}

constexpr std::uint64_t points[]{
    0, 1, 2, 3, 4, 256, 999, 1000, 0xFFFF, 0x10000, 0x10001, 67'000, 70'000, 140'000, 150'000, 0x1FFFF, 0x20000, 0x30000,
    0x30001, std::uint64_t{ 10 } << 16, ( std::uint64_t{ 20 } << 16 ) + 1, ( std::uint64_t{ 30 } << 16 ) + 50'000,
    ( std::uint64_t{ 200 } << 16 ) + ( 1 << 14 ), std::uint64_t{ 205 } << 16, 0xC000'0000U, 0xC000'0001U, 0xFFFE'FFFFU,
    0xFFFF'0000U, 0xFFFF'0001U, 0xFFFF'FFFEU, max32, universe, universe + 1, universe + 6, universe + max32, max64
};

// tests/toplevel_unit.c: test_intersect_with_range_matches_cardinality, test_contains_range_agrees_with_range_cardinality,
// test_range_cardinality_layouts and test_range_end_beyond_universe: the interval queries over a grid of ends
TYPED_TEST( UpstreamReference, interval_queries_with_ends_past_the_key_space ) {
    using Bitmap = TypeParam;
    for ( auto const & held : interval_values() ) {
        in_both<Bitmap> const both{ held };
        for ( std::size_t i{ 0 }; i < std::size( points ); ++i ) {
            for ( auto j{ i }; j < std::size( points ); ++j ) {
                auto const lo{ points[ i ] };
                auto const hi{ points[ j ] };
                auto const range{ key_range::half_open( lo, hi ) };
                ASSERT_EQ( both.ours.and_cardinality( range ), roaring_bitmap_range_cardinality( both.theirs.r, lo, hi ) ) << lo << ' ' << hi;
                ASSERT_EQ( both.ours.intersects( range ), roaring_bitmap_intersect_with_range( both.theirs.r, lo, hi ) ) << lo << ' ' << hi;
                ASSERT_EQ( both.ours.is_superset_of( range ), roaring_bitmap_contains_range( both.theirs.r, lo, hi ) ) << lo << ' ' << hi;
            }
        }
    }
}

// tests/toplevel_unit.c: test_range_end_beyond_universe and tests/cpp_unit.cpp: test_cpp_range_end_beyond_universe:
// the interval mutations (add_range, remove_range, flip) with ends past 2^32
TYPED_TEST( UpstreamReference, interval_mutations_with_ends_past_the_key_space ) {
    using Bitmap = TypeParam;
    constexpr std::uint64_t mutation_points[]{
        0, 1, 5, 7, max32 - 100'000, 0xFFFF'0000U, 0xFFFF'FFFEU, max32, universe, universe + 1, universe + 5, universe + 10, max64
    };
    for ( auto const & held : { values{}, values{ 0, 7, max32 }, values{ 1, 5, 0xFFFF'0010U, max32 } } ) {
        for ( auto const lo : mutation_points ) {
            for ( auto const hi : mutation_points ) {
                // keep the intervals that reach far below the top short: the bitmaps are compared value by value
                if ( lo < max32 - 100'000 && hi > lo + 1000 ) {
                    continue;
                }
                auto const range{ key_range::half_open( lo, hi ) };
                {
                    in_both<Bitmap> both{ held };
                    both.ours.or_with( range );
                    roaring_bitmap_add_range( both.theirs.r, lo, hi );
                    ASSERT_EQ( values_of( both.ours ), both.theirs.to_values() ) << "add " << lo << ' ' << hi;
                }
                {
                    in_both<Bitmap> both{ held };
                    both.ours.andnot_with( range );
                    roaring_bitmap_remove_range( both.theirs.r, lo, hi );
                    ASSERT_EQ( values_of( both.ours ), both.theirs.to_values() ) << "remove " << lo << ' ' << hi;
                }
                {
                    in_both<Bitmap> both{ held };
                    both.ours.xor_with( range );
                    roaring_bitmap_flip_inplace( both.theirs.r, lo, hi );
                    ASSERT_EQ( values_of( both.ours ), both.theirs.to_values() ) << "flip " << lo << ' ' << hi;
                    EXPECT_EQ( both.ours.validate(), nullptr );
                }
            }
        }
    }
}

// tests/toplevel_unit.c: test_full_range_mask_with_mixed_data (Go TestRoaring_AndMask / TestRoaring_OrThenAnd)
TYPED_TEST( UpstreamReference, test_full_range_mask_with_mixed_data ) {
    using Bitmap = TypeParam;
    std::mt19937 rng{ 123'456'789 };
    values random;
    for ( int i{ 0 }; i < 1'000'000; ++i ) {
        random.emplace_back( 5'000'000 + rng() % 10'000'000 );
    }
    std::ranges::sort( random );
    random.erase( std::ranges::unique( random ).begin(), random.end() );
    values mask_values;
    for ( std::uint32_t i{ 1 }; i <= 10'000'000; ++i ) {
        mask_values.emplace_back( i );
    }
    in_both<Bitmap> const rb{ random };
    in_both<Bitmap> mask{ mask_values };
    croaring const anded{ roaring_bitmap_and( rb.theirs.r, mask.theirs.r ) };
    EXPECT_EQ( values_of( rb.ours & mask.ours ), anded.to_values() );
    mask.ours &= rb.ours;
    EXPECT_EQ( values_of( mask.ours ), anded.to_values() );

    // many runs in one container at key 4272, then a union and an intersection with a range
    values runs;
    constexpr std::uint32_t base{ 4272U << 16 };
    for ( std::uint32_t i{ 0 }; i < 1135; ++i ) {
        for ( std::uint32_t v{ 0 }; v <= i % 20; ++v ) {
            runs.emplace_back( base + i * 57 + v );
        }
    }
    for ( std::uint32_t const extra : { 279'981'785U, 279'982'923U, 279'988'809U, 279'995'913U } ) {
        runs.emplace_back( extra );
    }
    std::ranges::sort( runs );
    runs.erase( std::ranges::unique( runs ).begin(), runs.end() );
    in_both<Bitmap> const many_runs{ runs };
    values range_values;
    for ( std::uint32_t i{ 279'900'001 }; i < 280'000'001; ++i ) {
        range_values.emplace_back( i );
    }
    in_both<Bitmap> const range{ range_values };
    croaring const runs_and{ roaring_bitmap_and( many_runs.theirs.r, range.theirs.r ) };
    croaring const runs_or{ roaring_bitmap_or( many_runs.theirs.r, range.theirs.r ) };
    EXPECT_EQ( values_of( many_runs.ours & range.ours ), runs_and.to_values() );
    EXPECT_EQ( values_of( many_runs.ours | range.ours ), runs_or.to_values() );
    auto in_place{ range.ours };
    in_place &= many_runs.ours;
    EXPECT_EQ( values_of( in_place ), runs_and.to_values() );
}

// tests/toplevel_unit.c: test_issue330_wide_or_with_runs and test_issue178_sparse_or_many: the wide union of
// run-optimized bitmaps whose chunks interleave, against roaring_bitmap_or_many
TYPED_TEST( UpstreamReference, wide_union_of_interleaved_run_bitmaps ) {
    using Bitmap = TypeParam;
    std::mt19937 rng{ 330 };
    for ( std::uint32_t const spread : { 1U << 19, 1U << 30 } ) {
        std::deque<in_both<Bitmap>> inputs;
        for ( int b{ 0 }; b < 3; ++b ) {
            values held;
            for ( int i{ 0 }; i < 500; ++i ) {
                auto const start{ 1'000'000U + rng() % spread };
                auto const length{ rng() % 4 };
                for ( std::uint32_t v{ 0 }; v < length; ++v ) {
                    held.emplace_back( start + v );
                }
            }
            std::ranges::sort( held );
            held.erase( std::ranges::unique( held ).begin(), held.end() );
            inputs.emplace_back( held );
        }
        std::vector<Bitmap const *> ours;
        std::vector<roaring_bitmap_t const *> theirs;
        for ( auto const & input : inputs ) {
            ours.emplace_back( &input.ours );
            theirs.emplace_back( input.theirs.r );
        }
        croaring const expected{ roaring_bitmap_or_many( theirs.size(), theirs.data() ) };
        Bitmap wide;
        wide.or_many_in_place( ours );
        EXPECT_EQ( values_of( wide ), expected.to_values() );
        Bitmap heap;
        heap.or_many_heap_in_place( ours );
        EXPECT_EQ( values_of( heap ), expected.to_values() );
        EXPECT_EQ( inputs[ 0 ].ours.or_cardinality( inputs[ 1 ].ours ), roaring_bitmap_or_cardinality( inputs[ 0 ].theirs.r, inputs[ 1 ].theirs.r ) );
    }
}

} // namespace
