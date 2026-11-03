// optimize() and optimize_keep_bitsets() settle every chunk in the form CRoaring's run_optimize picks, decided from
// the chunk's cardinality and run count by serialized size (runs: 2 + 4 per run, array: 2 per value, bitset: its
// words), the held kind keeping a tie. A chunk that keeps its kind is left as it is: nothing is allocated and a
// copy that shares it still does. The cases follow CRoaring's own run_optimize tests (array to run, array to
// itself, bitset to itself, bitset to run, run to itself), plus the size boundaries of each rule.

#include <frsr/roaring/bitmap.hpp>

#include "allocation_fault.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring;

using values_t = std::vector<std::uint32_t>;

constexpr std::uint32_t chunk{ 65'536U };

template <typename Bitmap>
class OptimizeStorage : public ::testing::Test {};

// The lazy policy builds arrays and bitsets only, so every run container below comes from the optimize call.
using OptimizeStorageBitmaps = ::testing::Types<
    bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>,
    bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_value_semantics, detail::run_selection_lazy>,
    bitmap<std::uint64_t, default_container_set<std::uint64_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>
>;
TYPED_TEST_SUITE( OptimizeStorage, OptimizeStorageBitmaps );

// `runs` runs of `length` values each, `gap` absent values apart, from `first` on.
[[nodiscard]] values_t runs_of( std::uint32_t const first, std::uint32_t const runs, std::uint32_t const length, std::uint32_t const gap ) {
    values_t values;
    for ( std::uint32_t run{ 0U }; run < runs; ++run ) {
        auto const begin{ first + run * ( length + gap ) };
        for ( std::uint32_t offset{ 0U }; offset < length; ++offset ) {
            values.push_back( begin + offset );
        }
    }
    return values;
}

// [first, last) but `hole`: two runs.
[[nodiscard]] values_t range_but( std::uint32_t const first, std::uint32_t const last, std::uint32_t const hole ) {
    values_t values;
    for ( auto value{ first }; value < last; ++value ) {
        if ( value != hole ) {
            values.push_back( value );
        }
    }
    return values;
}

template <typename Bitmap>
[[nodiscard]] Bitmap make( values_t const & values ) {
    Bitmap result;
    for ( auto const value : values ) {
        result.add( value );
    }
    return result;
}

template <typename Bitmap>
[[nodiscard]] values_t contents( Bitmap const & bitmap ) {
    values_t result;
    for ( auto const value : bitmap ) {
        result.push_back( static_cast<std::uint32_t>( value ) );
    }
    return result;
}

enum class kind { array, run, bitset };

template <typename Bitmap>
[[nodiscard]] kind only_kind( Bitmap const & bitmap ) {
    auto const statistics{ bitmap.statistics() };
    EXPECT_EQ( statistics.container_count(), 1U );
    if ( statistics.run_containers != 0 ) {
        return kind::run;
    }
    return statistics.bitset_containers != 0 ? kind::bitset : kind::array;
}

// The allocations the calling thread makes while `f` runs (0 when the binary does not count them).
template <typename F>
[[nodiscard]] std::uint32_t allocations_during( F && f ) {
    namespace fault = frsr::roaring::kernel_test::alloc_fault;
    fault::armed const armed{ 0x7fff'ffffU };
    auto const before{ fault::counter };
    f();
    return fault::counter - before;
}

// Both optimize calls, each on its own copy: the kind they leave and whether the chunk was left as it was.
template <typename Bitmap>
void expect_settles( values_t const & values, kind const before, kind const after ) {
    auto const source{ make<Bitmap>( values ) };
    ASSERT_EQ( only_kind( source ), before );
    for ( auto const keep_bitsets : { false, true } ) {
        auto copy{ source };
        auto const allocations{ allocations_during( [ & ] {
            if ( keep_bitsets ) {
                copy.optimize_keep_bitsets();
            } else {
                copy.optimize();
            }
        } ) };
        EXPECT_EQ( only_kind( copy ), after ) << "keep_bitsets=" << keep_bitsets;
        EXPECT_EQ( contents( copy ), values );
        EXPECT_EQ( contents( source ), values );
        EXPECT_EQ( only_kind( source ), before );
        EXPECT_EQ( copy.validate(), nullptr );
        if ( before == after ) {
            EXPECT_EQ( allocations, 0U ) << "keep_bitsets=" << keep_bitsets;
        }
    }
}

TYPED_TEST( OptimizeStorage, ArrayToRun ) {
    expect_settles<TypeParam>( range_but( chunk, chunk + 500U, chunk + 300U ), kind::array, kind::run );
}

TYPED_TEST( OptimizeStorage, ArrayToSelf ) {
    expect_settles<TypeParam>( runs_of( chunk, 250U, 1U, 1U ), kind::array, kind::array );
}

TYPED_TEST( OptimizeStorage, BitsetToSelf ) {
    expect_settles<TypeParam>( runs_of( chunk, 25'000U, 1U, 1U ), kind::bitset, kind::bitset );
}

TYPED_TEST( OptimizeStorage, BitsetToRun ) {
    expect_settles<TypeParam>( range_but( chunk, chunk + 50'000U, chunk + 300U ), kind::bitset, kind::run );
}

TYPED_TEST( OptimizeStorage, RunToSelf ) {
    auto const values{ range_but( chunk, chunk + 50'000U, chunk + 300U ) };
    auto source{ make<TypeParam>( values ) };
    source.optimize_keep_bitsets();
    ASSERT_EQ( only_kind( source ), kind::run );
    for ( auto const keep_bitsets : { false, true } ) {
        auto copy{ source };
        auto const allocations{ allocations_during( [ & ] {
            if ( keep_bitsets ) {
                copy.optimize_keep_bitsets();
            } else {
                copy.optimize();
            }
        } ) };
        EXPECT_EQ( only_kind( copy ), kind::run );
        EXPECT_EQ( contents( copy ), values );
        EXPECT_EQ( allocations, 0U ) << "keep_bitsets=" << keep_bitsets;
    }
}

// A handful of values in one run is smaller as runs, however few they are.
TYPED_TEST( OptimizeStorage, FewValuesInRunsBecomeRuns ) {
    expect_settles<TypeParam>( runs_of( chunk, 1U, 10U, 0U ), kind::array, kind::run );
    expect_settles<TypeParam>( runs_of( chunk, 3U, 2U, 5U ), kind::array, kind::array );
}

// An array of 2n + 1 values in n runs serializes to as many bytes as the runs (4n + 2): the array keeps the tie.
// One value more and the runs are smaller, one fewer and the array is.
TYPED_TEST( OptimizeStorage, ArrayRunBoundary ) {
    constexpr std::uint32_t runs{ 100U };
    auto tie{ runs_of( chunk, runs - 1U, 2U, 1U ) };
    for ( auto const value : runs_of( chunk + ( runs - 1U ) * 3U, 1U, 3U, 0U ) ) {
        tie.push_back( value );
    }
    ASSERT_EQ( tie.size(), 2U * runs + 1U );
    expect_settles<TypeParam>( tie, kind::array, kind::array );

    auto smaller_as_runs{ tie };
    smaller_as_runs.push_back( tie.back() + 1U );
    expect_settles<TypeParam>( smaller_as_runs, kind::array, kind::run );

    auto smaller_as_array{ runs_of( chunk, runs, 2U, 1U ) };
    expect_settles<TypeParam>( smaller_as_array, kind::array, kind::array );
}

// A bitset (8192 bytes) becomes runs below 2048 of them: 2047 runs serialize to 8190 bytes, 2048 to 8194.
TYPED_TEST( OptimizeStorage, BitsetRunBoundary ) {
    expect_settles<TypeParam>( runs_of( chunk, 2'047U, 3U, 1U ), kind::bitset, kind::run );
    expect_settles<TypeParam>( runs_of( chunk, 2'048U, 3U, 1U ), kind::bitset, kind::bitset );
}

// Runs that end the chunk: the last run reaches the domain's top value.
TYPED_TEST( OptimizeStorage, RunsToTheEndOfTheChunk ) {
    auto values{ runs_of( chunk, 100U, 7U, 9U ) };
    for ( auto const value : runs_of( 2U * chunk - 5'000U, 1U, 5'000U, 0U ) ) {
        values.push_back( value );
    }
    expect_settles<TypeParam>( values, kind::bitset, kind::run );
}

// The selective run pass and the full one agree: a chunk the first turned into runs is left alone by the second.
TYPED_TEST( OptimizeStorage, RunShapedPassThenOptimizeLeavesTheChunk ) {
    auto const values{ runs_of( chunk, 1U, 40U, 0U ) };
    auto bitmap{ make<TypeParam>( values ) };
    bitmap.optimize_run_shaped( 32U );
    ASSERT_EQ( only_kind( bitmap ), kind::run );
    auto const allocations{ allocations_during( [ & ] { bitmap.optimize_keep_bitsets(); } ) };
    EXPECT_EQ( only_kind( bitmap ), kind::run );
    EXPECT_EQ( allocations, 0U );
    EXPECT_EQ( contents( bitmap ), values );
}

// Every kind at once, each chunk left as it is or converted on its own; a second call changes nothing.
TYPED_TEST( OptimizeStorage, MixedChunksSettleOnceThenStay ) {
    values_t values;
    for ( auto const & part : {
        range_but( 0U, 500U, 300U ), // array -> runs
        runs_of( 1U * chunk, 250U, 1U, 1U ), // array
        runs_of( 2U * chunk, 25'000U, 1U, 1U ), // bitset
        range_but( 3U * chunk, 3U * chunk + 50'000U, 3U * chunk + 7U ), // bitset -> runs
    } ) {
        values.insert( values.end(), part.begin(), part.end() );
    }
    auto bitmap{ make<TypeParam>( values ) };
    bitmap.optimize_keep_bitsets();
    auto const once{ bitmap.statistics() };
    EXPECT_EQ( once.run_containers, 2U );
    EXPECT_EQ( once.array_containers, 1U );
    EXPECT_EQ( once.bitset_containers, 1U );
    EXPECT_EQ( contents( bitmap ), values );

    auto copy{ bitmap };
    auto const allocations{ allocations_during( [ & ] { copy.optimize_keep_bitsets(); } ) };
    EXPECT_EQ( allocations, 0U );
    auto const twice{ copy.statistics() };
    EXPECT_EQ( twice.run_containers, once.run_containers );
    EXPECT_EQ( twice.array_containers, once.array_containers );
    EXPECT_EQ( twice.bitset_containers, once.bitset_containers );
    EXPECT_EQ( contents( copy ), values );
}

} // namespace
