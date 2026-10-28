// remove_many_sorted over chunks of each kind: a run chunk is cut by one merge pass over its runs and the sorted
// values, a bitset chunk has its bits cleared in one pass, an array chunk is compacted in one pass. Every result
// takes the smallest form, a value that is absent or lies in a chunk that is absent is ignored, a chunk that loses
// everything is dropped, and a copy that shares the chunk keeps all of its values. Contents are checked against
// plain sorted value lists.

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/container_ops.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

namespace {

using namespace frsr::roaring;

using eager_bitmap       = bitmap<std::uint32_t>;
using lazy_bitmap        = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_value_semantics, detail::run_selection_lazy>;
using atomic_bitmap      = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using atomic_lazy_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>;
using array_run_set      = std::variant<array_container<std::uint32_t>, run_container<std::uint32_t>>;
using array_run_bitmap   = bitmap<std::uint32_t, array_run_set>;

using values_t = std::vector<std::uint32_t>;

constexpr std::uint32_t chunk_size{ 65'536 };

[[nodiscard]] values_t closed_range( std::uint32_t const first, std::uint32_t const last ) {
    values_t result;
    for ( auto value{ first }; value <= last; ++value ) {
        result.push_back( value );
    }
    return result;
}

[[nodiscard]] values_t without( values_t const & values, values_t const & taken ) {
    values_t result;
    std::ranges::set_difference( values, taken, std::back_inserter( result ) );
    return result;
}

template <typename Bitmap>
class RemoveManyChunks : public ::testing::Test {
protected:
    static constexpr bool has_bitsets{ !std::same_as<Bitmap, array_run_bitmap> };

    [[nodiscard]] static Bitmap from( values_t const & values, bool const run_optimized ) {
        Bitmap result;
        result.add_many_sorted( { values.data(), values.size() } );
        if ( run_optimized ) {
            result.optimize();
        }
        return result;
    }

    [[nodiscard]] static Bitmap removed( values_t const & values, bool const run_optimized, values_t const & doomed ) {
        auto result{ from( values, run_optimized ) };
        result.remove_many_sorted( { doomed.data(), doomed.size() } );
        return result;
    }

    // Removes `doomed` from a copy that shares the chunks and from a sole holder; both must hold exactly `expected`,
    // and the original must keep every value.
    static void expect_removal( values_t const & values, bool const run_optimized, values_t const & doomed, values_t const & expected, char const * const what ) {
        auto const original{ from( values, run_optimized ) };
        auto shared{ original };
        shared.remove_many_sorted( { doomed.data(), doomed.size() } );
        EXPECT_EQ( shared.validate(), nullptr ) << what;
        EXPECT_TRUE( std::ranges::equal( shared.to_vector(), expected ) ) << what << ": shared copy";
        EXPECT_EQ( shared.size(), expected.size() ) << what;
        EXPECT_TRUE( std::ranges::equal( original.to_vector(), values ) ) << what << ": the other holder";

        auto const sole{ removed( values, run_optimized, doomed ) };
        EXPECT_EQ( sole.validate(), nullptr ) << what;
        EXPECT_TRUE( std::ranges::equal( sole.to_vector(), expected ) ) << what << ": sole holder";
    }
};

using policies = ::testing::Types<eager_bitmap, lazy_bitmap, atomic_bitmap, atomic_lazy_bitmap, array_run_bitmap>;
TYPED_TEST_SUITE( RemoveManyChunks, policies );

TYPED_TEST( RemoveManyChunks, a_run_chunk_stays_runs_while_runs_are_its_smallest_form ) {
    auto values{ closed_range( 100, 30'000 ) };
    for ( auto const value : closed_range( 40'000, 65'535 ) ) {
        values.push_back( value );
    }
    for ( auto const value : closed_range( chunk_size + 5, chunk_size + 50'000 ) ) {
        values.push_back( value );
    }

    values_t const doomed{ 100, 101, 7'000, 7'001, 7'002, 30'000, 65'535, chunk_size + 5, chunk_size + 20'000, chunk_size + 50'000, chunk_size + 60'000 };
    TestFixture::expect_removal( values, true, doomed, without( values, doomed ), "run chunks with holes" );

    auto const stats{ TestFixture::removed( values, true, doomed ).statistics() };
    EXPECT_EQ( stats.run_containers, 2U );
    EXPECT_EQ( stats.container_count(), 2U );
}

TYPED_TEST( RemoveManyChunks, a_run_chunk_that_is_cut_to_its_ends_falls_to_an_array ) {
    auto const values{ closed_range( 0, 59'999 ) };
    auto const doomed{ closed_range( 1, 59'998 ) };
    TestFixture::expect_removal( values, true, doomed, { 0, 59'999 }, "only the ends left" );
    EXPECT_EQ( TestFixture::removed( values, true, doomed ).statistics().array_containers, 1U );
}

TYPED_TEST( RemoveManyChunks, a_run_chunk_that_loses_every_second_value_is_not_runs ) {
    auto const values{ closed_range( 0, 59'999 ) };
    values_t doomed;
    for ( std::uint32_t value{ 0 }; value < 59'999; value += 2U ) {
        doomed.push_back( value );
    }
    TestFixture::expect_removal( values, true, doomed, without( values, doomed ), "alternating" );
    EXPECT_EQ( TestFixture::removed( values, true, doomed ).statistics().run_containers, 0U );
}

TYPED_TEST( RemoveManyChunks, values_outside_a_run_chunk_and_absent_chunks_are_ignored ) {
    auto const values{ closed_range( 1'000, 2'000 ) };
    values_t const doomed{ 0, 999, 2'001, 65'535, chunk_size, 5 * chunk_size + 7 };
    TestFixture::expect_removal( values, true, doomed, values, "nothing present" );
    values_t const edge{ 1'000, 2'000, 2'001 };
    TestFixture::expect_removal( values, true, edge, closed_range( 1'001, 1'999 ), "both ends of the run" );
}

TYPED_TEST( RemoveManyChunks, a_chunk_that_loses_everything_is_dropped ) {
    for ( auto const run_optimized : { false, true } ) {
        auto const values{ closed_range( 10, 5'000 ) };
        TestFixture::expect_removal( values, run_optimized, values, {}, "all of it" );
        auto const emptied{ TestFixture::removed( values, run_optimized, values ) };
        EXPECT_TRUE( emptied.empty() );
        EXPECT_EQ( emptied.statistics().container_count(), 0U );
    }
}

TYPED_TEST( RemoveManyChunks, the_last_value_of_a_chunk_can_be_removed_from_a_run_ending_there ) {
    auto const values{ closed_range( 65'000, 65'535 ) };
    values_t const doomed{ 65'535 };
    TestFixture::expect_removal( values, true, doomed, closed_range( 65'000, 65'534 ), "run to the chunk end" );
}

TYPED_TEST( RemoveManyChunks, a_bitset_chunk_clears_its_bits_in_one_pass ) {
    if constexpr ( TestFixture::has_bitsets ) {
        values_t values;
        for ( std::uint32_t value{ 0 }; value < chunk_size; value += 3U ) {
            values.push_back( value );
        }
        values_t doomed;
        for ( std::uint32_t value{ 0 }; value < chunk_size; value += 6U ) {
            doomed.push_back( value );
        }
        doomed.push_back( 65'534 );  // absent
        TestFixture::expect_removal( values, false, doomed, without( values, doomed ), "a bitset losing half" );
        auto const half{ TestFixture::removed( values, false, doomed ).statistics() };
        EXPECT_EQ( half.bitset_containers, 1U );
        EXPECT_EQ( half.container_count(), 1U );

        // a bitset that falls below the array limit becomes an array
        auto const taken{ closed_range( 0, 65'000 ) };
        TestFixture::expect_removal( values, false, taken, without( values, taken ), "a bitset left with a few values" );
        EXPECT_EQ( TestFixture::removed( values, false, taken ).statistics().array_containers, 1U );
    }
}

TYPED_TEST( RemoveManyChunks, an_array_chunk_is_compacted ) {
    values_t values;
    for ( std::uint32_t value{ 0 }; value < 2'000; value += 7U ) {
        values.push_back( value * 11U );
    }
    values_t doomed{ values[ 0 ], values[ 10 ], values[ 11 ], values.back(), values.back() + 1U };
    std::ranges::sort( doomed );
    TestFixture::expect_removal( values, false, doomed, without( values, doomed ), "array" );
}

TYPED_TEST( RemoveManyChunks, one_call_cuts_chunks_of_every_kind ) {
    values_t values;
    for ( auto const value : closed_range( 0, 59'999 ) ) {                  // chunk 0: a run
        values.push_back( value );
    }
    for ( std::uint32_t value{ 0 }; value < chunk_size; value += 2U ) {     // chunk 1: scattered values, a bitset
        values.push_back( chunk_size + value );
    }
    for ( std::uint32_t value{ 0 }; value < 100; ++value ) {                // chunk 2: an array
        values.push_back( 2 * chunk_size + value * 13U );
    }
    values_t doomed;
    for ( std::uint32_t chunk{ 0 }; chunk < 4; ++chunk ) {
        for ( std::uint32_t low{ 6 }; low < 60'000; low += 997U ) {
            doomed.push_back( chunk * chunk_size + low );
        }
    }
    TestFixture::expect_removal( values, true, doomed, without( values, doomed ), "all kinds in one call" );
}

}  // namespace
