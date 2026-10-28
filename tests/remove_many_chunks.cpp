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
#include <type_traits>
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

TYPED_TEST( RemoveManyChunks, values_of_several_chunks_in_any_order ) {
    for ( auto const run_optimized : { false, true } ) {
        auto const values{ closed_range( 0, 2 * chunk_size - 1 ) };
        values_t const doomed{ chunk_size + 7, chunk_size + 5, 7, 5, 1, chunk_size + 1, chunk_size + 3, 3 };
        values_t sorted{ doomed };
        std::ranges::sort( sorted );

        auto bitmap{ TestFixture::from( values, run_optimized ) };
        bitmap.remove_many( doomed );
        EXPECT_EQ( bitmap.validate(), nullptr );
        EXPECT_TRUE( std::ranges::equal( bitmap.to_vector(), without( values, sorted ) ) ) << "run_optimized " << run_optimized;
        EXPECT_EQ( bitmap.size(), values.size() - doomed.size() );
    }
}

TYPED_TEST( RemoveManyChunks, no_values_and_values_that_are_not_there_change_nothing ) {
    auto const values{ closed_range( 500, 500 ) };
    auto bitmap{ TestFixture::from( values, false ) };
    bitmap.remove_many( values_t{} );
    bitmap.remove_many_sorted( {} );
    bitmap.remove_many( values_t{ 501, 80'000 } );
    EXPECT_EQ( bitmap.validate(), nullptr );
    EXPECT_TRUE( std::ranges::equal( bitmap.to_vector(), values ) );
}

// One value per call from a full chunk to nothing: the container changes kind on the way down (a bitset or a run
// becomes an array) and is dropped with its last value.
TYPED_TEST( RemoveManyChunks, one_call_per_value_empties_a_full_chunk ) {
    for ( auto const run_optimized : { false, true } ) {
        auto bitmap{ TestFixture::from( closed_range( 0, chunk_size - 1U ), run_optimized ) };
        for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
            bitmap.remove_many( values_t{ value } );
            ASSERT_EQ( bitmap.size(), chunk_size - 1U - value ) << "after removing " << value;
            if ( value % 4'099U == 0 ) {
                ASSERT_EQ( bitmap.validate(), nullptr ) << "after removing " << value;
            }
        }
        EXPECT_TRUE( bitmap.empty() );
        EXPECT_EQ( bitmap.statistics().container_count(), 0U );
    }
}

TYPED_TEST( RemoveManyChunks, a_value_repeated_in_the_sorted_batch_is_removed_once ) {
    values_t bitset_values;
    for ( std::uint32_t value{ 0 }; value < chunk_size; value += 3U ) {
        bitset_values.push_back( value );
    }
    values_t const array_values{ 4, 5, 6, 40, 90 };
    values_t const run_values{ closed_range( 100, 30'000 ) };
    values_t const doomed{ 6, 6, 6, 9, 9, 40, 40, 99, 100, 100, 101, 102, 102, 102, 30'000, 30'000 };
    values_t distinct{ doomed };
    distinct.erase( std::unique( distinct.begin(), distinct.end() ), distinct.end() );
    for ( auto const & values : { bitset_values, array_values, run_values } ) {
        for ( auto const run_optimized : { false, true } ) {
            auto const expected{ without( values, distinct ) };
            auto bitmap{ TestFixture::from( values, run_optimized ) };
            bitmap.remove_many_sorted( { doomed.data(), doomed.size() } );
            EXPECT_EQ( bitmap.validate(), nullptr );
            EXPECT_EQ( bitmap.size(), expected.size() );
            EXPECT_TRUE( std::ranges::equal( bitmap.to_vector(), expected ) );
        }
    }
}

// A bitmap in the middle of a lazy fold holds bitsets with a stale cardinality; values are removed from it correctly.
TYPED_TEST( RemoveManyChunks, a_bitmap_in_the_middle_of_a_fold_loses_values_correctly ) {
    if constexpr ( TestFixture::has_bitsets ) {
        values_t first;
        values_t second;
        for ( std::uint32_t value{ 0 }; value + 1U < chunk_size; value += 3U ) {
            first.push_back( value );
            second.push_back( value + 1U );
        }
        values_t both{ first };
        both.insert( both.end(), second.begin(), second.end() );
        std::ranges::sort( both );

        values_t doomed;
        for ( std::uint32_t value{ 1 }; value < chunk_size; value += 5U ) {
            doomed.push_back( value );
        }
        for ( auto const sorted_entry : { true, false } ) {
            auto deferred{ TestFixture::from( first, false ) };
            deferred.bulk_or_intermediate( TestFixture::from( second, false ) );
            if ( sorted_entry ) {
                deferred.remove_many_sorted( { doomed.data(), doomed.size() } );
            } else {
                deferred.remove_many( doomed );
            }
            deferred.bulk_or_finish();
            auto const expected{ without( both, doomed ) };
            EXPECT_EQ( deferred.validate(), nullptr );
            EXPECT_EQ( deferred.size(), expected.size() );
            EXPECT_TRUE( std::ranges::equal( deferred.to_vector(), expected ) );
        }
    }
}

// ---- copy on write -------------------------------------------------------------------------------------------
// The mutator must unshare a chunk before editing it, or it edits the chunk that the copies still own.

namespace cow_case {
constexpr std::uint32_t dense_end{ 1U << 17 };
constexpr std::uint32_t sparse_value{ 5U << 16 };

[[nodiscard]] values_t source_values() {
    auto result{ closed_range( 0, dense_end - 1U ) };
    result.push_back( sparse_value );
    return result;
}

[[nodiscard]] values_t several_chunks() { return { 1, 3, 5, 7, chunk_size + 1U, chunk_size + 3U, sparse_value }; }
[[nodiscard]] values_t one_chunk      () { return { 1, 2, 3 }; }
[[nodiscard]] values_t empties_chunk  () { return { sparse_value }; }
}  // namespace cow_case

TYPED_TEST( RemoveManyChunks, copies_keep_their_values_when_one_removes_many ) {
    auto const values{ cow_case::source_values() };
    for ( auto const run_optimized : { false, true } ) {
        for ( auto const & doomed : { cow_case::several_chunks(), cow_case::one_chunk(), cow_case::empties_chunk() } ) {
            TestFixture::expect_removal( values, run_optimized, doomed, without( values, doomed ), "copy on write" );
        }
    }
}

TYPED_TEST( RemoveManyChunks, four_copies_each_remove_without_touching_the_others ) {
    auto const values{ cow_case::source_values() };
    auto const source{ TestFixture::from( values, false ) };
    std::vector<std::remove_const_t<decltype( source )>> copies( 4, source );
    auto const doomed{ cow_case::several_chunks() };
    for ( std::size_t i{ 0 }; i < copies.size(); ++i ) {
        copies[ i ].remove_many( doomed );
        EXPECT_EQ( copies[ i ].validate(), nullptr );
        EXPECT_TRUE( std::ranges::equal( copies[ i ].to_vector(), without( values, doomed ) ) );
        EXPECT_TRUE( std::ranges::equal( source.to_vector(), values ) );
        for ( auto j{ i + 1 }; j < copies.size(); ++j ) {
            EXPECT_TRUE( std::ranges::equal( copies[ j ].to_vector(), values ) );
        }
    }
}

// ---- 64-bit keys ---------------------------------------------------------------------------------------------

TEST( RemoveMany64, removes_across_chunks_until_empty ) {
    bitmap<std::uint64_t> bitmap;
    std::vector<std::uint64_t> values;
    for ( std::uint64_t value{ 0 }; value < 1'000; ++value ) {
        values.push_back( value );
        bitmap.add( value );
    }
    bitmap.remove_many( values );
    EXPECT_TRUE( bitmap.empty() );
    EXPECT_EQ( bitmap.validate(), nullptr );
}

TEST( RemoveMany64, values_that_are_not_there ) {
    bitmap<std::uint64_t> bitmap;
    bitmap.add( 123 );
    bitmap.add( 124 );
    bitmap.remove_many( std::vector<std::uint64_t>{ 125 } );
    EXPECT_EQ( bitmap.size(), 2U );
    bitmap.remove_many( std::vector<std::uint64_t>{ 123, 124, 125 } );
    EXPECT_TRUE( bitmap.empty() );
    EXPECT_EQ( bitmap.validate(), nullptr );
}

TEST( RemoveMany64, a_chunk_emptied_by_its_first_value_and_a_value_repeated ) {
    bitmap<std::uint64_t> bitmap;
    bitmap.add( 0 );
    bitmap.add( std::uint64_t{ 1 } << 20 );
    bitmap.remove_many( std::vector<std::uint64_t>{ 0, 0 } );
    EXPECT_EQ( bitmap.validate(), nullptr );
    EXPECT_EQ( bitmap.size(), 1U );
    EXPECT_TRUE( bitmap.contains( std::uint64_t{ 1 } << 20 ) );

    decltype( bitmap ) other;
    other.add( 123 );
    other.add( 124 );
    other.remove_many( std::vector<std::uint64_t>{ 123, 124, 124 } );
    EXPECT_TRUE( other.empty() );
    EXPECT_EQ( other.validate(), nullptr );
}

// ---- the group removal on a bare container -------------------------------------------------------------------

TEST( RemoveManyContainers, a_bitset_with_an_unknown_cardinality_is_counted_before_its_bits_change ) {
    using layout_t    = detail::default_layout<std::uint32_t>;
    using container_t = detail::container_handle<layout_t, detail::cow_value_semantics>;

    auto container{ container_t::make_bitset_uninitialized() };
    auto & words{ container.as_bitset().words.as_array() };
    words.fill( 0 );
    std::uint32_t count{ 0 };
    for ( std::uint32_t value{ 10 }; value < chunk_size - 10U; value += 3U ) {
        words[ value / 64U ] |= std::uint64_t{ 1 } << ( value % 64U );
        ++count;
    }
    container.set_cardinality( count );
    container.as_bitset().mark_cardinality_unknown();

    // 13 and 16 are present, 14 and 15 are not; the endpoints (10 and the largest) stay
    std::vector<std::uint16_t> const doomed{ 13, 14, 15, 16 };
    auto const removed{ detail::container_remove_sorted<layout_t, detail::cow_value_semantics>( container, { doomed.data(), doomed.size() }, 4'096 ) };
    EXPECT_EQ( removed, 2U );
    EXPECT_EQ( detail::container_size<layout_t>( container ), count - 2U );
    EXPECT_EQ( container.min_value(), 10U );

    // the largest value goes: the endpoint moves
    std::vector<std::uint16_t> const largest{ container.max_value() };
    auto const second{ detail::container_remove_sorted<layout_t, detail::cow_value_semantics>( container, { largest.data(), largest.size() }, 4'096 ) };
    EXPECT_EQ( second, 1U );
    EXPECT_LT( container.max_value(), largest.front() );
    EXPECT_EQ( container.max_value() % 3U, 1U );
}

}  // namespace
