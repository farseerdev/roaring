// A subtraction from a run-encoded chunk (`-=`, `-`, `andnot_with`, `remove_closed_range`, `remove_many_sorted`)
// leaves the chunk's runs as runs while they are the smallest form, falls to an array when only a few values
// remain and to a bitset when the survivors are scattered. The shape is the same for every operand kind and every
// container policy, and a copy that shares the receiver's runs keeps them. Contents are checked against a plain
// membership table, never against a second implementation.

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/container_ops.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace frsr::roaring;

using eager_bitmap       = bitmap<std::uint32_t>;
using lazy_bitmap        = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_value_semantics, detail::run_selection_lazy>;
using atomic_bitmap      = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using atomic_lazy_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>;

// A container set without bitsets: a run result that would be a bitset must come out as an array.
using array_run_set    = std::variant<array_container<std::uint32_t>, run_container<std::uint32_t>>;
using array_run_bitmap = bitmap<std::uint32_t, array_run_set>;

enum class shape : std::uint8_t { empty, array, run, bitset };

constexpr std::uint32_t chunk_size{ 65'536 };
constexpr std::uint32_t run_end   { 59'999 };  // the receiver is the single run [0, run_end]

using members = std::vector<bool>;  // one chunk's membership table

[[nodiscard]] members receiver_members() {
    members result( chunk_size );
    for ( std::uint32_t value{ 0 }; value <= run_end; ++value ) {
        result[ value ] = true;
    }
    return result;
}

[[nodiscard]] members without( members result, members const & taken ) {
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        if ( taken[ value ] ) {
            result[ value ] = false;
        }
    }
    return result;
}

[[nodiscard]] std::vector<std::uint32_t> values_of( members const & table ) {
    std::vector<std::uint32_t> result;
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        if ( table[ value ] ) {
            result.push_back( value );
        }
    }
    return result;
}

[[nodiscard]] members members_of( std::vector<std::uint32_t> const & values ) {
    members result( chunk_size );
    for ( auto const value : values ) {
        result[ value ] = true;
    }
    return result;
}

[[nodiscard]] members closed_range_members( std::uint32_t const first, std::uint32_t const last ) {
    members result( chunk_size );
    for ( auto value{ first }; value <= last; ++value ) {
        result[ value ] = true;
    }
    return result;
}

// Nine values spread over the receiver: runs of them would take more bytes than the values themselves.
[[nodiscard]] std::vector<std::uint32_t> scattered_nine() {
    return { 3, 12, 13, 70, 71, 4'000, 4'001, 59'000, 59'999 };
}

// Every receiver value except `survivors`.
[[nodiscard]] std::vector<std::uint32_t> receiver_values_but( members const & survivors ) {
    std::vector<std::uint32_t> result;
    for ( std::uint32_t value{ 0 }; value <= run_end; ++value ) {
        if ( !survivors[ value ] ) {
            result.push_back( value );
        }
    }
    return result;
}

// The operand that leaves exactly `survivors` of the receiver: every other receiver value, plus every second value
// above the receiver so that its runs (thousands of them) are never smaller than a bitset and the operand is one.
[[nodiscard]] members bitset_operand_leaving( members const & survivors ) {
    members result( chunk_size );
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        result[ value ] = value <= run_end ? !survivors[ value ] : value % 2U == 0U;
    }
    return result;
}

template <typename Bitmap>
class AndnotKeepsRuns : public ::testing::Test {
protected:
    [[nodiscard]] static Bitmap run_receiver() {
        Bitmap result;
        result.add_closed_range( 0, run_end );
        result.optimize();
        return result;
    }

    [[nodiscard]] static Bitmap from_members( members const & table, bool const as_runs = false ) {
        Bitmap result;
        auto const values{ values_of( table ) };
        result.add_many_sorted( { values.data(), values.size() } );
        if ( as_runs ) {
            result.optimize();
        }
        return result;
    }

    [[nodiscard]] static shape shape_of( Bitmap const & bm ) {
        auto const stats{ bm.statistics() };
        EXPECT_LE( stats.container_count(), 1U );
        if ( stats.bitset_containers != 0 ) {
            return shape::bitset;
        }
        if ( stats.run_containers != 0 ) {
            return shape::run;
        }
        return stats.array_containers != 0 ? shape::array : shape::empty;
    }

    static void expect_result( Bitmap const & actual, members const & expected, shape const expected_shape, char const * const what ) {
        EXPECT_EQ( actual.validate(), nullptr ) << what;
        EXPECT_EQ( shape_of( actual ), expected_shape ) << what;
        auto const values{ values_of( expected ) };
        auto const actual_values{ actual.to_vector() };
        EXPECT_TRUE( std::ranges::equal( actual_values, values ) ) << what << ": contents differ, " << actual_values.size() << " vs " << values.size();
        EXPECT_EQ( actual.size(), values.size() ) << what;
    }

    // The receiver minus `operand` through `-=`, `-` and `andnot_with`, on a copy that shares the receiver's runs
    // (which must keep them) and on a receiver that is the sole holder.
    static void expect_subtraction( Bitmap const & operand, shape const expected_shape, char const * const what ) {
        std::vector<std::uint32_t> operand_values;
        for ( auto const value : operand.to_vector() ) {
            operand_values.push_back( value );
        }
        auto const operand_members{ members_of( operand_values ) };
        auto const expected{ without( receiver_members(), operand_members ) };

        auto const receiver{ run_receiver() };
        ASSERT_EQ( shape_of( receiver ), shape::run ) << what;

        {
            auto shared{ receiver };
            shared -= operand;
            expect_result( shared, expected, expected_shape, what );
            EXPECT_EQ( shape_of( receiver ), shape::run ) << what << ": the other holder of the runs";
            EXPECT_EQ( receiver.size(), run_end + 1U ) << what;
        }
        {
            auto sole{ run_receiver() };
            sole -= operand;
            expect_result( sole, expected, expected_shape, what );
        }
        expect_result( receiver - operand, expected, expected_shape, what );
        {
            auto shared{ receiver };
            shared.andnot_with( operand );
            expect_result( shared, expected, expected_shape, what );
        }
    }
};

using policies = ::testing::Types<eager_bitmap, lazy_bitmap, atomic_bitmap, atomic_lazy_bitmap>;
TYPED_TEST_SUITE( AndnotKeepsRuns, policies );

// ---- run minus array ---------------------------------------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, a_few_values_split_the_run ) {
    auto const operand{ TestFixture::from_members( members_of( { 100, 200, 300 } ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::array );
    TestFixture::expect_subtraction( operand, shape::run, "run - 3 values" );
}

TYPED_TEST( AndnotKeepsRuns, many_holes_are_more_runs_than_a_bitset_holds ) {
    std::vector<std::uint32_t> holes;
    for ( std::uint32_t value{ 0 }; value < 6'000U; value += 2U ) {
        holes.push_back( value );
    }
    auto const operand{ TestFixture::from_members( members_of( holes ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::array );
    TestFixture::expect_subtraction( operand, shape::bitset, "run - 3000 holes" );
}

TYPED_TEST( AndnotKeepsRuns, some_holes_keep_the_smaller_runs ) {
    std::vector<std::uint32_t> holes;
    for ( std::uint32_t value{ 0 }; value < 1'000U; value += 4U ) {
        holes.push_back( value );
    }
    auto const operand{ TestFixture::from_members( members_of( holes ) ) };
    TestFixture::expect_subtraction( operand, shape::run, "run - 250 holes" );
}

// ---- run minus run -----------------------------------------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, a_run_cuts_the_run_in_two ) {
    auto const operand{ TestFixture::from_members( closed_range_members( 1'000, 2'000 ), true ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::run );
    TestFixture::expect_subtraction( operand, shape::run, "run - run" );
}

TYPED_TEST( AndnotKeepsRuns, a_run_leaving_a_few_values ) {
    auto const taken{ receiver_values_but( members_of( scattered_nine() ) ) };
    auto const operand{ TestFixture::from_members( members_of( taken ), true ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::run );
    TestFixture::expect_subtraction( operand, shape::array, "run - run leaving 9 values" );
}

// ---- run minus bitset --------------------------------------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, a_bitset_leaving_one_stretch_keeps_a_run ) {
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( closed_range_members( 100, 40'099 ) ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    TestFixture::expect_subtraction( operand, shape::run, "run - bitset leaving one stretch" );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_leaving_a_few_stretches_keeps_runs ) {
    auto survivors{ closed_range_members( 10, 20 ) };
    for ( std::uint32_t value{ 5'000 }; value < 5'500; ++value ) {
        survivors[ value ] = true;
    }
    survivors[ 59'999 ] = true;
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( survivors ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    TestFixture::expect_subtraction( operand, shape::run, "run - bitset leaving three stretches" );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_leaving_a_few_values_makes_an_array ) {
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( members_of( scattered_nine() ) ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    TestFixture::expect_subtraction( operand, shape::array, "run - bitset leaving 9 values" );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_leaving_scattered_values_makes_a_bitset ) {
    members survivors( chunk_size );
    for ( std::uint32_t value{ 100 }; value < 20'100; value += 3U ) {
        survivors[ value ] = true;
    }
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( survivors ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    TestFixture::expect_subtraction( operand, shape::bitset, "run - bitset leaving scattered values" );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_that_misses_the_runs_leaves_them ) {
    TypeParam receiver;
    receiver.add_closed_range( 0, 19'999 );
    receiver.optimize();
    members tail( chunk_size );
    for ( std::uint32_t value{ 30'000 }; value < chunk_size; ++value ) {
        tail[ value ] = value % 3U != 0U;
    }
    auto const disjoint{ TestFixture::from_members( tail ) };
    ASSERT_EQ( TestFixture::shape_of( disjoint ), shape::bitset );

    auto shared{ receiver };
    shared -= disjoint;
    EXPECT_EQ( TestFixture::shape_of( shared ), shape::run );
    EXPECT_EQ( shared.size(), 20'000U );
    EXPECT_EQ( shared.validate(), nullptr );
    EXPECT_EQ( TestFixture::shape_of( receiver - disjoint ), shape::run );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_covering_the_runs_leaves_nothing ) {
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( members( chunk_size ) ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    auto receiver{ TestFixture::run_receiver() };
    receiver -= operand;
    EXPECT_TRUE( receiver.empty() );
    EXPECT_EQ( receiver.statistics().container_count(), 0U );
}

// ---- several runs, several chunks --------------------------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, every_chunk_is_decided_on_its_own ) {
    TypeParam receiver;
    receiver.add_closed_range( 0, run_end );
    receiver.add_closed_range( chunk_size, chunk_size + run_end );
    receiver.add_closed_range( 2U * chunk_size, 2U * chunk_size + run_end );
    receiver.optimize();
    ASSERT_EQ( receiver.statistics().run_containers, 3U );

    TypeParam operand;
    for ( auto const value : values_of( bitset_operand_leaving( closed_range_members( 100, 40'099 ) ) ) ) {
        operand.add( value );
        operand.add( 2U * chunk_size + value );
    }
    operand.add( chunk_size + 5U );

    receiver -= operand;
    auto const stats{ receiver.statistics() };
    EXPECT_EQ( stats.run_containers, 3U );
    EXPECT_EQ( stats.bitset_containers, 0U );
    EXPECT_EQ( receiver.size(), 40'000U + run_end + 40'000U );
    EXPECT_EQ( receiver.validate(), nullptr );
}

// ---- sorted keys and ranges --------------------------------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, a_span_of_sorted_keys ) {
    auto const check{ [ & ]( std::vector<std::uint32_t> const & keys, shape const expected_shape, char const * const what ) {
        auto const expected{ without( receiver_members(), members_of( keys ) ) };
        auto const receiver{ TestFixture::run_receiver() };
        auto shared{ receiver };
        shared.andnot_with( std::span<std::uint32_t const>{ keys } );
        TestFixture::expect_result( shared, expected, expected_shape, what );
        EXPECT_EQ( TestFixture::shape_of( receiver ), shape::run ) << what << ": the other holder of the runs";
    } };
    check( { 100, 200, 300 }, shape::run, "run - 3 keys" );
    std::vector<std::uint32_t> every_other;
    for ( std::uint32_t value{ 100 }; value < 10'100; value += 2U ) {
        every_other.push_back( value );
    }
    check( every_other, shape::bitset, "run - 5000 alternating keys" );
    check( receiver_values_but( members_of( scattered_nine() ) ), shape::array, "run - all but 9 keys" );
}

TYPED_TEST( AndnotKeepsRuns, remove_closed_range ) {
    auto const check{ [ & ]( std::uint32_t const first, std::uint32_t const last, shape const expected_shape, char const * const what ) {
        auto const expected{ without( receiver_members(), closed_range_members( first, last ) ) };
        auto const receiver{ TestFixture::run_receiver() };
        auto shared{ receiver };
        shared.remove_closed_range( first, last );
        TestFixture::expect_result( shared, expected, expected_shape, what );
        EXPECT_EQ( TestFixture::shape_of( receiver ), shape::run ) << what << ": the other holder of the runs";
        EXPECT_EQ( receiver.size(), run_end + 1U ) << what;
    } };
    check( 1'000, 2'000, shape::run, "middle" );
    check( 0, 2'000, shape::run, "prefix" );
    check( 50'000, 65'535, shape::run, "suffix beyond the end" );
    check( 60'000, 65'535, shape::run, "no overlap" );
    check( 1, run_end - 1U, shape::array, "only the two ends left" );
    check( 0, 65'535, shape::empty, "everything" );
}

TYPED_TEST( AndnotKeepsRuns, remove_many_sorted ) {
    auto const check{ [ & ]( std::vector<std::uint32_t> const & keys, shape const expected_shape, char const * const what ) {
        auto const expected{ without( receiver_members(), members_of( keys ) ) };
        auto const receiver{ TestFixture::run_receiver() };
        auto shared{ receiver };
        shared.remove_many_sorted( { keys.data(), keys.size() } );
        TestFixture::expect_result( shared, expected, expected_shape, what );
        EXPECT_EQ( TestFixture::shape_of( receiver ), shape::run ) << what << ": the other holder of the runs";
    } };
    check( { 100, 200, 300 }, shape::run, "3 keys" );
    std::vector<std::uint32_t> alternating;
    for ( std::uint32_t value{ 0 }; value < 6'000U; value += 2U ) {
        alternating.push_back( value );
    }
    check( alternating, shape::bitset, "3000 alternating keys" );
    check( receiver_values_but( members_of( scattered_nine() ) ), shape::array, "all but 9 keys" );
}

// ---- the size boundary between runs and a bitset ---------------------------------------------------------------
// A run container takes four bytes per run plus its two-byte count, so 2047 runs are smaller than the 8 KB bitset
// and 2048 are not. Past 2048 runs the bitset difference stops counting and finishes in words.

// The receiver minus isolated values 1, 3, ..., 2 * holes - 1: holes + 1 runs.
[[nodiscard]] members receiver_with_holes( std::uint32_t const holes ) {
    auto result{ receiver_members() };
    for ( std::uint32_t hole{ 0 }; hole != holes; ++hole ) {
        result[ 2U * hole + 1U ] = false;
    }
    return result;
}

TYPED_TEST( AndnotKeepsRuns, the_run_count_decides_between_runs_and_a_bitset ) {
    auto const check{ [ & ]( std::uint32_t const runs, shape const expected_shape ) {
        auto const survivors{ receiver_with_holes( runs - 1U ) };
        auto const operand{ TestFixture::from_members( bitset_operand_leaving( survivors ) ) };
        ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
        TestFixture::expect_subtraction( operand, expected_shape, "run - bitset leaving isolated holes" );

        members holes( chunk_size );
        for ( std::uint32_t hole{ 0 }; hole != runs - 1U; ++hole ) {
            holes[ 2U * hole + 1U ] = true;
        }
        TestFixture::expect_subtraction( TestFixture::from_members( holes ), expected_shape, "run - the holes as an array" );
    } };
    check( 2'047U, shape::run );
    check( 2'048U, shape::bitset );
    check( 2'049U, shape::bitset );
}

TYPED_TEST( AndnotKeepsRuns, a_bitset_leaving_more_runs_than_fit_but_few_values_makes_an_array ) {
    // 2100 isolated survivors: the runs outnumber what a bitset holds, the values are fewer than a bitset's threshold.
    members survivors( chunk_size );
    for ( std::uint32_t index{ 0 }; index != 2'100U; ++index ) {
        survivors[ 3U * index ] = true;
    }
    auto const operand{ TestFixture::from_members( bitset_operand_leaving( survivors ) ) };
    ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset );
    TestFixture::expect_subtraction( operand, shape::array, "run - bitset leaving 2100 isolated values" );
    auto const as_runs{ TestFixture::from_members( bitset_operand_leaving( survivors ), true ) };
    TestFixture::expect_subtraction( as_runs, shape::array, "run - run leaving 2100 isolated values" );
}

// ---- a receiver run that ends at the top of the chunk ------------------------------------------------------

TYPED_TEST( AndnotKeepsRuns, a_run_ending_at_the_top_of_the_chunk ) {
    auto const top_members{ closed_range_members( 10'000, 65'535 ) };
    auto const check{ [ & ]( members const & survivors_in_run, shape const expected_shape, char const * const what ) {
        members operand_members( chunk_size );
        for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
            operand_members[ value ] = value < 10'000U ? value % 2U == 0U : !survivors_in_run[ value ];
        }
        auto const operand{ TestFixture::from_members( operand_members ) };
        ASSERT_EQ( TestFixture::shape_of( operand ), shape::bitset ) << what;

        TypeParam receiver;
        receiver.add_closed_range( 10'000, 65'535 );
        receiver.optimize();
        ASSERT_EQ( TestFixture::shape_of( receiver ), shape::run ) << what;
        auto shared{ receiver };
        shared -= operand;
        TestFixture::expect_result( shared, without( top_members, operand_members ), expected_shape, what );
        EXPECT_EQ( TestFixture::shape_of( receiver ), shape::run ) << what << ": the other holder of the runs";
    } };
    check( closed_range_members( 65'000, 65'535 ), shape::run, "top stretch survives" );
    check( members_of( { 65'535 } ), shape::array, "only the top value survives" );

    // the array and run operands, a hole at the top
    TypeParam receiver;
    receiver.add_closed_range( 10'000, 65'535 );
    receiver.optimize();
    auto shared{ receiver };
    shared.remove_closed_range( 65'000, 65'535 );
    TestFixture::expect_result( shared, without( top_members, closed_range_members( 65'000, 65'535 ) ), shape::run, "top range removed" );
    auto top_value{ receiver };
    top_value.remove( 65'535 );
    TestFixture::expect_result( top_value, without( top_members, members_of( { 65'535 } ) ), shape::run, "top value removed" );
}

// ---- a container set without bitsets -------------------------------------------------------------------------

TEST( AndnotKeepsRunsArrayRunSet, results_that_would_be_bitsets_are_arrays ) {
    auto const holes{ [] {
        std::vector<std::uint32_t> result;
        for ( std::uint32_t value{ 1 }; value < 6'000U; value += 2U ) {
            result.push_back( value );
        }
        return result;
    }() };
    auto const expected{ without( receiver_members(), members_of( holes ) ) };
    auto const check{ [ & ]( auto const & subtract, char const * const what ) {
        array_run_bitmap receiver;
        receiver.add_closed_range( 0, run_end );
        receiver.optimize_for_speed();
        ASSERT_EQ( receiver.statistics().run_containers, 1U ) << what;
        subtract( receiver );
        auto const stats{ receiver.statistics() };
        EXPECT_EQ( stats.bitset_containers, 0U ) << what;
        EXPECT_EQ( receiver.validate(), nullptr ) << what;
        EXPECT_TRUE( std::ranges::equal( receiver.to_vector(), values_of( expected ) ) ) << what;
    } };
    check( [ & ]( array_run_bitmap & receiver ) { receiver.remove_many_sorted( { holes.data(), holes.size() } ); }, "remove_many_sorted" );
    check( [ & ]( array_run_bitmap & receiver ) {
        array_run_bitmap operand;
        operand.add_many_sorted( { holes.data(), holes.size() } );
        receiver -= operand;
    }, "-= an array" );
    check( [ & ]( array_run_bitmap & receiver ) {
        for ( std::uint32_t first{ 1 }; first < 6'000U; first += 2U ) {
            receiver.remove_closed_range( first, first );
        }
    }, "remove_closed_range one value at a time" );
}

// A range that misses every run leaves the container, and a copy that shares it, alone.
TYPED_TEST( AndnotKeepsRuns, a_range_that_misses_the_runs_keeps_the_shared_runs ) {
    auto const receiver{ TestFixture::run_receiver() };
    auto copy{ receiver };
    copy.remove_closed_range( 61'000, 62'000 );
    TestFixture::expect_result( copy, receiver_members(), shape::run, "range past the runs" );
}

// ---- the reference's own run-minus-bitset cases, on bare containers ------------------------------------------
// The data of the reference's run_andnot_test (tests/mixed_container_unit.c): runs and bitsets over the same
// membership rules, subtracted directly. The result takes the form the reference gives it.
// [croaring-ref] tests/mixed_container_unit.c:run_andnot_test

using container_layout_t = detail::default_layout<std::uint32_t>;
using container_t        = detail::container_handle<container_layout_t, detail::cow_value_semantics>;

template <typename Member>
[[nodiscard]] container_t run_container_of( Member const & member ) {
    auto result{ container_t::make_run() };
    auto run{ result.as_run() };
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        if ( member( value ) ) {
            std::uint32_t last{ value };
            while ( last + 1U < chunk_size && member( last + 1U ) ) {
                ++last;
            }
            run.add_closed_range( static_cast<std::uint16_t>( value ), static_cast<std::uint16_t>( last ) );
            value = last;
        }
    }
    return result;
}

template <typename Member>
[[nodiscard]] container_t bitset_container_of( Member const & member ) {
    auto result{ container_t::make_bitset_uninitialized() };
    auto & words{ result.as_bitset().words.as_array() };
    words.fill( 0 );
    std::uint32_t count{ 0 };
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        if ( member( value ) ) {
            words[ value / 64U ] |= std::uint64_t{ 1 } << ( value % 64U );
            ++count;
        }
    }
    result.set_cardinality( count );
    return result;
}

[[nodiscard]] container_t difference_of( container_t const & runs, container_t const & bitset ) {
    return detail::difference_run_bitset<container_layout_t, detail::cow_value_semantics>( runs.as_run(), bitset.as_bitset(), eager_bitmap::array_to_bitset_threshold );
}

TEST( AndnotKeepsRunsContainers, the_reference_cases ) {
    auto const r1_member{ []( std::uint32_t const x ) { return x % 5U < 3U; } };
    auto const m2_member{ []( std::uint32_t const x ) { return x % 62U < 37U; } };
    auto const m3_member{ []( std::uint32_t const x ) { return x % 5U < 2U || ( x % 5U < 3U && x > 10'000U ); } };
    auto const small    { []( std::uint32_t const x ) { return x >= 990U && x < 1'000U; } };

    auto const r1{ run_container_of( r1_member ) };
    auto const r_small{ run_container_of( small ) };
    auto const b1{ bitset_container_of( r1_member ) };
    auto const b2{ bitset_container_of( m2_member ) };
    auto const b3{ bitset_container_of( m3_member ) };

    EXPECT_EQ( detail::container_size( difference_of( r1, b1 ) ), 0U );

    auto const against_b3{ difference_of( r1, b3 ) };
    EXPECT_EQ( against_b3.kind(), detail::container_kind::array );
    EXPECT_EQ( detail::container_size( against_b3 ), 2'000U );

    std::uint32_t expected_against_b2{ 0 };
    for ( std::uint32_t x{ 0 }; x < chunk_size; ++x ) {
        expected_against_b2 += r1_member( x ) && !m2_member( x ) ? 1U : 0U;
    }
    auto const against_b2{ difference_of( r1, b2 ) };
    EXPECT_EQ( against_b2.kind(), detail::container_kind::bitset );
    EXPECT_EQ( detail::container_size( against_b2 ), expected_against_b2 );

    // 990 and 991 are all that is left of the ten values: two values, which as one run take the bytes they take as an array.
    auto const small_left{ difference_of( r_small, b2 ) };
    EXPECT_EQ( detail::container_size( small_left ), 2U );
    EXPECT_TRUE( small_left.kind() != detail::container_kind::bitset );
}

// ---- the reference's own array cases, on bare containers -----------------------------------------------------
// The data of the reference's run_andnot_test for the array operands (an array minus runs, runs minus an array),
// with the form of every result the reference gives it.
// [croaring-ref] tests/mixed_container_unit.c:run_andnot_test

template <typename Member>
[[nodiscard]] container_t array_container_of( Member const & member ) {
    std::vector<std::uint16_t> values;
    for ( std::uint32_t value{ 0 }; value < chunk_size; ++value ) {
        if ( member( value ) ) {
            values.push_back( static_cast<std::uint16_t>( value ) );
        }
    }
    return container_t::make_array_from_sorted( { values.data(), values.size() } );
}

[[nodiscard]] std::size_t array_minus_runs( container_t const & array, container_t const & runs ) {
    return detail::difference_array_run<container_layout_t, detail::cow_value_semantics>( array.as_array(), runs.as_run() ).size();
}

[[nodiscard]] container_t runs_minus_array( container_t const & runs, container_t const & array ) {
    return detail::difference_run_array<container_layout_t, detail::cow_value_semantics>( runs.as_run(), array.as_array(), eager_bitmap::array_to_bitset_threshold );
}

TEST( AndnotKeepsRunsContainers, the_reference_array_cases ) {
    auto const r1_member{ []( std::uint32_t const x ) { return x % 5U < 3U; } };
    auto const m2_member{ []( std::uint32_t const x ) { return x % 62U < 37U; } };
    auto const m3_member{ []( std::uint32_t const x ) { return x % 5U < 2U || ( x % 5U < 3U && x > 10'000U ); } };
    auto const m4_member{ []( std::uint32_t const x ) {
        // the reference's pseudo-random fill: three of every four positions
        static auto const table{ [] {
            std::vector<bool> result( chunk_size );
            int state{ 1 };
            for ( std::uint32_t at{ 0 }; at < chunk_size; ++at ) {
                result[ at ] = state % 4 != 0;
                state = ( 3432 * state + 6789 ) % 9973;
            }
            return result;
        }() };
        return static_cast<bool>( table[ x ] );
    } };
    auto const small{ []( std::uint32_t const x ) { return x >= 990U && x < 1'000U; } };

    auto const a1{ array_container_of( r1_member ) };
    auto const a2{ array_container_of( m2_member ) };
    auto const a3{ array_container_of( m3_member ) };
    auto const a4{ array_container_of( m4_member ) };
    auto const a_small{ array_container_of( small ) };
    auto const r1{ run_container_of( r1_member ) };
    auto const r2{ run_container_of( m2_member ) };
    auto const r3{ run_container_of( m3_member ) };
    auto const r_small{ run_container_of( small ) };

    auto const count_of{ []( auto const & keep ) {
        std::size_t result{ 0 };
        for ( std::uint32_t x{ 0 }; x < chunk_size; ++x ) {
            result += keep( x ) ? 1U : 0U;
        }
        return result;
    } };

    // an empty result, and a result of only the ends of the array
    EXPECT_EQ( array_minus_runs( a1, r1 ), 0U );
    EXPECT_EQ( array_minus_runs( a1, r3 ), 2'000U );
    EXPECT_EQ( array_minus_runs( a1, r2 ), count_of( [ & ]( std::uint32_t const x ) { return r1_member( x ) && !m2_member( x ); } ) );
    EXPECT_EQ( array_minus_runs( a_small, r2 ), 2U );
    EXPECT_EQ( array_minus_runs( a4, r3 ), count_of( [ & ]( std::uint32_t const x ) { return m4_member( x ) && !m3_member( x ); } ) );

    // nothing to take out leaves the array whole; a run over all of it leaves nothing
    auto const empty_runs{ container_t::make_run() };
    EXPECT_EQ( array_minus_runs( a1, empty_runs ), a1.as_array().values.size() );
    auto const full_runs{ run_container_of( []( std::uint32_t ) { return true; } ) };
    EXPECT_EQ( array_minus_runs( a1, full_runs ), 0U );

    auto const against_a3{ runs_minus_array( r1, a3 ) };
    EXPECT_EQ( against_a3.kind(), detail::container_kind::array );
    EXPECT_EQ( detail::container_size( against_a3 ), 2'000U );

    auto const small_left{ runs_minus_array( r_small, a2 ) };
    EXPECT_EQ( detail::container_size( small_left ), 2U );
    EXPECT_TRUE( small_left.kind() != detail::container_kind::bitset );

    // runs with a few holes stay runs: 13 values in five runs take less than the same values as an array
    auto const complex_member{ []( std::uint32_t const x ) {
        return x < 3U || ( x >= 10U && x < 12U ) || ( x >= 990U && x < 995U ) || ( x >= 10'000U && x < 10'003U ) || ( x >= 20'000U && x < 20'002U );
    } };
    auto const three_values{ container_t::make_array_from_sorted( std::vector<std::uint16_t>{ 993, 994, 2'000 } ) };
    auto const complex_left{ runs_minus_array( run_container_of( complex_member ), three_values ) };
    EXPECT_EQ( complex_left.kind(), detail::container_kind::run );
    EXPECT_EQ( detail::container_size( complex_left ), 13U );
}

// ---- an array minus runs, over every container set and layout --------------------------------------------------
// Random arrays minus random run operands, compared with a membership table: through the binary and the in-place
// operator, a shared and a sole receiver. The container set without bitsets and the narrow layouts take the same
// kernel with other element types and limits.

template <typename Bitmap>
void array_minus_runs_trials( std::uint64_t const seed ) {
    using key_t = typename Bitmap::key_type;
    constexpr std::uint64_t domain{ std::numeric_limits<key_t>::digits >= 32 ? std::uint64_t{ 3 } * chunk_size : std::uint64_t{ 1 } << std::numeric_limits<key_t>::digits };
    std::mt19937_64 rng{ seed };
    for ( int trial{ 0 }; trial < 300; ++trial ) {
        std::vector<bool> receiver_table( domain );
        std::vector<bool> operand_table( domain );
        auto const density{ 1U + static_cast<unsigned>( rng() % 40 ) };
        for ( std::uint64_t value{ 0 }; value < domain; ++value ) {
            receiver_table[ value ] = rng() % 1'000 < density;
        }
        auto const longest{ 1U + static_cast<std::uint64_t>( rng() % ( domain / 8 + 1 ) ) };
        auto const widest_gap{ 1U + static_cast<std::uint64_t>( rng() % ( domain / 4 + 1 ) ) };
        for ( std::uint64_t value{ rng() % widest_gap }; value < domain; value += 1 + rng() % widest_gap ) {
            auto const length{ 1 + rng() % longest };
            for ( std::uint64_t i{ 0 }; i < length && value < domain; ++i, ++value ) {
                operand_table[ value ] = true;
            }
        }
        auto const listed{ [ & ]( std::vector<bool> const & table ) {
            std::vector<key_t> result;
            for ( std::uint64_t value{ 0 }; value < domain; ++value ) {
                if ( table[ value ] ) {
                    result.push_back( static_cast<key_t>( value ) );
                }
            }
            return result;
        } };
        auto const receiver_values{ listed( receiver_table ) };
        auto const operand_values { listed( operand_table ) };
        std::vector<bool> left_table( domain );
        for ( std::uint64_t value{ 0 }; value < domain; ++value ) {
            left_table[ value ] = receiver_table[ value ] && !operand_table[ value ];
        }
        auto const expected{ listed( left_table ) };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " receiver " << receiver_values.size() << " operand " << operand_values.size() );

        auto const build{ [ & ]( std::vector<key_t> const & values, bool const run_optimized ) {
            Bitmap result;
            for ( auto const value : values ) {
                result.add( value );
            }
            if ( run_optimized ) {
                result.optimize();
            }
            return result;
        } };
        auto const receiver{ build( receiver_values, false ) };
        auto const operand { build( operand_values , true  ) };

        auto const difference{ receiver - operand };
        EXPECT_EQ( difference.validate(), nullptr ) << "binary";
        EXPECT_TRUE( std::ranges::equal( difference.to_vector(), expected ) ) << "binary";
        {
            auto shared{ receiver };
            shared -= operand;
            EXPECT_EQ( shared.validate(), nullptr ) << "-= on a shared copy";
            EXPECT_TRUE( std::ranges::equal( shared.to_vector(), expected ) ) << "-= on a shared copy";
            EXPECT_TRUE( std::ranges::equal( receiver.to_vector(), receiver_values ) ) << "the other holder";
        }
        {
            auto sole{ build( receiver_values, false ) };
            sole -= operand;
            EXPECT_EQ( sole.validate(), nullptr ) << "-= on a sole holder";
            EXPECT_TRUE( std::ranges::equal( sole.to_vector(), expected ) ) << "-= on a sole holder";
        }
    }
}

TEST( ArrayMinusRuns, default_set ) { array_minus_runs_trials<eager_bitmap>( 0xA2201 ); }
TEST( ArrayMinusRuns, array_and_run_set ) { array_minus_runs_trials<array_run_bitmap>( 0xA2202 ); }
TEST( ArrayMinusRuns, sixteen_bit_keys ) { array_minus_runs_trials<bitmap<std::uint16_t>>( 0xA2203 ); }
TEST( ArrayMinusRuns, eight_bit_keys ) { array_minus_runs_trials<bitmap<std::uint8_t>>( 0xA2204 ); }

}  // namespace
