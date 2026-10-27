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
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace frsr::roaring;

using eager_bitmap       = bitmap<std::uint32_t>;
using lazy_bitmap        = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_value_semantics, detail::run_selection_lazy>;
using atomic_bitmap      = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using atomic_lazy_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>;

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

}  // namespace
