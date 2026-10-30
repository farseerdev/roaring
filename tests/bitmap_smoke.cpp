#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/run_container.hpp>

#include "allocation_fault.hpp"

#include <gtest/gtest.h>
#include <cstring>

#include <algorithm>
#include <array>
#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <span>
#include <tuple>
#include <atomic>
#include <iterator>
#include <set>
#include <thread>
#include <vector>

namespace {

using TestBitmap = frsr::roaring::bitmap<std::uint32_t>;

TEST(FrsrRoaringSmoke, RangeConstructionAndOperatorsPreserveSortedValues) {
    TestBitmap const lhs{ std::vector<std::uint32_t>{ 1U, 2U, 65'536U, 65'538U } };
    TestBitmap const rhs{ std::vector<std::uint32_t>{ 2U, 3U, 65'536U, 131'072U } };

    EXPECT_EQ( ( lhs | rhs ).to_vector(), ( std::vector<std::uint32_t>{ 1U, 2U, 3U, 65'536U, 65'538U, 131'072U } ) );
    EXPECT_EQ( ( lhs & rhs ).to_vector(), ( std::vector<std::uint32_t>{ 2U, 65'536U } ) );
    EXPECT_EQ( ( lhs - rhs ).to_vector(), ( std::vector<std::uint32_t>{ 1U, 65'538U } ) );
    EXPECT_EQ( ( lhs & !rhs ).to_vector(), ( std::vector<std::uint32_t>{ 1U, 65'538U } ) );
}

TEST(FrsrRoaringSmoke, BulkContextAcceleratesSequentialChunkReuse) {
    TestBitmap bitmap;
    TestBitmap::bulk_context ctx;

    EXPECT_TRUE( bitmap.add_bulk( ctx, 1U ) );
    EXPECT_TRUE( bitmap.add_bulk( ctx, 2U ) );
    EXPECT_TRUE( bitmap.add_bulk( ctx, 65'536U ) );
    EXPECT_TRUE( bitmap.add_bulk( ctx, 65'537U ) );
    EXPECT_FALSE( bitmap.add_bulk( ctx, 65'537U ) );

    EXPECT_TRUE( bitmap.contains_bulk( ctx, 65'537U ) );
    EXPECT_FALSE( bitmap.contains_bulk( ctx, 65'539U ) );
}

// One context is carried across every kind of change to the chunk table: chunks appear before and
// after the cached one, the cached chunk empties and is erased, and a lookup in between lands on
// other chunks. Every answer is compared with a model.
template <typename Bitmap>
void bulk_context_survives_chunk_table_changes() {
    using key_type = typename Bitmap::key_type;
    constexpr key_type chunk_span{ key_type{ 1 } << 16 };
    std::mt19937_64 rng{ 77 };
    Bitmap bitmap;
    std::set<key_type> model;
    typename Bitmap::bulk_context ctx;
    auto const pick{ [&] {
        // few chunks with few values each, so that chunks are created and emptied constantly
        return static_cast<key_type>( ( rng() % 96 ) * chunk_span + rng() % 6 );
    } };
    for ( int step{ 0 }; step < 60'000; ++step ) {
        auto const value{ pick() };
        switch ( rng() % 5 ) {
            case 0: EXPECT_EQ( bitmap.add_bulk( ctx, value ), model.insert( value ).second ) << step; break;
            case 1: EXPECT_EQ( bitmap.add( value ), model.insert( value ).second ) << step; break;
            case 2: EXPECT_EQ( bitmap.remove( value ), model.erase( value ) != 0 ) << step; break;
            default: EXPECT_EQ( bitmap.contains_bulk( ctx, value ), model.contains( value ) ) << step; break;
        }
    }
    EXPECT_EQ( bitmap.size(), model.size() );
    EXPECT_STREQ( bitmap.validate() == nullptr ? "valid" : bitmap.validate(), "valid" );
    EXPECT_TRUE( std::ranges::equal( bitmap.to_vector(), model ) );
}

TEST(FrsrRoaringSmoke, BulkContextSurvivesChunkTableChanges) {
    bulk_context_survives_chunk_table_changes<TestBitmap>();
    bulk_context_survives_chunk_table_changes<frsr::roaring::bitmap<std::uint64_t>>();
}

// An intersection whose last pair has no common value abandons the container slot it opened for it,
// which on a full chunk table leaves a new empty last leaf behind. Whatever is asked of the result
// afterwards, that leaf must not show.
TEST(FrsrRoaringSmoke, AnIntersectionThatEndsOnAnEmptyPairLeavesACleanWideKeyResult) {
    using Wide = frsr::roaring::bitmap<std::uint64_t>;
    constexpr std::uint64_t span{ 1ULL << 16 };
    Wide a, b;
    for ( std::uint64_t chunk{ 0 }; chunk < 15; ++chunk ) {
        (void)a.add( chunk * span + 5 );
        (void)b.add( chunk * span + 5 );
    }
    (void)a.add( 15 * span + 7 );                                   // the last pair has nothing in common
    (void)b.add( 15 * span + 9 );
    Wide result{ a & b };
    ASSERT_TRUE( result.validate() == nullptr ) << result.validate();
    ASSERT_EQ( result.size(), 15U );
    EXPECT_TRUE( result.remove( 14 * span + 5 ) );                  // the real last leaf now ends below the gap
    ASSERT_TRUE( result.validate() == nullptr ) << result.validate();
    Wide probe;
    for ( std::uint64_t chunk{ 0 }; chunk < 40; ++chunk ) { (void)probe.add( chunk * span + 5 ); }
    std::array<Wide const *, 2> operands{ &result, &probe };
    auto const expected{ [&] {
        std::vector<std::uint64_t> values;
        for ( std::uint64_t chunk{ 0 }; chunk < 14; ++chunk ) { values.push_back( chunk * span + 5 ); }
        return values;
    }() };
    EXPECT_EQ( Wide::and_many( operands ).to_vector(), expected );
    EXPECT_EQ( Wide::and_many_cardinality( operands ), expected.size() );
    EXPECT_TRUE( result.intersects( probe ) );
    EXPECT_FALSE( result.contains( 14 * span + 5 ) );
    EXPECT_FALSE( result.contains( 15 * span + 7 ) );
    EXPECT_EQ( result.to_vector(), expected );
    EXPECT_EQ( result.back(), expected.back() );
    std::vector<std::uint64_t> iterated;
    for ( auto const value : result ) { iterated.push_back( value ); }
    EXPECT_EQ( iterated, expected );
}

TEST(FrsrRoaringSmoke, SortedBulkMutationsAndPromotionHelpersWork) {
    TestBitmap bitmap;
    std::vector<std::uint32_t> values;
    for ( std::uint32_t value{ 0U }; value < 2048U; ++value ) {
        values.push_back( value );
    }

    bitmap.add_sorted_many( values );
    EXPECT_EQ( bitmap.front(), 0U );
    EXPECT_EQ( bitmap.back(), 2047U );
    EXPECT_TRUE( bitmap.intersects( TestBitmap{ std::vector<std::uint32_t>{ 1024U, 999'999U } } ) );

    bitmap.remove_many_sorted( std::span{ values }.subspan( 0, 512 ) );
    EXPECT_FALSE( bitmap.contains( 1U ) );
    EXPECT_TRUE( bitmap.contains( 1024U ) );

    bitmap.optimize_for_storage();
    bitmap.optimize_for_speed();
    EXPECT_GT( bitmap.byte_size(), 0U );
    EXPECT_GT( bitmap.container_count(), 0U );
}

TEST(FrsrRoaringSmoke, OptimizingABitsetChoosesTheFormTheValuesWouldHave) {
    // The lazy run policy leaves a bitset a bitset even for contiguous values, which the eager policy would store as runs.
    using BitsetFormBitmap = frsr::roaring::bitmap<
        std::uint32_t,
        frsr::roaring::default_container_set<std::uint32_t>,
        frsr::roaring::detail::cow_value_semantics,
        frsr::roaring::detail::run_selection_lazy
    >;
    // A bitset with fewer values than the array limit becomes an array or, when runs hold them in fewer bytes, a
    // run container: the same form a container built from those values gets, whatever the shape of the values.
    auto const range_values = []( std::uint32_t const first, std::uint32_t const last, std::uint32_t const stride ) {
        std::vector<std::uint32_t> values;
        for ( auto value{ first }; value <= last; value += stride ) {
            values.push_back( value );
        }
        return values;
    };
    std::vector<std::vector<std::uint32_t>> const shapes{
        range_values( 0U, 59U, 1U ),          // under 64 values: never a run
        range_values( 0U, 99U, 1U ),          // one run
        range_values( 0U, 6000U, 100U ),      // isolated values
        range_values( 10U, 3000U, 1U ),       // one long run
        range_values( 0U, 60000U, 20U ),      // 3000 isolated values
        range_values( 0U, 5000U, 1U ),        // 5001 values in one run
        range_values( 0U, 65535U, 13U )       // 5043 isolated values: stays a bitset
    };
    for ( auto const & values : shapes ) {
        auto expected{ TestBitmap{ values } };
        expected.optimize();
        BitsetFormBitmap bitset_form;
        bitset_form.add_sorted_many( values, 1U );
        ASSERT_EQ( bitset_form.statistics().bitset_containers, 1U ) << values.size();
        bitset_form.optimize();
        EXPECT_EQ( bitset_form.to_vector(), values );
        EXPECT_EQ( bitset_form.statistics().array_containers, expected.statistics().array_containers ) << values.size();
        EXPECT_EQ( bitset_form.statistics().run_containers, expected.statistics().run_containers ) << values.size();
        EXPECT_EQ( bitset_form.statistics().bitset_containers, expected.statistics().bitset_containers ) << values.size();
    }
}

TEST(FrsrRoaringSmoke, SortedBulkConstructionHonorsAnExplicitBitsetThreshold) {
    // A chunk between an explicit (lowered) threshold and the byte-parity point:
    // the default decision keeps it an array (smaller), the explicit threshold
    // asks for a bitset regardless of bytes.
    std::vector<std::uint32_t> values;
    for ( std::uint32_t value{ 0U }; value < 2'048U; ++value ) {
        values.push_back( value * 3U );
    }
    ASSERT_LT( values.size(), TestBitmap::array_to_bitset_threshold );

    TestBitmap by_default;
    by_default.add_sorted_many( values );
    EXPECT_EQ( by_default.statistics().array_containers , 1U );
    EXPECT_EQ( by_default.statistics().bitset_containers, 0U );

    TestBitmap explicit_threshold;
    explicit_threshold.add_sorted_many( values, 1'024U );
    EXPECT_EQ( explicit_threshold.statistics().bitset_containers, 1U );
    EXPECT_EQ( explicit_threshold.statistics().array_containers , 0U );
    EXPECT_EQ( explicit_threshold.to_vector(), values );

    // Below the explicit threshold the chunk stays an array.
    TestBitmap under_threshold;
    under_threshold.add_sorted_many( std::span{ values }.first( 1'000U ), 1'024U );
    EXPECT_EQ( under_threshold.statistics().array_containers, 1U );
}

TEST(FrsrRoaringSmoke, PublicContainerTypesAreDirectlyUsable) {
    frsr::roaring::run<std::uint16_t> const segment{ 100U, 140U };
    EXPECT_EQ( segment.begin, 100U );
    EXPECT_EQ( segment.end, 140U );

    frsr::roaring::run_container<std::uint32_t> run;
    static_assert( std::same_as<decltype( run )::run, frsr::roaring::run<std::uint16_t>> );
    static_assert( std::same_as<decltype( run )::cardinality_type, std::uint32_t> );
    run.add_closed_range( 100U, 140U );
    run.remove_closed_range( 120U, 124U );
    EXPECT_TRUE( run.contains( 100U ) );
    EXPECT_FALSE( run.contains( 122U ) );
    EXPECT_EQ( run.size(), 36U );
}

TEST(FrsrRoaringSmoke, ContainerHandleMaintainsHeaderInvariants) {
    if constexpr ( FRSR_ROARING_NO_SBO ) { GTEST_SKIP() << "asserts inline-payload behaviour; FRSR_ROARING_NO_SBO deliberately removes it"; }
    using layout = frsr::roaring::detail::default_layout<std::uint32_t>;
    using handle = frsr::roaring::detail::container_handle<layout>;

    // array: inline SBO → heap spill, header (count/cardinality/[min,max]) in sync
    handle array_handle;
    auto array{ array_handle.as_array() };
    EXPECT_TRUE( array.add( 9U ) );
    EXPECT_TRUE( array.add( 7U ) );
    EXPECT_FALSE( array.add( 7U ) );
    EXPECT_TRUE( array.contains( 7U ) );
    EXPECT_EQ( array_handle.cardinality(), 2U );
    EXPECT_EQ( array_handle.min_value(), 7U );
    EXPECT_EQ( array_handle.max_value(), 9U );
    EXPECT_FALSE( array_handle.spilled() );
    for ( std::uint16_t value{ 100U }; value < 200U; ++value ) {
        EXPECT_TRUE( array.add( value ) );
    }
    EXPECT_TRUE( array_handle.spilled() );
    EXPECT_EQ( array_handle.cardinality(), 102U );
    EXPECT_EQ( array_handle.max_value(), 199U );
    EXPECT_TRUE( array.remove( 199U ) );
    EXPECT_EQ( array_handle.max_value(), 198U );

    // deep copy (owner unique) — copies diverge
    handle copy{ array_handle };
    EXPECT_TRUE( copy.as_array().remove( 7U ) );
    EXPECT_EQ( copy.min_value(), 9U );
    EXPECT_EQ( array_handle.min_value(), 7U );

    // bitset: add/remove maintain endpoints; sync_endpoints repairs after raw writes
    auto bitset_handle{ handle::make_bitset_zeroed() };
    auto bitset{ bitset_handle.as_bitset() };
    EXPECT_TRUE( bitset.add( 5U ) );
    EXPECT_TRUE( bitset.add( 60'000U ) );
    EXPECT_EQ( bitset_handle.cardinality(), 2U );
    EXPECT_EQ( bitset_handle.min_value(), 5U );
    EXPECT_EQ( bitset_handle.max_value(), 60'000U );
    EXPECT_TRUE( bitset.remove( 5U ) );
    EXPECT_EQ( bitset_handle.min_value(), 60'000U );

    // run handle mirrors the standalone run_container semantics
    auto run_handle{ handle::make_run() };
    auto run{ run_handle.as_run() };
    run.add_closed_range( 100U, 140U );
    run.remove_closed_range( 120U, 124U );
    EXPECT_TRUE( run.contains( 100U ) );
    EXPECT_FALSE( run.contains( 122U ) );
    EXPECT_EQ( run_handle.cardinality(), 36U );
    EXPECT_EQ( run_handle.min_value(), 100U );
    EXPECT_EQ( run_handle.max_value(), 140U );

    static_assert( sizeof( handle ) == 32U );
#if !FRSR_ROARING_NO_SBO
    static_assert( handle::inline_capacity<std::uint16_t> >= 8U );  // the sparse-regime SBO win
#endif
    static_assert( handle::is_trivially_moveable );
}

TEST(FrsrRoaringSmoke, BitmapCanUseCustomPublicContainerSets) {
    using ArrayBitsetSet = std::variant<
        frsr::roaring::array_container<std::uint32_t>,
        frsr::roaring::bitset_container<std::uint32_t>
    >;
    using ArrayBitsetBitmap = frsr::roaring::bitmap<std::uint32_t, ArrayBitsetSet>;
    using ArrayRunSet = std::variant<
        frsr::roaring::array_container<std::uint32_t>,
        frsr::roaring::run_container<std::uint32_t>
    >;
    using ArrayRunBitmap = frsr::roaring::bitmap<std::uint32_t, ArrayRunSet>;

    static_assert( std::same_as<ArrayBitsetBitmap::container_set_type, ArrayBitsetSet> );
    static_assert( std::same_as<ArrayRunBitmap::container_set_type, ArrayRunSet> );

    ArrayBitsetBitmap array_bitset;
    array_bitset.add_many( std::vector<std::uint32_t>{ 1U, 2U, 3U, 4'096U, 4'097U, 65'536U } );
    array_bitset.add_closed_range( 100U, 180U );
    array_bitset.remove_closed_range( 120U, 140U );
    array_bitset.optimize_for_storage();
    EXPECT_TRUE( array_bitset.contains( 4'096U ) );
    EXPECT_FALSE( array_bitset.contains( 130U ) );

    ArrayRunBitmap array_run;
    array_run.add_closed_range( 10U, 1'000U );
    array_run.flip_closed_range( 200U, 250U );
    array_run.remove_many_sorted( std::vector<std::uint32_t>{ 10U, 11U, 12U, 13U } );
    array_run.optimize_for_speed();
    EXPECT_TRUE( array_run.contains( 100U ) );
    EXPECT_FALSE( array_run.contains( 220U ) );
    EXPECT_FALSE( array_run.contains( 10U ) );

    auto const union_values{ ( array_run | ArrayRunBitmap{ std::vector<std::uint32_t>{ 5U, 6U, 7U } } ).to_vector() };
    EXPECT_EQ( union_values.front(), 5U );
    EXPECT_TRUE( std::ranges::binary_search( union_values, 100U ) );
}

TEST(FrsrRoaringSmoke, ClosedRangeOpsPreferRunStorageForContiguousRanges) {
    TestBitmap bitmap;
    bitmap.add_closed_range( 100U, 4'500U );

    EXPECT_EQ( bitmap.size(), 4'401U );
    EXPECT_TRUE( bitmap.contains( 100U ) );
    EXPECT_TRUE( bitmap.contains( 4'500U ) );

    auto const original_byte_size{ bitmap.byte_size() };
    bitmap.optimize_for_storage();

    EXPECT_LE( bitmap.byte_size(), original_byte_size );
    EXPECT_EQ( bitmap.container_count(), 1U );

    bitmap.flip_closed_range( 250U, 350U );
    EXPECT_FALSE( bitmap.contains( 300U ) );
    EXPECT_TRUE( bitmap.contains( 200U ) );

    bitmap.remove_closed_range( 100U, 149U );
    EXPECT_FALSE( bitmap.contains( 100U ) );
    EXPECT_TRUE( bitmap.contains( 150U ) );
}

TEST(FrsrRoaringSmoke, BulkOrFinishersPreserveUnionSemantics) {
    TestBitmap const lhs{ std::vector<std::uint32_t>{ 1U, 2U, 3U, 4'096U, 4'097U } };
    TestBitmap const rhs{ std::vector<std::uint32_t>{ 3U, 5U, 4'096U, 8'192U } };
    auto const expected{ ( lhs | rhs ).to_vector() };

    TestBitmap finish_storage;
    finish_storage.bulk_or_intermediate( lhs );
    finish_storage.bulk_or_intermediate( rhs );
    finish_storage.bulk_or_finish();
    EXPECT_EQ( finish_storage.to_vector(), expected );

    TestBitmap keep_bitsets;
    keep_bitsets.bulk_or_intermediate( lhs );
    keep_bitsets.bulk_or_intermediate( rhs );
    keep_bitsets.bulk_or_finish_keep_bitsets();
    EXPECT_EQ( keep_bitsets.to_vector(), expected );
}

// The array→bitset scatter picks between a per-value and a word-grouped loop
// from the array's span; both must produce identical bits, and the choice must
// not depend on which side of the threshold the OPERANDS sit on — only the array
// being scattered. Cases below straddle it in both directions, inside one chunk
// and across several, so a mis-taken branch cannot pass by luck.
TEST(FrsrRoaringSmoke, ArrayIntoBitsetScatterAgreesAcrossBothDensityRegimes) {
    auto const fold = []( std::vector<std::uint32_t> const & seed,
                          std::vector<std::uint32_t> const & scattered ) {
        TestBitmap const base{ seed };
        TestBitmap const operand{ scattered };
        TestBitmap accumulator;
        accumulator.bulk_or_intermediate( base );
        accumulator.bulk_or_intermediate( operand );
        accumulator.bulk_or_repair_after_lazy();
        return std::pair{ accumulator.to_vector(), ( base | operand ).to_vector() };
    };

    // A dense bitset seed forces the operand down the array-into-bitset arm.
    std::vector<std::uint32_t> dense_seed;
    for ( std::uint32_t i{ 0U }; i < 20'000U; ++i ) { dense_seed.push_back( i * 2U ); }

    for ( std::uint32_t stride : { 1U, 2U, 3U, 7U, 16U, 17U, 64U, 999U } ) {
        std::vector<std::uint32_t> operand;
        for ( std::uint32_t i{ 0U }; i < 3'000U; ++i ) { operand.push_back( i * stride ); }
        auto const [ folded, expected ]{ fold( dense_seed, operand ) };
        EXPECT_EQ( folded, expected ) << "stride " << stride;
    }

    // Degenerate shapes: single value, one full word, and a run that ends exactly
    // on a word boundary (the grouped loop's inner exit conditions).
    for ( auto const & operand : std::vector<std::vector<std::uint32_t>>{
              { 5U },
              { 64U, 65U, 66U },
              [] { std::vector<std::uint32_t> v; for ( std::uint32_t i{ 0U }; i < 64U; ++i ) { v.push_back( i ); } return v; }(),
              [] { std::vector<std::uint32_t> v; for ( std::uint32_t i{ 63U }; i < 130U; ++i ) { v.push_back( i ); } return v; }() } ) {
        auto const [ folded, expected ]{ fold( dense_seed, operand ) };
        EXPECT_EQ( folded, expected );
    }
}

// The bulk-union accumulator holds a bitset whose count is stale until the fold is repaired. An eager
// in-place operation that meets it first adjusts the count by the bits it changes, so it must count the
// words first: otherwise the header turns exact with a wrong value, and the repair that follows trusts it.
// The accumulator is a bitset (two arrays too large to merge as arrays) counted as 3'000 while it holds 6'000.
std::vector<std::uint32_t> const lazy_accumulator_values = [] {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t i{ 0U }; i < 6'000U; ++i ) { values.push_back( i ); }
    return values;
}();

TestBitmap lazy_accumulator() {
    std::vector<std::uint32_t> even_values;
    std::vector<std::uint32_t> odd_values;
    for ( std::uint32_t i{ 0U }; i < 3'000U; ++i ) {
        even_values.push_back( i * 2U );
        odd_values .push_back( i * 2U + 1U );
    }
    TestBitmap const evens{ even_values };
    TestBitmap const odds { odd_values  };
    TestBitmap accumulator;
    accumulator.bulk_or_intermediate( evens );
    accumulator.bulk_or_intermediate( odds  );
    return accumulator;
}

std::vector<std::uint32_t> sparse_values( std::uint32_t const first, std::uint32_t const step ) {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t i{ 0U }; i < 100U; ++i ) { values.push_back( first + i * step ); }
    return values;
}

TEST(FrsrRoaringSmoke, EagerUnionWithAnArrayOnALazyAccumulatorKeepsTheCardinalityExact) {
    auto const absent{ sparse_values( 7'000U, 3U ) };
    auto accumulator{ lazy_accumulator() };
    accumulator |= TestBitmap{ absent };
    accumulator.bulk_or_repair_after_lazy();
    std::vector<std::uint32_t> expected;
    std::ranges::set_union( lazy_accumulator_values, absent, std::back_inserter( expected ) );
    EXPECT_EQ( accumulator.to_vector(), expected );
    EXPECT_EQ( accumulator.size(), expected.size() );
}

TEST(FrsrRoaringSmoke, EagerDifferenceWithAnArrayOnALazyAccumulatorKeepsTheCardinalityExact) {
    auto const present{ sparse_values( 20U, 7U ) };
    auto accumulator{ lazy_accumulator() };
    accumulator -= TestBitmap{ present };
    accumulator.bulk_or_repair_after_lazy();
    std::vector<std::uint32_t> expected;
    std::ranges::set_difference( lazy_accumulator_values, present, std::back_inserter( expected ) );
    EXPECT_EQ( accumulator.to_vector(), expected );
    EXPECT_EQ( accumulator.size(), expected.size() );
}

TEST(FrsrRoaringSmoke, EagerXorWithAnArrayOnALazyAccumulatorKeepsTheCardinalityExact) {
    auto mixed{ sparse_values( 20U, 7U ) };
    auto const absent{ sparse_values( 7'000U, 3U ) };
    mixed.insert( mixed.end(), absent.begin(), absent.end() );
    std::ranges::sort( mixed );
    auto accumulator{ lazy_accumulator() };
    accumulator ^= TestBitmap{ mixed };
    accumulator.bulk_or_repair_after_lazy();
    std::vector<std::uint32_t> expected;
    std::ranges::set_symmetric_difference( lazy_accumulator_values, mixed, std::back_inserter( expected ) );
    EXPECT_EQ( accumulator.to_vector(), expected );
    EXPECT_EQ( accumulator.size(), expected.size() );
}

TEST(FrsrRoaringSmoke, LazyUnionFinishersExposeExpectedContainerForms) {
    std::vector<std::uint32_t> lhs_values;
    std::vector<std::uint32_t> rhs_values;
    std::vector<std::uint32_t> expected;
    for ( std::uint32_t i{ 0U }; i < 700U; ++i ) {
        lhs_values.push_back( i * 4U );
        rhs_values.push_back( i * 4U + 2U );
        expected.push_back( i * 4U );
        expected.push_back( i * 4U + 2U );
    }
    TestBitmap const lhs{ lhs_values };
    TestBitmap const rhs{ rhs_values };

    auto expect_forms = []( TestBitmap const & bitmap, std::size_t const arrays,
                            std::size_t const runs, std::size_t const bitsets ) {
        auto const stats{ bitmap.statistics() };
        EXPECT_EQ( stats.array_containers, arrays );
        EXPECT_EQ( stats.run_containers, runs );
        EXPECT_EQ( stats.bitset_containers, bitsets );
        EXPECT_EQ( stats.container_count(), 1U );
    };

    TestBitmap lazy;
    lazy.bulk_or_intermediate( lhs );
    lazy.bulk_or_intermediate( rhs );
    expect_forms( lazy, 0U, 0U, 1U );  // 1,400 > the 1,024 lazy-promotion gate

    auto storage_finish{ lazy };
    storage_finish.bulk_or_finish();
    EXPECT_EQ( storage_finish.to_vector(), expected );
    expect_forms( storage_finish, 1U, 0U, 0U );

    auto keep_bitsets{ lazy };
    keep_bitsets.bulk_or_finish_keep_bitsets();
    EXPECT_EQ( keep_bitsets.to_vector(), expected );
    expect_forms( keep_bitsets, 0U, 0U, 1U );

    auto explicit_optimize{ keep_bitsets };
    explicit_optimize.optimize();
    expect_forms( explicit_optimize, 1U, 0U, 0U );

    std::array<TestBitmap const *, 2U> const operands{ &lhs, &rhs };
    TestBitmap or_many;
    or_many.or_many_in_place( operands );
    EXPECT_EQ( or_many.to_vector(), expected );
    expect_forms( or_many, 1U, 0U, 0U );
}

TEST(FrsrRoaringSmoke, InPlaceArrayVsRunCombinesMatchMaterializingOps) {
    // Exercises the in-place array∩run / array\run spine arm (filter_array_run_inplace):
    // array-encoded accumulator, run-encoded rhs, both &= and -= polarities, spans
    // that start before / inside / after runs, and a shared (rc > 1) accumulator.
    TestBitmap array_side;
    for ( std::uint32_t value{ 0U }; value < 12'000U; value += 3U ) {
        std::ignore = array_side.add( value );
    }
    TestBitmap runs_side;
    for ( std::uint32_t run{ 0U }; run < 100U; ++run ) {
        runs_side.add_closed_range( run * 120U + 7U, run * 120U + 66U );
    }
    runs_side.optimize_for_storage();

    // Expected sets computed independently of the library's own combine kernels.
    std::vector<std::uint32_t> expected_and, expected_andnot;
    for ( std::uint32_t value{ 0U }; value < 12'000U; value += 3U ) {
        auto const in_run{ value % 120U >= 7U && value % 120U <= 66U };
        ( in_run ? expected_and : expected_andnot ).push_back( value );
    }

    TestBitmap and_accumulator{ array_side };   // shared containers: rc > 1 on first touch
    and_accumulator &= runs_side;
    EXPECT_EQ( and_accumulator.to_vector(), expected_and );

    TestBitmap subtract_accumulator{ array_side };
    subtract_accumulator -= runs_side;
    EXPECT_EQ( subtract_accumulator.to_vector(), expected_andnot );

    // Sole-referent accumulators (rc == 1) take the alloc-free compaction path.
    TestBitmap sole_and{ array_side & array_side };
    sole_and &= runs_side;
    EXPECT_EQ( sole_and.to_vector(), expected_and );

    TestBitmap sole_subtract{ array_side & array_side };
    sole_subtract -= runs_side;
    EXPECT_EQ( sole_subtract.to_vector(), expected_andnot );
}

TEST(FrsrRoaringSmoke, ArrayVsEmptiedRunChunkCombines) {
    // A 64-bit bitmap erases lazily: a run chunk emptied by removals stays in its slot holding no runs, and the
    // array-vs-run arms of both combine modes meet it as the right operand. Removing from the top leaves the run
    // [65536, 65536] last, so a walk that read the emptied run list would still find 65536 there.
    using WideBitmap = frsr::roaring::bitmap<std::uint64_t>;
    WideBitmap runs;
    runs.add_closed_range( 0U, 99U );
    runs.add_closed_range( 65'536U, 65'635U );
    runs.add_closed_range( 131'072U, 131'171U );
    runs.optimize();
    for ( std::uint64_t value{ 65'635U }; value >= 65'536U; --value ) {
        std::ignore = runs.remove( value );
    }
    ASSERT_EQ( runs.size(), 200U );

    WideBitmap array_side;
    std::ignore = array_side.add( 65'536U );
    std::ignore = array_side.add( 65'700U );
    std::vector<std::uint64_t> const all{ 65'536U, 65'700U };

    EXPECT_TRUE( ( array_side & runs ).empty() );
    EXPECT_EQ( ( array_side - runs ).to_vector(), all );

    WideBitmap and_accumulator{ array_side & array_side };
    and_accumulator &= runs;
    EXPECT_TRUE( and_accumulator.empty() );

    WideBitmap subtract_accumulator{ array_side & array_side };
    subtract_accumulator -= runs;
    EXPECT_EQ( subtract_accumulator.to_vector(), all );
}

// The n-way AND, checked against an ordered set over random operand mixes: arrays, bitsets and runs meeting in
// every order, empty and absent operands, 32- and 64-bit keys. Besides the values, the block form's contract is
// pinned: at most 256 values per block, never across a chunk, cut by count from the start of the chunk's result -
// and therefore identical for the same operands held as arrays and bitsets, run-optimized, or rebuilt.
template <typename Key>
void check_and_many_against_ordered_sets( std::uint64_t const seed, unsigned const trials ) {
    using Bitmap = frsr::roaring::bitmap<Key>;
    std::mt19937_64 rng{ seed };
    auto const pick{ [ & ]( std::uint64_t const n ) { return static_cast<std::uint64_t>( rng() % n ); } };
    constexpr std::uint64_t chunk_span{ 1ULL << 16 };
    constexpr std::uint64_t key_universe{ sizeof( Key ) > 4 ? 14 : 12 };
    // 64-bit keys are spread far apart, and built descending, so the chunk tables are not in order
    auto const chunk_base{ [ & ]( std::uint64_t const chunk ) -> std::uint64_t {
        return sizeof( Key ) > 4 ? ( ( key_universe - 1 - chunk ) << 24 ) * chunk_span : chunk * chunk_span;
    } };

    for ( unsigned trial{ 0 }; trial < trials; ++trial ) {
        auto const operand_count{ 1U + static_cast<unsigned>( pick( 9 ) ) };
        // per chunk key, a pool of low values every operand draws a subset of, and a range every run operand shares
        std::vector<std::vector<std::uint32_t>> pools( key_universe );
        std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges( key_universe );
        for ( std::uint64_t c{ 0 }; c < key_universe; ++c ) {
            auto const size{ 150U + static_cast<std::uint32_t>( pick( 7'000 ) ) };
            for ( std::uint32_t i{ 0 }; i < size; ++i ) { pools[ c ].push_back( static_cast<std::uint32_t>( pick( chunk_span ) ) ); }
            auto const first{ static_cast<std::uint32_t>( pick( 40'000 ) ) };
            ranges[ c ] = { first, first + static_cast<std::uint32_t>( 1 + pick( 20'000 ) ) };
        }
        std::vector<Bitmap> plain( operand_count ), rebuilt( operand_count ), optimized( operand_count );
        std::vector<std::set<std::uint64_t>> sets( operand_count );
        std::vector<bool> absent( operand_count, false );
        for ( unsigned o{ 0 }; o < operand_count; ++o ) {
            absent[ o ] = pick( 12 ) == 0;
            if ( pick( 14 ) == 0 ) { continue; }  // an empty operand
            static constexpr double fractions[]{ 0.04, 0.3, 0.7, 0.95 };
            auto const fraction{ fractions[ pick( 4 ) ] };
            auto const use_ranges{ pick( 3 ) == 0 };
            for ( std::uint64_t c{ 0 }; c < key_universe; ++c ) {
                if ( pick( 5 ) == 0 ) { continue; }  // the operand lacks this chunk
                auto const base{ chunk_base( c ) };
                if ( use_ranges ) {
                    plain[ o ].add_closed_range( static_cast<Key>( base + ranges[ c ].first ), static_cast<Key>( base + ranges[ c ].second ) );
                    for ( auto v{ ranges[ c ].first }; v <= ranges[ c ].second; ++v ) { sets[ o ].insert( base + v ); }
                } else {
                    for ( auto const low : pools[ c ] ) {
                        if ( pick( 1000 ) < static_cast<std::uint64_t>( fraction * 1000 ) ) {
                            std::ignore = plain[ o ].add( static_cast<Key>( base + low ) );
                            sets[ o ].insert( base + low );
                        }
                    }
                }
            }
            std::vector<Key> values;
            for ( auto const v : sets[ o ] ) { values.push_back( static_cast<Key>( v ) ); }
            if constexpr ( sizeof( Key ) <= 4 ) {
                rebuilt[ o ].add_many_sorted( values );
            } else {
                for ( auto const v : values ) { std::ignore = rebuilt[ o ].add( v ); }
            }
            optimized[ o ] = plain[ o ];
            optimized[ o ].optimize();
        }

        std::set<std::uint64_t> expected;
        bool first{ true };
        for ( unsigned o{ 0 }; o < operand_count; ++o ) {
            if ( absent[ o ] ) { continue; }
            if ( first ) { expected = sets[ o ]; first = false; continue; }
            std::set<std::uint64_t> next;
            std::set_intersection( expected.begin(), expected.end(), sets[ o ].begin(), sets[ o ].end(), std::inserter( next, next.end() ) );
            expected = std::move( next );
        }
        if ( first ) { expected.clear(); }  // only absent operands: nothing to intersect

        auto const pointers{ [ & ]( std::vector<Bitmap> const & held ) {
            std::vector<Bitmap const *> out;
            for ( unsigned o{ 0 }; o < operand_count; ++o ) { out.push_back( absent[ o ] ? nullptr : &held[ o ] ); }
            return out;
        } };
        auto const expected_vector{ std::vector<std::uint64_t>( expected.begin(), expected.end() ) };
        auto const blocks_of{ [ & ]( std::vector<Bitmap> const & held ) {
            std::vector<Bitmap const *> const named{ pointers( held ) };
            auto const before{ named };
            std::vector<std::size_t> sizes;
            std::vector<std::uint64_t> flat;
            std::uint64_t previous_chunk{ 0 };
            std::size_t   in_chunk{ 0 };
            bool          any{ false };
            Bitmap::and_many_for_each_block( std::span<Bitmap const * const>{ named }, [ & ]( std::span<Key const> const block ) {
                EXPECT_FALSE( block.empty() );
                EXPECT_LE( block.size(), 256U );
                auto const chunk{ static_cast<std::uint64_t>( block.front() ) / chunk_span };
                for ( auto const v : block ) {
                    EXPECT_EQ( static_cast<std::uint64_t>( v ) / chunk_span, chunk ) << "a block crossed a chunk";
                    flat.push_back( v );
                }
                if ( any && chunk == previous_chunk ) {
                    EXPECT_EQ( in_chunk % 256U, 0U ) << "a block ended early inside a chunk";
                } else {
                    in_chunk = 0;
                }
                in_chunk += block.size();
                previous_chunk = chunk;
                any = true;
                sizes.push_back( block.size() );
            } );
            EXPECT_EQ( named, before ) << "the operand span was modified";
            if ( flat != expected_vector ) {
                std::size_t at{ 0 };
                while ( at < flat.size() && at < expected_vector.size() && flat[ at ] == expected_vector[ at ] ) { ++at; }
                ADD_FAILURE() << "block values differ, trial " << trial << ": first difference at index " << at << " of " << flat.size() << " vs " << expected_vector.size()
                              << ", got " << ( at < flat.size() ? std::to_string( flat[ at ] ) : std::string{ "end" } )
                              << ", expected " << ( at < expected_vector.size() ? std::to_string( expected_vector[ at ] ) : std::string{ "end" } )
                              << ", operands " << operand_count;
            }
            return sizes;
        } };
        auto const blocks_plain{ blocks_of( plain ) };
        EXPECT_EQ( blocks_of( rebuilt ), blocks_plain ) << "the block cuts depend on how the operands were built, trial " << trial;
        EXPECT_EQ( blocks_of( optimized ), blocks_plain ) << "the block cuts depend on run-optimizing, trial " << trial;

        for ( auto const * held : { &plain, &rebuilt, &optimized } ) {
            auto const named{ pointers( *held ) };
            std::span<Bitmap const * const> const operands{ named };
            auto const combined{ Bitmap::and_many( operands ) };
            EXPECT_EQ( combined.validate( true ), nullptr ) << "and_many result, trial " << trial;  // canonical: a bitset only above the array limit
            for ( auto const * operand : named ) { if ( operand != nullptr ) { EXPECT_EQ( operand->validate(), nullptr ) << "operand, trial " << trial; } }
            auto const decoded{ combined.to_vector() };
            EXPECT_EQ( std::vector<std::uint64_t>( decoded.begin(), decoded.end() ), expected_vector ) << "and_many, trial " << trial;
            EXPECT_EQ( combined.size(), expected.size() );
            EXPECT_EQ( Bitmap::and_many_cardinality( operands ), expected.size() ) << "trial " << trial;
            EXPECT_EQ( Bitmap::and_many_intersects( operands ), !expected.empty() ) << "trial " << trial;
            std::vector<std::uint64_t> each;
            Bitmap::and_many_for_each( operands, [ & ]( Key const v ) { each.push_back( v ); } );
            EXPECT_EQ( each, expected_vector ) << "and_many_for_each, trial " << trial;
            // the same intersection through the extra-operand overload: hold the last live operand back
            for ( unsigned o{ operand_count }; o-- > 0; ) {
                if ( absent[ o ] ) { continue; }
                auto rest{ named };
                rest[ o ] = nullptr;
                EXPECT_EQ( Bitmap::and_many_intersects( std::span<Bitmap const * const>{ rest }, ( *held )[ o ] ), !expected.empty() ) << "trial " << trial;
                break;
            }
        }
    }
}

TEST(FrsrRoaringSmoke, AndManyMatchesAnOrderedSetForEveryOperandMix) {
    check_and_many_against_ordered_sets<std::uint32_t>( 0xA11D'0001ULL, 60 );
}

TEST(FrsrRoaringSmoke, AndManyOverWideKeysBuiltOutOfOrder) {
    check_and_many_against_ordered_sets<std::uint64_t>( 0xA11D'0002ULL, 20 );
}

TEST(FrsrRoaringSmoke, AndManyOfNothingAndOfOneOperand) {
    using Bitmap = frsr::roaring::bitmap<std::uint32_t>;
    Bitmap one;
    for ( std::uint32_t v{ 5 }; v < 70'000; v += 3 ) { std::ignore = one.add( v ); }
    std::vector<Bitmap const *> none;
    EXPECT_TRUE( Bitmap::and_many( none ).empty() );
    EXPECT_EQ( Bitmap::and_many_cardinality( none ), 0U );
    EXPECT_FALSE( Bitmap::and_many_intersects( none ) );
    std::vector<Bitmap const *> only_null{ nullptr, nullptr };
    EXPECT_TRUE( Bitmap::and_many( only_null ).empty() );
    // a lone operand of each container kind is walked as it lies
    Bitmap runs;
    runs.add_closed_range( 3U, 9'000U );
    runs.add_closed_range( 131'072U + 500U, 131'072U + 40'000U );
    Bitmap arrays;
    for ( std::uint32_t v{ 7 }; v < 3'000'000U; v += 40'009U ) { std::ignore = arrays.add( v ); }
    for ( auto const * lone : { &one, &runs, &arrays } ) {
        std::vector<Bitmap const *> single{ lone };
        std::vector<std::uint32_t> got;
        Bitmap::and_many_for_each( single, [ & ]( std::uint32_t const v ) { got.push_back( v ); } );
        auto const want{ lone->to_vector() };
        EXPECT_EQ( got, std::vector<std::uint32_t>( want.begin(), want.end() ) );
        EXPECT_EQ( Bitmap::and_many_cardinality( single ), lone->size() );
    }
    std::vector<Bitmap const *> alone{ &one };
    EXPECT_EQ( Bitmap::and_many( alone ).to_vector(), one.to_vector() );
    EXPECT_EQ( Bitmap::and_many_cardinality( alone ), one.size() );
    EXPECT_TRUE( Bitmap::and_many_intersects( alone ) );
    Bitmap empty;
    std::vector<Bitmap const *> with_empty{ &one, &empty };
    EXPECT_TRUE( Bitmap::and_many( with_empty ).empty() );
    EXPECT_FALSE( Bitmap::and_many_intersects( with_empty ) );
}

// Run-heavy operands in every kind pairing and both orders, through each binary operator and its in-place
// form: the values must match an ordered set and the result must be a legal bitmap - above all, no array
// may hold more values than the array limit, since the portable format tells an array from a bitset by
// cardinality alone, and the result kind must not depend on the order the operands meet in.
template <typename Bitmap>
void check_run_heavy_pairings() {
    constexpr std::uint32_t chunk{ 1U << 16 };
    enum class shape { long_run, short_runs, dense_bitset, barely_bitset, sparse_array, full_run, empty };
    constexpr shape shapes[]{ shape::long_run, shape::short_runs, shape::dense_bitset, shape::barely_bitset, shape::sparse_array, shape::full_run, shape::empty };
    // one operand per shape; each holds the same two chunks (3 and 4) so every pairing meets in both
    auto const build{ [ & ]( shape const kind, std::uint64_t const seed, std::set<std::uint32_t> & values ) {
        std::mt19937_64 rng{ seed };
        Bitmap bitmap;
        for ( std::uint32_t c : { 3U, 4U } ) {
            auto const base{ c * chunk };
            auto const add_range{ [ & ]( std::uint32_t const from, std::uint32_t const to ) {
                bitmap.add_closed_range( base + from, base + to );
                for ( auto v{ from }; v <= to; ++v ) { values.insert( base + v ); }
            } };
            auto const add_random{ [ & ]( std::uint32_t const count, std::uint32_t const range ) {
                for ( std::uint32_t i{ 0 }; i < count; ++i ) {
                    auto const low{ static_cast<std::uint32_t>( rng() % range ) };
                    std::ignore = bitmap.add( base + low );
                    values.insert( base + low );
                }
            } };
            switch ( kind ) {
                case shape::long_run     : add_range( 1'000U + c * 100U, 20'000U + c * 100U ); break;                          // 19,001 values, one run
                case shape::short_runs   : add_range( 100U, 300U ); add_range( 900U, 1'200U ); add_range( 40'000U, 40'500U ); break;
                case shape::dense_bitset : add_random( 40'000U, chunk ); break;
                case shape::barely_bitset: add_random( 6'500U, chunk ); break;                                                  // just past the array limit
                case shape::sparse_array : add_random( 1'800U, chunk ); break;
                case shape::full_run     : add_range( 0U, chunk - 1U ); break;
                case shape::empty        : break;
            }
        }
        return bitmap;
    } };

    auto const reference{ []( std::set<std::uint32_t> const & a, std::set<std::uint32_t> const & b, char const op ) {
        std::set<std::uint32_t> out;
        switch ( op ) {
            case '&': std::set_intersection( a.begin(), a.end(), b.begin(), b.end(), std::inserter( out, out.end() ) ); break;
            case '-': std::set_difference  ( a.begin(), a.end(), b.begin(), b.end(), std::inserter( out, out.end() ) ); break;
            default : std::set_union       ( a.begin(), a.end(), b.begin(), b.end(), std::inserter( out, out.end() ) ); break;
        }
        return out;
    } };
    auto const check{ [ & ]( Bitmap const & result, std::set<std::uint32_t> const & want, char const * const what ) {
        // legal, not canonical: a pairwise AND keeps a sparse bitset result a bitset on purpose (demoting a fold
        // accumulator costs the dense kernels of every later fold step), so only the array limit is asserted here
        EXPECT_EQ( result.validate(), nullptr ) << what;
        auto const got{ result.to_vector() };
        EXPECT_EQ( std::vector<std::uint32_t>( got.begin(), got.end() ), std::vector<std::uint32_t>( want.begin(), want.end() ) ) << what;
    } };

    unsigned pair_index{ 0 };
    // every pairing as built, and again with each operand storage-optimized (the run form of a run-shaped chunk)
    for ( bool const optimized : { false, true } ) {
    for ( auto const left_shape : shapes ) {
        for ( auto const right_shape : shapes ) {
            std::set<std::uint32_t> left_values, right_values;
            auto left { build( left_shape , 0x1000 + pair_index, left_values  ) };
            auto right{ build( right_shape, 0x2000 + pair_index, right_values ) };
            if ( optimized ) { left.optimize(); right.optimize(); }
            ++pair_index;
            ASSERT_EQ( left.validate(), nullptr );
            ASSERT_EQ( right.validate(), nullptr );
            auto const name{ [ & ]( char const * const op ) { return std::string{ optimized ? "optimized " : "" } + "shapes " + std::to_string( static_cast<int>( left_shape ) ) + op + std::to_string( static_cast<int>( right_shape ) ); } };
            for ( char const op : { '&', '-', '|' } ) {
                auto const want{ reference( left_values, right_values, op ) };
                Bitmap in_place{ left };
                switch ( op ) {
                    case '&': check( left & right, want, name( " & " ).c_str() ); in_place &= right; break;
                    case '-': check( left - right, want, name( " - " ).c_str() ); in_place -= right; break;
                    default : check( left | right, want, name( " | " ).c_str() ); in_place |= right; break;
                }
                check( in_place, want, name( ( std::string{ " " } + op + "= " ).c_str() ).c_str() );
            }
            // the result kind must not depend on the order the operands meet in
            auto const ab{ left & right };
            auto const ba{ right & left };
            auto const stats_ab{ ab.statistics() };
            auto const stats_ba{ ba.statistics() };
            EXPECT_EQ( stats_ab.array_containers , stats_ba.array_containers  ) << name( " & (order)" );
            EXPECT_EQ( stats_ab.bitset_containers, stats_ba.bitset_containers ) << name( " & (order)" );
            EXPECT_EQ( stats_ab.run_containers   , stats_ba.run_containers    ) << name( " & (order)" );
            // and the n-way questions over the same two operands
            Bitmap const * const operands[]{ &left, &right };
            auto const common{ reference( left_values, right_values, '&' ) };
            EXPECT_EQ( Bitmap::and_many_cardinality( operands ), common.size() ) << name( " (and_many_cardinality)" );
            EXPECT_EQ( Bitmap::and_many_intersects( operands ), !common.empty() ) << name( " (and_many_intersects)" );
        }
    }
    }
}

TEST(FrsrRoaringSmoke, RunHeavyPairingsStayLegalAndOrderIndependentByValueSemantics) {
    check_run_heavy_pairings<frsr::roaring::bitmap<std::uint32_t>>();
}

TEST(FrsrRoaringSmoke, RunHeavyPairingsStayLegalAndOrderIndependentWhenRefcountedAndLazy) {
    check_run_heavy_pairings<frsr::roaring::bitmap<
        std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>,
        frsr::roaring::detail::cow_atomic_refcount, frsr::roaring::detail::run_selection_lazy>>();
}

TEST(FrsrRoaringSmoke, WideKeyContainsReadsAMissOffTheChunkIndex) {
    // Past a chunk count of its own, a 64-bit bitmap resolves a membership query through its chunk
    // index map instead of a search. A chunk the valid map does not hold is absent, so that is the
    // answer - this pins it across the mutations that build, extend and invalidate the map, where a
    // stale "absent" would be a wrong answer rather than a slow one.
    using WideBitmap = frsr::roaring::bitmap<std::uint64_t>;
    constexpr std::uint64_t chunk_span { 1ULL << 16 };
    constexpr std::uint64_t chunk_count{ 900 };  // past the map's chunk-count threshold
    WideBitmap bitmap;
    std::set<std::uint64_t> reference;
    auto const insert{ [ & ]( std::uint64_t const value ) { std::ignore = bitmap.add   ( value ); reference.insert( value ); } };
    auto const erase { [ & ]( std::uint64_t const value ) { std::ignore = bitmap.remove( value ); reference.erase ( value ); } };
    // only even chunks, so the odd ones are the misses
    for ( std::uint64_t chunk{ 0 }; chunk < chunk_count; ++chunk ) {
        insert( 2 * chunk * chunk_span +     7 );
        insert( 2 * chunk * chunk_span + 4'242 );
    }
    auto const probe_every_chunk{ [ & ]( char const * const stage ) {
        for ( std::uint64_t chunk{ 0 }; chunk < 2 * chunk_count + 4; ++chunk ) {
            for ( std::uint64_t const low : { std::uint64_t{ 7 }, std::uint64_t{ 4'242 }, std::uint64_t{ 9'999 } } ) {
                auto const value{ chunk * chunk_span + low };
                EXPECT_EQ( bitmap.contains( value ), reference.contains( value ) ) << stage << ", value " << value;
            }
        }
    } };
    probe_every_chunk( "as built" );
    for ( std::uint64_t chunk{ 0 }; chunk < 40; ++chunk ) {
        erase( 2 * chunk * chunk_span +     7 );
        erase( 2 * chunk * chunk_span + 4'242 );
    }
    probe_every_chunk( "after emptying chunks" );
    // Past half the chunks emptied, every later chunk has moved down.
    for ( std::uint64_t chunk{ 40 }; chunk < 480; ++chunk ) {
        erase( 2 * chunk * chunk_span +     7 );
        erase( 2 * chunk * chunk_span + 4'242 );
    }
    probe_every_chunk( "after the long erase run" );
    for ( std::uint64_t chunk{ 0 }; chunk < 30; ++chunk ) { insert( ( 2 * chunk + 1 ) * chunk_span + 11 ); }
    probe_every_chunk( "after filling gaps that were misses" );
    bitmap.optimize();
    probe_every_chunk( "after optimize" );
}

TEST(FrsrRoaringSmoke, WideKeyChunksOutOfOrderStillReadAndCombineCorrectly) {
    // A 64-bit bitmap appends new chunks in arrival order and remembers whether that order
    // is still ascending, because the paths that sort, search and rebuild the chunk index
    // act on that answer. Build one the wrong way round - descending chunks, then a chunk
    // dropped in between - and pin that reads and set operations still agree with a plain
    // ordered set. A bitmap that believed itself sorted here would binary search a table
    // that is not, and answer wrongly rather than slowly.
    using WideBitmap = frsr::roaring::bitmap<std::uint64_t>;
    constexpr std::uint64_t chunk_span{ 1ULL << 16 };
    WideBitmap bitmap;
    std::set<std::uint64_t> reference;
    auto const insert{ [ & ]( std::uint64_t const value ) { std::ignore = bitmap.add( value ); reference.insert( value ); } };
    for ( std::uint64_t chunk{ 600 }; chunk > 0; --chunk ) {          // descending: every append breaks the order
        insert( chunk * chunk_span + 5 );
        insert( chunk * chunk_span + 4'000 );
    }
    for ( std::uint64_t chunk : { 250ULL, 1'000ULL, 3ULL, 777ULL } ) { // and now out of order among themselves
        insert( chunk * chunk_span + 9 );
    }
    EXPECT_EQ( bitmap.size(), reference.size() );
    EXPECT_EQ( bitmap.to_vector(), std::vector<std::uint64_t>( reference.begin(), reference.end() ) );
    for ( std::uint64_t chunk{ 0 }; chunk <= 1'001; ++chunk ) {
        for ( std::uint64_t const low : { std::uint64_t{ 5 }, std::uint64_t{ 9 }, std::uint64_t{ 4'000 }, std::uint64_t{ 61'234 } } ) {
            auto const value{ chunk * chunk_span + low };
            EXPECT_EQ( bitmap.contains( value ), reference.contains( value ) ) << "value " << value;
        }
    }
    WideBitmap every_other;
    std::set<std::uint64_t> every_other_reference;
    for ( std::uint64_t chunk{ 0 }; chunk <= 1'000; chunk += 2 ) {
        std::ignore = every_other.add( chunk * chunk_span + 5 );
        every_other_reference.insert( chunk * chunk_span + 5 );
    }
    std::set<std::uint64_t> intersection_reference;
    std::set_intersection( reference.begin(), reference.end(), every_other_reference.begin(), every_other_reference.end(),
                           std::inserter( intersection_reference, intersection_reference.end() ) );
    auto const intersection{ bitmap & every_other };
    EXPECT_EQ( intersection.to_vector(), std::vector<std::uint64_t>( intersection_reference.begin(), intersection_reference.end() ) );
    std::set<std::uint64_t> difference_reference;
    std::set_difference( reference.begin(), reference.end(), every_other_reference.begin(), every_other_reference.end(),
                         std::inserter( difference_reference, difference_reference.end() ) );
    auto subtracted{ bitmap };
    subtracted -= every_other;
    EXPECT_EQ( subtracted.to_vector(), std::vector<std::uint64_t>( difference_reference.begin(), difference_reference.end() ) );
}

TEST(FrsrRoaringSmoke, RunAccumulatorAndBitsetCombineArmsMatchExpectations) {
    // Exercises the dedicated combine-spine arms for run-encoded accumulators
    // (run∩array, run∩bitset — the dominant fold pairs when the accumulator is a
    // shallow copy of a storage-optimize()d bitmap) and the in-place bitset\array
    // arm. Expected sets computed independently of the library's combine kernels.
    TestBitmap run_seed;
    for ( std::uint32_t run{ 0U }; run < 100U; ++run ) {
        run_seed.add_closed_range( run * 120U + 7U, run * 120U + 66U );
    }
    run_seed.optimize_for_storage();

    TestBitmap array_operand;    // sparse: stays array-encoded
    for ( std::uint32_t value{ 0U }; value < 12'000U; value += 3U ) {
        std::ignore = array_operand.add( value );
    }
    TestBitmap bitset_operand;   // dense: crosses the array→bitset threshold
    for ( std::uint32_t value{ 0U }; value < 12'000U; value += 2U ) {
        std::ignore = bitset_operand.add( value );
    }

    auto const in_run{ []( std::uint32_t const value ) {
        return value % 120U >= 7U && value % 120U <= 66U;
    } };
    std::vector<std::uint32_t> expected_run_and_array, expected_run_and_bitset, expected_bitset_minus_array;
    for ( std::uint32_t value{ 0U }; value < 12'000U; ++value ) {
        if ( in_run( value ) && value % 3U == 0U ) { expected_run_and_array.push_back( value ); }
        if ( in_run( value ) && value % 2U == 0U ) { expected_run_and_bitset.push_back( value ); }
        if ( value % 2U == 0U && value % 3U != 0U ) { expected_bitset_minus_array.push_back( value ); }
    }

    TestBitmap run_and_array{ run_seed };    // shared containers (rc > 1), run-left
    run_and_array &= array_operand;
    EXPECT_EQ( run_and_array.to_vector(), expected_run_and_array );

    TestBitmap run_and_bitset{ run_seed };
    run_and_bitset &= bitset_operand;
    EXPECT_EQ( run_and_bitset.to_vector(), expected_run_and_bitset );

    TestBitmap bitset_minus_array{ bitset_operand };
    bitset_minus_array -= array_operand;
    EXPECT_EQ( bitset_minus_array.to_vector(), expected_bitset_minus_array );
}

TEST(FrsrRoaringSmoke, CopyMutationsStayIsolatedAcrossArrayRunAndBitsetChunks) {
    TestBitmap source;
    source.add_many_sorted( std::vector<std::uint32_t>{ 1U, 3U, 5U, 7U } );
    source.add_closed_range( 65'536U + 100U, 65'536U + 220U );

    std::vector<std::uint32_t> dense_values;
    dense_values.reserve( TestBitmap::array_to_bitset_threshold );
    for ( std::uint32_t index{ 0U }; index < TestBitmap::array_to_bitset_threshold; ++index ) {
        dense_values.push_back( 131'072U + index * 2U );
    }
    source.add_many_sorted( dense_values );
    source.optimize_for_speed();

    auto const baseline{ source.to_vector() };
    TestBitmap copy{ source };

    copy.add_closed_range( 0U, 4'095U );
    copy.remove_closed_range( 65'536U + 150U, 65'536U + 160U );
    copy.remove_many_sorted( std::span{ dense_values }.first( dense_values.size() / 2U ) );
    copy.optimize_for_storage();

    EXPECT_EQ( source.to_vector(), baseline );

    EXPECT_FALSE( source.contains( 1'024U ) );
    EXPECT_TRUE( copy.contains( 1'024U ) );

    EXPECT_TRUE( source.contains( 65'536U + 155U ) );
    EXPECT_FALSE( copy.contains( 65'536U + 155U ) );

    EXPECT_TRUE( source.contains( dense_values[ 10 ] ) );
    EXPECT_FALSE( copy.contains( dense_values[ 10 ] ) );
    EXPECT_TRUE( copy.contains( dense_values.back() ) );
}

TEST(FrsrRoaringSmoke, ConcurrentCopiesOfOneSourceAreSafe) {
    TestBitmap source;
    source.add_closed_range( 0U, 8'192U );
    source.remove_closed_range( 512U, 768U );

    std::atomic<bool> failed{ false };
    auto const copy_task = [&] {
        for ( int iteration{ 0 }; iteration < 1'000; ++iteration ) {
            TestBitmap copy{ source };
            if ( copy.size() != source.size() || copy.front() != source.front() || copy.back() != source.back() ) {
                failed.store( true, std::memory_order_relaxed );
                return;
            }
        }
    };

    std::jthread t1{ copy_task };
    std::jthread t2{ copy_task };
    std::jthread t3{ copy_task };

    EXPECT_FALSE( failed.load( std::memory_order_relaxed ) );
}

// Builds a bitmap spanning all three container representations (a sparse array
// chunk, a dense bitset chunk, and a contiguous run chunk after optimisation) so
// the serialization round-trips exercise every frozen_container_kind path.
[[nodiscard]] TestBitmap make_mixed_container_bitmap() {
    TestBitmap bitmap;
    for ( std::uint32_t const value : { 1U, 7U, 100U, 4'096U, 60'000U } ) {
        bitmap.add( value );
    }
    bitmap.add_closed_range( 65'536U, 70'536U );
    bitmap.add_closed_range( 131'072U, 134'072U );
    bitmap.optimize_for_storage();
    return bitmap;
}

TEST(FrsrRoaringSmoke, SerializePortableRoundTripPreservesValues) {
    auto const source{ make_mixed_container_bitmap() };
    auto const baseline{ source.to_vector() };

    TestBitmap::serialized_byte_vector buffer;
    source.serialize_to_vm_vector( buffer );
    auto const restored{ TestBitmap::deserialize_from_vm_vector( buffer ) };

    EXPECT_EQ( restored.size(), source.size() );
    EXPECT_EQ( restored.to_vector(), baseline );
}

TEST(FrsrRoaringSmoke, SerializeFrozenViewAndMaterializeRoundTrip) {
    auto const source{ make_mixed_container_bitmap() };
    auto const baseline{ source.to_vector() };

    TestBitmap::serialized_byte_vector buffer;
    source.serialize_frozen_to_vm_vector( buffer );

    auto const view{ TestBitmap::frozen_view_from_vm_vector( buffer ) };
    ASSERT_TRUE( static_cast<bool>( view ) );
    EXPECT_EQ( view.size(), source.size() );

    // Frozen-view membership must match the live bitmap across hits and misses in
    // each container kind (array / bitset / run) plus out-of-range keys.
    for ( std::uint32_t const value : {
        0U, 1U, 2U, 7U, 100U, 4'096U, 4'097U, 60'000U,
        65'535U, 65'536U, 67'000U, 70'536U, 70'537U,
        131'071U, 131'072U, 132'500U, 134'072U, 134'073U, 200'000U
    } ) {
        EXPECT_EQ( view.contains( value ), source.contains( value ) ) << "value " << value;
    }

    auto const materialized{ TestBitmap::deserialize_frozen_from_vm_vector( buffer ) };
    EXPECT_EQ( materialized.size(), source.size() );
    EXPECT_EQ( materialized.to_vector(), baseline );
}

// frozen_serialize() writes every byte of the image, padding included, so the
// destination need not be zero-filled: an image written into a dirty buffer is the
// one the vector path produces, and it reads back to the same set.
template <typename Bitmap>
void check_frozen_into_dirty_buffer( Bitmap const & source, std::set<std::uint64_t> const & oracle ) {
    typename Bitmap::serialized_byte_vector reference;
    source.serialize_frozen_to_vm_vector( reference );
    ASSERT_EQ( source.frozen_size_bytes(), reference.size() );

    std::vector<std::byte> dirty( reference.size() + 16, std::byte{ 0xA5 } );
    ASSERT_EQ( source.frozen_serialize( std::span{ dirty }.first( reference.size() ) ), reference.size() );
    EXPECT_TRUE( std::equal( reference.begin(), reference.end(), dirty.begin() ) );
    // Nothing past the image is touched.
    EXPECT_TRUE( std::all_of( dirty.begin() + static_cast<std::ptrdiff_t>( reference.size() ), dirty.end(), []( std::byte const b ) { return b == std::byte{ 0xA5 }; } ) );

    typename Bitmap::serialized_byte_vector copy;
    copy.resize( reference.size() );
    std::copy( dirty.begin(), dirty.begin() + static_cast<std::ptrdiff_t>( reference.size() ), copy.begin() );
    auto const materialized{ Bitmap::deserialize_frozen_from_vm_vector( copy ) };
    auto const values{ materialized.to_vector() };
    EXPECT_TRUE( std::equal( values.begin(), values.end(), oracle.begin(), oracle.end() ) );
}

TEST(FrsrRoaringSmoke, FrozenSerializeIntoDirtyBufferMatchesVectorPath) {
    auto const mixed{ make_mixed_container_bitmap() };
    auto const mixed_values{ mixed.to_vector() };
    check_frozen_into_dirty_buffer( mixed, { mixed_values.begin(), mixed_values.end() } );

    // Sparse 64-bit keys: many one-value array containers, odd-sized payloads
    // before aligned ones (run and bitset) force alignment gaps.
    using Wide = frsr::roaring::bitmap<std::uint64_t>;
    Wide wide;
    std::set<std::uint64_t> oracle;
    auto const add{ [&]( std::uint64_t const value ) {
        wide.add( value );
        oracle.insert( value );
    } };
    for ( std::uint64_t i{ 0 }; i < 300; ++i ) {
        add( ( i << 40 ) | ( i * 977U ) );
    }
    for ( std::uint64_t v{ 0 }; v < 9000; ++v ) {
        add( ( 400ULL << 40 ) + v * 7U );
    }
    for ( std::uint64_t v{ 5000 }; v < 5000 + 20000; ++v ) {
        add( ( 401ULL << 40 ) + v );
    }
    for ( std::uint64_t i{ 0 }; i < 3; ++i ) {
        add( ( 402ULL << 40 ) + i * 100 );
    }
    wide.optimize_for_storage();
    check_frozen_into_dirty_buffer( wide, oracle );
}

TEST(FrsrRoaringSmoke, FrozenViewRejectsAnEmptyContainerEntry) {
    TestBitmap single;
    single.add( 5U );
    TestBitmap::serialized_byte_vector buffer;
    single.serialize_frozen_to_vm_vector( buffer );
    ASSERT_TRUE( static_cast<bool>( TestBitmap::frozen_view_from_vm_vector( buffer ) ) );

    // header: value_count @16; index entry @24: payload_bytes @12, payload_count @16, cardinality @20.
    auto const poke32{ [&]( std::size_t const offset, std::uint32_t const value ) { std::memcpy( buffer.data() + offset, &value, sizeof( value ) ); } };
    std::memset( buffer.data() + 16, 0, sizeof( std::uint64_t ) );
    poke32( 24 + 12, 0 );
    poke32( 24 + 16, 0 );
    poke32( 24 + 20, 0 );
    EXPECT_FALSE( static_cast<bool>( TestBitmap::frozen_view_from_vm_vector( buffer ) ) );
}

// optimize_run_shaped() is exercised on a run_selection_lazy instantiation:
// under the eager policy ordinary construction would already pick run storage
// for contiguous input, leaving the selective pass nothing to decide.
using LazyBitmap = frsr::roaring::bitmap<
    std::uint32_t,
    frsr::roaring::default_container_set<std::uint32_t>,
    frsr::roaring::detail::cow_atomic_refcount,
    frsr::roaring::detail::run_selection_lazy
>;

// `runs` maximal runs of `run_length` values each, separated by `gap` absent
// values, all within the chunk at `base`.
[[nodiscard]] std::vector<std::uint32_t> run_shaped_values(
    std::uint32_t const base, std::uint32_t const runs, std::uint32_t const run_length, std::uint32_t const gap
) {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t run{ 0U }; run < runs; ++run ) {
        auto const start{ base + run * ( run_length + gap ) };
        for ( std::uint32_t offset{ 0U }; offset < run_length; ++offset ) {
            values.push_back( start + offset );
        }
    }
    return values;
}

TEST(FrsrRoaringSmoke, OptimizeRunShapedConvertsOnlyContainersAtOrAboveTheMeanRunLength) {
    constexpr std::uint32_t chunk{ 65'536U };
    constexpr std::size_t   min_mean_run_length{ 64U };

    // chunk 0: one contiguous block, array-sized          -> converts
    // chunk 1: one contiguous block, bitset-sized         -> converts
    // chunk 2: every other value, array-sized             -> stays an array
    // chunk 3: every other value, bitset-sized            -> stays a bitset
    // chunk 4: runs of exactly 64 (mean == threshold)     -> converts
    // chunk 5: runs of 63 (mean just below the threshold) -> stays an array
    std::vector<std::uint32_t> values;
    for ( auto const & part : {
        run_shaped_values( 0U * chunk,     1U,   600U, 0U ),
        run_shaped_values( 1U * chunk,     1U, 6'000U, 0U ),
        run_shaped_values( 2U * chunk,   600U,     1U, 1U ),
        run_shaped_values( 3U * chunk, 6'000U,     1U, 1U ),
        run_shaped_values( 4U * chunk,    50U,    64U, 1U ),
        run_shaped_values( 5U * chunk,    50U,    63U, 2U ),
    } ) {
        values.insert( values.end(), part.begin(), part.end() );
    }

    LazyBitmap bitmap{ values };
    {
        auto const before{ bitmap.statistics() };
        EXPECT_EQ( before.run_containers,    0U );
        EXPECT_EQ( before.array_containers,  4U );
        EXPECT_EQ( before.bitset_containers, 2U );
    }

    bitmap.optimize_run_shaped( min_mean_run_length );

    auto const after{ bitmap.statistics() };
    EXPECT_EQ( after.run_containers,    3U );
    EXPECT_EQ( after.array_containers,  2U );
    EXPECT_EQ( after.bitset_containers, 1U );
    EXPECT_EQ( bitmap.to_vector(), values );

    // Idempotent: run containers already satisfy the test, the rest was rejected once.
    bitmap.optimize_run_shaped( min_mean_run_length );
    auto const again{ bitmap.statistics() };
    EXPECT_EQ( again.run_containers,    3U );
    EXPECT_EQ( again.array_containers,  2U );
    EXPECT_EQ( again.bitset_containers, 1U );
}

TEST(FrsrRoaringSmoke, OptimizeRunShapedOnACopyLeavesTheSharedSourceUntouched) {
    auto const values{ run_shaped_values( 0U, 1U, 6'000U, 0U ) };  // above the array→bitset threshold
    LazyBitmap const source{ values };
    ASSERT_EQ( source.statistics().bitset_containers, 1U );

    LazyBitmap copy{ source };
    copy.optimize_run_shaped( 64U );

    EXPECT_EQ( copy  .statistics().run_containers,    1U );
    EXPECT_EQ( source.statistics().bitset_containers, 1U );
    EXPECT_EQ( source.statistics().run_containers,    0U );
    EXPECT_EQ( copy  .to_vector(), values );
    EXPECT_EQ( source.to_vector(), values );
}

// The listed pass: optimize_run_shaped( bar, touched_chunks ) over every key width and copy-on-write policy.
template <typename Bitmap>
class TouchedChunksTyped : public ::testing::Test {};

namespace detail_test = frsr::roaring::detail;

template <typename Key, typename Cow>
using LazyOf = frsr::roaring::bitmap<
    Key,
    frsr::roaring::default_container_set<Key>,
    Cow,
    detail_test::run_selection_lazy
>;

using TouchedChunksBitmaps = ::testing::Types<
    LazyOf<std::uint32_t, detail_test::cow_value_semantics>,
    LazyOf<std::uint32_t, detail_test::cow_atomic_refcount>,
    LazyOf<std::uint32_t, detail_test::cow_unsynchronized_refcount>,
    LazyOf<std::uint64_t, detail_test::cow_value_semantics>,
    LazyOf<std::uint64_t, detail_test::cow_atomic_refcount>,
    LazyOf<std::uint64_t, detail_test::cow_unsynchronized_refcount>
>;
TYPED_TEST_SUITE( TouchedChunksTyped, TouchedChunksBitmaps );

constexpr std::size_t listed_bar{ 64U };

enum class chunk_shape : std::uint8_t {
    run_array,         // one run of 600 values: an array that qualifies
    run_bitset,        // one run of 6000 values: a bitset that qualifies
    scattered_array,   // every other value, 600 of them: stays an array
    scattered_bitset,  // every other value, 6000 of them: stays a bitset
    short_run_array,   // one run of 100 values (cheap to build in bulk)
    short_scattered,   // every other value, 100 of them
};

template <typename Bitmap>
void add_chunk( std::vector<typename Bitmap::key_type> & values, std::uint64_t const chunk, chunk_shape const shape ) {
    std::vector<std::uint32_t> lows;
    switch ( shape ) {
        case chunk_shape::run_array:        lows = run_shaped_values( 0U,   1U,   600U, 0U ); break;
        case chunk_shape::run_bitset:       lows = run_shaped_values( 0U,   1U, 6'000U, 0U ); break;
        case chunk_shape::scattered_array:  lows = run_shaped_values( 0U, 600U,     1U, 1U ); break;
        case chunk_shape::scattered_bitset: lows = run_shaped_values( 0U, 6'000U,   1U, 1U ); break;
        case chunk_shape::short_run_array:  lows = run_shaped_values( 0U,   1U,   100U, 0U ); break;
        case chunk_shape::short_scattered:  lows = run_shaped_values( 0U, 100U,     1U, 1U ); break;
    }
    for ( auto const low : lows ) {
        values.push_back( static_cast<typename Bitmap::key_type>( ( chunk << 16 ) + low ) );
    }
}

// The first value of each of the chunks (distinct, ascending or not).
template <typename Bitmap>
[[nodiscard]] std::vector<typename Bitmap::key_type> first_values_of( std::initializer_list<std::uint64_t> const chunks ) {
    std::vector<typename Bitmap::key_type> values;
    for ( auto const chunk : chunks ) {
        values.push_back( static_cast<typename Bitmap::key_type>( chunk << 16 ) );
    }
    return values;
}

template <typename Bitmap>
[[nodiscard]] typename Bitmap::touched_chunks touched_of( std::initializer_list<std::uint64_t> const chunks ) {
    auto values{ first_values_of<Bitmap>( chunks ) };
    std::ranges::sort( values );
    typename Bitmap::touched_chunks touched;
    touched.add_values( values );
    return touched;
}

// The allocations the calling thread makes while `f` runs (the test binary's operator new counts them while a
// thread is armed; without it, the answer is always 0 and the checks that use it are void).
template <typename F>
[[nodiscard]] std::uint32_t allocations_during( F && f ) {
    namespace fault = frsr::roaring::kernel_test::alloc_fault;
    fault::armed const armed{ 0x7fff'ffffU };
    auto const before{ fault::counter };
    f();
    return fault::counter - before;
}

// chunk 0: run array, 1: run bitset, 2: scattered array, 3: short-run array (runs of 64), 4: scattered bitset,
// 5: run array
template <typename Bitmap>
[[nodiscard]] std::vector<typename Bitmap::key_type> mixed_chunks() {
    std::vector<typename Bitmap::key_type> values;
    add_chunk<Bitmap>( values, 0, chunk_shape::run_array );
    add_chunk<Bitmap>( values, 1, chunk_shape::run_bitset );
    add_chunk<Bitmap>( values, 2, chunk_shape::scattered_array );
    {
        auto const lows{ run_shaped_values( 0U, 50U, 64U, 1U ) };
        for ( auto const low : lows ) {
            values.push_back( static_cast<typename Bitmap::key_type>( ( std::uint64_t{ 3 } << 16 ) + low ) );
        }
    }
    add_chunk<Bitmap>( values, 4, chunk_shape::scattered_bitset );
    add_chunk<Bitmap>( values, 5, chunk_shape::run_array );
    return values;
}

TYPED_TEST( TouchedChunksTyped, OnlyTheListedChunksAreConsidered ) {
    using Bitmap = TypeParam;
    auto const values{ mixed_chunks<Bitmap>() };
    Bitmap bitmap{ values };
    auto const before{ bitmap.statistics() };
    ASSERT_EQ( before.run_containers,    0U );
    ASSERT_EQ( before.array_containers,  4U );
    ASSERT_EQ( before.bitset_containers, 2U );

    // 1 and 3 qualify, 4 does not; 0 and 5 qualify but are not listed
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched_of<Bitmap>( { 1, 3, 4 } ) );
    {
        auto const after{ bitmap.statistics() };
        EXPECT_EQ( after.run_containers,    2U );
        EXPECT_EQ( after.array_containers,  3U );
        EXPECT_EQ( after.bitset_containers, 1U );
        EXPECT_EQ( bitmap.to_vector(), values );
    }

    // listing everything makes it the whole pass
    typename Bitmap::touched_chunks all;
    all.add_chunks_of( bitmap );
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, all );
    {
        auto const after{ bitmap.statistics() };
        EXPECT_EQ( after.run_containers,    4U );
        EXPECT_EQ( after.array_containers,  1U );
        EXPECT_EQ( after.bitset_containers, 1U );
        EXPECT_EQ( bitmap.to_vector(), values );
    }

    Bitmap whole{ values };
    whole.optimize_run_shaped( listed_bar );
    EXPECT_EQ( whole.statistics().run_containers, bitmap.statistics().run_containers );
}

TYPED_TEST( TouchedChunksTyped, AnEmptySetDoesNothingAndAllocatesNothing ) {
    using Bitmap = TypeParam;
    auto const values{ mixed_chunks<Bitmap>() };
    Bitmap bitmap{ values };
    typename Bitmap::touched_chunks const none;
    EXPECT_TRUE( none.empty() );
    auto const allocations{ allocations_during( [ & ] { std::as_const( bitmap ).optimize_run_shaped( listed_bar, none ); } ) };
    if ( FRSR_TESTS_COUNT_ALLOCATIONS ) {
        EXPECT_EQ( allocations, 0U );
    }
    EXPECT_EQ( bitmap.statistics().run_containers, 0U );
    EXPECT_EQ( bitmap.to_vector(), values );

    // and an empty bitmap with a listed chunk
    Bitmap nothing;
    std::as_const( nothing ).optimize_run_shaped( listed_bar, touched_of<Bitmap>( { 0, 7 } ) );
    EXPECT_TRUE( nothing.empty() );
}

TYPED_TEST( TouchedChunksTyped, ListedChunksThatDoNotExistAreSkipped ) {
    using Bitmap = TypeParam;
    // chunks 2, 4 and 6 only
    std::vector<typename Bitmap::key_type> values;
    add_chunk<Bitmap>( values, 2, chunk_shape::run_array );
    add_chunk<Bitmap>( values, 4, chunk_shape::run_array );
    add_chunk<Bitmap>( values, 6, chunk_shape::run_array );
    Bitmap bitmap{ values };

    // below the first, between two, and above the last chunk: nothing there, and the walk goes on after each
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched_of<Bitmap>( { 0, 1, 3, 5, 9, 100 } ) );
    EXPECT_EQ( bitmap.statistics().run_containers, 0U );

    // the missing ones around the existing 4: only that one converts
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched_of<Bitmap>( { 1, 3, 4, 5, 100 } ) );
    EXPECT_EQ( bitmap.statistics().run_containers, 1U );

    // the first and the last, the walk past the end stops
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched_of<Bitmap>( { 2, 6, 7, 8 } ) );
    EXPECT_EQ( bitmap.statistics().run_containers, 3U );
    EXPECT_EQ( bitmap.to_vector(), values );
}

TYPED_TEST( TouchedChunksTyped, ARunContainerAndARepeatedPassAreLeftAlone ) {
    using Bitmap = TypeParam;
    auto const values{ mixed_chunks<Bitmap>() };
    Bitmap bitmap{ values };
    auto const touched{ touched_of<Bitmap>( { 0, 1, 2, 3, 4, 5 } ) };
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched );
    auto const once{ bitmap.statistics() };
    auto const allocations{ allocations_during( [ & ] { std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched ); } ) };
    if ( FRSR_TESTS_COUNT_ALLOCATIONS ) {
        EXPECT_EQ( allocations, 0U );
    }
    auto const twice{ bitmap.statistics() };
    EXPECT_EQ( twice.run_containers,    once.run_containers );
    EXPECT_EQ( twice.array_containers,  once.array_containers );
    EXPECT_EQ( twice.bitset_containers, once.bitset_containers );
    EXPECT_EQ( bitmap.to_vector(), values );
}

// A chunk that is not converted is not cloned even when its payload is shared; one that is converted is replaced
// in this bitmap only.
TYPED_TEST( TouchedChunksTyped, ASharedPayloadIsNotClonedUnlessConverted ) {
    using Bitmap = TypeParam;
    // 0: scattered bitset and 1: scattered array (neither qualifies), 2: array of 50 runs, 3: run bitset
    std::vector<typename Bitmap::key_type> values;
    add_chunk<Bitmap>( values, 0, chunk_shape::scattered_bitset );
    add_chunk<Bitmap>( values, 1, chunk_shape::scattered_array );
    for ( auto const low : run_shaped_values( 0U, 50U, 64U, 1U ) ) {   // 50 runs: a run payload that spills
        values.push_back( static_cast<typename Bitmap::key_type>( ( std::uint64_t{ 2 } << 16 ) + low ) );
    }
    add_chunk<Bitmap>( values, 3, chunk_shape::run_bitset );
    Bitmap const source{ values };
    Bitmap copy{ source };

    auto const failing{ touched_of<Bitmap>( { 0, 1 } ) };
    auto const allocations{ allocations_during( [ & ] { std::as_const( copy ).optimize_run_shaped( listed_bar, failing ); } ) };
    if ( FRSR_TESTS_COUNT_ALLOCATIONS ) {
        EXPECT_EQ( allocations, 0U );
    }
    EXPECT_EQ( copy.statistics().run_containers, 0U );

    auto const converting{ touched_of<Bitmap>( { 0, 1, 2, 3 } ) };
    auto const conversion_allocations{ allocations_during( [ & ] { std::as_const( copy ).optimize_run_shaped( listed_bar, converting ); } ) };
    if ( FRSR_TESTS_COUNT_ALLOCATIONS ) {
        EXPECT_GT( conversion_allocations, 0U );   // (the counter is live)
    }
    EXPECT_EQ( copy  .statistics().run_containers, 2U );
    EXPECT_EQ( source.statistics().run_containers, 0U );
    EXPECT_EQ( copy  .to_vector(), values );
    EXPECT_EQ( source.to_vector(), values );
    EXPECT_EQ( copy.size(), source.size() );
}

// The chunk table of a wide key is a tree or, for many evenly spread keys, a flat directory
// (the key shapes below are the ones chunk_tree.cpp pins as the one and the other): the walk is the same on both.
TEST( TouchedChunks, WideKeysOnADirectoryAndOnATree ) {
    using Wide = LazyOf<std::uint64_t, detail_test::cow_atomic_refcount>;
    struct layout {
        char const * name;
        std::vector<std::uint64_t> chunks; // ascending
    };
    std::vector<layout> layouts;
    {
        // evenly spread: the directory
        layout even{ "even", {} };
        for ( std::uint64_t k{ 0 }; k < 20'000; ++k ) {
            even.chunks.push_back( 4096 * k );
        }
        layouts.push_back( std::move( even ) );
    }
    {
        // dense clusters far apart: the tree
        layout clusters{ "clusters", {} };
        for ( std::uint64_t const base : { std::uint64_t{ 0 }, std::uint64_t{ 1 } << 34, std::uint64_t{ 1 } << 40, ( std::uint64_t{ 1 } << 44 ) + 12'345 } ) {
            for ( std::uint64_t k{ 0 }; k < 3'000; ++k ) {
                clusters.chunks.push_back( base + 2 * k );
            }
        }
        layouts.push_back( std::move( clusters ) );
    }
    for ( auto const & table : layouts ) {
        SCOPED_TRACE( table.name );
        // every even-numbered chunk qualifies (one run of 100), every odd one does not
        std::vector<std::uint64_t> values;
        std::size_t qualifying{ 0 };
        for ( std::size_t i{ 0 }; i < table.chunks.size(); ++i ) {
            auto const shape{ ( i % 2 == 0 ) ? chunk_shape::short_run_array : chunk_shape::short_scattered };
            qualifying += ( i % 2 == 0 );
            add_chunk<Wide>( values, table.chunks[ i ], shape );
        }
        Wide const pristine{ values };
        ASSERT_EQ( pristine.statistics().container_count(), table.chunks.size() );

        auto const touch_indices{ [ & ]( std::vector<std::size_t> const & indices ) {
            std::vector<Wide::key_type> firsts;
            for ( auto const i : indices ) {
                firsts.push_back( table.chunks[ i ] << 16 );
            }
            Wide::touched_chunks touched;
            touched.add_values( firsts );
            return touched;
        } };
        auto const converted_after{ [ & ]( Wide::touched_chunks const & touched ) {
            Wide bitmap{ pristine };
            std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched );
            EXPECT_EQ( bitmap.to_vector(), values );
            return bitmap.statistics().run_containers;
        } };

        // one chunk, qualifying and not
        EXPECT_EQ( converted_after( touch_indices( { 100 } ) ), 1U );
        EXPECT_EQ( converted_after( touch_indices( { 101 } ) ), 0U );
        // the first and the last
        EXPECT_EQ( converted_after( touch_indices( { 0, table.chunks.size() - 2 } ) ), 2U );
        // sixteen spread out (every 251st, even and odd alike)
        std::vector<std::size_t> sixteen;
        for ( std::size_t i{ 0 }; i < 16; ++i ) {
            sixteen.push_back( 17 + i * 251 );
        }
        auto const expected{ static_cast<std::size_t>( std::ranges::count_if( sixteen, []( std::size_t const i ) { return i % 2 == 0; } ) ) };
        EXPECT_EQ( converted_after( touch_indices( sixteen ) ), expected );
        // all of them: the full pass
        Wide::touched_chunks all;
        all.add_chunks_of( pristine );
        EXPECT_EQ( converted_after( all ), qualifying );
    }
}

// The set itself: ascending distinct chunks however the adds arrive.
using TouchedBitmap = LazyOf<std::uint32_t, detail_test::cow_atomic_refcount>;
using WideTouchedBitmap = LazyOf<std::uint64_t, detail_test::cow_atomic_refcount>;

template <typename Bitmap>
[[nodiscard]] typename Bitmap::touched_chunks touched_in_steps( std::initializer_list<std::initializer_list<std::uint64_t>> const steps ) {
    typename Bitmap::touched_chunks touched;
    for ( auto const step : steps ) {
        auto values{ first_values_of<Bitmap>( step ) };
        std::ranges::sort( values );
        touched.add_values( values );
    }
    return touched;
}

TEST( TouchedChunks, AddsMergeWhereverTheirChunksLand ) {
    using Touched = TouchedBitmap::touched_chunks;
    Touched const empty;
    EXPECT_TRUE( empty.empty() );
    EXPECT_EQ( empty, Touched{} );

    auto const expect = []( auto const & steps, std::initializer_list<std::uint64_t> const whole ) {
        EXPECT_EQ( touched_in_steps<TouchedBitmap>( steps ), touched_of<TouchedBitmap>( whole ) );
        EXPECT_EQ( touched_in_steps<WideTouchedBitmap>( steps ), touched_of<WideTouchedBitmap>( whole ) );
    };
    // after, below, inside, between, spanning, the same again
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 2 }, { 5, 9 } },                           { 1, 2, 5, 9 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 5, 9 }, { 1, 2 } },                           { 1, 2, 5, 9 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 9 }, { 3, 5 } },                           { 1, 3, 5, 9 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 2, 4 }, { 1, 3, 5 } },                        { 1, 2, 3, 4, 5 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 3, 4 }, { 1, 2, 5, 6 } },                     { 1, 2, 3, 4, 5, 6 } );
    // duplicates inside a step, equal to the last, equal to the first and to the last at once, wholly equal
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 1, 2, 2 } },                               { 1, 2 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 2 }, { 2 } },                              { 1, 2 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 2 }, { 2, 3 } },                           { 1, 2, 3 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 4 }, { 1, 4 } },                           { 1, 4 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 1, 4 }, { 0, 1, 4, 5 }, { 3, 4 } },           { 0, 1, 3, 4, 5 } );
    expect( std::initializer_list<std::initializer_list<std::uint64_t>>{ { 7 }, { 7 }, { 3 }, { 3 }, { 7 } },            { 3, 7 } );
}

TEST( TouchedChunks, ManyValuesInOneChunkAreOneEntry ) {
    std::vector<std::uint32_t> values;
    for ( std::uint32_t v{ 5U }; v < 3'000U; v += 7U ) {
        values.push_back( v );
        values.push_back( v + 65'536U );
        values.push_back( v + 3U * 65'536U );
    }
    std::ranges::sort( values );
    TouchedBitmap::touched_chunks touched;
    touched.add_values( values );
    EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0, 1, 3 } ) );
    touched.add_values( {} );
    EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0, 1, 3 } ) );
}

TEST( TouchedChunks, RangesCoverEveryChunkTheySpanAndMergeLikeValues ) {
    using Touched = TouchedBitmap::touched_chunks;
    {
        Touched touched;
        touched.add_range( 70'000U, 70'000U );
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 1 } ) );
        touched.add_range( 3U, 65'535U );   // chunk 0 only, below the 1 held
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0, 1 } ) );
        touched.add_range( 65'536U * 2U + 7U, 65'536U * 5U );   // 2..5, past it
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0, 1, 2, 3, 4, 5 } ) );
        touched.add_range( 65'536U * 3U, 65'536U * 8U + 1U );   // 3..8, overlapping the end
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0, 1, 2, 3, 4, 5, 6, 7, 8 } ) );
    }
    {
        // a range ending in the last chunk of the key domain does not wrap
        Touched touched;
        touched.add_range( 0xFFFE'0000U, 0xFFFF'FFFFU );
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0xFFFE, 0xFFFF } ) );
        using WideTouched = WideTouchedBitmap::touched_chunks;
        WideTouched wide;
        auto constexpr max{ std::numeric_limits<std::uint64_t>::max() };
        wide.add_range( max - 70'000U, max );
        auto const last_chunk{ max >> 16 };
        WideTouched expected;
        std::vector<std::uint64_t> firsts{ ( last_chunk - 1 ) << 16, last_chunk << 16 };
        expected.add_values( firsts );
        EXPECT_EQ( wide, expected );
    }
    {
        // at the very start of the domain
        Touched touched;
        touched.add_range( 0U, 0U );
        EXPECT_EQ( touched, touched_of<TouchedBitmap>( { 0 } ) );
    }
}

TEST( TouchedChunks, ChunksOfABitmapMergeWithWhatIsHeld ) {
    using Bitmap = TouchedBitmap;
    std::vector<std::uint32_t> values;
    add_chunk<Bitmap>( values, 2, chunk_shape::short_run_array );
    add_chunk<Bitmap>( values, 4, chunk_shape::short_run_array );
    add_chunk<Bitmap>( values, 9, chunk_shape::short_run_array );
    Bitmap const bitmap{ values };

    Bitmap::touched_chunks touched;
    touched.add_chunks_of( Bitmap{} );
    EXPECT_TRUE( touched.empty() );
    touched.add_chunks_of( bitmap );
    EXPECT_EQ( touched, touched_of<Bitmap>( { 2, 4, 9 } ) );
    touched.add_chunks_of( bitmap );
    EXPECT_EQ( touched, touched_of<Bitmap>( { 2, 4, 9 } ) );

    Bitmap::touched_chunks mixed{ touched_of<Bitmap>( { 0, 4, 11 } ) };
    mixed.add_chunks_of( bitmap );
    EXPECT_EQ( mixed, touched_of<Bitmap>( { 0, 2, 4, 9, 11 } ) );

    // the chunks of a 64-bit bitmap
    WideTouchedBitmap wide;
    (void)wide.add( 5ULL << 40 );
    (void)wide.add( 9ULL << 50 );
    WideTouchedBitmap::touched_chunks wide_touched;
    wide_touched.add_chunks_of( wide );
    EXPECT_EQ( wide_touched, touched_of<WideTouchedBitmap>( { 5ULL << 24, 9ULL << 34 } ) );
}

// A 16-bit key has a single chunk.
TEST( TouchedChunks, ASixteenBitKeyHasOneChunk ) {
    using Small = LazyOf<std::uint16_t, detail_test::cow_atomic_refcount>;
    std::vector<std::uint16_t> values;
    for ( std::uint16_t v{ 100U }; v < 700U; ++v ) {
        values.push_back( v );
    }
    Small bitmap{ values };
    ASSERT_EQ( bitmap.statistics().array_containers, 1U );
    Small::touched_chunks touched;
    touched.add_values( std::span<std::uint16_t const>{ values } );
    touched.add_range( 3U, 0xFFFFU );
    Small::touched_chunks expected;
    std::array<std::uint16_t, 1> const one{ 7U };
    expected.add_values( one );
    EXPECT_EQ( touched, expected );
    std::as_const( bitmap ).optimize_run_shaped( listed_bar, touched );
    EXPECT_EQ( bitmap.statistics().run_containers, 1U );
    EXPECT_EQ( bitmap.to_vector(), values );
}

// An allocation failure inside an add leaves the set as it was, and inside the pass leaves the bitmap valid and equal.
TEST( TouchedChunks, AllocationFailuresLeaveTheSetAndTheBitmapIntact ) {
    if constexpr ( detail_test::allocation_failure_terminates ) {
        GTEST_SKIP() << "an allocation failure terminates in this build";
    }
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    namespace fault = frsr::roaring::kernel_test::alloc_fault;
    using Bitmap = TouchedBitmap;
    std::uint32_t failed{ 0 };
    // adds that land below what is held, after some of their chunks were already appended
    for ( std::uint32_t const period : { 1U, 2U, 3U, 5U } ) {
        auto touched{ touched_in_steps<Bitmap>( { { 100, 200, 300, 400, 500 } } ) };
        auto const before{ touched_in_steps<Bitmap>( { { 100, 200, 300, 400, 500 } } ) };
        std::vector<std::uint32_t> values;
        for ( std::uint32_t chunk{ 0 }; chunk < 60; ++chunk ) {
            values.push_back( chunk << 16 );
        }
        try {
            fault::armed const armed{ period };
            touched.add_values( values );
        } catch ( std::bad_alloc const & ) {
            ++failed;
            EXPECT_EQ( touched, before );
        }
        try {
            fault::armed const armed{ period };
            touched.add_range( 0U, 60U << 16 );
        } catch ( std::bad_alloc const & ) {
            ++failed;
        }
        // whatever happened, the set is ascending and distinct: a later add and the pass behave
        touched.add_range( 0U, 600U << 16 );
        Bitmap::touched_chunks expected;
        expected.add_range( 0U, 600U << 16 );
        EXPECT_EQ( touched, expected );
    }
    EXPECT_GT( failed, 0U );

    auto const values{ mixed_chunks<Bitmap>() };
    auto const all{ touched_of<Bitmap>( { 0, 1, 2, 3, 4, 5 } ) };
    failed = 0;
    for ( std::uint32_t const period : { 1U, 2U, 3U, 5U, 11U, 23U } ) {
        Bitmap bitmap{ values };
        try {
            fault::armed const armed{ period };
            std::as_const( bitmap ).optimize_run_shaped( listed_bar, all );
        } catch ( std::bad_alloc const & ) {
            ++failed;
        }
        EXPECT_EQ( bitmap.validate(), nullptr );
        EXPECT_EQ( bitmap.to_vector(), values );
        EXPECT_EQ( bitmap.statistics().container_count(), 6U );
    }
    EXPECT_GT( failed, 0U );
#endif
}

TEST(FrsrRoaringSmoke, ShrinkToFitReturnsPayloadSlackAndKeepsContents) {
    // Array payloads grow by doubling, so a chunk that grew to N and then lost
    // most of its values sits on an oversized block. shrink_to_fit gives that
    // slack back without changing what the bitmap holds.
    constexpr std::uint32_t chunk{ 1U << 16 };
    std::vector<std::uint32_t> values;
    for ( std::uint32_t c{ 0U }; c < 4U; ++c ) {
        for ( std::uint32_t i{ 0U }; i < 3'000U; ++i ) { values.push_back( c * chunk + i * 5U ); }
    }
    TestBitmap bitmap;
    for ( auto const value : values ) { std::ignore = bitmap.add( value ); }

    std::vector<std::uint32_t> survivors;
    for ( std::size_t i{ 0 }; i < values.size(); ++i ) {
        if ( i % 16 == 0 ) { survivors.push_back( values[ i ] ); } else { std::ignore = bitmap.remove( values[ i ] ); }
    }
    ASSERT_EQ( bitmap.to_vector(), survivors );

    auto const freed{ bitmap.shrink_to_fit() };
    EXPECT_GT( freed, 0U );                       // the doubled blocks were at least halved
    EXPECT_EQ( bitmap.to_vector(), survivors );   // and the contents are untouched
    EXPECT_EQ( bitmap.shrink_to_fit(), 0U );      // idempotent: nothing left to give back

    // Still a normal bitmap afterwards: it grows, unions and intersects as before.
    std::ignore = bitmap.add( 7U * chunk + 1U );
    EXPECT_TRUE( bitmap.contains( 7U * chunk + 1U ) );
    TestBitmap other;
    for ( auto const value : survivors ) { std::ignore = other.add( value ); }
    EXPECT_EQ( ( bitmap & other ).to_vector(), survivors );
}

TEST(FrsrRoaringSmoke, ShrinkToFitLeavesASharedPayloadAloneAndKeepsBothCopies) {
    // A shared payload cannot shrink — reallocating it would clone rather than
    // release — so a bitmap sharing its containers keeps them, and both copies
    // still read correctly.
    constexpr std::uint32_t chunk{ 1U << 16 };
    std::vector<std::uint32_t> values;
    for ( std::uint32_t i{ 0U }; i < 3'000U; ++i ) { values.push_back( i * 5U ); }
    for ( std::uint32_t i{ 0U }; i < 3'000U; ++i ) { values.push_back( chunk + i * 5U ); }

    LazyBitmap source;
    for ( auto const value : values ) { std::ignore = source.add( value ); }
    for ( std::size_t i{ 1 }; i < values.size(); i += 2 ) { std::ignore = source.remove( values[ i ] ); }
    std::vector<std::uint32_t> survivors;
    for ( std::size_t i{ 0 }; i < values.size(); i += 2 ) { survivors.push_back( values[ i ] ); }
    std::ranges::sort( survivors );
    ASSERT_EQ( source.to_vector(), survivors );

    LazyBitmap copy{ source };
    EXPECT_EQ( copy.shrink_to_fit(), 0U );  // every payload is shared with `source`
    EXPECT_EQ( copy  .to_vector(), survivors );
    EXPECT_EQ( source.to_vector(), survivors );
}


TEST(FrsrRoaringSmoke, BulkContextAddsStillCloneASharedPayloadOnce) {
    // A bulk context skips the copy-on-write barrier for the run of adds after
    // the one that made its chunk private - so the FIRST add of a sequence must
    // still take it, however many values follow, or an outstanding copy would
    // see them.
    std::vector<std::uint32_t> values;
    for ( std::uint32_t value{ 0 }; value < 5'000U; ++value ) { values.push_back( value * 3U ); }

    LazyBitmap original{ std::span<std::uint32_t const>{ values } };
    LazyBitmap copy    { original };

    LazyBitmap::bulk_context ctx;
    for ( std::uint32_t i{ 0 }; i < 2'000U; ++i ) {
        ASSERT_TRUE( original.add_bulk( ctx, 1U + i * 3U ) );  // between the existing values
    }

    EXPECT_EQ( copy.to_vector(), values );
    EXPECT_EQ( copy.size(), values.size() );
    EXPECT_EQ( original.size(), values.size() + 2'000U );
    EXPECT_TRUE( original.contains( 1U ) );
    EXPECT_FALSE( copy.contains( 1U ) );

    // A second copy taken now, then a second sequence through a FRESH context:
    // the same protection has to apply again.
    LazyBitmap const copy2{ original };
    auto const copy2_values{ copy2.to_vector() };
    LazyBitmap::bulk_context ctx2;
    for ( std::uint32_t i{ 0 }; i < 2'000U; ++i ) {
        ASSERT_TRUE( original.add_bulk( ctx2, 2U + i * 3U ) );
    }
    EXPECT_EQ( copy2.to_vector(), copy2_values );
    EXPECT_EQ( original.size(), values.size() + 4'000U );
}

TEST(FrsrRoaringSmoke, AddManySortedIntoASharedPayloadLeavesTheCopyAlone) {
    // The group insert takes the copy-on-write barrier once for the whole group
    // instead of once per value; it still has to take it, so the copy must not
    // see the group's values.
    std::vector<std::uint32_t> values;
    for ( std::uint32_t value{ 0U }; value <= 20'000U; ++value ) {
        if ( ( value % 7U ) != 3U ) { values.push_back( value ); }
    }

    LazyBitmap original{ std::span<std::uint32_t const>{ values } };
    LazyBitmap copy    { original };

    std::vector<std::uint32_t> const group{ 20'001U, 20'002U, 20'003U, 30'000U };
    original.add_many_sorted( { group.data(), group.size() } );

    auto expected{ values };
    expected.insert( expected.end(), group.begin(), group.end() );

    EXPECT_EQ( original.to_vector(), expected );
    EXPECT_EQ( copy    .to_vector(), values   );
    EXPECT_EQ( original.size(), expected.size() );
    EXPECT_EQ( copy    .size(), values  .size() );
}

TEST(FrsrRoaringSmoke, ContainsMatchesTheModelForInterleavedBitmapsAndProbeOrders) {
    // Two bitmaps of different sizes probed in one thread, in ascending, descending,
    // random and alternating order, with hits, chunk misses (below, between and
    // above the stored chunks) and container misses: whatever state a lookup keeps
    // between calls must not change an answer.
    using wide = frsr::roaring::bitmap<std::uint64_t>;
    std::mt19937_64 rng{ 17 };
    std::set<std::uint64_t> model_a, model_b;
    wide a, b;
    for ( int i{ 0 }; i != 3000; ++i ) {
        auto const value{ ( ( rng() % 4000U ) << 16 ) | ( rng() & 0xFFFFU ) };
        model_a.insert( value ); a.add( value );
    }
    for ( int i{ 0 }; i != 40; ++i ) {
        auto const value{ ( ( rng() % 4000U ) << 16 ) | ( rng() & 0xFFFFU ) };
        model_b.insert( value ); b.add( value );
    }
    std::vector<std::uint64_t> probes;
    for ( auto const value : model_a ) { probes.push_back( value ); probes.push_back( value + 1 ); probes.push_back( value ^ ( 1ULL << 16 ) ); }
    for ( auto const value : model_b ) { probes.push_back( value ); probes.push_back( value - 1 ); }
    probes.push_back( 0 ); probes.push_back( ~std::uint64_t{ 0 } ); probes.push_back( 4000ULL << 16 );

    auto const check = [&]( std::vector<std::uint64_t> const & order ) {
        for ( auto const value : order ) {
            EXPECT_EQ( a.contains( value ), model_a.contains( value ) ) << value;
            EXPECT_EQ( b.contains( value ), model_b.contains( value ) ) << value;
        }
    };
    std::ranges::sort( probes );
    check( probes );                                   // ascending
    std::vector<std::uint64_t> descending{ probes.rbegin(), probes.rend() };
    check( descending );
    std::ranges::shuffle( probes, rng );
    check( probes );                                   // random
    for ( auto const value : probes ) {                // alternating between the two bitmaps
        EXPECT_EQ( a.contains( value ), model_a.contains( value ) ) << value;
        EXPECT_EQ( b.contains( value ), model_b.contains( value ) ) << value;
        EXPECT_EQ( b.contains( value ^ 0xFFFF0000ULL ), model_b.contains( value ^ 0xFFFF0000ULL ) ) << value;
    }
    wide const empty;
    EXPECT_FALSE( empty.contains( 0 ) );
    EXPECT_FALSE( empty.contains( ~std::uint64_t{ 0 } ) );
}

template <typename Bitmap, typename Key = typename Bitmap::key_type>
Bitmap make_mixed_chunks( std::uint32_t const chunk_count ) {
    Bitmap result;
    for ( std::uint32_t chunk{ 0 }; chunk < chunk_count; ++chunk ) {
        Key const base{ static_cast<Key>( chunk ) << 16 };
        switch ( chunk % 3 ) {
            case 0: for ( std::uint32_t i{ 0 }; i < 10; ++i ) { result.add( base + i * 7 ); } break;
            case 1: for ( std::uint32_t i{ 0 }; i < 20'000; ++i ) { result.add( base + ( i * 3 ) % 65'536 ); } break;
            default: for ( std::uint32_t i{ 0 }; i < 30'000; ++i ) { result.add( base + i ); } break;
        }
    }
    result.optimize_for_storage();
    return result;
}

TEST(FrsrRoaringSmoke, EstimatedDataByteSizeIsZeroForAnEmptyBitmap) {
    EXPECT_EQ( TestBitmap{}.estimated_data_byte_size(), 0U );
    EXPECT_EQ( frsr::roaring::bitmap<std::uint64_t>{}.estimated_data_byte_size(), 0U );
}

TEST(FrsrRoaringSmoke, EstimatedDataByteSizeIsExactUpToEightChunks) {
    for ( std::uint32_t chunks{ 1 }; chunks <= 8; ++chunks ) {
        auto const bitmap{ make_mixed_chunks<TestBitmap>( chunks ) };
        // Each sampled chunk is read once, so a uniform bitmap is exact and a mixed one is exact only when
        // every chunk is sampled: eight chunks or fewer, one step each.
        EXPECT_EQ( bitmap.estimated_data_byte_size(), bitmap.data_byte_size() ) << chunks << " chunks";
    }
}

TEST(FrsrRoaringSmoke, EstimatedDataByteSizeTracksDataByteSizeForManyMixedChunks) {
    for ( std::uint32_t const chunks : { 64U, 200U, 1'000U } ) {
        auto const bitmap{ make_mixed_chunks<TestBitmap>( chunks ) };
        auto const actual   { static_cast<double>( bitmap.data_byte_size() ) };
        auto const estimated{ static_cast<double>( bitmap.estimated_data_byte_size() ) };
        EXPECT_GT( estimated, actual / 4 ) << chunks << " chunks";
        EXPECT_LT( estimated, actual * 4 ) << chunks << " chunks";
    }
}

TEST(FrsrRoaringSmoke, EstimatedDataByteSizeOfAWideKeyBitmapIsDataByteSize) {
    using wide = frsr::roaring::bitmap<std::uint64_t>;
    auto const bitmap{ make_mixed_chunks<wide>( 100 ) };
    EXPECT_EQ( bitmap.estimated_data_byte_size(), bitmap.data_byte_size() );
}

} // namespace
