#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/run_container.hpp>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
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
    static_assert( handle::inline_capacity<std::uint16_t> >= 8U );  // the sparse-regime SBO win
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
        EXPECT_EQ( stats.staged_singleton_chunks, 0U );
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

} // namespace
