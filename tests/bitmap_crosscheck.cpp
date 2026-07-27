#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>
#include <roaring/roaring.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace {

using TestBitmap = frsr::roaring::bitmap<std::uint32_t>;

struct roaring_bitmap_holder {
    roaring_bitmap_holder() : bitmap{ roaring_bitmap_create() } {}
    explicit roaring_bitmap_holder( roaring_bitmap_t * raw ) : bitmap{ raw } {}

    roaring_bitmap_holder( roaring_bitmap_holder const & ) = delete;
    roaring_bitmap_holder & operator=( roaring_bitmap_holder const & ) = delete;

    roaring_bitmap_holder( roaring_bitmap_holder && other ) noexcept : bitmap{ other.bitmap } {
        other.bitmap = nullptr;
    }

    roaring_bitmap_holder & operator=( roaring_bitmap_holder && other ) noexcept {
        if ( this != &other ) {
            reset();
            bitmap = other.bitmap;
            other.bitmap = nullptr;
        }
        return *this;
    }

    ~roaring_bitmap_holder() { reset(); }

    void reset() noexcept {
        if ( bitmap != nullptr ) {
            roaring_bitmap_free( bitmap );
            bitmap = nullptr;
        }
    }

    roaring_bitmap_t * bitmap{};
};

[[nodiscard]] auto make_bitmap( std::span<std::uint32_t const> values ) {
    return TestBitmap{ values };
}

[[nodiscard]] auto make_roaring( std::span<std::uint32_t const> values ) {
    roaring_bitmap_holder result;
    for ( auto const value : values ) {
        roaring_bitmap_add( result.bitmap, value );
    }
    return result;
}

[[nodiscard]] auto to_vector( roaring_bitmap_t const * const bitmap ) {
    std::vector<std::uint32_t> result(
        static_cast<std::size_t>( roaring_bitmap_get_cardinality( bitmap ) )
    );
    roaring_bitmap_to_uint32_array( bitmap, result.data() );
    return result;
}

[[nodiscard]] auto make_dense_pattern_values(
    std::uint32_t const begin,
    std::uint32_t const end,
    std::uint32_t const modulus,
    std::uint32_t const remainder
) {
    std::vector<std::uint32_t> values;
    values.reserve( static_cast<std::size_t>( end - begin + 1U ) );
    for ( auto value{ begin }; value <= end; ++value ) {
        if ( ( value % modulus ) != remainder ) {
            values.push_back( value );
        }
    }
    return values;
}

TEST(FrsrRoaringCrosscheck, IterationAndBulkContextFollowSortedOrder) {
    TestBitmap bitmap;
    TestBitmap::bulk_context ctx;

    for ( auto const value : { 1U, 2U, 3U, 65'536U, 65'540U } ) {
        EXPECT_TRUE( bitmap.add_bulk( ctx, value ) );
    }

    EXPECT_FALSE( bitmap.add_bulk( ctx, 65'540U ) );
    EXPECT_TRUE( bitmap.contains_bulk( ctx, 65'540U ) );
    EXPECT_FALSE( bitmap.contains_bulk( ctx, 65'541U ) );

    std::vector<std::uint32_t> values;
    for ( auto const value : bitmap ) {
        values.push_back( value );
    }

    EXPECT_EQ( values, ( std::vector<std::uint32_t>{ 1U, 2U, 3U, 65'536U, 65'540U } ) );
}

TEST(FrsrRoaringCrosscheck, RandomizedBinaryOperationsMatchCRoaring) {
    std::mt19937 rng{ 0xC0FFEEU };
    std::uniform_int_distribution<std::uint32_t> value_dist{ 0U, 500'000U };
    std::uniform_int_distribution<int> size_dist{ 0, 600 };

    for ( int round{ 0 }; round < 48; ++round ) {
        std::vector<std::uint32_t> lhs_values;
        std::vector<std::uint32_t> rhs_values;
        lhs_values.reserve( static_cast<std::size_t>( size_dist( rng ) ) );
        rhs_values.reserve( static_cast<std::size_t>( size_dist( rng ) ) );

        auto const lhs_size{ size_dist( rng ) };
        auto const rhs_size{ size_dist( rng ) };
        for ( int index{ 0 }; index < lhs_size; ++index ) {
            lhs_values.push_back( value_dist( rng ) );
        }
        for ( int index{ 0 }; index < rhs_size; ++index ) {
            rhs_values.push_back( value_dist( rng ) );
        }

        auto const lhs{ make_bitmap( lhs_values ) };
        auto const rhs{ make_bitmap( rhs_values ) };
        auto const lhs_roaring{ make_roaring( lhs_values ) };
        auto const rhs_roaring{ make_roaring( rhs_values ) };

        auto const union_bitmap{ lhs | rhs };
        auto const intersection_bitmap{ lhs & rhs };
        auto const difference_bitmap{ lhs - rhs };

        roaring_bitmap_holder const union_roaring{ roaring_bitmap_or( lhs_roaring.bitmap, rhs_roaring.bitmap ) };
        roaring_bitmap_holder const intersection_roaring{ roaring_bitmap_and( lhs_roaring.bitmap, rhs_roaring.bitmap ) };
        roaring_bitmap_holder const difference_roaring{ roaring_bitmap_andnot( lhs_roaring.bitmap, rhs_roaring.bitmap ) };

        EXPECT_EQ( union_bitmap.to_vector(), to_vector( union_roaring.bitmap ) );
        EXPECT_EQ( intersection_bitmap.to_vector(), to_vector( intersection_roaring.bitmap ) );
        EXPECT_EQ( difference_bitmap.to_vector(), to_vector( difference_roaring.bitmap ) );

        for ( int probe{ 0 }; probe < 64; ++probe ) {
            auto const value{ value_dist( rng ) };
            EXPECT_EQ( lhs.contains( value ), roaring_bitmap_contains( lhs_roaring.bitmap, value ) );
            EXPECT_EQ( rhs.contains( value ), roaring_bitmap_contains( rhs_roaring.bitmap, value ) );
        }
    }
}

TEST(FrsrRoaringCrosscheck, ArrayPromotionStillMatchesCRoaring) {
    std::vector<std::uint32_t> values;
    values.reserve( 8'192 );
    for ( std::uint32_t value{ 0 }; value < 8'192U; ++value ) {
        values.push_back( value );
    }

    auto bitmap{ make_bitmap( values ) };
    auto roaring{ make_roaring( values ) };

    for ( std::uint32_t value{ 100U }; value < 3'000U; value += 3U ) {
        EXPECT_TRUE( bitmap.remove( value ) );
        roaring_bitmap_remove( roaring.bitmap, value );
    }

    EXPECT_EQ( bitmap.to_vector(), to_vector( roaring.bitmap ) );
}

TEST(FrsrRoaringCrosscheck, DensePatternBinaryOperationsMatchCRoaring) {
    auto lhs_values{ make_dense_pattern_values( 0U, 65'535U, 5U, 0U ) };
    auto rhs_values{ make_dense_pattern_values( 0U, 65'535U, 7U, 0U ) };
    auto lhs_tail{ make_dense_pattern_values( 131'072U, 196'607U, 9U, 4U ) };
    auto rhs_tail{ make_dense_pattern_values( 131'072U, 196'607U, 11U, 5U ) };
    lhs_values.insert( lhs_values.end(), lhs_tail.begin(), lhs_tail.end() );
    rhs_values.insert( rhs_values.end(), rhs_tail.begin(), rhs_tail.end() );

    auto const lhs{ make_bitmap( lhs_values ) };
    auto const rhs{ make_bitmap( rhs_values ) };
    auto const lhs_roaring{ make_roaring( lhs_values ) };
    auto const rhs_roaring{ make_roaring( rhs_values ) };

    auto const union_bitmap{ lhs | rhs };
    auto const intersection_bitmap{ lhs & rhs };
    auto const difference_bitmap{ lhs - rhs };

    roaring_bitmap_holder const union_roaring{ roaring_bitmap_or( lhs_roaring.bitmap, rhs_roaring.bitmap ) };
    roaring_bitmap_holder const intersection_roaring{ roaring_bitmap_and( lhs_roaring.bitmap, rhs_roaring.bitmap ) };
    roaring_bitmap_holder const difference_roaring{ roaring_bitmap_andnot( lhs_roaring.bitmap, rhs_roaring.bitmap ) };

    auto const expected_union{ to_vector( union_roaring.bitmap ) };
    auto const expected_intersection{ to_vector( intersection_roaring.bitmap ) };
    auto const expected_difference{ to_vector( difference_roaring.bitmap ) };

    EXPECT_EQ( union_bitmap.size(), expected_union.size() );
    EXPECT_EQ( intersection_bitmap.size(), expected_intersection.size() );
    EXPECT_EQ( difference_bitmap.size(), expected_difference.size() );
    EXPECT_EQ( union_bitmap.to_vector(), expected_union );
    EXPECT_EQ( intersection_bitmap.to_vector(), expected_intersection );
    EXPECT_EQ( difference_bitmap.to_vector(), expected_difference );
}

TEST(FrsrRoaringCrosscheck, ClosedRangeMutationsMatchCRoaring) {
    TestBitmap bitmap;
    roaring_bitmap_holder roaring;

    bitmap.add_closed_range( 128U, 8'192U );
    roaring_bitmap_add_range_closed( roaring.bitmap, 128U, 8'192U );

    bitmap.remove_closed_range( 777U, 1'333U );
    roaring_bitmap_remove_range_closed( roaring.bitmap, 777U, 1'333U );

    bitmap.flip_closed_range( 4'096U, 9'000U );
    roaring_bitmap_flip_inplace_closed( roaring.bitmap, 4'096U, 9'000U );

    EXPECT_EQ( bitmap.to_vector(), to_vector( roaring.bitmap ) );
}

TEST(FrsrRoaringCrosscheck, RangeHeavyStorageOptimizationPreservesSemantics) {
    TestBitmap bitmap;
    roaring_bitmap_holder roaring;

    for ( std::uint32_t base{ 0U }; base < 65'536U; base += 4'096U ) {
        bitmap.add_closed_range( 1'000U + base, 2'500U + base );
        roaring_bitmap_add_range_closed( roaring.bitmap, 1'000U + base, 2'500U + base );
    }

    bitmap.optimize_for_storage();

    EXPECT_EQ( bitmap.to_vector(), to_vector( roaring.bitmap ) );
    EXPECT_TRUE( bitmap.intersects( TestBitmap{ std::vector<std::uint32_t>{ 2'000U, 500'000U } } ) );
}

TEST(FrsrRoaringCrosscheck, LazyOrFinishersMatchCRoaringUnionResult) {
    std::vector<std::uint32_t> lhs_values;
    std::vector<std::uint32_t> rhs_values;
    for ( std::uint32_t value{ 0U }; value < 20'000U; value += 2U ) {
        lhs_values.push_back( value );
    }
    for ( std::uint32_t value{ 10'000U }; value < 30'000U; ++value ) {
        rhs_values.push_back( value );
    }

    auto const lhs{ make_bitmap( lhs_values ) };
    auto const rhs{ make_bitmap( rhs_values ) };
    auto const lhs_roaring{ make_roaring( lhs_values ) };
    auto const rhs_roaring{ make_roaring( rhs_values ) };
    roaring_bitmap_holder const union_roaring{ roaring_bitmap_or( lhs_roaring.bitmap, rhs_roaring.bitmap ) };

    TestBitmap finish_storage;
    finish_storage.bulk_or_intermediate( lhs );
    finish_storage.bulk_or_intermediate( rhs );
    finish_storage.bulk_or_finish();

    TestBitmap keep_bitsets;
    keep_bitsets.bulk_or_intermediate( lhs );
    keep_bitsets.bulk_or_intermediate( rhs );
    keep_bitsets.bulk_or_finish_keep_bitsets();

    auto const expected{ to_vector( union_roaring.bitmap ) };
    // size() must match too: the lazy bulk path defers the popcount, so a missing
    // repair_cardinality() would leave size() wrong while contents stayed correct.
    EXPECT_EQ( finish_storage.size(), expected.size() );
    EXPECT_EQ( finish_storage.to_vector(), expected );
    EXPECT_EQ( keep_bitsets.size(), expected.size() );
    EXPECT_EQ( keep_bitsets.to_vector(), expected );
}

TEST(FrsrRoaringCrosscheck, OrManyRepairsLazyContainerFormLikeCRoaring) {
    std::vector<std::uint32_t> lhs_values;
    std::vector<std::uint32_t> rhs_values;
    for ( std::uint32_t i{ 0U }; i < 700U; ++i ) {
        lhs_values.push_back( i * 4U );
        rhs_values.push_back( i * 4U + 2U );
    }

    auto const lhs{ make_bitmap( lhs_values ) };
    auto const rhs{ make_bitmap( rhs_values ) };
    std::vector<TestBitmap const *> const operands{ &lhs, &rhs };
    TestBitmap result;
    result.or_many_in_place( operands );

    auto const lhs_roaring{ make_roaring( lhs_values ) };
    auto const rhs_roaring{ make_roaring( rhs_values ) };
    roaring_bitmap_holder expected_roaring;
    roaring_bitmap_lazy_or_inplace( expected_roaring.bitmap, lhs_roaring.bitmap, true );
    roaring_bitmap_lazy_or_inplace( expected_roaring.bitmap, rhs_roaring.bitmap, true );
    roaring_bitmap_repair_after_lazy( expected_roaring.bitmap );

    roaring_statistics_t expected_stats{};
    roaring_bitmap_statistics( expected_roaring.bitmap, &expected_stats );
    auto const result_stats{ result.statistics() };

    EXPECT_EQ( result.to_vector(), to_vector( expected_roaring.bitmap ) );
    EXPECT_EQ( result_stats.array_containers, expected_stats.n_array_containers );
    EXPECT_EQ( result_stats.run_containers, expected_stats.n_run_containers );
    EXPECT_EQ( result_stats.bitset_containers, expected_stats.n_bitset_containers );
    EXPECT_EQ( result_stats.container_count(), expected_stats.n_containers );
    EXPECT_EQ( result_stats.array_containers, 1U );
    EXPECT_EQ( result_stats.bitset_containers, 0U );

    TestBitmap run_lhs;
    run_lhs.add_closed_range( 100U, 10'000U );
    run_lhs.optimize();
    std::vector<TestBitmap const *> const run_operands{ &run_lhs };
    TestBitmap run_result;
    run_result.or_many_in_place( run_operands );

    roaring_bitmap_holder run_lhs_roaring;
    roaring_bitmap_add_range_closed( run_lhs_roaring.bitmap, 100U, 10'000U );
    ASSERT_TRUE( roaring_bitmap_run_optimize( run_lhs_roaring.bitmap ) );
    roaring_bitmap_t const * run_inputs[]{ run_lhs_roaring.bitmap };
    roaring_bitmap_holder const run_expected{ roaring_bitmap_or_many( 1U, run_inputs ) };

    roaring_statistics_t run_expected_stats{};
    roaring_bitmap_statistics( run_expected.bitmap, &run_expected_stats );
    auto const run_result_stats{ run_result.statistics() };
    EXPECT_EQ( run_result.to_vector(), to_vector( run_expected.bitmap ) );
    EXPECT_EQ( run_result_stats.run_containers, run_expected_stats.n_run_containers );
    EXPECT_EQ( run_result_stats.run_containers, 1U );
}

TEST(FrsrRoaringCrosscheck, OrManyInPlaceMatchesCRoaringUnionAndCardinality) {
    // Several dense, overlapping operands spanning multiple chunks, so the N-way
    // union exercises the lazy bitset accumulation + one-shot repair_cardinality().
    constexpr std::size_t operand_count{ 8U };
    std::vector<std::vector<std::uint32_t>> operand_values( operand_count );
    for ( std::size_t k{ 0U }; k < operand_count; ++k ) {
        auto const base{ static_cast<std::uint32_t>( k * 20'000U ) };
        for ( std::uint32_t value{ base }; value < base + 30'000U; ++value ) {
            operand_values[ k ].push_back( value );
        }
    }

    std::vector<TestBitmap> operands;
    operands.reserve( operand_count );
    roaring_bitmap_holder expected_roaring;
    for ( auto const & values : operand_values ) {
        operands.push_back( make_bitmap( values ) );
        for ( auto const value : values ) {
            roaring_bitmap_add( expected_roaring.bitmap, value );
        }
    }

    std::vector<TestBitmap const *> operand_ptrs;
    operand_ptrs.reserve( operand_count );
    for ( auto const & operand : operands ) {
        operand_ptrs.push_back( &operand );
    }

    TestBitmap result;
    result.or_many_in_place( { operand_ptrs.data(), operand_ptrs.size() } );

    auto const expected{ to_vector( expected_roaring.bitmap ) };
    EXPECT_EQ( result.size(), expected.size() );
    EXPECT_EQ( result.to_vector(), expected );
}

TEST(FrsrRoaringCrosscheck, OrManyInPlaceMixedArrayBitsetChunksMatchCRoaring) {
    // Drives the lazy N-way union's same-key MIXED array↔bitset accumulation: when an
    // operand's chunk is an array (< array_to_bitset_threshold elements) but the
    // accumulator's same-key chunk is a bitset (or vice versa), the bulk path must OR
    // them in place (promoting the accumulator to a bitset as CRoaring's
    // LAZY_OR_BITSET_CONVERSION does) rather than fall back to a materialized combine.
    // Operand order is chosen so both sub-cases fire: array-accumulator promoted by an
    // incoming bitset, and bitset-accumulator scattered with an incoming array, on
    // chunk 0 and on higher chunks.
    auto const sparse_in = []( std::uint32_t const base, std::uint32_t const stride, std::uint32_t const n ) {
        std::vector<std::uint32_t> v;
        for ( std::uint32_t i{ 0U }; i < n; ++i ) { v.push_back( base + i * stride ); }
        return v;  // n well below 4096 -> array container
    };
    auto const dense_in = []( std::uint32_t const begin, std::uint32_t const end ) {
        std::vector<std::uint32_t> v;
        for ( std::uint32_t value{ begin }; value < end; ++value ) { v.push_back( value ); }
        return v;  // spans >= 4096 elements per chunk -> bitset container(s)
    };

    std::vector<std::vector<std::uint32_t>> operand_values;
    operand_values.push_back( sparse_in( 0U, 257U, 200U ) );             // op0: chunk0 ARRAY (accumulator seed)
    operand_values.push_back( dense_in( 0U, 10'000U ) );                 // op1: chunk0 BITSET -> array×bitset (promote left)
    operand_values.push_back( sparse_in( 1U, 333U, 150U ) );             // op2: chunk0 ARRAY -> bitset×array (scatter)
    operand_values.push_back( dense_in( 60'000U, 140'000U ) );           // op3: chunks 0,1,2 BITSET (new keys 1,2)
    {
        std::vector<std::uint32_t> sparse_high;                          // op4: chunk1 & chunk2 ARRAYs -> bitset×array
        for ( std::uint32_t i{ 0U }; i < 100U; ++i ) { sparse_high.push_back(  65'536U + i * 600U ); }
        for ( std::uint32_t i{ 0U }; i < 100U; ++i ) { sparse_high.push_back( 131'072U + i * 600U ); }
        operand_values.push_back( std::move( sparse_high ) );
    }

    std::vector<TestBitmap> operands;
    operands.reserve( operand_values.size() );
    roaring_bitmap_holder expected_roaring;
    for ( auto const & values : operand_values ) {
        operands.push_back( make_bitmap( values ) );
        for ( auto const value : values ) { roaring_bitmap_add( expected_roaring.bitmap, value ); }
    }

    std::vector<TestBitmap const *> operand_ptrs;
    operand_ptrs.reserve( operands.size() );
    for ( auto const & operand : operands ) { operand_ptrs.push_back( &operand ); }

    TestBitmap result;
    result.or_many_in_place( { operand_ptrs.data(), operand_ptrs.size() } );

    auto const expected{ to_vector( expected_roaring.bitmap ) };
    EXPECT_EQ( result.size(), expected.size() );
    EXPECT_EQ( result.to_vector(), expected );
}

TEST(FrsrRoaringCrosscheck, OrManyInPlaceInterleavedKeysSpliceMatchCRoaring) {
    // Drives bulk_or_inplace's other-only-key SPLICE branch — a key that falls BETWEEN
    // existing accumulator keys forces a mid-vector insert (the prior tests only append
    // higher keys to the tail). Operand keys arrive non-monotonically and mix array
    // (sparse) and bitset (dense) chunks, so splice, tail-append, same-key OR, and the
    // array↔bitset promotion all fire.
    auto const dense = []( std::uint32_t const chunk ) {  // >= threshold -> bitset chunk
        std::vector<std::uint32_t> v;
        std::uint32_t const base{ chunk * 65'536U };
        for ( std::uint32_t i{ 0U }; i < 10'000U; ++i ) { v.push_back( base + i ); }
        return v;
    };
    auto const sparse = []( std::uint32_t const chunk ) {  // < threshold -> array chunk
        std::vector<std::uint32_t> v;
        std::uint32_t const base{ chunk * 65'536U };
        for ( std::uint32_t i{ 0U }; i < 200U; ++i ) { v.push_back( base + i * 211U ); }
        return v;
    };
    auto const concat = []( std::vector<std::vector<std::uint32_t>> const & parts ) {
        std::vector<std::uint32_t> v;
        for ( auto const & p : parts ) { v.insert( v.end(), p.begin(), p.end() ); }
        return v;
    };

    std::vector<std::vector<std::uint32_t>> operand_values;
    operand_values.push_back( concat( { dense( 0 ), dense( 4 ) } ) );                 // seed accumulator: keys {0,4}
    operand_values.push_back( sparse( 2 ) );                                          // key 2 -> SPLICE between 0 and 4
    operand_values.push_back( concat( { dense( 1 ), dense( 3 ), dense( 6 ) } ) );     // 1,3 -> SPLICE; 6 -> tail
    operand_values.push_back( concat( { sparse( 0 ), dense( 2 ), sparse( 5 ) } ) );   // same-key OR on 0 (mixed) & 2; 5 -> SPLICE between 4 and 6

    std::vector<TestBitmap> operands;
    operands.reserve( operand_values.size() );
    roaring_bitmap_holder expected_roaring;
    for ( auto const & values : operand_values ) {
        operands.push_back( make_bitmap( values ) );
        for ( auto const value : values ) { roaring_bitmap_add( expected_roaring.bitmap, value ); }
    }

    std::vector<TestBitmap const *> operand_ptrs;
    operand_ptrs.reserve( operands.size() );
    for ( auto const & operand : operands ) { operand_ptrs.push_back( &operand ); }

    TestBitmap result;
    result.or_many_in_place( { operand_ptrs.data(), operand_ptrs.size() } );

    auto const expected{ to_vector( expected_roaring.bitmap ) };
    EXPECT_EQ( result.size(), expected.size() );
    EXPECT_EQ( result.to_vector(), expected );
}

TEST(FrsrRoaringCrosscheck, InPlaceUnionMatchesCRoaringAcrossMergeShapes) {
    // Exercises the in-place union_merge backing operator|=: same-key OR, LHS-only
    // keys, and RHS-only keys spliced in — across array/bitset/run container shapes,
    // checked after every accumulation step, plus the self-union and empty edges.
    std::vector<std::vector<std::uint32_t>> operand_values;
    {
        std::vector<std::uint32_t> dense_low;  // dense -> bitset chunk
        for ( std::uint32_t value{ 0U }; value < 5'000U; ++value ) {
            dense_low.push_back( value );
        }
        operand_values.push_back( std::move( dense_low ) );
    }
    {
        std::vector<std::uint32_t> sparse_high;  // scattered -> array chunks with new keys
        for ( std::uint32_t index{ 0U }; index < 400U; ++index ) {
            sparse_high.push_back( 100'000U + index * 977U );
        }
        operand_values.push_back( std::move( sparse_high ) );
    }
    {
        std::vector<std::uint32_t> dense_overlap;  // overlaps operand 0's chunk + a fresh chunk
        for ( std::uint32_t value{ 3'000U }; value < 9'000U; ++value ) {
            dense_overlap.push_back( value );
        }
        operand_values.push_back( std::move( dense_overlap ) );
    }
    {
        std::vector<std::uint32_t> contiguous_run;  // long run in a new chunk
        for ( std::uint32_t value{ 200'000U }; value < 215'000U; ++value ) {
            contiguous_run.push_back( value );
        }
        operand_values.push_back( std::move( contiguous_run ) );
    }

    TestBitmap accumulator;
    roaring_bitmap_holder reference;
    for ( auto const & values : operand_values ) {
        accumulator |= make_bitmap( values );
        for ( auto const value : values ) {
            roaring_bitmap_add( reference.bitmap, value );
        }
        auto const expected{ to_vector( reference.bitmap ) };
        EXPECT_EQ( accumulator.size(), expected.size() );
        EXPECT_EQ( accumulator.to_vector(), expected );
    }

    // Self-union is the identity (the in-place merge must not alias its operand).
    auto const before_self{ accumulator.to_vector() };
    accumulator |= accumulator;
    EXPECT_EQ( accumulator.to_vector(), before_self );

    // Union with an empty bitmap leaves the accumulator unchanged.
    TestBitmap const empty;
    accumulator |= empty;
    EXPECT_EQ( accumulator.to_vector(), before_self );

    // Empty |= dense yields exactly the dense operand.
    TestBitmap from_empty;
    from_empty |= make_bitmap( operand_values.front() );
    EXPECT_EQ( from_empty.to_vector(), make_bitmap( operand_values.front() ).to_vector() );

    // Interleaved mid-insertion: a chunk that lands strictly between two existing,
    // non-adjacent chunks goes through the splice path (not the tail append).
    {
        std::vector<std::uint32_t> const low_chunk{ 10U, 20U, 30U };                       // chunk 0
        std::vector<std::uint32_t> const high_chunk{ 5U * 65'536U + 1U, 5U * 65'536U + 9U }; // chunk 5
        std::vector<std::uint32_t> const mid_chunk{ 2U * 65'536U + 7U };                   // chunk 2, between 0 and 5
        TestBitmap gapped;
        roaring_bitmap_holder gapped_reference;
        for ( auto const & values : { low_chunk, high_chunk, mid_chunk } ) {
            gapped |= make_bitmap( values );
            for ( auto const value : values ) {
                roaring_bitmap_add( gapped_reference.bitmap, value );
            }
        }
        EXPECT_EQ( gapped.to_vector(), to_vector( gapped_reference.bitmap ) );
    }
}

TEST(FrsrRoaringCrosscheck, InPlaceUnionWithFullChunkLeftMatchesCRoaring) {
    // Guards the full-LHS-container shortcut in the in-place union: OR-ing into a
    // saturated chunk must be a no-op that leaves the result identical to CRoaring's
    // roaring_bitmap_or_inplace (which takes the same shortcut). A fully populated
    // low chunk [0, 65535] also exercises a full chunk that is NOT the merge's first.
    auto const full_chunk{ []( std::uint32_t const chunk ) {
        std::vector<std::uint32_t> values;
        values.reserve( 65'536U );
        for ( std::uint32_t low{ 0U }; low < 65'536U; ++low ) {
            values.push_back( chunk * 65'536U + low );
        }
        return values;
    } };

    std::vector<std::uint32_t> lhs_values{ full_chunk( 0U ) };              // chunk 0: full
    for ( auto const value : full_chunk( 3U ) ) { lhs_values.push_back( value ); }  // chunk 3: full
    lhs_values.push_back( 1U * 65'536U + 42U );                            // chunk 1: a stray bit

    std::vector<std::uint32_t> const rhs_subset_of_full{ 100U, 5'000U, 60'000U };   // all inside full chunk 0
    std::vector<std::uint32_t> rhs_spanning;                                        // overlaps chunks 0..4
    for ( std::uint32_t value{ 25'000U }; value < 4U * 65'536U + 10'000U; ++value ) {
        rhs_spanning.push_back( value );
    }

    for ( auto const & rhs_values : { rhs_subset_of_full, rhs_spanning } ) {
        auto accumulator{ make_bitmap( lhs_values ) };
        accumulator |= make_bitmap( rhs_values );

        roaring_bitmap_holder reference{ make_roaring( lhs_values ) };
        roaring_bitmap_holder const rhs_reference{ make_roaring( rhs_values ) };
        roaring_bitmap_or_inplace( reference.bitmap, rhs_reference.bitmap );

        EXPECT_EQ( accumulator.to_vector(), to_vector( reference.bitmap ) );
    }
}

TEST(FrsrRoaringCrosscheck, IntersectArraysSimdMatchesCRoaring) {
    // Exercises the SSE4.2 vectorized array×array intersection (intersect_array_array_sse42)
    // against CRoaring's roaring_bitmap_and across the kernel's edge cases: blocks well
    // over 8 elements (the SIMD path), the value 0 present in both operands (the
    // PCMPESTRM zero-tolerant phase), sizes that are not multiples of 8 (the scalar tail),
    // high/low overlap, identical and disjoint operands, and an empty operand. The arrays
    // stay below the bitset threshold so both chunks remain array containers.
    auto check{ []( std::vector<std::uint32_t> const & lhs_values,
                    std::vector<std::uint32_t> const & rhs_values ) {
        auto const lhs{ make_bitmap( lhs_values ) };
        auto const rhs{ make_bitmap( rhs_values ) };
        auto const lhs_roaring{ make_roaring( lhs_values ) };
        auto const rhs_roaring{ make_roaring( rhs_values ) };
        roaring_bitmap_holder const expected{ roaring_bitmap_and( lhs_roaring.bitmap, rhs_roaring.bitmap ) };
        EXPECT_EQ( ( lhs & rhs ).to_vector(), to_vector( expected.bitmap ) );
        EXPECT_EQ( ( rhs & lhs ).to_vector(), to_vector( expected.bitmap ) );
    } };

    auto const range{ []( std::uint32_t const begin, std::uint32_t const end, std::uint32_t const stride = 1U ) {
        std::vector<std::uint32_t> values;
        for ( std::uint32_t value{ begin }; value < end; value += stride ) {
            values.push_back( value );
        }
        return values;
    } };

    // Dense overlapping ranges including 0; result spans many full SIMD blocks + a tail.
    check( range( 0U, 1000U ), range( 250U, 1250U ) );          // high overlap, contains 0, tail (1000 % 8 != 0)
    check( range( 0U, 1003U ), range( 7U, 1011U ) );            // odd sizes -> scalar tail, 0 present
    check( range( 0U, 800U ), range( 720U, 1520U ) );           // low overlap
    check( range( 0U, 2000U, 2U ), range( 1U, 2000U, 2U ) );    // disjoint (evens vs odds)
    check( range( 0U, 500U ), range( 0U, 500U ) );              // identical
    check( range( 0U, 64U ), {} );                              // empty rhs
    check( {}, range( 0U, 64U ) );                              // empty lhs
    check( { 0U }, { 0U } );                                    // single shared element 0
    check( range( 0U, 17U ), range( 0U, 9U ) );                 // small, both with sub-8 tails

    // Scattered (non-contiguous) arrays that still stay array containers, with 0 present.
    std::vector<std::uint32_t> sparse_a, sparse_b;
    for ( std::uint32_t i{ 0U }; i < 600U; ++i ) { sparse_a.push_back( i * 5U ); }
    for ( std::uint32_t i{ 0U }; i < 600U; ++i ) { sparse_b.push_back( i * 3U ); }
    check( sparse_a, sparse_b );
}

} // namespace

