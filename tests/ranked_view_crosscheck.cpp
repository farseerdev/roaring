// The ranked frozen image against CRoaring: roaring_bitmap_rank (inclusive: ids <= x), roaring_bitmap_get_index and
// roaring_bitmap_select answer the same questions as count_below, index_of and select.

#include "ranked_view_support.hpp"

#include <roaring/roaring.h>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::ranked_test;

TEST( FrsrRoaringRankedCrosscheck, AgreesWithCRoaringOnEveryShapeAndBuild ) {
    for ( auto const & [ name, ids ] : shapes() ) {
        SCOPED_TRACE( name );
        auto * const reference{ roaring_bitmap_of_ptr( ids.size(), ids.data() ) };
        roaring_bitmap_run_optimize( reference );
        for ( auto const how : { build::plain, build::optimized, build::keep_bitsets } ) {
            SCOPED_TRACE( name_of( how ) );
            auto const frozen{ freeze( make_bitmap( ids, how ), { .ranked = true } ) };
            auto const view{ table_view::from( frozen.bytes() ) };
            ASSERT_TRUE( view );
            ASSERT_EQ( view.size(), roaring_bitmap_get_cardinality( reference ) );
            for ( auto const x : probes( ids ) ) {
                auto const present{ roaring_bitmap_contains( reference, x ) };
                ASSERT_EQ( view.contains( x ), present ) << x;
                ASSERT_EQ( view.count_below( x ), roaring_bitmap_rank( reference, x ) - ( present ? 1U : 0U ) ) << x;
                auto const index{ roaring_bitmap_get_index( reference, x ) };
                ASSERT_EQ( view.index_of( x ), index < 0 ? table_view::npos : static_cast<std::uint32_t>( index ) ) << x;
            }
            for ( std::uint32_t i{ 0 }; i < ids.size(); i += 1 + i % 7 ) {
                std::uint32_t expected{};
                ASSERT_TRUE( roaring_bitmap_select( reference, i, &expected ) );
                ASSERT_EQ( view.select( i ), expected ) << i;
            }
        }
        roaring_bitmap_free( reference );
    }
}

} // namespace
