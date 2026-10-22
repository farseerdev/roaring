// What an operation reading bitmaps writes of them (see view_support.hpp for how that is looked at). The n-way AND of a
// single bitmap used to take another reference to each of its containers.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include "interop_support.hpp"
#include "view_support.hpp"

#include <array>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace frsr::roaring;
using namespace frsr::roaring::interop_test;
using frsr::roaring::detail::bitmap_test_access;

template <typename Key, typename Cow, typename Run = detail::run_selection_eager>
using bitmap_of = bitmap<Key, default_container_set<Key>, Cow, Run>;

// The policies under which a copy of a bitmap shares its payloads by reference count: where a view's promise
// (it writes no count) means something.
using counted_bitmaps = ::testing::Types<
    bitmap_of<std::uint16_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, detail::cow_atomic_refcount, detail::run_selection_lazy>,
    bitmap_of<std::uint32_t, detail::cow_unsynchronized_refcount>,
    bitmap_of<std::uint64_t, detail::cow_atomic_refcount>
>;

template <typename Bitmap>
class ViewWitness : public ::testing::Test {};
TYPED_TEST_SUITE( ViewWitness, counted_bitmaps );

// The n-way AND of a single bitmap copies its containers where it used to take another reference to each: the bitmap
// is left byte-identical and the result shares nothing with it.
TYPED_TEST( ViewWitness, TheAndOfOneBitmapCopiesItsContainersAndCountsNothing ) {
    using Bitmap = TypeParam;
    using access = bitmap_test_access<Bitmap>;
    std::mt19937_64 rng{ 9 };
    for ( int round{ 0 }; round < 6; ++round ) {
        auto const published{ make_bitmap<Bitmap>( random_content<Bitmap>( rng ), round % 2 == 1 ) };
        auto const before{ access::snapshot( published ) };
        for ( auto const & state : before ) { ASSERT_LE( state.refcount, 1U ); }
        std::array<Bitmap const *, 1> const one{ &published };
        auto const result{ Bitmap::and_many( one ) };
        EXPECT_TRUE( before == access::snapshot( published ) ) << "the n-way AND of one bitmap wrote to it";
        ASSERT_EQ( result.validate(), nullptr );
        EXPECT_TRUE( result == published );
        for ( auto const & state : access::snapshot( result ) ) { EXPECT_LE( state.refcount, 1U ); }
        EXPECT_EQ( Bitmap::and_many_cardinality( one ), published.size() );
    }
}

} // namespace
