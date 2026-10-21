// The exception specifications of the bitmap operations under the allocation policy in force (noexcept_checks.hpp
// asserts them at compile time), and that the operations that never allocate behave.

#include "noexcept_checks.hpp"

#include <gtest/gtest.h>

namespace {

TEST( NoexceptPolicy, NonAllocatingOperationsNeverThrow ) {
    frsr::roaring::bitmap<std::uint32_t> a;
    a.add_many( std::vector<std::uint32_t>{ 1, 5, 70'000 } );
    auto b{ a };
    a.swap( b );
    a.clear();
    EXPECT_TRUE( a.empty() );
    EXPECT_EQ( b.size(), 3U );
    auto moved{ std::move( b ) };
    EXPECT_EQ( moved.size(), 3U );
}

TEST( NoexceptPolicy, TheDefaultBuildLetsAllocationFailuresThrow ) {
    // The test build keeps the default policy, so the operations that allocate are not noexcept and the
    // fault-injection tests can observe std::bad_alloc.
    EXPECT_FALSE( frsr::roaring::noexcept_checks::terminates );
}

} // namespace
