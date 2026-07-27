#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>
#include <roaring/roaring64.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using TestBitmap64 = frsr::roaring::bitmap<std::uint64_t>;

struct roaring64_holder {
    roaring64_holder() : bitmap{ roaring64_bitmap_create() } {}
    explicit roaring64_holder( roaring64_bitmap_t * raw ) : bitmap{ raw } {}
    ~roaring64_holder() {
        if ( bitmap != nullptr ) {
            roaring64_bitmap_free( bitmap );
        }
    }
    roaring64_holder( roaring64_holder const & ) = delete;
    roaring64_holder & operator=( roaring64_holder const & ) = delete;
    roaring64_holder( roaring64_holder && other ) noexcept : bitmap{ other.bitmap } { other.bitmap = nullptr; }
    roaring64_holder & operator=( roaring64_holder && other ) noexcept {
        if ( this != &other ) {
            if ( bitmap != nullptr ) {
                roaring64_bitmap_free( bitmap );
            }
            bitmap = other.bitmap;
            other.bitmap = nullptr;
        }
        return *this;
    }
    roaring64_bitmap_t * bitmap{};
};

[[nodiscard]] auto make_frsr( std::size_t const count, std::uint64_t const step ) {
    TestBitmap64 bitmap;
    for ( std::size_t index{ 0 }; index < count; ++index ) {
        bitmap.add( static_cast<std::uint64_t>( index ) * step );
    }
    return bitmap;
}

[[nodiscard]] auto make_croaring( std::size_t const count, std::uint64_t const step ) {
    roaring64_holder bitmap;
    for ( std::size_t index{ 0 }; index < count; ++index ) {
        roaring64_bitmap_add( bitmap.bitmap, static_cast<std::uint64_t>( index ) * step );
    }
    return bitmap;
}

TEST( FrsrRoaringFullCrosscheck, ContainsHitMissMatrixMatchesCRoaring ) {
    constexpr std::array<std::size_t, 3> counts{ 1000, 10000, 100000 };
    constexpr std::array<std::uint64_t, 7> steps{
        1ULL, 256ULL, 65536ULL, 16777216ULL, 4294967296ULL, 1099511627776ULL, 281474976710656ULL
    };

    for ( auto const count : counts ) {
        for ( auto const step : steps ) {
            auto frsr{ make_frsr( count, step ) };
            auto croaring{ make_croaring( count, step ) };

            for ( std::size_t iteration{ 0 }; iteration < 2048; ++iteration ) {
                auto const slot{ iteration % count };
                auto const hit{ static_cast<std::uint64_t>( slot ) * step };
                auto const miss{ ( static_cast<std::uint64_t>( slot + 1 ) * step ) - 1ULL };

                EXPECT_EQ( frsr.contains( hit ), roaring64_bitmap_contains( croaring.bitmap, hit ) );
                EXPECT_EQ( frsr.contains( miss ), roaring64_bitmap_contains( croaring.bitmap, miss ) );
            }
        }
    }

    // Keep one very-large regression guard without exploding runtime.
    {
        auto frsr{ make_frsr( 1000000, 1 ) };
        auto croaring{ make_croaring( 1000000, 1 ) };
        for ( std::size_t iteration{ 0 }; iteration < 512; ++iteration ) {
            auto const hit{ static_cast<std::uint64_t>( iteration % 1000000 ) };
            auto const miss{ static_cast<std::uint64_t>( 1000000 + iteration ) };
            EXPECT_EQ( frsr.contains( hit ), roaring64_bitmap_contains( croaring.bitmap, hit ) );
            EXPECT_EQ( frsr.contains( miss ), roaring64_bitmap_contains( croaring.bitmap, miss ) );
        }
    }
}

TEST( FrsrRoaringFullCrosscheck, InsertRemoveMatrixMatchesCRoaring ) {
    constexpr std::array<std::size_t, 3> counts{ 1000, 10000, 100000 };
    constexpr std::array<std::uint64_t, 7> steps{
        1ULL, 256ULL, 65536ULL, 16777216ULL, 4294967296ULL, 1099511627776ULL, 281474976710656ULL
    };

    for ( auto const count : counts ) {
        for ( auto const step : steps ) {
            TestBitmap64 frsr;
            roaring64_holder croaring;

            for ( std::size_t index{ 0 }; index < count; ++index ) {
                auto const value{ static_cast<std::uint64_t>( index ) * step };
                frsr.add( value );
                roaring64_bitmap_add( croaring.bitmap, value );
            }
            EXPECT_EQ( frsr.size(), roaring64_bitmap_get_cardinality( croaring.bitmap ) );

            for ( std::size_t index{ 0 }; index < count; ++index ) {
                auto const value{ static_cast<std::uint64_t>( index ) * step };
                frsr.remove( value );
                roaring64_bitmap_remove( croaring.bitmap, value );
            }
            EXPECT_EQ( frsr.size(), roaring64_bitmap_get_cardinality( croaring.bitmap ) );
        }
    }
}

TEST( FrsrRoaringFullCrosscheck, RandomBitmaskContainsAndMutationMatrixMatchesCRoaring ) {
    constexpr std::array<std::uint64_t, 10> bitmasks{
        0x00000000000FFFFFULL, 0x0000000FFFFF0000ULL, 0x000FFFFF00000000ULL, 0xFFFFF00000000000ULL,
        0x000000005DBFC83EULL, 0x00005DBFC83E0000ULL, 0x5DBFC83E00000000ULL, 0x0000493B189604B6ULL,
        0x493B189604B60000ULL, 0x420C684950A2D088ULL
    };

    for ( auto const mask : bitmasks ) {
        std::mt19937_64 rng{ 0xDEADBEEFULL + mask };
        auto next_value = [&]() { return rng() & mask; };

        TestBitmap64 frsr;
        roaring64_holder croaring;
        for ( std::size_t index{ 0 }; index < ( 1U << 14 ); ++index ) {
            auto const value{ next_value() };
            frsr.add( value );
            roaring64_bitmap_add( croaring.bitmap, value );
        }

        for ( std::size_t iteration{ 0 }; iteration < 2000; ++iteration ) {
            auto const probe{ next_value() };
            EXPECT_EQ( frsr.contains( probe ), roaring64_bitmap_contains( croaring.bitmap, probe ) );

            auto const inserted{ next_value() };
            auto const removed{ next_value() };
            frsr.add( inserted );
            frsr.remove( removed );
            roaring64_bitmap_add( croaring.bitmap, inserted );
            roaring64_bitmap_remove( croaring.bitmap, removed );
        }

        EXPECT_EQ( frsr.size(), roaring64_bitmap_get_cardinality( croaring.bitmap ) );

        for ( std::size_t iteration{ 0 }; iteration < 2048; ++iteration ) {
            auto const probe{ next_value() };
            EXPECT_EQ( frsr.contains( probe ), roaring64_bitmap_contains( croaring.bitmap, probe ) );
        }
    }
}

} // namespace
