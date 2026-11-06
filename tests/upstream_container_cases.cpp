// The container cases of the reference's unit tests (tests/bitset_container_unit.c, tests/array_container_unit.c,
// tests/run_container_unit.c, tests/util_unit.c, tests/threads_unit.cpp), replayed against bitmap and, where the
// case is about one container, against a container handle. The tests carry the names of the upstream ones.
//
// The reference's containers can be empty and are shrunk one at a time; here a chunk never holds an empty container
// and shrink_to_fit() is the bitmap's, which gives back every container's payload slack. A failed allocation throws
// std::bad_alloc instead of answering 0 (unless the build makes allocation failures terminate).

#include "allocation_fault.hpp"

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using namespace frsr::roaring;

using Bitmap    = bitmap<std::uint32_t>;
using Bitmap64  = bitmap<std::uint64_t>;
using CowBitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>;
using Layout    = detail::default_layout<std::uint32_t>;
using Handle    = detail::container_handle<Layout, detail::cow_value_semantics>;

constexpr std::uint32_t chunk{ 1U << 16 };

// An array container of 188 values whose payload grew to hold 3000: removals leave the payload as large as it was.
template <typename B>
void add_array_with_slack( B & bitmap, std::uint32_t const base ) {
    for ( std::uint32_t i{ 0 }; i < 3000; ++i ) {
        std::ignore = bitmap.add( base + 5U * i );
    }
    for ( std::uint32_t i{ 0 }; i < 3000; ++i ) {
        if ( i % 16 != 0 ) {
            std::ignore = bitmap.remove( base + 5U * i );
        }
    }
}

// A run container of 500 runs whose payload grew to hold 1000: one run cut into 1000 by removing every sixteenth
// value, then every second cut filled again.
template <typename B>
void add_runs_with_slack( B & bitmap, std::uint32_t const base ) {
    B range;
    range.add_closed_range( base, base + 15999U );
    range.optimize_keep_bitsets();
    bitmap |= range;
    for ( std::uint32_t i{ 0 }; i < 1000; ++i ) {
        std::ignore = bitmap.remove( base + 16U * i + 15U );
    }
    for ( std::uint32_t i{ 0 }; i < 1000; i += 2 ) {
        std::ignore = bitmap.add( base + 16U * i + 15U );
    }
}

// ---- tests/bitset_container_unit.c ----

// min_max_test: the smallest and the largest value of a bitset, also when they are in its first word (#925: the
// reference's bitset_container_maximum never looked at word 0). Through a bitmap whose only container is a bitset,
// and through a bitset handle whose cached endpoints were dropped and are recomputed from the words.
TEST( UpstreamContainers, min_max_test ) {
    for ( auto const & values : std::initializer_list<std::vector<std::uint32_t>>{
        { 3, 5 },
        { 0 },
        { 0, 63 },
        { 0, 63, 64 },
        { 0, 63, 64, 65535 },
    } ) {
        SCOPED_TRACE( ::testing::Message() << values.size() << " values up to " << values.back() );
        Bitmap bitmap{ values };
        bitmap.optimize_for_speed( 1 );
        ASSERT_EQ( bitmap.statistics().bitset_containers, 1U );
        EXPECT_EQ( bitmap.front(), values.front() );
        EXPECT_EQ( bitmap.back(), values.back() );

        Handle::word_array words{};
        for ( auto const value : values ) {
            words[ value >> 6U ] |= std::uint64_t{ 1 } << ( value & 63U );
        }
        auto handle{ Handle::make_bitset_from_words( words ) };
        handle.mark_endpoints_stale();
        EXPECT_EQ( handle.min_value(), values.front() );
        EXPECT_EQ( handle.max_value(), values.back() );
    }
}

// ---- tests/array_container_unit.c, tests/run_container_unit.c ----

// shrink_to_fit_answers_bytes_test of the run container: the slack of a run payload is given back, once, and the
// values stay. (The array container's case is FrsrRoaringSmoke.ShrinkToFitReturnsPayloadSlackAndKeepsContents.)
TEST( UpstreamContainers, run_shrink_to_fit_answers_bytes_test ) {
    Bitmap bitmap;
    add_runs_with_slack( bitmap, 0U );
    ASSERT_EQ( bitmap.statistics().run_containers, 1U );
    auto const values{ bitmap.to_vector() };
    ASSERT_EQ( values.size(), 15500U );

    EXPECT_GT( bitmap.shrink_to_fit(), 0U );
    EXPECT_EQ( bitmap.shrink_to_fit(), 0U );
    EXPECT_EQ( bitmap.validate(), nullptr );
    EXPECT_EQ( bitmap.statistics().run_containers, 1U );
    EXPECT_EQ( bitmap.to_vector(), values );
}

// shrink_to_fit_emptied_holds_no_more_than_new_test: a bitmap whose values were all removed holds, once shrunk, no
// more than a new one, and it is still usable.
template <typename B>
void emptied_holds_no_more_than_new() {
    auto const new_bytes{ B{}.byte_size() };
    B bitmap;
    for ( std::uint32_t key{ 0 }; key < 100; ++key ) {
        std::ignore = bitmap.add( key * chunk + 7U );
    }
    for ( std::uint32_t key{ 0 }; key < 100; ++key ) {
        std::ignore = bitmap.remove( key * chunk + 7U );
    }
    ASSERT_TRUE( bitmap.empty() );

    std::ignore = bitmap.shrink_to_fit();
    EXPECT_LE( bitmap.byte_size(), new_bytes );
    EXPECT_TRUE( bitmap.add( 3U ) );
    EXPECT_TRUE( bitmap.contains( 3U ) );
    EXPECT_EQ( bitmap.size(), 1U );
    EXPECT_EQ( bitmap.validate(), nullptr );
}

TEST( UpstreamContainers, shrink_to_fit_emptied_holds_no_more_than_new_test ) {
    emptied_holds_no_more_than_new<Bitmap>();
    emptied_holds_no_more_than_new<Bitmap64>();
}

// shrink_to_fit_failed_realloc_test (array and run containers): a shrink whose allocation fails leaves the values as
// they were, and the slack is still there to give back afterwards.
TEST( UpstreamContainers, shrink_to_fit_failed_realloc_test ) {
    if constexpr ( detail::allocation_failure_terminates ) {
        GTEST_SKIP() << "an allocation failure terminates in this build";
    }
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    namespace fault = frsr::roaring::kernel_test::alloc_fault;
    for ( bool const runs : { false, true } ) {
        SCOPED_TRACE( runs ? "run container" : "array container" );
        Bitmap bitmap;
        if ( runs ) {
            add_runs_with_slack( bitmap, 0U );
        } else {
            add_array_with_slack( bitmap, 0U );
        }
        auto const values{ bitmap.to_vector() };
        bool failed{ false };
        try {
            fault::armed const armed{ 1 };
            std::ignore = bitmap.shrink_to_fit();
        } catch ( std::bad_alloc const & ) {
            failed = true;
        }
        EXPECT_TRUE( failed );
        EXPECT_EQ( bitmap.validate(), nullptr );
        EXPECT_EQ( bitmap.to_vector(), values );
        EXPECT_GT( bitmap.shrink_to_fit(), 0U );
        EXPECT_EQ( bitmap.to_vector(), values );
    }
#endif
}

// ---- tests/threads_unit.cpp ----

// run_shrink_to_fit_race_test: shrinking a copy-on-write copy while another thread reads the source. The payloads
// the copy shares are not moved (the shrink gives back nothing), so the reader sees the source unchanged; a sole
// owner of the same containers does have slack to give back. The race itself is for a thread sanitizer to see.
TEST( UpstreamContainers, run_shrink_to_fit_race_test ) {
    auto const make{ [] {
        CowBitmap source;
        add_array_with_slack( source, 0U );
        add_runs_with_slack( source, chunk );
        return source;
    } };
    auto twin{ make() };
    ASSERT_EQ( twin.statistics().array_containers, 1U );
    ASSERT_EQ( twin.statistics().run_containers, 1U );
    ASSERT_GT( twin.shrink_to_fit(), 0U );

    auto const source{ make() };
    auto const expected{ source.to_vector() };
    CowBitmap copy{ source };
    bool read_ok{ true };
    std::size_t freed{ 1 };
    std::thread reader{ [ & ] {
        for ( int round{ 0 }; round < 100; ++round ) {
            read_ok = read_ok && source.to_vector() == expected;
        }
    } };
    std::thread shrinker{ [ & ] { freed = copy.shrink_to_fit(); } };
    reader.join();
    shrinker.join();

    EXPECT_TRUE( read_ok );
    EXPECT_EQ( freed, 0U );
    EXPECT_EQ( copy.to_vector(), expected );
    EXPECT_EQ( source.to_vector(), expected );
}

// ---- tests/util_unit.c ----

// avx512_extract_uint16_sparse_matches_scalar: the AVX-512 decoder of a bitset into 16-bit values, on sparse words
// spread out or in runs (so that some words hold more than 32 values), on both sides of 1024 values (where the
// decoder is first chosen) and of 2048, into an output of exactly the decoded count. The reference's decoder also
// stops at a given capacity; this one always decodes the whole block. Through a bitmap, the bitset of the same words
// demotes to the same values on any CPU.
TEST( UpstreamContainers, avx512_extract_uint16_sparse_matches_scalar ) {
    std::mt19937_64 rng{ 1234 };
    for ( std::size_t const target : { 1U, 7U, 64U, 500U, 1000U, 1023U, 1024U, 1025U, 2047U, 2048U, 3000U, 4096U } ) {
        for ( bool const runs : { false, true } ) {
            SCOPED_TRACE( ::testing::Message() << target << " values, runs " << runs );
            Handle::word_array words{};
            std::size_t cardinality{ 0 };
            while ( cardinality < target ) {
                auto const start{ static_cast<std::uint32_t>( rng() % chunk ) };
                auto const length{ runs ? 1U + static_cast<std::uint32_t>( rng() % 100 ) : 1U };
                for ( auto value{ start }; value < start + length && value < chunk && cardinality < target; ++value ) {
                    auto const bit{ std::uint64_t{ 1 } << ( value % 64U ) };
                    if ( ( words[ value / 64U ] & bit ) == 0 ) {
                        words[ value / 64U ] |= bit;
                        ++cardinality;
                    }
                }
            }
            std::vector<std::uint16_t> lows;
            for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
                for ( auto word{ words[ index ] }; word != 0; word &= word - 1 ) {
                    lows.emplace_back( static_cast<std::uint16_t>( index * 64U + static_cast<std::size_t>( std::countr_zero( word ) ) ) );
                }
            }
            ASSERT_EQ( lows.size(), cardinality );

#if FRSR_ROARING_X86_V4
            if ( detail::have_x86_v4() ) {
                std::vector<std::uint16_t> decoded( cardinality );
                EXPECT_EQ( detail::x86_v4::extract_setbits_uint16( words.data(), words.size(), decoded.data() ), cardinality );
                EXPECT_EQ( decoded, lows );
            }
#endif
            std::vector<std::uint32_t> values;
            for ( auto const low : lows ) {
                values.emplace_back( 3U * chunk + low );
            }
            Bitmap bitmap;
            for ( auto const value : values ) {
                std::ignore = bitmap.add( value );
            }
            bitmap.optimize_for_speed( 1 );
            ASSERT_EQ( bitmap.statistics().bitset_containers, 1U );
            bitmap.optimize();
            EXPECT_EQ( bitmap.validate(), nullptr );
            EXPECT_EQ( bitmap.to_vector(), values );
        }
    }
}

} // namespace
