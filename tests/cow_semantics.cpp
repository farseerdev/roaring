// Copy-on-write semantics of container_handle under the refcounted policies
// (Models 2 and 3) plus the Model-1 value-semantics baseline.
//
// The primary correctness gate is allocation accounting: this TU replaces the
// global operator new/delete with counting versions, so sharing (copy without
// allocation), the write barrier (exactly one clone allocation), and last-owner
// release (live-allocation balance) are asserted directly rather than inferred.
//
// Under ThreadSanitizer the counting harness is disabled: TSan's runtime ships
// its own global replacement operators (duplicate-symbol link error otherwise),
// and it intercepts allocation itself. The payload-pointer-identity assertions
// and the threaded race test — the parts TSan is for — remain active.

#include <frsr/roaring/container_handle.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

#if defined( __SANITIZE_THREAD__ )
#   define FRSR_TESTS_COUNT_ALLOCATIONS 0
#elif defined( __has_feature )
#   if __has_feature( thread_sanitizer )
#       define FRSR_TESTS_COUNT_ALLOCATIONS 0
#   endif
#endif
#ifndef FRSR_TESTS_COUNT_ALLOCATIONS
#   define FRSR_TESTS_COUNT_ALLOCATIONS 1
#endif

namespace {

constexpr bool allocation_counting_enabled{ FRSR_TESTS_COUNT_ALLOCATIONS != 0 };

// Allocation-count assertion, inert when the counting harness is disabled
// (TSan builds — the pointer-identity and race assertions still run there).
#define FRSR_EXPECT_ALLOC_EQ( actual, expected ) \
    do { if ( allocation_counting_enabled ) { EXPECT_EQ( actual, expected ); } } while ( false )

std::atomic<std::size_t> g_total_allocations{ 0 };
std::atomic<std::size_t> g_live_allocations { 0 };

#if FRSR_TESTS_COUNT_ALLOCATIONS
[[nodiscard]] void * counted_allocate( std::size_t const size ) {
    auto * const p{ std::malloc( size ? size : 1 ) };
    if ( !p ) {
        throw std::bad_alloc{};
    }
    g_total_allocations.fetch_add( 1, std::memory_order_relaxed );
    g_live_allocations .fetch_add( 1, std::memory_order_relaxed );
    return p;
}

void counted_free( void * const p ) noexcept {
    if ( p ) {
        g_live_allocations.fetch_sub( 1, std::memory_order_relaxed );
        std::free( p );
    }
}
#endif // FRSR_TESTS_COUNT_ALLOCATIONS

} // anonymous namespace

#if FRSR_TESTS_COUNT_ALLOCATIONS
void * operator new  ( std::size_t const size ) { return counted_allocate( size ); }
void * operator new[]( std::size_t const size ) { return counted_allocate( size ); }
void operator delete  ( void * const p ) noexcept { counted_free( p ); }
void operator delete[]( void * const p ) noexcept { counted_free( p ); }
void operator delete  ( void * const p, std::size_t ) noexcept { counted_free( p ); }
void operator delete[]( void * const p, std::size_t ) noexcept { counted_free( p ); }
#endif // FRSR_TESTS_COUNT_ALLOCATIONS

namespace {

using namespace frsr::roaring::detail;

using layout = default_layout<std::uint32_t>;

template <typename CowPolicy>
using handle_t = container_handle<layout, CowPolicy>;

using low_type = layout::low_type;

// Enough values to force the payload off the 16-byte inline body.
template <typename CowPolicy>
[[nodiscard]] handle_t<CowPolicy> make_spilled_array( std::uint32_t const size = 64 ) {
    std::vector<low_type> values;
    values.reserve( size );
    for ( std::uint32_t i{ 0 }; i < size; ++i ) {
        values.push_back( static_cast<low_type>( i * 3U ) );
    }
    return handle_t<CowPolicy>::make_array_from_sorted( { values.data(), values.size() } );
}

[[nodiscard]] void const * payload_of( auto const & handle ) noexcept { return handle.payload_data_raw(); }

template <typename CowPolicy>
class RefcountedCow : public ::testing::Test {};

using refcounted_policies = ::testing::Types<cow_atomic_refcount, cow_unsynchronized_refcount>;
TYPED_TEST_SUITE( RefcountedCow, refcounted_policies );

TYPED_TEST( RefcountedCow, SpilledHandleIsBornShared ) {
    auto const handle{ make_spilled_array<TypeParam>() };
    EXPECT_TRUE( handle.spilled() );
    EXPECT_EQ( handle.owner(), storage_ownership::shared );
}

TYPED_TEST( RefcountedCow, CopySharesWithoutAllocating ) {
    auto const original{ make_spilled_array<TypeParam>() };
    auto const allocations_before{ g_total_allocations.load() };
    auto const copy{ original };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    EXPECT_EQ( payload_of( copy ), payload_of( original ) );
    EXPECT_EQ( copy.cardinality(), original.cardinality() );
    EXPECT_TRUE( copy.as_array().contains( low_type{ 63 } ) );
}

TYPED_TEST( RefcountedCow, WriteBarrierClonesSharedPayload ) {
    auto original{ make_spilled_array<TypeParam>() };
    auto copy{ original };

    auto const allocations_before{ g_total_allocations.load() };
    auto mutable_copy{ copy.as_array() };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before + 1 );
    EXPECT_NE( payload_of( copy ), payload_of( original ) );

    ASSERT_TRUE( mutable_copy.add( low_type{ 1 } ) );
    EXPECT_TRUE ( copy    .as_array().contains( low_type{ 1 } ) );
    EXPECT_FALSE( std::as_const( original ).as_array().contains( low_type{ 1 } ) );
    EXPECT_EQ( original.cardinality(), 64U );
    EXPECT_EQ( copy    .cardinality(), 65U );
}

TYPED_TEST( RefcountedCow, SoleReferentMutatesInPlace ) {
    auto handle{ make_spilled_array<TypeParam>() };
    auto const * const payload_before{ payload_of( handle ) };
    auto const allocations_before{ g_total_allocations.load() };
    auto mutable_ref{ handle.as_array() };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    EXPECT_EQ( payload_of( handle ), payload_before );
    ASSERT_TRUE( mutable_ref.remove( low_type{ 0 } ) );
    EXPECT_EQ( handle.cardinality(), 63U );
}

TYPED_TEST( RefcountedCow, LastOwnerFreesPayload ) {
    auto const live_before{ g_live_allocations.load() };
    {
        auto const original{ make_spilled_array<TypeParam>() };
        auto first_copy { original };
        auto const second_copy{ original };
        std::ignore = first_copy.as_array().add( low_type{ 1 } );   // clone under rc == 3
        auto const third_copy{ first_copy };                        // reshare the clone
        std::ignore = third_copy;
        std::ignore = second_copy;
    }
    FRSR_EXPECT_ALLOC_EQ( g_live_allocations.load(), live_before );
}

TYPED_TEST( RefcountedCow, InlinePayloadCopiesByValue ) {
    low_type const values[]{ 1, 2, 3, 4 };
    auto const original{ handle_t<TypeParam>::make_array_from_sorted( values ) };
    ASSERT_FALSE( original.spilled() );

    auto const allocations_before{ g_total_allocations.load() };
    auto copy{ original };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );

    ASSERT_TRUE( copy.as_array().add( low_type{ 5 } ) );
    EXPECT_EQ( original.cardinality(), 4U );
    EXPECT_EQ( copy    .cardinality(), 5U );
}

TYPED_TEST( RefcountedCow, BitsetCopySharesAndBarrierClones ) {
    auto original{ handle_t<TypeParam>::make_bitset_zeroed() };
    for ( low_type value{ 0 }; value < 200; ++value ) {
        ASSERT_TRUE( original.as_bitset().add( value ) );
    }

    auto copy{ original };
    EXPECT_EQ( payload_of( copy ), payload_of( original ) );

    ASSERT_TRUE( copy.as_bitset().add( low_type{ 1000 } ) );
    EXPECT_NE( payload_of( copy ), payload_of( original ) );
    EXPECT_FALSE( std::as_const( original ).as_bitset().contains( low_type{ 1000 } ) );
    EXPECT_EQ( original.cardinality(), 200U );
    EXPECT_EQ( copy    .cardinality(), 201U );
}

TYPED_TEST( RefcountedCow, MoveTransfersWithoutRefcountTraffic ) {
    auto source{ make_spilled_array<TypeParam>() };
    auto const * const payload_before{ payload_of( source ) };
    auto const allocations_before{ g_total_allocations.load() };
    auto const live_before{ g_live_allocations.load() };

    auto const destination{ std::move( source ) };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    FRSR_EXPECT_ALLOC_EQ( g_live_allocations .load(), live_before );
    EXPECT_EQ( payload_of( destination ), payload_before );
    EXPECT_TRUE( source.empty() );
    EXPECT_FALSE( source.spilled() );
}

TYPED_TEST( RefcountedCow, AssignmentOverASharingPairStaysBalanced ) {
    auto const live_before{ g_live_allocations.load() };
    {
        auto const original{ make_spilled_array<TypeParam>() };
        auto other{ original };            // share
        other = original;                  // reassign over the same shared payload
        EXPECT_EQ( payload_of( other ), payload_of( original ) );
        other = make_spilled_array<TypeParam>( 32 );
        EXPECT_NE( payload_of( other ), payload_of( original ) );
    }
    FRSR_EXPECT_ALLOC_EQ( g_live_allocations.load(), live_before );
}

TEST( ValueSemanticsCow, CopyDeepClones ) {
    auto const original{ make_spilled_array<cow_value_semantics>() };
    EXPECT_EQ( original.owner(), storage_ownership::unique );

    auto const allocations_before{ g_total_allocations.load() };
    auto const copy{ original };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before + 1 );
    EXPECT_NE( payload_of( copy ), payload_of( original ) );
}

// ---- borrowed (mmap-master) ownership — Phase 3 ------------------------------
//
// A borrowed handle aliases caller-owned read-only bytes. Under EVERY policy:
// construction and copies allocate nothing and alias the master bytes; the
// write barrier clones to private storage (master bytes untouched); teardown
// frees nothing.

template <typename CowPolicy>
class BorrowedPayload : public ::testing::Test {};

using all_policies = ::testing::Types<cow_value_semantics, cow_atomic_refcount, cow_unsynchronized_refcount>;
TYPED_TEST_SUITE( BorrowedPayload, all_policies );

// A stand-in for an mmapped frozen payload: a plain sorted value buffer.
[[nodiscard]] std::vector<low_type> make_master_values( std::uint32_t const size = 64 ) {
    std::vector<low_type> values;
    values.reserve( size );
    for ( std::uint32_t i{ 0 }; i < size; ++i ) {
        values.push_back( static_cast<low_type>( i * 3U ) );
    }
    return values;
}

TYPED_TEST( BorrowedPayload, BorrowAliasesWithoutAllocating ) {
    auto const master{ make_master_values() };
    auto const allocations_before{ g_total_allocations.load() };
    auto const handle{ handle_t<TypeParam>::make_borrowed(
        container_kind::array, master.data(), 64U, 64U
    ) };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    EXPECT_EQ( handle.owner(), storage_ownership::borrowed );
    EXPECT_TRUE( handle.spilled() );
    EXPECT_EQ( payload_of( handle ), master.data() );
    EXPECT_EQ( handle.cardinality(), 64U );
    EXPECT_EQ( handle.min_value(), low_type{ 0 } );
    EXPECT_EQ( handle.max_value(), low_type{ 189 } );
    EXPECT_TRUE ( std::as_const( handle ).as_array().contains( low_type{ 63 } ) );
    EXPECT_FALSE( std::as_const( handle ).as_array().contains( low_type{ 62 } ) );
}

TYPED_TEST( BorrowedPayload, CopyOfBorrowedStaysBorrowedZeroCopy ) {
    auto const master{ make_master_values() };
    auto const original{ handle_t<TypeParam>::make_borrowed(
        container_kind::array, master.data(), 64U, 64U
    ) };
    auto const allocations_before{ g_total_allocations.load() };
    auto const copy{ original };
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    EXPECT_EQ( copy.owner(), storage_ownership::borrowed );
    EXPECT_EQ( payload_of( copy ), master.data() );
}

TYPED_TEST( BorrowedPayload, WriteBarrierClonesAndMasterBytesStayIntact ) {
    auto const master{ make_master_values() };
    auto const master_snapshot{ master };
    auto handle{ handle_t<TypeParam>::make_borrowed(
        container_kind::array, master.data(), 64U, 64U
    ) };

    auto const allocations_before{ g_total_allocations.load() };
    auto mutable_ref{ handle.as_array() };   // the barrier: clone-to-private
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before + 1 );
    EXPECT_NE( handle.owner(), storage_ownership::borrowed );
    EXPECT_NE( payload_of( handle ), master.data() );

    ASSERT_TRUE( mutable_ref.add( low_type{ 1 } ) );
    EXPECT_EQ( master, master_snapshot );   // the master is never written
    EXPECT_EQ( handle.cardinality(), 65U );
}

TYPED_TEST( BorrowedPayload, SmallBorrowedPayloadClonesInline ) {
    low_type const master[]{ 1, 2, 3, 4 };
    auto handle{ handle_t<TypeParam>::make_borrowed(
        container_kind::array, master, 4U, 4U
    ) };
    auto const allocations_before{ g_total_allocations.load() };
    ASSERT_TRUE( handle.as_array().add( low_type{ 5 } ) );
    FRSR_EXPECT_ALLOC_EQ( g_total_allocations.load(), allocations_before );
    EXPECT_FALSE( handle.spilled() );
    EXPECT_EQ( handle.owner(), storage_ownership::unique );
    EXPECT_EQ( handle.cardinality(), 5U );
    EXPECT_EQ( master[ 3 ], low_type{ 4 } );
}

TYPED_TEST( BorrowedPayload, TeardownFreesNothing ) {
    auto const master{ make_master_values() };
    auto const live_before{ g_live_allocations.load() };
    {
        auto const original{ handle_t<TypeParam>::make_borrowed(
            container_kind::array, master.data(), 64U, 64U
        ) };
        auto const copy{ original };
        std::ignore = copy;
    }
    FRSR_EXPECT_ALLOC_EQ( g_live_allocations.load(), live_before );
    EXPECT_EQ( master.front(), low_type{ 0 } );
}

TYPED_TEST( BorrowedPayload, BorrowedBitsetBarrierClones ) {
    // A master word block with bits 3 and 500 set.
    auto master{ std::vector<std::uint64_t>( layout::word_count, 0U ) };
    master[ 0 ] = std::uint64_t{ 1 } << 3U;
    master[ 500U >> 6U ] = std::uint64_t{ 1 } << ( 500U & 63U );
    auto const master_snapshot{ master };

    auto handle{ handle_t<TypeParam>::make_borrowed(
        container_kind::bitset, master.data(), static_cast<std::uint32_t>( layout::word_count ), 2U
    ) };
    EXPECT_TRUE ( std::as_const( handle ).as_bitset().contains( low_type{ 3 } ) );
    EXPECT_FALSE( std::as_const( handle ).as_bitset().contains( low_type{ 4 } ) );
    EXPECT_EQ( handle.min_value(), low_type{ 3 } );     // lazy endpoint recompute path
    EXPECT_EQ( handle.max_value(), low_type{ 500 } );

    ASSERT_TRUE( handle.as_bitset().add( low_type{ 1000 } ) );
    EXPECT_NE( payload_of( handle ), master.data() );
    EXPECT_EQ( master, master_snapshot );
    EXPECT_EQ( handle.cardinality(), 3U );
}

// The TSan target (gates on Linux): concurrent copy-a-shared-handle +
// mutate-your-own-copy + destroy, against one long-lived shared original.
TEST( AtomicCowThreaded, CopyMutateDestroyRace ) {
    auto const live_before{ g_live_allocations.load() };
    {
        auto const original{ make_spilled_array<cow_atomic_refcount>() };

        constexpr unsigned thread_count{ 8 };
        constexpr unsigned iterations  { 2000 };
        std::vector<std::thread> threads;
        threads.reserve( thread_count );
        for ( unsigned t{ 0 }; t < thread_count; ++t ) {
            threads.emplace_back( [&original, t] {
                for ( unsigned i{ 0 }; i < iterations; ++i ) {
                    auto copy{ original };
                    std::ignore = copy.as_array().add( static_cast<low_type>( 1 + t ) );
                }
            } );
        }
        for ( auto & thread : threads ) {
            thread.join();
        }

        EXPECT_EQ( original.cardinality(), 64U );
        EXPECT_FALSE( std::as_const( original ).as_array().contains( low_type{ 1 } ) );
    }
    FRSR_EXPECT_ALLOC_EQ( g_live_allocations.load(), live_before );
}

} // anonymous namespace
