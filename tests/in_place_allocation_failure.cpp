// An in-place set operation whose allocation fails leaves its left side valid (README, Exceptions): the walk over it
// yields size() keys, it validates, and it holds no key that neither side held. A union keeps every key it held, an
// intersection or a difference holds none it did not hold. Each operation is failed at each of its allocations in
// turn, for every copy-on-write policy and key width, on a left side of a few chunks (a union merges the tables) and
// of many (a union looks the operand's chunks up one at a time).

#include "allocation_fault.hpp"

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fr = frsr::roaring;
namespace alloc_fault = fr::kernel_test::alloc_fault;

template <typename Key, typename Cow>
using bitmap_of = fr::bitmap<Key, fr::default_container_set<Key>, Cow>;

template <typename Bitmap>
class InPlaceAllocationFailure : public ::testing::Test {};

using policies = ::testing::Types<
    bitmap_of<std::uint32_t, fr::detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, fr::detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, fr::detail::cow_unsynchronized_refcount>,
    bitmap_of<std::uint64_t, fr::detail::cow_value_semantics>,
    bitmap_of<std::uint64_t, fr::detail::cow_atomic_refcount>,
    bitmap_of<std::uint64_t, fr::detail::cow_unsynchronized_refcount>
>;
TYPED_TEST_SUITE( InPlaceAllocationFailure, policies );

// An array, a bitset and a run container, and a small array past a gap; `extra_chunks` more small arrays past the
// chunks of the right side.
template <typename Bitmap>
[[nodiscard]] Bitmap make_left( std::uint32_t const extra_chunks ) {
    Bitmap bitmap;
    for ( std::uint32_t i{ 0 }; i < 11; ++i ) {
        std::ignore = bitmap.add( i * 7 );
    }
    for ( std::uint32_t i{ 0 }; i < 5'000; ++i ) {
        std::ignore = bitmap.add( 65'536 + i * 3 );
    }
    bitmap.add_closed_range( 2 * 65'536 + 100, 2 * 65'536 + 5'000 );
    bitmap.optimize();
    for ( std::uint32_t i{ 0 }; i < 20; ++i ) {
        std::ignore = bitmap.add( 5 * 65'536 + i * 11 );
    }
    for ( std::uint32_t chunk{ 0 }; chunk < extra_chunks; ++chunk ) {
        for ( std::uint32_t i{ 0 }; i < 3; ++i ) {
            std::ignore = bitmap.add( ( 16 + chunk ) * 65'536 + i * 13 );
        }
    }
    return bitmap;
}

// Keys in chunks the left side holds (an array, the bitset, the run) and in chunks it does not (between and after).
template <typename Bitmap>
[[nodiscard]] Bitmap make_right() {
    Bitmap bitmap;
    for ( std::uint32_t i{ 0 }; i < 300; ++i ) {
        std::ignore = bitmap.add( i * 5 );
        std::ignore = bitmap.add( 65'536 + i * 4 );
        std::ignore = bitmap.add( 2 * 65'536 + 50 + i * 9 );
        std::ignore = bitmap.add( 3 * 65'536 + i * 5 );
        std::ignore = bitmap.add( 9 * 65'536 + i * 5 );
    }
    return bitmap;
}

template <typename Bitmap>
[[nodiscard]] std::set<typename Bitmap::key_type> keys_of( Bitmap const & bitmap ) {
    std::set<typename Bitmap::key_type> keys;
    for ( auto const value : bitmap ) {
        keys.insert( value );
    }
    return keys;
}

enum class bound { superset_of_left, subset_of_left, none };

template <typename Bitmap>
struct operation {
    std::string name;
    bound limit;
    std::function<void( Bitmap & )> apply;
};

// Fails each operation at its first, second, ... allocation until one runs through, and checks what is left of `left`.
template <typename Bitmap>
void check_failures( Bitmap const & left, Bitmap const & right, std::span<operation<Bitmap> const> const operations ) {
    auto const left_keys{ keys_of( left ) };
    auto both{ left_keys };
    for ( auto const key : right ) {
        both.insert( key );
    }
    for ( auto const & [ name, limit, apply ] : operations ) {
        bool failed_any{ false };
        for ( std::uint32_t fail_at{ 1 }; fail_at < 200; ++fail_at ) {
            Bitmap target{ left };
            bool threw{ false };
            alloc_fault::counter = 0;
            try {
                alloc_fault::armed const armed{ fail_at };
                apply( target );
            } catch ( std::bad_alloc const & ) {
                threw = true;
            }
            alloc_fault::counter = 0;
            failed_any |= threw;
            auto const invalid{ target.validate() };
            EXPECT_EQ( invalid, nullptr ) << name << " failing allocation " << fail_at << ": " << ( invalid != nullptr ? invalid : "" );
            auto const keys{ keys_of( target ) };
            EXPECT_EQ( keys.size(), target.size() ) << name << " failing allocation " << fail_at;
            EXPECT_TRUE( std::ranges::includes( both, keys ) ) << name << " failing allocation " << fail_at;
            if ( limit == bound::superset_of_left ) {
                EXPECT_TRUE( std::ranges::includes( keys, left_keys ) ) << name << " failing allocation " << fail_at;
            }
            if ( limit == bound::subset_of_left ) {
                EXPECT_TRUE( std::ranges::includes( left_keys, keys ) ) << name << " failing allocation " << fail_at;
            }
            EXPECT_EQ( keys_of( left ), left_keys ) << name;
            if ( !threw ) {
                break;
            }
        }
        EXPECT_TRUE( failed_any ) << name;
    }
}

TYPED_TEST( InPlaceAllocationFailure, LeavesTheLeftSideValid ) {
    using bitmap = TypeParam;
    using key_type = typename bitmap::key_type;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    auto const right{ make_right<bitmap>() };
    auto const right_keys{ right.to_vector() };
    std::vector<key_type> const right_values( right_keys.begin(), right_keys.end() );
    std::vector<operation<bitmap>> const operations{
        { "|=",                  bound::superset_of_left, [ & ]( bitmap & b ) { b |= right; } },
        { "&=",                  bound::subset_of_left,   [ & ]( bitmap & b ) { b &= right; } },
        { "-=",                  bound::subset_of_left,   [ & ]( bitmap & b ) { b -= right; } },
        { "^=",                  bound::none,             [ & ]( bitmap & b ) { b ^= right; } },
        { "or_with( keys )",     bound::superset_of_left, [ & ]( bitmap & b ) { b.or_with( std::span{ right_values } ); } },
        { "and_with( keys )",    bound::subset_of_left,   [ & ]( bitmap & b ) { b.and_with( std::span{ right_values } ); } },
        { "andnot_with( keys )", bound::subset_of_left,   [ & ]( bitmap & b ) { b.andnot_with( std::span{ right_values } ); } },
        { "xor_with( keys )",    bound::none,             [ & ]( bitmap & b ) { b.xor_with( std::span{ right_values } ); } },
        { "add_many_sorted",     bound::superset_of_left, [ & ]( bitmap & b ) { b.add_many_sorted( right_values ); } },
        { "remove_many_sorted",  bound::subset_of_left,   [ & ]( bitmap & b ) { b.remove_many_sorted( right_values ); } },
    };
    check_failures<bitmap>( make_left<bitmap>( 0 ), right, operations );
    check_failures<bitmap>( make_left<bitmap>( 48 ), right, operations );
#endif
}

// An add that makes a new chunk and fails leaves the bitmap as it was: no empty chunk is left in the table.
TYPED_TEST( InPlaceAllocationFailure, AnAddOfANewChunkLeavesNoEmptyChunk ) {
    using bitmap = TypeParam;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    constexpr auto key{ 12 * 65'536 + 7 };
    for ( auto const & left : { bitmap{}, make_left<bitmap>( 0 ) } ) {
        auto const left_keys{ keys_of( left ) };
        bool failed_any{ false };
        for ( std::uint32_t fail_at{ 1 }; fail_at < 50; ++fail_at ) {
            bitmap target{ left };
            bool threw{ false };
            alloc_fault::counter = 0;
            try {
                alloc_fault::armed const armed{ fail_at };
                std::ignore = target.add( key );
            } catch ( std::bad_alloc const & ) {
                threw = true;
            }
            alloc_fault::counter = 0;
            failed_any |= threw;
            auto const invalid{ target.validate() };
            EXPECT_EQ( invalid, nullptr ) << "failing allocation " << fail_at << ": " << ( invalid != nullptr ? invalid : "" );
            auto expected{ left_keys };
            if ( !threw ) {
                expected.insert( key );
            }
            EXPECT_EQ( keys_of( target ), expected ) << "failing allocation " << fail_at;
            EXPECT_EQ( target.size(), expected.size() ) << "failing allocation " << fail_at;
            if ( !threw ) {
                break;
            }
        }
        EXPECT_TRUE( failed_any || !left.empty() );
    }
#endif
}

} // namespace
