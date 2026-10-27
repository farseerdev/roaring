// Subtraction from run-encoded chunks cross-checked against CRoaring by contents: both libraries get the same
// receiver (a few runs, run-optimized) and the same operand (an array, runs or a bitset, over the receiver's
// chunks and beyond), the same operation is applied through every entry point, and the value sets must be equal.
// The shape each result takes is pinned in andnot_keeps_runs.cpp.

#include <frsr/roaring/bitmap.hpp>

#include <roaring/roaring.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace {

using namespace frsr::roaring;

using eager_bitmap       = bitmap<std::uint32_t>;
using atomic_lazy_bitmap = bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount, detail::run_selection_lazy>;
using values_t           = std::vector<std::uint32_t>;

struct croaring {
    explicit croaring( values_t const & values, bool const run_optimize ) : r{ roaring_bitmap_of_ptr( values.size(), values.data() ) } {
        if ( run_optimize ) {
            roaring_bitmap_run_optimize( r );
        }
    }
    explicit croaring( roaring_bitmap_t * const raw ) noexcept : r{ raw } {}
    croaring( croaring const & ) = delete;
    ~croaring() { roaring_bitmap_free( r ); }

    [[nodiscard]] values_t values() const {
        values_t out( roaring_bitmap_get_cardinality( r ) );
        roaring_bitmap_to_uint32_array( r, out.data() );
        return out;
    }

    roaring_bitmap_t * r;
};

template <typename Bitmap>
[[nodiscard]] ::testing::AssertionResult same_contents( Bitmap const & ours, croaring const & theirs ) {
    if ( auto const * const bad{ ours.validate() } ) {
        return ::testing::AssertionFailure() << "invariant violated: " << bad;
    }
    auto const mine{ ours.to_vector() };
    auto const other{ theirs.values() };
    if ( !std::ranges::equal( mine, other ) ) {
        return ::testing::AssertionFailure() << "contents differ: ours " << mine.size() << " values, CRoaring " << other.size();
    }
    return ::testing::AssertionSuccess();
}

// Runs of random lengths and gaps, two or three chunks wide, the first chunk starting at `first_chunk`.
[[nodiscard]] values_t random_runs( std::mt19937_64 & rng, std::uint32_t const first_chunk, unsigned const chunks, std::uint32_t const max_length, std::uint32_t const max_gap ) {
    values_t result;
    auto const end{ ( first_chunk + chunks ) << 16 };
    for ( auto position{ first_chunk << 16 }; position < end; ) {
        position += static_cast<std::uint32_t>( rng() % max_gap );
        auto const length{ 1U + static_cast<std::uint32_t>( rng() % max_length ) };
        for ( std::uint32_t i{ 0 }; i < length && position < end; ++i, ++position ) {
            result.push_back( position );
        }
        position += 1U;
    }
    return result;
}

[[nodiscard]] values_t random_values( std::mt19937_64 & rng, std::uint32_t const first_chunk, unsigned const chunks, std::size_t const count ) {
    values_t result;
    for ( std::size_t i{ 0 }; i < count; ++i ) {
        result.push_back( ( first_chunk << 16 ) + static_cast<std::uint32_t>( rng() % ( std::uint64_t{ chunks } << 16 ) ) );
    }
    std::ranges::sort( result );
    result.erase( std::unique( result.begin(), result.end() ), result.end() );
    return result;
}

// The receiver but for a handful of survivors, with every second value of the next chunk added so that the
// operand is a bitset (thousands of runs) rather than runs.
[[nodiscard]] values_t almost_everything( std::mt19937_64 & rng, values_t result, std::uint32_t const tail_chunk, unsigned const survivors ) {
    for ( unsigned i{ 0 }; i < survivors && !result.empty(); ++i ) {
        result.erase( result.begin() + static_cast<std::ptrdiff_t>( rng() % result.size() ) );
    }
    for ( std::uint32_t low{ 0 }; low < 65'536U; low += 2U ) {
        result.push_back( ( tail_chunk << 16 ) + low );
    }
    return result;
}

template <typename Bitmap>
void subtraction_trials( std::uint64_t const seed ) {
    std::mt19937_64 rng{ seed };
    for ( int trial{ 0 }; trial < 600; ++trial ) {
        auto const chunks{ 1U + static_cast<unsigned>( rng() % 3 ) };
        auto const receiver_values{ random_runs( rng, 2, chunks, 1U + static_cast<std::uint32_t>( rng() % 30'000 ), 1U + static_cast<std::uint32_t>( rng() % 3'000 ) ) };
        values_t operand_values;
        switch ( rng() % 5 ) {
            case 0: operand_values = random_values( rng, 2, chunks + 1, rng() % 40 ); break;
            case 1: operand_values = random_values( rng, 2, chunks + 1, 300 + rng() % 6'000 ); break;
            case 2: operand_values = random_runs( rng, 2, chunks + 1, 1U + static_cast<std::uint32_t>( rng() % 5'000 ), 1U + static_cast<std::uint32_t>( rng() % 5'000 ) ); break;
            case 3: operand_values = almost_everything( rng, receiver_values, 2 + chunks, static_cast<unsigned>( rng() % 20 ) ); break;
            default: operand_values = random_values( rng, 2, chunks + 1, 20'000 + rng() % 40'000 ); break;
        }
        bool const operand_runs{ rng() % 3 == 0 };
        SCOPED_TRACE( ::testing::Message() << "trial " << trial << " receiver " << receiver_values.size() << " operand " << operand_values.size() << " operand_runs " << operand_runs );

        Bitmap receiver;
        receiver.add_many_sorted( { receiver_values.data(), receiver_values.size() } );
        receiver.optimize();
        Bitmap operand;
        operand.add_many_sorted( { operand_values.data(), operand_values.size() } );
        if ( operand_runs ) {
            operand.optimize();
        }
        croaring const theirs_receiver{ receiver_values, true };
        croaring const theirs_operand{ operand_values, operand_runs };

        croaring const expected{ roaring_bitmap_andnot( theirs_receiver.r, theirs_operand.r ) };
        EXPECT_TRUE( same_contents( receiver - operand, expected ) ) << "binary";
        {
            auto copy{ receiver };
            copy -= operand;
            EXPECT_TRUE( same_contents( copy, expected ) ) << "-= on a shared copy";
            EXPECT_TRUE( same_contents( receiver, theirs_receiver ) ) << "the other holder";
        }
        {
            auto copy{ receiver };
            copy.andnot_with( std::span<std::uint32_t const>{ operand_values } );
            EXPECT_TRUE( same_contents( copy, expected ) ) << "andnot_with( span )";
        }
        {
            auto copy{ receiver };
            copy.remove_many_sorted( { operand_values.data(), operand_values.size() } );
            EXPECT_TRUE( same_contents( copy, expected ) ) << "remove_many_sorted";
        }
        {
            auto const first{ static_cast<std::uint32_t>( ( 2U << 16 ) + rng() % ( chunks << 16 ) ) };
            auto const last{ first + static_cast<std::uint32_t>( rng() % ( 1U << ( 4 + rng() % 13 ) ) ) };
            auto copy{ receiver };
            copy.remove_closed_range( first, last );
            auto reference{ roaring_bitmap_copy( theirs_receiver.r ) };
            roaring_bitmap_remove_range_closed( reference, first, last );
            croaring const reference_holder{ reference };
            EXPECT_TRUE( same_contents( copy, reference_holder ) ) << "remove_closed_range [" << first << ", " << last << "]";
        }
        // An intersection over whatever the subtraction left.
        {
            auto difference{ receiver - operand };
            auto const later{ random_runs( rng, 2, chunks, 20'000, 2'000 ) };
            Bitmap later_bitmap;
            later_bitmap.add_many_sorted( { later.data(), later.size() } );
            later_bitmap.optimize();
            croaring const theirs_later{ later, true };
            croaring const expected_and{ roaring_bitmap_and( expected.r, theirs_later.r ) };
            EXPECT_TRUE( same_contents( difference & later_bitmap, expected_and ) ) << "and after";
            difference &= later_bitmap;
            EXPECT_TRUE( same_contents( difference, expected_and ) ) << "&= after";
        }
    }
}

TEST( AndnotRunsCrosscheck, eager ) { subtraction_trials<eager_bitmap>( 0xA11D07 ); }
TEST( AndnotRunsCrosscheck, atomic_lazy ) { subtraction_trials<atomic_lazy_bitmap>( 0xA11D08 ); }

}  // namespace
