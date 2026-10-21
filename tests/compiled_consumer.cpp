// A program that includes only <frsr/roaring/bitmap.hpp>, never the member definitions: every call below must
// resolve to the library's compiled instantiations (or, in header-only mode, to the same code instantiated here).
// The results are checked against a std::set oracle.

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <vector>

#ifdef FRSR_ROARING_WITH_DEFINITIONS
#   error "this program must see only the declarations"
#endif

namespace {

using frsr::roaring::key_range;
using frsr::roaring::sorted_keys;

template <typename Bitmap>
std::vector<typename Bitmap::key_type> keys_of( Bitmap const & bitmap ) {
    std::vector<typename Bitmap::key_type> keys;
    for ( auto const key : bitmap ) { keys.push_back( key ); }
    return keys;
}

template <typename Key>
std::vector<Key> random_keys( std::mt19937_64 & rng, std::size_t const count, Key const bound ) {
    std::vector<Key> keys( count );
    for ( auto & key : keys ) { key = static_cast<Key>( rng() % bound ); }
    return keys;
}

template <typename Bitmap>
void exercise() {
    using key_type = typename Bitmap::key_type;
    constexpr key_type bound{ static_cast<key_type>( std::min<std::uint64_t>( std::numeric_limits<key_type>::max(), 1'000'000 ) ) };
    std::mt19937_64 rng{ 5 };

    auto const lhs_keys{ random_keys<key_type>( rng, 20'000, bound ) };
    auto const rhs_keys{ random_keys<key_type>( rng, 30'000, bound ) };
    std::set<key_type> const lhs_set{ lhs_keys.begin(), lhs_keys.end() };
    std::set<key_type> const rhs_set{ rhs_keys.begin(), rhs_keys.end() };

    Bitmap const lhs{ lhs_keys };
    Bitmap const rhs{ rhs_keys };
    auto const as_vector{ []( auto const & set ) { return std::vector<key_type>( set.begin(), set.end() ); } };
    EXPECT_EQ( keys_of( lhs ), as_vector( lhs_set ) );
    EXPECT_EQ( lhs.size(), lhs_set.size() );

    auto const check{ [ & ]( Bitmap const & result, auto const & oracle ) { EXPECT_EQ( keys_of( result ), as_vector( oracle ) ); } };
    std::set<key_type> oracle;
    std::ranges::set_union( lhs_set, rhs_set, std::inserter( oracle, oracle.end() ) );
    check( lhs | rhs, oracle );
    oracle.clear(); std::ranges::set_intersection( lhs_set, rhs_set, std::inserter( oracle, oracle.end() ) );
    check( lhs & rhs, oracle );
    EXPECT_EQ( lhs.and_cardinality( rhs ), oracle.size() );
    EXPECT_EQ( lhs.intersects( rhs ), !oracle.empty() );
    oracle.clear(); std::ranges::set_difference( lhs_set, rhs_set, std::inserter( oracle, oracle.end() ) );
    check( lhs - rhs, oracle );
    oracle.clear(); std::ranges::set_symmetric_difference( lhs_set, rhs_set, std::inserter( oracle, oracle.end() ) );
    check( lhs ^ rhs, oracle );
    EXPECT_EQ( lhs.xor_cardinality( rhs ), oracle.size() );

    Bitmap mutated{ lhs };
    mutated |= rhs; mutated &= lhs; mutated -= rhs; mutated ^= rhs;
    mutated.add_many( rhs_keys ); mutated.remove_many( lhs_keys );
    oracle = rhs_set; for ( auto const key : lhs_keys ) { oracle.erase( key ); }
    check( mutated, oracle );
    mutated.optimize();
    check( mutated, oracle );

    // operands that are not bitmaps
    Bitmap interval{ lhs };
    interval.and_with( key_range::closed( 1000, 5000 ) );
    oracle.clear(); for ( auto const key : lhs_set ) { if ( key >= 1000 && key <= 5000 ) { oracle.insert( key ); } }
    check( interval, oracle );
    EXPECT_EQ( lhs.cardinality_in_range( 1000, 5001 ), oracle.size() );
    EXPECT_TRUE( interval.is_subset_of( lhs ) );
    EXPECT_TRUE( lhs.is_superset_of( interval ) );

    Bitmap with_keys{ lhs };
    with_keys.or_with( std::span<key_type const>{ rhs_keys } );
    oracle = lhs_set; oracle.insert( rhs_keys.begin(), rhs_keys.end() );
    check( with_keys, oracle );
    EXPECT_EQ( lhs.and_cardinality( rhs_keys ), (lhs & rhs).size() );
    EXPECT_TRUE( with_keys.is_superset_of( rhs_keys ) );

    auto sorted{ rhs_keys }; std::ranges::sort( sorted );
    Bitmap without_sorted{ lhs };
    without_sorted.andnot_with( sorted_keys{ sorted } );
    oracle.clear(); std::ranges::set_difference( lhs_set, rhs_set, std::inserter( oracle, oracle.end() ) );
    check( without_sorted, oracle );

    std::array<std::uint64_t, 4> words{ 0xF0F0F0F0F0F0F0F0ULL, 0x1, ~0ULL, 0x8000000000000000ULL };
    Bitmap with_words;
    with_words.or_with( frsr::roaring::bit_words( std::span<std::uint64_t const>{ words }, 10 ) );
    std::vector<key_type> expected_words;
    for ( std::size_t bit{ 0 }; bit < 256; ++bit ) { if ( ( words[ bit / 64 ] >> ( bit % 64 ) ) & 1U ) { expected_words.push_back( static_cast<key_type>( 10 + bit ) ); } }
    EXPECT_EQ( keys_of( with_words ), expected_words );
    EXPECT_TRUE( with_words.equals( expected_words ) );

    // serialization
    if constexpr ( std::numeric_limits<key_type>::digits >= 32 ) {
        std::vector<std::byte> buffer( lhs.portable_size_in_bytes() );
        EXPECT_EQ( lhs.portable_serialize( buffer ), buffer.size() );
        auto const back{ Bitmap::portable_deserialize( buffer ) };
        ASSERT_TRUE( back.has_value() );
        EXPECT_EQ( keys_of( *back ), keys_of( lhs ) );
    }

    std::array<Bitmap const *, 3> const many{ &lhs, &rhs, &mutated };
    EXPECT_EQ( Bitmap::and_many_cardinality( many ), ( lhs & rhs & mutated ).size() );
    EXPECT_EQ( Bitmap::xor_many( many ).size(), ( lhs ^ rhs ^ mutated ).size() );
}

TEST( CompiledConsumer, DefaultBitmap32 ) { exercise<frsr::roaring::bitmap<std::uint32_t>>(); }
TEST( CompiledConsumer, DefaultBitmap64 ) { exercise<frsr::roaring::bitmap<std::uint64_t>>(); }
TEST( CompiledConsumer, DefaultBitmap16 ) { exercise<frsr::roaring::bitmap<std::uint16_t>>(); }
TEST( CompiledConsumer, AtomicLazyBitmap32 ) {
    exercise<frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>,
                                   frsr::roaring::detail::cow_atomic_refcount, frsr::roaring::detail::run_selection_lazy>>();
}
TEST( CompiledConsumer, UnsynchronizedBitmap32 ) {
    exercise<frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>,
                                   frsr::roaring::detail::cow_unsynchronized_refcount>>();
}

} // namespace
