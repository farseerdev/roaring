// Which operations of a bitmap are noexcept: the ones that cannot allocate always, the ones that may allocate
// exactly when a failed allocation cannot be observed as an exception (detail::allocation_failure_terminates).
// Compile-time only; included by the test binary (the default policy) and by a translation unit compiled with a
// non-failing allocation policy.
#pragma once

#include <frsr/roaring/bitmap.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace frsr::roaring::noexcept_checks {

inline constexpr bool terminates{ detail::allocation_failure_terminates };

template <typename Bitmap>
struct checks {
    using key_type = typename Bitmap::key_type;

    // operations that never allocate
    static_assert( noexcept( std::declval<Bitmap &>().clear() ) );
    static_assert( noexcept( std::declval<Bitmap &>().clear_keep_capacity() ) );
    static_assert( noexcept( std::declval<Bitmap &>().swap( std::declval<Bitmap &>() ) ) );
    static_assert( noexcept( std::declval<Bitmap &>().repair_cardinality() ) );
    static_assert( std::is_nothrow_move_constructible_v<Bitmap> );
    static_assert( std::is_nothrow_move_assignable_v<Bitmap> );
    static_assert( std::is_nothrow_destructible_v<Bitmap> );
    static_assert( noexcept( std::declval<Bitmap const &>().contains( std::declval<key_type>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().size() ) );
    static_assert( noexcept( std::declval<Bitmap const &>().intersects( std::declval<Bitmap const &>() ) ) );

    // operations that may allocate: noexcept exactly when allocation failure terminates
    static_assert( std::is_nothrow_copy_constructible_v<Bitmap> == terminates );
    static_assert( std::is_nothrow_copy_assignable_v<Bitmap> == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().add( std::declval<key_type>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().remove( std::declval<key_type>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().try_add( std::declval<key_type>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().try_remove( std::declval<key_type>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().add_bulk( std::declval<typename Bitmap::bulk_context &>(), std::declval<key_type>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().add_many_sorted( std::declval<std::span<key_type const>>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().remove_many_sorted( std::declval<std::span<key_type const>>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().add_many( std::declval<std::vector<key_type> const &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().remove_many( std::declval<std::span<key_type const>>() ) ) == terminates );
    static_assert( noexcept( Bitmap{ std::declval<std::vector<key_type> const &>() } ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().add_closed_range( key_type{}, key_type{} ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().remove_closed_range( key_type{}, key_type{} ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().flip_closed_range( key_type{}, key_type{} ) ) == terminates );

    static_assert( noexcept( std::declval<Bitmap &>() |= std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() &= std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() -= std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() ^= std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>() | std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>() & std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>() - std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>() ^ std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().or_with( std::declval<Bitmap const &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().xor_with( std::declval<Bitmap const &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().xor_into( std::declval<Bitmap const &>(), std::declval<Bitmap &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().union_into( std::declval<Bitmap const &>(), std::declval<Bitmap &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().intersect_into( std::declval<Bitmap const &>(), std::declval<Bitmap &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().difference_into( std::declval<Bitmap const &>(), std::declval<Bitmap &>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().or_many_in_place( std::declval<std::span<Bitmap const * const>>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().xor_many_in_place( std::declval<std::span<Bitmap const * const>>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().and_many_sorted_in_place( std::declval<std::span<Bitmap const *>>() ) ) == terminates );

    // against operands that are not bitmaps: the operand's own handling must not throw either
    static_assert( noexcept( std::declval<Bitmap &>().and_with( std::declval<key_range>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().or_with( std::declval<key_range>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().andnot_with( std::declval<key_range>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().xor_with( std::declval<key_range>() ) ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() ^= std::declval<bit_words<std::uint64_t>>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().and_range( key_type{}, key_type{} ) ) == terminates );
    static_assert( !noexcept( std::declval<Bitmap &>().xor_with( std::declval<std::vector<key_type> const &>() ) ), "a key sequence is walked by code that can throw" );

    // against a view: the reads allocate nothing, the updates allocate exactly as the operand they reduce to
    using view = typed_view<Bitmap>;
    static_assert( noexcept( view_of( std::declval<Bitmap const &>() ) ) );
    static_assert( std::is_nothrow_copy_constructible_v<view> );
    static_assert( noexcept( std::declval<Bitmap const &>().intersects( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().and_cardinality( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().xor_cardinality( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().is_subset_of( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().is_superset_of( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().equals( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap &>() &= std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() |= std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() -= std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() ^= std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<view>() & std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>() | std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().intersect_into( std::declval<view>(), std::declval<Bitmap &>() ) ) == terminates );
    static_assert( noexcept( Bitmap::and_many( std::declval<std::span<view const>>() ) ) == terminates );
    static_assert( noexcept( Bitmap::and_many_into( std::declval<std::span<view const>>(), std::declval<Bitmap &>() ) ) == terminates );

    // against the complement of an operand: as the operation each is rewritten to
    static_assert( noexcept( std::declval<Bitmap &>() &= !std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() -= !std::declval<view>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() |= !std::declval<key_range>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() ^= !std::declval<key_range>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>() &= !std::declval<Bitmap const &>() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap const &>().intersects( !std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().and_cardinality( !std::declval<key_range>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().is_subset_of( !std::declval<view>() ) ) );

    // the sizes of combinations
    static_assert( noexcept( std::declval<Bitmap const &>().andnot_cardinality( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().or_cardinality( std::declval<view>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().jaccard_index( std::declval<key_range>() ) ) );
    static_assert( noexcept( std::declval<Bitmap const &>().is_strict_subset_of( std::declval<view>() ) ) );

    static_assert( noexcept( std::declval<Bitmap &>().optimize() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().optimize_for_storage() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().optimize_for_speed() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().shrink_to_fit() ) == terminates );
    static_assert( noexcept( std::declval<Bitmap &>().bulk_or_finish() ) == terminates );
};

template struct checks<bitmap<std::uint16_t>>;
template struct checks<bitmap<std::uint32_t>>;
template struct checks<bitmap<std::uint64_t>>;
template struct checks<bitmap<std::uint32_t, default_container_set<std::uint32_t>, detail::cow_atomic_refcount>>;
template struct checks<bitmap<std::uint64_t, default_container_set<std::uint64_t>, detail::cow_unsynchronized_refcount, detail::run_selection_lazy>>;

} // namespace frsr::roaring::noexcept_checks
