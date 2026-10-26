#pragma once

// A frozen bitmap image read with constant-time rank (docs/ranked-view.md).
//
// A ranked image (`frozen_options{ .ranked = true }`, docs/frozen-format.md) carries, besides the payloads of the
// plain frozen image, the number of values before each chunk and a u16 rank block after each bitset and run
// payload. ranked_view turns that into
//   count_below( x )  the number of ids < x   (CRoaring's roaring_bitmap_rank is inclusive: it counts ids <= x)
//   index_of( x )     the ordinal of x among the ids if present, else npos
//   select( i )       the i-th id
// with one chunk lookup (O(1) when the chunks are consecutive, a binary search of the chunk index otherwise) and
// one in-container step: a masked popcount for a bitset chunk, a search of the values for an array chunk and a
// search of the runs for a run chunk.
//
// Every member is const, writes nothing and keeps no cache: any number of threads may read one view. The view
// borrows the image's bytes, which must outlive it.

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#if defined( __BMI2__ )
#include <immintrin.h>
#endif

namespace frsr::roaring {

// The bitmap to build a ranked table with: it picks no run containers by itself (see the build recipe below).
template <std::unsigned_integral Key>
using rank_table_bitmap = bitmap<Key, default_container_set<Key>, detail::cow_value_semantics, detail::run_selection_lazy>;

namespace detail {

// The position of the `rank`-th (0-based) set bit of `word`, which has more than `rank` set bits. A `rank` the word
// does not have gives a position of at most 64 and reads nothing (a rank block of an image that was not written by
// the library can say so).
[[nodiscard]] inline unsigned select_in_word( std::uint64_t word, unsigned rank ) noexcept {
#if defined( __BMI2__ )
    // pdep is microcoded on older AMD cores, which is why it is taken only when the build targets BMI2 explicitly
    // (the library does not dispatch on the CPU at run time). The mask is the shift's own (shlx masks as well).
    return static_cast<unsigned>( std::countr_zero( _pdep_u64( std::uint64_t{ 1 } << ( rank & 63U ), word ) ) );
#else
    // Halving: skip the low half when it holds no more than `rank` bits.
    unsigned position{ 0 };
    for ( unsigned half{ 32 }; half != 0; half >>= 1 ) {
        auto const below{ static_cast<unsigned>( std::popcount( word & ( ( std::uint64_t{ 1 } << half ) - 1 ) ) ) };
        if ( rank >= below ) {
            rank     -= below;
            word    >>= half;
            position += half;
        }
    }
    return position;
#endif
}

} // namespace detail

// Build recipe. Rank costs O(1) in a bitset chunk and O(log) in an array or run chunk, and the default bitmap picks a
// run container by itself whenever that serializes smaller, which for a nearly contiguous table (range/entries of 1.001
// to 1.1) means a binary search over dozens of runs where a bitset is one popcount (about 2 ns against 20 to 40 ns in
// bench/rank_bench.cpp). Build the table with a bitmap that does not select runs, and add runs only for chunks
// that are strongly run-shaped:
//
//   rank_table_bitmap<std::uint32_t> table;
//   table.add_sorted_many( ids );            // arrays and bitsets only
//   table.optimize_run_shaped( 4096 );       // optional: runs where the mean run length is at least the bar
//   std::vector<std::uint64_t> storage( ( table.frozen_size_bytes( { .ranked = true } ) + 7 ) / 8 );   // 8-byte aligned
//   std::span const image{ reinterpret_cast<std::byte *>( storage.data() ), table.frozen_size_bytes( { .ranked = true } ) };
//   table.frozen_serialize( image, { .ranked = true } );
//   auto const view{ ranked_view<std::uint32_t>::from( image ) };
//
// The image does not depend on the bitmap type that wrote it.
template <std::unsigned_integral Key, typename Bitmap = bitmap<Key>>
class ranked_view {
    static_assert( std::same_as<Key, std::uint32_t>, "a rank is a 32-bit count and a chunk key 16 bits: the keys are 32-bit" );
    static_assert( std::same_as<typename Bitmap::key_type, Key> );

    using view_type  = typename Bitmap::frozen_view;
    using entry_type = typename Bitmap::frozen_chunk_index;
    using kind_type  = typename Bitmap::frozen_container_kind;
    using low_type   = typename Bitmap::low_type;
    using run_type   = typename Bitmap::run_container_type::run;

public:
    using key_type   = Key;
    using index_type = std::uint32_t;
    // index_of's "absent". Opening a ranked image rejects 2^32 values or more, so no present id has this index.
    static constexpr index_type npos{ std::numeric_limits<index_type>::max() };

    ranked_view() = default;

    // Null if the image is not a valid ranked frozen image. The bytes must be 8-byte aligned and outlive the view.
    [[nodiscard]] static ranked_view from( std::span<std::byte const> const image ) { return from( Bitmap::frozen_view_from( image ) ); }

    [[nodiscard]] static ranked_view from( view_type const & view ) noexcept {
        ranked_view result;
        if ( !view || !view.ranked() ) {
            return result;
        }
        result.view_        = view;
        result.index_       = view.index_;
        result.data_        = view.data_;
        result.chunk_count_ = view.header_->chunk_count;
        result.size_        = static_cast<index_type>( view.size() );   // below 2^32: validated when the image was opened
        if ( result.chunk_count_ != 0 ) {
            result.first_chunk_ = static_cast<std::uint32_t>( result.index_[ 0 ].chunk );
            auto const last_chunk{ static_cast<std::uint32_t>( result.index_[ result.chunk_count_ - 1 ].chunk ) };
            result.contiguous_  = last_chunk - result.first_chunk_ + 1U == result.chunk_count_;
        }
        return result;
    }

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>( view_ ); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    // The bytes of the image.
    [[nodiscard]] std::size_t byte_size() const noexcept { return view_.byte_size(); }
    // Preconditions: !empty().
    [[nodiscard]] key_type front() const noexcept { return select( 0 ); }
    [[nodiscard]] key_type back () const noexcept { return select( size_ - 1 ); }

    [[nodiscard]] bool contains( key_type const x ) const noexcept { return find( x ).present; }

    // The number of ids below x.
    [[nodiscard]] index_type count_below( key_type const x ) const noexcept { return find( x ).below; }

    // The ordinal of x among the ids (count_below( x ) for a present x), else npos: one lookup.
    [[nodiscard]] index_type index_of( key_type const x ) const noexcept {
        auto const position{ find( x ) };
        return position.present ? position.below : npos;
    }

    // The id with ordinal i. Precondition: i < size().
    [[nodiscard]] key_type select( index_type const i ) const noexcept;

    // f( id, ordinal ) for every id in key order. The ordinal is a running count: the rank data is not read.
    template <typename F>
    void for_each_indexed( F && f ) const {
        index_type ordinal{};
        view_.for_each( [ & ]( key_type const id ) { f( id, ordinal++ ); } );
    }

    // The same from the first id >= `from`, with ordinals counted from count_below( from ): a slice of the table.
    // The chunk of `from` is decoded whole and filtered.
    template <typename F>
    void for_each_indexed_from( key_type const from, F && f ) const {
        auto ordinal{ count_below( from ) };
        view_.for_each_from( from, [ & ]( key_type const id ) { f( id, ordinal++ ); } );
    }

    // The plain frozen view of the same image: materialize(), borrow_into(), contains() work on it as on any other.
    [[nodiscard]] view_type frozen() const noexcept { return view_; }

private:
    using position = typename view_type::position;

    [[nodiscard]] [[gnu::always_inline]] position find( key_type const x ) const noexcept {
        auto const chunk{ static_cast<std::uint32_t>( x >> 16U ) };
        entry_type const * entry;
        if ( contiguous_ ) {
            // Two outcomes lie outside: below the first chunk nothing precedes x, above the last everything does.
            if ( chunk < first_chunk_ ) {
                return { 0U, false };
            }
            auto const at{ chunk - first_chunk_ };
            if ( at >= chunk_count_ ) {
                return { size_, false };
            }
            entry = index_ + at;
        } else {
            auto const it{ view_type::lower_bound_entry( index_, chunk_count_, chunk ) };
            if ( it == index_ + chunk_count_ ) {
                return { size_, false };
            }
            if ( static_cast<std::uint32_t>( it->chunk ) != chunk ) {
                return { it->base, false };   // the values of all chunks before the next one
            }
            entry = it;
        }

        auto const low{ static_cast<low_type>( x & 0xFFFFU ) };
        auto const * const payload{ data_ + entry->payload_offset };
        switch ( static_cast<kind_type>( entry->container_kind ) ) {
            case kind_type::bitset: {
                auto const * const words{ reinterpret_cast<std::uint64_t const *>( payload ) };
                auto const * const ranks{ reinterpret_cast<std::uint16_t const *>( payload + entry->payload_bytes ) };
                auto const word_at{ low >> 6U };
                auto const bit{ low & 63U };
                auto const word{ words[ word_at ] };
                return {
                    entry->base + ranks[ word_at ] + static_cast<std::uint32_t>( std::popcount( word & ( ( std::uint64_t{ 1 } << bit ) - 1U ) ) ),
                    ( ( word >> bit ) & 1U ) != 0
                };
            }
            case kind_type::array:
                return array_position( *entry, payload, low );
            case kind_type::run:
                return run_position( *entry, payload, low );
        }
        std::unreachable();
    }

    [[gnu::noinline]] static position array_position( entry_type const & entry, std::byte const * const payload, low_type const low ) noexcept {
        auto const * const values{ reinterpret_cast<low_type const *>( payload ) };
        auto const count{ static_cast<std::size_t>( entry.payload_count ) };
        auto const below{ static_cast<std::size_t>( view_type::array_lower_bound( values, count, low ) ) };
        return { entry.base + static_cast<std::uint32_t>( below ), below < count && values[ below ] == low };
    }

    [[gnu::noinline]] static position run_position( entry_type const & entry, std::byte const * const payload, low_type const low ) noexcept {
        auto const in_run{ view_type::run_position(
            reinterpret_cast<run_type const *>( payload ),
            reinterpret_cast<std::uint16_t const *>( payload + entry.payload_bytes ),
            static_cast<std::int32_t>( entry.payload_count ),
            low
        ) };
        return { entry.base + in_run.below, in_run.present };
    }

    view_type          view_{};
    entry_type const * index_{};
    std::byte const *  data_{};
    std::uint32_t      chunk_count_{};
    std::uint32_t      first_chunk_{};
    index_type         size_{};
    bool               contiguous_{};
};

template <std::unsigned_integral Key, typename Bitmap>
typename ranked_view<Key, Bitmap>::key_type ranked_view<Key, Bitmap>::select( index_type const i ) const noexcept {
    assert( i < size_ );
    // The last chunk whose first ordinal is <= i. The bases were checked when the image was opened to be the running
    // sum of the cardinalities, which end in size(): `within` is below the cardinality of the chunk.
    auto const span{ std::span<entry_type const>{ index_, chunk_count_ } };
    auto const it{ std::ranges::upper_bound( span, i, {}, &entry_type::base ) - 1 };
    auto const within{ i - it->base };
    auto const * const payload{ data_ + it->payload_offset };
    auto const compose{ [ & ]( std::uint32_t const low ) { return static_cast<key_type>( ( static_cast<std::uint32_t>( it->chunk ) << 16U ) | static_cast<low_type>( low ) ); } };
    // A rank block is searched from its second count on (the first is 0 by construction, so the result is an index
    // into the block whatever the counts are): a block that does not add up gives a wrong id, never a read outside
    // the payload or its rank block. Opening the image checked the sizes of both against the cardinality.
    switch ( static_cast<kind_type>( it->container_kind ) ) {
        case kind_type::array:
            assert( within < it->payload_count );
            return compose( reinterpret_cast<low_type const *>( payload )[ within ] );
        case kind_type::run: {
            auto const * const runs{ reinterpret_cast<run_type const *>( payload ) };
            auto const * const ranks{ reinterpret_cast<std::uint16_t const *>( payload + it->payload_bytes ) };
            auto const later{ std::span{ ranks + 1, it->payload_count - 1U } };
            auto const run_at{ static_cast<std::size_t>( std::ranges::upper_bound( later, within ) - later.begin() ) };
            return compose( runs[ run_at ].begin + ( within - ranks[ run_at ] ) );
        }
        case kind_type::bitset: {
            constexpr auto word_count{ Bitmap::layout_type::word_count };
            auto const * const words{ reinterpret_cast<std::uint64_t const *>( payload ) };
            auto const * const ranks{ reinterpret_cast<std::uint16_t const *>( payload + it->payload_bytes ) };
            auto const later{ std::span{ ranks + 1, word_count - 1U } };
            auto const word_at{ static_cast<std::size_t>( std::ranges::upper_bound( later, within ) - later.begin() ) };
            return compose( static_cast<std::uint32_t>( word_at * 64U ) + detail::select_in_word( words[ word_at ], within - ranks[ word_at ] ) );
        }
    }
    std::unreachable();
}

} // namespace frsr::roaring
