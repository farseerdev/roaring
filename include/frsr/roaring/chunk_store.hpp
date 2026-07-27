#pragma once

#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/containers.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>

namespace frsr::roaring::detail {

// SoA chunk storage: the sorted chunk keys live in their own dense array so the
// merge/lookup key walks touch only keys[] (2-8 B stride) and dereference the
// parallel 32 B container handle in slots[] on a key match only. Replaces the
// AoS heap_vector<chunk_entry>. Parallel-array invariant: keys.size() ==
// slots.size() at every public-API boundary.
// [croaring-ref] deps/croaring/include/roaring/containers/containers.h: roaring_array_t{keys;containers;typecodes}
template <typename Layout, typename CowPolicy = cow_value_semantics>
class chunk_store {
public:
    using chunk_type  = typename Layout::chunk_type;
    using handle_type = container_handle<Layout, CowPolicy>;

    // heap_vector subscripts take a 32-bit size_type; the public API stays in
    // std::size_t and narrows here (chunk counts are bounded far below 2^32).
    using index_type = typename heap_vector<chunk_type>::size_type;
    [[nodiscard]] static constexpr index_type idx( std::size_t const index ) noexcept { return static_cast<index_type>( index ); }

    [[nodiscard]] std::size_t size () const noexcept { return keys_.size (); }
    [[nodiscard]] bool        empty() const noexcept { return keys_.empty(); }

    [[nodiscard]] chunk_type key( std::size_t const index ) const noexcept { return keys_[ idx( index ) ]; }
    void set_key( std::size_t const index, chunk_type const key ) noexcept { keys_[ idx( index ) ] = key; }

    [[nodiscard]] handle_type       & slot( std::size_t const index )       noexcept { return slots_[ idx( index ) ]; }
    [[nodiscard]] handle_type const & slot( std::size_t const index ) const noexcept { return slots_[ idx( index ) ]; }

    [[nodiscard]] std::span<chunk_type  const> keys () const noexcept { return { keys_ .data(), keys_ .size() }; }
    [[nodiscard]] std::span<handle_type      > slots()       noexcept { return { slots_.data(), slots_.size() }; }
    [[nodiscard]] std::span<handle_type const> slots() const noexcept { return { slots_.data(), slots_.size() }; }

    // Storage footprint of the parallel arrays themselves (not the spilled payloads).
    [[nodiscard]] std::size_t entry_bytes() const noexcept {
        return size() * ( sizeof( chunk_type ) + sizeof( handle_type ) );
    }

    void reserve( std::size_t const capacity ) {
        keys_ .reserve( static_cast<std::uint32_t>( capacity ) );
        slots_.reserve( static_cast<std::uint32_t>( capacity ) );
    }

    // Retired slots (scratch payload reuse) are deliberately per-instance
    // transient state: copies start with none.
    chunk_store() = default;
    chunk_store( chunk_store const & other ) : keys_{ other.keys_ }, slots_{ other.slots_ } {}
    chunk_store( chunk_store && ) = default;
    chunk_store & operator=( chunk_store const & other ) {
        keys_  = other.keys_ ;
        slots_ = other.slots_;
        return *this;
    }
    chunk_store & operator=( chunk_store && ) = default;

    void clear() noexcept {
        keys_ .clear();
        slots_.clear();
        retired_slots_.clear();
        retired_array_next_ = retired_bitset_next_ = 0;
    }

    // Scratch payload reuse (CRoaring persistent-dst analog, see
    // container_handle::offers_reusable_payload): logically empties the store
    // but parks the current slots — payloads intact — for take_retired() to
    // hand back to the next run's materializing sites. The swap also recycles
    // the previous generation's retired vector storage back into slots_, so
    // the parallel arrays themselves keep their capacity across runs too.
    void clear_retiring_slots() noexcept {
        keys_.clear();
        retired_slots_.clear();
        retired_slots_.swap( slots_ );
        retired_array_next_ = retired_bitset_next_ = 0;
    }

    // Hands back a retired sole-owned spilled payload of `kind` (header reset
    // for an in-place rebuild), or an empty handle when none is left. A claimed
    // slot becomes an empty inline array (not spilled), so the other kind's
    // cursor skips it via the offers_reusable_payload() spilled check.
    // The payload capacity take_retired( kind ) would hand back, without claiming it —
    // zero when nothing is available. Lets a kernel with two implementations (one that
    // needs a result buffer, one that writes in place) pick the one that will not have
    // to grow an allocation.
    template <typename E = typename Layout::low_type>
    [[nodiscard]] std::uint32_t retired_capacity( container_kind const kind ) const noexcept {
        auto cursor{ ( kind == container_kind::bitset ) ? retired_bitset_next_ : retired_array_next_ };
        while ( cursor < retired_slots_.size() ) {
            auto const & candidate{ retired_slots_[ cursor++ ] };
            if ( candidate.offers_reusable_payload( kind ) ) {
                return candidate.template payload_capacity<E>();
            }
        }
        return 0;
    }

    [[nodiscard]] handle_type take_retired( container_kind const kind ) noexcept {
        auto & cursor{ kind == container_kind::bitset ? retired_bitset_next_ : retired_array_next_ };
        while ( cursor < retired_slots_.size() ) {
            auto & candidate{ retired_slots_[ cursor++ ] };
            if ( candidate.offers_reusable_payload( kind ) ) {
                auto handle{ std::move( candidate ) };
                if ( kind == container_kind::bitset ) { handle.reset_for_bitset_reuse(); }
                else                                  { handle.reset_for_array_reuse (); }
                return handle;
            }
        }
        return {};
    }

    // Parks a consumed handle's payload for take_retired() instead of freeing
    // it — the in-place combine walk pairs each such free with a fresh result
    // allocation one pair later, so this turns that free+alloc churn into
    // in-place reuse. Non-spilled handles have nothing to offer and are dropped.
    void retire( handle_type && handle ) {
        if ( handle.spilled() ) {
            retired_slots_.push_back( std::move( handle ) );
        }
    }

    // move_entry / truncate twins for the in-place combine walk: the payloads
    // they would free are retired instead.
    void move_entry_retiring( std::size_t const to, std::size_t const from ) {
        if ( to != from ) {
            keys_[ idx( to ) ] = keys_[ idx( from ) ];
            retire( std::move( slots_[ idx( to ) ] ) );
            slots_[ idx( to ) ] = std::move( slots_[ idx( from ) ] );
        }
    }

    void truncate_retiring( std::size_t const count ) {
        for ( auto i{ count }; i < size(); ++i ) {
            retire( std::move( slots_[ idx( i ) ] ) );
        }
        truncate( count );
    }

    void swap( chunk_store & other ) noexcept {
        keys_ .swap( other.keys_  );
        slots_.swap( other.slots_ );
        retired_slots_.swap( other.retired_slots_ );
        std::swap( retired_array_next_ , other.retired_array_next_  );
        std::swap( retired_bitset_next_, other.retired_bitset_next_ );
    }

    template <typename Handle>
    void push_back( chunk_type const key, Handle && handle ) {
        keys_ .push_back( key );
        slots_.push_back( std::forward<Handle>( handle ) );
    }

    template <typename Handle>
    void insert( std::size_t const index, chunk_type const key, Handle && handle ) {
        keys_ .insert( keys_ .begin() + static_cast<std::ptrdiff_t>( index ), key );
        slots_.insert( slots_.begin() + static_cast<std::ptrdiff_t>( index ), std::forward<Handle>( handle ) );
    }

    void erase( std::size_t const index ) {
        keys_ .erase( keys_ .begin() + static_cast<std::ptrdiff_t>( index ) );
        slots_.erase( slots_.begin() + static_cast<std::ptrdiff_t>( index ) );
    }

    // In-place compaction support for the shrinking set-ops (&=, -=): the entry
    // at `from` moves down to `to` (to <= from always holds on those walks).
    void move_entry( std::size_t const to, std::size_t const from ) {
        if ( to != from ) {
            keys_ [ idx( to ) ] = keys_[ idx( from ) ];
            slots_[ idx( to ) ] = std::move( slots_[ idx( from ) ] );
        }
    }

    void truncate( std::size_t const count ) {
        keys_ .erase( keys_ .begin() + static_cast<std::ptrdiff_t>( count ), keys_ .end() );
        slots_.erase( slots_.begin() + static_cast<std::ptrdiff_t>( count ), slots_.end() );
    }

    void erase_front( std::size_t const count ) {
        keys_ .erase( keys_ .begin(), keys_ .begin() + static_cast<std::ptrdiff_t>( count ) );
        slots_.erase( slots_.begin(), slots_.begin() + static_cast<std::ptrdiff_t>( count ) );
    }

    // Stable-compacts away every entry whose slot satisfies the predicate
    // (the SoA analog of remove_if over the entry vector).
    template <typename Pred>
    void erase_slots_if( Pred && pred ) {
        std::size_t out{ 0 };
        for ( std::size_t in{ 0 }; in < size(); ++in ) {
            if ( !pred( std::as_const( slots_[ idx( in ) ] ) ) ) {
                if ( out != in ) {
                    keys_ [ idx( out ) ] = keys_[ idx( in ) ];
                    slots_[ idx( out ) ] = std::move( slots_[ idx( in ) ] );
                }
                ++out;
            }
        }
        keys_ .erase( keys_ .begin() + static_cast<std::ptrdiff_t>( out ), keys_ .end() );
        slots_.erase( slots_.begin() + static_cast<std::ptrdiff_t>( out ), slots_.end() );
    }

    // Permutation sort over both arrays (cold: only the lazy-sort configuration
    // ever leaves the store unsorted).
    void sort_by_key() {
        auto const n{ size() };
        heap_vector<std::uint32_t> permutation;
        permutation.resize( static_cast<std::uint32_t>( n ) );
        std::iota( permutation.begin(), permutation.end(), 0U );
        std::ranges::sort( permutation, [&]( std::uint32_t const lhs, std::uint32_t const rhs ) {
            return keys_[ lhs ] < keys_[ rhs ];
        } );

        heap_vector<chunk_type>  sorted_keys;
        heap_vector<handle_type> sorted_slots;
        sorted_keys .reserve( static_cast<std::uint32_t>( n ) );
        sorted_slots.reserve( static_cast<std::uint32_t>( n ) );
        for ( auto const index : permutation ) {
            sorted_keys .push_back( keys_[ index ] );
            sorted_slots.push_back( std::move( slots_[ index ] ) );
        }
        keys_  = std::move( sorted_keys  );
        slots_ = std::move( sorted_slots );
    }

private:
    heap_vector<chunk_type>  keys_;
    heap_vector<handle_type> slots_;
    // Scratch payload reuse (clear_retiring_slots / retire / take_retired):
    // consumed slots offered for in-place rebuild; per-kind claim cursors.
    heap_vector<handle_type> retired_slots_;
    std::uint32_t            retired_array_next_ { 0 };
    std::uint32_t            retired_bitset_next_{ 0 };
};

} // namespace frsr::roaring::detail
