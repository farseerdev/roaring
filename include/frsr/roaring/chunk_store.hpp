#pragma once

#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/containers.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <numeric>
#include <span>
#include <utility>

namespace frsr::roaring::detail {

// SoA chunk storage: the sorted chunk keys live in their own dense array so the
// merge/lookup key walks touch only keys[] (2-8 B stride) and dereference the
// parallel 32 B container handle in slots[] on a key match only. Parallel-array
// invariant: the two arrays share one size at every public-API boundary.
//
// Both arrays live in ONE allocation, exactly as the reference lays out its
// roaring_array_t: the handles first, the keys behind them at the capacity
// boundary. Producing a result bitmap therefore costs one allocation and one
// free for its chunk table, not one per array, and growth relocates both arrays
// with a single trip through the allocator. (Two separate growth-managed
// vectors measured as the largest fixed per-result cost the library pays.)
// [croaring-ref] deps/croaring/include/roaring/containers/containers.h: roaring_array_t{keys;containers;typecodes}
// [croaring-ref] deps/croaring/src/roaring_array.c: ra_init_with_capacity / realloc_array (one bigalloc)
template <typename Layout, typename CowPolicy = cow_value_semantics>
class chunk_store {
public:
    using chunk_type  = typename Layout::chunk_type;
    using handle_type = container_handle<Layout, CowPolicy>;

    // The public API stays in std::size_t and narrows here (chunk counts are
    // bounded far below 2^32).
    using index_type = std::uint32_t;
    [[nodiscard]] static constexpr index_type idx( std::size_t const index ) noexcept { return static_cast<index_type>( index ); }

    [[nodiscard]] std::size_t size () const noexcept { return live_.size_;      }
    [[nodiscard]] bool        empty() const noexcept { return live_.size_ == 0; }

    [[nodiscard]] chunk_type key( std::size_t const index ) const noexcept { return live_.keys()[ idx( index ) ]; }
    void set_key( std::size_t const index, chunk_type const key ) noexcept { live_.keys()[ idx( index ) ] = key; }

    [[nodiscard]] handle_type       & slot( std::size_t const index )       noexcept { return live_.slots()[ idx( index ) ]; }
    [[nodiscard]] handle_type const & slot( std::size_t const index ) const noexcept { return live_.slots()[ idx( index ) ]; }

    [[nodiscard]] std::span<chunk_type  const> keys () const noexcept { return { live_.keys (), live_.size_ }; }
    [[nodiscard]] std::span<handle_type      > slots()       noexcept { return { live_.slots(), live_.size_ }; }
    [[nodiscard]] std::span<handle_type const> slots() const noexcept { return { live_.slots(), live_.size_ }; }

    // Storage footprint of the parallel arrays themselves (not the spilled payloads).
    [[nodiscard]] std::size_t entry_bytes() const noexcept {
        return size() * ( sizeof( chunk_type ) + sizeof( handle_type ) );
    }

    // Compaction (bitmap::shrink_to_fit): releases the parked scratch payloads
    // and the chunk table's spare capacity. Scratch reuse repopulates on the
    // next combine, so this is a footprint/throughput trade the CALLER asked for
    // — never call it from a set operation.
    void shrink_to_fit() {
        retired_.release();
        retired_array_next_ = retired_bitset_next_ = 0;
        live_.shrink_to_fit();
    }

    void reserve( std::size_t const capacity ) { live_.reserve( idx( capacity ) ); }

    // Retired slots (scratch payload reuse) are deliberately per-instance
    // transient state: copies start with none.
    chunk_store() = default;
    chunk_store( chunk_store const & other ) { live_.copy_from( other.live_ ); }
    chunk_store( chunk_store && ) noexcept = default;
    chunk_store & operator=( chunk_store const & other ) {
        if ( this != &other ) [[likely]] {
            live_.copy_from( other.live_ );
        }
        return *this;
    }
    chunk_store & operator=( chunk_store && ) noexcept = default;

    void clear() noexcept {
        live_   .destroy_all();
        retired_.destroy_all();
        retired_array_next_ = retired_bitset_next_ = 0;
    }

    // Scratch payload reuse (CRoaring persistent-dst analog, see
    // container_handle::offers_reusable_payload): logically empties the store
    // but parks the current slots — payloads intact — for take_retired() to
    // hand back to the next run's materializing sites. The swap also recycles
    // the previous generation's retired table back as the live one, so the
    // chunk table itself keeps its capacity across runs too.
    void clear_retiring_slots() noexcept {
        retired_.destroy_all();
        std::swap( live_, retired_ );
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
        auto const * const retired{ retired_.slots() };
        while ( cursor < retired_.size_ ) {
            auto const & candidate{ retired[ cursor++ ] };
            if ( candidate.offers_reusable_payload( kind ) ) {
                return candidate.template payload_capacity<E>();
            }
        }
        return 0;
    }

    [[nodiscard]] handle_type take_retired( container_kind const kind ) noexcept {
        auto & cursor{ kind == container_kind::bitset ? retired_bitset_next_ : retired_array_next_ };
        auto * const retired{ retired_.slots() };
        while ( cursor < retired_.size_ ) {
            auto & candidate{ retired[ cursor++ ] };
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
            retired_.push_back( chunk_type{}, std::move( handle ) );
        }
    }

    // move_entry / truncate twins for the in-place combine walk: the payloads
    // they would free are retired instead.
    void move_entry_retiring( std::size_t const to, std::size_t const from ) {
        if ( to != from ) {
            live_.keys()[ idx( to ) ] = live_.keys()[ idx( from ) ];
            retire( std::move( live_.slots()[ idx( to ) ] ) );
            live_.slots()[ idx( to ) ] = std::move( live_.slots()[ idx( from ) ] );
        }
    }

    void truncate_retiring( std::size_t const count ) {
        for ( auto i{ count }; i < size(); ++i ) {
            retire( std::move( live_.slots()[ idx( i ) ] ) );
        }
        truncate( count );
    }

    void swap( chunk_store & other ) noexcept {
        std::swap( live_   , other.live_    );
        std::swap( retired_, other.retired_ );
        std::swap( retired_array_next_ , other.retired_array_next_  );
        std::swap( retired_bitset_next_, other.retired_bitset_next_ );
    }

    template <typename Handle>
    void push_back( chunk_type const key, Handle && handle ) {
        live_.push_back( key, std::forward<Handle>( handle ) );
    }

    template <typename Handle>
    void insert( std::size_t const index, chunk_type const key, Handle && handle ) {
        live_.insert( idx( index ), key, std::forward<Handle>( handle ) );
    }

    void erase( std::size_t const index ) { live_.erase( idx( index ), idx( index ) + 1 ); }

    // In-place compaction support for the shrinking set-ops (&=, -=): the entry
    // at `from` moves down to `to` (to <= from always holds on those walks).
    void move_entry( std::size_t const to, std::size_t const from ) {
        if ( to != from ) {
            live_.keys ()[ idx( to ) ] = live_.keys()[ idx( from ) ];
            live_.slots()[ idx( to ) ] = std::move( live_.slots()[ idx( from ) ] );
        }
    }

    void truncate( std::size_t const count ) { live_.erase( idx( count ), live_.size_ ); }

    void erase_front( std::size_t const count ) { live_.erase( 0, idx( count ) ); }

    // Stable-compacts away every entry whose slot satisfies the predicate
    // (the SoA analog of remove_if over the entry vector).
    template <typename Pred>
    void erase_slots_if( Pred && pred ) {
        auto * const keys { live_.keys () };
        auto * const slots{ live_.slots() };
        index_type out{ 0 };
        for ( index_type in{ 0 }; in < live_.size_; ++in ) {
            if ( !pred( std::as_const( slots[ in ] ) ) ) {
                if ( out != in ) {
                    keys [ out ] = keys[ in ];
                    slots[ out ] = std::move( slots[ in ] );
                }
                ++out;
            }
        }
        truncate( out );
    }

    // Permutation sort over both arrays (cold: only the lazy-sort configuration
    // ever leaves the store unsorted).
    void sort_by_key() {
        auto const n{ live_.size_ };
        heap_vector<std::uint32_t> permutation;
        permutation.resize( n );
        std::iota( permutation.begin(), permutation.end(), 0U );
        auto const * const keys{ live_.keys() };
        std::ranges::sort( permutation, [=]( std::uint32_t const lhs, std::uint32_t const rhs ) {
            return keys[ lhs ] < keys[ rhs ];
        } );

        table sorted;
        sorted.reserve( n );
        auto * const slots{ live_.slots() };
        for ( auto const index : permutation ) {
            sorted.push_back( keys[ index ], std::move( slots[ index ] ) );
        }
        live_ = std::move( sorted );
    }

private:
    // The single-allocation chunk table: `capacity_` handles, then
    // `capacity_` keys. sizeof( handle_type ) is a multiple of every key
    // width's alignment, so the key array needs no padding. Both halves are
    // held as typed pointers (the key pointer is derived once, on relocation)
    // so a key or slot read is one load; deriving it from the capacity on each
    // read would reload the capacity after every handle store, which may alias
    // a same-width integer.
    struct table {
        static_assert( sizeof( handle_type ) % alignof( chunk_type ) == 0 );
        static_assert( handle_type::is_trivially_moveable ); // relocation below is a memcpy

        handle_type * slots_   { nullptr };
        chunk_type  * keys_    { nullptr };
        index_type    size_    { 0 };
        index_type    capacity_{ 0 };

        table() noexcept = default;
        table( table && other ) noexcept { steal( other ); }
        table & operator=( table && other ) noexcept {
            if ( this != &other ) [[likely]] {
                release();
                steal( other );
            }
            return *this;
        }
        table( table const & ) = delete;
        table & operator=( table const & ) = delete;
        ~table() noexcept { release(); }

        void steal( table & other ) noexcept {
            slots_    = std::exchange( other.slots_   , nullptr );
            keys_     = std::exchange( other.keys_    , nullptr );
            size_     = std::exchange( other.size_    , 0 );
            capacity_ = std::exchange( other.capacity_, 0 );
        }

        [[nodiscard]] handle_type       * slots()       noexcept { return slots_; }
        [[nodiscard]] handle_type const * slots() const noexcept { return slots_; }
        [[nodiscard]] chunk_type        * keys ()       noexcept { return keys_ ; }
        [[nodiscard]] chunk_type  const * keys () const noexcept { return keys_ ; }

        [[nodiscard]] static constexpr std::size_t keys_offset( index_type const capacity ) noexcept { return std::size_t{ capacity } * sizeof( handle_type ); }
        [[nodiscard]] static constexpr std::size_t bytes_for  ( index_type const capacity ) noexcept { return std::size_t{ capacity } * ( sizeof( handle_type ) + sizeof( chunk_type ) ); }

        void reserve( index_type const capacity ) {
            if ( capacity > capacity_ ) {
                relocate( capacity );
            }
        }

        template <typename Handle>
        void push_back( chunk_type const key, Handle && handle ) {
            if ( size_ == capacity_ ) [[unlikely]] {
                insert_growing( size_, key, std::forward<Handle>( handle ) );
                return;
            }
            new ( slots_ + size_ ) handle_type{ std::forward<Handle>( handle ) };
            keys_[ size_ ] = key;
            ++size_;
        }

        template <typename Handle>
        void insert( index_type const index, chunk_type const key, Handle && handle ) {
            if ( size_ == capacity_ ) [[unlikely]] {
                insert_growing( index, key, std::forward<Handle>( handle ) );
                return;
            }
            open_gap( index );
            new ( slots_ + index ) handle_type{ std::forward<Handle>( handle ) };
            keys_[ index ] = key;
            ++size_;
        }

        // The growing arm of push_back/insert, kept out of line: the source may
        // live in this very table, which the relocation would move from under
        // it, so it is materialized before the table grows. Only this (cold)
        // path pays for that copy; the common append constructs straight into
        // the slot.
        template <typename Handle>
        [[using gnu: cold, noinline]] void insert_growing( index_type const index, chunk_type const key, Handle && handle ) {
            handle_type value{ std::forward<Handle>( handle ) };
            grow( size_ + 1 );
            open_gap( index );
            new ( slots_ + index ) handle_type{ std::move( value ) };
            keys_[ index ] = key;
            ++size_;
        }

        // Shifts [index, size_) up by one (both arrays, memmove: trivially
        // moveable handles). Requires size_ < capacity_.
        void open_gap( index_type const index ) noexcept {
            auto const tail{ std::size_t{ size_ } - index };
            std::memmove( slots_ + index + 1, slots_ + index, tail * sizeof( handle_type ) );
            std::memmove( keys_  + index + 1, keys_  + index, tail * sizeof( chunk_type  ) );
        }

        // Destroys [first, last) and closes the gap.
        void erase( index_type const first, index_type const last ) noexcept {
            for ( auto i{ first }; i < last; ++i ) {
                slots()[ i ].~handle_type();
            }
            auto const tail{ std::size_t{ size_ } - last };
            std::memmove( slots() + first, slots() + last, tail * sizeof( handle_type ) );
            std::memmove( keys () + first, keys () + last, tail * sizeof( chunk_type  ) );
            size_ -= ( last - first );
        }

        void destroy_all() noexcept {
            for ( index_type i{ 0 }; i < size_; ++i ) {
                slots()[ i ].~handle_type();
            }
            size_ = 0;
        }

        void release() noexcept {
            destroy_all();
            if ( slots_ ) {
                free_block( reinterpret_cast<std::byte *>( slots_ ) );
                slots_    = nullptr;
                keys_     = nullptr;
                capacity_ = 0;
            }
        }

        void shrink_to_fit() {
            if ( size_ == 0 ) {
                release();
            } else if ( size_ < capacity_ ) {
                relocate( size_ );
            }
        }

        void copy_from( table const & other ) {
            destroy_all();
            reserve( other.size_ );
            std::memcpy( keys(), other.keys(), std::size_t{ other.size_ } * sizeof( chunk_type ) );
            // size_ tracks each constructed handle so a throwing clone leaves
            // the table consistent.
            for ( ; size_ < other.size_; ++size_ ) {
                new ( slots() + size_ ) handle_type{ other.slots()[ size_ ] };
            }
        }

        // [croaring-ref] roaring_array.c extend_array: double while small, 5/4 past 1024.
        [[using gnu: cold, noinline]] void grow( index_type const required ) {
            auto const geometric{ capacity_ < 1024 ? capacity_ * 2 : capacity_ + capacity_ / 4 };
            relocate( std::max( { required, geometric, index_type{ 4 } } ) );
        }

        // Both arrays move to a fresh block of `capacity` entries (never below
        // size_): handles by memcpy (trivially moveable), keys likewise.
        void relocate( index_type const capacity ) {
            auto * const fresh{ allocate_block( bytes_for( capacity ) ) };
            auto * const fresh_slots{ reinterpret_cast<handle_type *>( fresh ) };
            auto * const fresh_keys { reinterpret_cast<chunk_type  *>( fresh + keys_offset( capacity ) ) };
            if ( size_ != 0 ) {
                std::memcpy( fresh_slots, slots_, std::size_t{ size_ } * sizeof( handle_type ) );
                std::memcpy( fresh_keys , keys_ , std::size_t{ size_ } * sizeof( chunk_type  ) );
            }
            if ( slots_ ) {
                free_block( reinterpret_cast<std::byte *>( slots_ ) );
            }
            slots_    = fresh_slots;
            keys_     = fresh_keys;
            capacity_ = capacity;
        }

        [[nodiscard]] static std::byte * allocate_block( std::size_t const bytes ) {
#if FRSR_ROARING_HAS_MIMALLOC
            auto * const block{ static_cast<std::byte *>( mi_malloc( bytes ) ) };
            if ( block == nullptr ) [[unlikely]] { throw std::bad_alloc{}; }
            return block;
#else
            return static_cast<std::byte *>( ::operator new( bytes ) );
#endif
        }
        static void free_block( std::byte * const block ) noexcept {
#if FRSR_ROARING_HAS_MIMALLOC
            mi_free( block );
#else
            ::operator delete( block );
#endif
        }
    };

    table live_;
    // Scratch payload reuse (clear_retiring_slots / retire / take_retired):
    // consumed slots offered for in-place rebuild; per-kind claim cursors. Only
    // the handle half of this table is meaningful.
    table retired_;
    std::uint32_t retired_array_next_ { 0 };
    std::uint32_t retired_bitset_next_{ 0 };
};

} // namespace frsr::roaring::detail
