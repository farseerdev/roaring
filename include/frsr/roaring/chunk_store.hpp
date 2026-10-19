#pragma once

#include <frsr/roaring/chunk_array.hpp>
#include <frsr/roaring/chunk_tree.hpp>
#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/containers.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace frsr::roaring::detail {

// The chunk table of a bitmap: its chunk keys in ascending order, each with the container handle of
// the chunk. What holds them depends on the width of the chunk key. A 16-bit chunk key (any key type
// up to 32 bits) bounds the table at 65536 entries, which a sorted array serves best: contiguous,
// searched by a binary search, and just what the reference uses. A wider chunk key has no such
// bound, and an array's insertions and removals, which shift its tail, become the cost of the
// bitmap; those keys live in a B+ tree instead.
//
// Whichever it is, the table is walked through cursors, searched without writing anything (so
// concurrent readers of one table are safe), and rewritten by appenders and a rewriter, never by
// position.
//
// Beside the live entries the store keeps a pool of parked payloads for scratch reuse (see
// clear_retiring_slots), always a flat array.
template <typename Layout, typename CowPolicy = cow_value_semantics>
class chunk_store {
public:
    using chunk_type  = typename Layout::chunk_type;
    using handle_type = container_handle<Layout, CowPolicy>;

    // A chunk key wider than 16 bits.
    static constexpr bool wide{ sizeof( chunk_type ) > sizeof( std::uint16_t ) };
    static_assert( !wide || Layout::key_bits - Layout::low_bits < 63, "a wide chunk key must stay below the tree's pad" );

private:
    using flat_table = chunk_array<chunk_type, handle_type>;
    using live_table = std::conditional_t<wide, chunk_tree<handle_type>, flat_table>;

public:
    using cursor = typename live_table::cursor;

    [[nodiscard]] std::size_t size () const noexcept { return live_.size();  }
    [[nodiscard]] bool        empty() const noexcept { return live_.empty(); }

    // Structure generation: advances whenever a live slot may have moved or
    // died (relocation, insert, erase, compaction, swap, clear), so a
    // caller holding a pointer to a slot can tell in one compare whether it is
    // still the slot it took. Appends that do not relocate leave it alone.
    [[nodiscard]] std::uint32_t generation() const noexcept { return generation_; }

    [[nodiscard]] cursor begin() const noexcept { return live_.begin(); }
    [[nodiscard]] cursor end  () const noexcept { return live_.end(); }
    // The final entry. The table must not be empty.
    [[nodiscard]] cursor last () const noexcept { return live_.last(); }
    // The entry before `at` (end() included), which must not be begin().
    [[nodiscard]] cursor prev( cursor const at ) const noexcept { return live_.prev( at ); }

    // The entry at `at`, which must not be end().
    [[nodiscard]] chunk_type key( cursor const at ) const noexcept { return static_cast<chunk_type>( live_.key( at ) ); }
    [[nodiscard]] handle_type       & slot( cursor const at )       noexcept { return live_.slot( at ); }
    [[nodiscard]] handle_type const & slot( cursor const at ) const noexcept { return live_.slot( at ); }

    // The first entry whose key is not below `key` (end() if there is none). A pure read.
    [[nodiscard]] cursor lower_bound( chunk_type const key ) const noexcept { return live_.lower_bound( key ); }

    // lower_bound() for a caller that already stands at `from` and only moves forward (a
    // leapfrogging walk over several tables).
    [[nodiscard]] cursor lower_bound( chunk_type const key, cursor const from ) const noexcept { return live_.lower_bound( key, from ); }

    // The handles in key order.
    [[nodiscard]] auto slots()       noexcept { return live_.slots(); }
    [[nodiscard]] auto slots() const noexcept { return std::as_const( live_ ).slots(); }

    // Storage footprint of the table itself (not the spilled payloads).
    [[nodiscard]] std::size_t entry_bytes() const noexcept { return live_.node_bytes(); }

    // Checks the structural invariants; the first one that does not hold, or nullptr.
    [[nodiscard]] char const * validate() const noexcept { return live_.validate(); }

    // Compaction (bitmap::shrink_to_fit): releases the parked scratch payloads
    // and the chunk table's spare capacity. Scratch reuse repopulates on the
    // next combine, so this is a footprint/throughput trade the CALLER asked for
    // — never call it from a set operation.
    void shrink_to_fit() {
        retired_.release();
        retired_array_next_ = retired_bitset_next_ = 0;
        live_.shrink_to_fit();
        ++generation_;
    }

    // Makes room for `capacity` entries ahead of an append run. A tree grows by node and has no use for it.
    void reserve( std::size_t const capacity ) {
        if constexpr ( !wide ) {
            if ( live_.reserve( capacity ) ) { ++generation_; }
        } else {
            (void)capacity;
        }
    }

    // Retired slots (scratch payload reuse) are deliberately per-instance
    // transient state: copies start with none.
    chunk_store() = default;
    chunk_store( chunk_store const & other ) { live_.copy_from( other.live_ ); }
    chunk_store( chunk_store && ) noexcept = default;
    chunk_store & operator=( chunk_store const & other ) {
        if ( this != &other ) [[likely]] {
            live_.copy_from( other.live_ );
            ++generation_;
        }
        return *this;
    }
    chunk_store & operator=( chunk_store && other ) noexcept {
        live_                = std::move( other.live_ );
        retired_             = std::move( other.retired_ );
        retired_array_next_  = other.retired_array_next_;
        retired_bitset_next_ = other.retired_bitset_next_;
        ++generation_;
        return *this;
    }

    void clear() noexcept {
        live_   .destroy_all();
        retired_.destroy_all();
        retired_array_next_ = retired_bitset_next_ = 0;
        ++generation_;
    }

    // Scratch payload reuse (CRoaring persistent-dst analog, see
    // container_handle::offers_reusable_payload): logically empties the store
    // but parks the current slots — payloads intact — for take_retired() to
    // hand back to the next run's materializing sites. A flat table swaps with
    // the pool, which also recycles the previous generation's pool back as the
    // live table, so the chunk table itself keeps its capacity across runs too.
    // A tree moves its handles into the pool and frees its nodes.
    void clear_retiring_slots() noexcept {
        retired_.destroy_all();
        // An empty live table has nothing to park, and keeps its capacity: the
        // fresh result of every materializing operator arrives here empty.
        if ( live_.size() != 0 ) {
            if constexpr ( wide ) {
                retired_.reserve( live_.size() );
                for ( auto & slot : live_.slots() ) {
                    retired_.push_back( chunk_type{}, std::move( slot ) );
                }
                live_.destroy_all();
            } else {
                live_.swap( retired_ );
            }
        }
        retired_array_next_ = retired_bitset_next_ = 0;
        ++generation_;
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
        auto next{ ( kind == container_kind::bitset ) ? retired_bitset_next_ : retired_array_next_ };
        auto const * const retired{ retired_.data() };
        while ( next < retired_.size() ) {
            auto const & candidate{ retired[ next++ ] };
            if ( candidate.offers_reusable_payload( kind ) ) {
                return candidate.template payload_capacity<E>();
            }
        }
        return 0;
    }

    [[nodiscard]] handle_type take_retired( container_kind const kind ) noexcept {
        handle_type handle;
        take_retired_into( &handle, kind );
        return handle;
    }

    // take_retired( kind ) constructed straight into `destination`, which must be
    // uninitialized storage or a handle that owns no payload: the handle is
    // constructed over it without its destructor running.
    handle_type & take_retired_into( handle_type * const destination, container_kind const kind ) noexcept {
        auto & next{ kind == container_kind::bitset ? retired_bitset_next_ : retired_array_next_ };
        auto * const retired{ retired_.data() };
        while ( next < retired_.size() ) {
            auto & candidate{ retired[ next++ ] };
            if ( candidate.offers_reusable_payload( kind ) ) {
                auto & handle{ *std::construct_at( destination, std::move( candidate ) ) };
                if ( kind == container_kind::bitset ) { handle.reset_for_bitset_reuse(); }
                else                                  { handle.reset_for_array_reuse (); }
                return handle;
            }
        }
        return *std::construct_at( destination );
    }

    // In-slot construction for a table filled by appends (the destination of a materializing
    // combine): open_back() yields the storage past the last entry, the caller constructs and fills
    // a handle there (take_retired_into), and commit_back() makes it the last entry under `key` —
    // a finished container is never moved into the table. A flat table must have reserved the
    // capacity beforehand; neither call relocates then, so the opened handle stays put between the
    // two as long as nothing else appends meanwhile, and generation() does not advance (as for a
    // non-relocating push_back). The opened slot is not part of the table until committed: a caller
    // that abandons it must leave it owning no payload.
    [[nodiscard]] handle_type * open_back() {
        bool moved;
        auto * const slot{ live_.open_back( moved ) };
        if ( moved ) [[unlikely]] { ++generation_; }
        return slot;
    }

    void commit_back( chunk_type const key ) {
        live_.commit_back( key );
    }

    // Parks a consumed handle's payload for take_retired() instead of freeing
    // it — the in-place combine walk pairs each such free with a fresh result
    // allocation one pair later, so this turns that free+alloc churn into
    // in-place reuse. Only what take_retired() can hand out is parked: a
    // non-spilled handle has no payload to offer, and a shared one cannot be
    // rebuilt in place — parking it would allocate a retired table for nothing
    // and hold the co-owners' payload alive until the next clear.
    void retire( handle_type && handle ) {
        if ( handle.offers_reusable_payload( handle.kind() ) ) {
            retired_.push_back( chunk_type{}, std::move( handle ) );
        }
    }

    // The in-place rewrite of the table by the walk that is consuming it (the shrinking set
    // operations, &= and -=): the walk visits the entries in key order and, for each, either
    // keeps it (put back as it is, or as the walk modified it in place) or replaces it; every
    // entry it does not mention is dropped. The payloads that would be freed are retired for
    // take_retired() instead. finish() ends the rewrite and must be called once.
    class rewriter {
    public:
        explicit rewriter( chunk_store & store ) noexcept : store_{ &store }, inner_{ store.live_, retire_into{ &store } } {}

        // The entry at `from`, which the walk has finished with, is kept.
        void keep( cursor const from ) { inner_.keep( from ); }

        // The walk's current entry is replaced by (key, container).
        void put( chunk_type const key, handle_type && container ) { inner_.put( key, std::move( container ) ); }

        void finish() {
            inner_.finish();
            ++store_->generation_;
        }

    private:
        struct retire_into {
            chunk_store * store;
            void operator()( handle_type && handle ) const { store->retire( std::move( handle ) ); }
        };

        chunk_store *                                          store_;
        typename live_table::template rewriter<retire_into>    inner_;
    };

    void swap( chunk_store & other ) noexcept {
        live_   .swap( other.live_    );
        retired_.swap( other.retired_ );
        std::swap( retired_array_next_ , other.retired_array_next_  );
        std::swap( retired_bitset_next_, other.retired_bitset_next_ );
        ++generation_;
        ++other.generation_;
    }

    template <typename Handle>
    void push_back( chunk_type const key, Handle && handle ) {
        if ( live_.push_back( key, std::forward<Handle>( handle ) ) ) [[unlikely]] { ++generation_; }   // this append relocated
    }

    // Inserts before `at` (the lower_bound of `key`) and returns the position of the new entry.
    template <typename Handle>
    cursor insert( cursor const at, chunk_type const key, Handle && handle ) {
        auto const inserted{ live_.insert( at, key, std::forward<Handle>( handle ) ) };
        ++generation_;
        return inserted;
    }

    // Inserts a copy of every entry of `other` (another store's table, never this one) whose key is
    // not held here and for which `wanted( handle )` holds; `count` is how many that is. A flat table
    // is grown once and its tail moved once per run of insertions, from the back; a tree seeks and
    // inserts each.
    template <typename Wanted>
    void insert_absent_from( chunk_store const & other, std::size_t const count, Wanted const & wanted ) {
        live_.insert_absent_from( other.live_, static_cast<std::uint32_t>( count ), wanted );
        ++generation_;
    }

    // Removes the entry at `at` and returns the position of the entry that followed it.
    cursor erase( cursor const at ) {
        auto const following{ live_.erase( at ) };
        ++generation_;
        return following;
    }

private:
    live_table live_;
    // Scratch payload reuse (clear_retiring_slots / retire / take_retired):
    // consumed slots offered for in-place rebuild; per-kind claim cursors. Only
    // the handle half of this table is meaningful.
    flat_table retired_;
    std::uint32_t retired_array_next_ { 0 };
    std::uint32_t retired_bitset_next_{ 0 };
    std::uint32_t generation_         { 0 };
};

} // namespace frsr::roaring::detail
