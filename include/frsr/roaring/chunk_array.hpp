#pragma once

#include <frsr/roaring/container_handle.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <utility>

#if defined( FRSR_ROARING_HAS_PSI_VM )
#   include <psi/vm/containers/lookup.hpp>
#endif

namespace frsr::roaring::detail {

// The sorted chunk table of a bitmap whose chunk keys are 16 bits wide (so a table holds at most
// 65536 entries): the sorted chunk keys live in their own dense array, so the merge and lookup walks
// touch only the keys (2 bytes apart) and reach for the 32-byte container handle in the parallel
// handles array on a key match only.
//
// Both arrays live in ONE allocation, exactly as the reference lays out its roaring_array_t: the
// handles first, the keys behind them at the capacity boundary. Producing a result bitmap therefore
// costs one allocation and one free for its chunk table, not one per array, and growth relocates
// both arrays with a single trip through the allocator. (Two separate growth-managed vectors
// measured as the largest fixed per-result cost the library pays.)
// [croaring-ref] deps/croaring/include/roaring/containers/containers.h: roaring_array_t{keys;containers;typecodes}
// [croaring-ref] deps/croaring/src/roaring_array.c: ra_init_with_capacity / realloc_array (one bigalloc)
//
// `capacity` handles, then `capacity` keys. sizeof( Handle ) is a multiple of every key width's
// alignment, so the key array needs no padding. Both halves are held as typed pointers (the key
// pointer is derived once, on relocation) so a key or slot read is one load; deriving it from the
// capacity on each read would reload the capacity after every handle store, which may alias a
// same-width integer.
//
// The same table, with the same interface, is the flat pool of parked payloads that every chunk
// store keeps for scratch reuse.
template <typename ChunkType, typename Handle>
class chunk_array {
public:
    using chunk_type  = ChunkType;
    using handle_type = Handle;
    using index_type  = std::uint32_t;

    static_assert( sizeof( handle_type ) % alignof( chunk_type ) == 0 );
    static_assert( handle_type::is_trivially_moveable ); // relocation below is a memcpy

    // A position in key order; end() is the position past the last entry. It stays valid until
    // the table is restructured.
    class cursor {
    public:
        constexpr cursor() noexcept = default;
        constexpr cursor & operator++() noexcept { ++index_; return *this; }
        [[nodiscard]] friend constexpr bool operator==( cursor, cursor ) noexcept = default;

    private:
        friend class chunk_array;
        constexpr explicit cursor( index_type const index ) noexcept : index_{ index } {}
        index_type index_{ 0 };
    };

    chunk_array() noexcept = default;
    chunk_array( chunk_array && other ) noexcept { steal( other ); }
    chunk_array & operator=( chunk_array && other ) noexcept {
        if ( this != &other ) [[likely]] {
            release();
            steal( other );
        }
        return *this;
    }
    chunk_array( chunk_array const & ) = delete;
    chunk_array & operator=( chunk_array const & ) = delete;
    ~chunk_array() noexcept { release(); }

    [[nodiscard]] index_type size    () const noexcept { return size_; }
    [[nodiscard]] bool       empty   () const noexcept { return size_ == 0; }
    [[nodiscard]] index_type capacity() const noexcept { return capacity_; }

    [[nodiscard]] cursor begin() const noexcept { return cursor{ 0 }; }
    [[nodiscard]] cursor end  () const noexcept { return cursor{ size_ }; }
    // The final entry. The table must not be empty.
    [[nodiscard]] cursor last () const noexcept { assert( size_ != 0 ); return cursor{ size_ - 1 }; }
    // The entry before `at` (end() included), which must not be begin().
    [[nodiscard]] cursor prev( cursor const at ) const noexcept { assert( at.index_ != 0 ); return cursor{ at.index_ - 1 }; }

    // The entry at `at`, which must not be end().
    [[nodiscard]] chunk_type key( cursor const at ) const noexcept { assert( at.index_ < size_ ); return keys_[ at.index_ ]; }
    [[nodiscard]] handle_type       & slot( cursor const at )       noexcept { assert( at.index_ < size_ ); return slots_[ at.index_ ]; }
    [[nodiscard]] handle_type const & slot( cursor const at ) const noexcept { assert( at.index_ < size_ ); return slots_[ at.index_ ]; }

    // The first entry whose key is not below `key` (end() if there is none): a binary search over
    // the dense keys array only. A pure read - the search keeps no state between calls, which is
    // what lets concurrent readers share the table. With psi::vm the search is the branchless one
    // that prefetches both candidates of the next step: a lookup into a large table is a chain of
    // dependent cache misses, and overlapping them is what the reference's tree-shaped index gets
    // from its fan-out.
    [[nodiscard]] cursor lower_bound( chunk_type const key ) const noexcept {
        auto const * const first{ keys_ };
        auto const * const last { first + size_ };
#if defined( FRSR_ROARING_HAS_PSI_VM )
        auto const position{ psi::vm::prefetching_lower_bound( first, last, key ) };
#else
        auto const position{ std::lower_bound( first, last, key ) };
#endif
        return cursor{ static_cast<index_type>( position - first ) };
    }

    // lower_bound() for a caller that already stands at `from` and only moves forward (a
    // leapfrogging walk over several tables): the first entry at or after `from` whose key is
    // not below `key`. Gallops from `from`, so the cost follows the distance travelled.
    [[nodiscard]] cursor lower_bound( chunk_type const key, cursor const from ) const noexcept {
        auto const n{ size_ };
        auto low{ from.index_ };
        if ( low >= n || !( keys_[ low ] < key ) ) {
            return cursor{ low < n ? low : n };
        }
        auto step{ index_type{ 1 } };
        while ( low + step < n && keys_[ low + step ] < key ) {   // keys_[ low ] < key throughout
            low  += step;
            step <<= 1;
        }
        auto high{ std::min( low + step, n ) };                   // keys_[ high ] >= key, or high == n
        while ( low + 1 < high ) {
            auto const mid{ low + ( high - low ) / 2 };
            ( keys_[ mid ] < key ? low : high ) = mid;
        }
        return cursor{ high };
    }

    [[nodiscard]] std::span<handle_type      > slots()       noexcept { return { slots_, size_ }; }
    [[nodiscard]] std::span<handle_type const> slots() const noexcept { return { slots_, size_ }; }
    // The handle storage itself, for the pool of parked payloads (which reads it by number).
    [[nodiscard]] handle_type       * data()       noexcept { return slots_; }
    [[nodiscard]] handle_type const * data() const noexcept { return slots_; }

    // Bytes held, spare capacity included (not the spilled payloads).
    [[nodiscard]] std::size_t node_bytes() const noexcept { return bytes_for( capacity_ ); }

    // Inserts before `at` and returns the position of the new entry.
    template <typename H>
    cursor insert( cursor const at, chunk_type const key, H && handle ) {
        if ( size_ == capacity_ ) [[unlikely]] {
            insert_growing( at.index_, key, std::forward<H>( handle ) );
            return at;
        }
        open_gap( at.index_ );
        new ( slots_ + at.index_ ) handle_type{ std::forward<H>( handle ) };
        keys_[ at.index_ ] = key;
        ++size_;
        return at;
    }

    // Inserts a copy of every entry of `other` whose key is not held here and for which `wanted`
    // holds; `count` is how many that is. The table is grown once and the entries above each
    // insertion point are moved once, from the back: an entry takes its place among the keys below
    // the previous insertion point, so a few insertions into a long table move the table once
    // instead of once per entry.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_inplace_union_few (second loop)
    template <typename Wanted>
    void insert_absent_from( chunk_array const & other, index_type const count, Wanted const & wanted ) {
        if ( count == 0 ) { return; }
        assert( this != &other );
        auto const total{ size_ + count };
        // A table that has to grow is rebuilt in its new block, each run of entries going straight
        // to its final place, instead of being relocated first and moved again within the block.
        bool const relocating{ total > capacity_ };
        std::byte * fresh{ nullptr };
        auto new_capacity{ capacity_ };
        auto * dst_slots{ slots_ };
        auto * dst_keys { keys_ };
        if ( relocating ) {
            auto const capacity{ std::max( { total, capacity_ < 1024 ? capacity_ * 2 : capacity_ + capacity_ / 4, index_type{ 4 } } ) };
            fresh     = allocate_block( bytes_for( capacity ) );
            dst_slots = reinterpret_cast<handle_type *>( fresh );
            dst_keys  = reinterpret_cast<chunk_type  *>( fresh + keys_offset( capacity ) );
            new_capacity = capacity;
        }
        auto end{ size_ };                              // [ 0, end ) is still in place
        auto dst{ total };                              // [ dst, total ) is final
        try {
            for ( auto j{ other.size_ }; j-- != 0 && dst > end; ) {
                auto const key{ other.keys_[ j ] };
                auto const index{ static_cast<index_type>( std::lower_bound( keys_, keys_ + end, key ) - keys_ ) };
                if ( ( index < end && keys_[ index ] == key ) || !wanted( other.slots_[ j ] ) ) { continue; }
                handle_type copy{ other.slots_[ j ] };  // may throw: nothing has moved yet for this entry
                auto const tail{ end - index };
                dst -= tail;
                std::memmove( static_cast<void *>( dst_slots + dst ), static_cast<void const *>( slots_ + index ), std::size_t{ tail } * sizeof( handle_type ) );
                std::memmove( dst_keys + dst, keys_ + index, std::size_t{ tail } * sizeof( chunk_type ) );
                --dst;
                new ( dst_slots + dst ) handle_type{ std::move( copy ) };
                dst_keys[ dst ] = key;
                end = index;
            }
        } catch ( ... ) {
            if ( relocating ) {
                // The entries held are still in this table; those of the new block that are not among
                // them are the copies made so far.
                for ( auto i{ dst }; i < total; ++i ) {
                    if ( !std::binary_search( keys_, keys_ + size_, dst_keys[ i ] ) ) { dst_slots[ i ].~handle_type(); }
                }
                free_block( fresh );
            } else {
                close_gap( end, dst, total );
            }
            throw;
        }
        assert( dst == end && "the count was exact" );
        if ( relocating ) {
            if ( end != 0 ) {
                std::memcpy( static_cast<void *>( dst_slots ), static_cast<void const *>( slots_ ), std::size_t{ end } * sizeof( handle_type ) );
                std::memcpy( dst_keys, keys_, std::size_t{ end } * sizeof( chunk_type ) );
            }
            if ( slots_ ) { free_block( reinterpret_cast<std::byte *>( slots_ ) ); }
            slots_    = dst_slots;
            keys_     = dst_keys;
            capacity_ = new_capacity;
        }
        size_ = total;
    }

    // Removes the entry at `at` and returns the position of the entry that followed it.
    cursor erase( cursor const at ) {
        erase_range( at.index_, at.index_ + 1 );
        return at;
    }

    // Appends an entry whose key is above every key held. Returns whether the existing entries moved.
    template <typename H>
    bool push_back( chunk_type const key, H && handle ) {
        if ( size_ == capacity_ ) [[unlikely]] {
            insert_growing( size_, key, std::forward<H>( handle ) );
            return true;
        }
        new ( slots_ + size_ ) handle_type{ std::forward<H>( handle ) };
        keys_[ size_ ] = key;
        ++size_;
        return false;
    }

    // Storage for the next appended entry, to be filled with a handle and then committed under its
    // key. Capacity for it must have been reserved; nothing is moved. The opened slot is not part
    // of the table until committed.
    [[nodiscard]] handle_type * open_back( bool & moved ) noexcept {
        assert( size_ < capacity_ && "open_back() needs reserved capacity" );
        moved = false;
        return slots_ + size_;
    }
    void commit_back( chunk_type const key ) noexcept {
        assert( size_ < capacity_ );
        keys_[ size_ ] = key;
        ++size_;
    }

    // Makes room for `count` entries; returns whether the existing entries moved.
    bool reserve( std::size_t const count ) {
        if ( static_cast<index_type>( count ) > capacity_ ) {
            relocate( static_cast<index_type>( count ) );
            return true;
        }
        return false;
    }

    void shrink_to_fit() {
        if ( size_ == 0 ) {
            release();
        } else if ( size_ < capacity_ ) {
            relocate( size_ );
        }
    }

    // Destroys every entry; the capacity stays.
    void destroy_all() noexcept {
        for ( index_type i{ 0 }; i < size_; ++i ) {
            slots_[ i ].~handle_type();
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

    // Four field exchanges: the generation swap (temp + two move-assigns, each releasing an
    // already-empty table) showed up as a tenth of a small-result band.
    [[gnu::always_inline]] void swap( chunk_array & other ) noexcept {
        std::swap( slots_   , other.slots_    );
        std::swap( keys_    , other.keys_     );
        std::swap( size_    , other.size_     );
        std::swap( capacity_, other.capacity_ );
    }

    // Makes this a copy of `other`.
    void copy_from( chunk_array const & other ) {
        destroy_all();
        if ( other.size_ == 0 ) { return; }
        reserve( other.size_ );
        std::memcpy( keys_, other.keys_, std::size_t{ other.size_ } * sizeof( chunk_type ) );
        // size_ tracks each constructed handle so a throwing clone leaves the table consistent.
        for ( ; size_ < other.size_; ++size_ ) {
            new ( slots_ + size_ ) handle_type{ other.slots_[ size_ ] };
        }
    }

    // The in-place rewrite of the table by the walk that is consuming it (the shrinking set
    // operations, &= and -=): the walk visits the entries in key order and, for each, either keeps
    // it (put back as it is, or as the walk modified it in place) or replaces it; every entry it
    // does not mention is dropped. The entries written are always behind the walk, so none is
    // overwritten before it has been read. `retire` receives the payload of each entry that goes
    // away. finish() ends the rewrite and must be called once.
    template <typename Retire>
    class rewriter {
    public:
        rewriter( chunk_array & table, Retire retire ) noexcept : table_{ &table }, retire_{ std::move( retire ) } {}

        // The entry at `from`, which the walk has finished with, is kept.
        void keep( cursor const from ) {
            if ( write_ != from.index_ ) {
                table_->keys_[ write_ ] = table_->keys_[ from.index_ ];
                retire_( std::move( table_->slots_[ write_ ] ) );
                table_->slots_[ write_ ] = std::move( table_->slots_[ from.index_ ] );
            }
            ++write_;
        }

        // The walk's current entry is replaced by (key, container).
        void put( chunk_type const key, handle_type && container ) {
            table_->keys_[ write_ ] = key;
            retire_( std::move( table_->slots_[ write_ ] ) );
            table_->slots_[ write_ ] = std::move( container );
            ++write_;
        }

        void finish() {
            for ( auto i{ write_ }; i < table_->size_; ++i ) {
                retire_( std::move( table_->slots_[ i ] ) );
            }
            if ( write_ != table_->size_ ) {
                table_->erase_range( write_, table_->size_ );
            }
        }

    private:
        chunk_array * table_;
        Retire        retire_;
        index_type    write_{ 0 };
    };

    // Checks the structural invariants; the first one that does not hold, or nullptr.
    [[nodiscard]] char const * validate() const noexcept {
        for ( index_type i{ 1 }; i < size_; ++i ) {
            if ( !( keys_[ i - 1 ] < keys_[ i ] ) ) { return "chunk keys are not strictly ascending"; }
        }
        return size_ <= capacity_ ? nullptr : "the size exceeds the capacity";
    }

private:
    [[nodiscard]] static constexpr std::size_t bytes_for( index_type const capacity ) noexcept { return std::size_t{ capacity } * ( sizeof( handle_type ) + sizeof( chunk_type ) ); }
    [[nodiscard]] static constexpr std::size_t keys_offset( index_type const capacity ) noexcept { return std::size_t{ capacity } * sizeof( handle_type ); }

    void steal( chunk_array & other ) noexcept {
        slots_    = std::exchange( other.slots_   , nullptr );
        keys_     = std::exchange( other.keys_    , nullptr );
        size_     = std::exchange( other.size_    , 0 );
        capacity_ = std::exchange( other.capacity_, 0 );
    }

    // The growing arm of an insertion, kept out of line: the source may live in this very table,
    // which the relocation would move from under it, so it is materialized before the table grows.
    // Only this (cold) path pays for that copy; the common insertion constructs straight into the slot.
    template <typename H>
    [[using gnu: cold, noinline]] void insert_growing( index_type const index, chunk_type const key, H && handle ) {
        handle_type value{ std::forward<H>( handle ) };
        grow( size_ + 1 );
        open_gap( index );
        new ( slots_ + index ) handle_type{ std::move( value ) };
        keys_[ index ] = key;
        ++size_;
    }

    // Shifts [index, size_) up by one (both arrays, memmove: trivially moveable handles). Requires size_ < capacity_.
    void open_gap( index_type const index ) noexcept {
        auto const tail{ std::size_t{ size_ } - index };
        std::memmove( static_cast<void *>( slots_ + index + 1 ), static_cast<void const *>( slots_ + index ), tail * sizeof( handle_type ) );
        std::memmove( keys_  + index + 1, keys_  + index, tail * sizeof( chunk_type ) );
    }

    // Entries [ end, dst ) of the first `total` are a hole between the entries below and the ones above
    // it: moves the upper ones down and sets the size to what is held.
    void close_gap( index_type const end, index_type const dst, index_type const total ) noexcept {
        std::memmove( static_cast<void *>( slots_ + end ), static_cast<void const *>( slots_ + dst ), std::size_t{ total - dst } * sizeof( handle_type ) );
        std::memmove( keys_ + end, keys_ + dst, std::size_t{ total - dst } * sizeof( chunk_type ) );
        size_ = end + ( total - dst );
    }

    // Destroys [first, last) and closes the gap.
    void erase_range( index_type const first, index_type const last ) noexcept {
        for ( auto i{ first }; i < last; ++i ) {
            slots_[ i ].~handle_type();
        }
        auto const tail{ std::size_t{ size_ } - last };
        std::memmove( static_cast<void *>( slots_ + first ), static_cast<void const *>( slots_ + last ), tail * sizeof( handle_type ) );
        std::memmove( keys_ + first, keys_ + last, tail * sizeof( chunk_type ) );
        size_ -= ( last - first );
    }

    // [croaring-ref] roaring_array.c extend_array: double while small, 5/4 past 1024.
    [[using gnu: cold, noinline]] void grow( index_type const required ) {
        auto const geometric{ capacity_ < 1024 ? capacity_ * 2 : capacity_ + capacity_ / 4 };
        relocate( std::max( { required, geometric, index_type{ 4 } } ) );
    }

    // Both arrays move to a fresh block of `capacity` entries (never below size_): handles by
    // memcpy (trivially moveable), keys likewise.
    void relocate( index_type const capacity ) {
        auto * const fresh{ allocate_block( bytes_for( capacity ) ) };
        auto * const fresh_slots{ reinterpret_cast<handle_type *>( fresh ) };
        auto * const fresh_keys { reinterpret_cast<chunk_type  *>( fresh + keys_offset( capacity ) ) };
        if ( size_ != 0 ) {
            std::memcpy( static_cast<void *>( fresh_slots ), static_cast<void const *>( slots_ ), std::size_t{ size_ } * sizeof( handle_type ) );
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

    handle_type * slots_   { nullptr };
    chunk_type  * keys_    { nullptr };
    index_type    size_    { 0 };
    index_type    capacity_{ 0 };
};

} // namespace frsr::roaring::detail
