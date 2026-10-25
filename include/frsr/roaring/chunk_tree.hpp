#pragma once

#include <frsr/roaring/container_handle.hpp>   // FRSR_ROARING_HAS_MIMALLOC
#include <frsr/roaring/tuning.hpp>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace frsr::roaring::detail {

// A key slot that holds no key. Every key stored in a node is below it, so the unused tail of a
// node's key array never counts as "below the probe" and a node can be searched at its full,
// fixed width without ever reading how many keys it holds.
inline constexpr std::int64_t chunk_key_pad{ std::numeric_limits<std::int64_t>::max() };

// The node scans: how many of the keys, sorted ascending with the pad behind them, are below `key`.
// That is the position `key` has in the node.
#if FRSR_ROARING_NODE_SCAN_AVX2
template <unsigned Width>
[[nodiscard, gnu::always_inline]] inline unsigned count_below( std::int64_t const * const keys, std::int64_t const key ) noexcept {
    static_assert( Width % 4 == 0 );
    auto const probe{ _mm256_set1_epi64x( key ) };
    auto const below{ [&]( unsigned const i ) noexcept {
        return _mm256_cmpgt_epi64( probe, _mm256_loadu_si256( reinterpret_cast<__m256i const *>( keys + i ) ) );
    } };
    if constexpr ( Width % 16 == 0 ) {
        // sixteen keys narrow into one vector of bytes, two bytes per key, so one mask and one
        // popcount serve them all instead of one of each per four keys
        unsigned long long mask{ 0 };
        for ( unsigned i{ 0 }; i < Width; i += 16 ) {
            auto const lo{ _mm256_packs_epi32( below( i      ), below( i + 4  ) ) };
            auto const hi{ _mm256_packs_epi32( below( i + 8  ), below( i + 12 ) ) };
            mask |= static_cast<unsigned long long>( static_cast<unsigned>( _mm256_movemask_epi8( _mm256_packs_epi16( lo, hi ) ) ) ) << ( i * 2 );
        }
        return static_cast<unsigned>( std::popcount( mask ) ) / 2;
    } else {
        unsigned count{ 0 };
        for ( unsigned i{ 0 }; i < Width; i += 4 ) {
            count += static_cast<unsigned>( std::popcount( static_cast<unsigned>( _mm256_movemask_pd( _mm256_castsi256_pd( below( i ) ) ) ) ) );
        }
        return count;
    }
}
#else
// A branchless binary search over a fixed power-of-two width.
template <unsigned Width>
[[nodiscard, gnu::always_inline]] inline unsigned count_below( std::int64_t const * const keys, std::int64_t const key ) noexcept {
    static_assert( std::has_single_bit( Width ) );
    unsigned position{ 0 };
    for ( unsigned step{ Width / 2 }; step != 0; step /= 2 ) {
        position += ( keys[ position + step - 1 ] < key ) ? step : 0U;
    }
    return position + ( keys[ position ] < key ? 1U : 0U );
}
#endif

// The sorted chunk table of a bitmap whose chunk keys are wider than 16 bits: cache-line wide
// leaves in key order, in place of the always-sorted array a 16-bit key can afford, found through
// one of two indexes.
//
// A leaf holds up to 15 entries (a key and the handle). A leaf's keys are searched with one
// fixed-width scan (four vector compares, or four halving steps). The leaves are linked, so a walk
// in key order never goes back up. A bitmap of up to 15 chunks is a single leaf that grows like a
// vector (4, 8, then 15 entries), so a small bitmap stays small.
//
// The index over the leaves is either
//   - a B+ tree: inner nodes of 16 separators and 17 children, one scan per level, so the levels are
//     few (a few thousand chunks need two, a few hundred thousand three) and any key distribution
//     is served; or
//   - a flat directory, for a large table (see tuning.hpp) whose keys are spread evenly: one pointer
//     per bucket of the key space, so a lookup is one table read and a short walk along the leaf
//     chain. It replaces the inner levels: only one of the two exists at a time, and a table moves
//     between them as its keys do ("the leaf directory" below).
// Every const member is a pure read.
//
// Keys are held as signed 64-bit numbers, which is what the vector compares work on: a chunk key
// has to stay below the pad (a 64-bit bitmap's has 48 bits). A leaf block of capacity c is laid out
//     [ links, full leaf only ][ c handles, the last one first ][ head: size, capacity ][ keys ]
// and a leaf is named by the address of its first key, so the keys, the handle at an index and the
// head sit at fixed distances from it whatever the capacity.
template <typename Handle, std::uint64_t DirectoryMinEntries = chunk_directory_min_entries>
class chunk_tree {
public:
    using handle_type = Handle;
    using key_type    = std::uint64_t;

    static_assert( Handle::is_trivially_moveable );
    static_assert( alignof( Handle ) <= 8 );

    static constexpr unsigned leaf_capacity{ 15 };
    static constexpr unsigned inner_keys    { 16 };

private:
    using key_t    = std::int64_t;
    using leaf_ptr = key_t *;
    static constexpr key_t pad{ chunk_key_pad };

    struct leaf_head {
        std::uint32_t size;
        std::uint32_t cap;
    };

    struct alignas( 64 ) inner_node {
        key_t         keys [ inner_keys ];
        void *        child[ inner_keys + 1 ];
        std::uint32_t n;                          // separators in use; children = n + 1
    };

    [[nodiscard]] static constexpr unsigned padded( unsigned const cap ) noexcept { return ( cap + 3U ) & ~3U; }
    [[nodiscard]] static constexpr std::size_t links_bytes( unsigned const cap ) noexcept { return cap == leaf_capacity ? 2 * sizeof( leaf_ptr ) : 0; }
    [[nodiscard]] static constexpr std::size_t keys_offset( unsigned const cap ) noexcept { return links_bytes( cap ) + cap * sizeof( Handle ) + sizeof( leaf_head ); }
    [[nodiscard]] static constexpr std::size_t block_bytes( unsigned const cap ) noexcept { return keys_offset( cap ) + padded( cap ) * sizeof( key_t ); }
    // A full leaf's keys sit on a cache line: its block starts a few bytes into a 64-byte aligned allocation.
    static constexpr std::size_t full_leaf_slack{ 8 };
    static constexpr std::size_t full_leaf_alloc{ 640 };
    static_assert( keys_offset( leaf_capacity ) + full_leaf_slack == 512 );
    static_assert( block_bytes( leaf_capacity ) + full_leaf_slack == full_leaf_alloc );

    [[nodiscard]] static std::byte * bytes( leaf_ptr const leaf ) noexcept { return reinterpret_cast<std::byte *>( leaf ); }
    [[nodiscard]] static leaf_head & head( leaf_ptr const leaf ) noexcept { return *reinterpret_cast<leaf_head *>( bytes( leaf ) - sizeof( leaf_head ) ); }
    [[nodiscard]] static Handle * slot_base( leaf_ptr const leaf ) noexcept { return reinterpret_cast<Handle *>( bytes( leaf ) - sizeof( leaf_head ) ); }
    [[nodiscard]] static Handle & slot_at( leaf_ptr const leaf, unsigned const index ) noexcept { return *( slot_base( leaf ) - 1 - index ); }
    // Full leaves only: [ 0 ] is the next leaf, [ 1 ] the previous one.
    [[nodiscard]] static leaf_ptr * links( leaf_ptr const leaf ) noexcept {
        return reinterpret_cast<leaf_ptr *>( bytes( leaf ) - sizeof( leaf_head ) - leaf_capacity * sizeof( Handle ) - 2 * sizeof( leaf_ptr ) );
    }
    [[nodiscard]] static leaf_ptr next_leaf( leaf_ptr const leaf ) noexcept { return head( leaf ).cap == leaf_capacity ? links( leaf )[ 0 ] : nullptr; }
    [[nodiscard]] static leaf_ptr prev_leaf( leaf_ptr const leaf ) noexcept { return head( leaf ).cap == leaf_capacity ? links( leaf )[ 1 ] : nullptr; }

public:
    // A position in key order: a leaf and the index in it. end() is the null leaf.
    class cursor {
    public:
        constexpr cursor() noexcept = default;
        cursor & operator++() noexcept {
            assert( leaf_ != nullptr );
            if ( ++index_ == head( leaf_ ).size ) {
                leaf_  = next_leaf( leaf_ );
                index_ = 0;
                // an open_back() that was never committed leaves an empty last leaf behind
                if ( leaf_ != nullptr && head( leaf_ ).size == 0 ) [[unlikely]] { leaf_ = nullptr; }
            }
            return *this;
        }
        [[nodiscard]] friend constexpr bool operator==( cursor const lhs, cursor const rhs ) noexcept {
            return lhs.leaf_ == rhs.leaf_ && lhs.index_ == rhs.index_;
        }

    private:
        friend class chunk_tree;
        constexpr cursor( leaf_ptr const leaf, unsigned const index ) noexcept : leaf_{ leaf }, index_{ index } {}
        leaf_ptr leaf_ { nullptr };
        unsigned index_{ 0 };
    };

    chunk_tree() noexcept = default;
    chunk_tree( chunk_tree && other ) noexcept { steal( other ); }
    chunk_tree & operator=( chunk_tree && other ) noexcept {
        if ( this != &other ) [[likely]] {
            release();
            steal( other );
        }
        return *this;
    }
    chunk_tree( chunk_tree const & ) = delete;
    chunk_tree & operator=( chunk_tree const & ) = delete;
    ~chunk_tree() noexcept { release(); }

    [[nodiscard]] std::uint32_t size () const noexcept { return size_; }
    [[nodiscard]] bool          empty() const noexcept { return size_ == 0; }
    // Whether the flat directory stands in for the inner levels, and the bytes its table holds.
    [[nodiscard]] bool          indexed() const noexcept { return dir_ != nullptr; }
    [[nodiscard]] std::size_t   directory_bytes() const noexcept { return dir_buckets_ * sizeof( leaf_ptr ); }

    [[nodiscard]] cursor begin() const noexcept { return size_ == 0 ? cursor{} : cursor{ leftmost_leaf(), 0 }; }
    [[nodiscard]] cursor end  () const noexcept { return cursor{}; }
    // The final entry. The tree must not be empty.
    [[nodiscard]] cursor last () const noexcept {
        assert( size_ != 0 );
        auto leaf{ tail_ };
        if ( head( leaf ).size == 0 ) { leaf = prev_leaf( leaf ); }   // the empty leaf an abandoned open_back() leaves
        return cursor{ leaf, head( leaf ).size - 1 };
    }
    // The entry before `at` (end() included), which must not be begin().
    [[nodiscard]] cursor prev( cursor const at ) const noexcept {
        if ( at.leaf_ == nullptr ) { return last(); }
        if ( at.index_ != 0 ) { return cursor{ at.leaf_, at.index_ - 1 }; }
        auto const before{ prev_leaf( at.leaf_ ) };
        assert( before != nullptr );
        return cursor{ before, head( before ).size - 1 };
    }

    [[nodiscard]] key_type key( cursor const at ) const noexcept {
        assert( at.leaf_ != nullptr );
        return static_cast<key_type>( at.leaf_[ at.index_ ] );
    }
    [[nodiscard]] Handle       & slot( cursor const at )       noexcept { assert( at.leaf_ != nullptr ); return slot_at( at.leaf_, at.index_ ); }
    [[nodiscard]] Handle const & slot( cursor const at ) const noexcept { assert( at.leaf_ != nullptr ); return slot_at( at.leaf_, at.index_ ); }

    // The first entry whose key is not below `key` (end() if there is none).
    [[nodiscard]] cursor lower_bound( key_type const key ) const noexcept {
        if ( root_ == nullptr ) { return end(); }
        auto const probe{ static_cast<key_t>( key ) };
        if ( dir_ != nullptr ) {
            auto leaf{ search_start( probe ) };
            while ( leaf != nullptr ) {
                auto const index{ count_below<16>( leaf, probe ) };
                if ( leaf[ index ] != pad ) { return cursor{ leaf, index }; }
                leaf = links( leaf )[ 0 ];
                if ( leaf != nullptr && head( leaf ).size == 0 ) [[unlikely]] { break; }   // the empty last leaf of an abandoned open_back()
            }
            return end();
        }
        return seek_in_leaf( descend( probe ), probe );
    }

    // The entry holding `key`, or end() if there is none.
    [[nodiscard]] cursor find( key_type const key ) const noexcept {
        auto const at{ lower_bound( key ) };
        return ( at == end() || this->key( at ) != key ) ? end() : at;
    }

    // lower_bound() for a caller that already stands at `from` and only moves forward.
    [[nodiscard]] cursor lower_bound( key_type const key, cursor const from ) const noexcept {
        if ( from.leaf_ == nullptr ) { return end(); }
        auto const probe{ static_cast<key_t>( key ) };
        auto const & hd{ head( from.leaf_ ) };
        if ( from.leaf_[ hd.size - 1 ] >= probe ) {   // the answer is in this leaf
            return cursor{ from.leaf_, scan( from.leaf_, probe, hd.cap ) };
        }
        return lower_bound( key );
    }

    // Inserts before `at`, which is the lower_bound() of `key`, and returns the new entry's position.
    template <typename H>
    cursor insert( cursor const at, key_type const key, H && handle ) {
        stage staged{ std::forward<H>( handle ) };
        return insert_guarded( at, static_cast<key_t>( key ), staged );
    }

    // Inserts a copy of every entry of `other` whose key is not held here and for which `wanted`
    // holds. Each is sought and inserted on its own: a leaf takes an insertion without moving
    // anything but its own tail. `count` (how many that is) has no use here.
    template <typename Wanted>
    void insert_absent_from( chunk_tree const & other, std::uint32_t const /*count*/, Wanted const & wanted ) {
        assert( this != &other );
        auto at{ begin() };
        for ( auto from{ other.begin() }; from != other.end(); ++from ) {
            auto const key{ other.key( from ) };
            at = lower_bound( key, at );
            if ( ( at != end() && this->key( at ) == key ) || !wanted( other.slot( from ) ) ) { continue; }
            at = insert( at, key, other.slot( from ) );
        }
    }

    // Removes the entry at `at` and returns the position of the entry that followed it.
    cursor erase( cursor const at ) {
        if ( dir_ != nullptr && prepare_erase( at ) ) { return erase_indexed( at ); }
        auto const leaf{ at.leaf_ };
        auto const removed{ leaf[ at.index_ ] };
        leaf_remove( leaf, at.index_ );
        --size_;
        auto const left{ head( leaf ).size };
        if ( left != 0 && ( height_ == 0 || left != merge_below - 1 ) ) {
            // the leaf stays; it is only looked at for a merge in the call that makes it underfull
            if ( at.index_ < left ) { return cursor{ leaf, at.index_ }; }
            auto const following{ next_leaf( leaf ) };
            return ( following == nullptr || head( following ).size == 0 ) ? cursor{} : cursor{ following, 0 };
        }
        if ( height_ == 0 ) {                               // the only leaf has emptied
            free_leaf_block( leaf );
            root_ = tail_ = nullptr;
            return cursor{};
        }
        path_t path;
        [[maybe_unused]] auto const found{ descend( removed, path ) };
        assert( found == leaf );
        if ( left == 0 ) {
            auto const following{ next_leaf( leaf ) };
            unlink( leaf );
            remove_child( path, 0 );
            free_leaf_block( leaf );
            return ( following == nullptr || head( following ).size == 0 ) ? cursor{} : cursor{ following, 0 };
        }
        // an underfull leaf joins a neighbour under the same parent when the two fit one leaf comfortably
        auto * const parent{ path.node[ 0 ] };
        auto const   child { path.child[ 0 ] };
        if ( child < parent->n ) {
            auto const right{ static_cast<leaf_ptr>( parent->child[ child + 1 ] ) };
            if ( left + head( right ).size <= merge_limit ) {
                absorb( leaf, right );
                unlink( right );
                path.child[ 0 ] = child + 1;
                remove_child( path, 0 );
                free_leaf_block( right );
            }
            if ( at.index_ < head( leaf ).size ) { return cursor{ leaf, at.index_ }; }
            auto const following{ next_leaf( leaf ) };
            return ( following == nullptr || head( following ).size == 0 ) ? cursor{} : cursor{ following, 0 };
        }
        if ( child > 0 ) {
            auto const before{ static_cast<leaf_ptr>( parent->child[ child - 1 ] ) };
            if ( head( before ).size + left <= merge_limit ) {
                auto const offset{ head( before ).size + at.index_ };
                absorb( before, leaf );
                unlink( leaf );
                remove_child( path, 0 );
                free_leaf_block( leaf );
                if ( offset < head( before ).size ) { return cursor{ before, offset }; }
                auto const following{ next_leaf( before ) };
                return ( following == nullptr || head( following ).size == 0 ) ? cursor{} : cursor{ following, 0 };
            }
        }
        if ( at.index_ < left ) { return cursor{ leaf, at.index_ }; }
        auto const following{ next_leaf( leaf ) };
        return ( following == nullptr || head( following ).size == 0 ) ? cursor{} : cursor{ following, 0 };
    }

    // Appends an entry whose key is above every key held. Returns whether the existing entries moved.
    template <typename H>
    bool push_back( key_type const key, H && handle ) {
        assert( static_cast<key_t>( key ) >= 0 && static_cast<key_t>( key ) < pad - 1 );
        if ( tail_ != nullptr ) [[likely]] {
            auto & hd{ head( tail_ ) };
            if ( hd.size < hd.cap ) [[likely]] {
                // room in the last leaf: the handle is built where it stays and nothing moves, so
                // the source may even be an entry of this very tree
                std::construct_at( &slot_at( tail_, hd.size ), std::forward<H>( handle ) );
                tail_[ hd.size++ ] = static_cast<key_t>( key );
                ++size_;
                return false;
            }
        }
        stage staged{ std::forward<H>( handle ) };
        return append_guarded( static_cast<key_t>( key ), staged );
    }

    // Storage for the next appended entry, to be filled with a handle and then committed under its
    // key. An opened slot is not part of the tree until it is committed (the leaf it lies in may
    // be empty and last until then); opening again without committing hands back the same slot.
    // `moved` tells whether the existing entries moved to make room.
    [[nodiscard]] Handle * open_back( bool & moved ) {
        moved = false;
        if ( root_ == nullptr ) {
            root_ = tail_ = make_leaf( 4 );
            return &slot_at( tail_, 0 );
        }
        auto leaf{ tail_ };
        if ( head( leaf ).size == head( leaf ).cap ) {
            if ( height_ == 0 && head( leaf ).cap < leaf_capacity ) {
                leaf  = grow_root_leaf();
                moved = true;
            } else {
                // the last leaf is full: a fresh one is linked behind it and indexed under a
                // separator that is above everything held and not above the key to come (the
                // directory, which has no separators, names the leaf before it instead)
                prepare_new_last_leaf();
                auto const separator{ leaf[ head( leaf ).size - 1 ] + 1 };
                path_t path;
                inner_reserve reserve;
                if ( dir_ == nullptr ) {
                    descend_rightmost( path );
                    reserve.fill( inner_nodes_needed( path ) );
                }
                leaf = make_leaf( leaf_capacity );
                link_after( tail_, leaf );
                if ( dir_ == nullptr ) { insert_in_parent( path, reserve, 0, separator, leaf ); } else { name_leaf( prev_leaf( leaf ) ); }
            }
        }
        return &slot_at( leaf, head( leaf ).size );
    }

    // Makes the handle built in the slot open_back() returned the last entry, under `key`.
    void commit_back( key_type const key ) {
        auto & hd{ head( tail_ ) };
        assert( hd.size < hd.cap );
        tail_[ hd.size++ ] = static_cast<key_t>( key );
        ++size_;
    }

    void destroy_all() noexcept { release(); }
    void swap( chunk_tree & other ) noexcept {
        std::swap( root_, other.root_ );
        std::swap( tail_, other.tail_ );
        std::swap( size_, other.size_ );
        std::swap( height_, other.height_ );
        std::swap( dir_, other.dir_ );
        std::swap( dir_base_, other.dir_base_ );
        std::swap( dir_buckets_, other.dir_buckets_ );
        std::swap( dir_lo_, other.dir_lo_ );
        std::swap( dir_shift_, other.dir_shift_ );
        std::swap( next_eval_, other.next_eval_ );
    }

    // Gives back the spare capacity: a root leaf shrinks to the smallest size that holds its entries.
    void shrink_to_fit() {
        if ( size_ == 0 ) {
            release();
            return;
        }
        if ( head( tail_ ).size == 0 ) {                   // the empty last leaf of an abandoned open_back()
            auto const dropped{ tail_ };
            if ( dir_ != nullptr ) {
                unlink( dropped );                          // the directory never pointed at it
            } else {
                path_t path;
                descend_rightmost( path );
                unlink( dropped );
                remove_child( path, 0 );
            }
            free_leaf_block( dropped );
        }
        if ( height_ != 0 ) { return; }
        auto const leaf{ static_cast<leaf_ptr>( root_ ) };
        auto const cap{ root_capacity_for( head( leaf ).size ) };
        if ( cap < head( leaf ).cap ) {
            root_ = tail_ = relocate_leaf( leaf, cap );
        }
    }

    // Bytes held by the nodes.
    [[nodiscard]] std::size_t node_bytes() const noexcept {
        if ( dir_ != nullptr ) {
            std::size_t total{ dir_buckets_ * sizeof( leaf_ptr ) };
            for ( auto leaf{ first_leaf() }; leaf != nullptr; leaf = next_leaf( leaf ) ) { total += full_leaf_alloc; }
            return total;
        }
        return root_ == nullptr ? 0 : subtree_bytes( root_, height_ );
    }

    // Makes this tree a copy of `other`, the handles copy-constructed one by one. Leaves come out full.
    void copy_from( chunk_tree const & other ) {
        release();
        if ( other.size_ == 0 ) { return; }
        auto const count{ other.size_ };
        if ( count <= leaf_capacity ) {
            auto const leaf{ make_leaf( root_capacity_for( count ) ) };
            unsigned copied{ 0 };
            try {
                for ( auto from{ other.begin() }; !( from == other.end() ); ++from, ++copied ) {
                    std::construct_at( &slot_at( leaf, copied ), slot_at( from.leaf_, from.index_ ) );
                    leaf[ copied ] = from.leaf_[ from.index_ ];
                }
            } catch ( ... ) {
                head( leaf ).size = copied;
                destroy_leaf( leaf );
                throw;
            }
            head( leaf ).size = copied;
            root_ = tail_ = leaf;
            size_ = count;
            return;
        }
        leaf_ptr first{ nullptr };
        leaf_ptr last { nullptr };
        try {
            leaf_ptr to{ nullptr };
            for ( auto from{ other.leftmost_leaf() }; from != nullptr; from = next_leaf( from ) ) {
                auto const available{ head( from ).size };
                for ( unsigned copied{ 0 }; copied != available; ) {
                    if ( to == nullptr || head( to ).size == leaf_capacity ) {
                        auto const fresh{ make_leaf( leaf_capacity ) };
                        if ( first == nullptr ) { first = last = fresh; } else { link_after( last, fresh ); last = fresh; }
                        to = fresh;
                    }
                    auto & hd{ head( to ) };
                    auto const take{ std::min( available - copied, leaf_capacity - hd.size ) };
                    std::memcpy( to + hd.size, from + copied, take * sizeof( key_t ) );
                    for ( unsigned i{ 0 }; i != take; ++i ) {
                        std::construct_at( &slot_at( to, hd.size ), slot_at( from, copied + i ) );
                        ++hd.size;
                    }
                    copied += take;
                }
            }
        } catch ( ... ) {
            destroy_chain( first );
            tail_ = nullptr;
            throw;
        }
        size_ = count;
        tail_ = last;
        try {
            index_chain( first );
        } catch ( ... ) {
            destroy_chain( first );
            size_ = 0;
            tail_ = nullptr;
            throw;
        }
    }

    // The in-place rewrite of the table by the walk that is consuming it, in key order: an entry is
    // kept (moved down behind the walk), replaced, or left out. The rewritten entries are behind the
    // walk, so the leaves are compacted in place; finish() frees the leaves left empty and rebuilds
    // the levels above the leaves. `retire` receives the payload of each entry that goes away.
    template <typename Retire>
    class rewriter {
    public:
        rewriter( chunk_tree & tree, Retire retire ) noexcept
            : tree_{ &tree }, retire_{ std::move( retire ) }, leaf_{ tree.root_ == nullptr ? nullptr : tree.leftmost_leaf() } {}

        // The entry at `from`, which the walk has finished with, is kept.
        void keep( cursor const from ) {
            assert( leaf_ != nullptr );
            if ( !( cursor{ leaf_, index_ } == from ) ) {
                leaf_[ index_ ] = from.leaf_[ from.index_ ];
                store( std::move( slot_at( from.leaf_, from.index_ ) ) );
            }
            advance();
        }

        // The walk's current entry is replaced by (key, container).
        void put( key_type const key, Handle && container ) {
            assert( leaf_ != nullptr );
            leaf_[ index_ ] = static_cast<key_t>( key );
            store( std::move( container ) );
            advance();
        }

        void finish() { tree_->finish_rewrite( leaf_, index_, last_complete_, kept_, retire_ ); }

    private:
        // The write position holds a consumed entry, whose payload is retired, unless the walk
        // has packed entries from later leaves past the end of this one: then it holds nothing yet.
        void store( Handle && container ) {
            auto & target{ slot_at( leaf_, index_ ) };
            if ( index_ < head( leaf_ ).size ) {
                retire_( std::move( target ) );
                target = std::move( container );
            } else {
                std::construct_at( &target, std::move( container ) );
            }
        }
        void advance() noexcept {
            ++kept_;
            if ( ++index_ == head( leaf_ ).cap ) {
                head( leaf_ ).size = index_;
                last_complete_ = leaf_;
                leaf_  = next_leaf( leaf_ );
                index_ = 0;
            }
        }

        chunk_tree *  tree_;
        Retire        retire_;
        leaf_ptr      leaf_;                    // where the next kept entry goes; nullptr past the last leaf
        leaf_ptr      last_complete_{ nullptr };
        unsigned      index_{ 0 };
        std::uint32_t kept_ { 0 };
    };

    // The handles in key order.
    template <bool Const>
    class slot_range {
    public:
        class iterator {
        public:
            using value_type      = Handle;
            using difference_type = std::ptrdiff_t;
            iterator() = default;
            [[nodiscard]] std::conditional_t<Const, Handle const &, Handle &> operator*() const noexcept { return slot_at( at_.leaf_, at_.index_ ); }
            iterator & operator++() noexcept { ++at_; return *this; }
            void operator++( int ) noexcept { ++at_; }
            [[nodiscard]] friend bool operator==( iterator const lhs, iterator const rhs ) noexcept { return lhs.at_ == rhs.at_; }
        private:
            friend class slot_range;
            explicit iterator( cursor const at ) noexcept : at_{ at } {}
            cursor at_{};
        };
        explicit slot_range( chunk_tree const & tree ) noexcept : tree_{ &tree } {}
        [[nodiscard]] iterator begin() const noexcept { return iterator{ tree_->begin() }; }
        [[nodiscard]] iterator end  () const noexcept { return iterator{}; }
    private:
        chunk_tree const * tree_;
    };
    [[nodiscard]] slot_range<false> slots()       noexcept { return slot_range<false>{ *this }; }
    [[nodiscard]] slot_range<true > slots() const noexcept { return slot_range<true >{ *this }; }

    // Checks the structural invariants; the first one that does not hold, or nullptr.
    [[nodiscard]] char const * validate() const noexcept {
        if ( root_ == nullptr ) { return size_ == 0 && height_ == 0 && tail_ == nullptr && dir_ == nullptr ? nullptr : "an empty tree has a size, a height, a tail or a directory"; }
        if ( dir_ != nullptr ) { return validate_indexed(); }
        std::uint32_t counted{ 0 };
        leaf_ptr      previous{ nullptr };
        key_t         floor{ std::numeric_limits<key_t>::min() };
        char const *  problem{ nullptr };
        check_subtree( root_, height_, floor, pad, counted, previous, problem );
        if ( problem != nullptr ) { return problem; }
        if ( counted != size_ ) { return "the size differs from the number of entries"; }
        if ( previous != tail_ ) { return "the tail is not the last leaf"; }
        if ( next_leaf( tail_ ) != nullptr ) { return "the last leaf has a successor"; }
        return nullptr;
    }

private:
    // A handle built outside the tree and then relocated into it, so that a source living inside the
    // tree cannot be disturbed halfway through an insertion that moves entries.
    struct stage {
        alignas( Handle ) std::byte bytes[ sizeof( Handle ) ];
        template <typename H>
        explicit stage( H && handle ) { std::construct_at( reinterpret_cast<Handle *>( bytes ), std::forward<H>( handle ) ); }
        // For an insertion that failed: the staged handle was not relocated into the tree.
        void destroy() noexcept { std::destroy_at( reinterpret_cast<Handle *>( bytes ) ); }
    };
    // The staged handle is destroyed if the insertion throws before relocating it into a leaf.
    [[gnu::noinline]] bool append_guarded( key_t const key, stage & staged ) {
        try {
            return append_staged( key, staged.bytes );
        } catch ( ... ) {
            staged.destroy();
            throw;
        }
    }
    cursor insert_guarded( cursor const at, key_t const key, stage & staged ) {
        try {
            return insert_staged( at, key, staged.bytes );
        } catch ( ... ) {
            staged.destroy();
            throw;
        }
    }

    // A leaf that falls below this many entries is merged into a sibling it fits into.
    static constexpr unsigned merge_below{ 3 };
    static constexpr unsigned merge_limit{ leaf_capacity * 3 / 4 };

    // The last byte of an inner node is padding.
    static constexpr std::size_t inner_shift_at{ sizeof( inner_node ) - 1 };
    static_assert( inner_shift_at >= offsetof( inner_node, n ) + sizeof( std::uint32_t ) );

    // The way down to a leaf: the inner node and the child taken at each level (0 is the lowest).
    static constexpr unsigned max_height{ 12 };
    struct path_t {
        inner_node * node [ max_height ];
        unsigned     child[ max_height ];
    };

    // ---- memory -----------------------------------------------------------------------------
    // A block of `size` bytes. Memory the allocator hands out as it is (16-byte aligned).
    [[nodiscard]] static std::byte * allocate( std::size_t const size ) {
#if FRSR_ROARING_HAS_MIMALLOC
        auto * const block{ static_cast<std::byte *>( mi_malloc( size ) ) };
        if ( block == nullptr ) [[unlikely]] { throw std::bad_alloc{}; }
        return block;
#else
        return static_cast<std::byte *>( ::operator new( size ) );
#endif
    }
    static void deallocate( std::byte * const block ) noexcept {
#if FRSR_ROARING_HAS_MIMALLOC
        mi_free( block );
#else
        ::operator delete( block );
#endif
    }

    // A block of `size` bytes that starts on a cache line. Where the allocator can align, it does.
    // Elsewhere the block is cut out of a plain allocation, which is far cheaper than the aligned
    // form of the global operator new, and the distance back to the start of that allocation is kept
    // in the byte at `shift_at`, which the caller must leave unused.
    [[nodiscard]] static std::byte * allocate_line( std::size_t const size, [[maybe_unused]] std::size_t const shift_at ) {
#if FRSR_ROARING_HAS_MIMALLOC
        auto * const block{ static_cast<std::byte *>( mi_malloc_aligned( size, 64 ) ) };
        if ( block == nullptr ) [[unlikely]] { throw std::bad_alloc{}; }
        return block;
#else
        static_assert( __STDCPP_DEFAULT_NEW_ALIGNMENT__ >= 16 );
        auto * const raw  { static_cast<std::byte *>( ::operator new( size + 64 - __STDCPP_DEFAULT_NEW_ALIGNMENT__ ) ) };
        auto * const block{ reinterpret_cast<std::byte *>( ( reinterpret_cast<std::uintptr_t>( raw ) + 63U ) & ~std::uintptr_t{ 63 } ) };
        block[ shift_at ] = static_cast<std::byte>( block - raw );
        return block;
#endif
    }
    static void deallocate_line( std::byte * const block, [[maybe_unused]] std::size_t const shift_at ) noexcept {
#if FRSR_ROARING_HAS_MIMALLOC
        mi_free( block );
#else
        ::operator delete( block - std::to_integer<unsigned>( block[ shift_at ] ) );
#endif
    }

    // An empty leaf: no entries, all key slots the pad.
    [[nodiscard]] static leaf_ptr make_leaf( unsigned const cap ) {
        auto const full{ cap == leaf_capacity };
        auto * const block{ full ? allocate_line( full_leaf_alloc, 0 ) : allocate( block_bytes( cap ) ) };
        auto const leaf{ reinterpret_cast<leaf_ptr>( block + ( full ? full_leaf_slack : 0 ) + keys_offset( cap ) ) };
        head( leaf ) = leaf_head{ 0, cap };
        std::fill_n( leaf, padded( cap ), pad );
        if ( full ) { links( leaf )[ 0 ] = links( leaf )[ 1 ] = nullptr; }
        return leaf;
    }
    // Gives the block back; the handles must already be gone or moved out.
    static void free_leaf_block( leaf_ptr const leaf ) noexcept {
        auto const cap{ head( leaf ).cap };
        if ( cap == leaf_capacity ) {
            deallocate_line( bytes( leaf ) - ( keys_offset( cap ) + full_leaf_slack ), 0 );
        } else {
            deallocate( bytes( leaf ) - keys_offset( cap ) );
        }
    }
    static void destroy_leaf( leaf_ptr const leaf ) noexcept {
        for ( unsigned i{ 0 }; i < head( leaf ).size; ++i ) { slot_at( leaf, i ).~Handle(); }
        free_leaf_block( leaf );
    }
    [[nodiscard]] static inner_node * make_inner() {
        auto * const node{ reinterpret_cast<inner_node *>( allocate_line( sizeof( inner_node ), inner_shift_at ) ) };
        std::fill_n( node->keys, inner_keys, pad );
        node->n = 0;
        return node;
    }
    static void free_inner( inner_node * const node ) noexcept { deallocate_line( reinterpret_cast<std::byte *>( node ), inner_shift_at ); }

    // The inner nodes an insertion into the tree will need, allocated before it changes anything so
    // that it cannot fail halfway: one for every full node on the way up, and a new root if they all are.
    class inner_reserve {
    public:
        inner_reserve() noexcept = default;
        inner_reserve( inner_reserve const & ) = delete;
        ~inner_reserve() { while ( count_ != 0 ) { free_inner( node_[ --count_ ] ); } }
        void fill( unsigned const wanted ) {
            while ( count_ != wanted ) { node_[ count_ ] = make_inner(); ++count_; }
        }
        [[nodiscard]] inner_node * take() noexcept {
            assert( count_ != 0 );
            return node_[ --count_ ];
        }
    private:
        inner_node * node_[ max_height + 1 ];
        unsigned     count_{ 0 };
    };

    static void release_subtree( void * const node, unsigned const level ) noexcept {
        if ( level == 0 ) {
            destroy_leaf( static_cast<leaf_ptr>( node ) );
            return;
        }
        auto * const inner{ static_cast<inner_node *>( node ) };
        for ( unsigned i{ 0 }; i <= inner->n; ++i ) { release_subtree( inner->child[ i ], level - 1 ); }
        free_inner( inner );
    }
    // Frees the inner nodes of a subtree but leaves its leaves alone.
    static void release_inner( void * const node, unsigned const level ) noexcept {
        if ( level == 0 ) { return; }
        auto * const inner{ static_cast<inner_node *>( node ) };
        for ( unsigned i{ 0 }; i <= inner->n; ++i ) { release_inner( inner->child[ i ], level - 1 ); }
        free_inner( inner );
    }

    // ---- the leaf directory -----------------------------------------------------------------
    // Once a table is large and its keys are spread evenly enough, the inner levels give way to a flat
    // table indexed by the high bits of the key. It is then the only index: the inner levels are freed
    // when it is adopted and rebuilt when it is given up, while the leaves, their chain and the cursors
    // are the same blocks either way. It is held in memory from the same allocator as the nodes.
    //
    // Bucket b covers the keys from `base + (b << shift)` up (bucket 0 reaches down to minus infinity)
    // and holds the first leaf whose last key is not below that start. A search reads the probe's
    // bucket and walks the leaf chain from there. The walk is short because no more than the pile limit
    // of leaves end in one bucket, and a bucket is about half a leaf wide at the average gap between the
    // keys, so a leaf's span is two or three buckets.
    //   - A bucket that names no leaf stands for the last leaf (and for "no such key" when that has
    //     none either): appending to the last leaf, which is what builds a table, never writes the
    //     table. The last leaf is named once a leaf is linked behind it, as a batch for the whole of its
    //     span; the last leaf may also be named earlier, and is by adoption.
    //   - An insert goes to the first leaf whose last key is not below the key, so only a new largest
    //     key and a leaf split can change which leaf a bucket names, and only a split writes the table.
    //   - Buckets below the smallest key are not kept (`dir_lo_`), so erasing at the front never
    //     rewrites a prefix of the table.
    //   - The directory is adopted from the doubling sizes at which the table reaches a leaf boundary
    //     (and straight from the chain by a bulk build), and only if the keys span at most about half as
    //     many buckets as there are entries. It is given up when a leaf split, a new last leaf or an
    //     erase would pile too many leaves into a bucket, when a leaf ends too far above the others for
    //     the table to reach, and, after erases, when it has become much larger than the table needs: it
    //     is then rebuilt at the width that suits the table now, or given up if the table is small or uneven.
    //   - Everything that can allocate (growing, adopting, giving up) is done before the operation
    //     that needs it changes anything, so an allocation failure leaves the table as it was.
    static constexpr std::uint32_t initial_next_eval{ static_cast<std::uint32_t>( std::min<std::uint64_t>( DirectoryMinEntries, std::numeric_limits<std::uint32_t>::max() ) ) };
    static constexpr unsigned dir_pile_limit  { chunk_directory_pile_limit };
    // The shape of the table, measured over dense, scattered and random keys of 5 thousand to a million chunks:
    //   - a bucket is half a leaf's worth of entries wide at the average gap (rounded down to a power of
    //     two). A quarter of a leaf looks a key up about 15% faster up to a hundred thousand chunks but
    //     doubles the table, which is slower at a million; a whole leaf looks up 25-45% slower;
    //   - at most 6 leaves in a bucket (the 3 and 12 limits behave the same on evenly spread keys);
    //   - a directory is adopted for at most size/2 + 8 buckets, and may grow to size/2 + 64 while keys
    //     are appended (a quarter of that budget makes a table of a million chunks give the directory up
    //     for keys the full budget serves), growing by half again each time (a doubling or a quarter
    //     measure the same);
    //   - after a failed adoption the next is tried at twice the size (four times delays a table that
    //     becomes even later by two doublings, for nothing gained).
    static constexpr unsigned dir_adopt_slack { 8 };
    static constexpr unsigned dir_growth_slack{ 64 };
    static constexpr unsigned dir_table_slack { 16 };    // spare buckets of a fresh table; erases shrink a table beyond size + twice this

    [[nodiscard]] static std::uint32_t doubled( std::uint32_t const count ) noexcept {
        return count > std::numeric_limits<std::uint32_t>::max() / 2 ? std::numeric_limits<std::uint32_t>::max() : count * 2;
    }
    [[nodiscard]] static leaf_ptr * allocate_table( std::uint64_t const buckets ) {
        return reinterpret_cast<leaf_ptr *>( allocate( buckets * sizeof( leaf_ptr ) ) );
    }
    static void free_table( leaf_ptr * const table ) noexcept { deallocate( reinterpret_cast<std::byte *>( table ) ); }
    // Frees whichever index stands over the leaves (the root and the height are the caller's to reset).
    void free_index() noexcept {
        if ( dir_ != nullptr ) {
            free_table( dir_ );
            dir_ = nullptr;
            dir_buckets_ = 0;
        } else if ( height_ != 0 ) {
            release_inner( root_, height_ );
        }
    }

    [[nodiscard]] static key_t last_key( leaf_ptr const leaf ) noexcept { return leaf[ head( leaf ).size - 1 ]; }
    // The leaf behind this one, unless it is the empty last leaf an abandoned open_back() leaves.
    [[nodiscard]] static leaf_ptr next_nonempty( leaf_ptr const leaf ) noexcept {
        auto const following{ links( leaf )[ 0 ] };
        return ( following == nullptr || head( following ).size == 0 ) ? nullptr : following;
    }
    [[nodiscard]] leaf_ptr max_leaf() const noexcept { return head( tail_ ).size == 0 ? prev_leaf( tail_ ) : tail_; }
    [[nodiscard]] static bool is_max_leaf( leaf_ptr const leaf ) noexcept { return next_nonempty( leaf ) == nullptr; }
    [[nodiscard]] leaf_ptr first_leaf() const noexcept { return static_cast<leaf_ptr>( root_ ); }

    [[nodiscard]] std::uint64_t bucket_of( key_t const probe ) const noexcept {
        auto const rel{ probe - dir_base_ };
        return static_cast<std::uint64_t>( rel < 0 ? 0 : rel ) >> dir_shift_;
    }
    // The bucket a search for `probe` reads: inside the kept part of the table.
    [[nodiscard]] std::uint32_t bucket_clamped( key_t const probe ) const noexcept {
        return std::max( dir_lo_, static_cast<std::uint32_t>( std::min<std::uint64_t>( bucket_of( probe ), dir_buckets_ - 1 ) ) );
    }
    [[nodiscard]] key_t bucket_start( std::uint64_t const bucket ) const noexcept { return dir_base_ + static_cast<key_t>( bucket << dir_shift_ ); }

    // Where a search for `probe` starts walking the leaf chain.
    [[nodiscard]] leaf_ptr search_start( key_t const probe ) const noexcept {
        auto const leaf{ dir_[ bucket_clamped( probe ) ] };
        return leaf != nullptr ? leaf : tail_;
    }
    // The first leaf whose last key is not below `probe`; the last leaf if there is none.
    [[nodiscard]] leaf_ptr locate_indexed( key_t const probe ) const noexcept {
        auto leaf{ search_start( probe ) };
        for ( ;; ) {
            if ( leaf[ count_below<16>( leaf, probe ) ] != pad ) { return leaf; }
            auto const following{ next_nonempty( leaf ) };
            if ( following == nullptr ) { return leaf; }
            leaf = following;
        }
    }

    // How many leaves end in `bucket` (counted up to one past the pile limit). The table names the
    // first of them and the first leaf that ends above them, so they are the run of leaves between the
    // two, and only their links are read.
    [[nodiscard]] unsigned pile_at( std::uint64_t const bucket ) const noexcept {
        if ( bucket >= dir_buckets_ ) { return 0; }
        auto const above{ bucket + 1 < dir_buckets_ ? dir_[ bucket + 1 ] : nullptr };
        unsigned count{ 0 };
        for ( auto at{ dir_[ bucket ] }; at != nullptr && at != above && count <= dir_pile_limit; at = next_nonempty( at ) ) { ++count; }
        return count;
    }
    // Whether splitting `leaf` could put more leaves than the limit into a bucket: the right half ends
    // where the leaf ends, the left half in some bucket the leaf spans.
    [[nodiscard]] bool split_overflows( leaf_ptr const leaf ) const noexcept {
        auto const first{ bucket_clamped( leaf[ 0 ] ) };
        auto const last { bucket_clamped( last_key( leaf ) ) };
        return pile_at( last ) >= dir_pile_limit || ( first != last && pile_at( first ) >= dir_pile_limit );
    }

    // Grows the table to reach at least `at_least` buckets.
    void grow_table( std::uint64_t const at_least ) {
        auto const grown{ std::max<std::uint64_t>( at_least, dir_buckets_ + dir_buckets_ / 2 ) };
        auto * const table{ allocate_table( grown ) };
        std::memcpy( table, dir_, dir_buckets_ * sizeof( leaf_ptr ) );
        std::fill( table + dir_buckets_, table + grown, nullptr );
        free_table( dir_ );
        dir_         = table;
        dir_buckets_ = static_cast<std::uint32_t>( grown );
    }
    // A leaf that ends in `bucket` is about to be named: the table is made to reach the bucket, or the
    // directory is given up if the keys have drifted so far apart that it would be mostly empty.
    // May throw, changing nothing.
    void reserve_bucket( std::uint64_t const bucket ) {
        if ( bucket < dir_buckets_ ) [[likely]] { return; }
        if ( bucket + 1 > size_ / 2 + dir_growth_slack ) {
            abandon_directory();
        } else {
            grow_table( bucket + 1 );
        }
    }
    // The buckets `leaf` answers that the table does not name yet (the last leaf is not named as it grows).
    void name_leaf( leaf_ptr const leaf ) noexcept {
        for ( auto b{ static_cast<std::int64_t>( bucket_clamped( last_key( leaf ) ) ) }; b >= dir_lo_ && dir_[ b ] == nullptr; --b ) { dir_[ b ] = leaf; }
    }
    // `key` has become the smallest key and is held by `first`, the first leaf.
    void note_new_minimum( leaf_ptr const first, key_t const key ) noexcept {
        auto const bucket{ static_cast<std::uint32_t>( std::min<std::uint64_t>( bucket_of( key ), dir_lo_ ) ) };
        std::fill( dir_ + bucket, dir_ + dir_lo_, first );
        dir_lo_ = bucket;
    }
    // A leaf is about to be linked behind the full last leaf, which the table then has to name: it is
    // adopted first if it is time, and given up if naming the leaf could overfill a bucket.
    void prepare_new_last_leaf() {
        if ( dir_ == nullptr ) { adopt_if_due(); }
        if ( dir_ != nullptr ) {
            auto const last{ bucket_of( last_key( tail_ ) ) };
            if ( last < dir_buckets_ && dir_[ last ] != tail_ && pile_at( last ) >= dir_pile_limit ) {
                abandon_directory();
            } else {
                reserve_bucket( last );
            }
        }
    }
    // A full leaf is about to split: the left half ends no higher than the entry the split moves up,
    // and the table has to reach it and be able to take another leaf in the buckets it spans.
    void prepare_split( leaf_ptr const leaf ) {
        if ( split_overflows( leaf ) ) {
            abandon_directory();
        } else {
            reserve_bucket( bucket_of( leaf[ head( leaf ).size / 2 ] ) );
        }
    }
    // `right` has just been linked behind `left` and holds the upper part of what `left` held: `left`
    // is named for the buckets it spans, and `right` for those between the two ends.
    void split_indexed( leaf_ptr const left, leaf_ptr const right ) noexcept {
        name_leaf( left );
        auto const right_bucket{ std::min<std::uint64_t>( bucket_of( last_key( right ) ), dir_buckets_ - 1 ) };
        for ( auto b{ bucket_clamped( last_key( left ) ) + std::uint64_t{ 1 } }; b <= right_bucket; ++b ) { dir_[ b ] = right; }
    }
    // `leaf`, whose last key was `last`, goes away or is absorbed: whatever named it names `to`.
    void retarget( leaf_ptr const leaf, key_t const last, leaf_ptr const to ) noexcept {
        for ( auto b{ static_cast<std::int64_t>( bucket_clamped( last ) ) }; b >= dir_lo_ && ( dir_[ b ] == leaf || dir_[ b ] == nullptr ); --b ) { dir_[ b ] = to; }
    }
    // The last key of `leaf` fell from `was` to a lower one.
    void last_decreased( leaf_ptr const leaf, key_t const was ) noexcept {
        auto const floor{ bucket_clamped( last_key( leaf ) ) };
        auto const following{ next_nonempty( leaf ) };
        for ( auto b{ bucket_clamped( was ) }; b > floor && ( dir_[ b ] == leaf || dir_[ b ] == nullptr ); --b ) { dir_[ b ] = following; }
    }
    [[nodiscard]] cursor following_cursor( leaf_ptr const leaf ) const noexcept {
        auto const following{ next_nonempty( leaf ) };
        return following == nullptr ? cursor{} : cursor{ following, 0 };
    }

    // Makes the directory fit an erase of the entry at `at`: narrowed or given up if it has become much
    // larger than the table needs, given up if the erase would pile too many leaves into a bucket (when
    // the leaf's last key falls into the bucket of leaves before it). False if it was given up.
    bool prepare_erase( cursor const at ) {
        if ( dir_buckets_ > size_ + 2 * dir_table_slack ) {
            if ( size_ <= DirectoryMinEntries / 2 || !adopt_directory( first_leaf() ) ) { abandon_directory(); }
            if ( dir_ == nullptr ) { return false; }
        }
        auto const leaf{ at.leaf_ };
        auto const n   { head( leaf ).size };
        if ( at.index_ + 1 == n && n > 1 ) {
            auto const now{ bucket_clamped( leaf[ n - 2 ] ) };
            if ( now != bucket_clamped( leaf[ n - 1 ] ) && pile_at( now ) >= dir_pile_limit ) {
                abandon_directory();
                return false;
            }
        }
        return true;
    }

    cursor erase_indexed( cursor const at ) {
        auto const leaf{ at.leaf_ };
        auto const was{ last_key( leaf ) };
        auto const was_last{ at.index_ + 1 == head( leaf ).size };
        leaf_remove( leaf, at.index_ );
        --size_;
        auto const left{ head( leaf ).size };
        auto const was_first{ at.index_ == 0 && prev_leaf( leaf ) == nullptr };
        if ( left == 0 ) {
            auto const following{ next_nonempty( leaf ) };
            auto const raw_next{ links( leaf )[ 0 ] };
            unlink( leaf );
            retarget( leaf, was, following );
            if ( was_first && following != nullptr ) {
                dir_lo_ = std::max( dir_lo_, static_cast<std::uint32_t>( bucket_of( following[ 0 ] ) ) );
                root_   = following;
            }
            free_leaf_block( leaf );
            if ( size_ == 0 ) {
                if ( raw_next != nullptr ) { destroy_leaf( raw_next ); }   // the empty last leaf of an abandoned open_back()
                root_ = nullptr;                                            // no leaf is left to walk
                release();
                return cursor{};
            }
            return following == nullptr ? cursor{} : cursor{ following, 0 };
        }
        if ( was_last ) { last_decreased( leaf, was ); }
        if ( was_first ) { dir_lo_ = std::max( dir_lo_, static_cast<std::uint32_t>( bucket_of( leaf[ 0 ] ) ) ); }
        if ( left == merge_below - 1 ) {
            // an underfull leaf joins the leaf behind it, else the one before it, when the two fit one leaf comfortably
            auto const right{ next_nonempty( leaf ) };
            auto const before{ prev_leaf( leaf ) };
            if ( right != nullptr && left + head( right ).size <= merge_limit ) {
                auto const right_last{ last_key( right ) };
                absorb( leaf, right );
                unlink( right );
                retarget( right, right_last, leaf );
                free_leaf_block( right );
            } else if ( before != nullptr && head( before ).size + left <= merge_limit ) {
                auto const offset{ head( before ).size + at.index_ };
                auto const leaf_last{ last_key( leaf ) };
                absorb( before, leaf );
                unlink( leaf );
                retarget( leaf, leaf_last, before );
                free_leaf_block( leaf );
                return offset < head( before ).size ? cursor{ before, offset } : following_cursor( before );
            }
        }
        return at.index_ < head( leaf ).size ? cursor{ leaf, at.index_ } : following_cursor( leaf );
    }

    // ---- adopting and giving up the directory -----------------------------------------------
    // Called where a leaf is about to be added: adopts the directory when the table has reached the
    // size at which it is next considered. Returns whether a directory now stands. May throw, changing nothing.
    bool adopt_if_due() {
        if ( height_ == 0 || size_ < next_eval_ ) { return false; }
        if ( adopt_directory( leftmost_leaf() ) ) { return true; }
        next_eval_ = doubled( size_ );
        return false;
    }
    // Gives a chain of leaves (all full, `size_` entries, `tail_` set) the index that suits it: the
    // directory if it is time and the keys allow, else inner levels.
    void index_chain( leaf_ptr const first ) {
        if ( size_ >= next_eval_ ) {
            if ( adopt_directory( first ) ) { return; }
            next_eval_ = doubled( size_ );
        }
        rebuild_levels( first );
    }
    // Replaces the index (inner levels, an earlier directory, or nothing) by a directory whose buckets
    // are about half a leaf wide; false, with nothing changed, if the keys are spread too unevenly for
    // that. The new table is built and checked before the old index is touched; allocation may throw.
    bool adopt_directory( leaf_ptr const first ) {
        auto const last { last_key( max_leaf() ) };
        auto const base { first[ 0 ] };
        auto const span { static_cast<std::uint64_t>( last - base ) + 1 };
        auto const width{ std::max<std::uint64_t>( span * leaf_capacity / ( 2 * std::uint64_t{ size_ } ), 1 ) };
        auto const shift{ static_cast<unsigned>( std::bit_width( width ) - 1 ) };
        auto const used { ( ( span - 1 ) >> shift ) + 1 };
        if ( used > size_ / 2 + dir_adopt_slack ) { return false; }
        auto const capacity{ used + used / 8 + dir_table_slack };
        // (the table is written from the front as the leaves are walked, so an attempt that fails on
        // the first crowded bucket has touched only that much of it)
        auto * const table{ allocate_table( capacity ) };
        // one pass over the leaves: each answers the buckets from the one after the previous leaf's last
        // key's bucket up to its own, and no more than the pile limit may end in one bucket
        std::int64_t filled{ -1 };
        unsigned     run   { 0 };
        for ( auto leaf{ first }; leaf != nullptr; leaf = next_nonempty( leaf ) ) {
            auto const bucket{ static_cast<std::int64_t>( static_cast<std::uint64_t>( last_key( leaf ) - base ) >> shift ) };
            run = bucket == filled ? run + 1 : 1;
            if ( run > dir_pile_limit ) {
                free_table( table );
                return false;
            }
            for ( auto b{ filled + 1 }; b <= bucket; ++b ) { table[ b ] = leaf; }
            filled = bucket;
        }
        std::fill( table + filled + 1, table + capacity, nullptr );
        free_index();
        dir_         = table;
        dir_base_    = base;
        dir_buckets_ = static_cast<std::uint32_t>( capacity );
        dir_lo_      = 0;
        dir_shift_   = static_cast<std::uint8_t>( shift );
        root_        = first;
        height_      = 1;
        return true;
    }
    // Back to inner levels over the same leaves. May throw, changing nothing.
    void abandon_directory() {
        rebuild_levels( first_leaf() );
        next_eval_ = doubled( static_cast<std::uint32_t>( std::max<std::uint64_t>( size_, DirectoryMinEntries ) ) );
    }

    [[nodiscard]] char const * validate_indexed() const noexcept {
        if ( height_ != 1 || root_ == nullptr || prev_leaf( first_leaf() ) != nullptr ) { return "a directory's table does not begin at the first leaf"; }
        if ( dir_lo_ != bucket_of( first_leaf()[ 0 ] ) ) { return "the first kept bucket is not the bucket of the smallest key"; }
        std::uint32_t counted{ 0 };
        leaf_ptr      previous{ nullptr };
        for ( auto l{ first_leaf() }; l != nullptr; l = links( l )[ 0 ] ) {
            auto const & hd{ head( l ) };
            if ( hd.cap != leaf_capacity ) { return "a leaf of a directory tree is not full-sized"; }
            if ( hd.size > hd.cap || ( hd.size == 0 && l != tail_ ) ) { return "a leaf is empty or over capacity"; }
            if ( prev_leaf( l ) != previous ) { return "a leaf's back link is wrong"; }
            for ( unsigned i{ 0 }; i < hd.size; ++i ) {
                if ( i != 0 && l[ i - 1 ] >= l[ i ] ) { return "leaf keys are not strictly ascending"; }
            }
            if ( previous != nullptr && head( previous ).size != 0 && hd.size != 0 && last_key( previous ) >= l[ 0 ] ) { return "leaves are not in key order"; }
            for ( unsigned i{ hd.size }; i < padded( hd.cap ); ++i ) {
                if ( l[ i ] != pad ) { return "an unused key slot is not the pad"; }
            }
            counted += hd.size;
            previous = l;
            if ( hd.size != 0 && pile_at( bucket_of( last_key( l ) ) ) > dir_pile_limit ) { return "too many leaves end in one bucket"; }
            if ( hd.size != 0 && l != max_leaf() && bucket_of( last_key( l ) ) >= dir_buckets_ ) { return "a leaf ends beyond the table"; }
        }
        if ( counted != size_ ) { return "the size differs from the number of entries"; }
        if ( previous != tail_ ) { return "the tail is not the last leaf"; }
        // each bucket names the first leaf that reaches it; those the last leaf answers may be left unnamed
        auto expected{ first_leaf() };
        for ( std::uint64_t b{ dir_lo_ }; b != dir_buckets_; ++b ) {
            while ( expected != nullptr && b != dir_lo_ && last_key( expected ) < bucket_start( b ) ) { expected = next_nonempty( expected ); }
            if ( dir_[ b ] != expected && !( dir_[ b ] == nullptr && expected == max_leaf() ) ) { return "a directory bucket does not hold the first leaf that reaches it"; }
        }
        return nullptr;
    }

    static void destroy_chain( leaf_ptr leaf ) noexcept {
        while ( leaf != nullptr ) {
            auto const following{ next_leaf( leaf ) };
            destroy_leaf( leaf );
            leaf = following;
        }
    }
    void release() noexcept {
        if ( dir_ != nullptr ) {
            destroy_chain( first_leaf() );
            free_index();
        } else if ( root_ != nullptr ) {
            release_subtree( root_, height_ );
        }
        next_eval_ = initial_next_eval;
        root_      = nullptr;
        tail_      = nullptr;
        size_      = 0;
        height_    = 0;
    }
    void steal( chunk_tree & other ) noexcept {
        root_            = std::exchange( other.root_, nullptr );
        tail_            = std::exchange( other.tail_, nullptr );
        size_            = std::exchange( other.size_, 0 );
        height_          = std::exchange( other.height_, 0 );
        dir_             = std::exchange( other.dir_, nullptr );
        dir_base_        = other.dir_base_;
        dir_buckets_     = std::exchange( other.dir_buckets_, 0 );
        dir_lo_          = other.dir_lo_;
        dir_shift_       = other.dir_shift_;
        next_eval_       = std::exchange( other.next_eval_, initial_next_eval );
    }
    [[nodiscard]] static std::size_t subtree_bytes( void const * const node, unsigned const level ) noexcept {
        if ( level == 0 ) {
            auto const cap{ head( static_cast<leaf_ptr>( const_cast<void *>( node ) ) ).cap };
            return cap == leaf_capacity ? full_leaf_alloc : block_bytes( cap );
        }
        auto const * const inner{ static_cast<inner_node const *>( node ) };
        std::size_t total{ sizeof( inner_node ) };
        for ( unsigned i{ 0 }; i <= inner->n; ++i ) { total += subtree_bytes( inner->child[ i ], level - 1 ); }
        return total;
    }

    // ---- navigation -------------------------------------------------------------------------
    [[nodiscard]] leaf_ptr leftmost_leaf() const noexcept {
        if ( dir_ != nullptr ) { return first_leaf(); }
        auto node{ root_ };
        for ( unsigned level{ height_ }; level != 0; --level ) { node = static_cast<inner_node const *>( node )->child[ 0 ]; }
        return static_cast<leaf_ptr>( node );
    }
    // The position of `probe` in a leaf.
    [[nodiscard]] static unsigned scan( leaf_ptr const leaf, key_t const probe, unsigned const cap ) noexcept {
        switch ( cap ) {   // a leaf has one of three capacities, each a fixed-width scan
            case 4 : return count_below< 4>( leaf, probe );
            case 8 : return count_below< 8>( leaf, probe );
            default: assert( cap == leaf_capacity ); return count_below<16>( leaf, probe );
        }
    }
    // The leaf `probe` belongs to. A pure read.
    [[nodiscard]] leaf_ptr descend( key_t const probe ) const noexcept {
        auto node{ root_ };
        for ( unsigned level{ height_ }; level != 0; --level ) {
            auto const * const inner{ static_cast<inner_node const *>( node ) };
            node = inner->child[ count_below<inner_keys>( inner->keys, probe + 1 ) ];   // the separators <= probe
        }
        return static_cast<leaf_ptr>( node );
    }
    [[nodiscard]] leaf_ptr descend( key_t const probe, path_t & path ) const noexcept {
        auto node{ root_ };
        for ( unsigned level{ height_ }; level != 0; --level ) {
            auto * const inner{ static_cast<inner_node *>( node ) };
            auto const child{ count_below<inner_keys>( inner->keys, probe + 1 ) };
            path.node [ level - 1 ] = inner;
            path.child[ level - 1 ] = child;
            node = inner->child[ child ];
        }
        return static_cast<leaf_ptr>( node );
    }
    [[nodiscard]] leaf_ptr descend_rightmost( path_t & path ) const noexcept {
        auto node{ root_ };
        for ( unsigned level{ height_ }; level != 0; --level ) {
            auto * const inner{ static_cast<inner_node *>( node ) };
            path.node [ level - 1 ] = inner;
            path.child[ level - 1 ] = inner->n;
            node = inner->child[ inner->n ];
        }
        return static_cast<leaf_ptr>( node );
    }
    [[nodiscard]] cursor seek_in_leaf( leaf_ptr leaf, key_t const probe ) const noexcept {
        if ( height_ != 0 ) [[likely]] {
            // A leaf below the root is full-sized: it ends in a pad, so the position is past the
            // last entry exactly when the key found there is the pad, and the head is not read.
            auto const index{ count_below<16>( leaf, probe ) };
            if ( leaf[ index ] != pad ) [[likely]] { return cursor{ leaf, index }; }
            leaf = links( leaf )[ 0 ];
            return ( leaf == nullptr || head( leaf ).size == 0 ) ? cursor{} : cursor{ leaf, 0 };
        }
        auto const & hd{ head( leaf ) };
        auto const index{ scan( leaf, probe, hd.cap ) };
        return index < hd.size ? cursor{ leaf, index } : cursor{};
    }

    // ---- leaf operations --------------------------------------------------------------------
    [[nodiscard]] static unsigned root_capacity_for( std::uint32_t const count ) noexcept {
        return count <= 4 ? 4U : count <= 8 ? 8U : leaf_capacity;
    }

    // Puts the staged handle and `key` at `index` of a leaf that has room.
    static void leaf_put( leaf_ptr const leaf, unsigned const index, key_t const key, std::byte const * const staged ) noexcept {
        auto & hd{ head( leaf ) };
        auto const n{ hd.size };
        assert( n < hd.cap && index <= n );
        std::memmove( leaf + index + 1, leaf + index, ( n - index ) * sizeof( key_t ) );
        leaf[ index ] = key;
        // the handles from `index` on move one place down in memory (the last one is first)
        std::memmove( static_cast<void *>( slot_base( leaf ) - n - 1 ), static_cast<void const *>( slot_base( leaf ) - n ), ( n - index ) * sizeof( Handle ) );
        std::memcpy( static_cast<void *>( &slot_at( leaf, index ) ), staged, sizeof( Handle ) );
        ++hd.size;
    }
    // Puts the staged handle and `key` behind the last entry of a leaf that has room.
    static void leaf_append( leaf_ptr const leaf, key_t const key, std::byte const * const staged ) noexcept {
        auto & hd{ head( leaf ) };
        assert( hd.size < hd.cap );
        leaf[ hd.size ] = key;
        std::memcpy( static_cast<void *>( &slot_at( leaf, hd.size ) ), staged, sizeof( Handle ) );
        ++hd.size;
    }
    // Removes the entry at `index` of a leaf, destroying its handle.
    static void leaf_remove( leaf_ptr const leaf, unsigned const index ) noexcept {
        auto & hd{ head( leaf ) };
        auto const n{ hd.size };
        assert( index < n );
        slot_at( leaf, index ).~Handle();
        std::memmove( leaf + index, leaf + index + 1, ( n - index - 1 ) * sizeof( key_t ) );
        leaf[ n - 1 ] = pad;
        std::memmove( static_cast<void *>( slot_base( leaf ) - n + 1 ), static_cast<void const *>( slot_base( leaf ) - n ), ( n - index - 1 ) * sizeof( Handle ) );
        --hd.size;
    }
    // Appends the entries of `src` to `dst`, which must have room for them; `src` is left without entries.
    static void absorb( leaf_ptr const dst, leaf_ptr const src ) noexcept {
        auto & dh{ head( dst ) };
        auto & sh{ head( src ) };
        assert( dh.size + sh.size <= dh.cap );
        std::memcpy( dst + dh.size, src, sh.size * sizeof( key_t ) );
        std::memcpy( static_cast<void *>( slot_base( dst ) - dh.size - sh.size ), static_cast<void const *>( slot_base( src ) - sh.size ), sh.size * sizeof( Handle ) );
        dh.size += sh.size;
        sh.size = 0;
    }
    // Moves the entries of a leaf into a fresh leaf of capacity `cap` and frees the old one.
    [[nodiscard]] static leaf_ptr relocate_leaf( leaf_ptr const leaf, unsigned const cap ) {
        auto const fresh{ make_leaf( cap ) };
        auto const n{ head( leaf ).size };
        std::memcpy( fresh, leaf, n * sizeof( key_t ) );
        std::memcpy( static_cast<void *>( slot_base( fresh ) - n ), static_cast<void const *>( slot_base( leaf ) - n ), n * sizeof( Handle ) );
        head( fresh ).size = n;
        free_leaf_block( leaf );
        return fresh;
    }
    [[nodiscard]] leaf_ptr grow_root_leaf() {
        assert( height_ == 0 );
        auto const old{ static_cast<leaf_ptr>( root_ ) };
        auto const cap{ head( old ).cap };
        assert( cap < leaf_capacity );
        auto const fresh{ relocate_leaf( old, cap < 4 ? 4U : cap < 8 ? 8U : leaf_capacity ) };
        root_ = tail_ = fresh;
        return fresh;
    }
    // Copy-constructs `count` entries starting at `from` into an empty leaf; returns the position after them.
    static cursor fill_from( leaf_ptr const leaf, cursor from, std::uint32_t const count ) {
        auto & hd{ head( leaf ) };
        for ( std::uint32_t i{ 0 }; i != count; ++i, ++from ) {
            std::construct_at( &slot_at( leaf, hd.size ), slot_at( from.leaf_, from.index_ ) );
            leaf[ hd.size ] = from.leaf_[ from.index_ ];
            ++hd.size;
        }
        return from;
    }
    // Links a full leaf behind another full leaf.
    void link_after( leaf_ptr const leaf, leaf_ptr const fresh ) noexcept {
        auto * const lk{ links( leaf ) };
        auto * const fk{ links( fresh ) };
        fk[ 1 ] = leaf;
        fk[ 0 ] = lk[ 0 ];
        if ( lk[ 0 ] != nullptr ) { links( lk[ 0 ] )[ 1 ] = fresh; } else { tail_ = fresh; }
        lk[ 0 ] = fresh;
    }
    void unlink( leaf_ptr const leaf ) noexcept {
        auto * const lk{ links( leaf ) };
        if ( lk[ 1 ] != nullptr ) { links( lk[ 1 ] )[ 0 ] = lk[ 0 ]; }
        if ( lk[ 0 ] != nullptr ) { links( lk[ 0 ] )[ 1 ] = lk[ 1 ]; } else { tail_ = lk[ 1 ]; }
    }

    // ---- structure --------------------------------------------------------------------------
    cursor insert_staged( cursor const at, key_t const key, std::byte const * const staged ) {
        assert( key >= 0 && key < pad - 1 );
        if ( at.leaf_ == nullptr ) {                    // past every key: an append
            append_staged( key, staged );
            return last();
        }
        if ( at.index_ != 0 || prev_leaf( at.leaf_ ) == nullptr ) {   // between two keys of one leaf, or below every key: it belongs here
            auto const & hd{ head( at.leaf_ ) };
            if ( hd.size < hd.cap ) {
                leaf_put( at.leaf_, at.index_, key, staged );
                ++size_;
                if ( dir_ != nullptr && at.index_ == 0 ) { note_new_minimum( at.leaf_, key ); }
                return at;
            }
        }
        path_t path;
        auto leaf{ dir_ != nullptr ? locate_indexed( key ) : descend( key, path ) };
        auto const index{ scan( leaf, key, head( leaf ).cap ) };
        auto & hd{ head( leaf ) };
        if ( hd.size < hd.cap ) {
            leaf_put( leaf, index, key, staged );
            ++size_;
            // (a key above every other one never gets here: it is an append)
            if ( dir_ != nullptr && index == 0 && prev_leaf( leaf ) == nullptr ) { note_new_minimum( leaf, key ); }
            return cursor{ leaf, index };
        }
        if ( height_ == 0 && hd.cap < leaf_capacity ) {
            leaf = grow_root_leaf();
            leaf_put( leaf, index, key, staged );
            ++size_;
            return cursor{ leaf, index };
        }
        // a full leaf splits
        if ( dir_ != nullptr ) {
            prepare_split( leaf );
            if ( dir_ == nullptr ) { return insert_staged( at, key, staged ); }
        } else if ( adopt_if_due() ) {
            return insert_staged( at, key, staged );
        }
        inner_reserve reserve;
        if ( dir_ == nullptr ) { reserve.fill( inner_nodes_needed( path ) ); }
        auto const right{ make_leaf( leaf_capacity ) };
        auto const n{ hd.size };
        cursor result;
        key_t  separator;
        if ( index == n && next_leaf( leaf ) == nullptr ) {
            // past the end of the last leaf: the full leaf stays full and the key starts the next one
            leaf_put( right, 0, key, staged );
            separator = key;
            result = cursor{ right, 0 };
        } else {
            // A new smallest key in the full first leaf moves the whole leaf up and starts afresh in this
            // one, as a key past the last one starts a new leaf: a table built downward fills its leaves.
            auto const kept { ( index == 0 && prev_leaf( leaf ) == nullptr ) ? 0U : n / 2 };
            auto const moved{ n - kept };
            std::memcpy( right, leaf + kept, moved * sizeof( key_t ) );
            std::fill( leaf + kept, leaf + n, pad );
            std::memcpy( static_cast<void *>( slot_base( right ) - moved ), static_cast<void const *>( slot_base( leaf ) - n ), moved * sizeof( Handle ) );
            head( leaf  ).size = kept;
            head( right ).size = moved;
            if ( index <= kept ) {
                leaf_put( leaf, index, key, staged );
                result = cursor{ leaf, index };
            } else {
                leaf_put( right, index - kept, key, staged );
                result = cursor{ right, index - kept };
            }
            separator = right[ 0 ];
        }
        link_after( leaf, right );
        ++size_;
        if ( dir_ == nullptr ) {
            insert_in_parent( path, reserve, 0, separator, right );
        } else {
            split_indexed( leaf, right );
        }
        if ( dir_ != nullptr && result.index_ == 0 && prev_leaf( result.leaf_ ) == nullptr ) { note_new_minimum( result.leaf_, key ); }
        return result;
    }

    bool append_staged( key_t const key, std::byte const * const staged ) {
        assert( key >= 0 && key < pad - 1 );
        if ( root_ == nullptr ) {
            auto const leaf{ make_leaf( 4 ) };
            leaf_append( leaf, key, staged );
            root_ = tail_ = leaf;
            size_ = 1;
            return false;
        }
        auto leaf{ tail_ };
        auto const & hd{ head( leaf ) };
        if ( hd.size < hd.cap ) {
            leaf_append( leaf, key, staged );
            ++size_;
            return false;
        }
        if ( height_ == 0 && hd.cap < leaf_capacity ) {
            leaf = grow_root_leaf();
            leaf_append( leaf, key, staged );
            ++size_;
            return true;
        }
        prepare_new_last_leaf();
        path_t path;
        inner_reserve reserve;
        if ( dir_ == nullptr ) {
            descend_rightmost( path );
            reserve.fill( inner_nodes_needed( path ) );
        }
        auto const fresh{ make_leaf( leaf_capacity ) };
        leaf_append( fresh, key, staged );
        link_after( leaf, fresh );
        ++size_;
        if ( dir_ == nullptr ) { insert_in_parent( path, reserve, 0, key, fresh ); } else { name_leaf( leaf ); }
        return false;
    }

    [[nodiscard]] unsigned inner_nodes_needed( path_t const & path ) const noexcept {
        unsigned level{ 0 };
        while ( level < height_ && path.node[ level ]->n == inner_keys ) { ++level; }
        return level + ( level == height_ ? 1U : 0U );
    }

    // Registers `right` as the sibling that follows the node taken at `level` of `path`, separated by `separator`.
    void insert_in_parent( path_t & path, inner_reserve & reserve, unsigned const level, key_t const separator, void * const right ) noexcept {
        if ( level == height_ ) {                         // the root itself split: a new root above it
            auto const root{ reserve.take() };
            root->keys[ 0 ] = separator;
            root->child[ 0 ] = root_;
            root->child[ 1 ] = right;
            root->n = 1;
            root_ = root;
            ++height_;
            return;
        }
        auto * const parent{ path.node[ level ] };
        auto const   child { path.child[ level ] };
        if ( parent->n < inner_keys ) {
            std::memmove( parent->keys + child + 1, parent->keys + child, ( parent->n - child ) * sizeof( key_t ) );
            std::memmove( parent->child + child + 2, parent->child + child + 1, ( parent->n - child ) * sizeof( void * ) );
            parent->keys[ child ] = separator;
            parent->child[ child + 1 ] = right;
            ++parent->n;
            return;
        }
        if ( child == inner_keys ) {
            // past the last child of a full node: it stays full and the new child starts the next node
            auto const fresh{ reserve.take() };
            fresh->child[ 0 ] = right;
            insert_in_parent( path, reserve, level + 1, separator, fresh );
            return;
        }
        key_t  all_keys    [ inner_keys + 1 ];
        void * all_children[ inner_keys + 2 ];
        std::memcpy( all_keys, parent->keys, child * sizeof( key_t ) );
        all_keys[ child ] = separator;
        std::memcpy( all_keys + child + 1, parent->keys + child, ( inner_keys - child ) * sizeof( key_t ) );
        std::memcpy( all_children, parent->child, ( child + 1 ) * sizeof( void * ) );
        all_children[ child + 1 ] = right;
        std::memcpy( all_children + child + 2, parent->child + child + 1, ( inner_keys - child ) * sizeof( void * ) );
        constexpr unsigned kept{ ( inner_keys + 1 ) / 2 };      // all_keys[ 0, kept ) stay, all_keys[ kept ] moves up
        auto const sibling{ reserve.take() };
        std::memcpy( parent->keys, all_keys, kept * sizeof( key_t ) );
        std::fill( parent->keys + kept, parent->keys + inner_keys, pad );
        std::memcpy( parent->child, all_children, ( kept + 1 ) * sizeof( void * ) );
        parent->n = kept;
        constexpr unsigned moved{ inner_keys - kept };
        std::memcpy( sibling->keys, all_keys + kept + 1, moved * sizeof( key_t ) );
        std::memcpy( sibling->child, all_children + kept + 1, ( moved + 1 ) * sizeof( void * ) );
        sibling->n = moved;
        insert_in_parent( path, reserve, level + 1, all_keys[ kept ], sibling );
    }

    // Removes the child taken at `level` of `path`, whose node has already been unlinked and is about to be freed.
    void remove_child( path_t & path, unsigned const level ) {
        auto * const parent{ path.node[ level ] };
        auto const   child { path.child[ level ] };
        if ( parent->n == 0 ) {                           // its only child: the node goes with it
            if ( level + 1 == height_ ) {
                root_ = nullptr;
                height_ = 0;
            } else {
                remove_child( path, level + 1 );
            }
            free_inner( parent );
            return;
        }
        auto const separator{ child == 0 ? 0U : child - 1 };
        std::memmove( parent->keys + separator, parent->keys + separator + 1, ( parent->n - separator - 1 ) * sizeof( key_t ) );
        std::memmove( parent->child + child, parent->child + child + 1, ( parent->n - child ) * sizeof( void * ) );
        parent->keys[ --parent->n ] = pad;
        if ( parent == root_ && parent->n == 0 ) {        // a root with one child is no root
            root_ = parent->child[ 0 ];
            --height_;
            free_inner( parent );
        }
    }

    // An underfull leaf joins a neighbour under the same parent when the two fit one leaf comfortably.
    void merge_with_sibling( path_t & path, leaf_ptr const leaf ) {
        auto * const parent{ path.node[ 0 ] };
        auto const   child { path.child[ 0 ] };
        auto const   size  { head( leaf ).size };
        if ( child < parent->n ) {
            auto const right{ static_cast<leaf_ptr>( parent->child[ child + 1 ] ) };
            if ( size + head( right ).size <= merge_limit ) {
                absorb( leaf, right );
                unlink( right );
                path.child[ 0 ] = child + 1;
                remove_child( path, 0 );
                free_leaf_block( right );
            }
        } else if ( child > 0 ) {
            auto const left{ static_cast<leaf_ptr>( parent->child[ child - 1 ] ) };
            if ( head( left ).size + size <= merge_limit ) {
                absorb( left, leaf );
                unlink( leaf );
                remove_child( path, 0 );
                free_leaf_block( leaf );
            }
        }
    }

    // Builds the inner levels above a chain of full leaves, from `first`, and makes them the tree's
    // index in place of whichever one stands. The nodes are all allocated before anything is changed,
    // so an allocation failure leaves the table as it was.
    void rebuild_levels( leaf_ptr const first ) {
        struct item { void * node; key_t lowest; };
        std::uint32_t count{ 0 };
        for ( auto leaf{ first }; leaf != nullptr; leaf = next_leaf( leaf ) ) { ++count; }
        if ( count <= 1 ) {
            free_index();
            height_ = 0;
            root_   = first;
            return;
        }
        std::uint32_t nodes{ 0 };
        for ( auto level{ count }; level > 1; ) {
            level  = ( level + inner_keys ) / ( inner_keys + 1 );
            nodes += level;
        }
        auto const pool{ std::make_unique_for_overwrite<inner_node *[]>( nodes ) };
        std::uint32_t made{ 0 };
        try {
            for ( ; made != nodes; ++made ) { pool[ made ] = make_inner(); }
        } catch ( ... ) {
            while ( made != 0 ) { free_inner( pool[ --made ] ); }
            throw;
        }
        auto const items{ std::make_unique_for_overwrite<item[]>( count ) };
        {
            std::uint32_t i{ 0 };
            key_t previous_last{ 0 };
            for ( auto leaf{ first }; leaf != nullptr; leaf = next_leaf( leaf ) ) {
                // the empty last leaf of an abandoned open_back() is indexed under a key above everything held
                items[ i++ ] = item{ leaf, head( leaf ).size == 0 ? previous_last + 1 : leaf[ 0 ] };
                if ( head( leaf ).size != 0 ) { previous_last = last_key( leaf ); }
            }
        }
        unsigned height{ 0 };
        for ( std::uint32_t next{ 0 }; count > 1; ) {
            auto const groups{ ( count + inner_keys ) / ( inner_keys + 1 ) };      // children per node: at most 17
            auto const share { count / groups };
            auto const extra { count % groups };
            std::uint32_t from{ 0 };
            for ( std::uint32_t group{ 0 }; group != groups; ++group ) {
                auto const children{ share + ( group < extra ? 1U : 0U ) };
                auto const node{ pool[ next++ ] };
                for ( std::uint32_t j{ 0 }; j != children; ++j ) {
                    node->child[ j ] = items[ from + j ].node;
                    if ( j != 0 ) { node->keys[ j - 1 ] = items[ from + j ].lowest; }
                }
                node->n = children - 1;
                items[ group ] = item{ node, items[ from ].lowest };
                from += children;
            }
            count = groups;
            ++height;
        }
        free_index();
        height_ = static_cast<std::uint8_t>( height );
        root_   = items[ 0 ].node;
    }

    template <typename Retire>
    void finish_rewrite( leaf_ptr const write_leaf, unsigned const write_index, leaf_ptr const last_complete, std::uint32_t const kept, Retire & retire ) {
        if ( root_ == nullptr ) { return; }
        auto const first{ leftmost_leaf() };
        leaf_ptr last_kept{ last_complete };
        leaf_ptr dead     { nullptr };
        if ( write_leaf != nullptr ) {
            auto const written{ write_index };
            if ( written != 0 ) {
                for ( auto i{ written }; i < head( write_leaf ).size; ++i ) {
                    retire( std::move( slot_at( write_leaf, i ) ) );
                    slot_at( write_leaf, i ).~Handle();
                }
                std::fill( write_leaf + written, write_leaf + padded( head( write_leaf ).cap ), pad );
                head( write_leaf ).size = written;
                last_kept = write_leaf;
                dead = next_leaf( write_leaf );
            } else {
                dead = write_leaf;
            }
        }
        for ( auto leaf{ dead }; leaf != nullptr; ) {
            auto const following{ next_leaf( leaf ) };
            for ( unsigned i{ 0 }; i < head( leaf ).size; ++i ) { retire( std::move( slot_at( leaf, i ) ) ); }
            destroy_leaf( leaf );
            leaf = following;
        }
        if ( last_kept == nullptr ) {
            free_index();
            root_ = tail_ = nullptr;
            height_ = 0;
            size_ = 0;
            return;
        }
        if ( head( last_kept ).cap == leaf_capacity ) { links( last_kept )[ 0 ] = nullptr; }
        size_ = kept;
        tail_ = last_kept;
        if ( next_leaf( first ) != nullptr ) { index_chain( first ); } else { rebuild_levels( first ); }
    }

    // ---- invariants -------------------------------------------------------------------------
    // Checks the subtree at `level` whose keys must lie in [ floor, ceiling ).
    void check_subtree( void const * const node, unsigned const level, key_t const floor, key_t const ceiling,
                        std::uint32_t & counted, leaf_ptr & previous, char const * & problem ) const noexcept {
        if ( problem != nullptr ) { return; }
        if ( level == 0 ) {
            auto const leaf{ static_cast<leaf_ptr>( const_cast<void *>( node ) ) };
            auto const & hd{ head( leaf ) };
            if ( hd.size > hd.cap || ( hd.size == 0 && leaf != tail_ ) ) { problem = "a leaf is empty or over capacity"; return; }
            if ( hd.cap != leaf_capacity && !( leaf == root_ && height_ == 0 ) ) { problem = "a leaf below the root is not full-sized"; return; }
            for ( unsigned i{ 0 }; i < hd.size; ++i ) {
                if ( leaf[ i ] < floor || leaf[ i ] >= ceiling ) { problem = "a key is outside its leaf's range"; return; }
                if ( i != 0 && leaf[ i - 1 ] >= leaf[ i ] ) { problem = "leaf keys are not strictly ascending"; return; }
            }
            for ( unsigned i{ hd.size }; i < padded( hd.cap ); ++i ) {
                if ( leaf[ i ] != pad ) { problem = "an unused key slot is not the pad"; return; }
            }
            if ( prev_leaf( leaf ) != previous ) { problem = "a leaf's back link is wrong"; return; }
            if ( previous != nullptr && next_leaf( previous ) != leaf ) { problem = "a leaf's forward link is wrong"; return; }
            counted += hd.size;
            previous = leaf;
            return;
        }
        auto const * const inner{ static_cast<inner_node const *>( node ) };
        if ( inner->n > inner_keys ) { problem = "an inner node has too many separators"; return; }
        for ( unsigned i{ 0 }; i < inner->n; ++i ) {
            if ( inner->keys[ i ] < floor || inner->keys[ i ] >= ceiling ) { problem = "a separator is outside its node's range"; return; }
            if ( i != 0 && inner->keys[ i - 1 ] >= inner->keys[ i ] ) { problem = "separators are not strictly ascending"; return; }
        }
        for ( unsigned i{ inner->n }; i < inner_keys; ++i ) {
            if ( inner->keys[ i ] != pad ) { problem = "an unused separator is not the pad"; return; }
        }
        for ( unsigned i{ 0 }; i <= inner->n; ++i ) {
            check_subtree( inner->child[ i ], level - 1, i == 0 ? floor : inner->keys[ i - 1 ], i == inner->n ? ceiling : inner->keys[ i ], counted, previous, problem );
        }
    }

    void *        root_  { nullptr };   // an inner node when height_ != 0, else a leaf
    leaf_ptr      tail_  { nullptr };   // the last leaf, which is empty only between open_back() and commit_back()
    std::uint32_t size_  { 0 };
    std::uint8_t  height_{ 0 };
    // The flat directory that stands in for the inner levels (see "the leaf directory"). While it is
    // held, root_ is the first leaf and height_ is 1.
    std::uint8_t  dir_shift_      { 0 };
    leaf_ptr *    dir_            { nullptr };
    key_t         dir_base_       { 0 };
    std::uint32_t dir_buckets_    { 0 };        // the capacity of the table
    std::uint32_t dir_lo_         { 0 };        // the bucket of the smallest key; the buckets below it are not kept
    std::uint32_t next_eval_      { initial_next_eval };   // the table size at which a directory is next considered
};

} // namespace frsr::roaring::detail
