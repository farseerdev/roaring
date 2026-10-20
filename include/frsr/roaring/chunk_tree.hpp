#pragma once

#include <frsr/roaring/container_handle.hpp>   // FRSR_ROARING_HAS_MIMALLOC

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

// FRSR_ROARING_PORTABLE_NODE_SCAN forces the portable scan on a target that has the vector one.
#if defined( __AVX2__ ) && !defined( FRSR_ROARING_PORTABLE_NODE_SCAN )
#   define FRSR_ROARING_NODE_SCAN_AVX2 1
#   include <immintrin.h>
#else
#   define FRSR_ROARING_NODE_SCAN_AVX2 0
#endif

#ifndef FRSR_INNER_KEYS
#define FRSR_INNER_KEYS 16
#endif
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

// The sorted chunk table of a bitmap whose chunk keys are wider than 16 bits: a B+ tree with
// cache-line wide nodes, in place of the always-sorted array a 16-bit key can afford.
//
// A leaf holds up to 15 entries (a key and the handle), an inner node 16 separators and 17
// children. A leaf's keys are searched with one fixed-width scan (four vector compares, or four
// halving steps), an inner node's separators likewise, so a lookup is one scan per level and the
// levels are few: 15 chunks fit one leaf, a few thousand two levels, a few hundred thousand three.
// The leaves are linked, so a walk in key order never goes back up. A bitmap of up to 15 chunks is
// a single leaf that grows like a vector (4, 8, then 15 entries), so a small bitmap stays small.
// Every const member is a pure read.
//
// Keys are held as signed 64-bit numbers, which is what the vector compares work on: a chunk key
// has to stay below the pad (a 64-bit bitmap's has 48 bits). A leaf block of capacity c is laid out
//     [ links, full leaf only ][ c handles, the last one first ][ head: size, capacity ][ keys ]
// and a leaf is named by the address of its first key, so the keys, the handle at an index and the
// head sit at fixed distances from it whatever the capacity.
template <typename Handle>
class chunk_tree {
public:
    using handle_type = Handle;
    using key_type    = std::uint64_t;

    static_assert( Handle::is_trivially_moveable );
    static_assert( alignof( Handle ) <= 8 );

    static constexpr unsigned leaf_capacity{ 15 };
    static constexpr unsigned inner_keys    { FRSR_INNER_KEYS };

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
        return insert_staged( at, static_cast<key_t>( key ), staged.bytes );
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
        return append_staged( static_cast<key_t>( key ), staged.bytes );
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
                // separator that is above everything held and not above the key to come
                auto const separator{ leaf[ head( leaf ).size - 1 ] + 1 };
                path_t path;
                descend_rightmost( path );
                leaf = make_leaf( leaf_capacity );
                link_after( tail_, leaf );
                insert_in_parent( path, 0, separator, leaf );
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
    }

    // Gives back the spare capacity: a root leaf shrinks to the smallest size that holds its entries.
    void shrink_to_fit() {
        if ( size_ == 0 ) {
            release();
            return;
        }
        if ( head( tail_ ).size == 0 ) {                   // the empty last leaf of an abandoned open_back()
            path_t path;
            descend_rightmost( path );
            auto const dropped{ tail_ };
            unlink( dropped );
            remove_child( path, 0 );
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
            for ( auto leaf{ first }; leaf != nullptr; ) {
                auto const following{ next_leaf( leaf ) };
                destroy_leaf( leaf );
                leaf = following;
            }
            throw;
        }
        size_ = count;
        rebuild_levels( first );
        tail_ = last;
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
        if ( root_ == nullptr ) { return size_ == 0 && height_ == 0 && tail_ == nullptr ? nullptr : "an empty tree has a size, a height or a tail"; }
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
    };

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
    void release() noexcept {
        if ( root_ != nullptr ) { release_subtree( root_, height_ ); }
        root_ = nullptr;
        tail_ = nullptr;
        size_ = 0;
        height_ = 0;
    }
    void steal( chunk_tree & other ) noexcept {
        root_   = std::exchange( other.root_, nullptr );
        tail_   = std::exchange( other.tail_, nullptr );
        size_   = std::exchange( other.size_, 0 );
        height_ = std::exchange( other.height_, 0 );
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
        if ( at.index_ != 0 ) {   // between two keys of one leaf, so it belongs here
            auto const & hd{ head( at.leaf_ ) };
            if ( hd.size < hd.cap ) {
                leaf_put( at.leaf_, at.index_, key, staged );
                ++size_;
                return at;
            }
        }
        path_t path;
        auto leaf{ descend( key, path ) };
        auto const index{ scan( leaf, key, head( leaf ).cap ) };
        auto & hd{ head( leaf ) };
        if ( hd.size < hd.cap ) {
            leaf_put( leaf, index, key, staged );
            ++size_;
            return cursor{ leaf, index };
        }
        if ( height_ == 0 && hd.cap < leaf_capacity ) {
            leaf = grow_root_leaf();
            leaf_put( leaf, index, key, staged );
            ++size_;
            return cursor{ leaf, index };
        }
        // a full leaf splits
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
            auto const kept { n / 2 };
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
        insert_in_parent( path, 0, separator, right );
        ++size_;
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
        auto const fresh{ make_leaf( leaf_capacity ) };
        leaf_append( fresh, key, staged );
        path_t path;
        descend_rightmost( path );
        link_after( leaf, fresh );
        insert_in_parent( path, 0, key, fresh );
        ++size_;
        return false;
    }

    // Registers `right` as the sibling that follows the node taken at `level` of `path`, separated by `separator`.
    void insert_in_parent( path_t & path, unsigned const level, key_t const separator, void * const right ) {
        if ( level == height_ ) {                         // the root itself split: a new root above it
            auto const root{ make_inner() };
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
            auto const fresh{ make_inner() };
            fresh->child[ 0 ] = right;
            insert_in_parent( path, level + 1, separator, fresh );
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
        auto const sibling{ make_inner() };
        std::memcpy( parent->keys, all_keys, kept * sizeof( key_t ) );
        std::fill( parent->keys + kept, parent->keys + inner_keys, pad );
        std::memcpy( parent->child, all_children, ( kept + 1 ) * sizeof( void * ) );
        parent->n = kept;
        constexpr unsigned moved{ inner_keys - kept };
        std::memcpy( sibling->keys, all_keys + kept + 1, moved * sizeof( key_t ) );
        std::memcpy( sibling->child, all_children + kept + 1, ( moved + 1 ) * sizeof( void * ) );
        sibling->n = moved;
        insert_in_parent( path, level + 1, all_keys[ kept ], sibling );
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

    // Builds the levels above a chain of full leaves, from `first`, and makes them the tree's.
    void rebuild_levels( leaf_ptr const first ) {
        struct item { void * node; key_t lowest; };
        std::uint32_t count{ 0 };
        for ( auto leaf{ first }; leaf != nullptr; leaf = next_leaf( leaf ) ) { ++count; }
        height_ = 0;
        root_   = first;
        if ( count <= 1 ) { return; }
        auto const items{ std::make_unique_for_overwrite<item[]>( count ) };
        {
            std::uint32_t i{ 0 };
            for ( auto leaf{ first }; leaf != nullptr; leaf = next_leaf( leaf ) ) { items[ i++ ] = item{ leaf, leaf[ 0 ] }; }
        }
        while ( count > 1 ) {
            auto const groups{ ( count + inner_keys ) / ( inner_keys + 1 ) };      // children per node: at most 17
            auto const share { count / groups };
            auto const extra { count % groups };
            std::uint32_t from{ 0 };
            for ( std::uint32_t group{ 0 }; group != groups; ++group ) {
                auto const children{ share + ( group < extra ? 1U : 0U ) };
                auto const node{ make_inner() };
                for ( std::uint32_t j{ 0 }; j != children; ++j ) {
                    node->child[ j ] = items[ from + j ].node;
                    if ( j != 0 ) { node->keys[ j - 1 ] = items[ from + j ].lowest; }
                }
                node->n = children - 1;
                items[ group ] = item{ node, items[ from ].lowest };
                from += children;
            }
            count = groups;
            ++height_;
        }
        root_ = items[ 0 ].node;
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
        release_inner( root_, height_ );
        if ( last_kept == nullptr ) {
            root_ = tail_ = nullptr;
            height_ = 0;
            size_ = 0;
            return;
        }
        if ( head( last_kept ).cap == leaf_capacity ) { links( last_kept )[ 0 ] = nullptr; }
        size_ = kept;
        tail_ = last_kept;
        rebuild_levels( first );
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
    std::uint32_t height_{ 0 };
};

} // namespace frsr::roaring::detail
