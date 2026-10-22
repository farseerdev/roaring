#pragma once

// Operands a bitmap can be combined with directly, without first building a temporary bitmap:
//
//   - key_range              a closed interval of keys (also half-open, open-ended, the whole key space)
//   - bit_words              a span of raw bit-set words (bit i of the string is key `base + i`)
//   - std::bitset<N>         bit i is key `base + i`; `bits_at( bits, base )` gives the base
//   - dynamic bitset         anything shaped like boost::dynamic_bitset (no dependency on Boost headers)
//   - a range of keys        a contiguous span or any forward/input range of unsigned integers; ascending
//                            (duplicates allowed) is detected, or declared with `sorted_keys`
//   - typed_view<Bitmap>     a finished bitmap of the bitmap's own instantiation, read without being copied
//                            (`view_of( bitmap )`); see the contract on the class
//
// Each operand is presented to the bitmap as a stream of chunk-sized pieces in ascending chunk order, each a
// transient container handle that VIEWS the caller's memory (or a small per-source scratch buffer) instead of
// owning a copy. A piece is only ever the right-hand operand of a const kernel or of a combine that builds its
// result in fresh storage, so no handle that aliases the operand is ever stored in a bitmap.

#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/container_handle.hpp>
#include <frsr/roaring/run.hpp>
#include <frsr/roaring/run_container.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace frsr::roaring {

// ---- public operand descriptors -------------------------------------------------------------------------

// An interval of keys, both ends inclusive internally so that every key width, up to and including the whole
// 64-bit key space, is expressible. Bounds beyond the key type of the bitmap it is applied to are clipped to
// the key type's maximum (such keys are not members of any bitmap of that type).
struct key_range {
    std::uint64_t first{ 1 };
    std::uint64_t last { 0 };   // empty by default: last < first

    // [first, last]
    [[nodiscard]] static constexpr key_range closed( std::uint64_t const first, std::uint64_t const last ) noexcept {
        return last < first ? key_range{} : key_range{ first, last };
    }
    // [lo, hi), empty when hi <= lo
    [[nodiscard]] static constexpr key_range half_open( std::uint64_t const lo, std::uint64_t const hi ) noexcept {
        return hi <= lo ? key_range{} : key_range{ lo, hi - 1 };
    }
    // [first, maximum key]
    [[nodiscard]] static constexpr key_range from( std::uint64_t const first ) noexcept {
        return key_range{ first, std::numeric_limits<std::uint64_t>::max() };
    }
    // every key
    [[nodiscard]] static constexpr key_range all() noexcept {
        return key_range{ 0, std::numeric_limits<std::uint64_t>::max() };
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return last < first; }
};

// A string of bits in memory, bit i of which stands for the key `base + i`. The bits are read in memory order
// as little-endian words: bit i lives in bit (i % W) of word (i / W) for a word of W bits, which is the layout
// of every bit-set the interop recognizes. `bit_count` defaults to all the bits of `words`.
template <std::unsigned_integral Word>
struct bit_words {
    static_assert( std::endian::native == std::endian::little, "bit_words reads the bit string as little-endian words" );

    std::span<Word const> words;
    std::uint64_t         base{};
    std::uint64_t         bit_count{};

    constexpr bit_words( std::span<Word const> const words_, std::uint64_t const base_ = 0 ) noexcept
        : words{ words_ }, base{ base_ }, bit_count{ std::uint64_t{ words_.size() } * std::numeric_limits<Word>::digits } {}
    constexpr bit_words( std::span<Word const> const words_, std::uint64_t const base_, std::uint64_t const bit_count_ ) noexcept
        : words{ words_ }, base{ base_ }, bit_count{ std::min<std::uint64_t>( bit_count_, std::uint64_t{ words_.size() } * std::numeric_limits<Word>::digits ) } {}
};
template <std::unsigned_integral Word>
bit_words( std::span<Word const> ) -> bit_words<Word>;
template <std::unsigned_integral Word>
bit_words( std::span<Word const>, std::uint64_t ) -> bit_words<Word>;
template <std::unsigned_integral Word>
bit_words( std::span<Word const>, std::uint64_t, std::uint64_t ) -> bit_words<Word>;

// A bit-set operand (std::bitset, a dynamic bitset) whose bit 0 stands for the key `base`. Holds a reference:
// meant to be created in the call expression that uses it.
template <typename Bits>
struct based_bits {
    Bits const &  bits;
    std::uint64_t base;
};
template <typename Bits>
based_bits( Bits const &, std::uint64_t ) -> based_bits<Bits>;

template <typename Bits>
[[nodiscard]] constexpr based_bits<Bits> bits_at( Bits const & bits, std::uint64_t const base ) noexcept { return { bits, base }; }

// Declares a range of keys to be in ascending order (equal neighbours allowed), so it is not checked. Holds a
// reference, like based_bits. Declaring a range sorted that is not is a programming error (asserted in debug
// builds, a wrong answer otherwise).
template <std::ranges::forward_range Range>
struct sorted_keys {
    Range const & keys;
};
template <std::ranges::forward_range Range>
sorted_keys( Range const & ) -> sorted_keys<Range>;

// A read-only, non-owning handle on a finished bitmap, for use as the operand of the set operations of a bitmap
// of the same instantiation (`a &= view_of( b )`, `a.and_cardinality( view_of( b ) )`, ...). It is one pointer.
//
// What it is for: reading a bitmap that other threads read too, without writing anything it owns. Nothing the
// operation does touches the viewed bitmap, not a reference count, not a container header, not the chunk table,
// and nothing the operation keeps refers to it: whatever a result takes from the view is a fresh copy, so a
// result outlives the viewed bitmap. (A bitmap passed as such, `a &= b`, keeps sharing the containers of `b` by
// reference count where that is cheaper for a single thread, and that is a write to the counts of `b`.)
//
// Contract, the caller's to keep (as with std::string_view; release builds do not check it):
//   - the bitmap is finished: not in the middle of a lazy fold (asserted in debug builds);
//   - it is not modified, moved, swapped or destroyed while a view of it, or an operation reading through one,
//     is alive. Any number of threads may read through views of it and read it directly at the same time.
// Applying a view of a bitmap to the bitmap itself is defined: `a &= view_of( a )` and `a |= view_of( a )` leave
// `a` as it is, `a -= view_of( a )` and `a ^= view_of( a )` empty it. A bitmap that is a copy-on-write copy of the
// viewed one may be updated through its own view: dropping a payload it shares releases that payload's reference,
// which is the copy's own.
//
// It is deliberately not a range (no begin / end), so it is never taken for a sequence of keys, and it is not the
// type-erased bitmap_view of bitmap_view.hpp, which has no set operations (it converts to that one).
template <typename Bitmap>
class typed_view {
public:
    using bitmap_type = Bitmap;
    using key_type    = typename Bitmap::key_type;
    using size_type   = typename Bitmap::size_type;

    explicit typed_view( Bitmap const & bitmap ) noexcept : bitmap_{ &bitmap } { assert( !bitmap.counts_deferred() && "a view is taken of a finished bitmap" ); }

    [[nodiscard]] Bitmap const & get() const noexcept { return *bitmap_; }

    [[nodiscard]] bool      contains( key_type const key ) const noexcept { return bitmap_->contains( key ); }
    [[nodiscard]] size_type size    () const noexcept { return bitmap_->size(); }
    [[nodiscard]] bool      empty   () const noexcept { return bitmap_->empty(); }
    [[nodiscard]] key_type  front   () const noexcept { return bitmap_->front(); }
    [[nodiscard]] key_type  back    () const noexcept { return bitmap_->back(); }

    // Whether both name the same bitmap.
    [[nodiscard]] friend bool operator==( typed_view const lhs, typed_view const rhs ) noexcept { return lhs.bitmap_ == rhs.bitmap_; }

private:
    Bitmap const * bitmap_;
};

template <typename Bitmap>
    requires requires { typename Bitmap::chunk_type; typename Bitmap::layout_type; typename Bitmap::container_set_type; }
[[nodiscard]] typed_view<Bitmap> view_of( Bitmap const & bitmap ) noexcept { return typed_view<Bitmap>{ bitmap }; }

namespace detail {

template <typename Bitmap>
struct interop_engine;

template <typename T>
inline constexpr bool is_typed_view_v{ false };
template <typename Bitmap>
inline constexpr bool is_typed_view_v<typed_view<Bitmap>>{ true };

template <typename T>
inline constexpr bool is_sorted_keys_v{ false };
template <typename Range>
inline constexpr bool is_sorted_keys_v<sorted_keys<Range>>{ true };

template <typename T>
inline constexpr bool is_based_bits_v{ false };
template <typename Bits>
inline constexpr bool is_based_bits_v<based_bits<Bits>>{ true };

template <typename T>
inline constexpr bool is_bit_words_v{ false };
template <typename Word>
inline constexpr bool is_bit_words_v<bit_words<Word>>{ true };

template <typename T>
inline constexpr bool is_std_bitset_v{ false };
template <std::size_t N>
inline constexpr bool is_std_bitset_v<std::bitset<N>>{ true };

// A frsr bitmap, recognized by shape so that this header does not need the class.
template <typename T>
concept bitmap_like = requires { typename T::chunk_type; typename T::layout_type; typename T::container_set_type; };

template <typename T>
concept key_element = std::unsigned_integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>;

// boost::dynamic_bitset shape: a block type, a bit count and the ADL function that copies out the blocks.
template <typename T>
concept block_bitset = requires ( T const & bits, std::vector<typename T::block_type> & out ) {
    requires std::unsigned_integral<typename T::block_type>;
    { bits.size() } -> std::convertible_to<std::size_t>;
    to_block_range( bits, std::back_inserter( out ) );
};

// A bit-set that can only be walked by its set bits.
template <typename T>
concept find_bitset = !block_bitset<T> && requires ( T const & bits, std::size_t pos ) {
    { bits.size() } -> std::convertible_to<std::size_t>;
    { bits.find_first() } -> std::convertible_to<std::size_t>;
    { bits.find_next( pos ) } -> std::convertible_to<std::size_t>;
    { T::npos } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept bit_container = is_std_bitset_v<T> || block_bitset<T> || find_bitset<T>;

template <typename T>
concept key_sequence = !bitmap_like<T> && !is_bit_words_v<T> && !is_based_bits_v<T> && !is_sorted_keys_v<T> && !bit_container<T> &&
    !std::same_as<T, key_range> && std::ranges::input_range<T> && key_element<std::ranges::range_reference_t<T>>;

template <typename T>
concept sorted_key_sequence = is_sorted_keys_v<T> && key_sequence<std::remove_cvref_t<decltype( std::declval<T const &>().keys )>>;

// Whether the read-only interop operations against an operand of this type can be told never to throw:
// they allocate only to copy a large bit-set out, or to sort / collect keys, and run on stack scratch otherwise.
template <typename T>
inline constexpr bool interop_nothrow_v{ false };
template <>
inline constexpr bool interop_nothrow_v<key_range>{ true };
template <typename Word>
inline constexpr bool interop_nothrow_v<bit_words<Word>>{ true };
template <std::size_t N>
inline constexpr bool interop_nothrow_v<std::bitset<N>>{ ( N + 63 ) / 64 <= 2048 };
template <std::size_t N>
inline constexpr bool interop_nothrow_v<based_bits<std::bitset<N>>>{ ( N + 63 ) / 64 <= 2048 };
template <typename Bitmap>
inline constexpr bool interop_nothrow_v<typed_view<Bitmap>>{ true };

} // namespace detail

// What a bitmap can be combined with by the interop operations (and_with, or_with, andnot_with, intersects,
// and_cardinality, is_subset_of, is_superset_of, equals).
template <typename T>
concept interop_operand =
    std::same_as<T, key_range> ||
    detail::is_bit_words_v<T> ||
    ( detail::is_based_bits_v<T> && detail::bit_container<std::remove_cvref_t<decltype( std::declval<T const &>().bits )>> ) ||
    detail::bit_container<T> ||
    detail::key_sequence<T> ||
    detail::sorted_key_sequence<T> ||
    detail::is_typed_view_v<T>;

namespace detail {

// ---- operands the library holds compiled ------------------------------------------------------------------
//
// The interop operations are templates over the operand; instantiating the engine for every operand a caller
// names would put all of it into the caller. The operands that account for nearly every use are therefore
// reduced to one type-erased view, which the library's bitmap members take: an interval of keys, a bit string
// (of any word width: the bits are read as bytes), and a contiguous run of keys of the bitmap's key type,
// declared sorted or not. Any other operand (a std::bitset, a dynamic bit-set, a key range that is not
// contiguous or holds another integer type, a bit string reached through based_bits) still goes through the
// engine itself, which needs bitmap_definitions.hpp when the bitmap is compiled in the library.

enum class interop_op : std::uint8_t {
    and_with, or_with, andnot_with, xor_with,
    xor_cardinality, and_cardinality,
    intersects, is_subset_of, is_superset_of, equals
};

struct operand_view {
    enum class shape_type : std::uint8_t { interval, bits, sorted_keys, unsorted_keys, bitmap };

    shape_type    shape{};
    key_range     interval{};   // interval
    void const *  data{};       // bits: the first byte; keys: the first key; bitmap: the viewed bitmap (of the applying bitmap's own type)
    std::size_t   count{};      // keys
    std::uint64_t bit_count{};  // bits
    std::uint64_t base{};       // bits: the key bit 0 stands for
};

template <typename T, typename Key>
concept contiguous_keys_of = std::ranges::contiguous_range<T const> && std::ranges::sized_range<T const>
    && std::same_as<std::remove_cv_t<std::ranges::range_value_t<T const>>, Key>;

template <typename Operand, typename Key>
inline constexpr bool sorted_keys_erasable{ false };
template <typename Range, typename Key>
inline constexpr bool sorted_keys_erasable<sorted_keys<Range>, Key>{ contiguous_keys_of<Range, Key> };

template <typename Operand, typename Bitmap>
inline constexpr bool interop_erasable{
    std::same_as<Operand, key_range> || is_bit_words_v<Operand> || contiguous_keys_of<Operand, typename Bitmap::key_type> ||
    sorted_keys_erasable<Operand, typename Bitmap::key_type> || std::same_as<Operand, typed_view<Bitmap>>
};

template <typename Operand>
[[nodiscard]] operand_view make_operand_view( Operand const & operand ) noexcept {
    operand_view view;
    if constexpr ( std::same_as<Operand, key_range> ) {
        view.shape    = operand_view::shape_type::interval;
        view.interval = operand;
    } else if constexpr ( is_bit_words_v<Operand> ) {
        view.shape     = operand_view::shape_type::bits;
        view.data      = operand.words.data();
        view.bit_count = operand.bit_count;
        view.base      = operand.base;
    } else if constexpr ( is_typed_view_v<Operand> ) {
        view.shape = operand_view::shape_type::bitmap;
        view.data  = &operand.get();
    } else if constexpr ( is_sorted_keys_v<Operand> ) {
        view.shape = operand_view::shape_type::sorted_keys;
        view.data  = std::ranges::data( operand.keys );
        view.count = static_cast<std::size_t>( std::ranges::size( operand.keys ) );
    } else {
        view.shape = operand_view::shape_type::unsorted_keys;
        view.data  = std::ranges::data( operand );
        view.count = static_cast<std::size_t>( std::ranges::size( operand ) );
    }
    return view;
}

// ---- chunk sources ---------------------------------------------------------------------------------------
//
// A chunk source yields the non-overlapping pieces of an operand in ascending chunk order:
//
//   at_end()      no piece is left
//   key()         chunk key of the current piece (cheap; the piece itself is not built)
//   piece()       the current piece as a transient handle, valid until the source moves; its cardinality
//                 may be zero (the source does not look for empty pieces)
//   advance()     the next piece
//   seek( key )   the first piece whose chunk key is not below `key` (does nothing if already there)
//   truncated()   whether members of the operand lie beyond the bitmap's key type (they match nothing)
//
// A source is not copyable: its pieces point into its own scratch.

template <typename Layout, typename CowPolicy>
class interval_source {
public:
    using chunk_type  = typename Layout::chunk_type;
    using key_type    = typename Layout::key_type;
    using low_type    = typename Layout::low_type;
    using handle_type = container_handle<Layout, CowPolicy>;

    explicit interval_source( key_range const range ) noexcept {
        constexpr auto max_key{ static_cast<std::uint64_t>( std::numeric_limits<key_type>::max() ) };
        if ( range.empty() || range.first > max_key ) {
            done_ = true;
            return;
        }
        auto const first{ static_cast<key_type>( range.first ) };
        auto const last { static_cast<key_type>( std::min( range.last, max_key ) ) };
        first_chunk_ = Layout::chunk_key( first );
        last_chunk_  = Layout::chunk_key( last  );
        first_low_   = Layout::low_key  ( first );
        last_low_    = Layout::low_key  ( last  );
        chunk_       = first_chunk_;
    }
    interval_source( interval_source const & ) = delete;
    interval_source & operator=( interval_source const & ) = delete;

    [[nodiscard]] bool       at_end   () const noexcept { return done_; }
    [[nodiscard]] chunk_type key      () const noexcept { return chunk_; }
    [[nodiscard]] bool       truncated() const noexcept { return false; }   // bounds past the key type are clipped

    [[nodiscard]] handle_type const & piece() noexcept {
        auto const begin{ chunk_ == first_chunk_ ? first_low_ : low_type{ 0 } };
        auto const end  { chunk_ == last_chunk_  ? last_low_  : std::numeric_limits<low_type>::max() };
        if ( begin == 0 && end == std::numeric_limits<low_type>::max() ) {
            // a chunk the interval covers entirely: the same piece every time, made once
            if ( !full_made_ ) {
                full_run_  = { low_type{ 0 }, std::numeric_limits<low_type>::max() };
                full_piece_ = handle_type::make_borrowed( container_kind::run, &full_run_, 1, static_cast<std::uint32_t>( Layout::low_domain_size ) );
                full_made_ = true;
            }
            return full_piece_;
        }
        run_ = { begin, end };
        piece_ = handle_type::make_borrowed( container_kind::run, &run_, 1, static_cast<std::uint32_t>( end - begin ) + 1U );
        return piece_;
    }

    void advance() noexcept {
        if ( chunk_ == last_chunk_ ) { done_ = true; } else { ++chunk_; }
    }

    void seek( chunk_type const key ) noexcept {
        if ( done_ || !( chunk_ < key ) ) { return; }
        if ( last_chunk_ < key ) { done_ = true; } else { chunk_ = key; }
    }

private:
    chunk_type         first_chunk_{};
    chunk_type         last_chunk_ {};
    chunk_type         chunk_      {};
    low_type           first_low_  {};
    low_type           last_low_   {};
    bool               done_       { false };
    bool               full_made_  { false };
    run<low_type>      run_        {};
    run<low_type>      full_run_   {};
    handle_type        piece_      {};
    handle_type        full_piece_ {};
};

template <typename Source>
inline constexpr bool is_interval_source_v{ false };
template <typename Layout, typename CowPolicy>
inline constexpr bool is_interval_source_v<interval_source<Layout, CowPolicy>>{ true };

// The pieces of a sorted range of keys (equal neighbours allowed): the keys of a chunk are gathered into an
// array piece, or a bitset piece when there are more of them than an array holds.
template <typename Layout, typename CowPolicy, typename It, typename Sent>
class keys_source {
public:
    using chunk_type  = typename Layout::chunk_type;
    using key_type    = typename Layout::key_type;
    using low_type    = typename Layout::low_type;
    using handle_type = container_handle<Layout, CowPolicy>;

    static constexpr std::size_t array_limit{ Layout::low_domain_size >= 1024 ? Layout::low_domain_size / 16 : 64 };

    keys_source( It first, Sent last ) : cur_{ std::move( first ) }, end_{ std::move( last ) }, group_end_{ cur_ } { refresh(); }
    keys_source( keys_source const & ) = delete;
    keys_source & operator=( keys_source const & ) = delete;

    [[nodiscard]] bool       at_end   () const { return cur_ == end_ || !in_domain( *cur_ ); }
    [[nodiscard]] chunk_type key      () const noexcept { return chunk_; }
    [[nodiscard]] bool       truncated() const { return cur_ != end_ && !in_domain( *cur_ ); }

    [[nodiscard]] handle_type const & piece() {
        if ( !built_ ) { build(); }
        return piece_;
    }

    void advance() {
        if ( !built_ ) { build_extent(); }
        cur_ = group_end_;
        built_ = false;
        refresh();
    }

    void seek( chunk_type const key ) {
        if ( at_end() || !( chunk_ < key ) ) { return; }
        built_ = false;
        auto const bound{ static_cast<std::uint64_t>( Layout::compose( key, 0 ) ) };
        if constexpr ( std::random_access_iterator<It> && std::sized_sentinel_for<Sent, It> ) {
            // gallop, then bisect: the next wanted chunk is usually near, sometimes far
            auto const count{ static_cast<std::size_t>( end_ - cur_ ) };
            std::size_t step{ 1 };
            std::size_t low { 0 };
            while ( step < count && static_cast<std::uint64_t>( cur_[ static_cast<std::ptrdiff_t>( step ) ] ) < bound ) {
                low  = step;
                step *= 2;
            }
            auto const high{ std::min( step + 1, count ) };
            cur_ = std::lower_bound( cur_ + static_cast<std::ptrdiff_t>( low ), cur_ + static_cast<std::ptrdiff_t>( high ), bound,
                []( auto const element, std::uint64_t const wanted ) noexcept { return static_cast<std::uint64_t>( element ) < wanted; } );
        } else {
            while ( cur_ != end_ && static_cast<std::uint64_t>( *cur_ ) < bound ) { ++cur_; }
        }
        refresh();
    }

private:
    template <typename E>
    [[nodiscard]] static constexpr bool in_domain( E const element ) noexcept {
        if constexpr ( std::numeric_limits<E>::digits <= std::numeric_limits<key_type>::digits ) {
            return true;
        } else {
            return element <= static_cast<E>( std::numeric_limits<key_type>::max() );
        }
    }

    void refresh() {
        if ( !at_end() ) { chunk_ = Layout::chunk_key( static_cast<key_type>( *cur_ ) ); }
    }

    // The end of the current chunk's keys, without gathering them.
    void build_extent() {
        auto it{ cur_ };
        while ( it != end_ && in_domain( *it ) && Layout::chunk_key( static_cast<key_type>( *it ) ) == chunk_ ) { ++it; }
        group_end_ = it;
    }

    void build() {
        std::size_t count{ 0 };
        bool        as_bitset{ false };
        low_type    previous{};
        auto        it{ cur_ };
        for ( ; it != end_ && in_domain( *it ) ; ++it ) {
            auto const key{ static_cast<key_type>( *it ) };
            if ( Layout::chunk_key( key ) != chunk_ ) { break; }
            auto const low{ Layout::low_key( key ) };
            assert( count == 0 || !( low < previous ) );
            if ( count != 0 && low == previous ) { continue; }
            previous = low;
            if ( !as_bitset && count == array_limit ) {
                as_bitset = true;
                words_.fill( 0 );
                for ( std::size_t i{ 0 }; i < count; ++i ) {
                    words_[ static_cast<std::size_t>( array_[ i ] ) >> 6U ] |= std::uint64_t{ 1 } << ( static_cast<unsigned>( array_[ i ] ) & 63U );
                }
            }
            if ( as_bitset ) {
                words_[ static_cast<std::size_t>( low ) >> 6U ] |= std::uint64_t{ 1 } << ( static_cast<unsigned>( low ) & 63U );
            } else {
                array_[ count ] = low;
            }
            ++count;
        }
        group_end_ = it;
        built_ = true;
        piece_ = as_bitset
            ? handle_type::make_borrowed( container_kind::bitset, words_.data(), static_cast<std::uint32_t>( Layout::word_count ), static_cast<std::uint32_t>( count ) )
            : handle_type::make_borrowed( container_kind::array , array_.data(), static_cast<std::uint32_t>( count ), static_cast<std::uint32_t>( count ) );
    }

    It                                              cur_;
    Sent                                            end_;
    It                                              group_end_;
    chunk_type                                      chunk_{};
    bool                                            built_{ false };
    std::array<low_type, array_limit>               array_;
    std::array<std::uint64_t, Layout::word_count>   words_;
    handle_type                                     piece_{};
};

// The pieces of a bit string in memory: bit i stands for the key `base + i`. A chunk that is covered by
// whole, 8-byte aligned words of the string is viewed in place; every other chunk (a base that is not aligned
// to a chunk or a word, the ends of the string) is assembled in the scratch block with shifts and masks, so the
// bits past the end of the string never leak.
template <typename Layout, typename CowPolicy>
class bits_source {
public:
    using chunk_type  = typename Layout::chunk_type;
    using key_type    = typename Layout::key_type;
    using handle_type = container_handle<Layout, CowPolicy>;
    using low_type    = typename Layout::low_type;

    static constexpr std::uint64_t chunk_bits{ Layout::low_domain_size };
    static constexpr std::size_t   array_limit{ Layout::low_domain_size >= 1024 ? Layout::low_domain_size / 16 : 64 };

    bits_source( void const * const data, std::uint64_t const bit_count, std::uint64_t const base ) noexcept
        : bytes_{ static_cast<unsigned char const *>( data ) }, byte_count_{ ( bit_count + 7 ) / 8 }, bit_count_{ bit_count }, base_{ base }
    {
        constexpr auto max_key{ static_cast<std::uint64_t>( std::numeric_limits<key_type>::max() ) };
        if ( bit_count == 0 || base > max_key ) {
            if ( bit_count != 0 ) { truncated_ = any_set_from( 0 ); }
            done_ = true;
            return;
        }
        // the last bit that is a key of the bitmap's type, as an index into the string
        auto const last_index{ bit_count - 1 };
        if ( last_index > max_key - base ) {
            last_rel_  = max_key - base;
            truncated_ = any_set_from( last_rel_ + 1 );
        } else {
            last_rel_ = last_index;
        }
        first_chunk_ = Layout::chunk_key( static_cast<key_type>( base ) );
        last_chunk_  = Layout::chunk_key( static_cast<key_type>( base + last_rel_ ) );
        chunk_       = first_chunk_;
    }
    bits_source( bits_source const & ) = delete;
    bits_source & operator=( bits_source const & ) = delete;

    [[nodiscard]] bool       at_end   () const noexcept { return done_; }
    [[nodiscard]] chunk_type key      () const noexcept { return chunk_; }
    [[nodiscard]] bool       truncated() const noexcept { return truncated_; }

    [[nodiscard]] handle_type const & piece() noexcept {
        auto const words{ chunk_words() };
        auto const cardinality{ popcount_words( words, Layout::word_count ) };
        if ( cardinality > array_limit ) {
            piece_ = handle_type::make_borrowed( container_kind::bitset, words, static_cast<std::uint32_t>( Layout::word_count ), static_cast<std::uint32_t>( cardinality ) );
        } else {
            // few bits: an array piece, as every container of that size is
            std::size_t count{ 0 };
            for ( std::size_t w{ 0 }; w < Layout::word_count; ++w ) {
                for ( auto word{ words[ w ] }; word != 0; word &= word - 1 ) {
                    array_[ count++ ] = static_cast<low_type>( w * 64 + static_cast<std::size_t>( std::countr_zero( word ) ) );
                }
            }
            piece_ = handle_type::make_borrowed( container_kind::array, array_.data(), static_cast<std::uint32_t>( count ), static_cast<std::uint32_t>( count ) );
        }
        return piece_;
    }

    void advance() noexcept {
        if ( chunk_ == last_chunk_ ) { done_ = true; } else { ++chunk_; }
    }

    void seek( chunk_type const key ) noexcept {
        if ( done_ || !( chunk_ < key ) ) { return; }
        if ( last_chunk_ < key ) { done_ = true; } else { chunk_ = key; }
    }

private:
    // The words of the current chunk: in place when whole aligned words of the string cover it, else assembled
    // in the scratch block.
    [[nodiscard]] std::uint64_t const * chunk_words() noexcept {
        auto const chunk_start{ static_cast<std::uint64_t>( Layout::compose( chunk_, 0 ) ) };
        if ( chunk_start >= base_ ) {
            auto const first{ chunk_start - base_ };
            if ( ( first & 63U ) == 0 && last_rel_ - first >= chunk_bits - 1 ) {
                auto const * const aligned{ bytes_ + ( first >> 3U ) };
                if ( ( reinterpret_cast<std::uintptr_t>( aligned ) & 7U ) == 0 ) {
                    return reinterpret_cast<std::uint64_t const *>( aligned );
                }
            }
            for ( std::size_t w{ 0 }; w < Layout::word_count; ++w ) {
                // `first <= last_rel_` holds for every chunk of the string
                scratch_[ w ] = w * 64 <= last_rel_ - first ? load_bits( first + w * 64 ) : 0;
            }
        } else {
            // the chunk starts below the string: its bit 0 is key `base_`, offset bits into the chunk
            auto const offset{ base_ - chunk_start };
            for ( std::size_t w{ 0 }; w < Layout::word_count; ++w ) {
                auto const dest{ static_cast<std::uint64_t>( w ) * 64 };
                if ( dest + 63 < offset ) { scratch_[ w ] = 0; continue; }
                scratch_[ w ] = dest >= offset ? load_bits( dest - offset ) : load_bits( 0 ) << ( offset - dest );
            }
        }
        return scratch_.data();
    }

    [[nodiscard]] std::uint8_t byte_at( std::uint64_t const index ) const noexcept { return index < byte_count_ ? bytes_[ index ] : std::uint8_t{ 0 }; }

    [[nodiscard]] std::uint64_t read_word( std::uint64_t const byte ) const noexcept {
        std::uint64_t word{ 0 };
        if ( byte + 8 <= byte_count_ ) {
            std::memcpy( &word, bytes_ + byte, 8 );
        } else {
            for ( unsigned i{ 0 }; i < 8; ++i ) { word |= static_cast<std::uint64_t>( byte_at( byte + i ) ) << ( 8U * i ); }
        }
        return word;
    }

    // The 64 bits of the string starting at bit `index`, the bits beyond its last as zeros.
    [[nodiscard]] std::uint64_t load_bits( std::uint64_t const index ) const noexcept {
        if ( index > last_rel_ ) { return 0; }
        auto const byte { index >> 3U };
        auto const shift{ static_cast<unsigned>( index & 7U ) };
        auto word{ read_word( byte ) >> shift };
        if ( shift != 0 ) { word |= static_cast<std::uint64_t>( byte_at( byte + 8 ) ) << ( 64U - shift ); }
        if ( auto const left{ last_rel_ - index }; left < 63 ) {
            word &= ( std::uint64_t{ 2 } << left ) - 1U;
        }
        return word;
    }

    // Whether any bit at or after `index` is set (the part of a string that lies beyond the key type).
    [[nodiscard]] bool any_set_from( std::uint64_t const index ) const noexcept {
        auto const bits{ bit_count_ };
        for ( auto i{ index }; i < bits; ) {
            if ( ( i & 7U ) == 0 && i + 64 <= bits ) {
                if ( read_word( i >> 3U ) != 0 ) { return true; }
                i += 64;
            } else {
                if ( ( byte_at( i >> 3U ) >> ( i & 7U ) ) & 1U ) { return true; }
                ++i;
            }
        }
        return false;
    }

    unsigned char const *                           bytes_;
    std::uint64_t                                   byte_count_;
    std::uint64_t                                   bit_count_;
    std::uint64_t                                   base_;
    std::uint64_t                                   last_rel_{};
    chunk_type                                      first_chunk_{};
    chunk_type                                      last_chunk_ {};
    chunk_type                                      chunk_      {};
    bool                                            done_     { false };
    bool                                            truncated_{ false };
    std::array<std::uint64_t, Layout::word_count>   scratch_;
    std::array<low_type, array_limit>               array_;
    handle_type                                     piece_{};
};

// A forward iterator over the set bits of a bit-set that offers find_first / find_next, as keys `base + bit`.
template <typename Bits>
class find_bits_iterator {
public:
    using value_type        = std::uint64_t;
    using difference_type   = std::ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    find_bits_iterator() = default;
    find_bits_iterator( Bits const & bits, std::uint64_t const base ) : bits_{ &bits }, base_{ base }, position_{ static_cast<std::size_t>( bits.find_first() ) } {}

    [[nodiscard]] value_type operator*() const { return base_ + position_; }
    find_bits_iterator & operator++() { position_ = static_cast<std::size_t>( bits_->find_next( position_ ) ); return *this; }
    find_bits_iterator operator++( int ) { auto const previous{ *this }; ++*this; return previous; }
    [[nodiscard]] friend bool operator==( find_bits_iterator const & lhs, find_bits_iterator const & rhs ) noexcept { return lhs.position_ == rhs.position_; }

    [[nodiscard]] static find_bits_iterator end_of( Bits const & bits, std::uint64_t const base ) {
        find_bits_iterator result;
        result.bits_ = &bits; result.base_ = base; result.position_ = static_cast<std::size_t>( Bits::npos );
        return result;
    }

private:
    Bits const *  bits_{};
    std::uint64_t base_{};
    std::size_t   position_{ static_cast<std::size_t>( Bits::npos ) };
};

// Whether std::bitset<N> stores its bits as a little-endian string, bit i at bit (i % 8) of byte (i / 8), with
// its padding bits clear: true for every standard library in use, checked once at run time rather than assumed.
template <std::size_t N>
[[nodiscard]] inline bool std_bitset_is_bit_string() noexcept {
    static bool const result{ []() noexcept {
        if constexpr ( !std::is_trivially_copyable_v<std::bitset<N>> || N == 0 || std::endian::native != std::endian::little ) {
            return false;
        } else {
            constexpr std::size_t probes[]{ 0, 1, 8, 9, 63, 64, N / 2, N - 1 };
            std::bitset<N> bits;
            for ( auto const probe : probes ) { if ( probe < N ) { bits.set( probe ); } }
            std::array<unsigned char, ( sizeof( std::bitset<N> ) > ( N + 7 ) / 8 ? sizeof( std::bitset<N> ) : ( N + 7 ) / 8 )> bytes{};
            std::memcpy( bytes.data(), &bits, sizeof( bits ) );
            std::size_t set{ 0 };
            for ( std::size_t i{ 0 }; i < bytes.size() * 8; ++i ) {
                auto const stored{ ( ( bytes[ i / 8 ] >> ( i % 8 ) ) & 1U ) != 0 };
                if ( stored != ( i < N && bits.test( i ) ) ) { return false; }
                set += stored;
            }
            return set == bits.count();
        }
    }() };
    return result;
}

} // namespace detail

} // namespace frsr::roaring
