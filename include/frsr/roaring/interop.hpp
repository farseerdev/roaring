#pragma once

// The set operations of bitmap against operands that are not bitmaps (see interop_operands.hpp for the
// operand kinds): and_with / or_with / andnot_with / xor_with, intersects, and_cardinality, xor_cardinality,
// is_subset_of, is_superset_of, equals.
//
// The operand is walked as a stream of chunk-sized pieces and merged with the chunk table the way two bitmaps
// are: both sides are searched for each other's next key (a gallop on the bitmap's table, a seek on the
// source), and a piece meets only the container of its own chunk, with the same container kernels that serve
// bitmap against bitmap. No temporary bitmap is built, and chunks of either side the other lacks are
// not visited one by one.
//
//   - A sorted range of keys is grouped by chunk into array pieces (bitset pieces when a chunk holds more keys
//     than an array does) in a small scratch buffer.
//   - A bit string is viewed in place when whole aligned words cover a chunk, and is assembled by shifts and
//     masks into one scratch block otherwise.
//   - A key interval is a run piece per chunk, produced on demand; a chunk it covers entirely is never
//     opened: it is kept, dropped or its cardinality read, whatever the operation calls for.
//   - An UNSORTED range of keys: intersects and is_superset_of probe it key by key (a binary search of the
//     chunk table plus one container lookup; they stop at the first answer). and_with and and_cardinality
//     probe below a few thousand keys, where the cost follows the keys and not the bitmap, and sort a copy
//     above, which costs less than a probe per key. The other operations, which have to see the operand as a
//     set, sort a copy and merge it like a sorted range.
//
// Keys of the operand that the bitmap's key type cannot hold are not members: they match nothing, and make
// is_superset_of / equals false. All const operations are pure reads.

// Included by bitmap.hpp, after the class.
#include <frsr/roaring/interop_operands.hpp>

namespace frsr::roaring::detail {

// The pieces of another bitmap, so that the queries which have an interop form work against a bitmap too.
template <typename Store>
class store_source {
public:
    using chunk_type  = typename Store::chunk_type;
    using handle_type = typename Store::handle_type;

    explicit store_source( Store const & store ) noexcept : store_{ &store }, cur_{ store.begin() } {}

    [[nodiscard]] bool                at_end   () const noexcept { return cur_ == store_->end(); }
    [[nodiscard]] chunk_type          key      () const noexcept { return store_->key( cur_ ); }
    [[nodiscard]] handle_type const & piece    () noexcept { return store_->slot( cur_ ); }
    [[nodiscard]] bool                truncated() const noexcept { return false; }
    void advance() noexcept { ++cur_; }
    void seek( chunk_type const key ) noexcept {
        if ( !at_end() && store_->key( cur_ ) < key ) { cur_ = store_->lower_bound( key, cur_ ); }
    }

private:
    Store const *                store_;
    typename Store::cursor       cur_;
};

template <typename... Fs>
struct overloaded : Fs... { using Fs::operator()...; };
template <typename... Fs>
overloaded( Fs... ) -> overloaded<Fs...>;

struct unsorted_keys_tag {};

// Sorts keys ascending. Below a few hundred keys that is the comparison sort; above, a least-significant-digit
// radix sort over bytes (digits that every key shares are skipped), which does not pay the comparison sort's
// log factor.
template <std::unsigned_integral E>
void sort_keys( heap_vector<E> & keys ) {
    constexpr std::size_t radix_threshold{ 512 };
    auto const count{ keys.size() };
    if ( count < radix_threshold ) {
        std::ranges::sort( keys );
        return;
    }
    constexpr unsigned digits{ sizeof( E ) };
    std::array<std::array<std::uint32_t, 256>, digits> histogram{};
    for ( auto const key : keys ) {
        for ( unsigned d{ 0 }; d < digits; ++d ) { ++histogram[ d ][ ( key >> ( 8U * d ) ) & 0xFFU ]; }
    }
    heap_vector<E> scratch;
    resize_uninitialized( scratch, count );
    E * from{ keys.data() };
    E * to  { scratch.data() };
    for ( unsigned d{ 0 }; d < digits; ++d ) {
        auto & bucket{ histogram[ d ] };
        if ( bucket[ ( from[ 0 ] >> ( 8U * d ) ) & 0xFFU ] == count ) { continue; }   // every key has this digit
        std::uint32_t offset{ 0 };
        for ( auto & slot : bucket ) { auto const n{ slot }; slot = offset; offset += n; }
        for ( std::size_t i{ 0 }; i < count; ++i ) { to[ bucket[ ( from[ i ] >> ( 8U * d ) ) & 0xFFU ]++ ] = from[ i ]; }
        std::swap( from, to );
    }
    if ( from != keys.data() ) { std::memcpy( keys.data(), from, count * sizeof( E ) ); }
}

template <typename Bitmap>
struct interop_engine {
    using key_type    = typename Bitmap::key_type;
    using layout_type = typename Bitmap::layout_type;
    using cow_policy  = typename Bitmap::cow_policy;
    using chunk_type  = typename Bitmap::chunk_type;
    using low_type    = typename Bitmap::low_type;
    using handle_type = typename Bitmap::handle_type;
    using store_type  = typename Bitmap::chunk_store_type;
    using cursor      = typename store_type::cursor;
    using size_type   = std::size_t;

    static constexpr size_type chunk_capacity{ layout_type::low_domain_size };

    // Unsorted keys are probed one by one below this many (a probe costs a binary search of the chunk table and
    // a container lookup, a few tens of nanoseconds); above, the keys are sorted into a copy and merged.
    static constexpr size_type probe_limit{ 2048 };

    template <typename Range>
    [[nodiscard]] static bool probe_is_cheaper( Range const & keys ) {
        if constexpr ( std::ranges::sized_range<Range const> ) {
            return static_cast<size_type>( std::ranges::size( keys ) ) < probe_limit;
        } else {
            return static_cast<size_type>( std::ranges::distance( keys ) ) < probe_limit;
        }
    }

    // ---- operand -> source ----------------------------------------------------------------------------

    // Calls one of the visitor's overloads: v( source& ) for an operand presented as a chunk stream, and
    // v( unsorted_keys_tag, range ) for a range of keys in no particular order (an operation that cannot probe
    // sorts a copy through sorted_copy_visit and merges that).
    template <typename Operand, typename Visitor>
    static decltype( auto ) visit( Operand const & operand, Visitor && v ) {
        if constexpr ( std::same_as<Operand, key_range> ) {
            interval_source<layout_type, cow_policy> source{ operand };
            return v( source );
        } else if constexpr ( is_bit_words_v<Operand> ) {
            bits_source<layout_type, cow_policy> source{ operand.words.data(), operand.bit_count, operand.base };
            return v( source );
        } else if constexpr ( is_based_bits_v<Operand> ) {
            return visit_bits( operand.bits, operand.base, v );
        } else if constexpr ( bit_container<Operand> ) {
            return visit_bits( operand, std::uint64_t{ 0 }, v );
        } else if constexpr ( is_sorted_keys_v<Operand> ) {
            return visit_sorted( operand.keys, v );
        } else {
            static_assert( key_sequence<Operand> );
            if constexpr ( std::ranges::forward_range<Operand const> ) {
                return visit_keys( operand, v );
            } else {
                heap_vector<std::ranges::range_value_t<Operand const>> collected;
                for ( auto const element : operand ) { collected.push_back( element ); }
                return visit_keys( collected, v );
            }
        }
    }

    template <typename Bits, typename Visitor>
    static decltype( auto ) visit_bits( Bits const & bits, std::uint64_t const base, Visitor & v ) {
        if constexpr ( is_std_bitset_v<Bits> ) {
            constexpr std::size_t bit_count{ Bits{}.size() };
            constexpr std::size_t word_count{ ( bit_count + 63 ) / 64 };
            auto const fill{ [ & ]( std::uint64_t * const words ) {
                if ( std_bitset_is_bit_string<bit_count>() ) {
                    std::memcpy( words, &bits, std::min( sizeof( bits ), word_count * sizeof( std::uint64_t ) ) );
                } else {
                    for ( std::size_t i{ 0 }; i < bit_count; ++i ) {
                        if ( bits.test( i ) ) { words[ i / 64 ] |= std::uint64_t{ 1 } << ( i % 64 ); }
                    }
                }
            } };
            // a small bit-set is copied to the stack; a large one, to the heap
            if constexpr ( word_count <= 2048 ) {
                std::array<std::uint64_t, word_count> words{};
                fill( words.data() );
                bits_source<layout_type, cow_policy> source{ words.data(), bit_count, base };
                return v( source );
            } else {
                std::vector<std::uint64_t> words( word_count );
                fill( words.data() );
                bits_source<layout_type, cow_policy> source{ words.data(), bit_count, base };
                return v( source );
            }
        } else if constexpr ( block_bitset<Bits> ) {
            std::vector<typename Bits::block_type> blocks;
            if constexpr ( requires { bits.num_blocks(); } ) { blocks.reserve( static_cast<std::size_t>( bits.num_blocks() ) ); }
            to_block_range( bits, std::back_inserter( blocks ) );
            bits_source<layout_type, cow_policy> source{ blocks.data(), static_cast<std::uint64_t>( bits.size() ), base };
            return v( source );
        } else {
            using iterator = find_bits_iterator<Bits>;
            keys_source<layout_type, cow_policy, iterator, iterator> source{ iterator{ bits, base }, iterator::end_of( bits, base ) };
            return v( source );
        }
    }

    template <typename Range, typename Visitor>
    static decltype( auto ) visit_sorted( Range const & keys, Visitor & v ) {
        assert( is_ascending( keys ) && "a range declared sorted is not" );
        keys_source<layout_type, cow_policy, std::ranges::iterator_t<Range const>, std::ranges::sentinel_t<Range const>> source{
            std::ranges::begin( keys ), std::ranges::end( keys )
        };
        return v( source );
    }

    // Whether the keys are in ascending order (repeats allowed). Contiguous keys are checked a block at a time
    // without a branch per key, which the compiler turns into vector compares; the first block that is not
    // ascending ends the check.
    template <typename Range>
    [[nodiscard]] static bool is_ascending( Range const & keys ) {
        if constexpr ( std::ranges::contiguous_range<Range const> ) {
            auto const * const data{ std::ranges::data( keys ) };
            auto const count{ static_cast<size_type>( std::ranges::size( keys ) ) };
            constexpr size_type block{ 512 };
            for ( size_type at{ 0 }; at + 1 < count; ) {
                auto const end{ std::min( count - 1, at + block ) };
                unsigned descents{ 0 };
                for ( auto i{ at }; i < end; ++i ) { descents |= static_cast<unsigned>( data[ i + 1 ] < data[ i ] ); }
                if ( descents != 0 ) { return false; }
                at = end;
            }
            return true;
        } else {
            return std::ranges::is_sorted( keys );
        }
    }

    template <typename Range, typename Visitor>
    static decltype( auto ) visit_keys( Range const & keys, Visitor & v ) {
        if ( is_ascending( keys ) ) { return visit_sorted( keys, v ); }
        return v( unsorted_keys_tag{}, keys );
    }

    template <typename Range, typename Visitor>
    static decltype( auto ) sorted_copy_visit( Range const & keys, Visitor & v ) {
        heap_vector<std::ranges::range_value_t<Range const>> copy;
        if constexpr ( std::ranges::sized_range<Range const> ) { copy.reserve( static_cast<std::uint32_t>( std::ranges::size( keys ) ) ); }
        for ( auto const element : keys ) { copy.push_back( element ); }
        sort_keys( copy );
        return visit_sorted( copy, v );
    }

    template <typename E>
    [[nodiscard]] static constexpr bool in_domain( E const element ) noexcept {
        if constexpr ( std::numeric_limits<E>::digits <= std::numeric_limits<key_type>::digits ) {
            return true;
        } else {
            return element <= static_cast<E>( std::numeric_limits<key_type>::max() );
        }
    }

    // ---- container level --------------------------------------------------------------------------------

    [[nodiscard]] static bool is_full( handle_type const & piece ) noexcept { return piece.cardinality() == chunk_capacity; }

    // `lhs op rhs` as a legal container of the bitmap's container set.
    [[nodiscard]] static handle_type combine( Bitmap & self, handle_type const & lhs, handle_type const & rhs, set_operation const op ) {
        auto result{ Bitmap::combine_containers_for_policy( lhs, rhs, op ) };
        if ( op != set_operation::bit_or ) { result = Bitmap::demote_sparse_bitset( std::move( result ), self.chunks_ ); }
        assert( result.payload_is_unshared() && "a combine result aliases its operand" );
        return result;
    }

    // `slot = slot op piece` for the shrinking operations (and, and-not). A container of ours that nobody else
    // holds is changed in place where the result is a subset of it and the kernels of bitmap against bitmap
    // do so (bitset against bitset, an array filtered by a bitset or a run, small array against array, a bitset
    // lessened by an array); everything else is computed into fresh storage.
    static void combine_into( Bitmap & self, handle_type & slot, handle_type const & piece, set_operation const op ) {
        assert( op != set_operation::bit_or );
        if constexpr ( Bitmap::uses_default_container_set ) {
            if ( slot.payload_is_unshared() ) {
                if ( slot.holds_bitset() ) {
                    if ( piece.holds_bitset() ) {
                        combine_bitset_bitset_inplace<layout_type, cow_policy>( slot.as_bitset(), piece.as_bitset(), op );
                        slot = Bitmap::demote_sparse_bitset( std::move( slot ), self.chunks_ );
                        return;
                    }
                    if ( op == set_operation::bit_andnot && piece.holds_array() ) {
                        difference_bitset_array_inplace<layout_type, cow_policy>( slot.as_bitset(), piece.as_array() );
                        return;
                    }
                } else if ( slot.holds_array() ) {
                    auto const keep_matches{ op == set_operation::bit_and };
                    if ( piece.holds_bitset() ) {
                        auto array{ slot.as_array() };
                        filter_array_bitset_inplace<layout_type, cow_policy>( array, piece.as_bitset(), keep_matches );
                        finish_filtered( slot, array );
                        return;
                    }
                    if ( piece.holds_run() ) {
                        auto array{ slot.as_array() };
                        filter_array_run_inplace<layout_type, cow_policy>( array, piece.as_run(), keep_matches );
                        finish_filtered( slot, array );
                        return;
                    }
                    if ( op == set_operation::bit_and && piece.holds_array() && std::min( slot.count(), piece.count() ) <= kInplaceIntersectMaxElements ) {
                        intersect_array_array_inplace<layout_type, cow_policy>( slot.as_array(), piece.as_array() );
                        return;
                    }
                }
            }
        }
        slot = combine( self, slot, piece, op );
    }

    // The header of an array that was filtered in place; an array filtered to nothing becomes an empty container.
    template <typename ArrayRef>
    static void finish_filtered( handle_type & slot, ArrayRef & array ) noexcept {
        if ( array.values.empty() ) { slot = handle_type{}; } else { array.sync_header(); }
    }

    // An owning container of the contents of a piece (which views memory that is not ours).
    [[nodiscard]] static handle_type own( handle_type const & piece ) {
        handle_type result;
        switch ( piece.kind() ) {
            case container_kind::array : result = handle_type::make_array_from_sorted( { piece.as_array().values.data(), piece.as_array().values.size() } ); break;
            case container_kind::bitset: result = handle_type::make_bitset_from_words( piece.as_bitset().words.as_array() ); break;
            case container_kind::run   : return Bitmap::combine_containers_for_policy( handle_type{}, piece, set_operation::bit_or );
        }
        if constexpr ( !Bitmap::uses_default_container_set ) { result = Bitmap::optimize_container_for_policy( std::move( result ) ); }
        assert( result.payload_is_unshared() && "an owned copy aliases its piece" );
        return result;
    }

    // The piece is one run (an interval cut to a chunk): [first, last].
    [[nodiscard]] static bool is_single_run( handle_type const & piece ) noexcept { return piece.holds_run() && piece.count() == 1; }

    // Whether `slot` holds a value of [first, last]: a search of its sorted values or runs, a masked word scan.
    [[nodiscard]] static bool any_in_range( handle_type const & slot, low_type const first, low_type const last ) noexcept {
        switch ( slot.kind() ) {
            case container_kind::array: {
                auto const values{ slot.as_array().values };
                auto const at{ std::lower_bound( values.begin(), values.end(), first ) };
                return at != values.end() && *at <= last;
            }
            case container_kind::run: {
                auto const runs{ slot.as_run().runs };
                auto const at{ std::lower_bound( runs.begin(), runs.end(), first, []( auto const & r, low_type const value ) noexcept { return r.end < value; } ) };
                return at != runs.end() && at->begin <= last;
            }
            case container_kind::bitset: {
                auto const & words{ slot.as_bitset().words };
                auto const first_word{ static_cast<size_type>( first ) >> 6U };
                auto const last_word { static_cast<size_type>( last  ) >> 6U };
                for ( auto w{ first_word }; w <= last_word; ++w ) {
                    auto mask{ ~std::uint64_t{ 0 } };
                    if ( w == first_word ) { mask &= ~std::uint64_t{ 0 } << ( static_cast<unsigned>( first ) & 63U ); }
                    if ( w == last_word  ) { mask &= ~std::uint64_t{ 0 } >> ( 63U - ( static_cast<unsigned>( last ) & 63U ) ); }
                    if ( ( words[ w ] & mask ) != 0 ) { return true; }
                }
                return false;
            }
        }
        std::unreachable();
    }

    // How many values of `slot` are in [first, last].
    [[nodiscard]] static size_type count_in_range( handle_type const & slot, low_type const first, low_type const last ) noexcept {
        switch ( slot.kind() ) {
            case container_kind::array: {
                auto const values{ slot.as_array().values };
                auto const from{ std::lower_bound( values.begin(), values.end(), first ) };
                return static_cast<size_type>( std::upper_bound( from, values.end(), last ) - from );
            }
            case container_kind::run: {
                auto const runs{ slot.as_run().runs };
                size_type count{ 0 };
                for ( auto at{ std::lower_bound( runs.begin(), runs.end(), first, []( auto const & r, low_type const value ) noexcept { return r.end < value; } ) };
                      at != runs.end() && at->begin <= last; ++at ) {
                    count += static_cast<size_type>( std::min( at->end, last ) - std::max( at->begin, first ) ) + 1U;
                }
                return count;
            }
            case container_kind::bitset: {
                auto const & words{ slot.as_bitset().words };
                auto const first_word{ static_cast<size_type>( first ) >> 6U };
                auto const last_word { static_cast<size_type>( last  ) >> 6U };
                size_type count{ 0 };
                for ( auto w{ first_word }; w <= last_word; ++w ) {
                    auto mask{ ~std::uint64_t{ 0 } };
                    if ( w == first_word ) { mask &= ~std::uint64_t{ 0 } << ( static_cast<unsigned>( first ) & 63U ); }
                    if ( w == last_word  ) { mask &= ~std::uint64_t{ 0 } >> ( 63U - ( static_cast<unsigned>( last ) & 63U ) ); }
                    count += static_cast<size_type>( std::popcount( words[ w ] & mask ) );
                }
                return count;
            }
        }
        std::unreachable();
    }

    // |lhs ∩ rhs|
    [[nodiscard]] static size_type and_card( handle_type const & lhs, handle_type const & rhs ) noexcept {
        if ( is_single_run( rhs ) ) {
            auto const run{ rhs.as_run().runs.front() };
            return count_in_range( lhs, run.begin, run.end );
        }
        if ( lhs.holds_array() && rhs.holds_array() ) {
            auto const left { lhs.as_array().values };
            auto const right{ rhs.as_array().values };
            auto const * const small{ left.size() <= right.size() ? &left : &right };
            auto const * const large{ left.size() <= right.size() ? &right : &left };
            size_type count{ 0 };
            if ( small->size() * 16 < large->size() ) {
                for ( auto const value : *small ) { count += std::binary_search( large->begin(), large->end(), value ); }
                return count;
            }
#if defined( __SSE4_2__ )
            if constexpr ( std::is_same_v<low_type, std::uint16_t> ) {
                if ( small->size() >= sse42_intersect_min_side ) {
                    return intersect_cardinality_array_array_sse42( left.data(), left.size(), right.data(), right.size() );
                }
            }
#endif
            auto li{ left.begin() };
            auto ri{ right.begin() };
            while ( li != left.end() && ri != right.end() ) {
                count += *li == *ri;
                auto const advance_left { *li <= *ri };
                auto const advance_right{ *ri <= *li };
                li += advance_left;
                ri += advance_right;
            }
            return count;
        }
        return container_intersect_cardinality<layout_type, cow_policy>( lhs, rhs );
    }

    // Whether every value of [first, last] is in `slot`, stopping at the first one that is not.
    [[nodiscard]] static bool covers_range( handle_type const & slot, low_type const first, low_type const last ) noexcept {
        auto const length{ static_cast<size_type>( last - first ) + 1U };
        if ( slot.cardinality() < length ) { return false; }
        switch ( slot.kind() ) {
            case container_kind::array: {
                // sorted without repeats: the range is all there iff the value `length - 1` places on is `last`
                auto const values{ slot.as_array().values };
                auto const at{ static_cast<size_type>( std::lower_bound( values.begin(), values.end(), first ) - values.begin() ) };
                return at + length <= values.size() && values[ at ] == first && values[ at + length - 1 ] == last;
            }
            case container_kind::run: {
                auto const runs{ slot.as_run().runs };
                auto const run{ std::lower_bound( runs.begin(), runs.end(), first, []( auto const & r, low_type const value ) noexcept { return r.end < value; } ) };
                return run != runs.end() && run->begin <= first && last <= run->end;
            }
            case container_kind::bitset: {
                auto const & words{ slot.as_bitset().words };
                auto const first_word{ static_cast<size_type>( first ) >> 6U };
                auto const last_word { static_cast<size_type>( last  ) >> 6U };
                for ( auto w{ first_word }; w <= last_word; ++w ) {
                    auto mask{ ~std::uint64_t{ 0 } };
                    if ( w == first_word ) { mask &= ~std::uint64_t{ 0 } << ( static_cast<unsigned>( first ) & 63U ); }
                    if ( w == last_word  ) { mask &= ~std::uint64_t{ 0 } >> ( 63U - ( static_cast<unsigned>( last ) & 63U ) ); }
                    if ( ( words[ w ] & mask ) != mask ) { return false; }
                }
                return true;
            }
        }
        std::unreachable();
    }

    // Whether every value of `piece` is in `slot`, stopping at the first one that is not.
    [[nodiscard]] static bool covers( handle_type const & slot, handle_type const & piece ) noexcept {
        if ( piece.cardinality() > slot.cardinality() ) { return false; }
        switch ( piece.kind() ) {
            case container_kind::run:
                for ( auto const & run : piece.as_run().runs ) {
                    if ( !covers_range( slot, run.begin, run.end ) ) { return false; }
                }
                return true;
            case container_kind::array:
                for ( auto const value : piece.as_array().values ) {
                    if ( !container_contains<layout_type, cow_policy>( slot, value ) ) { return false; }
                }
                return true;
            case container_kind::bitset:
                if ( slot.holds_bitset() ) {
                    auto const & mine{ slot.as_bitset().words };
                    auto const & theirs{ piece.as_bitset().words };
                    for ( size_type w{ 0 }; w < layout_type::word_count; ++w ) {
                        if ( ( theirs[ w ] & ~mine[ w ] ) != 0 ) { return false; }
                    }
                    return true;
                }
                return and_card( slot, piece ) == piece.cardinality();
        }
        std::unreachable();
    }

    // The smallest and largest value of a non-empty container, read without touching its header (a bitset's
    // endpoints are maintained lazily, which a read of a shared bitmap must not do).
    [[nodiscard]] static std::pair<low_type, low_type> bounds_of( handle_type const & container ) noexcept {
        switch ( container.kind() ) {
            case container_kind::array: { auto const values{ container.as_array().values }; return { values.front(), values.back() }; }
            case container_kind::run  : { auto const runs{ container.as_run().runs }; return { runs.front().begin, runs.back().end }; }
            case container_kind::bitset: {
                auto const & words{ container.as_bitset().words };
                size_type low{ 0 };
                while ( words[ low ] == 0 ) { ++low; }
                size_type high{ layout_type::word_count - 1 };
                while ( words[ high ] == 0 ) { --high; }
                return {
                    static_cast<low_type>( low * 64 + static_cast<size_type>( std::countr_zero( words[ low ] ) ) ),
                    static_cast<low_type>( high * 64 + 63 - static_cast<size_type>( std::countl_zero( words[ high ] ) ) )
                };
            }
        }
        std::unreachable();
    }

    // Whether every value of `slot` is in `piece`.
    [[nodiscard]] static bool within( handle_type const & slot, handle_type const & piece ) noexcept {
        if ( slot.cardinality() > piece.cardinality() ) { return false; }
        if ( piece.holds_run() && piece.count() == 1 ) {
            // one run (an interval): the bounds of the container decide
            auto const run{ piece.as_run().runs.front() };
            auto const [ low, high ]{ bounds_of( slot ) };
            return run.begin <= low && high <= run.end;
        }
        return and_card( slot, piece ) == slot.cardinality();
    }

    // ---- the merge walk ---------------------------------------------------------------------------------

    // Calls `visit( cursor, piece )` for each pair of a chunk of `self` and a non-empty piece of the same key,
    // in ascending order, until it returns false. Either side jumps over the other's absent keys.
    template <typename Self, typename Source, typename Visit>
    static void leapfrog( Self & self, Source & source, Visit && visit ) {
        auto at{ self.chunks_.begin() };
        while ( at != self.chunks_.end() && !source.at_end() ) {
            auto const self_key  { self.chunks_.key( at ) };
            auto const source_key{ source.key() };
            if ( self_key < source_key ) {
                at = self.chunks_.lower_bound( source_key, at );
            } else if ( source_key < self_key ) {
                source.seek( self_key );
            } else {
                auto const & piece{ source.piece() };
                if ( piece.cardinality() != 0 && !visit( at, piece ) ) { return; }
                ++at;
                source.advance();
            }
        }
    }

    // ---- stream operations ------------------------------------------------------------------------------

    template <typename Source>
    static void and_stream( Bitmap & self, Source & source ) {
        size_type new_size{ 0 };
        typename store_type::rewriter rewrite{ self.chunks_ };
        leapfrog( self, source, [ & ]( cursor const at, handle_type const & piece ) {
            auto & slot{ self.chunks_.slot( at ) };
            if ( is_full( piece ) ) {
                new_size += slot.cardinality();
                rewrite.keep( at );
            } else {
                combine_into( self, slot, piece, set_operation::bit_and );
                if ( slot.cardinality() != 0 ) {
                    new_size += slot.cardinality();
                    rewrite.keep( at );
                }
            }
            return true;
        } );
        rewrite.finish();
        self.size_ = new_size;
    }

    template <typename Source>
    static void andnot_stream( Bitmap & self, Source & source ) {
        bool any_empty{ false };
        leapfrog( self, source, [ & ]( cursor const at, handle_type const & piece ) {
            auto & slot{ self.chunks_.slot( at ) };
            auto const before{ slot.cardinality() };
            if ( is_full( piece ) ) { slot = handle_type{}; } else { combine_into( self, slot, piece, set_operation::bit_andnot ); }
            self.size_ -= before - slot.cardinality();
            any_empty |= slot.cardinality() == 0;
            return true;
        } );
        if ( any_empty ) { drop_empty_chunks( self ); }
    }

    template <typename Source>
    static void or_stream( Bitmap & self, Source & source ) {
        std::vector<std::pair<chunk_type, handle_type>> added;
        size_type added_size{ 0 };
        small_array_values<low_type> union_scratch;
        auto at{ self.chunks_.begin() };
        while ( !source.at_end() ) {
            auto const key{ source.key() };
            if ( at != self.chunks_.end() && self.chunks_.key( at ) < key ) { at = self.chunks_.lower_bound( key, at ); }
            auto const & piece{ source.piece() };
            if ( auto const cardinality{ piece.cardinality() }; cardinality != 0 ) {
                if ( at != self.chunks_.end() && self.chunks_.key( at ) == key ) {
                    auto & slot{ self.chunks_.slot( at ) };
                    auto const before{ slot.cardinality() };
                    if ( is_full( piece ) ) { slot = own( piece ); } else { self.union_same_key( self.chunks_, slot, piece, false, union_scratch ); }
                    self.size_ += slot.cardinality() - before;
                    ++at;
                } else {
                    added.emplace_back( key, own( piece ) );
                    added_size += cardinality;
                }
            }
            source.advance();
        }
        self.size_ += added_size;
        insert_added( self, added );
    }

    // `container ^ piece` for a piece that is an interval cut to its chunk (one run): the flip of that range of the
    // container, which every container form has a kernel for. An absent chunk is an empty container. So an interval
    // is flipped in one pass over the chunks it covers, whatever the chunks hold.
    [[nodiscard]] static handle_type flip_piece( handle_type && container, handle_type const & piece ) {
        auto const run{ piece.as_run().runs.front() };
        // an array may hold as many values as the bitmap's own limit: the flip kernels take the first cardinality
        // that is a bitset
        auto result{ apply_range_to_container<layout_type, cow_policy, typename Bitmap::run_selection_policy>(
            std::move( container ), run.begin, run.end, range_operation::flip, Bitmap::array_to_bitset_threshold + 1U ) };
        if constexpr ( !Bitmap::uses_default_container_set ) { result = Bitmap::optimize_container_for_policy( std::move( result ) ); }
        return result;
    }

    template <typename Source>
    static void xor_stream( Bitmap & self, Source & source ) {
        std::vector<std::pair<chunk_type, handle_type>> added;
        size_type added_size{ 0 };
        bool any_empty{ false };
        auto at{ self.chunks_.begin() };
        while ( !source.at_end() ) {
            auto const key{ source.key() };
            if ( at != self.chunks_.end() && self.chunks_.key( at ) < key ) { at = self.chunks_.lower_bound( key, at ); }
            auto const & piece{ source.piece() };
            if ( auto const cardinality{ piece.cardinality() }; cardinality != 0 ) {
                if ( at != self.chunks_.end() && self.chunks_.key( at ) == key ) {
                    auto & slot{ self.chunks_.slot( at ) };
                    assert( slot.cardinality() != 0 );
                    self.size_ -= slot.cardinality();
                    if constexpr ( is_interval_source_v<Source> ) { slot = flip_piece( std::move( slot ), piece ); } else { self.xor_same_key( self.chunks_, slot, piece ); }
                    self.size_ += slot.cardinality();
                    any_empty |= slot.cardinality() == 0;
                    ++at;
                } else {
                    if constexpr ( is_interval_source_v<Source> ) { added.emplace_back( key, flip_piece( handle_type{}, piece ) ); } else { added.emplace_back( key, own( piece ) ); }
                    added_size += cardinality;
                }
            }
            source.advance();
        }
        self.size_ += added_size;
        if ( any_empty ) { drop_empty_chunks( self ); }
        insert_added( self, added );
    }

    template <typename Source>
    [[nodiscard]] static size_type xor_cardinality_stream( Bitmap const & self, Source & source ) {
        size_type common{ 0 };
        size_type total { 0 };
        auto at{ self.chunks_.begin() };
        while ( !source.at_end() ) {
            auto const key{ source.key() };
            auto const & piece{ source.piece() };
            if ( auto const cardinality{ piece.cardinality() }; cardinality != 0 ) {
                total += cardinality;
                if ( at != self.chunks_.end() && self.chunks_.key( at ) < key ) { at = self.chunks_.lower_bound( key, at ); }
                if ( at != self.chunks_.end() && self.chunks_.key( at ) == key ) {
                    auto const & slot{ self.chunks_.slot( at ) };
                    common += is_full( piece ) ? size_type{ slot.cardinality() } : and_card( slot, piece );
                }
            }
            source.advance();
        }
        return self.size_ + total - 2 * common;
    }

    template <typename Source>
    [[nodiscard]] static bool intersects_stream( Bitmap const & self, Source & source ) {
        bool found{ false };
        leapfrog( self, source, [ & ]( cursor const at, handle_type const & piece ) {
            auto const & slot{ self.chunks_.slot( at ) };
            if ( is_full( piece ) ) {
                found = true;
            } else if ( is_single_run( piece ) ) {
                auto const run{ piece.as_run().runs.front() };
                found = any_in_range( slot, run.begin, run.end );
            } else {
                found = container_intersects<layout_type, cow_policy>( slot, piece );
            }
            return !found;
        } );
        return found;
    }

    template <typename Source>
    [[nodiscard]] static size_type and_cardinality_stream( Bitmap const & self, Source & source ) {
        size_type count{ 0 };
        leapfrog( self, source, [ & ]( cursor const at, handle_type const & piece ) {
            auto const & slot{ self.chunks_.slot( at ) };
            count += is_full( piece ) ? size_type{ slot.cardinality() } : and_card( slot, piece );
            return true;
        } );
        return count;
    }

    // Whether the operand holds every key of `self`.
    template <typename Source>
    [[nodiscard]] static bool subset_stream( Bitmap const & self, Source & source ) {
        auto at{ self.chunks_.begin() };
        while ( at != self.chunks_.end() ) {
            if ( source.at_end() ) { return false; }
            auto const self_key  { self.chunks_.key( at ) };
            auto const source_key{ source.key() };
            if ( source_key < self_key ) { source.seek( self_key ); continue; }
            if ( self_key < source_key ) { return false; }
            auto const & piece{ source.piece() };
            if ( piece.cardinality() == 0 ) { source.advance(); continue; }
            auto const & slot{ self.chunks_.slot( at ) };
            if ( !is_full( piece ) && !within( slot, piece ) ) { return false; }
            ++at;
            source.advance();
        }
        return true;
    }

    // Whether `self` holds every key of the operand; the operand's total cardinality is added to `total`
    // while it holds.
    template <typename Source>
    [[nodiscard]] static bool superset_stream( Bitmap const & self, Source & source, size_type & total ) {
        auto at{ self.chunks_.begin() };
        while ( !source.at_end() ) {
            auto const key{ source.key() };
            auto const & piece{ source.piece() };
            auto const cardinality{ piece.cardinality() };
            if ( cardinality == 0 ) { source.advance(); continue; }
            if ( at != self.chunks_.end() && self.chunks_.key( at ) < key ) { at = self.chunks_.lower_bound( key, at ); }
            if ( at == self.chunks_.end() || self.chunks_.key( at ) != key ) { return false; }
            auto const & slot{ self.chunks_.slot( at ) };
            if ( is_full( piece ) ? slot.cardinality() != chunk_capacity : !covers( slot, piece ) ) { return false; }
            total += cardinality;
            ++at;
            source.advance();
        }
        return !source.truncated();
    }

    // Removes the chunks whose container went empty.
    static void drop_empty_chunks( Bitmap & self ) {
        typename store_type::rewriter rewrite{ self.chunks_ };
        for ( auto at{ self.chunks_.begin() }; at != self.chunks_.end(); ++at ) {
            if ( self.chunks_.slot( at ).cardinality() != 0 ) { rewrite.keep( at ); }
        }
        rewrite.finish();
    }

    // Puts the chunks an or_stream collected (ascending, none present) into the table.
    static void insert_added( Bitmap & self, std::vector<std::pair<chunk_type, handle_type>> & added ) {
        if ( added.empty() ) { return; }
        if ( added.size() <= 8 ) {
            for ( auto & [ key, container ] : added ) {
                self.chunks_.insert( self.chunks_.lower_bound( key ), key, std::move( container ) );
            }
            return;
        }
        store_type merged;
        merged.reserve( self.chunks_.size() + added.size() );
        auto at{ self.chunks_.begin() };
        std::size_t next{ 0 };
        while ( at != self.chunks_.end() || next != added.size() ) {
            if ( next == added.size() || ( at != self.chunks_.end() && self.chunks_.key( at ) < added[ next ].first ) ) {
                merged.push_back( self.chunks_.key( at ), std::move( self.chunks_.slot( at ) ) );
                ++at;
            } else {
                merged.push_back( added[ next ].first, std::move( added[ next ].second ) );
                ++next;
            }
        }
        self.chunks_ = std::move( merged );
    }

    // ---- probe operations (unsorted keys) ---------------------------------------------------------------

    template <typename Range>
    [[nodiscard]] static heap_vector<key_type> contained_keys( Bitmap const & self, Range const & keys ) {
        heap_vector<key_type> hits;
        for ( auto const element : keys ) {
            if ( in_domain( element ) && self.contains( static_cast<key_type>( element ) ) ) { hits.push_back( static_cast<key_type>( element ) ); }
        }
        std::ranges::sort( hits );
        return hits;
    }

    template <typename Range>
    static void and_probe( Bitmap & self, Range const & keys ) {
        if ( self.empty() ) { return; }
        auto const hits{ contained_keys( self, keys ) };
        Bitmap result;
        result.add_sorted_many( { hits.data(), hits.size() } );
        self.swap( result );
    }

    template <typename Range>
    [[nodiscard]] static bool intersects_probe( Bitmap const & self, Range const & keys ) {
        if ( self.empty() ) { return false; }
        for ( auto const element : keys ) {
            if ( in_domain( element ) && self.contains( static_cast<key_type>( element ) ) ) { return true; }
        }
        return false;
    }

    template <typename Range>
    [[nodiscard]] static size_type and_cardinality_probe( Bitmap const & self, Range const & keys ) {
        if ( self.empty() ) { return 0; }
        auto hits{ contained_keys( self, keys ) };
        return static_cast<size_type>( std::unique( hits.begin(), hits.end() ) - hits.begin() );
    }

    template <typename Range>
    [[nodiscard]] static bool superset_probe( Bitmap const & self, Range const & keys ) {
        for ( auto const element : keys ) {
            if ( !in_domain( element ) || !self.contains( static_cast<key_type>( element ) ) ) { return false; }
        }
        return true;
    }

    // ---- the operations ---------------------------------------------------------------------------------

    template <typename Operand>
    static void and_with( Bitmap & self, Operand const & operand ) {
        visit( operand, overloaded{
            [ & ]( auto & source ) { and_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                if ( probe_is_cheaper( keys ) ) {
                    and_probe( self, keys );
                } else {
                    auto merge{ overloaded{ [ & ]( auto & source ) { and_stream( self, source ); } } };
                    sorted_copy_visit( keys, merge );
                }
            }
        } );
    }

    template <typename Operand>
    static void or_with( Bitmap & self, Operand const & operand ) {
        visit( operand, overloaded{
            [ & ]( auto & source ) { or_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                auto merge{ overloaded{ [ & ]( auto & source ) { or_stream( self, source ); } } };
                sorted_copy_visit( keys, merge );
            }
        } );
    }

    template <typename Operand>
    static void andnot_with( Bitmap & self, Operand const & operand ) {
        visit( operand, overloaded{
            [ & ]( auto & source ) { andnot_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                auto merge{ overloaded{ [ & ]( auto & source ) { andnot_stream( self, source ); } } };
                sorted_copy_visit( keys, merge );
            }
        } );
    }

    template <typename Operand>
    static void xor_with( Bitmap & self, Operand const & operand ) {
        visit( operand, overloaded{
            [ & ]( auto & source ) { xor_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                auto merge{ overloaded{ [ & ]( auto & source ) { xor_stream( self, source ); } } };
                sorted_copy_visit( keys, merge );
            }
        } );
    }

    template <typename Operand>
    [[nodiscard]] static size_type xor_cardinality( Bitmap const & self, Operand const & operand ) {
        return visit( operand, overloaded{
            [ & ]( auto & source ) { return xor_cardinality_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                auto merge{ overloaded{ [ & ]( auto & source ) { return xor_cardinality_stream( self, source ); } } };
                return sorted_copy_visit( keys, merge );
            }
        } );
    }

    template <typename Operand>
    [[nodiscard]] static bool intersects( Bitmap const & self, Operand const & operand ) {
        return visit( operand, overloaded{
            [ & ]( auto & source ) { return intersects_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) { return intersects_probe( self, keys ); }
        } );
    }

    template <typename Operand>
    [[nodiscard]] static size_type and_cardinality( Bitmap const & self, Operand const & operand ) {
        return visit( operand, overloaded{
            [ & ]( auto & source ) { return and_cardinality_stream( self, source ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) {
                if ( probe_is_cheaper( keys ) ) { return and_cardinality_probe( self, keys ); }
                auto merge{ overloaded{ [ & ]( auto & source ) { return and_cardinality_stream( self, source ); } } };
                return sorted_copy_visit( keys, merge );
            }
        } );
    }

    template <typename Operand>
    [[nodiscard]] static bool is_subset_of( Bitmap const & self, Operand const & operand ) {
        if ( self.empty() ) { return true; }
        auto visitor{ overloaded{ [ & ]( auto & source ) { return subset_stream( self, source ); } } };
        if constexpr ( key_sequence<Operand> ) {
            // the operand has to be seen as a set: an unsorted range is sorted (a copy) first
            return visit( operand, overloaded{
                [ & ]( auto & source ) { return subset_stream( self, source ); },
                [ & ]( unsorted_keys_tag, auto const & keys ) {
                    auto sorted_visitor{ overloaded{ [ & ]( auto & source ) { return subset_stream( self, source ); } } };
                    return sorted_copy_visit( keys, sorted_visitor );
                }
            } );
        } else {
            return visit( operand, visitor );
        }
    }

    template <typename Operand>
    [[nodiscard]] static bool is_superset_of( Bitmap const & self, Operand const & operand ) {
        return visit( operand, overloaded{
            [ & ]( auto & source ) { size_type total{ 0 }; return superset_stream( self, source, total ); },
            [ & ]( unsorted_keys_tag, auto const & keys ) { return superset_probe( self, keys ); }
        } );
    }

    template <typename Operand>
    [[nodiscard]] static bool equals( Bitmap const & self, Operand const & operand ) {
        auto const test{ [ & ]( auto & source ) {
            size_type total{ 0 };
            return superset_stream( self, source, total ) && total == self.size_;
        } };
        if constexpr ( key_sequence<Operand> ) {
            return visit( operand, overloaded{
                test,
                [ & ]( unsorted_keys_tag, auto const & keys ) {
                    auto sorted_visitor{ overloaded{ test } };
                    return sorted_copy_visit( keys, sorted_visitor );
                }
            } );
        } else {
            return visit( operand, overloaded{ test } );
        }
    }

    // against another bitmap

    [[nodiscard]] static bool intersects( Bitmap const & self, Bitmap const & other ) {
        store_source source{ other.chunks_ };
        return intersects_stream( self, source );
    }
    [[nodiscard]] static size_type and_cardinality( Bitmap const & self, Bitmap const & other ) {
        store_source source{ other.chunks_ };
        return and_cardinality_stream( self, source );
    }
    [[nodiscard]] static bool is_subset_of( Bitmap const & self, Bitmap const & other ) {
        if ( self.size_ > other.size_ ) { return false; }
        store_source source{ other.chunks_ };
        return subset_stream( self, source );
    }
};

} // namespace frsr::roaring::detail
