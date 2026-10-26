#pragma once

// Out-of-line definitions of frsr::roaring::bitmap's serialization members.
// Three formats: the standard Roaring portable format (portable_serialize /
// portable_deserialize, container-level and interoperable with CRoaring), the
// add()-replay format (serialize_to_vm_vector / deserialize_from_vm_vector) and
// the zero-copy frozen format
// (serialize_frozen_to_vm_vector + frozen_view + materialize). The nested
// wire-format types (serialized_header, frozen_header, frozen_chunk_index,
// frozen_container_kind, frozen_view), the magic constants and the payload-layout
// helpers stay declared in the bitmap class; only the function bodies live here.
//
// Includes bitmap.hpp so it is self-contained; the include cycle is broken by
// #pragma once (bitmap.hpp pulls this in after the class is complete).

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/run_container.hpp>

#include <cstddef>
#include <cstdint>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace frsr::roaring {

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::serialized_size_bytes() const noexcept {
    return sizeof( serialized_header ) + ( static_cast<std::size_t>( size_ ) * sizeof( key_type ) );
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_size_bytes( frozen_options const options ) const {
    // No container is empty (validate()), so the index holds one entry per chunk.
    auto offset{ sizeof( frozen_header ) + chunks_.size() * sizeof( frozen_chunk_index ) };
    for ( auto const & slot : chunks_.slots() ) {
        auto const [alignment, payload_size] = frozen_payload_layout( slot );
        offset = align_up( offset, alignment );
        offset += payload_size;
        if ( options.ranked ) {
            // before the next align_up: the rank block sits between this payload and the next
            offset += frozen_rank_bytes( slot );
        }
    }
    return offset;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
void bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::serialize_to_vm_vector( serialized_byte_vector & out ) const {
    auto const values{ to_vector() };
    auto const byte_count{
        static_cast<std::uint32_t>( sizeof( serialized_header ) + values.size() * sizeof( key_type ) )
    };
#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)
    if ( out.has_attached_storage() ) {
        // Pre-mapped storage (e.g. file-backed via persistent_bitmap::store):
        // size within the existing mapping instead of detaching it for RAM.
        out.resize( byte_count, psi::vm::no_init );
    } else {
        out.map_memory( byte_count );
    }
#else
    out.resize( byte_count );
#endif
    auto * const data{ out.data() };
    serialized_header const header{
        .magic = serialization_magic,
        .key_bits = static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits ),
        .reserved = 0,
        .value_count = static_cast<std::uint64_t>( values.size() )
    };
    std::memcpy( data, &header, sizeof( header ) );
    if ( !values.empty() ) {
        std::memcpy( data + sizeof( header ), values.data(), values.size() * sizeof( key_type ) );
    }
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
void bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::serialize_frozen_to_vm_vector( serialized_byte_vector & out, frozen_options const options ) const {
    auto const byte_count{ static_cast<std::uint32_t>( frozen_size_bytes( options ) ) };
#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)
    if ( out.has_attached_storage() ) {
        // Pre-mapped storage (e.g. file-backed via persistent_bitmap::store):
        // size within the existing mapping instead of detaching it for RAM.
        out.resize( byte_count, psi::vm::no_init );
    } else {
        out.map_memory( byte_count );
    }
#else
    out.resize( byte_count );
#endif
    frozen_serialize( std::span{ out.data(), out.size() }, options );
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_serialize( std::span<std::byte> const out, frozen_options const options ) const {
    assert( out.size() >= frozen_size_bytes( options ) );
    assert( reinterpret_cast<std::uintptr_t>( out.data() ) % alignof( std::uint64_t ) == 0 );
    assert( ( !options.ranked || ( std::numeric_limits<key_type>::digits <= 32 && size_ < ( std::size_t{ 1 } << 32 ) ) ) && "a ranked image holds a key type of at most 32 bits and fewer than 2^32 values" );

    auto * const data{ out.data() };
    auto const chunk_count{ static_cast<std::uint32_t>( chunks_.size() ) };

    // Every byte of the output is written, padding included: the destination
    // need not be zero-filled.
    std::memset( data, 0, sizeof( frozen_header ) );
    auto & header{ *reinterpret_cast<frozen_header *>( data ) };
    header.magic       = frozen_serialization_magic;
    header.key_bits    = static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits );
    header.version     = options.ranked ? frozen_ranked_format_version : frozen_format_version;
    header.chunk_count = chunk_count;
    header.value_count = static_cast<std::uint64_t>( size_ );

    auto * const index{
        reinterpret_cast<frozen_chunk_index *>( data + sizeof( frozen_header ) )
    };
    std::size_t payload_offset{ sizeof( frozen_header ) + std::size_t{ chunk_count } * sizeof( frozen_chunk_index ) };
    std::uint32_t frozen_index{};
    std::uint32_t base{};
    for ( auto chunk_at{ chunks_.begin() }; chunk_at != chunks_.end(); ++chunk_at ) {
        auto const & container{ chunks_.slot( chunk_at ) };

        assert( container.cardinality() != 0 );
        auto const [alignment, payload_size] = frozen_payload_layout( container );
        auto const aligned_offset{ align_up( payload_offset, alignment ) };
        std::memset( data + payload_offset, 0, aligned_offset - payload_offset );
        payload_offset = aligned_offset;

        frozen_container_kind kind;
        std::uint32_t payload_count;
        std::uint32_t cardinality;
        auto * const payload{ data + payload_offset };
        switch ( container.kind() ) {
            case detail::container_kind::array: {
                auto const array{ container.as_array() };
                kind          = frozen_container_kind::array;
                payload_count = static_cast<std::uint32_t>( array.values.size() );
                cardinality   = payload_count;
                std::memcpy( payload, array.values.data(), payload_size );
                break;
            }
            case detail::container_kind::run: {
                auto const run{ container.as_run() };
                kind          = frozen_container_kind::run;
                payload_count = static_cast<std::uint32_t>( run.runs.size() );
                cardinality   = static_cast<std::uint32_t>( container.cardinality() );
                std::memcpy( payload, run.runs.data(), payload_size );
                if ( options.ranked ) {
                    auto * const ranks{ reinterpret_cast<std::uint16_t *>( payload + payload_size ) };
                    std::uint32_t before{};
                    for ( std::uint32_t i{ 0 }; i < payload_count; ++i ) {
                        ranks[ i ] = static_cast<std::uint16_t>( before );   // at most 65535: the last run holds at least one value
                        before += static_cast<std::uint32_t>( run.runs[ i ].end - run.runs[ i ].begin ) + 1U;
                    }
                }
                break;
            }
            case detail::container_kind::bitset: {
                auto const bitset{ container.as_bitset() };
                kind          = frozen_container_kind::bitset;
                payload_count = static_cast<std::uint32_t>( layout_type::word_count );
                cardinality   = static_cast<std::uint32_t>( container.cardinality() );
                std::memcpy( payload, bitset.words.as_array().data(), payload_size );
                if ( options.ranked ) {
                    auto const * const words{ bitset.words.as_array().data() };
                    auto * const ranks{ reinterpret_cast<std::uint16_t *>( payload + payload_size ) };
                    std::uint32_t before{};
                    for ( std::uint32_t i{ 0 }; i < payload_count; ++i ) {
                        ranks[ i ] = static_cast<std::uint16_t>( before );   // at most 64 * 1023
                        before += static_cast<std::uint32_t>( std::popcount( words[ i ] ) );
                    }
                }
                break;
            }
        }
        index[ frozen_index++ ] = frozen_chunk_index{
            .chunk          = static_cast<std::uint64_t>( chunks_.key( chunk_at ) ),
            .payload_offset = static_cast<std::uint32_t>( payload_offset ),
            .payload_bytes  = static_cast<std::uint32_t>( payload_size ),
            .payload_count  = payload_count,
            .cardinality    = cardinality,
            .container_kind = static_cast<std::uint8_t>( kind ),
            .base           = options.ranked ? base : 0U
        };
        base += cardinality;
        payload_offset += payload_size;
        if ( options.ranked ) {
            payload_offset += frozen_rank_bytes( container );
        }
    }
    return payload_offset;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy> bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::deserialize_from_vm_vector( serialized_byte_vector const & in ) {
    if ( in.size() < sizeof( serialized_header ) ) {
        return {};
    }
    serialized_header header{};
    std::memcpy( &header, in.data(), sizeof( header ) );
    if (
        header.magic != serialization_magic ||
        header.key_bits != static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits )
    ) {
        return {};
    }

    auto const payload_size{ static_cast<std::size_t>( header.value_count ) * sizeof( key_type ) };
    if ( in.size() != sizeof( serialized_header ) + payload_size ) {
        return {};
    }

    detail::heap_vector<key_type> values;
    values.resize( static_cast<std::uint32_t>( header.value_count ) );
    if ( !values.empty() ) {
        std::memcpy( values.data(), in.data() + sizeof( serialized_header ), payload_size );
    }
    bitmap result;
    result.add_many_sorted( std::span<key_type const>{ values.data(), values.size() } );
    return result;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
typename bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view_from_vm_vector( serialized_byte_vector const & in ) {
    return frozen_view_from( std::span<std::byte const>{ in.data(), in.size() } );
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
typename bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view_from( std::span<std::byte const> const in ) {
    if ( in.size() < sizeof( frozen_header ) ) {
        return {};
    }
    assert( reinterpret_cast<std::uintptr_t>( in.data() ) % alignof( std::uint64_t ) == 0 && "the image must be 8-byte aligned" );

    auto const * const data{ in.data() };
    auto const * const header{ reinterpret_cast<frozen_header const *>( data ) };
    if (
        header->magic != frozen_serialization_magic ||
        header->key_bits != static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits ) ||
        ( header->version != frozen_format_version && header->version != frozen_ranked_format_version )
    ) {
        return {};
    }
    auto const ranked{ header->version == frozen_ranked_format_version };
    // The rank data is 16 bits per count and a rank is 32 bits, with one value of that range left for "absent".
    if ( ranked && ( std::numeric_limits<key_type>::digits > 32 || header->value_count >= ( std::uint64_t{ 1 } << 32 ) ) ) {
        return {};
    }

    auto const index_offset{ sizeof( frozen_header ) };
    auto const index_bytes{ static_cast<std::size_t>( header->chunk_count ) * sizeof( frozen_chunk_index ) };
    if ( in.size() < index_offset + index_bytes ) {
        return {};
    }

    auto const * const index{
        reinterpret_cast<frozen_chunk_index const *>( data + index_offset )
    };
    // Where the next payload may start: the end of the index, and, in a ranked image, where the previous chunk's
    // payload and rank block end (the payloads are in index order).
    std::uint64_t payload_floor{ index_offset + index_bytes };
    // The end of the last payload (and rank block): what frozen_view::byte_size() reports for either version.
    std::uint64_t image_end{ payload_floor };
    std::uint64_t total_cardinality{};
    std::uint64_t previous_chunk{};
    bool first_chunk{ true };
    bool any_empty{ false };
    for ( std::uint32_t i = 0; i < header->chunk_count; ++i ) {
        auto const & entry{ index[ i ] };
        if ( !first_chunk && entry.chunk <= previous_chunk ) {
            return {};
        }
        if ( entry.chunk > std::numeric_limits<chunk_type>::max() ) {
            return {};  // narrowed to the chunk key type it would alias another chunk
        }
        first_chunk = false;
        previous_chunk = entry.chunk;

        auto const expected_alignment{ frozen_payload_alignment( entry ) };
        if ( expected_alignment == 0U || ( entry.payload_offset % expected_alignment ) != 0U ) {
            return {};
        }
        if ( entry.payload_offset < payload_floor || entry.payload_offset > in.size() ) {
            return {};
        }
        if ( entry.payload_bytes > in.size() - entry.payload_offset ) {
            return {};
        }
        if ( entry.payload_bytes != frozen_expected_payload_bytes( entry ) ) {
            return {};
        }
        auto rank_end{ std::uint64_t{ entry.payload_offset } + entry.payload_bytes };
        if ( ranked ) {
            if ( entry.base != total_cardinality || !frozen_ranked_counts_match( entry ) ) {
                return {};
            }
            rank_end += frozen_rank_bytes( static_cast<frozen_container_kind>( entry.container_kind ), entry.payload_count );
            if ( rank_end > in.size() ) {
                return {};
            }
            payload_floor = rank_end;
        }
        image_end = std::max( image_end, rank_end );
        total_cardinality += entry.cardinality;
        any_empty |= ( entry.cardinality == 0 );
    }
    // a chunk never holds an empty container
    if ( any_empty || total_cardinality != header->value_count ) {
        return {};
    }

    return frozen_view{ header, index, data, static_cast<std::size_t>( image_end ) };
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy> bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::deserialize_frozen_from_vm_vector( serialized_byte_vector const & in ) {
    return frozen_view_from_vm_vector( in ).materialize();
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bool bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::contains( key_type const value ) const noexcept {
    if ( header_ == nullptr || header_->chunk_count == 0U ) {
        return false;
    }
    auto const chunk{ static_cast<std::uint64_t>( layout_type::chunk_key( value ) ) };
    auto const count{ static_cast<std::size_t>( header_->chunk_count ) };
    auto const it{ lower_bound_entry( index_, count, chunk ) };
    if ( it == index_ + count || it->chunk != chunk ) {
        return false;
    }

    auto const low{ layout_type::low_key( value ) };
    auto const * const payload{ data_ + it->payload_offset };
    switch ( static_cast<frozen_container_kind>( it->container_kind ) ) {
        case frozen_container_kind::array:
            return array_contains_payload(
                reinterpret_cast<low_type const *>( payload ),
                static_cast<std::size_t>( it->payload_count ),
                low
            );
        case frozen_container_kind::run:
            return run_contains_payload(
                reinterpret_cast<typename run_container_type::run const *>( payload ),
                static_cast<int32_t>( it->payload_count ),
                low
            );
        case frozen_container_kind::bitset: {
            auto const * const words{ reinterpret_cast<std::uint64_t const *>( payload ) };
            auto const word_index{ static_cast<std::size_t>( low ) >> 6U };
            auto const bit_index{ static_cast<unsigned>( low ) & 63U };
            return ( words[ word_index ] & ( std::uint64_t{ 1 } << bit_index ) ) != 0;
        }
    }
    return false;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy> bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::materialize() const {
    bitmap result;
    if ( header_ == nullptr || header_->chunk_count == 0U ) {
        return result;
    }

    result.chunks_.reserve( header_->chunk_count );
    result.size_ = static_cast<std::size_t>( header_->value_count );
    for ( std::uint32_t i = 0; i < header_->chunk_count; ++i ) {
        auto const & entry{ index_[ i ] };
        auto const * const payload{ data_ + entry.payload_offset };
        detail::container_handle<layout_type, CowPolicy> container;
        switch ( static_cast<frozen_container_kind>( entry.container_kind ) ) {
            case frozen_container_kind::array: {
                auto const * const values{ reinterpret_cast<low_type const *>( payload ) };
                container = detail::container_handle<layout_type, CowPolicy>::make_array_from_sorted(
                    { values, entry.payload_count }
                );
                break;
            }
            case frozen_container_kind::run: {
                auto handle{ detail::container_handle<layout_type, CowPolicy>::make_run() };
                auto run{ handle.as_run() };
                auto const * const runs{
                    reinterpret_cast<typename detail::container_handle<layout_type, CowPolicy>::run_type const *>( payload )
                };
                run.runs.assign( { runs, entry.payload_count } );
                handle.set_cardinality( entry.cardinality );
                if ( entry.payload_count != 0 ) {
                    handle.set_endpoints( runs[ 0 ].begin, runs[ entry.payload_count - 1 ].end );
                }
                container = std::move( handle );
                break;
            }
            case frozen_container_kind::bitset: {
                auto handle{ detail::container_handle<layout_type, CowPolicy>::make_bitset_uninitialized() };
                std::memcpy(
                    handle.payload_data_raw(),
                    payload,
                    static_cast<std::size_t>( entry.payload_bytes )
                );
                handle.set_cardinality( entry.cardinality );
                container = std::move( handle );
                break;
            }
        }
        result.chunks_.push_back( static_cast<chunk_type>( entry.chunk ), std::move( container ) );
    }
    return result;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
void bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::borrow_into( bitmap & out ) const {
    out = bitmap{};
    if ( header_ == nullptr || header_->chunk_count == 0U ) {
        return;
    }

    out.chunks_.reserve( header_->chunk_count );
    out.size_ = static_cast<std::size_t>( header_->value_count );
    for ( std::uint32_t i = 0; i < header_->chunk_count; ++i ) {
        auto const & entry{ index_[ i ] };
        auto const kind{ container_kind_of( static_cast<frozen_container_kind>( entry.container_kind ) ) };
        out.chunks_.push_back(
            static_cast<chunk_type>( entry.chunk ),
            detail::container_handle<layout_type, CowPolicy>::make_borrowed(
                kind,
                data_ + entry.payload_offset,
                entry.payload_count,
                entry.cardinality
            )
        );
    }
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::int32_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::array_block_search(
    low_type const * values,
    std::int32_t const num_blocks,
    low_type const value
) noexcept {
    constexpr int32_t gap = 16;
    int32_t base = 0;
    int32_t n = num_blocks;
    while ( n > 3 ) {
        auto const quarter{ n >> 2 };
        auto const k1{ values[ ( base + quarter + 1 ) * gap - 1 ] };
        auto const k2{ values[ ( base + 2 * quarter + 1 ) * gap - 1 ] };
        auto const k3{ values[ ( base + 3 * quarter + 1 ) * gap - 1 ] };
        auto const c1{ k1 < value };
        auto const c2{ k2 < value };
        auto const c3{ k3 < value };
        base += ( c1 + c2 + c3 ) * quarter;
        n -= 3 * quarter;
    }
    while ( n > 1 ) {
        auto const half{ n >> 1 };
        base = ( values[ ( base + half + 1 ) * gap - 1 ] < value ) ? base + half : base;
        n -= half;
    }

    // Every value of the blocks before the result is below `value`: a block is passed only when its last value is.
    return ( values[ ( base + 1 ) * gap - 1 ] < value ) ? base + 1 : base;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bool bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::array_contains_payload(
    low_type const * values,
    std::size_t const count,
    low_type const value
) noexcept {
    // The equality kernel of the block: measured faster than array_lower_bound's position kernel (3 to 8 percent on
    // 12 to 4000 values) for a lookup that needs no position.
    constexpr int32_t gap = 16;
    auto const cardinality{ static_cast<int32_t>( count ) };
    if ( cardinality < gap ) {
        for ( int32_t j = 0; j < cardinality; ++j ) {
            if ( values[ j ] >= value ) {
                return values[ j ] == value;
            }
        }
        return false;
    }

    auto const num_blocks{ cardinality / gap };
    auto const lo{ array_block_search( values, num_blocks, value ) };
    if ( lo < num_blocks ) {
        auto const * const block{ values + static_cast<std::ptrdiff_t>( lo * gap ) };
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
        __m128i const needle{ _mm_set1_epi16( static_cast<short>( value ) ) };
        __m128i const v0{ _mm_loadu_si128( reinterpret_cast<__m128i const *>( block ) ) };
        __m128i const v1{ _mm_loadu_si128( reinterpret_cast<__m128i const *>( block + 8 ) ) };
        __m128i const hit{ _mm_or_si128( _mm_cmpeq_epi16( v0, needle ), _mm_cmpeq_epi16( v1, needle ) ) };
        return _mm_movemask_epi8( hit ) != 0;
#else
        for ( int32_t j = 0; j < gap; ++j ) {
            if ( block[ j ] >= value ) {
                return block[ j ] == value;
            }
        }
        return false;
#endif
    }

    for ( int32_t j = num_blocks * gap; j < cardinality; ++j ) {
        auto const current{ values[ j ] };
        if ( current >= value ) {
            return current == value;
        }
    }
    return false;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::int32_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::array_lower_bound(
    low_type const * values,
    std::size_t const count,
    low_type const value
) noexcept {
    constexpr int32_t gap = 16;
    auto const cardinality{ static_cast<int32_t>( count ) };
    if ( cardinality < gap ) {
        for ( int32_t j = 0; j < cardinality; ++j ) {
            if ( values[ j ] >= value ) {
                return j;
            }
        }
        return cardinality;
    }

    auto const num_blocks{ cardinality / gap };
    auto const lo{ array_block_search( values, num_blocks, value ) };
    if ( lo < num_blocks ) {
        auto const * const block{ values + static_cast<std::ptrdiff_t>( lo * gap ) };
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
        // The values are unsigned and the vector compare is signed: both sides are biased by 0x8000.
        __m128i const bias{ _mm_set1_epi16( static_cast<short>( 0x8000 ) ) };
        __m128i const needle{ _mm_xor_si128( _mm_set1_epi16( static_cast<short>( value ) ), bias ) };
        __m128i const v0{ _mm_xor_si128( _mm_loadu_si128( reinterpret_cast<__m128i const *>( block ) ), bias ) };
        __m128i const v1{ _mm_xor_si128( _mm_loadu_si128( reinterpret_cast<__m128i const *>( block + 8 ) ), bias ) };
        auto const below{ _mm_movemask_epi8( _mm_packs_epi16( _mm_cmplt_epi16( v0, needle ), _mm_cmplt_epi16( v1, needle ) ) ) };
        return lo * gap + std::popcount( static_cast<unsigned>( below ) );
#else
        int32_t below{ 0 };
        for ( int32_t j = 0; j < gap; ++j ) {
            below += block[ j ] < value;
        }
        return lo * gap + below;
#endif
    }

    for ( int32_t j = num_blocks * gap; j < cardinality; ++j ) {
        if ( values[ j ] >= value ) {
            return j;
        }
    }
    return cardinality;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
typename bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::position bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::run_position(
    typename run_container_type::run const * runs,
    std::uint16_t const * run_ranks,
    std::int32_t const count,
    low_type const value
) noexcept {
    // The last run that begins at or below `value`; an exact begin match is that run, otherwise
    // interleaved_binary_search returns -(insertion point + 1). The arithmetic is in 32 bits: a run rank tops out at
    // 65535 and end + 1 reaches 65536, neither of which fits low_type.
    auto last{ interleaved_binary_search( runs, count, value ) };
    if ( last < 0 ) {
        last = -last - 2;
        if ( last < 0 ) {
            return { 0U, false };
        }
    }
    auto const & run{ runs[ last ] };
    std::uint32_t const covered_end{ std::uint32_t{ run.end } + 1U };
    std::uint32_t const covered{ std::min<std::uint32_t>( value, covered_end ) - run.begin };
    return { std::uint32_t{ run_ranks[ last ] } + covered, value <= run.end };
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bool bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::run_contains_payload(
    typename run_container_type::run const * runs,
    int32_t const count,
    low_type const value
) noexcept {
    // A run covers the closed interval [begin, end]. interleaved_binary_search
    // returns the run index on an exact begin match, else -(insertion_point + 1);
    // on an inexact match the value may still fall inside the run immediately
    // before the insertion point, so test that run's end. Mirrors
    // run_container::contains (the frozen path previously matched begins only,
    // missing every interior and end value of a run).
    auto index{ interleaved_binary_search( runs, count, value ) };
    if ( index >= 0 ) {
        return true;
    }
    index = -index - 2;
    if ( index < 0 ) {
        return false;
    }
    return value <= runs[ static_cast<std::uint32_t>( index ) ].end;
}

// ---- standard Roaring portable format ---------------------------------------
// [croaring-ref] deps/croaring/src/roaring_array.c: ra_portable_serialize / ra_portable_deserialize
// [croaring-ref] deps/croaring/src/roaring64.c: roaring64_bitmap_portable_serialize / _deserialize_safe

namespace detail {

inline constexpr std::uint32_t portable_cookie_no_run{ 12346U };
inline constexpr std::uint32_t portable_cookie_with_run{ 12347U };
// Below this many containers a run-carrying bitmap omits the offset header.
inline constexpr std::uint32_t portable_no_offset_threshold{ 4U };
// A non-run container is an array up to and including this cardinality, a bitset above it:
// the reader derives the kind from the descriptive header alone.
inline constexpr std::uint32_t portable_array_max_cardinality{ 4096U };
inline constexpr std::size_t   portable_bitset_bytes{ 8192U };

static_assert( std::endian::native == std::endian::little, "the portable format is little-endian; no byte swapping is implemented" );

enum class portable_kind : std::uint8_t { array, bitset, run };

struct portable_wire {
    portable_kind kind;
    std::uint32_t cardinality;
    std::size_t   bytes;
};

// The wire kind follows the cardinality, not the in-memory kind, for non-run
// containers: an in-memory array above 4096 values (or a bitset at or below it)
// is re-encoded so the output is what the reference would hold.
template <typename Handle>
[[nodiscard]] inline portable_wire portable_wire_of( Handle const & container ) noexcept {
    auto const cardinality{ container.cardinality() };
    if ( container.holds_run() ) {
        return { portable_kind::run, cardinality, sizeof( std::uint16_t ) + std::size_t{ container.count() } * 2U * sizeof( std::uint16_t ) };
    }
    if ( cardinality > portable_array_max_cardinality ) {
        return { portable_kind::bitset, cardinality, portable_bitset_bytes };
    }
    return { portable_kind::array, cardinality, std::size_t{ cardinality } * sizeof( std::uint16_t ) };
}

inline void portable_put( std::byte * & out, std::uint16_t const value ) noexcept {
    std::memcpy( out, &value, sizeof( value ) );
    out += sizeof( value );
}
inline void portable_put( std::byte * & out, std::uint32_t const value ) noexcept {
    std::memcpy( out, &value, sizeof( value ) );
    out += sizeof( value );
}
template <typename T>
[[nodiscard]] inline T portable_get( std::byte const * const in ) noexcept {
    T value;
    std::memcpy( &value, in, sizeof( value ) );
    return value;
}

// Whether the values are strictly ascending. The compiler turns the obvious loop into one that
// carries the previous lane across vectors and widens every mask to 32 bits, which costs more than
// the copy of the values it checks; this compares the neighbouring pairs of a whole vector at a time
// and reads one verdict at the end. The last vector overlaps the one before it instead of leaving a
// scalar remainder.
[[nodiscard]] inline bool strictly_ascending( std::byte const * const values, std::size_t const count ) noexcept {
    if ( count < 2U ) {
        return true;
    }
    auto const pairs{ count - 1U };
#if defined( __AVX2__ )
    constexpr std::size_t lanes{ 16 };
    if ( pairs >= lanes ) {
        __m256i unordered{ _mm256_setzero_si256() };
        auto const compare{ [&]( std::size_t const at ) noexcept {
            auto const current{ _mm256_loadu_si256( reinterpret_cast<__m256i const *>( values + at * sizeof( std::uint16_t ) ) ) };
            auto const next   { _mm256_loadu_si256( reinterpret_cast<__m256i const *>( values + ( at + 1U ) * sizeof( std::uint16_t ) ) ) };
            // current >= next  <=>  max( current, next ) == current
            unordered = _mm256_or_si256( unordered, _mm256_cmpeq_epi16( _mm256_max_epu16( current, next ), current ) );
        } };
        std::size_t at{ 0 };
        for ( ; at + lanes <= pairs; at += lanes ) {
            compare( at );
        }
        if ( at != pairs ) {
            compare( pairs - lanes );
        }
        return _mm256_testz_si256( unordered, unordered ) != 0;
    }
#endif
    std::uint32_t unordered{ 0 };
    for ( std::size_t i{ 0 }; i < pairs; ++i ) {
        unordered |= std::uint32_t{ portable_get<std::uint16_t>( values + i * sizeof( std::uint16_t ) ) >= portable_get<std::uint16_t>( values + ( i + 1U ) * sizeof( std::uint16_t ) ) };
    }
    return unordered == 0;
}

template <typename Handle>
[[nodiscard]] inline std::byte * portable_write_container( Handle const & container, portable_wire const & wire, std::byte * out ) noexcept {
    switch ( wire.kind ) {
        case portable_kind::run: {
            auto const view{ container.as_run() };
            portable_put( out, static_cast<std::uint16_t>( view.runs.size() ) );
            for ( auto const & run : std::span{ view.runs.data(), view.runs.size() } ) {
                portable_put( out, static_cast<std::uint16_t>( run.begin ) );
                portable_put( out, static_cast<std::uint16_t>( run.end - run.begin ) );
            }
            return out;
        }
        case portable_kind::array:
            if ( container.holds_array() ) {
                auto const view{ container.as_array() };
                std::memcpy( out, view.values.data(), wire.bytes );
                return out + wire.bytes;
            } else {
                auto const view{ container.as_bitset() };
                auto const & words{ view.words.as_array() };
                for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
                    auto word{ words[ index ] };
                    while ( word != 0 ) {
                        portable_put( out, static_cast<std::uint16_t>( index * 64U + static_cast<unsigned>( std::countr_zero( word ) ) ) );
                        word &= word - 1;
                    }
                }
                return out;
            }
        case portable_kind::bitset:
            if ( container.holds_bitset() ) {
                auto const view{ container.as_bitset() };
                std::memcpy( out, view.words.as_array().data(), portable_bitset_bytes );
            } else {
                // Only a cardinality above 4096 reaches here, i.e. an oversized in-memory array.
                std::memset( out, 0, portable_bitset_bytes );
                auto const view{ container.as_array() };
                for ( auto const value : std::span{ view.values.data(), view.values.size() } ) {
                    out[ value >> 3U ] |= static_cast<std::byte>( 1U << ( value & 7U ) );
                }
            }
            return out + portable_bitset_bytes;
    }
    std::unreachable();
}

} // namespace detail

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
typename bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_plan bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_plan_for( chunk_cursor const first, chunk_cursor const last ) const noexcept {
    portable_plan plan;
    for ( auto index{ first }; index != last; ++index ) {
        auto const & slot{ chunks_.slot( index ) };
        ++plan.containers;
        plan.has_run |= slot.holds_run();
        plan.payload_bytes += detail::portable_wire_of( slot ).bytes;
    }
    // Run-carrying: cookie (carries the count), run flags, key+cardinality pairs, offsets
    // only from four containers up. Otherwise: cookie, count, pairs, offsets.
    if ( plan.has_run ) {
        plan.header_bytes = 4U + ( ( std::size_t{ plan.containers } + 7U ) / 8U ) + 4U * std::size_t{ plan.containers }
                          + ( plan.containers >= detail::portable_no_offset_threshold ? 4U * std::size_t{ plan.containers } : 0U );
    } else {
        plan.header_bytes = 8U + 8U * std::size_t{ plan.containers };
    }
    return plan;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::byte * bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_write_32( chunk_cursor const first, chunk_cursor const last, std::uint32_t const count, std::byte * const out ) const noexcept {
    // The header layout depends on whether any container is a run: one scan of the
    // handles settles it, then the header and the payloads are written in a single pass.
    bool has_run{ false };
    for ( auto index{ first }; index != last; ++index ) {
        if ( chunks_.slot( index ).holds_run() ) {
            has_run = true;
            break;
        }
    }

    auto * cursor{ out };
    std::byte * flags{ nullptr };
    if ( has_run ) {
        detail::portable_put( cursor, detail::portable_cookie_with_run | ( ( count - 1U ) << 16 ) );
        auto const flag_bytes{ ( count + 7U ) / 8U };
        std::memset( cursor, 0, flag_bytes );
        flags = cursor;
        cursor += flag_bytes;
    } else {
        detail::portable_put( cursor, detail::portable_cookie_no_run );
        detail::portable_put( cursor, count );
    }
    auto * descriptive{ cursor };
    cursor += 4U * std::size_t{ count };
    std::byte * offsets{ nullptr };
    if ( !has_run || count >= detail::portable_no_offset_threshold ) {
        offsets = cursor;
        cursor += 4U * std::size_t{ count };
    }

    std::uint32_t position{ 0 };
    for ( auto index{ first }; index != last; ++index, ++position ) {
        auto const & slot{ chunks_.slot( index ) };
        assert( slot.cardinality() != 0 );
        auto const wire{ detail::portable_wire_of( slot ) };
        detail::portable_put( descriptive, static_cast<std::uint16_t>( chunks_.key( index ) ) );
        detail::portable_put( descriptive, static_cast<std::uint16_t>( wire.cardinality - 1U ) );
        if ( offsets != nullptr ) {
            detail::portable_put( offsets, static_cast<std::uint32_t>( cursor - out ) );
        }
        if ( slot.holds_run() ) {
            flags[ position / 8U ] |= static_cast<std::byte>( 1U << ( position % 8U ) );
        }
        cursor = detail::portable_write_container( slot, wire, cursor );
    }
    return cursor;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_size_in_bytes() const requires ( std::numeric_limits<Key>::digits >= 32 ) {
    static_assert( std::numeric_limits<key_type>::digits == 32 || std::numeric_limits<key_type>::digits == 64, "the portable format is defined for 32- and 64-bit keys" );

    if constexpr ( std::numeric_limits<key_type>::digits == 32 ) {
        auto const plan{ portable_plan_for( chunks_.begin(), chunks_.end() ) };
        return plan.header_bytes + plan.payload_bytes;
    } else {
        std::size_t total{ sizeof( std::uint64_t ) };
        for ( auto first{ chunks_.begin() }; first != chunks_.end(); ) {
            auto const high{ chunks_.key( first ) >> 16 };
            auto last{ first };
            ++last;
            while ( last != chunks_.end() && ( chunks_.key( last ) >> 16 ) == high ) {
                ++last;
            }
            auto const plan{ portable_plan_for( first, last ) };
            if ( plan.containers != 0 ) {
                total += sizeof( std::uint32_t ) + plan.header_bytes + plan.payload_bytes;
            }
            first = last;
        }
        return total;
    }
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_serialize( std::span<std::byte> const out ) const requires ( std::numeric_limits<Key>::digits >= 32 ) {
    static_assert( std::numeric_limits<key_type>::digits == 32 || std::numeric_limits<key_type>::digits == 64, "the portable format is defined for 32- and 64-bit keys" );
    assert( out.size() >= portable_size_in_bytes() );

    auto * cursor{ out.data() };
    if constexpr ( std::numeric_limits<key_type>::digits == 32 ) {
        cursor = portable_write_32( chunks_.begin(), chunks_.end(), static_cast<std::uint32_t>( chunks_.size() ), cursor );
    } else {
        // The buckets (runs of chunks sharing the high 32 bits) in key order, behind
        // their count, which is only known once they have been walked.
        auto * const bucket_count_at{ cursor };
        cursor += sizeof( std::uint64_t );
        std::uint64_t buckets{ 0 };
        for ( auto first{ chunks_.begin() }; first != chunks_.end(); ) {
            auto const high{ chunks_.key( first ) >> 16 };
            auto last{ first };
            std::uint32_t containers{ 0 };
            do {
                ++last;
                ++containers;
            } while ( last != chunks_.end() && ( chunks_.key( last ) >> 16 ) == high );
            detail::portable_put( cursor, static_cast<std::uint32_t>( high ) );
            cursor = portable_write_32( first, last, containers, cursor );
            ++buckets;
            first = last;
        }
        std::memcpy( bucket_count_at, &buckets, sizeof( buckets ) );
    }
    return static_cast<std::size_t>( cursor - out.data() );
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bool bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_read_32(
    std::byte const * & cursor, std::byte const * const end, chunk_type const high_bits, bitmap & result, std::size_t & total
) {
    using namespace detail;
    auto const * p{ cursor };
    auto const available{ [&] { return static_cast<std::size_t>( end - p ); } };

    if ( available() < sizeof( std::uint32_t ) ) {
        return false;
    }
    auto const cookie{ portable_get<std::uint32_t>( p ) };
    p += sizeof( cookie );
    bool has_run;
    std::size_t count;
    if ( ( cookie & 0xFFFFU ) == portable_cookie_with_run ) {
        has_run = true;
        count = ( cookie >> 16 ) + 1U;
    } else if ( cookie == portable_cookie_no_run ) {
        has_run = false;
        if ( available() < sizeof( std::uint32_t ) ) {
            return false;
        }
        count = portable_get<std::uint32_t>( p );
        p += sizeof( std::uint32_t );
        if ( count > 65536U ) {
            return false;
        }
    } else {
        return false;
    }

    // `count` <= 65536 here, so none of the header arithmetic below can overflow.
    std::byte const * run_flags{ nullptr };
    if ( has_run ) {
        auto const flag_bytes{ ( count + 7U ) / 8U };
        if ( available() < flag_bytes ) {
            return false;
        }
        run_flags = p;
        p += flag_bytes;
    }
    if ( available() < count * 4U ) {
        return false;
    }
    auto const * const descriptive{ p };
    p += count * 4U;
    if ( !has_run || count >= portable_no_offset_threshold ) {
        // The offsets are redundant with the descriptive header and are not read (as in
        // the reference): the payloads are consumed back to back.
        if ( available() < count * 4U ) {
            return false;
        }
        p += count * 4U;
    }

    if ( result.chunks_.size() == 0 ) {
        result.chunks_.reserve( count );
    }
    std::int32_t previous_key{ -1 };
    for ( std::size_t k{ 0 }; k < count; ++k ) {
        auto const key{ portable_get<std::uint16_t>( descriptive + 4U * k ) };
        if ( std::int32_t{ key } <= previous_key ) {
            return false;
        }
        previous_key = key;
        std::uint32_t const cardinality{ std::uint32_t{ portable_get<std::uint16_t>( descriptive + 4U * k + 2U ) } + 1U };
        bool const is_run{ has_run && ( ( std::to_integer<unsigned>( run_flags[ k / 8U ] ) >> ( k % 8U ) ) & 1U ) != 0 };

        handle_type container;
        if ( is_run ) {
            if ( available() < sizeof( std::uint16_t ) ) {
                return false;
            }
            std::size_t const run_count{ portable_get<std::uint16_t>( p ) };
            p += sizeof( std::uint16_t );
            if ( run_count == 0 || available() < run_count * 4U ) {
                return false;
            }
            container = handle_type::make_run();
            auto view{ container.as_run() };
            view.runs.resize_uninitialized( static_cast<std::uint32_t>( run_count ) );
            auto * const dst{ view.runs.data() };
            std::uint32_t covered{ 0 };
            std::int32_t previous_end{ -1 };
            std::size_t written{ 0 };
            for ( std::size_t r{ 0 }; r < run_count; ++r, p += 4 ) {
                std::int32_t const begin{ portable_get<std::uint16_t>( p ) };
                std::int32_t const run_end{ begin + portable_get<std::uint16_t>( p + 2 ) };
                if ( run_end > 0xFFFF || begin <= previous_end ) {
                    return false;   // out of the 16-bit domain, unsorted or overlapping
                }
                covered += static_cast<std::uint32_t>( run_end - begin + 1 );
                if ( begin == previous_end + 1 && written != 0 ) {
                    // Adjacent runs are one run here (the writer is allowed to leave them split).
                    dst[ written - 1 ].end = static_cast<low_type>( run_end );
                } else {
                    dst[ written++ ] = { static_cast<low_type>( begin ), static_cast<low_type>( run_end ) };
                }
                previous_end = run_end;
            }
            if ( covered != cardinality ) {
                return false;
            }
            view.runs.resize_uninitialized( static_cast<std::uint32_t>( written ) );
            container.set_cardinality( cardinality );
            container.set_endpoints( dst[ 0 ].begin, dst[ written - 1 ].end );
        } else if ( cardinality > portable_array_max_cardinality ) {
            if ( available() < portable_bitset_bytes ) {
                return false;
            }
            container = handle_type::make_bitset_uninitialized();
            std::memcpy( container.payload_data_raw(), p, portable_bitset_bytes );
            p += portable_bitset_bytes;
            auto const * const words{ static_cast<std::uint64_t const *>( container.payload_data_raw() ) };
            if ( popcount_words( words, layout_type::word_count ) != cardinality ) {
                return false;
            }
            container.set_cardinality( cardinality );
        } else {
            if ( available() < std::size_t{ cardinality } * sizeof( std::uint16_t ) ) {
                return false;
            }
            // Checked where it lies: the values are read back from the buffer, not from the
            // copy that was just written, which a load straddling two stores would have to wait for.
            if ( !strictly_ascending( p, cardinality ) ) {
                return false;
            }
            container = handle_type::make_array();
            auto array{ container.as_array() };
            array.values.resize_uninitialized( cardinality );
            std::memcpy( array.values.data(), p, std::size_t{ cardinality } * sizeof( std::uint16_t ) );
            p += std::size_t{ cardinality } * sizeof( std::uint16_t );
            array.sync_header();
        }
        total += cardinality;
        result.chunks_.push_back( static_cast<chunk_type>( high_bits | key ), std::move( container ) );
    }
    cursor = p;
    return true;
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
std::optional<bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>> bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::portable_deserialize( std::span<std::byte const> const in ) requires ( std::numeric_limits<Key>::digits >= 32 ) {
    static_assert( std::numeric_limits<key_type>::digits == 32 || std::numeric_limits<key_type>::digits == 64, "the portable format is defined for 32- and 64-bit keys" );
    bitmap result;
    std::size_t total{ 0 };
    auto const * cursor{ in.data() };
    auto const * const end{ in.data() + in.size() };

    if constexpr ( std::numeric_limits<key_type>::digits == 32 ) {
        if ( !portable_read_32( cursor, end, chunk_type{ 0 }, result, total ) ) {
            return std::nullopt;
        }
    } else {
        if ( in.size() < sizeof( std::uint64_t ) ) {
            return std::nullopt;
        }
        auto const buckets{ detail::portable_get<std::uint64_t>( cursor ) };
        cursor += sizeof( std::uint64_t );
        // Every bucket costs at least its key and a minimal 32-bit bitmap header.
        if ( buckets > std::numeric_limits<std::uint32_t>::max() || buckets > static_cast<std::size_t>( end - cursor ) / 12U ) {
            return std::nullopt;
        }
        std::int64_t previous_high{ -1 };
        for ( std::uint64_t bucket{ 0 }; bucket < buckets; ++bucket ) {
            if ( static_cast<std::size_t>( end - cursor ) < sizeof( std::uint32_t ) ) {
                return std::nullopt;
            }
            auto const high{ detail::portable_get<std::uint32_t>( cursor ) };
            cursor += sizeof( std::uint32_t );
            if ( std::int64_t{ high } <= previous_high ) {
                return std::nullopt;
            }
            previous_high = high;
            if ( !portable_read_32( cursor, end, static_cast<chunk_type>( chunk_type{ high } << 16 ), result, total ) ) {
                return std::nullopt;
            }
        }
    }

    result.size_ = total;
    return result;
}

} // namespace frsr::roaring
