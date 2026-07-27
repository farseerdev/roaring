#pragma once

// Out-of-line definitions of frsr::roaring::bitmap's serialization members.
// Two formats: the portable add()-replay format (serialize_to_vm_vector /
// deserialize_from_vm_vector) and the zero-copy frozen format
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
#include <cstring>
#include <limits>
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
std::size_t bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_size_bytes() const {
    if constexpr ( kUseSingletonChunkMap ) {
        materialize_singleton_chunks();
    }
    ensure_sorted();

    std::size_t chunk_count{};
    std::size_t offset{ sizeof( frozen_header ) };
    for ( auto const & slot : chunks_.slots() ) {
        if ( detail::container_size( slot ) != 0 ) {
            ++chunk_count;
        }
    }
    offset += chunk_count * sizeof( frozen_chunk_index );
    for ( auto const & slot : chunks_.slots() ) {
        if ( detail::container_size( slot ) == 0 ) {
            continue;
        }
        auto const [alignment, payload_size] = frozen_payload_layout( slot );
        offset = align_up( offset, alignment );
        offset += payload_size;
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
void bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::serialize_frozen_to_vm_vector( serialized_byte_vector & out ) const {
    if constexpr ( kUseSingletonChunkMap ) {
        materialize_singleton_chunks();
    }
    ensure_sorted();

    std::uint32_t chunk_count{};
    for ( auto const & slot : chunks_.slots() ) {
        if ( detail::container_size( slot ) != 0 ) {
            ++chunk_count;
        }
    }

    auto const byte_count{ static_cast<std::uint32_t>( frozen_size_bytes() ) };
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
    frozen_header const header{
        .magic = frozen_serialization_magic,
        .key_bits = static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits ),
        .version = frozen_format_version,
        .chunk_count = chunk_count,
        .value_count = static_cast<std::uint64_t>( size_ )
    };
    std::memcpy( data, &header, sizeof( header ) );

    auto * const index{
        reinterpret_cast<frozen_chunk_index *>( data + sizeof( frozen_header ) )
    };
    std::size_t payload_offset{
        sizeof( frozen_header ) + static_cast<std::size_t>( chunk_count ) * sizeof( frozen_chunk_index )
    };
    std::uint32_t frozen_index{};
    for ( size_type chunk_index{ 0 }; chunk_index < chunks_.size(); ++chunk_index ) {
        auto const & container{ chunks_.slot( chunk_index ) };
        if ( detail::container_size( container ) == 0 ) {
            continue;
        }

        auto const [alignment, payload_size] = frozen_payload_layout( container );
        payload_offset = align_up( payload_offset, alignment );
        auto & frozen_chunk{ index[ frozen_index++ ] };
        frozen_chunk.chunk = static_cast<std::uint64_t>( chunks_.key( chunk_index ) );
        frozen_chunk.payload_offset = static_cast<std::uint32_t>( payload_offset );
        frozen_chunk.payload_bytes = static_cast<std::uint32_t>( payload_size );

        auto * const payload{ data + payload_offset };
        switch ( container.kind() ) {
            case detail::container_kind::array: {
                auto const array{ container.as_array() };
                frozen_chunk.payload_count = static_cast<std::uint32_t>( array.values.size() );
                frozen_chunk.cardinality = static_cast<std::uint32_t>( array.values.size() );
                frozen_chunk.container_kind = static_cast<std::uint8_t>( frozen_container_kind::array );
                std::memcpy( payload, array.values.data(), payload_size );
                break;
            }
            case detail::container_kind::run: {
                auto const run{ container.as_run() };
                frozen_chunk.payload_count = static_cast<std::uint32_t>( run.runs.size() );
                frozen_chunk.cardinality = static_cast<std::uint32_t>( container.cardinality() );
                frozen_chunk.container_kind = static_cast<std::uint8_t>( frozen_container_kind::run );
                std::memcpy( payload, run.runs.data(), payload_size );
                break;
            }
            case detail::container_kind::bitset: {
                auto const bitset{ container.as_bitset() };
                frozen_chunk.payload_count = static_cast<std::uint32_t>( layout_type::word_count );
                frozen_chunk.cardinality = static_cast<std::uint32_t>( container.cardinality() );
                frozen_chunk.container_kind = static_cast<std::uint8_t>( frozen_container_kind::bitset );
                std::memcpy( payload, bitset.words.as_array().data(), payload_size );
                break;
            }
        }
        payload_offset += payload_size;
    }
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
    if ( in.size() < sizeof( frozen_header ) ) {
        return {};
    }

    auto const * const data{ in.data() };
    auto const * const header{ reinterpret_cast<frozen_header const *>( data ) };
    if (
        header->magic != frozen_serialization_magic ||
        header->key_bits != static_cast<std::uint16_t>( std::numeric_limits<key_type>::digits ) ||
        header->version != frozen_format_version
    ) {
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
    auto const min_payload_offset{ index_offset + index_bytes };
    std::uint64_t total_cardinality{};
    std::uint64_t previous_chunk{};
    bool first_chunk{ true };
    for ( std::uint32_t i = 0; i < header->chunk_count; ++i ) {
        auto const & entry{ index[ i ] };
        if ( !first_chunk && entry.chunk <= previous_chunk ) {
            return {};
        }
        first_chunk = false;
        previous_chunk = entry.chunk;

        auto const expected_alignment{ frozen_payload_alignment( entry ) };
        if ( expected_alignment == 0U || ( entry.payload_offset % expected_alignment ) != 0U ) {
            return {};
        }
        if ( entry.payload_offset < min_payload_offset || entry.payload_offset > in.size() ) {
            return {};
        }
        if ( entry.payload_bytes > in.size() - entry.payload_offset ) {
            return {};
        }
        if ( entry.payload_bytes != frozen_expected_payload_bytes( entry ) ) {
            return {};
        }
        total_cardinality += entry.cardinality;
    }
    if ( total_cardinality != header->value_count ) {
        return {};
    }

    return frozen_view{ header, index, data };
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
    auto const index_span{
        std::span<frozen_chunk_index const>{ index_, static_cast<std::size_t>( header_->chunk_count ) }
    };
    auto const it{ std::ranges::lower_bound(
        index_span,
        chunk,
        {},
        &frozen_chunk_index::chunk
    ) };
    if ( it == index_span.end() || it->chunk != chunk ) {
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
    if constexpr ( kUseLazySort ) {
        result.chunks_sorted_ = true;
    }
    result.invalidate_chunk_index_map();
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
        auto const kind{ [&] {
            switch ( static_cast<frozen_container_kind>( entry.container_kind ) ) {
                case frozen_container_kind::array : return detail::container_kind::array ;
                case frozen_container_kind::run   : return detail::container_kind::run   ;
                case frozen_container_kind::bitset: return detail::container_kind::bitset;
            }
            std::unreachable();
        }() };
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
    if constexpr ( kUseLazySort ) {
        out.chunks_sorted_ = true;
    }
    out.invalidate_chunk_index_map();
}

template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy>
    requires detail::valid_container_set<Key, ContainerSet>
bool bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>::frozen_view::array_contains_payload(
    low_type const * values,
    std::size_t const count,
    low_type const value
) noexcept {
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

    auto const lo{ ( values[ ( base + 1 ) * gap - 1 ] < value ) ? base + 1 : base };
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

} // namespace frsr::roaring
