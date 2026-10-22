#pragma once

// Out-of-class definitions of the bitmap members that are too large to inline into a caller. They are
// compiled once, into the library, for the instantiations bitmap.hpp declares extern; a program that
// instantiates any other bitmap, or that builds header-only, includes this header (bitmap.hpp does it itself
// under FRSR_ROARING_HEADER_ONLY or FRSR_ROARING_WITH_DEFINITIONS).

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/array_ops.hpp>
#include <frsr/roaring/run_ops.hpp>
#include <frsr/roaring/container_ops.hpp>
#include <frsr/roaring/interop.hpp>
#include <frsr/roaring/serialization.hpp>

#define FRSR_ROARING_BITMAP bitmap<Key, ContainerSet, CowPolicy, RunSelectionPolicy>
#define FRSR_ROARING_BITMAP_TEMPLATE \
    template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy, typename RunSelectionPolicy> \
        requires detail::valid_container_set<Key, ContainerSet>

// Out of class a definition is not inline unless said so, and inline is exactly what a header-only program wants
// back for the members that were defined in the class; in the compiled mode it would make every program instantiate
// them in spite of the extern declarations.
#ifdef FRSR_ROARING_HEADER_ONLY
#   define FRSR_ROARING_DEFINITION_INLINE inline
#else
#   define FRSR_ROARING_DEFINITION_INLINE
#endif

namespace frsr::roaring {

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::byte_size() const noexcept -> size_type {
    auto total{ chunks_.entry_bytes() };
    for ( auto const & slot : chunks_.slots() ) {
        total += detail::container_byte_size( slot );
    }
    return total;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::data_byte_size() const noexcept -> size_type {
    auto total{ chunks_.size() * chunk_descriptor_bytes };
    for ( auto const & slot : chunks_.slots() ) {
        total += detail::container_byte_size( slot );
    }
    return total;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::estimated_data_byte_size() const noexcept -> size_type {
    if constexpr ( chunk_store_type::wide ) {
        return data_byte_size();
    } else {
        auto const slots{ chunks_.slots() };
        auto const n{ slots.size() };
        if ( n == 0 ) { return 0; }
        constexpr std::size_t max_samples{ 8 };
        auto const step{ std::max<std::size_t>( 1, n / max_samples ) };
        std::size_t sum{ 0 }, taken{ 0 };
        for ( auto i{ step / 2 }; i < n && taken < max_samples; i += step, ++taken ) {
            sum += detail::container_byte_size( slots[ i ] );
        }
        return static_cast<size_type>( n * chunk_descriptor_bytes + sum * n / taken );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE char const * FRSR_ROARING_BITMAP::validate( bool const canonical_kinds ) const noexcept {
    if ( auto const * const problem{ chunks_.validate() } ) {
        return problem;
    }
    size_type total{ 0 };
    chunk_type previous{};
    for ( auto at{ chunks_.begin() }; at != chunks_.end(); ++at ) {
        auto const key{ chunks_.key( at ) };
        if ( at != chunks_.begin() && !( previous < key ) ) {
            return "chunk keys are not strictly ascending";
        }
        previous = key;
        auto const & slot{ chunks_.slot( at ) };
        auto const cardinality{ static_cast<size_type>( slot.cardinality() ) };
        if ( cardinality == 0 ) {
            return "a chunk holds an empty container";
        }
        switch ( slot.kind() ) {
            case detail::container_kind::array: {
                auto const & values{ slot.as_array().values };
                if ( static_cast<size_type>( values.size() ) != cardinality ) { return "an array's cardinality header differs from its length"; }
                if ( cardinality > array_to_bitset_threshold ) { return "an array holds more values than the array limit"; }
                for ( size_type i{ 1 }; i < cardinality; ++i ) {
                    if ( !( values[ i - 1 ] < values[ i ] ) ) { return "an array's values are not strictly ascending"; }
                }
                break;
            }
            case detail::container_kind::bitset: {
                auto const & words{ slot.as_bitset().words.as_array() };
                size_type bits{ 0 };
                for ( size_type w{ 0 }; w < layout_type::word_count; ++w ) { bits += static_cast<size_type>( std::popcount( words[ w ] ) ); }
                if ( bits != cardinality ) { return "a bitset's cardinality header differs from its popcount"; }
                if ( canonical_kinds && cardinality <= array_to_bitset_threshold ) { return "a bitset holds no more values than the array limit"; }
                break;
            }
            case detail::container_kind::run: {
                size_type counted{ 0 };
                bool      first_run{ true };
                low_type  previous_end{};
                for ( auto const & run : slot.as_run().runs ) {
                    if ( run.end < run.begin ) { return "a run ends before it begins"; }
                    if ( !first_run && !( previous_end < run.begin ) ) { return "runs overlap or are out of order"; }
                    first_run    = false;
                    previous_end = run.end;
                    counted     += static_cast<size_type>( run.end - run.begin ) + 1U;
                }
                if ( counted != cardinality ) { return "a run container's cardinality header differs from its runs"; }
                break;
            }
        }
        total += cardinality;
    }
    if ( total != size_ ) { return "the size differs from the sum of the container cardinalities"; }
    return nullptr;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::statistics() const noexcept -> container_statistics {
    container_statistics result;
    for ( auto const & slot : chunks_.slots() ) {
        if ( slot.holds_array() ) {
            ++result.array_containers;
        } else if ( slot.holds_run() ) {
            ++result.run_containers;
        } else {
            ++result.bitset_containers;
        }
    }
    return result;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::back() const noexcept -> key_type {
    auto at{ chunks_.end() };
    while ( at != chunks_.begin() ) {
        at = chunks_.prev( at );
        auto const & container{ chunks_.slot( at ) };
        auto next{ detail::container_first( container ) };
        if ( !next ) {
            continue;
        }
        while ( auto const after{ detail::container_next_after( container, *next ) } ) {
            next = after;
        }
        return layout_type::compose( chunks_.key( at ), *next );
    }
    return {};
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE bool FRSR_ROARING_BITMAP::intersects( bitmap const & other ) const noexcept {

    auto left { chunks_.begin() };
    auto right{ other.chunks_.begin() };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            ++left;
        } else if ( right_key < left_key ) {
            ++right;
        } else {
            if ( detail::container_intersects( chunks_.slot( left ), other.chunks_.slot( right ) ) ) {
                return true;
            }
            ++left;
            ++right;
        }
    }
    return false;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::add_many_sorted( std::span<key_type const> const sorted_values ) noexcept( detail::allocation_failure_terminates ) {
    settle_counts();
    if ( sorted_values.empty() ) {
        return;
    }

    chunk_type current_chunk{};
    detail::small_array_values<low_type> grouped_values;
    auto flush_group{ [&]() {
        if ( grouped_values.empty() ) {
            return;
        }

        auto const at{ chunks_.lower_bound( current_chunk ) };
        if ( at == chunks_.end() || chunks_.key( at ) != current_chunk ) {
            chunks_.insert( at, current_chunk, make_policy_container_from_sorted_values(
                { grouped_values.data(), grouped_values.size() },
                array_to_bitset_threshold
            ) );
            size_ += grouped_values.size();
        } else {
            size_ += detail::container_add_sorted(
                chunks_.slot( at ), { grouped_values.data(), grouped_values.size() }
            );
            promote_if_needed( chunks_.slot( at ) );
        }

        grouped_values.clear();
    } };

    for ( auto const value : sorted_values ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        if ( grouped_values.empty() ) {
            current_chunk = chunk;
        } else if ( chunk != current_chunk ) {
            flush_group();
            current_chunk = chunk;
        }

        if ( grouped_values.empty() || grouped_values.back() != low ) {
            grouped_values.push_back( low );
        }
    }

    flush_group();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::remove_many_sorted( std::span<key_type const> const sorted_values ) noexcept( detail::allocation_failure_terminates ) {
    settle_counts();
    if ( sorted_values.empty() ) {
        return;
    }

    chunk_type current_chunk{};
    detail::small_array_values<low_type> grouped_values;
    auto flush_group{ [&]() {
        if ( grouped_values.empty() ) {
            return;
        }

        auto const at{ chunks_.lower_bound( current_chunk ) };
        if ( at != chunks_.end() && chunks_.key( at ) == current_chunk ) {
            auto & slot{ chunks_.slot( at ) };
            if ( slot.holds_array() ) {
                // One merge walk over the array for the whole group; the
                // per-value remove is a lower_bound + memmove each.
                size_ -= static_cast<size_type>( slot.as_array().remove_sorted( grouped_values ) );
            } else {
                for ( auto const low : grouped_values ) {
                    if ( detail::container_remove( slot, low ) ) {
                        --size_;
                    }
                }
            }
            if ( detail::container_size( chunks_.slot( at ) ) == 0 ) {
                chunks_.erase( at );
            } else {
                chunks_.slot( at ) = optimize_container_for_policy( std::move( chunks_.slot( at ) ) );
            }
        }

        grouped_values.clear();
    } };

    for ( auto const value : sorted_values ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        if ( grouped_values.empty() ) {
            current_chunk = chunk;
        } else if ( chunk != current_chunk ) {
            flush_group();
            current_chunk = chunk;
        }

        if ( grouped_values.empty() || grouped_values.back() != low ) {
            grouped_values.push_back( low );
        }
    }

    flush_group();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::add_sorted_many( std::span<key_type const> const sorted_values, std::uint16_t const bitset_threshold ) {
    if ( !empty() ) {
        add_many_sorted( sorted_values );
        return;
    }

    chunks_.clear();
    size_ = 0;
    if ( sorted_values.empty() ) {
        return;
    }

    chunk_type current_chunk{};
    detail::small_array_values<low_type> grouped_values;
    auto flush_group{ [&]() {
        if ( grouped_values.empty() ) {
            return;
        }
        chunks_.push_back( current_chunk, make_policy_container_from_sorted_values(
            { grouped_values.data(), grouped_values.size() },
            bitset_threshold
        ) );
        size_ += grouped_values.size();
        grouped_values.clear();
    } };

    for ( auto const value : sorted_values ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        if ( grouped_values.empty() ) {
            current_chunk = chunk;
        } else if ( chunk != current_chunk ) {
            flush_group();
            current_chunk = chunk;
        }
        if ( grouped_values.empty() || grouped_values.back() != low ) {
            grouped_values.push_back( low );
        }
    }
    flush_group();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::add_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates ) {
    apply_closed_range_operation( begin_value, end_value, detail::range_operation::add );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::remove_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates ) {
    apply_closed_range_operation( begin_value, end_value, detail::range_operation::remove );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::flip_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates ) {
    apply_closed_range_operation( begin_value, end_value, detail::range_operation::flip );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE std::size_t FRSR_ROARING_BITMAP::to_array_into( std::span<key_type> const out ) const {
    // Values come out in order, so the chunks must be in order first: a wide
    // chunk key defers the sort until a path needs it, and this is such a path.
    assert( out.size() >= size_ );
    auto * out_cursor{ out.data() };
    auto * const end{ out.data() + out.size() };
    for ( auto at{ chunks_.begin() }; at != chunks_.end(); ++at ) {
        out_cursor = detail::container_decode_into( chunks_.slot( at ), chunks_.key( at ), out_cursor, end );
    }
    return static_cast<std::size_t>( out_cursor - out.data() );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::to_vector() const -> detail::heap_vector<key_type> {
    detail::heap_vector<key_type> result;
    detail::resize_uninitialized( result, static_cast<std::uint32_t>( size_ ) );
    auto const written{ to_array_into( { result.data(), result.size() } ) };
    assert( written == size_ );
    result.resize( static_cast<std::uint32_t>( written ) );
    return result;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::union_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates ) {
    if ( this == &other ) {
        scratch = *this;
        return;
    }


    scratch.clear_keep_capacity();
    scratch.chunks_.reserve( chunks_.size() + other.chunks_.size() );
    detail::small_array_values<low_type> array_union_scratch;
#if FRSR_ROARING_COMBINE_STATS
    detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif

    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            auto const & left_container{ chunks_.slot( left ) };
            if ( detail::container_size( left_container ) != 0 ) {
                scratch.size_ += detail::container_size( left_container );
                scratch.chunks_.push_back( left_key, left_container );
            }
            ++left;
        } else if ( right_key < left_key ) {
            auto const & right_container{ other.chunks_.slot( right ) };
            if ( detail::container_size( right_container ) != 0 ) {
                scratch.size_ += detail::container_size( right_container );
                scratch.chunks_.push_back( right_key, right_container );
            }
            ++right;
        } else {
            auto const & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_pair( left_container, right_container, detail::set_operation::bit_or );
#endif
            if ( left_container.holds_array() && right_container.holds_array() ) {
                auto container{ make_array_union_container_reusing(
                    scratch.chunks_,
                    left_container.as_array(),
                    right_container.as_array(),
                    array_union_scratch
                ) };
                scratch.size_ += detail::container_size( container );
                scratch.chunks_.push_back( left_key, std::move( container ) );
                ++left;
                ++right;
                continue;
            }
            if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
                auto container{ combine_bitset_bitset_for_policy(
                    left_container.as_bitset(),
                    right_container.as_bitset(),
                    detail::set_operation::bit_or,
                    scratch.chunks_.take_retired( detail::container_kind::bitset )
                ) };
                scratch.size_ += detail::container_size( container );
                scratch.chunks_.push_back( left_key, std::move( container ) );
                ++left;
                ++right;
                continue;
            }
            auto container{ combine_containers_for_policy(
                left_container,
                right_container,
                detail::set_operation::bit_or
            ) };
            scratch.size_ += detail::container_size( container );
            scratch.chunks_.push_back( left_key, std::move( container ) );
            ++left;
            ++right;
        }
    }

    while ( left != chunks_.end() ) {
        auto const & left_container{ chunks_.slot( left ) };
        if ( detail::container_size( left_container ) != 0 ) {
            scratch.size_ += detail::container_size( left_container );
            scratch.chunks_.push_back( chunks_.key( left ), left_container );
        }
        ++left;
    }
    while ( right != other.chunks_.end() ) {
        auto const & right_container{ other.chunks_.slot( right ) };
        if ( detail::container_size( right_container ) != 0 ) {
            scratch.size_ += detail::container_size( right_container );
            scratch.chunks_.push_back( other.chunks_.key( right ), right_container );
        }
        ++right;
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::difference_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates ) {
    if ( this == &other ) {
        scratch.clear_keep_capacity();
        return;
    }


    scratch.clear_keep_capacity();
    scratch.chunks_.reserve( chunks_.size() );

    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    while ( left != chunks_.end() ) {
        auto const left_key{ chunks_.key( left ) };
        while ( right != other.chunks_.end() && other.chunks_.key( right ) < left_key ) {
            ++right;
        }

        if ( right == other.chunks_.end() || left_key < other.chunks_.key( right ) ) {
            auto const & left_container{ chunks_.slot( left ) };
            if ( detail::container_size( left_container ) != 0 ) {
                scratch.size_ += detail::container_size( left_container );
                scratch.chunks_.push_back( left_key, left_container );
            }
        } else {
            auto const & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
            if ( left_container.holds_array() && right_container.holds_array() ) {
                // Difference directly into the result array's storage: a\b ⊆ a is
                // always an array (|result| ≤ |a| < bitset threshold), so skip the
                // scratch buffer and the scratch→container copy (see intersect_into).
                handle_type result_handle{ scratch.chunks_.take_retired( detail::container_kind::array ) };
                auto result_array{ result_handle.as_array() };
                detail::difference_array_array_to_vector<layout_type>(
                    left_container.as_array(),
                    right_container.as_array(),
                    result_array.values
                );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    scratch.size_ += result_array.values.size();
                    scratch.chunks_.push_back( left_key, std::move( result_handle ) );
                }
                ++left;
                continue;
            }
            if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
                auto container{ demote_sparse_bitset( combine_bitset_bitset_for_policy(
                    left_container.as_bitset(),
                    right_container.as_bitset(),
                    detail::set_operation::bit_andnot,
                    scratch.chunks_.take_retired( detail::container_kind::bitset )
                ), scratch.chunks_ ) };
                if ( auto const bitset_size{ detail::container_size( container ) }; bitset_size != 0 ) {
                    scratch.size_ += bitset_size;
                    scratch.chunks_.push_back( left_key, std::move( container ) );
                }
                ++left;
                continue;
            }
            if ( left_container.holds_array() && right_container.holds_bitset() ) {
                // array\bitset: result ⊆ array ⇒ write straight into a fresh/retired
                // array handle (same D2a direct-fill as materializing array∩bitset
                // in combine). Without this arm every mixed pair fell through to
                // combine_containers_for_policy's visit/scratch path.
                handle_type result_handle{ scratch.chunks_.take_retired( detail::container_kind::array ) };
                auto result_array{ result_handle.as_array() };
                detail::filter_array_bitset_into<layout_type>(
                    left_container.as_array(),
                    right_container.as_bitset(),
                    false,
                    result_array.values
                );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    scratch.size_ += result_array.values.size();
                    scratch.chunks_.push_back( left_key, std::move( result_handle ) );
                }
                ++left;
                continue;
            }
            auto container{ combine_containers_for_policy(
                left_container,
                right_container,
                detail::set_operation::bit_andnot
            ) };
            auto const container_size{ detail::container_size( container ) };
            if ( container_size != 0 ) {
                scratch.size_ += container_size;
                scratch.chunks_.push_back( left_key, std::move( container ) );
            }
        }
        ++left;
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
void FRSR_ROARING_BITMAP::union_same_key( detail::chunk_store<layout_type, CowPolicy> & store, handle_type & left_container, handle_type const & right_container, bool const lazy, detail::small_array_values<low_type> & array_union_scratch ) {
    bool const mutate_left{ left_container.payload_is_unshared() };
    if ( left_container.holds_array() && right_container.holds_array() ) {
        if ( mutate_left ) {
            detail::union_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
            left_container = make_fast_container( std::move( left_container ) );
        } else {
            left_container = make_array_union_container_reusing(
                store,
                std::as_const( left_container ).as_array(),
                right_container.as_array(),
                array_union_scratch
            );
        }
        return;
    }
    if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
        if ( !mutate_left ) {
            // A saturated left is the result, shared as it is: the O(1) analog of the
            // in-place arm's full-bitset shortcut.
            if ( left_container.cardinality() != layout_type::word_count * 64U ) {
                left_container = combine_bitset_bitset_for_policy(
                    std::as_const( left_container ).as_bitset(),
                    right_container.as_bitset(),
                    detail::set_operation::bit_or,
                    store.take_retired( detail::container_kind::bitset )
                );
            }
            return;
        }
        auto left_bitset{ left_container.as_bitset() };
        if ( lazy ) {
            detail::or_bitset_bitset_inplace_lazy<layout_type>( left_bitset, right_container.as_bitset() );
        } else {
            detail::combine_bitset_bitset_inplace<layout_type>( left_bitset, right_container.as_bitset(), detail::set_operation::bit_or );
        }
        return;
    }
    // An array into an unshared bitset: set its bits in place and count the new ones, as CRoaring's
    // bitset_container_ior does, rather than materializing a copy of the 8 KB bitset.
    // [croaring-ref] deps/croaring/src/containers/containers.c:container_ior (BITSET_CONTAINER x ARRAY_CONTAINER)
    if ( !lazy && mutate_left && left_container.holds_bitset() && right_container.holds_array() ) {
        auto left_bitset{ left_container.as_bitset() };
        detail::or_array_into_bitset_inplace<false, layout_type>( left_bitset, right_container.as_array() );
        return;
    }
    // Same-key mixed array/bitset in the lazy bulk-union path: accumulate
    // into a bitset in place, mirroring CRoaring's LAZY_OR_BITSET_CONVERSION
    // (roaring.c:roaring_bitmap_lazy_or_inplace). Promote the accumulator
    // (left) to a bitset if it is still an array, then OR the operand into
    // its existing 8 KB block: no 8 KB copy, no fresh allocation, no
    // popcount (repaired once at the end), unlike combine_containers_for_policy.
    // Eager (operator|=) and run-typed operands fall through to the general
    // path below, which preserves their sparse-result down-conversion.
    if ( lazy ) {
        auto const left_is_bitset { left_container.holds_bitset()  };
        auto const left_is_array  { left_container.holds_array()   };
        auto const right_is_bitset{ right_container.holds_bitset() };
        auto const right_is_array { right_container.holds_array()  };
        if ( ( left_is_bitset || left_is_array ) && ( right_is_bitset || right_is_array ) ) {
            if ( !left_is_bitset ) {
                // std::as_const avoids an unneeded write-barrier clone under a
                // refcounted CowPolicy: left_container is about to be wholly
                // replaced below, so any clone the mutable as_array() would
                // trigger is discarded immediately.
                auto const values{ std::as_const( left_container ).as_array().values };
                left_container = detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
            }
            auto left_bitset{ left_container.as_bitset() };
            if ( right_is_bitset ) {
                detail::or_bitset_bitset_inplace_lazy<layout_type>( left_bitset, right_container.as_bitset() );
            } else {
                detail::or_array_into_bitset_inplace<true, layout_type>( left_bitset, right_container.as_array() );
            }
            return;
        }
    }
    left_container = combine_containers_for_policy( left_container, right_container, detail::set_operation::bit_or );
}

FRSR_ROARING_BITMAP_TEMPLATE
template <bool Lazy>
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::union_few( bitmap const & other ) -> bitmap & {
    constexpr bool lazy{ Lazy };
    detail::small_array_values<low_type> array_union_scratch;
    std::size_t added{ 0 };
    std::size_t added_size{ 0 };
    auto at{ chunks_.begin() };
    for ( auto right{ other.chunks_.begin() }; right != other.chunks_.end(); ++right ) {
        auto const key{ other.chunks_.key( right ) };
        at = chunks_.lower_bound( key, at );
        auto const & right_container{ other.chunks_.slot( right ) };
        if ( at == chunks_.end() || chunks_.key( at ) != key ) {
            if ( auto const right_size{ detail::container_size( right_container ) }; right_size != 0 ) {
                ++added;
                added_size += right_size;
            }
            continue;
        }
        auto & left_container{ chunks_.slot( at ) };
        if ( !lazy ) {
            size_ -= detail::container_size( left_container );
        }
        union_same_key( chunks_, left_container, right_container, lazy, array_union_scratch );
        if ( !lazy ) {
            size_ += detail::container_size( left_container );
        }
    }
    if ( added != 0 ) {
        chunks_.insert_absent_from( other.chunks_, added, []( handle_type const & container ) noexcept { return detail::container_size( container ) != 0; } );
        size_ += added_size;
    }
    return *this;
}

FRSR_ROARING_BITMAP_TEMPLATE
template <bool Lazy>
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::union_merge( bitmap const & other ) -> bitmap & {
    constexpr bool lazy{ Lazy };
    if ( std::uint64_t{ other.chunks_.size() } * 32 < chunks_.size() ) {
        return union_few<lazy>( other );
    }
    detail::small_array_values<low_type> array_union_scratch;
#if FRSR_ROARING_COMBINE_STATS
    detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif

    // Walk both tables with the containers of shared keys united in place. The table is
    // rebuilt only from the first key the other bitmap holds and this one lacks; while none
    // is met, nothing here moves (`size_` is kept current as the walk goes).
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_or_inplace
    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            ++left;
        } else if ( right_key < left_key ) {
            break;
        } else {
            auto       & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_pair( left_container, right_container, detail::set_operation::bit_or );
#endif
            if ( !lazy ) {
                size_ -= detail::container_size( left_container );
            }
            union_same_key( chunks_, left_container, right_container, lazy, array_union_scratch );
            if ( !lazy ) {
                size_ += detail::container_size( left_container );
            }
            ++left;
            ++right;
        }
    }
    if ( right == other.chunks_.end() ) {
        return *this;
    }

    // The rest of the other bitmap is merged into the rest of this table (or appended to it).
    detail::chunk_store<layout_type, CowPolicy> merged;
    merged.reserve( chunks_.size() + other.chunks_.size() );
    for ( auto kept{ chunks_.begin() }; kept != left; ++kept ) {
        merged.push_back( chunks_.key( kept ), std::move( chunks_.slot( kept ) ) );
    }
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            merged.push_back( left_key, std::move( chunks_.slot( left ) ) );
            ++left;
        } else if ( right_key < left_key ) {
            auto const & right_container{ other.chunks_.slot( right ) };
            if ( detail::container_size( right_container ) != 0 ) {
                if ( !lazy ) {
                    size_ += detail::container_size( right_container );
                }
                merged.push_back( right_key, right_container );
            }
            ++right;
        } else {
            auto       & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_pair( left_container, right_container, detail::set_operation::bit_or );
#endif
            if ( !lazy ) {
                size_ -= detail::container_size( left_container );
            }
            union_same_key( merged, left_container, right_container, lazy, array_union_scratch );
            if ( !lazy ) {
                size_ += detail::container_size( left_container );
            }
            merged.push_back( left_key, std::move( left_container ) );
            ++left;
            ++right;
        }
    }

    while ( left != chunks_.end() ) {
        merged.push_back( chunks_.key( left ), std::move( chunks_.slot( left ) ) );
        ++left;
    }
    while ( right != other.chunks_.end() ) {
        auto const & right_container{ other.chunks_.slot( right ) };
        if ( detail::container_size( right_container ) != 0 ) {
            if ( !lazy ) {
                size_ += detail::container_size( right_container );
            }
            merged.push_back( other.chunks_.key( right ), right_container );
        }
        ++right;
    }

    chunks_ = std::move( merged );
    return *this;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::repair_cardinality() noexcept {
    size_ = 0;
    for ( auto & slot : chunks_.slots() ) {
        if ( slot.holds_bitset() && !slot.cardinality_valid() ) {
            auto bitset{ slot.as_bitset() };
            std::size_t cardinality{ 0 };
            for ( auto const word : bitset.words ) {
                cardinality += static_cast<std::size_t>( std::popcount( word ) );
            }
            bitset.cardinality = static_cast<std::uint32_t>( cardinality );
            bitset.mark_endpoints_stale();
        }
        size_ += detail::container_size( slot );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
template <detail::set_operation Op, bool InPlace>
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::combine( bitmap const & other, bitmap * const out ) -> bitmap & {
    constexpr auto op      { Op };
    constexpr auto in_place{ InPlace };
    assert( in_place == ( out == nullptr ) );
    auto const keep_unmatched_left{ op == detail::set_operation::bit_andnot };
#if FRSR_ROARING_COMBINE_STATS
    detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif
    assert( ( in_place || !keep_unmatched_left ) && "materializing subtract has no caller/arm" );

    typename chunk_store_type::rewriter rewrite{ chunks_ };
    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    if ( in_place ) { size_ = 0; }
    auto & result_size { in_place ? size_    : out->size_    };
    auto & result_chunks{ in_place ? chunks_ : out->chunks_ };
    if ( !in_place ) {
        // One slot per possible match (chunk keys are unique on both sides): result
        // containers are built in their destination slots (chunk_store::open_back),
        // which must not relocate.
        result_chunks.reserve( std::min( chunks_.size(), other.chunks_.size() ) );
    }
    // Store a finished result container for the current left_key. In-place mode
    // rewrites the already-consumed prefix of our own chunk table; materializing
    // mode appends to the (pre-reserved) destination.
    auto const emit{ [&]( auto const key, handle_type && container, size_type const count ) {
#if FRSR_ROARING_COMBINE_STATS
        detail::combine_stats().record_result( container );
#endif
        result_size += count;
        if ( in_place ) {
            // Retire (not free) the consumed slot being overwritten: its
            // payload becomes the next pair's / next call's result buffer
            // via take_retired (scratch-reuse path).
            rewrite.put( key, std::move( container ) );
        } else {
            result_chunks.push_back( key, std::move( container ) );
        }
    } };
    // A result container an arm fills itself (rather than receiving one by
    // value from a helper) is built where it can stay. Materializing mode builds
    // it in the destination's next slot — intersect_into reserved one per
    // possible match — and commits it where it lies. In-place mode cannot: its
    // destination slot may be the very operand the kernel is reading, so it
    // builds in `staged` and emit() moves the finished result into the
    // rewritten prefix. An opened slot stays put until committed, since nothing
    // appends to the destination's live table in between (retire() and
    // demote_sparse_bitset() push only to its retired table). `staged` owns no
    // payload while no result is open, so it is bare storage: each opened result
    // constructs a handle in it, and none is ever destroyed there (a committed
    // result is moved out and an abandoned one emptied), which keeps a handle
    // construction and destruction off every call that opens no result.
    union staged_storage {
        handle_type handle;
        staged_storage() noexcept {}
        ~staged_storage() {}
    } staged;
    // Owns the result under construction until commit_result() hands it on.
    // Leaving scope uncommitted (an empty result, or a kernel whose payload
    // growth threw) empties it, so no payload is left in a slot past the
    // destination's size, nor in `staged`.
    struct result_under_construction {
        handle_type * handle;
        explicit result_under_construction( handle_type & result ) noexcept : handle{ &result } {}
        result_under_construction( result_under_construction const & ) = delete;
        ~result_under_construction() noexcept {
            if ( handle != nullptr ) { *handle = handle_type{}; }
        }
    };
    auto const open_result{ [&]( detail::container_kind const kind ) {
        return result_under_construction{ result_chunks.take_retired_into( in_place ? &staged.handle : result_chunks.open_back(), kind ) };
    } };
    auto const commit_result{ [&]( auto const key, result_under_construction & result, size_type const count ) {
        if ( in_place ) {
            emit( key, std::move( *result.handle ), count );
        } else {
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_result( *result.handle );
#endif
            result_size += count;
            result_chunks.commit_back( key );
        }
        result.handle = nullptr;
    } };

    while ( left != chunks_.end() ) {
        auto const left_key{ chunks_.key( left ) };
        while ( right != other.chunks_.end() && other.chunks_.key( right ) < left_key ) {
            ++right;
        }

        if ( right == other.chunks_.end() || left_key < other.chunks_.key( right ) ) {
            if ( !keep_unmatched_left ) {
                if ( right == other.chunks_.end() ) {
                    break;  // bit_and: nothing to the right can match — drop the remainder via truncate
                }
                ++left;
                continue;
            }
            size_ += detail::container_size( chunks_.slot( left ) );
            rewrite.keep( left );
            ++left;
            continue;
        }

        auto       & left_container { chunks_.slot( left ) };
        auto const & right_container{ other.chunks_.slot( right ) };
#if FRSR_ROARING_COMBINE_STATS
        detail::combine_stats().record_pair( left_container, right_container, op );
#endif
        // A shared left payload is never mutated in place: cloning it only to
        // overwrite the clone costs a full payload copy on top of the kernel,
        // where computing straight into a fresh result costs the kernel alone.
        // So the in-place kernels below run only when we are the sole holder,
        // and a shared left container takes the materializing arm for its pair —
        // emitted into our own table exactly as the materializing arms already
        // do in in-place mode. This is the rule CRoaring applies to a
        // SHARED_CONTAINER_TYPE operand (container_and rather than container_iand).
        bool const mutate_left{ in_place && left_container.payload_is_unshared() };
        if ( left_container.holds_array() && right_container.holds_array() ) {
            // The materializing arm below needs min(|la|,|ra|) + one SIMD vector of
            // result capacity. When the recycled scratch cannot hold that, it has to
            // GROW the payload — an allocation per pair, which at small container
            // cardinalities costs far more than the whole intersection. The in-place
            // kernel writes the result into the left slot's own payload, needs no
            // result buffer at all, and so is taken exactly in that case; when the
            // recycled buffer already fits, the materializing arm is the cheaper of
            // the two (it stores matches straight out instead of through the
            // overwrite-hazard staging the in-place kernel must use).
            // [croaring-ref] deps/croaring/src/array_util.c:intersect_vector16_inplace
            // The in-place kernel pays a fixed per-block staging overhead (it cannot
            // store matches straight into a payload it is still reading), so it only
            // repays the avoided allocation while the blocks are few: measured, it
            // wins by ~30% at 64-element operands and loses by ~30% at ~1000.
            auto const intersect_result_count{ std::min( left_container.count(), right_container.count() ) };
            if ( op == detail::set_operation::bit_and && mutate_left &&
                 intersect_result_count <= detail::kInplaceIntersectMaxElements &&
                 result_chunks.retired_capacity( detail::container_kind::array ) < intersect_result_count + 8U ) {
                detail::intersect_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                if ( auto const count{ detail::container_size( left_container ) }; count != 0 ) {
                    size_ += count;
                    rewrite.keep( left );
                }
            } else if ( op == detail::set_operation::bit_and ) {
                // Intersect directly into a fresh result array's storage: array∩array
                // is always an array (|result| ≤ min(|la|,|ra|) < bitset threshold).
                // A fresh handle is mandatory anyway — the SIMD kernels store full
                // vectors past the true count, so the result cannot alias the input —
                // and writing it inline skips the reusable-scratch buffer plus the
                // scratch→container copy (CRoaring likewise writes the result inline).
                // std::as_const avoids an unneeded write-barrier clone under a
                // refcounted CowPolicy. Seeding from a retired scratch slot makes
                // the inline write land in a reused payload (scratch-reuse path).
                auto result{ open_result( detail::container_kind::array ) };
                auto result_array{ result.handle->as_array() };
                detail::combine_array_array_into<layout_type>(
                    std::as_const( left_container ).as_array(),
                    right_container.as_array(),
                    detail::set_operation::bit_and,
                    result_array.values
                );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    auto const count{ static_cast<size_type>( result_array.values.size() ) };
                    commit_result( left_key, result, count );
                }
            } else if ( mutate_left ) {
                detail::difference_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                if ( left_container.count() != 0 ) {
                    auto container{ make_fast_container( std::move( left_container ) ) };
                    emit( left_key, std::move( container ), detail::container_size( container ) );
                }
            } else {
                // array \ array with a shared left payload: write the survivors into
                // a fresh handle (what difference_into does for every pair).
                auto result{ open_result( detail::container_kind::array ) };
                auto result_array{ result.handle->as_array() };
                detail::difference_array_array_to_vector<layout_type>(
                    std::as_const( left_container ).as_array(),
                    right_container.as_array(),
                    result_array.values
                );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    auto const count{ static_cast<size_type>( result_array.values.size() ) };
                    commit_result( left_key, result, count );
                }
            }
            ++left;
            continue;
        }
        if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
            if ( mutate_left ) {
                auto left_bitset{ left_container.as_bitset() };
                detail::combine_bitset_bitset_inplace<layout_type>( left_bitset, right_container.as_bitset(), op );
                if ( left_bitset.size() != 0 ) {
                    size_ += left_bitset.size();
                    rewrite.keep( left );
                }
            } else {
                // Direct-init (not assignment into a pre-declared handle): the prvalue
                // return is elided straight into `container`, halving the
                // copy_fields_from traffic vs move-assigning into an already-live handle.
                auto container{ [ & ]() -> handle_type {
                    // Opt-in (tuning.hpp): a materializing AND counts the common bits first and, below the
                    // array limit, extracts them straight from the operands' words, as the reference does,
                    // instead of writing an intermediate bitset. The in-place walk keeps its accumulators.
                    if constexpr ( detail::kAndDemotesSparseBitsets ) {
                        if ( !in_place && op == detail::set_operation::bit_and ) {
                            auto const left_bitset { std::as_const( left_container ).as_bitset() };
                            auto const right_bitset{ right_container.as_bitset() };
                            auto const cardinality { detail::and_popcount_words( left_bitset.words.begin(), right_bitset.words.begin(), left_bitset.words.size() ) };
                            if ( cardinality < array_to_bitset_threshold ) {
                                auto array{ detail::intersect_bitset_bitset_to_array<layout_type, CowPolicy>(
                                    left_bitset, right_bitset, cardinality, result_chunks.take_retired( detail::container_kind::array )
                                ) };
                                if constexpr ( uses_default_container_set ) {
                                    return array;
                                } else {
                                    return optimize_container_for_policy( std::move( array ) );
                                }
                            }
                        }
                    }
                    return combine_bitset_bitset_for_policy(
                        std::as_const( left_container ).as_bitset(),
                        right_container.as_bitset(),
                        op,
                        result_chunks.take_retired( detail::container_kind::bitset )
                    );
                }() };
                // AND results NOT demoted (measured): the in-place fold walk is
                // where the run-heavy witness's AND-fold accumulator lives —
                // demoting it to array cost ~12% regardless of threshold or
                // run∩bitset exemption (the demoted accumulator loses the dense
                // arms for every subsequent fold pair). ANDNOT results are
                // demoted: subtraction traffic is not fold-accumulator traffic.
                if ( op == detail::set_operation::bit_andnot ) {
                    container = demote_sparse_bitset( std::move( container ), result_chunks );
                }
                if ( auto const bitset_size{ detail::container_size( container ) }; bitset_size != 0 ) {
                    emit( left_key, std::move( container ), bitset_size );
                }
            }
            ++left;
            continue;
        }
        if ( mutate_left && left_container.holds_array() && right_container.holds_bitset() ) {
            // In-place array∩bitset / array\bitset: the result is a subset of our
            // own array payload, so filter it in place instead of building a fresh
            // container through combine_containers. The mutable as_array() write
            // barrier keeps this alloc-free when we are the sole referent (rc == 1,
            // the common case for an accumulator after its first combine); a shared
            // array clones once — the same one allocation the generic path would
            // pay — then compacts. Result ⊆ the array ⇒ always still an array, so
            // no representation re-decision (deferred to optimize(), exactly as the
            // bitset∩bitset arm above defers). Mirrors that arm's slot handling;
            // only the kernel (and the filter polarity, keyed off op) differs.
            // (Materializing mode must not mutate *this — it takes the generic arm.)
            // mutate_left has already established sole ownership, so the payload
            // is written without a second write-barrier test; and a filter that
            // drops nothing leaves the same set, whose header is still exact.
            auto left_array{ detail::as_array_already_private( left_container ) };
            auto const unfiltered_count{ left_array.values.size() };
            detail::filter_array_bitset_inplace<layout_type>( left_array, right_container.as_bitset(), op == detail::set_operation::bit_and );
            if ( !left_array.values.empty() ) {
                if ( left_array.values.size() != unfiltered_count ) {
                    left_array.sync_header();
                }
                size_ += left_array.values.size();
                rewrite.keep( left );
            }
            ++left;
            continue;
        }
        if ( mutate_left && left_container.holds_array() && right_container.holds_run() ) {
            // In-place array∩run / array\run: same contract as the array-vs-
            // bitset arm above (result ⊆ our own array payload — compact in
            // place, alloc-free at rc == 1, always still an array). Without
            // this arm every RUN-involved pair falls through to the generic
            // combine_containers visit machinery, whose per-pair dispatch +
            // scratch + representation re-decision dominates N-way-AND-fold
            // shapes over run-encoded (storage-optimize()d) operands.
            auto left_array{ left_container.as_array() };
            detail::filter_array_run_inplace<layout_type>( left_array, right_container.as_run(), op == detail::set_operation::bit_and );
            if ( !left_array.values.empty() ) {
                left_array.sync_header();
                size_ += left_array.values.size();
                rewrite.keep( left );
            }
            ++left;
            continue;
        }
        bool const left_is_array { left_container.holds_array()  };
        bool const right_is_array{ right_container.holds_array() };
        if ( ( op == detail::set_operation::bit_and &&
               ( ( left_is_array && right_container.holds_bitset() ) ||
                 ( left_container.holds_bitset() && right_is_array ) ) ) ||
             ( op == detail::set_operation::bit_andnot && left_is_array && right_container.holds_bitset() ) ) {
            // Materializing array∩bitset (either orientation — the in-place
            // left-array case is already consumed by the alloc-free arm above):
            // the result is an array ⊆ the array side, so write it inline into a
            // fresh handle, CRoaring-style, instead of paying the generic
            // visit/variant dispatch plus representation re-decision of
            // combine_containers (the dominant per-pair overhead in profiles).
            auto const & array_side { left_is_array ? std::as_const( left_container ) : right_container };
            auto const & bitset_side{ left_is_array ? right_container : std::as_const( left_container ) };
            // Saturated bitset (its header keeps an exact count, see
            // bitset_ops.hpp's repair note): the intersection IS the array
            // side — share it via a CoW handle copy instead of probing every
            // key against all-ones words (the all-survive regime is the
            // filter kernel's slowest — store-to-load aliasing on the
            // coinciding cursors). Same shape as the full-domain-run arm below.
            if ( op == detail::set_operation::bit_and && detail::container_is_known_full<layout_type, CowPolicy>( bitset_side ) ) {
                if ( auto const count{ static_cast<size_type>( array_side.as_array().values.size() ) }; count != 0 ) {
                    emit( left_key, handle_type{ array_side }, count );
                }
                ++left;
                continue;
            }
            auto result{ open_result( detail::container_kind::array ) };
            auto result_array{ result.handle->as_array() };
            detail::filter_array_bitset_into<layout_type>(
                array_side.as_array(),
                bitset_side.as_bitset(),
                op == detail::set_operation::bit_and,
                result_array.values
            );
            if ( !result_array.values.empty() ) {
                result_array.sync_header();
                auto const count{ static_cast<size_type>( result_array.values.size() ) };
                commit_result( left_key, result, count );
            }
            ++left;
            continue;
        }
        if ( op == detail::set_operation::bit_and &&
             ( ( left_container.holds_run() && right_is_array ) ||
               ( left_is_array && right_container.holds_run() ) ) ) {
            // array-side ∩ run-side (run on either side; the in-place left-array
            // case is already consumed by the alloc-free arm above): the result is
            // an array ⊆ the array side — write it inline into a fresh/retired
            // handle instead of paying the generic visit dispatch + scratch +
            // representation re-decision. This is the dominant fold pair when the
            // ACCUMULATOR is run-encoded (a shallow copy of a storage-optimize()d
            // bitmap, a downstream N-way AND fold seed): run-left probe counts ran
            // ~30x the array-left ones on the run-heavy witness model. Both sides are only
            // read (std::as_const), so a shared left container needs no clone.
            // Outlined: cold for array/bitset-dominated workloads — keeping these
            // bodies inline in the walk cost the bitset-heavy witness ~4% (spine
            // i-cache footprint), same mechanism as the AVX-512 in-place dispatch
            // finding in bitset_ops.hpp.
            [ & ] [[ gnu::noinline ]] () {
                auto const & array_side{ left_is_array ? std::as_const( left_container ) : right_container };
                auto const & run_side  { left_is_array ? right_container : std::as_const( left_container ) };
                // Full-domain run (a storage-optimize()d saturated chunk — the
                // common accumulator seed): the intersection IS the array side, so
                // share its payload via a CoW handle copy instead of galloping and
                // memcpy-ing it into a fresh container (CRoaring clones here; the
                // CoW copy is strictly cheaper). Mirrors intersect_run_run's
                // is_full fast path and the saturated-bitset arm above.
                if ( detail::container_is_known_full<layout_type, CowPolicy>( run_side ) ) {
                    if ( auto const count{ static_cast<size_type>( array_side.as_array().values.size() ) }; count != 0 ) {
                        emit( left_key, handle_type{ array_side }, count );
                    }
                    return;
                }
                auto result{ open_result( detail::container_kind::array ) };
                auto result_array{ result.handle->as_array() };
                detail::filter_array_run_into<layout_type>( array_side.as_array(), run_side.as_run(), result_array.values );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    auto const count{ static_cast<size_type>( result_array.values.size() ) };
                    commit_result( left_key, result, count );
                }
            }();
            ++left;
            continue;
        }
        if ( op == detail::set_operation::bit_and && left_container.holds_run() && right_container.holds_run() ) {
            // run ∩ run: the merge writes its runs straight into a fresh result payload (the
            // reference's shape: the intersection lands in the destination run container, then
            // one size decision), instead of building a standalone run_container and copying
            // it into a handle — two allocations and a copy per pair, most of the cost of a
            // pair of short run lists. Outlined like the other run arms.
            // [croaring-ref] deps/croaring/src/roaring.c:container_and (run, run)
            [ & ] [[ gnu::noinline ]] () {
                auto const & lhs{ std::as_const( left_container ) };
                // A full-domain side leaves the other: share its payload.
                if ( detail::container_is_known_full<layout_type, CowPolicy>( lhs ) ) {
                    if ( auto const count{ static_cast<size_type>( detail::container_size( right_container ) ) }; count != 0 ) {
                        emit( left_key, handle_type{ right_container }, count );
                    }
                    return;
                }
                if ( detail::container_is_known_full<layout_type, CowPolicy>( right_container ) ) {
                    if ( auto const count{ static_cast<size_type>( detail::container_size( lhs ) ) }; count != 0 ) {
                        emit( left_key, handle_type{ lhs }, count );
                    }
                    return;
                }
                // A retired payload is offered by kind, and only the array and bitset kinds are
                // parked, so a run result starts from an empty run handle.
                result_under_construction result{ *std::construct_at( in_place ? &staged.handle : result_chunks.open_back(), handle_type::make_run() ) };
                auto result_run{ result.handle->as_run() };
                auto const cardinality{ detail::intersect_run_run_into<layout_type>( lhs.as_run(), right_container.as_run(), result_run.runs ) };
                if ( cardinality == 0 ) {
                    return;
                }
                auto const run_count{ result_run.runs.size() };
                auto const run_bytes{ run_count * sizeof( typename detail::run_container<layout_type>::run ) };
                auto const array_bytes{ cardinality * sizeof( typename layout_type::low_type ) };
                constexpr auto bitset_bytes{ layout_type::word_count * sizeof( std::uint64_t ) };
                if ( run_bytes <= array_bytes && !( cardinality >= array_to_bitset_threshold && bitset_bytes <= run_bytes ) ) {
                    result.handle->set_cardinality( static_cast<std::uint32_t>( cardinality ) );
                    result.handle->set_endpoints( result_run.runs.front().begin, result_run.runs.back().end );
                    commit_result( left_key, result, static_cast<size_type>( cardinality ) );
                    return;
                }
                // The runs are not the smallest form: the size decision the other run results
                // take, over the runs just written.
                detail::run_container<layout_type> merged;
                merged.runs.reserve( static_cast<std::uint32_t>( run_count ) );
                for ( auto const & current : std::as_const( result_run.runs ) ) { merged.runs.push_back( current ); }
                merged.cardinality = static_cast<typename detail::run_container<layout_type>::cardinality_type>( cardinality );
                *result.handle = handle_type{};
                result.handle = nullptr;  // the slot is free for the emit below
                emit( left_key, detail::make_container_from_runs<layout_type, CowPolicy>( std::move( merged ), array_to_bitset_threshold ), static_cast<size_type>( cardinality ) );
            }();
            ++left;
            continue;
        }
        if ( op == detail::set_operation::bit_and &&
             ( ( left_container.holds_run   () && right_container.holds_bitset() ) ||
               ( left_container.holds_bitset() && right_container.holds_run   () ) ) ) {
            // run ∩ bitset (either orientation): masked source words land directly
            // in a fresh/retired bitset payload with the popcount fused into the
            // fill — the generic path pays a zero-initialized stack word_array, an
            // 8 KB container-build copy, a separate popcount pass, and the visit
            // dispatch. The other dominant pair of the run-encoded-accumulator
            // fold shape (see the array-side ∩ run-side arm above).
            [ & ] [[ gnu::noinline ]] () {   // outlined: see the arm above
                auto const & run_side   { left_container.holds_run() ? std::as_const( left_container ) : right_container };
                auto const & bitset_side{ left_container.holds_run() ? right_container : std::as_const( left_container ) };
                // Sparse runs (result ⊆ runs, guaranteed under the array
                // threshold): extract directly into an array — the full-block
                // fill reads/writes all Layout::word_count words regardless of
                // run coverage, ~an order of magnitude more instructions on the
                // production fold's sparse shapes (HW-counter A/B vs a downstream
                // engine's sparse run∩bitset kernel, which croaring-arm ran here).
                auto const run_cardinality{ detail::container_size( run_side ) };
                if ( run_cardinality >= layout_type::low_domain_size / 2U ) {
                    // Runs covering at least half the domain (incl. the full-domain
                    // run — CRoaring's run_container_is_full short-circuit): clone
                    // the bitset payload and clear only the gaps, with the
                    // cardinality maintained by subtraction. The masked fill below
                    // costs per covered word, this form per gap word plus one 8 KB
                    // copy; measured on 64 runs the two tie at half coverage and
                    // this form wins 40 % at three quarters — cheaper than masking
                    // all 8 KB through the fill kernel below.
                    auto container{ detail::intersect_run_bitset_dense_runs<layout_type, CowPolicy>(
                        std::as_const( run_side ).as_run(),
                        std::as_const( bitset_side ).as_bitset(),
                        detail::container_size( bitset_side ),
                        result_chunks.take_retired( detail::container_kind::bitset )
                    ) };
                    if constexpr ( !uses_default_container_set ) {
                        container = optimize_container_for_policy( std::move( container ) );
                    }
                    if ( auto const bitset_size{ detail::container_size( container ) }; bitset_size != 0 ) {
                        emit( left_key, std::move( container ), bitset_size );
                    }
                    return;
                }
                if ( run_cardinality < array_to_bitset_threshold ) {
                    auto container{ detail::intersect_run_bitset_sparse<layout_type, CowPolicy>(
                        run_side.as_run(),
                        bitset_side.as_bitset(),
                        static_cast<std::uint32_t>( std::min( run_cardinality, detail::container_size( bitset_side ) ) ),
                        result_chunks.take_retired( detail::container_kind::array )
                    ) };
                    if ( auto const array_size{ detail::container_size( container ) }; array_size != 0 ) {
                        emit( left_key, std::move( container ), array_size );
                    }
                    return;
                }
                // Many runs covering 3/16 to 7/16 of the domain: clone, clear the gaps
                // and recount once. The masked fill below grows with the covered words
                // (a mask setup, a store and a popcount each, per run); this form costs
                // a block copy, the gap stores and one vector recount, flat across that
                // range. Below it the fill's covered words cost less than the copy and
                // recount, and on 64 runs the per-gap clears of this form lose from 7/16
                // on; from half coverage the subtracting clone above serves.
                if ( std::as_const( run_side ).as_run().runs.size() > 32U &&
                     run_cardinality >= layout_type::low_domain_size * 3U / 16U &&
                     run_cardinality <  layout_type::low_domain_size * 7U / 16U ) {
                    auto container{ detail::intersect_run_bitset_clone_and_recount<layout_type, CowPolicy>(
                        std::as_const( run_side ).as_run(),
                        std::as_const( bitset_side ).as_bitset(),
                        result_chunks.take_retired( detail::container_kind::bitset )
                    ) };
                    if constexpr ( !uses_default_container_set ) {
                        container = optimize_container_for_policy( std::move( container ) );
                    }
                    if ( auto const bitset_size{ detail::container_size( container ) }; bitset_size != 0 ) {
                        emit( left_key, std::move( container ), bitset_size );
                    }
                    return;
                }
                // Dense runs: materialize into a (retired or fresh) bitset. NOT
                // demoted (unlike the bitset×bitset sites below/above): the
                // run-heavy fold witness regressed ~12% with demotion here — its
                // chained folds keep re-intersecting runs against the SAME
                // accumulator, and a demoted array accumulator loses the dense
                // word-masked arm on every subsequent pair.
                auto container{ intersect_run_bitset_for_policy(
                    run_side.as_run(),
                    bitset_side.as_bitset(),
                    result_chunks.take_retired( detail::container_kind::bitset )
                ) };
                if ( auto const bitset_size{ detail::container_size( container ) }; bitset_size != 0 ) {
                    emit( left_key, std::move( container ), bitset_size );
                }
            }();
            ++left;
            continue;
        }
        if ( mutate_left && op == detail::set_operation::bit_andnot &&
             left_container.holds_bitset() && right_container.holds_array() ) {
            // In-place bitset\array: clear the array's bits from our own 8 KB
            // block (exact cardinality bookkeeping, no allocation; the mutable
            // as_bitset() write barrier clones once at rc > 1, as elsewhere).
            [ & ] [[ gnu::noinline ]] () {   // outlined: see the array∩run arm above
                auto left_bitset{ left_container.as_bitset() };
                detail::difference_bitset_array_inplace<layout_type>( left_bitset, right_container.as_array() );
                if ( left_bitset.size() != 0 ) {
                    size_ += left_bitset.size();
                    rewrite.keep( left );
                }
            }();
            ++left;
            continue;
        }
        auto container{ combine_containers_for_policy(
            left_container,
            right_container,
            op
        ) };
        auto const container_size{ detail::container_size( container ) };
        if ( container_size != 0 ) {
            emit( left_key, std::move( container ), container_size );
        }
        ++left;
    }

    if ( in_place ) {
        rewrite.finish();
    } else {
    }
    return *this;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_or_inplace( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
    // Counts are not maintained by the fold.
    size_ = size_deferred;
    auto const & rhs{ other.chunks_ };
    auto pos1{ chunks_.begin() };
    auto pos2{ rhs.begin() };
    while ( pos1 != chunks_.end() && pos2 != rhs.end() ) {
        auto const left_key { chunks_.key( pos1 ) };
        auto const right_key{ rhs.key( pos2 ) };
        if ( left_key < right_key ) {
            ++pos1;  // this-only key: already in place, no move
            continue;
        }
        if ( right_key < left_key ) {  // other-only key: splice a copy in (rare on the leading edge)
            if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
                pos1 = chunks_.insert( pos1, right_key, rhs.slot( pos2 ) );
                ++pos1;
            }
            ++pos2;
            continue;
        }
        // same key: OR rhs[pos2] into chunks_[pos1] in place (lazy — no popcount).
        auto       & left_container { chunks_.slot( pos1 ) };
        auto const & right_container{ rhs.slot( pos2 ) };
        // [croaring-ref] roaring_bitmap_lazy_or_inplace skips the combine when
        // container_is_full(lhs) — OR into a saturated chunk is a no-op. A lazy
        // bitset may be full without us knowing (cardinality stale), so this only
        // fires when the header still carries an exact full count (fresh copy of a
        // full operand, or a prior card-updating bulk OR below).
        if ( detail::container_is_known_full<layout_type, CowPolicy>( left_container ) ) {
            ++pos1;
            ++pos2;
            continue;
        }
        // A small array pair merges in place (lazy_union_array_bound, tuning.hpp:
        // the reference converts unconditionally; the bound is where that
        // measurably loses).
        if ( left_container.holds_array() && right_container.holds_array() &&
             ( static_cast<std::size_t>( left_container.count() ) + right_container.count() ) <= detail::lazy_union_array_bound ) {
            detail::union_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
            ++pos1;
            ++pos2;
            continue;
        }
        if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
            // Prefer the card-updating bulk OR (outlined) so chunks that
            // saturate mid-fold become known-full for later skips. nocard
            // lazy OR left for non-bulk callers.
            detail::or_bitset_bitset_inplace_bulk<layout_type>( left_container.as_bitset(), right_container.as_bitset() );
            ++pos1;
            ++pos2;
            continue;
        }
        // Array/bitset pairs: the accumulator becomes a bitset on its first
        // same-key fold, whatever its size, and every operand is then OR'd into
        // that block in place. That is the reference's in-place lazy union with
        // bitsetconversion (LAZY_OR_BITSET_CONVERSION is true): an array-merge
        // arm that kept small accumulators as arrays rewrote the whole
        // accumulator on every fold — O(K·n) per chunk — and measured 2.4x the
        // reference on a 16-way union of 32-element operands.
        {
            auto const left_is_bitset { left_container.holds_bitset()  };
            auto const left_is_array  { left_container.holds_array()   };
            auto const right_is_bitset{ right_container.holds_bitset() };
            auto const right_is_array { right_container.holds_array()  };
            if ( ( left_is_bitset || left_is_array ) && ( right_is_bitset || right_is_array ) ) {
                if ( !left_is_bitset ) {
                    // std::as_const avoids an unneeded write-barrier clone under a
                    // refcounted CowPolicy: left_container is about to be wholly
                    // replaced below, so any clone the mutable as_array() would
                    // trigger is discarded immediately.
                    auto const values{ std::as_const( left_container ).as_array().values };
                    left_container = detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
                }
                auto left_bitset{ left_container.as_bitset() };
                if ( right_is_bitset ) {
                    detail::or_bitset_bitset_inplace_bulk<layout_type>( left_bitset, right_container.as_bitset() );
                } else {
                    detail::or_array_into_bitset_inplace<true, layout_type>( left_bitset, right_container.as_array() );
                }
                ++pos1;
                ++pos2;
                continue;
            }
        }
        // Run-typed operand (no SIMD in-place form): materialized combine into the slot, which reads the
        // exact count of a bitset it meets.
        left_container.settle_cardinality();
        left_container = combine_containers_for_policy( left_container, right_container, detail::set_operation::bit_or );
        ++pos1;
        ++pos2;
    }
    // Tail: append other's remaining chunks (copies).
    while ( pos2 != rhs.end() ) {
        if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
            chunks_.push_back( rhs.key( pos2 ), rhs.slot( pos2 ) );
        }
        ++pos2;
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_or_finish() noexcept( detail::allocation_failure_terminates ) {
    repair_cardinality();
    for ( auto & slot : chunks_.slots() ) {
        slot = optimize_container_for_policy( std::move( slot ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_or_finish_keep_bitsets() noexcept( detail::allocation_failure_terminates ) {
    repair_cardinality();  // bulk_or_intermediate left bitset cardinalities stale
    for ( auto & slot : chunks_.slots() ) {
        slot = optimize_container_keep_bitsets_for_policy<detail::run_selection_eager>( std::move( slot ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_or_repair_after_lazy() noexcept( detail::allocation_failure_terminates ) {
    repair_cardinality();
    for ( auto & slot : chunks_.slots() ) {
        if ( slot.holds_bitset() && slot.cardinality() <= array_to_bitset_threshold ) {
            auto array{ detail::array_from_bitset<layout_type, CowPolicy>(
                std::as_const( slot ).as_bitset(),
                slot.cardinality()
            ) };
            chunks_.retire( std::move( slot ) );
            slot = std::move( array );
        } else if ( slot.holds_run() ) {
            slot = optimize_container_for_policy<detail::run_selection_eager>( std::move( slot ) );
        }
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::or_many_in_place( std::span<bitmap const * const> const others ) noexcept( detail::allocation_failure_terminates ) {
    for ( auto const other : others ) {
        if ( other != nullptr ) {
            bulk_or_intermediate( *other );
        }
    }
    bulk_or_repair_after_lazy();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::or_many_sorted_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates ) {
    std::ranges::sort( others, []( bitmap const * const lhs, bitmap const * const rhs ) {
        return lhs->data_byte_size() > rhs->data_byte_size();
    } );
    or_many_in_place( { others.data(), others.size() } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::or_many_heap_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates ) {
    or_many_sorted_in_place( others );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::xor_result_for_policy( handle_type container ) -> handle_type {
    if constexpr ( uses_default_container_set ) {
        return container;
    } else {
        return optimize_container_for_policy( std::move( container ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::xor_containers_for_policy( handle_type const & lhs, handle_type const & rhs ) -> handle_type {
    return xor_result_for_policy( detail::xor_containers<layout_type, CowPolicy>( lhs, rhs, array_to_bitset_threshold ) );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::settle_xor_bitset( handle_type && container, detail::chunk_store<layout_type, CowPolicy> & store ) -> handle_type {
    return detail::settle_xor_bitset<layout_type, CowPolicy>(
        std::move( container ), array_to_bitset_threshold, [ &store ]( handle_type && vacated ) noexcept {
            try { store.retire( std::move( vacated ) ); } catch ( std::bad_alloc const & ) {}
        } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::xor_pair( detail::chunk_store<layout_type, CowPolicy> & store, handle_type const & lhs, handle_type const & rhs ) -> handle_type {
    if ( lhs.holds_array() && rhs.holds_array() ) {
        return xor_result_for_policy( detail::xor_array_array<layout_type, CowPolicy>( lhs.as_array(), rhs.as_array(), array_to_bitset_threshold ) );
    }
    if ( lhs.holds_bitset() && rhs.holds_bitset() ) {
        return settle_xor_bitset( combine_bitset_bitset_for_policy(
            lhs.as_bitset(), rhs.as_bitset(), detail::set_operation::bit_xor, store.take_retired( detail::container_kind::bitset ) ), store );
    }
    return xor_containers_for_policy( lhs, rhs );
}

FRSR_ROARING_BITMAP_TEMPLATE
void FRSR_ROARING_BITMAP::xor_same_key( detail::chunk_store<layout_type, CowPolicy> & store, handle_type & left_container, handle_type const & right_container ) {
    if ( left_container.holds_bitset() && left_container.payload_is_unshared() ) {
        if ( right_container.holds_bitset() ) {
            detail::combine_bitset_bitset_inplace<layout_type>( left_container.as_bitset(), right_container.as_bitset(), detail::set_operation::bit_xor );
        } else if ( right_container.holds_array() ) {
            detail::xor_array_into_bitset_inplace<false, layout_type>( left_container.as_bitset(), right_container.as_array() );
        } else {
            detail::xor_runs_into_bitset_inplace<false, layout_type>( left_container.as_bitset(), right_container.as_run() );
        }
        left_container = settle_xor_bitset( std::move( left_container ), store );
        return;
    }
    left_container = xor_pair( store, left_container, right_container );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::xor_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates ) {
    scratch.clear_keep_capacity();
    if ( this == &other ) {
        return;
    }
    scratch.chunks_.reserve( chunks_.size() + other.chunks_.size() );
    auto const push_alone{ [ & ]( chunk_type const key, handle_type const & container ) {
        if ( auto const count{ detail::container_size( container ) }; count != 0 ) {
            scratch.size_ += count;
            scratch.chunks_.push_back( key, container );
        }
    } };
    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            push_alone( left_key, chunks_.slot( left ) );
            ++left;
        } else if ( right_key < left_key ) {
            push_alone( right_key, other.chunks_.slot( right ) );
            ++right;
        } else {
            auto container{ xor_pair( scratch.chunks_, chunks_.slot( left ), other.chunks_.slot( right ) ) };
            if ( auto const count{ detail::container_size( container ) }; count != 0 ) {
                scratch.size_ += count;
                scratch.chunks_.push_back( left_key, std::move( container ) );
            }
            ++left;
            ++right;
        }
    }
    for ( ; left != chunks_.end(); ++left ) {
        push_alone( chunks_.key( left ), chunks_.slot( left ) );
    }
    for ( ; right != other.chunks_.end(); ++right ) {
        push_alone( other.chunks_.key( right ), other.chunks_.slot( right ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::xor_merge( bitmap const & other ) -> bitmap & {
    settle_counts();
    if ( this == &other ) {
        clear();
        return *this;
    }
    if ( other.chunks_.empty() ) {
        return *this;
    }
    if ( chunks_.empty() ) {
        return *this = other;
    }
    auto left{ chunks_.begin() };
    auto right{ other.chunks_.begin() };
    bool cancelled{ false };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            ++left;
        } else if ( right_key < left_key ) {
            break;
        } else {
            auto       & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
            {
                size_reconciliation const reconcile{ size_, left_container };
                xor_same_key( chunks_, left_container, right_container );
            }
            ++right;
            if ( detail::container_size( left_container ) == 0 ) {
                cancelled = true;
                break;   // `left` stays on the emptied slot, which the rebuild below drops
            }
            ++left;
        }
    }
    if ( right == other.chunks_.end() && !cancelled ) {
        return *this;
    }

    try {
    detail::chunk_store<layout_type, CowPolicy> merged;
    merged.reserve( chunks_.size() + other.chunks_.size() );
    for ( auto kept{ chunks_.begin() }; kept != left; ++kept ) {
        merged.push_back( chunks_.key( kept ), std::move( chunks_.slot( kept ) ) );
    }
    auto const keep_left{ [ & ]( auto const at ) {
        if ( detail::container_size( chunks_.slot( at ) ) != 0 ) {
            merged.push_back( chunks_.key( at ), std::move( chunks_.slot( at ) ) );
        }
    } };
    auto const take_right{ [ & ]( auto const at ) {
        auto const & container{ other.chunks_.slot( at ) };
        if ( auto const count{ detail::container_size( container ) }; count != 0 ) {
            size_ += count;
            merged.push_back( other.chunks_.key( at ), container );
        }
    } };
    while ( left != chunks_.end() && right != other.chunks_.end() ) {
        auto const left_key { chunks_.key( left ) };
        auto const right_key{ other.chunks_.key( right ) };
        if ( left_key < right_key ) {
            keep_left( left );
            ++left;
        } else if ( right_key < left_key ) {
            take_right( right );
            ++right;
        } else {
            auto       & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
            {
                size_reconciliation const reconcile{ size_, left_container };
                xor_same_key( merged, left_container, right_container );
            }
            keep_left( left );
            ++left;
            ++right;
        }
    }
    for ( ; left != chunks_.end(); ++left ) { keep_left( left ); }
    for ( ; right != other.chunks_.end(); ++right ) { take_right( right ); }
    chunks_ = std::move( merged );
    } catch ( ... ) {
        clear();   // slots have been moved into `merged`: what is left of the table is not the bitmap
        throw;
    }
    return *this;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_xor_intermediate( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
    // Counts are not maintained by the fold.
    size_ = size_deferred;
    if ( this == &other ) {
        clear();
        return;
    }
    auto const & rhs{ other.chunks_ };
    auto pos1{ chunks_.begin() };
    auto pos2{ rhs.begin() };
    while ( pos1 != chunks_.end() && pos2 != rhs.end() ) {
        auto const left_key { chunks_.key( pos1 ) };
        auto const right_key{ rhs.key( pos2 ) };
        if ( left_key < right_key ) {
            ++pos1;
            continue;
        }
        if ( right_key < left_key ) {
            if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
                pos1 = chunks_.insert( pos1, right_key, rhs.slot( pos2 ) );
                ++pos1;
            }
            ++pos2;
            continue;
        }
        lazy_xor_same_key( chunks_, chunks_.slot( pos1 ), rhs.slot( pos2 ) );
        ++pos1;
        ++pos2;
    }
    for ( ; pos2 != rhs.end(); ++pos2 ) {
        if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
            chunks_.push_back( rhs.key( pos2 ), rhs.slot( pos2 ) );
        }
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::lazy_xor_same_key( detail::chunk_store<layout_type, CowPolicy> & store, handle_type & left_container, handle_type const & right_container ) {
    if ( left_container.holds_array() && right_container.holds_array() &&
         left_container.count() + right_container.count() > detail::lazy_xor_array_bound ) {
        left_container = detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>(
            { std::as_const( left_container ).as_array().values.data(), left_container.count() } );
    } else if ( left_container.holds_array() && right_container.holds_bitset() ) {
        auto result{ detail::clone_bitset_words<layout_type, CowPolicy>( right_container.as_bitset() ) };
        detail::xor_array_into_bitset_inplace<true, layout_type>( result.as_bitset(), std::as_const( left_container ).as_array() );
        left_container = std::move( result );
        return;
    }
    if ( left_container.holds_bitset() ) {
        if ( right_container.holds_bitset() ) {
            detail::xor_bitset_bitset_inplace_lazy<layout_type>( left_container.as_bitset(), right_container.as_bitset() );
            return;
        }
        if ( right_container.holds_array() ) {
            detail::xor_array_into_bitset_inplace<true, layout_type>( left_container.as_bitset(), right_container.as_array() );
            return;
        }
        detail::xor_runs_into_bitset_inplace<true, layout_type>( left_container.as_bitset(), right_container.as_run() );
        return;
    }
    left_container = xor_pair( store, left_container, right_container );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::bulk_xor_repair_after_lazy() noexcept( detail::allocation_failure_terminates ) {
    repair_cardinality();
    bool any_empty{ false };
    for ( auto & slot : chunks_.slots() ) {
        if ( slot.holds_bitset() && slot.cardinality() <= array_to_bitset_threshold ) {
            slot = settle_xor_bitset( std::move( slot ), chunks_ );
        } else if ( slot.holds_run() ) {
            slot = optimize_container_for_policy<detail::run_selection_eager>( std::move( slot ) );
        }
        any_empty |= slot.cardinality() == 0;
    }
    if ( any_empty ) {
        detail::interop_engine<bitmap>::drop_empty_chunks( *this );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::xor_many_in_place( std::span<bitmap const * const> const others ) noexcept( detail::allocation_failure_terminates ) {
    try {
        for ( auto const other : others ) {
            if ( other != nullptr ) {
                bulk_xor_intermediate( *other );
            }
        }
    } catch ( ... ) {
        clear();   // the folded containers are not counted and some may have cancelled: valid, contents unspecified
        throw;
    }
    bulk_xor_repair_after_lazy();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::xor_many( std::span<bitmap const * const> const operands ) noexcept( detail::allocation_failure_terminates ) -> bitmap {
    bitmap result;
    result.xor_many_in_place( operands );
    return result;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::and_many_sorted_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates ) {
    std::ranges::sort( others, []( bitmap const * const lhs, bitmap const * const rhs ) {
        return lhs->data_byte_size() < rhs->data_byte_size();
    } );
    for ( auto const other : others ) {
        if ( other != nullptr ) {
            *this &= *other;
            if ( empty() ) {
                return;
            }
        }
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::and_many_cardinality( std::span<bitmap const * const> const operands ) -> size_type {
    and_many_scratch scratch;
    size_type total{ 0 };
    and_many_walk( operands, [ & ]( chunk_type const, std::span<handle_type const * const> const ordered ) {
        // The largest container is only counted against the intersection of the others, never
        // combined with it.
        if ( ordered.size() == 1 ) {
            total += ordered[ 0 ]->cardinality();
        } else if ( ordered.size() == 2 ) {
            total += and_many_count_pair( *ordered[ 0 ], *ordered[ 1 ], scratch );
        } else {
            bool owned{ false };
            if ( auto const * const acc{ and_chunk( ordered.first( ordered.size() - 1 ), scratch, owned ) } ) {
                total += and_many_count_pair( *acc, *ordered.back(), scratch );
            }
        }
        return true;
    } );
    return total;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE bool FRSR_ROARING_BITMAP::and_many_intersects( std::span<bitmap const * const> const operands ) {
    and_many_scratch scratch;
    bool found{ false };
    and_many_walk( operands, [ & ]( chunk_type const, std::span<handle_type const * const> const ordered ) {
        if ( ordered.size() == 1 ) {
            found = true;
        } else if ( ordered.size() == 2 ) {
            found = detail::container_intersects( *ordered[ 0 ], *ordered[ 1 ] );
        } else {
            bool owned{ false };
            auto const * const acc{ and_chunk( ordered.first( ordered.size() - 1 ), scratch, owned ) };
            found = acc != nullptr && detail::container_intersects( *acc, *ordered.back() );
        }
        return !found;
    } );
    return found;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE bool FRSR_ROARING_BITMAP::and_many_intersects( std::span<bitmap const * const> const operands, bitmap const & other ) {
    and_many_buffer<bitmap const *> all{ operands.size() + 1 };
    std::size_t count{ 0 };
    for ( auto const * const operand : operands ) {
        if ( operand != nullptr ) { all.data[ count++ ] = operand; }
    }
    all.data[ count++ ] = &other;
    return and_many_intersects( std::span<bitmap const * const>{ all.data, count } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::intersect_into( typed_view<bitmap> const view, bitmap & out ) const noexcept( detail::allocation_failure_terminates ) {
    assert( &out != this && &out != &view.get() );
    if ( &view.get() == this ) {
        // the view names this bitmap: the answer is this bitmap, copied (a counted copy would write the counts of a view's bitmap)
        std::array<bitmap const *, 1> const same{ this };
        and_many_into( same, out );
        return;
    }
    detail::store_source source{ view.get().chunks_ };
    detail::interop_engine<bitmap>::and_stream_into( *this, source, out );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::and_many( std::span<bitmap const * const> const operands ) -> bitmap {
    bitmap result;
    and_many_into( operands, result );
    return result;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::and_many_into( std::span<bitmap const * const> const operands, bitmap & result ) noexcept( detail::allocation_failure_terminates ) {
    assert( std::ranges::none_of( operands, [ &result ]( bitmap const * const operand ) { return operand == &result; } ) && "the result is none of the operands" );
    result.clear_keep_capacity();
    and_many_scratch scratch;
    and_many_walk( operands, [ & ]( chunk_type const key, std::span<handle_type const * const> const ordered ) {
        bool owned{ false };
        auto const * const acc{ and_chunk( ordered, scratch, owned ) };
        if ( acc == nullptr ) {
            return true;
        }
        auto const count{ static_cast<size_type>( acc->cardinality() ) };
        if ( !owned ) {
            // the only operand: a copy of its container that shares nothing with it (a counted copy would write the
            // operand's reference count)
            result.chunks_.push_back( key, detail::interop_engine<bitmap>::own( *acc ) );
        } else if ( auto & mine{ const_cast<handle_type &>( *acc ) }; mine.holds_bitset() && count <= array_to_bitset_threshold ) {
            result.chunks_.push_back( key, detail::array_handle_from_container<layout_type, CowPolicy>( mine ) );
        } else {
            result.chunks_.push_back( key, std::move( mine ) );
        }
        result.size_ += count;
        return true;
    } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::and_many_seed_bitset( and_many_scratch & scratch, handle_type const & source ) {
    auto & acc{ scratch.bitset_acc };
    if ( !acc.holds_bitset() ) {
        acc = handle_type::make_bitset_uninitialized();
    }
    auto words{ acc.as_bitset() };
    std::memcpy( words.words.as_array().data(), source.as_bitset().words.as_array().data(), layout_type::word_count * sizeof( std::uint64_t ) );
    words.cardinality = static_cast<std::uint32_t>( source.cardinality() );
    words.mark_endpoints_stale();
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::and_many_count_pair( handle_type const & acc, handle_type const & last, and_many_scratch & scratch ) -> size_type {
    if ( !( acc.holds_array() && last.holds_array() ) ) {
        return static_cast<size_type>( detail::container_intersect_cardinality<layout_type, CowPolicy>( acc, last ) );
    }
    auto common{ scratch.array_alt.as_array() };
    detail::combine_array_array_into<layout_type>( acc.as_array(), last.as_array(), detail::set_operation::bit_and, common.values );
    return static_cast<size_type>( common.values.size() );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::and_chunk( std::span<handle_type const * const> const ordered, and_many_scratch & scratch, bool & owned ) -> handle_type const * {
    owned = false;
    if ( ordered.size() == 1 ) {
        return ordered[ 0 ];
    }
    owned = true;
    auto const & seed{ *ordered[ 0 ] };
    handle_type * acc{};
    switch ( seed.kind() ) {
        case detail::container_kind::array: {
            auto const source{ seed.as_array() };
            auto       target{ scratch.array_acc.as_array() };
            detail::resize_uninitialized( target.values, source.values.size() );
            std::memcpy( target.values.data(), source.values.data(), source.values.size() * sizeof( low_type ) );
            target.sync_header();
            acc = &scratch.array_acc;
            break;
        }
        case detail::container_kind::bitset:
            and_many_seed_bitset( scratch, seed );
            acc = &scratch.bitset_acc;
            break;
        case detail::container_kind::run: {
            auto const runs{ seed.as_run() };
            if ( seed.cardinality() <= array_to_bitset_threshold ) {
                auto target{ scratch.array_acc.as_array() };
                detail::resize_uninitialized( target.values, seed.cardinality() );
                auto * out{ target.values.data() };
                for ( auto const & run : runs.runs ) {
                    for ( auto value{ static_cast<std::uint32_t>( run.begin ) }; value <= run.end; ++value ) {
                        *out++ = static_cast<low_type>( value );
                    }
                }
                target.sync_header();
                acc = &scratch.array_acc;
            } else {
                auto & bits{ scratch.bitset_acc };
                if ( !bits.holds_bitset() ) { bits = handle_type::make_bitset_uninitialized(); }
                auto words{ bits.as_bitset() };
                std::memset( words.words.as_array().data(), 0, layout_type::word_count * sizeof( std::uint64_t ) );
                for ( auto const & run : runs.runs ) {
                    detail::apply_range_to_words<layout_type>( words.words.as_array(), run.begin, run.end, detail::range_operation::add );
                }
                words.cardinality = static_cast<std::uint32_t>( seed.cardinality() );
                words.mark_endpoints_stale();
                acc = &scratch.bitset_acc;
            }
            break;
        }
    }

    for ( std::size_t i{ 1 }; i < ordered.size(); ++i ) {
        auto const & other{ *ordered[ i ] };
        if ( acc->holds_array() ) {
            auto array{ acc->as_array() };
            switch ( other.kind() ) {
                case detail::container_kind::array : detail::intersect_array_array_inplace<layout_type>( array, other.as_array() ); break;
                case detail::container_kind::bitset: detail::filter_array_bitset_inplace  <layout_type>( array, other.as_bitset(), true ); break;
                case detail::container_kind::run   : detail::filter_array_run_inplace     <layout_type>( array, other.as_run(), true ); break;
            }
            array.sync_header();
        } else {
            switch ( other.kind() ) {
                case detail::container_kind::bitset:
                    detail::combine_bitset_bitset_inplace<layout_type, CowPolicy>( acc->as_bitset(), other.as_bitset(), detail::set_operation::bit_and );
                    break;
                case detail::container_kind::array: {
                    auto alt{ scratch.array_alt.as_array() };
                    detail::filter_array_bitset_into<layout_type>( other.as_array(), std::as_const( *acc ).as_bitset(), true, alt.values );
                    alt.sync_header();
                    std::swap( scratch.array_acc, scratch.array_alt );
                    acc = &scratch.array_acc;
                    break;
                }
                case detail::container_kind::run:
                    scratch.bitset_alt = detail::intersect_run_bitset<layout_type, CowPolicy>( other.as_run(), std::as_const( *acc ).as_bitset(), std::move( scratch.bitset_alt ) );
                    std::swap( scratch.bitset_acc, scratch.bitset_alt );
                    acc = &scratch.bitset_acc;
                    break;
            }
        }
        if ( acc->cardinality() == 0 ) {
            return nullptr;
        }
    }
    return acc;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE std::size_t FRSR_ROARING_BITMAP::shrink_to_fit() noexcept( detail::allocation_failure_terminates ) {
    std::size_t freed{ 0 };
    for ( auto & slot : chunks_.slots() ) {
        freed += detail::container_shrink_to_fit( slot );
    }
    chunks_.shrink_to_fit();
    return freed;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::promote_large_arrays( std::uint16_t const threshold ) noexcept( detail::allocation_failure_terminates ) {
    for ( auto & slot : chunks_.slots() ) {
        if ( slot.holds_array() && slot.count() >= threshold ) {
            if constexpr ( supports_bitset_container ) {
                // std::as_const avoids an unneeded write-barrier clone under a
                // refcounted CowPolicy: slot is about to be wholly replaced below.
                auto const values{ std::as_const( slot ).as_array().values };
                slot = detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
            } else {
                slot = optimize_container_for_policy( std::move( slot ) );
            }
        }
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::optimize() noexcept( detail::allocation_failure_terminates ) {
    settle_counts();
    for ( auto & slot : chunks_.slots() ) {
        slot = optimize_container_for_policy<detail::run_selection_eager>( std::move( slot ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::optimize_keep_bitsets() noexcept( detail::allocation_failure_terminates ) {
    settle_counts();
    for ( auto & slot : chunks_.slots() ) {
        if constexpr ( supports_bitset_container && uses_default_container_set ) {
            if ( slot.holds_bitset() ) {
                // Reconsider run encoding, but keep the bitset we already hold if
                // the optimizer would down-convert to an array. Decided from the
                // words: the previous formulation passed the slot BY VALUE so it
                // could fall back to it, which cloned the whole 8 KB payload of
                // every bitset in the bitmap on a call that usually changes nothing.
                slot = detail::optimize_bitset_run_only<layout_type, CowPolicy, detail::run_selection_eager>(
                    std::move( slot ), array_to_bitset_threshold
                );
                continue;
            }
        } else if constexpr ( supports_bitset_container ) {
            if ( slot.holds_bitset() ) {
                auto optimized{ optimize_container_for_policy<detail::run_selection_eager>( slot ) };
                if ( !optimized.holds_array() ) {
                    slot = std::move( optimized );
                }
                continue;
            }
        }
        slot = optimize_container_keep_bitsets_for_policy<detail::run_selection_eager>( std::move( slot ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::optimize_run_shaped( std::size_t const min_mean_run_length ) noexcept( detail::allocation_failure_terminates ) {
    settle_counts();
    for ( auto & slot : chunks_.slots() ) {
        if ( slot.holds_run() ) {
            continue;
        }
        auto const cardinality{ detail::container_size( slot ) };
        // runs * min_mean_run_length <= cardinality  ⇔  runs <= cardinality / min_mean_run_length
        auto const max_runs{ cardinality / min_mean_run_length };
        auto const runs{
            slot.holds_array()
                ? [ & ] {
                      auto const values{ std::as_const( slot ).as_array().values };
                      return detail::run_count_from_sorted_values<layout_type>( { values.data(), values.size() }, max_runs + 1U );
                  }()
                : detail::run_count_from_bitset_words<layout_type>(
                      std::span<std::uint64_t const>{ std::as_const( slot ).as_bitset().words.as_array() },
                      max_runs + 1U
                  )
        };
        if ( runs > max_runs ) {
            continue;
        }
        auto const values{ detail::sorted_values_from_container<layout_type>( slot ) };
        slot = detail::run_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::optimize_for_speed( std::uint16_t const threshold ) noexcept( detail::allocation_failure_terminates ) {
    promote_large_arrays( threshold );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::make_fast_container_from_scratch_reusing( detail::chunk_store<layout_type, CowPolicy> & store, detail::small_array_values<low_type> const & values ) -> handle_type {
    if constexpr ( supports_bitset_container ) {
        if ( values.size() >= array_to_bitset_threshold ) {
            return detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>(
                { values.data(), values.size() },
                store.take_retired( detail::container_kind::bitset )
            );
        }
    }
    auto dst{ store.take_retired( detail::container_kind::array ) };
    if ( !dst.spilled() || values.empty() ) [[unlikely]] {
        return handle_type::make_array_from_sorted( { values.data(), values.size() } );
    }
    auto const count{ static_cast<std::uint32_t>( values.size() ) };
    dst.template ensure_payload_capacity<low_type>( count );
    std::memcpy( dst.template payload_data<low_type>(), values.data(), std::size_t{ count } * sizeof( low_type ) );
    dst.set_count( count );
    dst.set_cardinality( count );
    dst.set_endpoints( values.front(), values.back() );
    return dst;
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::make_array_union_container_reusing( detail::chunk_store<layout_type, CowPolicy> & store, detail::array_cref<layout_type, CowPolicy> const lhs, detail::array_cref<layout_type, CowPolicy> const rhs, detail::small_array_values<low_type> & scratch ) -> handle_type {
    if ( lhs.values.size() + rhs.values.size() < array_to_bitset_threshold ) {
        handle_type result_handle{ store.take_retired( detail::container_kind::array ) };
        auto result_array{ result_handle.as_array() };
        detail::union_array_array_to_vector<layout_type>( lhs, rhs, result_array.values );
        result_array.sync_header();
        return result_handle;
    }
    if constexpr ( supports_bitset_container ) {
        // The union may still fall below the limit (shared values): demote, as the reference does.
        return demote_sparse_bitset(
            detail::bitset_handle_from_array_union<layout_type, CowPolicy>( lhs, rhs, store.take_retired( detail::container_kind::bitset ) ),
            store
        );
    } else {
        detail::union_array_array_to_vector<layout_type>( lhs, rhs, scratch );
        return make_fast_container_from_scratch_reusing( store, scratch );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::make_container_from_words_for_policy( detail::word_array<layout_type> const & words ) -> handle_type {
    auto container{ detail::make_container_from_words<layout_type, CowPolicy, RunSelectionPolicy>(
        words,
        array_to_bitset_threshold
    ) };
    if constexpr ( uses_default_container_set ) {
        return container;
    } else {
        return optimize_container_for_policy( std::move( container ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::combine_containers_for_policy( handle_type const & lhs, handle_type const & rhs, detail::set_operation const op ) -> handle_type {
    auto container{ detail::combine_containers<layout_type, CowPolicy, RunSelectionPolicy>(
        lhs,
        rhs,
        array_to_bitset_threshold,
        op
    ) };
    if constexpr ( uses_default_container_set ) {
        return container;
    } else {
        return optimize_container_for_policy( std::move( container ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::combine_bitset_bitset_for_policy( detail::bitset_cref<layout_type, CowPolicy> const lhs, detail::bitset_cref<layout_type, CowPolicy> const rhs, detail::set_operation const op, handle_type && reuse ) -> handle_type {
    auto container{ detail::combine_bitset_bitset<layout_type, CowPolicy>( lhs, rhs, op, std::move( reuse ) ) };
    if constexpr ( uses_default_container_set ) {
        return container;
    } else {
        return optimize_container_for_policy( std::move( container ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::intersect_run_bitset_for_policy( detail::run_cref<layout_type, CowPolicy> const runs, detail::bitset_cref<layout_type, CowPolicy> const bitset, handle_type && reuse ) -> handle_type {
    auto container{ detail::intersect_run_bitset<layout_type, CowPolicy>( runs, bitset, std::move( reuse ) ) };
    if constexpr ( uses_default_container_set ) {
        return container;
    } else {
        return optimize_container_for_policy( std::move( container ) );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::apply_closed_range_operation( key_type const begin_value, key_type const end_value, detail::range_operation const operation ) {
    settle_counts();
    if ( begin_value > end_value ) {
        return;
    }

    auto current{ begin_value };
    while ( true ) {
        auto const chunk{ layout_type::chunk_key( current ) };
        auto const low_begin{ layout_type::low_key( current ) };
        auto const chunk_end_value{ layout_type::compose( chunk, std::numeric_limits<low_type>::max() ) };
        auto const segment_end_value{ std::min( end_value, chunk_end_value ) };
        auto const low_end{ layout_type::low_key( segment_end_value ) };
        apply_chunk_range_operation( chunk, low_begin, low_end, operation );

        if ( segment_end_value == end_value || segment_end_value == std::numeric_limits<key_type>::max() ) {
            break;
        }
        current = static_cast<key_type>( segment_end_value + 1 );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::apply_chunk_range_operation( chunk_type const chunk, low_type const low_begin, low_type const low_end, detail::range_operation const operation ) {
    auto at{ chunks_.lower_bound( chunk ) };
    if ( at == chunks_.end() || chunks_.key( at ) != chunk ) {
        if ( operation == detail::range_operation::remove ) { return; }
        at = chunks_.insert( at, chunk, handle_type{} );
    }

    auto const previous_size{ detail::container_size( chunks_.slot( at ) ) };
    {
        auto & slot{ chunks_.slot( at ) };
        slot = detail::apply_range_to_container<layout_type, CowPolicy, RunSelectionPolicy>( std::move( slot ), low_begin, low_end, operation, array_to_bitset_threshold );
        if constexpr ( !uses_default_container_set ) {
            slot = optimize_container_for_policy( std::move( slot ) );
        }
    }
    auto const updated_size{ detail::container_size( chunks_.slot( at ) ) };

    if ( updated_size > previous_size ) {
        size_ += updated_size - previous_size;
    } else {
        size_ -= previous_size - updated_size;
    }
    if ( updated_size == 0 ) {
        chunks_.erase( at );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::add_many_unsorted( std::span<key_type const> const values ) noexcept( detail::allocation_failure_terminates ) {
    if ( try_add_many_single_chunk_dense( values ) ) {
        return;
    }
    detail::heap_vector<key_type> sorted_values;
    sorted_values.reserve( static_cast<std::uint32_t>( values.size() ) );
    for ( auto const value : values ) {
        sorted_values.push_back( value );
    }
    std::ranges::sort( sorted_values );
    if ( empty() ) {
        add_sorted_many( sorted_values );
    } else {
        add_many_sorted( sorted_values );
    }
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::remove_many_unsorted( std::span<key_type const> const values ) noexcept( detail::allocation_failure_terminates ) {
    detail::heap_vector<key_type> sorted_values;
    sorted_values.reserve( static_cast<std::uint32_t>( values.size() ) );
    for ( auto const value : values ) {
        sorted_values.push_back( value );
    }
    std::ranges::sort( sorted_values );
    remove_many_sorted( sorted_values );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::operator|=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) -> bitmap & {
    settle_counts();
    return union_merge<false>( other );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::operator&=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) -> bitmap & {
    settle_counts();
    return combine<detail::set_operation::bit_and, true>( other, nullptr );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::operator-=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) -> bitmap & {
    settle_counts();
    return combine<detail::set_operation::bit_andnot, true>( other, nullptr );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::and_cardinality( bitmap const & other ) const -> size_type {
    return detail::interop_engine<bitmap>::and_cardinality( *this, other );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE bool FRSR_ROARING_BITMAP::is_subset_of( bitmap const & other ) const {
    return detail::interop_engine<bitmap>::is_subset_of( *this, other );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE void FRSR_ROARING_BITMAP::interop_mutate( detail::interop_op const op, detail::operand_view const & operand ) noexcept( detail::allocation_failure_terminates ) {
    using engine = detail::interop_engine<bitmap>;
    detail::with_operand<bitmap>( operand, [ & ]( auto const & concrete ) {
        switch ( op ) {
            case detail::interop_op::and_with:    engine::and_with   ( *this, concrete ); return;
            case detail::interop_op::or_with:     engine::or_with    ( *this, concrete ); return;
            case detail::interop_op::andnot_with: engine::andnot_with( *this, concrete ); return;
            case detail::interop_op::xor_with:    engine::xor_with   ( *this, concrete ); return;
            default: std::unreachable();
        }
    } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE auto FRSR_ROARING_BITMAP::interop_count( detail::interop_op const op, detail::operand_view const & operand ) const -> size_type {
    using engine = detail::interop_engine<bitmap>;
    return detail::with_operand<bitmap>( operand, [ & ]( auto const & concrete ) -> size_type {
        return op == detail::interop_op::and_cardinality ? engine::and_cardinality( *this, concrete ) : engine::xor_cardinality( *this, concrete );
    } );
}

FRSR_ROARING_BITMAP_TEMPLATE
FRSR_ROARING_DEFINITION_INLINE bool FRSR_ROARING_BITMAP::interop_test( detail::interop_op const op, detail::operand_view const & operand ) const {
    using engine = detail::interop_engine<bitmap>;
    return detail::with_operand<bitmap>( operand, [ & ]( auto const & concrete ) -> bool {
        switch ( op ) {
            case detail::interop_op::intersects:     return engine::intersects    ( *this, concrete );
            case detail::interop_op::is_subset_of:   return engine::is_subset_of  ( *this, concrete );
            case detail::interop_op::is_superset_of: return engine::is_superset_of( *this, concrete );
            case detail::interop_op::equals:         return engine::equals        ( *this, concrete );
            default: std::unreachable();
        }
    } );
}

} // namespace frsr::roaring

#undef FRSR_ROARING_BITMAP_TEMPLATE
#undef FRSR_ROARING_DEFINITION_INLINE
#undef FRSR_ROARING_BITMAP
