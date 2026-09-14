#pragma once

#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/array_ops.hpp>
#include <frsr/roaring/run_ops.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>


namespace frsr::roaring::detail {

// RunSelectionPolicy gates only the array/bitset-vs-run size-estimate decisions
// this function forwards into make_container_from_sorted_vector (the array∩array,
// array∩run, array-andnot-run, and array∩bitset/array-andnot-bitset arms below).
// The run∩run / run∪run / run-andnot-run / union(run,array) arms feed an
// already-run-shaped kernel result into make_container_from_runs, a materially
// different decision (keep-vs-downgrade an existing run result, not "should sorted
// values become a run") that is intentionally left unconditional: under
// run_selection_lazy those kernels are only reachable at all if an operand was
// already run-encoded (i.e. produced by an explicit optimize()/deserialization),
// so it is not an implicit-run-creation site in the sense this policy targets.
template <typename Layout, typename CowPolicy = cow_value_semantics, typename RunSelectionPolicy = run_selection_eager>
[[nodiscard]] inline container_variant<Layout, CowPolicy> combine_containers(
    container_variant<Layout, CowPolicy> const & lhs,
    container_variant<Layout, CowPolicy> const & rhs,
    std::size_t const bitset_threshold,
    set_operation const op
) {
    auto combine_run_bitset_words = []( run_cref<Layout, CowPolicy> const run, bitset_cref<Layout, CowPolicy> const bitset, set_operation const run_op ) {
        if ( run_op == set_operation::bit_or ) {
            auto words{ bitset.words.as_array() };
            for ( auto const & current : run.runs ) {
                apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::add );
            }
            return words;
        }

        // Hoist the and-vs-andnot choice out of the innermost per-word loop (run_op is
        // bit_and or bit_andnot here — bit_or already returned above), matching the
        // sibling combine_bitset_bitset kernel's "hoist the operation out of the loop"
        // principle instead of re-branching on every word of every run.
        word_array<Layout> words{};
        auto const combine_run{ [ & ]( ::frsr::roaring::run<typename Layout::low_type> const & current, auto const combine_word ) {
            auto const first_word{ static_cast<std::size_t>( current.begin ) >> 6U };
            auto const last_word{ static_cast<std::size_t>( current.end ) >> 6U };
            auto const first_bit{ static_cast<unsigned>( current.begin ) & 63U };
            auto const last_bit{ static_cast<unsigned>( current.end ) & 63U };
            for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                auto mask{ std::numeric_limits<std::uint64_t>::max() };
                if ( word_index == first_word ) {
                    mask &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                }
                if ( word_index == last_word && last_bit != 63U ) {
                    mask &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                }
                words[ word_index ] |= combine_word( bitset.words[ word_index ], mask );
            }
        } };
        if ( run_op == set_operation::bit_and ) {
            for ( auto const & current : run.runs ) {
                combine_run( current, []( std::uint64_t const bits, std::uint64_t const mask ) { return bits & mask; } );
            }
        } else {
            for ( auto const & current : run.runs ) {
                combine_run( current, []( std::uint64_t const bits, std::uint64_t const mask ) { return ( ~bits ) & mask; } );
            }
        }
        return words;
    };

    // array∩bitset / array\bitset: run the membership filter straight into the
    // result container's own array payload (no intermediate small_array_values
    // scratch, no scratch→payload copy). |result| ≤ |array operand| < the bitset
    // threshold on the common path, so make_container_from_filled_array keeps this
    // very handle — matching CRoaring's write-into-preallocated-result shape.
    auto filter_into_container{ [ bitset_threshold ](
        array_cref<Layout, CowPolicy> const array,
        bitset_cref<Layout, CowPolicy> const bitset,
        bool const keep_matches
    ) -> container_handle<Layout, CowPolicy> {
        container_handle<Layout, CowPolicy> result_handle;
        auto result_array{ result_handle.as_array() };
        filter_array_bitset_into<Layout>( array, bitset, keep_matches, result_array.values );
        if ( result_array.values.empty() ) {
            return container_handle<Layout, CowPolicy>{};
        }
        result_array.sync_header();
        return make_container_from_filled_array<Layout, CowPolicy, RunSelectionPolicy>( std::move( result_handle ), bitset_threshold );
    } };

    return visit_container_pair<Layout, CowPolicy>( [&]( auto const & left, auto const & right ) -> container_handle<Layout, CowPolicy> {
        using left_type = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;

        if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( combine_array_array<Layout, CowPolicy>( left, right, op ), bitset_threshold );
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_array<Layout, CowPolicy>( right, left ), bitset_threshold );
            }
            if ( op == set_operation::bit_and ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( intersect_array_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( difference_array_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_array<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_and ) {
                return make_container_from_sorted_vector<Layout, CowPolicy, RunSelectionPolicy>( intersect_array_run<Layout, CowPolicy>( right, left ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return make_container_from_runs<Layout, CowPolicy>( difference_run_array<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or ) {
                return make_bitset_container_from_words<Layout, CowPolicy>( combine_bitset_array_words<Layout, CowPolicy>( right, left, op ) );
            }
            if ( op == set_operation::bit_and ) {
                return filter_into_container( left, right, true );
            }
            if ( op == set_operation::bit_andnot ) {
                return filter_into_container( left, right, false );
            }
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_or || op == set_operation::bit_andnot ) {
                return make_bitset_container_from_words<Layout, CowPolicy>( combine_bitset_array_words<Layout, CowPolicy>( left, right, op ) );
            }
            return filter_into_container( right, left, true );
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            return make_bitset_container_from_words<Layout, CowPolicy>( combine_run_bitset_words( left, right, op ) );
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_andnot ) {
                auto words{ left.words.as_array() };
                for ( auto const & current : right.runs ) {
                    apply_range_to_words<Layout>( words, current.begin, current.end, range_operation::remove );
                }
                return make_bitset_container_from_words<Layout, CowPolicy>( words );
            }
            return make_bitset_container_from_words<Layout, CowPolicy>( combine_run_bitset_words( right, left, op ) );
        } else if constexpr ( std::same_as<left_type, bitset_cref<Layout, CowPolicy>> && std::same_as<right_type, bitset_cref<Layout, CowPolicy>> ) {
            return combine_bitset_bitset<Layout, CowPolicy>( left, right, op );
        } else if constexpr ( std::same_as<left_type, run_cref<Layout, CowPolicy>> && std::same_as<right_type, run_cref<Layout, CowPolicy>> ) {
            if ( op == set_operation::bit_and ) {
                return make_container_from_runs<Layout, CowPolicy>( intersect_run_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_or ) {
                return make_container_from_runs<Layout, CowPolicy>( union_run_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
            if ( op == set_operation::bit_andnot ) {
                return make_container_from_runs<Layout, CowPolicy>( difference_run_run<Layout, CowPolicy>( left, right ), bitset_threshold );
            }
        }

        auto words{ combine_words<Layout>( words_from_container<Layout>( lhs ), words_from_container<Layout>( rhs ), op ) };
        return make_container_from_words<Layout, CowPolicy>( words, bitset_threshold );
    }, lhs, rhs );
}

template <typename Layout, typename CowPolicy = cow_value_semantics>
[[nodiscard]] inline bool container_intersects(
    container_variant<Layout, CowPolicy> const & lhs,
    container_variant<Layout, CowPolicy> const & rhs
) noexcept {
    return visit_container_pair<Layout, CowPolicy>( []( auto const & left, auto const & right ) noexcept {
        using left_type = std::remove_cvref_t<decltype( left )>;
        using right_type = std::remove_cvref_t<decltype( right )>;

        if constexpr (
            std::same_as<left_type, array_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, array_cref<Layout, CowPolicy>>
        ) {
            auto li{ left.values.begin() };
            auto ri{ right.values.begin() };
            while ( li != left.values.end() && ri != right.values.end() ) {
                if ( *li < *ri ) {
                    ++li;
                } else if ( *ri < *li ) {
                    ++ri;
                } else {
                    return true;
                }
            }
            return false;
        } else if constexpr ( std::same_as<left_type, array_cref<Layout, CowPolicy>> ) {
            for ( auto const value : left.values ) {
                if ( right.contains( value ) ) {
                    return true;
                }
            }
            return false;
        } else if constexpr ( std::same_as<right_type, array_cref<Layout, CowPolicy>> ) {
            for ( auto const value : right.values ) {
                if ( left.contains( value ) ) {
                    return true;
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, run_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, run_cref<Layout, CowPolicy>>
        ) {
            auto li{ left.runs.begin() };
            auto ri{ right.runs.begin() };
            while ( li != left.runs.end() && ri != right.runs.end() ) {
                if ( li->end < ri->begin ) {
                    ++li;
                } else if ( ri->end < li->begin ) {
                    ++ri;
                } else {
                    return true;
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, run_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, bitset_cref<Layout, CowPolicy>>
        ) {
            for ( auto const & run : left.runs ) {
                auto const first_word{ static_cast<std::size_t>( run.begin ) >> 6U };
                auto const last_word{ static_cast<std::size_t>( run.end ) >> 6U };
                auto const first_bit{ static_cast<unsigned>( run.begin ) & 63U };
                auto const last_bit{ static_cast<unsigned>( run.end ) & 63U };
                for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                    auto word{ right.words[ word_index ] };
                    if ( word == 0 ) {
                        continue;
                    }
                    if ( word_index == first_word ) {
                        word &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                    }
                    if ( word_index == last_word && last_bit != 63U ) {
                        word &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                    }
                    if ( word != 0 ) {
                        return true;
                    }
                }
            }
            return false;
        } else if constexpr (
            std::same_as<left_type, bitset_cref<Layout, CowPolicy>> &&
            std::same_as<right_type, run_cref<Layout, CowPolicy>>
        ) {
            for ( auto const & run : right.runs ) {
                auto const first_word{ static_cast<std::size_t>( run.begin ) >> 6U };
                auto const last_word{ static_cast<std::size_t>( run.end ) >> 6U };
                auto const first_bit{ static_cast<unsigned>( run.begin ) & 63U };
                auto const last_bit{ static_cast<unsigned>( run.end ) & 63U };
                for ( auto word_index{ first_word }; word_index <= last_word; ++word_index ) {
                    auto word{ left.words[ word_index ] };
                    if ( word == 0 ) {
                        continue;
                    }
                    if ( word_index == first_word ) {
                        word &= std::numeric_limits<std::uint64_t>::max() << first_bit;
                    }
                    if ( word_index == last_word && last_bit != 63U ) {
                        word &= ( std::uint64_t{ 1 } << ( last_bit + 1U ) ) - 1U;
                    }
                    if ( word != 0 ) {
                        return true;
                    }
                }
            }
            return false;
        } else {
            for ( std::size_t index{ 0 }; index < left.words.size(); ++index ) {
                if ( ( left.words[ index ] & right.words[ index ] ) != 0 ) {
                    return true;
                }
            }
            return false;
        }
    }, lhs, rhs );
}

// Writes every value of `container`, composed with `chunk_key` into the full key
// space, straight into `out` (which MUST hold container_size() entries — the
// contract is the reference's, and `out_end` is a slack guard for the vector
// stores, not a bounds check) and returns the end pointer. The equivalent of a for_each with a push_back per value
// minus the per-value capacity check and size increment: each kind writes through a
// plain pointer, which is what lets the array kind's widening loop vectorise.
// [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_to_uint32_array
// Inlined into its per-chunk callers (to_array_into, to_vector): outlined, the
// decode loops lose the callers' knowledge of the output span and the run and
// array kinds decode at half speed.
template <typename Layout, typename CowPolicy>
[[ using gnu: hot, always_inline ]] inline typename Layout::key_type * container_decode_into(
    container_handle<Layout, CowPolicy> const & container,
    typename Layout::chunk_type const chunk_key,
    typename Layout::key_type * out,
    typename Layout::key_type * const out_end
) noexcept {
    using key_type = typename Layout::key_type;
    auto const base{ Layout::compose( chunk_key, 0 ) };
    switch ( container.kind() ) {
        case container_kind::array: {
            auto const & values{ container.as_array().values };
            auto const   count { values.size() };
            auto const * const source{ values.data() };
#if FRSR_ROARING_X86_V4
            if constexpr ( std::is_same_v<key_type, std::uint32_t> && std::is_same_v<typename Layout::low_type, std::uint16_t> ) {
                if ( count >= x86_v4::decode_array_min_count && have_x86_v4() ) {
                    return x86_v4::decode_array_uint32( source, count, out, base );
                }
            }
#endif
            // Widening add over a contiguous run: clang lowers this to unpack +
            // add + store (vpmovzxwd/vpaddd on x86, ushll/add on AArch64).
            for ( std::uint32_t i{ 0 }; i < count; ++i ) {
                out[ i ] = base + static_cast<key_type>( source[ i ] );
            }
            return out + count;
        }
        case container_kind::run: {
            for ( auto const & run : container.as_run().runs ) {
                auto const first{ base + static_cast<key_type>( run.begin ) };
                auto const count{ static_cast<std::size_t>( run.end - run.begin ) + 1U };  // end is inclusive
                // Counted, index-addressed loop (not a pointer-incrementing one):
                // this is the shape clang vectorises into a broadcast + index-vector
                // add + store, which a run of thousands of values needs.
                // [croaring-ref] deps/croaring/src/containers/run.c:run_container_to_uint32_array
                for ( std::size_t i{ 0 }; i < count; ++i ) {
                    out[ i ] = first + static_cast<key_type>( i );
                }
                out += count;
            }
            return out;
        }
        case container_kind::bitset: {
            auto const & words{ container.as_bitset().words.as_array() };
            if constexpr ( std::is_same_v<key_type, std::uint32_t> ) {
                return extract_setbits_uint32( words.data(), words.size(), out, out_end, base );
            } else {
                for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
                    auto word{ words[ index ] };
                    auto const word_base{ base + static_cast<key_type>( index << 6U ) };
                    while ( word != 0 ) {
                        *out++ = word_base + static_cast<key_type>( std::countr_zero( word ) );
                        word &= word - 1;
                    }
                }
                return out;
            }
        }
    }
    std::unreachable();
}

} // namespace frsr::roaring::detail
