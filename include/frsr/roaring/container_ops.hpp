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

} // namespace frsr::roaring::detail
