#pragma once

#include <frsr/roaring/container_layout.hpp>
#include <frsr/roaring/run_container.hpp>
#include <frsr/roaring/run_selection_policy.hpp>
#include <frsr/roaring/tuning.hpp>
#include <frsr/roaring/operations.hpp>
#include <frsr/roaring/hw_info.hpp>
#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/chunk_store.hpp>
#include <frsr/roaring/bitset_ops.hpp>
#include <frsr/roaring/array_ops.hpp>
#include <frsr/roaring/run_ops.hpp>
#include <frsr/roaring/container_ops.hpp>

#include <algorithm>
#include <array>
#ifndef FRSR_ROARING_COMBINE_STATS
#   define FRSR_ROARING_COMBINE_STATS 0
#endif
#if FRSR_ROARING_COMBINE_STATS
#   include <atomic>
#   include <cstdio>
#endif
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>   // the serialized_byte_vector fallback uses std::vector even with psi::vm present
#ifdef FRSR_ROARING_HAS_PSI_VM
#   include <psi/vm/containers/small_vector.hpp>
#   ifdef FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION
#       include <psi/vm/containers/vm_vector.hpp>
#   endif
    // heap_vector already pulled in via run_container.hpp → psi/vm/containers/heap_vector.hpp
#endif

// Prefer Boost's open-addressing flat_hash_map for the chunk-index cache:
// it eliminates pointer chasing in std::unordered_map and greatly reduces
// cache misses on large sparse bitmaps (1M+ distinct chunk keys).
#if __has_include(<boost/unordered/unordered_flat_map.hpp>)
#   include <boost/unordered/unordered_flat_map.hpp>
#   define FRSR_USE_BOOST_FLAT_MAP 1
#endif

namespace frsr::roaring {

// Container-set policy tags. Concrete storage is the type-erased
// detail::container_handle (one 32-byte representation for every kind); these
// types only name the alternatives a bitmap policy admits — the ContainerSet
// typelist that gates run/bitset conversions. run_container doubles as the
// standalone result-builder type the run kernels use.
template <std::unsigned_integral Key>
struct array_container { using layout_type = container_layout<Key>; };

template <std::unsigned_integral Key>
struct bitset_container { using layout_type = container_layout<Key>; };

template <std::unsigned_integral Key>
using default_container_set = std::variant<array_container<Key>, run_container<Key>, bitset_container<Key>>;

namespace detail {

// A bitmap bookkeeping member that forwards to a real T while its feature flag
// is Active, but is an empty (combined with [[no_unique_address]]) stand-in when
// the flag is compiled off: reads yield OffValue, writes/increments are no-ops.
// This lets every novel-feature counter (lazy-tombstone / lazy-sort / hot-chunk /
// adaptive chunk-index bookkeeping) cost zero footprint when its flag is off
// WITHOUT guarding each of its dozens of use sites — the flags-off bitmap shrinks
// by construction and the existing read / write / ++ / compare sites keep working
// through these operators. Feature-off semantics are correct by construction:
// e.g. tombstone_count reads 0 so every `tombstone_count_ != 0` guard is false,
// exactly matching "no lazy tombstones exist when the feature is off".
template <typename T, bool Active, T OffValue = T{}>
struct feature_field {
    T value_{ OffValue };
    constexpr feature_field(                    ) noexcept = default;
    constexpr feature_field( feature_field const & ) noexcept = default;
    constexpr feature_field & operator=( feature_field const & ) noexcept = default;
    constexpr operator T() const noexcept { return value_; }
    constexpr feature_field & operator= ( T const v ) noexcept { value_ = v; return *this; }
    constexpr feature_field & operator++(          ) noexcept { ++value_; return *this; }
    constexpr T               operator++( int      ) noexcept { return value_++; }
    constexpr feature_field & operator--(          ) noexcept { --value_; return *this; }
    constexpr T               operator--( int      ) noexcept { return value_--; }
};
template <typename T, T OffValue>
struct feature_field<T, false, OffValue> {
    constexpr feature_field(                    ) noexcept = default;
    constexpr feature_field( feature_field const & ) noexcept = default;
    constexpr feature_field & operator=( feature_field const & ) noexcept = default;
    constexpr operator T() const noexcept { return OffValue; }
    constexpr feature_field & operator= ( T          ) noexcept { return *this; }
    constexpr feature_field & operator++(            ) noexcept { return *this; }
    constexpr T               operator++( int        ) noexcept { return OffValue; }
    constexpr feature_field & operator--(            ) noexcept { return *this; }
    constexpr T               operator--( int        ) noexcept { return OffValue; }
};

template <typename... Types>
struct unique_types : std::true_type {};

template <typename Type, typename... Rest>
struct unique_types<Type, Rest...>
    : std::bool_constant<(!(std::same_as<Type, Rest>) && ...) && unique_types<Rest...>::value> {};

template <typename Variant, typename Type>
struct variant_contains : std::false_type {};

template <typename... Types, typename Type>
struct variant_contains<std::variant<Types...>, Type> : std::bool_constant<(std::same_as<Types, Type> || ...)> {};

template <typename Key, typename Variant>
struct container_set_validator : std::false_type {};

template <std::unsigned_integral Key, typename... Types>
struct container_set_validator<Key, std::variant<Types...>>
    : std::bool_constant<
          unique_types<Types...>::value &&
          ( ( std::same_as<Types, ::frsr::roaring::array_container<Key>> ||
              std::same_as<Types, ::frsr::roaring::run_container<Key>> ||
              std::same_as<Types, ::frsr::roaring::bitset_container<Key>> ) && ... ) &&
          variant_contains<std::variant<Types...>, ::frsr::roaring::array_container<Key>>::value> {};

template <typename Variant, typename Type>
inline constexpr bool variant_contains_v{ variant_contains<Variant, Type>::value };

template <typename Key, typename Variant>
concept valid_container_set = std::unsigned_integral<Key> && container_set_validator<Key, Variant>::value;

} // namespace detail

// Diagnostic-only combine() telemetry (compile-time gated, default off): a
// per-container-type-pair op histogram + result-form (run-mint) counters +
// per-call operand chunk-count sums, dumped to stderr at process exit. The
// "pathological usage" detector distilled from a downstream incident where an
// accumulate-in-arbitrary-order caller silently degenerated into a run×run
// AND cascade — one instrumented run of this hook answers what previously
// took several bespoke probe cycles (pair mix, operand widths, run minting).
//
// Extended for a coverage census (operation × form-pair × cardinality-band ×
// saturation): bucketed operand-cardinality histograms, an approximate
// operand-payload-bytes tally (array: 2 bytes/element; run: conservatively
// treated like array, i.e. an upper bound, since exact run-count is not
// cheaply available here; bitset: the fixed per-chunk word block), and a
// full/near-full bitset counter for the saturated-bitset→run question. This
// stays a pure aggregate counter set — no per-call logging — so it is safe to
// leave compiled into an instrumented diagnostic build of a large workload.
#ifndef FRSR_ROARING_COMBINE_STATS
#   define FRSR_ROARING_COMBINE_STATS 0
#endif
#if FRSR_ROARING_COMBINE_STATS
namespace detail {
struct combine_stats_t {
    static constexpr char const * kind_names[ 3 ]{ "array", "run", "bitset" };
    static constexpr char const * op_names[ 3 ]{ "and", "andnot", "or" };
    // Power-of-two-ish cardinality bands; the top band is capped at the
    // per-chunk domain size (65536), which is also the "full bitset" mark.
    static constexpr std::uint32_t kBucketHi[ 7 ]{ 16, 64, 256, 1024, 4096, 16384, 65536 };
    static constexpr int kBucketCount{ 7 };
    static int bucket_of( std::uint64_t const card ) noexcept {
        for ( auto i{ 0 }; i < kBucketCount; ++i ) { if ( card <= kBucketHi[ i ] ) return i; }
        return kBucketCount - 1;
    }
    static constexpr std::uint32_t kBitsetFull{ 65536 };
    static constexpr std::uint32_t kBitsetNearFull{ 62259 }; // ~95% of 65536

    std::atomic<std::uint64_t> pairs        [ 3 ][ 3 ][ 3 ]{};   // [left kind][right kind][op: and/andnot/or]
    std::atomic<std::uint64_t> pair_min_card[ 3 ][ 3 ][ 3 ]{};   // Σ min(cardinality): element-volume bound per combo
    std::atomic<std::uint64_t> pair_bytes_l [ 3 ][ 3 ][ 3 ]{};   // Σ approx payload bytes, left operand
    std::atomic<std::uint64_t> pair_bytes_r [ 3 ][ 3 ][ 3 ]{};   // Σ approx payload bytes, right operand
    std::atomic<std::uint64_t> card_hist    [ 3 ][ 3 ][ kBucketCount ]{}; // [op][kind][bucket], both operand sides folded in
    std::atomic<std::uint64_t> results[ 3 ]{};           // result container form at emit
    std::atomic<std::uint64_t> calls{}, left_chunks{}, right_chunks{};
    std::atomic<std::uint64_t> full_bitset_operands{}, near_full_bitset_operands{};

    ~combine_stats_t() {
        std::fprintf( stderr, "[combine-stats] calls=%llu left_chunks=%llu right_chunks=%llu\n",
            (unsigned long long)calls.load(), (unsigned long long)left_chunks.load(), (unsigned long long)right_chunks.load() );
        std::fprintf( stderr, "[combine-stats] full_bitset_operands=%llu near_full_bitset_operands=%llu (>=%u/%u)\n",
            (unsigned long long)full_bitset_operands.load(), (unsigned long long)near_full_bitset_operands.load(),
            kBitsetNearFull, kBitsetFull );
        for ( auto l{ 0 }; l < 3; ++l ) for ( auto r{ 0 }; r < 3; ++r ) for ( auto o{ 0 }; o < 3; ++o ) {
            if ( auto const n{ pairs[ l ][ r ][ o ].load() } ) {
                std::fprintf( stderr, "[combine-stats]   %s x %s %s: %llu min_card_sum=%llu bytes_l=%llu bytes_r=%llu\n",
                    kind_names[ l ], kind_names[ r ], op_names[ o ], (unsigned long long)n,
                    (unsigned long long)pair_min_card[ l ][ r ][ o ].load(),
                    (unsigned long long)pair_bytes_l[ l ][ r ][ o ].load(),
                    (unsigned long long)pair_bytes_r[ l ][ r ][ o ].load() );
            }
        }
        for ( auto o{ 0 }; o < 3; ++o ) for ( auto k{ 0 }; k < 3; ++k ) {
            std::fprintf( stderr, "[combine-stats]   card_hist %s %s:", op_names[ o ], kind_names[ k ] );
            for ( auto b{ 0 }; b < kBucketCount; ++b ) {
                std::fprintf( stderr, " <=%u:%llu", kBucketHi[ b ], (unsigned long long)card_hist[ o ][ k ][ b ].load() );
            }
            std::fprintf( stderr, "\n" );
        }
        for ( auto k{ 0 }; k < 3; ++k ) {
            if ( auto const n{ results[ k ].load() } ) {
                std::fprintf( stderr, "[combine-stats]   result %s: %llu\n", kind_names[ k ], (unsigned long long)n );
            }
        }
    }
    template <typename Handle>
    static int kind_of( Handle const & h ) noexcept { return h.holds_array() ? 0 : h.holds_run() ? 1 : 2; }
    static std::uint64_t approx_bytes( int const kind, std::uint64_t const card ) noexcept {
        // array: 2-byte in-chunk offsets. run: no cheap run-count access here,
        // so treated as an upper bound identical to array (real run containers
        // are never bigger than this). bitset: fixed 65536-bit block.
        return kind == 2 ? std::uint64_t{ 65536 / 8 } : card * 2;
    }
    template <typename Handle>
    void record_pair( Handle const & l, Handle const & r, set_operation const op ) noexcept {
        auto const o{ op == set_operation::bit_and ? 0 : op == set_operation::bit_andnot ? 1 : 2 };
        auto const lk{ kind_of( l ) }, rk{ kind_of( r ) };
        auto const lc{ l.cardinality() }, rc{ r.cardinality() };
        pairs        [ lk ][ rk ][ o ].fetch_add( 1, std::memory_order_relaxed );
        pair_min_card[ lk ][ rk ][ o ].fetch_add( std::min<std::uint64_t>( lc, rc ), std::memory_order_relaxed );
        pair_bytes_l [ lk ][ rk ][ o ].fetch_add( approx_bytes( lk, lc ), std::memory_order_relaxed );
        pair_bytes_r [ lk ][ rk ][ o ].fetch_add( approx_bytes( rk, rc ), std::memory_order_relaxed );
        card_hist[ o ][ lk ][ bucket_of( lc ) ].fetch_add( 1, std::memory_order_relaxed );
        card_hist[ o ][ rk ][ bucket_of( rc ) ].fetch_add( 1, std::memory_order_relaxed );
        if ( lk == 2 ) {
            if      ( lc >= kBitsetFull     ) full_bitset_operands     .fetch_add( 1, std::memory_order_relaxed );
            else if ( lc >= kBitsetNearFull ) near_full_bitset_operands.fetch_add( 1, std::memory_order_relaxed );
        }
        if ( rk == 2 ) {
            if      ( rc >= kBitsetFull     ) full_bitset_operands     .fetch_add( 1, std::memory_order_relaxed );
            else if ( rc >= kBitsetNearFull ) near_full_bitset_operands.fetch_add( 1, std::memory_order_relaxed );
        }
    }
    template <typename Handle>
    void record_result( Handle const & h ) noexcept {
        results[ kind_of( h ) ].fetch_add( 1, std::memory_order_relaxed );
    }
    void record_call( std::uint64_t const lchunks, std::uint64_t const rchunks ) noexcept {
        calls       .fetch_add( 1     , std::memory_order_relaxed );
        left_chunks .fetch_add( lchunks, std::memory_order_relaxed );
        right_chunks.fetch_add( rchunks, std::memory_order_relaxed );
    }
};
inline combine_stats_t & combine_stats() noexcept { static combine_stats_t s; return s; }
} // namespace detail
#endif // FRSR_ROARING_COMBINE_STATS

// RunSelectionPolicy defaults to run_selection_eager (the library's historical,
// unconditional smallest-serialized-size heuristic) so every EXISTING
// instantiation — bitmap<u16>, bitmap<u32>, bitmap<u64>, and the CoW-refcounted
// bitmap<u32, default_container_set<u32>, cow_atomic_refcount> — keeps today's
// behavior byte-for-byte. Opt into run_selection_lazy (CRoaring parity: never
// auto-picks run encoding outside optimize()/optimize_for_storage()) explicitly,
// as a downstream consumer's bitmap wrapper does.
template <
    std::unsigned_integral Key,
    typename ContainerSet = default_container_set<Key>,
    typename CowPolicy = detail::cow_value_semantics,
    typename RunSelectionPolicy = detail::run_selection_eager
>
    requires detail::valid_container_set<Key, ContainerSet>
class bitmap {
public:
    using key_type = Key;
    using container_set_type = ContainerSet;
    using cow_policy = CowPolicy;
    using run_selection_policy = RunSelectionPolicy;
    using layout_type = detail::default_layout<key_type>;
    using chunk_type = typename layout_type::chunk_type;
    using low_type = typename layout_type::low_type;
    using size_type = std::size_t;
    using array_container_type = array_container<key_type>;
    using run_container_type = run_container<key_type>;
    using bitset_container_type = bitset_container<key_type>;
    using default_container_set_type = default_container_set<key_type>;

private:
    using handle_type = detail::container_handle<layout_type, CowPolicy>;

public:

    static constexpr size_type array_to_bitset_threshold{
        layout_type::low_domain_size >= 1024 ? layout_type::low_domain_size / 16 : 64
    };
    static constexpr size_type invalid_index{ std::numeric_limits<size_type>::max() };

    struct container_statistics {
        size_type array_containers{};
        size_type run_containers{};
        size_type bitset_containers{};
        size_type staged_singleton_chunks{};

        [[nodiscard]] constexpr size_type container_count() const noexcept {
            return array_containers + run_containers + bitset_containers + staged_singleton_chunks;
        }
    };

    struct inverse_proxy {
        bitmap const & bm;
    };

    struct bulk_context {
        chunk_type chunk{};
        size_type index{ invalid_index };

        void reset() noexcept { index = invalid_index; }
    };

    class const_iterator {
    public:
        using value_type = key_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using iterator_concept = std::forward_iterator_tag;

        const_iterator() = default;

        [[nodiscard]] value_type operator*() const noexcept { return current_value_; }

        const_iterator & operator++() {
            advance();
            return *this;
        }

        const_iterator operator++( int ) {
            auto copy{ *this };
            ++( *this );
            return copy;
        }

        [[nodiscard]] friend bool operator==( const_iterator const & lhs, const_iterator const & rhs ) noexcept {
            return lhs.owner_ == rhs.owner_ && lhs.chunk_index_ == rhs.chunk_index_ && lhs.current_low_ == rhs.current_low_;
        }

    private:
        friend class bitmap;

        explicit const_iterator( bitmap const * owner, bool at_begin ) : owner_{ owner } {
            if ( owner_ == nullptr ) {
                return;
            }
            if ( at_begin ) {
                seek_first();
            } else {
                finish();
            }
        }

        void finish() noexcept {
            owner_ = nullptr;
            chunk_index_ = 0;
            current_low_.reset();
            current_value_ = {};
        }

        void seek_first() noexcept {
            for ( chunk_index_ = 0; chunk_index_ < owner_->chunks_.size(); ++chunk_index_ ) {
                if ( auto const low{ detail::container_first( owner_->chunks_.slot( chunk_index_ ) ) } ) {
                    current_low_ = low;
                    current_value_ = layout_type::compose( owner_->chunks_.key( chunk_index_ ), *low );
                    return;
                }
            }
            finish();
        }

        void advance() noexcept {
            if ( owner_ == nullptr ) {
                return;
            }

            if ( auto const next_low{ detail::container_next_after( owner_->chunks_.slot( chunk_index_ ), *current_low_ ) } ) {
                current_low_ = next_low;
                current_value_ = layout_type::compose( owner_->chunks_.key( chunk_index_ ), *next_low );
                return;
            }

            for ( ++chunk_index_; chunk_index_ < owner_->chunks_.size(); ++chunk_index_ ) {
                if ( auto const low{ detail::container_first( owner_->chunks_.slot( chunk_index_ ) ) } ) {
                    current_low_ = low;
                    current_value_ = layout_type::compose( owner_->chunks_.key( chunk_index_ ), *low );
                    return;
                }
            }

            finish();
        }

        bitmap const * owner_{};
        size_type chunk_index_{};
        std::optional<low_type> current_low_{};
        value_type current_value_{};
    };

    bitmap() = default;

    // User-defined copy (the boxed chunk-index cache makes the implicit one
    // deleted): copies the real state — chunks, singleton staging, tombstone /
    // sort bookkeeping, cardinality — and deliberately NOT the derived caches
    // (chunk index map, read index, hot indices): a copy starts with a cold
    // cache instead of paying a map clone for state it may never look up.
    bitmap( bitmap const & other )
        : chunks_{ other.chunks_ }
        , front_tombstones_{ other.front_tombstones_ }
        , singleton_chunks_{ other.singleton_chunks_ }
        , removes_since_structural_add_{ other.removes_since_structural_add_ }
        , chunks_sorted_{ other.chunks_sorted_ }
        , tombstone_count_{ other.tombstone_count_ }
        , size_{ other.size_ }
    {}

    bitmap( bitmap && ) noexcept = default;

    bitmap & operator=( bitmap const & other ) {
        if ( this != &other ) [[likely]] {
            bitmap copy{ other };
            swap( copy );
        }
        return *this;
    }

    bitmap & operator=( bitmap && ) noexcept = default;

    ~bitmap() = default;

    // The !same_as<bitmap> constraint keeps this greedy template from hijacking
    // copies from NON-CONST lvalues (bitmap models input_range of key_type, so
    // without it `bitmap c{ mutable_src }` would prefer Range&& over
    // bitmap const & and rebuild every container element-by-element —
    // O(cardinality) decode + fresh payload allocations instead of a zero-alloc
    // CoW copy).
    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
              && ( !std::same_as<std::remove_cvref_t<Range>, bitmap> )
    explicit bitmap( Range && values ) {
        add_many( std::forward<Range>( values ) );
    }

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool test( key_type const value ) const noexcept { return contains( value ); }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type byte_size() const noexcept {
        auto total{ chunks_.entry_bytes() };
        for ( auto const & slot : chunks_.slots() ) {
            total += detail::container_byte_size( slot );
        }
        if constexpr ( kUseSingletonChunkMap ) {
            total += singleton_chunks_.size() * sizeof( singleton_chunk_entry );
        }
        return total;
    }
    [[nodiscard]] std::uint16_t container_count() const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            return static_cast<std::uint16_t>( chunks_.size() + singleton_chunks_.size() );
        } else {
            return static_cast<std::uint16_t>( chunks_.size() );
        }
    }
    [[nodiscard]] container_statistics statistics() const noexcept {
        container_statistics result;
        for ( auto const & slot : chunks_.slots() ) {
            if ( detail::container_size( slot ) == 0 ) {
                continue;  // lazy-removal tombstone, not a logical container
            }
            if ( slot.holds_array() ) {
                ++result.array_containers;
            } else if ( slot.holds_run() ) {
                ++result.run_containers;
            } else {
                ++result.bitset_containers;
            }
        }
        if constexpr ( kUseSingletonChunkMap ) {
            result.staged_singleton_chunks = singleton_chunks_.size();
        }
        return result;
    }
    [[nodiscard]] key_type front() const noexcept { return *begin(); }
    [[nodiscard]] key_type back() const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        auto chunk_index{ chunks_.size() };
        while ( chunk_index > 0 ) {
            --chunk_index;
            auto const & container{ chunks_.slot( chunk_index ) };
            auto next{ detail::container_first( container ) };
            if ( !next ) {
                continue;
            }
            while ( auto const after{ detail::container_next_after( container, *next ) } ) {
                next = after;
            }
            return layout_type::compose( chunks_.key( chunk_index ), *next );
        }
        return {};
    }

    void clear() noexcept {
        chunks_.clear();
        clear_bookkeeping();
    }
    // Scratch reuse (CRoaring persistent-dst analog): logically empty, but the
    // container payloads are retired for chunk_store::take_retired() reuse by
    // the next materializing run instead of being freed (and immediately
    // reallocated) on every *_into combine.
    void clear_keep_capacity() noexcept {
        chunks_.clear_retiring_slots();
        clear_bookkeeping();
    }

private:
    void clear_bookkeeping() noexcept {
        front_tombstones_ = 0;
        hot_chunk_index_ = invalid_index;
        if constexpr ( kUseSingletonChunkMap ) {
            singleton_chunks_.clear();
            reset_singleton_read_index_empty();
        }
        if ( chunk_index_map_box_ ) {
            chunk_index_map_box_->clear();
        }
        chunk_index_map_valid_ = false;
        chunk_index_lookup_probes_ = 0;
        if constexpr ( kUseLazySort ) { chunks_sorted_ = true; }
        if constexpr ( kUseLazyTombstoning ) { tombstone_count_ = 0; }
        size_ = 0;
    }

public:

    [[nodiscard]] bool contains( key_type const value ) const noexcept {
        auto const chunk{ layout_type::chunk_key( value ) };
        if constexpr ( kUseSingletonChunkMap ) {
            singleton_chunk_entry const * indexed_entry{};
            switch ( try_singleton_read_index_lookup( chunk, indexed_entry, false ) ) {
                case singleton_lookup_result::found:
                    return singleton_contains( *indexed_entry, layout_type::low_key( value ) );
                case singleton_lookup_result::missing:
                    return false;
                case singleton_lookup_result::not_used:
                    break;
            }
            auto const it{ singleton_chunks_.find( chunk ) };
            if ( it != singleton_chunks_.end() ) {
                return singleton_contains( it->second, layout_type::low_key( value ) );
            }
        }
        // Parity shape (all novel flags off): a pure binary search, nothing
        // written. [croaring-ref] roaring_bitmap_contains -> containerptr_roaring
        // is likewise a read-only container lookup; amortization is opt-in
        // through roaring_bitmap_contains_bulk, mirrored here by contains_bulk.
        if constexpr ( kUseHotChunkIndex ) {
            size_type hot_index{};
            if ( try_hot_chunk_index( chunk, hot_index ) ) {
                return detail::container_contains( chunks_.slot( hot_index ), layout_type::low_key( value ) );
            }
            if ( try_adjacent_chunk_index( chunk, hot_chunk_index_, hot_index ) ) {
                hot_chunk_index_ = hot_index;
                return detail::container_contains( chunks_.slot( hot_index ), layout_type::low_key( value ) );
            }
        }
        if constexpr ( kUseChunkHashMap ) {
            size_type indexed_pos{};
            if ( try_chunk_index_lookup( chunk, indexed_pos, true ) ) {
                note_hot_chunk( indexed_pos );
                return detail::container_contains( chunks_.slot( indexed_pos ), layout_type::low_key( value ) );
            }
        }
        ensure_sorted();
        auto const pos{ lower_bound( chunk ) };
        if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) { return false; }
        note_hot_chunk( pos );
        return detail::container_contains( chunks_.slot( pos ), layout_type::low_key( value ) );
    }

    [[nodiscard]] bool contains_bulk( bulk_context & ctx, key_type const value ) const noexcept {
        auto const chunk{ layout_type::chunk_key( value ) };
        if constexpr ( kUseSingletonChunkMap ) {
            auto const it{ singleton_chunks_.find( chunk ) };
            if ( it != singleton_chunks_.end() ) {
                ctx.reset();
                return singleton_contains( it->second, layout_type::low_key( value ) );
            }
        }
        if (
            ctx.index != invalid_index &&
            ctx.index < chunks_.size() &&
            chunks_.key( ctx.index ) == chunk
        ) {
            return detail::container_contains( chunks_.slot( ctx.index ), layout_type::low_key( value ) );
        }
        if (
            ctx.index != invalid_index &&
            ctx.index < chunks_.size() &&
            try_adjacent_chunk_index( chunk, ctx.index, ctx.index )
        ) {
            ctx.chunk = chunk;
            return detail::container_contains( chunks_.slot( ctx.index ), layout_type::low_key( value ) );
        }

        if constexpr ( kUseChunkHashMap ) {
            ensure_chunk_index_map();
            auto const it{ chunk_index_map_().find( chunk ) };
            if ( it == chunk_index_map_().end() ) {
                ctx.reset();
                return false;
            }
            ctx.chunk = chunk;
            ctx.index = it->second;
        } else {
            ensure_sorted();
            auto const pos{ lower_bound( chunk ) };
            if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) {
                ctx.reset();
                return false;
            }
            ctx.chunk = chunk;
            ctx.index = pos;
        }
        return detail::container_contains( chunks_.slot( ctx.index ), layout_type::low_key( value ) );
    }

    [[nodiscard]] bool add( key_type const value ) {
        return add_impl( value, nullptr );
    }

    [[nodiscard]] bool add_bulk( bulk_context & ctx, key_type const value ) {
        return add_impl( value, &ctx );
    }

    [[nodiscard]] bool remove( key_type const value ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        if constexpr ( kUseSingletonChunkMap ) {
            auto singleton_it{ singleton_chunks_.find( chunk ) };
            if ( singleton_it != singleton_chunks_.end() ) {
                if ( !singleton_remove( singleton_it->second, low ) ) {
                    return false;
                }
                invalidate_singleton_read_index();
                --size_;
                if ( singleton_it->second.count == 0 ) {
                    singleton_chunks_.erase( singleton_it );
                }
                return true;
            }
        }

        size_type pos;
        if constexpr ( kUseChunkHashMap ) {
            // O(1) hash-map lookup path (handles both sorted and unsorted chunks).
            if ( chunk_index_map_valid_ ) {
                auto const it{ chunk_index_map_().find( chunk ) };
                if ( it == chunk_index_map_().end() ) { return false; }
                pos = it->second;
            } else if constexpr ( !kUseLazySort ) {
                // Always sorted when lazy-sort is off; use lower_bound with lazy rebuild.
                static constexpr std::uint32_t kLazyMapBuildThreshold{ 2 };
                if ( ++removes_since_structural_add_ >= kLazyMapBuildThreshold ) {
                    ensure_chunk_index_map();
                    auto const it{ chunk_index_map_().find( chunk ) };
                    if ( it == chunk_index_map_().end() ) { return false; }
                    pos = it->second;
                } else {
                    pos = lower_bound( chunk );
                    if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) { return false; }
                }
            } else if ( chunks_sorted_ ) {
                // Map invalid but chunks sorted; lazy rebuild threshold logic.
                static constexpr std::uint32_t kLazyMapBuildThreshold{ 2 };
                if ( ++removes_since_structural_add_ >= kLazyMapBuildThreshold ) {
                    ensure_chunk_index_map();
                    auto const it{ chunk_index_map_().find( chunk ) };
                    if ( it == chunk_index_map_().end() ) { return false; }
                    pos = it->second;
                } else {
                    pos = lower_bound( chunk );
                    if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) { return false; }
                }
            } else {
                // Unsorted chunks + invalid map: must rebuild map.
                ensure_chunk_index_map();
                auto const it{ chunk_index_map_().find( chunk ) };
                if ( it == chunk_index_map_().end() ) { return false; }
                pos = it->second;
            }
        } else {
            // CRoaring-style: always-sorted chunks, O(log n) binary search.
            size_type hot_index{};
            if ( try_hot_chunk_index( chunk, hot_index ) ) {
                pos = hot_index;
            } else if ( try_chunk_index_lookup( chunk, hot_index, true ) ) {
                pos = hot_index;
            } else {
                pos = lower_bound( chunk );
                if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) { return false; }
            }
        }
        auto const removed{ detail::container_remove( chunks_.slot( pos ), low ) };
        if ( !removed ) { return false; }

        --size_;
        note_hot_chunk( pos );
        if constexpr ( kUseSingletonChunkMap ) {
            if (
                detail::container_size( chunks_.slot( pos ) ) <= singleton_chunk_demote_threshold
            ) {
                auto entry{ singleton_from_container( chunks_.slot( pos ) ) };
                if ( entry.count != 0 ) {
                    singleton_chunks_.emplace( chunks_.key( pos ), std::move( entry ) );
                    invalidate_singleton_read_index();
                    chunks_.erase( pos );
                    invalidate_chunk_index_map();
                    return true;
                }
            }
        }
        if ( detail::container_size( chunks_.slot( pos ) ) == 0 ) {
            if constexpr ( kUseLazyTombstoning ) {
                ++tombstone_count_;
                if ( tombstone_count_ * 2 > static_cast<std::uint32_t>( chunks_.size() ) ) {
                    compact_tombstones();
                }
                // Don't invalidate hash map: key→index remains valid for tombstone entries.
            } else {
                // Parity shape: erase immediately.
                // [croaring-ref] deps/croaring/src/roaring_array.c:ra_remove_at_index
                if constexpr ( kUseFrontTombstones ) {
                    // Batched front compaction: avoid O(n^2) memmove when
                    // removing monotonically from the front.
                    if ( pos == front_tombstones_ ) {
                        ++front_tombstones_;
                        compact_front_tombstones( false );
                        return true;
                    }
                }
                chunks_.erase( pos );
                invalidate_chunk_index_map();
            }
        }
        return true;
    }

    [[nodiscard]] bool try_add( key_type const value ) { return add( value ); }
    [[nodiscard]] bool try_remove( key_type const value ) { return remove( value ); }

    [[nodiscard]] bool intersects( bitmap const & other ) const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        const_cast<bitmap*>(this)->ensure_sorted();
        const_cast<bitmap*>(&other)->ensure_sorted();

        size_type left{ 0 };
        size_type right{ 0 };
        while ( left != chunks_.size() && right != other.chunks_.size() ) {
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

    [[ gnu::hot ]] void add_many_sorted( std::span<key_type const> const sorted_values ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        if ( sorted_values.empty() ) {
            return;
        }

        chunk_type current_chunk{};
        detail::small_array_values<low_type> grouped_values;
        auto flush_group{ [&]() {
            if ( grouped_values.empty() ) {
                return;
            }

            // If chunks are sorted (or lazy sort is off), use lower_bound+insert.
            // Otherwise use hash map to append without O(n) shifting.
            size_type pos;
            if constexpr ( !kUseLazySort ) {
                // Always sorted path.
                pos = lower_bound( current_chunk );
                if ( pos == chunks_.size() || chunks_.key( pos ) != current_chunk ) {
                    chunks_.insert( pos, current_chunk, make_policy_container_from_sorted_values(
                        { grouped_values.data(), grouped_values.size() },
                        array_to_bitset_threshold
                    ) );
                    size_ += grouped_values.size();
                } else {
                    bool const was_tombstone{ kUseLazyTombstoning && tombstone_count_ != 0 && detail::container_size( chunks_.slot( pos ) ) == 0 };
                    for ( auto const low : grouped_values ) {
                        if ( detail::container_add( chunks_.slot( pos ), low ) ) { ++size_; }
                    }
                    if ( was_tombstone ) { --tombstone_count_; }
                    promote_if_needed( chunks_.slot( pos ) );
                }
            } else if ( chunks_sorted_ ) {
                pos = lower_bound( current_chunk );
                if ( pos == chunks_.size() || chunks_.key( pos ) != current_chunk ) {
                    chunks_.insert( pos, current_chunk, make_policy_container_from_sorted_values(
                        { grouped_values.data(), grouped_values.size() },
                        array_to_bitset_threshold
                    ) );
                    size_ += grouped_values.size();
                } else {
                    bool const was_tombstone{ tombstone_count_ != 0 && detail::container_size( chunks_.slot( pos ) ) == 0 };
                    for ( auto const low : grouped_values ) {
                        if ( detail::container_add( chunks_.slot( pos ), low ) ) {
                            ++size_;
                        }
                    }
                    if ( was_tombstone ) { --tombstone_count_; }
                    promote_if_needed( chunks_.slot( pos ) );
                }
            } else {
                // Chunks unsorted; check hash map or append
                ensure_chunk_index_map();
                auto const it{ chunk_index_map_().find( current_chunk ) };
                if ( it != chunk_index_map_().end() ) {
                    pos = it->second;
                    bool const was_tombstone{ tombstone_count_ != 0 && detail::container_size( chunks_.slot( pos ) ) == 0 };
                    for ( auto const low : grouped_values ) {
                        if ( detail::container_add( chunks_.slot( pos ), low ) ) {
                            ++size_;
                        }
                    }
                    if ( was_tombstone ) { --tombstone_count_; }
                    promote_if_needed( chunks_.slot( pos ) );
                } else {
                    chunks_.push_back( current_chunk, make_policy_container_from_sorted_values(
                        { grouped_values.data(), grouped_values.size() },
                        array_to_bitset_threshold
                    ) );
                    size_ += grouped_values.size();
                    chunk_index_map_().emplace( current_chunk, chunks_.size() - 1 );
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
        if constexpr ( kUseLazySort ) {
            chunks_sorted_ = false;  // Mark chunks as potentially unsorted after add_many
        }
        invalidate_chunk_index_map();
    }

    [[ gnu::hot ]] void remove_many_sorted( std::span<key_type const> const sorted_values ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        if ( sorted_values.empty() ) {
            return;
        }

        chunk_type current_chunk{};
        detail::small_array_values<low_type> grouped_values;
        auto flush_group{ [&]() {
            if ( grouped_values.empty() ) {
                return;
            }

            auto const pos{ lower_bound( current_chunk ) };
            if ( pos != chunks_.size() && chunks_.key( pos ) == current_chunk ) {
                for ( auto const low : grouped_values ) {
                    if ( detail::container_remove( chunks_.slot( pos ), low ) ) {
                        --size_;
                    }
                }
                if ( detail::container_size( chunks_.slot( pos ) ) == 0 ) {
                    if constexpr ( kUseLazyTombstoning ) {
                        ++tombstone_count_;
                    } else {
                        chunks_.erase( pos );
                        // Note: lower_bound in next iteration is still valid since
                        // remove_many_sorted works in sorted order (monotone increasing chunks).
                    }
                } else {
                    chunks_.slot( pos ) = optimize_container_for_policy( std::move( chunks_.slot( pos ) ) );
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
        if constexpr ( kUseLazyTombstoning ) {
            if ( tombstone_count_ * 2 > static_cast<std::uint32_t>( chunks_.size() ) ) {
                compact_tombstones();
            } else {
                invalidate_chunk_index_map();
            }
        } else {
            invalidate_chunk_index_map();
        }
    }

    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
    [[ gnu::hot ]] void add_many( Range && values ) {
        if ( try_add_many_single_chunk_dense( values ) ) {
            return;
        }
        if ( try_add_many_grouped_by_chunk( values ) ) {
            return;
        }

        detail::heap_vector<key_type> sorted_values;
        if constexpr ( requires { std::ranges::size( values ); } ) {
            sorted_values.reserve( static_cast<std::uint32_t>( std::ranges::size( values ) ) );
        }
        for ( auto && value : values ) {
            sorted_values.push_back( static_cast<key_type>( value ) );
        }
        std::ranges::sort( sorted_values );
        if ( empty() ) {
            add_sorted_many( sorted_values );
        } else {
            add_many_sorted( sorted_values );
        }
    }

    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
    [[ gnu::hot ]] void remove_many( Range && values ) {
        detail::heap_vector<key_type> sorted_values;
        if constexpr ( requires { std::ranges::size( values ); } ) {
            sorted_values.reserve( static_cast<std::uint32_t>( std::ranges::size( values ) ) );
        }
        for ( auto && value : values ) {
            sorted_values.push_back( static_cast<key_type>( value ) );
        }
        std::ranges::sort( sorted_values );
        remove_many_sorted( sorted_values );
    }

    [[ gnu::cold ]] void add_sorted_many(
        std::span<key_type const> const sorted_values,
        std::uint16_t const bitset_threshold = static_cast<std::uint16_t>( array_to_bitset_threshold )
    ) {
        if ( !empty() ) {
            add_many_sorted( sorted_values );
            return;
        }

        chunks_.clear();
        size_ = 0;
        tombstone_count_ = 0;
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
        invalidate_chunk_index_map();
    }

    void add_closed_range( key_type const begin_value, key_type const end_value ) {
        apply_closed_range_operation( begin_value, end_value, detail::range_operation::add );
        invalidate_chunk_index_map();
    }

    void remove_closed_range( key_type const begin_value, key_type const end_value ) {
        apply_closed_range_operation( begin_value, end_value, detail::range_operation::remove );
        if ( tombstone_count_ * 2 > static_cast<std::uint32_t>( chunks_.size() ) ) {
            compact_tombstones();
        } else {
            invalidate_chunk_index_map();
        }
    }

    void flip_closed_range( key_type const begin_value, key_type const end_value ) {
        apply_closed_range_operation( begin_value, end_value, detail::range_operation::flip );
        invalidate_chunk_index_map();
    }

    template <typename F>
    void for_each( F && f ) const {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        for ( size_type index{ 0 }; index < chunks_.size(); ++index ) {
            auto const chunk_key{ chunks_.key( index ) };
            detail::container_for_each( chunks_.slot( index ), [&]( low_type const low ) {
                f( layout_type::compose( chunk_key, low ) );
            } );
        }
    }

    [[nodiscard]] detail::heap_vector<key_type> to_vector() const {
        detail::heap_vector<key_type> result;
        result.reserve( static_cast<std::uint32_t>( size_ ) );
        for_each( [&]( key_type const value ) { result.push_back( value ); } );
        return result;
    }
    [[nodiscard]] detail::heap_vector<key_type> to_array() const { return to_vector(); }

#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)
    using serialized_byte_vector = psi::vm::vm_vector<std::byte, std::uint32_t>;
#else
    using serialized_byte_vector = std::vector<std::byte>;
#endif

    struct serialized_header {
        std::uint32_t magic{};
        std::uint16_t key_bits{};
        std::uint16_t reserved{};
        std::uint64_t value_count{};
    };

    struct frozen_header {
        std::uint32_t magic{};
        std::uint16_t key_bits{};
        std::uint16_t version{};   // frozen_format_version; the format is a compatibility surface (persistent_bitmap)
        std::uint32_t chunk_count{};
        std::uint64_t value_count{};
    };

    struct frozen_chunk_index {
        std::uint64_t chunk{};
        std::uint32_t payload_offset{};
        std::uint32_t payload_bytes{};
        std::uint32_t payload_count{};
        std::uint32_t cardinality{};
        std::uint8_t container_kind{};
        std::uint8_t reserved[ 7 ]{};
    };

    enum class frozen_container_kind : std::uint8_t {
        array = 1,
        run = 2,
        bitset = 3
    };

    class frozen_view {
    public:
        frozen_view() = default;

        [[nodiscard]] explicit operator bool() const noexcept { return header_ != nullptr; }
        [[nodiscard]] std::size_t size() const noexcept {
            return header_ == nullptr ? 0U : static_cast<std::size_t>( header_->value_count );
        }
        [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

        [[nodiscard]] bool contains( key_type const value ) const noexcept;

        [[nodiscard]] bitmap materialize() const;

        // The zero-copy sibling of materialize(): rebuilds `out` with container
        // handles that BORROW the frozen payloads in place (the mmap-master /
        // CoW-scratchpad seed). No payload copy — only the two SoA chunk arrays
        // are built. The viewed buffer must outlive `out` and every bitmap
        // copied from it; mutating `out` CoW-clones the touched containers into
        // private storage and never writes the viewed bytes.
        void borrow_into( bitmap & out ) const;

    private:
        friend class bitmap;

        frozen_view(
            frozen_header const * header,
            frozen_chunk_index const * index,
            std::byte const * data
        ) noexcept
            : header_{ header }, index_{ index }, data_{ data } {}

        [[nodiscard]] static bool array_contains_payload(
            low_type const * values,
            std::size_t const count,
            low_type const value
        ) noexcept;

        [[nodiscard]] static bool run_contains_payload(
            typename run_container_type::run const * runs,
            int32_t const count,
            low_type const value
        ) noexcept;

        frozen_header const * header_{};
        frozen_chunk_index const * index_{};
        std::byte const * data_{};
    };

    static constexpr std::uint32_t serialization_magic{ 0x314D4252U }; // "RBM1"
    static constexpr std::uint32_t frozen_serialization_magic{ 0x31464252U }; // "RBF1"
    // Bump on any layout change to frozen_header / frozen_chunk_index / the
    // payload encodings; spec: docs/frozen-format.md.
    static constexpr std::uint16_t frozen_format_version{ 1 };

    [[nodiscard]] std::size_t serialized_size_bytes() const noexcept;

    [[nodiscard]] std::size_t frozen_size_bytes() const;

    void serialize_to_vm_vector( serialized_byte_vector & out ) const;

    void serialize_frozen_to_vm_vector( serialized_byte_vector & out ) const;

    [[nodiscard]] static bitmap deserialize_from_vm_vector( serialized_byte_vector const & in );

    [[nodiscard]] static frozen_view frozen_view_from_vm_vector( serialized_byte_vector const & in );

    [[nodiscard]] static bitmap deserialize_frozen_from_vm_vector( serialized_byte_vector const & in );

    template <typename F>
    [[nodiscard]] bool iterate( F && f ) const {
        for ( auto const value : *this ) {
            if ( !f( value ) ) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        if ( empty() ) {
            return end();
        }
        return const_iterator{ this, true };
    }

    [[nodiscard]] const_iterator end() const noexcept {
        return const_iterator{ this, false };
    }

    void swap( bitmap & other ) noexcept {
        chunks_.swap( other.chunks_ );
        if constexpr ( kUseSingletonChunkMap ) {
            singleton_chunks_.swap( other.singleton_chunks_ );
        }
        // The chunk-index cache exists (adaptively, past the probe threshold)
        // even when kUseChunkHashMap == false — its state must follow the
        // chunks_ it indexes or a swapped-in bitmap would consult the other
        // bitmap's (still-valid-flagged) index.
        chunk_index_map_box_.swap( other.chunk_index_map_box_ );
        std::swap( chunk_index_map_valid_, other.chunk_index_map_valid_ );
        std::swap( chunk_index_lookup_probes_, other.chunk_index_lookup_probes_ );
        std::swap( hot_chunk_index_, other.hot_chunk_index_ );
        std::swap( front_tombstones_, other.front_tombstones_ );
        std::swap( removes_since_structural_add_, other.removes_since_structural_add_ );
        if constexpr ( kUseLazySort ) {
            std::swap( chunks_sorted_, other.chunks_sorted_ );
        }
        if constexpr ( kUseLazyTombstoning ) {
            std::swap( tombstone_count_, other.tombstone_count_ );
        }
        std::swap( size_, other.size_ );
    }

    [[ gnu::hot ]] void intersect_into( bitmap const & other, bitmap & scratch ) const {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        if ( this == &other ) {
            scratch = *this;
            return;
        }

        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        const_cast<bitmap*>(this)->ensure_sorted();
        const_cast<bitmap*>(&other)->ensure_sorted();

        scratch.clear_keep_capacity();
        scratch.chunks_.reserve( std::min( chunks_.size(), other.chunks_.size() ) );

        // Materializing mode never mutates *this (the spine's in-place arms are
        // gated on out == nullptr), so the const_cast is only formal.
        const_cast<bitmap*>(this)->combine( other, detail::set_operation::bit_and, &scratch );
    }

    [[ gnu::hot ]] void union_into( bitmap const & other, bitmap & scratch ) const {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        if ( this == &other ) {
            scratch = *this;
            return;
        }

        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        const_cast<bitmap*>(this)->ensure_sorted();
        const_cast<bitmap*>(&other)->ensure_sorted();

        scratch.clear_keep_capacity();
        scratch.chunks_.reserve( chunks_.size() + other.chunks_.size() );
        detail::small_array_values<low_type> array_union_scratch;
#if FRSR_ROARING_COMBINE_STATS
        detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif

        size_type left{ 0 };
        size_type right{ 0 };
        while ( left != chunks_.size() && right != other.chunks_.size() ) {
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
                    detail::union_array_array_to_vector<layout_type>(
                        left_container.as_array(),
                        right_container.as_array(),
                        array_union_scratch
                    );
                    auto container{ make_fast_container_from_scratch_reusing( scratch.chunks_, array_union_scratch ) };
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

        while ( left != chunks_.size() ) {
            auto const & left_container{ chunks_.slot( left ) };
            if ( detail::container_size( left_container ) != 0 ) {
                scratch.size_ += detail::container_size( left_container );
                scratch.chunks_.push_back( chunks_.key( left ), left_container );
            }
            ++left;
        }
        while ( right != other.chunks_.size() ) {
            auto const & right_container{ other.chunks_.slot( right ) };
            if ( detail::container_size( right_container ) != 0 ) {
                scratch.size_ += detail::container_size( right_container );
                scratch.chunks_.push_back( other.chunks_.key( right ), right_container );
            }
            ++right;
        }
        scratch.invalidate_chunk_index_map();
    }

    [[ gnu::hot ]] void difference_into( bitmap const & other, bitmap & scratch ) const {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        if ( this == &other ) {
            scratch.clear_keep_capacity();
            return;
        }

        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        const_cast<bitmap*>(this)->ensure_sorted();
        const_cast<bitmap*>(&other)->ensure_sorted();

        scratch.clear_keep_capacity();
        scratch.chunks_.reserve( chunks_.size() );

        size_type left{ 0 };
        size_type right{ 0 };
        while ( left != chunks_.size() ) {
            auto const left_key{ chunks_.key( left ) };
            while ( right != other.chunks_.size() && other.chunks_.key( right ) < left_key ) {
                ++right;
            }

            if ( right == other.chunks_.size() || left_key < other.chunks_.key( right ) ) {
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
        scratch.invalidate_chunk_index_map();
    }

    // Shared union merge. When `lazy` is true, same-key bitset×bitset chunks are
    // OR'd without recomputing cardinality and `size_` is left invalid — the
    // caller MUST call repair_cardinality() afterwards. The N-way bulk-union path
    // uses this so the per-bitset popcount runs once at the end instead of K
    // times (CRoaring lazy-OR + repair). LTO folds `lazy` at each call site, so
    // operator|= keeps its eager codegen. The container_size(rhs)!=0 empty-skip
    // checks are kept in both modes — they read operand cardinalities, which are
    // always valid (only the accumulator's bitsets are lazily dirtied).
    [[ gnu::hot ]] bitmap & union_merge( bitmap const & other, bool const lazy ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        compact_tombstones();  // Ensure no tombstones in our chunk vector before merge
        detail::chunk_store<layout_type, CowPolicy> merged;
        merged.reserve( chunks_.size() + other.chunks_.size() );
#if FRSR_ROARING_COMBINE_STATS
        detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif

        size_type left{ 0 };
        size_type right{ 0 };
        size_ = 0;
        while ( left != chunks_.size() && right != other.chunks_.size() ) {
            auto const left_key { chunks_.key( left ) };
            auto const right_key{ other.chunks_.key( right ) };
            if ( left_key < right_key ) {
                if ( !lazy ) {
                    size_ += detail::container_size( chunks_.slot( left ) );
                }
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
                if ( left_container.holds_array() && right_container.holds_array() ) {
                    detail::union_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                    auto container{ make_fast_container( std::move( left_container ) ) };
                    if ( !lazy ) {
                        size_ += detail::container_size( container );
                    }
                    merged.push_back( left_key, std::move( container ) );
                    ++left;
                    ++right;
                    continue;
                }
                if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
                    auto left_bitset{ left_container.as_bitset() };
                    if ( lazy ) {
                        detail::or_bitset_bitset_inplace_lazy<layout_type>( left_bitset, right_container.as_bitset() );
                    } else {
                        detail::combine_bitset_bitset_inplace<layout_type>( left_bitset, right_container.as_bitset(), detail::set_operation::bit_or );
                        size_ += left_bitset.size();
                    }
                    merged.push_back( left_key, std::move( left_container ) );
                    ++left;
                    ++right;
                    continue;
                }
                // Same-key mixed array↔bitset in the lazy bulk-union path: accumulate
                // into a bitset in place, mirroring CRoaring's LAZY_OR_BITSET_CONVERSION
                // (roaring.c:roaring_bitmap_lazy_or_inplace). Promote the accumulator
                // (left) to a bitset if it is still an array, then OR the operand into
                // its existing 8 KB block — no 8 KB copy, no fresh allocation, no
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
                            detail::or_array_into_bitset_inplace_lazy<layout_type>( left_bitset, right_container.as_array() );
                        }
                        merged.push_back( left_key, std::move( left_container ) );
                        ++left;
                        ++right;
                        continue;
                    }
                }
                auto container{ combine_containers_for_policy(
                    left_container,
                    right_container,
                    detail::set_operation::bit_or
                ) };
                if ( !lazy ) {
                    size_ += detail::container_size( container );
                }
                merged.push_back( left_key, std::move( container ) );
                ++left;
                ++right;
            }
        }

        while ( left != chunks_.size() ) {
            if ( !lazy ) {
                size_ += detail::container_size( chunks_.slot( left ) );
            }
            merged.push_back( chunks_.key( left ), std::move( chunks_.slot( left ) ) );
            ++left;
        }
        while ( right != other.chunks_.size() ) {
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
        invalidate_chunk_index_map();
        return *this;
    }

    [[ gnu::hot ]] bitmap & operator|=( bitmap const & other ) { return union_merge( other, false ); }

    // Recomputes every chunk's cardinality when a nocard lazy word kernel left it
    // stale, and rebuilds size_. Card-updating bulk OR clears the stale flag, so a
    // dense K-way union finishes by summing headers instead of re-scanning every
    // saturated 8 KB bitset (CRoaring's repair is cheap on the same shape because
    // TO_FULL converted those to runs — we keep bitsets but skip the redundant
    // popcount when the header is already live).
    void repair_cardinality() noexcept {
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

    // The shared in-place merge-walk spine behind operator&= (op == bit_and) and
    // operator-= (op == bit_andnot) — ONE copy of the forward-compacting chunk walk
    // and its bookkeeping instead of two monomorphized spines (a differential
    // profile on a downstream workload: the three separate walk spines' combined
    // self-time ran ~2x CRoaring's single shared merge-walk spine — an
    // i-cache/layout cost, matching the general "erasure wins on size at speed
    // parity" rule).
    // op is loop-invariant, so the per-arm op branches predict perfectly; the only
    // structural difference between the two callers is the unmatched-left policy
    // (bit_and drops, bit_andnot keeps) and the array∩array kernel (scratch-based
    // intersect vs in-place difference).
    //
    // True in-place: every surviving key already exists at position >= write in our
    // own store, so the walk forward-compacts keys/slots without a scratch store
    // (no per-call allocation — the dominant cost at count=1000).
    // The ONE shared merge-walk spine for pairwise intersect/subtract combines:
    // out == nullptr → in-place, forward-compacting walk over
    // our own chunk table (operator&= / operator-=); out != nullptr → materializing,
    // *this is only read and matched results are appended to *out (intersect_into).
    // bit_andnot keeps unmatched left chunks; bit_and drops them (and early-outs
    // once the right side is exhausted). Materializing bit_andnot is unsupported
    // (no caller; the arr\arr arm below is in-place only).
    [[ gnu::hot ]] bitmap & combine( bitmap const & other, detail::set_operation const op, bitmap * const out = nullptr ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        auto const in_place           { out == nullptr };
        auto const keep_unmatched_left{ op == detail::set_operation::bit_andnot };
#if FRSR_ROARING_COMBINE_STATS
        detail::combine_stats().record_call( chunks_.size(), other.chunks_.size() );
#endif
        assert( ( in_place || !keep_unmatched_left ) && "materializing subtract has no caller/arm" );
        if ( in_place && keep_unmatched_left ) {
            compact_tombstones();  // Ensure no tombstones carried into the subtraction result
        }

        size_type write{ 0 };
        size_type left{ 0 };
        size_type right{ 0 };
        if ( in_place ) { size_ = 0; }
        auto & result_size { in_place ? size_    : out->size_    };
        auto & result_chunks{ in_place ? chunks_ : out->chunks_ };
        // Store a finished result container for the current left_key. In-place mode
        // rewrites the already-consumed prefix of our own chunk table; materializing
        // mode appends to the (pre-reserved) destination.
        auto const emit{ [&]( auto const key, handle_type && container, size_type const count ) {
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_result( container );
#endif
            result_size += count;
            if ( in_place ) {
                chunks_.set_key( write, key );
                // Retire (not free) the consumed slot being overwritten: its
                // payload becomes the next pair's / next call's result buffer
                // via take_retired (scratch-reuse path).
                chunks_.retire( std::move( chunks_.slot( write ) ) );
                chunks_.slot( write ) = std::move( container );
                ++write;
            } else {
                result_chunks.push_back( key, std::move( container ) );
            }
        } };

        while ( left != chunks_.size() ) {
            auto const left_key{ chunks_.key( left ) };
            while ( right != other.chunks_.size() && other.chunks_.key( right ) < left_key ) {
                ++right;
            }

            if ( right == other.chunks_.size() || left_key < other.chunks_.key( right ) ) {
                if ( !keep_unmatched_left ) {
                    if ( right == other.chunks_.size() ) {
                        break;  // bit_and: nothing to the right can match — drop the remainder via truncate
                    }
                    ++left;
                    continue;
                }
                size_ += detail::container_size( chunks_.slot( left ) );
                chunks_.move_entry_retiring( write, left );
                ++write;
                ++left;
                continue;
            }

            auto       & left_container { chunks_.slot( left ) };
            auto const & right_container{ other.chunks_.slot( right ) };
#if FRSR_ROARING_COMBINE_STATS
            detail::combine_stats().record_pair( left_container, right_container, op );
#endif
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
                if ( op == detail::set_operation::bit_and && in_place &&
                     intersect_result_count <= detail::kInplaceIntersectMaxElements &&
                     result_chunks.retired_capacity( detail::container_kind::array ) < intersect_result_count + 8U ) {
                    detail::intersect_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                    if ( auto const count{ detail::container_size( left_container ) }; count != 0 ) {
                        size_ += count;
                        chunks_.move_entry_retiring( write, left );
                        ++write;
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
                    handle_type result_handle{ result_chunks.take_retired( detail::container_kind::array ) };
                    auto result_array{ result_handle.as_array() };
                    detail::combine_array_array_into<layout_type>(
                        std::as_const( left_container ).as_array(),
                        right_container.as_array(),
                        detail::set_operation::bit_and,
                        result_array.values
                    );
                    if ( !result_array.values.empty() ) {
                        result_array.sync_header();
                        auto const count{ static_cast<size_type>( result_array.values.size() ) };
                        emit( left_key, std::move( result_handle ), count );
                    }
                } else {
                    detail::difference_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                    if ( left_container.count() != 0 ) {
                        auto container{ make_fast_container( std::move( left_container ) ) };
                        emit( left_key, std::move( container ), detail::container_size( container ) );
                    }
                }
                ++left;
                continue;
            }
            if ( left_container.holds_bitset() && right_container.holds_bitset() ) {
                if ( in_place ) {
                    auto left_bitset{ left_container.as_bitset() };
                    detail::combine_bitset_bitset_inplace<layout_type>( left_bitset, right_container.as_bitset(), op );
                    if ( left_bitset.size() != 0 ) {
                        size_ += left_bitset.size();
                        chunks_.move_entry_retiring( write, left );
                        ++write;
                    }
                } else {
                    // Direct-init (not assignment into a pre-declared handle): the prvalue
                    // return is elided straight into `container`, halving the
                    // copy_fields_from traffic vs move-assigning into an already-live handle.
                    auto container{ combine_bitset_bitset_for_policy(
                        std::as_const( left_container ).as_bitset(),
                        right_container.as_bitset(),
                        op,
                        result_chunks.take_retired( detail::container_kind::bitset )
                    ) };
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
            if ( in_place && left_container.holds_array() && right_container.holds_bitset() ) {
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
                auto left_array{ left_container.as_array() };
                detail::filter_array_bitset_inplace<layout_type>( left_array, right_container.as_bitset(), op == detail::set_operation::bit_and );
                if ( !left_array.values.empty() ) {
                    left_array.sync_header();
                    size_ += left_array.values.size();
                    chunks_.move_entry_retiring( write, left );
                    ++write;
                }
                ++left;
                continue;
            }
            if ( in_place && left_container.holds_array() && right_container.holds_run() ) {
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
                    chunks_.move_entry_retiring( write, left );
                    ++write;
                }
                ++left;
                continue;
            }
            bool const left_is_array { left_container.holds_array()  };
            bool const right_is_array{ right_container.holds_array() };
            if ( op == detail::set_operation::bit_and &&
                 ( ( left_is_array && right_container.holds_bitset() ) ||
                   ( left_container.holds_bitset() && right_is_array ) ) ) {
                // Materializing array∩bitset (either orientation — the in-place
                // left-array case is already consumed by the alloc-free arm above):
                // the result is an array ⊆ the array side, so write it inline into a
                // fresh handle, CRoaring-style, instead of paying the generic
                // visit/variant dispatch plus representation re-decision of
                // combine_containers (the dominant per-pair overhead in profiles).
                auto const & array_side { left_is_array ? std::as_const( left_container ) : right_container };
                auto const & bitset_side{ left_is_array ? right_container : std::as_const( left_container ) };
                handle_type result_handle{ result_chunks.take_retired( detail::container_kind::array ) };
                auto result_array{ result_handle.as_array() };
                detail::filter_array_bitset_into<layout_type>(
                    array_side.as_array(),
                    bitset_side.as_bitset(),
                    true,
                    result_array.values
                );
                if ( !result_array.values.empty() ) {
                    result_array.sync_header();
                    auto const count{ static_cast<size_type>( result_array.values.size() ) };
                    emit( left_key, std::move( result_handle ), count );
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
                    handle_type result_handle{ result_chunks.take_retired( detail::container_kind::array ) };
                    auto result_array{ result_handle.as_array() };
                    detail::filter_array_run_into<layout_type>( array_side.as_array(), run_side.as_run(), result_array.values );
                    if ( !result_array.values.empty() ) {
                        result_array.sync_header();
                        auto const count{ static_cast<size_type>( result_array.values.size() ) };
                        emit( left_key, std::move( result_handle ), count );
                    }
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
                    if ( run_cardinality >= layout_type::low_domain_size - layout_type::low_domain_size / 8U ) {
                        // Near-full runs (≥ 7/8 of the domain, incl. the full-domain
                        // run — CRoaring's run_container_is_full short-circuit): clone
                        // the bitset payload and clear only the gaps, with the
                        // cardinality maintained by subtraction — cheaper than masking
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
            if ( in_place && op == detail::set_operation::bit_andnot &&
                 left_container.holds_bitset() && right_container.holds_array() ) {
                // In-place bitset\array: clear the array's bits from our own 8 KB
                // block (exact cardinality bookkeeping, no allocation; the mutable
                // as_bitset() write barrier clones once at rc > 1, as elsewhere).
                [ & ] [[ gnu::noinline ]] () {   // outlined: see the array∩run arm above
                    auto left_bitset{ left_container.as_bitset() };
                    detail::difference_bitset_array_inplace<layout_type>( left_bitset, right_container.as_array() );
                    if ( left_bitset.size() != 0 ) {
                        size_ += left_bitset.size();
                        chunks_.move_entry_retiring( write, left );
                        ++write;
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
            chunks_.truncate_retiring( write );
            invalidate_chunk_index_map();
        } else {
            out->invalidate_chunk_index_map();
        }
        return *this;
    }

    [[ gnu::hot ]] bitmap & operator&=( bitmap const & other ) { return combine( other, detail::set_operation::bit_and ); }

    [[ gnu::hot ]] bitmap & operator-=( bitmap const & other ) { return combine( other, detail::set_operation::bit_andnot ); }

    bitmap & operator+=( bitmap const & other ) { return ( *this |= other ); }

    bitmap & operator&=( inverse_proxy const inverse ) {
        return ( *this -= inverse.bm );
    }

    // In-place N-way bulk-union step (mirrors CRoaring roaring_bitmap_lazy_or_inplace):
    // OR `other` into our existing chunk vector WITHOUT rebuilding it — same-key
    // containers are OR'd in place, this-only keys stay untouched (no move), other-only
    // keys are spliced in, and the tail is appended. cardinality is left STALE (the caller
    // must repair_cardinality()). This avoids the per-fold chunk-store allocation + full
    // chunk-move that union_merge(…, /*lazy*/true) incurs on every one of the K folds — the
    // profiler-confirmed core-bound / per-call-alloc cost that dominated the N-way union.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_lazy_or_inplace
    [[ gnu::hot ]] void bulk_or_inplace( bitmap const & other ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
            other.materialize_singleton_chunks();
        }
        compact_front_tombstones( true );
        other.compact_front_tombstones( true );
        compact_tombstones();
        auto const & rhs{ other.chunks_ };
        size_type pos1{ 0 };
        size_type pos2{ 0 };
        while ( pos1 < chunks_.size() && pos2 < rhs.size() ) {
            auto const left_key { chunks_.key( pos1 ) };
            auto const right_key{ rhs.key( pos2 ) };
            if ( left_key < right_key ) {
                ++pos1;  // this-only key: already in place, no move
                continue;
            }
            if ( right_key < left_key ) {  // other-only key: splice a copy in (rare on the leading edge)
                if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
                    chunks_.insert( pos1, right_key, rhs.slot( pos2 ) );
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
            // array∪array: merge in place only while the accumulator stays SMALL.
            // Past kLazyUnionArrayLowerBound the pair falls through to the mixed arm
            // below, which promotes the accumulator to a bitset once and then scatters
            // every later operand into it in O(|operand|) — CRoaring's lazy-union form
            // rule. Merging arrays unconditionally (as this arm used to) rewrites the
            // whole accumulator on EVERY fold, i.e. O(K·n) per chunk across a K-way
            // union; that dominated the union phase of the lazy-union-fold benchmark
            // at accumulator cardinalities above the bound (~2.6x CRoaring at 2048).
            // The finishers still re-decide the final form.
            if ( left_container.holds_array() && right_container.holds_array() &&
                 ( static_cast<std::size_t>( left_container.count() ) + right_container.count() ) <= detail::kLazyUnionArrayLowerBound ) {
                detail::union_array_array_inplace<layout_type>( left_container.as_array(), right_container.as_array() );
                left_container = make_fast_container( std::move( left_container ) );
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
            // Mixed array↔bitset: promote the accumulator to a bitset, OR the operand into
            // its existing block in place (CRoaring LAZY_OR_BITSET_CONVERSION).
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
                        detail::or_array_into_bitset_inplace_lazy<layout_type>( left_bitset, right_container.as_array() );
                    }
                    ++pos1;
                    ++pos2;
                    continue;
                }
            }
            // Run-typed operand (no SIMD in-place form): materialized combine into the slot.
            left_container = combine_containers_for_policy( left_container, right_container, detail::set_operation::bit_or );
            ++pos1;
            ++pos2;
        }
        // Tail: append other's remaining chunks (copies).
        while ( pos2 < rhs.size() ) {
            if ( detail::container_size( rhs.slot( pos2 ) ) != 0 ) {
                chunks_.push_back( rhs.key( pos2 ), rhs.slot( pos2 ) );
            }
            ++pos2;
        }
        invalidate_chunk_index_map();
    }

    // Lazy accumulation step: OR `other` in, deferring the per-bitset popcount.
    // size_ is invalid until a finisher (or or_many_in_place) calls repair_cardinality().
    [[ gnu::hot ]] void bulk_or_intermediate( bitmap const & other ) { bulk_or_inplace( other ); }
    // Reshapes with the bitmap's AMBIENT RunSelectionPolicy, not a forced eager
    // one: this finishes an ordinary (lazy-accumulated) merge result, so it must
    // NOT reintroduce eager run-selection under run_selection_lazy — that's
    // exactly the "merge results" case this policy targets. Deliberately does not
    // call the public optimize() (which forces run_selection_eager, mirroring
    // CRoaring's explicit run_optimize() and must stay caller-requested-only).
    void bulk_or_finish() noexcept {
        repair_cardinality();
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        for ( auto & slot : chunks_.slots() ) {
            slot = optimize_container_for_policy( std::move( slot ) );
        }
    }
    void bulk_or_finish_keep_bitsets() noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        repair_cardinality();  // bulk_or_intermediate left bitset cardinalities stale
        for ( auto & slot : chunks_.slots() ) {
            slot = optimize_container_keep_bitsets_for_policy( std::move( slot ) );
        }
    }

    // Exact CRoaring repair_after_lazy contract: compute deferred bitset
    // cardinalities, demote bitsets at/below the array ceiling, re-decide only
    // pre-existing run containers, and leave arrays untouched. This is narrower
    // than bulk_or_finish(), which applies the ambient storage policy to every
    // slot and needlessly scans dense bitsets for run encoding.
    void bulk_or_repair_after_lazy() noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
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

    [[ gnu::hot ]] void or_many_in_place( std::span<bitmap const * const> const others ) {
        for ( auto const other : others ) {
            if ( other != nullptr ) {
                bulk_or_intermediate( *other );
            }
        }
        bulk_or_repair_after_lazy();
    }

    [[ gnu::hot ]] void or_many_sorted_in_place( std::span<bitmap const *> others ) {
        std::ranges::sort( others, []( bitmap const * const lhs, bitmap const * const rhs ) {
            return lhs->byte_size() > rhs->byte_size();
        } );
        or_many_in_place( { others.data(), others.size() } );
    }

    void or_many_heap_in_place( std::span<bitmap const *> others ) {
        or_many_sorted_in_place( others );
    }

    [[ gnu::hot ]] void and_many_sorted_in_place( std::span<bitmap const *> others ) {
        std::ranges::sort( others, []( bitmap const * const lhs, bitmap const * const rhs ) {
            return lhs->byte_size() < rhs->byte_size();
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

    [[nodiscard]] static bitmap and_many( std::span<bitmap const *> others ) {
        if ( others.empty() ) {
            return {};
        }
        std::ranges::sort( others, []( bitmap const * const lhs, bitmap const * const rhs ) {
            return lhs->byte_size() < rhs->byte_size();
        } );
        bitmap result{ *others.front() };
        for ( auto index{ std::size_t{ 1 } }; index < others.size(); ++index ) {
            if ( others[ index ] != nullptr ) {
                result &= *others[ index ];
                if ( result.empty() ) {
                    return {};
                }
            }
        }
        return result;
    }

    void promote_large_arrays( std::uint16_t const threshold = static_cast<std::uint16_t>( array_to_bitset_threshold ) ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
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

    // Explicit, caller-requested compaction — the CRoaring run_optimize() analog.
    // Always reconsiders run-encoding (forces run_selection_eager) regardless of
    // the bitmap's ambient RunSelectionPolicy: under run_selection_lazy, ordinary
    // insert/merge never auto-picks run encoding, but this call still can, on
    // request, exactly like CRoaring never auto-run-encodes except via an
    // explicit run_optimize() call.
    void optimize() {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
        for ( auto & slot : chunks_.slots() ) {
            slot = optimize_container_for_policy<detail::run_selection_eager>( std::move( slot ) );
        }
    }

    // Faithful analog of CRoaring's roaring_bitmap_run_optimize(): run encoding is
    // reconsidered exactly as in optimize(), but an existing bitset is never
    // down-converted to an array.
    //
    // This exists for API parity, because optimize() is NOT that analog despite its
    // comment saying so. optimize() rebuilds each container from its sorted values at
    // array_to_bitset_threshold, which can turn a bitset below that threshold into an
    // array — the smaller stored form (an n-element array is 2n bytes against a
    // bitset's fixed Layout::word_count * 8), which is what optimize() is for.
    // CRoaring's run_optimize() reconsiders run encoding ONLY and has no array/bitset
    // down-conversion, so callers porting from it want this call.
    //
    // Run selection is still evaluated for a bitset (run_optimize converts bitset→run
    // when that is smaller); only the bitset→array step is suppressed.
    //
    // NOTE: no performance claim is attached to this call. On the workload it was
    // written for, optimize() was measured never to down-convert a bitset at all, so
    // the two are behaviourally identical there; the divergence from run_optimize() is
    // real in principle but was not observed to matter. Do not cite this as a
    // speed optimization without measuring it.
    void optimize_keep_bitsets() {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
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

    void optimize_for_storage() { optimize(); }
    void optimize_for_speed( std::uint16_t const threshold = static_cast<std::uint16_t>( array_to_bitset_threshold ) ) {
        promote_large_arrays( threshold );
    }
    [[nodiscard]] inverse_proxy operator!() const noexcept { return inverse_proxy{ *this }; }

    [[nodiscard]] friend bitmap operator|( bitmap const & lhs, bitmap const & rhs ) {
        bitmap result;
        lhs.union_into( rhs, result );
        return result;
    }

    [[nodiscard]] friend bitmap operator+( bitmap const & lhs, bitmap const & rhs ) {
        return lhs | rhs;
    }

    [[nodiscard]] friend bitmap operator&( bitmap const & lhs, bitmap const & rhs ) {
        bitmap result;
        lhs.intersect_into( rhs, result );
        return result;
    }

    [[nodiscard]] friend bitmap operator&( bitmap lhs, inverse_proxy const rhs ) {
        lhs &= rhs;
        return lhs;
    }

    [[nodiscard]] friend bitmap operator-( bitmap const & lhs, bitmap const & rhs ) {
        bitmap result;
        lhs.difference_into( rhs, result );
        return result;
    }

    [[nodiscard]] friend bool operator==( bitmap const & lhs, bitmap const & rhs ) {
        return lhs.to_vector() == rhs.to_vector();
    }

private:
    static constexpr bool uses_default_container_set{ std::same_as<container_set_type, default_container_set_type> };
    static constexpr bool supports_run_container{
        detail::variant_contains_v<container_set_type, run_container_type>
    };
    static constexpr bool supports_bitset_container{
        detail::variant_contains_v<container_set_type, bitset_container_type>
    };

    [[nodiscard]] static constexpr std::size_t align_up(
        std::size_t const offset,
        std::size_t const alignment
    ) noexcept {
        return ( offset + alignment - 1U ) & ~( alignment - 1U );
    }

    [[nodiscard]] static std::pair<std::size_t, std::size_t> frozen_payload_layout(
        handle_type const & container
    ) noexcept {
        switch ( container.kind() ) {
            case detail::container_kind::array:
                return {
                    alignof( low_type ),
                    static_cast<std::size_t>( container.count() ) * sizeof( low_type )
                };
            case detail::container_kind::run:
                return {
                    alignof( typename run_container_type::run ),
                    static_cast<std::size_t>( container.count() ) * sizeof( typename run_container_type::run )
                };
            case detail::container_kind::bitset:
                return {
                    alignof( std::uint64_t ),
                    static_cast<std::size_t>( layout_type::word_count ) * sizeof( std::uint64_t )
                };
        }
        std::unreachable();
    }

    [[nodiscard]] static constexpr std::size_t frozen_payload_alignment( frozen_chunk_index const & entry ) noexcept {
        switch ( static_cast<frozen_container_kind>( entry.container_kind ) ) {
            case frozen_container_kind::array:
                return alignof( low_type );
            case frozen_container_kind::run:
                return alignof( typename run_container_type::run );
            case frozen_container_kind::bitset:
                return alignof( std::uint64_t );
        }
        return 0U;
    }

    [[nodiscard]] static constexpr std::size_t frozen_expected_payload_bytes(
        frozen_chunk_index const & entry
    ) noexcept {
        switch ( static_cast<frozen_container_kind>( entry.container_kind ) ) {
            case frozen_container_kind::array:
                return static_cast<std::size_t>( entry.payload_count ) * sizeof( low_type );
            case frozen_container_kind::run:
                return static_cast<std::size_t>( entry.payload_count ) * sizeof( typename run_container_type::run );
            case frozen_container_kind::bitset:
                return static_cast<std::size_t>( layout_type::word_count ) * sizeof( std::uint64_t );
        }
        return 0U;
    }

    using chunk_store_type = detail::chunk_store<layout_type, CowPolicy>;
    static constexpr std::uint8_t singleton_chunk_capacity{ 8 };
    struct singleton_chunk_entry {
        std::array<low_type, singleton_chunk_capacity> lows{};
        std::uint8_t count{};
    };
#ifdef FRSR_USE_BOOST_FLAT_MAP
    using singleton_chunk_map = boost::unordered_flat_map<chunk_type, singleton_chunk_entry>;
#else
    using singleton_chunk_map = std::unordered_map<chunk_type, singleton_chunk_entry>;
#endif

    // Index of the first live entry with key >= chunk (chunks_.size() if none):
    // a branch-light binary search over the dense SoA keys[] array only.
    [[nodiscard]] size_type lower_bound( chunk_type const chunk ) const noexcept {
        auto const keys{ chunks_.keys() };
        auto const first{ keys.begin() + static_cast<std::ptrdiff_t>( front_tombstones_ ) };
        auto const pos{ std::lower_bound( first, keys.end(), chunk ) };
        return static_cast<size_type>( pos - keys.begin() );
    }

    // Single funnel for every hot-slot store, so the parity build provably
    // performs no write at all (rather than relying on the optimizer to sink a
    // dead store). Const because contains() must stay a read-only operation.
    void note_hot_chunk( size_type const index ) const noexcept {
        if constexpr ( kUseHotChunkIndex ) { hot_chunk_index_ = index; }
        else { (void)index; }
    }

    [[nodiscard]] bool try_hot_chunk_index( chunk_type const chunk, size_type & index ) const noexcept {
        if constexpr ( !kUseHotChunkIndex ) { (void)chunk; (void)index; return false; }
        else {
        if ( hot_chunk_index_ == invalid_index ) {
            return false;
        }
        if ( hot_chunk_index_ >= chunks_.size() ) {
            return false;
        }
        if ( hot_chunk_index_ < front_tombstones_ ) {
            return false;
        }
        if ( chunks_.key( hot_chunk_index_ ) != chunk ) {
            return false;
        }
        index = hot_chunk_index_;
        return true;
        }
    }

    [[nodiscard]] bool try_adjacent_chunk_index(
        chunk_type const chunk,
        size_type const anchor_index,
        size_type & result_index
    ) const noexcept {
        if ( anchor_index == invalid_index || anchor_index >= chunks_.size() ) {
            return false;
        }
        auto const current_key{ chunks_.key( anchor_index ) };
        if ( chunk > current_key ) {
            auto const next{ anchor_index + 1U };
            if ( next < chunks_.size() && chunks_.key( next ) == chunk ) {
                result_index = next;
                return true;
            }
        } else if ( chunk < current_key && anchor_index > front_tombstones_ ) {
            auto const prev{ anchor_index - 1U };
            if ( chunks_.key( prev ) == chunk ) {
                result_index = prev;
                return true;
            }
        }
        return false;
    }

    void compact_front_tombstones( bool const force ) const {
        // Compiled out in the parity shape, which is what keeps the const_cast
        // below off every const query path (intersects, set-op const RHS, ...).
        if constexpr ( !kUseFrontTombstones ) { (void)force; return; } else {
        if ( front_tombstones_ == 0 ) {
            return;
        }
        if ( !force && front_tombstones_ < front_tombstone_compact_threshold ) {
            return;
        }
        auto & mutable_chunks = const_cast<chunk_store_type &>( chunks_ );
        mutable_chunks.erase_front( front_tombstones_ );
        front_tombstones_ = 0;
        chunk_index_map_valid_ = false;
        hot_chunk_index_ = invalid_index;
        }
    }

    void invalidate_chunk_index_map() const noexcept {
        hot_chunk_index_ = invalid_index;
        chunk_index_map_valid_ = false;
        chunk_index_lookup_probes_ = 0;
    }

    void ensure_sorted() const {
        if constexpr ( !kUseLazySort ) { return; } // always sorted when lazy sort is off
        if ( chunks_sorted_ ) { return; }
        auto & mutable_chunks = const_cast<chunk_store_type &>( chunks_ );
        mutable_chunks.sort_by_key();
        chunks_sorted_ = true;
        chunk_index_map_valid_ = false;  // Indices changed; map must be rebuilt on next access
    }

    void ensure_chunk_index_map() const {
        if ( chunk_index_map_valid_ ) { return; }
        ensure_sorted();
        auto & map{ chunk_index_map_() };
        map.clear();
        map.reserve( chunks_.size() );
        for ( size_type i = 0; i < chunks_.size(); ++i ) {
            map.emplace( chunks_.key( i ), i );
        }
        chunk_index_map_valid_ = true;
        chunk_index_lookup_probes_ = 0;
    }

    [[nodiscard]] bool try_chunk_index_lookup(
        chunk_type const chunk,
        size_type & index,
        bool const allow_build
    ) const {
        // The adaptive chunk-index map is the kUseChunkHashMap feature. It used
        // to be reachable regardless of that flag (a runtime size threshold was
        // the only gate), which silently put a lazily-allocated unordered_map —
        // and a mutable probe counter — on the "parity baseline" lookup path
        // that CRoaring resolves with a plain binary search.
        if constexpr ( !kUseChunkHashMap ) {
            (void)chunk; (void)index; (void)allow_build;
            return false;
        } else {
        if ( chunks_.size() < chunk_index_lookup_threshold ) {
            return false;
        }
        if ( !chunk_index_map_valid_ ) {
            if ( !allow_build ) {
                return false;
            }
            if ( ++chunk_index_lookup_probes_ < chunk_index_probe_build_threshold ) {
                return false;
            }
            ensure_chunk_index_map();
        }
        auto const it{ chunk_index_map_().find( chunk ) };
        if ( it == chunk_index_map_().end() ) {
            return false;
        }
        index = it->second;
        return true;
        }
    }

    [[nodiscard]] static bool singleton_contains( singleton_chunk_entry const & entry, low_type const low ) noexcept {
        for ( std::uint8_t i = 0; i < entry.count; ++i ) {
            if ( entry.lows[ i ] == low ) {
                return true;
            }
        }
        return false;
    }

    enum class singleton_add_result : std::uint8_t {
        inserted,
        duplicate,
        full
    };

    enum class singleton_lookup_result : std::uint8_t {
        not_used,
        missing,
        found
    };

    [[nodiscard]] static singleton_add_result singleton_add( singleton_chunk_entry & entry, low_type const low ) noexcept {
        std::uint8_t pos{};
        while ( pos < entry.count && entry.lows[ pos ] < low ) {
            ++pos;
        }
        if ( pos < entry.count && entry.lows[ pos ] == low ) {
            return singleton_add_result::duplicate;
        }
        if ( entry.count >= singleton_chunk_capacity ) {
            return singleton_add_result::full;
        }
        for ( std::uint8_t i = entry.count; i > pos; --i ) {
            entry.lows[ i ] = entry.lows[ i - 1 ];
        }
        entry.lows[ pos ] = low;
        ++entry.count;
        return singleton_add_result::inserted;
    }

    [[nodiscard]] static bool singleton_remove( singleton_chunk_entry & entry, low_type const low ) noexcept {
        std::uint8_t pos{ singleton_chunk_capacity };
        for ( std::uint8_t i = 0; i < entry.count; ++i ) {
            if ( entry.lows[ i ] == low ) {
                pos = i;
                break;
            }
        }
        if ( pos == singleton_chunk_capacity ) {
            return false;
        }
        for ( std::uint8_t i = pos + 1; i < entry.count; ++i ) {
            entry.lows[ i - 1 ] = entry.lows[ i ];
        }
        --entry.count;
        return true;
    }

    [[nodiscard]] static singleton_chunk_entry singleton_from_container( handle_type const & container ) {
        singleton_chunk_entry entry{};
        auto next{ detail::container_first( container ) };
        while ( next && entry.count < singleton_chunk_capacity ) {
            entry.lows[ entry.count++ ] = *next;
            next = detail::container_next_after( container, *next );
        }
        return entry;
    }

    void invalidate_singleton_read_index() const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
        singleton_read_index_valid_ = false;
        singleton_read_index_probes_ = 0;
        hot_singleton_read_index_ = invalid_index;
        }
    }

    void reset_singleton_read_index_empty() const noexcept {
        if constexpr ( kUseSingletonChunkMap ) {
        singleton_read_index_.clear();
        singleton_read_index_valid_ = true;
        singleton_read_index_probes_ = 0;
        hot_singleton_read_index_ = invalid_index;
        }
    }

    void note_singleton_insert( chunk_type const chunk, singleton_chunk_entry const & entry ) const {
        if constexpr ( !kUseSingletonChunkMap ) { return; } else {
        if ( !singleton_read_index_valid_ ) {
            if ( singleton_chunks_.size() == 1 ) {
                reset_singleton_read_index_empty();
                singleton_read_index_.emplace_back( chunk, entry );
            }
            return;
        }
        if ( singleton_read_index_.empty() || singleton_read_index_.back().first < chunk ) {
            singleton_read_index_.emplace_back( chunk, entry );
            hot_singleton_read_index_ = invalid_index;
            return;
        }
        invalidate_singleton_read_index();
        }
    }

    void ensure_singleton_read_index() const {
        if constexpr ( !kUseSingletonChunkMap ) { return; } else {
        if ( singleton_read_index_valid_ ) {
            return;
        }
        auto & mutable_index{ const_cast<decltype(singleton_read_index_)&>( singleton_read_index_ ) };
        mutable_index.clear();
        mutable_index.reserve( static_cast<std::uint32_t>( singleton_chunks_.size() ) );
        for ( auto const & [key, entry] : singleton_chunks_ ) {
            mutable_index.emplace_back( key, entry );
        }
        std::ranges::sort( mutable_index, {}, &std::pair<chunk_type, singleton_chunk_entry>::first );
        singleton_read_index_valid_ = true;
        singleton_read_index_probes_ = 0;
        hot_singleton_read_index_ = invalid_index;
        }
    }

    [[nodiscard]] singleton_lookup_result try_singleton_read_index_lookup(
        chunk_type const chunk,
        singleton_chunk_entry const *& entry,
        bool const allow_build
    ) const {
        if constexpr ( !kUseSingletonChunkMap ) {
            return singleton_lookup_result::not_used;
        } else {
        if ( singleton_chunks_.size() < singleton_read_index_threshold ) {
            return singleton_lookup_result::not_used;
        }
        if ( !singleton_read_index_valid_ ) {
            if ( !allow_build ) {
                return singleton_lookup_result::not_used;
            }
            if ( ++singleton_read_index_probes_ < singleton_read_index_probe_build_threshold ) {
                return singleton_lookup_result::not_used;
            }
            ensure_singleton_read_index();
        }
        if ( hot_singleton_read_index_ != invalid_index && hot_singleton_read_index_ < singleton_read_index_.size() ) {
            auto const current_key{ singleton_read_index_[ hot_singleton_read_index_ ].first };
            if ( current_key == chunk ) {
                entry = &singleton_read_index_[ hot_singleton_read_index_ ].second;
                return singleton_lookup_result::found;
            }
            if ( chunk > current_key ) {
                auto const next{ hot_singleton_read_index_ + 1U };
                if ( next < singleton_read_index_.size() && singleton_read_index_[ next ].first == chunk ) {
                    hot_singleton_read_index_ = next;
                    entry = &singleton_read_index_[ next ].second;
                    return singleton_lookup_result::found;
                }
            } else if ( hot_singleton_read_index_ > 0 ) {
                auto const prev{ hot_singleton_read_index_ - 1U };
                if ( singleton_read_index_[ prev ].first == chunk ) {
                    hot_singleton_read_index_ = prev;
                    entry = &singleton_read_index_[ prev ].second;
                    return singleton_lookup_result::found;
                }
            }
        }
        auto const it{
            std::lower_bound(
                singleton_read_index_.begin(),
                singleton_read_index_.end(),
                chunk,
                []( auto const & current, chunk_type const key ) { return current.first < key; }
            )
        };
        if ( it == singleton_read_index_.end() || it->first != chunk ) {
            return singleton_lookup_result::missing;
        }
        hot_singleton_read_index_ = static_cast<size_type>( it - singleton_read_index_.begin() );
        entry = &it->second;
        return singleton_lookup_result::found;
        }
    }

    void materialize_singleton_chunks() const {
        if constexpr ( !kUseSingletonChunkMap ) { return; } else {
        if ( singleton_chunks_.empty() ) { return; }
        auto & mutable_chunks{ const_cast<chunk_store_type &>( chunks_ ) };
        chunk_store_type merged;
        merged.reserve( mutable_chunks.size() + singleton_chunks_.size() );

        detail::heap_vector<std::pair<chunk_type, singleton_chunk_entry>> sorted_singletons;
        sorted_singletons.reserve( static_cast<std::uint32_t>( singleton_chunks_.size() ) );
        for ( auto const & [key, low] : singleton_chunks_ ) {
            sorted_singletons.emplace_back( key, low );
        }
        std::ranges::sort( sorted_singletons, {}, &std::pair<chunk_type, singleton_chunk_entry>::first );

        size_type chunk_index{ 0 };
        auto const chunk_count{ mutable_chunks.size() };
        auto single_it{ sorted_singletons.begin() };
        auto const single_end{ sorted_singletons.end() };

        while ( chunk_index != chunk_count && single_it != single_end ) {
            if ( single_it->first < mutable_chunks.key( chunk_index ) ) {
                merged.push_back(
                    single_it->first,
                    handle_type::make_array_from_sorted(
                        { single_it->second.lows.data(), single_it->second.count }
                    )
                );
                ++single_it;
            } else {
                merged.push_back( mutable_chunks.key( chunk_index ), std::move( mutable_chunks.slot( chunk_index ) ) );
                ++chunk_index;
            }
        }
        while ( chunk_index != chunk_count ) {
            merged.push_back( mutable_chunks.key( chunk_index ), std::move( mutable_chunks.slot( chunk_index ) ) );
            ++chunk_index;
        }
        while ( single_it != single_end ) {
            merged.push_back(
                single_it->first,
                handle_type::make_array_from_sorted(
                    { single_it->second.lows.data(), single_it->second.count }
                )
            );
            ++single_it;
        }

        mutable_chunks.swap( merged );
        auto & mutable_singletons{ const_cast<decltype(singleton_chunks_)&>( singleton_chunks_ ) };
        mutable_singletons.clear();
        reset_singleton_read_index_empty();
        invalidate_chunk_index_map();
        }
    }

    void compact_tombstones() {
        if constexpr ( !kUseLazyTombstoning ) { return; }
        if ( tombstone_count_ == 0 ) { return; }
        if constexpr ( kUseLazySort ) {
            if ( !chunks_sorted_ ) { ensure_sorted(); }
        }
        chunks_.erase_slots_if( []( handle_type const & slot ) {
            return detail::container_size( slot ) == 0;
        } );
        tombstone_count_ = 0;
        removes_since_structural_add_ = 0;
        invalidate_chunk_index_map();
    }

    // Fast container creation from an owned array handle: skips run detection for hot
    // binary-op paths. Run conversion (if desired) is deferred to optimize_for_storage()
    // / bulk_or_finish().
    [[nodiscard, gnu::always_inline]] static handle_type
    make_fast_container( handle_type && array_handle ) noexcept {
        if constexpr ( supports_bitset_container ) {
            if ( array_handle.count() >= array_to_bitset_threshold ) {
                auto const values{ std::as_const( array_handle ).as_array().values };
                return detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
            }
        }
        return std::move( array_handle );
    }

    // Fast container creation from a scratch span: copies values and skips run detection.
    // Preserves the scratch buffer's capacity for reuse across loop iterations.
    [[nodiscard, gnu::always_inline]] static handle_type
    make_fast_container_from_scratch( detail::small_array_values<low_type> const & values ) {
        if constexpr ( supports_bitset_container ) {
            if ( values.size() >= array_to_bitset_threshold ) {
                return detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
            }
        }
        return handle_type::make_array_from_sorted( { values.data(), values.size() } );
    }

    // Reuse-aware twin of make_fast_container_from_scratch: the result is
    // rebuilt into a retired scratch payload of the matching kind when one is
    // available (scratch-reuse path), allocating only on a miss.
    [[nodiscard]] static handle_type make_fast_container_from_scratch_reusing(
        detail::chunk_store<layout_type, CowPolicy> & store,
        detail::small_array_values<low_type> const & values
    ) {
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

    // ForcedRunSelectionPolicy defaults to the bitmap's own ambient RunSelectionPolicy
    // (used by every ordinary insert/merge call site below) but can be overridden
    // explicitly — the public, caller-requested optimize()/optimize_for_storage()
    // API forces ForcedRunSelectionPolicy = run_selection_eager regardless of the
    // ambient policy, exactly mirroring CRoaring's explicit run_optimize().
    template <typename ForcedRunSelectionPolicy = RunSelectionPolicy>
    [[nodiscard]] static handle_type make_policy_container_from_sorted_vector(
        detail::small_array_values<low_type> && sorted_values,
        std::uint16_t const bitset_threshold
    ) {
        if constexpr ( uses_default_container_set ) {
            return detail::make_container_from_sorted_vector<layout_type, CowPolicy, ForcedRunSelectionPolicy>(
                std::move( sorted_values ),
                bitset_threshold
            );
        } else {
            if ( sorted_values.empty() ) {
                return handle_type{};
            }

            auto const value_count{ sorted_values.size() };
            auto const array_bytes{ value_count * sizeof( low_type ) };
            auto const bitset_bytes{ layout_type::word_count * sizeof( std::uint64_t ) };
            auto best_non_run_bytes{ array_bytes };
            if constexpr ( supports_bitset_container ) {
                if ( value_count >= bitset_threshold && bitset_bytes <= array_bytes ) {
                    best_non_run_bytes = bitset_bytes;
                }
            }

            if constexpr ( supports_run_container && ForcedRunSelectionPolicy::eager ) {
                if ( auto run{ detail::try_make_run_from_sorted_values_capped<layout_type>(
                         { sorted_values.data(), sorted_values.size() },
                         best_non_run_bytes
                     ) } ) {
                    return detail::handle_from_run_container<layout_type, CowPolicy>( *run );
                }
            }

            if constexpr ( supports_bitset_container ) {
                if ( value_count >= bitset_threshold && bitset_bytes <= array_bytes ) {
                    return detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { sorted_values.data(), sorted_values.size() } );
                }
            }
            return handle_type::make_array_from_sorted( { sorted_values.data(), sorted_values.size() } );
        }
    }

    template <typename ForcedRunSelectionPolicy = RunSelectionPolicy>
    [[nodiscard]] static handle_type make_policy_container_from_sorted_values(
        std::span<low_type const> const sorted_values,
        std::uint16_t const bitset_threshold
    ) {
        if constexpr ( uses_default_container_set ) {
            return detail::make_container_from_sorted_values<layout_type, CowPolicy, ForcedRunSelectionPolicy>( sorted_values, bitset_threshold );
        } else {
            detail::small_array_values<low_type> values;
            values.assign( sorted_values.begin(), sorted_values.end() );
            return make_policy_container_from_sorted_vector<ForcedRunSelectionPolicy>( std::move( values ), bitset_threshold );
        }
    }

    template <typename ForcedRunSelectionPolicy = RunSelectionPolicy>
    [[nodiscard]] static handle_type optimize_container_for_policy(
        handle_type container
    ) {
        if constexpr ( uses_default_container_set ) {
            return detail::optimize_container_for_storage<layout_type, CowPolicy, ForcedRunSelectionPolicy>(
                std::move( container ),
                array_to_bitset_threshold
            );
        } else {
            auto values{ detail::sorted_values_from_container<layout_type>( container ) };
            detail::small_array_values<low_type> normalized_values;
            normalized_values.assign( values.begin(), values.end() );
            return make_policy_container_from_sorted_vector<ForcedRunSelectionPolicy>(
                std::move( normalized_values ),
                static_cast<std::uint16_t>( array_to_bitset_threshold )
            );
        }
    }

    template <typename ForcedRunSelectionPolicy = RunSelectionPolicy>
    [[nodiscard]] static handle_type optimize_container_keep_bitsets_for_policy(
        handle_type container
    ) {
        if constexpr ( uses_default_container_set ) {
            return detail::optimize_container_keep_bitsets<layout_type, CowPolicy, ForcedRunSelectionPolicy>(
                std::move( container ),
                array_to_bitset_threshold
            );
        } else {
            if constexpr ( supports_bitset_container ) {
                if ( container.holds_bitset() ) {
                    return container;
                }
            }
            return optimize_container_for_policy<ForcedRunSelectionPolicy>( std::move( container ) );
        }
    }

    [[nodiscard]] static handle_type make_container_from_words_for_policy(
        detail::word_array<layout_type> const & words
    ) {
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

    [[nodiscard]] static handle_type combine_containers_for_policy(
        handle_type const & lhs,
        handle_type const & rhs,
        detail::set_operation const op
    ) {
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

    // Same-key bitset×bitset materialized combine, bypassing the 2-D std::visit in
    // combine_containers (whose bitset×bitset arm just forwards to combine_bitset_bitset
    // with no array-downgrade, so this is identical minus the dispatch). The dense
    // count=100k scenarios are bitset×bitset, so this is the hot materialized arm —
    // the materialized analog of the in-place fast path in union_merge/operator&=/-=.
    // Demote a low-cardinality bitset combine result to an array container,
    // retiring the vacated 8 KB payload for take_retired reuse. CRoaring
    // converts AND/ANDNOT bitset results to array below DEFAULT_MAX_SIZE so
    // chained folds run array kernels instead of full word loops; doing the
    // demotion inside the kernel instead destroyed the adopted scratch bitset
    // (breaking the chunk_store scratch cycle) and regressed the fold-heavy
    // witness up to +14% — hence spine-side, with retirement.
    template <typename Store>
    [[nodiscard]] static handle_type demote_sparse_bitset( handle_type && container, Store & store ) {
        if ( container.holds_bitset() ) {
            auto const cardinality{ container.cardinality() };
            if ( cardinality < array_to_bitset_threshold ) {
                auto array{ detail::array_from_bitset<layout_type, CowPolicy>(
                    std::as_const( container ).as_bitset(),
                    cardinality
                ) };
                store.retire( std::move( container ) );
                return array;
            }
        }
        return std::move( container );
    }

    [[nodiscard]] static handle_type combine_bitset_bitset_for_policy(
        detail::bitset_cref<layout_type, CowPolicy> const lhs,
        detail::bitset_cref<layout_type, CowPolicy> const rhs,
        detail::set_operation const op,
        handle_type && reuse = {}
    ) {
        auto container{ detail::combine_bitset_bitset<layout_type, CowPolicy>( lhs, rhs, op, std::move( reuse ) ) };
        if constexpr ( uses_default_container_set ) {
            return container;
        } else {
            return optimize_container_for_policy( std::move( container ) );
        }
    }

    // Policy wrapper for the combine spine's run∩bitset arm — same contract as
    // combine_bitset_bitset_for_policy above (identical minus the kernel).
    [[nodiscard]] static handle_type intersect_run_bitset_for_policy(
        detail::run_cref<layout_type, CowPolicy> const runs,
        detail::bitset_cref<layout_type, CowPolicy> const bitset,
        handle_type && reuse = {}
    ) {
        auto container{ detail::intersect_run_bitset<layout_type, CowPolicy>( runs, bitset, std::move( reuse ) ) };
        if constexpr ( uses_default_container_set ) {
            return container;
        } else {
            return optimize_container_for_policy( std::move( container ) );
        }
    }

    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
    [[nodiscard]] bool try_add_many_grouped_by_chunk( Range && values ) {
        if ( !empty() ) {
            return false;
        }

        if constexpr ( !requires { std::ranges::size( values ); } ) {
            return false;
        }

        auto const count{ std::ranges::size( values ) };
        if ( count < 128U || count > 2048U ) {
            return false;
        }

        struct chunk_bucket {
            chunk_type key{};
            detail::small_array_values<low_type> lows;
        };

        detail::heap_vector<chunk_bucket> buckets;
        buckets.reserve( static_cast<std::uint32_t>( std::min<std::size_t>( count / 8U, 128U ) ) );
        for ( auto && value : values ) {
            auto const full_value{ static_cast<key_type>( value ) };
            auto const chunk{ layout_type::chunk_key( full_value ) };
            auto const low{ layout_type::low_key( full_value ) };

            auto const it{ std::find_if( buckets.begin(), buckets.end(), [chunk]( chunk_bucket const & bucket ) {
                return bucket.key == chunk;
            } ) };
            if ( it == buckets.end() ) {
                auto & bucket{ buckets.emplace_back( chunk_bucket{ .key = chunk } ) };
                bucket.lows.reserve( 16U );
                bucket.lows.push_back( low );
            } else {
                it->lows.push_back( low );
            }
        }
        if ( buckets.empty() ) {
            return true;
        }

        std::ranges::sort( buckets, {}, &chunk_bucket::key );

        chunks_.clear();
        size_ = 0;
        tombstone_count_ = 0;
        chunks_.reserve( buckets.size() );
        for ( auto & bucket : buckets ) {
            std::ranges::sort( bucket.lows );
            auto const unique_end{ std::unique( bucket.lows.begin(), bucket.lows.end() ) };
            bucket.lows.erase( unique_end, bucket.lows.end() );
            if ( bucket.lows.empty() ) {
                continue;
            }

            handle_type container;
            if constexpr ( uses_default_container_set ) {
                if ( bucket.lows.size() < 64U ) {
                    container = handle_type::make_array_from_sorted(
                        { bucket.lows.data(), bucket.lows.size() }
                    );
                } else {
                    container = make_policy_container_from_sorted_vector(
                        std::move( bucket.lows ),
                        array_to_bitset_threshold
                    );
                }
            } else {
                container = make_policy_container_from_sorted_vector(
                    std::move( bucket.lows ),
                    array_to_bitset_threshold
                );
            }
            size_ += detail::container_size( container );
            chunks_.push_back( bucket.key, std::move( container ) );
        }
        invalidate_chunk_index_map();
        return true;
    }

    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
    [[nodiscard]] bool try_add_many_single_chunk_dense( Range && values ) {
        if constexpr ( !std::ranges::forward_range<Range> || !requires { std::ranges::size( values ); } ) {
            return false;
        } else {
            if ( !chunks_.empty() ) {
                return false;
            }

            auto const count{ std::ranges::size( values ) };
            if ( count < array_to_bitset_threshold ) {
                return false;
            }

            auto it{ std::ranges::begin( values ) };
            auto const last{ std::ranges::end( values ) };
            if ( it == last ) {
                return true;
            }

            auto const first_value{ static_cast<key_type>( *it ) };
            auto const chunk{ layout_type::chunk_key( first_value ) };
            auto bitset_handle{ handle_type::make_bitset_zeroed() };
            auto bitset{ bitset_handle.as_bitset() };
            auto min_low{ layout_type::low_key( first_value ) };
            auto max_low{ min_low };

            for ( ; it != last; ++it ) {
                auto const value{ static_cast<key_type>( *it ) };
                if ( layout_type::chunk_key( value ) != chunk ) {
                    return false;
                }

                auto const low{ layout_type::low_key( value ) };
                static_cast<void>( bitset.add( low ) );
                min_low = std::min( min_low, low );
                max_low = std::max( max_low, low );
            }

            auto const bitset_cardinality{ bitset_handle.cardinality() };
            if ( bitset_cardinality == 0 ) {
                return true;
            }

            auto const span{ static_cast<std::size_t>( max_low ) - static_cast<std::size_t>( min_low ) + 1U };
            if ( bitset_cardinality == span ) {
                // Implicit run-creation site (a fully contiguous range within one
                // chunk from an ordinary add_many call) — gated by RunSelectionPolicy
                // like every other "should sorted/added values become a run"
                // decision. Under run_selection_lazy this falls through to the
                // bitset-cardinality check below, same as for container sets that
                // don't support run containers at all.
                if constexpr ( supports_run_container && RunSelectionPolicy::eager ) {
                    detail::run_container<layout_type> run;
                    run.runs.push_back( frsr::roaring::run<low_type>{ min_low, max_low } );
                    run.cardinality = static_cast<typename detail::run_container<layout_type>::cardinality_type>( bitset_cardinality );
                    chunks_.push_back( chunk, detail::handle_from_run_container<layout_type, CowPolicy>( run ) );
                    size_ = bitset_cardinality;
                    invalidate_chunk_index_map();
                    return true;
                } else if constexpr ( !supports_bitset_container ) {
                    return false;
                }
            }

            auto const array_bytes{ bitset_cardinality * sizeof( low_type ) };
            auto const bitset_bytes{ layout_type::word_count * sizeof( std::uint64_t ) };
            if constexpr ( supports_bitset_container ) {
                if ( bitset_cardinality >= array_to_bitset_threshold && bitset_bytes <= array_bytes ) {
                    size_ = bitset_cardinality;
                    chunks_.push_back( chunk, std::move( bitset_handle ) );
                    invalidate_chunk_index_map();
                    return true;
                }
            }

            return false;
        }
    }

    [[nodiscard]] bool add_impl( key_type const value, bulk_context * const ctx ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        if constexpr ( kUseSingletonChunkMap ) {
            auto singleton_it{ singleton_chunks_.find( chunk ) };
            if ( singleton_it != singleton_chunks_.end() ) {
                switch ( singleton_add( singleton_it->second, low ) ) {
                    case singleton_add_result::inserted:
                        invalidate_singleton_read_index();
                        ++size_;
                        if ( ctx != nullptr ) {
                            ctx->chunk = chunk;
                            ctx->index = invalid_index;
                        }
                        return true;
                    case singleton_add_result::duplicate:
                        return false;
                    case singleton_add_result::full:
                        break;
                }
                auto const existing{ singleton_it->second };
                singleton_chunks_.erase( singleton_it );
                invalidate_singleton_read_index();
                auto const sorted_pos{ lower_bound( chunk ) };
                handle_type array_handle;
                auto array{ array_handle.as_array() };
                auto const insert_pos{
                    std::lower_bound( existing.lows.begin(), existing.lows.begin() + existing.count, low )
                };
                auto const inserted_index{
                    static_cast<std::size_t>( insert_pos - existing.lows.begin() )
                };
                array.values.resize_uninitialized( existing.count + 1U );
                for ( std::uint8_t i = 0; i < existing.count; ++i ) {
                    auto const dest_index{ static_cast<std::size_t>( i ) + ( i >= inserted_index ? 1U : 0U ) };
                    array.values[ dest_index ] = existing.lows[ i ];
                }
                array.values[ inserted_index ] = low;
                array.sync_header();
                chunks_.insert( sorted_pos, chunk, std::move( array_handle ) );
                if ( ctx != nullptr ) {
                    ctx->chunk = chunk;
                    ctx->index = sorted_pos;
                }
                note_hot_chunk( sorted_pos );
                ++size_;
                return true;
            }

            size_type sorted_pos;
            size_type indexed_pos{};
            if ( try_chunk_index_lookup( chunk, indexed_pos, false ) ) {
                sorted_pos = indexed_pos;
            } else {
                sorted_pos = lower_bound( chunk );
            }
            if ( sorted_pos == chunks_.size() || chunks_.key( sorted_pos ) != chunk ) {
                singleton_chunk_entry entry{};
                entry.lows[ 0 ] = low;
                entry.count = 1;
                singleton_chunks_.emplace( chunk, entry );
                note_singleton_insert( chunk, entry );
                ++size_;
                if ( ctx != nullptr ) {
                    ctx->chunk = chunk;
                    ctx->index = invalid_index;
                }
                return true;
            } else {
                if ( ctx != nullptr ) {
                    ctx->chunk = chunk;
                    ctx->index = sorted_pos;
                }
                note_hot_chunk( sorted_pos );
                bool const was_tombstone{
                    kUseLazyTombstoning && tombstone_count_ != 0 &&
                    detail::container_size( chunks_.slot( sorted_pos ) ) == 0
                };
                auto const added{ detail::container_add( chunks_.slot( sorted_pos ), low ) };
                if ( !added ) { return false; }
                ++size_;
                if ( was_tombstone ) { --tombstone_count_; }
                promote_if_needed( chunks_.slot( sorted_pos ) );
                return true;
            }
        }

        size_type pos;
        bool found_existing{ false };
        bool structural_insert{ false };
        if (
            ctx != nullptr &&
            ctx->index != invalid_index &&
            ctx->index < chunks_.size() &&
            chunks_.key( ctx->index ) == chunk
        ) {
            pos = ctx->index;
            found_existing = true;
        } else if constexpr ( kUseChunkHashMap ) {
            // Fast path for already-added chunks via hash map.
            ensure_chunk_index_map();
            auto const it{ chunk_index_map_().find( chunk ) };
            if ( it != chunk_index_map_().end() ) {
                pos = it->second;
                found_existing = true;
            } else if constexpr ( kUseLazySort ) {
                // New chunk: push_back (O(1)), mark unsorted, update map.
                chunks_.push_back( chunk, handle_type{} );
                pos = chunks_.size() - 1;
                chunks_sorted_ = false;
                removes_since_structural_add_ = 0;
                chunk_index_map_().emplace( chunk, chunks_.size() - 1 );
            } else {
                // Hash map lookup but always-sorted: sorted insert, then invalidate map.
                auto const sorted_pos{ lower_bound( chunk ) };
                chunks_.insert( sorted_pos, chunk, handle_type{} );
                pos = sorted_pos;
                invalidate_chunk_index_map();
            }
        } else {
            // CRoaring-style: O(log n) binary search + O(n) sorted insert.
            size_type hot_index{};
            if ( try_hot_chunk_index( chunk, hot_index ) ) {
                pos = hot_index;
                found_existing = true;
            } else if ( try_chunk_index_lookup( chunk, hot_index, false ) ) {
                pos = hot_index;
                found_existing = true;
            } else {
                auto const sorted_pos{ lower_bound( chunk ) };
                if ( sorted_pos != chunks_.size() && chunks_.key( sorted_pos ) == chunk ) {
                    pos = sorted_pos;
                    found_existing = true;
                } else {
                    // If remove() left a front tombstone window and there are no active
                    // chunks, reuse the last tombstone slot instead of growing + shifting.
                    if (
                        kUseFrontTombstones &&
                        front_tombstones_ != 0 &&
                        sorted_pos == chunks_.size() &&
                        sorted_pos == front_tombstones_
                    ) {
                        --front_tombstones_;
                        pos = front_tombstones_;
                        chunks_.set_key( pos, chunk );
                        chunks_.slot( pos ) = handle_type{};
                        structural_insert = true;
                    } else {
                        chunks_.insert( sorted_pos, chunk, handle_type{} );
                        pos = sorted_pos;
                        structural_insert = true;
                    }
                }
            }
        }

        if ( structural_insert ) {
            invalidate_chunk_index_map();
        }
        if ( ctx != nullptr ) {
            ctx->chunk = chunk;
            ctx->index = pos;
        }
        note_hot_chunk( pos );

        bool const was_tombstone{
            kUseLazyTombstoning && found_existing &&
            tombstone_count_ != 0 &&
            detail::container_size( chunks_.slot( pos ) ) == 0
        };
        auto const added{ detail::container_add( chunks_.slot( pos ), low ) };
        if ( !added ) { return false; }

        ++size_;
        if ( was_tombstone ) { --tombstone_count_; }
        promote_if_needed( chunks_.slot( pos ) );
        // No invalidate here: adding to an existing chunk doesn't shift any
        // chunk's index, so the key→index map remains valid.
        return true;
    }

    void promote_if_needed( handle_type & slot ) {
        if ( slot.holds_array() && slot.count() >= array_to_bitset_threshold ) {
            if constexpr ( supports_bitset_container ) {
                // [croaring-ref] deps/croaring/include/roaring/roaring.h: array→bitset promotion concept
                // std::as_const avoids an unneeded write-barrier clone under a
                // refcounted CowPolicy: slot is about to be wholly replaced below.
                auto const values{ std::as_const( slot ).as_array().values };
                slot = detail::bitset_handle_from_sorted_values<layout_type, CowPolicy>( { values.data(), values.size() } );
            } else {
                slot = optimize_container_for_policy( std::move( slot ) );
            }
        }
    }

    void apply_closed_range_operation(
        key_type const begin_value,
        key_type const end_value,
        detail::range_operation const operation
    ) {
        if constexpr ( kUseSingletonChunkMap ) {
            materialize_singleton_chunks();
        }
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

    void apply_chunk_range_operation(
        chunk_type const chunk,
        low_type const low_begin,
        low_type const low_end,
        detail::range_operation const operation
    ) {
        size_type pos;
        if constexpr ( !kUseLazySort ) {
            // Always sorted: use lower_bound.
            pos = lower_bound( chunk );
            if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) {
                if ( operation == detail::range_operation::remove ) { return; }
                chunks_.insert( pos, chunk, handle_type{} );
                invalidate_chunk_index_map();
            }
        } else if ( chunks_sorted_ ) {
            pos = lower_bound( chunk );
            if ( pos == chunks_.size() || chunks_.key( pos ) != chunk ) {
                if ( operation == detail::range_operation::remove ) { return; }
                chunks_.insert( pos, chunk, handle_type{} );
            }
        } else {
            ensure_chunk_index_map();
            auto const it{ chunk_index_map_().find( chunk ) };
            if ( it != chunk_index_map_().end() ) {
                pos = it->second;
            } else {
                if ( operation == detail::range_operation::remove ) { return; }
                chunks_.push_back( chunk, handle_type{} );
                pos = chunks_.size() - 1;
                chunk_index_map_().emplace( chunk, chunks_.size() - 1 );
            }
        }

        auto const previous_size{ detail::container_size( chunks_.slot( pos ) ) };
        auto words{ detail::words_from_container<layout_type>( chunks_.slot( pos ) ) };
        detail::apply_range_to_words<layout_type>( words, low_begin, low_end, operation );
        chunks_.slot( pos ) = make_container_from_words_for_policy( words );
        auto const updated_size{ detail::container_size( chunks_.slot( pos ) ) };

        if ( updated_size > previous_size ) {
            size_ += updated_size - previous_size;
        } else {
            size_ -= previous_size - updated_size;
        }
        if ( updated_size == 0 ) {
            if constexpr ( kUseLazyTombstoning ) {
                ++tombstone_count_;
            } else {
                chunks_.erase( pos );
                invalidate_chunk_index_map();
            }
        }
    }

    // ── Feature flags ─────────────────────────────────────────────────────────
    // All false ⇒ the plain CRoaring strategy: always-sorted chunks, O(log n)
    // binary-search lookup, immediate erase, no singleton-chunk staging. This is
    // the parity baseline — frsr must match CRoaring here before any of the novel
    // optimizations below are enabled to try to beat it.
    // Note: kUseLazySort requires kUseChunkHashMap (unsorted chunks need the map).
    static constexpr bool kUseLazySort        = false; // novel: defer chunk sort to set-op paths
    static constexpr bool kUseChunkHashMap    = false; // novel: hash-map chunk lookup vs binary search
    static constexpr bool kUseLazyTombstoning = false; // novel: mark-erased vs immediate erase
    // Novel: remember the last resolved chunk slot and probe it (plus its two
    // neighbours) before the binary search. CRoaring has no analog on the plain
    // lookup path — it keeps this amortization opt-in via roaring_bulk_context_t
    // (our bulk_context / contains_bulk), so the library never guesses the
    // caller's access pattern. With this off, contains() is a *pure* read: no
    // store to a mutable member, hence no write-sharing between threads that
    // hold the same bitmap by const reference.
    static constexpr bool kUseHotChunkIndex   = false; // novel: implicit last-chunk cache in contains()
    // Novel: when remove() empties the front-most chunk, mark it instead of
    // erasing, and batch the erase (avoids O(n^2) memmove on monotonic
    // front-removal). CRoaring erases immediately (ra_remove_at_index). The
    // batching is why compact_front_tombstones() has to const_cast and
    // structurally erase from *const* query paths — intersects(), and every
    // set op, force it on their const RHS. With this off the deferral never
    // happens, front_tombstones_ stays 0, and those const paths mutate nothing,
    // restoring CRoaring's documented "concurrent access to the same bitmap is
    // safe as long as you do not modify it" guarantee.
    static constexpr bool kUseFrontTombstones = false; // novel: batched front-chunk erase
    // Novel: stage sparse single-key chunks in a side map to cut add()/remove()
    // cost on random workloads (no CRoaring analog). To re-enable in the beat phase,
    // gate on uint32/uint64 key_type (the staging assumes a 16-bit chunk-key split):
    //   = !kUseLazySort && !kUseChunkHashMap && !kUseLazyTombstoning &&
    //     ( std::same_as<key_type, std::uint32_t> || std::same_as<key_type, std::uint64_t> )
    static constexpr bool kUseSingletonChunkMap = false;
    static constexpr std::uint8_t singleton_chunk_demote_threshold{ singleton_chunk_capacity };
    static constexpr size_type chunk_index_lookup_threshold{ 512 };
    static constexpr std::uint32_t chunk_index_probe_build_threshold{ 8 };
    static constexpr size_type singleton_read_index_threshold{ 16384 };
    static constexpr std::uint32_t singleton_read_index_probe_build_threshold{ 8 };
    static_assert( !kUseLazySort || kUseChunkHashMap,
        "kUseLazySort requires kUseChunkHashMap" );
    // ─────────────────────────────────────────────────────────────────────────

    chunk_store_type chunks_;
#ifdef FRSR_USE_BOOST_FLAT_MAP
    using chunk_index_map_type = boost::unordered_flat_map<chunk_type, size_type>;
#else
    using chunk_index_map_type = std::unordered_map<chunk_type, size_type>;
#endif
    // Boxed: the adaptive chunk-index cache is built only for large bitmaps
    // (past chunk_index_lookup_threshold probes), so the common small bitmap
    // pays one pointer of footprint instead of the whole map object (48 B),
    // keeping sizeof(bitmap) lean for by-value storage in flat containers.
    mutable std::unique_ptr<chunk_index_map_type> chunk_index_map_box_{};
    [[nodiscard]] chunk_index_map_type & chunk_index_map_() const {
        if ( !chunk_index_map_box_ ) [[unlikely]] {
            chunk_index_map_box_ = std::make_unique<chunk_index_map_type>();
        }
        return *chunk_index_map_box_;
    }
#if defined( _MSC_VER )
#   define FRSR_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#   define FRSR_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif
    // Flags-off bookkeeping members are empty feature_field stand-ins (see detail::
    // feature_field): with every novel flag currently off these collapse to zero
    // footprint under [[no_unique_address]], shrinking the by-value bitmap that a
    // downstream N-way AND fold and every downstream bitmap-wrapper copy move
    // around, WITHOUT guarding their many use sites. The map *box* itself stays a
    // live unique_ptr — its
    // machinery (ensure/chunk_index_map_) is only reached from kUseChunkHashMap-
    // guarded call sites, so no body needs conditioning.
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<bool,          kUseChunkHashMap, false> chunk_index_map_valid_{};
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<std::uint32_t, kUseChunkHashMap       > chunk_index_lookup_probes_{};
    static constexpr std::uint32_t front_tombstone_compact_threshold{ 4096 };
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<std::uint32_t, kUseFrontTombstones     > front_tombstones_{};
    // The singleton-chunk staging state exists only when the feature flag is on:
    // with kUseSingletonChunkMap == false these members would be pure dead
    // footprint (~80 B of the handle) copied/moved with every bitmap and padding
    // out by-value storage of bitmaps in flat containers, so they are compiled
    // out to empty stand-ins ([[msvc::no_unique_address]] keeps them size-free
    // under the MSVC ABI clang-cl targets; plain [[no_unique_address]] is a
    // no-op there).
    struct empty_member_stub {};
    template <typename T> using iff_singleton_map = std::conditional_t<kUseSingletonChunkMap, T, empty_member_stub>;
    static constexpr iff_singleton_map<size_type> initial_hot_singleton_read_index() noexcept {
        if constexpr ( kUseSingletonChunkMap ) { return invalid_index; }
        else                                   { return {};            }
    }
    FRSR_NO_UNIQUE_ADDRESS mutable iff_singleton_map<singleton_chunk_map> singleton_chunks_{};
    FRSR_NO_UNIQUE_ADDRESS mutable iff_singleton_map<detail::heap_vector<std::pair<chunk_type, singleton_chunk_entry>>> singleton_read_index_{};
    FRSR_NO_UNIQUE_ADDRESS mutable iff_singleton_map<bool> singleton_read_index_valid_{};
    FRSR_NO_UNIQUE_ADDRESS mutable iff_singleton_map<std::uint32_t> singleton_read_index_probes_{};
    FRSR_NO_UNIQUE_ADDRESS mutable iff_singleton_map<size_type> hot_singleton_read_index_{ initial_hot_singleton_read_index() };
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<size_type, kUseHotChunkIndex, invalid_index> hot_chunk_index_{};
    // Counts consecutive remove() calls since the last structural add (chunk insert).
    // When this reaches the lazy-build threshold we rebuild the map so bulk-remove
    // sessions amortise to O(1) per remove, while alternating add/remove patterns
    // avoid O(n) map rebuilds.  Reset to 0 by add_impl() new-chunk inserts and by
    // compact_tombstones() (both change vector structure / indices).
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<std::uint32_t, kUseChunkHashMap> removes_since_structural_add_{};
    // Tracks whether chunks_ needs sorting. When true, any set operation or iteration
    // path must call ensure_sorted() before proceeding. This defers expensive O(n log n)
    // sorts from hot add() paths to less-hot set-operation paths.
    // Invariant: always true when kUseLazySort == false.
    FRSR_NO_UNIQUE_ADDRESS mutable detail::feature_field<bool, kUseLazySort, true> chunks_sorted_{};
    // Non-zero only when kUseLazyTombstoning == true.
    FRSR_NO_UNIQUE_ADDRESS detail::feature_field<std::uint32_t, kUseLazyTombstoning> tombstone_count_{};
#undef FRSR_NO_UNIQUE_ADDRESS
    size_type size_{};
};

} // namespace frsr::roaring

#include <frsr/roaring/serialization.hpp>

namespace frsr::roaring {
extern template class bitmap<std::uint16_t>;
extern template class bitmap<std::uint32_t>;
extern template class bitmap<std::uint64_t>;
extern template class bitmap<
    std::uint32_t,
    default_container_set<std::uint32_t>,
    detail::cow_atomic_refcount
>;
// A downstream consumer's actual instantiation: CoW Model 2 +
// run_selection_lazy (CRoaring parity — never auto-picks run encoding outside
// optimize()/optimize_for_storage()).
extern template class bitmap<
    std::uint32_t,
    default_container_set<std::uint32_t>,
    detail::cow_atomic_refcount,
    detail::run_selection_lazy
>;
} // namespace frsr::roaring
