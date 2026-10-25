#pragma once

// frsr::roaring::bitmap. This header declares the class and defines what has to inline into a caller (one-key
// lookups and updates, the iterators, size queries, and the templates that adapt a caller's range or operand).
// The bodies of the set operations, conversions, merges, range updates, optimization and serialization are in
// bitmap_definitions.hpp, compiled once into the frsr_roaring library for the bitmaps declared extern at the end
// of this file. Defining FRSR_ROARING_HEADER_ONLY instead brings them into every translation unit, and
// FRSR_ROARING_WITH_DEFINITIONS (or including bitmap_definitions.hpp) lets a program instantiate a bitmap the
// library does not hold. See the README, "Compiled library and header-only mode".

#include <frsr/roaring/allocation_policy.hpp>
#include <frsr/roaring/container_layout.hpp>
#include <frsr/roaring/run_container.hpp>
#include <frsr/roaring/run_selection_policy.hpp>
#include <frsr/roaring/tuning.hpp>
#include <frsr/roaring/operations.hpp>
#include <frsr/roaring/hw_info.hpp>
#include <frsr/roaring/containers.hpp>
#include <frsr/roaring/chunk_store.hpp>
#include <frsr/roaring/interop_operands.hpp>

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
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>   // the serialized_byte_vector fallback uses std::vector even with psi::vm present
#ifdef FRSR_ROARING_HAS_PSI_VM
#   include <psi/vm/containers/small_vector.hpp>
#   include <psi/vm/containers/lookup.hpp>
#   ifdef FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION
#       include <psi/vm/containers/vm_vector.hpp>
#   endif
    // heap_vector already pulled in via run_container.hpp → psi/vm/containers/heap_vector.hpp
#endif

namespace frsr::roaring {

// Options of the frozen image writer (docs/frozen-format.md). A ranked image carries, per chunk, the data that
// turns count_below / index_of / select into constant-time reads (see ranked_view.hpp); it is format version 2,
// while an unranked image stays byte-identical version 1. A ranked image needs a key type of at most 32 bits and
// fewer than 2^32 values.
struct frozen_options {
    bool ranked{ false };
};

namespace detail {
// Defined by the tests that look at the containers of a bitmap (see tests/view_support.hpp).
template <typename Bitmap>
struct bitmap_test_access;
// Defined by ranked_view.hpp: the reader of the frozen image's rank data, a friend of frozen_view.
template <typename Bitmap>
struct frozen_reader_access;
}

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
//
// Concurrency. Any number of threads may read one bitmap at the same time as long
// as none of them modifies it, for every key width - the guarantee the reference
// implementation's bitmaps give. A const member function stores nothing to the
// bitmap: point queries (contains, and contains_bulk with a context private to the
// calling thread), cardinality, iteration, conversion to a vector, and
// the set operations that read this bitmap as an operand (the result is a new
// bitmap owned by the caller). Copying a const bitmap is such a read, and so is a
// set operation whose result shares chunk payloads with its operands, for the
// value-semantics and the atomic-refcount CowPolicy; the unsynchronized-refcount
// policy updates a plain shared count on both, so there those need one thread at
// a time. Anything that can modify the bitmap - every non-const member - needs
// exclusive access, and a bulk_context is the caller's own state: it is not shared
// between threads.
// The chunk table is therefore always kept sorted and is searched, never cached or
// indexed lazily from a const path; a lookup accelerator may be added only if all of
// its state is repaired inside non-const operations.
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
    using handle_type      = detail::container_handle<layout_type, CowPolicy>;
    using chunk_store_type = detail::chunk_store<layout_type, CowPolicy>;
    using chunk_cursor     = typename chunk_store_type::cursor;

public:

    static constexpr size_type array_to_bitset_threshold{
        layout_type::low_domain_size >= 1024 ? layout_type::low_domain_size / 16 : 64
    };

    struct container_statistics {
        size_type array_containers{};
        size_type run_containers{};
        size_type bitset_containers{};

        [[nodiscard]] constexpr size_type container_count() const noexcept {
            return array_containers + run_containers + bitset_containers;
        }
    };

    // What `!bitmap` is: the complement of a bitmap as an operand (see complement in interop_operands.hpp).
    using inverse_proxy = complement<bitmap>;

    // Caches the chunk a sequence of add_bulk()/contains_bulk() calls is working
    // on. `payload_private` records that this chunk's payload has already been
    // made private for this sequence, so the run's remaining adds skip the
    // copy-on-write barrier - the reference implementation's bulk context does
    // the same ([croaring-ref] deps/croaring/src/roaring.c:add_bulk_impl adds to
    // its cached container directly, having unshared it once on the miss path).
    //
    // Precondition, shared with that reference: no copy of the bitmap may be
    // taken while a context on it is alive. A copy is what re-shares a payload,
    // and a writer that has already made one private would then mutate the
    // copy's data. Callers already have to guarantee this - a copy taken during
    // any in-place write sequence would observe a half-applied one anyway - and
    // in practice it falls out of the lock discipline: writers hold the
    // exclusive lock, readers copy under a shared one.
    //
    // A context belongs to the bitmap it was filled from. It holds the resolved slot
    // itself, as the reference caches its container, together with the chunk table's
    // generation at the time: any call that restructures the table (a chunk added or
    // removed, a set operation, a swap) advances the generation, the context is then
    // stale, and the next call locates the chunk again. The slot pointer and the cursor
    // are never dereferenced before the generation has been compared.
    struct bulk_context {
        chunk_type chunk{};
        chunk_cursor at{};
        handle_type const * slot{ nullptr };   // nullptr: nothing is cached
        std::uint32_t generation{ 0 };
        bool payload_private{ false };

        void reset() noexcept { slot = nullptr; payload_private = false; }
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
            return lhs.owner_ == rhs.owner_ && lhs.chunk_ == rhs.chunk_ && lhs.current_low_ == rhs.current_low_;
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
            chunk_ = chunk_cursor{};
            current_low_.reset();
            current_value_ = {};
        }

        void seek_first() noexcept {
            for ( chunk_ = owner_->chunks_.begin(); chunk_ != owner_->chunks_.end(); ++chunk_ ) {
                if ( auto const low{ detail::container_first( owner_->chunks_.slot( chunk_ ) ) } ) {
                    current_low_ = low;
                    current_value_ = layout_type::compose( owner_->chunks_.key( chunk_ ), *low );
                    return;
                }
            }
            finish();
        }

        void advance() noexcept {
            if ( owner_ == nullptr ) {
                return;
            }

            if ( auto const next_low{ detail::container_next_after( owner_->chunks_.slot( chunk_ ), *current_low_ ) } ) {
                current_low_ = next_low;
                current_value_ = layout_type::compose( owner_->chunks_.key( chunk_ ), *next_low );
                return;
            }

            for ( ++chunk_; chunk_ != owner_->chunks_.end(); ++chunk_ ) {
                if ( auto const low{ detail::container_first( owner_->chunks_.slot( chunk_ ) ) } ) {
                    current_low_ = low;
                    current_value_ = layout_type::compose( owner_->chunks_.key( chunk_ ), *low );
                    return;
                }
            }

            finish();
        }

        bitmap const * owner_{};
        chunk_cursor chunk_{};
        std::optional<low_type> current_low_{};
        value_type current_value_{};
    };

    bitmap() = default;

    bitmap( bitmap const & ) noexcept( detail::allocation_failure_terminates ) = default;

    bitmap( bitmap && ) noexcept = default;

    bitmap & operator=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
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
    explicit bitmap( Range && values ) noexcept( detail::allocation_failure_terminates && detail::nothrow_key_range<Range, key_type> ) {
        add_many( std::forward<Range>( values ) );
    }

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool test( key_type const value ) const noexcept { return contains( value ); }
    // Not meaningful while the bitmap is in the middle of a lazy fold (see bulk_or_intermediate).
    [[nodiscard]] size_type size() const noexcept { assert( !counts_deferred() ); return size_; }
    [[nodiscard]] size_type byte_size() const noexcept;

    // Administrative charge per chunk in data_byte_size(), fixed at the reference
    // format's descriptor width rather than at this implementation's handle width.
    static constexpr size_type chunk_descriptor_bytes{ 8 };

    // Size estimate for ordering a multi-way fold: the set data each chunk holds,
    // plus one fixed descriptor charge per chunk so that a bitmap made of many
    // nearly empty chunks is not judged free.
    //
    // Deliberately excludes the container handles. A handle is 32 bytes of index
    // belonging to this implementation — four times the descriptor charge — so
    // counting it makes a sparse many-chunk bitmap look several times heavier than
    // the data it actually holds, which reorders a smallest-data-first fold away
    // from the smallest data. byte_size() answers the other question, how much
    // memory the bitmap occupies, and is the wrong key for a cost heuristic.
    [[nodiscard]] size_type data_byte_size() const noexcept;
    // data_byte_size() read off at most eight evenly spaced chunks and scaled to the chunk count: a cost
    // key for ordering a fold when the operands' chunk tables are too long to walk once per call. Exact for
    // a bitmap of up to eight chunks. Constant time for the flat (16-bit key) table; a wide-key bitmap
    // has no random access to its chunks and answers with data_byte_size().
    [[nodiscard]] size_type estimated_data_byte_size() const noexcept;
    [[nodiscard]] std::uint16_t container_count() const noexcept {
        return static_cast<std::uint16_t>( chunks_.size() );
    }
    // Checks the structural invariants the set operations and the portable format rely on, and returns a
    // short description of the first one that does not hold, or nullptr when they all do. The portable
    // format tells an array from a bitset by cardinality alone, so an array above the array limit is not a
    // legal container however its values are ordered. A bitmap in the middle of a bulk_or_intermediate
    // fold carries stale bitset cardinalities and size by design and is reported as broken until it is
    // finished. With canonical_kinds, a bitset at or below the array limit is reported too: the kind a
    // result of a set operation settles in is then a function of its cardinality, not of the operation's
    // path. Linear in the bitmap's size: a debugging and testing aid, never a hot-path check.
    [[nodiscard]] char const * validate( bool const canonical_kinds = false ) const noexcept;

    [[nodiscard]] container_statistics statistics() const noexcept;
    [[nodiscard]] key_type front() const noexcept { return *begin(); }
    [[nodiscard]] key_type back() const noexcept;

    void clear() noexcept {
        chunks_.clear();
        size_ = 0;
    }
    // Scratch reuse (CRoaring persistent-dst analog): logically empty, but the
    // container payloads are retired for chunk_store::take_retired() reuse by
    // the next materializing run instead of being freed (and immediately
    // reallocated) on every *_into combine.
    void clear_keep_capacity() noexcept {
        chunks_.clear_retiring_slots();
        size_ = 0;
    }

public:

    // A pure binary search over the chunk keys, nothing written, so concurrent readers of one
    // bitmap are safe. [croaring-ref] roaring_bitmap_contains -> containerptr_roaring is
    // likewise a read-only container lookup; amortization is opt-in through
    // roaring_bitmap_contains_bulk, mirrored here by contains_bulk.
    [[nodiscard]] [[gnu::always_inline]] bool contains( key_type const value ) const noexcept {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const at{ chunks_.find( chunk ) };
        if ( at == chunks_.end() ) { return false; }
        return detail::container_contains( chunks_.slot( at ), layout_type::low_key( value ) );
    }

    [[nodiscard]] bool contains_bulk( bulk_context & ctx, key_type const value ) const noexcept {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low  { layout_type::low_key( value ) };
        if ( ctx.slot != nullptr && ctx.generation == chunks_.generation() ) [[likely]] {
            if ( ctx.chunk == chunk ) {
                return detail::container_contains( *ctx.slot, low );
            }
            if ( try_adjacent_chunk( chunk, ctx.at ) ) {
                ctx.chunk           = chunk;
                ctx.slot            = &chunks_.slot( ctx.at );
                ctx.payload_private = false;
                return detail::container_contains( *ctx.slot, low );
            }
        }
        auto const at{ lower_bound_outlined( chunk ) };
        if ( at == chunks_.end() || chunks_.key( at ) != chunk ) {
            ctx.reset();
            return false;
        }
        ctx.chunk           = chunk;
        ctx.at              = at;
        ctx.slot            = &chunks_.slot( at );
        ctx.generation      = chunks_.generation();
        ctx.payload_private = false;
        return detail::container_contains( *ctx.slot, low );
    }

    [[nodiscard]] bool add( key_type const value ) noexcept( detail::allocation_failure_terminates ) {
        return add_impl( value, nullptr );
    }

    // The context-hit lane is the whole function for a run of adds into one
    // chunk, as the reference's add_bulk_impl is: the cached chunk is checked
    // and the value goes straight into its container. Everything else (a new
    // chunk) is the shared add_impl.
    // [croaring-ref] deps/croaring/src/roaring.c:add_bulk_impl
    [[nodiscard]] bool add_bulk( bulk_context & ctx, key_type const value ) noexcept( detail::allocation_failure_terminates ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        if ( ctx.slot != nullptr && ctx.generation == chunks_.generation() && ctx.chunk == chunk ) [[likely]] {
            // The context was filled from this (non-const) bitmap, whose slots are not const.
            auto       & slot{ const_cast<handle_type &>( *ctx.slot ) };
            auto const   low { layout_type::low_key( value ) };
            // The common hit is an in-order append into a private array: one
            // inline lane, as the reference's is, before the kind dispatch.
            if ( ctx.payload_private && slot.holds_array() ) [[likely]] {
                if ( detail::as_array_already_private( slot ).try_append( low ) ) [[likely]] {
                    ++size_;
                    promote_if_needed( slot );
                    return true;
                }
            }
            auto const added{
                ctx.payload_private
                    ? detail::container_add_already_private( slot, low )
                    : detail::container_add               ( slot, low )
            };
            if ( !added ) { return false; }
            ctx.payload_private = true;
            ++size_;
            promote_if_needed( slot );
            return true;
        }
        return add_impl( value, &ctx );
    }

    [[nodiscard]] bool remove( key_type const value ) noexcept( detail::allocation_failure_terminates ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };
        auto const at{ chunks_.lower_bound( chunk ) };
        if ( at == chunks_.end() || chunks_.key( at ) != chunk ) { return false; }
        if ( !detail::container_remove( chunks_.slot( at ), low ) ) { return false; }

        --size_;
        if ( detail::container_size( chunks_.slot( at ) ) == 0 ) {
            // [croaring-ref] deps/croaring/src/roaring_array.c:ra_remove_at_index
            chunks_.erase( at );
        }
        return true;
    }

    [[nodiscard]] bool try_add( key_type const value ) noexcept( detail::allocation_failure_terminates ) { return add( value ); }
    [[nodiscard]] bool try_remove( key_type const value ) noexcept( detail::allocation_failure_terminates ) { return remove( value ); }

    [[nodiscard]] bool intersects( bitmap const & other ) const noexcept;

    // ---- operations against operands that are not bitmaps: keys in a span or any range, std::bitset and
    // dynamic bit-sets, raw bit words, key intervals. See interop_operands.hpp for the operand kinds and
    // interop.hpp for how they are applied. Keys of an operand that key_type cannot hold are not members.
    //
    // and_with / or_with / andnot_with (and &= |= -=) update this bitmap; they accept a bitmap too, so that
    // the name works for every operand.

    template <interop_operand Operand>
    bitmap & and_with( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { interop_dispatch<detail::interop_op::and_with>( *this, operand ); return *this; }
    template <interop_operand Operand>
    bitmap & or_with( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { interop_dispatch<detail::interop_op::or_with>( *this, operand ); return *this; }
    template <interop_operand Operand>
    bitmap & andnot_with( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { interop_dispatch<detail::interop_op::andnot_with>( *this, operand ); return *this; }
    // Keeps the keys that are in exactly one of this bitmap and the operand. With a key_range this is the flip of
    // the interval.
    template <interop_operand Operand>
    bitmap & xor_with( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { interop_dispatch<detail::interop_op::xor_with>( *this, operand ); return *this; }
    bitmap & and_with   ( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return *this &= other; }
    bitmap & or_with    ( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return *this |= other; }
    bitmap & andnot_with( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return *this -= other; }
    bitmap & xor_with   ( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return *this ^= other; }

    template <interop_operand Operand>
    bitmap & operator&=( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { return and_with( operand ); }
    template <interop_operand Operand>
    bitmap & operator|=( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { return or_with( operand ); }
    template <interop_operand Operand>
    bitmap & operator-=( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { return andnot_with( operand ); }
    template <interop_operand Operand>
    bitmap & operator^=( Operand const & operand ) noexcept( detail::allocation_failure_terminates && detail::interop_nothrow_v<Operand> ) { return xor_with( operand ); }

    // Whether this bitmap and the operand share a key.
    template <interop_operand Operand>
    [[nodiscard]] bool intersects( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::intersects>( *this, operand ); }
    // How many keys this bitmap and the operand share (each key once, whatever the operand repeats).
    template <interop_operand Operand>
    [[nodiscard]] size_type and_cardinality( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::and_cardinality>( *this, operand ); }
    [[nodiscard]] size_type and_cardinality( bitmap const & other ) const;
    // How many keys are in exactly one of this bitmap and the operand.
    template <interop_operand Operand>
    [[nodiscard]] size_type xor_cardinality( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::xor_cardinality>( *this, operand ); }
    // Whether every key of this bitmap is in the operand.
    template <interop_operand Operand>
    [[nodiscard]] bool is_subset_of( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::is_subset_of>( *this, operand ); }
    [[nodiscard]] bool is_subset_of( bitmap const & other ) const;
    // Whether every key of the operand is in this bitmap.
    template <interop_operand Operand>
    [[nodiscard]] bool is_superset_of( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::is_superset_of>( *this, operand ); }
    [[nodiscard]] bool is_superset_of( bitmap const & other ) const { return other.is_subset_of( *this ); }
    // Whether this bitmap and the operand hold the same keys.
    template <interop_operand Operand>
    [[nodiscard]] bool equals( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) { return interop_dispatch<detail::interop_op::equals>( *this, operand ); }
    [[nodiscard]] bool equals( bitmap const & other ) const { return size_ == other.size_ && is_subset_of( other ); }

    // ---- against the complement of an operand (`!x`, `complement_of( x )`; see interop_operands.hpp). Each of these
    // is the operation it is rewritten to; none builds the complement.
    // The result of a union or an exclusive-or with a complement is the rest of the key space and is wanted for
    // no operand but an interval, and the queries that would walk the whole complement are not offered.
#if defined( __cpp_deleted_function ) && __cpp_deleted_function >= 202403L
#   define FRSR_ROARING_DELETED_COMPLEMENT( WHY ) delete( WHY )
#else
#   define FRSR_ROARING_DELETED_COMPLEMENT( WHY ) delete
#endif
    template <complementable Operand>
    bitmap & and_with( complement<Operand> const & c ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) { return andnot_with( c.operand ); }
    template <complementable Operand>
    bitmap & andnot_with( complement<Operand> const & c ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) { return and_with( c.operand ); }
    // The two intervals outside the interval.
    bitmap & or_with( complement<key_range> const & c ) noexcept( detail::allocation_failure_terminates ) {
        auto const [ below, above ]{ detail::outside( c.operand ) };
        or_with( below );
        return or_with( above );
    }
    bitmap & xor_with( complement<key_range> const & c ) noexcept( detail::allocation_failure_terminates ) {
        auto const [ below, above ]{ detail::outside( c.operand ) };
        xor_with( below );
        return xor_with( above );
    }
    template <typename Operand>
    bitmap & or_with( complement<Operand> const & ) = FRSR_ROARING_DELETED_COMPLEMENT( "the union with the complement of an operand is the rest of the key space: only the complement of a key_range is offered" );
    template <typename Operand>
    bitmap & xor_with( complement<Operand> const & ) = FRSR_ROARING_DELETED_COMPLEMENT( "the exclusive-or with the complement of an operand is the rest of the key space: only the complement of a key_range is offered" );

    template <complementable Operand>
    bitmap & operator&=( complement<Operand> const & c ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) { return and_with( c ); }
    template <complementable Operand>
    bitmap & operator-=( complement<Operand> const & c ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) { return andnot_with( c ); }
    bitmap & operator|=( complement<key_range> const & c ) noexcept( detail::allocation_failure_terminates ) { return or_with( c ); }
    bitmap & operator^=( complement<key_range> const & c ) noexcept( detail::allocation_failure_terminates ) { return xor_with( c ); }
    template <typename Operand>
    bitmap & operator|=( complement<Operand> const & ) = FRSR_ROARING_DELETED_COMPLEMENT( "the union with the complement of an operand is the rest of the key space: only the complement of a key_range is offered" );
    template <typename Operand>
    bitmap & operator^=( complement<Operand> const & ) = FRSR_ROARING_DELETED_COMPLEMENT( "the exclusive-or with the complement of an operand is the rest of the key space: only the complement of a key_range is offered" );

    // Whether this bitmap has a key outside the operand; how many.
    template <complementable Operand>
    [[nodiscard]] bool intersects( complement<Operand> const & c ) const noexcept( detail::complement_query_nothrow_v<Operand> ) { return !is_subset_of( c.operand ); }
    template <complementable Operand>
    [[nodiscard]] size_type and_cardinality( complement<Operand> const & c ) const noexcept( detail::complement_query_nothrow_v<Operand> ) { return size() - and_cardinality( c.operand ); }
    // Whether no key of this bitmap is in the operand.
    template <complementable Operand>
    [[nodiscard]] bool is_subset_of( complement<Operand> const & c ) const noexcept( detail::complement_query_nothrow_v<Operand> ) { return !intersects( c.operand ); }
    template <typename Operand>
    [[nodiscard]] size_type xor_cardinality( complement<Operand> const & ) const = FRSR_ROARING_DELETED_COMPLEMENT( "the complement of an operand is the rest of the key space: its exclusive-or cardinality is not offered" );
    template <typename Operand>
    [[nodiscard]] bool is_superset_of( complement<Operand> const & ) const = FRSR_ROARING_DELETED_COMPLEMENT( "the complement of an operand is the rest of the key space: no bitmap is expected to hold it" );
    template <typename Operand>
    [[nodiscard]] bool equals( complement<Operand> const & ) const = FRSR_ROARING_DELETED_COMPLEMENT( "the complement of an operand is the rest of the key space: no bitmap is expected to equal it" );
#undef FRSR_ROARING_DELETED_COMPLEMENT

    // ---- sizes of combinations and the strict subset, without building the combination (the operand is a bitmap, a
    // view of one or an interval where its size is needed; both sides are finished bitmaps).
    // How many keys of this bitmap are not in the operand. [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_andnot_cardinality
    template <complementable Operand>
    [[nodiscard]] size_type andnot_cardinality( Operand const & operand ) const noexcept( detail::complement_query_nothrow_v<Operand> ) { return size() - and_cardinality( operand ); }
    // How many keys of this bitmap are in the operand's complement: the ones the operand has.
    template <complementable Operand>
    [[nodiscard]] size_type andnot_cardinality( complement<Operand> const & c ) const noexcept( detail::complement_query_nothrow_v<Operand> ) { return and_cardinality( c.operand ); }
    // How many keys are in this bitmap or the operand. [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_or_cardinality
    // (An interval that holds every 64-bit key has a size no size_type can hold: not a valid operand here.)
    template <typename Operand>
        requires ( std::same_as<Operand, bitmap> || std::same_as<Operand, typed_view<bitmap>> || std::same_as<Operand, key_range> )
    [[nodiscard]] size_type or_cardinality( Operand const & operand ) const noexcept( detail::complement_query_nothrow_v<Operand> ) {
        return size() + operand_size( operand ) - and_cardinality( operand );
    }
    // |this & operand| / |this | operand|, NaN for two empty sets, as the reference. [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_jaccard_index
    template <typename Operand>
        requires ( std::same_as<Operand, bitmap> || std::same_as<Operand, typed_view<bitmap>> || std::same_as<Operand, key_range> )
    [[nodiscard]] double jaccard_index( Operand const & operand ) const noexcept( detail::complement_query_nothrow_v<Operand> ) {
        auto const common{ and_cardinality( operand ) };
        return static_cast<double>( common ) / ( static_cast<double>( size() ) + operand_size_as_double( operand ) - static_cast<double>( common ) );
    }
    // Whether every key of this bitmap is in the operand and the operand has more. [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_is_strict_subset
    template <typename Operand>
        requires ( std::same_as<Operand, bitmap> || std::same_as<Operand, typed_view<bitmap>> || std::same_as<Operand, key_range> )
    [[nodiscard]] bool is_strict_subset_of( Operand const & operand ) const noexcept( detail::complement_query_nothrow_v<Operand> ) {
        return operand_has_more_than( operand, size() ) && is_subset_of( operand );
    }
    template <interop_operand Operand>
        requires ( !std::same_as<Operand, typed_view<bitmap>> && !std::same_as<Operand, key_range> )
    [[nodiscard]] bool is_strict_subset_of( Operand const & operand ) const noexcept( detail::interop_nothrow_v<Operand> ) {
        return is_subset_of( operand ) && !equals( operand );
    }

    // The half-open interval [lo, hi) (empty if hi <= lo); the whole key space and intervals that run to the
    // largest key are key_range::all() / key_range::from( lo ), through the operations above.
    [[nodiscard]] bool      intersects_range    ( key_type const lo, key_type const hi ) const noexcept { return intersects( key_range::half_open( lo, hi ) ); }
    // Whether every key of [lo, hi) is in this bitmap (true for an empty interval).
    [[nodiscard]] bool      contains_range      ( key_type const lo, key_type const hi ) const noexcept { return is_superset_of( key_range::half_open( lo, hi ) ); }
    [[nodiscard]] size_type cardinality_in_range( key_type const lo, key_type const hi ) const noexcept { return and_cardinality( key_range::half_open( lo, hi ) ); }
    // Keeps only the keys of [lo, hi).
    bitmap &                and_range           ( key_type const lo, key_type const hi )       noexcept( detail::allocation_failure_terminates ) { return and_with( key_range::half_open( lo, hi ) ); }


    [[ gnu::hot ]] void add_many_sorted( std::span<key_type const> const sorted_values ) noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] void remove_many_sorted( std::span<key_type const> const sorted_values ) noexcept( detail::allocation_failure_terminates );

    // The values in any order, possibly repeated; a contiguous range of key_type reaches these through add_many /
    // remove_many, and the cost of the sort and the merge is paid in the library, not in the caller.
    [[ gnu::hot ]] void add_many_unsorted( std::span<key_type const> values ) noexcept( detail::allocation_failure_terminates );
    [[ gnu::hot ]] void remove_many_unsorted( std::span<key_type const> values ) noexcept( detail::allocation_failure_terminates );

    template <std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
    [[ gnu::hot ]] void add_many( Range && values ) noexcept( detail::allocation_failure_terminates && detail::nothrow_key_range<Range, key_type> ) {
        if constexpr ( detail::contiguous_key_values<Range, key_type> ) {
            add_many_unsorted( std::span<key_type const>{ std::ranges::data( values ), static_cast<std::size_t>( std::ranges::size( values ) ) } );
            return;
        }
        if ( try_add_many_single_chunk_dense( values ) ) {
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
    [[ gnu::hot ]] void remove_many( Range && values ) noexcept( detail::allocation_failure_terminates && detail::nothrow_key_range<Range, key_type> ) {
        if constexpr ( detail::contiguous_key_values<Range, key_type> ) {
            remove_many_unsorted( std::span<key_type const>{ std::ranges::data( values ), static_cast<std::size_t>( std::ranges::size( values ) ) } );
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
        remove_many_sorted( sorted_values );
    }

    [[ gnu::cold ]] void add_sorted_many(
        std::span<key_type const> const sorted_values,
        std::uint16_t const bitset_threshold = static_cast<std::uint16_t>( array_to_bitset_threshold )
    );

    void add_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates );

    void remove_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates );

    void flip_closed_range( key_type const begin_value, key_type const end_value ) noexcept( detail::allocation_failure_terminates );

    template <typename F>
    void for_each( F && f ) const {
        // Values come out in order, so the chunks must be in order first: a wide
        // chunk key defers the sort until a path needs it, and this is such a path.
        for ( auto at{ chunks_.begin() }; at != chunks_.end(); ++at ) {
            auto const chunk_key{ chunks_.key( at ) };
            detail::container_for_each( chunks_.slot( at ), [&]( low_type const low ) {
                f( layout_type::compose( chunk_key, low ) );
            } );
        }
    }

    // Decodes into a CALLER-OWNED buffer (which must hold size() values) and returns
    // how many were written. The buffer-owning form is the primitive: a caller that
    // decodes repeatedly — a read path handing values out to its own consumer — keeps
    // one buffer instead of paying an allocation and, more expensively, the first-touch
    // page faults of a fresh one on every call.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_to_uint32_array
    [[ gnu::hot ]] std::size_t to_array_into( std::span<key_type> const out ) const;

    [[nodiscard]] detail::heap_vector<key_type> to_vector() const;
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
        std::uint8_t reserved[ 3 ]{};
        std::uint32_t base{};   // version 2: the number of values in all earlier chunks; zero in version 1
    };

    enum class frozen_container_kind : std::uint8_t {
        array = 1,
        run = 2,
        bitset = 3
    };

    class frozen_view {
    public:
        frozen_view() = default;

        // The result of a rank lookup in one chunk: the number of values below the probe, and whether the probe is present.
        struct position {
            std::uint32_t below;
            bool          present;
        };

        [[nodiscard]] explicit operator bool() const noexcept { return header_ != nullptr; }
        // Whether the image carries the rank data (format version 2).
        [[nodiscard]] bool ranked() const noexcept { return header_ != nullptr && header_->version == frozen_ranked_format_version; }
        // The bytes of the image the view covers (header, index, payloads and rank data).
        [[nodiscard]] std::size_t byte_size() const noexcept { return bytes_; }
        [[nodiscard]] std::size_t size() const noexcept {
            return header_ == nullptr ? 0U : static_cast<std::size_t>( header_->value_count );
        }
        [[nodiscard]] bool empty() const noexcept { return size() == 0U; }

        [[nodiscard]] bool contains( key_type const value ) const noexcept;

        [[nodiscard]] bitmap materialize() const;

        // f( key ) for every value >= `first`, in key order. The first chunk is decoded whole and filtered.
        template <typename F>
        void for_each_from( key_type const first, F && f ) const {
            if ( header_ == nullptr || header_->chunk_count == 0U ) {
                return;
            }
            auto const first_chunk{ static_cast<std::uint64_t>( layout_type::chunk_key( first ) ) };
            auto const first_low{ layout_type::low_key( first ) };
            auto const index_span{ std::span<frozen_chunk_index const>{ index_, static_cast<std::size_t>( header_->chunk_count ) } };
            auto at{ std::ranges::lower_bound( index_span, first_chunk, {}, &frozen_chunk_index::chunk ) };
            for ( ; at != index_span.end(); ++at ) {
                auto const & entry{ *at };
                auto const container{ detail::container_handle<layout_type, CowPolicy>::make_borrowed(
                    container_kind_of( static_cast<frozen_container_kind>( entry.container_kind ) ),
                    data_ + entry.payload_offset,
                    entry.payload_count,
                    entry.cardinality
                ) };
                auto const chunk_key{ static_cast<chunk_type>( entry.chunk ) };
                // only the chunk of `first` can hold values below it
                auto const from_low{ entry.chunk == first_chunk ? first_low : low_type{} };
                detail::container_for_each( container, [&]( low_type const low ) {
                    if ( low >= from_low ) {
                        f( layout_type::compose( chunk_key, low ) );
                    }
                } );
        }
        }
        template <typename F>
        void for_each( F && f ) const { for_each_from( key_type{}, std::forward<F>( f ) ); }

        // The zero-copy sibling of materialize(): rebuilds `out` with container
        // handles that BORROW the frozen payloads in place (the mmap-master /
        // CoW-scratchpad seed). No payload copy — only the two SoA chunk arrays
        // are built. The viewed buffer must outlive `out` and every bitmap
        // copied from it; mutating `out` CoW-clones the touched containers into
        // private storage and never writes the viewed bytes.
        void borrow_into( bitmap & out ) const;

    private:
        friend class bitmap;
        template <typename>
        friend struct detail::frozen_reader_access;

        frozen_view(
            frozen_header const * header,
            frozen_chunk_index const * index,
            std::byte const * data,
            std::size_t const bytes
        ) noexcept
            : header_{ header }, index_{ index }, data_{ data }, bytes_{ bytes } {}

        // The number of values below `value` (the insertion point) in a sorted array payload.
        [[nodiscard]] static std::int32_t array_lower_bound(
            low_type const * values,
            std::size_t const count,
            low_type const value
        ) noexcept;

        [[nodiscard]] static bool array_contains_payload(
            low_type const * values,
            std::size_t const count,
            low_type const value
        ) noexcept {
            auto const below{ static_cast<std::size_t>( array_lower_bound( values, count, value ) ) };
            return below < count && values[ below ] == value;
        }

        // Rank of `value` in a run payload whose per-run cumulative counts are `run_ranks` (the ranked image's rank block).
        [[nodiscard]] static position run_position(
            typename run_container_type::run const * runs,
            std::uint16_t const * run_ranks,
            std::int32_t const count,
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
        std::size_t bytes_{};
    };

    static constexpr std::uint32_t serialization_magic{ 0x314D4252U }; // "RBM1"
    static constexpr std::uint32_t frozen_serialization_magic{ 0x31464252U }; // "RBF1"
    // Bump on any layout change to frozen_header / frozen_chunk_index / the
    // payload encodings; spec: docs/frozen-format.md.
    static constexpr std::uint16_t frozen_format_version{ 1 };
    // The version of an image that carries the rank data (frozen_options::ranked); readers accept both.
    static constexpr std::uint16_t frozen_ranked_format_version{ 2 };

    [[nodiscard]] std::size_t serialized_size_bytes() const noexcept;

    [[nodiscard]] std::size_t frozen_size_bytes( frozen_options options = {} ) const;

    void serialize_to_vm_vector( serialized_byte_vector & out ) const;

    void serialize_frozen_to_vm_vector( serialized_byte_vector & out, frozen_options options = {} ) const;

    // The frozen image written into caller storage, which needs
    // `out.size() >= frozen_size_bytes()` and an 8-byte aligned `out.data()` (the image's
    // offsets are relative to its start; a programming error otherwise) and need not
    // be zeroed; returns the bytes written (== frozen_size_bytes( options )).
    std::size_t frozen_serialize( std::span<std::byte> out, frozen_options options = {} ) const;

    [[nodiscard]] static bitmap deserialize_from_vm_vector( serialized_byte_vector const & in );

    [[nodiscard]] static frozen_view frozen_view_from_vm_vector( serialized_byte_vector const & in );

    // The validating reader over caller-owned bytes (a flat buffer, a mapped file): null if the image is malformed.
    // `image.data()` must be 8-byte aligned, and the bytes must outlive the view.
    [[nodiscard]] static frozen_view frozen_view_from( std::span<std::byte const> image );

    [[nodiscard]] static bitmap deserialize_frozen_from_vm_vector( serialized_byte_vector const & in );

    // The standard Roaring portable format (https://github.com/RoaringBitmap/RoaringFormatSpec):
    // 32-bit key bitmaps are byte-compatible with roaring_bitmap_portable_serialize,
    // 64-bit key bitmaps with roaring64_bitmap_portable_serialize (u64 bucket count,
    // then per distinct high 32 bits: the u32 key and a 32-bit portable bitmap of the
    // low 32 bits). Work is O(containers + payload bytes): array and bitset payloads
    // are copied whole, run payloads are re-encoded (begin,end) <-> (begin,length-1)
    // per run; no value is visited. Little-endian hosts only.
    //
    // portable_serialize() needs `out.size() >= portable_size_in_bytes()` (programming
    // error otherwise) and returns the number of bytes written (== the size query).
    // portable_deserialize() never trusts the input: every length is checked against
    // `in` before it is read, keys and array values must be strictly increasing, runs
    // in range, bitset popcounts equal the declared cardinalities. It returns nullopt
    // on any violation; bytes after the encoded bitmap are ignored.
    [[nodiscard]] std::size_t portable_size_in_bytes() const requires ( std::numeric_limits<Key>::digits >= 32 );

    std::size_t portable_serialize( std::span<std::byte> out ) const requires ( std::numeric_limits<Key>::digits >= 32 );

    [[nodiscard]] static std::optional<bitmap> portable_deserialize( std::span<std::byte const> in ) requires ( std::numeric_limits<Key>::digits >= 32 );

    template <typename F>
    [[nodiscard]] bool iterate( F && f ) const {
        for ( auto const value : *this ) {
            if ( !f( value ) ) {
                return false;
            }
        }
        return true;
    }

    // Not noexcept: ordering the chunks of a wide-key bitmap allocates, and this is
    // where a deferred sort comes due. end() stays noexcept - it reads nothing.
    [[nodiscard]] const_iterator begin() const {
        // Values come out in order, so the chunks must be in order first: a wide
        // chunk key defers the sort until a path needs it, and this is such a path.
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
        std::swap( size_, other.size_ );
    }

    // Defined here, not with the other large members: it is a thin forwarder to the out-of-line merge walk, and a
    // caller that hands it a fresh scratch bitmap lets the compiler drop the clearing of that scratch.
    [[ gnu::hot ]] void intersect_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates ) {
        if ( this == &other ) {
            scratch = *this;
            return;
        }

        scratch.clear_keep_capacity();

        // Materializing mode never mutates *this (the spine's in-place arms are
        // gated on out == nullptr), so the const_cast is only formal.
        const_cast<bitmap*>(this)->template combine<detail::set_operation::bit_and, false>( other, &scratch );
    }

    [[ gnu::hot ]] void union_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] void difference_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates );

    // Unites `right_container` into `left_container`, the containers of one key (the result replaces
    // the left one). A shared left payload is not cloned and then mutated: the union is written
    // straight into a fresh (or retired, from `store`) payload, as CRoaring's container_or does for a
    // SHARED container: the clone is a full read+write of the payload that the materializing combine
    // never pays. Same rule as the general spine's mutate_left (combine()). When `lazy`, bitset
    // cardinalities are left stale (see union_merge).
    // inline: the interop engine, instantiated in a program that applies an operand the library does not hold, calls it
    // too, and an always_inline function whose body is not in the program cannot be called at all.
    [[ gnu::always_inline ]] inline void union_same_key(
        detail::chunk_store<layout_type, CowPolicy> & store,
        handle_type & left_container,
        handle_type const & right_container,
        bool const lazy,
        detail::small_array_values<low_type> & array_union_scratch
    );

    // `|=` of a bitmap with far fewer chunks than this one: each of the other's keys is found by
    // galloping through our chunk table, the containers of the keys we hold are united in place, and
    // the keys we lack are inserted into the table in one go (see chunk_store::insert_absent_from),
    // where the merge would visit and move every chunk we hold.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_inplace_union_few
    template <bool Lazy>
    [[ gnu::hot ]] bitmap & union_few( bitmap const & other );

    // Shared union merge. When `lazy` is true, same-key bitset×bitset chunks are
    // OR'd without recomputing cardinality and `size_` is left invalid — the
    // caller MUST call repair_cardinality() afterwards. The N-way bulk-union path
    // uses this so the per-bitset popcount runs once at the end instead of K
    // times (CRoaring lazy-OR + repair). LTO folds `lazy` at each call site, so
    // operator|= keeps its eager codegen. The container_size(rhs)!=0 empty-skip
    // checks are kept in both modes — they read operand cardinalities, which are
    // always valid (only the accumulator's bitsets are lazily dirtied).
    template <bool Lazy>
    [[ gnu::hot ]] bitmap & union_merge( bitmap const & other );

    [[ gnu::hot ]] bitmap & operator|=( bitmap const & other ) noexcept( detail::allocation_failure_terminates );

    // Recomputes every chunk's cardinality when a nocard lazy word kernel left it
    // stale, and rebuilds size_. Card-updating bulk OR clears the stale flag, so a
    // dense K-way union finishes by summing headers instead of re-scanning every
    // saturated 8 KB bitset (CRoaring's repair is cheap on the same shape because
    // TO_FULL converted those to runs — we keep bitsets but skip the redundant
    // popcount when the header is already live).
    void repair_cardinality() noexcept;

    // The state a lazy fold leaves: size_ and the counts of the bitsets it dirtied are not known. size_ then
    // holds a value no size can reach (the top bit; the half below it absorbs what operations subtract), so that an operation which adds to it or subtracts from it keeps it
    // recognizable, and empty() is false whatever the fold held. repair_cardinality() ends it.
    static constexpr size_type size_deferred{ size_type{ 1 } << ( std::numeric_limits<size_type>::digits - 1 ) };
    [[nodiscard]] bool counts_deferred() const noexcept { return size_ >= size_deferred / 2; }
    // The entry of an in-place operation that reads container counts: ends a deferred state first.
    void settle_counts() noexcept { if ( counts_deferred() ) [[unlikely]] { repair_cardinality(); } }

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
    template <detail::set_operation Op, bool InPlace>
    [[ gnu::hot ]] bitmap & combine( bitmap const & other, bitmap * const out );

    [[ gnu::hot ]] bitmap & operator&=( bitmap const & other ) noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] bitmap & operator-=( bitmap const & other ) noexcept( detail::allocation_failure_terminates );

    bitmap & operator+=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return ( *this |= other ); }

    bitmap & operator^=( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return xor_merge( other ); }

    // In-place N-way bulk-union step (mirrors CRoaring roaring_bitmap_lazy_or_inplace):
    // OR `other` into our existing chunk vector WITHOUT rebuilding it — same-key
    // containers are OR'd in place, this-only keys stay untouched (no move), other-only
    // keys are spliced in, and the tail is appended. cardinality is left STALE (the caller
    // must repair_cardinality()). This avoids the per-fold chunk-store allocation + full
    // chunk-move that union_merge(…, /*lazy*/true) incurs on every one of the K folds — the
    // profiler-confirmed core-bound / per-call-alloc cost that dominated the N-way union.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_lazy_or_inplace
    [[ gnu::hot ]] void bulk_or_inplace( bitmap const & other ) noexcept( detail::allocation_failure_terminates );

    // Lazy accumulation step: OR `other` in, deferring the per-bitset popcount.
    // The bitmap is then deferred until a finisher counts it. Meanwhile further folds, every in-place update
    // (add, remove, ranges, add_many/remove_many, the set-operation assignments and *_with, optimize*) and contains
    // are allowed and leave a correct bitmap; anything that needs a size or a count (size, iteration,
    // cardinalities, equals, serialization, using it as an operand) needs the finished bitmap. docs/lazy-folds.md.
    [[ gnu::hot ]] void bulk_or_intermediate( bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { bulk_or_inplace( other ); }
    // Reshapes with the bitmap's AMBIENT RunSelectionPolicy, not a forced eager
    // one: this finishes an ordinary (lazy-accumulated) merge result, so it must
    // NOT reintroduce eager run-selection under run_selection_lazy — that's
    // exactly the "merge results" case this policy targets. Deliberately does not
    // call the public optimize() (which forces run_selection_eager, mirroring
    // CRoaring's explicit run_optimize() and must stay caller-requested-only).
    void bulk_or_finish() noexcept( detail::allocation_failure_terminates );
    // Forces the eager run re-decision, exactly as bulk_or_repair_after_lazy()
    // does for its run slots: a lazily accumulated merge can carry run containers
    // in verbatim from its operands, and under the ambient lazy policy a run can
    // never be re-chosen, so every such container would be materialized out and
    // re-encoded on the caller's next optimize().
    void bulk_or_finish_keep_bitsets() noexcept( detail::allocation_failure_terminates );

    // Exact CRoaring repair_after_lazy contract: compute deferred bitset
    // cardinalities, demote bitsets at/below the array ceiling, re-decide only
    // pre-existing run containers, and leave arrays untouched. This is narrower
    // than bulk_or_finish(), which applies the ambient storage policy to every
    // slot and needlessly scans dense bitsets for run encoding.
    void bulk_or_repair_after_lazy() noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] void or_many_in_place( std::span<bitmap const * const> const others ) noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] void or_many_sorted_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates );

    void or_many_heap_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates );

    // ---- exclusive-or (symmetric difference) --------------------------------------------------------------
    //
    // Unlike the other set operations the result can lose a chunk the operands hold (two equal containers cancel)
    // and gains the chunks only the other operand holds, so the walks are the union's with empty results dropped.
    // A result is an array up to array_to_bitset_threshold values and a bitset above, as the reference has it.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_xor, roaring_bitmap_xor_inplace

    [[nodiscard]] static handle_type xor_result_for_policy( handle_type container );

    [[nodiscard]] static handle_type xor_containers_for_policy( handle_type const & lhs, handle_type const & rhs );

    // A bitset result in its final form (see detail::settle_xor_bitset); the vacated payload of a demotion is
    // parked in `store` for the next result to take. Parking is only a reuse of the memory: a table that cannot
    // grow to hold it frees it instead, so the demotion that has been made is not undone by the failure.
    [[nodiscard]] static handle_type settle_xor_bitset( handle_type && container, detail::chunk_store<layout_type, CowPolicy> & store );

    // `lhs ^ rhs` of the containers of one key, as a new container; neither operand is touched.
    [[nodiscard]] static handle_type xor_pair( detail::chunk_store<layout_type, CowPolicy> & store, handle_type const & lhs, handle_type const & rhs );

    // `left ^= right` of the containers of one key. A payload nobody else holds is changed where it stands when
    // that is the cheaper form: a bitset takes a bitset, the values of an array or the runs of a run container
    // by flipping the bits in its own words, with the cardinality kept exact. Everything else is computed into
    // a new container.
    // [croaring-ref] deps/croaring/src/containers/containers.c:container_ixor
    // inline: the interop engine, instantiated in a program that applies an operand the library does not hold, calls it
    // too, and an always_inline function whose body is not in the program cannot be called at all.
    [[ gnu::always_inline ]] inline void xor_same_key(
        detail::chunk_store<layout_type, CowPolicy> & store,
        handle_type & left_container,
        handle_type const & right_container
    );

    // `this ^ other` into `scratch` (whose previous contents are discarded; its payloads are reused).
    [[ gnu::hot ]] void xor_into( bitmap const & other, bitmap & scratch ) const noexcept( detail::allocation_failure_terminates );

    // Keeps `size_` equal to the sum of the containers' sizes across an exclusive-or of one container, whether it
    // returns or throws (a container is replaced only by a finished result, so one that is left is whole).
    struct size_reconciliation {
        size_type &          total;
        handle_type const &  container;
        size_type            before;
        size_reconciliation( size_type & total_, handle_type const & container_ ) noexcept
            : total{ total_ }, container{ container_ }, before{ detail::container_size( container_ ) } {}
        size_reconciliation( size_reconciliation const & ) = delete;
        ~size_reconciliation() { total = total - before + detail::container_size( container ); }
    };

    // The containers of shared keys are exclusive-ored in place while the walk finds no key the other bitmap
    // holds and this one lacks, and no container that cancels out; from the first of either the rest of the
    // table is rebuilt. If an allocation fails the bitmap is left valid but with unspecified contents (the
    // containers the walk had already combined are changed, and once the table is being rebuilt it is emptied).
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_xor_inplace
    [[ gnu::hot ]] bitmap & xor_merge( bitmap const & other );

    // How many keys are in exactly one of the two bitmaps.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_xor_cardinality
    [[nodiscard]] size_type xor_cardinality( bitmap const & other ) const { return size_ + other.size_ - 2 * and_cardinality( other ); }

    // One fold of xor_many_in_place: `other` is exclusive-ored into the chunk table without rebuilding
    // it (shared keys in place, keys only this holds untouched, keys only `other` holds spliced in), and
    // the cardinalities of the bitsets it dirties are left stale for bulk_xor_repair_after_lazy().
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_lazy_xor_inplace
    [[ gnu::hot ]] void bulk_xor_intermediate( bitmap const & other ) noexcept( detail::allocation_failure_terminates );

    // `left ^= right` of the containers of one key, leaving a bitset's cardinality stale where that saves its
    // count: a bitset against a bitset, an array or runs; two arrays that grow past what an array merge is worth
    // become a bitset, as does an array against a bitset. Anything else is computed whole.
    // [croaring-ref] deps/croaring/src/containers/containers.c:container_lazy_ixor
    void lazy_xor_same_key(
        detail::chunk_store<layout_type, CowPolicy> & store,
        handle_type & left_container,
        handle_type const & right_container
    );

    // Puts the bitmap right after a fold of bulk_xor_intermediate() steps: the deferred bitset cardinalities are
    // counted, bitsets that fell to the array limit become arrays, run containers get their form decided again,
    // and the containers that cancelled out are dropped.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_repair_after_lazy
    void bulk_xor_repair_after_lazy() noexcept( detail::allocation_failure_terminates );

    // `this ^= *others[ 0 ] ^= ...`: the fold of the exclusive-or over many bitmaps, which defers the
    // bitsets' cardinality counts to the end. Null entries are skipped.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_xor_many
    [[ gnu::hot ]] void xor_many_in_place( std::span<bitmap const * const> const others ) noexcept( detail::allocation_failure_terminates );

    [[nodiscard]] static bitmap xor_many( std::span<bitmap const * const> const operands ) noexcept( detail::allocation_failure_terminates );

    [[ gnu::hot ]] void and_many_sorted_in_place( std::span<bitmap const *> others ) noexcept( detail::allocation_failure_terminates );

    // ---- n-way AND ------------------------------------------------------------------------------
    //
    // The AND of N bitmaps, answered one chunk at a time. Operands are only read: never reordered,
    // copied or mutated, and neither is the span that names them. The walk writes nothing the
    // operands own, whatever the key width (no reference count, container header or chunk table), so
    // any number of threads may walk the same operands at once, and a result never keeps one of
    // their containers. Operands must be finalized, i.e. not in the middle of a bulk_or_intermediate
    // fold: the walk reads cardinalities straight from the container headers.
    //
    // The chunk tables are walked together, leapfrogging from the operand with the fewest chunks, so
    // a chunk any operand lacks costs a gallop and no container is opened. Where every operand has
    // the chunk its containers meet in ascending order of their exact cardinalities - read from the
    // headers, so the ordering is free and involves no estimate - which makes the accumulator start
    // as the smallest container and only ever shrink, runs the cheap array probes before the dense
    // word-wise steps, and ends the chunk at the first empty accumulator. The accumulator lives in a
    // few scratch containers; no intermediate bitmap or chunk table exists.
    struct and_many_scratch {
        handle_type array_acc;
        handle_type array_alt;
        handle_type bitset_acc;
        handle_type bitset_alt;
    };

    // The values common to every operand, ascending, handed over in blocks. A block is a run of
    // consecutive result values inside one chunk: the first 256 values of the chunk's result, the next
    // 256, and so on, the last one shorter. The cut points are a function of the result SET alone, not
    // of the container kinds it passed through, so a reduction written inside f groups its operands the
    // same way whatever the result's representation - which a per-value callback cannot promise once
    // the compiler reassociates a floating-point sum across the loop around it.
    template <typename F>
    static void and_many_for_each_block( std::span<bitmap const * const> const operands, F && f, and_many_scratch & scratch ) {
        and_many_walk( operands, [ & ]( chunk_type const key, std::span<handle_type const * const> const ordered ) {
            bool owned{ false };
            if ( auto const * const acc{ and_chunk( ordered, scratch, owned ) } ) {
                and_many_emit_blocks( *acc, key, f );
            }
            return true;
        } );
    }
    template <typename F>
    static void and_many_for_each_block( std::span<bitmap const * const> const operands, F && f ) {
        and_many_scratch scratch;
        and_many_for_each_block( operands, std::forward<F>( f ), scratch );
    }

    // One call per common value, in ascending order. The grouping of a floating-point reduction fused
    // into f is unspecified; use and_many_for_each_block where it matters.
    template <typename F>
    static void and_many_for_each( std::span<bitmap const * const> const operands, F && f ) {
        and_many_for_each_block( operands, [ & ]( std::span<key_type const> const block ) {
            for ( auto const value : block ) { f( value ); }
        } );
    }

    [[nodiscard]] static size_type and_many_cardinality( std::span<bitmap const * const> const operands );

    // Whether the operands have any value in common. Stops at the first chunk that does: the last
    // container of that chunk is only tested against the accumulator, not combined with it.
    [[nodiscard]] static bool and_many_intersects( std::span<bitmap const * const> const operands );
    [[nodiscard]] static bool and_many_intersects( std::span<bitmap const * const> const operands, bitmap const & other );

    // The AND as a bitmap. A result bitset that fell to array size is stored as an array.
    [[nodiscard]] static bitmap and_many( std::span<bitmap const * const> const operands );

    // A copy of the viewed bitmap that shares its payloads without counting them: each spilled container of the copy
    // is a borrowed alias of the viewed one (a small one is copied by value), so making the copy writes nothing to the
    // viewed bitmap, where a copy of a bitmap under a counting policy writes one reference count per container. The first
    // update of a container of the copy clones it into private storage, so the copy is as mutable as any bitmap and the
    // viewed bitmap is never changed by it.
    // Unlike everything else that reads through a view, the result DOES refer to the viewed bitmap, and so does every bitmap
    // that takes a container of it by copying its handle: a copy of it (`bitmap{ copy }`, which stays borrowed under every
    // policy), an assignment from it, and the receiver of `x |= copy` or `x ^= copy` (the containers x lacks) and of
    // `x &= copy` (a full chunk), as a bitmap served from a mapped file does (see persistent_bitmap). All of them must be
    // dropped, or have every container written once, before the viewed bitmap is changed or destroyed. Through a VIEW of the
    // copy nothing is adopted: `x |= view_of( copy )` is a copy of the viewed contents that outlives the master, as is any
    // other operation that takes a view.
    // The cost is one pass over the chunk table, a 32-byte handle per container and no payload copy.
    [[nodiscard]] static bitmap borrowed_copy( typed_view<bitmap> const view ) noexcept( detail::allocation_failure_terminates );

    // The AND written into `out`, which is cleared first (its chunk table keeps its capacity). `out` is none of the operands.
    static void and_many_into( std::span<bitmap const * const> const operands, bitmap & out ) noexcept( detail::allocation_failure_terminates );

    // The same over views of finished bitmaps (see typed_view): the walk is the one above, it writes nothing the
    // viewed bitmaps own, and the result keeps none of their containers.
    template <typename F>
    static void and_many_for_each_block( std::span<typed_view<bitmap> const> const views, F && f, and_many_scratch & scratch ) {
        with_view_pointers( views, [ & ]( auto const operands ) { and_many_for_each_block( operands, std::forward<F>( f ), scratch ); } );
    }
    template <typename F>
    static void and_many_for_each_block( std::span<typed_view<bitmap> const> const views, F && f ) {
        with_view_pointers( views, [ & ]( auto const operands ) { and_many_for_each_block( operands, std::forward<F>( f ) ); } );
    }
    template <typename F>
    static void and_many_for_each( std::span<typed_view<bitmap> const> const views, F && f ) {
        with_view_pointers( views, [ & ]( auto const operands ) { and_many_for_each( operands, std::forward<F>( f ) ); } );
    }
    [[nodiscard]] static size_type and_many_cardinality( std::span<typed_view<bitmap> const> const views ) {
        return with_view_pointers( views, []( auto const operands ) { return and_many_cardinality( operands ); } );
    }
    [[nodiscard]] static bool and_many_intersects( std::span<typed_view<bitmap> const> const views ) {
        return with_view_pointers( views, []( auto const operands ) { return and_many_intersects( operands ); } );
    }
    [[nodiscard]] static bitmap and_many( std::span<typed_view<bitmap> const> const views ) noexcept( detail::allocation_failure_terminates ) {
        return with_view_pointers( views, []( auto const operands ) { return and_many( operands ); } );
    }
    static void and_many_into( std::span<typed_view<bitmap> const> const views, bitmap & out ) noexcept( detail::allocation_failure_terminates ) {
        with_view_pointers( views, [ & ]( auto const operands ) { and_many_into( operands, out ); } );
    }


    // `*this & view` into `out` (cleared first, its parked payloads reused), as intersect_into does for a bitmap, without sharing a
    // container of the view or of this bitmap: only the chunks both hold are visited, and what is kept is built in `out`.
    void intersect_into( typed_view<bitmap> const view, bitmap & out ) const noexcept( detail::allocation_failure_terminates );
    // `*this - view` into `out`, built the same way: only the chunks that are kept are built.
    void difference_into( typed_view<bitmap> const view, bitmap & out ) const noexcept( detail::allocation_failure_terminates );

private:
    static constexpr std::size_t and_many_inline_operands{ 16 };
    static constexpr std::size_t and_many_block_values{ 256 };

    template <typename F>
    static decltype( auto ) with_view_pointers( std::span<typed_view<bitmap> const> const views, F && f ) {
        and_many_buffer<bitmap const *> pointers{ views.size() };
        for ( std::size_t i{ 0 }; i != views.size(); ++i ) { pointers.data[ i ] = &views[ i ].get(); }
        return f( std::span<bitmap const * const>{ pointers.data, views.size() } );
    }

    template <typename T>
    struct and_many_buffer {
        std::array<T, and_many_inline_operands> inline_;
        std::unique_ptr<T[]>                    heap_;
        T *                                     data;
        explicit and_many_buffer( std::size_t const n )
            : data{ n <= inline_.size() ? inline_.data() : ( heap_ = std::make_unique<T[]>( n ), heap_.get() ) } {}
    };

    // Calls chunk( key, handles ) for every chunk key every operand holds a live container for, in
    // ascending key order, with `handles` ordered by ascending container cardinality. chunk returns
    // false to stop the walk.
    template <typename Chunk>
    static void and_many_walk( std::span<bitmap const * const> const operands, Chunk && chunk ) {
        std::size_t count{ 0 };
        for ( auto const * const operand : operands ) {
            count += operand != nullptr;
        }
        if ( count == 0 ) {
            return;
        }
        and_many_buffer<bitmap const *>      ops    { count };
        and_many_buffer<chunk_cursor>         cursors{ count };
        and_many_buffer<handle_type const *> handles{ count };
        count = 0;
        for ( auto const * const operand : operands ) {
            if ( operand == nullptr ) {
                continue;
            }
            if ( operand->chunks_.size() == 0 ) {
                return;
            }
            ops.data[ count ]     = operand;
            cursors.data[ count ] = operand->chunks_.begin();
            ++count;
        }

        // The driver has the fewest chunks; only keys inside every operand's key range can match.
        std::size_t driver{ 0 };
        auto lowest { std::numeric_limits<chunk_type>::lowest() };
        auto highest{ std::numeric_limits<chunk_type>::max() };
        for ( std::size_t j{ 0 }; j < count; ++j ) {
            auto const & chunks{ ops.data[ j ]->chunks_ };
            if ( chunks.size() < ops.data[ driver ]->chunks_.size() ) { driver = j; }
            lowest  = std::max( lowest , chunks.key( chunks.begin() ) );
            highest = std::min( highest, chunks.key( chunks.last() ) );
        }
        if ( highest < lowest ) {
            return;
        }

        auto const & driving{ *ops.data[ driver ] };
        auto at{ driving.chunks_.lower_bound( lowest, driving.chunks_.begin() ) };
        while ( at != driving.chunks_.end() ) {
            auto const key{ driving.chunks_.key( at ) };
            if ( highest < key ) {
                break;
            }
            bool all_present{ true };
            for ( std::size_t j{ 0 }; j < count; ++j ) {
                auto const & bm{ *ops.data[ j ] };
                auto const   found_at{ bm.chunks_.lower_bound( key, cursors.data[ j ] ) };
                cursors.data[ j ] = found_at;
                if ( found_at == bm.chunks_.end() ) {
                    return;  // nothing at or above this key in this operand: nothing further can intersect
                }
                auto const found{ bm.chunks_.key( found_at ) };
                if ( found != key ) {
                    // leapfrog: the driver jumps to the operand's next key
                    at = driving.chunks_.lower_bound( found, at );
                    all_present = false;
                    break;
                }
                handles.data[ j ] = &bm.chunks_.slot( found_at );
            }
            if ( !all_present ) {
                continue;
            }
            for ( std::size_t i{ 1 }; i < count; ++i ) {  // ascending cardinality, insertion sort
                auto const * const moving{ handles.data[ i ] };
                auto j{ i };
                for ( ; j > 0 && moving->cardinality() < handles.data[ j - 1 ]->cardinality(); --j ) {
                    handles.data[ j ] = handles.data[ j - 1 ];
                }
                handles.data[ j ] = moving;
            }
            if ( !chunk( key, std::span<handle_type const * const>{ handles.data, count } ) ) {
                return;
            }
            ++at;
        }
    }

    static void and_many_seed_bitset( and_many_scratch & scratch, handle_type const & source );

    // The AND of the containers of one chunk, ascending by cardinality, as a handle: a view of the only
    // container when there is just one operand, otherwise one of the scratch handles (owned = true).
    // nullptr when the intersection is empty.
    // |acc ∩ last| without storing it. Two arrays are the exception: their kernel is the materializing
    // one, so the intersection is written to scratch and measured there.
    [[nodiscard]] static size_type and_many_count_pair( handle_type const & acc, handle_type const & last, and_many_scratch & scratch );

    [[nodiscard]] static handle_type const * and_chunk( std::span<handle_type const * const> const ordered, and_many_scratch & scratch, bool & owned );

    // Hands the values of an accumulator over in blocks of and_many_block_values, cut by count from the
    // start of the chunk. A bitset is decoded a slice of words at a time with the library's vector
    // extractor and the surplus over whole blocks is carried to the front of the buffer; a scalar
    // one-bit-at-a-time loop here would cost more than building and decoding an array result.
    template <typename F>
    static void and_many_emit_blocks( handle_type const & acc, chunk_type const chunk, F & f ) {
        constexpr std::size_t block{ and_many_block_values };
        constexpr bool        vector_decode{ std::is_same_v<key_type, std::uint32_t> };
        constexpr std::size_t slice_words{ vector_decode ? 64U : 4U };
        constexpr std::size_t capacity{ slice_words * 64U + block + 64U };  // a slice's values, the carry, and the extractor's slack
        auto const base{ layout_type::compose( chunk, 0 ) };
        std::array<key_type, capacity> buffer;
        if ( acc.holds_array() ) {
            auto const & values{ acc.as_array().values };
            auto const   n{ static_cast<std::size_t>( values.size() ) };
            for ( std::size_t at{ 0 }; at < n; at += block ) {
                auto const m{ std::min( block, n - at ) };
                for ( std::size_t j{ 0 }; j < m; ++j ) {
                    buffer[ j ] = base + static_cast<key_type>( values[ at + j ] );
                }
                f( std::span<key_type const>{ buffer.data(), m } );
            }
        } else if ( acc.holds_run() ) {
            // only a lone operand's own container reaches here: the walk expands a run seed
            std::size_t fill{ 0 };
            for ( auto const & run : acc.as_run().runs ) {
                auto value    { static_cast<std::uint32_t>( run.begin ) };
                auto remaining{ static_cast<std::size_t>( run.end - run.begin ) + 1U };
                while ( remaining != 0 ) {
                    auto const n{ std::min( remaining, block - fill ) };
                    for ( std::size_t j{ 0 }; j < n; ++j ) {
                        buffer[ fill + j ] = base + static_cast<key_type>( value + j );
                    }
                    fill      += n;
                    value     += static_cast<std::uint32_t>( n );
                    remaining -= n;
                    if ( fill == block ) {
                        f( std::span<key_type const>{ buffer.data(), block } );
                        fill = 0;
                    }
                }
            }
            if ( fill != 0 ) {
                f( std::span<key_type const>{ buffer.data(), fill } );
            }
        } else {
            auto const & words{ acc.as_bitset().words.as_array() };
            std::size_t fill{ 0 };
            for ( std::size_t w{ 0 }; w < layout_type::word_count; w += slice_words ) {
                auto const slice_base{ base + static_cast<key_type>( w << 6U ) };
                key_type * out{ buffer.data() + fill };
                if constexpr ( vector_decode ) {
                    out = detail::extract_setbits_uint32( words.data() + w, slice_words, out, buffer.data() + capacity, slice_base );
                } else {
                    for ( std::size_t k{ 0 }; k < slice_words; ++k ) {
                        auto word{ words[ w + k ] };
                        auto const word_base{ slice_base + static_cast<key_type>( k << 6U ) };
                        while ( word != 0 ) {
                            *out++ = word_base + static_cast<key_type>( std::countr_zero( word ) );
                            word &= word - 1U;
                        }
                    }
                }
                fill = static_cast<std::size_t>( out - buffer.data() );
                std::size_t at{ 0 };
                for ( ; fill - at >= block; at += block ) {
                    f( std::span<key_type const>{ buffer.data() + at, block } );
                }
                if ( at != 0 ) {
                    std::copy( buffer.begin() + at, buffer.begin() + fill, buffer.begin() );
                    fill -= at;
                }
            }
            if ( fill != 0 ) {
                f( std::span<key_type const>{ buffer.data(), fill } );
            }
        }
    }

public:

    // Gives back every byte the bitmap holds beyond its contents: each container's
    // payload slack, the parked scratch payloads, and the chunk arrays' spare
    // capacity. Returns the bytes freed. Explicit and cold — the growth slack and
    // the scratch pool are what make the mutation paths cheap, so this belongs
    // next to a caller-requested compaction (optimize()) or before a long idle
    // period, never inside a set operation.
    // [croaring-ref] deps/croaring/src/roaring.c:roaring_bitmap_shrink_to_fit
    [[ gnu::cold ]] std::size_t shrink_to_fit() noexcept( detail::allocation_failure_terminates );

    void promote_large_arrays( std::uint16_t const threshold = static_cast<std::uint16_t>( array_to_bitset_threshold ) ) noexcept( detail::allocation_failure_terminates );

    // Explicit, caller-requested compaction — the CRoaring run_optimize() analog.
    // Always reconsiders run-encoding (forces run_selection_eager) regardless of
    // the bitmap's ambient RunSelectionPolicy: under run_selection_lazy, ordinary
    // insert/merge never auto-picks run encoding, but this call still can, on
    // request, exactly like CRoaring never auto-run-encodes except via an
    // explicit run_optimize() call.
    void optimize() noexcept( detail::allocation_failure_terminates );

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
    void optimize_keep_bitsets() noexcept( detail::allocation_failure_terminates );

    // Selective run-encoding: convert exactly the containers whose mean run length
    // (cardinality / number of maximal runs) is at least `min_mean_run_length`, and
    // touch nothing else — no representation re-decision for the other containers
    // and no payload access beyond a read-only run count (an array is scanned in
    // place, a bitset is counted from its words, both with the count capped at the
    // first value that already fails the test). Under a refcounted CowPolicy a
    // shared payload is therefore never cloned unless it is the one being replaced.
    // Run containers already satisfy the test and are left as they are.
    //
    // This is the maintenance pass for data whose ordinary mutations degrade run
    // shapes over time: unlike optimize()/optimize_keep_bitsets(), which reconsider
    // every container by serialized size, only strongly run-shaped containers move
    // — a marginal run container costs more in consumers' run×array/run×bitset
    // kernels than it saves in bytes, so the caller sets the bar.
    void optimize_run_shaped( std::size_t const min_mean_run_length ) noexcept( detail::allocation_failure_terminates );

    void optimize_for_storage() noexcept( detail::allocation_failure_terminates ) { optimize(); }
    void optimize_for_speed( std::uint16_t const threshold = static_cast<std::uint16_t>( array_to_bitset_threshold ) ) noexcept( detail::allocation_failure_terminates );
    [[nodiscard]] inverse_proxy operator!() const noexcept { return inverse_proxy{ *this }; }

    [[nodiscard]] friend bitmap operator|( bitmap const & lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.union_into( rhs, result );
        return result;
    }

    [[nodiscard]] friend bitmap operator+( bitmap const & lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        return lhs | rhs;
    }

    [[nodiscard]] friend bitmap operator&( bitmap const & lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.intersect_into( rhs, result );
        return result;
    }

    // Against a bitmap or a view, materializes straight into the result, exactly as the operator- below
    // does: only the chunks that are kept are built. Against any other operand it copies lhs and subtracts
    // from the copy, which under a refcounting CowPolicy writes one count per container of lhs and then
    // clones every payload the in-place difference touches, because the copy left them shared.
    template <complementable Operand>
    [[nodiscard]] friend bitmap operator&( bitmap const & lhs, complement<Operand> const & rhs ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) {
        if constexpr ( std::same_as<Operand, bitmap> ) {
            bitmap result;
            lhs.difference_into( rhs.operand, result );
            return result;
        } else if constexpr ( std::same_as<Operand, typed_view<bitmap>> ) {
            bitmap result;
            lhs.difference_into( rhs.operand, result );
            return result;
        } else {
            bitmap result{ lhs };
            result.andnot_with( rhs.operand );
            return result;
        }
    }
    template <complementable Operand>
    [[nodiscard]] friend bitmap operator&( complement<Operand> const & lhs, bitmap const & rhs ) noexcept( noexcept( rhs & lhs ) ) { return rhs & lhs; }
    template <complementable Operand>
    [[nodiscard]] friend bitmap operator&( typed_view<bitmap> const lhs, complement<Operand> const & rhs ) noexcept( detail::allocation_failure_terminates && detail::complement_mutation_nothrow_v<Operand> ) {
        bitmap result;
        if constexpr ( std::same_as<Operand, bitmap> || std::same_as<Operand, typed_view<bitmap>> ) {
            lhs.get().difference_into( typed_view<bitmap>{ rhs.operand }, result );
        } else {
            result |= lhs;
            result.andnot_with( rhs.operand );
        }
        return result;
    }
    template <complementable Operand>
    [[nodiscard]] friend bitmap operator&( complement<Operand> const & lhs, typed_view<bitmap> const rhs ) noexcept( noexcept( rhs & lhs ) ) { return rhs & lhs; }

    [[nodiscard]] friend bitmap operator-( bitmap const & lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.difference_into( rhs, result );
        return result;
    }

    [[nodiscard]] friend bitmap operator^( bitmap const & lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.xor_into( rhs, result );
        return result;
    }

    // The set operations with a view on either side, materialized like the ones between bitmaps. Nothing of a
    // view is shared with the result.
    [[nodiscard]] friend bitmap operator&( typed_view<bitmap> const lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.get().intersect_into( rhs, result );
        return result;
    }
    [[nodiscard]] friend bitmap operator&( bitmap const & lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.intersect_into( rhs, result );
        return result;
    }
    [[nodiscard]] friend bitmap operator&( typed_view<bitmap> const lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) { return rhs & lhs; }
    [[nodiscard]] friend bitmap operator|( typed_view<bitmap> const lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        result |= lhs;
        result |= rhs;
        return result;
    }
    [[nodiscard]] friend bitmap operator|( bitmap const & lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result{ lhs };
        result |= rhs;
        return result;
    }
    [[nodiscard]] friend bitmap operator|( typed_view<bitmap> const lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) { return rhs | lhs; }
    [[nodiscard]] friend bitmap operator^( typed_view<bitmap> const lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        result |= lhs;
        result ^= rhs;
        return result;
    }
    [[nodiscard]] friend bitmap operator^( bitmap const & lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result{ lhs };
        result ^= rhs;
        return result;
    }
    [[nodiscard]] friend bitmap operator^( typed_view<bitmap> const lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) { return rhs ^ lhs; }
    [[nodiscard]] friend bitmap operator-( typed_view<bitmap> const lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.get().difference_into( rhs, result );
        return result;
    }
    [[nodiscard]] friend bitmap operator-( bitmap const & lhs, typed_view<bitmap> const rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.difference_into( rhs, result );
        return result;
    }
    [[nodiscard]] friend bitmap operator-( typed_view<bitmap> const lhs, bitmap const & rhs ) noexcept( detail::allocation_failure_terminates ) {
        bitmap result;
        lhs.get().difference_into( view_of( rhs ), result );
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

    // The bytes of the rank block that follows a payload in a ranked image: a u16 per word of a bitset (the values in the
    // earlier words) and a u16 per run of a run container (the values in the earlier runs); none for an array, whose
    // position is its rank.
    [[nodiscard]] static constexpr std::size_t frozen_rank_bytes( frozen_container_kind const kind, std::size_t const payload_count ) noexcept {
        switch ( kind ) {
            case frozen_container_kind::array : return 0U;
            case frozen_container_kind::run   : return payload_count * sizeof( std::uint16_t );
            case frozen_container_kind::bitset: return static_cast<std::size_t>( layout_type::word_count ) * sizeof( std::uint16_t );
        }
        return 0U;
    }

    [[nodiscard]] static std::size_t frozen_rank_bytes( handle_type const & container ) noexcept {
        switch ( container.kind() ) {
            case detail::container_kind::array : return 0U;
            case detail::container_kind::run   : return frozen_rank_bytes( frozen_container_kind::run, container.count() );
            case detail::container_kind::bitset: return frozen_rank_bytes( frozen_container_kind::bitset, layout_type::word_count );
        }
        std::unreachable();
    }

    [[nodiscard]] static constexpr detail::container_kind container_kind_of( frozen_container_kind const kind ) noexcept {
        switch ( kind ) {
            case frozen_container_kind::array : return detail::container_kind::array ;
            case frozen_container_kind::run   : return detail::container_kind::run   ;
            case frozen_container_kind::bitset: return detail::container_kind::bitset;
        }
        std::unreachable();
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

    // ---- portable (standard Roaring) format helpers; defined in serialization.hpp ----
    struct portable_plan {
        std::uint32_t containers{};   // non-empty containers in the range
        bool          has_run{};
        std::size_t   header_bytes{};
        std::size_t   payload_bytes{};
    };
    [[nodiscard]] portable_plan portable_plan_for( chunk_cursor first, chunk_cursor last ) const noexcept;
    std::byte * portable_write_32( chunk_cursor first, chunk_cursor last, std::uint32_t count, std::byte * out ) const noexcept;
    [[nodiscard]] static bool portable_read_32(
        std::byte const * & cursor, std::byte const * end, chunk_type high_bits, bitmap & result, std::size_t & total
    );


    // lower_bound() out of line, for call sites whose fast path is something else
    // (a bulk context hit): keeps the search loop out of their hot code.
    [[nodiscard, gnu::noinline]] chunk_cursor lower_bound_outlined( chunk_type const chunk ) const noexcept {
        return chunks_.lower_bound( chunk );
    }

    // Whether the entry next to (or before) `at` holds `chunk`: the sequential-access
    // probe of a bulk context. On success `at` moves onto it. `at` must not be end().
    [[nodiscard]] bool try_adjacent_chunk( chunk_type const chunk, chunk_cursor & at ) const noexcept {
        auto const current_key{ chunks_.key( at ) };
        if ( chunk > current_key ) {
            auto following{ at };
            ++following;
            if ( following != chunks_.end() && chunks_.key( following ) == chunk ) {
                at = following;
                return true;
            }
        } else if ( chunk < current_key && at != chunks_.begin() ) {
            auto const before{ chunks_.prev( at ) };
            if ( chunks_.key( before ) == chunk ) {
                at = before;
                return true;
            }
        }
        return false;
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
    );

    // array ∪ array as a result container, decided on the operands' total the way
    // the reference decides it: a total below the bitset threshold bounds the
    // result below it too, so the union is merged straight into a retired (or
    // fresh) array payload — the kernel sizes that payload for its own worst case,
    // store slack included. A larger total may well be a bitset, so both arrays are
    // scattered straight into one (demoted to an array if the shared values leave
    // it below the limit). A retired payload is only ever a sole-owned one, so it
    // never aliases an operand the kernel is still reading.
    // [croaring-ref] deps/croaring/src/containers/mixed_union.c:array_array_container_union
    [[nodiscard]] static handle_type make_array_union_container_reusing(
        detail::chunk_store<layout_type, CowPolicy> & store,
        detail::array_cref<layout_type, CowPolicy> const lhs,
        detail::array_cref<layout_type, CowPolicy> const rhs,
        detail::small_array_values<low_type> & scratch
    );

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
    );

    [[nodiscard]] static handle_type combine_containers_for_policy(
        handle_type const & lhs,
        handle_type const & rhs,
        detail::set_operation const op
    );

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
    );

    // Policy wrapper for the combine spine's run∩bitset arm — same contract as
    // combine_bitset_bitset_for_policy above (identical minus the kernel).
    [[nodiscard]] static handle_type intersect_run_bitset_for_policy(
        detail::run_cref<layout_type, CowPolicy> const runs,
        detail::bitset_cref<layout_type, CowPolicy> const bitset,
        handle_type && reuse = {}
    );

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
                    return true;
                }
            }

            return false;
        }
    }

    // Calls process( chunk, lows ) for every chunk the ascending `sorted_values` touch, in chunk order, with the low
    // parts of that chunk's values, duplicates dropped, in a buffer that is reused from chunk to chunk.
    template <typename Process>
    static void for_each_chunk_group( std::span<key_type const> const sorted_values, Process && process ) {
        if ( sorted_values.empty() ) {
            return;
        }
        // A chunk holds at most 65536 distinct lows (one more slot takes the store of a duplicate).
        constexpr std::size_t max_lows{ std::size_t{ 1 } << 16 };
        detail::small_array_values<low_type> lows;
        auto capacity{ std::min( sorted_values.size(), std::size_t{ 2048 } ) + 1 };
        detail::resize_uninitialized( lows, capacity );
        auto *       out     { lows.data() };
        auto         chunk   { layout_type::chunk_key( sorted_values.front() ) };
        std::size_t  count   { 0 };
        auto         previous{ layout_type::low_key( sorted_values.front() ) };   // the value before the current one: ascending, so a duplicate equals it
        for ( auto const value : sorted_values ) {
            auto const value_chunk{ layout_type::chunk_key( value ) };
            auto const low        { layout_type::low_key  ( value ) };
            if ( value_chunk != chunk ) [[unlikely]] {
                process( chunk, std::span<low_type const>{ out, count } );
                chunk = value_chunk;
                count = 0;
            }
            if ( count + 1 == capacity ) [[unlikely]] {
                capacity = std::min( capacity * 2, max_lows + 1 );
                detail::resize_uninitialized( lows, capacity );
                out = lows.data();
            }
            out[ count ] = low;
            count       += ( count == 0 ) | ( low != previous );
            previous     = low;
        }
        process( chunk, std::span<low_type const>{ out, count } );
    }

    [[nodiscard]] bool add_impl( key_type const value, bulk_context * const ctx ) {
        auto const chunk{ layout_type::chunk_key( value ) };
        auto const low{ layout_type::low_key( value ) };

        // CRoaring-style: O(log n) binary search + sorted insert.
        auto at{ chunks_.lower_bound( chunk ) };
        if ( at == chunks_.end() || chunks_.key( at ) != chunk ) {
            at = chunks_.insert( at, chunk, handle_type{} );
        }

        if ( ctx != nullptr ) {
            ctx->chunk      = chunk;
            ctx->at         = at;
            ctx->slot       = &chunks_.slot( at );
            ctx->generation = chunks_.generation();
            // The add below makes this payload private (or finds it so), and no copy can
            // intervene before the next call on this context.
            ctx->payload_private = true;
        }

        if ( !detail::container_add( chunks_.slot( at ), low ) ) { return false; }

        ++size_;
        promote_if_needed( chunks_.slot( at ) );
        return true;
    }

    // The threshold test is the inline part (two loads on the handle line the
    // add just touched); the conversion itself is one cold call per 4096 adds.
    [[gnu::always_inline]] void promote_if_needed( handle_type & slot ) {
        if ( slot.holds_array() && slot.count() >= array_to_bitset_threshold ) [[unlikely]] {
            promote_array( slot );
        }
    }
    [[gnu::cold, gnu::noinline]] void promote_array( handle_type & slot ) {
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

    void apply_closed_range_operation(
        key_type const begin_value,
        key_type const end_value,
        detail::range_operation const operation
    );

    void apply_chunk_range_operation(
        chunk_type const chunk,
        low_type const low_begin,
        low_type const low_end,
        detail::range_operation const operation
    );

    // The interop operations against the operands the library holds compiled (see interop_operands.hpp) reach
    // these three through a type-erased view; any other operand is applied by the engine itself.
    void interop_mutate( detail::interop_op op, detail::operand_view const & operand ) noexcept( detail::allocation_failure_terminates );
    [[nodiscard]] size_type interop_count( detail::interop_op op, detail::operand_view const & operand ) const;
    [[nodiscard]] bool interop_test( detail::interop_op op, detail::operand_view const & operand ) const;

    // The number of keys of an operand whose size is known (the keys of an interval that the key type can hold).
    [[nodiscard]] static size_type operand_size( bitmap const & operand ) noexcept { return operand.size(); }
    [[nodiscard]] static size_type operand_size( typed_view<bitmap> const operand ) noexcept { return operand.size(); }
    [[nodiscard]] static size_type operand_size( key_range const operand ) noexcept {
        if ( !operand_interval_holds_keys( operand ) ) { return 0; }
        auto const count{ interval_keys_after_first( operand ) + 1 };
        assert( count != 0 && "an interval of every 64-bit key has no size" );
        return static_cast<size_type>( count );
    }

    // The sizes where an interval of every 64-bit key (which has none a size_type holds) is a legal operand: the comparison
    // with a count and the size as a double.
    [[nodiscard]] static bool operand_has_more_than( bitmap const & operand, size_type const count ) noexcept { return operand.size() > count; }
    [[nodiscard]] static bool operand_has_more_than( typed_view<bitmap> const operand, size_type const count ) noexcept { return operand.size() > count; }
    [[nodiscard]] static bool operand_has_more_than( key_range const operand, size_type const count ) noexcept {
        return operand_interval_holds_keys( operand ) && count <= interval_keys_after_first( operand );
    }
    [[nodiscard]] static double operand_size_as_double( bitmap const & operand ) noexcept { return static_cast<double>( operand.size() ); }
    [[nodiscard]] static double operand_size_as_double( typed_view<bitmap> const operand ) noexcept { return static_cast<double>( operand.size() ); }
    [[nodiscard]] static double operand_size_as_double( key_range const operand ) noexcept {
        return operand_interval_holds_keys( operand ) ? static_cast<double>( interval_keys_after_first( operand ) ) + 1.0 : 0.0;
    }
    [[nodiscard]] static bool operand_interval_holds_keys( key_range const operand ) noexcept {
        return !operand.empty() && operand.first <= static_cast<std::uint64_t>( std::numeric_limits<key_type>::max() );
    }
    // the keys of a non-empty interval, clipped to the key type, but the first (one less than its size, which cannot wrap)
    [[nodiscard]] static std::uint64_t interval_keys_after_first( key_range const operand ) noexcept {
        return std::min( operand.last, static_cast<std::uint64_t>( std::numeric_limits<key_type>::max() ) ) - operand.first;
    }

    template <detail::interop_op Op, typename Self, typename Operand>
    static decltype( auto ) interop_dispatch( Self & self, Operand const & operand ) {
        constexpr bool mutating{ Op == detail::interop_op::and_with || Op == detail::interop_op::or_with
                              || Op == detail::interop_op::andnot_with || Op == detail::interop_op::xor_with };
        if constexpr ( mutating ) {
            self.settle_counts();
        }
        if constexpr ( detail::is_typed_view_v<Operand> ) {
            static_assert( std::same_as<typename Operand::bitmap_type, bitmap>, "a view is an operand of bitmaps of its own instantiation" );
            if constexpr ( mutating ) {
                // the table that is read is the table that is rewritten: the answer is known without reading it
                if ( &operand.get() == &self ) {
                    if constexpr ( Op == detail::interop_op::andnot_with || Op == detail::interop_op::xor_with ) { self.clear(); }
                    return;
                }
            }
        }
        if constexpr ( detail::interop_erasable<Operand, bitmap> ) {
            auto const view{ detail::make_operand_view( operand ) };
            if constexpr ( Op == detail::interop_op::and_with || Op == detail::interop_op::or_with
                        || Op == detail::interop_op::andnot_with || Op == detail::interop_op::xor_with ) {
                self.interop_mutate( Op, view );
            } else if constexpr ( Op == detail::interop_op::and_cardinality || Op == detail::interop_op::xor_cardinality ) {
                return self.interop_count( Op, view );
            } else {
                return self.interop_test( Op, view );
            }
        } else {
            return detail::interop_engine<bitmap>::template run<Op>( self, operand );
        }
    }

    friend struct detail::interop_engine<bitmap>;
    friend struct detail::bitmap_test_access<bitmap>;
    template <typename> friend class typed_view;

    chunk_store_type chunks_;
    size_type size_{};
};

} // namespace frsr::roaring


#if defined( FRSR_ROARING_HEADER_ONLY ) || defined( FRSR_ROARING_WITH_DEFINITIONS )
#   include <frsr/roaring/bitmap_definitions.hpp>
#endif

#ifndef FRSR_ROARING_HEADER_ONLY

// The bitmaps the library holds compiled: every key width, copy-on-write policy and run selection policy
// over the default container set. A program that instantiates any other bitmap (another container set or
// policy) includes bitmap_definitions.hpp, or defines FRSR_ROARING_WITH_DEFINITIONS before this header.
#define FRSR_ROARING_EXTERN_BITMAP( KEY, COW, RUN_SELECTION ) \
    extern template class bitmap<KEY, default_container_set<KEY>, detail::COW, detail::RUN_SELECTION>; \
    extern template bitmap<KEY, default_container_set<KEY>, detail::COW, detail::RUN_SELECTION> & \
    bitmap<KEY, default_container_set<KEY>, detail::COW, detail::RUN_SELECTION>::combine<detail::set_operation::bit_and, false>( \
        bitmap const &, bitmap * );
#define FRSR_ROARING_EXTERN_BITMAP_RUN( KEY, COW ) \
    FRSR_ROARING_EXTERN_BITMAP( KEY, COW, run_selection_eager ) \
    FRSR_ROARING_EXTERN_BITMAP( KEY, COW, run_selection_lazy )
#define FRSR_ROARING_EXTERN_BITMAP_COW( KEY ) \
    FRSR_ROARING_EXTERN_BITMAP_RUN( KEY, cow_value_semantics ) \
    FRSR_ROARING_EXTERN_BITMAP_RUN( KEY, cow_atomic_refcount ) \
    FRSR_ROARING_EXTERN_BITMAP_RUN( KEY, cow_unsynchronized_refcount )

namespace frsr::roaring {
FRSR_ROARING_EXTERN_BITMAP_COW( std::uint16_t )
FRSR_ROARING_EXTERN_BITMAP_COW( std::uint32_t )
FRSR_ROARING_EXTERN_BITMAP_COW( std::uint64_t )
} // namespace frsr::roaring

#undef FRSR_ROARING_EXTERN_BITMAP_COW
#undef FRSR_ROARING_EXTERN_BITMAP_RUN
#undef FRSR_ROARING_EXTERN_BITMAP

#include <frsr/roaring/library_configuration.hpp>

#endif // FRSR_ROARING_HEADER_ONLY
