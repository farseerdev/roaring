#pragma once

// The concrete, storage-owning container representation replacing
// std::variant<array_container, run_container, bitset_container>.
//
// A container_handle is a fixed 32-byte (½ cacheline) trivially-relocatable
// value: a 16-byte header { count, cardinality, [min,max] endpoints, kind,
// ownership, flags } + a 16-byte body holding either the payload elements
// inline (small arrays / runs) or a spilled { data, capacity } record
// (large arrays / runs, and always the 8 KB bitset word block).
//
// - count / cardinality / [min,max] live in the header so the merge walk and
//   the value-search reject (v < min || v > max) never dereference the body.
// - Ownership carries the CoW / mmap-borrow dimension from birth: `unique`
//   deep-owned (Model 1 value semantics), `shared` refcounted born-shared
//   (Models 2/3), `borrowed` aliasing read-only mapped bytes with no refcount
//   word — copies stay borrowed under every policy and the write barrier
//   clones to private storage on first mutable access.
// - Typed access goes through array_ref / run_ref / bitset_ref proxies which
//   expose the same payload members (.values / .runs / .words) and the same
//   membership / mutation algorithms the former standalone container types
//   had, keeping header fields in sync.
//
// See the container-representation design notes.

#include <frsr/roaring/container_layout.hpp>
#include <frsr/roaring/cow_policy.hpp>
#include <frsr/roaring/run.hpp>
#include <frsr/roaring/run_container.hpp>

// Payload allocator backend. Host projects that route their allocations through
// mimalloc (signalled by psi::vm's PSI_VM_HAS_MIMALLOC, or explicitly via
// FRSR_ROARING_HAS_MIMALLOC) get the container payloads there too. This matters
// beyond consistency: on macOS a statically linked mimalloc does not interpose
// the global operator new, so without this the per-container payload traffic
// lands on Apple's (slower for this pattern) system zone allocator while the
// rest of the host's allocations use mimalloc.
#ifndef FRSR_ROARING_HAS_MIMALLOC
#   ifdef PSI_VM_HAS_MIMALLOC
#       define FRSR_ROARING_HAS_MIMALLOC PSI_VM_HAS_MIMALLOC
#   else
#       define FRSR_ROARING_HAS_MIMALLOC 0
#   endif
#endif
#if FRSR_ROARING_HAS_MIMALLOC
#   include <mimalloc.h>
#endif

// Diagnostic-only payload-allocation attribution (counts per originating site
// + size histogram, dumped to stderr at exit). Compile-time gated, default off.
#ifndef FRSR_ROARING_PAYLOAD_ALLOC_STATS
#   define FRSR_ROARING_PAYLOAD_ALLOC_STATS 0
#endif
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
#   include <atomic>
#   include <cstdio>
#endif

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace frsr::roaring::detail {

#if FRSR_ROARING_HAS_MIMALLOC
// Container payloads allocate from a DEDICATED per-thread mimalloc heap rather
// than the thread's default heap. mimalloc retires-and-reuses an emptied page
// only while it is the SOLE page in its size-bin queue (_mi_page_retire); in
// the default heap any long-lived co-tenant allocation that lands in the same
// size bin defeats that protection, and the hot intermediate-payload stream
// then cycles a fresh OS page every few allocations (measured on a downstream
// Apple-Silicon workload: a large number of fresh pages and a large amount of
// touched memory, and a large CPU regression, triggered purely by an unrelated
// struct changing size bins). A payload-only
// heap restores the single-page fast path by construction instead of by
// bin-tenancy luck. Frees stay heap-agnostic (mi_free resolves the owning page
// from the address) so payloads may be freed from any thread; mi_heap_delete
// at thread exit migrates still-live blocks to the default heap.
// Split into a POD fast path + cold-outlined creation: a non-trivial
// thread_local (guarded init + registered dtor) costs a guard check on every
// access on top of the TLS resolution itself — on Darwin, where every access
// already pays the _tlv_get_addr thunk, the guarded form measured ~58% dearer
// than a POD slot, and the guard/creation code inlined into payload_heap()
// defeated allocate_payload's own inlining at its call sites. The POD cached
// pointer is zero-initialized (no guard, no thunk beyond the base TLS cost);
// heap creation and end-of-thread cleanup live in the cold path only (the
// lifetime-managing holder there is touched once per thread). A mi_heap_new
// failure leaves the cache null: the cold path is then re-entered on every
// allocation, each falling back to the default heap (same fallback semantics
// as before, failure case only).
[[nodiscard, gnu::cold, gnu::noinline]] inline mi_heap_t * create_payload_heap() noexcept {
    struct holder_t {
        mi_heap_t * const heap{ mi_heap_new() };
        ~holder_t() noexcept { if ( heap ) { mi_heap_delete( heap ); } }
    };
    static thread_local holder_t const holder;
    return holder.heap;
}
[[nodiscard]] inline mi_heap_t * payload_heap() noexcept {
    static thread_local mi_heap_t * cached; // POD/zero-init: guard-free access
    if ( cached == nullptr ) [[unlikely]] {
        cached = create_payload_heap();
    }
    return cached;
}
#endif // FRSR_ROARING_HAS_MIMALLOC

enum class container_kind : std::uint8_t { array = 0, run = 1, bitset = 2 };

enum class storage_ownership : std::uint8_t { unique = 0, shared = 1, borrowed = 2 };

template <typename Layout, typename CowPolicy = cow_value_semantics> class array_ref;
template <typename Layout, typename CowPolicy = cow_value_semantics> class array_cref;
template <typename Layout, typename CowPolicy = cow_value_semantics> class run_ref;
template <typename Layout, typename CowPolicy = cow_value_semantics> class run_cref;
template <typename Layout, typename CowPolicy = cow_value_semantics> class bitset_ref;
template <typename Layout, typename CowPolicy = cow_value_semantics> class bitset_cref;

template <typename Layout, typename CowPolicy = cow_value_semantics>
class container_handle {
public:
    using cow_policy = CowPolicy;

    // Bytes reserved before a spilled payload for its refcount word. 0 under the
    // value-semantics default (no refcount word); the copy path deep-clones.
    static constexpr std::size_t rc_prefix_bytes{ CowPolicy::rc_prefix_bytes };

    using low_type  = typename Layout::low_type;
    using run_type  = ::frsr::roaring::run<low_type>;
    using word_array = std::array<std::uint64_t, Layout::word_count>;

    static constexpr std::size_t handle_size{ 32 };
    static constexpr std::size_t body_size{ 16 };

    template <typename E>
    static constexpr std::uint32_t inline_capacity{ static_cast<std::uint32_t>( body_size / sizeof( E ) ) };

    constexpr container_handle() noexcept = default;   // empty inline array

    container_handle( container_handle const & other ) { clone_from( other ); }

    // Field-wise (not whole-*this memcpy) so locally-built handles stay
    // SROA-able — a memcpy over `this` forces every stack-constructed result
    // through memory.
    container_handle( container_handle && other ) noexcept { adopt_fields_from( other ); }

    container_handle & operator=( container_handle const & other ) {
        if ( this != &other ) [[likely]] {
            release_payload();
            clone_from( other );
        }
        return *this;
    }

    container_handle & operator=( container_handle && other ) noexcept {
        if ( this != &other ) [[likely]] {
            release_payload();
            adopt_fields_from( other );
        }
        return *this;
    }

    ~container_handle() noexcept { release_payload(); }

    // psi::vm relocation opt-in: the spilled pointer is not self-referential and
    // inline payloads are POD element arrays, so memcpy + skip-source-destroy is
    // a correct move+destroy.
    static constexpr bool is_trivially_moveable{ true };

    [[nodiscard]] container_kind    kind () const noexcept { return kind_ ; }
    [[nodiscard]] storage_ownership owner() const noexcept { return owner_; }

    [[nodiscard]] bool holds_array () const noexcept { return kind_ == container_kind::array ; }
    [[nodiscard]] bool holds_run   () const noexcept { return kind_ == container_kind::run   ; }
    [[nodiscard]] bool holds_bitset() const noexcept { return kind_ == container_kind::bitset; }

    // Element count in the payload's own unit: array #values, run #runs, bitset word_count.
    [[nodiscard]] std::uint32_t count      () const noexcept { return count_      ; }
    [[nodiscard]] std::uint32_t cardinality() const noexcept { return cardinality_; }
    [[nodiscard]] bool          empty      () const noexcept { return cardinality_ == 0; }

    // Valid iff !empty(): smallest / largest value present. Enables the
    // header-only reject (v < min || v > max) without touching the payload.
    // LAZY for bitsets: the bulk word kernels mark the endpoints stale instead
    // of paying an endpoint scan per combine (nothing consumes them mid-merge);
    // the first read recomputes them. Arrays/runs keep them exact for free.
    [[nodiscard]] low_type min_value() const noexcept { ensure_endpoints(); return min_; }
    [[nodiscard]] low_type max_value() const noexcept { return ensure_endpoints(), max_; }

    [[nodiscard]] bool endpoints_valid() const noexcept { return ( flags_ & endpoints_stale_flag ) == 0; }
    void mark_endpoints_stale() noexcept { flags_ |= endpoints_stale_flag; }

    // Bitset cardinality is exact unless a nocard lazy word kernel dirtied it.
    // repair_cardinality() only re-popcounts when this is set — card-updating
    // bulk OR (and ordinary add/remove) leave it clear so a dense K-way union
    // does not re-scan every saturated 8 KB block at finish.
    [[nodiscard]] bool cardinality_valid() const noexcept { return ( flags_ & cardinality_stale_flag ) == 0; }
    void mark_cardinality_stale() noexcept { flags_ |= cardinality_stale_flag; }

    [[nodiscard]] bool spilled() const noexcept { return ( flags_ & spilled_flag ) != 0; }

    // ---- typed access ------------------------------------------------------
    //
    // The non-const accessors are the CoW write barrier: under a refcounted
    // policy they first make the payload sole-referent (clone if rc > 1), and
    // under EVERY policy they clone a borrowed (read-only-mapping) payload to
    // private storage, so a mutable ref can never alias another handle's — or
    // a mapping's — payload. Hence they are the only sanctioned way to obtain
    // a mutating view of a possibly-shared or borrowed handle. They can
    // allocate: throwing under a refcounted policy, terminating on allocation
    // failure under the non-refcounted ones (kept noexcept so the Model-1 hot
    // path does not grow exception edges).

    [[nodiscard]] array_ref <Layout, CowPolicy> as_array ()       noexcept( !CowPolicy::refcounted );
    [[nodiscard]] array_cref<Layout, CowPolicy> as_array () const noexcept;
    [[nodiscard]] run_ref   <Layout, CowPolicy> as_run   ()       noexcept( !CowPolicy::refcounted );
    [[nodiscard]] run_cref  <Layout, CowPolicy> as_run   () const noexcept;
    [[nodiscard]] bitset_ref <Layout, CowPolicy> as_bitset()       noexcept( !CowPolicy::refcounted );
    [[nodiscard]] bitset_cref<Layout, CowPolicy> as_bitset() const noexcept;

    template <typename F> decltype( auto ) visit( F && f );
    template <typename F> decltype( auto ) visit( F && f ) const;

    // ---- factories ---------------------------------------------------------

    [[nodiscard]] static container_handle make_array() noexcept { return {}; }

    [[nodiscard]] static container_handle make_array_from_sorted( std::span<low_type const> sorted_values );

    [[nodiscard]] static container_handle make_run() noexcept {
        container_handle handle;
        handle.kind_ = container_kind::run;
        return handle;
    }

    // Zero-initialized word block, cardinality 0.
    [[nodiscard]] static container_handle make_bitset_zeroed() {
        auto handle{ make_bitset_uninitialized() };
        std::memset( handle.payload_data_raw(), 0, sizeof( word_array ) );
        return handle;
    }

    // Uninitialized word block: the caller must overwrite every word and then
    // restore the header invariants (cardinality + endpoints).
    [[nodiscard]] static container_handle make_bitset_uninitialized() {
        container_handle handle;
        handle.kind_  = container_kind::bitset;
        handle.count_ = static_cast<std::uint32_t>( Layout::word_count );
        handle.mark_endpoints_stale();
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
        payload_alloc_stats().bitset_spill.fetch_add( 1, std::memory_order_relaxed );
#endif
        handle.spill_to( allocate_payload( sizeof( word_array ) ), static_cast<std::uint32_t>( Layout::word_count ) );
        return handle;
    }

    [[nodiscard]] static container_handle make_bitset_from_words( word_array const & words );

    // A handle whose payload aliases caller-owned read-only bytes (an mmapped
    // frozen buffer): always spilled, carries no refcount word, and is never
    // freed by the handle — the mapped bytes must outlive every handle (and
    // every copy of it) borrowing them. Copies alias the same bytes and stay
    // borrowed under every CoW policy; the first mutable access
    // (as_array / as_run / as_bitset) clones the payload to private storage.
    // `count` and `cardinality` follow the usual header semantics; the
    // endpoints are derived from the payload (arrays / runs eagerly, bitsets
    // lazily via the stale flag).
    [[nodiscard]] static container_handle make_borrowed(
        container_kind const kind,
        void const * const payload,
        std::uint32_t const count,
        std::uint32_t const cardinality
    ) noexcept {
        container_handle handle;
        handle.kind_        = kind;
        handle.count_       = count;
        handle.cardinality_ = cardinality;
        // Immutable payload ⇒ capacity == count; growth always goes through the
        // write-barrier clone first.
        handle.body_.spilled = spilled_rep{ const_cast<void *>( payload ), count };
        handle.flags_ = spilled_flag;
        handle.owner_ = storage_ownership::borrowed;
        switch ( kind ) {
            case container_kind::array:
                if ( count != 0 ) {
                    auto const * const values{ static_cast<low_type const *>( payload ) };
                    handle.set_endpoints( values[ 0 ], values[ count - 1 ] );
                }
                break;
            case container_kind::run:
                if ( count != 0 ) {
                    auto const * const runs{ static_cast<run_type const *>( payload ) };
                    handle.set_endpoints( runs[ 0 ].begin, runs[ count - 1 ].end );
                }
                break;
            case container_kind::bitset:
                handle.mark_endpoints_stale();
                break;
        }
        return handle;
    }

    // ---- header maintenance (used by refs and by direct-write call sites) ---

    void set_count      ( std::uint32_t const n ) noexcept { count_       = n; }
    void set_cardinality( std::uint32_t const n ) noexcept {
        cardinality_ = n;
        flags_ &= static_cast<std::uint8_t>( ~cardinality_stale_flag );
    }
    void set_endpoints  ( low_type const min, low_type const max ) noexcept {
        min_ = min;
        max_ = max;
        flags_ &= static_cast<std::uint8_t>( ~endpoints_stale_flag );
    }

    // ---- scratch payload reuse (CRoaring persistent-dst analog) -------------
    //
    // A scratch bitmap's clear_keep_capacity() retires its slots instead of
    // destroying them, and the in-place combine walk retires each consumed
    // slot; the next materializing site rebuilds its result into a retired
    // payload via chunk_store::take_retired(), so the per-combine
    // free-everything/reallocate-everything churn collapses to in-place writes
    // — the same buffer-reuse CRoaring gets from growing a persistent dst
    // container (array_container_grow) instead of allocating fresh ones.

    // True when this handle solely owns a spilled payload of `kind` that a
    // rebuilt result may overwrite in place. Sole ownership (unique, or shared
    // with rc == 1) naturally excludes retired pass-through entries that still
    // co-own a live source bitmap's payload.
    [[nodiscard]] bool offers_reusable_payload( container_kind const kind ) const noexcept {
        if ( !spilled() || kind_ != kind ) {
            return false;
        }
        switch ( owner_ ) {
            case storage_ownership::unique  : return true;
            case storage_ownership::shared  :
                if constexpr ( CowPolicy::refcounted ) {
                    return CowPolicy::rc_load( rc_slot_of( spill().data ) ) == 1;
                }
                return false;
            case storage_ownership::borrowed: return false;
        }
        return false; // unreachable
    }

    // Header resets for a payload handed back by take_retired(): the storage
    // (and ownership/refcount word) stays, the logical contents restart empty.
    void reset_for_array_reuse() noexcept {
        count_       = 0;
        cardinality_ = 0;
        flags_       = spilled_flag;
    }
    // Bitsets restart as a full uninitialized word block (the reuse sites
    // overwrite every word, mirroring make_bitset_uninitialized).
    void reset_for_bitset_reuse() noexcept {
        count_       = static_cast<std::uint32_t>( Layout::word_count );
        cardinality_ = 0;
        flags_       = spilled_flag | endpoints_stale_flag;
    }

    // ---- raw payload plumbing (used by the ref proxies) ---------------------

    [[nodiscard]] void       * payload_data_raw()       noexcept { return spilled() ? spill().data : body_.inline_bytes; }
    [[nodiscard]] void const * payload_data_raw() const noexcept { return spilled() ? spill().data : body_.inline_bytes; }

    template <typename E> [[nodiscard]] E       * payload_data()       noexcept { return static_cast<E       *>( payload_data_raw() ); }
    template <typename E> [[nodiscard]] E const * payload_data() const noexcept { return static_cast<E const *>( payload_data_raw() ); }

    template <typename E>
    [[nodiscard]] std::uint32_t payload_capacity() const noexcept {
        return spilled() ? spill().capacity : inline_capacity<E>;
    }

    // Grows the payload to hold at least `required` elements of E, preserving
    // the first count() elements. Never shrinks. The grow path is outlined cold:
    // it inlines rc-header allocation + memcpy + release, and the combine kernels
    // call ensure-capacity from several resize sites — inlining the slow path at
    // each bloated those kernels ~10x vs their CRoaring counterparts (I-cache
    // footprint measured off the real Release binary in the round-6 asm pass).
    template <typename E>
    [[gnu::always_inline]] void ensure_payload_capacity( std::uint32_t const required ) {
        if ( required <= payload_capacity<E>() ) [[likely]] {
            return;
        }
        grow_payload_capacity<E>( required );
    }

    template <typename E>
    [[using gnu: cold, noinline]] void grow_payload_capacity( std::uint32_t const required ) {
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
        ( spilled() ? payload_alloc_stats().grow_chain : payload_alloc_stats().grow_first ).fetch_add( 1, std::memory_order_relaxed );
#endif
        auto const grown{ std::max( required, payload_capacity<E>() * 2U ) };
        auto * const fresh{ allocate_payload( std::size_t{ grown } * sizeof( E ) ) };
        std::memcpy( fresh, payload_data_raw(), std::size_t{ count_ } * sizeof( E ) );
        if ( spilled() ) {
            free_payload( spill().data );
        }
        spill_to( fresh, grown );
    }

private:
    static constexpr std::uint8_t spilled_flag          { 0x01 };
    static constexpr std::uint8_t endpoints_stale_flag  { 0x02 };
    static constexpr std::uint8_t cardinality_stale_flag{ 0x04 };

    // Recomputes stale [min,max] from the payload (bitset word scan; arrays and
    // runs keep them exact eagerly). Header-only mutation, hence the mutable
    // endpoint fields.
    void ensure_endpoints() const noexcept;

    struct spilled_rep {
        void        * data;
        std::uint32_t capacity;   // in elements of the payload's element type
    };

    // The two alternatives ever held in a handle's inline body: raw bytes for
    // small in-body payloads (arrays/runs staying within SBO capacity), or the
    // { data, capacity } record once the payload spills to a separate
    // allocation. Per [class.union]/[basic.life], writing a named union member
    // and later reading that SAME named member is well-defined without
    // std::launder — laundering is only required when a raw byte buffer is
    // reinterpret_cast to a different type, which a union member access never
    // does. Both alternatives are trivially copyable, so the union itself is
    // trivially copyable/relocatable — whole-object memcpy over `body_`
    // (copy_fields_from / clone_borrowed_payload) stays exactly as safe as it
    // was over the raw byte array.
    union body_union_t {
        std::byte   inline_bytes[ body_size ];
        spilled_rep spilled;
    };

    [[nodiscard]] spilled_rep       & spill()       noexcept { return body_.spilled; }
    [[nodiscard]] spilled_rep const & spill() const noexcept { return body_.spilled; }

    void spill_to( void * const data, std::uint32_t const capacity ) noexcept {
        body_.spilled = spilled_rep{ data, capacity };
        flags_ |= spilled_flag;
        if constexpr ( CowPolicy::refcounted ) {
            // Born shared (rc == 1), never promoted in place later: copies only
            // ever touch the refcount, so they cannot disturb concurrent readers
            // of the source handle. Invariant under a refcounted policy:
            // owner == shared ⟺ spilled.
            owner_ = storage_ownership::shared;
        }
    }

    // Payloads carry an rc_prefix_bytes-wide refcount word immediately before
    // the returned pointer ([ rc ][ payload ] in one allocation); the returned
    // pointer addresses the payload so accessors never see the prefix. Under the
    // value-semantics default rc_prefix_bytes == 0 and both reduce to a plain
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
    // Diagnostic-only (compile-time gated, default off): attributes payload
    // allocations to their originating site + a size histogram, printed to
    // stderr at process exit. Counts only — do not use for timing.
    struct payload_alloc_stats_t {
        std::atomic<std::uint64_t> total{}, cls448{}, bitset_spill{}, grow_first{}, grow_chain{}, cow_clone{}, borrowed_clone{};
        std::atomic<std::uint64_t> size_log2[ 24 ]{};
        ~payload_alloc_stats_t() {
            std::fprintf( stderr, "[payload-alloc] total=%llu cls(432,448]=%llu bitset_spill=%llu grow_first=%llu grow_chain=%llu cow_clone=%llu borrowed_clone=%llu\n",
                (unsigned long long)total.load(), (unsigned long long)cls448.load(), (unsigned long long)bitset_spill.load(),
                (unsigned long long)grow_first.load(), (unsigned long long)grow_chain.load(),
                (unsigned long long)cow_clone.load(), (unsigned long long)borrowed_clone.load() );
            for ( auto i{ 0U }; i < 24U; ++i ) {
                if ( auto const n{ size_log2[ i ].load() } ) {
                    std::fprintf( stderr, "[payload-alloc]   size<=%u B: %llu\n", 1U << i, (unsigned long long)n );
                }
            }
        }
    };
    static payload_alloc_stats_t & payload_alloc_stats() { static payload_alloc_stats_t s; return s; }
#endif

    // new/delete. free_payload always rebases by the *compile-time* offset (not
    // runtime owner) so a demoted-to-unique shared payload still frees its base.
    [[nodiscard]] static void * allocate_payload( std::size_t const bytes ) {
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
        {
            auto & s{ payload_alloc_stats() };
            auto const sz{ rc_prefix_bytes + bytes };
            s.total.fetch_add( 1, std::memory_order_relaxed );
            if ( sz > 432 && sz <= 448 ) { s.cls448.fetch_add( 1, std::memory_order_relaxed ); }
            auto bucket{ 0U }; while ( ( std::size_t{ 1 } << bucket ) < sz && bucket < 23U ) { ++bucket; }
            s.size_log2[ bucket ].fetch_add( 1, std::memory_order_relaxed );
        }
#endif
#if FRSR_ROARING_HAS_MIMALLOC
        void * raw;
        if ( auto * const heap{ payload_heap() } ) [[likely]] {
            raw = mi_heap_malloc( heap, rc_prefix_bytes + bytes );
            if ( raw == nullptr ) [[unlikely]] { throw std::bad_alloc{}; }
        } else { // mi_heap_new failure fallback: default heap, mi_new OOM semantics
            raw = mi_new( rc_prefix_bytes + bytes );
        }
        auto * const base{ static_cast<std::byte *>( raw ) };
#else
        auto * const base{ static_cast<std::byte *>( ::operator new( rc_prefix_bytes + bytes ) ) };
#endif
        if constexpr ( CowPolicy::refcounted ) {
            CowPolicy::rc_construct( base );
        }
        return base + rc_prefix_bytes;
    }

    static void free_payload( void * const data ) noexcept {
#if FRSR_ROARING_HAS_MIMALLOC
        mi_free( static_cast<std::byte *>( data ) - rc_prefix_bytes );
#else
        ::operator delete( static_cast<std::byte *>( data ) - rc_prefix_bytes );
#endif
    }

    [[nodiscard]] static void * rc_slot_of( void * const payload ) noexcept {
        return static_cast<std::byte *>( payload ) - rc_prefix_bytes;
    }

    void release_payload() noexcept {
        if ( !spilled() ) {
            return;
        }
        if constexpr ( CowPolicy::refcounted ) {
            if ( owner_ == storage_ownership::shared ) {
                if ( CowPolicy::rc_decrement_is_last( rc_slot_of( spill().data ) ) ) {
                    free_payload( spill().data );
                }
                return;
            }
        }
        if ( owner_ == storage_ownership::unique ) {
            free_payload( spill().data );
        }
    }

    // CoW write barrier: makes the payload private before handing out a mutable
    // view. A borrowed payload (read-only mapping, no refcount word — reading
    // one would stray outside the mapping) clones under EVERY policy; a shared
    // one only when co-owned. rc == 1 (the common case for a handle that was
    // never copied, or already cloned) mutates in place; rc > 1 clones to a
    // fresh born-shared payload and drops this handle's reference to the old
    // one. Under a non-refcounted policy this reduces to the single
    // predictably-never-taken borrowed check.
    void make_payload_unique() noexcept( !CowPolicy::refcounted ) {
        if ( owner_ == storage_ownership::borrowed ) [[unlikely]] {
            clone_borrowed_payload();
            return;
        }
        if constexpr ( CowPolicy::refcounted ) {
            if ( !spilled() ) {
                return;
            }
            auto * const old_payload{ spill().data };
            if ( CowPolicy::rc_load( rc_slot_of( old_payload ) ) == 1 ) {
                return;
            }
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
            payload_alloc_stats().cow_clone.fetch_add( 1, std::memory_order_relaxed );
#endif
            auto * const fresh{ allocate_payload( std::size_t{ spill().capacity } * element_size() ) };
            std::memcpy( fresh, old_payload, std::size_t{ count_ } * element_size() );
            if ( CowPolicy::rc_decrement_is_last( rc_slot_of( old_payload ) ) ) {
                free_payload( old_payload );
            }
            spill().data = fresh;
        }
    }

    // Clones a borrowed payload into private storage: inline when it fits the
    // body (small arrays / runs — no allocation), else a fresh exact-capacity
    // allocation (born shared under a refcounted policy, unique otherwise).
    // The mapped source bytes are only read. noexcept-shaped like the barrier:
    // under a non-refcounted policy an allocation failure here terminates
    // rather than widening every Model-1 mutable accessor into a throwing one.
    [[gnu::cold]] void clone_borrowed_payload() noexcept( !CowPolicy::refcounted ) {
        auto const bytes{ std::size_t{ count_ } * element_size() };
        void const * const source{ spill().data };
        if ( bytes <= body_size ) {   // bitsets (8 KB block) never fit ⇒ always take the spilled arm
            flags_ &= static_cast<std::uint8_t>( ~spilled_flag );
            owner_ = storage_ownership::unique;
            std::memcpy( body_.inline_bytes, source, bytes );
        } else {
#if FRSR_ROARING_PAYLOAD_ALLOC_STATS
            payload_alloc_stats().borrowed_clone.fetch_add( 1, std::memory_order_relaxed );
#endif
            auto * const fresh{ allocate_payload( bytes ) };
            std::memcpy( fresh, source, bytes );
            spill_to( fresh, count_ );
            if constexpr ( !CowPolicy::refcounted ) {
                owner_ = storage_ownership::unique;
            }
        }
    }

    void reset_to_empty() noexcept {
        count_       = 0;
        cardinality_ = 0;
        kind_        = container_kind::array;
        owner_       = storage_ownership::unique;
        flags_       = 0;
    }

    [[nodiscard]] std::size_t element_size() const noexcept {
        switch ( kind_ ) {
            case container_kind::array : return sizeof( low_type );
            case container_kind::run   : return sizeof( run_type );
            case container_kind::bitset: return sizeof( std::uint64_t );
        }
        return 0; // unreachable
    }

    void copy_fields_from( container_handle const & other ) noexcept {
        count_       = other.count_;
        cardinality_ = other.cardinality_;
        min_         = other.min_;
        max_         = other.max_;
        kind_        = other.kind_;
        owner_       = other.owner_;
        flags_       = other.flags_;
        std::memcpy( &body_, &other.body_, body_size );   // whole-object byte copy; a union of trivially-copyable alternatives is itself trivially copyable
    }

    void adopt_fields_from( container_handle & other ) noexcept {
        copy_fields_from( other );
        other.reset_to_empty();
    }

    void clone_from( container_handle const & other ) {
        copy_fields_from( other );
        if ( owner_ == storage_ownership::borrowed ) {
            // Borrowed payloads live in a read-only mapping with no refcount
            // word: the copy aliases the same mapped bytes and stays borrowed
            // (zero-copy under every policy — this is what makes copying an
            // all-borrowed mmapped master a pure SoA-array memcpy); the write
            // barrier clones on first mutation.
            return;
        }
        if constexpr ( CowPolicy::refcounted ) {
            // Spilled payloads are shared by refcount (owner_ == shared came over
            // with the fields); inline bodies were value-copied by the memcpy.
            if ( other.spilled() ) {
                CowPolicy::rc_increment( rc_slot_of( spill().data ) );
            }
        } else {
            owner_ = storage_ownership::unique;
            if ( other.spilled() ) {
                auto const bytes{ std::size_t{ other.spill().capacity } * other.element_size() };
                auto * const fresh{ allocate_payload( bytes ) };
                std::memcpy( fresh, other.spill().data, std::size_t{ other.count_ } * other.element_size() );
                spill().data = fresh;
            }
        }
    }

    friend class array_ref <Layout, CowPolicy>;
    friend class array_cref<Layout, CowPolicy>;
    friend class run_ref   <Layout, CowPolicy>;
    friend class run_cref  <Layout, CowPolicy>;
    friend class bitset_ref <Layout, CowPolicy>;
    friend class bitset_cref<Layout, CowPolicy>;

private:
    std::uint32_t             count_      { 0 };
    std::uint32_t             cardinality_{ 0 };
    mutable low_type          min_        { 0 };
    mutable low_type          max_        { 0 };
    container_kind            kind_       { container_kind::array };
    storage_ownership         owner_      { storage_ownership::unique };
    mutable std::uint8_t      flags_      { 0 };
    std::uint8_t              reserved_   { 0 };
    alignas( 8 ) body_union_t body_;
};

// ---- payload vector proxy ----------------------------------------------------
//
// A vector-like view over a handle's payload for element type E. Mutators keep
// the handle's `count` in sync; the algorithm-level header fields (cardinality,
// endpoints) are maintained by the typed refs / explicit sync at direct-write
// call sites.

template <typename Layout, typename E, typename CowPolicy = cow_value_semantics>
class payload_vector {
public:
    using value_type = E;

    explicit payload_vector( container_handle<Layout, CowPolicy> & handle ) noexcept : handle_{ &handle } {}

    [[nodiscard]] E       * data()       noexcept { return handle_->template payload_data<E>(); }
    [[nodiscard]] E const * data() const noexcept { return std::as_const( *handle_ ).template payload_data<E>(); }

    [[nodiscard]] std::uint32_t size    () const noexcept { return handle_->count(); }
    [[nodiscard]] bool          empty   () const noexcept { return size() == 0; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return handle_->template payload_capacity<E>(); }

    [[nodiscard]] E       * begin()       noexcept { return data(); }
    [[nodiscard]] E const * begin() const noexcept { return data(); }
    [[nodiscard]] E       * end  ()       noexcept { return data() + size(); }
    [[nodiscard]] E const * end  () const noexcept { return data() + size(); }

    [[nodiscard]] E       & operator[]( std::size_t const i )       noexcept { return data()[ i ]; }
    [[nodiscard]] E const & operator[]( std::size_t const i ) const noexcept { return data()[ i ]; }

    [[nodiscard]] E       & front()       noexcept { return data()[ 0 ]; }
    [[nodiscard]] E const & front() const noexcept { return data()[ 0 ]; }
    [[nodiscard]] E       & back ()       noexcept { return data()[ size() - 1 ]; }
    [[nodiscard]] E const & back () const noexcept { return data()[ size() - 1 ]; }

    void reserve( std::uint32_t const n ) { handle_->template ensure_payload_capacity<E>( n ); }

    void push_back( E const value ) {
        auto const n{ size() };
        handle_->template ensure_payload_capacity<E>( n + 1U );
        data()[ n ] = value;
        handle_->set_count( n + 1U );
    }

    // Grows with zero-fill (std::vector semantics for these POD payloads) or
    // shrinks by dropping the tail.
    void resize( std::uint32_t const n ) {
        auto const old_size{ size() };
        resize_uninitialized( n );
        if ( n > old_size ) {
            std::memset( data() + old_size, 0, std::size_t{ n - old_size } * sizeof( E ) );
        }
    }

    void resize_uninitialized( std::uint32_t const n ) {
        handle_->template ensure_payload_capacity<E>( n );
        handle_->set_count( n );
    }

    void assign( std::span<E const> const values ) {
        resize_uninitialized( static_cast<std::uint32_t>( values.size() ) );
        std::memcpy( data(), values.data(), values.size_bytes() );
    }

    void clear() noexcept { handle_->set_count( 0 ); }

private:
    container_handle<Layout, CowPolicy> * handle_;
};

template <typename Layout, typename E, typename CowPolicy = cow_value_semantics>
class payload_vector_const_view {
public:
    using value_type = E;

    // Re-derives the payload pointer + count from the handle on every
    // data()/size() call rather than caching them at construction. A cached
    // copy is unsound: a const view's own members can't be mutated, but the
    // *handle it wraps* can be — the two-container combine algorithms
    // (array_ops.hpp / run_ops.hpp / bitset_ops.hpp) build a mutable ref for
    // one operand and a const view for the other from what may be the SAME
    // handle (in-place self-union/self-combine), so growing the mutable side
    // can reallocate the payload out from under an already-cached pointer on
    // the const side. payload_data_raw() (container_handle.hpp) is now a
    // plain `body_.spilled`/`body_.inline_bytes` union-member read — an
    // ordinary, cheap, hoistable/CSE-able struct field access, not the
    // std::launder-guarded derivation this view used to work around — so
    // re-deriving per call costs nothing and closes the dangling-pointer
    // window entirely.
    explicit payload_vector_const_view( container_handle<Layout, CowPolicy> const & handle ) noexcept
        : handle_{ &handle } {}

    [[nodiscard]] E const * data () const noexcept { return handle_->template payload_data<E>(); }
    [[nodiscard]] std::uint32_t size() const noexcept { return handle_->count(); }
    [[nodiscard]] bool  empty() const noexcept { return size() == 0; }
    [[nodiscard]] E const * begin() const noexcept { return data(); }
    [[nodiscard]] E const * end  () const noexcept { return data() + size(); }
    [[nodiscard]] E const & operator[]( std::size_t const i ) const noexcept { return data()[ i ]; }
    [[nodiscard]] E const & front() const noexcept { return data()[ 0 ]; }
    [[nodiscard]] E const & back () const noexcept { return data()[ size() - 1 ]; }

private:
    container_handle<Layout, CowPolicy> const * handle_;
};

// Assignable / incrementable view of the handle's header cardinality, so ref
// proxies can expose `.cardinality` with the same syntax the former standalone
// container types had.
template <typename Layout, typename CowPolicy = cow_value_semantics>
class header_cardinality_proxy {
public:
    explicit header_cardinality_proxy( container_handle<Layout, CowPolicy> & handle ) noexcept : handle_{ &handle } {}

    [[nodiscard]] operator std::uint32_t() const noexcept { return handle_->cardinality(); }

    header_cardinality_proxy & operator= ( std::uint32_t const n ) noexcept { handle_->set_cardinality( n ); return *this; }
    header_cardinality_proxy & operator+=( std::uint32_t const n ) noexcept { handle_->set_cardinality( handle_->cardinality() + n ); return *this; }
    header_cardinality_proxy & operator-=( std::uint32_t const n ) noexcept { handle_->set_cardinality( handle_->cardinality() - n ); return *this; }
    header_cardinality_proxy & operator++() noexcept { return *this += 1U; }
    header_cardinality_proxy & operator--() noexcept { return *this -= 1U; }

private:
    container_handle<Layout, CowPolicy> * handle_;
};

// ---- shared payload algorithms -----------------------------------------------

// SIMD block + branchless binary-search membership test over a sorted array.
// Algorithm by Daniel Lemire, "You can beat the binary search"
// (https://lemire.me/blog/2026/04/27/you-can-beat-the-binary-search/);
// reference code lemire/Code-used-on-Daniel-Lemire-s-blog/extra/simd/binsearch/simdbinsearch16.c
// (public domain). [lemire-ref] simdbinsearch16.c
template <typename E>
[[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] inline bool sorted_array_contains(
    E const * const carr,
    std::uint32_t const size,
    E const value
) noexcept {
    const int32_t gap = 16;
    auto const cardinality{ static_cast<int32_t>( size ) };
    if ( cardinality < gap ) {
        for ( int32_t j = 0; j < cardinality; ++j ) {
            if ( carr[ j ] >= value ) {
                return carr[ j ] == value;
            }
        }
        return false;
    }

    auto const num_blocks{ cardinality / gap };
    int32_t base = 0;
    int32_t n = num_blocks;
    while ( n > 3 ) {
        auto const quarter{ n >> 2 };
        auto const k1{ carr[ ( base + quarter + 1 ) * gap - 1 ] };
        auto const k2{ carr[ ( base + 2 * quarter + 1 ) * gap - 1 ] };
        auto const k3{ carr[ ( base + 3 * quarter + 1 ) * gap - 1 ] };
        auto const c1{ k1 < value };
        auto const c2{ k2 < value };
        auto const c3{ k3 < value };
        base += ( c1 + c2 + c3 ) * quarter;
        n -= 3 * quarter;
    }
    while ( n > 1 ) {
        auto const half{ n >> 1 };
        base = ( carr[ ( base + half + 1 ) * gap - 1 ] < value ) ? base + half : base;
        n -= half;
    }

    auto const lo{ ( carr[ ( base + 1 ) * gap - 1 ] < value ) ? base + 1 : base };
    if ( lo < num_blocks ) {
        auto const * blk{ carr + static_cast<std::ptrdiff_t>( lo * gap ) };
        // The whole point of the block search is to end on a BRANCHLESS 16-element
        // probe: the scalar fallback below is an early-exit loop whose per-element
        // compare is data-dependent and essentially unpredictable (~8 mispredicts
        // per lookup), which costs more than the branchy binary search this
        // algorithm is meant to beat. Every SIMD ISA we build for therefore needs
        // its own arm here — leaving one on the scalar path silently makes that
        // platform slower than plain binary search, which is exactly what happened
        // on AArch64 (array-container contains ran 1.6-3.0x the CRoaring arm,
        // while x86 sat at parity and dense/run-container shapes were unaffected).
#if ( defined(_M_X64) || defined(__x86_64__) || defined(__i386__) )
        if constexpr ( sizeof( E ) == 2 ) {
            __m128i const needle{ _mm_set1_epi16( static_cast<short>( value ) ) };
            __m128i const v0{ _mm_loadu_si128( reinterpret_cast<__m128i const *>( blk ) ) };
            __m128i const v1{ _mm_loadu_si128( reinterpret_cast<__m128i const *>( blk + 8 ) ) };
            __m128i const hit{ _mm_or_si128( _mm_cmpeq_epi16( v0, needle ), _mm_cmpeq_epi16( v1, needle ) ) };
            return _mm_movemask_epi8( hit ) != 0;
        }
#elif ( defined(__aarch64__) || defined(_M_ARM64) )
        if constexpr ( sizeof( E ) == 2 ) {
            // Mirrors the SSE2 arm: two 8-lane compares, OR'd, then a horizontal
            // "any lane set" reduction. vmaxvq_u16 yields 0xFFFF iff some lane
            // matched (vceqq produces all-ones lanes), and is AArch64-only --
            // hence the __aarch64__/_M_ARM64 guard rather than __ARM_NEON.
            uint16x8_t const needle{ vdupq_n_u16( static_cast<std::uint16_t>( value ) ) };
            uint16x8_t const v0{ vld1q_u16( reinterpret_cast<std::uint16_t const *>( blk ) ) };
            uint16x8_t const v1{ vld1q_u16( reinterpret_cast<std::uint16_t const *>( blk + 8 ) ) };
            uint16x8_t const hit{ vorrq_u16( vceqq_u16( v0, needle ), vceqq_u16( v1, needle ) ) };
            return vmaxvq_u16( hit ) != 0;
        }
#endif
        for ( int32_t j = 0; j < gap; ++j ) {
            if ( blk[ j ] >= value ) {
                return blk[ j ] == value;
            }
        }
        return false;
    }

    for ( int32_t j = num_blocks * gap; j < cardinality; ++j ) {
        auto const v{ carr[ j ] };
        if ( v >= value ) {
            return v == value;
        }
    }
    return false;
}

// ---- array refs ---------------------------------------------------------------
//
// Arrays are sorted value lists: cardinality == count and the endpoints are
// front()/back(), so header sync after a mutation is O(1) (sync_header()).

template <typename Layout, typename CowPolicy>
class array_cref {
public:
    using low_type = typename Layout::low_type;

    explicit array_cref( container_handle<Layout, CowPolicy> const & handle ) noexcept : values{ handle } {}

    [[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] bool contains( low_type const value ) const noexcept {
        return sorted_array_contains( values.data(), values.size(), value );
    }

    [[nodiscard]] std::size_t size() const noexcept { return values.size(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept {
        if ( values.empty() ) {
            return std::nullopt;
        }
        return values.front();
    }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        auto const pos{ std::upper_bound( values.begin(), values.end(), value ) };
        if ( pos == values.end() ) {
            return std::nullopt;
        }
        return *pos;
    }

    template <typename F>
    void for_each( F && f ) const {
        for ( auto const value : values ) {
            f( value );
        }
    }

    payload_vector_const_view<Layout, low_type, CowPolicy> values;
};

template <typename Layout, typename CowPolicy>
class array_ref {
public:
    using low_type = typename Layout::low_type;

    explicit array_ref( container_handle<Layout, CowPolicy> & handle ) noexcept : values{ handle }, handle_{ &handle } {}

    [[nodiscard]] operator array_cref<Layout, CowPolicy>() const noexcept { return array_cref<Layout, CowPolicy>{ *handle_ }; }

    [[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] bool contains( low_type const value ) const noexcept {
        return sorted_array_contains( values.data(), values.size(), value );
    }

    [[nodiscard]] bool add( low_type const value ) {
        auto const cardinality{ values.size() };
        if ( cardinality == 0 || values.back() < value ) {
            values.push_back( value );
            sync_header();
            return true;
        }

        auto const * const begin{ values.data() };
        auto const loc{ std::lower_bound( begin, begin + static_cast<std::ptrdiff_t>( cardinality ), value ) };
        if ( loc != begin + static_cast<std::ptrdiff_t>( cardinality ) && *loc == value ) {
            return false;
        }
        auto const insert_idx{ static_cast<std::size_t>( loc - begin ) };
        values.resize_uninitialized( cardinality + 1 );
        auto * const data{ values.data() };
        std::memmove(
            data + insert_idx + 1,
            data + insert_idx,
            ( cardinality - insert_idx ) * sizeof( low_type )
        );
        data[ insert_idx ] = value;
        sync_header();
        return true;
    }

    [[nodiscard]] bool remove( low_type const value ) {
        auto const cardinality{ values.size() };
        if ( cardinality == 0 || values.back() < value ) {
            return false;
        }
        auto * const begin{ values.data() };
        auto * const end{ begin + static_cast<std::ptrdiff_t>( cardinality ) };
        auto * const loc{ std::lower_bound( begin, end, value ) };
        if ( loc == end || *loc != value ) {
            return false;
        }
        auto const remove_idx{ static_cast<std::size_t>( loc - begin ) };
        std::memmove(
            begin + remove_idx,
            begin + remove_idx + 1,
            ( cardinality - remove_idx - 1 ) * sizeof( low_type )
        );
        values.resize_uninitialized( cardinality - 1 );
        sync_header();
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return values.size(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept {
        if ( values.empty() ) {
            return std::nullopt;
        }
        return values.front();
    }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        auto const pos{ std::upper_bound( values.begin(), values.end(), value ) };
        if ( pos == values.end() ) {
            return std::nullopt;
        }
        return *pos;
    }

    template <typename F>
    void for_each( F && f ) const {
        for ( auto const value : values ) {
            f( value );
        }
    }

    // Restores the header invariants (cardinality == count, endpoints ==
    // front/back) after any direct mutation of `values`. Every raw-payload
    // write site (resize_uninitialized + fill, assign, …) must end with this.
    void sync_header() noexcept {
        auto const n{ values.size() };
        handle_->set_cardinality( n );
        if ( n != 0 ) {
            handle_->set_endpoints( values.front(), values.back() );
        }
    }

    payload_vector<Layout, low_type, CowPolicy> values;

private:
    container_handle<Layout, CowPolicy> * handle_;
};

// ---- run refs -------------------------------------------------------------------
//
// Runs are sorted, coalesced, non-adjacent closed intervals; count == #runs,
// cardinality == Σ run lengths, endpoints == front().begin / back().end.

template <typename Layout, typename CowPolicy>
class run_cref {
public:
    using low_type = typename Layout::low_type;
    using run_type = ::frsr::roaring::run<low_type>;

    explicit run_cref( container_handle<Layout, CowPolicy> const & handle ) noexcept : runs{ handle }, handle_{ &handle } {}

    [[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] bool contains( low_type const value ) const noexcept {
        // interleaved_binary_search returns the run index when value exactly equals a run's
        // begin, else -(insertion_point + 1). A run covers the closed interval [begin, end],
        // so on an inexact match we must still test the run immediately before the insertion
        // point (the last run with begin < value) against its end.
        // [croaring-ref] deps/croaring/src/containers/run.c:run_container_contains
        auto index{ ::frsr::roaring::interleaved_binary_search( runs.data(), static_cast<int32_t>( runs.size() ), value ) };
        if ( index >= 0 ) {
            return true;
        }
        index = -index - 2;
        if ( index < 0 ) {
            return false;
        }
        return value <= runs[ static_cast<std::uint32_t>( index ) ].end;
    }

    [[nodiscard]] std::size_t size() const noexcept { return handle_->cardinality(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept {
        if ( runs.empty() ) {
            return std::nullopt;
        }
        return runs.front().begin;
    }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        auto const it{ std::lower_bound(
            runs.begin(), runs.end(), value,
            []( run_type const & current, low_type const v ) noexcept { return current.end < v; }
        ) };
        if ( it == runs.end() ) {
            return std::nullopt;
        }
        if ( value < it->begin ) {
            return it->begin;
        }
        if ( value < it->end && value != std::numeric_limits<low_type>::max() ) {
            return static_cast<low_type>( value + 1U );
        }
        auto next{ it + 1 };
        if ( next == runs.end() ) {
            return std::nullopt;
        }
        return next->begin;
    }

    template <typename F>
    void for_each( F && f ) const {
        for ( auto const & current : runs ) {
            for ( auto value{ current.begin }; ; ++value ) {
                f( value );
                if ( value == current.end ) {
                    break;
                }
            }
        }
    }

    payload_vector_const_view<Layout, run_type, CowPolicy> runs;

private:
    container_handle<Layout, CowPolicy> const * handle_;
};

template <typename Layout, typename CowPolicy>
class run_ref {
public:
    using low_type = typename Layout::low_type;
    using run_type = ::frsr::roaring::run<low_type>;
    using cref     = run_cref<Layout, CowPolicy>;

    explicit run_ref( container_handle<Layout, CowPolicy> & handle ) noexcept
        : runs{ handle }, cardinality{ handle }, handle_{ &handle } {}

    [[nodiscard]] operator cref() const noexcept { return cref{ *handle_ }; }

    [[nodiscard]] bool contains( low_type const value ) const noexcept {
        return cref{ *handle_ }.contains( value );
    }

    [[nodiscard]] bool add( low_type const value ) {
        auto const previous_size{ handle_->cardinality() };
        add_closed_range( value, value );
        return handle_->cardinality() != previous_size;
    }

    [[nodiscard]] bool remove( low_type const value ) {
        auto const previous_size{ handle_->cardinality() };
        remove_closed_range( value, value );
        return handle_->cardinality() != previous_size;
    }

    void add_closed_range( low_type const begin, low_type const end ) {
        if ( begin > end ) {
            return;
        }

        ::frsr::roaring::detail::heap_vector<run_type> merged;
        merged.reserve( static_cast<std::uint32_t>( runs.size() + 1U ) );

        auto new_begin{ begin };
        auto new_end{ end };
        auto inserted{ false };
        auto const separated_before{ []( run_type const & current, low_type const next_begin ) noexcept {
            return static_cast<std::size_t>( current.end ) + 1U < static_cast<std::size_t>( next_begin );
        } };
        auto const separated_after{ []( low_type const previous_end, run_type const & current ) noexcept {
            return static_cast<std::size_t>( previous_end ) + 1U < static_cast<std::size_t>( current.begin );
        } };

        for ( auto const & current : runs ) {
            if ( separated_before( current, begin ) ) {
                merged.push_back( current );
                continue;
            }
            if ( separated_after( end, current ) ) {
                if ( !inserted ) {
                    merged.push_back( run_type{ new_begin, new_end } );
                    inserted = true;
                }
                merged.push_back( current );
                continue;
            }
            new_begin = std::min( new_begin, current.begin );
            new_end = std::max( new_end, current.end );
        }

        if ( !inserted ) {
            merged.push_back( run_type{ new_begin, new_end } );
        }

        runs.assign( { merged.data(), merged.size() } );
        sync_header();
    }

    void remove_closed_range( low_type const begin, low_type const end ) {
        if ( begin > end || runs.empty() ) {
            return;
        }

        ::frsr::roaring::detail::heap_vector<run_type> updated;
        updated.reserve( static_cast<std::uint32_t>( runs.size() + 1U ) );
        for ( auto const & current : runs ) {
            if ( current.end < begin || end < current.begin ) {
                updated.push_back( current );
                continue;
            }

            if ( current.begin < begin ) {
                updated.push_back( run_type{ current.begin, static_cast<low_type>( begin - 1U ) } );
            }
            if ( end < current.end && end != std::numeric_limits<low_type>::max() ) {
                updated.push_back( run_type{ static_cast<low_type>( end + 1U ), current.end } );
            }
        }

        runs.assign( { updated.data(), updated.size() } );
        sync_header();
    }

    [[nodiscard]] std::size_t size() const noexcept { return handle_->cardinality(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept { return cref{ *handle_ }.first(); }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        return cref{ *handle_ }.next_after( value );
    }

    template <typename F>
    void for_each( F && f ) const { cref{ *handle_ }.for_each( std::forward<F>( f ) ); }

    void from_sorted_values( std::span<low_type const> const sorted_values ) {
        runs.clear();
        if ( sorted_values.empty() ) {
            sync_header();
            return;
        }

        auto current_begin{ sorted_values.front() };
        auto current_end{ current_begin };
        for ( auto index{ std::size_t{ 1 } }; index < sorted_values.size(); ++index ) {
            auto const value{ sorted_values[ index ] };
            if ( value == current_end || ( current_end != std::numeric_limits<low_type>::max() && value == static_cast<low_type>( current_end + 1U ) ) ) {
                current_end = value;
                continue;
            }
            runs.push_back( run_type{ current_begin, current_end } );
            current_begin = value;
            current_end = value;
        }
        runs.push_back( run_type{ current_begin, current_end } );
        sync_header();
    }

    // Restores the header invariants (cardinality == Σ run lengths, endpoints
    // == front().begin / back().end) after any direct mutation of `runs`.
    void sync_header() noexcept {
        auto total{ std::uint32_t{ 0 } };
        for ( auto const & current : runs ) {
            total += static_cast<std::uint32_t>(
                static_cast<std::size_t>( current.end ) - static_cast<std::size_t>( current.begin ) + 1U
            );
        }
        handle_->set_cardinality( total );
        if ( !runs.empty() ) {
            handle_->set_endpoints( runs.front().begin, runs.back().end );
        }
    }

    payload_vector<Layout, run_type, CowPolicy> runs;
    header_cardinality_proxy<Layout, CowPolicy> cardinality;

private:
    container_handle<Layout, CowPolicy> * handle_;
};

// ---- bitset refs ------------------------------------------------------------------
//
// The word block is always spilled (8 KB for the u16 layout); count ==
// word_count. Single add/remove maintain cardinality and endpoints in O(1)
// amortized; bulk kernels write words directly and then restore the header
// via set_cardinality + sync_endpoints.

template <typename Layout, typename CowPolicy = cow_value_semantics>
class bitset_words_view {
public:
    using word_array = std::array<std::uint64_t, Layout::word_count>;

    explicit bitset_words_view( container_handle<Layout, CowPolicy> & handle ) noexcept
        : words_{ handle.template payload_data<std::uint64_t>() } {}

    [[nodiscard]] std::uint64_t & operator[]( std::size_t const i ) noexcept { return words_[ i ]; }
    [[nodiscard]] std::uint64_t const & operator[]( std::size_t const i ) const noexcept { return words_[ i ]; }

    [[nodiscard]] static constexpr std::size_t size() noexcept { return Layout::word_count; }

    [[nodiscard]] std::uint64_t       * begin()       noexcept { return words_; }
    [[nodiscard]] std::uint64_t const * begin() const noexcept { return words_; }
    [[nodiscard]] std::uint64_t       * end  ()       noexcept { return words_ + size(); }
    [[nodiscard]] std::uint64_t const * end  () const noexcept { return words_ + size(); }

    // The payload block is exactly word_count u64s, so viewing it as the
    // std::array the word-level kernels take is layout-safe.
    [[nodiscard]] word_array       & as_array()       noexcept { return *reinterpret_cast<word_array       *>( words_ ); }
    [[nodiscard]] word_array const & as_array() const noexcept { return *reinterpret_cast<word_array const *>( words_ ); }

private:
    std::uint64_t * words_;
};

template <typename Layout, typename CowPolicy = cow_value_semantics>
class bitset_words_cview {
public:
    using word_array = std::array<std::uint64_t, Layout::word_count>;

    explicit bitset_words_cview( container_handle<Layout, CowPolicy> const & handle ) noexcept
        : words_{ handle.template payload_data<std::uint64_t>() } {}

    [[nodiscard]] std::uint64_t const & operator[]( std::size_t const i ) const noexcept { return words_[ i ]; }

    [[nodiscard]] static constexpr std::size_t size() noexcept { return Layout::word_count; }

    [[nodiscard]] std::uint64_t const * begin() const noexcept { return words_; }
    [[nodiscard]] std::uint64_t const * end  () const noexcept { return words_ + size(); }

    [[nodiscard]] word_array const & as_array() const noexcept { return *reinterpret_cast<word_array const *>( words_ ); }

private:
    std::uint64_t const * words_;
};

template <typename Layout, typename CowPolicy>
class bitset_cref {
public:
    using low_type = typename Layout::low_type;

    explicit bitset_cref( container_handle<Layout, CowPolicy> const & handle ) noexcept : words{ handle }, handle_{ &handle } {}

    // Matches array_cref::contains / run_cref::contains: this is the hot per-element
    // membership test inside filter_array_bitset's array∩bitset / array-andnot-bitset
    // loop, and must be force-inlined there for the same reason those are — without it,
    // the call crosses a real function boundary (bitset_words_cview::operator[] +
    // container_handle::payload_data indirection re-derived per call) instead of
    // collapsing to a single word load + mask.
    [[nodiscard]] [[gnu::hot]] [[gnu::always_inline]] bool contains( low_type const value ) const noexcept {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        return ( words[ word_index ] & ( std::uint64_t{ 1 } << bit_index ) ) != 0;
    }

    [[nodiscard]] std::size_t size() const noexcept { return handle_->cardinality(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept {
        for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
            auto const word{ words[ index ] };
            if ( word != 0 ) {
                return static_cast<low_type>( ( index << 6U ) + std::countr_zero( word ) );
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        auto word_index{ static_cast<std::size_t>( value ) >> 6U };
        if ( word_index >= words.size() ) {
            return std::nullopt;
        }

        auto bit_index{ static_cast<unsigned>( value ) & 63U };
        auto word{ words[ word_index ] };
        auto const consumed_mask{
            bit_index == 63U
                ? std::numeric_limits<std::uint64_t>::max()
                : ( std::uint64_t{ 1 } << ( bit_index + 1U ) ) - 1U
        };
        word &= ~consumed_mask;
        if ( word != 0 ) {
            return static_cast<low_type>( ( word_index << 6U ) + std::countr_zero( word ) );
        }

        for ( ++word_index; word_index < words.size(); ++word_index ) {
            word = words[ word_index ];
            if ( word != 0 ) {
                return static_cast<low_type>( ( word_index << 6U ) + std::countr_zero( word ) );
            }
        }
        return std::nullopt;
    }

    template <typename F>
    void for_each( F && f ) const {
        for ( std::size_t index{ 0 }; index < words.size(); ++index ) {
            auto word{ words[ index ] };
            while ( word != 0 ) {
                auto const bit{ std::countr_zero( word ) };
                f( static_cast<low_type>( ( index << 6U ) + bit ) );
                word &= word - 1U;
            }
        }
    }

    bitset_words_cview<Layout, CowPolicy> words;

private:
    container_handle<Layout, CowPolicy> const * handle_;
};

template <typename Layout, typename CowPolicy>
class bitset_ref {
public:
    using low_type = typename Layout::low_type;
    using cref     = bitset_cref<Layout, CowPolicy>;

    explicit bitset_ref( container_handle<Layout, CowPolicy> & handle ) noexcept
        : words{ handle }, cardinality{ handle }, handle_{ &handle } {}

    [[nodiscard]] operator cref() const noexcept { return cref{ *handle_ }; }

    [[nodiscard]] bool contains( low_type const value ) const noexcept {
        return cref{ *handle_ }.contains( value );
    }

    [[nodiscard]] bool add( low_type const value ) noexcept {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        auto & word{ words[ word_index ] };
        auto const mask{ std::uint64_t{ 1 } << bit_index };
        if ( ( word & mask ) != 0 ) {
            return false;
        }
        word |= mask;
        auto const previous_cardinality{ handle_->cardinality() };
        handle_->set_cardinality( previous_cardinality + 1U );
        if ( previous_cardinality == 0 ) {
            handle_->set_endpoints( value, value );
        } else if ( handle_->endpoints_valid() ) {
            handle_->set_endpoints(
                std::min( handle_->min_value(), value ),
                std::max( handle_->max_value(), value )
            );
        }
        return true;
    }

    [[nodiscard]] bool remove( low_type const value ) noexcept {
        auto const word_index{ static_cast<std::size_t>( value ) >> 6U };
        auto const bit_index{ static_cast<unsigned>( value ) & 63U };
        auto & word{ words[ word_index ] };
        auto const mask{ std::uint64_t{ 1 } << bit_index };
        if ( ( word & mask ) == 0 ) {
            return false;
        }
        word &= ~mask;
        handle_->set_cardinality( handle_->cardinality() - 1U );
        if (
            !handle_->empty() && handle_->endpoints_valid() &&
            ( value == handle_->min_value() || value == handle_->max_value() )
        ) {
            handle_->mark_endpoints_stale();
        }
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return handle_->cardinality(); }

    [[nodiscard]] std::optional<low_type> first() const noexcept { return cref{ *handle_ }.first(); }

    [[nodiscard]] std::optional<low_type> next_after( low_type const value ) const noexcept {
        return cref{ *handle_ }.next_after( value );
    }

    template <typename F>
    void for_each( F && f ) const { cref{ *handle_ }.for_each( std::forward<F>( f ) ); }

    // Marks the header endpoints stale after bulk word writes; the next
    // min_value()/max_value() read recomputes them.
    void mark_endpoints_stale() noexcept { handle_->mark_endpoints_stale(); }

    // Marks cardinality unknown after a nocard lazy word OR (repair must popcount).
    void mark_cardinality_stale() noexcept { handle_->mark_cardinality_stale(); }

    // Eagerly recomputes the [min,max] header endpoints from the word block.
    void sync_endpoints() noexcept {
        if ( handle_->cardinality() == 0 ) {
            return;
        }
        auto const & w{ words };
        std::size_t front{ 0 };
        while ( w[ front ] == 0 ) {
            ++front;
        }
        std::size_t back{ w.size() - 1 };
        while ( w[ back ] == 0 ) {
            --back;
        }
        handle_->set_endpoints(
            static_cast<low_type>( ( front << 6U ) + static_cast<unsigned>( std::countr_zero( w[ front ] ) ) ),
            static_cast<low_type>( ( back  << 6U ) + ( 63U - static_cast<unsigned>( std::countl_zero( w[ back ] ) ) ) )
        );
    }

    bitset_words_view<Layout, CowPolicy>        words;
    header_cardinality_proxy<Layout, CowPolicy> cardinality;

private:
    container_handle<Layout, CowPolicy> * handle_;
};

// ---- deferred container_handle member definitions ---------------------------------

template <typename Layout, typename CowPolicy>
array_ref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_array() noexcept( !CowPolicy::refcounted ) {
    make_payload_unique();
    return array_ref<Layout, CowPolicy>{ *this };
}
template <typename Layout, typename CowPolicy>
array_cref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_array() const noexcept { return array_cref<Layout, CowPolicy>{ *this }; }
template <typename Layout, typename CowPolicy>
run_ref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_run() noexcept( !CowPolicy::refcounted ) {
    make_payload_unique();
    return run_ref<Layout, CowPolicy>{ *this };
}
template <typename Layout, typename CowPolicy>
run_cref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_run() const noexcept { return run_cref<Layout, CowPolicy>{ *this }; }
template <typename Layout, typename CowPolicy>
bitset_ref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_bitset() noexcept( !CowPolicy::refcounted ) {
    make_payload_unique();
    return bitset_ref<Layout, CowPolicy>{ *this };
}
template <typename Layout, typename CowPolicy>
bitset_cref<Layout, CowPolicy> container_handle<Layout, CowPolicy>::as_bitset() const noexcept { return bitset_cref<Layout, CowPolicy>{ *this }; }

template <typename Layout, typename CowPolicy>
template <typename F>
decltype( auto ) container_handle<Layout, CowPolicy>::visit( F && f ) {
    switch ( kind_ ) {
        case container_kind::array : return f( as_array () );
        case container_kind::run   : return f( as_run   () );
        case container_kind::bitset: return f( as_bitset() );
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
template <typename F>
decltype( auto ) container_handle<Layout, CowPolicy>::visit( F && f ) const {
    switch ( kind_ ) {
        case container_kind::array : return f( as_array () );
        case container_kind::run   : return f( as_run   () );
        case container_kind::bitset: return f( as_bitset() );
    }
    std::unreachable();
}

template <typename Layout, typename CowPolicy>
container_handle<Layout, CowPolicy> container_handle<Layout, CowPolicy>::make_array_from_sorted( std::span<low_type const> const sorted_values ) {
    container_handle handle;
    auto array{ handle.as_array() };
    array.values.assign( sorted_values );
    array.sync_header();
    return handle;
}

template <typename Layout, typename CowPolicy>
container_handle<Layout, CowPolicy> container_handle<Layout, CowPolicy>::make_bitset_from_words( word_array const & words ) {
    auto handle{ make_bitset_uninitialized() };
    std::memcpy( handle.payload_data_raw(), words.data(), sizeof( word_array ) );
    auto cardinality{ std::uint32_t{ 0 } };
    for ( auto const word : words ) {
        cardinality += static_cast<std::uint32_t>( std::popcount( word ) );
    }
    handle.set_cardinality( cardinality );
    handle.mark_endpoints_stale();
    return handle;
}

template <typename Layout, typename CowPolicy>
void container_handle<Layout, CowPolicy>::ensure_endpoints() const noexcept {
    if ( endpoints_valid() || cardinality_ == 0 ) {
        return;
    }
    // Only bitsets go stale (bulk word kernels defer the scan); array/run
    // mutation keeps the endpoints exact.
    auto const * const words{ payload_data<std::uint64_t>() };
    std::size_t front{ 0 };
    while ( words[ front ] == 0 ) {
        ++front;
    }
    std::size_t back{ Layout::word_count - 1 };
    while ( words[ back ] == 0 ) {
        --back;
    }
    min_ = static_cast<low_type>( ( front << 6U ) + static_cast<unsigned>( std::countr_zero( words[ front ] ) ) );
    max_ = static_cast<low_type>( ( back  << 6U ) + ( 63U - static_cast<unsigned>( std::countl_zero( words[ back ] ) ) ) );
    flags_ &= static_cast<std::uint8_t>( ~endpoints_stale_flag );
}

// Two-container dispatch as one flat switch( lhs_kind * 3 + rhs_kind ) — the 2-D
// analog of visit() (replaces the former 2-D std::visit). A single function (not
// nested visit() calls) so the 9 arms are all dependent on both refs.
template <typename Layout, typename CowPolicy, typename F>
decltype( auto ) visit_container_pair(
    F && f,
    container_handle<Layout, CowPolicy> const & lhs,
    container_handle<Layout, CowPolicy> const & rhs
) {
    auto const combined{ static_cast<unsigned>( lhs.kind() ) * 3U + static_cast<unsigned>( rhs.kind() ) };
    switch ( combined ) {
        case 0: return f( lhs.as_array (), rhs.as_array () );
        case 1: return f( lhs.as_array (), rhs.as_run   () );
        case 2: return f( lhs.as_array (), rhs.as_bitset() );
        case 3: return f( lhs.as_run   (), rhs.as_array () );
        case 4: return f( lhs.as_run   (), rhs.as_run   () );
        case 5: return f( lhs.as_run   (), rhs.as_bitset() );
        case 6: return f( lhs.as_bitset(), rhs.as_array () );
        case 7: return f( lhs.as_bitset(), rhs.as_run   () );
        case 8: return f( lhs.as_bitset(), rhs.as_bitset() );
    }
    std::unreachable();
}

static_assert( sizeof( container_handle<default_layout<std::uint32_t>> ) == container_handle<default_layout<std::uint32_t>>::handle_size );
static_assert( sizeof( container_handle<default_layout<std::uint64_t>> ) == container_handle<default_layout<std::uint64_t>>::handle_size );
static_assert( container_handle<default_layout<std::uint32_t>>::inline_capacity<std::uint16_t> >= 8 ); // the sparse-regime SBO win — HARD constraint
static_assert( container_handle<default_layout<std::uint32_t>>::is_trivially_moveable );

// The CoW policy shapes only the payload prefix and copy behavior — never the handle itself.
static_assert( sizeof( container_handle<default_layout<std::uint32_t>, cow_atomic_refcount        > ) == 32 );
static_assert( sizeof( container_handle<default_layout<std::uint32_t>, cow_unsynchronized_refcount> ) == 32 );
static_assert( container_handle<default_layout<std::uint32_t>, cow_atomic_refcount>::is_trivially_moveable );

} // namespace frsr::roaring::detail

