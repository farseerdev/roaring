#pragma once

// Copy-on-write policy for the container payload storage.
//
// Selects how a container_handle's *spilled* (heap) payload is duplicated when
// the handle is copied, and — orthogonally — the shape of the refcount word
// that precedes such a payload. The policy is a compile-time trait so LTO folds
// the unused model's branches to nothing: Model 1 (the default) carries no
// refcount word and deep-clones on copy, exactly the pre-Phase-2 behavior.
//
//   Model 1 — value semantics      : unique payloads, deep-clone on copy.
//   Model 2 — atomic-refcount CoW   : born-shared payloads, ++rc on copy,
//                                     clone-on-write; thread-safe.
//   Model 3 — non-atomic-refcount   : same, plain u32 rc; single-threaded.
//
// A refcounted payload is laid out as [ rc_word ][ payload ] in one allocation;
// the handle's spill pointer addresses the payload, so payload accessors never
// see the prefix. Only the allocate / free / clone plumbing, keyed on the
// compile-time rc_prefix_bytes constant, knows about it.
//
// See the container-representation design notes, "CoW models as a
// zero-overhead Policy".

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

namespace frsr::roaring::detail {

// Model 1 — value semantics (the default; no refcount word, deep-clone on copy).
struct cow_value_semantics {
    static constexpr bool        refcounted     { false };
    static constexpr std::size_t rc_prefix_bytes{ 0 };
};

// Model 2 — atomic-refcount CoW, thread-safe. Spilled payloads are born shared
// (rc = 1) and are never promoted in place: copying a handle only ever touches
// the refcount, so concurrent readers of the source handle are undisturbed.
struct cow_atomic_refcount {
    static constexpr bool        refcounted     { true };
    // 8 so the payload keeps the word block's 8-byte alignment.
    static constexpr std::size_t rc_prefix_bytes{ 8 };

    using rc_word = std::atomic<std::uint32_t>;
    static_assert( sizeof( rc_word ) <= rc_prefix_bytes );

    static void rc_construct( void * const rc_slot ) noexcept { new ( rc_slot ) rc_word{ 1 }; }

    // acquire so a sole-referent (== 1) observation synchronizes with the
    // release-decrement of the previous co-owner before we mutate in place.
    [[nodiscard]] static std::uint32_t rc_load( void const * const rc_slot ) noexcept {
        return std::launder( static_cast<rc_word const *>( rc_slot ) )->load( std::memory_order_acquire );
    }

    static void rc_increment( void * const rc_slot ) noexcept {
        std::launder( static_cast<rc_word *>( rc_slot ) )->fetch_add( 1, std::memory_order_relaxed );
    }

    // release-decrement + acquire fence on the last release: the freeing thread
    // must observe every co-owner's writes before the payload is reused.
    [[nodiscard]] static bool rc_decrement_is_last( void * const rc_slot ) noexcept {
        auto & rc{ *std::launder( static_cast<rc_word *>( rc_slot ) ) };
        if ( rc.fetch_sub( 1, std::memory_order_release ) == 1 ) {
            std::atomic_thread_fence( std::memory_order_acquire );
            return true;
        }
        return false;
    }
};

// Model 3 — non-atomic refcount: same born-shared sharing structure as Model 2
// with a plain counter word. Single-threaded (or externally synchronized) use only.
struct cow_unsynchronized_refcount {
    static constexpr bool        refcounted     { true };
    static constexpr std::size_t rc_prefix_bytes{ 8 };

    using rc_word = std::uint32_t;

    static void rc_construct( void * const rc_slot ) noexcept { new ( rc_slot ) rc_word{ 1 }; }

    [[nodiscard]] static std::uint32_t rc_load( void const * const rc_slot ) noexcept {
        return *std::launder( static_cast<rc_word const *>( rc_slot ) );
    }

    static void rc_increment( void * const rc_slot ) noexcept {
        ++*std::launder( static_cast<rc_word *>( rc_slot ) );
    }

    [[nodiscard]] static bool rc_decrement_is_last( void * const rc_slot ) noexcept {
        return --*std::launder( static_cast<rc_word *>( rc_slot ) ) == 0;
    }
};

} // namespace frsr::roaring::detail
