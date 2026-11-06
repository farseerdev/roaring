#pragma once

// shared_bitmap<Bitmap>: a counted handle to a bitmap. The copies of a handle share one bitmap, which none of them
// changes while it is shared: the first one to change it takes a copy of its own first (copy on write), and a change
// that would leave the bitmap as it is is found out first and leaves it shared. Copying a handle costs one count
// whatever the bitmap holds, and a reader may keep a handle to a bitmap that a writer goes on changing through its own
// copy of the handle.
//
// - The count is atomic, unless the bitmap's copy-on-write policy is cow_unsynchronized_refcount, which is for one
//   thread at a time (as the bitmap itself is). With an atomic count any number of threads may copy, read, change and
//   drop handles that share a bitmap, each using handle objects of its own.
// - A handle holds the cardinality of its bitmap next to the pointer, so size() and empty() read the handle only. A
//   shared bitmap does not change, and a handle refreshes it whenever it changes its own, also when the change fails.
// - A default-constructed or moved-from handle holds no bitmap and reads as an empty one; nothing is allocated for it
//   until it is changed. clear() gives the bitmap up instead of emptying it.
// - A change that has to copy the bitmap first leaves the handle (and the bitmap it shares) as it was when the copy
//   fails; past the copy the bitmap's own guarantees apply (README, "Exceptions").
//
// bitmap knows nothing of the handle: a bitmap that is not put in one pays nothing for it.

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/cow_policy.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace frsr::roaring {

template <typename Bitmap>
class shared_bitmap {
public:
    using bitmap_type = Bitmap;
    using key_type    = typename Bitmap::key_type;
    using size_type   = typename Bitmap::size_type;

    class editor;

    shared_bitmap() noexcept = default;
    explicit shared_bitmap( Bitmap const & value ) noexcept( detail::allocation_failure_terminates )
        : block_{ make_block( value ) }, size_{ block_->value.size() } {}
    explicit shared_bitmap( Bitmap && value ) noexcept( detail::allocation_failure_terminates )
        : block_{ make_block( std::move( value ) ) }, size_{ block_->value.size() } {}

    shared_bitmap( shared_bitmap const & other ) noexcept : block_{ other.block_ }, size_{ other.size_ } {
        if ( block_ != nullptr ) {
            count_policy::rc_increment( block_->count );
        }
    }
    shared_bitmap( shared_bitmap && other ) noexcept
        : block_{ std::exchange( other.block_, nullptr ) }, size_{ std::exchange( other.size_, 0 ) } {}

    shared_bitmap & operator=( shared_bitmap const & other ) noexcept {
        shared_bitmap{ other }.swap( *this );
        return *this;
    }
    shared_bitmap & operator=( shared_bitmap && other ) noexcept {
        shared_bitmap{ std::move( other ) }.swap( *this );
        return *this;
    }

    ~shared_bitmap() { release( block_ ); }

    void swap( shared_bitmap & other ) noexcept {
        std::swap( block_, other.block_ );
        std::swap( size_, other.size_ );
    }
    friend void swap( shared_bitmap & left, shared_bitmap & right ) noexcept { left.swap( right ); }

    // ---- reads ------------------------------------------------------------------------------------------------------

    [[nodiscard]] Bitmap const & get() const noexcept { return block_ != nullptr ? block_->value : empty_.value; }
    [[nodiscard]] Bitmap const & operator*() const noexcept { return get(); }
    [[nodiscard]] Bitmap const * operator->() const noexcept { return &get(); }

    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] size_type container_count() const noexcept { return block_ != nullptr ? block_->value.container_count() : 0; }
    [[nodiscard]] bool contains( key_type const value ) const noexcept { return block_ != nullptr && block_->value.contains( value ); }

    // Whether no other handle shares this handle's bitmap (a handle that holds none is unique).
    [[nodiscard]] bool is_unique() const noexcept { return block_ == nullptr || count_policy::rc_load( block_->count ) == 1; }
    // Whether the two handles share one bitmap.
    [[nodiscard]] bool shares_storage_with( shared_bitmap const & other ) const noexcept { return block_ != nullptr && block_ == other.block_; }

    // ---- changes ----------------------------------------------------------------------------------------------------
    // Each makes the bitmap the handle's own (a copy, if it is shared) only when it changes it; of a shared bitmap it
    // asks first. An operand may be the handle's own bitmap (h -= h.get()).

    // Whether `value` was added (it was not held).
    [[nodiscard]] bool add( key_type const value ) noexcept( detail::allocation_failure_terminates ) {
        if ( !is_unique() && block_->value.contains( value ) ) {
            return false;
        }
        bool added{ false };
        change_own( [ & ]( Bitmap & own ) { added = own.add( value ); } );
        return added;
    }

    // Whether `value` was removed (it was held).
    [[nodiscard]] bool remove( key_type const value ) noexcept( detail::allocation_failure_terminates ) {
        if ( block_ == nullptr || ( !is_unique() && !block_->value.contains( value ) ) ) {
            return false;
        }
        bool removed{ false };
        change_own( [ & ]( Bitmap & own ) { removed = own.remove( value ); } );
        return removed;
    }

    // A handle that holds nothing takes the bitmap of the handle it is united or exclusive-ored with (shares it).
    shared_bitmap & operator|=( shared_bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( other.empty() || shares_storage_with( other ) ) {
            return *this;
        }
        if ( empty() ) {
            return *this = other;
        }
        return unite( other.get() );
    }
    shared_bitmap & operator|=( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( other.empty() ) {
            return *this;
        }
        return unite( other );
    }

    shared_bitmap & operator&=( shared_bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( shares_storage_with( other ) ) {
            return *this;
        }
        return intersect( other.get() );
    }
    shared_bitmap & operator&=( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return intersect( other ); }

    shared_bitmap & operator-=( shared_bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( shares_storage_with( other ) ) {
            clear();
            return *this;
        }
        return subtract( other.get() );
    }
    shared_bitmap & operator-=( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) { return subtract( other ); }

    shared_bitmap & operator^=( shared_bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( other.empty() ) {
            return *this;
        }
        if ( shares_storage_with( other ) ) {
            clear();
            return *this;
        }
        if ( empty() ) {
            return *this = other;
        }
        return change_own( [ & ]( Bitmap & own ) { own ^= other.get(); } );
    }
    shared_bitmap & operator^=( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( other.empty() ) {
            return *this;
        }
        return change_own( [ & ]( Bitmap & own ) { own ^= other; } );
    }

    // Gives the bitmap up (it is freed if no other handle shares it).
    void clear() noexcept {
        release( std::exchange( block_, nullptr ) );
        size_ = 0;
    }

    // Any other change: the bitmap made the handle's own (copied first if it is shared, or made for a handle that holds
    // none), to change through the editor.
    [[nodiscard]] editor edit() noexcept( detail::allocation_failure_terminates ) {
        release( take_own() );
        return editor{ *this };
    }

private:
    // The count of a block: plain for the policy that is for one thread at a time, atomic for the others.
    using count_policy = std::conditional_t<
        std::is_same_v<typename Bitmap::cow_policy, detail::cow_unsynchronized_refcount>,
        detail::cow_unsynchronized_refcount,
        detail::cow_atomic_refcount
    >;

    struct block {
        alignas( std::uint64_t ) std::byte count[ count_policy::rc_prefix_bytes ];
        Bitmap value;

        template <typename... Args>
        explicit block( Args &&... args ) : value( std::forward<Args>( args )... ) { count_policy::rc_construct( count ); }
    };

    template <typename... Args>
    [[nodiscard]] static block * make_block( Args &&... args ) { return new block{ std::forward<Args>( args )... }; }

    static void release( block * const held ) noexcept {
        if ( held != nullptr && count_policy::rc_decrement_is_last( held->count ) ) {
            delete held;
        }
    }

    // Makes the bitmap the handle's own: one is made for a handle that holds none, and a shared one is copied. When that
    // fails the handle is as it was. Returns the shared block it copied (else nullptr), whose count the handle still
    // holds for the caller to release once nothing reads it any more.
    [[nodiscard]] block * take_own() noexcept( detail::allocation_failure_terminates ) {
        if ( block_ == nullptr ) {
            block_ = make_block();
            return nullptr;
        }
        if ( is_unique() ) {
            return nullptr;
        }
        return std::exchange( block_, make_block( std::as_const( block_->value ) ) );
    }

    void refresh() noexcept { size_ = block_ != nullptr ? block_->value.size() : 0; }

    // Ends a change of the handle's own bitmap, also one that fails part way: the handle is refreshed and the bitmap it
    // shared before (which an operand may be) is let go.
    class change_end {
    public:
        change_end( shared_bitmap & handle, block * const left ) noexcept : handle_{ handle }, left_{ left } {}
        change_end( change_end const & ) = delete;
        ~change_end() {
            handle_.refresh();
            release( left_ );
        }

    private:
        shared_bitmap & handle_;
        block *         left_;
    };

    // `change` applied to the handle's own bitmap.
    template <typename Change>
    shared_bitmap & change_own( Change const & change ) noexcept( detail::allocation_failure_terminates ) {
        change_end const end{ *this, take_own() };
        change( block_->value );
        return *this;
    }

    shared_bitmap & unite( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( !is_unique() && other.is_subset_of( block_->value ) ) {
            return *this;
        }
        return change_own( [ & ]( Bitmap & own ) { own |= other; } );
    }

    shared_bitmap & intersect( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( empty() ) {
            return *this;
        }
        if ( other.empty() ) {
            clear();
            return *this;
        }
        if ( !is_unique() && block_->value.is_subset_of( other ) ) {
            return *this;
        }
        return change_own( [ & ]( Bitmap & own ) { own &= other; } );
    }

    shared_bitmap & subtract( Bitmap const & other ) noexcept( detail::allocation_failure_terminates ) {
        if ( empty() || other.empty() ) {
            return *this;
        }
        if ( !is_unique() && !block_->value.intersects( other ) ) {
            return *this;
        }
        return change_own( [ & ]( Bitmap & own ) { own -= other; } );
    }

    // What a handle that holds no bitmap reads: built at compile time and never destroyed, so it may be read at any
    // time, also from the destructor of a static object.
    union empty_storage {
        Bitmap value;
        constexpr empty_storage() : value{} {}
        ~empty_storage() {}
    };
    static constinit inline empty_storage const empty_{};

    block *   block_{ nullptr };
    size_type size_{ 0 };
}; // class shared_bitmap

// Write access to the bitmap of a handle, which edit() made the handle's own; the handle refreshes what it holds of the
// bitmap (its cardinality) when the editor goes. While an editor is alive its handle is not copied, cleared or assigned
// (a copy would share the bitmap the editor writes; asserted), as no copy of a bitmap is taken while a bulk_context on
// it is alive, and an edit leaves the bitmap with its counts settled (a lazy fold begun through the editor is finished
// through it).
template <typename Bitmap>
class shared_bitmap<Bitmap>::editor {
public:
    editor( editor const & ) = delete;
    editor & operator=( editor const & ) = delete;
    ~editor() {
        assert( handle_->block_ != nullptr && handle_->is_unique() && "an editor's handle is not copied, cleared or assigned" );
        handle_->refresh();
    }

    [[nodiscard]] Bitmap & operator*() const noexcept {
        assert( handle_->block_ != nullptr && handle_->is_unique() && "an editor's handle is not copied, cleared or assigned" );
        return handle_->block_->value;
    }
    [[nodiscard]] Bitmap * operator->() const noexcept { return &**this; }

private:
    friend class shared_bitmap;
    explicit editor( shared_bitmap & handle ) noexcept : handle_{ &handle } {}

    shared_bitmap * handle_;
}; // class shared_bitmap::editor

} // namespace frsr::roaring
