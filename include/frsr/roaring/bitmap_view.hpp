#pragma once

// Type-erased, non-owning views over bitmap<Key, ContainerSet> instantiations —
// the std::span / std::string_view / function_ref analog for bitmaps, so
// non-template consumers (a downstream application's data layers) can take *any*
// bitmap by reference across a compiled boundary.
//
// - bitmap_view : read-only; binds to any const bitmap<...> &.
// - bitmap_ref  : read+write; binds to any bitmap<...> &; converts to bitmap_view.
//
// Representation: a fat pointer { void const * obj; vtable const * vt } — two
// pointers, no allocation, trivially copyable. The vtable is a static constexpr
// per bound instantiation; binding captures &bm + &vtable_for<Bitmap>.
//
// The erased value type is std::uint64_t: every key_type widens losslessly, and
// values outside the bound bitmap's key domain are simply not members (contains /
// remove return false; add asserts — adding an unrepresentable value is a caller
// bug, not a set property).
//
// Hot-path rule: the view is for boundary/query use where one indirect call per
// op is fine — NOT the inner merge loop. Set-ops between two erased views are
// deliberately out of the vtable (no concrete result type); materialize through
// a concrete bitmap or recover the concrete type via try_as<>().
//
// See the container-representation design notes, "Type-erased bitmap view / ref" section.

#include <frsr/roaring/bitmap.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace frsr::roaring {

namespace detail {

template <typename T>
inline constexpr bool is_frsr_bitmap{ false };
template <std::unsigned_integral Key, typename ContainerSet, typename CowPolicy>
inline constexpr bool is_frsr_bitmap<bitmap<Key, ContainerSet, CowPolicy>>{ true };

} // namespace detail

template <typename T>
concept erasable_bitmap = detail::is_frsr_bitmap<std::remove_const_t<T>>;

class bitmap_view {
public:
    using value_type = std::uint64_t;
    using size_type = std::size_t;
    using iterate_callback = bool (*)( value_type, void * );

    struct vtable {
        bool      (*contains )( void const *, value_type ) noexcept;
        size_type (*size     )( void const * ) noexcept;
        size_type (*byte_size)( void const * ) noexcept;
        value_type(*front    )( void const * ) noexcept;
        value_type(*back     )( void const * ) noexcept;
        // Visits values in ascending order; stops and returns false when the
        // callback returns false, returns true otherwise.
        bool      (*iterate  )( void const *, iterate_callback, void * );
        // Both operands are the same concrete instantiation.
        bool      (*equal    )( void const *, void const * );
    };

    template <erasable_bitmap Bitmap>
    static constexpr vtable vtable_for{
        .contains = []( void const * const obj, value_type const value ) noexcept {
            using key_type = typename Bitmap::key_type;
            if ( value > std::numeric_limits<key_type>::max() ) { return false; }
            return static_cast<Bitmap const *>( obj )->contains( static_cast<key_type>( value ) );
        },
        .size      = []( void const * const obj ) noexcept { return static_cast<Bitmap const *>( obj )->size     (); },
        .byte_size = []( void const * const obj ) noexcept { return static_cast<Bitmap const *>( obj )->byte_size(); },
        .front     = []( void const * const obj ) noexcept { return value_type{ static_cast<Bitmap const *>( obj )->front() }; },
        .back      = []( void const * const obj ) noexcept { return value_type{ static_cast<Bitmap const *>( obj )->back () }; },
        .iterate   = []( void const * const obj, iterate_callback const callback, void * const context ) {
            return static_cast<Bitmap const *>( obj )->iterate( [=]( auto const value ) {
                return callback( value_type{ value }, context );
            } );
        },
        .equal = []( void const * const lhs, void const * const rhs ) {
            return *static_cast<Bitmap const *>( lhs ) == *static_cast<Bitmap const *>( rhs );
        },
    };

    bitmap_view() = delete;

    template <erasable_bitmap Bitmap>
    bitmap_view( Bitmap const & bm ) noexcept
        : obj_{ &bm }, vt_{ &vtable_for<Bitmap> } {}

    [[nodiscard]] bool contains( value_type const value ) const noexcept { return vt_->contains( obj_, value ); }

    [[nodiscard]] size_type size     () const noexcept { return vt_->size     ( obj_ ); }
    [[nodiscard]] size_type byte_size() const noexcept { return vt_->byte_size( obj_ ); }
    [[nodiscard]] bool      empty    () const noexcept { return size() == 0; }

    // Precondition: !empty()
    [[nodiscard]] value_type front() const noexcept { return vt_->front( obj_ ); }
    [[nodiscard]] value_type back () const noexcept { return vt_->back ( obj_ ); }

    // Visits values in ascending order until f returns false; returns whether
    // the full set was visited.
    template <typename F>
    bool iterate( F && f ) const {
        return vt_->iterate(
            obj_,
            []( value_type const value, void * const context ) { return ( *static_cast<F *>( context ) )( value ); },
            &f
        );
    }

    template <typename F>
    void for_each( F && f ) const {
        iterate( [&]( value_type const value ) { f( value ); return true; } );
    }

    // Recover the concrete bitmap (the with_concrete escape hatch for set-ops,
    // which are deliberately not part of the erased surface).
    template <erasable_bitmap Bitmap>
    [[nodiscard]] Bitmap const * try_as() const noexcept {
        return vt_ == &vtable_for<Bitmap> ? static_cast<Bitmap const *>( obj_ ) : nullptr;
    }

    [[nodiscard]] friend bool operator==( bitmap_view const lhs, bitmap_view const rhs ) {
        if ( lhs.vt_ == rhs.vt_ ) { return lhs.vt_->equal( lhs.obj_, rhs.obj_ ); }
        // Heterogeneous instantiations: equal sets iff equal cardinality and
        // one is a subset of the other.
        return
            lhs.size() == rhs.size() &&
            lhs.iterate( [&]( value_type const value ) { return rhs.contains( value ); } );
    }

private:
    friend class bitmap_ref;

    bitmap_view( void const * const obj, vtable const * const vt ) noexcept : obj_{ obj }, vt_{ vt } {}

    void const * obj_;
    vtable const * vt_;
};

class bitmap_ref {
public:
    using value_type = bitmap_view::value_type;
    using size_type = bitmap_view::size_type;

    struct vtable {
        // Pointer to the canonical bitmap_view vtable (not an embedded copy) so
        // views obtained through ref->view conversion share vtable identity —
        // and thus try_as / homogeneous equality — with directly-bound views.
        bitmap_view::vtable const * query;
        // Returns whether the value was newly added / actually removed.
        bool (*add   )( void *, value_type );
        bool (*remove)( void *, value_type );
        void (*clear )( void * ) noexcept;
    };

    template <erasable_bitmap Bitmap>
    static constexpr vtable vtable_for{
        .query = &bitmap_view::vtable_for<Bitmap>,
        .add = []( void * const obj, value_type const value ) {
            using key_type = typename Bitmap::key_type;
            if ( value > std::numeric_limits<key_type>::max() ) {
                assert( !"bitmap_ref::add: value not representable in the bound bitmap's key type" );
                return false;
            }
            return static_cast<Bitmap *>( obj )->add( static_cast<key_type>( value ) );
        },
        .remove = []( void * const obj, value_type const value ) {
            using key_type = typename Bitmap::key_type;
            if ( value > std::numeric_limits<key_type>::max() ) { return false; }
            return static_cast<Bitmap *>( obj )->remove( static_cast<key_type>( value ) );
        },
        .clear = []( void * const obj ) noexcept { static_cast<Bitmap *>( obj )->clear(); },
    };

    bitmap_ref() = delete;

    template <erasable_bitmap Bitmap>
        requires ( !std::is_const_v<Bitmap> )
    bitmap_ref( Bitmap & bm ) noexcept
        : obj_{ &bm }, vt_{ &vtable_for<Bitmap> } {}

    [[nodiscard]] operator bitmap_view() const noexcept { return { obj_, vt_->query }; }
    [[nodiscard]] bitmap_view view() const noexcept { return *this; }

    [[nodiscard]] bool contains( value_type const value ) const noexcept { return vt_->query->contains( obj_, value ); }

    [[nodiscard]] size_type size     () const noexcept { return vt_->query->size     ( obj_ ); }
    [[nodiscard]] size_type byte_size() const noexcept { return vt_->query->byte_size( obj_ ); }
    [[nodiscard]] bool      empty    () const noexcept { return size() == 0; }

    // Precondition: !empty()
    [[nodiscard]] value_type front() const noexcept { return vt_->query->front( obj_ ); }
    [[nodiscard]] value_type back () const noexcept { return vt_->query->back ( obj_ ); }

    template <typename F>
    bool iterate( F && f ) const { return view().iterate( std::forward<F>( f ) ); }

    template <typename F>
    void for_each( F && f ) const { view().for_each( std::forward<F>( f ) ); }

    // Precondition: value representable in the bound bitmap's key type.
    bool add   ( value_type const value ) const { return vt_->add   ( obj_, value ); }
    bool remove( value_type const value ) const { return vt_->remove( obj_, value ); }
    void clear () const noexcept { vt_->clear( obj_ ); }

    template <erasable_bitmap Bitmap>
    [[nodiscard]] Bitmap * try_as() const noexcept {
        return vt_ == &vtable_for<Bitmap> ? static_cast<Bitmap *>( obj_ ) : nullptr;
    }

    [[nodiscard]] friend bool operator==( bitmap_ref const lhs, bitmap_ref const rhs ) {
        return lhs.view() == rhs.view();
    }

private:
    void * obj_;
    vtable const * vt_;
};

} // namespace frsr::roaring
