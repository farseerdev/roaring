#pragma once

// The mmap-master: a bitmap served zero-copy from the frozen on-disk format
// (docs/frozen-format.md) mapped in place. open() is mmap + validation + a
// rebuild of the two SoA chunk arrays with every container payload BORROWED
// from the mapping — no payload copy, O(#chunks) work regardless of value
// count. Mutations CoW-clone the touched containers into private storage
// through the ownership write barrier; the mapped master bytes are never
// written through this type (store() rewrites a file wholesale from a live
// bitmap). Copies of the borrowed bitmap alias the same mapped payloads, so
// independent scratchpads over one master are cheap and mutually isolated.
//
// Requires psi::vm (file mapping); compiled out otherwise.
//
// See the container-representation design notes, "Persistence: the on-disk
// format is the live store".

#include <frsr/roaring/bitmap.hpp>

#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)

#include <concepts>
#include <optional>

namespace frsr::roaring {

template <std::unsigned_integral Key, typename ContainerSet = default_container_set<Key>>
    requires detail::valid_container_set<Key, ContainerSet>
class persistent_bitmap {
public:
    using bitmap_type = bitmap<Key, ContainerSet>;

    // Maps `file_name` and borrows its frozen payloads in place. Disengaged on
    // mapping failure or frozen-format validation failure (wrong magic /
    // key_bits / version, malformed index).
    [[nodiscard]] static std::optional<persistent_bitmap> open( auto const * const file_name ) {
        std::optional<persistent_bitmap> result{ persistent_bitmap{} };
        if ( !result->storage_.map_file( file_name, psi::vm::flags::named_object_construction_policy::open_existing ).succeeded() ) {
            return std::nullopt;
        }
        auto const view{ bitmap_type::frozen_view_from_vm_vector( result->storage_ ) };
        if ( !view ) {   // store() writes a header even for an empty bitmap — headerless ⇒ not ours
            return std::nullopt;
        }
        view.borrow_into( result->bitmap_ );
        return result;
    }

    // Writes `source` to `file_name` in the frozen format (creating or
    // truncating it), through a file mapping — the same bytes open() borrows.
    [[nodiscard]] static bool store( auto const * const file_name, bitmap_type const & source ) {
        typename bitmap_type::serialized_byte_vector out;
        if ( !out.map_file( file_name, psi::vm::flags::named_object_construction_policy::create_new_or_truncate_existing ).succeeded() ) {
            return false;
        }
        source.serialize_frozen_to_vm_vector( out );
        return true;
    }

    // The live bitmap over the mapping. Its borrowed payloads (and those of
    // any copy made of it) are valid only while this persistent_bitmap is
    // alive; containers the caller has mutated are private and unaffected.
    [[nodiscard]] bitmap_type       & operator*()        noexcept { return  bitmap_; }
    [[nodiscard]] bitmap_type const & operator*()  const noexcept { return  bitmap_; }
    [[nodiscard]] bitmap_type       * operator->()       noexcept { return &bitmap_; }
    [[nodiscard]] bitmap_type const * operator->() const noexcept { return &bitmap_; }

    persistent_bitmap( persistent_bitmap && )              = default;
    persistent_bitmap & operator=( persistent_bitmap && )  = default;
    persistent_bitmap( persistent_bitmap const & )             = delete;
    persistent_bitmap & operator=( persistent_bitmap const & ) = delete;

private:
    persistent_bitmap() = default;

    typename bitmap_type::serialized_byte_vector storage_;
    bitmap_type                                  bitmap_;
};

} // namespace frsr::roaring

#endif // FRSR_ROARING_HAS_PSI_VM && FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION
