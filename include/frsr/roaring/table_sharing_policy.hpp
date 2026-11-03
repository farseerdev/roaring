#pragma once

// The knobs of chunk-table sharing, in one place.
//
// Under a refcounted copy-on-write policy a copy of a bitmap whose chunk keys are 16 bits wide holds the chunk
// table of its source, and the first call that changes either of them gives it a table of its own (see
// chunk_array.hpp, chunk_store.hpp and bitmap::is_unique). The parts of that machinery that trade one cost for
// another are compile-time constants here; a build sets any of them otherwise with the macro of the same name
// (-DFRSR_ROARING_TABLE_SHARING_...=<value>), and the branches it turns off compile to nothing. None of them changes
// anything under value semantics, and with sharing off (FRSR_ROARING_TABLE_SHARING=0) a refcounted bitmap is laid
// out and compiled as it is without this machinery.
//
// The knobs change the code of the compiled bitmaps, so a program has to be built with the values its library was
// built with; library_configuration.hpp makes any other combination fail to link.
//
// Only the defaults, sharing off, and the defaults with one knob set to one of its tested values are supported
// (tests/table_sharing_matrix.sh builds and tests each); any other combination fails to compile here. To support one
// more, add it to that script and to supported_configuration below.

#include <cstdint>

namespace frsr::roaring::detail {

struct table_sharing_defaults {
    // Whether copies share the chunk table at all. Off, a copy copies the table and takes a count on every payload,
    // as it did before tables were shared.
    static constexpr bool enabled{ true };

    // A table whose block holds room for at most this many chunks carries no count and is never shared: a copy
    // copies it (0: every table that holds a chunk is shared). Sharing moves the cost of copying a table from the copy
    // to the first change, which then pays for a table of its own out of the line of the call that makes it; for a
    // table of a few chunks that costs more than the copy. A table grown from empty has room for 4 chunks, then 8,
    // 16, ...; a copied one for as many as it holds.
    static constexpr std::uint32_t small_table_capacity{ 4 };

    // `a |= b` on a shared table: true asks first whether b adds anything to a (b ⊆ a) and leaves the table shared
    // when it does not; false makes the table a's own and unites in place, as a union with a copy always did. The
    // question reads both bitmaps whole when b adds nothing, and is wasted when it adds something.
    static constexpr bool union_checks_first{ false };

    // `a -= b` on a shared table: true asks first whether a and b intersect (stopping at the first common key) and
    // leaves the table shared when they do not, otherwise makes the table a's own and subtracts in place; false builds
    // the difference into a table of its own, which takes the place of the shared one only when it holds fewer keys,
    // so the shared table is never copied for it, but every container a keeps is.
    static constexpr bool subtract_checks_first{ true };

    // and_with / or_with / andnot_with of a key sequence, a key range or a view on a shared table: true asks first
    // whether the call would change the bitmap (a ⊆ operand, operand ⊆ a, a ∩ operand empty) and leaves the table
    // shared when it would not; false makes the table the bitmap's own and runs the call. An operand that can be read
    // only once is never read ahead of the call.
    static constexpr bool key_list_checks_first{ true };

    // optimize() and optimize_keep_bitsets() on a shared table: true runs the pass on a copy with a table of its
    // own, which takes the place of the shared one only when the pass changed the kind of a container (a pass that
    // changes nothing leaves the table shared); false makes the table the bitmap's own and runs the pass in place.
    static constexpr bool optimize_aside{ true };

    // How many handles ahead a walk that writes the counts of their payloads (the copy of a shared table that gives
    // a bitmap a table of its own, and the release of the last holder of a table) starts fetching those counts (0:
    // none). Each count sits at the head of its payload, a cache line of its own, and the walk takes them one at a
    // time. Under the atomic policy each is a locked instruction, which waits for the fetches issued before it, so
    // fetching ahead serializes the walk on them: measured slower (a copy then a first change of a table of 1024
    // bitset chunks took up to a sixth longer at a distance of 8).
    static constexpr std::uint32_t prefetch_distance{ 0 };
};

struct table_sharing_policy {
#ifdef FRSR_ROARING_TABLE_SHARING
    static constexpr bool enabled{ FRSR_ROARING_TABLE_SHARING != 0 };
#else
    static constexpr bool enabled{ table_sharing_defaults::enabled };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_SMALL_CAPACITY
    static constexpr std::uint32_t small_table_capacity{ FRSR_ROARING_TABLE_SHARING_SMALL_CAPACITY };
#else
    static constexpr std::uint32_t small_table_capacity{ table_sharing_defaults::small_table_capacity };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_UNION_CHECKS_FIRST
    static constexpr bool union_checks_first{ FRSR_ROARING_TABLE_SHARING_UNION_CHECKS_FIRST != 0 };
#else
    static constexpr bool union_checks_first{ table_sharing_defaults::union_checks_first };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_SUBTRACT_CHECKS_FIRST
    static constexpr bool subtract_checks_first{ FRSR_ROARING_TABLE_SHARING_SUBTRACT_CHECKS_FIRST != 0 };
#else
    static constexpr bool subtract_checks_first{ table_sharing_defaults::subtract_checks_first };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_KEY_LIST_CHECKS_FIRST
    static constexpr bool key_list_checks_first{ FRSR_ROARING_TABLE_SHARING_KEY_LIST_CHECKS_FIRST != 0 };
#else
    static constexpr bool key_list_checks_first{ table_sharing_defaults::key_list_checks_first };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_OPTIMIZE_ASIDE
    static constexpr bool optimize_aside{ FRSR_ROARING_TABLE_SHARING_OPTIMIZE_ASIDE != 0 };
#else
    static constexpr bool optimize_aside{ table_sharing_defaults::optimize_aside };
#endif
#ifdef FRSR_ROARING_TABLE_SHARING_PREFETCH_DISTANCE
    static constexpr std::uint32_t prefetch_distance{ FRSR_ROARING_TABLE_SHARING_PREFETCH_DISTANCE };
#else
    static constexpr std::uint32_t prefetch_distance{ table_sharing_defaults::prefetch_distance };
#endif

    // The values that define a compiled library (library_configuration.hpp): the knobs in one number.
    static constexpr std::uint64_t signature{
        ( std::uint64_t{ enabled               } <<  0 ) |
        ( std::uint64_t{ union_checks_first    } <<  1 ) |
        ( std::uint64_t{ subtract_checks_first } <<  2 ) |
        ( std::uint64_t{ key_list_checks_first } <<  3 ) |
        ( std::uint64_t{ optimize_aside        } <<  4 ) |
        ( std::uint64_t{ small_table_capacity  } <<  8 ) |
        ( std::uint64_t{ prefetch_distance     } << 32 )
    };

private:
    using defaults = table_sharing_defaults;
    // How many knobs a build sets otherwise than their defaults.
    static constexpr unsigned knobs_off_default{
        unsigned{ small_table_capacity  != defaults::small_table_capacity  } +
        unsigned{ union_checks_first    != defaults::union_checks_first    } +
        unsigned{ subtract_checks_first != defaults::subtract_checks_first } +
        unsigned{ key_list_checks_first != defaults::key_list_checks_first } +
        unsigned{ optimize_aside        != defaults::optimize_aside        } +
        unsigned{ prefetch_distance     != defaults::prefetch_distance     }
    };
    static constexpr bool tested_small_table_capacity{
        ( small_table_capacity == defaults::small_table_capacity ) ||
        ( small_table_capacity == 0                              ) ||
        ( small_table_capacity == 16                             )
    };
    static constexpr bool tested_prefetch_distance{
        ( prefetch_distance == defaults::prefetch_distance ) ||
        ( prefetch_distance == 8                           )
    };

public:
    // The tested configurations: sharing off, the defaults, and the defaults with one knob set to one of its tested
    // values (a small-table capacity of 0 or 16, a prefetch distance of 8, or a flag flipped).
    static constexpr bool supported_configuration{
        ( !enabled ) ||
        ( knobs_off_default <= 1 && tested_small_table_capacity && tested_prefetch_distance )
    };
};

static_assert( table_sharing_policy::supported_configuration,
               "an untested combination of FRSR_ROARING_TABLE_SHARING_* knobs: set at most one of them otherwise than its "
               "default, to a tested value, or add the combination to tests/table_sharing_matrix.sh and to "
               "table_sharing_policy::supported_configuration" );

} // namespace frsr::roaring::detail
