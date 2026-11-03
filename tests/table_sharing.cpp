// Copies of a bitmap whose chunk keys are 16 bits wide share the chunk table under a refcounted copy-on-write policy:
// a copy takes one count on the table, the first call that changes either bitmap gives it a table of its own, and a
// call that changes nothing leaves the table shared. Under value semantics, with sharing off, and for a small table
// (table_sharing_policy.hpp), a copy has a table of its own from the start; the tests that do not depend on the
// sharing run there too.
//
// Mirrors the copy tests of the reference (tests/toplevel_unit.c: can_copy_empty, can_add_to_copies,
// can_remove_from_copies, leaks_with_empty, test_stress_memory, test_example; tests/cpp_unit.cpp: test_example_cpp,
// issue316; tests/threads_unit.cpp: the threads that copy and combine, and the clone that races with the release of
// the other reference), each with every copy-on-write policy instead of the reference's copy_on_write flag.

#include "allocation_fault.hpp"
#include "view_support.hpp"

#include <frsr/roaring/bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace fr = frsr::roaring;
namespace alloc_fault = fr::kernel_test::alloc_fault;
using fr::detail::cow_atomic_refcount;
using fr::detail::cow_unsynchronized_refcount;
using fr::detail::cow_value_semantics;
using fr::detail::run_selection_eager;
using fr::detail::run_selection_lazy;

template <typename Key, typename Cow, typename Run = run_selection_eager>
using bitmap_of = fr::bitmap<Key, fr::default_container_set<Key>, Cow, Run>;

using sharing_policy = fr::detail::table_sharing_policy;

// The chunks of a table that is not small: the bitmaps of 32-bit keys these tests share hold at least this many.
inline constexpr std::uint32_t shared_chunks{ sharing_policy::small_table_capacity + 4 };

// The room a table that grows from empty makes for its first chunks.
inline constexpr std::uint32_t first_capacity{ 4 };

// Whether a copy of the bitmaps these tests build shares their table: a refcounted policy with sharing on, and a table
// that is not small. A bitmap of 16-bit keys has one chunk, in a table with room for first_capacity of them.
template <typename Bitmap>
inline constexpr bool shares_tables{
    Bitmap::cow_policy::refcounted && sharing_policy::enabled &&
    ( sizeof( typename Bitmap::key_type ) > sizeof( std::uint16_t ) || sharing_policy::small_table_capacity < first_capacity )
};

template <typename Bitmap>
using access = fr::detail::bitmap_test_access<Bitmap>;

// The containers of a bitmap as bytes, but for the counts of the payloads: what a co-owner of a payload may change.
template <typename Bitmap>
[[nodiscard]] auto containers_of( Bitmap const & bitmap ) {
    auto states{ access<Bitmap>::snapshot( bitmap ) };
    for ( auto & state : states ) {
        state.refcount = 0;
    }
    return states;
}

template <typename Bitmap>
[[nodiscard]] std::vector<typename Bitmap::key_type> values_of( Bitmap const & bitmap ) {
    auto const values{ bitmap.to_vector() };
    return { values.begin(), values.end() };
}

// A bitmap with an array, a bitset and a run container (in chunks 0, 1, 2 and 5 of a 32-bit key, and one value in each
// of the chunks from 8 on that make the table more than small; a 16-bit key has one chunk, which ends up a bitset).
template <typename Bitmap>
[[nodiscard]] Bitmap make_mixed() {
    using key = typename Bitmap::key_type;
    Bitmap bitmap;
    if constexpr ( sizeof( key ) == sizeof( std::uint16_t ) ) {
        for ( std::uint32_t i{ 0 }; i < 6'000; ++i ) {
            std::ignore = bitmap.add( static_cast<key>( i * 3 ) );
        }
    } else {
        for ( key i{ 0 }; i < 11; ++i ) {
            std::ignore = bitmap.add( i * 7 );
        }
        for ( key i{ 0 }; i < 5'000; ++i ) {
            std::ignore = bitmap.add( 65'536 + i * 3 );
        }
        bitmap.add_closed_range( 2 * 65'536 + 100, 2 * 65'536 + 5'000 );
        bitmap.optimize();
        for ( key i{ 0 }; i < 20; ++i ) {
            std::ignore = bitmap.add( 5 * 65'536 + i * 11 );
        }
        for ( key chunk{ 8 }; chunk < 8 + shared_chunks; ++chunk ) {
            std::ignore = bitmap.add( chunk * 65'536 + 1 );
        }
    }
    return bitmap;
}

// A value the mixed bitmap holds, and one it does not hold, in its first chunk.
template <typename Bitmap>
inline constexpr typename Bitmap::key_type held_value{ 21 };
template <typename Bitmap>
inline constexpr typename Bitmap::key_type absent_value{ 22 };

template <typename Bitmap>
class TableSharing : public ::testing::Test {};

using sharing_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, cow_value_semantics>,
    bitmap_of<std::uint32_t, cow_atomic_refcount>,
    bitmap_of<std::uint32_t, cow_unsynchronized_refcount>,
    bitmap_of<std::uint32_t, cow_atomic_refcount, run_selection_lazy>,
    bitmap_of<std::uint16_t, cow_value_semantics>,
    bitmap_of<std::uint16_t, cow_atomic_refcount>,
    bitmap_of<std::uint16_t, cow_unsynchronized_refcount>
>;
TYPED_TEST_SUITE( TableSharing, sharing_bitmaps );

// The table is shared and nothing in it, nor in its payloads, is touched by the copy: every handle, payload count and
// payload byte is what it was. Dropping the copy leaves the source the table's only holder.
TYPED_TEST( TableSharing, CopySharesTheChunkTable ) {
    using bitmap = TypeParam;
    auto const source{ make_mixed<bitmap>() };
    auto const before{ access<bitmap>::snapshot( source ) };
    auto const containers_before{ containers_of( source ) };
    EXPECT_TRUE( source.is_unique() );
    {
        bitmap const copy{ source };
        EXPECT_EQ( copy.shares_storage_with( source ), shares_tables<bitmap> );
        EXPECT_EQ( source.shares_storage_with( copy ), shares_tables<bitmap> );
        EXPECT_EQ( source.is_unique(), !shares_tables<bitmap> );
        EXPECT_EQ( copy.is_unique(), !shares_tables<bitmap> );
        // a copy of a table that is not shared takes a count on each payload instead
        if constexpr ( shares_tables<bitmap> ) {
            EXPECT_EQ( access<bitmap>::snapshot( source ), before );
        }
        EXPECT_EQ( containers_of( source ), containers_before );
        EXPECT_EQ( values_of( copy ), values_of( source ) );
        EXPECT_EQ( copy.size(), source.size() );
        EXPECT_EQ( copy.validate(), nullptr );

        bitmap assigned;
        assigned = copy;
        EXPECT_EQ( assigned.shares_storage_with( source ), shares_tables<bitmap> );
        EXPECT_EQ( values_of( assigned ), values_of( source ) );
    }
    EXPECT_TRUE( source.is_unique() );
    EXPECT_EQ( access<bitmap>::snapshot( source ), before );
}

// An empty table never shares: neither a copy of an empty bitmap nor one of a bitmap emptied with its capacity kept.
TYPED_TEST( TableSharing, AnEmptyTableIsNotShared ) {
    using bitmap = TypeParam;
    auto source{ make_mixed<bitmap>() };
    source.clear_keep_capacity();
    bitmap const copy{ source };
    EXPECT_FALSE( copy.shares_storage_with( source ) );
    EXPECT_TRUE( source.is_unique() );
    EXPECT_TRUE( copy.empty() );
    bitmap const empty;
    bitmap const copy_of_empty{ empty };
    EXPECT_FALSE( copy_of_empty.shares_storage_with( empty ) );
}

// Every call that would change nothing, on a copy: the copy still holds the source's table afterwards, and both hold
// the source's keys.
TYPED_TEST( TableSharing, CallsThatChangeNothingLeaveTheTableShared ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    auto source{ make_mixed<bitmap>() };
    source.optimize();
    auto const expected{ values_of( source ) };
    bitmap absent;
    std::ignore = absent.add( absent_value<bitmap> );
    bitmap subset;
    std::ignore = subset.add( held_value<bitmap> );
    auto const superset{ [ & ] {
        auto result{ source };
        std::ignore = result.add( absent_value<bitmap> );
        return result;
    }() };
    std::vector<key> const held{ expected.front(), held_value<bitmap>, expected.back() };
    std::vector<key> const held_unsorted{ expected.back(), held_value<bitmap>, expected.front() };
    std::vector<key> const absent_keys{ absent_value<bitmap>, static_cast<key>( absent_value<bitmap> + 3 ) };
    std::vector<key> const absent_unsorted{ static_cast<key>( absent_value<bitmap> + 3 ), absent_value<bitmap> };
    std::vector<key> superset_keys{ expected };
    superset_keys.push_back( absent_value<bitmap> );
    std::ranges::sort( superset_keys );

    std::vector<std::pair<std::string, std::function<void( bitmap & )>>> const calls{
        { "remove( absent )",              [ & ]( bitmap & b ) { EXPECT_FALSE( b.remove( absent_value<bitmap> ) ); } },
        { "add( held )",                   [ & ]( bitmap & b ) { EXPECT_FALSE( b.add( held_value<bitmap> ) ); } },
        { "add_bulk( held )", [ & ]( bitmap & b ) {
            typename bitmap::bulk_context ctx;
            EXPECT_FALSE( b.add_bulk( ctx, held_value<bitmap> ) );
        } },
        { "add_many_sorted( held )",       [ & ]( bitmap & b ) { b.add_many_sorted( held ); } },
        { "add_many( held, unsorted )",    [ & ]( bitmap & b ) { b.add_many( held_unsorted ); } },
        { "remove_many_sorted( absent )",  [ & ]( bitmap & b ) { b.remove_many_sorted( absent_keys ); } },
        { "remove_many( absent )",         [ & ]( bitmap & b ) { b.remove_many( absent_unsorted ); } },
        { "add_closed_range( held )",      [ & ]( bitmap & b ) { b.add_closed_range( held_value<bitmap>, held_value<bitmap> ); } },
        { "remove_closed_range( absent )", [ & ]( bitmap & b ) { b.remove_closed_range( absent_value<bitmap>, absent_value<bitmap> ); } },
        { "flip_closed_range( empty )",    [ & ]( bitmap & b ) { b.flip_closed_range( 5, 4 ); } },
        { "-= disjoint",                   [ & ]( bitmap & b ) { b -= absent; } },
        { "^= empty",                      [ & ]( bitmap & b ) { b ^= bitmap{}; } },
        { "&= superset",                   [ & ]( bitmap & b ) { b &= superset; } },
        { "&= a copy", [ & ]( bitmap & b ) {
            auto const other{ b };
            b &= other;
        } },
        { "|= a copy", [ & ]( bitmap & b ) {
            auto const other{ b };
            b |= other;
        } },
        { "&= view of a copy", [ & ]( bitmap & b ) {
            auto const other{ b };
            b &= fr::view_of( other );
        } },
        { "and_with( superset keys )",     [ & ]( bitmap & b ) { b.and_with( superset_keys ); } },
        { "and_with( every key )",         [ & ]( bitmap & b ) { b.and_with( fr::key_range::closed( 0, std::numeric_limits<key>::max() ) ); } },
        { "or_with( held keys )",          [ & ]( bitmap & b ) { b.or_with( held_unsorted ); } },
        { "andnot_with( absent keys )",    [ & ]( bitmap & b ) { b.andnot_with( absent_keys ); } },
        { "andnot_with( absent range )",   [ & ]( bitmap & b ) { b.andnot_with( fr::key_range::closed( absent_value<bitmap>, absent_value<bitmap> ) ); } },
        { "or_many_in_place( subset )", [ & ]( bitmap & b ) {
            std::array<bitmap const *, 1> const operands{ &subset };
            b.or_many_in_place( operands );
        } },
        { "xor_many_in_place( empty )", [ & ]( bitmap & b ) {
            bitmap const none;
            std::array<bitmap const *, 1> const operands{ &none };
            b.xor_many_in_place( operands );
        } },
        { "optimize_run_shaped( none )",   [ & ]( bitmap & b ) { b.optimize_run_shaped( 1U << 20 ); } },
        { "optimize_run_shaped( touched )", [ & ]( bitmap & b ) {
            typename bitmap::touched_chunks touched;
            touched.add_chunks_of( b );
            b.optimize_run_shaped( 1U << 20, touched );
        } },
        { "optimize( optimized )",         [ & ]( bitmap & b ) { b.optimize(); } },
        { "promote_large_arrays( none )",  [ & ]( bitmap & b ) { b.promote_large_arrays(); } },
        { "shrink_to_fit", [ & ]( bitmap & b ) {
            auto const freed{ b.shrink_to_fit() };
            EXPECT_TRUE( !shares_tables<bitmap> || freed == 0U );
        } },
    };
    // The calls a knob has make the table the bitmap's own whatever they would change.
    auto const stays_shared{ [ & ]( std::string const & name ) {
        bool const key_list{ name.starts_with( "and_with" ) ||
                             name.starts_with( "or_with" ) ||
                             name.starts_with( "andnot_with" ) };
        bool const optimizing{ name.starts_with( "optimize( " ) };
        return shares_tables<bitmap> &&
               ( !key_list || sharing_policy::key_list_checks_first ) &&
               ( !optimizing || sharing_policy::optimize_aside );
    } };
    for ( auto const & [ name, call ] : calls ) {
        bitmap copy{ source };
        call( copy );
        EXPECT_EQ( copy.shares_storage_with( source ), stays_shared( name ) ) << name;
        EXPECT_EQ( values_of( copy ), expected ) << name;
        EXPECT_EQ( copy.size(), expected.size() ) << name;
        EXPECT_EQ( values_of( source ), expected ) << name;
    }
}

// A call that changes the copy gives it a table of its own and leaves the source as it was, for every call; the table
// the source holds is its alone again.
TYPED_TEST( TableSharing, ACallThatChangesTheCopyLeavesTheSourceAsItWas ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    auto const source{ make_mixed<bitmap>() };
    auto const expected{ values_of( source ) };
    auto const before{ containers_of( source ) };
    bitmap other;
    std::ignore = other.add( held_value<bitmap> );
    std::ignore = other.add( absent_value<bitmap> );
    std::vector<key> const keys{ held_value<bitmap>, absent_value<bitmap> };

    std::vector<std::pair<std::string, std::function<void( bitmap & )>>> const calls{
        { "add",                 [ & ]( bitmap & b ) { EXPECT_TRUE( b.add( absent_value<bitmap> ) ); } },
        { "add_bulk", [ & ]( bitmap & b ) {
            typename bitmap::bulk_context ctx;
            EXPECT_TRUE( b.add_bulk( ctx, absent_value<bitmap> ) );
            EXPECT_TRUE( b.add_bulk( ctx, absent_value<bitmap> + 3 ) );
        } },
        { "contains_bulk then add_bulk", [ & ]( bitmap & b ) {
            typename bitmap::bulk_context ctx;
            EXPECT_TRUE( b.contains_bulk( ctx, held_value<bitmap> ) );
            EXPECT_TRUE( b.add_bulk( ctx, absent_value<bitmap> ) );
        } },
        { "remove",              [ & ]( bitmap & b ) { EXPECT_TRUE( b.remove( held_value<bitmap> ) ); } },
        { "add_many_sorted",     [ & ]( bitmap & b ) { b.add_many_sorted( keys ); } },
        { "remove_many_sorted",  [ & ]( bitmap & b ) { b.remove_many_sorted( keys ); } },
        { "add_closed_range",    [ & ]( bitmap & b ) { b.add_closed_range( held_value<bitmap>, absent_value<bitmap> + 40 ); } },
        { "remove_closed_range", [ & ]( bitmap & b ) { b.remove_closed_range( 0, held_value<bitmap> ); } },
        { "flip_closed_range",   [ & ]( bitmap & b ) { b.flip_closed_range( 0, absent_value<bitmap> ); } },
        { "|=",                  [ & ]( bitmap & b ) { b |= other; } },
        { "&=",                  [ & ]( bitmap & b ) { b &= other; } },
        { "-=",                  [ & ]( bitmap & b ) { b -= other; } },
        { "^=",                  [ & ]( bitmap & b ) { b ^= other; } },
        { "-= a copy", [ & ]( bitmap & b ) {
            auto const same{ b };
            b -= same;
        } },
        { "^= a copy", [ & ]( bitmap & b ) {
            auto const same{ b };
            b ^= same;
        } },
        { "and_with( keys )",    [ & ]( bitmap & b ) { b.and_with( keys ); } },
        { "or_with( keys )",     [ & ]( bitmap & b ) { b.or_with( keys ); } },
        { "andnot_with( keys )", [ & ]( bitmap & b ) { b.andnot_with( keys ); } },
        { "xor_with( range )",   [ & ]( bitmap & b ) { b.xor_with( fr::key_range::closed( 0, 100 ) ); } },
        { "or_many_in_place", [ & ]( bitmap & b ) {
            std::array<bitmap const *, 1> const operands{ &other };
            b.or_many_in_place( operands );
        } },
        { "xor_many_in_place", [ & ]( bitmap & b ) {
            std::array<bitmap const *, 1> const operands{ &other };
            b.xor_many_in_place( operands );
        } },
        { "optimize_run_shaped", [ & ]( bitmap & b ) { b.optimize_run_shaped( 1 ); } },
        { "optimize_for_speed", [ & ]( bitmap & b ) {
            b.remove_closed_range( 0, 17'000 );
            b.optimize_for_speed( 1 );
        } },
        { "optimize", [ & ]( bitmap & b ) {
            b.add_closed_range( 0, 3'000 );
            b.optimize();
        } },
        { "clear",               [ & ]( bitmap & b ) { b.clear(); } },
        { "clear_keep_capacity", [ & ]( bitmap & b ) {
            b.clear_keep_capacity();
            std::ignore = b.add( 1 );
        } },
    };
    // [croaring-ref] tests/toplevel_unit.c:check_unshares: each call on either holder of the table, from a source as
    // built and from one run-optimized; the other holder keeps its containers and keys.
    auto const optimized{ [] {
        auto result{ make_mixed<bitmap>() };
        result.optimize();
        return result;
    }() };
    for ( auto const * const original : { &source, &optimized } ) {
        auto const original_before{ containers_of( *original ) };
        auto const original_keys{ values_of( *original ) };
        for ( auto const & [ name, call ] : calls ) {
            for ( bool const change_the_copy : { true, false } ) {
                bitmap held{ *original };
                bitmap copy{ held };
                auto & changed{ change_the_copy ? copy : held };
                auto const & unchanged{ change_the_copy ? held : copy };
                auto const unchanged_before{ containers_of( unchanged ) };
                call( changed );
                EXPECT_FALSE( changed.shares_storage_with( unchanged ) ) << name;
                EXPECT_TRUE( changed.is_unique() ) << name;
                EXPECT_TRUE( unchanged.is_unique() || unchanged.shares_storage_with( *original ) ) << name;
                EXPECT_EQ( changed.validate(), nullptr ) << name;
                EXPECT_EQ( unchanged.validate(), nullptr ) << name;
                EXPECT_EQ( containers_of( unchanged ), unchanged_before ) << name;
                EXPECT_EQ( values_of( unchanged ), original_keys ) << name;
            }
        }
        EXPECT_EQ( containers_of( *original ), original_before );
    }
    EXPECT_TRUE( source.is_unique() );
    EXPECT_EQ( containers_of( source ), before );
    EXPECT_EQ( values_of( source ), expected );
}

// [croaring-ref] tests/toplevel_unit.c:test_remove_many_copy_on_write_many_copies: four copies of one table, each
// changed in turn by a batch removal (one that empties a chunk, one inside a single chunk): every copy that has not
// been changed yet keeps the keys of the source.
TYPED_TEST( TableSharing, ManyCopiesChangedOneAfterAnother ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    auto const source{ make_mixed<bitmap>() };
    auto const expected{ values_of( source ) };
    std::array<bitmap, 4> copies{ source, source, source, source };
    std::vector<key> const one_chunk{ held_value<bitmap>, static_cast<key>( 2 * held_value<bitmap> ) };
    auto const first_chunk{ [ & ] {
        std::vector<key> keys;
        for ( auto const value : expected ) {
            if ( value >> 16 == expected.front() >> 16 ) {
                keys.push_back( value );
            }
        }
        return keys;
    }() };
    for ( std::size_t i{ 0 }; i < copies.size(); ++i ) {
        if ( i % 2 == 0 ) {
            copies[ i ].remove_many_sorted( first_chunk );
            EXPECT_EQ( copies[ i ].size(), expected.size() - first_chunk.size() );
        } else {
            copies[ i ].remove_many( one_chunk );
            EXPECT_EQ( copies[ i ].size(), expected.size() - one_chunk.size() );
        }
        EXPECT_EQ( copies[ i ].validate(), nullptr );
        for ( std::size_t later{ i + 1 }; later < copies.size(); ++later ) {
            EXPECT_EQ( values_of( copies[ later ] ), expected ) << i << " " << later;
            EXPECT_EQ( copies[ later ].shares_storage_with( source ), shares_tables<bitmap> ) << i << " " << later;
        }
    }
    EXPECT_EQ( values_of( source ), expected );
    EXPECT_TRUE( source.is_unique() );
}

// [croaring-ref] tests/toplevel_unit.c:issue288 (roaring_bitmap_overwrite): a bitmap that holds a shared table is
// assigned an empty bitmap, another table, itself and a copy of its own table; the other holders keep their keys.
TYPED_TEST( TableSharing, AssignmentOverASharedTable ) {
    using bitmap = TypeParam;
    auto const source{ make_mixed<bitmap>() };
    auto const expected{ values_of( source ) };
    bitmap target{ source };
    target = bitmap{};
    EXPECT_TRUE( target.empty() );
    EXPECT_TRUE( source.is_unique() );
    EXPECT_EQ( values_of( source ), expected );

    target = source;
    EXPECT_EQ( target.shares_storage_with( source ), shares_tables<bitmap> );
    auto const & same{ target };
    target = same;
    EXPECT_EQ( values_of( target ), expected );
    bitmap other;
    std::ignore = other.add( absent_value<bitmap> );
    target = other;
    EXPECT_EQ( values_of( target ), values_of( other ) );
    EXPECT_TRUE( source.is_unique() );
    EXPECT_EQ( values_of( source ), expected );
    bitmap const third{ source };
    target = third;
    EXPECT_EQ( target.shares_storage_with( source ), shares_tables<bitmap> );
    target = source;
    EXPECT_EQ( values_of( target ), expected );
    target = {};
    EXPECT_EQ( values_of( third ), expected );
}

// A table that is small (table_sharing_policy::small_table_capacity) is copied, not shared: the copy has a table of its
// own, which takes a count on every spilled payload instead, and either bitmap changes without copying a table.
TYPED_TEST( TableSharing, ASmallTableIsCopied ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    if constexpr ( !( bitmap::cow_policy::refcounted && sharing_policy::enabled && sharing_policy::small_table_capacity >= first_capacity ) ) {
        GTEST_SKIP() << "no table is copied for being small";
    } else {
        bitmap source;
        for ( std::uint32_t i{ 0 }; i < 3'000; ++i ) {
            std::ignore = source.add( static_cast<key>( i * 3 ) );
        }
        ASSERT_LE( source.statistics().container_count(), sharing_policy::small_table_capacity );
        auto const before{ access<bitmap>::snapshot( source ) };
        bitmap copy{ source };
        EXPECT_FALSE( copy.shares_storage_with( source ) );
        EXPECT_TRUE( copy.is_unique() );
        EXPECT_TRUE( source.is_unique() );
        auto const after{ access<bitmap>::snapshot( source ) };
        ASSERT_EQ( after.size(), before.size() );
        for ( std::size_t i{ 0 }; i < after.size(); ++i ) {
            EXPECT_EQ( after[ i ].refcount, before[ i ].refcount == 0 ? 0 : before[ i ].refcount + 1 );
        }
        std::ignore = copy.add( 1 );
        EXPECT_FALSE( source.contains( 1 ) );
        EXPECT_EQ( access<bitmap>::snapshot( source ), before );
        bitmap assigned;
        assigned = source;
        EXPECT_FALSE( assigned.shares_storage_with( source ) );
        EXPECT_EQ( values_of( assigned ), values_of( source ) );
    }
}

// A copy, an assignment and the first change of a copy that fail at any of their allocations leave every bitmap they
// involve as it was: the source, the target (which may hold a shared table of its own) and the copy.
TYPED_TEST( TableSharing, AFailedCopyOrAssignmentChangesNothing ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    auto const large{ make_mixed<bitmap>() };
    auto const small{ [] {
        bitmap result;
        for ( std::uint32_t i{ 0 }; i < 3'000; ++i ) {
            std::ignore = result.add( static_cast<key>( i * 5 ) );
        }
        return result;
    }() };
    for ( auto const * const source : { &large, &small } ) {
        auto const source_before{ containers_of( *source ) };
        auto const source_keys{ values_of( *source ) };
        auto const target_holder{ make_mixed<bitmap>() };
        auto const target_keys{ values_of( target_holder ) };
        bool failed_any{ false };
        bool completed{ false };
        for ( std::uint32_t fail_at{ 1 }; fail_at < 4'000 && !completed; ++fail_at ) {
            bitmap target{ target_holder };
            auto const target_before{ containers_of( target ) };
            bool threw{ false };
            alloc_fault::counter = 0;
            try {
                alloc_fault::armed const armed{ fail_at };
                bitmap copy{ *source };
                target = *source;
                std::ignore = copy.add( absent_value<bitmap> );
            } catch ( std::bad_alloc const & ) {
                threw = true;
            }
            alloc_fault::counter = 0;
            failed_any |= threw;
            EXPECT_EQ( containers_of( *source ), source_before ) << fail_at;
            EXPECT_EQ( values_of( *source ), source_keys ) << fail_at;
            EXPECT_EQ( values_of( target_holder ), target_keys ) << fail_at;
            EXPECT_EQ( target_holder.validate(), nullptr ) << fail_at;
            EXPECT_EQ( target.validate(), nullptr ) << fail_at;
            if ( threw && values_of( target ) != source_keys ) {
                EXPECT_EQ( values_of( target ), target_keys ) << fail_at;
                EXPECT_EQ( containers_of( target ), target_before ) << fail_at;
            }
            if ( !threw ) {
                EXPECT_EQ( values_of( target ), source_keys ) << fail_at;
                completed = true;
            }
        }
        EXPECT_TRUE( failed_any );
        EXPECT_TRUE( completed );
    }
#endif
}

// An iterator on the source is not disturbed by changes of a copy.
TYPED_TEST( TableSharing, AnIteratorOnTheSourceSurvivesAChangeOfTheCopy ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    auto const source{ make_mixed<bitmap>() };
    auto const expected{ values_of( source ) };
    std::vector<key> walked;
    auto at{ source.begin() };
    for ( int i{ 0 }; i < 5; ++i ) {
        walked.push_back( *at );
        ++at;
    }
    bitmap copy{ source };
    copy.remove_closed_range( 0, std::numeric_limits<key>::max() / 2 );
    std::ignore = copy.add( absent_value<bitmap> );
    copy.optimize();
    for ( ; at != source.end(); ++at ) {
        walked.push_back( *at );
    }
    EXPECT_EQ( walked, expected );
}

// The copy that took the change detaches once: the context it filled is then good for the run of adds into its chunk.
TYPED_TEST( TableSharing, AddBulkOnASharedTableDetachesOnce ) {
    using bitmap = TypeParam;
    if constexpr ( !shares_tables<bitmap> ) {
        GTEST_SKIP() << "a copy does not share the table";
    } else {
        auto const source{ make_mixed<bitmap>() };
        auto const expected{ values_of( source ) };
        bitmap copy{ source };
        typename bitmap::bulk_context ctx;
        auto const generation{ access<bitmap>::generation( copy ) };
        EXPECT_TRUE( copy.add_bulk( ctx, absent_value<bitmap> ) );
        auto const after_detach{ access<bitmap>::generation( copy ) };
        EXPECT_NE( after_detach, generation );
        EXPECT_TRUE( copy.is_unique() );
        EXPECT_TRUE( source.is_unique() );
        for ( typename bitmap::key_type i{ 1 }; i < 5; ++i ) {
            EXPECT_TRUE( copy.add_bulk( ctx, absent_value<bitmap> + 48 + 3 * i ) );   // beyond the first chunk's values, none a multiple of 3
        }
#if !FRSR_ROARING_TEST_SHARE_EVERY_TABLE
        EXPECT_EQ( access<bitmap>::generation( copy ), after_detach );   // that build counts every call's table as shared
#endif
        EXPECT_EQ( values_of( source ), expected );
        EXPECT_EQ( copy.size(), expected.size() + 5 );
    }
}

// The listed run-shaped pass on a bitmap whose table a copy shares: this bitmap is converted, the copy keeps its
// containers, payloads included.
TYPED_TEST( TableSharing, OptimizeRunShapedOfTouchedChunksOnASharedTable ) {
    using bitmap = TypeParam;
    using key = typename bitmap::key_type;
    bitmap published;
    for ( key value{ 0 }; value < 6'000; ++value ) {  // a bitset that is two runs
        std::ignore = published.add( value );
    }
    std::ignore = published.add( static_cast<key>( 8'000 ) );
    if constexpr ( sizeof( key ) > sizeof( std::uint16_t ) ) {
        for ( key chunk{ 1 }; chunk <= shared_chunks; ++chunk ) {  // a table that is not small
            std::ignore = published.add( chunk * 65'536 );
        }
    }
    ASSERT_EQ( published.statistics().bitset_containers, 1U );
    auto const expected{ values_of( published ) };
    auto const reader{ published };
    auto const before{ containers_of( reader ) };
    typename bitmap::touched_chunks touched;
    touched.add_chunks_of( published );
    published.optimize_run_shaped( 64, touched );
    EXPECT_EQ( published.statistics().run_containers, 1U );
    EXPECT_EQ( values_of( published ), expected );
    EXPECT_EQ( reader.statistics().run_containers, 0U );
    EXPECT_EQ( containers_of( reader ), before );
    EXPECT_EQ( values_of( reader ), expected );
    EXPECT_FALSE( published.shares_storage_with( reader ) );
}

// A bitmap emptied with its capacity kept while it shares its table parks nothing: a result built into it afterwards
// leaves the other holder of the table as it was.
TYPED_TEST( TableSharing, ClearKeepCapacityOfASharedTableParksNothing ) {
    using bitmap = TypeParam;
    auto const source{ make_mixed<bitmap>() };
    auto const before{ containers_of( source ) };
    auto const expected{ values_of( source ) };
    bitmap scratch{ source };
    scratch.clear_keep_capacity();
    EXPECT_TRUE( source.is_unique() );
    auto const other{ make_mixed<bitmap>() };
    source.intersect_into( other, scratch );
    EXPECT_EQ( values_of( scratch ), expected );
    other.union_into( source, scratch );
    other.difference_into( source, scratch );
    EXPECT_TRUE( scratch.empty() );
    EXPECT_EQ( containers_of( source ), before );
    EXPECT_EQ( values_of( source ), expected );
}

// Compaction of a shared table frees nothing while the other holder keeps it: the call is skipped.
TYPED_TEST( TableSharing, ShrinkToFitOfASharedTableIsSkipped ) {
    using bitmap = TypeParam;
    auto source{ make_mixed<bitmap>() };
    bitmap copy{ source };
    if constexpr ( shares_tables<bitmap> ) {
        EXPECT_EQ( copy.shrink_to_fit(), 0U );
        EXPECT_TRUE( copy.shares_storage_with( source ) );
    }
    EXPECT_EQ( values_of( copy ), values_of( source ) );
}

// A frozen image borrowed into a bitmap: a copy shares the borrowed table, a change of the copy gives it a table of
// its own (its payloads still borrowed until written) and leaves the borrowed bitmap and the image as they were; a
// shared bitmap serializes as its keys.
TYPED_TEST( TableSharing, ABorrowedTableIsSharedAndTheImageStaysIntact ) {
    using bitmap = TypeParam;
    auto const source{ make_mixed<bitmap>() };
    auto const expected{ values_of( source ) };
    typename bitmap::serialized_byte_vector image;
    source.serialize_frozen_to_vm_vector( image );
    std::vector<std::byte> const image_bytes( image.data(), image.data() + image.size() );
    auto const view{ bitmap::frozen_view_from_vm_vector( image ) };
    ASSERT_TRUE( static_cast<bool>( view ) );
    bitmap borrowed;
    view.borrow_into( borrowed );
    {
        bitmap copy{ borrowed };
        EXPECT_EQ( copy.shares_storage_with( borrowed ), shares_tables<bitmap> );
        typename bitmap::serialized_byte_vector reimage;
        copy.serialize_frozen_to_vm_vector( reimage );
        EXPECT_EQ( values_of( bitmap::deserialize_frozen_from_vm_vector( reimage ) ), expected );
        std::ignore = copy.add( absent_value<bitmap> );
        EXPECT_TRUE( copy.remove( held_value<bitmap> ) );
        EXPECT_FALSE( copy.shares_storage_with( borrowed ) );
        EXPECT_TRUE( copy.contains( absent_value<bitmap> ) );
    }
    EXPECT_EQ( values_of( borrowed ), expected );
    EXPECT_TRUE( std::equal( image_bytes.begin(), image_bytes.end(), image.data() ) );
}

// A failed allocation of the table a change needs leaves both bitmaps as they were, still sharing the table.
TYPED_TEST( TableSharing, AFailedDetachLeavesBothBitmapsAsTheyWere ) {
    using bitmap = TypeParam;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations of the table cannot be made to fail in this build";
#else
    if constexpr ( !shares_tables<bitmap> ) {
        GTEST_SKIP() << "a copy does not share the table";
    } else {
        auto const source{ make_mixed<bitmap>() };
        auto const expected{ values_of( source ) };
        auto const before{ access<bitmap>::snapshot( source ) };
        bitmap other;
        std::ignore = other.add( absent_value<bitmap> );
        std::vector<std::pair<std::string, std::function<void( bitmap & )>>> const calls{
            { "add", [ & ]( bitmap & b ) { std::ignore = b.add( absent_value<bitmap> ); } },
            { "|=",  [ & ]( bitmap & b ) { b |= other; } },
            { "^=",  [ & ]( bitmap & b ) { b ^= other; } },
        };
        for ( auto const & [ name, call ] : calls ) {
            bitmap copy{ source };
            {
                alloc_fault::armed const armed{ 1 };
                EXPECT_THROW( call( copy ), std::bad_alloc ) << name;
            }
            EXPECT_TRUE( copy.shares_storage_with( source ) ) << name;
            EXPECT_EQ( values_of( copy ), expected ) << name;
            EXPECT_EQ( access<bitmap>::snapshot( source ), before ) << name;
            call( copy );
            EXPECT_FALSE( copy.shares_storage_with( source ) ) << name;
        }
    }
#endif
}

// A sequence of random calls, each on a bitmap that a copy taken just before shares: the copy keeps the keys it was
// taken with, the bitmap gets the keys a std::set gets, and both stay valid. Three earlier copies are kept alive and
// checked as well. The calls reach every chunk of the table, so most changes start past its first chunk.
template <typename Bitmap>
class TableSharingDifferential : public ::testing::Test {};

using differential_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, cow_atomic_refcount>,
    bitmap_of<std::uint32_t, cow_unsynchronized_refcount, run_selection_lazy>,
    bitmap_of<std::uint32_t, cow_value_semantics>
>;
TYPED_TEST_SUITE( TableSharingDifferential, differential_bitmaps );

TYPED_TEST( TableSharingDifferential, CopiesKeepTheirKeysThroughRandomCalls ) {
    using bitmap = TypeParam;
    using key = std::uint32_t;
    std::mt19937 rng{ 20261101 };
    constexpr key domain{ 8 * 65'536 };
    auto const random_key{ [ & ] { return std::uniform_int_distribution<key>{ 0, domain - 1 }( rng ); } };
    auto const random_keys{ [ & ]( std::size_t const count ) {
        std::vector<key> keys( count );
        for ( auto & each : keys ) {
            each = random_key();
        }
        return keys;
    } };

    bitmap current;
    std::set<key> oracle;
    std::array<std::pair<bitmap, std::vector<key>>, 3> kept{};
#if FRSR_ROARING_TEST_SHARE_EVERY_TABLE
    constexpr int steps{ 300 };
#else
    constexpr int steps{ 1'500 };
#endif
    for ( int step{ 0 }; step < steps; ++step ) {
        bitmap const copy{ current };
        auto const copied{ values_of( copy ) };
        auto const choice{ std::uniform_int_distribution<int>{ 0, 15 }( rng ) };
        switch ( choice ) {
            case 0: {
                auto const value{ random_key() };
                EXPECT_EQ( current.add( value ), oracle.insert( value ).second );
                break;
            }
            case 1: {
                auto const value{ oracle.empty() || rng() % 2 ? random_key() : *oracle.begin() };
                EXPECT_EQ( current.remove( value ), oracle.erase( value ) == 1 );
                break;
            }
            case 2: {
                auto keys{ random_keys( 1 + rng() % 300 ) };
                std::ranges::sort( keys );
                current.add_many_sorted( keys );
                oracle.insert( keys.begin(), keys.end() );
                break;
            }
            case 3: {
                auto keys{ random_keys( 1 + rng() % 300 ) };
                keys.insert( keys.end(), oracle.begin(), std::next( oracle.begin(), static_cast<std::ptrdiff_t>( std::min<std::size_t>( oracle.size(), 50 ) ) ) );
                std::ranges::sort( keys );
                current.remove_many_sorted( keys );
                for ( auto const each : keys ) {
                    oracle.erase( each );
                }
                break;
            }
            case 4: case 5: case 6: {
                auto const first{ random_key() };
                auto const last{ std::min<key>( domain - 1, first + rng() % 9'000 ) };
                if ( choice == 4 ) {
                    current.add_closed_range( first, last );
                } else if ( choice == 5 ) {
                    current.remove_closed_range( first, last );
                } else {
                    current.flip_closed_range( first, last );
                }
                for ( auto value{ first }; ; ++value ) {
                    if ( choice == 4 || ( choice == 6 && !oracle.contains( value ) ) ) {
                        oracle.insert( value );
                    } else {
                        oracle.erase( value );
                    }
                    if ( value == last ) {
                        break;
                    }
                }
                break;
            }
            case 7: case 8: case 9: case 10: {
                bitmap operand;
                auto const keys{ random_keys( rng() % 2'000 ) };
                operand.add_many( keys );
                if ( rng() % 3 == 0 ) {
                    operand.add_closed_range( random_key() / 2, random_key() / 2 + 5'000 );
                }
                std::set<key> const other{ operand.begin(), operand.end() };
                std::set<key> result;
                if ( choice == 7 ) {
                    current |= operand;
                    std::ranges::set_union( oracle, other, std::inserter( result, result.end() ) );
                } else if ( choice == 8 ) {
                    // half the time an operand that holds most of the bitmap, so that most chunks keep their keys
                    auto const mask{ rng() % 2 ? operand : ( operand | current ) - bitmap{ std::vector<key>{ random_key() } } };
                    std::set<key> const masked{ mask.begin(), mask.end() };
                    current &= mask;
                    std::ranges::set_intersection( oracle, masked, std::inserter( result, result.end() ) );
                } else if ( choice == 9 ) {
                    current -= operand;
                    std::ranges::set_difference( oracle, other, std::inserter( result, result.end() ) );
                } else {
                    current ^= operand;
                    std::ranges::set_symmetric_difference( oracle, other, std::inserter( result, result.end() ) );
                }
                oracle = std::move( result );
                break;
            }
            case 11: {
                current.optimize();
                break;
            }
            case 12: {
                current.optimize_run_shaped( 1 + rng() % 64 );
                break;
            }
            case 13: {
                auto const keys{ random_keys( 1 + rng() % 100 ) };
                current.and_with( std::span<key const>{ keys } );
                std::set<key> result;
                for ( auto const each : keys ) {
                    if ( oracle.contains( each ) ) {
                        result.insert( each );
                    }
                }
                oracle = std::move( result );
                break;
            }
            case 14: {
                auto const first{ random_key() };
                current.xor_with( fr::key_range::closed( first, first + 3'000 ) );
                for ( auto value{ first }; value <= first + 3'000; ++value ) {
                    if ( !oracle.erase( value ) ) {
                        oracle.insert( value );
                    }
                }
                break;
            }
            default: {
                typename bitmap::bulk_context ctx;
                for ( auto const value : random_keys( 1 + rng() % 200 ) ) {
                    EXPECT_EQ( current.add_bulk( ctx, value ), oracle.insert( value ).second );
                }
                break;
            }
        }
        ASSERT_EQ( values_of( copy ), copied ) << "step " << step << " call " << choice;
        ASSERT_EQ( values_of( current ), std::vector<key>( oracle.begin(), oracle.end() ) ) << "step " << step << " call " << choice;
        ASSERT_EQ( current.validate(), nullptr ) << "step " << step;
        ASSERT_EQ( copy.validate(), nullptr ) << "step " << step;
        if ( step % 7 == 0 ) {
            auto & slot{ kept[ static_cast<std::size_t>( step / 7 ) % kept.size() ] };
            ASSERT_EQ( values_of( slot.first ), slot.second ) << "step " << step;
            slot = { current, values_of( current ) };
        }
    }
    for ( auto const & [ held, keys ] : kept ) {
        EXPECT_EQ( values_of( held ), keys );
    }
}

// ---- the copy tests of the reference, with each copy-on-write policy -----------------------------------------------

template <typename Bitmap>
class ReferenceCopies : public ::testing::Test {};

using reference_bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, cow_value_semantics>,
    bitmap_of<std::uint32_t, cow_atomic_refcount>,
    bitmap_of<std::uint32_t, cow_unsynchronized_refcount>
>;
TYPED_TEST_SUITE( ReferenceCopies, reference_bitmaps );

// [croaring-ref] tests/toplevel_unit.c:can_copy_empty
TYPED_TEST( ReferenceCopies, CanCopyEmpty ) {
    TypeParam bm1;
    TypeParam bm2{ bm1 };
    EXPECT_EQ( bm1.size(), 0U );
    EXPECT_EQ( bm2.size(), 0U );
    EXPECT_TRUE( bm1.empty() );
    EXPECT_TRUE( bm2.empty() );
    EXPECT_TRUE( bm1.add( 3 ) );
    EXPECT_TRUE( bm2.add( 5 ) );
    EXPECT_EQ( bm1.size(), 1U );
    EXPECT_EQ( bm2.size(), 1U );
    EXPECT_TRUE( bm1.contains( 3 ) );
    EXPECT_TRUE( bm2.contains( 5 ) );
    EXPECT_FALSE( bm2.contains( 3 ) );
    EXPECT_FALSE( bm1.contains( 5 ) );
}

// [croaring-ref] tests/toplevel_unit.c:can_add_to_copies
TYPED_TEST( ReferenceCopies, CanAddToCopies ) {
    TypeParam bm1;
    EXPECT_TRUE( bm1.add( 3 ) );
    TypeParam bm2{ bm1 };
    EXPECT_EQ( bm1.size(), 1U );
    EXPECT_EQ( bm2.size(), 1U );
    EXPECT_TRUE( bm2.add( 4 ) );
    EXPECT_TRUE( bm1.add( 5 ) );
    EXPECT_EQ( bm1.size(), 2U );
    EXPECT_EQ( bm2.size(), 2U );
}

// [croaring-ref] tests/toplevel_unit.c:can_remove_from_copies
TYPED_TEST( ReferenceCopies, CanRemoveFromCopies ) {
    TypeParam bm1;
    EXPECT_TRUE( bm1.add( 3 ) );
    TypeParam bm2{ bm1 };
    EXPECT_EQ( bm1.size(), 1U );
    EXPECT_EQ( bm2.size(), 1U );
    EXPECT_TRUE( bm2.add( 4 ) );
    EXPECT_TRUE( bm1.add( 5 ) );
    EXPECT_EQ( bm1.size(), 2U );
    EXPECT_EQ( bm2.size(), 2U );
    EXPECT_TRUE( bm1.remove( 5 ) );
    EXPECT_EQ( bm1.size(), 1U );
    EXPECT_FALSE( bm1.remove( 4 ) );
    EXPECT_EQ( bm1.size(), 1U );
    EXPECT_EQ( bm2.size(), 2U );
    EXPECT_TRUE( bm2.remove( 4 ) );
    EXPECT_EQ( bm2.size(), 1U );
}

// [croaring-ref] tests/toplevel_unit.c:leaks_with_empty
TYPED_TEST( ReferenceCopies, LeaksWithEmpty ) {
    TypeParam empty;
    TypeParam r1;
    for ( std::uint32_t i{ 100 }; i < 70'000; i += 3 ) {
        std::ignore = r1.add( i );
    }
    auto const ror    { r1 | empty };
    auto const rxor   { r1 ^ empty };
    auto const rand   { r1 & empty };
    auto const randnot{ r1 - empty };
    empty = {};
    EXPECT_EQ( ror, r1 );
    EXPECT_EQ( rxor, r1 );
    EXPECT_EQ( randnot, r1 );
    r1 = {};
    EXPECT_TRUE( rand.empty() );
}

// [croaring-ref] tests/toplevel_unit.c:test_stress_memory
TYPED_TEST( ReferenceCopies, StressMemory ) {
#if FRSR_ROARING_TEST_SHARE_EVERY_TABLE
    constexpr std::size_t rounds{ 1 };
    constexpr std::size_t adds  { 20'000 };
#else
    constexpr std::size_t rounds{ 5 };
    constexpr std::size_t adds  { 1'000'000 };
#endif
    std::mt19937 rng{ 1 };
    for ( std::size_t i{ 0 }; i < rounds; ++i ) {
        TypeParam r1;
        for ( std::size_t k{ 0 }; k < adds; ++k ) {
            std::ignore = r1.add( static_cast<std::uint32_t>( rng() % 100'000'000 ) );
        }
        r1.optimize();
        auto const compact_size{ r1.portable_size_in_bytes() };
        std::vector<std::byte> bytes( compact_size );
        EXPECT_EQ( r1.portable_serialize( bytes ), compact_size );
        auto const t{ TypeParam::portable_deserialize( bytes ) };
        ASSERT_TRUE( t.has_value() );
        EXPECT_EQ( *t, r1 );
        auto const copy{ r1 };
        EXPECT_EQ( copy, r1 );
    }
}

// [croaring-ref] tests/toplevel_unit.c:test_example, tests/cpp_unit.cpp:test_example_cpp (the parts that copy,
// combine and compare)
TYPED_TEST( ReferenceCopies, Example ) {
    TypeParam r1;
    for ( std::uint32_t i{ 100 }; i < 1'000; ++i ) {
        std::ignore = r1.add( i );
        ASSERT_EQ( r1.validate(), nullptr );
    }
    EXPECT_TRUE( r1.contains( 500 ) );
    r1.optimize();
    ASSERT_EQ( r1.validate(), nullptr );
    TypeParam const r2{ std::vector<std::uint32_t>{ 1, 2, 3, 5, 6 } };
    TypeParam const r3{ std::vector<std::uint32_t>{ 2, 3, 4 } };
    auto const values{ r1.to_vector() };
    TypeParam const r1f{ std::vector<std::uint32_t>( values.begin(), values.end() ) };
    EXPECT_EQ( r1, r1f );

    TypeParam const z{ r3 };
    EXPECT_EQ( z.validate(), nullptr );
    EXPECT_EQ( r3, z );

    auto r1_2_3{ r1 | r2 };
    EXPECT_EQ( r1_2_3.validate(), nullptr );
    EXPECT_EQ( r1_2_3.size(), r1.size() + 5 );
    r1_2_3 |= r3;
    std::array<TypeParam const *, 3> const all{ &r1, &r2, &r3 };
    auto bigunion{ r1 };
    bigunion.or_many_in_place( { all.data() + 1, 2 } );
    EXPECT_EQ( r1_2_3, bigunion );

    auto rx1_2_3{ r1 ^ r2 };
    rx1_2_3 ^= r3;
    EXPECT_EQ( rx1_2_3, TypeParam::xor_many( all ) );

    auto const i1_2{ r1 & r2 };
    EXPECT_TRUE( i1_2.empty() );

    std::uint32_t counter{ 0 };
    for ( [[maybe_unused]] auto const value : r1 ) {
        ++counter;
    }
    EXPECT_EQ( counter, r1.size() );
}

// [croaring-ref] tests/cpp_unit.cpp:issue316
TYPED_TEST( ReferenceCopies, UnionIntoAnEmptyBitmapAndACopyOfIt ) {
    TypeParam r1;
    r1.add_closed_range( 1, 99 );
    TypeParam r2;
    r2 |= r1;
    EXPECT_TRUE( r2.is_subset_of( r1 ) );
    EXPECT_TRUE( r1.is_subset_of( r2 ) );
    EXPECT_EQ( r1, r2 );
    TypeParam const r3{ r2 };
    EXPECT_TRUE( r3.is_subset_of( r1 ) );
    EXPECT_TRUE( r1.is_subset_of( r3 ) );
    EXPECT_EQ( r1, r3 );
    EXPECT_EQ( r1, r2 );
}

// [croaring-ref] tests/roaring64_unit.cpp: roaring64_bitmap_copy, roaring64_bitmap_overwrite (with keys, with an empty
// bitmap and with itself): a bitmap of 64-bit keys, whose table every copy copies.
TYPED_TEST( ReferenceCopies, CopiesAndOverwritesOfWideKeys ) {
    using wide = bitmap_of<std::uint64_t, typename TypeParam::cow_policy>;
    wide source;
    for ( std::uint64_t i{ 0 }; i < 1'000; ++i ) {
        std::ignore = source.add( i * 0x1'0000'0001ULL );
    }
    wide copy{ source };
    EXPECT_EQ( copy, source );
    std::ignore = copy.add( 3 );
    EXPECT_FALSE( source.contains( 3 ) );
    wide target;
    std::ignore = target.add( 7 );
    target = source;
    EXPECT_EQ( target, source );
    target = wide{};
    EXPECT_TRUE( target.empty() );
    EXPECT_EQ( source.size(), 1'000U );
    copy = source;
    auto const & same{ copy };
    copy = same;
    EXPECT_EQ( copy, source );
}

// ---- threads -------------------------------------------------------------------------------------------------------

using atomic_bitmap = bitmap_of<std::uint32_t, cow_atomic_refcount>;

// [croaring-ref] tests/threads_unit.cpp:run_threads_unit_tests: two threads, each on copies of its own copies of
// three bitmaps (one run-optimized), copy them again and combine the copies in place, while the bitmaps they were
// copied from, which share their tables, are read.
TEST( TableSharingThreads, CopiesCombinedOnTwoThreads ) {
    atomic_bitmap r1;
    for ( std::uint32_t i{ 0 }; i < 50'000; ++i ) {
        if ( i != 300 ) {
            std::ignore = r1.add( 65'536 + i );
        }
    }
    for ( std::uint32_t i{ 50'000 }; i < 150'000; ++i ) {
        if ( i % 500 == 0 ) {
            std::ignore = r1.add( i );
        }
    }
    for ( std::uint32_t i{ 150'000 }; i < 200'000; ++i ) {
        if ( i % 2 == 0 ) {
            std::ignore = r1.add( i );
        }
    }
    for ( std::uint32_t chunk{ 8 }; chunk < 8 + shared_chunks; ++chunk ) {  // a table that is not small
        std::ignore = r1.add( chunk * 65'536 );
    }
    r1.optimize();
    atomic_bitmap const r2{ std::vector<std::uint32_t>{ 10'010, 10'020, 10'030, 10'040, 10'050 } };
    atomic_bitmap const r3{ r1 };
    auto const expected{ ( r1 & r2 ) - r3 };
    std::array<atomic_bitmap, 3> const first{ r1, r2, r3 };
    std::array<atomic_bitmap, 3> const second{ r1, r2, r3 };
    std::atomic<int> wrong{ 0 };
    auto const run{ [ & ]( std::array<atomic_bitmap, 3> const & operands ) {
        for ( int i{ 0 }; i < 100; ++i ) {
            atomic_bitmap a{ operands[ 0 ] };
            atomic_bitmap const b{ operands[ 1 ] };
            atomic_bitmap const c{ operands[ 2 ] };
            a &= b;
            a -= c;
            if ( a != expected || !operands[ 0 ].contains( 65'536 ) ) {
                wrong.fetch_add( 1, std::memory_order_relaxed );
            }
        }
    } };
    std::thread thread1{ run, std::cref( first ) };
    std::thread thread2{ run, std::cref( second ) };
    thread1.join();
    thread2.join();
    EXPECT_EQ( wrong.load(), 0 );
    EXPECT_EQ( r1, r3 );
}

// Threads copy one published bitmap, change their copies and read the published one meanwhile: the published bitmap
// never changes, every copy ends with the published keys and its own change.
TEST( TableSharingThreads, CopiesOfAPublishedBitmapChangedByManyThreads ) {
    static_assert( sharing_policy::small_table_capacity < 64, "the published table is not small" );
    atomic_bitmap published;
    for ( std::uint32_t chunk{ 0 }; chunk < 64; ++chunk ) {
        for ( std::uint32_t i{ 0 }; i < 100; ++i ) {
            std::ignore = published.add( chunk * 65'536 + i * 5 );
        }
    }
    auto const expected{ values_of( published ) };
    constexpr int thread_count{ 6 };
    std::atomic<int> wrong{ 0 };
    std::atomic<bool> go{ false };
    std::vector<std::thread> threads;
    for ( int t{ 0 }; t < thread_count; ++t ) {
        threads.emplace_back( [ &, t ] {
            while ( !go.load( std::memory_order_acquire ) ) {
            }
            for ( int i{ 0 }; i < 200; ++i ) {
                atomic_bitmap copy{ published };
                auto const value{ static_cast<std::uint32_t>( ( t * 7 + i ) % 64 ) * 65'536 + 1 };
                if ( !copy.add( value ) || !copy.contains( value ) || published.contains( value ) ) {
                    wrong.fetch_add( 1, std::memory_order_relaxed );
                }
                if ( copy.size() != expected.size() + 1 || published.size() != expected.size() ) {
                    wrong.fetch_add( 1, std::memory_order_relaxed );
                }
                copy -= published;
                if ( copy.size() != 1 ) {
                    wrong.fetch_add( 1, std::memory_order_relaxed );
                }
            }
        } );
    }
    go.store( true, std::memory_order_release );
    for ( auto & thread : threads ) {
        thread.join();
    }
    EXPECT_EQ( wrong.load(), 0 );
    EXPECT_EQ( values_of( published ), expected );
    EXPECT_TRUE( published.is_unique() );
}

// The two holders of a table change it at the same time: each detaches, and the last one to let go of the table
// frees it.
TEST( TableSharingThreads, TwoHoldersOfATableChangeItAtOnce ) {
    for ( int round{ 0 }; round < 200; ++round ) {
        auto source{ std::make_unique<atomic_bitmap>() };
        for ( std::uint32_t chunk{ 0 }; chunk < shared_chunks; ++chunk ) {
            std::ignore = source->add( chunk * 65'536 );
        }
        atomic_bitmap left{ *source };
        atomic_bitmap right{ *source };
        source.reset();
        std::atomic<int> ready{ 0 };
        auto const change{ [ & ]( atomic_bitmap & bitmap, std::uint32_t const value ) {
            ready.fetch_add( 1, std::memory_order_acq_rel );
            while ( ready.load( std::memory_order_acquire ) < 2 ) {
            }
            std::ignore = bitmap.add( value );
        } };
        std::thread a{ change, std::ref( left ), 7U };
        std::thread b{ change, std::ref( right ), 9U };
        a.join();
        b.join();
        ASSERT_TRUE( left.is_unique() );
        ASSERT_TRUE( right.is_unique() );
        ASSERT_TRUE( left.contains( 7 ) && !left.contains( 9 ) && left.size() == shared_chunks + 1 );
        ASSERT_TRUE( right.contains( 9 ) && !right.contains( 7 ) && right.size() == shared_chunks + 1 );
    }
}

// [croaring-ref] tests/threads_unit.cpp:run_shared_container_race_test, for the table: the copy of a shared table
// must not race with the other holder letting go of it. The detach is paused at its allocation while the other holder
// drops the table; the detach then finds itself the last holder and frees the table it left.
TEST( TableSharingThreads, ADetachRacesTheReleaseOfTheOtherHolder ) {
#if !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be intercepted in this build";
#else
    std::mutex mutex;
    std::condition_variable cv;
    bool release_may_go{ false };
    bool released{ false };
    static thread_local std::function<void()> * pause{ nullptr };
    std::function<void()> on_allocate{ [ & ] {
        std::unique_lock lock{ mutex };
        release_may_go = true;
        cv.notify_all();
        cv.wait( lock, [ & ] { return released; } );
    } };

    auto r1{ std::make_unique<atomic_bitmap>() };
    for ( std::uint32_t i{ 0 }; i < 10'000; ++i ) {
        std::ignore = r1->add( 2 * i + 1 + ( i % shared_chunks ) * 65'536 );
    }
    auto r2{ std::make_unique<atomic_bitmap>( *r1 ) };
    if ( !shares_tables<atomic_bitmap> ) {
        GTEST_SKIP() << "a copy does not share the table";
    }
    ASSERT_TRUE( r1->shares_storage_with( *r2 ) );
    bool intercepted{ false };
    std::thread modifier{ [ & ] {
        alloc_fault::on_allocate = [] {
            if ( pause != nullptr ) {
                auto * const once{ pause };
                pause = nullptr;
                ( *once )();
            }
        };
        pause = &on_allocate;
        std::ignore = r1->add( 2 );
        intercepted = pause == nullptr;
        pause = nullptr;
        alloc_fault::on_allocate = nullptr;
    } };
    std::thread releaser{ [ & ] {
        std::unique_lock lock{ mutex };
        cv.wait( lock, [ & ] { return release_may_go; } );
        lock.unlock();
        r2.reset();
        lock.lock();
        released = true;
        cv.notify_all();
    } };
    modifier.join();
    releaser.join();
    EXPECT_TRUE( intercepted ) << "the shared table was never copied, the test is ineffective";
    EXPECT_TRUE( r1->contains( 2 ) );
    EXPECT_EQ( r1->size(), 10'001U );
    EXPECT_EQ( r1->validate(), nullptr );
    EXPECT_TRUE( r1->is_unique() );
#endif
}

// What a bitmap of value semantics pays for the sharing: nothing in its layout.
#if !FRSR_ROARING_TEST_SHARE_EVERY_TABLE
static_assert( sizeof( fr::bitmap<std::uint32_t> ) == sizeof( fr::bitmap<std::uint32_t, fr::default_container_set<std::uint32_t>, cow_atomic_refcount> ) );
static_assert( sizeof( fr::bitmap<std::uint32_t> ) == 72 );
#endif

} // namespace
