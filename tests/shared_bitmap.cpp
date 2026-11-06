// shared_bitmap: copies of a handle share one bitmap until one of them changes it; a change that would leave it as it
// is leaves it shared; the cardinality a handle holds follows its bitmap; a copy that fails leaves the handle as it
// was; threads copy, read, change and drop handles that share a bitmap. Every copy-on-write policy, 32- and 64-bit
// keys, and (through tests/table_sharing_matrix.sh) chunk-table sharing off and on.

#include "allocation_fault.hpp"

#include <frsr/roaring/shared_bitmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace fr = frsr::roaring;
namespace alloc_fault = fr::kernel_test::alloc_fault;

template <typename Key, typename Cow>
using bitmap_of = fr::bitmap<Key, fr::default_container_set<Key>, Cow>;

template <typename Bitmap>
class SharedBitmap : public ::testing::Test {};

using bitmaps = ::testing::Types<
    bitmap_of<std::uint32_t, fr::detail::cow_value_semantics>,
    bitmap_of<std::uint32_t, fr::detail::cow_atomic_refcount>,
    bitmap_of<std::uint32_t, fr::detail::cow_unsynchronized_refcount>,
    bitmap_of<std::uint64_t, fr::detail::cow_value_semantics>,
    bitmap_of<std::uint64_t, fr::detail::cow_atomic_refcount>
>;
TYPED_TEST_SUITE( SharedBitmap, bitmaps );

// Arrays, a bitset and a run.
template <typename Bitmap>
[[nodiscard]] Bitmap make_sample() {
    Bitmap bitmap;
    for ( std::uint32_t i{ 0 }; i < 40; ++i ) {
        std::ignore = bitmap.add( i * 9 );
        std::ignore = bitmap.add( 3 * 65'536 + i * 5 );
    }
    for ( std::uint32_t i{ 0 }; i < 6'000; ++i ) {
        std::ignore = bitmap.add( 65'536 + i * 3 );
    }
    bitmap.add_closed_range( 5 * 65'536 + 10, 5 * 65'536 + 4'000 );
    bitmap.optimize();
    return bitmap;
}

template <typename Bitmap>
[[nodiscard]] std::set<typename Bitmap::key_type> keys_of( Bitmap const & bitmap ) {
    std::set<typename Bitmap::key_type> keys;
    for ( auto const value : bitmap ) {
        keys.insert( value );
    }
    return keys;
}

// What a handle holds of its bitmap agrees with the bitmap.
template <typename Handle>
void expect_coherent( Handle const & handle ) {
    EXPECT_EQ( handle.size(), handle.get().size() );
    EXPECT_EQ( handle.empty(), handle.get().empty() );
    EXPECT_EQ( handle.container_count(), handle.get().container_count() );
    EXPECT_EQ( handle.get().validate(), nullptr );
}

TYPED_TEST( SharedBitmap, AnEmptyHandleHoldsNothing ) {
    using bitmap = TypeParam;
    fr::shared_bitmap<bitmap> handle;
    EXPECT_TRUE( handle.empty() );
    EXPECT_TRUE( handle.is_unique() );
    EXPECT_EQ( handle.container_count(), 0U );
    EXPECT_FALSE( handle.contains( 7 ) );
    EXPECT_FALSE( handle.remove( 7 ) );
    EXPECT_FALSE( handle.shares_storage_with( fr::shared_bitmap<bitmap>{} ) );
    auto const sample{ make_sample<bitmap>() };
#if !FRSR_ROARING_HAS_MIMALLOC && FRSR_TESTS_COUNT_ALLOCATIONS
    {
        // nothing here allocates: every allocation would fail
        alloc_fault::armed const armed{ 1 };
        handle.clear();
        handle -= sample;
        handle &= sample;
        handle -= fr::shared_bitmap<bitmap>{};
        EXPECT_FALSE( handle.remove( 3 ) );
        EXPECT_FALSE( handle.contains( 3 ) );
        EXPECT_TRUE( handle.get().empty() );
    }
    alloc_fault::counter = 0;
#endif
    EXPECT_TRUE( handle.empty() );
    EXPECT_TRUE( handle.add( 7 ) );
    EXPECT_TRUE( handle.contains( 7 ) );
    EXPECT_EQ( handle.size(), 1U );
    expect_coherent( handle );
    auto moved{ std::move( handle ) };
    EXPECT_TRUE( handle.empty() );
    EXPECT_EQ( moved.size(), 1U );
}

TYPED_TEST( SharedBitmap, ACopySharesUntilAChange ) {
    using bitmap = TypeParam;
    auto const sample{ make_sample<bitmap>() };
    fr::shared_bitmap<bitmap> source{ sample };
    auto copy{ source };
    EXPECT_TRUE( copy.shares_storage_with( source ) );
    EXPECT_FALSE( copy.is_unique() );
    EXPECT_FALSE( source.is_unique() );
    EXPECT_EQ( &copy.get(), &source.get() );
    EXPECT_EQ( copy.size(), sample.size() );

    EXPECT_TRUE( copy.add( 2 * 65'536 + 1 ) );
    EXPECT_FALSE( copy.shares_storage_with( source ) );
    EXPECT_TRUE( copy.is_unique() );
    EXPECT_TRUE( source.is_unique() );
    EXPECT_EQ( keys_of( source.get() ), keys_of( sample ) );
    EXPECT_EQ( copy.size(), sample.size() + 1 );
    expect_coherent( copy );
    expect_coherent( source );

    // the handle's own bitmap is changed where it is
    auto const * const own{ &copy.get() };
    EXPECT_TRUE( copy.remove( 2 * 65'536 + 1 ) );
    EXPECT_EQ( &copy.get(), own );
    EXPECT_EQ( keys_of( copy.get() ), keys_of( sample ) );
}

TYPED_TEST( SharedBitmap, AChangeThatChangesNothingLeavesTheBitmapShared ) {
    using bitmap = TypeParam;
    auto const sample{ make_sample<bitmap>() };
    fr::shared_bitmap<bitmap> const source{ sample };
    bitmap subset;
    std::ignore = subset.add( 65'536 + 3 );
    std::ignore = subset.add( 5 * 65'536 + 100 );
    bitmap disjoint;
    std::ignore = disjoint.add( 1 );
    std::ignore = disjoint.add( 9 * 65'536 );
    auto superset{ sample };
    std::ignore = superset.add( 1 );

    auto const stays_shared{ [ & ]( char const * const name, auto const & change ) {
        auto copy{ source };
        change( copy );
        EXPECT_TRUE( copy.shares_storage_with( source ) ) << name;
        EXPECT_EQ( keys_of( copy.get() ), keys_of( sample ) ) << name;
        expect_coherent( copy );
    } };
    stays_shared( "add of a held key",       []( auto & h ) { EXPECT_FALSE( h.add( 65'536 + 3 ) ); } );
    stays_shared( "remove of an absent key", []( auto & h ) { EXPECT_FALSE( h.remove( 1 ) ); } );
    stays_shared( "|= itself",               [ & ]( auto & h ) { h |= source; } );
    stays_shared( "&= itself",               [ & ]( auto & h ) { h &= source; } );
    stays_shared( "|= its own bitmap",       []( auto & h ) { h |= h.get(); } );
    stays_shared( "&= its own bitmap",       []( auto & h ) { h &= h.get(); } );
    stays_shared( "|= an empty handle",      []( auto & h ) { h |= fr::shared_bitmap<bitmap>{}; } );
    stays_shared( "|= an empty bitmap",      []( auto & h ) { h |= bitmap{}; } );
    stays_shared( "-= an empty bitmap",      []( auto & h ) { h -= bitmap{}; } );
    stays_shared( "^= an empty bitmap",      []( auto & h ) { h ^= bitmap{}; } );
    stays_shared( "|= a subset",             [ & ]( auto & h ) { h |= subset; } );
    stays_shared( "&= a superset",           [ & ]( auto & h ) { h &= superset; } );
    stays_shared( "-= a disjoint bitmap",    [ & ]( auto & h ) { h -= disjoint; } );
}

TYPED_TEST( SharedBitmap, SetOperationsWithAHandleShareWhatTheyCan ) {
    using bitmap = TypeParam;
    auto const sample{ make_sample<bitmap>() };
    fr::shared_bitmap<bitmap> const source{ sample };
    {
        fr::shared_bitmap<bitmap> empty;
        empty |= source;
        EXPECT_TRUE( empty.shares_storage_with( source ) );
    }
    {
        fr::shared_bitmap<bitmap> empty;
        empty ^= source;
        EXPECT_TRUE( empty.shares_storage_with( source ) );
    }
    {
        auto copy{ source };
        copy -= source;
        EXPECT_TRUE( copy.empty() );
        EXPECT_FALSE( copy.shares_storage_with( source ) );
        EXPECT_TRUE( source.is_unique() );
        EXPECT_EQ( source.size(), sample.size() );
    }
    {
        auto copy{ source };
        copy ^= source;
        EXPECT_TRUE( copy.empty() );
        EXPECT_EQ( keys_of( source.get() ), keys_of( sample ) );
    }
    {
        auto copy{ source };
        copy &= bitmap{};
        EXPECT_TRUE( copy.empty() );
        EXPECT_EQ( keys_of( source.get() ), keys_of( sample ) );
    }
    {
        auto copy{ source };
        copy.clear();
        EXPECT_TRUE( copy.empty() );
        EXPECT_TRUE( source.is_unique() );
    }
    {
        // a handle whose bitmap went empty takes the operand's too
        fr::shared_bitmap<bitmap> emptied{ bitmap{} };
        EXPECT_TRUE( emptied.add( 3 ) );
        EXPECT_TRUE( emptied.remove( 3 ) );
        emptied |= source;
        EXPECT_TRUE( emptied.shares_storage_with( source ) );
        emptied.clear();
        EXPECT_TRUE( emptied.add( 3 ) );
        EXPECT_TRUE( emptied.remove( 3 ) );
        emptied ^= source;
        EXPECT_TRUE( emptied.shares_storage_with( source ) );
    }
    // the handle's own bitmap as the operand, shared or not
    for ( bool const shared : { false, true } ) {
        fr::shared_bitmap<bitmap> copy{ sample };
        auto const other{ shared ? copy : fr::shared_bitmap<bitmap>{} };
        copy -= copy.get();
        EXPECT_TRUE( copy.empty() );
        expect_coherent( copy );
        EXPECT_EQ( other.size(), shared ? sample.size() : 0U );
        copy = fr::shared_bitmap<bitmap>{ sample };
        auto const again{ shared ? copy : fr::shared_bitmap<bitmap>{} };
        copy ^= copy.get();
        EXPECT_TRUE( copy.empty() );
        expect_coherent( copy );
        EXPECT_EQ( again.size(), shared ? sample.size() : 0U );
    }
}

// A random sequence of changes and copies over a few handles agrees with plain bitmaps changed the same way, and no
// change shows through a handle that shared the bitmap before it.
TYPED_TEST( SharedBitmap, ChangesAgreeWithPlainBitmaps ) {
    using bitmap = TypeParam;
    using key_type = typename bitmap::key_type;
    using handle_type = fr::shared_bitmap<bitmap>;
    constexpr std::size_t slots{ 6 };
    std::vector<handle_type> handles( slots );
    std::vector<bitmap> models( slots );
    std::mt19937 rng{ 20261106 };
    auto const random_key{ [ & ] { return static_cast<key_type>( ( rng() % 6 ) * 65'536 + rng() % 9'000 ); } };
    auto const random_bitmap{ [ & ] {
        bitmap result;
        auto const count{ rng() % 3 == 0 ? 0U : rng() % 3'000 };
        for ( std::uint32_t i{ 0 }; i < count; ++i ) {
            std::ignore = result.add( random_key() );
        }
        if ( rng() % 2 == 0 ) {
            result.optimize();
        }
        return result;
    } };
    for ( std::uint32_t step{ 0 }; step < 3'000; ++step ) {
        auto const at   { rng() % slots };
        auto const other{ rng() % slots };
        auto & handle{ handles[ at ] };
        auto & model { models [ at ] };
        switch ( rng() % 14 ) {
            case 0:
                handle = handles[ other ];
                model = models[ other ];
                break;
            case 1: {
                auto const key{ random_key() };
                EXPECT_EQ( handle.add( key ), model.add( key ) );
                break;
            }
            case 2: {
                auto const key{ random_key() };
                EXPECT_EQ( handle.remove( key ), model.remove( key ) );
                break;
            }
            case 3:
                handle |= handles[ other ];
                model |= models[ other ];
                break;
            case 4:
                handle &= handles[ other ];
                model &= models[ other ];
                break;
            case 5:
                handle -= handles[ other ];
                model -= models[ other ];
                break;
            case 6:
                handle ^= handles[ other ];
                model ^= models[ other ];
                break;
            case 7: {
                auto const operand{ random_bitmap() };
                handle |= operand;
                model |= operand;
                break;
            }
            case 8: {
                auto const operand{ random_bitmap() };
                handle &= operand;
                model &= operand;
                break;
            }
            case 9: {
                auto const operand{ random_bitmap() };
                handle -= operand;
                model -= operand;
                break;
            }
            case 10: {
                auto const operand{ random_bitmap() };
                handle ^= operand;
                model ^= operand;
                break;
            }
            case 11: {
                // an operand that is another handle's bitmap, which may be this handle's own
                auto const operand_model{ models[ other ] };
                handle |= handles[ other ].get();
                model |= operand_model;
                break;
            }
            case 12: {
                auto const first{ random_key() };
                auto editor{ handle.edit() };
                editor->add_closed_range( first, first + 300 );
                editor->optimize();
                model.add_closed_range( first, first + 300 );
                break;
            }
            default:
                handle.clear();
                model.clear();
                break;
        }
        for ( std::size_t i{ 0 }; i < slots; ++i ) {
            ASSERT_EQ( handles[ i ].size(), models[ i ].size() ) << "step " << step << ", handle " << i;
            ASSERT_TRUE( handles[ i ].get() == models[ i ] ) << "step " << step << ", handle " << i;
            ASSERT_EQ( handles[ i ].container_count(), handles[ i ].get().container_count() );
        }
    }
}

TYPED_TEST( SharedBitmap, AnEditorChangesTheHandlesOwnBitmap ) {
    using bitmap = TypeParam;
    auto const sample{ make_sample<bitmap>() };
    fr::shared_bitmap<bitmap> const source{ sample };
    auto copy{ source };
    {
        auto editor{ copy.edit() };
        EXPECT_TRUE( copy.is_unique() );
        EXPECT_FALSE( copy.shares_storage_with( source ) );
        editor->add_closed_range( 7 * 65'536, 7 * 65'536 + 99 );
        std::ignore = ( *editor ).remove( 0 );
    }
    EXPECT_EQ( copy.size(), sample.size() + 100 - 1 );
    expect_coherent( copy );
    EXPECT_EQ( keys_of( source.get() ), keys_of( sample ) );

    // an empty handle is given a bitmap of its own to edit
    fr::shared_bitmap<bitmap> empty;
    {
        auto editor{ empty.edit() };
        std::ignore = editor->add( 3 );
    }
    EXPECT_EQ( empty.size(), 1U );
}

// A change that has to copy a shared bitmap and fails to leaves the handle as it was: it still shares the bitmap, which
// holds what it held, and the handle's cardinality is the bitmap's.
TYPED_TEST( SharedBitmap, AFailedCopyLeavesTheHandleAsItWas ) {
    using bitmap = TypeParam;
#if FRSR_ROARING_HAS_MIMALLOC || !FRSR_TESTS_COUNT_ALLOCATIONS
    GTEST_SKIP() << "allocations cannot be made to fail in this build";
#else
    auto const sample{ make_sample<bitmap>() };
    auto const sample_keys{ keys_of( sample ) };
    bitmap operand;
    std::ignore = operand.add( 2 * 65'536 + 4 );
    std::ignore = operand.add( 65'536 + 3 );
    fr::shared_bitmap<bitmap> const source{ sample };
    struct change {
        char const * name;
        void ( *apply )( fr::shared_bitmap<bitmap> &, bitmap const & );
    };
    change const changes[]{
        { "add",    []( fr::shared_bitmap<bitmap> & h, bitmap const & ) { std::ignore = h.add( 2 * 65'536 + 4 ); } },
        { "remove", []( fr::shared_bitmap<bitmap> & h, bitmap const & ) { std::ignore = h.remove( 65'536 + 3 ); } },
        { "|=",     []( fr::shared_bitmap<bitmap> & h, bitmap const & o ) { h |= o; } },
        { "&=",     []( fr::shared_bitmap<bitmap> & h, bitmap const & o ) { h &= o; } },
        { "&= a handle", []( fr::shared_bitmap<bitmap> & h, bitmap const & o ) { h &= fr::shared_bitmap<bitmap>{ o }; } },
        { "-=",     []( fr::shared_bitmap<bitmap> & h, bitmap const & o ) { h -= o; } },
        { "^=",     []( fr::shared_bitmap<bitmap> & h, bitmap const & o ) { h ^= o; } },
        { "edit",   []( fr::shared_bitmap<bitmap> & h, bitmap const & ) { std::ignore = h.edit()->add( 9 ); } },
    };
    for ( auto const & [ name, apply ] : changes ) {
        bool failed_any{ false };
        for ( std::uint32_t fail_at{ 1 }; fail_at < 400; ++fail_at ) {
            SCOPED_TRACE( testing::Message() << name << " failing allocation " << fail_at );
            auto copy{ source };
            bool threw{ false };
            alloc_fault::counter = 0;
            try {
                alloc_fault::armed const armed{ fail_at };
                apply( copy, operand );
            } catch ( std::bad_alloc const & ) {
                threw = true;
            }
            alloc_fault::counter = 0;
            EXPECT_EQ( keys_of( source.get() ), sample_keys ) << name << " failing allocation " << fail_at;
            expect_coherent( copy );
            if ( threw && copy.shares_storage_with( source ) ) {
                failed_any = true;
                EXPECT_FALSE( source.is_unique() ) << name << " failing allocation " << fail_at;
                EXPECT_EQ( copy.size(), sample.size() ) << name << " failing allocation " << fail_at;
            }
            if ( !threw ) {
                break;
            }
        }
        EXPECT_TRUE( failed_any ) << name;
        EXPECT_TRUE( source.is_unique() ) << name;
    }
#endif
}

// Threads copy a published handle, read through their copies, change some of them (which copies the bitmap) and drop
// them, while the handle stays published; under ThreadSanitizer any unsynchronized write to a shared bitmap or count is
// reported. Not for the policy that is for one thread at a time.
TYPED_TEST( SharedBitmap, ThreadsShareAHandle ) {
    using bitmap = TypeParam;
    if constexpr ( std::is_same_v<typename bitmap::cow_policy, fr::detail::cow_unsynchronized_refcount> ) {
        GTEST_SKIP() << "the unsynchronized policy is for one thread at a time";
    } else {
        auto const sample{ make_sample<bitmap>() };
        auto const expected_size{ sample.size() };
        fr::shared_bitmap<bitmap> const published{ sample };
        constexpr unsigned thread_count{ 6 };
        constexpr unsigned rounds{ 300 };
        std::atomic<bool> go{ false };
        std::atomic<unsigned> failures{ 0 };
        {
            std::vector<std::jthread> threads;
            for ( unsigned t{ 0 }; t != thread_count; ++t ) {
                threads.emplace_back( [ &, t ] {
                    while ( !go.load( std::memory_order_acquire ) ) {
                        std::this_thread::yield();
                    }
                    for ( unsigned round{ 0 }; round != rounds; ++round ) {
                        auto copy{ published };
                        bool ok{ copy.size() == expected_size && copy.contains( 65'536 + 3 ) && copy.get().size() == expected_size };
                        if ( ( round + t ) % 3 == 0 ) {
                            ok &= copy.add( 8 * 65'536 + t );
                            ok &= copy.size() == expected_size + 1;
                            ok &= copy.is_unique();
                        }
                        if ( ( round + t ) % 5 == 0 ) {
                            auto again{ copy };
                            again -= published;
                            ok &= again.size() == ( copy.shares_storage_with( published ) ? 0U : 1U );
                        }
                        if ( !ok ) {
                            failures.fetch_add( 1, std::memory_order_relaxed );
                        }
                    }
                } );
            }
            go.store( true, std::memory_order_release );
        }
        EXPECT_EQ( failures.load(), 0U );
        EXPECT_TRUE( published.is_unique() );
        EXPECT_EQ( keys_of( published.get() ), keys_of( sample ) );
    }
}

// Two handles share a bitmap and nothing else holds it: one thread changes its handle (which copies the bitmap, or
// changes it in place once the other has let it go) with the shared bitmap as the operand, while the other drops its
// handle. Under ThreadSanitizer a change that reads a bitmap freed under it, or writes one still shared, is reported.
TYPED_TEST( SharedBitmap, AChangeRacesTheOtherHandlesDrop ) {
    using bitmap = TypeParam;
    if constexpr ( std::is_same_v<typename bitmap::cow_policy, fr::detail::cow_unsynchronized_refcount> ) {
        GTEST_SKIP() << "the unsynchronized policy is for one thread at a time";
    } else {
        auto const sample{ make_sample<bitmap>() };
        bitmap disjoint;
        std::ignore = disjoint.add( 9 * 65'536 + 1 );
        unsigned failures{ 0 };
        for ( unsigned round{ 0 }; round != 200; ++round ) {
            fr::shared_bitmap<bitmap> changed{ sample };
            auto dropped{ changed };
            std::atomic<bool> go{ false };
            std::jthread dropper{ [ & ] {
                while ( !go.load( std::memory_order_acquire ) ) {
                    std::this_thread::yield();
                }
                dropped.clear();
            } };
            go.store( true, std::memory_order_release );
            switch ( round % 3 ) {
                case 0:
                    changed -= changed.get();
                    failures += changed.empty() ? 0U : 1U;
                    break;
                case 1:
                    changed |= disjoint;
                    failures += changed.size() == sample.size() + 1 ? 0U : 1U;
                    break;
                default:
                    changed ^= changed.get();
                    failures += changed.empty() ? 0U : 1U;
                    break;
            }
            dropper.join();
            failures += changed.is_unique() ? 0U : 1U;
        }
        EXPECT_EQ( failures, 0U );
    }
}

} // namespace
