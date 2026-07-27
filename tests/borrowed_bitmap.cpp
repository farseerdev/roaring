// Phase-3 mmap-borrow gates (design doc "Migration", Phase 3): frozen-buffer
// borrow round-trip, master immutability under scratchpad mutation, scratchpad
// independence, format-version rejection, and the persistent_bitmap file
// round-trip. The frozen buffer stands in for the read-only mmapped master;
// byte-level snapshots (not page protection) assert it is never written.

#include <frsr/roaring/bitmap.hpp>
#include <frsr/roaring/persistent_bitmap.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace {

using TestBitmap = frsr::roaring::bitmap<std::uint32_t>;

// Spans all three container kinds (array, bitset, run after optimisation) so
// the borrow paths exercise every frozen_container_kind.
[[nodiscard]] TestBitmap make_mixed_container_bitmap() {
    TestBitmap bitmap;
    for ( std::uint32_t const value : { 1U, 7U, 100U, 4'096U, 60'000U } ) {
        std::ignore = bitmap.add( value );
    }
    bitmap.add_closed_range( 65'536U, 70'536U );
    bitmap.add_closed_range( 131'072U, 134'072U );
    for ( std::uint32_t value{ 196'608U }; value < 216'608U; value += 2U ) {
        std::ignore = bitmap.add( value );   // dense alternating chunk: survives optimisation as a bitset
    }
    bitmap.optimize_for_storage();
    return bitmap;
}

[[nodiscard]] std::vector<std::byte> snapshot( TestBitmap::serialized_byte_vector const & buffer ) {
    return { buffer.data(), buffer.data() + buffer.size() };
}

[[nodiscard]] bool bytes_equal( TestBitmap::serialized_byte_vector const & buffer, std::vector<std::byte> const & baseline ) {
    return buffer.size() == baseline.size()
        && std::memcmp( buffer.data(), baseline.data(), baseline.size() ) == 0;
}

TEST(FrsrRoaringBorrow, BorrowIntoRoundTripsWithoutPayloadCopies) {
    auto const source{ make_mixed_container_bitmap() };
    auto const baseline{ source.to_vector() };

    TestBitmap::serialized_byte_vector buffer;
    source.serialize_frozen_to_vm_vector( buffer );

    auto const view{ TestBitmap::frozen_view_from_vm_vector( buffer ) };
    ASSERT_TRUE( static_cast<bool>( view ) );

    TestBitmap borrowed;
    view.borrow_into( borrowed );
    EXPECT_EQ( borrowed.size(), source.size() );
    EXPECT_EQ( borrowed.to_vector(), baseline );
    for ( std::uint32_t const value : {
        0U, 1U, 2U, 7U, 100U, 4'096U, 4'097U, 60'000U,
        65'535U, 65'536U, 67'000U, 70'536U, 70'537U,
        131'071U, 131'072U, 132'500U, 134'072U, 134'073U, 200'000U
    } ) {
        EXPECT_EQ( borrowed.contains( value ), source.contains( value ) ) << "value " << value;
    }
}

// THE Phase-3 gate: a scratchpad over an mmapped master, mutated across every
// container kind — the master bytes must be bit-identical afterwards and a
// second scratchpad over the same master must be unaffected.
TEST(FrsrRoaringBorrow, ScratchpadMutationLeavesMasterAndSiblingScratchpadIntact) {
    auto const source{ make_mixed_container_bitmap() };
    auto const baseline{ source.to_vector() };

    TestBitmap::serialized_byte_vector master;
    source.serialize_frozen_to_vm_vector( master );
    auto const master_snapshot{ snapshot( master ) };

    auto const view{ TestBitmap::frozen_view_from_vm_vector( master ) };
    ASSERT_TRUE( static_cast<bool>( view ) );

    TestBitmap scratchpad;
    view.borrow_into( scratchpad );
    TestBitmap sibling{ scratchpad };   // second scratchpad: shares the borrowed payloads

    // Mutate every container kind in the first scratchpad: the array chunk
    // (add + remove), the run chunk (punch a hole), the bitset chunk (add),
    // plus a brand-new chunk.
    EXPECT_TRUE ( scratchpad.add( 2U ) );
    EXPECT_TRUE ( scratchpad.remove( 7U ) );
    EXPECT_TRUE ( scratchpad.remove( 67'000U ) );
    EXPECT_TRUE ( scratchpad.add( 131'000U ) );
    EXPECT_TRUE ( scratchpad.add( 196'609U ) );
    EXPECT_TRUE ( scratchpad.remove( 196'608U ) );
    EXPECT_TRUE ( scratchpad.add( 1'000'000U ) );

    EXPECT_TRUE( bytes_equal( master, master_snapshot ) );   // the master is never written

    EXPECT_TRUE ( scratchpad.contains( 2U ) );
    EXPECT_FALSE( scratchpad.contains( 7U ) );
    EXPECT_FALSE( scratchpad.contains( 67'000U ) );

    // The sibling scratchpad and a fresh borrow are both unaffected.
    EXPECT_EQ( sibling.to_vector(), baseline );
    TestBitmap fresh;
    view.borrow_into( fresh );
    EXPECT_EQ( fresh.to_vector(), baseline );
}

TEST(FrsrRoaringBorrow, SetOperationsOnBorrowedBitmapsLeaveMasterIntact) {
    auto const source{ make_mixed_container_bitmap() };

    TestBitmap::serialized_byte_vector master;
    source.serialize_frozen_to_vm_vector( master );
    auto const master_snapshot{ snapshot( master ) };

    auto const view{ TestBitmap::frozen_view_from_vm_vector( master ) };
    ASSERT_TRUE( static_cast<bool>( view ) );

    TestBitmap borrowed;
    view.borrow_into( borrowed );

    TestBitmap other;
    other.add_closed_range( 60'000U, 66'000U );
    std::ignore = other.add( 131'072U );

    EXPECT_EQ( ( borrowed & other ).to_vector(), ( source & other ).to_vector() );
    EXPECT_EQ( ( borrowed | other ).to_vector(), ( source | other ).to_vector() );
    EXPECT_EQ( ( borrowed - other ).to_vector(), ( source - other ).to_vector() );

    borrowed &= other;   // in-place mutation over borrowed slots
    EXPECT_EQ( borrowed.to_vector(), ( source & other ).to_vector() );
    EXPECT_TRUE( bytes_equal( master, master_snapshot ) );
}

TEST(FrsrRoaringBorrow, WrongFormatVersionIsRejected) {
    auto const source{ make_mixed_container_bitmap() };

    TestBitmap::serialized_byte_vector buffer;
    source.serialize_frozen_to_vm_vector( buffer );
    ASSERT_TRUE( static_cast<bool>( TestBitmap::frozen_view_from_vm_vector( buffer ) ) );

    // The version field is the u16 at byte offset 6 (docs/frozen-format.md).
    std::uint16_t bogus_version{ 0xFFFFU };
    std::memcpy( buffer.data() + 6, &bogus_version, sizeof( bogus_version ) );
    EXPECT_FALSE( static_cast<bool>( TestBitmap::frozen_view_from_vm_vector( buffer ) ) );
}

#if defined(FRSR_ROARING_HAS_PSI_VM) && defined(FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION)

TEST(FrsrRoaringBorrow, PersistentBitmapFileRoundTrip) {
    auto const source{ make_mixed_container_bitmap() };
    auto const baseline{ source.to_vector() };
    auto const file_name{ std::string{ ::testing::TempDir() } + "frsr_persistent_bitmap.frozen" };

    ASSERT_TRUE( ( frsr::roaring::persistent_bitmap<std::uint32_t>::store( file_name.c_str(), source ) ) );

    {
        auto opened{ frsr::roaring::persistent_bitmap<std::uint32_t>::open( file_name.c_str() ) };
        ASSERT_TRUE( opened.has_value() );
        EXPECT_EQ( ( *opened )->size(), source.size() );
        EXPECT_EQ( ( *opened )->to_vector(), baseline );

        // Mutations CoW into private storage; a reopen must see the original.
        EXPECT_TRUE( ( *opened )->add( 5U ) );
        EXPECT_TRUE( ( *opened )->remove( 7U ) );
    }
    {
        auto const reopened{ frsr::roaring::persistent_bitmap<std::uint32_t>::open( file_name.c_str() ) };
        ASSERT_TRUE( reopened.has_value() );
        EXPECT_EQ( ( *reopened )->to_vector(), baseline );
    }

    std::remove( file_name.c_str() );
}

TEST(FrsrRoaringBorrow, PersistentBitmapOpenRejectsGarbage) {
    auto const file_name{ std::string{ ::testing::TempDir() } + "frsr_persistent_bitmap.garbage" };
    {
        auto * const f{ std::fopen( file_name.c_str(), "wb" ) };
        ASSERT_NE( f, nullptr );
        std::fputs( "definitely not a frozen bitmap", f );
        std::fclose( f );
    }
    EXPECT_FALSE( frsr::roaring::persistent_bitmap<std::uint32_t>::open( file_name.c_str() ).has_value() );
    std::remove( file_name.c_str() );
}

#endif // FRSR_ROARING_HAS_PSI_VM && FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION

} // anonymous namespace
