// Shared by the tests of views as operands: a look at the containers of a bitmap, to prove what an operation
// reading through a view writes (nothing) and what a result keeps of the viewed bitmap (nothing).
#pragma once

#include <frsr/roaring/bitmap.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace frsr::roaring::detail {

template <typename Bitmap>
struct bitmap_test_access {
    using handle_type = typename Bitmap::handle_type;

    // Everything an operation could write into a published container, as bytes taken before and after it:
    // the 32-byte handle, the count of the payload, and the payload itself.
    struct container_state {
        std::uint64_t                  key{};
        std::uint32_t                  refcount{};
        bool                           endpoints_valid{};
        void const *                   payload{};
        std::vector<unsigned char>     header;
        std::vector<unsigned char>     content;

        [[nodiscard]] friend bool operator==( container_state const &, container_state const & ) = default;
    };

    [[nodiscard]] static std::vector<container_state> snapshot( Bitmap const & bitmap ) {
        std::vector<container_state> result;
        auto const & chunks{ bitmap.chunks_ };
        for ( auto at{ chunks.begin() }; at != chunks.end(); ++at ) {
            auto const & slot{ chunks.slot( at ) };
            container_state state;
            state.key      = static_cast<std::uint64_t>( chunks.key( at ) );
            state.refcount        = slot.payload_refcount();
            state.endpoints_valid = slot.endpoints_valid();
            state.header.resize( sizeof( handle_type ) );
            std::memcpy( state.header.data(), static_cast<void const *>( &slot ), sizeof( handle_type ) );
            auto const keep{ [ & ]( void const * const data, std::size_t const bytes ) {
                state.payload = slot.spilled() ? data : nullptr;
                state.content.resize( bytes );
                if ( bytes != 0 ) { std::memcpy( state.content.data(), data, bytes ); }
            } };
            switch ( slot.kind() ) {
                case container_kind::array : { auto const values{ slot.as_array().values }; keep( values.data(), values.size() * sizeof( typename Bitmap::low_type ) ); break; }
                case container_kind::run   : { auto const runs{ slot.as_run().runs }; keep( runs.data(), runs.size() * sizeof( runs[ 0 ] ) ); break; }
                case container_kind::bitset: { auto const & words{ slot.as_bitset().words.as_array() }; keep( words.data(), sizeof( words ) ); break; }
            }
            result.push_back( std::move( state ) );
        }
        return result;
    }
};

} // namespace frsr::roaring::detail
