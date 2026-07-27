#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace frsr::roaring {

namespace detail {

template <std::unsigned_integral Key>
struct default_layout {
    static constexpr unsigned key_bits{ std::numeric_limits<Key>::digits };
    static constexpr unsigned low_bits{ key_bits < 16 ? key_bits : 16 };
    static constexpr std::size_t low_domain_size{ std::size_t{ 1 } << low_bits };
    static constexpr std::size_t word_count{ low_domain_size / 64 };
    static constexpr Key low_mask{
        low_bits == key_bits
            ? std::numeric_limits<Key>::max()
            : static_cast<Key>( ( Key{ 1 } << low_bits ) - 1 )
    };

    using key_type = Key;
    using chunk_type = std::conditional_t<( key_bits <= 32 ), std::uint16_t, Key>;
    using low_type = std::conditional_t<( low_bits <= 8 ), std::uint8_t, std::uint16_t>;

    static constexpr chunk_type chunk_key( key_type const value ) noexcept {
        if constexpr ( key_bits == low_bits ) {
            return static_cast<chunk_type>( 0 );
        } else {
            return static_cast<chunk_type>( value >> low_bits );
        }
    }

    static constexpr low_type low_key( key_type const value ) noexcept {
        return static_cast<low_type>( value & low_mask );
    }

    static constexpr key_type compose( chunk_type const chunk, low_type const low ) noexcept {
        if constexpr ( key_bits == low_bits ) {
            return static_cast<key_type>( low );
        } else {
            return static_cast<key_type>( ( static_cast<key_type>( chunk ) << low_bits ) | static_cast<key_type>( low ) );
        }
    }
};

} // namespace detail

template <std::unsigned_integral Key>
using container_layout = detail::default_layout<Key>;

} // namespace frsr::roaring
