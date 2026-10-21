// Shared by the interop tests: bitmaps of every container kind, operands related to them, and a cheap
// oracle (sorted vectors) to compare them with.
#pragma once

#include <frsr/roaring/bitmap.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace frsr::roaring::interop_test {

// The members of a set, ascending without repeats (a sorted vector: the oracle must be cheap beside the code under test).
using oracle_set = std::vector<std::uint64_t>;

inline void finalize( oracle_set & keys ) {
    std::ranges::sort( keys );
    keys.erase( std::unique( keys.begin(), keys.end() ), keys.end() );
}

// ---- bit-set doubles ------------------------------------------------------------------------------------

// The shape of boost::dynamic_bitset with 32-bit blocks (the block type of the Windows default), without Boost.
class block_bits {
public:
    using block_type = std::uint32_t;
    explicit block_bits( std::size_t const bits ) : bits_{ bits }, blocks_( ( bits + 31 ) / 32 ) {}
    void set( std::size_t const i ) { blocks_[ i / 32 ] |= block_type{ 1 } << ( i % 32 ); }
    [[nodiscard]] std::size_t size() const { return bits_; }
    [[nodiscard]] std::size_t num_blocks() const { return blocks_.size(); }
    template <typename Out>
    friend void to_block_range( block_bits const & bits, Out out ) { std::copy( bits.blocks_.begin(), bits.blocks_.end(), out ); }
private:
    std::size_t bits_;
    std::vector<block_type> blocks_;
};

// A bit-set that can only be walked by its set bits.
class find_only_bits {
public:
    static constexpr std::size_t npos{ ~std::size_t{ 0 } };
    explicit find_only_bits( std::size_t const bits ) : bits_( bits, false ) {}
    void set( std::size_t const i ) { bits_[ i ] = true; }
    [[nodiscard]] std::size_t size() const { return bits_.size(); }
    [[nodiscard]] std::size_t find_first() const { return scan( 0 ); }
    [[nodiscard]] std::size_t find_next( std::size_t const pos ) const { return scan( pos + 1 ); }
private:
    [[nodiscard]] std::size_t scan( std::size_t i ) const {
        for ( ; i < bits_.size(); ++i ) { if ( bits_[ i ] ) { return i; } }
        return npos;
    }
    std::vector<bool> bits_;
};

// ---- bitmaps with every container kind ------------------------------------------------------------------

template <typename Bitmap>
struct domain {
    using key_type = typename Bitmap::key_type;
    static constexpr std::uint64_t max{ std::numeric_limits<key_type>::max() };
    static constexpr bool single_chunk{ std::numeric_limits<key_type>::digits <= 16 };

    // chunks that hold the data: the first ones, a distant one, the last ones
    static std::vector<std::uint64_t> chunks() {
        if constexpr ( single_chunk ) {
            return { 0 };
        } else if constexpr ( std::numeric_limits<key_type>::digits == 32 ) {
            return { 0, 1, 2, 3, 7, 1000, 65534, 65535 };
        } else {
            return { 0, 1, 2, 5, 65535, std::uint64_t{ 1 } << 30, std::uint64_t{ 1 } << 40, ( max >> 16 ) - 1, max >> 16 };
        }
    }
};

enum class shape { sparse, array, dense_bitset, runs, full, ends, single, mixed };

// The keys of one chunk in the given shape
inline void fill_chunk( oracle_set & keys, std::uint64_t const chunk, shape const kind, std::mt19937_64 & rng, std::uint64_t const max ) {
    std::uint64_t const base{ chunk << 16 };
    auto const add{ [ & ]( std::uint64_t const low ) {
        if ( base + low <= max ) { keys.push_back( base + low ); }
    } };
    switch ( kind ) {
        case shape::sparse : for ( int i{ 0 }; i < 5; ++i ) { add( rng() & 0xFFFF ); } break;
        case shape::array  : for ( int i{ 0 }; i < 3000; ++i ) { add( rng() & 0xFFFF ); } break;
        case shape::dense_bitset: for ( int i{ 0 }; i < 30000; ++i ) { add( rng() & 0xFFFF ); } break;
        case shape::runs: {
            for ( int r{ 0 }; r < 12; ++r ) {
                auto const begin{ rng() & 0xFFFF };
                auto const length{ 1 + rng() % 1500 };
                for ( std::uint64_t i{ 0 }; i < length && begin + i <= 0xFFFF; ++i ) { add( begin + i ); }
            }
            break;
        }
        case shape::full  : for ( std::uint64_t i{ 0 }; i < 65536; ++i ) { add( i ); } break;
        case shape::ends  : add( 0 ); add( 1 ); add( 0xFFFE ); add( 0xFFFF ); break;
        case shape::single: add( rng() & 0xFFFF ); break;
        case shape::mixed : for ( int i{ 0 }; i < 9000; ++i ) { add( ( rng() & 0xFFFF ) & ( rng() & 0xFFFF ) ); } break;
    }
}

template <typename Bitmap>
Bitmap make_bitmap( oracle_set const & keys, bool const optimize ) {
    std::vector<typename Bitmap::key_type> values( keys.begin(), keys.end() );
    Bitmap bitmap{ values };
    if ( optimize ) { bitmap.optimize(); }
    return bitmap;
}

template <typename Bitmap>
oracle_set random_content( std::mt19937_64 & rng ) {
    oracle_set keys;
    constexpr shape shapes[]{ shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::ends, shape::single, shape::mixed };
    for ( auto const chunk : domain<Bitmap>::chunks() ) {
        if ( rng() % 4 == 0 ) { continue; }
        fill_chunk( keys, chunk, shapes[ rng() % std::size( shapes ) ], rng, domain<Bitmap>::max );
    }
    finalize( keys );
    return keys;
}

// keys related to `content`: part of it, plus keys near it and elsewhere, possibly beyond the key type
template <typename Bitmap>
oracle_set random_related( std::mt19937_64 & rng, oracle_set const & content, bool const allow_out_of_domain ) {
    oracle_set keys;
    constexpr double fractions[]{ 0.0, 0.01, 0.5, 1.0 };
    auto const fraction{ fractions[ rng() % std::size( fractions ) ] };
    std::uniform_real_distribution<double> unit{ 0.0, 1.0 };
    for ( auto const key : content ) { if ( unit( rng ) < fraction ) { keys.push_back( key ); } }
    auto const chunks{ domain<Bitmap>::chunks() };
    constexpr shape shapes[]{ shape::sparse, shape::array, shape::dense_bitset, shape::runs, shape::full, shape::ends, shape::single, shape::mixed };
    for ( auto const chunk : chunks ) {
        if ( rng() % 3 == 0 ) { fill_chunk( keys, chunk, shapes[ rng() % std::size( shapes ) ], rng, domain<Bitmap>::max ); }
    }
    if ( allow_out_of_domain && domain<Bitmap>::max < std::numeric_limits<std::uint64_t>::max() && rng() % 3 == 0 ) {
        keys.push_back( domain<Bitmap>::max + 1 );
    }
    finalize( keys );
    return keys;
}

// ---- oracle ---------------------------------------------------------------------------------------------

inline oracle_set in_domain( oracle_set const & keys, std::uint64_t const max ) {
    oracle_set result;
    for ( auto const key : keys ) { if ( key <= max ) { result.push_back( key ); } }
    return result;
}
inline oracle_set intersection( oracle_set const & a, oracle_set const & b ) {
    oracle_set result;
    std::set_intersection( a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( result ) );
    return result;
}
inline oracle_set unite( oracle_set const & a, oracle_set const & b ) {
    oracle_set result;
    std::set_union( a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( result ) );
    return result;
}
inline oracle_set difference( oracle_set const & a, oracle_set const & b ) {
    oracle_set result;
    std::set_difference( a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( result ) );
    return result;
}

inline oracle_set symmetric_difference( oracle_set const & a, oracle_set const & b ) {
    oracle_set result;
    std::set_symmetric_difference( a.begin(), a.end(), b.begin(), b.end(), std::back_inserter( result ) );
    return result;
}

// random bit string over [base, base + bits) as words and as the set of its keys
struct bit_string {
    std::vector<std::uint64_t> words;
    std::uint64_t              base;
    std::uint64_t              bits;
    oracle_set                 members;
};

inline bit_string random_bit_string( std::mt19937_64 & rng, std::uint64_t const base, std::uint64_t const bits, std::uint64_t const max ) {
    bit_string result{ std::vector<std::uint64_t>( ( bits + 63 ) / 64 ), base, bits, {} };
    constexpr double densities[]{ 0.0, 0.0005, 0.02, 0.5, 0.97, 1.0 };
    auto const density{ densities[ rng() % std::size( densities ) ] };
    std::uniform_real_distribution<double> unit{ 0.0, 1.0 };
    for ( std::uint64_t i{ 0 }; i < bits; ++i ) {
        if ( unit( rng ) < density ) {
            result.words[ i / 64 ] |= std::uint64_t{ 1 } << ( i % 64 );
            result.members.push_back( base + i );
        }
    }
    (void)max;
    // the padding bits of the last word are set when the data has any, to prove they never leak
    if ( bits % 64 != 0 && !result.words.empty() ) { result.words.back() |= ~std::uint64_t{ 0 } << ( bits % 64 ); }
    return result;
}


} // namespace frsr::roaring::interop_test
