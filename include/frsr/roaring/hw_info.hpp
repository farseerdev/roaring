#pragma once

#include <frsr/roaring/operations.hpp>

#include <cstddef>
#include <cstdint>

namespace frsr::roaring::detail {

// Compile-time SIMD characteristics of the build target. The bitset word kernels
// size their portable wide-vector tiles from these constants — one kernel, no
// per-ISA duplication: the compiler lowers a wide gnu-vector op to that many native
// instructions (AVX2 vpor, AVX-512 vpord, NEON orr, WASM v128.or). Try the portable
// route first; add a target-specific path only where it measurably wins.
struct hw_info {
    static constexpr std::size_t simd_register_bytes{
#if defined( __AVX512F__ )
        64
#elif defined( __AVX2__ )
        32
#elif defined( __SSE2__ ) || defined( _M_X64 ) || defined( __aarch64__ ) || defined( __ARM_NEON ) || defined( __wasm_simd128__ )
        16
#else
        8
#endif
    };
    static constexpr std::size_t simd_register_count{
#if defined( __aarch64__ ) || defined( __ARM_NEON ) || defined( __AVX512F__ )
        32
#else
        16
#endif
    };
    // Native SIMD vectors processed per kernel tile. Half the register file keeps a
    // pure-bitwise tile (result + addressing) resident without spilling on x86 (the
    // op takes a memory operand, ~1 reg/lane) and ARM (orr needs both operands in
    // registers, ~2 regs/lane); capped at 8 — CRoaring's hand-tuned unroll — past
    // which the 8 KB block is bandwidth-bound and extra unrolling stops paying.
    static constexpr std::size_t kernel_tile_vectors{ ( simd_register_count / 2U < 8U ) ? simd_register_count / 2U : 8U };
    static constexpr std::size_t bitset_tile_words{ simd_register_bytes * kernel_tile_vectors / sizeof( std::uint64_t ) };
};

// Wide gnu-vector of uint64 spanning one kernel tile (kernel_tile_vectors native
// SIMD registers). aligned(1) → unaligned access (the 8 KB block is only 16-byte
// aligned, like CRoaring's lddqu); may_alias → safe to reach through a uint64 array.
typedef std::uint64_t bitset_word_tile
    __attribute__(( vector_size( hw_info::bitset_tile_words * sizeof( std::uint64_t ) ), aligned( 1 ), may_alias ));

// One native SIMD register of uint64 lanes (AVX2: 4, AVX-512: 8, SSE2/NEON: 2). The
// Harley-Seal carry-save tree below operates one register at a time so its ~12 live
// accumulators fit the register file; the wider bitset_word_tile is for the
// cardinality-free lazy OR, where a single streaming store never spills.
typedef std::uint64_t bitset_reg
    __attribute__(( vector_size( hw_info::simd_register_bytes ), aligned( 1 ), may_alias ));

// Carry-save adder over one SIMD register: a full-adder on bit-vectors. Given a, b, c
// each contributing 0/1 per bit position, writes the weight-2 part to h and the
// weight-1 part to l. The kernel of the Harley-Seal population count.
// [croaring-ref] deps/croaring/include/roaring/bitset_util.h:CSA
template <typename V>
[[ gnu::always_inline ]] inline void carry_save_add( V & h, V & l, V const a, V const b, V const c ) noexcept {
    V const u{ a ^ b };
    h = ( a & b ) | ( u & c );
    l = u ^ c;
}

template <set_operation Op, typename V>
[[ gnu::always_inline ]] inline V apply_bitwise_op( V const a, V const b ) noexcept {
    if      constexpr ( Op == set_operation::bit_or  ) { return a | b;  }
    else if constexpr ( Op == set_operation::bit_and ) { return a & b;  }
    else                                               { return a & ~b; }  // bit_andnot
}

} // namespace frsr::roaring::detail
