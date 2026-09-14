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

// ---- x86 runtime dispatch (function multi-versioning) ----------------------
// The shipped baselines are AVX-512-less (-march=skylake class) while current
// server silicon has AVX-512 incl. VPOPCNTDQ/VBMI2, which roughly halves the 8 KB
// bitset word loops (512-bit ops + native vector popcount) and enables the
// 32-lane bitonic array merge and the compress-store decoders — CRoaring
// runtime-dispatches the same kernels. clang does not allow target attributes on
// templates, so the dispatchable kernels are plain functions over raw spans; the
// templated entry points funnel into them when the CPU qualifies. When the whole
// TU already targets the tier (-march=native builds) the same kernels compile as
// ordinary inlinable functions and the runtime check folds to true.
//
// ⚠ Site selection is MEASURED, not uniform (bisected on an AVX-512 x86 server,
// 2026-07-19): dispatching the in-place bitset combine — which inlines into the
// merge-walk spine — regressed the bitset-heavy workload ~4% in context despite
// winning its microbench (outlining cost inside the spine), and the
// repair_cardinality popcount dispatch measured neutral-to-negative. Don't add
// sites without an in-context full-scale A/B.
#if ( defined( __x86_64__ ) || defined( _M_X64 ) ) && defined( __clang__ )
#   if defined( __AVX512VPOPCNTDQ__ ) && defined( __AVX512VBMI2__ ) && defined( __AVX512BW__ ) && defined( __AVX512VL__ )
#       define FRSR_ROARING_X86_V4_NATIVE   1
#       define FRSR_ROARING_X86_V4_DISPATCH 0
#   else
#       define FRSR_ROARING_X86_V4_NATIVE   0
#       define FRSR_ROARING_X86_V4_DISPATCH 1
#   endif
#else
#   define FRSR_ROARING_X86_V4_NATIVE   0
#   define FRSR_ROARING_X86_V4_DISPATCH 0
#endif
#define FRSR_ROARING_X86_V4 ( FRSR_ROARING_X86_V4_NATIVE || FRSR_ROARING_X86_V4_DISPATCH )

#if FRSR_ROARING_X86_V4

// The tier is x86-64-v4 (AVX-512 F/BW/DQ/CD/VL) PLUS VPOPCNTDQ and VBMI2: the
// vector popcount and the 16-bit compress/expand are where the wins are, and
// every AVX-512 CPU still in service that matters (Ice Lake+, Zen 4+) has both.
#define FRSR_ROARING_X86_V4_TARGET "avx512f,avx512bw,avx512dq,avx512cd,avx512vl,avx512vpopcntdq,avx512vbmi2"

#if FRSR_ROARING_X86_V4_DISPATCH
// Attribute set of a dispatched kernel: compiled for the tier, kept out of line so
// the tier's code never leaks into the baseline caller. Its helpers take
// FRSR_ROARING_X86_V4_HELPER instead: same target, forced inline — a helper with a
// target attribute is otherwise a real call per use (measured: the bitonic
// compare-exchange helpers ran as separate functions, 60% of the union kernel).
#   define FRSR_ROARING_X86_V4_KERNEL [[ using gnu: target( FRSR_ROARING_X86_V4_TARGET ), noinline, hot ]]
#   define FRSR_ROARING_X86_V4_HELPER [[ using gnu: target( FRSR_ROARING_X86_V4_TARGET ), always_inline ]]

// Detection is hand-rolled cpuid/xgetbv (GNU inline asm — works under clang-cl
// too, where __builtin_cpu_supports lacks its libgcc-style runtime).
[[nodiscard]] inline bool have_x86_v4() noexcept {
    static bool const value{ []() noexcept {
        std::uint32_t eax, ebx, ecx, edx;
        __asm__ volatile ( "cpuid" : "=a"( eax ), "=b"( ebx ), "=c"( ecx ), "=d"( edx ) : "a"( 1U ), "c"( 0U ) );
        if ( !( ecx & ( 1U << 27 ) ) ) { return false; }  // OSXSAVE
        std::uint32_t xlo, xhi;
        __asm__ volatile ( "xgetbv" : "=a"( xlo ), "=d"( xhi ) : "c"( 0U ) );
        if ( ( xlo & 0xE6U ) != 0xE6U ) { return false; }  // XMM+YMM+opmask+ZMM state enabled by the OS
        __asm__ volatile ( "cpuid" : "=a"( eax ), "=b"( ebx ), "=c"( ecx ), "=d"( edx ) : "a"( 7U ), "c"( 0U ) );
        constexpr std::uint32_t ebx_need{ ( 1U << 16 ) | ( 1U << 17 ) | ( 1U << 28 ) | ( 1U << 30 ) | ( 1U << 31 ) };  // F, DQ, CD, BW, VL
        if ( ( ebx & ebx_need ) != ebx_need ) { return false; }
        constexpr std::uint32_t ecx_need{ ( 1U << 6 ) | ( 1U << 14 ) };  // VBMI2, VPOPCNTDQ
        return ( ecx & ecx_need ) == ecx_need;
    }() };
    return value;
}
#else // native
#   define FRSR_ROARING_X86_V4_KERNEL [[ gnu::hot ]]
#   define FRSR_ROARING_X86_V4_HELPER [[ gnu::always_inline ]]
[[nodiscard]] constexpr bool have_x86_v4() noexcept { return true; }
#endif


// ---------------------------------------------------------------------------
// AVX-512 VP2INTERSECT — a tier of its own, deliberately NOT folded into x86-64-v4.
//
// VP2INTERSECT computes BOTH operands' match masks for a 16x16 all-pairs compare in
// a single instruction. That matters because the array intersection's match finding
// is PCMPISTRM, an SSE4.2 *string* instruction with no 256/512-bit form: widening it
// otherwise means all-pairs by rotation, which costs ~2N ops for N^2 pairs and
// measured 4.1-5.6x SLOWER at 32 lanes than the 128-bit kernel it would replace.
// VP2INTERSECT is the only wide instruction that avoids that.
//
// Separate tier because it is NOT part of x86-64-v4 and its availability does not
// track the rest: absent on most Intel server parts, present on Zen 5. Where it is
// missing the caller keeps the SSE4.2 kernel, which is why this is a pure addition.
#if defined( __AVX512VP2INTERSECT__ )
#   define FRSR_ROARING_VP2_NATIVE   1
#   define FRSR_ROARING_VP2_DISPATCH 0
#else
#   define FRSR_ROARING_VP2_NATIVE   0
#   define FRSR_ROARING_VP2_DISPATCH 1
#endif
#define FRSR_ROARING_VP2 ( FRSR_ROARING_VP2_NATIVE || FRSR_ROARING_VP2_DISPATCH )

#define FRSR_ROARING_VP2_TARGET FRSR_ROARING_X86_V4_TARGET ",avx512vp2intersect"

#if FRSR_ROARING_VP2_DISPATCH
#   define FRSR_ROARING_VP2_KERNEL [[ using gnu: target( FRSR_ROARING_VP2_TARGET ), noinline, hot ]]

// CPUID.(EAX=07H,ECX=0):EDX[8]. Layered on have_x86_v4() rather than repeating it:
// the kernel also uses AVX-512 F/BW and the 32->16 bit compress, and that check
// already covers the OS ZMM-state enable, which a bare feature bit does not.
[[nodiscard]] inline bool have_vp2intersect() noexcept {
    static bool const value{ []() noexcept {
        if ( !have_x86_v4() ) { return false; }
        std::uint32_t eax, ebx, ecx, edx;
        __asm__ volatile ( "cpuid" : "=a"( eax ), "=b"( ebx ), "=c"( ecx ), "=d"( edx ) : "a"( 7U ), "c"( 0U ) );
        return ( edx & ( 1U << 8 ) ) != 0;
    }() };
    return value;
}
#else // native
#   define FRSR_ROARING_VP2_KERNEL [[ gnu::hot ]]
[[nodiscard]] constexpr bool have_vp2intersect() noexcept { return true; }
#endif

#endif // FRSR_ROARING_X86_V4

#ifndef FRSR_ROARING_VP2
#   define FRSR_ROARING_VP2 0
#endif

} // namespace frsr::roaring::detail
