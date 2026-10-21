# roaring

Modern C++26 roaring bitmaps for Farseer.

## Status

This repository is the bootstrap of a from-scratch C++ roaring-bitmap implementation in `frsr::roaring`. The first slice intentionally focuses on:

- a clean standalone + embedded CMake shape,
- a generic `bitmap<Key>` API,
- public array + bitset containers,
- iterator/value-semantic operators,
- CRoaring-backed cross-check tests.

It is **not** yet a full CRoaring replacement. Run containers, deeper policy configurability, CoW, richer range adaptors, and the remaining consumer-specific low-level optimizations are planned follow-up work.

## Data-model reference

The current design follows the container model described in the Roaring bitmap literature, especially:

- **Daniel Lemire, Leonid Boytsov, Nathan Kurz — _Roaring Bitmaps: Efficient Compressed Bitmap Indexes_**
- **Daniel Lemire et al. — _CRoaring: A Fast Integer Set Library with Roaring Bitmaps_**

The library intentionally keeps those container types visible in the public API.

## Public container types

The current public container surface is:

- `frsr::roaring::array_container<Key>`
- `frsr::roaring::run_container<Key>`
- `frsr::roaring::bitset_container<Key>`
- `frsr::roaring::default_container_set<Key>`

Users can construct and manipulate these types directly today, and `bitmap<Key, ContainerSet>` accepts any `std::variant` composed from those public containers for the same `Key`.

For example, to instantiate a bitmap that only uses array + bitset containers:

```cpp
using array_bitset_set = std::variant<
    frsr::roaring::array_container<std::uint32_t>,
    frsr::roaring::bitset_container<std::uint32_t>
>;

frsr::roaring::bitmap<std::uint32_t, array_bitset_set> bitmap;
```

## Combining with plain data

A bitmap can be and-ed, or-ed, and-not-ed, xor-ed, tested for intersection, subset, superset and equality with key
spans and ranges, `std::bitset`, dynamic bit-sets, raw bit words and key intervals directly, without building
a temporary bitmap:

```cpp
frsr::roaring::bitmap<std::uint32_t> bitmap;
std::vector<std::uint32_t> keys{ 3, 70000, 70001 };

bitmap.and_with( keys );                                  // keep these keys
bitmap.or_with( frsr::roaring::bits_at( std::bitset<4096>{ 0xF0F0 }, 65536 ) );
bool any = bitmap.intersects( frsr::roaring::key_range::half_open( 100, 200 ) );
auto n = bitmap.cardinality_in_range( 0, 1000 );
```

See [docs/interop.md](docs/interop.md).

## Exceptions

An operation that can allocate (`add`, `add_many`, the set operations and their `_with` forms, `optimize`, copying)
is declared `noexcept( detail::allocation_failure_terminates )`: in a build where a failed allocation is an exception
(the default) it is not `noexcept` and throws `std::bad_alloc`; in a build where an allocation cannot fail, or must end
the program, it is `noexcept`, which lets the caller's own operations be `noexcept` at no cost and takes the
unwinding paths out of the generated code. The policy is the allocator's, spelled `PSI_NOEXCEPT_EXCEPT_BADALLOC` when
psi::vm is in use (`noexcept` or empty); a build without it can define `FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC` to
`noexcept` or to nothing. The operations against a range of keys, which walk the caller's range, stay non-`noexcept`
unless the range is contiguous storage of keys.

What never allocates is always `noexcept`: `clear`, `swap`, moves, the destructor, the queries (`contains`,
`size`, `intersects`, `validate`, ...) and the interop queries against a `key_range`, bit words or a bit-set of up to
131072 bits.

When an allocation does throw, a bitmap stays valid. The chunk table is left as it was; a materializing set
operation never touches its operands; an in-place `^=` or `xor_many_in_place` leaves its left side valid with
unspecified contents.

## Design constraints

- Namespace: `frsr::roaring`
- Naming: `snake_case`
- CMake target: `frsr_roaring` with alias `frsr::roaring`
- License: MIT
- Primary development environment: embedded as `deps/roaring` inside `farseerdev/rama`

## Source reference markers

When an implementation detail is informed by CRoaring, code uses a breadcrumb comment like:

```cpp
// [croaring-ref] deps/croaring/src/...:<function-or-region>
```

These markers preserve review and porting context without claiming direct file mirroring.

## Build

```bash
cmake -S . -B build -G Ninja -DFRSR_ROARING_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
