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
