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
