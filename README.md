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

A finished bitmap is also an operand through a view, and the complement of an operand through `!x` /
`complement_of( x )`, without a copy or a temporary:

```cpp
bitmap.and_with( keys );
bitmap &= !frsr::roaring::view_of( other );            // the keys of bitmap that other lacks
bitmap |= frsr::roaring::view_of( other );             // other is only read
bitmap -= frsr::roaring::complement_of( std::span<std::uint32_t const>{ keys } );
auto common = bitmap.and_cardinality( !frsr::roaring::key_range::closed( 5, 9 ) );
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

When an allocation does throw, a bitmap stays valid. A materializing set operation never touches its operands. An
in-place set operation that fails part way (`|=`, `&=`, `-=`, `or_with`, `and_with`, `andnot_with`) leaves its left
side with the operation applied to part of it: a union still holds every key it held, an intersection or a difference
holds no key it did not hold. An in-place `^=`, `xor_with` or `xor_many_in_place` leaves its left side valid with
unspecified contents.

## Copies

Under a refcounted copy-on-write policy (`cow_atomic_refcount`, `cow_unsynchronized_refcount`) a copy of a bitmap whose
keys are up to 32 bits wide shares the whole chunk table with the bitmap it was copied from: it takes one count on the
table and touches nothing else, and dropping it gives the count back. The first call that changes either bitmap gives
that bitmap a table of its own, which costs what copying the table cost before (one allocation, and one count on each
payload, which stays shared until it is written); a call that would change nothing (`remove` of an absent key, `&=` with
a superset, `-=` of a disjoint operand, an `optimize` that converts nothing, ...) finds that out on the shared table and
leaves it shared. A `|=` with a bitmap is, by default, the exception: it makes the table its own whatever it adds (a
`|=` with a view, keys or a range still asks first). `is_unique()` tells whether another bitmap holds the table and
`shares_storage_with( other )` whether two bitmaps hold the same one. A table with room for only a few chunks is copied
instead (see the knobs below), and so is the chunk table of a bitmap of 64-bit keys; both copies share the payloads.
Under value semantics a copy copies everything.

A call that changes nothing on a shared table leaves its containers as they are, where on a table of its own some such
calls re-encode a container they rewrite (an `&=` with a superset may turn a run into an array, for instance): the keys
are the same either way. `byte_size()` counts a shared table in full for each bitmap that holds it.

Copying a bitmap is a read of it: under the atomic policy any number of threads may copy a bitmap that others read, and
each copy is its holder's own to change.

### Sharing knobs

The parts of the sharing that trade one cost for another are compile-time knobs, all in
`include/frsr/roaring/table_sharing_policy.hpp`, set with macros (and CMake cache variables) of the names below. They
change the compiled bitmaps, so a program must be built with the values of the library it links (a mismatch fails to
link). None of them changes anything under value semantics. The supported configurations are the defaults, sharing
off, and the defaults with one knob set to another tested value (a small-table capacity of 0 or 16, a prefetch
distance of 8, a flag flipped); `tests/table_sharing_matrix.sh` builds and tests each, and any other combination does
not compile.

| knob | default | what it trades |
|---|---|---|
| `FRSR_ROARING_TABLE_SHARING` | 1 | 0: a copy copies the table and counts every payload, as before tables were shared |
| `FRSR_ROARING_TABLE_SHARING_SMALL_CAPACITY` | 4 | a table with room for at most this many chunks carries no count and is copied: for a table of a few chunks a copy costs less than the first change of a shared one, which pays for a table of its own out of line (0: share every table) |
| `FRSR_ROARING_TABLE_SHARING_UNION_CHECKS_FIRST` | 0 | 1: `a \|= b` on a shared table asks whether `b` adds anything and stays shared when it does not; costs a read of both bitmaps when it does add |
| `FRSR_ROARING_TABLE_SHARING_SUBTRACT_CHECKS_FIRST` | 1 | `a -= b` on a shared table asks whether they intersect (stopping at the first common key) and stays shared when they do not; 0 builds the difference aside, copying every container `a` keeps, and keeps it only when it lost keys |
| `FRSR_ROARING_TABLE_SHARING_KEY_LIST_CHECKS_FIRST` | 1 | `and_with` / `or_with` / `andnot_with` of keys, ranges and views on a shared table ask whether they change anything first; 0 makes the table the bitmap's own straight away |
| `FRSR_ROARING_TABLE_SHARING_OPTIMIZE_ASIDE` | 1 | `optimize()` / `optimize_keep_bitsets()` on a shared table run on a private copy that replaces the bitmap only when a container changed its kind; 0 makes the table the bitmap's own and runs in place |
| `FRSR_ROARING_TABLE_SHARING_PREFETCH_DISTANCE` | 0 | how far ahead the walks that write payload counts (the copy that gives a bitmap a table of its own, the release of a table's last holder) fetch those counts; under the atomic policy each count is taken with a locked instruction that waits for the fetches before it, and fetching ahead measured slower |

## Compiled library and header-only mode

`bitmap` is split in two. The headers a program includes (`<frsr/roaring/bitmap.hpp>`) declare the class and
define only what has to inline into a caller: `contains`, `add`, `remove` of one key, the iterators, the size
queries and the thin templates that adapt a caller's range or operand. The set operations, conversions,
merges, range updates, optimization, serialization and the interop engine are defined in
`<frsr/roaring/bitmap_definitions.hpp>`, which a program does not parse or instantiate: the library
(`frsr_roaring`, a static library) holds them compiled for every key width (`uint16_t`, `uint32_t`,
`uint64_t`), every copy-on-write policy (value, atomic refcount, unsynchronized refcount) and both run
selection policies, over the default container set. `bitmap.hpp` declares those instantiations `extern`.

| mode | CMake | what a program compiles | notes |
| --- | --- | --- | --- |
| compiled (default) | `FRSR_ROARING_HEADER_ONLY=OFF` | declarations and the inline fast paths | link `frsr::roaring` |
| header-only | `-DFRSR_ROARING_HEADER_ONLY=ON` (`frsr::roaring` becomes an `INTERFACE` target defining `FRSR_ROARING_HEADER_ONLY`) | everything, in every translation unit that uses it | nothing to link |

Which operands reach the compiled interop code: a `key_range`, a `bit_words` of any word width, a
contiguous range of `key_type` (also as `sorted_keys`), and a `typed_view` of a bitmap of the same instantiation
(`view_of( bitmap )`, the operand to use for a bitmap other threads read: it writes nothing the bitmap owns). Every other operand (a `std::bitset`, a dynamic
bit-set, a range of another integer type or one that is not contiguous, `based_bits`) is applied by the
engine in the calling translation unit.

A bitmap that the library does not hold (another key type, container set or policy), and an operand of the
last kind above, needs the definitions in the program. Include `<frsr/roaring/bitmap_definitions.hpp>`, or define
`FRSR_ROARING_WITH_DEFINITIONS` before `bitmap.hpp`: the program then instantiates what it uses, exactly
as in header-only mode, and still calls the library for the instantiations it holds.

The library honours the toolchain's link-time optimization: set `CMAKE_INTERPROCEDURAL_OPTIMIZATION`
(or `-flto` / `-flto=thin` in the flags, with the matching archiver) for the library and for the programs that
link it, and calls into it inline across the boundary.

The options that change the compiled code (`FRSR_ROARING_NO_SBO`, `FRSR_ROARING_COMBINE_STATS`, the
allocation policy, psi::vm containers and file-mapped serialization) must be the same in the library and in
the program. A program built with other settings fails to link instead of running mismatched code.

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
