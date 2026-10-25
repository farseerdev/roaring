# frsr::roaring frozen format — v1 and v2

The zero-copy serialization format produced by
`bitmap::serialize_frozen_to_vm_vector` (or `bitmap::frozen_serialize` into
caller storage of `frozen_size_bytes()` bytes, every byte of which it writes) and consumed by
`bitmap::frozen_view_from_vm_vector` (validating view),
`frozen_view::materialize()` (eager copy-out), `frozen_view::borrow_into()`
(zero-copy borrow), and `persistent_bitmap` (file-mapped master). Once written
by `persistent_bitmap::store`, these bytes are a compatibility surface: any
layout change below requires a new version. Readers accept versions 1 and 2.

- **v1** is the plain image.
- **v2** is a v1 image plus the rank data of [ranked-view.md](ranked-view.md): a `base` per index entry and a rank
  block after each bitset and run payload. The writer stamps 2 only for `frozen_options{ .ranked = true }`; an
  unranked image is written as v1, byte for byte as before. There is no flags word: the four bytes at offset 12 were
  not reliably zero in images written by early writers, so the version alone says whether rank data is present.

## Global rules

- **Little-endian only.** No byte-swapping variant exists (x86 / ARM LE targets).
- **Position-independent**: all offsets are relative to byte 0 of the buffer.
- **Payload layout == live in-memory payload layout** for every container kind,
  so a borrow is a pure reinterpret (no transform on load).
- All structs are packed as written — no implicit padding (verified by
  `sizeof` at the writer; explicit `reserved` bytes pad where needed).

## Layout

```
[ frozen_header | frozen_chunk_index[chunk_count] | payload … payload ]
```

### frozen_header (24 bytes)

| offset | type | field       | contents |
|--------|------|-------------|----------|
| 0      | u32  | magic       | `0x31464252` ("RBF1") |
| 4      | u16  | key_bits    | `numeric_limits<Key>::digits` of the writing bitmap (16 / 32 / 64) |
| 6      | u16  | version     | `frozen_format_version` == **1**, or `frozen_ranked_format_version` == **2** |
| 8      | u32  | chunk_count | number of non-empty chunks (== index entries) |
| 12     | —    | (padding)   | 4 bytes of struct padding, not interpreted |
| 16     | u64  | value_count | total cardinality (Σ index cardinality — validated on load); below 2^32 in v2 |

### frozen_chunk_index (32 bytes each), sorted strictly ascending by `chunk`

| offset | type | field          | contents |
|--------|------|----------------|----------|
| 0      | u64  | chunk          | chunk key (high bits of the value) |
| 8      | u32  | payload_offset | buffer-relative byte offset of this chunk's payload |
| 12     | u32  | payload_bytes  | payload size in bytes (must equal the kind-derived size) |
| 16     | u32  | payload_count  | array: #values · run: #runs · bitset: word_count |
| 20     | u32  | cardinality    | values present in the chunk |
| 24     | u8   | container_kind | 1 = array, 2 = run, 3 = bitset |
| 25     | u8×3 | reserved       | zero |
| 28     | u32  | base           | v2: the number of values in all earlier chunks (the running Σ cardinality); zero in v1 |

### Payloads

Concatenated after the index, each aligned with `align_up(offset, alignment)`:

| kind   | element              | alignment | payload_bytes            |
|--------|----------------------|-----------|--------------------------|
| array  | `low_type` (u8/u16)  | `alignof(low_type)` | `payload_count * sizeof(low_type)`, sorted ascending |
| run    | `run{begin, end}` (closed intervals, sorted, coalesced, non-adjacent) | `alignof(run)` | `payload_count * sizeof(run)` |
| bitset | u64 words            | 8         | `word_count * 8`         |

### Rank blocks (v2 only)

Right after a bitset or run payload, before the next payload's alignment padding (`payload_bytes` is unchanged, so
borrowing and materializing never see the rank block). Each count is a little-endian u16:

| kind   | rank block                                                                    | size                  |
|--------|-------------------------------------------------------------------------------|-----------------------|
| array  | none: the position of a value is its rank                                     | 0                     |
| run    | `run_rank[i]` = values in runs `0..i-1` (at most 65535)                       | `payload_count * 2`   |
| bitset | `word_rank[w]` = values in words `0..w-1` of the chunk (at most 65472)        | `word_count * 2`      |

Payloads and their rank blocks follow the index in index order, so `payload_offset` is non-decreasing and a payload
starts at or after the end of the previous chunk's rank block.

## Validation performed by `frozen_view_from` / `frozen_view_from_vm_vector`

Rejects (returns a null view) on: short buffer, magic / key_bits / version
mismatch, unsorted or duplicate chunk keys, chunk keys wider than the reading
bitmap's chunk key type, misaligned or out-of-bounds
payload offsets, payload_bytes not matching the kind-derived size, and
Σ cardinality ≠ header value_count. A v2 image is also rejected for a key type wider than 32 bits, for `value_count`
of 2^32 or more (so that "absent" is never a valid rank), for a `base` that is not the running Σ cardinality, for a
payload that starts before the previous chunk's rank block ends, and for a rank block that runs past the image.
The counts themselves are trusted like the declared cardinalities: a wrong count gives wrong ranks, never a read
outside the image, because every index into a rank block is bounded by the kind.

`frozen_view_from( std::span<std::byte const> )` is the primitive (a flat buffer or a mapping; the data must be
8-byte aligned); the vm_vector overload forwards to it.

## Version history

- **v2**: the rank data above (`frozen_options{ .ranked = true }`); `reserved[7]` became `reserved[3]` and `base`.

- **v1** (2026-07-11): initial locked version — the `version` field replaced a
  reserved u16 in the header at the same offset; buffers written before the
  field existed carry 0 there and are rejected.
