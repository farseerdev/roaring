# frsr::roaring frozen format — v1

The zero-copy serialization format produced by
`bitmap::serialize_frozen_to_vm_vector` and consumed by
`bitmap::frozen_view_from_vm_vector` (validating view),
`frozen_view::materialize()` (eager copy-out), `frozen_view::borrow_into()`
(zero-copy borrow), and `persistent_bitmap` (file-mapped master). Once written
by `persistent_bitmap::store`, these bytes are a compatibility surface: any
layout change below requires bumping `frozen_format_version`.

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
| 6      | u16  | version     | `frozen_format_version` == **1** |
| 8      | u32  | chunk_count | number of non-empty chunks (== index entries) |
| 12     | —    | (padding)   | 4 bytes of struct padding, zero |
| 16     | u64  | value_count | total cardinality (Σ index cardinality — validated on load) |

### frozen_chunk_index (32 bytes each), sorted strictly ascending by `chunk`

| offset | type | field          | contents |
|--------|------|----------------|----------|
| 0      | u64  | chunk          | chunk key (high bits of the value) |
| 8      | u32  | payload_offset | buffer-relative byte offset of this chunk's payload |
| 12     | u32  | payload_bytes  | payload size in bytes (must equal the kind-derived size) |
| 16     | u32  | payload_count  | array: #values · run: #runs · bitset: word_count |
| 20     | u32  | cardinality    | values present in the chunk |
| 24     | u8   | container_kind | 1 = array, 2 = run, 3 = bitset |
| 25     | u8×7 | reserved       | zero |

### Payloads

Concatenated after the index, each aligned with `align_up(offset, alignment)`:

| kind   | element              | alignment | payload_bytes            |
|--------|----------------------|-----------|--------------------------|
| array  | `low_type` (u8/u16)  | `alignof(low_type)` | `payload_count * sizeof(low_type)`, sorted ascending |
| run    | `run{begin, end}` (closed intervals, sorted, coalesced, non-adjacent) | `alignof(run)` | `payload_count * sizeof(run)` |
| bitset | u64 words            | 8         | `word_count * 8`         |

## Validation performed by `frozen_view_from_vm_vector`

Rejects (returns a null view) on: short buffer, magic / key_bits / version
mismatch, unsorted or duplicate chunk keys, misaligned or out-of-bounds
payload offsets, payload_bytes not matching the kind-derived size, and
Σ cardinality ≠ header value_count.

## Version history

- **v1** (2026-07-11): initial locked version — the `version` field replaced a
  reserved u16 in the header at the same offset; buffers written before the
  field existed carry 0 there and are rejected.
