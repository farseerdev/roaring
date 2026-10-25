# Ranked frozen images: rank, index and select in constant time

`ranked_view<std::uint32_t>` reads a frozen image (see [frozen-format.md](frozen-format.md)) that carries rank data, and
answers, without decoding anything:

| call | meaning |
|---|---|
| `contains( x )` | whether `x` is present |
| `count_below( x )` | the number of ids `< x` |
| `index_of( x )` | the ordinal of `x` among the ids when present, else `npos`; one lookup |
| `select( i )` | the id with ordinal `i` (`i < size()`) |
| `for_each_indexed( f )` | `f( id, ordinal )` in key order |
| `for_each_indexed_from( first, f )` | the same from the first id `>= first`, with ordinals counted from `count_below( first )` |
| `size()`, `byte_size()`, `front()`, `back()` | cardinality, bytes of the image, first and last id |

`count_below` is exclusive. CRoaring's `roaring_bitmap_rank` is inclusive (ids `<= x`); the two differ by
`contains( x )`. `index_of( x )` is CRoaring's `roaring_bitmap_get_index` with `npos` for its `-1`.

The use case is an immutable table from sparse 32-bit ids to dense ordinals, read on a hot path and rebuilt rather than
edited.

## Building and reading

```cpp
using table_type = frsr::roaring::bitmap<std::uint32_t, frsr::roaring::default_container_set<std::uint32_t>,
                                         frsr::roaring::detail::cow_value_semantics, frsr::roaring::detail::run_selection_lazy>;
table_type table;
table.add_sorted_many( ids );                        // table: a bitmap that selects no runs, see "Choosing the containers"

frsr::roaring::frozen_options const options{ .ranked = true };
std::vector<std::uint64_t> storage( ( table.frozen_size_bytes( options ) + 7 ) / 8 );   // 8-byte aligned
std::span const image{ reinterpret_cast<std::byte *>( storage.data() ), table.frozen_size_bytes( options ) };
table.frozen_serialize( image, options );

auto const view{ frsr::roaring::ranked_view<std::uint32_t>::from( image ) };   // null if not a valid ranked image
auto const ordinal{ view.index_of( id ) };
```

The same option goes to `serialize_frozen_to_vm_vector( out, options )` and `persistent_bitmap::store( file, bitmap,
options )`; `persistent_bitmap::open` accepts both versions and `persistent_bitmap::frozen()` is the image to hand to
`ranked_view::from`. `ranked_view::frozen()` returns the plain `frozen_view` of the same image (`materialize`,
`borrow_into`, `contains`).

The view borrows the image: the bytes must outlive it and be 8-byte aligned. All members are `const`, write nothing and
keep no cache, so concurrent readers need no synchronisation.

## How a lookup works

1. **Chunk.** `chunk = x >> 16`. If the image's chunks are consecutive (the common case for a table over a dense id
   range) the chunk is `index[ chunk - first_chunk ]`, after two compares that tell below-the-range (nothing precedes
   `x`) from above-the-range (everything does). Otherwise the chunk index is searched by chunk key (`O(log chunks)`,
   its own search over the frozen index rather than the bitmap's chunk array); a miss between chunks is the next
   chunk's `base`.
2. **Inside the chunk**, with `base` the number of ids in earlier chunks:
   - bitset: `base + word_rank[ low >> 6 ] + popcount( word & mask )`. Two independent loads, no search.
   - array: the position of the insertion point (the existing 16-wide quaternary search) is the rank.
   - run: a binary search of the runs, then `base + run_rank[ j ] + min( low, end_j + 1 ) - begin_j`.

`select( i )` binary-searches the chunk `base` column, then the bitset's `word_rank` (and a select inside the word:
`pdep` when the build targets BMI2, otherwise a halving popcount search), the runs' `run_rank`, or indexes the array.

## Choosing the containers

Rank is `O(1)` only in a bitset chunk; array and run chunks cost a search. The default bitmap picks a run container by
itself whenever it serializes smaller (`run_selection_eager`), and so does `optimize()`. For a nearly contiguous table
(range / entries of about 1.001 to 1.03) that means dozens of short runs per chunk and a binary search where a bitset is
one popcount. Measured with `bench/ranked_view_bench.cpp` (`index_of`, independent probes of present ids, one
x86-64 core, 16.7M ids of range):

| range / entries | default bitmap + `optimize()` | bitmap with `run_selection_lazy` | lazy + `optimize_run_shaped( 1024 )` |
|---|---|---|---|
| 1.001 | 26 ns, 0.007 B/entry | 4.4 ns, 0.157 B/entry | 14 ns, 0.097 B/entry |
| 1.01 | 42 ns | 5.1 ns | 5.1 ns |
| 1.03 | 56 ns | 5.1 ns | 4.4 ns |
| 1.1 | 5.1 ns | 4.9 ns | 4.4 ns |
| 16 / 68 / 590 (arrays) | 14 / 16 / 12 ns | the same | the same |

So build a ranked table with a bitmap that selects no runs (`detail::run_selection_lazy`) and, only when memory matters
more than the last nanoseconds, let `optimize_run_shaped( bar )` turn chunks with a mean run length of at least `bar`
into runs. The image does not depend on the bitmap type that wrote it. A flat bit string with a `u32` prefix count per
64 ids is faster still on dense tables (1.2 to 1.9 ns) but costs 0.19 bytes per id of range at any density, 110 bytes
per entry at 590x, where the ranked image takes about 2.3; CRoaring's `roaring_bitmap_get_index` walks the chunk
cardinalities and costs 0.13 to 13 microseconds on the dense shapes.

## Memory

Per chunk, on top of the v1 payload: 2048 bytes for a bitset (`word_rank`), 2 bytes per run, nothing for an array, and
4 bytes of `base` in the index entry. A bitset chunk is 10272 bytes with its index entry, i.e. about 0.16 bytes per
possible id; arrays and runs keep their small size.

## Limits

- 32-bit keys only; fewer than 2^32 ids (so `npos` is never a valid ordinal). Opening rejects an image that says otherwise.
- The counts in the rank blocks are not re-verified on open (open stays `O(chunks)`, as it is for the declared
  cardinalities): a corrupted count gives wrong answers, never an out-of-bounds read.
- Not on the mutable `bitmap`: its chunk bases would go stale on every write. Rank is a property of the frozen form.
