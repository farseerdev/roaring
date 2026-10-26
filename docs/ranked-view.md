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
| `size()`, `byte_size()`, `front()`, `back()` | cardinality, bytes of the image (up to the end of its last payload and rank block), first and last id |

`count_below` is exclusive. CRoaring's `roaring_bitmap_rank` is inclusive (ids `<= x`); the two differ by
`contains( x )`. `index_of( x )` is CRoaring's `roaring_bitmap_get_index` with `npos` for its `-1`.

The use case is an immutable table from sparse 32-bit ids to dense ordinals, read on a hot path and rebuilt rather than
edited.

## Building and reading

```cpp
frsr::roaring::rank_table_bitmap<std::uint32_t> table;
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
(range / entries of about 1.001 to 1.1) that means a binary search of dozens of short runs where a bitset is one popcount.
Build a ranked table with a bitmap that selects no runs (`rank_table_bitmap`), and, only when memory matters more
than the last nanoseconds, let `optimize_run_shaped( 4096 )` turn chunks with a mean run length of at least 4096 (at most
16 runs per chunk) into runs. Lower bars (64, 1024) still pick runs for tables with a range / entries of 1.001 to 1.01
and are 5 to 10 times slower there.

Measured with `bench/rank_bench.cpp` on one pinned core of a Zen 4 machine, clang 23 and g++ 16 (glibc), `index_of` of
present ids, independent probes, a table of 10^6 ids, nanoseconds per lookup, the fastest of 9 rounds (clang; g++ is
within 15% of it except for the sorted array, which is up to 2x slower, and CRoaring on the dense tables):

| range / entries | bit string + u32 per 64 ids | ranked, default bitmap | ranked, lazy bitmap | sorted array | CRoaring `get_index` |
|---|---|---|---|---|---|
| 1.001 | 0.8 | 21 | 2.2 | 44 | 79 |
| 1.01 | 0.8 | 33 | 2.2 | 44 | 329 |
| 1.03 | 0.8 | 40 | 2.2 | 45 | 833 |
| 1.1 | 0.8 | 2.2 | 2.2 | 43 | 228 |
| 2 | 0.8 | 2.3 | 2.3 | 44 | 234 |
| 16 | 1.4 | 13 | 13 | 44 | 254 |
| 68 | 1.3 | 22 | 22 | 44 | 473 |
| 590 | 1.4 | 17 | 16 | 45 | 3815 |

Bytes per id of the same tables: the bit string takes 0.19 (1.01), 3.0 (16), 12.7 (68) and 111 (590); the ranked image
0.16 to 0.18 for every range / entries up to 1.1 with the lazy bitmap (0.006 to 0.17 with the default bitmap, whose
smaller size is paid for in the lookup time above), 0.32 at 2, 2.3 at 16 and 2.0 to 2.3 at 68 and 590. A sorted array takes 4.

Where it wins and loses:

- A dense table (up to about 2x): a flat bit string with a count per word is 2.5 to 3 times faster than the ranked image
  for independent lookups and 1.5 to 1.7 times faster for dependent ones, at the same size. A ranked image there is for
  a consumer that wants the bytes in a file or a mapping, with `select` and ordered iteration, not for speed.
- A sparse table (16x and beyond): the bit string takes 3 to 111 bytes per id and the ranked image about 2; lookups cost
  13 to 22 ns instead of 1.3 to 1.4 when the table is larger than the caches, comparable to a sorted array's binary
  search at smaller sizes (8 to 12 ns against 17 for 10^4 ids) and 2 to 4 times faster than it at 10^6 ids.
  Dependent chains are the worst case for it (21 to 34 ns at 10^6 ids).
- Against CRoaring's `roaring_bitmap_get_index`, which walks the chunk cardinalities, the ranked image (lazy build) is 8 to
  400 times faster for independent lookups in tables of 10^4 or more ids, and for dependent chains from 1.7 times
  (10^4 ids, 68x sparse: a tie) to several hundred times. Below about a hundred ids CRoaring is slightly faster
  (4.5 against 5.7 ns independent, 4.8 against 16 ns dependent).
- Fewer than about a hundred ids (one chunk): 5.7 to 6.1 ns independent and 14 to 16 ns dependent, against 0.6 and 3.5 for
  the bit string; a sorted array is as fast independently (4.7 to 5.1 ns) and faster dependently (10 ns).

## Memory

Per chunk, on top of the v1 payload: 2048 bytes for a bitset (`word_rank`), 2 bytes per run, nothing for an array, and
4 bytes of `base` in the index entry. A bitset chunk is 10272 bytes with its index entry, i.e. about 0.16 bytes per
possible id; arrays and runs keep their small size.

## Limits

- 32-bit keys only; fewer than 2^32 ids (so `npos` is never a valid ordinal). Opening rejects an image that says otherwise.
- The counts in the rank blocks are not re-verified on open (open stays `O(chunks)`, as it is for the declared
  cardinalities): a corrupted count gives wrong answers, never an out-of-bounds read.
- Not on the mutable `bitmap`: its chunk bases would go stale on every write. Rank is a property of the frozen form.
