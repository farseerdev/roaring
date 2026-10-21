# Combining a bitmap with plain arrays, bit-sets and ranges

A `frsr::roaring::bitmap` can be combined with data that is not a bitmap without first turning that data
into one. The operations, all members of `bitmap`:

| operation | meaning |
| --- | --- |
| `and_with( x )`, `&=` | keep the keys that are also in `x` |
| `or_with( x )`, `\|=` | add the keys of `x` |
| `andnot_with( x )`, `-=` | remove the keys of `x` |
| `xor_with( x )`, `^=` | keep the keys that are in exactly one of the bitmap and `x`; with a `key_range`, flip the interval |
| `intersects( x )` | whether a key is in both |
| `and_cardinality( x )` | how many keys are in both (each key once) |
| `xor_cardinality( x )` | how many keys are in exactly one |
| `is_subset_of( x )` | every key of the bitmap is in `x` |
| `is_superset_of( x )` | every key of `x` is in the bitmap |
| `equals( x )` | the same keys |
| `intersects_range( lo, hi )`, `contains_range( lo, hi )`, `cardinality_in_range( lo, hi )`, `and_range( lo, hi )` | the half-open interval `[lo, hi)` |

`x` is a bitmap (the operations then are the ones between two bitmaps) or any of the following.

## Operands

* **`key_range`**: an interval of keys. `key_range::closed( first, last )`, `half_open( lo, hi )`,
  `from( first )` (to the largest key) and `all()`. The bounds are 64-bit and both ends are inclusive inside,
  so the whole key space of any key width can be named. An interval is clipped to the key type of the bitmap.
* **A range of keys**: a `std::span`, a `std::vector`, a `std::set`, any range of unsigned integers whose
  elements may be narrower or wider than the bitmap's key type. A key the key type cannot hold is not a member
  of any bitmap of that type: it matches nothing, is not added by `or_with`, and makes `is_superset_of` and
  `equals` false. A range in ascending order (equal neighbours allowed) is detected; `sorted_keys( range )`
  declares it and skips the check.
* **`bit_words<Word>( words, base, bit_count )`**: a string of bits in memory, bit `i` standing for the key
  `base + i`. `Word` is any unsigned type; the bits are read in memory order as little-endian words.
* **`std::bitset<N>`**: bit `i` is the key `i`; `bits_at( bitset, base )` makes it `base + i`.
* **A dynamic bit-set** such as `boost::dynamic_bitset`, recognized by shape, with no dependency on Boost: a
  `block_type`, `size()` and an ADL `to_block_range( bits, out )` (the blocks are copied out), or else
  `size()`, `find_first()`, `find_next( pos )` and `npos` (walked by its set bits). `bits_at` adds a base.

Bits at or past the end of the bit string never count, whatever the storage holds there.

## How it works

The operand is presented to the bitmap as a stream of pieces, one per 65536-key chunk, in ascending chunk
order. A piece is a container handle that views memory: the caller's own words when whole aligned words cover
the chunk, a small scratch buffer otherwise (a chunk's keys gathered into an array, shifted bits assembled into
a word block, or a run for an interval). The bitmap's chunk table and the stream are merged by jumping each
over the other's absent keys, and a piece meets only the container of its own chunk, through the container
kernels that serve bitmap against bitmap. Nothing is built for the chunks the operand lacks, and no piece is
ever stored in a bitmap: what an operation keeps is built in fresh storage.

* An **interval** is a run piece per chunk made on demand. A chunk it covers entirely is kept (`and_with`),
  dropped (`andnot_with`) or counted from its header (`and_cardinality`) without being opened; a long interval
  over a few chunks costs the few chunks. `xor_with( key_range )` flips the interval in one pass over the chunks
  it covers (a chunk it covers entirely is complemented without being read twice), as `flip_closed_range` does.
* A **sorted range of keys** is grouped by chunk in a scratch array (or a bitset past the array limit);
  random-access ranges are searched by galloping, so a few keys against a large bitmap cost the keys.
* An **unsorted range of keys** has three treatments. `intersects` and `is_superset_of` probe key by key (a
  binary search of the chunk table and one container lookup) at any size, since they stop at the first answer.
  `and_with` and `and_cardinality` probe below 2048 keys, where the cost follows the keys and not the bitmap;
  from there on they sort a copy of the keys (a radix sort) and merge it like a sorted range, which costs less
  than a probe per key. `or_with`, `andnot_with`, `xor_with`, `is_subset_of` and `equals` need the operand as a set: they
  sort a copy and merge it like a sorted range.
* A **bit string** is merged a chunk at a time; a chunk-aligned string of aligned words is used where it is.

All const operations are pure reads and are safe to run concurrently on one bitmap. The queries are `noexcept`
against a `key_range`, a `bit_words` and a `std::bitset` of up to 131072 bits, which they read without
allocating; against a range of keys or a larger bit-set they may allocate (to sort a copy, or to copy the bits
out) and are not.
