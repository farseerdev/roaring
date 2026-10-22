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
* **`view_of( bitmap )`**: a read-only, non-owning view of a finished bitmap of the bitmap's own instantiation (see
  [Views and complements](#views-and-complements)).
* **`!x`, `complement_of( x )`**: the complement of any of these (and of a bitmap), as a lazy operand.
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
* A **view** is merged chunk against chunk with the other bitmap's table, each of its containers meeting ours as a
  piece does. The intersection and the difference build a result container with the kernel of the pair of kinds
  (array, bitset, run), in a payload the table retired where there is one, as the operations between two bitmaps do;
  where those share an operand's container as the result (against a saturated bitset or a full-domain run) a view
  computes the result instead, which is the price of writing nothing to the viewed bitmap. A difference that leaves a
  bitset below the array limit makes it an array; an intersection leaves it a bitset, as the operation between two
  bitmaps does (`a &= view` is `a &= bitmap`). Against every other operand an intersection makes it an array too, as the
  reference does.

All const operations are pure reads and are safe to run concurrently on one bitmap. The queries are `noexcept`
against a `key_range`, a `bit_words` and a `std::bitset` of up to 131072 bits, which they read without
allocating; against a range of keys or a larger bit-set they may allocate (to sort a copy, or to copy the bits
out) and are not.

The operations that change the bitmap (`and_with`, `or_with`, `andnot_with`, `xor_with` and the compound operators)
allocate, so they are `noexcept` exactly when a failed allocation cannot be observed as an exception (see
[README](../README.md), Exceptions) and the operand is a `key_range`, a `bit_words` or a bit-set of up to 131072 bits.

## Views and complements

### A view as an operand

`view_of( b )` is a `typed_view`, one pointer wide, that stands for the finished bitmap `b` as an operand of the
operations above: `a &= view_of( b )`, `a |= view_of( b )`, `a -= view_of( b )`, `a ^= view_of( b )`, and the queries
(`intersects`, `and_cardinality`, `xor_cardinality`, `is_subset_of`, `is_superset_of`, `equals`). It is the operand
for a reader of a bitmap that other threads read too: an operation reading through a view **writes nothing the viewed
bitmap owns**, not a reference count, not a container header, not the chunk table, and **keeps nothing of it**: what a
result takes from the view is a fresh copy, so a result is valid after the viewed bitmap is gone.

Passing the bitmap itself (`a &= b`) is the operation between two bitmaps, which is the cheapest for one thread because
it shares the containers of `b` by reference count where it can; under a counting copy-on-write policy that sharing is
a write to the counts of `b`, to cache lines the other readers use. A view is the operand to use there. The two give
the same answers.

A view belongs to the bitmap type it was made from: `bitmap<K, S, P, R>` takes `typed_view<bitmap<K, S, P, R>>`
only. It is not a range (it has no `begin` or `end`, so it is never taken for a sequence of keys) and it is not the
type-erased `bitmap_view` of `bitmap_view.hpp`, which has no set operations; it converts to that.

The contract, which the caller keeps (as with `std::string_view`; a release build does not check it):

1. **Finished.** The viewed bitmap is not in the middle of a lazy fold (a debug build asserts; see
   [lazy-folds.md](lazy-folds.md)).
2. **Unchanged while viewed.** It is not modified, moved, swapped or destroyed while a view of it, or an operation
   reading through one, is alive. Any number of threads may read through views of it, and read it directly, at the
   same time.
3. **Nothing reachable from a view is written.** The bitmap may carry bitset containers whose endpoints are not
   settled (a lazy fold leaves them so); operations through a view do not settle them.
4. **Results do not refer to the view.** The one exception is `borrowed_copy` below, which says so by its name.
5. **A bitmap updated through a view of itself is defined.** `a &= view_of( a )` and `a |= view_of( a )` leave `a`
   as it is, `a -= view_of( a )` and `a ^= view_of( a )` empty it.
6. **A copy-on-write copy of the viewed bitmap may be updated through its view.** Releasing a payload the copy shares
   with the viewed bitmap drops the copy's own reference to it; that is the copy's, not a write through the view.

The binary operators take views on either side and build a bitmap, as `bitmap & bitmap` does: `view & view`,
`bitmap & view`, `view | bitmap`, `view - view`, `view ^ bitmap`, and so on. `a.intersect_into( view, scratch )` is
`a & view` into a scratch bitmap the caller reuses.

A view travels through the same compiled code as the other erased operands: a program that uses one instantiates
nothing of the engine.

### Many bitmaps: the n-way AND

`bitmap::and_many`, `and_many_cardinality`, `and_many_intersects`, `and_many_for_each` and
`and_many_for_each_block` take a `std::span<typed_view<bitmap> const>` as well as a span of pointers. The walk is the
n-way AND that starts from the operand with the fewest chunks and leapfrogs from chunk to chunk, so a chunk any
operand lacks is a gallop; where every operand has the chunk, the containers meet in ascending order of their
cardinality and the walk ends the chunk at the first empty accumulator. It writes nothing the operands own and
the result keeps none of their containers (the single operand is copied too). `bitmap::and_many_into( views, out )`
writes the result into a bitmap of the caller's, whose chunk table keeps its capacity and whose payloads the next call
builds in.

The walk of `and_many` over views is the walk over pointers (which already copies nothing and writes no count), so views do
not make it faster; they make it applicable to bitmaps that have to be named by view. What the n-way AND with
views and a reused result saves over the pointer form is the allocation of the result, and what `a.intersect_into( view,
scratch )` followed by `scratch &= view` of the rest saves over a counted copy is the reference count it does not write:
see `bench/view_contention_bench.cpp`, arms `and-many-ptr`, `views`, `many-into` and `views-into`.

### Complements

`!x` is the complement of `x` within the key space, for a `typed_view`, a `key_range`, a `bit_words`, a `sorted_keys`,
a `based_bits` and a bitmap; `complement_of( x )` is the same for any operand, which is how to write it for a
`std::span`, a `std::vector` or a `std::bitset`, whose namespace is not ours to put an operator in:
`a &= complement_of( std::span<std::uint32_t const>{ keys } )` keeps the keys of `a` that are not in `keys`. `!!x` is `x`. Nothing is built: the operation is rewritten to the one it stands for.

| expression | is |
| --- | --- |
| `a &= !x`, `a & !x` | `a -= x`, `a - x` |
| `a -= !x` | `a &= x` |
| `a.intersects( !x )` | `!a.is_subset_of( x )` |
| `a.is_subset_of( !x )` | `!a.intersects( x )` |
| `a.and_cardinality( !x )`, `a.andnot_cardinality( x )` | `a.size() - a.and_cardinality( x )` |
| `a.andnot_cardinality( !x )` | `a.and_cardinality( x )` |
| `a \|= !r`, `a ^= !r` for a `key_range` `r` | the two intervals outside `r`, one after the other |

A complement is the rest of the key space, so the other combinations are not defined, and do not compile: `a |= !x`
and `a ^= !x` for any `x` but a `key_range`, and `xor_cardinality( !x )`, `is_superset_of( !x )` and `equals( !x )` for
any `x`. The union with the complement of an interval is as large as the intervals it adds.

A complement holds a small descriptor (an interval, a view, a span, a bit string) by value and any
other operand by reference, so it has to be created in the expression that uses it, as `sorted_keys` and `based_bits`
have to be.

### Sizes, and the strict subset

`andnot_cardinality( x )`, `or_cardinality( x )`, `jaccard_index( x )` and `is_strict_subset_of( x )` answer without
building the combination, as the reference's `roaring_bitmap_andnot_cardinality`, `_or_cardinality`, `_jaccard_index`
and `_is_strict_subset` do. The operand of the second and third is a bitmap, a view or a `key_range` (an operand
whose size is known); `jaccard_index` of two empty sets is NaN, as in the reference.

### A cheap copy of a published bitmap

`bitmap::borrowed_copy( view_of( b ) )` is a copy of the finished bitmap `b` whose spilled containers are borrowed
aliases of those of `b` (the way a bitmap served from a mapped file borrows its payloads): making it writes nothing to
`b`, where copying a bitmap under a counting policy writes one count per container, and it copies no payload. The first
update of a container of the copy clones that container into private storage, so the copy is as mutable as any bitmap
and `b` is never changed through it.

Because it does refer to `b`, so does everything that takes a container of the copy by copying its handle, which a
borrowed container keeps borrowed under every policy: `bitmap{ copy }`, an assignment from the copy, the receiver of
`x |= copy` or `x ^= copy` (the containers `x` lacks) and of `x &= copy` (a full chunk). All of them must be dropped, or have
each of their containers written once, before `b` is changed or destroyed; this is the aliasing of a bitmap served from a
mapped file. To get a bitmap that outlives `b`, take the copy through a view, which adopts nothing:

```cpp
bitmap detached;
detached |= view_of( copy );   // or view_of( b ): a copy of the contents, shared with nothing
```

Taking the copy by `bitmap{ copy }` does not do that, and a copy of a `bitmap` under a counting policy
(`bitmap{ b }`) is the one to use when the reference counts of `b` may be written.

A bitmap served from a mapped file has `persistent_bitmap::view()`: an operation through it never aliases the
mapping, where `x |= *persistent` adopts the borrowed payloads and `x` is then valid only as long as the mapping is.

