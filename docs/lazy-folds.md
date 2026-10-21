# Lazy folds: what may run between a fold and its finisher

`bulk_or_intermediate`, `bulk_xor_intermediate` (and the `or_many*` / `xor_many*` folds built on them) update
an accumulator without counting. A bitset the fold touched keeps a *stale* count, and `size()` is not known.
The bitmap is then **deferred** until a finisher counts it:

| finisher | for |
|---|---|
| `bulk_or_finish` | an OR fold, re-deciding every container by the ambient run-selection policy |
| `bulk_or_finish_keep_bitsets` | the same, keeping bitsets that the optimizer would turn into arrays |
| `bulk_or_repair_after_lazy` | the reference's repair: counts, small bitsets to arrays, existing runs re-decided |
| `bulk_xor_repair_after_lazy` | an XOR fold, or a mix of both folds: it also drops the containers that cancelled out |

## The contract

While a bitmap is deferred:

* further `bulk_or_intermediate` / `bulk_xor_intermediate` folds are allowed;
* every in-place update is allowed and leaves a correct bitmap: `add`, `add_bulk`, `remove`, `add_many*`,
  `remove_many*`, `add_closed_range`, `remove_closed_range`, `flip_closed_range`, `|=`, `&=`, `-=`, `^=` and
  the `*_with` operations (a bitmap or any other operand), `optimize*`, `shrink_to_fit`, `clear`;
* `contains` and `contains_bulk` are allowed and exact;
* a finished bitmap is exact on every operation.

What is **not** allowed until a finisher has run: anything that needs a size or a count, namely `size`,
`equals`, `is_subset_of`, `and_cardinality` and the other cardinalities, iteration, `to_vector`, `front` / `back`,
serialization, `validate`, and using the deferred bitmap as the operand of another operation. A build without
`NDEBUG` asserts on the first such read (`container_handle::cardinality()` asserts that the count is exact,
`bitmap::size()` that the bitmap is not deferred).

The finisher that matches the folds has to run: `bulk_or_*` after OR folds only, `bulk_xor_repair_after_lazy` after
an XOR fold or a mix.

## How the library keeps this true

* A stale bitset count is never above the true one: a kernel that only adds bits keeps the count it had and
  sets the stale flag, a kernel that may clear bits sets it to zero (`mark_cardinality_unknown`). The saturated
  container shortcut compares the stored count with the full size and so can only say "full" when it is.
* The incremental updates (`bitset_ref::add`, the range and array kernels) move the count with
  `adjust_cardinality`, which keeps the stale flag. Storing an adjusted count (`set_cardinality`) declares it
  exact, so it is used for counts that were just computed.
* `bitset_ref::remove` counts a stale bitset first: the caller drops a container that this empties.
* A fold sets `size_` to `size_deferred`, a value no size reaches, so `empty()` stays false and operations that add
  to it or subtract from it keep it recognizable. `repair_cardinality` rebuilds it.
* Every other in-place bitmap operation starts with `settle_counts()`: when the bitmap is deferred, one
  pass counts the stale bitsets. Single-key `add` and `remove` do not, because they never read a count.
* `ensure_endpoints` scans a bitset whose count is stale with a bound, since the fold may have cleared every bit.

## Where a count is read or adjusted

| site | reads / writes | state it can meet | handling |
|---|---|---|---|
| `bitset_ref::add`, `add_sorted` | adjusts | stale | `adjust_cardinality` keeps the flag; `add_sorted` decides "was empty" only from an exact zero |
| `bitset_ref::remove` | adjusts, then the caller drops an emptied container | stale | settles first |
| array-into-bitset union, bitset minus array, array xor into bitset (eager forms) | adjusts by bits changed | stale | `settle_stale_cardinality` first |
| the same kernels, lazy forms; run xor; bitset xor | write | any | OR-only kernels keep the count and set the flag; kernels that may clear bits set it unknown (zero) |
| bitset-with-bitset in-place combines, the bulk OR step | recompute from the words; the saturated shortcut compares `stored_cardinality()` | stale | the stored count is a lower bound, so "full" is never wrong |
| `container_is_known_full` | reads | stale | `stored_cardinality()` |
| run-typed operand combined into a bitset slot during a fold | reads the bitset's count | stale | settles that slot first |
| `ensure_endpoints` | reads | stale and possibly empty | bounded word scan |
| range updates, `add_many*`, `remove_many*`, `|=`, `&=`, `-=`, `^=`, `*_with`, `optimize*` | read container counts | deferred bitmap | `settle_counts()` at entry |
| single-key `add`, `remove`, `add_bulk`, `contains*` | no bitmap-level count read | deferred bitmap | none needed |
| `size`, iteration, cardinalities, `equals`, `is_subset_of`, serialization, `validate`, `front`, `back`, operand use | read counts | deferred bitmap | not permitted; debug builds assert |
| array and run containers | counts are always exact | | |

`tests/lazy_fold_state.cpp` interleaves every permitted operation with the folds against a bit-per-key model, on random
dense, sparse, run and boundary containers, and checks a finished copy after every step. To run it with the
exact-count assertions and sanitizers:

```
cmake -DCMAKE_BUILD_TYPE=None -DCMAKE_CXX_FLAGS="-O1 -g -DBOOST_DISABLE_ASSERTS -fsanitize=address,undefined -fno-sanitize-recover=undefined" ...
ASAN_OPTIONS=alloc_dealloc_mismatch=0:detect_leaks=0 ./frsr_roaring_tests --gtest_filter='lazy_fold_state*'
```
