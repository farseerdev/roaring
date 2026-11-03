# Keeping runs under subtraction

A run chunk that loses values stays a run chunk whenever that is its smallest form. `bitmap -= other`,
`remove_many*`, `remove_closed_range` and the materialized difference `a - b` all settle through one rule
(`choose_run_container_kind`):

| form | serialized size |
|---|---|
| runs | `4 * runs + 2` bytes (four bytes per run, plus the two-byte run count) |
| array | `2 * cardinality` bytes |
| bitset | 8192 bytes, a candidate only from the array/bitset threshold on |

The smallest wins and runs keep a tie with the array, as in CRoaring's `convert_run_to_efficient_container`. A
result that is one run after a cut stays one run; a run chunk holed by a few thousand values still has fewer runs than
values and stays runs; only when the holes outnumber what a run saves does it become an array or a bitset.

The point is what happens next. A bitmap that is mostly long runs (ids allocated in blocks, ranges of dates, whole
sub-trees of a hierarchy) and loses values in batches stays small, and the intersections that follow still walk runs,
not values. CRoaring does not decide again after `roaring_bitmap_andnot_inplace`; its users call
`roaring_bitmap_run_optimize` to get the smallest form back, which re-examines every container of the bitmap.

## What is measured

`bench/bitmap_bench.cpp` has two families.

* **Single chunk** (`RunAndNot*`, `AndAfterRunAndNot`, `AndInplaceAfterRunAndNot`): the one-run chunk `[0, 59999]`
  minus an array, a run and a bitset (residues of one stretch, three stretches, nine values, a dense one, and a
  bitset that does not overlap), as `-=` on a copy that shares the runs, as the materialized difference, and followed by
  `&` and `&=` with a run covering half the receiver.
* **End to end** (`KeepRuns*`): a bitmap of about a million ids, mostly long runs over 64 chunks, plus eight
  bitset chunks and sixteen array chunks, goes through four rounds of batch deletion. A round's sorted doomed set is
  scattered singles (12 000 ids), whole sub-ranges (100 ranges of 200 to 1500 ids), bitset-dense chunks (six chunks at
  55%), or those in turn (`mixed`). The rounds are `-=` only (`KeepRunsRemove`), `-=` then `&=` a run-heavy filter
  (`KeepRunsRemoveAnd`), `-=` then the intersection of the bitmap with four run-heavy filters
  (`KeepRunsRemoveAndMany`; CRoaring has no n-ary intersection, so its arm is `and` then three `and_inplace`), and
  `remove_many` of the sorted doomed ids (`KeepRunsRemoveMany`). Every timed run starts from a copy of the bitmap in
  every arm. The portable serialized size of the final bitmap is printed once per entry as a `#bytes` line.

Arms of the end-to-end bands: `frsr-rc-lazy` (the library as shipped), `cpp-cow` (CRoaring, nothing between the
deletion and the intersection) and `cpp-cow-opt` (CRoaring with `roaring_bitmap_run_optimize` after each deletion,
the best a CRoaring user can do to keep the result small).

`bench/tools/keep_runs_report.py <outputs>` prints the table below from the bench output (`--markdown`), and with
`--check` is the regression gate: every band is expected at or below 1.05x each CRoaring arm, unless the tool lists a
ceiling for it. `bench/tools/band_ratios.py` pairs these bands with the plain CRoaring arm like every other band
(`band_ratios.py <output> frsr-rc-lazy cpp-cow-opt` pairs the end-to-end ones with the `run_optimize` arm).

## Numbers

Linux, AMD Zen 4, clang 23 and g++ 16, Release, `FRSR_ROARING_NO_SBO=1`, CRoaring 5.3.0. One process per arm of a
band, pinned to a core under SCHED_FIFO, minimum of five runs; four builds (clang and g++, glibc and mimalloc). Times
are the clang/glibc build in microseconds for the whole four rounds; the ratio columns give the range over the four
builds; below 1.0 is faster than CRoaring.

| rounds of | doomed ids | frsr us | CRoaring us | CRoaring + run_optimize us | frsr / CRoaring | frsr / CRoaring + run_optimize | frsr bytes | CRoaring bytes | CRoaring + run_optimize bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `-=` | singles | 113 | 86 | 940 | 1.21-1.40 | 0.08-0.12 | 245,989 | 635,934 | 245,989 |
| `-=` | ranges | 15 | 21 | 53 | 0.59-0.70 | 0.10-0.28 | 115,517 | 115,517 | 115,517 |
| `-=` | dense chunks | 8 | 9 | 71 | 0.95-1.08 | 0.05-0.12 | 244,825 | 244,825 | 244,825 |
| `-=` | mixed | 29 | 38 | 165 | 0.64-0.92 | 0.08-0.17 | 184,223 | 636,878 | 184,223 |
| `remove_many` of the sorted ids | singles | 189 | 898 | 927 | 0.19-0.21 | 0.17-0.20 | 245,989 | 245,989 | 245,989 |
| `remove_many` of the sorted ids | ranges | 469 | 1634 | 1664 | 0.29-0.34 | 0.28-0.32 | 115,517 | 115,517 | 115,517 |
| `remove_many` of the sorted ids | dense chunks | 1251 | 6899 | 7084 | 0.17-0.18 | 0.16-0.18 | 244,825 | 306,695 | 244,825 |
| `remove_many` of the sorted ids | mixed | 642 | 3025 | 2992 | 0.19-0.21 | 0.19-0.21 | 184,223 | 193,557 | 184,223 |
| `-=`, `&=` a filter | singles | 192 | 161 | 1051 | 1.07-1.20 | 0.11-0.18 | 171,843 | 617,234 | 171,843 |
| `-=`, `&=` a filter | ranges | 62 | 99 | 132 | 0.55-0.65 | 0.29-0.47 | 95,915 | 95,915 | 95,915 |
| `-=`, `&=` a filter | dense chunks | 72 | 133 | 189 | 0.46-0.56 | 0.23-0.38 | 187,225 | 187,225 | 187,225 |
| `-=`, `&=` a filter | mixed | 116 | 145 | 323 | 0.63-0.80 | 0.20-0.36 | 142,301 | 602,872 | 142,301 |
| `-=`, and-many of four filters | singles | 546 | 479 | 1803 | 1.06-1.14 | 0.24-0.30 | 245,989 | 635,934 | 245,989 |
| `-=`, and-many of four filters | ranges | 203 | 358 | 394 | 0.54-0.60 | 0.43-0.54 | 115,517 | 115,517 | 115,517 |
| `-=`, and-many of four filters | dense chunks | 288 | 472 | 537 | 0.56-0.67 | 0.42-0.58 | 244,825 | 244,825 | 244,825 |
| `-=`, and-many of four filters | mixed | 344 | 474 | 791 | 0.61-0.72 | 0.33-0.46 | 184,223 | 636,878 | 184,223 |

What it shows:

* The bytes are CRoaring's best, without asking. Where the deletion leaves holes in run chunks (`singles`, `mixed`)
  plain CRoaring ends at 2.6x to 4.2x the size (636 KB against 184 KB to 246 KB for `-=`; 603 KB against 142 KB with
  the `&=` rounds); frsr ends at exactly the size CRoaring reaches after `run_optimize`, with no extra pass.
* Against the fair baseline (CRoaring that re-optimizes after each deletion to stay that small) frsr is faster in
  every row, by 2x to 20x on the `-=` rows, since `run_optimize` costs more than the deletion it follows.
* `remove_many` of the sorted doomed ids is 3x to 6x faster than CRoaring's `roaring_bitmap_remove_many` (0.17x to
  0.34x its time) on every doomed set, and ends at the same bytes or fewer (a dense chunk: 245 KB against 307 KB).
* Against plain CRoaring the deletion by `-=` alone is faster with sub-ranges and mixed doomed sets, level on
  bitset-dense chunks, and slower with scattered singles (see below).

## How the deletion kernels are shaped

* **Runs minus values** (`difference_run_values`: `-=` an array chunk, the materialized difference and
  `remove_many_sorted`) walks the values one by one. A value past the run the one before it cut writes what is left
  of that run and then the runs wholly below the value, counted by their ends eight at a time (one AVX2 compare where
  the build has it) and copied in 32-byte blocks; a value inside the current run cuts it without a branch on where it
  lands. The one branch per value, whether the value has left the current run, goes the same way for most values both
  when the values are scattered over many runs (each lands in a later run) and when many cut one run. A merge that
  chooses between the next run and the next value at every step mispredicts about once per value on scattered values,
  which is what the singles rounds used to pay (about 27 thousand mispredicted branches per four rounds, now about a
  hundred). The cardinality is the runs' one less the values they held, not a recount.
* **Run minus runs** copies the left runs below the first right run and past the last one in one piece each.
* **Bitset minus runs** clears the runs in the bitmap's own block when it holds the chunk alone, as the reference does.
* **Array minus a few values** (the subtrahend at least 64 times shorter) searches the few in the array four at a time
  and moves the stretches between them whole, instead of a merge over every value of the array.

## Bands still above 1.0x CRoaring

The gate lists these with their ceilings.

* `KeepRunsRemove` (`-=`) with scattered singles: 1.2x to 1.4x plain CRoaring, `KeepRunsRemoveAnd` 1.1x to 1.2x and
  `KeepRunsRemoveAndMany` 1.1x with the same doomed set. This is the cost of keeping the smaller form. A run chunk
  that loses a few hundred values to an array is, in CRoaring, converted to a bitset from the runs and cleared bit by
  bit (`run_array_container_andnot`), and every later round on that chunk is a bitset minus an array, a few bit clears
  per value. frsr keeps the runs (a chunk of 190 holes is 760 bytes against 8 KB), so each round writes the chunk's
  runs again (up to about six hundred after four rounds); the walk above costs about eight cycles per doomed value,
  most of it the dependency from one value's run count to the next value's. Taking CRoaring's decision instead (a run
  chunk above the array limit minus an array becomes a bitset) brings these rows to 0.84x to 0.93x CRoaring on the
  clang/glibc build, but the bitmap ends at CRoaring's size (636 KB against 246 KB), the `mixed` rows become slower
  (1.8x), and the single-chunk run-minus-array bands become 2x to 3x slower than now.
* The materialized difference of many tiny array chunks (`Difference/overlap=sparse`) pays a test for the skew per
  pair: 1.00x to 1.06x the time without the array-minus-a-few-values kernel on clang with mimalloc, level on the other
  builds.
* Under 0.1 us per operation, `RunAndNot*` of a run minus a run (`run_run/one_cut`) and a few single-chunk bitset
  residues sit at 1.0x to 1.3x; at that size a few nanoseconds of timer and loop overhead move the ratio.

Running one entry: `frsr_roaring_bench --exact --filter "kernel/frsr-rc-lazyKeepRunsRemove/mixed"`; `--list` shows
every entry.
