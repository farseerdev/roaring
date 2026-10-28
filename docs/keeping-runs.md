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
| `-=` | singles | 275 | 87 | 880 | 3.05-3.70 | 0.21-0.31 | 245,989 | 635,934 | 245,989 |
| `-=` | ranges | 37 | 22 | 54 | 1.68-2.44 | 0.41-0.69 | 115,517 | 115,517 | 115,517 |
| `-=` | dense chunks | 8 | 9 | 70 | 0.91-1.17 | 0.05-0.12 | 244,825 | 244,825 | 244,825 |
| `-=` | mixed | 53 | 37 | 176 | 1.29-1.85 | 0.18-0.30 | 184,223 | 636,878 | 184,223 |
| `-=`, `&=` a filter | singles | 432 | 164 | 998 | 2.64-2.80 | 0.30-0.43 | 171,843 | 617,234 | 171,843 |
| `-=`, `&=` a filter | ranges | 126 | 101 | 133 | 1.24-1.47 | 0.73-1.04 | 95,915 | 95,915 | 95,915 |
| `-=`, `&=` a filter | dense chunks | 130 | 132 | 186 | 0.90-1.06 | 0.46-0.74 | 187,225 | 187,225 | 187,225 |
| `-=`, `&=` a filter | mixed | 205 | 149 | 326 | 1.34-1.47 | 0.44-0.63 | 142,301 | 602,872 | 142,301 |
| `-=`, and-many of four filters | singles | 1107 | 466 | 1766 | 2.32-2.55 | 0.56-0.64 | 245,989 | 635,934 | 245,989 |
| `-=`, and-many of four filters | ranges | 620 | 348 | 387 | 1.78-2.27 | 1.60-1.97 | 115,517 | 115,517 | 115,517 |
| `-=`, and-many of four filters | dense chunks | 623 | 458 | 526 | 1.36-1.62 | 1.06-1.39 | 244,825 | 244,825 | 244,825 |
| `-=`, and-many of four filters | mixed | 720 | 467 | 775 | 1.54-1.67 | 0.89-1.03 | 184,223 | 636,878 | 184,223 |

What it shows:

* The bytes are CRoaring's best, without asking. Where the deletion leaves holes in run chunks (`singles`, `mixed`)
  plain CRoaring ends at 2.6x to 4.2x the size (636 KB against 184 KB to 246 KB for `-=`; 603 KB against 142 KB with
  the `&=` rounds); frsr ends at exactly the size CRoaring reaches after `run_optimize`, with no extra pass.
* Against the fair baseline (CRoaring that re-optimizes after each deletion to stay that small) frsr is faster in
  every `-=` row, by 1.4x to 20x, since `run_optimize` costs more than the deletion it follows; with a `&=` after each
  deletion it is faster by 1.4x to 3.3x on singles, mixed and dense chunks and level on sub-ranges (0.73 to 1.04).
  With the four-filter intersection it is faster on singles, level on mixed, and slower on sub-ranges and dense chunks
  (see below).
* Against plain CRoaring the deletion alone is slower with scattered singles and mixed doomed sets (the time buys the
  size above); it is level on bitset-dense chunks.
* The single-chunk bands, against the same bands before the runs were kept: a run minus a bitset is 1.2x to 4.7x faster
  (the intersection after it up to 4.5x; a dense residue is unchanged), a run minus an array with a thousand holes is
  1.4x to 1.9x faster, the intersection after that one and the other single-chunk bands are unchanged.

## Bands still above 1.0x CRoaring

The gate lists these with their ceilings; none is new with keeping runs, except the first.

* `KeepRunsRemoveMany` (`remove_many_sorted` of the doomed ids): 3x to 116x slower than CRoaring's
  `roaring_bitmap_remove_many`, and the dense-chunk and mixed rows regressed against the commit before runs were
  kept. `remove_many_sorted` removes the values of a non-array chunk one at a time, and a run chunk that keeps its
  runs pays a search and a shift in the run vector for each one. It needs the same merge walk the array chunk has.
  The gate has no ceiling for these bands and fails on them until that is done.
* `KeepRunsRemoveAndMany` (the intersection of the bitmap with four run-heavy filters): 1.4x to 2.6x slower than
  CRoaring's chained `and`, and 1.1x to 2x slower than the `run_optimize` arm on ranges and dense chunks. This is
  `and_many`, not the deletion.
* `KeepRunsRemove`, `KeepRunsRemoveAnd` with singles, sub-ranges or mixed doomed sets: 1.2x to 3.7x plain CRoaring,
  which does not pay for the size (see above); against the `run_optimize` arm they are faster.
* `AndAfterRunAndNot` and `AndInplaceAfterRunAndNot` on a run with a thousand holes: 2.2x to 3.8x, an intersection
  of two run containers of a thousand runs each against CRoaring's array path. Unchanged by keeping runs.
* Under 0.1 us per operation, `RunAndNot*` of a run minus a run (`run_run/one_cut`) and a few single-chunk bitset
  residues sit at 1.0x to 1.3x; at that size a few nanoseconds of timer and loop overhead move the ratio.

Running one entry: `frsr_roaring_bench --exact --filter "kernel/frsr-rc-lazyKeepRunsRemove/mixed"`; `--list` shows
every entry.
