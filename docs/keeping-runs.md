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

Linux, AMD Zen 4, clang 23 and g++ 16, Release, `FRSR_ROARING_NO_SBO=1`, `-falign-functions=64`, CRoaring 5.3.0. One process per arm of a
band, pinned to a core under SCHED_FIFO, minimum of five runs; four builds (clang and g++, glibc and mimalloc). Times
are the clang/glibc build in microseconds for the whole four rounds; the ratio columns give the range over the four
builds; below 1.0 is faster than CRoaring.

| rounds of | doomed ids | frsr us | CRoaring us | CRoaring + run_optimize us | frsr / CRoaring | frsr / CRoaring + run_optimize | frsr bytes | CRoaring bytes | CRoaring + run_optimize bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `-=` | singles | 256 | 86 | 915 | 2.91-3.57 | 0.21-0.28 | 245,989 | 635,934 | 245,989 |
| `-=` | ranges | 27 | 22 | 54 | 1.15-1.59 | 0.26-0.50 | 115,517 | 115,517 | 115,517 |
| `-=` | dense chunks | 8 | 9 | 70 | 0.88-1.19 | 0.05-0.12 | 244,825 | 244,825 | 244,825 |
| `-=` | mixed | 51 | 38 | 184 | 1.22-1.80 | 0.18-0.28 | 184,223 | 636,878 | 184,223 |
| `remove_many` of the sorted ids | singles | 308 | 870 | 915 | 0.34-0.39 | 0.33-0.35 | 245,989 | 245,989 | 245,989 |
| `remove_many` of the sorted ids | ranges | 533 | 1547 | 1560 | 0.32-0.35 | 0.31-0.34 | 115,517 | 115,517 | 115,517 |
| `remove_many` of the sorted ids | dense chunks | 1489 | 6707 | 6831 | 0.19-0.22 | 0.18-0.22 | 244,825 | 306,695 | 244,825 |
| `remove_many` of the sorted ids | mixed | 696 | 2947 | 2929 | 0.20-0.24 | 0.20-0.24 | 184,223 | 193,557 | 184,223 |
| `-=`, `&=` a filter | singles | 422 | 176 | 1010 | 2.39-2.63 | 0.29-0.42 | 171,843 | 617,234 | 171,843 |
| `-=`, `&=` a filter | ranges | 126 | 116 | 147 | 1.08-1.22 | 0.64-0.91 | 95,915 | 95,915 | 95,915 |
| `-=`, `&=` a filter | dense chunks | 132 | 158 | 212 | 0.79-0.91 | 0.41-0.66 | 187,225 | 187,225 | 187,225 |
| `-=`, `&=` a filter | mixed | 201 | 166 | 342 | 1.18-1.23 | 0.39-0.59 | 142,301 | 602,872 | 142,301 |
| `-=`, and-many of four filters | singles | 1068 | 517 | 1722 | 2.07-2.44 | 0.53-0.65 | 245,989 | 635,934 | 245,989 |
| `-=`, and-many of four filters | ranges | 598 | 403 | 438 | 1.48-2.10 | 1.37-1.67 | 115,517 | 115,517 | 115,517 |
| `-=`, and-many of four filters | dense chunks | 623 | 544 | 608 | 1.15-1.50 | 0.98-1.17 | 244,825 | 244,825 | 244,825 |
| `-=`, and-many of four filters | mixed | 713 | 529 | 796 | 1.35-1.57 | 0.81-0.97 | 184,223 | 636,878 | 184,223 |

What it shows:

* The bytes are CRoaring's best, without asking. Where the deletion leaves holes in run chunks (`singles`, `mixed`)
  plain CRoaring ends at 2.6x to 4.2x the size (636 KB against 184 KB to 246 KB for `-=`; 603 KB against 142 KB with
  the `&=` rounds); frsr ends at exactly the size CRoaring reaches after `run_optimize`, with no extra pass.
* Against the fair baseline (CRoaring that re-optimizes after each deletion to stay that small) frsr is faster in
  every `-=` row, by 2x to 20x, since `run_optimize` costs more than the deletion it follows; with a `&=` after each
  deletion it is faster by 1.1x to 3.4x. With the four-filter intersection it is faster on singles, level to faster on
  mixed, level to slower on dense chunks and slower on sub-ranges (see below).
* `remove_many` of the sorted doomed ids is 3x to 5x faster than CRoaring's `roaring_bitmap_remove_many` (0.19x to
  0.39x its time) on every doomed set, and ends at the same bytes or fewer (a dense chunk: 245 KB against 307 KB).
* Against plain CRoaring the deletion by `-=` alone is slower with scattered singles and mixed doomed sets and with
  sub-ranges (the time buys the size above); it is level on bitset-dense chunks.
* The single-chunk bands, against the same bands before the runs were kept: a run minus a bitset is 1.2x to 4.7x faster
  (the intersection after it up to 4.5x; a dense residue is unchanged), a run minus an array with a thousand holes is
  1.4x to 1.9x faster, the intersection after that one and the other single-chunk bands are unchanged.

## Bands still above 1.0x CRoaring

The gate lists these with their ceilings. `remove_many_sorted` is not among them: it cuts a run chunk in one merge
pass over the runs and the sorted values, clears the bits of a bitset chunk in one pass and compacts an array chunk in
one pass, and is 0.19x to 0.39x the time of CRoaring's `roaring_bitmap_remove_many` on all four doomed sets.

* `KeepRunsRemove` (`-=`) with scattered singles, and with the mixed set that starts with them: 2.9x to 3.6x plain
  CRoaring. This is the price of the smaller form, not overhead around it. A run chunk that loses more than a few
  hundred values to an array is, in CRoaring, converted to a bitset from the runs and cleared bit by bit
  (`run_array_container_andnot`), which costs the same whatever the number of runs, and every later round on that
  chunk is a bitset minus an array. frsr keeps the runs (a chunk of 190 holes is 760 bytes against 8 KB), so each round
  is one merge of the chunk's runs (up to about a thousand after four rounds) against the doomed values. Per run of
  four rounds, from the instruction counters, that is about 4.0 million instructions and 39 thousand mispredicted
  branches against 1.7 million and 0.4 thousand, at 2.1 instructions per cycle against 3.0; the merge kernel
  (`difference_run_values`) is more than half of the time, allocation under 1%, and the per-chunk work around it
  about a fifth. The mispredictions are the data-dependent choice between taking the next run and the next value, and
  a branch-free form of the same merge measured 2x to 5x slower (a load-latency chain per step). Against the arm that
  re-optimizes after each deletion to stay as small, the same rounds are 3.6x to 4.8x faster.
* `KeepRunsRemove` with sub-ranges: 1.15x to 1.6x, and `KeepRunsRemoveAnd` with sub-ranges 1.1x to 1.2x. The first
  round's difference of an array chunk and the runs used to test every array value against the runs; it now gallops
  past the stretch a run covers and copies the stretch it keeps (the sub-ranges rows were 1.8x to 2.5x and 1.1x to
  1.3x before).
* `KeepRunsRemoveAnd` with singles and mixed doomed sets: the deletion above, plus the intersection with the filter.
* `KeepRunsRemoveAndMany` (the intersection of the bitmap with four run-heavy filters): 1.2x to 2.4x plain CRoaring's
  chained `and`, and on ranges 1.4x to 1.7x the arm that re-optimizes, whose containers are the same forms and the same
  bytes. That gap is the intersection, not the deletion: on ranges, 44% of the time is `array & runs` and 27% is
  `run & bitset`, both of them kernels shared with `&=`; the array-with-runs and run-run intersections are the subject
  of their own branch.
* `AndAfterRunAndNot` and `AndInplaceAfterRunAndNot` on a run with a thousand holes: 2.2x to 3.8x, an intersection
  of two run containers of a thousand runs each against CRoaring's array path. Unchanged by keeping runs.
* Under 0.1 us per operation, `RunAndNot*` of a run minus a run (`run_run/one_cut`) and a few single-chunk bitset
  residues sit at 1.0x to 1.3x; at that size a few nanoseconds of timer and loop overhead move the ratio.

Running one entry: `frsr_roaring_bench --exact --filter "kernel/frsr-rc-lazyKeepRunsRemove/mixed"`; `--list` shows
every entry.
