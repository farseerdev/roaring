#pragma once

// Run-selection policy for the DEFAULT container set (default_container_set<Key>):
// whether ordinary construction from sorted values (inserts via add_many_sorted /
// add_sorted_many, add_closed_range, and merge results from union/intersect/
// difference) automatically considers run-length encoding, or only ever picks
// array (below the bitset threshold) / bitset (at/above).
//
// This is orthogonal to whether the ContainerSet typelist admits run_container as
// a TYPE (bitmap::supports_run_container) — that gates structural support; this
// policy gates the AUTOMATIC/implicit selection made during ordinary insert/merge.
// Run containers remain fully producible via the explicit optimize() /
// optimize_for_storage() compaction path regardless of this policy — exactly
// mirroring CRoaring, which never auto-creates run containers during normal
// insert/merge and only run-encodes via an explicit run_optimize()-style call.
//
//   run_selection_lazy  — never auto-picks run encoding outside optimize()/
//                         optimize_for_storage(); CRoaring parity. Best for data
//                         that is mutated/merged far more often than it is
//                         explicitly compacted (e.g. a downstream engine's live index bitmaps).
//   run_selection_eager — unconditionally considers run encoding via the
//                         smallest-serialized-size estimate during ordinary
//                         construction too. The library's historical default —
//                         favors storage-biased workloads (built once, read many,
//                         rarely mutated) where paying the selection cost per
//                         insert/merge is worth the steady-state compactness.
//
// See the container-representation design notes.

namespace frsr::roaring::detail {

struct run_selection_lazy {
    static constexpr bool eager{ false };
};

struct run_selection_eager {
    static constexpr bool eager{ true };
};

} // namespace frsr::roaring::detail
