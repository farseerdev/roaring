#include <frsr/roaring/bitmap.hpp>

template class frsr::roaring::bitmap<std::uint16_t>;
template class frsr::roaring::bitmap<std::uint32_t>;
template class frsr::roaring::bitmap<std::uint64_t>;

// Refcounted (CoW Model 2) instantiation for a downstream consumer's shared-master adoption path.
template class frsr::roaring::bitmap<
    std::uint32_t,
    frsr::roaring::default_container_set<std::uint32_t>,
    frsr::roaring::detail::cow_atomic_refcount
>;

// A downstream consumer's actual instantiation: CoW Model 2 +
// run_selection_lazy (CRoaring parity — never auto-picks run encoding outside
// optimize()/optimize_for_storage()).
template class frsr::roaring::bitmap<
    std::uint32_t,
    frsr::roaring::default_container_set<std::uint32_t>,
    frsr::roaring::detail::cow_atomic_refcount,
    frsr::roaring::detail::run_selection_lazy
>;

