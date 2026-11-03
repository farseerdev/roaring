#include "instantiate.hpp"

template <bool SmallBuffers, bool CombineStats, bool AllocationFailureTerminates, bool PsiVm, bool VmSerialization, std::uint64_t TableSharing>
int const frsr::roaring::detail::library_configuration<SmallBuffers, CombineStats, AllocationFailureTerminates, PsiVm, VmSerialization, TableSharing>::witness{ 0 };

template struct frsr::roaring::detail::library_configuration<FRSR_ROARING_LIBRARY_CONFIGURATION_ARGUMENTS>;
