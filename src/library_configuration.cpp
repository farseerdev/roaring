#include "instantiate.hpp"

template <bool SmallBuffers, bool CombineStats, bool AllocationFailureTerminates, bool PsiVm, bool VmSerialization>
int const frsr::roaring::detail::library_configuration<SmallBuffers, CombineStats, AllocationFailureTerminates, PsiVm, VmSerialization>::witness{ 0 };

template struct frsr::roaring::detail::library_configuration<FRSR_ROARING_LIBRARY_CONFIGURATION_ARGUMENTS>;
