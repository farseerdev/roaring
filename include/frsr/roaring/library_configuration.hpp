#pragma once

// The options that change the code of a bitmap member (the small-buffer containers, the combine statistics, the
// allocation failure policy, psi::vm containers and file-mapped serialization) are visible to the library only
// when it is compiled. A program that sets any of them differently from the library it links would run
// instantiations laid out and behaving otherwise than it expects; the program therefore refers to a symbol
// that only a library built with the same settings defines, and fails to link against any other.

#include <frsr/roaring/allocation_policy.hpp>
#include <frsr/roaring/tuning.hpp>

#ifndef FRSR_ROARING_COMBINE_STATS
#   define FRSR_ROARING_COMBINE_STATS 0
#endif

namespace frsr::roaring::detail {

template <bool SmallBuffers, bool CombineStats, bool AllocationFailureTerminates, bool PsiVm, bool VmSerialization>
struct library_configuration {
    static int const witness;
};

#ifdef FRSR_ROARING_HAS_PSI_VM
inline constexpr bool configured_psi_vm{ true };
#else
inline constexpr bool configured_psi_vm{ false };
#endif

#ifdef FRSR_ROARING_ENABLE_VM_VECTOR_SERIALIZATION
inline constexpr bool configured_vm_serialization{ true };
#else
inline constexpr bool configured_vm_serialization{ false };
#endif

#define FRSR_ROARING_LIBRARY_CONFIGURATION_ARGUMENTS \
    !FRSR_ROARING_NO_SBO, FRSR_ROARING_COMBINE_STATS != 0, ::frsr::roaring::detail::allocation_failure_terminates, \
    ::frsr::roaring::detail::configured_psi_vm, ::frsr::roaring::detail::configured_vm_serialization

using this_library_configuration = library_configuration<FRSR_ROARING_LIBRARY_CONFIGURATION_ARGUMENTS>;

#ifndef FRSR_ROARING_DEFINES_LIBRARY_CONFIGURATION
[[ gnu::used ]] inline int const * const library_configuration_witness{ &this_library_configuration::witness };
#endif

} // namespace frsr::roaring::detail
