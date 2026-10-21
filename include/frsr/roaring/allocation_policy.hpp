#pragma once

// What a failed allocation does, as a compile-time property of the build.
//
// The operations that may allocate (add, the set operations, optimize, ...) are declared
// `noexcept( detail::allocation_failure_terminates )`. In a build where an allocation failure is an
// exception (the default) they are not noexcept and throw std::bad_alloc, leaving the bitmaps valid (see
// the notes on each operation). In a build where it cannot happen, or must end the program, they are
// noexcept, which lets a caller's own operations be noexcept for free and removes the unwinding paths
// from the generated code.
//
// The policy is that of the allocator the build was configured with, spelled
// PSI_NOEXCEPT_EXCEPT_BADALLOC when psi::vm is in use (`noexcept` for an allocator that cannot fail, empty
// otherwise). A build without it may name the policy itself by defining FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC
// to `noexcept` or to nothing (which is also what it is when neither is defined).

#ifndef FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC
#   ifdef PSI_NOEXCEPT_EXCEPT_BADALLOC
#       define FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC PSI_NOEXCEPT_EXCEPT_BADALLOC
#   else
#       define FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC
#   endif
#endif

#include <concepts>
#include <ranges>

namespace frsr::roaring::detail {

// Declared, never defined: names the policy in a function type so that noexcept() can read it.
void allocation_policy_probe() FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC;

// True when an allocation failure cannot be observed as an exception.
inline constexpr bool allocation_failure_terminates{ noexcept( allocation_policy_probe() ) };

// A range of keys that is walked without calling anything that can throw (contiguous storage of the keys).
template <typename Range, typename Key>
inline constexpr bool nothrow_key_range{ std::ranges::contiguous_range<Range> && std::convertible_to<std::ranges::range_reference_t<Range>, Key> };

} // namespace frsr::roaring::detail
