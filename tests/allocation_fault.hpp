#pragma once

// Allocation failures on demand for tests. The binary's global operator new (tests/cow_semantics.cpp)
// consults this: while a thread is armed, every `period`-th allocation of that thread throws
// std::bad_alloc. Disarmed (the default) it costs one thread-local load.
//
// Under ThreadSanitizer the global operators are the sanitizer's own and cannot be replaced, so
// nothing is injected there: FRSR_TESTS_COUNT_ALLOCATIONS is 0 and tests that need failures skip.

#include <cstdint>

#if defined( __SANITIZE_THREAD__ )
#   define FRSR_TESTS_COUNT_ALLOCATIONS 0
#elif defined( __has_feature )
#   if __has_feature( thread_sanitizer )
#       define FRSR_TESTS_COUNT_ALLOCATIONS 0
#   endif
#endif
#ifndef FRSR_TESTS_COUNT_ALLOCATIONS
#   define FRSR_TESTS_COUNT_ALLOCATIONS 1
#endif

namespace frsr::roaring::kernel_test::alloc_fault {

inline thread_local std::uint32_t period { 0 };
inline thread_local std::uint32_t counter{ 0 };

[[nodiscard]] inline bool should_fail() noexcept { return period != 0 && ++counter % period == 0; }

struct armed {
    explicit armed( std::uint32_t const every ) noexcept { period = every; }
    armed( armed const & ) = delete;
    ~armed() { period = 0; }
};

} // namespace frsr::roaring::kernel_test::alloc_fault
