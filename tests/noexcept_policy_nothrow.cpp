// Compiled alone, with an allocation policy under which a failed allocation cannot be observed as an exception:
// every operation that allocates is then noexcept (see noexcept_checks.hpp), which is what lets a caller's own
// operations be noexcept at no cost. Not linked into a test binary: a different policy changes the exception
// specifications of the very same templates, so it must not share an executable with the default one.
#define FRSR_ROARING_NOEXCEPT_EXCEPT_BADALLOC noexcept
#include "noexcept_checks.hpp"

static_assert( frsr::roaring::noexcept_checks::terminates );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> &>().add( 1U ) ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint64_t> &>() ^= std::declval<frsr::roaring::bitmap<std::uint64_t> const &>() ) );
static_assert( noexcept( std::declval<frsr::roaring::bitmap<std::uint32_t> &>().xor_with( std::declval<frsr::roaring::key_range>() ) ) );
