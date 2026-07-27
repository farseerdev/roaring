#pragma once

#include <cstdint>

namespace frsr::roaring::detail {

enum class range_operation : std::uint8_t { add, remove, flip };
enum class set_operation : std::uint8_t { bit_or, bit_and, bit_andnot };

} // namespace frsr::roaring::detail
