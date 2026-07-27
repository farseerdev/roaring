#pragma once

#include <concepts>

namespace frsr::roaring {

template <std::unsigned_integral Value>
struct run {
    Value begin{};
    Value end{};
};

} // namespace frsr::roaring
