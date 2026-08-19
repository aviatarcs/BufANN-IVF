#pragma once

#include <cstddef>

namespace folly {

inline std::size_t nextPowTwo(std::size_t value) {
  if (value <= 1) {
    return 1;
  }
  --value;
  for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
    value |= (value >> shift);
  }
  return value + 1;
}

} // namespace folly
