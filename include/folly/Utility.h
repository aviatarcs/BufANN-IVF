#pragma once

#include <utility>

namespace folly {

struct Identity {
  template <class T>
  constexpr T&& operator()(T&& value) const noexcept {
    return std::forward<T>(value);
  }
};

} // namespace folly
