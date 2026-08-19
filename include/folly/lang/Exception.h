#pragma once

#include <utility>

namespace folly {

template <class Exception, class... Args>
[[noreturn]] inline void throw_exception(Args&&... args) {
  throw Exception(std::forward<Args>(args)...);
}

} // namespace folly
