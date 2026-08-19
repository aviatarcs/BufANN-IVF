#pragma once

#include <thread>

namespace folly {
namespace detail {

template <class Pred>
inline void atomic_hash_spin_wait(Pred&& pred) {
  while (pred()) {
    std::this_thread::yield();
  }
}

} // namespace detail
} // namespace folly
