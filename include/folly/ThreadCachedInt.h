#pragma once

#include <atomic>
#include <cstdint>

namespace folly {

template <class IntT, class Tag = IntT>
class ThreadCachedInt {
 public:
  explicit ThreadCachedInt(IntT initialVal = 0, uint32_t cacheSize = 1000)
      : target_(initialVal), cacheSize_(cacheSize) {}

  ThreadCachedInt(const ThreadCachedInt&) = delete;
  ThreadCachedInt& operator=(const ThreadCachedInt&) = delete;

  void increment(IntT inc) { target_.fetch_add(inc, std::memory_order_relaxed); }

  IntT readFast() const { return target_.load(std::memory_order_relaxed); }

  IntT readFull() const { return readFast(); }

  IntT readFastAndReset() { return target_.exchange(0, std::memory_order_release); }

  IntT readFullAndReset() { return readFastAndReset(); }

  void setCacheSize(uint32_t newSize) {
    cacheSize_.store(newSize, std::memory_order_release);
  }

  uint32_t getCacheSize() const {
    return cacheSize_.load(std::memory_order_relaxed);
  }

  ThreadCachedInt& operator+=(IntT inc) {
    increment(inc);
    return *this;
  }

  ThreadCachedInt& operator-=(IntT inc) {
    increment(-inc);
    return *this;
  }

  ThreadCachedInt& operator++() {
    increment(1);
    return *this;
  }

  ThreadCachedInt& operator--() {
    increment(IntT(-1));
    return *this;
  }

  void set(IntT newVal) { target_.store(newVal, std::memory_order_release); }

 private:
  std::atomic<IntT> target_;
  std::atomic<uint32_t> cacheSize_;
};

} // namespace folly
