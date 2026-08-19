#pragma once

#include <iterator>
#include <memory>

namespace folly {
namespace detail {

template <class Derived, class Value, class Category>
class IteratorFacade {
 public:
  using value_type = Value;
  using reference = Value&;
  using pointer = Value*;
  using difference_type = std::ptrdiff_t;
  using iterator_category = Category;

  Derived& operator++() {
    derived().increment();
    return derived();
  }

  Derived operator++(int) {
    Derived tmp(derived());
    derived().increment();
    return tmp;
  }

  reference operator*() const { return derived().dereference(); }

  pointer operator->() const { return std::addressof(derived().dereference()); }

  bool operator==(const Derived& other) const { return derived().equal(other); }

  bool operator!=(const Derived& other) const { return !derived().equal(other); }

 private:
  Derived& derived() { return *static_cast<Derived*>(this); }

  const Derived& derived() const { return *static_cast<const Derived*>(this); }
};

} // namespace detail
} // namespace folly
