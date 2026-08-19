#pragma once

#include <type_traits>

#define FOR_EACH_RANGE(i, begin, end)                                         \
  for (std::decay_t<decltype(end)> i =                                        \
           static_cast<std::decay_t<decltype(end)>>(begin),                   \
       i##_folly_end = static_cast<std::decay_t<decltype(end)>>(end);         \
       i < i##_folly_end;                                                      \
       ++i)
