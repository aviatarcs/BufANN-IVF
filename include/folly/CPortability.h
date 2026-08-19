#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>

#ifndef FOLLY_EXPORT
#define FOLLY_EXPORT
#endif

#ifndef CHECK
#define CHECK(expr)                                                           \
  do {                                                                        \
    if (!(expr)) {                                                            \
      std::cerr << "CHECK failed: " #expr << " at " << __FILE__ << ":"      \
                << __LINE__ << std::endl;                                     \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(a, b) CHECK((a) == (b))
#endif

#ifndef CHECK_NE
#define CHECK_NE(a, b) CHECK((a) != (b))
#endif

#ifndef CHECK_LT
#define CHECK_LT(a, b) CHECK((a) < (b))
#endif

#ifndef CHECK_LE
#define CHECK_LE(a, b) CHECK((a) <= (b))
#endif

#ifndef CHECK_GT
#define CHECK_GT(a, b) CHECK((a) > (b))
#endif

#ifndef CHECK_GE
#define CHECK_GE(a, b) CHECK((a) >= (b))
#endif

#ifndef DCHECK
#define DCHECK(expr) CHECK(expr)
#endif

#ifndef DCHECK_EQ
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#endif

#ifndef DCHECK_NE
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#endif

#ifndef DCHECK_LT
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#endif

#ifndef DCHECK_LE
#define DCHECK_LE(a, b) CHECK_LE(a, b)
#endif

#ifndef DCHECK_GT
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#endif

#ifndef DCHECK_GE
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#endif
