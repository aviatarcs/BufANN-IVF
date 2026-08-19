#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define FOLLY_LIKELY(x) (__builtin_expect(!!(x), 1))
#define FOLLY_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define FOLLY_LIKELY(x) (x)
#define FOLLY_UNLIKELY(x) (x)
#endif
