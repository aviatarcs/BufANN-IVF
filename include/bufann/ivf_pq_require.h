#pragma once

#include "ann_exception.h"

// Precondition check; throws ANNException recording the call site.
#define IVF_PQ_REQUIRE(cond, msg)                                                         \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            throw diskann::ANNException((msg), -1, __FUNCSIG__, __FILE__, __LINE__);      \
        }                                                                                 \
    } while (0)
