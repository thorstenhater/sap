#pragma once

#include <cstdint>
#define noalias __restrict__

using f64 = double;
using i32 = int;
using u64 = std::uint64_t;
using i64 = std::int64_t;

template<typename T>
struct v3 {
    T x, y, z;
};

using u64x3 = v3<u64>;
using i32x3 = v3<i64>;
using f64x3 = v3<f64>;
