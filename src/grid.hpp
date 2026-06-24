#pragma once

#include "common.hpp"
#include <cmath>

enum class bc_kind {
    dirichlet,
    neumann,
};

struct bc_type {
    bc_kind kind = bc_kind::neumann;
    f64 value = NAN;
};

constexpr auto neumann_bc() { return bc_type { .kind=bc_kind::neumann, .value=NAN }; }
constexpr auto dirichlet_bc(f64 v) { return bc_type { .kind=bc_kind::dirichlet, .value=v }; }

struct grid_type {
    v3<bc_type> bnd = { neumann_bc(), neumann_bc(), neumann_bc()}; 
    i32x3 dim = {0, 0, 0};
    f64x3 low = {0.0, 0.0, 0.0};
    f64x3 ext = {0.0, 0.0, 0.0};
    f64x3 del = {ext.x/dim.x, ext.y/dim.y, ext.z/dim.z};

    auto constexpr size() const { return dim.x*dim.y*dim.z; }
};

constexpr auto i2x(grid_type const& grid, u64 ix) { return grid.low.x + ix*grid.del.x; }
constexpr auto i2y(grid_type const& grid, u64 iy) { return grid.low.y + iy*grid.del.y; }
constexpr auto i2z(grid_type const& grid, u64 iz) { return grid.low.z + iz*grid.del.z; }

constexpr auto x2i(grid_type const& grid, f64 x) { return i64((x - grid.low.x)/grid.del.x); }
constexpr auto y2i(grid_type const& grid, f64 y) { return i64((y - grid.low.y)/grid.del.y); }
constexpr auto z2i(grid_type const& grid, f64 z) { return i64((z - grid.low.z)/grid.del.z); }
