#pragma once

#include "common.hpp"
#include "grid.hpp"

#include <cmath>

inline constexpr void
make_const(grid_type const& grid,
           matrix state,
           f64 const u) {
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                state[ix*grid.dim.y*grid.dim.z + iy*grid.dim.z + iz] = u; 
            }
        }
    }
}

inline constexpr void
add_gaussian(grid_type const& grid,
             matrix state,
             f64 const A, f64 const sigma, f64 const bg) {
    auto const cx = 0.5*grid.ext.x, cy = 0.5*grid.ext.y, cz = 0.5*grid.ext.z;
    auto const width = 1/(2.0*sigma*sigma);
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        auto const x = i2x(grid, ix) - cx;
        for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            auto const y = i2y(grid, iy) - cy;
            for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                auto const z = i2z(grid, iz) - cz;
                auto const d = x*x + y*y + z*z;
                auto const u = (A - bg)*std::exp(-d*width) + bg;
                state[ix*grid.dim.y*grid.dim.z + iy*grid.dim.z + iz] += u; 
            }
        }
    }
}

inline constexpr void
make_gaussian(grid_type const& grid,
              matrix state,
              f64 const A, f64 const sigma, f64 const bg) {
    make_const(grid, state, 0.0);
    add_gaussian(grid, state, A, sigma, bg);
}


inline constexpr void
add_step(grid_type const& grid,
          matrix state,
          f64 const A, f64 const radius, f64 const bg) {
    auto const cx = 0.5*grid.ext.x, cy = 0.5*grid.ext.y, cz = 0.5*grid.ext.z;
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        auto const x = i2x(grid, ix) - cx;
        for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            auto const y = i2y(grid, iy) - cy;
            for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                auto const z = i2z(grid, iz) - cz;
                auto const d = x*x + y*y + z*z;
                if (d > radius*radius) {
                    state[ix*grid.dim.y*grid.dim.z + iy*grid.dim.z + iz] += bg;
                }
                else {
                    state[ix*grid.dim.y*grid.dim.z + iy*grid.dim.z + iz] += A;
                }
            }
        }
    }
}

inline constexpr void
make_step(grid_type const& grid,
          matrix state,
         f64 const A, f64 const radius, f64 const bg) {
    make_const(grid, state, 0.0);
    add_step(grid, state, A, radius, bg);
}
