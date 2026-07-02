#include <cassert>
#include <cmath>
#include <iostream>

#include "common.hpp"
#include "diff.hpp"
#include "grid.hpp"
#include "state.hpp"

inline constexpr void tridia_solve(i32 lo, i32 hi, f64 *sap_noalias const sub,
                                   f64 *sap_noalias const dia,
                                   f64 *sap_noalias const sup,
                                   f64 *sap_noalias const rhs) {
  sup[lo] = sup[lo] / dia[lo];
  rhs[lo] = rhs[lo] / dia[lo];

  for (i32 ix = lo + 1; ix < hi - 1; ++ix) {
    auto const den = dia[ix] - sub[ix] * sup[ix - 1];
    sup[ix] = (ix < hi - 1) ? sup[ix] / den : 0.0;
    rhs[ix] = (rhs[ix] - sub[ix] * rhs[ix - 1]) / den;
  }
  for (i32 ix = hi - 2; ix >= lo; --ix) {
    rhs[ix] = rhs[ix] - sup[ix] * rhs[ix + 1];
  }
}

// Compute Laplacians of field u per-direction
// x-direction w/ zero flux bc
inline constexpr auto lap_x(const grid_type &grid, i32 ix, i32 iy, i32 iz,
                            matrix const u) {
  auto const u_xc = u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
  auto const u_xm =
      u[(ix - 1) * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
  auto const u_xp =
      u[(ix + 1) * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
  return u_xm - 2.0 * u_xc + u_xp;
}

// y-direction w/ zero flux bc
inline constexpr auto lap_y(const grid_type &grid, i32 ix, i32 iy, i32 iz,
                            matrix const u) {
  auto const u_yc = u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
  auto const u_ym =
      u[ix * grid.dim.y * grid.dim.z + (iy - 1) * grid.dim.z + iz];
  auto const u_yp =
      u[ix * grid.dim.y * grid.dim.z + (iy + 1) * grid.dim.z + iz];
  return u_ym - 2.0 * u_yc + u_yp;
}

// z-direction w/ zero flux bc
inline constexpr auto lap_z(const grid_type &grid, i32 ix, i32 iy, i32 iz,
                            matrix const u) {
  auto const u_zc = u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
  auto const u_zm = u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz - 1];
  auto const u_zp = u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz + 1];
  return u_zm - 2.0 * u_zc + u_zp;
}

inline void make_neumann_bc(grid_type const &grid, matrix u) {
  auto const off = 2;
#pragma omp for
  for (i32 ix = 0; ix < grid.dim.x; ++ix) {
    for (i32 iy = 0; iy < grid.dim.y; ++iy) {
      {
        i32 iz = 0;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz + off];
      }
      {
        i32 iz = grid.dim.z - 1;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz - off];
      }
    }
  }
#pragma omp for
  for (i32 ix = 0; ix < grid.dim.x; ++ix) {
    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
      {
        i32 iy = 0;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[ix * grid.dim.y * grid.dim.z + (iy + off) * grid.dim.z + iz];
      }
      {
        i32 iy = grid.dim.y - 1;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[ix * grid.dim.y * grid.dim.z + (iy - off) * grid.dim.z + iz];
      }
    }
  }
#pragma omp for
  for (i32 iy = 0; iy < grid.dim.y; ++iy) {
    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
      {
        i32 ix = 0;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[(ix + off) * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
      }
      {
        i32 ix = grid.dim.x - 1;
        u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
            u[(ix - off) * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
      }
    }
  }
}

template <u64 N>
void adi_step(state_type<N> &state, f64 t, f64 dt) {
  auto const &grid = state.grid;
  auto const &dif_fn = state.diffusion;
  auto const dx = grid.del.x;
  auto const dy = grid.del.y;
  auto const dz = grid.del.z;

  auto const i_dx2 = 0.5 * dt / (dx * dx);
  auto const i_dy2 = 0.5 * dt / (dy * dy);
  auto const i_dz2 = 0.5 * dt / (dz * dz);

  for (i32 ion = 0; ion < state.n_species; ++ion) {
    auto u_cur = state.species[ion];
    auto u_nxt = state.tmp;
    make_neumann_bc(grid, u_cur);

    auto const& dif_fn = state.diffusion[ion];
    if (dynamic_cast<non_diffusive*>(dif_fn.get())) continue;
    
    // NOTE: We _could_ make this parallel over the ion species _but_
    // this would require extra space, in fact Nx Ny Nz f64 _per ion_.
    // Currently it's one working array, not Nion.
    #pragma omp parallel
    {
      auto const tid = omp_get_thread_num();
      auto rhs = state.rhs + tid * state.linear_ext;
      auto sub = state.sub + tid * state.linear_ext;
      auto dia = state.dia + tid * state.linear_ext;
      auto sup = state.sup + tid * state.linear_ext;

      // X-implicit, explicit in y, z.
      #pragma omp for collapse(2)
      for (i32 iy = 1; iy < grid.dim.y - 1; ++iy) {
        for (i32 iz = 1; iz < grid.dim.z - 1; ++iz) {
          auto y = i2y(grid, iy);
          auto z = i2z(grid, iz);
          for (i32 ix = 0; ix < grid.dim.x; ++ix) {
            auto const x = i2x(grid, ix);
            auto const Dy = dif_fn->y(x, y, z);
            auto const Dz = dif_fn->z(x, y, z);
            auto const yyu = lap_y(grid, ix, iy, iz, u_cur) * i_dy2;
            auto const zzu = lap_z(grid, ix, iy, iz, u_cur) * i_dz2;
            rhs[ix] = u_cur[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz]
                    + Dy * yyu + Dz * zzu;
            auto const Dxm = dif_fn->x(x - dx, y, z);
            auto const Dxp = dif_fn->x(x + dx, y, z);
            sub[ix] = -Dxm * i_dx2;
            dia[ix] = 1.0 + (Dxm + Dxp) * i_dx2;
            sup[ix] = -Dxp * i_dx2;
          }
          // Fix BC: Neumanr
          auto const Dxp = dif_fn->x(dx, y, z);
          sub[0] = 0.0;
          dia[0] = 1.0 + Dxp * i_dx2;
          sup[0] = -Dxp * i_dx2;

          auto const Dxm = dif_fn->x(grid.ext.x - dx, y, z);
          sub[grid.dim.x - 1] = -Dxm * i_dx2;
          dia[grid.dim.x - 1] = 1.0 + Dxm * i_dx2;
          sup[grid.dim.x - 1] = 0.0;

          //  solve line
          tridia_solve(0, grid.dim.x, sub, dia, sup, rhs);
          // write line into cube: cur -> nxt
          for (i32 ix = 1; ix < grid.dim.x - 1; ++ix) {
            u_nxt[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
                rhs[ix];
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        state.swap(ion);
        u_cur = state.species[ion];
        u_nxt = state.tmp;
      }
      
      #pragma omp barrier
      make_neumann_bc(grid, u_cur);

      // Y-implicit, explicit in x,z.
      #pragma omp for collapse(2)
      for (i32 ix = 1; ix < grid.dim.x - 1; ++ix) {
        for (i32 iz = 1; iz < grid.dim.z - 1; ++iz) {
          auto const x = i2x(grid, ix);
          auto const z = i2z(grid, iz);
          for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            auto const y = i2y(grid, iy);
            auto const xxu = lap_x(grid, ix, iy, iz, u_cur) * i_dx2;
            auto const zzu = lap_z(grid, ix, iy, iz, u_cur) * i_dz2;
            auto const Dx = dif_fn->x(x, y, z);
            auto const Dz = dif_fn->z(x, y, z);
            rhs[iy] =
                u_cur[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] +
                Dx * xxu + Dz * zzu;
            auto const Dym = dif_fn->y(x, y - dy, z);
            auto const Dyp = dif_fn->y(x, y + dy, z);
            sub[iy] = -Dym * i_dy2;
            dia[iy] = 1.0 + (Dym + Dyp) * i_dy2;
            sup[iy] = -Dyp * i_dy2;
          }
          // Fix BC: Neumann
          auto const Dyp = dif_fn->y(x, dy, z);
          sub[0] = 0.0;
          dia[0] = 1.0 + Dyp * i_dy2;
          sup[0] = -Dyp * i_dy2;

          auto const Dym = dif_fn->y(x, grid.ext.y - dy, z);
          sub[grid.dim.y - 1] = -Dym * i_dy2;
          dia[grid.dim.y - 1] = 1.0 + Dym * i_dy2;
          sup[grid.dim.y - 1] = 0.0;

          //  solve line
          tridia_solve(0, grid.dim.y, sub, dia, sup, rhs);
          // write line into cube
          for (i32 iy = 1; iy < grid.dim.y - 1; ++iy) {
            u_nxt[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] = rhs[iy];
          }
        }
      }

      #pragma omp barrier
      #pragma omp master
      {
        state.swap(ion);
        u_cur = state.species[ion];
        u_nxt = state.tmp;
      }
      #pragma omp barrier
      make_neumann_bc(grid, u_cur);

      // Z-implicit, explicit in x,y.
      #pragma omp for collapse(2)
      for (i32 ix = 1; ix < grid.dim.x - 1; ++ix) {
        for (i32 iy = 1; iy < grid.dim.y - 1; ++iy) {
          auto const x = i2x(grid, ix);
          auto const y = i2y(grid, iy);
          for (i32 iz = 0; iz < grid.dim.z; ++iz) {
            auto const z = i2z(grid, iz);
            auto const xxu = lap_x(grid, ix, iy, iz, u_cur) * i_dx2;
            auto const yyu = lap_y(grid, ix, iy, iz, u_cur) * i_dy2;
            auto const Dx = dif_fn->x(x, y, z);
            auto const Dy = dif_fn->y(x, y, z);
            rhs[iz] =
                u_cur[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] +
                Dx * xxu + Dy * yyu;
            auto const Dzm = dif_fn->z(x, y, z - dz);
            auto const Dzp = dif_fn->z(x, y, z + dz);
            sub[iz] = -Dzm * i_dz2;
            dia[iz] = 1.0 + (Dzm + Dzp) * i_dz2;
            sup[iz] = -Dzp * i_dz2;
          }
          auto const Dzp = dif_fn->z(x, y, dz);
          sub[0] = 0.0;
          dia[0] = 1.0 + Dzp * i_dz2;
          sup[0] = -Dzp * i_dz2;
          auto const Dzm = dif_fn->z(x, y, grid.ext.z - dz);
          sub[grid.dim.z - 1] = -Dzm * i_dz2;
          dia[grid.dim.z - 1] = 1.0 + Dzm * i_dz2;
          sup[grid.dim.z - 1] = 0.0;
          //  solve line
          tridia_solve(0, grid.dim.z, sub, dia, sup, rhs);
          // write line into cube: cur -> nxt
          for (i32 iz = 1; iz < grid.dim.z - 1; ++iz) {
            u_nxt[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz] =
                rhs[iz];
          }
        }
      }

#pragma omp barrier
#pragma omp master
      {
        state.swap(ion);
        u_cur = state.species[ion];
        u_nxt = state.tmp;
      }
#pragma omp barrier
      make_neumann_bc(grid, u_cur);
    }
  } // ions
}
