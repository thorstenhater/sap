#pragma once

#include "common.hpp"
#include "diff.hpp"
#include "grid.hpp"
#include "react.hpp"

#include <array>
#include <fstream>

#include <omp.h>

template <u64 N> struct state_type {
  grid_type grid;
  i32 n_species = -1;
  i32 n_threads = -1;
  i32 linear_ext = -1;
  reaction_ptr<N> reaction;
  std::array<diffusion_ptr, N> diffusion;
  std::array<f64 *, N> species; // current concentration field per species
  f64 *tmp = nullptr;           // swap/working concentration field
  f64 *sub = nullptr;           // subdiagonals
  f64 *dia = nullptr;           // diagonals
  f64 *sup = nullptr;           // superdiagonals
  f64 *rhs = nullptr;           // right hand sides

  state_type(grid_type const &grid,
             reaction_ptr<N> r,
             std::array<diffusion_ptr, N> d):
               grid(grid), n_species(N),
        linear_ext(std::max(grid.dim.x, std::max(grid.dim.y, grid.dim.z))),
        reaction(std::move(r)), diffusion(std::move(d)) {
    for (i32 ix = 0; ix < n_species; ++ix) {
      species[ix] = new f64[grid.size()];
    }
    tmp = new f64[grid.size()];
    #pragma omp parallel
    {
      n_threads = omp_get_num_threads();
    }
    sub = new f64[n_threads * linear_ext];
    dia = new f64[n_threads * linear_ext];
    sup = new f64[n_threads * linear_ext];
    rhs = new f64[n_threads * linear_ext];
  }

  state_type() = delete;
  state_type(state_type const&) = delete;
  state_type(state_type&&) = default;
  
  ~state_type() {
    delete[] tmp;
    for (auto &cur : species) {
      delete[] cur;
      cur = nullptr;
    }
  }

  constexpr void swap(i32 ix) { std::swap(species[ix], tmp); }
};

template <u64 N>
inline void write_slices_csv(state_type<N> const &state, f64 t) {
    auto const &grid = state.grid;
    for (i32 ion = 0; ion < state.n_species; ++ion) {
        auto const u = state.species[ion];
        {
            {
                auto fd = std::ofstream("species-" + std::to_string(ion) + "-xy-" +
                                        std::to_string(t) + ".csv");
                for (i32 ix = 0; ix < grid.dim.x; ++ix) {
                    for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                        fd << u[ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + N / 2];
                        if (iy < grid.dim.y - 1) fd << ',';
                    }
                    fd << '\n';
                }
            }
            {
                auto fd = std::ofstream("species-" + std::to_string(ion) + "-xz-" +
                                        std::to_string(t) + ".csv");
                for (i32 ix = 0; ix < grid.dim.x; ++ix) {
                    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                        fd << u[ix * grid.dim.y * grid.dim.z + N / 2 * grid.dim.z + iz];
                        if (iz < grid.dim.z - 1) fd << ',';
                    }
                    fd << '\n';
                }
            }
            {
                auto fd = std::ofstream("species-" + std::to_string(ion) + "-yz-" +
                                        std::to_string(t) + ".csv");
                for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                        fd << u[N / 2 * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz];
                        if (iz < grid.dim.z - 1) fd << ',';
                    }
                    fd << '\n';
                }
            }
        }
    }
}
