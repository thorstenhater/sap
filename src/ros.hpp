#pragma once

#include "common.hpp"
#include "grid.hpp"
#include "react.hpp"
#include "state.hpp"

#include <cmath>

constexpr static f64 gamma = 1.0 - 1.0 / sqrt(2.0);

template <u64 N> inline void
ros_step_conc_dependent(f64** species,
                        reaction_type<N> const& reaction,
                        grid_type const& grid,
                        f64 const t,
                        f64 const dt) {
    using M = Matrix<N>;
    using V = Vector<N>;
    #pragma omp parallel
    {
        V c;
        #pragma omp for collapse(3)
        for (i32 ix = 0; ix < grid.dim.x; ++ix) {
            for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                    auto const off = ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz;
                    // Read concentrations
                    for (i32 ion = 0; ion < N; ++ion) c[ion] = species[ion][off];
                    // prepare solver
                    auto const dec = (M::Identity() - gamma * dt * reaction.jac(c, t))
                                    .partialPivLu();
                    // do the two ROS2 sub-steps
                    auto const d0 = gamma * dt * dt * reaction.dRdt(c, t);
                    V const k1 = dec.solve(dt * reaction.rhs(c, t) + d0);
                    V const k2 = dec.solve(dt * reaction.rhs(c + k1, t + dt) - d0 - 2.0 * k1);
                    // Write back solution
                    for (i32 ion = 0; ion < N; ++ion) species[ion][off] += 1.5 * k1[ion] + 0.5 * k2[ion];
                }
            }
        }
    } // omp parallel
}

template <u64 N> inline void
ros_step_conc_independent(f64** species,
                          reaction_type<N> const& reaction,
                          grid_type const& grid,
                          f64 const t,
                          f64 const dt) {
    using M = Matrix<N>;
    using V = Vector<N>;
    #pragma omp parallel
    {
        V c;
        auto jac      = reaction.jac(V::Zero(), t);
        auto const dec = (M::Identity() - gamma * dt * jac).partialPivLu();        
        #pragma omp for collapse(3)
        for (i32 ix = 0; ix < grid.dim.x; ++ix) {
            for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                    auto const off = ix * grid.dim.y * grid.dim.z + iy * grid.dim.z + iz;
                    // Read concentrations
                    for (i32 ion = 0; ion < N; ++ion) c[ion] = species[ion][off];
                    // do the two ROS2 sub-steps
                    V const d0 = gamma * dt * dt * reaction.dRdt(c, t);
                    V const k1 = dec.solve(dt * reaction.rhs(c, t) + d0);
                    V const k2 = dec.solve(dt * reaction.rhs(c + k1, t + dt) - d0 - 2.0 * k1);
                    // Write back solution
                    for (i32 ion = 0; ion < N; ++ion) species[ion][off] += 1.5 * k1[ion] + 0.5 * k2[ion];
                }
            }
        }
    } // omp parallel
}


template <u64 N> inline void
ros_step(state_type<N> &state, f64 const t, f64 const dt) {
    if (state.reaction->is_conc_dependent()) {
        ros_step_conc_dependent(state.species.data(), *state.reaction, state.grid, t, dt);
    }
    else {
        ros_step_conc_independent(state.species.data(), *state.reaction, state.grid, t, dt);
    }
}
