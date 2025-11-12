#include <mdspan>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cassert>

#include "common.hpp"
#include "grid.hpp"

constexpr auto to_domain(i32x3 const& dim, f64 const * noalias const data) { return std::mdspan(data, dim.x, dim.y, dim.z); }
constexpr auto to_domain(i32x3 const& dim, f64 * noalias const data) { return std::mdspan(data, dim.x, dim.y, dim.z); }

using matrix = std::mdspan<f64, std::dextents<std::size_t, 3>>;

template<typename T>
concept diffusion_function = requires(T fn) { { fn(0.0, 0.0, 0.0) } -> std::convertible_to<f64>; };

template<typename T>
concept source_function = requires(T fn) { { fn(0.0, 0.0, 0.0, 0.0) } -> std::convertible_to<f64>; };

struct constant {
    constant(f64 v): val{v} {}
    auto operator()(f64, f64, f64) const { return val; }
private:
    f64 val;
};

template <diffusion_function X, diffusion_function Y, diffusion_function Z>
struct diffusion_function_3d {
    X x;
    Y y;
    Z z;

    diffusion_function_3d(f64 d):
        x{constant(d)}, y{constant(d)}, z{constant(d)}
    {}

    diffusion_function_3d(const f64x3 ds):
        x{constant(ds.x)}, y{constant(ds.y)}, z(constant(ds.z))
    {}

    diffusion_function_3d(f64 dx, f64 dy, f64 dz):
        x{constant(dx)}, y{constant(dy)}, z(constant(dz))
    {}

    template<diffusion_function D>
    diffusion_function_3d(D d): x{d}, y{d}, z{d}
    {}

    template<diffusion_function DX, diffusion_function DY, diffusion_function DZ>
    diffusion_function_3d(DX dx, DY dy, DZ dz): x{dx}, y{dy}, z{dz}
    {}    
};

diffusion_function_3d(f64 x) -> diffusion_function_3d<constant, constant, constant>;
diffusion_function_3d(f64 x, f64 y, f64 z) -> diffusion_function_3d<constant, constant, constant>;
template<diffusion_function D>
diffusion_function_3d(D d) -> diffusion_function_3d<D, D, D>;

inline constexpr void
tridia_solve(i32 len,
             f64 const * noalias const sub,
             f64       * noalias const dia,
             f64 const * noalias const sup,
             f64       * noalias const rhs) {
    assert(len >= 2);
    // forward
    for (i32 ix = 1; ix < len; ++ix) {
        auto const tmp  = sub[ix] / dia[ix - 1];
        dia[ix] -= tmp * sup[ix - 1];
        rhs[ix] -= tmp * rhs[ix - 1];
    }
    // root
    rhs[len - 1] /= dia[len - 1];
    // backward
    for (i32 ix = len - 2; ix >= 0; --ix) {
        rhs[ix] = (rhs[ix] - sup[ix] * rhs[ix + 1]) / dia[ix];
    }
}

// Compute Laplacians of field u per-direction
// x-direction w/ zero flux bc
inline constexpr auto
lap_x(const grid_type& grid,
      i32 ix, i32 iy, i32 iz,
      matrix const u) {
    auto const nx = grid.dim.x;
    auto u_xc = NAN;
    if (grid.bnd.x.kind == bc_kind::neumann) {
      u_xc = u[ix, iy, iz];
    }
    else if (grid.bnd.x.kind == bc_kind::dirichlet) {
      u_xc = grid.bnd.x.value;
    }                
    auto const u_xm = (ix ==    0) ? u_xc : u[ix-1, iy, iz];
    auto const u_xp = (ix == nx-1) ? u_xc : u[ix+1, iy, iz];
    return u_xm - 2.0*u_xc + u_xp;
}

// y-direction w/ zero flux bc
inline constexpr auto
lap_y(const grid_type& grid,
      i32 ix, i32 iy, i32 iz,
      matrix const u) {
    auto const ny = grid.dim.y;
    auto u_yc = NAN;
    if (grid.bnd.y.kind == bc_kind::neumann) {
      u_yc = u[ix, iy, iz];
    }
    else if (grid.bnd.y.kind == bc_kind::dirichlet) {
      u_yc = grid.bnd.y.value;
    }                
    auto const u_ym = (iy ==    0) ? u_yc : u[ix, iy-1, iz]; 
    auto const u_yp = (iy == ny-1) ? u_yc : u[ix, iy+1, iz];
    return u_ym - 2.0*u_yc + u_yp;
}

// z-direction w/ zero flux bc
inline constexpr auto
lap_z(const grid_type& grid,
      i32 ix, i32 iy, i32 iz,
      matrix const u) {
    auto const nz = grid.dim.z;
    auto u_zc = NAN;
    if (grid.bnd.z.kind == bc_kind::neumann) {
      u_zc = u[ix, iy, iz];
    }
    else if (grid.bnd.z.kind == bc_kind::dirichlet) {
      u_zc = grid.bnd.z.value;
    }            
    auto const u_zm = (iz == 0)    ? u_zc : u[ix, iy, iz-1]; 
    auto const u_zp = (iz == nz-1) ? u_zc : u[ix, iy, iz+1];
    return u_zm - 2.0*u_zc + u_zp;
}

struct state_type {
    f64 * cur = nullptr; // current concentration field
    f64 * nxt = nullptr; // swap/working concentration field
    f64 * sub = nullptr; // subdiagonal
    f64 * dia = nullptr; // diagonal
    f64 * sup = nullptr; // superdiagonal
    f64 * rhs = nullptr; // right hand side

    state_type(grid_type const& grid):
        cur{alloc(grid)},
        nxt{alloc(grid)},
        sub{alloc_line(grid)},
        dia{alloc_line(grid)},
        sup{alloc_line(grid)},
        rhs{alloc_line(grid)}
    {}

    ~state_type() {
        delete [] rhs;
        delete [] sup;
        delete [] dia;
        delete [] sub;
        delete [] nxt;
        delete [] cur;
    }

    constexpr void swap() { std::swap(cur, nxt); }
};

constexpr void
make_gaussian(grid_type const& grid,
              state_type& state) {
    auto const cx = 0.5, cy = 0.5, cz = 0.5;
    auto const sigma = 0.04;
    auto const width = 1/(2.0*sigma*sigma);
    
    auto domain = to_domain(grid.dim, state.cur);
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                auto const x = i2x(grid, ix) - cx;
                auto const y = i2y(grid, iy) - cy;
                auto const z = i2z(grid, iz) - cz;
                auto const d = x*x + y*y + z*z;
                auto const u = std::exp(-d*width);
                domain[ix, iy, iz] = u; 
            }
        }
    }
}



template <diffusion_function DX, diffusion_function DY, diffusion_function DZ, source_function S>
constexpr void
adi_step(grid_type const& grid,
         state_type& state,
         f64 dt, f64 t,
         diffusion_function_3d<DX, DY, DZ> dif_fn,
         S src_fn) {

    auto const& [dx, dy, dz] = grid.del; 
    
    auto const i_dx2 = 0.5*dt/(dx*dx);
    auto const i_dy2 = 0.5*dt/(dy*dy);
    auto const i_dz2 = 0.5*dt/(dz*dz);

    auto u_cur = to_domain(grid.dim, state.cur);
    auto u_nxt = to_domain(grid.dim, state.nxt);
    auto sub = state.sub;
    auto dia = state.dia;
    auto sup = state.sup;
    auto rhs = state.rhs;    
    
    // X-implicit, explicit in y, z.
    for (i32 iy = 0; iy < grid.dim.y; ++iy) {
        for (i32 iz = 0; iz < grid.dim.z; ++iz) {
            for (i32 ix = 0; ix < grid.dim.x; ++ix) {
                auto const x = i2x(grid, ix), y = i2y(grid, iy), z = i2z(grid, iz);
                auto const src = src_fn(x, y, z, t);
                auto const yyu = lap_y(grid, ix, iy, iz, u_cur) * i_dy2;
                auto const zzu = lap_z(grid, ix, iy, iz, u_cur) * i_dz2;
                rhs[ix] = u_cur[ix, iy, iz] + dif_fn.y(x, y, z) * yyu + dif_fn.z(x, y, z) * zzu + 0.5 * dt * src;
                
                auto const dif = dif_fn.x(x, y, z);
                auto const d_xm = (ix == 0)            ? dif : dif_fn.x(x-dx, y, z);
                auto const d_xp = (ix == grid.dim.x-1) ? dif : dif_fn.x(x+dx, y, z);
                sub[ix] = -d_xm * i_dx2;
                dia[ix] = 1.0 + (d_xm + d_xp)*i_dx2;
                sup[ix] = -d_xp * i_dx2;
            }
            //  solve line
            tridia_solve(grid.dim.x, sub, dia, sup, rhs);
            // write line into cube: cur -> nxt
            for (i32 ix = 0; ix < grid.dim.x; ++ix) u_nxt[ix, iy, iz] = rhs[ix];            
        }   
    }

    // Y-implicit, explicit in x,z.
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        for (i32 iz = 0; iz < grid.dim.z; ++iz) {
            for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                auto const x = i2x(grid, ix), y = i2y(grid, iy), z = i2z(grid, iz);
                auto const src = src_fn(x, y, z, t);
                auto const xxu = lap_x(grid, ix, iy, iz, u_nxt)*i_dx2;
                auto const zzu = lap_z(grid, ix, iy, iz, u_nxt)*i_dz2;
                rhs[iy] = u_nxt[ix, iy, iz] + dif_fn.x(x, y, z)*xxu + dif_fn.z(x, y, z)*zzu + 0.5*dt*src;
                // NOTE: This is always Neumann BC... 
                auto const dif = dif_fn.y(x, y, z);
                auto const d_ym = (iy == 0)            ? dif : dif_fn.y(x, y-dy, z);
                auto const d_yp = (iy == grid.dim.y-1) ? dif : dif_fn.y(x, y+dy, z);
                sub[iy] = -d_ym * i_dy2;
                dia[iy] = 1.0 + (d_ym + d_yp) * i_dy2;
                sup[iy] = -d_yp * i_dy2;
            }
            //  solve line
            tridia_solve(grid.dim.y, sub, dia, sup, rhs);
            // write line into cube: nxt -> cur
            for (i32 iy = 0; iy < grid.dim.y; ++iy) u_cur[ix, iy, iz] = rhs[iy];            
        }
    }

    // Z-implicit, explicit in x,y.
    for (i32 ix = 0; ix < grid.dim.x; ++ix) {
        for (i32 iy = 0; iy < grid.dim.y; ++iy) {
            for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                auto const x = i2x(grid, ix), y = i2y(grid, iy), z = i2z(grid, iz);
                auto const src = src_fn(x, y, z, t);
                auto const xxu = lap_x(grid, ix, iy, iz, u_cur)*i_dx2;
                auto const yyu = lap_y(grid, ix, iy, iz, u_cur)*i_dy2;
                rhs[iz] = u_cur[ix, iy, iz] + dif_fn.x(x, y, z) * xxu + dif_fn.y(x, y, z) * yyu + 0.5*dt*src;

                auto const dif = dif_fn.z(x, y, z);
                auto const d_zm = (iz == 0)            ? dif : dif_fn.z(x, y, z-dz);
                auto const d_zp = (iz == grid.dim.z-1) ? dif : dif_fn.z(x, y, z+dz);
                sub[iz] = -d_zm * i_dz2;
                dia[iz] = 1.0 + (d_zm + d_zp) * i_dz2;
                sup[iz] = -d_zp * i_dz2;
            }
            //  solve line
            tridia_solve(grid.dim.z, sub, dia, sup, rhs);
            // write line into cube: cur -> nxt
            for (i32 iz = 0; iz < grid.dim.z; ++iz) u_nxt[ix, iy, iz] = rhs[iz];            
        }
    }
}

int main() {
    f64 const dt =  2.5;    // ms
    f64 const T  = 1000.0;    // ms
    i32 const N  =  256;      // dx ~ 4um
    f64 const L  =  100.0;    // um
    auto const grid = grid_type {
        .dim = {N, N, N},
        .ext = {L, L, L},
    };

    auto state = state_type(grid);
    
    f64 t_out = 0.0;

    auto dif = diffusion_function_3d(0.75); // 0.7–0.8 µm²/ms
    auto src = [](f64, f64, f64, f64) { return 0.0; };
    
    make_gaussian(grid, state);
    for (f64 t = 0.0; t < T; t += dt) {
        std::cerr << "time=" << t << '\n';
        if (t >= t_out) {
            auto u_cur = to_domain(grid.dim, state.cur);
            {
                auto fd = std::ofstream("out-xy-" + std::to_string(t) + ".csv");
                for (i32 ix = 0; ix < grid.dim.x; ++ix) {
                    for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                        fd << u_cur[ix, iy, 0];
                        if (iy < grid.dim.y-1) fd << ',';
                    }
                    fd << '\n';
                }
            }
            {
                auto fd = std::ofstream("out-xz-" + std::to_string(t) + ".csv");
                for (i32 ix = 0; ix < grid.dim.x; ++ix) {
                    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                        fd << u_cur[ix, 0, iz];
                        if (iz < grid.dim.z-1) fd << ',';
                    }
                    fd << '\n';
                }
            }
            {
                auto fd = std::ofstream("out-yz-" + std::to_string(t) + ".csv");
                for (i32 iy = 0; iy < grid.dim.y; ++iy) {
                    for (i32 iz = 0; iz < grid.dim.z; ++iz) {
                        fd << u_cur[0, iy, iz];
                        if (iz < grid.dim.z-1) fd << ',';
                    }
                    fd << '\n';
                }
            }            
            t_out += 10*dt;
        }
        adi_step(grid, state, dt, t, dif, src);
        state.swap();
    }
}
