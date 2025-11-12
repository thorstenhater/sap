#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "common.hpp"

enum class bc_kind {
    dirichlet,
    neumann,
};

struct bc_type {
    bc_kind kind = bc_kind::neumann;
    f64 value = NAN;
};

constexpr bool use_shm = false;
const char* shm_key = "diffusion";

constexpr auto neumann_bc() { return bc_type { .kind=bc_kind::neumann, .value=NAN }; }
constexpr auto dirichlet_bc(f64 v) { return bc_type { .kind=bc_kind::dirichlet, .value=v }; }

struct grid_type {
    v3<bc_type> bnd = { neumann_bc(), neumann_bc(), neumann_bc()};
    i32x3 dim = {0, 0, 0};
    f64x3 low = {0.0, 0.0, 0.0};
    f64x3 ext = {0.0, 0.0, 0.0};
    f64x3 del = {ext.x/dim.x, ext.y/dim.y, ext.z/dim.z};
};

constexpr auto i2x(grid_type const& grid, u64 ix) { return grid.low.x + ix*grid.del.x; }
constexpr auto i2y(grid_type const& grid, u64 iy) { return grid.low.y + iy*grid.del.y; }
constexpr auto i2z(grid_type const& grid, u64 iz) { return grid.low.z + iz*grid.del.z; }

constexpr auto x2i(grid_type const& grid, f64 x) { return i64((x - grid.low.x)/grid.del.x); }
constexpr auto y2i(grid_type const& grid, f64 y) { return i64((y - grid.low.y)/grid.del.y); }
constexpr auto z2i(grid_type const& grid, f64 z) { return i64((z - grid.low.z)/grid.del.z); }


auto alloc(grid_type const& grid) {
    auto N = grid.dim.x*grid.dim.y*grid.dim.z;
    if constexpr (use_shm) {
        shm_unlink(shm_key);
        auto fd = shm_open(shm_key, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            perror("SHM");
            exit(-1);
        }

        if (ftruncate(fd, N) == -1) {
            perror("TRNC");
            exit(-1);
        }        
        
        auto mem = (char*) mmap((void*)NULL, N, PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED) {
            perror("MMAP");
            exit(-1);
        }        
    }
    return new f64[N];
}

auto alloc_line(grid_type const& grid) { return new f64[std::max({grid.dim.x, grid.dim.y, grid.dim.z})]; }

