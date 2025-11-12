#include <arbor/cable_cell.hpp>
#include <arbor/units.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arborio/neurolucida.hpp>

#include "common.hpp"
#include "grid.hpp"

#include <filesystem>
#include <iostream>

using namespace arb::units::literals;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "mrf <morphology.asc>\n";
        return -42;
    }
    f64 const dt = 0.001;
    f64 const T  = 0.1;
    i32 const N = 1024;
    f64 const L = 2000.0;
    auto const grid = grid_type {
        .dim = {N, N, N},
        .ext = {L, L, L},
    };
    
    auto const fn = std::filesystem::path{argv[1]};
    auto const asc = arborio::load_asc(fn);
    auto const iso = arb::isometry();
    auto const mrf = asc.morphology;
    auto const cvp = arb::cv_policy_max_extent(10_um);
    auto const cell = arb::cable_cell(mrf, {}, {}, cvp);
    auto const cvs = arb::cv_data(cell);
    auto const pwl = arb::place_pwlin(mrf, iso);
    std::cout << "Cables\n";
    f64 lx = std::numeric_limits<f64>::max(), hx = std::numeric_limits<f64>::lowest();
    f64 ly = std::numeric_limits<f64>::max(), hy = std::numeric_limits<f64>::lowest();
    f64 lz = std::numeric_limits<f64>::max(), hz = std::numeric_limits<f64>::lowest();    
    for (int cv = 0; cv < cvs->size(); ++cv) {
        for (auto const& cable: cvs->cables(cv)) {
            if (cable.dist_pos == cable.prox_pos) continue;
            auto const point = pwl.at(arb::mlocation{ .branch=cable.branch, .pos=0.5*(cable.prox_pos + cable.dist_pos)}); 
            std::cout << " = " << cv
                      << " (cable " << cable.branch << ' ' << cable.prox_pos << ' ' << cable.dist_pos << ')'
                      << " (point " << point.x      << ' ' << point.y         << ' ' << point.z << ')'
                      << '\n';
            lx = std::min(lx, point.x), hx = std::max(hx, point.x);
            ly = std::min(ly, point.y), hy = std::max(hy, point.y);
            lz = std::min(lz, point.z), hz = std::max(hz, point.z);
        }
    }
    std::cout << "x = [" << lx << " -- " << hx << "]\n"
              << "y = [" << ly << " -- " << hy << "]\n"
              << "z = [" << lz << " -- " << hz << "]\n";
}

 
