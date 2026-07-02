#include <arbor/cable_cell.hpp>
#include <arbor/morph/cv_data.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>
#include <arbor/units.hpp>
#include <arborio/neurolucida.hpp>

#include "common.hpp"
#include "diff.hpp"
#include "grid.hpp"
#include "init.hpp"
#include "integrate.hpp"
#include "react.hpp"
#include "shm.hpp"
#include "state.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <numbers>
#include <random>

constexpr const char* shm_key = "shmem-diffusion";
constexpr bool use_shm = true;

using namespace arb::units::literals;

arb::segment_tree make_branchy_tree(i32);

/// Create a recipe with a single cell that depends on a dynamic Calcium
/// reversal potential
struct recipe : public arb::recipe {
    recipe() {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.catalogue.extend(arb::load_catalogue("cat/the-catalogue.so"), "cat:");
    }
    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind) const override { return gprop; }
    arb::util::unique_any
    get_cell_description(arb::cell_gid_type gid) const override {
        auto tree = make_branchy_tree(gid);
        auto decor = arb::decor{}
                     .paint(arb::reg::all(),
                            arb::init_membrane_potential(-40.0_mV))
                     .paint(arb::reg::tagged(1), arb::density("hh"))
                     .paint(arb::reg::tagged(3), arb::density("pas"))
                     .set_default(arb::ion_reversal_potential_method("na", "nernst/x=na"))
                     .place(arb::ls::location(0, 0.5),
                            arb::i_clamp::box(0.1_s, 0.8_s, 0.1_nA), "ic0")
                     .place(arb::ls::location(0, 0.5),
                            arb::i_clamp::box(1.1_s, 0.8_s, 0.1_nA), "ic0")
                     .place(arb::ls::location(0, 0.5),
                            arb::i_clamp::box(2.1_s, 0.8_s, 0.1_nA), "ic0")
        
        ;

        for (const auto& seg: tree.segments()) {
            auto loc = arb::ls::support(arb::ls::on_components(0.5,
                                                               arb::reg::segment(seg.id)));
            decor.place(loc,
                        arb::synapse("cat:read/x=na",
                                     {{"px", 0.5*(seg.prox.x + seg.dist.x)},
                                      {"py", 0.5*(seg.prox.y + seg.dist.y)},
                                      {"pz", 0.5*(seg.prox.z + seg.dist.z)}}),
                        "rd-" + std::to_string(seg.id));
        }


        return arb::cable_cell({tree}, decor, {}, arb::cv_policy_every_segment());
    }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override {
        return {
            {arb::cable_probe_ion_ext_concentration_cell{"na"}, "nao"},
            {arb::cable_probe_ion_int_concentration_cell{"na"}, "nai"},
            {arb::cable_probe_membrane_voltage_cell{}, "Um"},
        };
    }
    arb::cable_cell_global_properties gprop;
};

struct sampler {

    sampler(std::string fn_): fn(std::move(fn_)) {
        // truncate output
        std::ofstream(fn, std::ios_base::trunc);
    }

    void operator()(arb::probe_metadata pm, std::size_t n,
                    const arb::sample_record *samples) {
        auto out = std::ofstream(fn, std::ios_base::app);
        auto ptr = arb::util::any_cast<const arb::mcable_list *>(pm.meta);
        assert(ptr);
        auto n_cable = ptr->size();
        for (u64 i = 0; i < n; ++i) {
            const auto &[val, _ig] = *arb::util::any_cast<const arb::cable_sample_range *>(samples[i].data);
            out << samples[i].time;
            for (u64 j = 0; j < n_cable; ++j) {
                arb::mcable loc = (*ptr)[j];
                out << ',' << val[j];
            }
            out << '\n';
        }
    }

  std::string fn;
};

struct test_reaction : public reaction_type<2> {
    Vec rhs(Vec const &c, f64) const override { return Vec{{-1000.0 * c[0], -1.0 * c[1]}}; }
    Mat jac(Vec const &c, f64) const override { return Mat{{-1000.0, 0.0}, {0.0, -1.0}}; }
    test_reaction() = default;
};

struct calcium_binding: public reaction_type<2> {
    calcium_binding() = default;

    Vec rhs(Vec const &c, f64) const override final {
        // state vector (no de-structuring bind allowed :/)
        f64 ca = c[0], ca_bound = c[1];
        f64 free_buffer = buffer_max - ca_bound;
        // rates
        f64 r_bind   = k_on  * free_buffer * ca;
        f64 r_unbind = k_off * ca_bound;

        return Vec{{
            - r_bind + r_unbind,
            + r_bind - r_unbind
        }};
    }

    Mat jac(Vec const& c, f64) const override final {
        f64 ca = c[0], ca_bound = c[1];
        f64 free_buffer = buffer_max - ca_bound;
        return Mat {
            { - k_on * free_buffer,   k_on * ca + k_off },
            {   k_on * free_buffer, - k_on * ca - k_off }
        };
    }

    f64 constexpr static buffer_max =   0.1; // mM
    f64 constexpr static k_on       = 100.0; // mM/ms
    f64 constexpr static k_off      =   1.0; // 1/ms
};

auto make_ca_binding_problem(f64 L, i32 N) {
    auto state = state_type<2>(grid_type {
                                   .bnd = {neumann_bc(), neumann_bc(), neumann_bc()},
                                   .dim = {N, N, N},
                                   .ext = {L, L, L},
                                   .del = {L / N, L / N, L / N}
                               },
                               std::make_unique<calcium_binding>(),
                               {
                                   std::make_unique<const_diffusion>(7.7e-3), // Ca
                                   std::make_unique<non_diffusive>(),         // bound Ca
                               });
    make_const(state.grid, state.species[0], 0.0);
    make_step(state.grid, state.species[0], 20.0, 50.0, 1.0);
    make_const(state.grid, state.species[1], 0.0);
    return state;
}

struct calcium_buffer: public reaction_type<2> {
    calcium_buffer() = default;

    Vec rhs(Vec const &c, f64) const override final {
        // state vector (no de-structuring bind allowed :/)
        f64 ca = c[0], ca_bound = c[1];
        f64 free_buffer = buffer_max - ca_bound;
        // rates
        f64 r_bind   = k_on  * free_buffer * ca;
        f64 r_unbind = k_off * ca_bound;

        f64 pump = vmax_pump * ca / (k_pump + ca);

        return Vec{{
            - r_bind + r_unbind - pump + leak,
            + r_bind - r_unbind
        }};
    }

    Mat jac(Vec const& c, f64) const override final {
        f64 ca = c[0], ca_bound = c[1];
        f64 free_buffer = buffer_max - ca_bound;
        f64 dpump = vmax_pump * k_pump / ((k_pump + ca) * (k_pump + ca));
        return Mat {
            { - k_on * free_buffer - dpump,   k_on * ca + k_off },
            {   k_on * free_buffer,         - k_on * ca - k_off }
        };
    }

    f64 constexpr static buffer_max =   0.1;  // mM
    f64 constexpr static k_on       = 100.0;  // mM/ms
    f64 constexpr static k_off      =   1.0;  // 1/ms
    f64 constexpr static vmax_pump  =  10.0;  // mM/ms
    f64 constexpr static k_pump     =   0.3;  // mM
    f64 constexpr static ca_rest    =   1.5;  // mM
    f64 constexpr static leak = vmax_pump * ca_rest / (k_pump + ca_rest);
};

auto make_ca_buffer_problem(f64 L, i32 N) {
    auto state = state_type<2>(grid_type {
                                   .bnd = {neumann_bc(), neumann_bc(), neumann_bc()},
                                   .dim = {N, N, N},
                                   .ext = {L, L, L},
                                   .del = {L / N, L / N, L / N}
                               },
                               std::make_unique<calcium_binding>(),
                               {
                                   std::make_unique<const_diffusion>(7.7e-3), // Ca
                                   std::make_unique<non_diffusive>(),         // bound Ca
                               });
    make_step(state.grid, state.species[0], 5.0, 50.0, 0.0);
    make_const(state.grid, state.species[1], 0.0);

    return state;
}


struct glutamate_reaction: public reaction_type<4> {
    Vec rhs(Vec const &c, f64) const override final {
        // state vector (no de-structuring bind allowed :/)
        f64 ko = c[0], nao = c[1], glu = c[2], k_buff = c[3];
        // Glutamate uptake and degradation (units: µM for Glu)
        f64 glu_take = Vmax_Glu_up * glu / (Km_Glu_up + glu);
        f64 glu_deg  = Vmax_Glu_deg * glu / (Km_Glu_deg + glu);
        // Na clearance (saturable)
        f64 na_clear = Vmax_Na * nao / (Km_Na + nao);
        // K pump removal
        f64 k_pump = Vmax_pumpK * ko / (Km_pumpK + ko);
        // K uptake by astrocytes (Michaelis-Menten)
        f64 k_take = Vmax_K * ko / (Km_K + ko);
        // Buffer recycling (slow)
        f64 k_rec = k_rec * k_buff;

        return Vec{{
            -k_take - k_pump + k_rec,
            -na_clear,
            -glu_take - glu_deg,
            k_take - k_rec,
        }};
    }

    bool is_conc_dependent() const override final { return false; }

    Mat jac(Vec const &c, f64) const override final {
        // common derivatives for Michaelis-Menten form
        //
        // d    V x          V K
        // -- ------- = ------------
        // dx  K + x      (K + x)^2
        auto mm_deriv = [](auto const &V, auto const &K, auto const &x) {
            return V * K / ((K + x) * (K + x));
        };
        // state vector
        f64 ko = c[0], nao = c[1], glu = c[2], k_buff = c[3];
        // derivatives
        f64 dk_take_dko    = mm_deriv(Vmax_K, Km_K, ko);
        f64 dk_pump_dko    = mm_deriv(Vmax_pumpK, Km_pumpK, ko);
        f64 dna_clear_dnao = mm_deriv(Vmax_Na, Km_Na, nao);
        f64 dglu_take_dglu = mm_deriv(Vmax_Glu_up, Km_Glu_up, glu);
        f64 dglu_deg_dglu  = mm_deriv(Vmax_Glu_deg, Km_Glu_deg, glu);

        return Mat{
            {-dk_take_dko - dk_pump_dko, 0.0,             0.0,                             k_rec},
            {0.0,                        -dna_clear_dnao, 0.0,                             0.0},
            {0.0,                        0.0,             -dglu_take_dglu - dglu_deg_dglu, 0.0},
            {dk_take_dko, 0.0, 0.0, -k_rec},
        };
    }

    // Paraemters
    // K uptake (astrocyte)
    f64 Vmax_K       =     1.0e-3;    // mM/ms
    f64 Km_K         =     2.0;       // mM
    // K pump removal
    f64 Vmax_pumpK   =     0.5e-3;    // mM/ms
    f64 Km_pumpK     =     1.0;       // mM
    // Na clearance
    f64 Vmax_Na      =     0.5e-3;    // mM/ms
    f64 Km_Na        =    10.0;       // mM
    // Glutamate uptake/degradation (units µM -> mM)
    f64 Vmax_Glu_up  =     2.0e-6;    // mM/ms  2 µM/s = 2 1/1000 mM / 1000 ms = 2e-6 mM/ms
    f64 Km_Glu_up    =     5.0e-3;    // mM
    f64 Vmax_Glu_deg =     0.5e-6;    // mM/ms
    f64 Km_Glu_deg   =    10.0e-3;    // mM
    // buffer recycling
    f64 k_rec        =     0.01e-3;   // 1/ms
    // input
    f64 A_K          =    10.0e-3;    // mM/s pulse amplitude (very brief)
    f64 A_Na         =     0.0e-3;    // mM/s (often small compared to K)
    f64 A_Glu        = 10000.0e-6;    // µM/s pulse amplitude (large to produce µM-scale spike)
    f64 t_pulse      =     1.0;       // ms pulse duration
};

auto make_glutamate_problem(f64 L, i32 N) {
    auto state = state_type<4>(grid_type {
                                   .bnd = {neumann_bc(), neumann_bc(), neumann_bc()},
                                   .dim = {N, N, N},
                                   .ext = {L, L, L},
                                   .del = {L / N, L / N, L / N}
                               },
                               std::make_unique<glutamate_reaction>(),
                               {
                                   std::make_unique<const_diffusion>(7.7e-10), // K
                                   std::make_unique<const_diffusion>(5.2e-10), // Na
                                   std::make_unique<const_diffusion>(3.0e-10), // Glu
                                   std::make_unique<non_diffusive>(),
                               });

    make_step(state.grid, state.species[0], 140.0, 10, 100.0);
    make_const(state.grid, state.species[1], 10.0);

    return state;
}

int main(int argc, char **argv) {
    f64 dt = 0.25; // ms
    f64 T = 500.0; // ms
    i32 N = 64;    // dx ~ 4um
    f64 L = 100.0; // um

    auto state = make_ca_buffer_problem(L, N);

    // iff we export our species via SHMEM, facilitate that
    if constexpr (use_shm) {
        auto* tmp = state.species[0];
        state.species[0] = set_shm(shm_key, state.grid.size());
        std::memcpy(state.species[0], tmp, state.grid.size()*sizeof(f64));
        delete[] tmp;
    }

    
    auto rec = recipe{};
    auto sim = arb::simulation{rec};
    auto sched = arb::regular_schedule(1_ms);
    for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) {
        sim.add_sampler(arb::one_probe({gid, "Um"}),  sched, sampler{"Um"  + std::to_string(gid) + ".csv"});
        sim.add_sampler(arb::one_probe({gid, "nao"}), sched, sampler{"nao" + std::to_string(gid) + ".csv"});
    }

    f64 dx = L/N;
    for (const auto& [px, py, pz]: std::vector<std::tuple<f64, f64, f64>>{{0, 25.0, 26.2927}}) {
            auto idx = int(px/dx);
            auto idy = int(py/dx);
            auto idz = int(pz/dx);
        
            auto off = idx*N*N + idy*N + idz;
            std::cout << state.species[0][off] << "\n";
    }
    
    f64 t_out = 0.0;
    for (f64 t = 0.0; t < T; t += dt) {
        if (t >= t_out) {
            std::cerr << t << '\n';
            write_slices_csv(state, t);
            t_out += 10 * dt;
        }
        run(state, t, dt);
        sim.run((t + dt) * arb::units::ms, 5_us);
    }

    if constexpr (use_shm) {
        del_shm(state.species[0], shm_key, state.grid.size());
        state.species[0] = nullptr;
    }

    for (arb::cell_gid_type gid = 0; gid < rec.num_cells(); ++gid) {
        auto tree = make_branchy_tree(gid);
        auto out = std::ofstream("mrf"  + std::to_string(gid) + ".csv");
        for (const auto& seg: tree.segments()) {
            out << seg.prox.x << ',' << seg.prox.y << ',' << seg.prox.z << ',' << seg.prox.radius << ','
                << seg.dist.x << ',' << seg.dist.y << ',' << seg.dist.z << ',' << seg.dist.radius << ','
                << seg.id     << ',' << seg.tag    << '\n';
        }
    }
}

arb::segment_tree make_branchy_tree(i32 seed) {
    using std::numbers::pi;
    auto interp = [](auto range, i32 i, i32 n) {
        auto p = i * 1./(n - 1);
        auto [r0, r1] = range;
        return r0 + p*(r1 - r0);
    };

    f64 area_um = 42;
    f64 dl = sqrtf(0.5 * area_um / pi);

    // shift to somewhere near centre (um, um, um)
    f64 sx = 0.0;
    f64 sy = 25.0;
    f64 sz = 25.0;

    auto tree = arb::segment_tree();
    auto root = arb::mnpos;
    root = tree.append(root,
                       {0.0 + sx, 0.0 + sy, 0.0 + sz, dl}, {0.0 + sx, 0.0 + sy, dl + sz, dl},
                       1);

    i32 max_depth     = 5;
    auto branch_probs = std::pair{1.0, 0.25};
    auto lengths      = std::pair{20.0, 2.0};
    auto levels       = std::vector<std::vector<i32>>{{0}};
    auto dys          = std::unordered_map<i32, f64>{{0, 0.0}};

    auto gen = std::default_random_engine(seed);
    std::uniform_real_distribution dis(0.0, 1.0);

    //gen = default_rng(seed=gid);
    // Diameter of 1 μm for each dendrite cable.
    f64 drad = 0.05*dl;
    // Start dendrite at the edge of the soma.
    f64 dist_from_soma = dl;
    f64 width = 30;
    for (i32 ix = 0; ix < max_depth; ++ix) {
        // Branch prob at this level.
        f64 bp = interp(branch_probs, ix, max_depth);
        // Length at this level.
        f64 dl = interp(lengths, ix, max_depth);
        auto sec_ids = std::vector<i32>{};
        for (const auto sec: levels[ix]) {
            for(auto iy: {0, 1}) {
                if (dis(gen) >= bp) continue;
                f64 z = dist_from_soma;
                f64 dz = sqrt(dl*dl - 0.25*width*width);
                f64 dy = dys[sec];
                auto p = tree.append(sec, {0.0 + sx, dy + sy, z + dz + sz, drad}, 3);
                dys[p] = dys[sec];
                dys[sec] += width;
                sec_ids.push_back(p);
            }
        }
        width *= 0.5;
        if (sec_ids.empty()) break;
        levels.push_back(sec_ids);
        dist_from_soma += dl;
    }

    f64 wx = 0.5*dis(gen) - 0.5;
    f64 wy = 0.5*dis(gen) - 0.5;
    f64 wz = 0.5*dis(gen) - 0.5;
    auto iso = arb::isometry()
    // * arb::isometry::rotate(wx * pi, 1, 0, 0)
    // * arb::isometry::rotate(wy * pi, 0, 1, 0)
    // * arb::isometry::rotate(wz * pi, 0, 0, 1)
    ;
    return arb::apply(tree, iso);
}
