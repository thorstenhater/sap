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

using namespace arb::units::literals;

/// Create a recipe with a single cell that depends on a dynamic Calcium
/// reversal potential
struct recipe: public arb::recipe {

  recipe() {
    gprop.default_parameters = arb::neuron_parameter_defaults;
    gprop.catalogue.extend(arb::load_catalogue("cat/the-catalogue.so"), "cat:");
  }

  arb::cell_size_type num_cells() const override { return 5; }
  arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
    return arb::cell_kind::cable;
  }
  std::any get_global_properties(arb::cell_kind) const override {
    return gprop;
  }
  arb::util::unique_any
  get_cell_description(arb::cell_gid_type gid) const override {
    arb::segment_tree tree;
    auto par = arb::mnpos;
    par = tree.append(par, {0, -2, 0, 5}, {0, 2, 0, 5}, 1);
    auto decor = arb::decor{}
                     .paint(arb::reg::all(),
                            arb::membrane_capacitance(0.1_F / arb::units::m2))
                     .paint(arb::reg::tagged(1), arb::density("hh"))
                     .paint(arb::reg::tagged(1),
                            arb::density("cat:read/x=na",
                                         {{"px", 50.0 - 20.0 + 10.0 * gid},
                                          {"py", 50.0},
                                          {"pz", 50.0}}))
                     .set_default(arb::ion_reversal_potential_method(
                         "na", "nernst/x=na"))
                     .place(arb::ls::location(0, 0.5),
                            arb::i_clamp::box(0.01_s, 0.8_s, 0.1_nA), "ic0");
    return arb::cable_cell({tree}, decor, {}, arb::cv_policy_max_extent(10_um));
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

auto print_cvs(arb::morphology const &mrf, arb::isometry const &iso = {}) {
  auto const cvp = arb::cv_policy_max_extent(10_um);
  auto const cell = arb::cable_cell(mrf, {}, {}, cvp);
  auto const cvs = arb::cv_data(cell);
  auto const pwl = arb::place_pwlin(mrf, iso);
  std::cout << "Cables\n";
  f64 lx = std::numeric_limits<f64>::max(),
      hx = std::numeric_limits<f64>::lowest();
  f64 ly = std::numeric_limits<f64>::max(),
      hy = std::numeric_limits<f64>::lowest();
  f64 lz = std::numeric_limits<f64>::max(),
      hz = std::numeric_limits<f64>::lowest();
  for (int cv = 0; cv < cvs->size(); ++cv) {
    for (auto const &cable : cvs->cables(cv)) {
      if (cable.dist_pos == cable.prox_pos)
        continue;
      auto const point = pwl.at(
          arb::mlocation{.branch = cable.branch,
                         .pos = 0.5 * (cable.prox_pos + cable.dist_pos)});
      std::cout << " = " << cv << " (cable " << cable.branch << ' '
                << cable.prox_pos << ' ' << cable.dist_pos << ')' << " (point "
                << point.x << ' ' << point.y << ' ' << point.z << ')' << '\n';
      lx = std::min(lx, point.x), hx = std::max(hx, point.x);
      ly = std::min(ly, point.y), hy = std::max(hy, point.y);
      lz = std::min(lz, point.z), hz = std::max(hz, point.z);
    }
  }
  std::cout << "x = [" << lx << " -- " << hx << "]\n"
            << "y = [" << ly << " -- " << hy << "]\n"
            << "z = [" << lz << " -- " << hz << "]\n";
}

struct sampler {
  void operator()(arb::probe_metadata pm, std::size_t n,
                  const arb::sample_record *samples) {
    std::ofstream out(fn, std::ios_base::app);
    auto ptr = arb::util::any_cast<const arb::mcable_list *>(pm.meta);
    assert(ptr);
    auto n_cable = ptr->size();
    for (std::size_t i = 0; i < n; ++i) {
      const auto &[val, _ig] =
        *arb::util::any_cast<const arb::cable_sample_range *>(samples[i].data);
      out << samples[i].time;
      for (unsigned j = 0; j < n_cable; ++j) {
        arb::mcable loc = (*ptr)[j];
        out << ',' << val[j];
      }
      out << '\n';
    }
  }

  std::string fn;
};

struct test_reaction: public reaction_type<2> {
  Vector<2> rhs(Vector<2> const &c, f64) const override {
    return Vector<2>{{-1000.0 * c[0], -1.0 * c[1]}};
  }
  Matrix<2> jac(Vector<2> const &, f64) const override {
    return Matrix<2>{{-1000.0, 0.0}, {0.0, -1.0}};
  }
  test_reaction() = default;
};

struct neuro_reaction: public reaction_type<4> {
  Vector<4> rhs(const Vector<4>& c, f64) const override {
    // state vector
    f64 ko = c[0], nao = c[1], glu = c[2], k_buff = c[3];
    // Glutamate uptake and degradation (units: µM for Glu)
    f64 const glu_take = Vmax_Glu_up * glu / (Km_Glu_up + glu);
    f64 const glu_deg = Vmax_Glu_deg * glu / (Km_Glu_deg + glu);
    // Na clearance (saturable)
    f64 const na_clear = Vmax_Na * nao / (Km_Na + nao);
    // K pump removal
    f64 const k_pump = Vmax_pumpK * ko / (Km_pumpK + ko);    
    // K uptake by astrocytes (Michaelis-Menten)
    f64 const k_take = Vmax_K * ko / (Km_K + ko);    
    // Buffer recycling (slow)
    f64 const k_rec = k_rec * k_buff;

    return Vector<4>{
      {
        - k_take - k_pump + k_rec,
        - na_clear,
        - glu_take - glu_deg,
          k_take - k_rec,
      }
    };
  }

  Matrix<4> jac(const Vector<4>& c, f64) const override {
    // common derivatives for Michaelis-Menten form
    // 
    // d    V x          V K
    // -- ------- = ------------
    // dx  K + x      (K + x)^2
    auto mm_deriv = [](auto const& V, auto const& K, auto const& x) {
      return V * K / ((K + x)*(K + x));
    };
    // state vector
    f64 ko = c[0], nao = c[1], glu = c[2], k_buff = c[3];
    // derivatives
    f64 const& dk_take_dko    = mm_deriv(Vmax_K, Km_K, ko);
    f64 const& dk_pump_dko    = mm_deriv(Vmax_pumpK, Km_pumpK, ko);
    f64 const& dna_clear_dnao = mm_deriv(Vmax_Na, Km_Na, nao);
    f64 const& dglu_take_dglu = mm_deriv(Vmax_Glu_up, Km_Glu_up, glu);
    f64 const& dglu_deg_dglu  = mm_deriv(Vmax_Glu_deg, Km_Glu_deg, glu);

    return Matrix<4> {
      {-dk_take_dko - dk_pump_dko,              0.0,                              0.0,  k_rec},
      { 0.0,                        -dna_clear_dnao,                              0.0,    0.0},
      { 0.0,                                    0.0, - dglu_take_dglu - dglu_deg_dglu,    0.0},
      {dk_take_dko,                             0.0,                              0.0, -k_rec},
    };
}
  
  /// Paraemters
  // K uptake (astrocyte)
  f64 Vmax_K       = 1.0;    // mM/s
  f64 Km_K         =   2.0;  // mM

  // K pump removal
  f64 Vmax_pumpK   = 0.5;    // mM/s
  f64 Km_pumpK     =   1.0;  // mM

  // Na clearance
  f64 Vmax_Na      = 0.5;    // mM/s
  f64 Km_Na        =   10.0; // mM

  // Glutamate uptake/degradation (units µM)
  f64 Vmax_Glu_up  = 2.0;    // µM/s
  f64 Km_Glu_up    = 5.0;    // µM
  f64 Vmax_Glu_deg = 0.5;    // µM/s
  f64 Km_Glu_deg   = 10.0;   // µM

  // buffer recycling
  f64 k_rec        = 0.01;   // s^-1

  f64 A_K          = 10.0;   // mM/s pulse amplitude (very brief)
  f64 A_Na         = 0.0;    // mM/s (often small compared to K)
  f64 A_Glu        = 1e4;    // µM/s pulse amplitude (large to produce µM-scale spike)
  f64 t_pulse      = 0.001;  // pulse duration in seconds  
};

constexpr bool use_shm = true;

int main(int argc, char **argv) {
  f64 const dt = 0.25; // ms
  f64 const T = 200.0; // ms
  i32 const N = 128;   // dx ~ 4um
  f64 const L = 100.0; // um

  auto state = state_type<4>(
      grid_type{
          .bnd = {neumann_bc(), neumann_bc(), neumann_bc()},
          .dim = {N, N, N},
          .ext = {L, L, L},
          .del = {L / N, L / N, L / N},
      },
      // std::make_unique<test_reaction>(),
      // std::array<std::unique_ptr<diffusion_function_3d>, 2>{
        // std::make_unique<const_diffusion>(0.25),
        // std::make_unique<const_diffusion>(0.25),
      // }
      std::make_unique<neuro_reaction>(),
      {
        std::make_unique<const_diffusion>(7.7e-10), // K
        std::make_unique<const_diffusion>(5.2e-10), // Na
        std::make_unique<const_diffusion>(3.0e-10), // Glu
        std::make_unique<non_diffusive>(),
      }
    );

  if constexpr (use_shm) {
    delete[] state.species[0];
    state.species[0] = set_shm("shmem-diffusion", state.grid.size());
  }

  // make_gaussian(grid, state);
  make_step(state.grid, state.species[0], 140.0, 10, 100.0);
  make_const(state.grid, state.species[1], 10.0);
  // make_const(grid, state, 140);

  auto rec = recipe{};
  auto sim = arb::simulation{rec};
  auto sched = arb::regular_schedule(1_ms);
  sim.add_sampler(arb::one_probe({0, "Um"}),  sched, sampler{.fn = "Um0.csv"});
  sim.add_sampler(arb::one_probe({1, "Um"}),  sched, sampler{.fn = "Um1.csv"});
  sim.add_sampler(arb::one_probe({2, "Um"}),  sched, sampler{.fn = "Um2.csv"});
  sim.add_sampler(arb::one_probe({3, "Um"}),  sched, sampler{.fn = "Um3.csv"});
  sim.add_sampler(arb::one_probe({4, "Um"}),  sched, sampler{.fn = "Um4.csv"});
  sim.add_sampler(arb::one_probe({0, "nao"}), sched, sampler{.fn = "nao0.csv"});
  sim.add_sampler(arb::one_probe({1, "nao"}), sched, sampler{.fn = "nao1.csv"});
  sim.add_sampler(arb::one_probe({2, "nao"}), sched, sampler{.fn = "nao2.csv"});
  sim.add_sampler(arb::one_probe({3, "nao"}), sched, sampler{.fn = "nao3.csv"});
  sim.add_sampler(arb::one_probe({4, "nao"}), sched, sampler{.fn = "nao4.csv"});

  f64 t_out = 0.0;
  for (f64 t = 0.0; t < T; t += dt) {
    std::cerr << t << '\n';
    if (t >= t_out) {
      write_slices_csv(state, t);
      t_out += 10 * dt;
    }
    run(state, t, dt);
    sim.run((t + dt) * arb::units::ms, 5_us);
  }
}
