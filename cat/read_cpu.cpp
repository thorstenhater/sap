#include <algorithm>
#include <cstddef>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>
#include <arbor/simd/simd.hpp>
#undef NDEBUG
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>


namespace arb {
namespace the_catalogue {
namespace kernel_read {
        
namespace S = ::arb::simd;
using S::index_constraint;
using S::simd_cast;
using S::indirect;
using S::assign;
using simd_value = S::simd<arb_value_type, S::simd_abi::native_width<double>::value, S::simd_abi::native>;
using simd_index = S::simd<arb_index_type, S::simd_abi::native_width<double>::value, S::simd_abi::native>;
using simd_mask  = S::simd_mask<arb_value_type, S::simd_abi::native_width<double>::value, S::simd_abi::native>;
static constexpr unsigned simd_width_ = S::simd_abi::native_width<double>::value;
static constexpr unsigned min_align_ = std::max(S::min_align(simd_value{}), S::min_align(simd_index{}));

inline simd_value safeinv(simd_value x) {
    simd_value ones = simd_cast<simd_value>(1.0);
    auto mask = S::cmp_eq(S::add(x,ones), ones);
    S::where(mask, x) = simd_cast<simd_value>(DBL_EPSILON);
    return S::div(ones, x);
}

inline simd_value log(const simd_value& v) { return S::log(v); }
inline simd_value log(arb_value_type v) { return S::log(S::simd_cast<simd_value>(v)); }

#define PPACK_IFACE_BLOCK \
[[maybe_unused]] auto _pp_var_width                                                 = pp->width;\
[[maybe_unused]] auto _pp_var_n_detectors                                           = pp->n_detectors;\
[[maybe_unused]] auto _pp_var_dt                                                    = pp->dt;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_vec_ci                       = pp->vec_ci;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_v                        = pp->vec_v;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_i                        = pp->vec_i;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_g                        = pp->vec_g;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_temperature_degC             = pp->temperature_degC;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_diam_um                      = pp->diam_um;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_area_um2                     = pp->area_um2;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_time_since_spike             = pp->time_since_spike;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_node_index                   = pp->node_index;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_peer_index                   = pp->peer_index;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_multiplicity                 = pp->multiplicity;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_weight                       = pp->weight;\
[[maybe_unused]] auto& _pp_var_events                                               = pp->events;\
[[maybe_unused]] auto _pp_var_mechanism_id                                          = pp->mechanism_id;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_contiguous               = pp->index_constraints.n_contiguous;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_constant                 = pp->index_constraints.n_constant;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_independent              = pp->index_constraints.n_independent;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_none                     = pp->index_constraints.n_none;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_contiguous  = pp->index_constraints.contiguous;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_constant    = pp->index_constraints.constant;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_independent = pp->index_constraints.independent;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_none        = pp->index_constraints.none;\
[[maybe_unused]] auto const * const * _pp_var_random_numbers = pp->random_numbers;\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_xo = pp->state_vars[0];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_t = pp->state_vars[1];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_px = pp->parameters[0];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_py = pp->parameters[1];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_pz = pp->parameters[2];\
[[maybe_unused]] auto& _pp_var_ion_x = pp->ion_states[0];\
[[maybe_unused]] auto* __restrict__ _pp_var_ion_x_index = pp->ion_states[0].index;\
//End of IFACEBLOCK

static double * mem = nullptr;
static int fd = -1;

constexpr int N = 64;
constexpr double L = 100.0;
constexpr double dx = L/N;
    
// interface methods
static void init(arb_mechanism_ppack* pp) {
    if (nullptr == mem) { 
        auto fn = "shmem-diffusion";
        fd = shm_open(fn, O_RDONLY, 0600);
        if (fd < 0) {
            perror("SHM");
            exit(-42);
        }
        
        mem = (double*) mmap(nullptr, N*N*N*sizeof(double), PROT_READ, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED) {
            perror("MMAP");
            exit(-42);
        }
    }

    PPACK_IFACE_BLOCK;

    assert(simd_width_ <= (unsigned)S::width(simd_cast<simd_value>(0)));
    for (arb_size_type i_ = 0; i_ < _pp_var_width; i_ += simd_width_) {
        indirect(_pp_var_xo+i_, simd_width_) = simd_cast<simd_value>((double)0.0);
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 2; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    assert(simd_width_ <= (unsigned)S::width(simd_cast<simd_value>(0)));
    
    for (arb_size_type i_ = 0; i_ < _pp_var_width; i_ += simd_width_) {
        simd_value val = simd_cast<simd_value>(2.0);
        for (arb_size_type j_ = 0; j_ < simd_width_; ++j_) {            
            auto px = _pp_var_px[j_ + i_];
            auto idx = int(px/dx);
            auto py = _pp_var_py[j_ + i_];
            auto idy = int(py/dx);
            auto pz = _pp_var_pz[j_ + i_];
            auto idz = int(pz/dx);
        
            auto off = idx*N*N + idy*N + idz;
            val[j_] = mem[off];
        }
        indirect(_pp_var_xo + i_, simd_width_) = val;
    }
    // exit(-23);
}

static void write_ions(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    assert(simd_width_ <= (unsigned)S::width(simd_cast<simd_value>(0)));
    for (auto i_ = 0ul; i_ < _pp_var_index_constraints_n_contiguous; i_ += 2) {
        for (auto index_ = _pp_var_index_constraints_contiguous[i_]; index_ < _pp_var_index_constraints_contiguous[i_+1]; index_ += simd_width_) {
            simd_value w_;
            assign(w_, indirect((_pp_var_weight+index_), simd_width_));
            auto ion_x_indexi_ = _pp_var_ion_x_index[index_];
            simd_value xo_shadowed_ = simd_cast<simd_value>(0);
            assign(xo_shadowed_, indirect(_pp_var_xo+index_, simd_width_));
            simd_value t_xo_shadowed_;
            assign(t_xo_shadowed_, indirect(_pp_var_ion_x.external_concentration + ion_x_indexi_, simd_width_));
            t_xo_shadowed_ = S::fma(simd_cast<simd_value>(1.0), xo_shadowed_, t_xo_shadowed_);
            indirect(_pp_var_ion_x.external_concentration + ion_x_indexi_, simd_width_) = t_xo_shadowed_;
        }
    }
    for (auto i_ = 0ul; i_ < _pp_var_index_constraints_n_independent; i_++) {
        arb_index_type index_ = _pp_var_index_constraints_independent[i_];
        simd_value w_;
        assign(w_, indirect((_pp_var_weight+index_), simd_width_));
        auto ion_x_indexi_ = simd_cast<simd_index>(indirect(&_pp_var_ion_x_index[0] + index_, simd_width_));
        simd_value xo_shadowed_ = simd_cast<simd_value>(0);
        assign(xo_shadowed_, indirect(_pp_var_xo+index_, simd_width_));
        indirect(_pp_var_ion_x.external_concentration, ion_x_indexi_, simd_width_, index_constraint::independent) += S::mul(simd_cast<simd_value>(1.0), xo_shadowed_);
    }
    for (auto i_ = 0ul; i_ < _pp_var_index_constraints_n_none; i_++) {
        arb_index_type index_ = _pp_var_index_constraints_none[i_];
        simd_value w_;
        assign(w_, indirect((_pp_var_weight+index_), simd_width_));
        auto ion_x_indexi_ = simd_cast<simd_index>(indirect(&_pp_var_ion_x_index[0] + index_, simd_width_));
        simd_value xo_shadowed_ = simd_cast<simd_value>(0);
        assign(xo_shadowed_, indirect(_pp_var_xo+index_, simd_width_));
        indirect(_pp_var_ion_x.external_concentration, ion_x_indexi_, simd_width_, index_constraint::none) += S::mul(simd_cast<simd_value>(1.0), xo_shadowed_);
    }
    for (auto i_ = 0ul; i_ < _pp_var_index_constraints_n_constant; i_++) {
        arb_index_type index_ = _pp_var_index_constraints_constant[i_];
        simd_value w_;
        assign(w_, indirect((_pp_var_weight+index_), simd_width_));
        auto ion_x_indexi_ = _pp_var_ion_x_index[index_];
        simd_value xo_shadowed_ = simd_cast<simd_value>(0);
        assign(xo_shadowed_, indirect(_pp_var_xo+index_, simd_width_));
        indirect(_pp_var_ion_x.external_concentration, simd_cast<simd_index>(ion_x_indexi_), simd_width_, index_constraint::constant) += S::mul(simd_cast<simd_value>(1.0), xo_shadowed_);
    }
}
    
static void apply_events(arb_mechanism_ppack*, arb_deliverable_event_stream*) {}

static void post_event(arb_mechanism_ppack*) {}
#undef PPACK_IFACE_BLOCK
} // namespace kernel_read
} // namespace the_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_the_catalogue_read_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::the_catalogue::kernel_read::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::the_catalogue::kernel_read::min_align_;
    result.init_mechanism = arb::the_catalogue::kernel_read::init;
    result.compute_currents = arb::the_catalogue::kernel_read::compute_currents;
    result.apply_events = arb::the_catalogue::kernel_read::apply_events;
    result.advance_state = arb::the_catalogue::kernel_read::advance_state;
    result.write_ions = arb::the_catalogue::kernel_read::write_ions;
    result.post_event = arb::the_catalogue::kernel_read::post_event;
    return &result;
  }
}

