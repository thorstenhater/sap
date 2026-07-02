#pragma once

#include "common.hpp"

struct diffusion_function_3d {
    virtual f64 x(f64 x, f64 y, f64 z) = 0;
    virtual f64 y(f64 x, f64 y, f64 z) = 0;
    virtual f64 z(f64 x, f64 y, f64 z) = 0;
    virtual ~diffusion_function_3d() {}
};

struct const_diffusion: public diffusion_function_3d {
    virtual f64 x(f64 x, f64 y, f64 z) { return val; };
    virtual f64 y(f64 x, f64 y, f64 z) { return val; };
    virtual f64 z(f64 x, f64 y, f64 z) { return val; };    

    const_diffusion(f64 v): val(v) {}
    
    f64 val = NAN;
};

struct non_diffusive: diffusion_function_3d {
  virtual f64 x(f64 x, f64 y, f64 z) { return 0.0; };
  virtual f64 y(f64 x, f64 y, f64 z) { return 0.0; };
  virtual f64 z(f64 x, f64 y, f64 z) { return 0.0; };    

  non_diffusive() {}
};

using diffusion_ptr = std::unique_ptr<diffusion_function_3d>;
