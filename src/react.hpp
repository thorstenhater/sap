#pragma once

#include "common.hpp" 

/// reaction system
/// 
/// dC                \dR
/// -- = rhs(c, t) +  ---
/// dt                \dt
///
/// with the Jacobian
///
/// d rhs
/// -----
/// d C
///
/// with
/// 
/// C:    1xN 
/// jac:  (C, t) -> NxN
/// rhs:  (C, t) -> 1xN
/// dRdt: (C, t) -> 1xN

template <u64 N>
struct reaction_type {
  using Vec = Vector<N>;
  using Mat = Matrix<N>;  
  constexpr static i32 size = N;
  virtual Vec rhs(Vec const&, f64)  const = 0;
  virtual Mat jac(Vec const&, f64)  const = 0;
  virtual Vec dRdt(Vec const&, f64) const { return Vec::Zero(); };
  virtual bool is_conc_dependent() const { return true; }    
  virtual ~reaction_type() {}
    
};

template <u64 N>
using reaction_ptr = std::unique_ptr<reaction_type<N>>;
