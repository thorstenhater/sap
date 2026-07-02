#pragma once

#include "common.hpp"
#include "state.hpp"
#include "adi.hpp"
#include "ros.hpp"

template<u64 N> inline void
run(state_type<N>& state, f64 t_cur, f64 dt) {
  ros_step(state, t_cur, 0.5*dt);
  adi_step(state, t_cur, dt);
  ros_step(state, t_cur, 0.5*dt);
}
