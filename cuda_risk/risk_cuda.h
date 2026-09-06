//
// Nexus-LOB :: risk engine (subsystem 3) — GPU host-facing API
// Owner: Person A.
//
// Declares the CUDA launcher. Only included when a CUDA toolkit is available
// (risk_bench.cpp guards on NEXUS_HAS_CUDA). The implementation lives in
// risk_cuda.cu and is inert on machines without nvcc — the CPU reference
// (risk_cpu.hpp) is the always-available path.
#pragma once

#include "risk_common.hpp"

namespace nexus::risk {

// Parallel VaR/CVaR. `gpu_ms` (optional) receives the pure device-kernel wall
// time (path simulation only; the quantile is reduced on the host, which is
// microseconds vs. the simulation's milliseconds). Same math + RNG as the CPU
// reference, so results are bit-for-bit identical for the same seed.
RiskResult compute_var_cvar_gpu(const RiskParams& p, double* gpu_ms = nullptr);

}  // namespace nexus::risk