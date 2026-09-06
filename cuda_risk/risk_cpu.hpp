//
// Nexus-LOB :: risk engine (subsystem 3) — CPU reference (header-only)
// Owner: Person A.
//
// Serial Monte-Carlo VaR/CVaR. This is the *oracle* the CUDA kernel must match
// bit-for-bit (same splitmix64 draws, same path recursion — see risk_common.hpp),
// and the fallback used when no GPU is present.
//
// Compiles with plain C++ (no CUDA). Callable from Python via the pybind
// binding (bindings/pybind_wrapper.cpp) and diff-tested against a NumPy
// re-implementation in python_quant/tests/test_risk_parity.py.
#pragma once

#include <cstdint>
#include <vector>

#include "risk_common.hpp"

namespace nexus::risk {

// Serial VaR/CVaR over `p.n_paths` paths. Deterministic given `p.seed`.
inline RiskResult compute_var_cvar_cpu(const RiskParams& p) {
    std::vector<double> losses;
    losses.reserve(static_cast<std::size_t>(p.n_paths));
    for (std::int64_t i = 0; i < p.n_paths; ++i) {
        losses.push_back(terminal_loss(p, static_cast<std::uint64_t>(i)));
    }
    return quantile_from_losses(losses, p.alpha, p.n_paths);
}

}  // namespace nexus::risk