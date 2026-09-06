//
// Nexus-LOB :: risk engine (subsystem 3) — shared model + deterministic RNG
// Owner: Person A.
//
// Monte-Carlo VaR / CVaR over a GBM (+ optional Merton jump-diffusion) price
// process. The whole engine is built around ONE reproducible design decision:
//
//   * The per-path random draws are a PURE function of (seed, path, step) via
//     splitmix64 — a counter-based generator with no mutable state.
//
// That makes the CPU reference (risk_cpu.hpp), the CUDA kernel (risk_cuda.cu)
// and the NumPy oracle (python_quant/tests/test_risk_parity.py) compute the
// *identical* losses for a given seed, so parity is EXACT (not Monte-Carlo
// noise) and the GPU/CPU speedup is measured on identical work.
//
// This header is compiled three ways:
//   * plain C++ (MSVC / g++ / clang) for the CPU reference + tests + pybind,
//   * by nvcc for the device kernel (the __host__ __device__ path),
//   * mirrored in Python (test_risk_parity.py) for the cross-language oracle.
//
// VaR/CVaR are reported as FRACTIONS of the initial notional (loss = 1 - S_T/S0,
// so a positive number is a loss). Prices stay plain doubles here: unlike the
// order book, risk analytics are statistical aggregates, not bit-exact LOB state.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#ifdef __CUDACC__
#define NEXUS_RISK_HOSTDEV __host__ __device__
#else
#define NEXUS_RISK_HOSTDEV
#endif

namespace nexus::risk {

// ---------------------------------------------------------------------------
// Model / parameters
// ---------------------------------------------------------------------------
struct RiskParams {
    double s0        = 100.0;   // initial price
    double mu        = 0.05;    // annualized drift
    double sigma     = 0.25;    // annualized volatility
    double T         = 1.0;     // horizon in years
    int    steps     = 252;     // steps per path (trading days / year)
    int    n_paths   = 200000;  // number of Monte-Carlo paths
    double alpha     = 0.95;    // VaR/CVaR confidence
    double lambda_jump = 0.0;   // per-step jump intensity (0 = pure GBM)
    double jump_mu     = 0.0;   // log jump-size mean
    double jump_sigma  = 0.0;   // log jump-size volatility
    std::uint64_t seed = 0x51ED;
};

struct RiskResult {
    double var       = 0.0;  // VaR_alpha (fractional loss, >0 = bad)
    double cvar      = 0.0;  // CVaR_alpha (mean of losses beyond VaR)
    double mean_loss = 0.0;  // mean fractional loss across all paths
    double n         = 0.0;  // paths used (for reporting)
};

// ---------------------------------------------------------------------------
// Deterministic counter-based RNG (host + device)
// ---------------------------------------------------------------------------
// splitmix64 is a bijective scramble of its input: no state, fully parallel.
NEXUS_RISK_HOSTDEV inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Uniform in [0, 1). The salts keep the normal pair and the jump draw on
// disjoint counter lanes, so no two draws share a splitmix input.
NEXUS_RISK_HOSTDEV inline double rng_uniform(std::uint64_t seed,
                                             std::uint64_t path,
                                             std::uint64_t lane) {
    const std::uint64_t x =
        seed ^ (path * 0x9E3779B97F4A7C15ULL) ^ (lane * 0xBF58476D1CE4E5B9ULL);
    return static_cast<double>(splitmix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

// Box–Muller from two uniforms (cos form), u1 clamped off 0 to keep log finite.
NEXUS_RISK_HOSTDEV inline double box_muller(double u1, double u2) {
    const double a = u1 > 1e-300 ? u1 : 1e-300;
    return sqrt(-2.0 * log(a)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// ---------------------------------------------------------------------------
// Per-path terminal fractional loss. (seed, path) -> one double. Same on CPU/GPU.
// ---------------------------------------------------------------------------
NEXUS_RISK_HOSTDEV inline double terminal_loss(const RiskParams& p,
                                               std::uint64_t path) {
    const double dt     = p.T / static_cast<double>(p.steps);
    const double drift  = (p.mu - 0.5 * p.sigma * p.sigma) * dt;
    const double vol_sd = p.sigma * sqrt(dt);
    const double lambda_dt = p.lambda_jump * dt;
    double logS = log(p.s0);
    for (int t = 0; t < p.steps; ++t) {
        const std::uint64_t tl = static_cast<std::uint64_t>(t);
        double inc = drift + vol_sd * box_muller(
            rng_uniform(p.seed, path, tl * 2u),
            rng_uniform(p.seed, path, tl * 2u + 1u));
        if (p.lambda_jump > 0.0 &&
            rng_uniform(p.seed, path, tl * 2u + 1'000'000u) < lambda_dt) {
            inc += p.jump_mu + p.jump_sigma * box_muller(
                rng_uniform(p.seed, path, tl * 2u + 2'000'000u),
                rng_uniform(p.seed, path, tl * 2u + 3'000'000u));
        }
        logS += inc;
    }
    const double st = exp(logS);
    return 1.0 - st / p.s0;
}

// ---------------------------------------------------------------------------
// Quantile: VaR_alpha = (1-alpha) tail threshold; CVaR_alpha = tail mean.
// Shared by the CPU reference and the GPU host side so they always agree.
// (Host-only: sorts a std::vector in place.)
// ---------------------------------------------------------------------------
inline RiskResult quantile_from_losses(
    std::vector<double>& losses,  // mutated by std::sort
    double alpha,
    std::int64_t n_paths) {
    std::sort(losses.begin(), losses.end());
    std::int64_t k = static_cast<std::int64_t>(std::ceil((1.0 - alpha) * n_paths));
    if (k < 1) k = 1;
    if (k > n_paths) k = n_paths;
    const std::int64_t idx = n_paths - k;
    double tail_sum = 0.0, sum = 0.0;
    for (std::int64_t i = 0; i < n_paths; ++i) {
        sum += losses[static_cast<std::size_t>(i)];
        if (i >= idx) tail_sum += losses[static_cast<std::size_t>(i)];
    }
    RiskResult r;
    r.var       = losses[static_cast<std::size_t>(idx)];
    r.cvar      = tail_sum / static_cast<double>(k);
    r.mean_loss = sum / static_cast<double>(n_paths);
    r.n         = static_cast<double>(n_paths);
    return r;
}

}  // namespace nexus::risk