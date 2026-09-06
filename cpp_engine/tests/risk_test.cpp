//
// Nexus-LOB :: risk engine (subsystem 3) — CPU reference statistical tests
// Owner: Person A.
//
// Validates the CPU Monte-Carlo VaR/CVaR reference against its mathematical
// expectations. GPU parity is a separate concern: the CUDA kernel shares the
// exact same RNG + path recursion, so cross-checking is via risk_bench.cpp
// (exact bit-for-bit on a CUDA machine). These run on plain C++/CTest.
//
//   g++ -std=c++20 -O2 -I cuda_risk cpp_engine/tests/risk_test.cpp -o risk_test
#include <cmath>
#include <cstdio>

#include "risk_cpu.hpp"

using nexus::risk::RiskParams;
using nexus::risk::RiskResult;
using nexus::risk::compute_var_cvar_cpu;

namespace {
int g_checks = 0;
int g_failed = 0;

void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
}  // namespace

int main() {
    // 1. Determinism: same seed -> identical result.
    {
        RiskParams p;
        RiskResult a = compute_var_cvar_cpu(p);
        RiskResult b = compute_var_cvar_cpu(p);
        check(a.var == b.var && a.cvar == b.cvar && a.mean_loss == b.mean_loss,
              "deterministic under a fixed seed");
    }

    // 2. No volatility, no drift -> every path stays at S0 -> zero loss.
    {
        RiskParams p;
        p.sigma = 0.0;
        p.mu    = 0.0;
        RiskResult r = compute_var_cvar_cpu(p);
        // Transcendental exp/log rounding leaves ~1e-16, so allow a tiny tol.
        check(close(r.var, 0.0, 1e-9) && close(r.cvar, 0.0, 1e-9) &&
              close(r.mean_loss, 0.0, 1e-9),
              "sigma=mu=0 gives ~zero VaR/CVaR");
    }

    // 3. Mean loss converges to the closed-form 1 - exp(mu*T) (GBM: E[S_T]=S0 e^{muT}).
    {
        RiskParams p;
        p.n_paths = 400000;
        const double expected = 1.0 - std::exp(p.mu * p.T);  // mu=0.05,T=1 -> ~-0.0513
        RiskResult r = compute_var_cvar_cpu(p);
        // MC mean error ~ sigma/sqrt(N) = 0.25/632 ~ 0.0004; allow 3x.
        check(close(r.mean_loss, expected, 0.002), "mean loss matches 1-exp(mu*T)");
    }

    // 4. Quantile sanity: VaR increases with alpha; CVaR >= VaR.
    {
        RiskParams p;
        RiskParams p95 = p; p95.alpha = 0.95;
        RiskParams p99 = p; p99.alpha = 0.99;
        RiskResult r95 = compute_var_cvar_cpu(p95);
        RiskResult r99 = compute_var_cvar_cpu(p99);
        check(r99.var > r95.var, "VaR_0.99 > VaR_0.95");
        check(r95.var > 0.0, "VaR_0.95 positive under vol");
        check(r99.cvar >= r99.var, "CVaR >= VaR (tail mean is beyond the threshold)");
        check(r99.cvar > r95.cvar, "CVaR increases with confidence");
    }

    // 5. Jump diffusion fattens the tail: same sigma, added jumps raise CVaR.
    //    Use a clearly-fattening config (~20% of paths carry a ~30% jump) so the
    //    effect is far beyond Monte-Carlo noise.
    {
        RiskParams base;
        RiskParams jump = base;
        jump.lambda_jump = 0.20;   // expected 0.20 jumps per path
        jump.jump_mu     = 0.0;
        jump.jump_sigma  = 0.30;   // log jump size ~30%
        RiskResult b = compute_var_cvar_cpu(base);
        RiskResult j = compute_var_cvar_cpu(jump);
        check(j.cvar > b.cvar * 1.10, "jumps increase CVaR (fatter tail)");
    }

    // 6. Reported path count is honored.
    {
        RiskParams p;
        p.n_paths = 12345;
        check(compute_var_cvar_cpu(p).n == 12345.0, "n_paths reported");
    }

    std::printf("risk_test: %d checks, %d failed\n", g_checks, g_failed);
    std::printf(g_failed ? "FAIL\n" : "ALL PASS\n");
    return g_failed ? 1 : 0;
}