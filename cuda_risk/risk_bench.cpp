//
// Nexus-LOB :: risk engine (subsystem 3) — CPU vs GPU benchmark + parity
// Owner: Person A.
//
// Times the Monte-Carlo VaR/CVaR reference and, when built with a CUDA
// toolkit, the GPU kernel — and checks the two agree bit-for-bit (they share
// the splitmix64 RNG + path recursion). The GPU timing is the kernel wall
// clock only (path simulation); the quantile is reduced on the host.
//
//   cmake build:  ./risk_bench [--paths N] [--steps S] [--alpha A]
//                 [--sigma S] [--mu M] [--seed X]
//   (also standalone without CMake: g++ -std=c++20 -O3 -I cuda_risk -I cpp_engine/include
//    cuda_risk/risk_bench.cpp -o risk_bench  — CPU-only when nvcc is absent)
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "risk_cpu.hpp"
#ifdef NEXUS_HAS_CUDA
#include "risk_cuda.h"
#endif

using namespace nexus::risk;

namespace {
double now_ms() {
    using namespace std::chrono;
    return std::chrono::duration<double, std::milli>(
               steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

int main(int argc, char** argv) {
    RiskParams p;
    for (int i = 1; i < argc; ++i) {
        auto val = [&] { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--paths"))      p.n_paths      = std::atoi(val());
        else if (!std::strcmp(argv[i], "--steps")) p.steps        = std::atoi(val());
        else if (!std::strcmp(argv[i], "--alpha")) p.alpha        = std::atof(val());
        else if (!std::strcmp(argv[i], "--sigma")) p.sigma        = std::atof(val());
        else if (!std::strcmp(argv[i], "--mu"))    p.mu           = std::atof(val());
        else if (!std::strcmp(argv[i], "--seed"))  p.seed         = std::strtoull(val(), nullptr, 0);
    }

    std::printf("risk_bench: %d paths x %d steps, alpha=%.2f, sigma=%.2f, seed=%llu\n",
                p.n_paths, p.steps, p.alpha, p.sigma,
                static_cast<unsigned long long>(p.seed));

    const double t0 = now_ms();
    const RiskResult cpu = compute_var_cvar_cpu(p);
    const double cpu_ms = now_ms() - t0;
    std::printf("CPU : var=%.6f  cvar=%.6f  mean_loss=%.6f   %7.1f ms\n",
                cpu.var, cpu.cvar, cpu.mean_loss, cpu_ms);

#ifdef NEXUS_HAS_CUDA
    double gpu_ms = 0.0;
    const RiskResult gpu = compute_var_cvar_gpu(p, &gpu_ms);
    std::printf("GPU : var=%.6f  cvar=%.6f  mean_loss=%.6f   %7.2f ms\n",
                gpu.var, gpu.cvar, gpu.mean_loss, gpu_ms);
    const double dvar = std::fabs(cpu.var - gpu.var);
    const double dcvar = std::fabs(cpu.cvar - gpu.cvar);
    std::printf("parity: |var diff|=%.3e  |cvar diff|=%.3e  (bit-for-bit -> 0)\n",
                dvar, dcvar);
    std::printf("speedup (CPU/GPU kernel): %.1fx\n", cpu_ms / gpu_ms);
#else
    std::printf("GPU : not compiled (no CUDA toolkit) — speedup measured on a CUDA box\n");
#endif
    return 0;
}