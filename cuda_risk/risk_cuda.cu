//
// Nexus-LOB :: risk engine (subsystem 3) — CUDA kernel + launcher
// Owner: Person A.
//
// One CUDA thread simulates one Monte-Carlo path (see terminal_loss in
// risk_common.hpp). The counter-based splitmix64 RNG means every thread writes
// its own path's terminal loss with NO shared state and NO atomics — the
// embarrassingly-parallel core that gives the ~40x-over-CPU claim its
// structure. The (1-alpha)-tail quantile / CVaR is then reduced on the host:
// sorting 200k doubles is microseconds next to simulating 50M path-steps.
//
// Requires a CUDA toolkit (nvcc). Not part of the non-CUDA build; CMake only
// compiles this into `nexus_risk` when CUDAToolkit is found.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "risk_common.hpp"
#include "risk_cuda.h"

namespace nexus::risk {

namespace {

__global__ void simulate_kernel(const RiskParams p, double* __restrict__ losses) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.n_paths) {
        losses[i] = terminal_loss(p, static_cast<std::uint64_t>(i));
    }
}

// Minimal error reporting — we stay off the C++ exceptions in this TU so the
// build needs no extra runtime handling.
[[noreturn]] void die(const char* what, cudaError_t e) {
    std::fprintf(stderr, "nexus_risk: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

}  // namespace

RiskResult compute_var_cvar_gpu(const RiskParams& p, double* gpu_ms) {
    double* d_losses = nullptr;
    if (cudaMalloc(&d_losses, static_cast<std::size_t>(p.n_paths) * sizeof(double))
        != cudaSuccess) {
        die("cudaMalloc", cudaGetLastError());
    }

    const int threads = 256;
    const int blocks  = (p.n_paths + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    simulate_kernel<<<blocks, threads>>>(p, d_losses);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    if (gpu_ms) *gpu_ms = static_cast<double>(ms);

    std::vector<double> losses(static_cast<std::size_t>(p.n_paths));
    if (cudaMemcpy(losses.data(), d_losses,
                   static_cast<std::size_t>(p.n_paths) * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_losses);
        die("cudaMemcpy", cudaGetLastError());
    }
    cudaFree(d_losses);

    return quantile_from_losses(losses, p.alpha, p.n_paths);
}

}  // namespace nexus::risk