#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#include "dynamic_path_planning/mc_collision_kernel.cuh"

__global__ void mc_collision_kernel(
    float robot_x,
    float robot_y,
    float delta,
    const PedestrianGaussian* peds,
    int num_peds,
    float* out_joint_cp,
    int num_samples,
    unsigned long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float r = delta * sqrtf(curand_uniform(&state));
    float theta = 2.0f * M_PI * curand_uniform(&state);

    float x = robot_x + r * cosf(theta);
    float y = robot_y + r * sinf(theta);

    float prod = 1.0f;

    for (int o = 0; o < num_peds; ++o) {
        const auto& p = peds[o];

        float dx = x - p.mu_x;
        float dy = y - p.mu_y;

        float exponent =
            -0.5f * (
                dx * (p.inv_cov_xx * dx + p.inv_cov_xy * dy) +
                dy * (p.inv_cov_xy * dx + p.inv_cov_yy * dy)
            );

        float p_o = p.norm_factor * expf(exponent);
        prod *= (1.0f - p_o);
    }

    out_joint_cp[idx] = 1.0f - prod;
}

#include <chrono>

void launch_mc_collision_kernel(
    float robot_x,
    float robot_y,
    float delta,
    const PedestrianGaussian* d_peds,
    int num_peds,
    float* d_joint_cp,
    int num_samples
) {
    int threads = 256;
    int blocks = (num_samples + threads - 1) / threads;
    unsigned long seed =
        static_cast<unsigned long>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count()
        );

    mc_collision_kernel<<<blocks, threads>>>(
        robot_x,
        robot_y,
        delta,
        d_peds,
        num_peds,
        d_joint_cp,
        num_samples,
        seed
    );
}