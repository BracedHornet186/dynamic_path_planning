#pragma once

#include <curand_kernel.h>
#include <mppi/cost_functions/cost.cuh>
#include "dynamic_path_planning/pedestrian_gaussian.hpp"

template <class DYN_T, int SIM_TIME_HORIZON>
struct RiskAwareCostParams : public CostParams<DYN_T::CONTROL_DIM>
{
  static constexpr int TIME_HORIZON = SIM_TIME_HORIZON;

  // Tracking
  float goal[DYN_T::OUTPUT_DIM * TIME_HORIZON] = {0};
  float tracking_weights[DYN_T::OUTPUT_DIM] = {1};

  // Speed / rotation
  float v_ref = 0.0f;
  float w_ref = 0.0f;
  float w_speed = 0.0f;
  float w_rotation = 0.0f;

  // Risk
  PedestrianGaussian* pedestrians = nullptr;
  int num_peds = 12;

  float delta = 0.5f;
  int num_mc_samples = 256;

  float w_soft = 0.0f;
  float w_hard = 0.0f;`
  float sigma = 0.05f;

  unsigned long seed = 0;
};

template <class CLASS_T, class DYN_T, class PARAMS_T>
class RiskAwareCostImpl
  : public Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>
{
public:
  using PARENT = Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>;

  __device__ float computeRunningCost(
    float* s,
    float* u,
    int timestep,
    float* theta_c,
    int* /*crash_status*/)
  {
    float cost = 0.0f;

    // ---------------- Tracking ----------------
    float* goal = this->params_.goal + timestep * DYN_T::OUTPUT_DIM;
    for (int i = 0; i < DYN_T::OUTPUT_DIM; i++)
    {
      float e = s[i] - goal[i];
      cost += this->params_.tracking_weights[i] * e * e;
    }

    // ---------------- Speed ----------------
    float dv = u[0] - this->params_.v_ref;
    cost += this->params_.w_speed * dv * dv;

    // ---------------- Rotation ----------------
    float dw = u[1] - this->params_.w_ref;
    cost += this->params_.w_rotation * dw * dw;

    // ---------------- Risk (FUSED MC) ----------------
    float P = 0.0f;

    if (this->params_.pedestrians && this->params_.num_peds > 0)
    {
      int rollout_id = static_cast<int>(theta_c[0]);

      curandState rng;
      curand_init(
        this->params_.seed,
        rollout_id * PARAMS_T::TIME_HORIZON + timestep,
        0,
        &rng
      );

      float rx = s[0];
      float ry = s[1];

      float acc = 0.0f;

      for (int i = 0; i < this->params_.num_mc_samples; ++i)
      {
        float r = this->params_.delta * sqrtf(curand_uniform(&rng));
        float th = 2.0f * M_PI * curand_uniform(&rng);

        float x = rx + r * cosf(th);
        float y = ry + r * sinf(th);

        float prod = 1.0f;

        for (int o = 0; o < this->params_.num_peds; ++o)
        {
          const auto& p = this->params_.pedestrians[o];

          float dx = x - p.mu_x;
          float dy = y - p.mu_y;

          float e =
            -0.5f * (
              dx * (p.inv_cov_xx * dx + p.inv_cov_xy * dy) +
              dy * (p.inv_cov_xy * dx + p.inv_cov_yy * dy)
            );

          float p_o = p.norm_factor * expf(e);
          prod *= (1.0f - p_o);
        }

        acc += (1.0f - prod);
      }

      P = acc / this->params_.num_mc_samples;
    }

    cost += this->params_.w_soft * P;
    if (P > this->params_.sigma)
      cost += this->params_.w_hard;

    return cost;
  }

  __device__ float terminalCost(float*, float*) { return 0.0f; }
};

template <class DYN_T, int SIM_TIME_HORIZON>
class RiskAwareCost
  : public RiskAwareCostImpl<RiskAwareCost<DYN_T, SIM_TIME_HORIZON>, DYN_T,
                             RiskAwareCostParams<DYN_T, SIM_TIME_HORIZON>>
{
public:
  RiskAwareCost() = default;
}; // DPP_COST_FUNCTIONS_RISK_AWARE_COST_CUH_