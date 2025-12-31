#pragma once
/*
 * Created on Thu Jan 1 2026 by Yash
 */

#ifndef DPP_COST_FUNCTIONS_RISK_AWARE_COST_CUH_
#define DPP_COST_FUNCTIONS_RISK_AWARE_COST_CUH_


#include <mppi/cost_functions/cost.cuh>
#include <mppi/utils/math_utils.h>

template <class DYN_T, int SIM_TIME_HORIZON>
struct RiskAwareCostParams : public CostParams<DYN_T::CONTROL_DIM>
{
  static constexpr int TIME_HORIZON = SIM_TIME_HORIZON;

  // ===== Tracking =====
  float goal[DYN_T::OUTPUT_DIM * TIME_HORIZON] = {0};
  float tracking_weights[DYN_T::OUTPUT_DIM] = {1};

  // ===== Speed / rotation =====
  float v_ref = 0.0f;
  float w_ref = 0.0f;
  float w_speed = 0.0f;
  float w_rotation = 0.0f;

  // ===== Risk =====
  float* collision_prob = nullptr;   // size = TIME_HORIZON
  float w_soft = 0.0f;
  float w_hard = 0.0f;
  float sigma = 0.05f;

  int current_time = 0;

  __host__ __device__
  int idx(int t) const
  {
    int tt = current_time + t;
    if (tt >= TIME_HORIZON) tt = TIME_HORIZON - 1;
    return tt;
  }

  __host__ __device__
  float getCollisionProb(int t) const
  {
    return collision_prob ? collision_prob[idx(t)] : 0.0f;
  }
};


template <class CLASS_T, class DYN_T, class PARAMS_T>
class RiskAwareCostImpl
  : public Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>
{
public:
  using PARENT = Cost<CLASS_T, PARAMS_T, typename DYN_T::DYN_PARAMS_T>;
  using output_array = typename PARENT::output_array;

  RiskAwareCostImpl(cudaStream_t stream = nullptr)
  {
    this->bindToStream(stream);
  }

  std::string getCostFunctionName() const override
  {
    return "RiskAwareCost";
  }

  __device__ float computeRunningCost(
    float* s,
    float* u,
    float* noise,
    float* std_dev,
    float lambda,
    float alpha,
    int timestep,
    float* theta_c,
    int* crash_status)
  {
    float cost = 0.0f;

    // ================= Tracking =================
    float* goal = this->params_.goal +
                  timestep * DYN_T::OUTPUT_DIM;

    for (int i = 0; i < DYN_T::OUTPUT_DIM; i++)
    {
      float e = s[i] - goal[i];
      cost += this->params_.tracking_weights[i] * e * e;
    }

    // ================= Speed =================
    float dv = u[0] - this->params_.v_ref;
    cost += this->params_.w_speed * dv * dv;

    // ================= Rotation =================
    float dw = u[1] - this->params_.w_ref;
    cost += this->params_.w_rotation * dw * dw;

    // ================= Risk =================
    float P = this->params_.getCollisionProb(timestep);

    cost += this->params_.w_soft * P;

    if (P > this->params_.sigma)
    {
      cost += this->params_.w_hard;
    }

    return cost;
  }

   __device__ float terminalCost(float* s, float* theta_c)
  {
    return 0.0f;
  }
};


#if __CUDACC__
#include "risk_aware_cost.cu"
#endif

template <class DYN_T, int SIM_TIME_HORIZON>
class RiskAwareCost : public RiskAwareCostImpl<RiskAwareCost<DYN_T, SIM_TIME_HORIZON>, DYN_T, RiskAwareCostParams<DYN_T, SIM_TIME_HORIZON>>
{
public:
  RiskAwareCost(cudaStream_t stream = nullptr)
    : RiskAwareCostImpl<RiskAwareCost, DYN_T, RiskAwareCostParams<DYN_T, SIM_TIME_HORIZON>>(
          stream){};
};

#endif  // DPP_COST_FUNCTIONS_RISK_AWARE_COST_CUH_
