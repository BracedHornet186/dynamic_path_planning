#pragma once

#include "dynamic_path_planning/pedestrian_gaussian.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void launch_mc_collision_kernel(
    float robot_x,
    float robot_y,
    float delta,
    const PedestrianGaussian* d_peds,
    int num_peds,
    float* d_joint_cp,
    int num_samples
);

#ifdef __cplusplus
}
#endif
