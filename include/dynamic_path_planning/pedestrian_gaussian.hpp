#pragma once

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct PedestrianGaussian {
    float mu_x;
    float mu_y;

    // Inverse covariance (symmetric 2x2)
    float inv_cov_xx;
    float inv_cov_xy;
    float inv_cov_yy;

    // 1 / (2π√|Σ|)
    float norm_factor;

    HOST_DEVICE
    PedestrianGaussian() = default;

    HOST_DEVICE
    PedestrianGaussian(
        float mx, float my,
        float ic_xx, float ic_xy, float ic_yy,
        float norm
    )
        : mu_x(mx), mu_y(my),
          inv_cov_xx(ic_xx),
          inv_cov_xy(ic_xy),
          inv_cov_yy(ic_yy),
          norm_factor(norm) {}
};
