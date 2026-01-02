#include <memory>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include "dynamic_path_planning/pedestrian_gaussian.hpp"
#include "dynamic_path_planning/risk_aware_cost.cuh"

// MPPI includes
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/dynamics/dubins/dubins.cuh>
#include <mppi/feedback_controllers/DDP/ddp.cuh>

using std::placeholders::_1;

// ===================== MPPI typedefs =====================
constexpr int NUM_TIMESTEPS = 100;
constexpr int NUM_ROLLOUTS  = 2048;

using DYN_T = DubinsDynamics;
using COST_T = RiskAwareCost<DYN_T, NUM_TIMESTEPS>;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T =
  mppi::sampling_distributions::GaussianDistribution<DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T =
  VanillaMPPIController<DYN_T, COST_T, FB_T,
                         NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;
using CONTROLLER_PARAMS_T = CONTROLLER_T::TEMPLATED_PARAMS;

// ===================== Utility =====================
static double yawFromQuat(const geometry_msgs::msg::Quaternion& q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

// ===================== Node =====================
class MPPIControllerNode : public rclcpp::Node
{
public:
  MPPIControllerNode()
  : Node("mppi_cmd_vel_node")
  {
    // ---------------- Parameters ----------------
    num_peds_        = this->declare_parameter("num_actors", 12);
    delta_           = this->declare_parameter("delta_radius", 0.5);
    num_mc_samples_  = this->declare_parameter("num_mc_samples", 256);
    dt_              = this->declare_parameter("dt", 0.05);
    max_v_           = this->declare_parameter("max_linear_vel", 1.5);
    max_w_           = this->declare_parameter("max_angular_vel", 0.5);
    rate_            = this->declare_parameter("update_rate", 10.0);

    // ---------------- ROS I/O ----------------
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&MPPIControllerNode::odomCallback, this, _1));

    for (int i = 1; i <= num_peds_; ++i)
    {
      ped_subs_.push_back(
        this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "/pedestrian" + std::to_string(i) + "/est_pose",
          10,
          [this, i](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
          {
            pedCallback(msg, i);
          })
      );
    }

    cmd_pub_ =
      this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    // ---------------- MPPI setup ----------------
    dynamics_ = std::make_shared<DYN_T>();
    cost_     = std::make_shared<COST_T>();
    fb_       = std::make_shared<FB_T>(dynamics_.get(), dt_);

    // Allocate pedestrian buffer on GPU
    cudaMalloc(&d_peds_, num_peds_ * sizeof(PedestrianGaussian));

    // Sampler
    SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
    sampler_params.std_dev[0] = 0.3f;
    sampler_params.std_dev[1] = 0.5f;
    sampler_ = std::make_shared<SAMPLING_T>(sampler_params);

    CONTROLLER_PARAMS_T params;
    params.dt_ = dt_;
    params.lambda_ = 1.0f;
    params.dynamics_rollout_dim_ = dim3(64, DYN_T::STATE_DIM, 1);
    params.cost_rollout_dim_     = dim3(NUM_TIMESTEPS, 1, 1);

    controller_ = std::make_shared<CONTROLLER_T>(
      dynamics_.get(), cost_.get(), fb_.get(), sampler_.get(), params);

    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_),
      std::bind(&MPPIControllerNode::controlLoop, this));

    RCLCPP_INFO(this->get_logger(), "MPPI controller node started");
  }

  ~MPPIControllerNode()
  {
    if (d_peds_) cudaFree(d_peds_);
  }

private:
  // ---------------- Callbacks ----------------
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_odom_ = *msg;
    have_odom_ = true;
  }

  void pedCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg,
    int id)
  {
    PedestrianGaussian p;

    p.mu_x = msg->pose.pose.position.x;
    p.mu_y = msg->pose.pose.position.y;

    const auto& C = msg->pose.covariance;
    float det = C[0] * C[7] - C[1] * C[6];
    if (det < 1e-6f) return;

    p.inv_cov_xx =  C[7] / det;
    p.inv_cov_xy = -C[1] / det;
    p.inv_cov_yy =  C[0] / det;
    p.norm_factor = 1.0f / (2.0f * M_PI * sqrtf(det));

    pedestrians_[id - 1] = p;
  }

  // ---------------- Control Loop ----------------
  void controlLoop()
  {
    if (!have_odom_) return;

    // Pack pedestrians
    host_peds_.clear();
    for (auto& kv : pedestrians_)
      host_peds_.push_back(kv.second);

    if (!host_peds_.empty())
    {
      cudaMemcpy(
        d_peds_,
        host_peds_.data(),
        host_peds_.size() * sizeof(PedestrianGaussian),
        cudaMemcpyHostToDevice);
    }

    auto params = cost_->getParams();

    constexpr float GOAL_X = 50.0f;
    constexpr float GOAL_Y = 0.0f;
    constexpr float GOAL_YAW = 0.0f;

    // ---------- Tracking weights ----------
    params.tracking_weights[0] = 1.0f;
    params.tracking_weights[1] = 1.0f;
    params.tracking_weights[2] = 0.0f;

    for (int t = 0; t < NUM_TIMESTEPS; ++t)
    {
      params.goal[t * 3 + 0] = GOAL_X;
      params.goal[t * 3 + 1] = GOAL_Y;
      params.goal[t * 3 + 2] = GOAL_YAW;
    }

    params.pedestrians   = d_peds_;
    params.num_peds      = host_peds_.size();
    params.delta         = delta_;
    params.num_mc_samples = num_mc_samples_;
    params.seed =
      static_cast<unsigned long>(
        std::chrono::high_resolution_clock::now()
          .time_since_epoch().count());
    params.v_ref   = 1.0f;   // desired forward speed
    params.w_speed = 5.0f;   // strong enough

    cost_->setParams(params);

    // Build state
    DYN_T::state_array x;
    x[0] = last_odom_.pose.pose.position.x;
    x[1] = last_odom_.pose.pose.position.y;
    x[2] = yawFromQuat(last_odom_.pose.pose.orientation);

    controller_->computeControl(x, 1);

    auto u0 = controller_->getControlSeq().col(0);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x  = std::clamp(u0[0], -max_v_, max_v_);
    cmd.angular.z = std::clamp(u0[1], -max_w_, max_w_);
    cmd_pub_->publish(cmd);
  }

  // ---------------- Members ----------------
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> ped_subs_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Odometry last_odom_;
  bool have_odom_{false};

  std::map<int, PedestrianGaussian> pedestrians_;
  std::vector<PedestrianGaussian> host_peds_;
  PedestrianGaussian* d_peds_{nullptr};

  std::shared_ptr<DYN_T> dynamics_;
  std::shared_ptr<COST_T> cost_;
  std::shared_ptr<FB_T> fb_;
  std::shared_ptr<SAMPLING_T> sampler_;
  std::shared_ptr<CONTROLLER_T> controller_;

  double dt_, rate_;
  float max_v_, max_w_;
  int num_peds_;
  float delta_;
  int num_mc_samples_;
};

// ===================== main =====================
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPPIControllerNode>());
  rclcpp::shutdown();
  return 0;
} // MPPI_CONTROLLER_NODE