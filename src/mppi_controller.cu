#include <memory>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"

// MPPI includes
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include "dynamic_path_planning/risk_aware_cost.cuh"
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
    dt_ = this->declare_parameter("dt", 0.05);
    max_v_ = this->declare_parameter("max_linear_vel", 1.5);
    max_w_ = this->declare_parameter("max_angular_vel", 1.0);
    rate_  = this->declare_parameter("rate", 10.0);

    // ---------------- ROS I/O ----------------
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&MPPIControllerNode::odomCallback, this, _1));

    cmd_pub_ =
      this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    cp_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "/collision_probability", 10,
      [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg)
      {
        if (msg->data.size() < NUM_TIMESTEPS) return;

        for (int t = 0; t < NUM_TIMESTEPS; ++t)
          collision_prob_host_[t] = msg->data[t];
      });


    // ---------------- MPPI setup ----------------
    dynamics_ = std::make_shared<DYN_T>();
    cost_     = std::make_shared<COST_T>();
    fb_       = std::make_shared<FB_T>(dynamics_.get(), dt_);

    collision_prob_host_.resize(NUM_TIMESTEPS, 0.0f);

    // Allocate device memory
    cudaMalloc(&collision_prob_device_,
              NUM_TIMESTEPS * sizeof(float));
    
    auto err = cudaMalloc(&collision_prob_device_,
                      NUM_TIMESTEPS * sizeof(float));
    if (err != cudaSuccess)
    {
      throw std::runtime_error("cudaMalloc failed for collision_prob_device_");
    }
    
    // Attach pointer to cost params
    auto& cost_params = cost_->getParams();
    cost_params.collision_prob = collision_prob_device_;

    // Set reasonable defaults
    cost_params.w_soft = 10.0f;
    cost_params.w_hard = 1e5f;
    cost_params.sigma  = 0.05f;
    cost_params.v_ref  = 0.5f;
    cost_params.w_ref  = 0.0f;
    cost_params.w_speed    = 1.0f;
    cost_params.w_rotation = 0.5f;


    SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
    sampler_params.std_dev[0] = 0.3;  // v
    sampler_params.std_dev[1] = 0.5;  // yaw rate
    sampler_ = std::make_shared<SAMPLING_T>(sampler_params);

    CONTROLLER_PARAMS_T params;
    params.dt_ = dt_;
    params.lambda_ = 1.0;
    params.dynamics_rollout_dim_ = dim3(64, DYN_T::STATE_DIM, 1);
    params.cost_rollout_dim_ = dim3(NUM_TIMESTEPS, 1, 1);

    controller_ = std::make_shared<CONTROLLER_T>(
      dynamics_.get(), cost_.get(), fb_.get(), sampler_.get(), params);

    // ---------------- Timer ----------------
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_),
      std::bind(&MPPIControllerNode::controlLoop, this));

    RCLCPP_INFO(this->get_logger(), "MPPI cmd_vel node started");
  }

  ~MPPIControllerNode()
  {
    if (collision_prob_device_)
      cudaFree(collision_prob_device_);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    last_odom_ = *msg;
    have_odom_ = true;
  }

  void controlLoop()
  {
    if (!have_odom_) return;

    // --- Update risk on GPU ---
    cudaMemcpyAsync(
      collision_prob_device_,
      collision_prob_host_.data(),
      NUM_TIMESTEPS * sizeof(float),
      cudaMemcpyHostToDevice,
      0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      RCLCPP_ERROR(this->get_logger(),
                  "CUDA error: %s", cudaGetErrorString(err));
    }

    // --- Build state ---
    DYN_T::state_array x;
    x[0] = last_odom_.pose.pose.position.x;
    x[1] = last_odom_.pose.pose.position.y;
    x[2] = yawFromQuat(last_odom_.pose.pose.orientation);

    // --- MPPI ---
    controller_->computeControl(x, 1);
    auto u0 = controller_->getControlSeq().col(0);

    // --- Clamp ---
    float v = std::clamp(u0[0], -max_v_, max_v_);
    float w = std::clamp(u0[1], -max_w_, max_w_);

    // --- Publish ---
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x  = v;
    cmd.angular.z = w;
    cmd_pub_->publish(cmd);
  }


  // ---------------- ROS ----------------
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr cp_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Odometry last_odom_;
  bool have_odom_{false};

  // -------------- Risk buffers --------------
  std::vector<float> collision_prob_host_;   // size NUM_TIMESTEPS
  float* collision_prob_device_{nullptr};

  // ---------------- MPPI ----------------
  std::shared_ptr<DYN_T> dynamics_;
  std::shared_ptr<COST_T> cost_;
  std::shared_ptr<FB_T> fb_;
  std::shared_ptr<SAMPLING_T> sampler_;
  std::shared_ptr<CONTROLLER_T> controller_;

  // ---------------- Params ----------------
  double dt_;
  double rate_;
  double max_v_;
  double max_w_;
};

// ===================== main =====================
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPPIControllerNode>());
  rclcpp::shutdown();
  return 0;
}
