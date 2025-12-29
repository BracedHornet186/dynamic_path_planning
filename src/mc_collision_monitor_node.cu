#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/float32.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "dynamic_path_planning/mc_collision_kernel.cuh"
#include "dynamic_path_planning/pedestrian_gaussian.hpp"

class MCCollisionMonitor : public rclcpp::Node {
public:
    MCCollisionMonitor() : Node("mc_collision_monitor_cuda") {

        delta_ = this->declare_parameter("delta_radius", 0.5);
        num_samples_ = this->declare_parameter("num_mc_samples", 256);
        update_rate_ = this->declare_parameter("update_rate", 10.0);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&MCCollisionMonitor::odom_cb, this, std::placeholders::_1)
        );

        // Subscribe to multiple pedestrians
        for (int i = 1; i < max_peds_ + 1; ++i) {
            auto sub = this->create_subscription<
                geometry_msgs::msg::PoseWithCovarianceStamped>(
                "/pedestrian" + std::to_string(i) + "/est_pose",
                10,
                [this, i](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                    ped_cb(msg, i);
                }
            );
            ped_subs_.push_back(sub);
        }

        cp_pub_ = this->create_publisher<std_msgs::msg::Float32>(
            "/collision_probability", 10
        );

        timer_ = this->create_wall_timer(
            std::chrono::seconds(1) / update_rate_,
            std::bind(&MCCollisionMonitor::compute_cp, this)
        );

        cudaMalloc(&d_joint_cp_, num_samples_ * sizeof(float));
        d_peds_ = nullptr;

        RCLCPP_INFO(this->get_logger(), "MC Collision Monitor (CUDA) started");
    }

    ~MCCollisionMonitor() {
        cudaFree(d_joint_cp_);
        if (d_peds_) cudaFree(d_peds_);
    }

private:
    // ---------------- Callbacks ----------------

    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
        robot_x_ = msg->pose.pose.position.x;
        robot_y_ = msg->pose.pose.position.y;
        has_odom_ = true;
    }

    void ped_cb(
        const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg,
        int id
    ) {
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

        pedestrians_[id] = p;
    }

    // ---------------- Main computation ----------------

    void compute_cp() {
        if (!has_odom_ || pedestrians_.empty()) return;

        // Pack active pedestrians
        host_peds_.clear();
        for (auto& kv : pedestrians_) {
            host_peds_.push_back(kv.second);
        }

        // Allocate GPU memory if needed
        if (host_peds_.size() != last_num_peds_) {
            if (d_peds_) cudaFree(d_peds_);
            cudaMalloc(&d_peds_, host_peds_.size() * sizeof(PedestrianGaussian));
            last_num_peds_ = host_peds_.size();
        }

        cudaMemcpy(
            d_peds_,
            host_peds_.data(),
            host_peds_.size() * sizeof(PedestrianGaussian),
            cudaMemcpyHostToDevice
        );

        launch_mc_collision_kernel(
            robot_x_,
            robot_y_,
            delta_,
            d_peds_,
            host_peds_.size(),
            d_joint_cp_,
            num_samples_
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(),
                "CUDA kernel launch failed: %s",
                cudaGetErrorString(err));
            return;
        }

        cudaDeviceSynchronize();

        thrust::device_ptr<float> ptr(d_joint_cp_);
        float cp = thrust::reduce(ptr, ptr + num_samples_, 0.0f) / num_samples_;

        std_msgs::msg::Float32 msg;
        msg.data = cp;
        cp_pub_->publish(msg);
    }

    // ---------------- Members ----------------

    float delta_;
    int num_samples_;
    float update_rate_;
    float robot_x_{0.0f}, robot_y_{0.0f};
    bool has_odom_{false};

    static constexpr int max_peds_ = 10;

    std::unordered_map<int, PedestrianGaussian> pedestrians_;
    std::vector<PedestrianGaussian> host_peds_;

    size_t last_num_peds_{0};

    float* d_joint_cp_{nullptr};
    PedestrianGaussian* d_peds_{nullptr};

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::vector<rclcpp::Subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> ped_subs_;

    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr cp_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

// ============================================================

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MCCollisionMonitor>());
    rclcpp::shutdown();
    return 0;
}
