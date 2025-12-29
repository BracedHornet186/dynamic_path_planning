#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32

class MCCollisionMonitor(Node):

    def __init__(self):
        super().__init__('mc_collision_monitor')

        # -------- Parameters --------
        self.declare_parameter('delta', 0.5)
        self.declare_parameter('num_mc_samples', 20)

        self.delta = self.get_parameter('delta').value
        self.num_mc = self.get_parameter('num_mc_samples').value

        self.robot_pose = None
        self.pedestrians = {}  # key: topic, value: (mu, cov)

        # -------- Subscribers --------
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # You can add these dynamically later if needed
        for i in range(10):
            self.create_subscription(PoseWithCovarianceStamped,f'/pedestrian{i}/est_pose',self.make_ped_cb(i), 10)

        # -------- Publisher --------
        self.cp_pub = self.create_publisher(Float32, '/collision_probability', 10)

        self.timer = self.create_timer(0.1, self.compute_cp)

        self.get_logger().info("MC Collision Monitor started")

    # ------------------------------------------------------

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.robot_pose = np.array([p.x, p.y])

    def make_ped_cb(self, ped_id):
        def ped_cb(msg):
            mu = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            ])
            cov = np.array(msg.pose.covariance).reshape(6, 6)[:2, :2]
            self.pedestrians[ped_id] = (mu, cov)
        return ped_cb

    # ------------------------------------------------------

    def gaussian_pdf(self, x, mu, cov):
        """2D Gaussian PDF"""
        diff = x - mu
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        if det_cov < 1e-6:
            return 0.0

        norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
        exponent = -0.5 * diff.T @ inv_cov @ diff
        return float(norm * np.exp(exponent))

    # ------------------------------------------------------

    def sample_disk(self, center):
        """Uniform sampling in disk of radius delta"""
        r = self.delta * np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        return center + np.array([r * np.cos(theta), r * np.sin(theta)])

    # ------------------------------------------------------

    def compute_cp(self):
        if self.robot_pose is None or len(self.pedestrians) == 0:
            return

        joint_probs = []

        for _ in range(self.num_mc):
            xj = self.sample_disk(self.robot_pose)

            prod = 1.0
            for mu, cov in self.pedestrians.values():
                p_o = self.gaussian_pdf(xj, mu, cov)
                prod *= (1.0 - p_o)

            p_joint = 1.0 - prod
            joint_probs.append(p_joint)

        cp = float(np.mean(joint_probs))

        msg = Float32()
        msg.data = cp
        self.cp_pub.publish(msg)

# -----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MCCollisionMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()