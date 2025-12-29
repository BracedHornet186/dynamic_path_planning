#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

class PedestrianPredictor(Node):
    """
    Predicts pedestrian motion using a Mixture of Gaussians
    with constant velocity and stochastic direction switching.
    """

    def __init__(self):
        super().__init__('pedestrian_predictor')

        # ---------------- Parameters ----------------
        self.declare_parameter('actor_id', 1)
        self.declare_parameter('horizon', 10)
        self.declare_parameter('linear_speed', 1.0)
        self.declare_parameter('diagonal_prob', 0.1)
        self.declare_parameter('linear_prob', 0.9)
        self.declare_parameter('update_rate', 10.0)

        self.actor_id = self.get_parameter('actor_id').value
        self.horizon = self.get_parameter('horizon').value
        self.speed = self.get_parameter('linear_speed').value
        self.diagonal_prob = self.get_parameter('diagonal_prob').value
        self.linear_prob = self.get_parameter('linear_prob').value
        self.update_rate = self.get_parameter('update_rate').value

        self.pose = None
        self.dt = 0.2
        self.weights = [self.linear_prob, self.diagonal_prob]
        self.num_modes = len(self.weights)

        # Noise growth
        self.base_cov = 0.05 * np.eye(2)

        # Subscriber
        self.create_subscription(PoseStamped, f'/actor{self.actor_id}/pose', self.pose_callback, 10)

        # Publisher
        self.pred_pub = self.create_publisher(PoseWithCovarianceStamped, f'/pedestrian{self.actor_id}/est_pose', 10)

        self.timer = self.create_timer(1.0 / self.update_rate, self.predict)

        self.get_logger().info("Pedestrian predictor node started")

    # -------------------------------------------------

    def pose_callback(self, msg):
        p = msg.pose.position
        self.pose = np.array([p.x, p.y])

    # -------------------------------------------------

    def predict(self):
        if self.pose is None:
            return

        pos = self.pose
        lambda_decay = 0.3

        mus = []
        Sigmas = []
        weights_t = []

        for t in range(self.horizon):
            mean_modes = []
            cov_modes = []

            for mode in range(self.num_modes):
                if mode == 0:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = np.array([1.0 / np.sqrt(2),
                                        1.0 / np.sqrt(2)])

                direction /= np.linalg.norm(direction)
                velocity = self.speed * direction

                mean = pos + velocity * self.dt * (t + 1)
                cov = self.base_cov * (t + 1)

                mean_modes.append(mean)
                cov_modes.append(cov)

            # MoG moment matching
            mu_t = np.average(mean_modes, axis=0, weights=self.weights)
            Sigma_t = np.zeros((2, 2))

            for w, m, C in zip(self.weights, mean_modes, cov_modes):
                diff = (m - mu_t).reshape(2, 1)
                Sigma_t += w * (C + diff @ diff.T)

            alpha_t = np.exp(-lambda_decay * t)

            mus.append(mu_t)
            Sigmas.append(Sigma_t)
            weights_t.append(alpha_t)

        weights_t = np.array(weights_t)
        weights_t /= np.sum(weights_t)

        # Time-decayed aggregation
        mu_bar = np.sum([w * m for w, m in zip(weights_t, mus)], axis=0)

        Sigma_bar = np.zeros((2, 2))
        for w, m, S in zip(weights_t, mus, Sigmas):
            diff = (m - mu_bar).reshape(2, 1)
            Sigma_bar += w * (S + diff @ diff.T)

        # Publish PoseWithCovarianceStamped
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.pose.position.x = float(mu_bar[0])
        msg.pose.pose.position.y = float(mu_bar[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        cov = np.zeros((6, 6))
        cov[0, 0] = Sigma_bar[0, 0]
        cov[0, 1] = Sigma_bar[0, 1]
        cov[1, 0] = Sigma_bar[1, 0]
        cov[1, 1] = Sigma_bar[1, 1]
        cov[2, 2] = 1.0       # z uncertainty
        cov[3, 3] = 10.0      # roll
        cov[4, 4] = 10.0      # pitch
        cov[5, 5] = 1.0       # yaw

        msg.pose.covariance = cov.flatten().tolist()
        self.pred_pub.publish(msg)

# -----------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PedestrianPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
