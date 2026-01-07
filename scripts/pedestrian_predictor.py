#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped

class PedestrianPredictor(Node):

    def __init__(self):
        super().__init__('pedestrian_predictor')

        # ---------------- Parameters ----------------
        self.declare_parameter('num_actors', 12)
        self.declare_parameter('horizon', 5)
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('process_noise', 0.05)

        self.num_actors = self.get_parameter('num_actors').value
        self.horizon = self.get_parameter('horizon').value
        self.dt = 1.0 / self.get_parameter('update_rate').value
        self.q = self.get_parameter('process_noise').value

        # State storage
        self.poses = {f'actor{i}': None for i in range(1, self.num_actors + 1)}
        self.velocities = {f'actor{i}': np.zeros(2) for i in range(1, self.num_actors + 1)}

        # Subscribers
        self.subscribers = {
            f'actor{i}': self.create_subscription(
                Pose,
                f'/actor{i}/pose',
                self.make_pose_callback(i),
                10
            )
            for i in range(1, self.num_actors + 1)
        }

        # Publishers
        self.est_pose_publishers = {
            f'pedestrian{i}': self.create_publisher(
                PoseWithCovarianceStamped,
                f'/pedestrian{i}/est_pose',
                10
            )
            for i in range(1, self.num_actors + 1)
        }

        self.timer = self.create_timer(self.dt, self.predict)

        self.get_logger().info("Gaussian pedestrian predictor started")

    # -------------------------------------------------

    def make_pose_callback(self, actor_id):
        def callback(msg: Pose):
            key = f'actor{actor_id}'
            p = np.array([msg.position.x, msg.position.y])
            noise = np.random.normal(0, 0.01, size=2)
            p += noise

            if self.poses[key] is None:
                self.velocities[key] = np.zeros(2)
            else:
                self.velocities[key] = (p - self.poses[key]) / self.dt

            self.poses[key] = p
        return callback

    # -------------------------------------------------

    def predict(self):
        for i in range(1, self.num_actors + 1):
            key = f'actor{i}'
            pos = self.poses[key]

            if pos is None:
                continue

            v = self.velocities[key]

            # ---- Mean prediction (1-step) ----
            mu = pos + v * self.dt

            # ---- Covariance growth ----
            # Random walk uncertainty
            Sigma = self.q * self.horizon * np.eye(2)

            # ---- Publish ----
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.pose.position.x = float(mu[0])
            msg.pose.pose.position.y = float(mu[1])
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.w = 1.0

            cov = np.zeros((6, 6))
            cov[0, 0] = Sigma[0, 0]
            cov[1, 1] = Sigma[1, 1]
            cov[0, 1] = cov[1, 0] = 0.0
            cov[2, 2] = 1e-3
            cov[3, 3] = 1.0
            cov[4, 4] = 1.0
            cov[5, 5] = 1.0

            msg.pose.covariance = cov.flatten().tolist()
            self.est_pose_publishers[f'pedestrian{i}'].publish(msg)

# -----------------------------------------------------

def main():
    rclpy.init()
    node = PedestrianPredictor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
