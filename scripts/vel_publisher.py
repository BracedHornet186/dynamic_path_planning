#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import random
import math

class HumanLikeWalker(Node):

    def __init__(self):
        super().__init__('human_like_walker')

        # Parameters
        self.declare_parameter('actor_id', 1)
        self.declare_parameter('linear_speed', 1.0)
        self.declare_parameter('diagonal_prob', 0.08)
        self.declare_parameter('max_ang_speed', 0.4)
        self.declare_parameter('update_rate', 10.0)

        actor_id = self.get_parameter('actor_id').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.diagonal_prob = self.get_parameter('diagonal_prob').value
        self.max_ang_speed = self.get_parameter('max_ang_speed').value
        update_rate = self.get_parameter('update_rate').value

        topic = f'/actor{actor_id}/cmd_vel'
        self.publisher = self.create_publisher(Twist, topic, 10)

        self.timer = self.create_timer(1.0 / update_rate, self.update)

        self.current_ang_vel = 0.0
        self.steps_remaining = 0

        self.get_logger().info(f'Publishing human-like velocity to {topic}')

    def update(self):
        msg = Twist()
        msg.linear.x = self.linear_speed

        # Decide if we start diagonal motion
        if self.steps_remaining <= 0:
            if random.random() < self.diagonal_prob:
                self.current_ang_vel = random.uniform(
                    -self.max_ang_speed, self.max_ang_speed
                )
                self.steps_remaining = random.randint(10, 30)  # persists ~1–3 sec
            else:
                self.current_ang_vel = 0.0
                self.steps_remaining = random.randint(20, 50)

        msg.angular.z = self.current_ang_vel
        self.steps_remaining -= 1

        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = HumanLikeWalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
