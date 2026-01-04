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
        self.declare_parameter('num_actors', 12)
        self.declare_parameter('linear_speed', 1.0)
        self.declare_parameter('diagonal_prob', 0.1)
        self.declare_parameter('update_rate', 1.0)

        num_actors = self.get_parameter('num_actors').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.diagonal_prob = self.get_parameter('diagonal_prob').value
        update_rate = self.get_parameter('update_rate').value

        self.vel_publishers = {f'actor{i+1}': self.create_publisher(Twist, f'/actor{i+1}/cmd_vel', 10) for i in range(num_actors)}

        self.timer = self.create_timer(1.0 / update_rate, self.update)

        self.steps_remaining = 0

        self.get_logger().info(f'Publishing velocities for {num_actors} actors')

    def update(self):
        for publisher in self.vel_publishers.values():
            msg = Twist()
            msg.linear.x = self.linear_speed

            # Decide if we start diagonal motion
            if self.steps_remaining <= 0:
                if random.random() < self.diagonal_prob:
                    msg.linear.y = self.linear_speed
                    self.steps_remaining = random.randint(10, 30)  # persists ~1–3 sec
                else:
                    msg.linear.y = 0.0
                    self.steps_remaining = random.randint(20, 50)

            self.steps_remaining -= 1
            publisher.publish(msg)


def main():
    rclpy.init()
    node = HumanLikeWalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
