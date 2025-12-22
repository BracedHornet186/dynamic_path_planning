#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    num_actors = int(LaunchConfiguration('num_actors').perform(context))

    vel_publisher_nodes = []

    for actor_id in range(1, num_actors + 1):
        vel_publisher_nodes.append(
            Node(
                package='dynamic_path_planning',
                executable='vel_publisher.py',
                name=f'vel_publisher_actor_{actor_id}',
                parameters=[{
                    'actor_id': actor_id,
                    'linear_speed': 1.0,
                    'diagonal_prob': 0.08,
                    'max_ang_speed': 0.4,
                    'update_rate': 10.0
                }],
                output='screen'
            )
        )

    return vel_publisher_nodes


def generate_launch_description():

    declare_num_actors = DeclareLaunchArgument(
        'num_actors',
        default_value='12',
        description='Number of actors'
    )

    return LaunchDescription([
        declare_num_actors,
        OpaqueFunction(function=launch_setup)
    ])
