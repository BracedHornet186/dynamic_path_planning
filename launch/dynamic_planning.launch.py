#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    num_actors = int(LaunchConfiguration('num_actors').perform(context))
    delta_radius = float(LaunchConfiguration('delta_radius').perform(context))
    linear_speed = float(LaunchConfiguration('linear_speed').perform(context))
    diagonal_prob = float(LaunchConfiguration('diagonal_prob').perform(context))
    linear_prob = 1 - diagonal_prob
    max_ang_speed = float(LaunchConfiguration('max_ang_speed').perform(context))
    horizon = int(LaunchConfiguration('horizon').perform(context))
    update_rate = float(LaunchConfiguration('update_rate').perform(context))

    vel_publisher_nodes = []
    ped_predictor_nodes = []

    for actor_id in range(1, num_actors + 1):

        vel_publisher_nodes.append(
            Node(
                package='dynamic_path_planning',
                executable='vel_publisher.py',
                name=f'vel_publisher_actor_{actor_id}',
                parameters=[{
                    'actor_id': actor_id,
                    'linear_speed': linear_speed,
                    'diagonal_prob': diagonal_prob,
                    'max_ang_speed': max_ang_speed,
                    'update_rate': update_rate
                }],
                output='screen'
            )
        )

        ped_predictor_nodes.append(
            Node(
                package='dynamic_path_planning',
                executable='pedestrian_predictor.py',
                name=f'pedestrian_predictor_{actor_id}',
                parameters=[{
                    'actor_id': actor_id,
                    'horizon': horizon,
                    'linear_speed': linear_speed,
                    'diagonal_prob': diagonal_prob,
                    'linear_prob': linear_prob,
                    'update_rate': update_rate
                }],
                output='screen'
            )
        )

    collision_monitor_node = Node(
        package='dynamic_path_planning',
        executable='mc_collision_monitor',
        name='mc_collision_monitor_cuda',
        parameters=[{
            'delta_radius': delta_radius,
            'num_mc_samples': 256,
            'update_rate': update_rate
        }],
        output='screen'
    )
    return vel_publisher_nodes + ped_predictor_nodes + [collision_monitor_node]


def generate_launch_description():

    declare_num_actors = DeclareLaunchArgument(
        'num_actors',
        default_value='12',
        description='Number of actors'
    )
    declare_linear_speed = DeclareLaunchArgument(
        'linear_speed',
        default_value='1.0',
        description='Linear speed of pedestrians'
    )
    declare_diagonal_prob = DeclareLaunchArgument(
        'diagonal_prob',
        default_value='0.1',
        description='Probability of diagonal movement'
    )
    declare_max_ang_speed = DeclareLaunchArgument(
        'max_ang_speed',
        default_value='0.4',
        description='Maximum angular speed of pedestrians'
    )
    declare_horizon = DeclareLaunchArgument(
        'horizon',
        default_value='10',
        description='Time Horizon of the predictor'
    )
    declare_update_rate = DeclareLaunchArgument(
        'update_rate',
        default_value='10.0',
        description='Update rate of the predictor'
    )
    declare_delta_radius = DeclareLaunchArgument(
        'delta_radius',
        default_value='0.5',
        description='Delta radius for collision monitoring'
    )

    return LaunchDescription([
        declare_num_actors,
        declare_linear_speed,
        declare_diagonal_prob,
        declare_max_ang_speed,
        declare_horizon,
        declare_update_rate,
        declare_delta_radius,
        OpaqueFunction(function=launch_setup)
    ])
