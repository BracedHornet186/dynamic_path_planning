#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    IncludeLaunchDescription,
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def launch_setup(context, *args, **kwargs):

    pkg_gazebo_ros_actor_plugin = get_package_share_directory('gazebo_ros_actor_plugin')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    world_file = os.path.join(pkg_gazebo_ros_actor_plugin, 'config', 'worlds', 'move_actor.world')
    model_path = os.path.join(pkg_gazebo_ros_actor_plugin, 'config', 'skins')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    verbose = LaunchConfiguration('verbose').perform(context)
    headless = LaunchConfiguration('headless').perform(context)

    gz_resource_path = AppendEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=model_path)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': PythonExpression([
                f"'{world_file} -r'",
                " + (' -v' if '", verbose, "' == 'True' else '')",
                " + (' -s' if '", headless, "' == 'True' else '')"
            ])
        }.items()
    )
    num_actors = 12
    ros_gz_bridges = []
    for i in range(1, num_actors + 1):
        ros_gz_bridges.append(
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f'ros_gz_bridge_actor_{i}',
            arguments=[
                f'/actor{i}/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                f'/actor{i}/cmd_path@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V'
            ],
            output='screen'
        )
    )

    return ([
        gz_resource_path,
        gz_sim,
        *ros_gz_bridges
    ])

def generate_launch_description():

    # Declare the launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use simulation (Gazebo) clock if true'
    )

    verbose_arg = DeclareLaunchArgument(
        'verbose', default_value='True', description='Enable verbose mode for Gazebo'
    )
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='False', description='Enable headless mode for Gazebo'
    )

    return LaunchDescription([
        declare_use_sim_time,
        verbose_arg,
        headless_arg,
        OpaqueFunction(function=launch_setup)
    ])