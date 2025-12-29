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

    pkg_dir = get_package_share_directory('dynamic_path_planning')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    world_file = os.path.join(pkg_dir, 'config', 'worlds', 'move_actor.world')
    actor_path = os.path.join(pkg_dir, 'config', 'skins')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    verbose = LaunchConfiguration('verbose').perform(context)
    headless = LaunchConfiguration('headless').perform(context)
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    gz_resource_path = AppendEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=actor_path)

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
                f'/actor{i}/cmd_path@geometry_msgs/msg/PoseArray@gz.msgs.Pose_V',
                f'/actor{i}/pose@geometry_msgs/msg/Pose@gz.msgs.Pose'
            ],
            output='screen'
        )
    )
        
    # Load model and URDF
    TURTLEBOT3_MODEL = 'waffle'
    model_dir = f'turtlebot3_{TURTLEBOT3_MODEL}'
    # sdf_file_name = 'model.sdf'
    sdf_file_name = 'minimal_model.sdf'
    sdf_path = os.path.join(pkg_dir, 'models', model_dir, sdf_file_name)

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    frame_prefix = LaunchConfiguration('frame_prefix', default='')

    # urdf_file_name = 'turtlebot3_' + TURTLEBOT3_MODEL + '.urdf'
    urdf_file_name = 'minimal_urdf.urdf'
    urdf_path = os.path.join(pkg_dir, 'urdf', urdf_file_name)

    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()

    robot_state_publisher_cmd = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
                'frame_prefix': PythonExpression(["'", frame_prefix, "/'"])
            }]
    )

    start_gazebo_ros_spawner_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', TURTLEBOT3_MODEL,
            '-file', sdf_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01'
        ],
        output='screen',
    )

    bridge_params = os.path.join(pkg_dir, 'config', TURTLEBOT3_MODEL + '_bridge.yaml')

    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen',
    )


    return ([
        gz_resource_path,
        gz_sim,
        *ros_gz_bridges,
        robot_state_publisher_cmd,
        start_gazebo_ros_spawner_cmd,
        start_gazebo_ros_bridge_cmd
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

    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Specify namespace of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Specify namespace of the robot')
    
    return LaunchDescription([
        declare_use_sim_time,
        verbose_arg,
        headless_arg,
        declare_x_position_cmd,
        declare_y_position_cmd,
        OpaqueFunction(function=launch_setup)
    ])