from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = 'visual_graph_slam'
    pkg_share = get_package_share_directory(package_name)
    
    default_common_params = os.path.join(pkg_share, 'config', 'common_params.yaml')
    main_params_file = os.path.join(pkg_share, 'config', 'slam_params.yaml')
    default_rviz = os.path.join(pkg_share, 'rviz', 'graph_slam.rviz')

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=main_params_file,
        description='Path to the main ROS2 parameters file (containing the mode)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 with SLAM visualization config'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz,
        description='Path to RViz2 config file'
    )

    publish_static_tf_arg = DeclareLaunchArgument(
        'publish_static_tf',
        default_value='true',
        description='Publish static TF base_link -> cam0_link'
    )

    tf_parent_arg = DeclareLaunchArgument(
        'tf_parent_frame',
        default_value='base_link',
        description='Parent frame for static transform'
    )

    tf_child_arg = DeclareLaunchArgument(
        'tf_child_frame',
        default_value='cam0_link',

        description='Child frame for static transform'
    )

    tf_x_arg = DeclareLaunchArgument(
        'tf_x',
        default_value='-0.08',
        description='Static TF translation X (m)'
    )

    tf_y_arg = DeclareLaunchArgument(
        'tf_y',
        default_value='0.0',
        description='Static TF translation Y (m)'
    )

    tf_z_arg = DeclareLaunchArgument(
        'tf_z',
        default_value='-0.035',
        description='Static TF translation Z (m)'
    )

    tf_roll_arg = DeclareLaunchArgument(
        'tf_roll',
        default_value='-1.57079632679',
        description='Static TF roll (rad) base_link -> camera optical'
    )

    tf_pitch_arg = DeclareLaunchArgument(
        'tf_pitch',
        default_value='0.0',
        description='Static TF pitch (rad) base_link -> camera optical'
    )

    tf_yaw_arg = DeclareLaunchArgument(
        'tf_yaw',
        default_value='-1.57079632679',
        description='Static TF yaw (rad) base_link -> camera optical'
    )

    from launch.actions import OpaqueFunction

    import yaml

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='',
        description='Override the SLAM mode (mono|mono_imu|mono_imu_wheel). '
                    'When set, takes priority over the mode in params_file.'
    )

    def launch_setup(context, *args, **kwargs):
        main_params_path = context.launch_configurations['params_file']

        # Read mode from YAML as the base
        mode = 'mono'  # safe default
        try:
            with open(main_params_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'graph_slam' in config and 'ros__parameters' in config['graph_slam']:
                    mode = config['graph_slam']['ros__parameters'].get('mode', 'mono')
        except Exception as e:
            print(f"Warning: Could not parse mode from {main_params_path}: {e}")

        # CLI / launch arg overrides YAML — this is the key fix
        mode_override = context.launch_configurations.get('mode', '').strip()
        if mode_override:
            print(f"[graph_slam.launch] Mode override: '{mode}' → '{mode_override}'")
            mode = mode_override

        # ── Inject mode override BEFORE slam_params.yaml ──────────────────────
        # In ROS 2, parameter files are merged in order (last wins).
        # We inject the mode as a flat dict right after the mode-YAML so that
        # slam_params.yaml (which may have a different mode) does NOT override it.
        # The flat dict form (no node-name prefix) is what ROS 2 actually uses
        # when the dict is in the `parameters` list of a Node().
        mode_override_dict = {
            'mode': mode,
        }

        common_file = os.path.join(pkg_share, 'config', 'common_params.yaml')
        mode_file = os.path.join(pkg_share, 'config', 'modes', f'{mode}.yaml')

        final_params = [common_file]
        if os.path.exists(mode_file):
            final_params.append(mode_file)
        else:
            print(f"Warning: Mode file {mode_file} not found.")


        # Inject mode right after the mode YAML so slam_params.yaml can't overwrite it
        final_params.append(mode_override_dict)
        final_params.append(main_params_path)
        # Inject again at the very end to guarantee it survives any later file
        final_params.append(mode_override_dict)
        final_params.append({'use_sim_time': LaunchConfiguration('use_sim_time')})


        slam_node = Node(
            package=package_name,
            executable='graph_slam',
            output='screen',
            parameters=final_params
        )

        static_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='kitti_base_to_cam_static_tf',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('tf_x'),
                '--y', LaunchConfiguration('tf_y'),
                '--z', LaunchConfiguration('tf_z'),
                '--yaw', LaunchConfiguration('tf_yaw'),
                '--pitch', LaunchConfiguration('tf_pitch'),
                '--roll', LaunchConfiguration('tf_roll'),
                '--frame-id', LaunchConfiguration('tf_parent_frame'),
                '--child-frame-id', LaunchConfiguration('tf_child_frame'),
            ],
            condition=IfCondition(LaunchConfiguration('publish_static_tf')),
        )

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )

        imu_static_tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='kitti_base_to_imu_static_tf',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'base_link', '--child-frame-id', 'oxts_link'
            ],
            condition=IfCondition(LaunchConfiguration('publish_static_tf')),
        )

        gps_visualizer_node = Node(
            package=package_name,
            executable='gps_visualizer_node.py',
            name='gps_visualizer_node',
            output='screen'
        )

        return [static_tf_node, imu_static_tf_node, slam_node, gps_visualizer_node, rviz_node]

    return LaunchDescription([
        params_file_arg,
        mode_arg,
        use_sim_time_arg,
        use_rviz_arg,
        rviz_config_arg,
        publish_static_tf_arg,
        tf_parent_arg,
        tf_child_arg,
        tf_x_arg,
        tf_y_arg,
        tf_z_arg,
        tf_roll_arg,
        tf_pitch_arg,
        tf_yaw_arg,
        OpaqueFunction(function=launch_setup)
    ])

