# Reference: https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_bringup/launch/ur5e.launch.py
"""
Launch fake-hardware UR5e bringup + a parallel robot_state_publisher for the
OnRobot RG2 + a static TF tool0 -> rg2_base_link, then bring up RViz with a
config that shows both robot models.

This is Phase 7's standalone-URDF-plus-static-TF integration (07-CONTEXT.md
D-4..D-6). The RG2 URDF is at `compliant_insertion_studio/urdf/rg2/rg2.urdf`
and uses `package://compliant_insertion_studio/...` mesh paths; we rewrite
those to absolute `file://` paths at launch time because the repo is not a
ROS2 package.

Usage:
    source ~/ros2_ws/install/setup.bash
    ros2 launch /home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/launch/ur5e_with_rg2.launch.py

Arguments:
    rviz       (default: true)  Launch RViz2 with the dual-RobotModel config.
    rg2_only   (default: false) Skip the UR5e bringup; useful for fast RG2
                                 visual iteration without spinning up the
                                 controller manager.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_URDF_PATH = REPO / "compliant_insertion_studio" / "urdf" / "rg2" / "rg2.urdf"
RVIZ_CONFIG = REPO / "compliant_insertion_studio" / "rviz" / "ur5e_with_rg2.rviz"


def _load_rg2_description(_context):
    """Read the RG2 URDF and rewrite package:// mesh paths to file:// absolute
    paths so RViz can resolve them without requiring a ROS2 package install.
    """
    text = RG2_URDF_PATH.read_text()
    text = text.replace(
        "package://compliant_insertion_studio/",
        f"file://{REPO}/compliant_insertion_studio/",
    )
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace="rg2",
            name="rg2_state_publisher",
            output="screen",
            parameters=[{"robot_description": text}],
        ),
    ]


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 with the dual-RobotModel config.",
    )
    rg2_only_arg = DeclareLaunchArgument(
        "rg2_only",
        default_value="false",
        description="Skip UR5e bringup; only load RG2 URDF + static TF + RViz.",
    )

    # UR5e fake-hardware bringup. Includes ur_control.launch.py directly
    # (rather than ur5e.launch.py) so we can pass launch_rviz:=false — the
    # default UR5e launch hard-codes rviz on, and Phase 7 ships its own
    # dual-RobotModel rviz config.
    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("ur_robot_driver"),
                "/launch/ur_control.launch.py",
            ]
        ),
        launch_arguments={
            "ur_type": "ur5e",
            "robot_ip": "192.0.2.1",
            "use_fake_hardware": "true",
            "fake_sensor_commands": "true",
            "activate_joint_controller": "false",
            "launch_rviz": "false",
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("rg2_only")),
    )

    # Static TF tool0 -> rg2_base_link.
    # The per-mesh <visual><origin rpy> blocks in the RG2 URDF already encode
    # each link's USD-authored -90°/+90°-about-Z orientations. So the v1
    # hypothesis here is identity — RG2 base mounts at the UR mechanical
    # flange (= tool0) with no extra rotation. Iteration handled via the
    # static_tf_rpy launch argument (see below).
    # Default rpy_z = π = 3.1415926 — derived empirically from a live read
    # of the isaac-sim-mcp ur5e-dt quickstart attach scene, anchored on the
    # /World/RG2_Gripper asset root (NOT the leaf onrobot_rg2_base_link prim:
    # the leaf's own -90° about Z is already encoded in each per-mesh
    # <visual><origin rpy> block of rg2.urdf).
    # See _references/articles/rg2_attach_transform.md.
    # Translation is ≈ (0, 0, 0) — the RG2 base mounts at the UR mechanical
    # flange = tool0.
    rpy_arg = DeclareLaunchArgument(
        "tf_rpy",
        default_value="0 0 3.141592653589793",
        description="Roll Pitch Yaw in radians (xyz) for tool0 -> rg2_base_link.",
    )
    xyz_arg = DeclareLaunchArgument(
        "tf_xyz",
        default_value="0 0 0",
        description="Translation in meters (x y z) for tool0 -> rg2_base_link.",
    )

    def _make_static_tf(context):
        rpy = context.launch_configurations.get("tf_rpy", "0 0 0").split()
        xyz = context.launch_configurations.get("tf_xyz", "0 0 0").split()
        return [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="tool0_to_rg2_base_link",
                output="screen",
                arguments=[
                    "--x", xyz[0], "--y", xyz[1], "--z", xyz[2],
                    "--roll", rpy[0], "--pitch", rpy[1], "--yaw", rpy[2],
                    "--frame-id", "tool0",
                    "--child-frame-id", "rg2_base_link",
                ],
            ),
        ]

    static_tf_tool0_to_rg2 = OpaqueFunction(function=_make_static_tf)

    rg2_state_publisher_action = OpaqueFunction(function=_load_rg2_description)

    # When --rg2_only, we still need a base_link → tool0 chain so RViz can
    # render the gripper. Use a single static TF world → tool0 to anchor.
    static_tf_world_to_tool0 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_tool0",
        output="screen",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0.5",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "world",
            "--child-frame-id", "tool0",
        ],
        condition=IfCondition(LaunchConfiguration("rg2_only")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_ur5e_rg2",
        output="screen",
        arguments=["-d", str(RVIZ_CONFIG)],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            rviz_arg,
            rg2_only_arg,
            rpy_arg,
            xyz_arg,
            ur_launch,
            static_tf_tool0_to_rg2,
            rg2_state_publisher_action,
            static_tf_world_to_tool0,
            rviz,
        ]
    )
