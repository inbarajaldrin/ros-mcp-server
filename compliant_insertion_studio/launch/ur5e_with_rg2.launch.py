# Reference: https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/humble/ur_bringup/launch/ur5e.launch.py
"""
Launch UR5e bringup (real OR fake hardware) + a parallel robot_state_publisher
for the OnRobot RG2 + a static TF tool0 -> rg2_base_link, then optionally bring
up RViz with the dual-RobotModel config.

This is Phase 7's standalone-URDF-plus-static-TF integration (07-CONTEXT.md
D-4..D-6). The RG2 URDF is at `compliant_insertion_studio/urdf/rg2/rg2_articulated.urdf`
and uses `package://compliant_insertion_studio/...` mesh paths; rewritten to
absolute `file://` paths at launch time because the repo is not a ROS2 package.

The RG2 is fully articulated (master finger_joint + 5 mimic joints). A
joint_state_publisher evaluates the mimics and publishes /rg2/joint_states; it is
fed by /rg2/gripper_command, so commanding finger_joint alone opens/closes both
fingers symmetrically. Drive it with scripts/rg2_gripper_command.py open|close.

Usage:
    # Fake hardware (visualization-only)
    ros2 launch .../ur5e_with_rg2.launch.py

    # Real robot
    ros2 launch .../ur5e_with_rg2.launch.py use_fake_hardware:=false robot_ip:=192.168.1.111

Arguments:
    rviz                (default: true)   Launch RViz2 with dual-RobotModel config.
    rg2_only            (default: false)  Skip UR5e bringup; only RG2 + static TFs + RViz.
    use_fake_hardware   (default: true)   true=fake, false=real hardware.
    fake_sensor_commands(default: true)   Only used in fake mode.
    robot_ip            (default: 192.0.2.1) Robot IP — set when use_fake_hardware:=false.
    tf_rpy              (default: "0 0 π/2") Roll/Pitch/Yaw for tool0 -> rg2_base_link static TF
                        (π/2 matches the articulated RG2 to isaac-sim-mcp; old rigid model used π).
    tf_xyz              (default: "0 0 0") Translation for tool0 -> rg2_base_link static TF.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
# Articulated RG2 (master finger_joint + 5 mimic joints). The old single-rigid-body
# rg2.urdf (visualization-only) is kept alongside for any collision-geometry consumers.
RG2_URDF_PATH = REPO / "compliant_insertion_studio" / "urdf" / "rg2" / "rg2_articulated.urdf"
RVIZ_CONFIG = REPO / "compliant_insertion_studio" / "rviz" / "ur5e_with_rg2.rviz"


def _load_rg2_description(context):
    """Bring up the articulated RG2 in the /rg2 namespace:

      * robot_state_publisher — publishes TF for every gripper link from
        /rg2/joint_states.
      * joint_state_publisher — evaluates the URDF `mimic` joints and publishes
        /rg2/joint_states. Fed by /rg2/gripper_command (source_list, JointState
        carrying finger_joint), so one angle drives all six joints and both
        fingers move symmetrically. This is the articulated core.

    Two ways to drive the core:
      * DEFAULT (gripper_bridge:=true) — rg2_command_bridge.py translates the
        established /gripper_command (std_msgs/String "open"/"close"/width*10)
        into finger_joint on /rg2/gripper_command and echoes /gripper_width_sim.
        This is the same interface isaac-sim-mcp and primitives/control_gripper.py
        speak, so `control_gripper.py open|close` drives the RViz gripper too.
      * LOW-LEVEL (gripper_bridge:=false) — no bridge; publish finger_joint
        directly with scripts/rg2_gripper_command.py.

    package:// mesh paths are rewritten to file:// absolute paths so RViz can
    resolve them without a ROS2 package install.
    """
    text = RG2_URDF_PATH.read_text()
    text = text.replace(
        "package://compliant_insertion_studio/",
        f"file://{REPO}/compliant_insertion_studio/",
    )
    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace="rg2",
            name="rg2_state_publisher",
            output="screen",
            parameters=[{"robot_description": text}],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            namespace="rg2",
            name="rg2_joint_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": text,
                "source_list": ["gripper_command"],
                "rate": 30.0,
            }],
        ),
    ]
    # Default: bridge the established /gripper_command (String) interface onto the
    # articulated core. Run as a plain python process since the repo is not a
    # ROS2 package. Toggle off with gripper_bridge:=false for the low-level path.
    if context.launch_configurations.get("gripper_bridge", "true").lower() != "false":
        actions.append(
            ExecuteProcess(
                cmd=["python3", str(REPO / "compliant_insertion_studio" / "scripts" / "rg2_command_bridge.py")],
                name="rg2_command_bridge",
                output="screen",
            )
        )
    return actions


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
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="true = fake hardware (no robot), false = real hardware (use --robot_ip).",
    )
    fake_sensor_commands_arg = DeclareLaunchArgument(
        "fake_sensor_commands",
        default_value="true",
        description="Only honored when use_fake_hardware:=true.",
    )
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.0.2.1",
        description="Robot IP — set when use_fake_hardware:=false.",
    )
    gripper_bridge_arg = DeclareLaunchArgument(
        "gripper_bridge",
        default_value="true",
        description="true (default): bridge /gripper_command (String width) -> RG2 "
                    "articulation, like isaac-sim-mcp/control_gripper.py. "
                    "false: low-level finger_joint path (rg2_gripper_command.py).",
    )

    # UR5e bringup — real or fake. Includes ur_control.launch.py directly
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
            "robot_ip": LaunchConfiguration("robot_ip"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "activate_joint_controller": "false",
            "launch_rviz": "false",
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("rg2_only")),
    )

    # Static TF tool0 -> rg2_base_link. Default yaw = π/2 for the ARTICULATED URDF.
    # Derivation (ROS frames only): the old rigid rg2.urdf baked rpy=(0,0,-π/2) into
    # every mesh origin and mounted at yaw=π, giving a net mesh orientation of
    # π + (-π/2) = +π/2 vs tool0 (the orientation validated for the rigid model).
    # The articulated URDF (from onrobot_ros2) has identity mesh origins, so the
    # static TF must itself carry the +π/2 to reproduce that same world orientation
    # and match the isaac-sim-mcp twin. Translation ≈ (0,0,0): the RG2 base mounts
    # at the UR mechanical flange = tool0. Tunable via tf_rpy / tf_xyz
    # (pass tf_rpy:="0 0 3.14159" for the old rigid-model yaw).
    rpy_arg = DeclareLaunchArgument(
        "tf_rpy",
        default_value="0 0 1.5707963267948966",
        description="Roll Pitch Yaw in radians (xyz) for tool0 -> rg2_base_link. "
                    "Default π/2 matches the articulated RG2 to the isaac-sim-mcp "
                    "twin (the old rigid model used π).",
    )
    xyz_arg = DeclareLaunchArgument(
        "tf_xyz",
        default_value="0 0 0",
        description="Translation in meters (x y z) for tool0 -> rg2_base_link.",
    )

    def _make_static_tf(context):
        rpy = context.launch_configurations.get("tf_rpy", "0 0 0").split()
        xyz = context.launch_configurations.get("tf_xyz", "0 0 0").split()
        adapter = context.launch_configurations.get("adapter", "true").lower() != "false"
        if adapter:
            # Insert the 15 mm OnRobot Quick-Changer between tool0 and the RG2:
            #   tool0 -> quick_changer_link        : coincident at the flange; the QC
            #                                        mesh extends +15 mm along its +Z.
            #   quick_changer_link -> rg2_base_link: +15 mm along +Z (= tool0 +Z) plus
            #                                        the gripper's yaw (tf_rpy).
            # Net tool0 -> rg2_base = the old single-TF transform + 15 mm in tool0 +Z,
            # so the gripper keeps its orientation and moves 15 mm out, with the QC
            # rendered in the gap. Toggle off with adapter:=false (original behaviour).
            z_rg2 = str(float(xyz[2]) + 0.015)
            return [
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="tool0_to_quick_changer",
                    output="screen",
                    arguments=[
                        "--x", "0", "--y", "0", "--z", "0",
                        "--roll", "0", "--pitch", "0", "--yaw", "0",
                        "--frame-id", "tool0",
                        "--child-frame-id", "quick_changer_link",
                    ],
                ),
                Node(
                    package="tf2_ros",
                    executable="static_transform_publisher",
                    name="quick_changer_to_rg2_base_link",
                    output="screen",
                    arguments=[
                        "--x", xyz[0], "--y", xyz[1], "--z", z_rg2,
                        "--roll", rpy[0], "--pitch", rpy[1], "--yaw", rpy[2],
                        "--frame-id", "quick_changer_link",
                        "--child-frame-id", "rg2_base_link",
                    ],
                ),
            ]
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

    def _load_adapter_description(context):
        """robot_state_publisher for the standalone Quick-Changer URDF (namespace
        /quick_changer) so RViz renders the QC mesh at the quick_changer_link frame.
        Skipped if adapter:=false. package:// is rewritten to file:// like the RG2."""
        if context.launch_configurations.get("adapter", "true").lower() == "false":
            return []
        path = REPO / "compliant_insertion_studio" / "urdf" / "adapter" / "quick_changer.urdf"
        text = path.read_text().replace(
            "package://compliant_insertion_studio/",
            f"file://{REPO}/compliant_insertion_studio/",
        )
        return [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace="quick_changer",
                name="quick_changer_state_publisher",
                output="screen",
                parameters=[{"robot_description": text}],
            ),
        ]

    static_tf_tool0_to_rg2 = OpaqueFunction(function=_make_static_tf)

    rg2_state_publisher_action = OpaqueFunction(function=_load_rg2_description)

    adapter_state_publisher_action = OpaqueFunction(function=_load_adapter_description)

    adapter_arg = DeclareLaunchArgument(
        "adapter",
        default_value="true",
        description="true (default): insert the 15 mm Quick-Changer between tool0 and "
                    "rg2_base_link (two static TFs + QC robot_state_publisher). "
                    "false: original single tool0 -> rg2_base_link TF.",
    )

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
            use_fake_hardware_arg,
            fake_sensor_commands_arg,
            robot_ip_arg,
            gripper_bridge_arg,
            rpy_arg,
            xyz_arg,
            adapter_arg,
            ur_launch,
            static_tf_tool0_to_rg2,
            rg2_state_publisher_action,
            adapter_state_publisher_action,
            static_tf_world_to_tool0,
            rviz,
        ]
    )
