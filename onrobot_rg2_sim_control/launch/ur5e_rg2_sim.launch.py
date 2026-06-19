"""One-command sim twin: UR5e (fake hw) + husarion-native RG2 (CAD-exact) + the curated
gripper driver + RViz. Replaces the ad-hoc studio_stack manager.

  ros2 launch onrobot_rg2_sim_control ur5e_rg2_sim.launch.py
  # then drive it:
  python3 .../onrobot_rg2_sim_control/scripts/rg2_gripper.py open|close|<mm>

Gripper hangs off tool0 through the quick-changer (+90 mount baked in the URDF: reach +Z, jaws flange-X).
The driver maps CONTACT-width commands -> rg2_gripper_joint through the CAD arc model.
"""
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PKG = "/home/aaugus11/Documents/ros-mcp-server/onrobot_rg2_sim_control"
URDF = os.path.join(PKG, "urdf", "rg2_husarion_native.urdf")
RVIZ = os.path.join(PKG, "rviz", "ur5e_rg2.rviz")
DRIVER = os.path.join(PKG, "scripts", "rg2_gripper_driver.py")


def generate_launch_description():
    # UR5e fake hardware; scaled trajectory controller active; no built-in rviz
    ur = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_robot_driver"), "/launch/ur_control.launch.py"]),
        launch_arguments={
            "ur_type": "ur5e", "robot_ip": "0.0.0.0",
            "use_fake_hardware": "true", "fake_sensor_commands": "true",
            "launch_rviz": "false", "activate_joint_controller": "true",
            "initial_joint_controller": "scaled_joint_trajectory_controller",
        }.items(),
    )

    with open(URDF) as f:
        rg2_description = f.read()

    # husarion-native RG2, rooted at tool0 (mount baked in URDF). Joints come from the driver.
    rg2_rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        namespace="rg2ref", name="rg2_state_publisher",
        parameters=[{"robot_description": rg2_description}],
    )

    # curated gripper driver: /rg2sim/gripper_cmd (CONTACT mm) -> arc -> /rg2ref/joint_states
    gripper_driver = ExecuteProcess(cmd=["python3", DRIVER], name="rg2_gripper_driver", output="screen")

    rviz = Node(package="rviz2", executable="rviz2", name="rviz_ur5e_rg2",
                arguments=["-d", RVIZ], output="screen")

    return LaunchDescription([ur, rg2_rsp, gripper_driver, rviz])
