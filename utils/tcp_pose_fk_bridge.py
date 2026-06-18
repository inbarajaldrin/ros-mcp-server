#!/usr/bin/env python3
"""tcp_pose_fk_bridge.py — publish /tcp_pose_broadcaster/pose from /joint_states via FK.

WHY: the UR ros2_control driver under use_fake_hardware:=true (scripts/launch_ur_fake.sh) does
NOT compute the TCP/end-effector pose — mock_components just echoes joint commands, so the
tcp_pose state interface (and hence the real tcp_pose_broadcaster) reads NaN. move_home only
needs /joint_states so it works, but move_to_grasp blocks forever waiting for a valid
current_ee_pose from /tcp_pose_broadcaster/pose.

This bridge closes that gap WITHOUT URSim/real hardware: it subscribes to /joint_states and
publishes the flange pose computed with ros-mcp-server's OWN forward_kinematics(dh_params, ...) —
the exact same FK move_to_grasp uses for its IK targets, so the pose convention is guaranteed
consistent (flange frame, base coordinate frame, meters). Deactivate the NaN tcp_pose_broadcaster
controller first so this is the only publisher on the topic.

Usage:
    python3 utils/tcp_pose_fk_bridge.py [--topic /tcp_pose_broadcaster/pose] [--frame base]
"""
import sys
import argparse
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from primitives.shared.ik import dh_params, forward_kinematics

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class TcpPoseFkBridge(Node):
    def __init__(self, topic: str, frame: str):
        super().__init__("tcp_pose_fk_bridge")
        self.frame = frame
        # match move_to_grasp's ee_pose subscription QoS (RELIABLE + VOLATILE)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE, depth=10)
        self.pub = self.create_publisher(PoseStamped, topic, qos)
        self.sub = self.create_subscription(JointState, "/joint_states",
                                            self._on_js, qos)
        self._warned = False
        self.get_logger().info(f"FK bridge up: /joint_states -> {topic} (frame={frame})")

    def _on_js(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            angles = [name_to_pos[n] for n in JOINT_NAMES]
        except KeyError:
            if not self._warned:
                self.get_logger().warn(f"/joint_states missing UR joints; have {list(msg.name)}")
                self._warned = True
            return
        T = np.array(forward_kinematics(dh_params, angles))
        quat = R.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame
        out.pose.position.x = float(T[0, 3])
        out.pose.position.y = float(T[1, 3])
        out.pose.position.z = float(T[2, 3])
        out.pose.orientation.x = float(quat[0])
        out.pose.orientation.y = float(quat[1])
        out.pose.orientation.z = float(quat[2])
        out.pose.orientation.w = float(quat[3])
        self.pub.publish(out)


def main():
    ap = argparse.ArgumentParser(description="FK bridge: /joint_states -> /tcp_pose_broadcaster/pose")
    ap.add_argument("--topic", default="/tcp_pose_broadcaster/pose")
    ap.add_argument("--frame", default="base")
    a = ap.parse_args()
    rclpy.init()
    node = TcpPoseFkBridge(a.topic, a.frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
