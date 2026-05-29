#!/usr/bin/env python3
"""Open / close the articulated RG2 in RViz (and any joint_state_publisher-driven
view) by publishing the master `finger_joint` angle on /rg2/gripper_command.

The joint_state_publisher launched by ur5e_with_rg2.launch.py evaluates the URDF
mimic joints, so commanding finger_joint alone moves both fingers symmetrically --
the ROS2 equivalent of the isaac-sim-mcp ur5e-dt RG2 open/close drive.

finger_joint convention (verified against the articulated URDF):
    -0.558505 rad  -> fully OPEN   (fingertip frames ~172 mm apart)
    +0.785398 rad  -> fully CLOSED (fingertip frames ~59 mm apart)

Usage:
    python3 rg2_gripper_command.py open
    python3 rg2_gripper_command.py close
    python3 rg2_gripper_command.py 0.0          # arbitrary angle (rad)
    python3 rg2_gripper_command.py demo         # cycle open<->close a few times
"""
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

OPEN = -0.558505
CLOSE = 0.785398


def _angle_from_arg(arg: str) -> float:
    a = arg.lower()
    if a == "open":
        return OPEN
    if a == "close":
        return CLOSE
    return float(arg)  # raw radians


class RG2Command(Node):
    def __init__(self):
        super().__init__("rg2_gripper_command")
        self.pub = self.create_publisher(JointState, "/rg2/gripper_command", 10)

    def send(self, angle: float, hold: float = 1.5):
        msg = JointState()
        msg.name = ["finger_joint"]
        msg.position = [float(angle)]
        end = time.time() + hold
        while time.time() < end:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            time.sleep(0.05)
        self.get_logger().info(f"finger_joint -> {angle:+.4f} rad")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rclpy.init()
    node = RG2Command()
    arg = sys.argv[1].lower()
    try:
        if arg == "demo":
            for _ in range(3):
                node.send(OPEN)
                node.send(CLOSE)
        else:
            node.send(_angle_from_arg(sys.argv[1]))
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
