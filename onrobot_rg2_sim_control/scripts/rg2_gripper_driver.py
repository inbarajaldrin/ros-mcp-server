#!/usr/bin/env python3
"""Curated husarion-RG2 gripper driver for the sim twin.

Persistent node: holds + smoothly ramps the gripper to a commanded CONTACT width (rubber
pad-to-pad, mm), mapped to rg2_gripper_joint through the CAD arc model, and publishes the
master + 5 mimic joints on /rg2ref/joint_states for the gripper robot_state_publisher.

Command it with std_msgs/Float64 (contact mm) on /rg2sim/gripper_cmd, or the rg2_gripper.py CLI.
Single persistent publisher -> no fighting, no /tmp one-shots, no pgrep kills.
"""
import sys
sys.path.insert(0, "/home/aaugus11/Documents/ros-mcp-server")
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from onrobot_rg2_sim_control.onrobot_rg2_sim_control.rg2_arc import Rg2Arc

MULT = {"rg2_gripper_joint": 1.0,
        "rg2_l_finger_2_joint": -1.0, "rg2_l_finger_passive_joint": 1.0,
        "rg2_r_finger_1_joint": -1.0, "rg2_r_finger_2_joint": 1.0, "rg2_r_finger_passive_joint": -1.0}
NAMES = list(MULT.keys())


class Rg2GripperDriver(Node):
    def __init__(self):
        super().__init__("rg2_gripper_driver")
        self.arc = Rg2Arc()
        self.rate_hz = 30.0
        self.ramp_mm_s = 250.0
        self.target = float(self.declare_parameter("start_contact_mm", 100.0).value)  # default open
        self.current = self.target
        self.pub = self.create_publisher(JointState, "/rg2ref/joint_states", 10)
        self.create_subscription(Float64, "/rg2sim/gripper_cmd", self._on_cmd, 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"rg2_gripper_driver up: cmd /rg2sim/gripper_cmd (CONTACT mm, 0..{self.arc.max_contact_mm():.1f}); "
            f"start {self.target:.0f}mm")

    def _on_cmd(self, msg: Float64):
        self.target = max(0.0, min(self.arc.max_contact_mm(), msg.data))
        self.get_logger().info(f"target contact = {self.target:.1f} mm "
                               f"(raw {self.arc.raw_of_contact(self.target):.1f})")

    def _tick(self):
        step = self.ramp_mm_s / self.rate_hz
        if abs(self.target - self.current) <= step:
            self.current = self.target
        else:
            self.current += step if self.target > self.current else -step
        theta = self.arc.theta_of_contact(self.current)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = NAMES
        js.position = [MULT[k] * theta for k in NAMES]
        self.pub.publish(js)


def main():
    rclpy.init()
    n = Rg2GripperDriver()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
