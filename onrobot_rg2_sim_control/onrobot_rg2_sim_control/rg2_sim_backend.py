#!/usr/bin/env python3
"""
RG2 sim backend (sim-first Python path) — the CONTACT-width control seam for Isaac.

GPT-gated (both sides). Fixes baked in:
  #1  open limit DERIVED from the active offset (rg2_arc.theta_at_raw_max) — never the YAML constant.
  #2  command interface = CONTACT (rubber pad-to-pad) gap, clamped to [0, max_contact=100.2mm]. NOT raw 0-110.
  #3  /rg2_sim/joint_target = std_msgs/Float64 RADIANS; reading /rg2_sim/joint_state selects
      'rg2_gripper_joint' BY NAME (not index 0); /gripper_width_sim stays deprecated-compat (contact mm),
      never fed back into control; NO fallback-to-commanded — actual state is STALE until a fresh joint_state.

Units are explicit at every seam: ROS topics = METERS / radians; the arc module works in mm internally.

Pipeline:
  /rg2_sim/finger_width_cmd (Float64, CONTACT m)  --arc.theta_of_contact-->  /rg2_sim/joint_target (Float64 rad)
  Isaac applies theta to rg2_gripper_joint, publishes /rg2_sim/joint_state (JointState)
  here: joint_state.theta --arc--> contact/raw/depth  ->  /rg2_sim/{contact_width(m),raw_width(m),depth(m),
        theta(rad), actual_valid(Bool)} + /gripper_width_sim (Float64, contact mm, compat)
"""
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import JointState

from onrobot_rg2_sim_control.onrobot_rg2_sim_control.rg2_arc import Rg2Arc, Rg2ArcParams

JOINT = "rg2_gripper_joint"
STALE_SEC = 0.5
DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "rg2_gripper.yaml")


class Rg2SimBackend(Node):
    def __init__(self):
        super().__init__("rg2_sim_backend")
        yaml_path = self.declare_parameter("gripper_yaml", DEFAULT_YAML).value
        try:
            self.arc = Rg2Arc(Rg2ArcParams.from_yaml(yaml_path))
            self.get_logger().info(f"loaded arc params from {yaml_path}")
        except Exception as e:
            self.arc = Rg2Arc()
            self.get_logger().warn(f"yaml load failed ({e}); using built-in CAD defaults")

        # state (STALE until first fresh joint_state; never fall back to commanded — fix #3/#6)
        self._last_theta = None
        self._last_js_t = None

        # command in: CONTACT gap in METERS
        self.create_subscription(Float64, "/rg2_sim/finger_width_cmd", self._on_cmd, 10)
        # joint feedback from Isaac
        self.create_subscription(JointState, "/rg2_sim/joint_state", self._on_js, 10)
        # joint target out: RADIANS
        self.pub_target = self.create_publisher(Float64, "/rg2_sim/joint_target", 10)
        # diagnostics / state (meters, except the deprecated mm compat)
        self.pub_contact = self.create_publisher(Float64, "/rg2_sim/contact_width", 10)   # m
        self.pub_raw = self.create_publisher(Float64, "/rg2_sim/raw_width", 10)           # m
        self.pub_depth = self.create_publisher(Float64, "/rg2_sim/depth", 10)             # m
        self.pub_theta = self.create_publisher(Float64, "/rg2_sim/theta", 10)             # rad
        self.pub_valid = self.create_publisher(Bool, "/rg2_sim/actual_valid", 10)
        self.pub_compat = self.create_publisher(Float64, "/gripper_width_sim", 10)        # contact MM (deprecated)

        self.create_timer(1.0 / 20.0, self._tick)
        self.get_logger().info(
            f"rg2_sim_backend up: cmd /rg2_sim/finger_width_cmd (CONTACT m, max "
            f"{self.arc.max_contact_mm()/1000.0:.4f}m) -> /rg2_sim/joint_target (rad)")

    def _on_cmd(self, msg: Float64):
        contact_mm = msg.data * 1000.0                       # m -> mm (explicit seam)
        theta = self.arc.theta_of_contact(contact_mm)        # clamps to [0, max_contact] then arc inverse
        self.pub_target.publish(Float64(data=float(theta)))  # rad

    def _on_js(self, msg: JointState):
        try:
            i = list(msg.name).index(JOINT)                  # BY NAME, not index 0 (fix #3)
        except ValueError:
            return
        self._last_theta = float(msg.position[i])
        self._last_js_t = self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        fresh = (self._last_theta is not None and self._last_js_t is not None
                 and (now - self._last_js_t) <= STALE_SEC)
        self.pub_valid.publish(Bool(data=bool(fresh)))
        if not fresh:
            return  # STALE -> publish no actual values (never fall back to commanded)
        th = self._last_theta
        contact_mm = self.arc.contact_of_theta(th)
        self.pub_contact.publish(Float64(data=contact_mm / 1000.0))      # m
        self.pub_raw.publish(Float64(data=self.arc.raw_width_mm(th) / 1000.0))
        self.pub_depth.publish(Float64(data=self.arc.depth_mm(th) / 1000.0))
        self.pub_theta.publish(Float64(data=th))
        self.pub_compat.publish(Float64(data=contact_mm))                # mm, deprecated compat


def main():
    rclpy.init()
    node = Rg2SimBackend()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
