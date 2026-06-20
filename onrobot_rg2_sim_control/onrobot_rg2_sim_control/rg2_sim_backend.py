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
  Isaac applies theta to rg2_gripper_joint, publishes /rg2_sim/joint_state (JointState) AND the geometric
  measured pad-to-pad gap /rg2_sim/pad_gap_m (Float64 m).
  here: contact_width(m) = MEASURED /rg2_sim/pad_gap_m (the TRUE physical gap; gates actual_valid).
        actuated_width(m) = arc(joint_state.theta) = DIAGNOSTIC (free-space-faithful, under-reports under
        contact-stall). raw/depth/theta also arc(theta) diagnostics. /gripper_width_sim = TRUE gap mm (compat).
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
        # MEASURED pad-to-pad gap (the TRUE contact width) from Isaac's geometry. contact_width must be
        # the physical rubber-pad-to-pad gap, NOT arc(theta): under contact-stall the force drive pushes
        # the (rigid) linkage ~2.45deg past the free-space position for the gap, so arc(theta) under-reports
        # the real gap (~0 at open, ~2mm at tight grips). Verified parity 0.10mm at proper grips (u_brown).
        self._last_pad_gap_m = None
        self._last_pad_t = None

        # command in: CONTACT gap in METERS
        self.create_subscription(Float64, "/rg2_sim/finger_width_cmd", self._on_cmd, 10)
        # joint feedback from Isaac
        self.create_subscription(JointState, "/rg2_sim/joint_state", self._on_js, 10)
        # MEASURED pad-to-pad gap from Isaac (geometric inner-pad-face distance) -> the TRUE contact width
        self.create_subscription(Float64, "/rg2_sim/pad_gap_m", self._on_pad_gap, 10)
        # joint target out: RADIANS
        self.pub_target = self.create_publisher(Float64, "/rg2_sim/joint_target", 10)
        # diagnostics / state (meters, except the deprecated mm compat)
        self.pub_contact = self.create_publisher(Float64, "/rg2_sim/contact_width", 10)   # m — TRUE measured pad gap
        self.pub_actuated = self.create_publisher(Float64, "/rg2_sim/actuated_width", 10) # m — arc(theta) DIAGNOSTIC (under-reports under contact-stall load)
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

    def _on_pad_gap(self, msg: Float64):
        # the measured physical pad-to-pad gap (meters) from Isaac -> the TRUE contact width
        self._last_pad_gap_m = float(msg.data)
        self._last_pad_t = self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        theta_fresh = (self._last_theta is not None and self._last_js_t is not None
                       and (now - self._last_js_t) <= STALE_SEC)
        pad_fresh = (self._last_pad_gap_m is not None and self._last_pad_t is not None
                     and (now - self._last_pad_t) <= STALE_SEC)
        # actual_valid gates the TRUE contact width (= measured pad gap). NEVER fall back to arc(theta) for
        # contact_width — arc(theta) under-reports under contact-stall; it is published as actuated_width only.
        self.pub_valid.publish(Bool(data=bool(pad_fresh)))

        # TRUE contact width = the measured rubber-pad-to-pad gap from Isaac
        if pad_fresh:
            g = float(self._last_pad_gap_m)
            self.pub_contact.publish(Float64(data=g))                    # m, TRUE measured gap
            self.pub_compat.publish(Float64(data=g * 1000.0))            # mm, deprecated compat (now the TRUE gap)

        # actuated-joint diagnostics (arc inference of the rg2_gripper_joint angle): faithful in free space,
        # under-reports the gap under contact-stall -> diagnostic only, NOT the contact width.
        if theta_fresh:
            th = self._last_theta
            self.pub_actuated.publish(Float64(data=self.arc.contact_of_theta(th) / 1000.0))  # m, arc(theta) diagnostic
            self.pub_raw.publish(Float64(data=self.arc.raw_width_mm(th) / 1000.0))
            self.pub_depth.publish(Float64(data=self.arc.depth_mm(th) / 1000.0))
            self.pub_theta.publish(Float64(data=th))


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
