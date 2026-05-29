#!/usr/bin/env python3
"""Bridge the established /gripper_command (std_msgs/String) interface to the
articulated RG2 in RViz.

Reference (command contract): isaac-sim-mcp ur5e-dt extension
`_on_physics_step_gripper` + primitives/control_gripper.py. Both speak:

    /gripper_command  (std_msgs/String):
        "open"   -> 100 mm
        "close"  ->   0 mm
        "<n>"    -> float(n)/10 mm     (control_gripper sends width*10; e.g. "350"=35mm)
        "" / "None" / "0" -> no-op     (matches Isaac's parser exactly)

This node makes ONE command drive both the Isaac sim gripper and this RViz model:
  * subscribes /gripper_command, maps width(mm) -> finger_joint(rad), ramps smoothly,
    and publishes sensor_msgs/JointState{finger_joint} on /rg2/gripper_command
    (the rg2 joint_state_publisher evaluates the 5 URDF mimic joints from it).
  * echoes /gripper_width_sim (std_msgs/Float64, mm) at a steady rate so
    control_gripper.py's width-feedback verification works against the RViz-only
    stack too (no Isaac required).

width(mm) -> finger_joint(rad): linear over the URDF joint limits
    0 mm  (closed) -> +0.785398   (finger_joint upper limit)
  100 mm (open)    -> -0.558505   (finger_joint lower limit)

This is the width-layer front-end for the articulated core. To drive the core
directly with finger_joint angles instead (the low-level path), launch with
gripper_bridge:=false and use scripts/rg2_gripper_command.py.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState

WIDTH_MAX = 100.0
FJ_CLOSE = 0.785398    # width 0 mm
FJ_OPEN = -0.558505    # width 100 mm


def width_to_finger_joint(width_mm: float) -> float:
    w = max(0.0, min(WIDTH_MAX, width_mm))
    return FJ_CLOSE + (w / WIDTH_MAX) * (FJ_OPEN - FJ_CLOSE)


class RG2CommandBridge(Node):
    def __init__(self):
        super().__init__("rg2_command_bridge")
        self.rate_hz = 50.0
        self.ramp_mm_per_s = 250.0   # ~0.4 s for a full 100 mm stroke
        self.current_mm = 100.0      # start open
        self.target_mm = 100.0

        self.js_pub = self.create_publisher(JointState, "/rg2/gripper_command", 10)
        self.width_pub = self.create_publisher(Float64, "/gripper_width_sim", 10)
        self.create_subscription(String, "/gripper_command", self._on_cmd, 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            "RG2 bridge up: /gripper_command (String) -> /rg2/gripper_command "
            "(finger_joint) + /gripper_width_sim echo")

    def _on_cmd(self, msg: String):
        s = msg.data.strip()
        if s in ("", "None", "0"):
            return  # no-op, matches Isaac's parser
        if s == "open":
            w = 100.0
        elif s == "close":
            w = 0.0
        else:
            try:
                w = float(s) / 10.0
            except ValueError:
                self.get_logger().warn(f"unparseable /gripper_command: {s!r}")
                return
        self.target_mm = max(0.0, min(WIDTH_MAX, w))
        self.get_logger().info(f"/gripper_command {s!r} -> target {self.target_mm:.1f} mm")

    def _tick(self):
        step = self.ramp_mm_per_s / self.rate_hz
        if abs(self.target_mm - self.current_mm) <= step:
            self.current_mm = self.target_mm
        else:
            self.current_mm += step if self.target_mm > self.current_mm else -step

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ["finger_joint"]
        js.position = [width_to_finger_joint(self.current_mm)]
        self.js_pub.publish(js)
        self.width_pub.publish(Float64(data=float(self.current_mm)))


def main():
    rclpy.init()
    node = RG2CommandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
