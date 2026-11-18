#!/usr/bin/env python3
"""Drive the RViz tool model's RG2 fingers from the real gripper: /gripper_width (raw mm, onrobot_ros) ->
onrobot_rg2_control's arc model (Rg2Arc.theta_of_raw_width, the CAD 4-bar the lab's RG2 driver uses) -> the drive
joint + every mimic of it, on /tool/joint_states for the tool's robot_state_publisher (cell_tool_v31.urdf.xacro).

The cell runs onrobot_ros's gripper bridge, which publishes widths but no joint states; this is the read-only half of
onrobot_rg2_control's /joint_states path. It only subscribes and publishes joint states: it never commands the gripper.
The mimic tree is parsed from /tool/robot_description, as the driver parses its own.

    python3 compliant_insertion_studio/scripts/tool_joint_states.py
"""
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from onrobot_rg2_control.rg2_arc import Rg2Arc

DRIVE = 'tool_rg2_gripper_joint'


class ToolJointStates(Node):
    def __init__(self):
        super().__init__('tool_joint_states')
        self.arc = Rg2Arc()
        self.raw = None
        self.mimics = []
        self.create_subscription(Float32, '/gripper_width', self._on_width, 10)
        self.create_subscription(Float32, '/gripper_width_offset', self._on_gap, 10)
        self.create_subscription(String, '/tool/robot_description', self._on_description,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.pub = self.create_publisher(JointState, '/tool/joint_states', 10)
        self.create_timer(0.05, self._publish)

    def _on_width(self, msg):
        self.raw = float(msg.data)

    def _on_gap(self, msg):
        # fingertip offset = (raw - jaw gap) / 2: the device's own reg-258 value (4.6 mm on this RG2)
        if self.raw is not None and 0.0 < self.raw - msg.data < 20.0:
            self.arc.p.fingertip_offset_mm = (self.raw - float(msg.data)) / 2.0

    def _on_description(self, msg):
        mimics = []
        for j in ET.fromstring(msg.data).findall('joint'):
            m = j.find('mimic')
            if m is not None and m.get('joint') == DRIVE:
                mimics.append((j.get('name'), float(m.get('multiplier', '1')), float(m.get('offset', '0'))))
        self.mimics = mimics
        self.get_logger().info(f'{len(mimics)} mimics of {DRIVE} from /tool/robot_description')

    def _publish(self):
        if self.raw is None or not self.mimics:
            return
        th = self.arc.theta_of_raw_width(self.raw)
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [DRIVE] + [n for n, _, _ in self.mimics]
        msg.position = [th] + [k * th + o for _, k, o in self.mimics]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ToolJointStates()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
