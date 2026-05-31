#!/usr/bin/env python3
"""
get_robot_state.py — one-shot snapshot of the robot's kinematic state.

Reads the current joint positions (/joint_states), gripper width (mode-aware
topic), and end-effector pose (/tcp_pose_broadcaster/pose), prints one JSON
result, and exits. Intended to be called by the server BEFORE and AFTER each
primitive tool call (observe-only timing/IK baseline) — it does NOT command any
motion, so it cannot perturb a deterministic replay.

Usage:  python3 get_robot_state.py --mode {sim|real}

Output (between __RESULT_JSON__ markers):
  {"joints_rad": [6], "joints_deg": [6], "gripper_mm": float|null,
   "ee_pos": [x,y,z]|null, "ee_quat": [x,y,z,w]|null, "have": {...}}
Fields are null / omitted from `have` when the topic did not publish within the
timeout (fail-loud: the gap is explicit, never a fabricated default).
"""

import argparse
import json
import math

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Float32
from tf2_msgs.msg import TFMessage

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def output_result(result):
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "real"], required=True)
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="Max seconds to wait for all topics (default 3).")
    args = ap.parse_args()

    gripper_topic = "/gripper_width_sim" if args.mode == "sim" else "/gripper_width_offset"
    gripper_type = Float64 if args.mode == "sim" else Float32
    objects_topic = "/objects_poses_sim" if args.mode == "sim" else "/objects_poses_real"

    rclpy.init()
    node = rclpy.create_node("robot_state_reader")

    # objects: the IK PREMISE — each object's pose (position + full orientation) relative
    # to the arm, which the primitives turn into the IK target. Keyed by object name.
    state = {"joints": None, "gripper": None, "ee_pos": None, "ee_quat": None, "objects": {}}

    def joint_cb(msg):
        if len(msg.name) >= 6 and len(msg.position) >= 6:
            jd = dict(zip(msg.name, msg.position))
            if all(n in jd for n in JOINT_NAMES):
                state["joints"] = [float(jd[n]) for n in JOINT_NAMES]

    def gripper_cb(msg):
        state["gripper"] = float(msg.data)

    def ee_cb(msg):
        state["ee_pos"] = [float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z)]
        state["ee_quat"] = [float(msg.pose.orientation.x), float(msg.pose.orientation.y),
                            float(msg.pose.orientation.z), float(msg.pose.orientation.w)]

    def objects_cb(msg):
        # TFMessage: one transform per object, child_frame_id = object name.
        for tf in msg.transforms:
            name = tf.child_frame_id
            t, q = tf.transform.translation, tf.transform.rotation
            state["objects"][name] = {
                "pos": [float(t.x), float(t.y), float(t.z)],
                "quat": [float(q.x), float(q.y), float(q.z), float(q.w)],
            }

    volatile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE, depth=10)
    node.create_subscription(JointState, "/joint_states", joint_cb, 10)
    node.create_subscription(gripper_type, gripper_topic, gripper_cb, 10)
    node.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", ee_cb, volatile)
    node.create_subscription(TFMessage, objects_topic, objects_cb, 5)

    # Spin until all three arrive or timeout. Joints are the critical field.
    import time as _t
    t0 = _t.monotonic()
    try:
        while rclpy.ok() and (_t.monotonic() - t0) < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["joints"] is not None and state["gripper"] is not None and state["ee_pos"] is not None:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    joints = state["joints"]
    objects = state["objects"] or None
    result = {
        "joints_rad": joints,
        "joints_deg": ([round(math.degrees(x), 3) for x in joints] if joints else None),
        "gripper_mm": state["gripper"],
        "ee_pos": state["ee_pos"],
        "ee_quat": state["ee_quat"],
        "objects": objects,  # {name: {pos:[x,y,z], quat:[x,y,z,w]}} — the IK premise
        "have": {
            "joints": joints is not None,
            "gripper": state["gripper"] is not None,
            "ee": state["ee_pos"] is not None,
            "objects": bool(objects),
        },
        "mode": args.mode,
    }
    output_result(result)


if __name__ == "__main__":
    main()
