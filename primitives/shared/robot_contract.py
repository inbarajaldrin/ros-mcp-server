"""Declared DroidStack robot names used by the cell's simulation mode.

Real-mode defaults remain the deployed UR controller names.  A DroidStack
sim run sets ``ROS_MCP_DROIDSTACK_SIM=1`` and may override the instance.
"""
from __future__ import annotations

import os

INSTANCE = os.environ.get("ROS_MCP_ROBOT_INSTANCE", "ur5e_0")
DROIDSTACK_SIM = os.environ.get("ROS_MCP_DROIDSTACK_SIM") == "1"

BARE_ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ARM_ACTION = (f"/{INSTANCE}/arm_controller/follow_joint_trajectory" if DROIDSTACK_SIM
              else "/scaled_joint_trajectory_controller/follow_joint_trajectory")
JOINT_STATES_TOPIC = f"/{INSTANCE}/joint_states" if DROIDSTACK_SIM else "/joint_states"
GRIP_ACTION = f"/{INSTANCE}/rg2/grip"
BASE_FRAME = f"{INSTANCE}_base_link" if DROIDSTACK_SIM else "base_link"
TOOL_FRAME = f"{INSTANCE}_tool0" if DROIDSTACK_SIM else "tool0"
WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench_sim"
RG2_WIDTH_COEFFICIENTS_M = (-0.013997, -0.025746, 0.097033, 0.047604)


def public_joint(name: str) -> str:
    return f"{INSTANCE}_{name}" if DROIDSTACK_SIM else name


def public_arm_joints() -> list[str]:
    return [public_joint(name) for name in BARE_ARM_JOINTS]


# Assigned after public_joint exists; kept as one declared name for subscribers.
GRIPPER_JOINT = public_joint("rg2_gripper_joint")


def rg2_width_m(joint_rad: float) -> float:
    """Evaluate the lab robot record's descending-power width polynomial."""
    answer = 0.0
    for coefficient in RG2_WIDTH_COEFFICIENTS_M:
        answer = answer * float(joint_rad) + coefficient
    return answer


def arm_positions(message):
    """Return the six bare-order arm positions from a public joint sample."""
    values = dict(zip(message.name, message.position))
    names = public_arm_joints()
    if any(name not in values for name in names):
        return None
    return [values[name] for name in names]
