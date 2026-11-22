"""R40, the operator's tool0 floor: one pure check, and the guards that apply it before a motion is sent.

R40 (operator 2026-11-21 ~08:00 on WhatsApp, confirmed ~13:50 MST in the terminal "yes keep both"; docs/PLAN.md "R40"):
tool0 must not go lower than the real cell ever flew.  From the lead's check over the 547 October real-run motion logs:
  - tool pointing down (tool z within TOP_DOWN_MAX_DEG of straight down): tool0 z >= 0.1478 m, the lowest top-down stop;
  - any other orientation: tool0 z >= -0.0002 m, the lowest stop of any kind (the horizontal clear-area set-down).
z is the UR base frame's (base and base_link differ by 180 deg about z: the same z, the same tool-axis tilt); tool0 is the
flange (/tcp_pose_broadcaster/pose on this cell, the FK of primitives.shared.ik).  The same constants and the same
top-down test as the board program's planners/boardport/checks.py (fmb-cell-lab), so both refuse the same poses.
The October real inserts bottomed at tool0 z 182.9 mm (u_brown): the floor refuses nothing the cell has flown.

Refused by name (R40_TOOL0_FLOOR), never clamped.  Enforced where ENV is '1' (drive-mate sets it for its motion
children, which pass it on): primitives.move_cartesian.plan_move, every FollowJointTrajectory goal of a process that
installed the send guard (goal_guard.init_ros primitives, _run_hover), and the force-mode loops of the two inserts.
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOOL0_FLOOR_TOP_DOWN_M = 0.1478      # R40: the lowest top-down tool0 stop in the October real-run logs
TOOL0_FLOOR_ANY_M = -0.0002          # R40: the lowest tool0 stop of any orientation (horizontal clear-area set-down)
TOP_DOWN_MAX_DEG = 10.0              # R40: "top-down" = tool z within this of straight down
CODE = 'R40_TOOL0_FLOOR'
ENV = 'R40_TOOL0_FLOOR'              # '1': enforce in this process (and, inherited, its children)
_EPS_M = 1e-7


def enforced() -> bool:
    return os.environ.get(ENV) == '1'


def is_top_down(T) -> bool:
    return float(np.asarray(T, float)[2, 2]) <= -np.cos(np.radians(TOP_DOWN_MAX_DEG))


def tool0_floor(T) -> float:
    """The R40 floor (m, UR base z) for a tool0 pose's orientation."""
    return TOOL0_FLOOR_TOP_DOWN_M if is_top_down(T) else TOOL0_FLOOR_ANY_M


def pose_matrix(xyz, quat_xyzw) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_quat([float(v) for v in quat_xyzw]).as_matrix()
    T[:3, 3] = [float(v) for v in xyz]
    return T


def r40_refusal(T_tool0) -> str | None:
    """None when the tool0 pose (4x4, UR base frame) keeps the R40 floor, else 'R40_TOOL0_FLOOR: ...'."""
    T = np.asarray(T_tool0, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return f'{CODE}: pose not checkable (not a finite 4x4)'
    z, f = float(T[2, 3]), tool0_floor(T)
    if f - z <= _EPS_M:
        return None
    return (f'{CODE}: tool0 z {1000 * z:.1f} mm < {1000 * f:.1f} mm '
            f"({'top-down' if is_top_down(T) else 'any orientation'}, tool z tilt "
            f'{np.degrees(np.arccos(np.clip(-T[2, 2], -1.0, 1.0))):.1f} deg; {1000 * (f - z):.1f} mm short)')


def path_refusal(poses, what='pose') -> str | None:
    """The refusal of the lowest offending pose of a sequence (named by its index), or None."""
    worst = None
    poses = list(poses)
    for k, T in enumerate(poses):
        T = np.asarray(T, float)
        why = r40_refusal(T)
        if why is None:
            continue
        short = tool0_floor(T) - float(T[2, 3]) if np.all(np.isfinite(T)) and T.shape == (4, 4) else np.inf
        if worst is None or short > worst[0]:
            worst = (short, k, why)
    if worst is None:
        return None
    return f'{worst[2]} at {what} {worst[1] + 1}/{len(poses)}'


def trajectory_refusal(joint_names, positions) -> str | None:
    """R40 over every point of a joint trajectory (FK of each point; the six UR arm joints by name, any prefix).  A
    trajectory whose arm joints cannot all be named is refused (not checkable).  The joint-space segment from the
    arm's current pose to the first point is not in the goal: senders here put the start in as point 1."""
    from primitives.shared.ik import dh_params, forward_kinematics
    from primitives.shared.robot_contract import BARE_ARM_JOINTS
    names = [str(n) for n in joint_names]
    idx = []
    for bare in BARE_ARM_JOINTS:
        hits = [i for i, n in enumerate(names) if n == bare or n.endswith('_' + bare) or n.endswith('/' + bare)]
        if len(hits) != 1:
            return f'{CODE}: trajectory not checkable (arm joint {bare} not named once in {names})'
        idx.append(hits[0])
    poses = []
    for p in positions:
        p = list(p)
        if len(p) != len(names):
            return f'{CODE}: trajectory not checkable ({len(p)} positions for {len(names)} joints)'
        poses.append(forward_kinematics(dh_params, [float(p[i]) for i in idx]))
    if not poses:
        return f'{CODE}: trajectory not checkable (no points)'
    return path_refusal(poses, 'trajectory point')


def log_refusal(why: str, where: str) -> None:
    print(f'[R40] {where}: {why}', file=sys.stderr, flush=True)


class Tool0FloorRefused(RuntimeError):
    """An arm goal R40 refused before it was sent: nothing moved."""


def _is_arm_goal(client) -> bool:
    action_type = getattr(client, '_action_type', None)
    return getattr(action_type, '__name__', '') == 'FollowJointTrajectory'


def goal_refusal(client, goal) -> str | None:
    """R40 on one action goal: None for a non-arm goal or with ENV off; else the trajectory's refusal."""
    if not enforced() or not _is_arm_goal(client):
        return None
    traj = getattr(goal, 'trajectory', None)
    if traj is None:
        return f'{CODE}: arm goal without a trajectory - not checkable'
    return trajectory_refusal(traj.joint_names, [pt.positions for pt in traj.points])


_send_guard_installed = False


def install_send_guard() -> None:
    """Check every FollowJointTrajectory goal of this process before rclpy sends it (ENV read at each send).  A
    refused goal raises Tool0FloorRefused from send_goal_async: it never reaches the controller."""
    global _send_guard_installed
    if _send_guard_installed:
        return
    from rclpy.action import ActionClient
    original_send = ActionClient.send_goal_async

    def send_goal_async(self, goal, *args, **kwargs):
        why = goal_refusal(self, goal)
        if why:
            log_refusal(why, 'arm goal NOT sent')
            raise Tool0FloorRefused(why)
        return original_send(self, goal, *args, **kwargs)

    ActionClient.send_goal_async = send_goal_async
    _send_guard_installed = True
