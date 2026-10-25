#!/usr/bin/env python3
"""Move the flange in a straight Cartesian line to a target position (2026-10-14).

    python3 -m primitives.move_cartesian --mode real --xyz X Y Z [--yaw-deg Y] [--keep-orientation]

The safe general move: dense Cartesian waypoints solved by differential IK SEEDED FROM THE
CURRENT JOINTS (the method of move_to_safe_height and the calibration survey, ~150 moves on
2026-10-13/14), never the fixed-seed single-point move of primitives.core.move_xyz, which
swung the arm on 2026-10-07. Orientation: top-down at --yaw-deg, or the current orientation
with --keep-orientation. Refuses (no motion) when the IK path ends > 2 mm off target, any
joint would move more than --max-joint-step rad, or the flange would pass within
--min-radius of the base axis. Never cancels a running trajectory (that is a hard stop) —
except with --max-force: a sustained wrench change above it (median of 51 samples, ~100 ms:
the robot-side glitch is usually 4-6 ms but runs up to 28 ms, which dragged a 15-sample
median 20 N and stopped a descent at rest, 2026-10-14) stops the move. Pair it with --min-duration so that
stop happens at low speed (descents onto parts).
"""
import argparse
import json
import sys
import time
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from primitives.shared.config import SAFE_HEIGHT
from primitives.shared.ik import compute_cartesian_waypoints_ik, dh_params, forward_kinematics
from primitives.shared.velocity_profiles import compute_duration, s_curve_profile
WRIST3_ABS_MAX = 4.0             # rad: the wrist camera's cable (operator: a worn D415 cable/plug)

ACTION = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def result(d):
    print("__RESULT_JSON__"); print(json.dumps(d)); print("__END_RESULT_JSON__")


class Mover(Node):
    def __init__(self):
        super().__init__('move_cartesian')
        self.q = None
        self.create_subscription(JointState, '/joint_states', self._cb, 10)
        self.F = deque(maxlen=51)
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench',
                                 lambda m: self.F.append((m.wrench.force.x, m.wrench.force.y, m.wrench.force.z)), 50)
        self.ac = ActionClient(self, FollowJointTrajectory, ACTION)

    def _cb(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS])


def plan_move(n, xyz, orient=('keep', None), max_force=None, min_duration=0.0, min_z=SAFE_HEIGHT - 0.10,
              max_joint_step=1.5, max_wrist3_step=3.2, min_radius=0.25, duration=None, on_sample=None):
    """One straight Cartesian move on a live Mover node (the CLI's whole behaviour, 2026-10-22: callers that make many
    small moves - the touch insert's probes - keep one node instead of a process, rclpy init and joint wait per move).
    orient: ('quat', [qx, qy, qz, qw]) | ('keep', None) | ('yaw', deg or None). duration: exact seconds, for guarded
    probe moves of a few mm (otherwise compute_duration's floor, >= min_duration). on_sample(t, dF, F, q): called with
    each guarded-loop sample (s since the goal, |F - baseline|, F, joints) - the force record of a tap or press.
    Returns the result dict."""
    target = np.array(xyz, dtype=float)
    out = {"result": "failure", "mode": "real", "movement_type": "move_cartesian", "target": list(map(float, xyz))}
    if target[2] < min_z:
        out["error"] = f"target z {target[2]:.3f} below min_z {min_z:.3f}"; return out
    if np.hypot(target[0], target[1]) < min_radius:
        out["error"] = f"target within {min_radius} m of the base axis"; return out
    n.q = None                                           # this move's start: a fresh joint sample
    end = time.time() + 5
    while n.q is None and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    if n.q is None or not n.ac.wait_for_server(timeout_sec=10.0):
        out["error"] = "no joint states or trajectory action server"; return out
    q0 = n.q.copy(); T0 = forward_kinematics(dh_params, q0)
    kind, val = orient
    if kind == 'quat':
        Rt = R.from_quat(val).as_matrix()
    elif kind == 'keep':
        Rt = T0[:3, :3]
    else:
        yaw = val if val is not None else R.from_matrix(T0[:3, :3]).as_euler('xyz', degrees=True)[2]
        Rt = R.from_euler('xyz', [180.0, 0.0, yaw], degrees=True).as_matrix()
    # sample the straight line for the near-base guard
    for s in np.linspace(0, 1, 21):
        p = T0[:3, 3] + s * (target - T0[:3, 3])
        if np.hypot(p[0], p[1]) < min_radius:
            out["error"] = f"path passes within {min_radius} m of the base axis"; return out
    dist = float(np.linalg.norm(target - T0[:3, 3]))
    nw = int(np.clip(dist / 0.005, 30, 200))
    wps = compute_cartesian_waypoints_ik(q0, target[2], num_waypoints=nw, target_pos=target, target_orientation=Rt)
    if wps is None:
        out["error"] = "no Cartesian IK path"; return out
    Tend = forward_kinematics(dh_params, wps[-1])
    if np.linalg.norm(Tend[:3, 3] - target) > 0.002:
        out["error"] = f"IK end error {1000*np.linalg.norm(Tend[:3, 3]-target):.1f} mm"; return out
    dq = np.abs(np.array(wps[-1]) - q0)
    if float(np.max(dq[:5])) > max_joint_step:                      # arm swing: the 2026-10-07 lesson
        out["error"] = f"joint step {float(np.max(dq[:5])):.2f} rad > {max_joint_step}"; return out
    if abs(float(wps[-1][5])) > WRIST3_ABS_MAX:
        # 2026-10-20: views and pushes wound wrist_3 to 6.30 rad over the day (past its 2*pi limit: two
        # protective stops, the camera cable a full turn around the wrist). Every move was within its own step
        # limit; nothing bounded the total. Refuse instead.
        out["error"] = f"wrist_3 would end at {float(wps[-1][5]):+.2f} rad (|.| > {WRIST3_ABS_MAX}) - unwind first"; return out
    if float(dq[5]) > max_wrist3_step:
        out["error"] = f"wrist_3 step {float(dq[5]):.2f} rad > {max_wrist3_step}"; return out
    step = float(np.max(dq))                                           # the duration still follows the largest joint
    dur = (float(duration) if duration is not None
           else max(compute_duration(joint_distance=step, cartesian_distance=dist, profile='s_curve'), min_duration))
    F0 = None
    if max_force is not None:
        # Baseline only once the wrench is steady: two consecutive 0.15 s medians within 1 N.
        # A gripper still moving from the previous command tripped the guard at 14 N before
        # the arm had moved (2026-10-14).
        def window(sec):
            buf = []; t_end = time.time() + sec
            while time.time() < t_end:
                rclpy.spin_once(n, timeout_sec=0.005)
                if n.F:
                    buf.append(n.F[-1])
            return np.median(np.array(buf), axis=0) if len(buf) > 10 else None
        F0 = None; end = time.time() + 4.0
        prev = window(0.15)
        while time.time() < end:
            cur = window(0.15)
            if prev is not None and cur is not None and np.linalg.norm(cur - prev) < 1.0:
                F0 = cur; break
            prev = cur
        if F0 is None:
            out["error"] = "wrench not steady (or absent) for the force-guard baseline"; return out
    traj = JointTrajectory(); traj.joint_names = JOINTS
    for p, v, t in s_curve_profile([q0] + list(wps), dur):
        traj.points.append(JointTrajectoryPoint(positions=p, velocities=v,
                                                time_from_start=Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))))
    goal = FollowJointTrajectory.Goal(); goal.trajectory = traj; goal.goal_time_tolerance = Duration(sec=1)
    fut = n.ac.send_goal_async(goal); rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    gh = fut.result()
    if gh is None or not gh.accepted:
        out["error"] = "goal rejected"; return out
    res = gh.get_result_async()
    if F0 is None:
        rclpy.spin_until_future_complete(n, res, timeout_sec=dur + 15.0)
    else:
        end = time.time() + dur + 15.0; t_go = time.time()
        while not res.done() and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.005)
            Fm_ = np.median(np.array(n.F), axis=0)
            dF = float(np.linalg.norm(Fm_ - F0))
            out['dF_peak_N'] = round(max(out.get('dF_peak_N', 0.0), dF), 2)   # the push's peak load (side taps log it)
            if on_sample is not None:
                on_sample(time.time() - t_go, dF, Fm_, n.q)
            if dF > max_force:
                cf = gh.cancel_goal_async(); rclpy.spin_until_future_complete(n, cf, timeout_sec=3.0)
                T = forward_kinematics(dh_params, n.q)
                Fm = np.median(np.array(n.F), axis=0)
                out.update(error=f"contact: |dF| {dF:.1f} N > {max_force} N", contact=True,
                           t_after_start_s=round(time.time() - t_go, 3),
                           F_baseline=[round(float(v), 2) for v in F0], F_now=[round(float(v), 2) for v in Fm],
                           stopped_at=[round(float(v), 4) for v in T[:3, 3]])
                return out
    if res.result() is None or res.result().result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
        out["error"] = "trajectory did not complete"; return out
    out.update(result="success", duration_s=round(dur, 2), joint_step_rad=round(step, 3), distance_m=round(dist, 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='real', choices=['real'])
    ap.add_argument('--xyz', type=float, nargs=3, required=True, help='flange target, robot base frame (m)')
    ap.add_argument('--yaw-deg', type=float, default=None, help='top-down at this yaw (EE rpy z); default: current yaw')
    ap.add_argument('--keep-orientation', action='store_true', help='keep the current flange orientation')
    ap.add_argument('--quat', type=float, nargs=4, default=None, help='flange orientation qx qy qz qw')
    ap.add_argument('--min-z', type=float, default=SAFE_HEIGHT - 0.10, help='refuse targets below this flange z')
    ap.add_argument('--max-joint-step', type=float, default=1.5,
                    help='refuse if any of the five arm joints (shoulder..wrist_2) would move more than this (rad)')
    ap.add_argument('--max-wrist3-step', type=float, default=3.2,
                    help='the same cap for wrist_3 alone: it only spins the tool about its axis, so a turn up to '
                         '~180 deg is harmless (an 89 deg wrist turn hit the 1.5 rad arm cap, 2026-10-19)')
    ap.add_argument('--min-radius', type=float, default=0.25)
    ap.add_argument('--max-force', type=float, default=None, help='stop on a sustained |dF| above this (N)')
    ap.add_argument('--min-duration', type=float, default=0.0, help='slow the move to at least this (s)')
    ap.add_argument('--duration', type=float, default=None,
                    help='exact seconds, for a short move of a few mm only (a tap: nudge_part.py); refused over 10 mm')
    a = ap.parse_args()
    rclpy.init(); n = Mover()
    try:
        if a.quat is not None:
            orient = ('quat', a.quat)
        elif a.keep_orientation:
            orient = ('keep', None)
        else:
            orient = ('yaw', a.yaw_deg)
        if a.duration is not None:
            # an exact short duration only for a short move: a few mm in 0.3 s is a tap, a long one would be a lunge
            t0 = time.time()
            while n.q is None and time.time() - t0 < 3.0:
                rclpy.spin_once(n, timeout_sec=0.05)
            if n.q is None:
                result({'result': 'failure', 'error': 'no joint state'}); return 1
            cur = forward_kinematics(dh_params, n.q)[:3, 3]
            if float(np.linalg.norm(np.asarray(a.xyz) - cur)) > 0.010:
                result({'result': 'failure', 'error': '--duration is for moves of <= 10 mm'}); return 1
        out = plan_move(n, a.xyz, orient, max_force=a.max_force, min_duration=a.min_duration, min_z=a.min_z,
                        max_joint_step=a.max_joint_step, max_wrist3_step=a.max_wrist3_step, min_radius=a.min_radius,
                        duration=a.duration)
        result(out); return 0 if out.get('result') == 'success' else 1
    finally:
        n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
