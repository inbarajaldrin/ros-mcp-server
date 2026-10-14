#!/usr/bin/env python3
"""Move the flange in a straight Cartesian line to a target position (2026-10-14).

    python3 -m primitives.move_cartesian --mode real --xyz X Y Z [--yaw-deg Y] [--keep-orientation]

The safe general move: dense Cartesian waypoints solved by differential IK SEEDED FROM THE
CURRENT JOINTS (the method of move_to_safe_height and the calibration survey, ~150 moves on
2026-10-13/14), never the fixed-seed single-point move of primitives.core.move_xyz, which
swung the arm on 2026-10-07. Orientation: top-down at --yaw-deg, or the current orientation
with --keep-orientation. Refuses (no motion) when the IK path ends > 2 mm off target, any
joint would move more than --max-joint-step rad, or the flange would pass within
--min-radius of the base axis. Never cancels a running trajectory (that is a hard stop).
"""
import argparse
import json
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from primitives.shared.config import SAFE_HEIGHT
from primitives.shared.ik import compute_cartesian_waypoints_ik, dh_params, forward_kinematics
from primitives.shared.velocity_profiles import compute_duration, s_curve_profile

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
        self.ac = ActionClient(self, FollowJointTrajectory, ACTION)

    def _cb(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in JOINTS):
            self.q = np.array([d[j] for j in JOINTS])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='real', choices=['real'])
    ap.add_argument('--xyz', type=float, nargs=3, required=True, help='flange target, robot base frame (m)')
    ap.add_argument('--yaw-deg', type=float, default=None, help='top-down at this yaw (EE rpy z); default: current yaw')
    ap.add_argument('--keep-orientation', action='store_true', help='keep the current flange orientation')
    ap.add_argument('--min-z', type=float, default=SAFE_HEIGHT - 0.10, help='refuse targets below this flange z')
    ap.add_argument('--max-joint-step', type=float, default=1.5)
    ap.add_argument('--min-radius', type=float, default=0.25)
    a = ap.parse_args()
    target = np.array(a.xyz, dtype=float)
    out = {"result": "failure", "mode": "real", "movement_type": "move_cartesian", "target": a.xyz}
    if target[2] < a.min_z:
        out["error"] = f"target z {target[2]:.3f} below min_z {a.min_z:.3f}"; result(out); return 1
    if np.hypot(target[0], target[1]) < a.min_radius:
        out["error"] = f"target within {a.min_radius} m of the base axis"; result(out); return 1

    rclpy.init(); n = Mover()
    try:
        end = time.time() + 5
        while n.q is None and time.time() < end:
            rclpy.spin_once(n, timeout_sec=0.05)
        if n.q is None or not n.ac.wait_for_server(timeout_sec=10.0):
            out["error"] = "no joint states or trajectory action server"; result(out); return 1
        q0 = n.q.copy(); T0 = forward_kinematics(dh_params, q0)
        if a.keep_orientation:
            Rt = T0[:3, :3]
        else:
            yaw = a.yaw_deg if a.yaw_deg is not None else R.from_matrix(T0[:3, :3]).as_euler('xyz', degrees=True)[2]
            Rt = R.from_euler('xyz', [180.0, 0.0, yaw], degrees=True).as_matrix()
        # sample the straight line for the near-base guard
        for s in np.linspace(0, 1, 21):
            p = T0[:3, 3] + s * (target - T0[:3, 3])
            if np.hypot(p[0], p[1]) < a.min_radius:
                out["error"] = f"path passes within {a.min_radius} m of the base axis"; result(out); return 1
        dist = float(np.linalg.norm(target - T0[:3, 3]))
        nw = int(np.clip(dist / 0.005, 30, 200))
        wps = compute_cartesian_waypoints_ik(q0, target[2], num_waypoints=nw, target_pos=target, target_orientation=Rt)
        if wps is None:
            out["error"] = "no Cartesian IK path"; result(out); return 1
        Tend = forward_kinematics(dh_params, wps[-1])
        if np.linalg.norm(Tend[:3, 3] - target) > 0.002:
            out["error"] = f"IK end error {1000*np.linalg.norm(Tend[:3, 3]-target):.1f} mm"; result(out); return 1
        step = float(np.max(np.abs(np.array(wps[-1]) - q0)))
        if step > a.max_joint_step:
            out["error"] = f"joint step {step:.2f} rad > {a.max_joint_step}"; result(out); return 1
        dur = compute_duration(joint_distance=step, cartesian_distance=dist, profile='s_curve')
        traj = JointTrajectory(); traj.joint_names = JOINTS
        for p, v, t in s_curve_profile([q0] + list(wps), dur):
            traj.points.append(JointTrajectoryPoint(positions=p, velocities=v,
                                                    time_from_start=Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))))
        goal = FollowJointTrajectory.Goal(); goal.trajectory = traj; goal.goal_time_tolerance = Duration(sec=1)
        fut = n.ac.send_goal_async(goal); rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            out["error"] = "goal rejected"; result(out); return 1
        res = gh.get_result_async(); rclpy.spin_until_future_complete(n, res, timeout_sec=dur + 15.0)
        if res.result() is None or res.result().result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            out["error"] = "trajectory did not complete"; result(out); return 1
        out.update(result="success", duration_s=round(dur, 2), joint_step_rad=round(step, 3), distance_m=round(dist, 4))
        result(out); return 0
    finally:
        n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
