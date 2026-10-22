#!/usr/bin/env python3
"""
Prismatic Peg-in-Hole Insertion
================================

Trajectory approach to target -> force-mode insertion with tilt-responsive descent.

Slows down on high torque/lateral forces, aborts on sustained tilt.
Retries with retract + re-approach if insertion fails.
#ros2 control switch_controllers --deactivate passthrough_trajectory_controller force_mode_controller --activate scaled_joint_trajectory_controller
"""

import argparse
import json
import os
import numpy as np
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Wrench, WrenchStamped, Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from ur_msgs.srv import SetForceMode
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation as R

_project_root = str(Path(__file__).parent.parent.parent)  # primitives/_real_mode_stash/<file> → repo root
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET, held_tool_offset, held_tool_offset_for
BAR_ALONG_MAX_M = 0.030   # a bar may slide this far along its free axis while the compliance finds the slot
from primitives.shared.ik import compute_cartesian_waypoints_ik
from utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir, get_symmetry_dir, find_assembly_json_by_base_name

# ============================================================================
# CONFIGURATION
# ============================================================================

# Force mode (matched to legacy peg_in_hole_insert)
TRAJECTORY_FORCE_Z = 2.0        # Light force during trajectory (N) — just enough for Z compliance, gentle contact
DEFAULT_FORCE_Z = 5.0           # Downward force during force-mode descent (N)
SLOW_FORCE_Z = 5.0              # Same force when disturbed — legacy doesn't change force
SETTLING_FORCE_Z = 2.0          # Light force at depth — just enough to keep contact, not jam tilt (N)

# Speed limits - descent (matched to legacy)
NORMAL_Z_SPEED = 0.03           # m/s - normal descent
SLOW_Z_SPEED = 0.03             # m/s - same as normal (legacy uses fixed 0.03)
COMPLIANCE_XY_SPEED = 0.005     # m/s - very tight lateral (keeps peg centered over hole)
COMPLIANCE_RX_RY_SPEED = 0.1    # rad/s - slow tilt compliance (legacy value when enabled)
COMPLIANCE_RZ_SPEED = 0.05      # rad/s - yaw compliance (legacy value)

# Speed limits - settling at depth
SETTLING_RX_RY_SPEED = 0.02     # rad/s - tilt correction at depth (was 0.1; tightened
                                 # 2026-05-07 to kill ~13s of post-at-target oscillation
                                 # observed in line_green / inverted_u_yellow runs)
SETTLING_XY_SPEED = 0.005       # m/s - tight lateral (peg is in hole)

# Torque thresholds (hysteresis: trigger > recovery to avoid oscillation)
TORQUE_TRIGGER_TX = 0.10        # N-m - roll tilt trigger
TORQUE_TRIGGER_TY = 0.10        # N-m - pitch tilt trigger
TORQUE_RECOVERY_TX = 0.04       # N-m - roll recovery
TORQUE_RECOVERY_TY = 0.04       # N-m - pitch recovery

# Lateral force thresholds
LATERAL_TRIGGER = 6.0           # N - detect wall contact
LATERAL_RECOVERY = 3.0          # N - wall contact cleared

# Tilt timing
TILT_RECOVERY_TIME = 0.5        # seconds - sustain low torque before resuming normal speed
MAX_CONTINUOUS_TILT_TIME = 20.0 # seconds - abort if tilt persists (global timeout is the real safety net)

# Stuck detection — escalate compliance when Z stops progressing
STUCK_WINDOW = 3.0              # seconds — check Z progress over this window
STUCK_THRESHOLD = 0.0005        # m — must move at least 0.5mm per window to count as progress

# Trajectory
APPROACH_SPEED = 0.02           # m/s — trajectory descent speed (duration = distance / speed)
DEFAULT_NUM_WAYPOINTS = 60
SETTLING_TIME = 0.5             # seconds - ignore sensor data after controller switch

# Force zero (2026-10-07, mirrors compliant_insert's DEFAULT_BIAS_ABORT_N): zero only once the
# measured TCP is still, then refuse force mode if the post-zero residual is a phantom load.
ZERO_STILL_WINDOW_S = 0.5       # TCP must stay within ZERO_STILL_TOL_M over this window
ZERO_STILL_TOL_M = 0.0002
ZERO_STILL_TIMEOUT_S = 5.0
ZERO_RESIDUAL_ABORT_N = 5.0
ZERO_GOOD_N = 1.0               # median residual of a good zero is 0.05-0.40 N (2026-10-14, 80 trials)
ZERO_EFFECT_LATENCY_S = 0.018   # zero_ftsensor takes effect ~18 ms after the call (std 15 ms)
ZERO_GLITCH_PERIOD_S = 0.102    # robot-side wrench glitch period

# Exit criteria
Z_TOLERANCE = 0.005             # m — 5mm tolerance (peg may not reach exact target due to tilt/friction)
EXIT_TORQUE_AVG_THRESHOLD = 0.09  # N-m - rolling average Tx/Ty must be below this to exit
EXIT_TORQUE_WINDOW = 0.5        # seconds - rolling average window
EXIT_SUSTAINED_TIME = 1.0       # seconds - Z at target + low avg torques must hold this long

# 2026-05-07: GEOMETRIC EXIT (option 1, replaces torque-based exit for line_green
# specifically). When the gripper jaws rest on base outer rim while only the bar
# tip is in the slot, sustained jaws-on-rim contact creates persistent Tx/Ty far
# above EXIT_TORQUE_AVG_THRESHOLD. The torque check then never fires even when
# the part is correctly seated. Geometric exit replaces "low torque" with
# "depth reached + low Fz + zero z-velocity":
# 2026-10-14: 1.0 -> 2.5 mm. inverted_u_yellow sat seated and still at +0.99..+1.03 mm for two
# 60 s attempts (the 1 s hold reset each time noise crossed 1.0 mm); u_orange seated at +0.84.
# Good seats land within +-1.7 mm (CLAUDE.md); a part on the rim sits many mm higher.
EXIT_GEOMETRIC_Z_TOL_ABOVE_M = 0.0025  # above predicted (rim contact = false success)
EXIT_GEOMETRIC_Z_TOL_BELOW_M = 0.0080  # 8 mm below predicted (going deeper = real seat)
EXIT_GEOMETRIC_FZ_MAX_N      = 4.0     # |Fz| must be light (no hard wedge)
EXIT_GEOMETRIC_DZ_DT_MAX     = 0.0005  # 0.5 mm/s — z-velocity ≈ 0 (stopped descending)
# 2026-05-07: 1.5s window matches compliant_insert.contact_search_fsm
# `insert_motion_window_s` — micro-oscillation under settling compliance
# averages out over this window. 0.5s was too short; spikes reset every cycle.
EXIT_GEOMETRIC_VEL_WINDOW    = 1.5     # seconds for dz/dt estimation

# Deviation limits
# 2026-05-07: tightened from 25mm/6° to 5mm/3°. With the calibrated
# PER_OBJECT_BASE_OFFSET_M, real seats land within 3-5mm of predicted; the
# old 25mm let line_green declare success at 8.95mm pos_dev (clearly tilted
# on the rim, not seated). 5mm forces a real seat or triggers retry.
POSITION_DEVIATION_XY = 0.005   # m - max XY drift from target at verify
SEATED_XY_ACCEPT_M = 0.015      # m - at seat depth, an xy offset up to this is a found slot, not an exit
DRIFT_EXEMPT_DROP_M = 0.005     # m - this far below the first [CONTACT] z: in the slot, drift allowed
ORIENTATION_DEVIATION = 3.0     # degrees - max orientation change at verify

# Retry
MAX_RETRIES = int(__import__('os').environ.get('PRISMATIC_MAX_RETRIES', '3'))   # the replay runs 1 and re-reads the base between (2026-10-20)
# Retry aim (2026-10-19): a stall above the seat with the part slid off the aim is an xy error, not a
# compliance problem — the pockets are sub-mm (CAD: yellow x +-0.6 / y +-0.5 mm; feet round
# line_green +-1 mm), so the same descent at the same xy ends the same way (3 attempts, 5.0 mm each,
# yellow stuck 32 mm up). Attempt 2 aims where the part slid to (aim + measured deviation), attempt 3
# the mirror; the shift is only taken between these bounds. Rz/tilt searches do not help here.
RETRY_SHIFT_MIN_M = 0.0015      # below: the stall is not an xy miss (noise / a seated part), no shift
RETRY_SHIFT_MAX_M = 0.010       # above: more than the aim or grasp can be off (half a part thickness) -> no shift
STAGE_MATCH_TOL_M = 0.008       # a stall within this of a CAD contact height is reported as that stage
# Corridor stage (INSERT_GEOMETRY.md, 2026-10-19): a stall BELOW a seated part's top is the part wedged
# inside that part's opening (line_green 5-9 mm into the U openings, +-1.1 mm clearance; yellow's legs
# astride line_green). The old retry retracted 10 mm above the stall - still inside the opening - and
# locked Rz, so attempts 2 and 3 restarted wedged (3 x 5.0 mm, 20:56 and 21:12). A corridor retry
# retracts above the entrance, keeps Rz compliant and re-aims by at most the clearance scale.
CORRIDOR_DEPTH_M = 0.020        # how far below a seated part's top counts as inside its opening
CORRIDOR_EXIT_M = 0.008         # retract to this above the entrance before the next attempt
CORRIDOR_SHIFT_MAX_M = 0.002    # re-aim bound inside a corridor (the measured slide is mostly the wedge pivot)
DEFAULT_TIMEOUT = 60.0

# ============================================================================

SCALED_CONTROLLER = 'scaled_joint_trajectory_controller'
PASSTHROUGH_CONTROLLER = 'passthrough_trajectory_controller'
FORCE_MODE_CONTROLLER = 'force_mode_controller'

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]


def switch_controllers(activate, deactivate):
    """Switch ROS2 controllers."""
    result = subprocess.run(['ros2', 'control', 'list_controllers'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print(f"ERROR: list_controllers failed: {result.stderr}")
        return False

    # The state is the LAST column; `'active' in line` also matched "inactive", so every
    # controller looked active and nothing was ever switched on (2026-10-14).
    import re
    active = set()
    for line in result.stdout.splitlines():
        parts = re.sub(r'\x1b\[[0-9;]*m', '', line).split()
        if len(parts) >= 2 and parts[-1] == 'active':
            active.add(parts[0])

    to_activate = [c for c in activate if c not in active]
    to_deactivate = [c for c in deactivate if c in active]

    if not to_activate and not to_deactivate:
        return True

    cmd = ['ros2', 'control', 'switch_controllers']
    if to_activate:
        cmd += ['--activate'] + to_activate
    if to_deactivate:
        cmd += ['--deactivate'] + to_deactivate

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print(f"ERROR: Controller switch failed: {result.stderr}")
    return result.returncode == 0


def restore_controllers():
    """Restore scaled_joint_trajectory_controller."""
    print("[CLEANUP] Restoring controllers...")
    # One switch: the driver refuses scaled_joint_trajectory_controller while passthrough is
    # still active ("Starting position together with trajectory_passthrough is not allowed"),
    # and the old two-call version left the arm uncommandable when the second call was rejected.
    if switch_controllers([SCALED_CONTROLLER], [PASSTHROUGH_CONTROLLER, FORCE_MODE_CONTROLLER]):
        print("[CLEANUP] Controllers restored")
    else:
        print("[CLEANUP] ERROR restoring controllers (stop_run.sh restores them)")


def load_grasp_point_position(object_name, grasp_id):
    """Load grasp point position from grasp points JSON."""
    data_dir = Path(get_aruco_data_dir()) / "grasp_points"
    json_path = data_dir / f"{object_name}_grasp_points.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for gp in data.get('grasp_points', []):
                if gp['id'] == grasp_id:
                    pos = gp['position']
                    return np.array([pos['x'], pos['y'], pos['z']])
        except Exception:
            pass
    return None


def build_trapezoidal_trajectory(all_q, duration):
    """Build a JointTrajectory with trapezoidal velocity profile.

    Args:
        all_q: List of joint configurations (including initial position as first element).
        duration: Total trajectory duration in seconds.

    Returns:
        JointTrajectory message.
    """
    n = len(all_q)

    # Arc-length parameterization
    segment_dists = [max(np.linalg.norm(all_q[i] - all_q[i - 1]), 1e-6) for i in range(1, n)]
    cumulative_s = [0.0]
    for d in segment_dists:
        cumulative_s.append(cumulative_s[-1] + d)
    total_s = cumulative_s[-1]

    # Trapezoidal profile: 20% accel, 60% cruise, 20% decel
    t_accel = 0.2 * duration
    t_decel = 0.2 * duration
    t_cruise = duration - t_accel - t_decel
    v_max = total_s / (0.5 * t_accel + t_cruise + 0.5 * t_decel)
    a_accel = v_max / t_accel
    a_decel = v_max / t_decel

    def trapez_s_and_v(t):
        if t <= t_accel:
            return 0.5 * a_accel * t ** 2, a_accel * t
        elif t <= t_accel + t_cruise:
            s_a = 0.5 * v_max * t_accel
            return s_a + v_max * (t - t_accel), v_max
        else:
            s_a = 0.5 * v_max * t_accel
            s_c = v_max * t_cruise
            dt = t - t_accel - t_cruise
            return s_a + s_c + v_max * dt - 0.5 * a_decel * dt ** 2, max(v_max - a_decel * dt, 0.0)

    def find_time_for_s(target_s):
        lo, hi = 0.0, duration
        for _ in range(50):
            mid = (lo + hi) / 2
            if trapez_s_and_v(mid)[0] < target_s:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    waypoint_times = [find_time_for_s(s) for s in cumulative_s]
    waypoint_times[0] = 0.0
    waypoint_times[-1] = duration

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    for i in range(n):
        t_i = waypoint_times[i]
        _, speed_scalar = trapez_s_and_v(t_i)

        if i == 0 or i == n - 1:
            velocities = [0.0] * 6
        else:
            delta = all_q[i + 1] - all_q[i - 1]
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 1e-8:
                direction = delta / delta_norm
                velocities = [float(speed_scalar * direction[j]) for j in range(6)]
            else:
                velocities = [0.0] * 6

        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in all_q[i]]
        point.velocities = velocities
        point.time_from_start = Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
        traj.points.append(point)

    return traj



def bar_free_axis_world(object_name, base_name, base_quat_xyzw, ratio=2.0):
    """Unit world xy vector along the seated part's footprint long axis if the footprint is a bar (long >= ratio x
    short, from CAD), else None. Along it a bar slides freely in its slot - INSERT_GEOMETRY.md: line_green is
    constrained +-1.1 mm across, free along (10-22: its compliance slid 10-13 mm along the bar to seat)."""
    try:
        repo = Path(__file__).resolve().parents[2]
        for p_ in (str(repo), str(repo / 'ablations')):
            if p_ not in sys.path:
                sys.path.insert(0, p_)
        from grasp_depth import load_mesh, surface_points
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        sp, sq = load_seat_pose_in_base(base_name, object_name)
        V, F = load_mesh(object_name)
        P = R.from_quat(sq).apply(surface_points(V, F, 0.003))[:, :2]
        P = P - P.mean(0)
        w, U = np.linalg.eigh(np.cov(P.T))
        ext = [float(np.ptp(P @ U[:, k])) for k in (0, 1)]
        if max(ext) < ratio * min(ext):
            return None
        ax_b = np.array([*U[:, int(np.argmax(ext))], 0.0])
        ax_w = R.from_quat(base_quat_xyzw).apply(ax_b)[:2]
        return ax_w / np.linalg.norm(ax_w)
    except Exception:
        return None

class PrismaticPegInsertion(Node):
    """Prismatic peg-in-hole insertion with tilt-responsive descent."""

    def __init__(self, force_z=DEFAULT_FORCE_Z, timeout=DEFAULT_TIMEOUT,
                 num_waypoints=DEFAULT_NUM_WAYPOINTS,
                 object_name=None, base_name=None, grasp_id=None,
                 final_base_pos=None, final_base_orientation=None,
                 use_default_base=False, object_orientation=None):
        super().__init__('prismatic_peg_insertion')

        self.force_z = force_z
        self.timeout = timeout
        self.num_waypoints = num_waypoints

        self.object_name = object_name
        self.base_name = base_name
        self.grasp_id = grasp_id
        self.final_base_pos = final_base_pos
        self.final_base_orientation = final_base_orientation
        self.use_default_base = use_default_base
        self.object_orientation = object_orientation

        self.current_joints = None
        self.ee_pose = None
        self.wrench = None
        self.target_xyz = None
        self.target_z = None

        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._pose_cb, 10)
        self._wbuf = __import__('collections').deque(maxlen=600)      # ~1.2 s for the phase-locked zero
        self._zero_cli = self.create_client(Trigger, '/io_and_status_controller/zero_ftsensor')
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self._wrench_cb, 10)

        self.traj_client = ActionClient(self, FollowJointTrajectory,
                                        f'/{PASSTHROUGH_CONTROLLER}/follow_joint_trajectory')
        self.start_force_mode_client = self.create_client(SetForceMode, '/force_mode_controller/start_force_mode')
        self.stop_force_mode_client = self.create_client(Trigger, '/force_mode_controller/stop_force_mode')

        # 2026-05-07: telemetry — writes CSV+meta in same schema as
        # compliant_insertion_studio/wrapper logs so the analyzer dashboard
        # can ingest prismatic runs alongside compliant_insert runs.
        from compliant_insertion_studio.wrapper.telemetry import (
            CSVWriter, MetaJSONBuilder, filename_timestamp,
        )
        from compliant_insertion_studio.wrapper.schema_v1 import (
            csv_path_for, meta_path_for,
        )
        from pathlib import Path
        log_dir = Path(__file__).parent.parent.parent / "compliant_insertion_studio" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = filename_timestamp()
        self._csv_path = csv_path_for(object_name or "unknown", ts, str(log_dir))
        self._meta_path = meta_path_for(self._csv_path)
        self._csv_writer = CSVWriter(self._csv_path)
        self._meta_builder = MetaJSONBuilder()
        self._tele_t0 = time.time()
        self._tele_phase = "PRE"
        self._tele_event = ""
        self._tele_commanded_fz = 0.0
        # Reference quat stored on first pose for delta computation
        self._tele_target_xyzq = None
        self.get_logger().info(f"[telemetry] CSV: {self._csv_path}")
        self.get_logger().info(f"[telemetry] meta: {self._meta_path}")

    def _tele_log_row(self):
        """Write one CSV row from the latest pose+wrench. Safe to call any tick."""
        if self.ee_pose is None or self.wrench is None:
            return
        from scipy.spatial.transform import Rotation as _R
        t_s = time.time() - self._tele_t0
        p = self.ee_pose.position
        q = self.ee_pose.orientation
        w = self.wrench
        tx_q, ty_q, tz_q, tw_q = q.x, q.y, q.z, q.w
        # Target = current target if known else hover anchor (first pose seen)
        if self.target_xyz is not None:
            tgt = (float(self.target_xyz[0]), float(self.target_xyz[1]), float(self.target_xyz[2]))
        elif self._tele_target_xyzq is not None:
            tgt = self._tele_target_xyzq[:3]
        else:
            tgt = (p.x, p.y, p.z)
            self._tele_target_xyzq = (p.x, p.y, p.z, tx_q, ty_q, tz_q, tw_q)
        if self._tele_target_xyzq is None:
            self._tele_target_xyzq = (tgt[0], tgt[1], tgt[2], tx_q, ty_q, tz_q, tw_q)
        tgt_q = self._tele_target_xyzq[3:]
        # delta euler
        try:
            R_curr = _R.from_quat([tx_q, ty_q, tz_q, tw_q])
            R_tgt = _R.from_quat(list(tgt_q))
            d_eul = (R_tgt.inv() * R_curr).as_euler("xyz", degrees=False)
        except Exception:
            d_eul = (0.0, 0.0, 0.0)
        row = {
            "t_s": t_s, "phase": self._tele_phase, "event_marker": self._tele_event,
            "hands_off": "false", "zero_event": "",
            "tcp_x": p.x, "tcp_y": p.y, "tcp_z": p.z,
            "tcp_qx": tx_q, "tcp_qy": ty_q, "tcp_qz": tz_q, "tcp_qw": tw_q,
            "target_x": tgt[0], "target_y": tgt[1], "target_z": tgt[2],
            "target_qx": tgt_q[0], "target_qy": tgt_q[1], "target_qz": tgt_q[2], "target_qw": tgt_q[3],
            "dx": p.x - tgt[0], "dy": p.y - tgt[1], "dz": p.z - tgt[2],
            "droll": float(d_eul[0]), "dpitch": float(d_eul[1]), "dyaw": float(d_eul[2]),
            "fx": w.force.x, "fy": w.force.y, "fz": w.force.z,
            "tx": w.torque.x, "ty": w.torque.y, "tz": w.torque.z,
            "gripper_width": 0.0, "commanded_fz": self._tele_commanded_fz,
            "wrench_frame_id": "tool0_controller",
            "obj_x": p.x, "obj_y": p.y, "obj_z": p.z,
            "obj_qx": tx_q, "obj_qy": ty_q, "obj_qz": tz_q, "obj_qw": tw_q,
        }
        try:
            self._csv_writer.write(row)
            self._tele_event = ""
        except Exception as e:
            self.get_logger().warning(f"[telemetry] csv write failed: {e}")

    def _tele_close(self, outcome: str, reason: str = ""):
        """Close CSV + write meta JSON."""
        try:
            import json as _json
            from compliant_insertion_studio.wrapper.telemetry import iso_local_now
            self._csv_writer.close()
            meta = {
                "wrapper_version": "prismatic_peg_insertion@stash",
                "object": self.object_name,
                "base": self.base_name,
                "grasp_id": self.grasp_id,
                "outcome": outcome,
                "outcome_reason": reason,
                "samples_logged": self._csv_writer.row_count,
                "duration_s": round(time.time() - self._tele_t0, 4),
                "end_iso": iso_local_now(),
                "csv_path": self._csv_path,
                "predicted_seat_z": float(self.target_z) if self.target_z is not None else None,
                "predicted_seat_xyz": list(self.target_xyz) if self.target_xyz is not None else None,
                "router": "stash/prismatic_peg_insertion",
            }
            with open(self._meta_path, "w") as fh:
                _json.dump(meta, fh, indent=2, default=str)
            self.get_logger().info(
                f"[telemetry] wrote {self._csv_writer.row_count} samples to {self._csv_path}"
            )
        except Exception as e:
            self.get_logger().warning(f"[telemetry] close failed: {e}")

    # -- ROS callbacks --

    def _joint_cb(self, msg):
        if len(msg.name) == 6 and len(msg.position) == 6:
            ordered = [0.0] * 6
            for i, name in enumerate(msg.name):
                if name in JOINT_NAMES:
                    ordered[JOINT_NAMES.index(name)] = msg.position[i]
            self.current_joints = np.array(ordered)

    def _pose_cb(self, msg):
        self.ee_pose = msg.pose

    def _wrench_cb(self, msg):
        self.wrench = msg.wrench
        w = msg.wrench
        self._wbuf.append((time.time(), w.force.x, w.force.y, w.force.z))

    # -- Helpers --

    def wait_for_data(self, timeout=5.0):
        start = time.time()
        while (self.current_joints is None or self.ee_pose is None or self.wrench is None) \
                and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_joints is not None and self.ee_pose is not None

    def call_service(self, client, request, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def get_ee_position(self):
        if self.ee_pose is None:
            return None
        return np.array([self.ee_pose.position.x, self.ee_pose.position.y, self.ee_pose.position.z])

    def get_ee_orientation_quat(self):
        if self.ee_pose is None:
            return None
        return np.array([self.ee_pose.orientation.x, self.ee_pose.orientation.y,
                         self.ee_pose.orientation.z, self.ee_pose.orientation.w])

    # -- Force sensor --

    def _wait_until_still(self):
        """Block until the TCP has moved < ZERO_STILL_TOL_M over ZERO_STILL_WINDOW_S (measured
        pose, not the trajectory's "complete"), or ZERO_STILL_TIMEOUT_S. True if still."""
        hist = []
        end = time.time() + ZERO_STILL_TIMEOUT_S
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.ee_pose is None:
                continue
            now = time.time()
            hist.append((now, self.get_ee_position()))
            hist = [h for h in hist if now - h[0] <= ZERO_STILL_WINDOW_S]
            if now - hist[0][0] >= ZERO_STILL_WINDOW_S * 0.95:
                pts = np.array([p for _, p in hist])
                if np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < ZERO_STILL_TOL_M:
                    return True
        return False

    def _glitch_mask(self, F):
        dev = np.linalg.norm(F - np.median(F, axis=0), axis=1)
        mad = np.median(np.abs(dev - np.median(dev))) + 1e-6
        return dev > max(2.0, 8.0 * mad)

    def _phase_locked_zero(self):
        end = time.time() + 0.7
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.002)
        buf = np.array(self._wbuf)
        if len(buf) >= 100:
            g = self._glitch_mask(buf[:, 1:4])
            onsets = buf[np.where(g & ~np.r_[False, g[:-1]])[0], 0]
            if len(onsets) >= 2:
                d = np.diff(onsets); d = d[(d > 0.06) & (d < 0.15)]
                period = float(np.median(d)) if len(d) else ZERO_GLITCH_PERIOD_S
                target = onsets[-1] + period / 2.0
                while target - ZERO_EFFECT_LATENCY_S < time.time() + 0.003:
                    target += period
                while time.time() < target - ZERO_EFFECT_LATENCY_S:
                    rclpy.spin_once(self, timeout_sec=0.001)
        self.get_logger().info("Zeroing force sensor (phase-locked)...")
        if not self._zero_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("[ZERO] zero_ftsensor service unavailable")
            return False
        fut = self._zero_cli.call_async(Trigger.Request())
        deadline = time.time() + 5.0
        while time.time() < deadline and not fut.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        return fut.done() and bool(getattr(fut.result(), "success", False))

    def _median_residual(self, wait_s=0.15, window_s=0.3):
        t0 = time.time() + wait_s; end = t0 + window_s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.002)
        buf = np.array([r for r in self._wbuf if r[0] >= t0])
        if len(buf) < 20:
            return None
        good = buf[~self._glitch_mask(buf[:, 1:4]), 1:4]
        return float(np.max(np.abs(np.median(good, axis=0))))

    def zero_force_sensor(self):
        # 2026-10-07: same guard as compliant_insert (2026-08-16): a zero taken while the arm
        # is still moving leaves a phantom load that force mode "relieves" by driving the TCP
        # up. inverted_u_yellow: -9 N in free air, TCP rose 150-250 mm instead of descending.
        if not self._wait_until_still():
            self.get_logger().warn(
                f"[ZERO] arm not still within {ZERO_STILL_TIMEOUT_S:.0f}s — zeroing anyway, residual check follows")
        # 2026-10-14: phase-locked against the robot-side wrench glitch (one 4-6 ms burst every
        # ~102 ms; a zero that lands on one bakes ~11 N in) and verified with a glitch-proof
        # median — same method and numbers as compliant_insert.py (80 trials: 1.25% first-try
        # miss, residual 0.05-0.40 N when good). Native service client: the ros2 CLI took 1.5-2 s
        # with random latency, which makes timing the zero impossible.
        residual = None
        for attempt in (1, 2):
            if not self._phase_locked_zero():
                return False
            residual = self._median_residual()
            if residual is None or residual <= ZERO_GOOD_N:
                break
            self.get_logger().warn(f"[ZERO] median residual {residual:.2f} N > {ZERO_GOOD_N} N after attempt "
                                   f"{attempt}" + (" — zeroing again" if attempt == 1 else ""))
        if residual is None:
            self.get_logger().error("[ZERO] no wrench sample after zero")
            return False
        if residual > ZERO_RESIDUAL_ABORT_N:
            self.get_logger().error(
                f"[ZERO] post-zero residual {residual:.2f} N > {ZERO_RESIDUAL_ABORT_N} N after 2 zeros — "
                f"a real load, refusing force mode this attempt")
            return False
        self.get_logger().info(f"[ZERO] ok, post-zero residual {residual:.2f} N")
        return True

    # -- Force mode --

    def start_force_mode(self, z_speed=NORMAL_Z_SPEED, force_z=None,
                         xy_compliant=False, rx_compliant=False, ry_compliant=False, rz_compliant=True):
        """Start force mode with configurable compliance axes.

        Default: Z + Rz compliance (XY tight, Rx/Ry rigid).
        Keeps orientation rigid in roll/pitch so the peg stays aligned during descent.
        Rz (yaw) is compliant by default for prismatic peg alignment.
        """
        fz = force_z if force_z is not None else self.force_z
        axes = ["Z"]
        if xy_compliant:
            axes = ["XYZ"]
        if rx_compliant:
            axes.append("Rx")
        if ry_compliant:
            axes.append("Ry")
        if rz_compliant:
            axes.append("Rz")
        self.get_logger().info(f"Force mode: {'+'.join(axes)}, Fz={-fz:.1f}N, Vz={z_speed}m/s")

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base_link'
        req.task_frame.pose.orientation.w = 1.0

        req.selection_vector_x = xy_compliant
        req.selection_vector_y = xy_compliant
        req.selection_vector_z = True
        req.selection_vector_rx = rx_compliant
        req.selection_vector_ry = ry_compliant
        req.selection_vector_rz = rz_compliant

        req.wrench = Wrench()
        req.wrench.force.z = -fz
        req.type = 2

        req.speed_limits = Twist()
        req.speed_limits.linear.x = COMPLIANCE_XY_SPEED if xy_compliant else 0.1
        req.speed_limits.linear.y = COMPLIANCE_XY_SPEED if xy_compliant else 0.1
        req.speed_limits.linear.z = z_speed
        req.speed_limits.angular.x = COMPLIANCE_RX_RY_SPEED if rx_compliant else 1.0
        req.speed_limits.angular.y = COMPLIANCE_RX_RY_SPEED if ry_compliant else 1.0
        req.speed_limits.angular.z = COMPLIANCE_RZ_SPEED if rz_compliant else 0.5

        req.damping_factor = 0.0
        req.gain_scaling = 1.0

        # 2026-05-07: retry with backoff. force_mode_controller's start RPC
        # can return success=False for ~0.5-1.5s (sometimes longer on first
        # iteration of a smoke loop) after activation. 6 attempts × 0.8s
        # backoff = ~4s total budget.
        for attempt_n in range(6):
            resp = self.call_service(self.start_force_mode_client, req)
            if resp is not None and resp.success:
                if attempt_n > 0:
                    self.get_logger().info(f"start_force_mode succeeded on attempt {attempt_n+1}")
                return True
            if attempt_n < 5:
                time.sleep(0.8)
        self.get_logger().error("start_force_mode failed after 6 attempts")
        return False

    def start_settling_mode(self, attempt: int = 1):
        """Light downward force + Rx/Ry compliance for tilt correction at depth.

        2026-05-07: Rz (yaw) is LOCKED on ALL attempts. Operator observation:
        yaw consistently rotates OPPOSITE to where the robot needs to move
        under jaws-on-rim contact, so free Rz works against alignment.
        Rx/Ry remain compliant so the part can tilt to settle properly.
        XY remain compliant for fine-positioning during descent.
        """
        rxry_compliant = True
        rz_compliant = False  # Rz locked on all attempts
        self.get_logger().info(
            f"Settling mode: Fz={-SETTLING_FORCE_Z:.1f}N, attempt={attempt}, "
            f"Rx/Ry=free Rz=LOCKED"
        )

        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base_link'
        req.task_frame.pose.orientation.w = 1.0

        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = True
        req.selection_vector_rx = rxry_compliant
        req.selection_vector_ry = rxry_compliant
        req.selection_vector_rz = rz_compliant

        req.wrench = Wrench()
        req.wrench.force.z = -SETTLING_FORCE_Z
        req.type = 2

        req.speed_limits = Twist()
        req.speed_limits.linear.x = SETTLING_XY_SPEED
        req.speed_limits.linear.y = SETTLING_XY_SPEED
        req.speed_limits.linear.z = 0.01  # near-zero Z speed — already at depth
        req.speed_limits.angular.x = SETTLING_RX_RY_SPEED
        req.speed_limits.angular.y = SETTLING_RX_RY_SPEED
        req.speed_limits.angular.z = COMPLIANCE_RZ_SPEED

        # 2026-05-07: high damping during settling. damping=1.0 is max
        # dissipation = stiffest compliance feel (energy bled from any
        # rotational motion, no oscillation). Was 0.0 (springy, took 13s
        # for the part to stop oscillating after reaching depth).
        req.damping_factor = 0.9
        req.gain_scaling = 1.0

        # 2026-05-07: retry with backoff. force_mode_controller's start RPC
        # can return success=False for ~0.5-1.5s (sometimes longer on first
        # iteration of a smoke loop) after activation. 6 attempts × 0.8s
        # backoff = ~4s total budget.
        for attempt_n in range(6):
            resp = self.call_service(self.start_force_mode_client, req)
            if resp is not None and resp.success:
                if attempt_n > 0:
                    self.get_logger().info(f"start_force_mode succeeded on attempt {attempt_n+1}")
                return True
            if attempt_n < 5:
                time.sleep(0.8)
        self.get_logger().error("start_force_mode failed after 6 attempts")
        return False

    def stop_force_mode(self):
        resp = self.call_service(self.stop_force_mode_client, Trigger.Request())
        return resp is not None and resp.success

    # -- Trajectory --

    def send_trajectory(self, waypoints, duration):
        """Send joint trajectory with trapezoidal velocity profile."""
        if not self.traj_client.wait_for_server(timeout_sec=5.0):
            return False

        all_q = [self.current_joints] + list(waypoints)
        traj = build_trapezoidal_trajectory(all_q, duration)

        self.get_logger().info(f"Sending trajectory ({duration:.1f}s)...")
        goal = FollowJointTrajectory.Goal(trajectory=traj, goal_time_tolerance=Duration(sec=2))
        future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.02)

        return result_future.result().status == 4

    # -- Target computation --

    def compute_target_xyz(self):
        """Compute EE target XYZ from assembly config + grasp point + fold symmetry."""
        from primitives.rotate_object import ExtendedCardinalOrientations
        from primitives.shared.fold_symmetry import (
            load_symmetry_data as _load_symmetry_data,
            equivalent_orientations as _equivalent_orientations,
        )

        if not all([self.grasp_id, self.object_name, self.base_name, self.object_orientation]):
            self.get_logger().error("Missing required parameters")
            return None

        assembly_dir = str(get_assembly_data_dir())
        assembly_json = find_assembly_json_by_base_name(self.base_name, assembly_dir)
        if not assembly_json:
            self.get_logger().error(f"No assembly JSON found for '{self.base_name}'")
            return None

        with open(assembly_json, 'r') as f:
            assembly_config = json.load(f)

        target_pos_rel = None
        target_ori_rel = None
        for component in assembly_config.get('components', []):
            if component.get('name') == self.object_name:
                pos = component.get('position', {})
                target_pos_rel = np.array([pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)])
                rot = component.get('rotation', {})
                quat = rot.get('quaternion', {})
                target_ori_rel = np.array([quat.get('x', 0), quat.get('y', 0), quat.get('z', 0), quat.get('w', 1.0)])
                break

        if target_pos_rel is None:
            self.get_logger().error(f"Object '{self.object_name}' not found in assembly config")
            return None

        if self.use_default_base:
            from primitives.shared.config import DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION
            base_pos = np.array(DEFAULT_BASE_POSITION)
            base_ori_quat = np.array(DEFAULT_BASE_ORIENTATION)
        elif self.final_base_pos and self.final_base_orientation:
            base_pos = np.array(self.final_base_pos)
            base_ori_quat = np.array(self.final_base_orientation)
        else:
            self.get_logger().error("No base position provided")
            return None

        R_base = R.from_quat(base_ori_quat).as_matrix()
        self._free_axis = bar_free_axis_world(self.object_name, self.base_name, base_ori_quat)
        if self._free_axis is not None:
            self.get_logger().info(f"[DRIFT] bar footprint: drift along {self._free_axis.round(3).tolist()} is free up to "
                                   f"{1000*BAR_ALONG_MAX_M:.0f} mm, across it the usual tolerance")
        target_pos_world = base_pos + R_base @ target_pos_rel
        R_target_world = R_base @ R.from_quat(target_ori_rel).as_matrix()

        grasp_offset = load_grasp_point_position(self.object_name, self.grasp_id)
        if grasp_offset is None:
            self.get_logger().error(f"Could not load grasp point {self.grasp_id}")
            return None

        R_object_current = R.from_quat(self.object_orientation).as_matrix()
        symmetry_dir = str(get_symmetry_dir())
        fold_data = _load_symmetry_data(self.object_name, symmetry_dir)

        if fold_data:
            equivalents = _equivalent_orientations(R_target_world, fold_data)
            best_dist = float('inf')
            R_grasp_rot = R_target_world
            for R_eq in equivalents:
                dist = ExtendedCardinalOrientations.rotation_matrix_distance(R_object_current, R_eq)
                if dist < best_dist:
                    best_dist = dist
                    R_grasp_rot = R_eq
        else:
            R_grasp_rot = R_object_current

        grasp_world_offset = R_grasp_rot @ grasp_offset
        ee_target = target_pos_world + grasp_world_offset

        if self.ee_pose:
            ee_quat = np.array([self.ee_pose.orientation.x, self.ee_pose.orientation.y,
                               self.ee_pose.orientation.z, self.ee_pose.orientation.w])
            R_ee = R.from_quat(ee_quat).as_matrix()
            ee_target -= R_ee @ held_tool_offset_for(self.object_name)   # as this part was actually grasped

        self.get_logger().info(f"Target XYZ: [{ee_target[0]:.4f}, {ee_target[1]:.4f}, {ee_target[2]:.4f}]")
        return tuple(ee_target)

    # -- Insertion monitoring --

    def monitor_insertion(self, attempt: int = 1):
        """Monitor force-mode descent with tilt-responsive speed adjustment.

        Uses rolling average torques for exit criteria (smooths sensor noise).
        Tilt abort only applies above target Z (not at depth).

        2026-05-07: on retry attempts (attempt > 1) the rotational axes are
        LOCKED during settling and the XY tolerance is tighter — operator
        request to prevent the part drifting further off-axis when attempt 1
        already showed it landing on the rim instead of in the slot.
        """
        self.get_logger().info(f"[INSERTION] Descending to Z={self.target_z:.4f} (attempt={attempt})")
        start_time = time.time()
        # XY tolerance: 10 mm on attempt 1 (initial chamfer nav), 5 mm on
        # retries (operator request — drift past 5mm on retry means we're
        # off-target).
        retry_xy_tol = 0.010 if attempt == 1 else 0.005

        settled = False
        is_tilted = False
        is_stuck = False
        in_settling_mode = False
        tilt_recovery_start = None
        tilt_start = None
        z_at_target_since = None
        last_log_time = 0.0

        # Stuck detection
        stuck_check_z = None
        stuck_check_time = None

        # Rolling torque history: list of (timestamp, tx, ty)
        torque_history = []
        # Rolling Z history for geometric exit's dz/dt estimation
        z_history: list = []
        z_contact = None               # z at this attempt's first [CONTACT]: where the part met something

        def avg_torques(now):
            """Compute rolling average Tx, Ty over EXIT_TORQUE_WINDOW."""
            cutoff = now - EXIT_TORQUE_WINDOW
            # Prune old entries
            while torque_history and torque_history[0][0] < cutoff:
                torque_history.pop(0)
            if not torque_history:
                return 1.0, 1.0  # no data = high torque
            avg_tx = sum(t[1] for t in torque_history) / len(torque_history)
            avg_ty = sum(t[2] for t in torque_history) / len(torque_history)
            return avg_tx, avg_ty

        # Telemetry: from now on we're in the ACTIVE-equivalent phase
        self._tele_phase = "ACTIVE"
        last_tele_t = 0.0
        TELE_PERIOD_S = 0.02   # log at ~50 Hz

        while (time.time() - start_time) < self.timeout:
            rclpy.spin_once(self, timeout_sec=0.005)
            now = time.time()
            elapsed = now - start_time

            if self.ee_pose is None or self.wrench is None:
                continue

            # Per-tick CSV row (rate-limited)
            if (now - last_tele_t) >= TELE_PERIOD_S:
                self._tele_log_row()
                last_tele_t = now

            z = self.ee_pose.position.z
            fx = self.wrench.force.x
            fy = self.wrench.force.y
            fz = self.wrench.force.z
            tx = abs(self.wrench.torque.x)
            ty = abs(self.wrench.torque.y)

            # Wait for sensor to stabilize
            if not settled:
                if elapsed >= SETTLING_TIME:
                    settled = True
                    self.get_logger().info(f"[WALL] Starting insertion at Z={z:.4f} (Fz={fz:.1f}N)")
                continue

            # Accumulate torque history
            torque_history.append((now, tx, ty))

            # Detect disturbance (high torque or lateral force)
            has_lateral = abs(fx) > LATERAL_TRIGGER or abs(fy) > LATERAL_TRIGGER
            has_tilt = tx > TORQUE_TRIGGER_TX or ty > TORQUE_TRIGGER_TY
            disturbed = has_lateral or has_tilt

            # Recovery requires forces below recovery thresholds (hysteresis)
            recovered = (abs(fx) < LATERAL_RECOVERY and abs(fy) < LATERAL_RECOVERY
                         and tx < TORQUE_RECOVERY_TX and ty < TORQUE_RECOVERY_TY)

            if disturbed and not is_tilted:
                is_tilted = True
                if z_contact is None:
                    z_contact = z
                tilt_recovery_start = None
                stuck_check_z = z
                stuck_check_time = now
                # 2026-05-07: on retry, lock Rz too. Operator observation:
                # yaw consistently rotates opposite-direction-from-helpful
                # under jaws-on-rim contact, defeating the retry.
                rz_at_contact = (attempt == 1) or bool(getattr(self, '_corridor_retry', False))
                self.get_logger().warn(
                    f"[CONTACT] Fx={fx:.2f} Fy={fy:.2f} Tx={tx:.4f} Ty={ty:.4f} "
                    f"-> XYZ+Rx/Ry compliance (Rz={'free' if rz_at_contact else 'LOCKED'}), "
                    f"slowing to {SLOW_Z_SPEED}m/s, Fz={SLOW_FORCE_Z}N, attempt={attempt}")
                self.start_force_mode(z_speed=SLOW_Z_SPEED, force_z=SLOW_FORCE_Z,
                                     xy_compliant=True, rz_compliant=rz_at_contact)

            elif is_tilted and disturbed:
                tilt_recovery_start = None
                # Stuck detection: abort only if Z isn't progressing (not just time-based)
                at_depth = z <= self.target_z + Z_TOLERANCE
                if not at_depth and stuck_check_time is not None:
                    if (now - stuck_check_time) >= STUCK_WINDOW:
                        z_progress = stuck_check_z - z  # positive = descending
                        if z_progress < STUCK_THRESHOLD:
                            if not is_stuck:
                                is_stuck = True
                                stuck_start = now
                                self.get_logger().warn(f"[STUCK] No Z progress in {STUCK_WINDOW}s (Z={z:.4f})")
                            elif (now - stuck_start) > MAX_CONTINUOUS_TILT_TIME:
                                self.get_logger().error(
                                    f"[ABORT] Stuck for > {MAX_CONTINUOUS_TILT_TIME}s with no Z progress "
                                    f"(Z={z:.4f}, target={self.target_z:.4f})")
                                return 'tilt_unrecoverable'
                        else:
                            if is_stuck:
                                self.get_logger().info(f"[PROGRESS] Z moved {z_progress*1000:.1f}mm — no longer stuck")
                                is_stuck = False
                        stuck_check_z = z
                        stuck_check_time = now

            elif is_tilted and recovered:
                if tilt_recovery_start is None:
                    tilt_recovery_start = now
                elif (now - tilt_recovery_start) >= TILT_RECOVERY_TIME:
                    is_tilted = False
                    is_stuck = False
                    tilt_recovery_start = None
                    stuck_check_z = None
                    stuck_check_time = None
                    self.get_logger().info(f"[ALIGNED] Resuming {NORMAL_Z_SPEED}m/s, Fz={self.force_z}N")
                    self.start_force_mode(z_speed=NORMAL_Z_SPEED,
                                         xy_compliant=True,
                                         rz_compliant=False)  # Rz LOCKED — yaw drift fights alignment

            elif is_tilted:
                tilt_recovery_start = None

            # XY deviation check (after grace period)
            if elapsed >= SETTLING_TIME + 2.0:
                dxy_ = np.array([self.ee_pose.position.x - self.target_xyz[0], self.ee_pose.position.y - self.target_xyz[1]])
                xy_dev = float(np.hypot(*dxy_))
                if getattr(self, '_free_axis', None) is not None:
                    # a bar: only drift ACROSS it is a miss; along it the compliance is finding the slot (<= 30 mm)
                    along_ = float(abs(dxy_ @ self._free_axis))
                    across_ = float(abs(dxy_[0] * self._free_axis[1] - dxy_[1] * self._free_axis[0]))
                    xy_dev = across_ if along_ <= BAR_ALONG_MAX_M else max(across_, along_)
                # Once the part has dropped below its first-contact height it is IN the slot and
                # sideways motion is the compliance following it — 2026-10-15 line_green dropped
                # into the U channel on attempt 3 and this check pulled it straight out. Drift is
                # only an exit while the part still rides on top of something.
                in_slot = z_contact is not None and (z_contact - z) > DRIFT_EXEMPT_DROP_M
                # or below the highest CAD first-contact surface: the part is already in a corridor stage (10-22 07:20:
                # yellow's legs were astride line_green at +33 mm when force mode began, so its 'first contact' was that
                # height and the compliance sliding along line_green tripped the 10 mm drift abort)
                cand_ = [v for k_, v in (self.contact_candidates or {}).items() if k_ != 'SEAT']
                if cand_ and z is not None and (max(cand_) - z) > DRIFT_EXEMPT_DROP_M:
                    in_slot = True
                if xy_dev > (SEATED_XY_ACCEPT_M if in_slot else retry_xy_tol):
                    self.get_logger().warn(
                        f"[XY_DRIFT] {xy_dev*1000:.1f}mm > tol {retry_xy_tol*1000:.1f}mm "
                        f"(attempt={attempt}) -> aborting"
                    )
                    return 'xy_deviation'

            # ---- GEOMETRIC EXIT (option 1, 2026-05-07) ----
            # Track z history for dz/dt estimation
            z_history.append((now, z))
            while z_history and (now - z_history[0][0]) > EXIT_GEOMETRIC_VEL_WINDOW:
                z_history.pop(0)
            if len(z_history) >= 2:
                t0, z0 = z_history[0]
                dz_dt = (z - z0) / max(1e-6, now - t0)   # m/s; negative = descending
            else:
                dz_dt = float('inf')
            depth_err = z - self.target_z   # negative = below predicted seat
            geo_at_depth = (depth_err <=  EXIT_GEOMETRIC_Z_TOL_ABOVE_M and
                            depth_err >= -EXIT_GEOMETRIC_Z_TOL_BELOW_M)
            # 2026-05-07: Fz gate removed (was 4 N) — line_green's jaws-on-rim
            # contact produces oscillating Fz (mean 5 N, max 13 N) even when
            # the part is correctly seated and motionless. compliant_insert
            # wrapper's seat detector doesn't check Fz either — depth +
            # velocity + tilt are sufficient.
            geo_light_fz = True
            geo_stopped  = abs(dz_dt) <= EXIT_GEOMETRIC_DZ_DT_MAX

            # Exit: Z at target + rolling avg torques below threshold
            at_depth = z <= self.target_z + Z_TOLERANCE
            if at_depth:
                # Switch to settling mode on first arrival at depth
                if not in_settling_mode:
                    in_settling_mode = True
                    self.get_logger().info(f"[AT_DEPTH] Z={z:.4f} — switching to settling mode (Fz={SETTLING_FORCE_Z}N, Rx/Ry={SETTLING_RX_RY_SPEED}rad/s)")
                    self.start_settling_mode(attempt=attempt)

                # Geometric exit fires regardless of torque state
                if geo_at_depth and geo_light_fz and geo_stopped:
                    if z_at_target_since is None:
                        z_at_target_since = now
                        self.get_logger().info(
                            f"[CONFIRMING-GEO] depth_err={depth_err*1000:+.2f}mm "
                            f"|Fz|={abs(fz):.2f}N |dz/dt|={abs(dz_dt)*1000:.3f}mm/s — holding..."
                        )
                    elif (now - z_at_target_since) >= EXIT_SUSTAINED_TIME:
                        avg_tx_now, avg_ty_now = avg_torques(now)
                        self.get_logger().info(
                            f"[SUCCESS-GEO] Z={z:.4f} depth_err={depth_err*1000:+.2f}mm "
                            f"|Fz|={abs(fz):.2f}N |dz/dt|={abs(dz_dt)*1000:.3f}mm/s "
                            f"(avg_Tx={avg_tx_now:.3f} avg_Ty={avg_ty_now:.3f} ignored) "
                            f"sustained {EXIT_SUSTAINED_TIME}s"
                        )
                        return 'success'
                else:
                    if z_at_target_since is not None:
                        miss = []
                        if not geo_at_depth: miss.append(f"depth_err={depth_err*1000:+.2f}mm")
                        if not geo_light_fz: miss.append(f"|Fz|={abs(fz):.2f}N")
                        if not geo_stopped:  miss.append(f"|dz/dt|={abs(dz_dt)*1000:.3f}mm/s")
                        self.get_logger().info(f"[SETTLING-GEO] reset: {', '.join(miss)}")
                    z_at_target_since = None
            else:
                z_at_target_since = None
                if in_settling_mode:
                    # Bounced above target — back to descent mode
                    in_settling_mode = False
                    self.get_logger().info(f"[ABOVE_TARGET] Z={z:.4f} — resuming descent")
                    if is_tilted:
                        self.start_force_mode(z_speed=SLOW_Z_SPEED, force_z=SLOW_FORCE_Z,
                                             xy_compliant=True,
                                             rz_compliant=False)  # Rz LOCKED — yaw drift fights alignment
                    else:
                        self.start_force_mode(z_speed=NORMAL_Z_SPEED,
                                             xy_compliant=True,
                                             rz_compliant=False)  # Rz LOCKED — yaw drift fights alignment

            # Periodic logging (every 2s)
            if (now - last_log_time) >= 2.0:
                last_log_time = now
                avg_tx, avg_ty = avg_torques(now)
                tilt_str = " [TILTED]" if is_tilted else ""
                depth_str = " [AT_DEPTH]" if z <= self.target_z + Z_TOLERANCE else ""
                self.get_logger().info(
                    f"Z={z:.4f} Fz={fz:.1f} Fx={fx:.2f} Fy={fy:.2f} "
                    f"Tx={tx:.4f}(avg={avg_tx:.4f}) Ty={ty:.4f}(avg={avg_ty:.4f}){tilt_str}{depth_str}")

        self.get_logger().error(f"[TIMEOUT] {self.timeout}s exceeded")
        return 'timeout'

    # -- Retract --

    def retract_upward(self, target_z, timeout=5.0):
        """Retract upward using force mode (all axes compliant)."""
        self.get_logger().info(f"[RETRACT] To Z={target_z:.4f}...")
        req = SetForceMode.Request()
        req.task_frame.header.frame_id = 'base_link'
        req.task_frame.pose.orientation.w = 1.0
        req.selection_vector_x = True
        req.selection_vector_y = True
        req.selection_vector_z = True
        req.selection_vector_rx = True
        req.selection_vector_ry = True
        req.selection_vector_rz = True
        req.wrench = Wrench()
        req.wrench.force.z = 5.0  # upward
        req.type = 2
        req.speed_limits = Twist()
        req.speed_limits.linear.z = NORMAL_Z_SPEED
        req.speed_limits.linear.x = 0.01
        req.speed_limits.linear.y = 0.01
        req.speed_limits.angular.x = 0.1
        req.speed_limits.angular.y = 0.1
        req.speed_limits.angular.z = 0.1

        resp = self.call_service(self.start_force_mode_client, req)
        if resp is None or not resp.success:
            return False

        start = time.time()
        while (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.ee_pose and self.ee_pose.position.z >= target_z:
                self.get_logger().info(f"[RETRACT] Reached Z={self.ee_pose.position.z:.4f}")
                self.stop_force_mode()
                return True

        self.stop_force_mode()
        return True

    # -- Main execution --

    def execute(self):
        """Run insertion with retry logic."""
        if not self.wait_for_data():
            self.get_logger().error("Timeout waiting for sensor data")
            self._tele_close("abort", "wait_for_data_timeout")
            return False

        self.get_logger().info("=" * 60)
        self.get_logger().info("PRISMATIC PEG-IN-HOLE INSERTION")
        self.get_logger().info("=" * 60)

        target_result = self.compute_target_xyz()
        if target_result is None:
            self._tele_close("abort", "compute_target_xyz_failed")
            return False

        target_x, target_y, target_z = target_result
        self.target_xyz = np.array([target_x, target_y, target_z])
        self.target_z = target_z
        self.aim_xyz = self.target_xyz.copy()           # the CAD aim; retries shift target_xyz from it
        self.last_dev_vec, self.last_stall_z = None, None
        self.contact_candidates = self._cad_contact_candidates()

        initial_joints = self.current_joints.copy()
        initial_pos = self.get_ee_position().copy()
        initial_ori = self.get_ee_orientation_quat().copy()

        self.get_logger().info(f"Current: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f}]")
        self.get_logger().info(f"Target:  [{target_x:.4f}, {target_y:.4f}, {target_z:.4f}]")

        for attempt in range(1, MAX_RETRIES + 1):
            self.get_logger().info(f"\n--- ATTEMPT {attempt}/{MAX_RETRIES} ---")

            if not self.zero_force_sensor():
                self.get_logger().error("Force sensor zero failed")
                continue

            # Phase 1: Start force mode (Z-only, light force for gentle contact during trajectory)
            self.get_logger().info(f"[PHASE 1] Starting force mode (Z-only, Fz={TRAJECTORY_FORCE_Z}N)")
            if not self.start_force_mode(force_z=TRAJECTORY_FORCE_Z, rz_compliant=False):
                self.get_logger().error("Failed to start force mode")
                continue

            # Phase 2: Trajectory approach to target Z (force mode absorbs contact)
            rclpy.spin_once(self, timeout_sec=0.1)
            current_z = self.get_ee_position()[2] if self.get_ee_position() is not None else initial_pos[2]
            distance = abs(current_z - target_z)
            # Retries use slower speed (closer to hole, need precision)
            speed = APPROACH_SPEED if attempt == 1 else APPROACH_SPEED / 2
            duration = max(3.0, distance / speed)

            if attempt == 1:
                self.get_logger().info(f"[PHASE 2] Trajectory to target Z={target_z:.4f} ({distance*1000:.0f}mm, {duration:.1f}s)")
                waypoints = compute_cartesian_waypoints_ik(
                    np.array(self.current_joints, dtype=float),
                    target_z,
                    num_waypoints=self.num_waypoints
                )
            else:
                self.get_logger().info(f"[PHASE 2] Retry trajectory from Z={current_z:.4f} ({distance*1000:.0f}mm, {duration:.1f}s, slow)")
                waypoints = compute_cartesian_waypoints_ik(
                    np.array(self.current_joints, dtype=float),
                    target_z,
                    num_waypoints=max(10, self.num_waypoints // 2)
                )

            if waypoints is None:
                self.get_logger().error("Motion planning failed: no collision-free approach path could be computed")
                self.stop_force_mode()
                continue
            self.send_trajectory(waypoints, duration)

            # Post-trajectory check: if at target Z with low torques, exit immediately
            rclpy.spin_once(self, timeout_sec=0.1)
            post_z = self.get_ee_position()[2] if self.get_ee_position() is not None else 999
            if post_z <= target_z + Z_TOLERANCE and self.wrench is not None:
                post_tx = abs(self.wrench.torque.x)
                post_ty = abs(self.wrench.torque.y)
                if post_tx <= 0.05 and post_ty <= 0.05:
                    self.get_logger().info(
                        f"[SUCCESS] Reached target via trajectory (Z={post_z:.4f}, Tx={post_tx:.4f}, Ty={post_ty:.4f})")
                    self.stop_force_mode()
                    # telemetry only: this exit left a CSV with no meta (line_green, 2026-10-19 20:12)
                    self._tele_close("success", f"reached_target_via_trajectory z={post_z:.4f} tx={post_tx:.4f} ty={post_ty:.4f}")
                    return True
                else:
                    self.get_logger().info(
                        f"[PHASE 2] At target Z={post_z:.4f} but tilted (Tx={post_tx:.4f}, Ty={post_ty:.4f}) — continuing to force mode")

            # Phase 3: Force-mode descent + compliance (force mode already active)
            self.get_logger().info(f"[PHASE 3] Force-mode descent from Z={post_z:.4f}")
            # 2026-05-07: Phase 3 enters with XY LOCKED. Mirrors the
            # compliant_insert wrapper's INSERT_DESCENT path — peg goes
            # straight down at predicted target, no XY drift before contact.
            # The [CONTACT] handler inside monitor_insertion is what unlocks
            # XY (xy_compliant=True) once a Fxy/T spike is detected, enabling
            # the slide-into-slot under jaw-on-rim contact for wide-grip parts.
            # If prediction is dead-on (no rim hit), peg seats with zero drift.
            self.start_force_mode(xy_compliant=False, rz_compliant=False)
            result = self.monitor_insertion(attempt=attempt)
            self.stop_force_mode()

            # Check result + deviation
            within, pos_dev, ori_dev = self._check_deviation(initial_ori)
            self.get_logger().info(f"[VERIFY] result={result}, pos_dev={pos_dev*1000:.1f}mm, ori_dev={ori_dev:.2f}deg")

            # Seated = at seat depth and still (SUCCESS-GEO). A part that missed its slot rests on
            # the base top, above that depth, so an xy offset here is the compliance having slid it
            # to where the slot really is — not an exit. 2026-10-15: yellow seated 6.6 mm from the
            # aim, the old 5 mm rule called it a failure, retracted 10 mm WITH the part gripped
            # (pulling the seated parts with it) and pushed it in again at 5 N sideways.
            if result == 'success' and not within and pos_dev <= SEATED_XY_ACCEPT_M \
                    and ori_dev <= ORIENTATION_DEVIATION:
                self.get_logger().warn(f"[VERIFY] seated {pos_dev*1000:.1f} mm from the aim (compliance found "
                                       f"the slot) -> accepted, no retract")
                within = True
            if result == 'success' and within:
                self.get_logger().info(f"[SUCCESS] Insertion complete on attempt {attempt}")
                self._tele_phase = "DONE"
                self._tele_event = f"success_attempt_{attempt}_pos_dev_{pos_dev*1000:.1f}mm_ori_dev_{ori_dev:.2f}deg"
                self._tele_log_row()
                self._tele_close("success",
                    f"attempt={attempt} result={result} within={within} pos_dev_mm={pos_dev*1000:.2f} ori_dev_deg={ori_dev:.2f}")
                return True
            elif result == 'success' and not within:
                # Geometry says we exited the slot region — that's a real failure
                self._tele_event = f"declared_success_but_over_tolerance_pos_dev_{pos_dev*1000:.1f}mm_ori_dev_{ori_dev:.2f}deg"
                self._tele_log_row()

            self.get_logger().warn(f"[ATTEMPT {attempt}] result={result}, within_limits={within}")

            # Retry: retract 1cm and restore orientation
            if attempt < MAX_RETRIES:
                current_pos = self.get_ee_position()
                retry_z = current_pos[2] + 0.01 if current_pos is not None else initial_pos[2]
                cor = self._corridor_of(self.last_stall_z)
                self._corridor_retry = cor is not None
                if cor is not None:
                    retry_z = max(retry_z, cor[1] + CORRIDOR_EXIT_M)
                    self.get_logger().warn(f"[RETRY {attempt + 1}] wedged in {cor[0]}'s opening -> retract above its "
                                           f"entrance (Z={retry_z:.4f}), Rz stays compliant")
                self.retract_upward(retry_z)
                rclpy.spin_once(self, timeout_sec=0.5)

                # Where did it stall, and where does the next attempt aim (2026-10-19)
                stage = self._stage_at(self.last_stall_z)
                shift, why = self._retry_shift(attempt + 1)
                self.target_xyz = np.array([self.aim_xyz[0] + shift[0], self.aim_xyz[1] + shift[1], self.aim_xyz[2]])
                self.get_logger().warn(
                    f"[RETRY {attempt + 1}] stalled at Z={self.last_stall_z if self.last_stall_z is not None else float('nan'):.4f}"
                    f" = {stage}; aim shift {why}")
                self._tele_event = f"retry_{attempt + 1}_shift_{1000*shift[0]:+.1f}_{1000*shift[1]:+.1f}mm_stage_{stage}"
                self._tele_log_row()

                # Restore orientation via trajectory
                rclpy.spin_once(self, timeout_sec=0.1)
                retry_target = np.array([self.target_xyz[0], self.target_xyz[1], retry_z])
                ori_waypoints = compute_cartesian_waypoints_ik(
                    np.array(self.current_joints, dtype=float),
                    retry_z,
                    target_pos=retry_target,
                    target_orientation=R.from_quat(initial_ori).as_matrix(),
                    num_waypoints=20
                )
                if ori_waypoints is not None:
                    self.send_trajectory(ori_waypoints, 2.0)
                    rclpy.spin_once(self, timeout_sec=0.5)

        # All retries exhausted - return to initial pose
        self.get_logger().error(f"[FAILED] All {MAX_RETRIES} attempts exhausted")
        # 2026-10-20: a part that ended wedged just above its seat (x/y at the aim) is held there instead of pulled
        # out, so the caller can release it and nudge it in (PRISMATIC_HOLD_IF_WEDGED=1; line_brown stalled 4.5-4.8 mm
        # above the seat at the aim four times and every retract-and-retry did the same)
        cur = self.get_ee_position()
        if (os.environ.get('PRISMATIC_HOLD_IF_WEDGED') == '1' and cur is not None and self.target_xyz is not None):
            dz = 1000 * (cur[2] - self.target_xyz[2]); dxy = 1000 * float(np.hypot(cur[0] - self.target_xyz[0], cur[1] - self.target_xyz[1]))
            if 0 < dz <= 7.0 and dxy <= 5.0:
                self.get_logger().warn(f"[WEDGED] {dz:.1f} mm above the seat, {dxy:.1f} mm from the aim - holding here (no retract)")
                self._tele_phase = "ABORT"; self._tele_event = "wedged_hold"
                self._tele_log_row()
                self._tele_close("abort", f"wedged {dz:.1f} mm above the seat, held")
                restore_controllers()
                return False
        self._tele_phase = "ABORT"
        self._tele_event = "all_retries_exhausted"
        self._tele_log_row()
        self._tele_close("abort", f"all_retries_exhausted ({MAX_RETRIES} attempts)")
        self.retract_upward(initial_pos[2])
        restore_controllers()
        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.1)
        self.send_trajectory([initial_joints], 3.0)
        return False

    def _cad_contact_candidates(self):
        """{SEAT, BASE_RIM, OBJ:<seated part>} -> TCP z (m), the same chain the compliant path uses
        (compliant_insertion_studio.wrapper.cad_geometry). Tells which stage a stall is at:
        yellow clears line_green first (feet astride it), then the pocket rim. Best effort."""
        try:
            from compliant_insertion_studio.wrapper.cad_geometry import build_contact_candidates, get_seated_so_far
            base_pos = np.array(self.final_base_pos) if self.final_base_pos else None
            if base_pos is None or self.object_orientation is None:
                return None
            c = build_contact_candidates(
                target_object=self.object_name, target_grasp_id=int(self.grasp_id), base_name=self.base_name,
                base_world_z=float(base_pos[2]), R_grasp_quat_xyzw=list(self.object_orientation),
                predicted_tcp_z_at_seat=float(self.target_z),
                seated_objects=get_seated_so_far(self.base_name, self.object_name),
                flange_offset_world_z=-float(held_tool_offset_for(self.object_name)[2]))
            self.get_logger().info("Contact candidates (CAD-derived): " + ", ".join(f"{k}={1000*v:.1f}mm" for k, v in c.items()))
            return c
        except Exception as e:
            self.get_logger().warn(f"contact candidates unavailable ({e})")
            return None

    def _stage_at(self, z):
        """Which CAD contact height a stall z matches ('SEAT' / 'BASE_RIM' / 'OBJ:line_green' ...), or None."""
        if not self.contact_candidates or z is None:
            return None
        k, v = min(self.contact_candidates.items(), key=lambda kv: abs(kv[1] - z))
        return f"{k} ({1000*(z - v):+.1f} mm)" if abs(v - z) <= STAGE_MATCH_TOL_M else f"between contacts ({1000*(z - self.target_z):+.1f} mm above the seat)"

    def _corridor_of(self, z):
        """(name, top_tcp_z) of the seated part whose opening a stall at z sits inside, else None."""
        if not self.contact_candidates or z is None:
            return None
        for k, top in self.contact_candidates.items():
            if k.startswith('OBJ:') and top - CORRIDOR_DEPTH_M <= z <= top + 0.002:
                return k, top
        return None

    def _retry_shift(self, attempt):
        """xy shift of the aim for `attempt` from the last stall's measured slide (see RETRY_SHIFT_*)."""
        d = self.last_dev_vec
        if d is None:
            return np.zeros(2), "no deviation measured"
        n = float(np.linalg.norm(d))
        if self._corridor_of(self.last_stall_z) is not None and n >= RETRY_SHIFT_MIN_M:
            sign = 1.0 if attempt == 2 else -1.0
            d2 = sign * d * min(1.0, CORRIDOR_SHIFT_MAX_M / n)
            return d2, f"corridor: {'toward' if sign > 0 else 'away from'} the slide, bounded ({1000*d2[0]:+.1f}, {1000*d2[1]:+.1f}) mm"
        if n < RETRY_SHIFT_MIN_M:
            return np.zeros(2), f"slide {1000*n:.1f} mm < {1000*RETRY_SHIFT_MIN_M:.1f}: not an xy miss"
        if n > RETRY_SHIFT_MAX_M:
            return np.zeros(2), f"slide {1000*n:.1f} mm > {1000*RETRY_SHIFT_MAX_M:.0f}: beyond what aim/grasp can be off"
        sign = 1.0 if attempt == 2 else -1.0
        return sign * d, f"{'where the part slid to' if sign > 0 else 'the mirror of the slide'} ({1000*sign*d[0]:+.1f}, {1000*sign*d[1]:+.1f}) mm"

    def _check_deviation(self, initial_ori_quat):
        """Check XY position and orientation deviation. Returns (within_limits, pos_dev_m, ori_dev_deg)."""
        rclpy.spin_once(self, timeout_sec=0.1)
        current_pos = self.get_ee_position()
        current_ori = self.get_ee_orientation_quat()
        if current_pos is None or current_ori is None:
            return True, 0.0, 0.0

        self.last_dev_vec = np.asarray(current_pos[:2]) - np.asarray(self.target_xyz[:2])
        self.last_stall_z = float(current_pos[2])
        pos_dev = np.linalg.norm(self.last_dev_vec)
        ori_dev = np.degrees((R.from_quat(initial_ori_quat).inv() * R.from_quat(current_ori)).magnitude())
        within = pos_dev <= POSITION_DEVIATION_XY and ori_dev <= ORIENTATION_DEVIATION
        return within, pos_dev, ori_dev


def main():
    parser = argparse.ArgumentParser(description='Prismatic Peg-in-Hole Insertion')
    parser.add_argument('--force', type=float, default=DEFAULT_FORCE_Z)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument('--num-waypoints', type=int, default=DEFAULT_NUM_WAYPOINTS)
    parser.add_argument('--object-name', type=str, required=True)
    parser.add_argument('--base-name', type=str, required=True)
    parser.add_argument('--grasp-id', type=int, required=True)
    parser.add_argument('--current-object-orientation', type=float, nargs=4, required=True,
                        metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--use-default-base-position', action='store_true')

    args, remaining = parser.parse_known_args()

    # Controller setup: deactivate scaled, activate passthrough + force_mode
    print("Deactivating scaled_joint_trajectory_controller...")
    result = subprocess.run(
        ['ros2', 'control', 'switch_controllers', '--deactivate', SCALED_CONTROLLER],
        capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return 1
    time.sleep(0.5)

    print("Activating passthrough + force_mode controllers...")
    if not switch_controllers([PASSTHROUGH_CONTROLLER, FORCE_MODE_CONTROLLER], []):
        print("Controller activation failed")
        return 1
    # 2026-05-07: bumped 1.0 → 2.5s. force_mode_controller's start RPC was
    # rejecting calls even after 4 retries (~1.8s of backoff) on the first
    # attempt of the first iteration of a smoke loop. 2.5s plus the retry
    # loop in start_force_mode covers the worst-case ready-time we've seen.
    time.sleep(2.5)

    try:
        rclpy.init(args=remaining)
        node = PrismaticPegInsertion(
            force_z=args.force,
            timeout=args.timeout,
            num_waypoints=args.num_waypoints,
            object_name=args.object_name,
            base_name=args.base_name,
            grasp_id=args.grasp_id,
            final_base_pos=args.final_base_pos,
            final_base_orientation=args.final_base_orientation,
            use_default_base=args.use_default_base_position,
            object_orientation=args.current_object_orientation,
        )

        success = node.execute()
        time.sleep(0.5)

        print(f"\nRESULT: {'SUCCESS' if success else 'FAILED'}")
        rclpy.shutdown()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C - cleaning up...")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        restore_controllers()


if __name__ == '__main__':
    exit(main())
