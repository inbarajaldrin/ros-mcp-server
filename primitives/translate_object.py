#!/usr/bin/env python3
"""
Translate Object Primitive - Move held object to assembly position

Modes:
- --insert: Move to hover above base then insert (sim: both hover + descend, real: subprocess to prismatic_peg_insertion.py)
- --place-down: Move laterally to clear area then lower onto table

Note: --insert and --place-down are mutually exclusive.

Usage:
    # Sim mode - move to base and insert
    python3 translate_object.py --mode sim --object-name fork_orange --base-name base --insert

    # Real mode - insert (peg-in-hole force control)
    python3 translate_object.py --mode real --insert

    # Place on clear area (lateral move + lower)
    python3 translate_object.py --mode real --place-down
"""

import sys
import os

# Add project root to path so primitives package can be imported when running directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import subprocess
import argparse
import threading
import json
import re
import time

import logging


# ---------------------------------------------------------------------------
# Fast path for subprocess-only actions. These paths just spawn a child
# process and forward its JSON output — they don't need rclpy, numpy, scipy,
# IK solvers, collision checkers, rotate_object, etc.  By handling them here
# we skip ~2-3s of import overhead.
# ---------------------------------------------------------------------------

def _subprocess_fast_path():
    """Exit early for subprocess-only invocations (before heavy imports)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--mode', type=str, default='sim')
    pre.add_argument('--place-down', action='store_true', dest='place_down')
    pre.add_argument('--insert', action='store_true')
    pre.add_argument('--insertion-type', type=str, default='compliant')
    pre.add_argument('--object-name', type=str, default=None)
    pre.add_argument('--base-name', type=str, default=None)
    pre.add_argument('--grasp-id', type=int, default=None)
    pre.add_argument('--current-object-orientation', type=float, nargs=4, default=None)
    pre.add_argument('--use-default-base-position', action='store_true',
                     dest='use_default_base_position')
    pre.add_argument('--final-base-pos', type=float, nargs=3, default=None)
    pre.add_argument('--final-base-orientation', type=float, nargs=4, default=None)
    args, _ = pre.parse_known_args()

    # Real-mode insert with the compliant_insert wrapper is a thin proxy:
    # delegate to the autonomous-search FSM in compliant_insertion_studio,
    # skipping the IK-heavy hover step (the wrapper has its own _run_hover).
    # The actual insert algorithm lives in compliant_insertion_studio/wrapper/
    # so that future improvements there land here automatically without
    # touching translate_object's main path. Sim-mode insert is unchanged.
    is_compliant_insert = (
        args.insert and args.mode == 'real' and args.insertion_type == 'compliant'
    )
    is_fast = args.place_down or is_compliant_insert
    if not is_fast:
        return  # Fall through to heavy imports and full main()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    log = logging.getLogger('translate_object')

    # Validate quaternion if provided (catches SDK parser corruption like 1e-6 → 16)
    if args.current_object_orientation is not None:
        if any(abs(v) > 1.0 for v in args.current_object_orientation):
            print("__RESULT_JSON__")
            print(json.dumps({"result": "failure", "mode": args.mode,
                              "error": f"current_object_orientation has component(s) outside [-1, 1]: {args.current_object_orientation}"}))
            print("__END_RESULT_JSON__")
            sys.exit(1)

    def _output(result):
        print("__RESULT_JSON__")
        print(json.dumps(result))
        print("__END_RESULT_JSON__")

    def _extract_json(text):
        if "__RESULT_JSON__" in text and "__END_RESULT_JSON__" in text:
            s = text.find("__RESULT_JSON__") + len("__RESULT_JSON__")
            e = text.find("__END_RESULT_JSON__")
            try:
                return json.loads(text[s:e].strip())
            except json.JSONDecodeError:
                pass
        return None

    def _make_env():
        env = os.environ.copy()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env['PYTHONPATH'] = f"{root}:{env['PYTHONPATH']}" if 'PYTHONPATH' in env else root
        return env

    def _stream(pipe, lines):
        for line in iter(pipe.readline, ''):
            line = line.rstrip()
            if line:
                lines.append(line)
                log.info(line)
        pipe.close()

    def _run(script, cmd_args=None, timeout=None):
        cmd = [sys.executable, script] + (cmd_args or [])
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=_make_env(),
        )
        lines = []
        t = threading.Thread(target=_stream, args=(proc.stdout, lines), daemon=True)
        t.start()
        try:
            rc = proc.wait(timeout=timeout)
            t.join(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            t.join(timeout=1.0)
            return False, '\n'.join(lines)
        return rc == 0, '\n'.join(lines)

    def _finish(success, output_text, movement_type):
        rj = _extract_json(output_text)
        if rj:
            rj["movement_type"] = movement_type
            rj["mode"] = args.mode  # Subprocess mode may differ from MCP mode
            _output(rj)
        else:
            r = {"result": "success" if success else "failure", "mode": args.mode,
                 "movement_type": movement_type}
            if not success:
                r["error"] = f"{movement_type} failed"
            _output(r)
        sys.exit(0 if success else 1)

    pdir = os.path.dirname(os.path.abspath(__file__))

    if args.place_down:
        if not args.object_name:
            _output({"result": "failure", "mode": args.mode, "movement_type": "place_down",
                     "error": "object_name is required for place_down"})
            sys.exit(1)
        log.info(f"Verifying grasp on {args.object_name} before placing down")
        qdir = os.path.join(os.path.dirname(pdir), 'queries')
        ok, out = _run(os.path.join(qdir, 'verify_grasp.py'),
                       ['--object-name', args.object_name, '--mode', args.mode, '--width-only'], timeout=15)
        vj = _extract_json(out)
        if not ok or (vj and vj.get('result') == 'failure'):
            err = vj.get('error', 'grasp check failed') if vj else 'grasp check failed'
            _output({"result": "failure", "mode": args.mode, "movement_type": "place_down",
                     "error": f"Grasp check failed before place_down: {err}"})
            sys.exit(1)
        log.info("Placing object on clear area")
        ca = ['--move', '--mode', args.mode, '--object-name', args.object_name]
        ok, out = _run(os.path.join(pdir, 'core', 'move_to_clear_area.py'), ca, timeout=45)
        if not ok:
            _finish(ok, out, "place_down")
        log.info("Lowering object onto table")
        ok, out = _run(os.path.join(pdir, 'core', 'move_down.py'), ['--mode', args.mode], timeout=310)
        _finish(ok, out, "place_down")

    if is_compliant_insert:
        # Validate inputs that the wrapper requires up front (so we fail fast
        # with a clean error JSON before subprocess spin-up).
        missing = []
        if not args.object_name:
            missing.append('--object-name')
        if not args.base_name:
            missing.append('--base-name')
        if args.grasp_id is None:
            missing.append('--grasp-id')
        if args.current_object_orientation is None:
            missing.append('--current-object-orientation')
        if not (args.use_default_base_position
                or (args.final_base_pos is not None and args.final_base_orientation is not None)):
            missing.append('--use-default-base-position OR --final-base-pos+--final-base-orientation')
        if missing:
            _output({"result": "failure", "mode": args.mode, "movement_type": "insert",
                     "error": f"compliant insert requires: {', '.join(missing)}"})
            sys.exit(1)

        # 2026-05-07: apply PER_OBJECT_BASE_OFFSET_M for ALL objects routed
        # through this thin shell. Both branches below (line_green/yellow via
        # prismatic, others via compliant_insert) used to load bare
        # DEFAULT_BASE_POSITION, missing the per-object calibration. Apply
        # once here; both branches then pass --final-base-pos through.
        if args.use_default_base_position and args.final_base_pos is None:
            from primitives.shared.config import (
                DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION,
                get_object_base_offset_m,
            )
            offs = get_object_base_offset_m(args.object_name)
            if offs != (0.0, 0.0, 0.0):
                base_pos_eff = [
                    float(DEFAULT_BASE_POSITION[0]) + float(offs[0]),
                    float(DEFAULT_BASE_POSITION[1]) + float(offs[1]),
                    float(DEFAULT_BASE_POSITION[2]) + float(offs[2]),
                ]
                args.final_base_pos = base_pos_eff
                args.final_base_orientation = list(DEFAULT_BASE_ORIENTATION)
                args.use_default_base_position = False
                print(
                    f"[translate_object] applying PER_OBJECT_BASE_OFFSET_M for "
                    f"{args.object_name}: offset={tuple(round(1000*o,3) for o in offs)} mm "
                    f"→ final-base-pos={tuple(round(1000*v,3) for v in base_pos_eff)} mm",
                    flush=True,
                )

        # 2026-05-07: line_green and inverted_u_yellow route to the stash
        # prismatic_peg_insertion primitive (validated working for line_green —
        # XYZ + Rx/Ry compliance with geometric settling exit). The autonomous
        # SEARCH spiral can't break the gripper-jaws-on-base-rim standoff for
        # these wide-grasp parts; prismatic's tiered approach handles it.
        # u_brown / u_orange keep the validated compliant_insert path.
        if args.object_name in ('line_green', 'inverted_u_yellow'):
            # Per-object base offset already applied above (one branch handles
            # all objects). Continue with hover + prismatic dispatch.

            # Step 1: position the part above the slot via _run_hover
            hover_argv = [
                sys.executable, '-u', '-m',
                'compliant_insertion_studio.wrapper._run_hover',
                '--object-name', args.object_name,
                '--base-name', args.base_name,
                '--grasp-id', str(args.grasp_id),
                '--current-object-orientation',
                *[f"{v:.6f}" for v in args.current_object_orientation],
            ]
            if args.use_default_base_position:
                hover_argv.append('--use-default-base-position')
            if args.final_base_pos is not None:
                hover_argv += ['--final-base-pos', *[f"{v:.6f}" for v in args.final_base_pos]]
            if args.final_base_orientation is not None:
                hover_argv += ['--final-base-orientation',
                               *[f"{v:.6f}" for v in args.final_base_orientation]]
            print(f"[line_green route] hover: {' '.join(hover_argv)}", flush=True)
            r = subprocess.run(hover_argv, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if r.returncode != 0:
                _output({"result": "failure", "mode": args.mode, "movement_type": "insert",
                         "object_name": args.object_name,
                         "error": f"hover subprocess returned {r.returncode}"})
                sys.exit(1)
            # Step 2: run prismatic_peg_insertion (stash primitive)
            prism_argv = [
                sys.executable, '-u', '-m',
                'primitives._real_mode_stash.prismatic_peg_insertion',
                '--object-name', args.object_name,
                '--base-name', args.base_name,
                '--grasp-id', str(args.grasp_id),
                '--current-object-orientation',
                *[f"{v:.6f}" for v in args.current_object_orientation],
                '--force', '2.0',
                '--timeout', '60',
            ]
            if args.use_default_base_position:
                prism_argv.append('--use-default-base-position')
            if args.final_base_pos is not None:
                prism_argv += ['--final-base-pos', *[f"{v:.6f}" for v in args.final_base_pos]]
            if args.final_base_orientation is not None:
                prism_argv += ['--final-base-orientation',
                               *[f"{v:.6f}" for v in args.final_base_orientation]]
            print(f"[line_green route] prismatic: {' '.join(prism_argv)}", flush=True)
            r = subprocess.run(prism_argv, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            success = (r.returncode == 0)
            _output({
                "result": "success" if success else "failure",
                "mode": args.mode,
                "movement_type": "insert",
                "object_name": args.object_name,
                "base_name": args.base_name,
                "router": "line_green→prismatic_peg_insertion",
            })
            sys.exit(0 if success else 1)

        # Build wrapper argv. Hardcoded autonomous-search params match the
        # working production config validated against u_brown / u_orange runs
        # (see compliant_insertion_studio/scripts/loop_autonomous_insert.sh).
        wrapper_argv = [
            '-u', '-m', 'compliant_insertion_studio.wrapper.compliant_insert',
            '--object-name', args.object_name,
            '--base-name', args.base_name,
            '--grasp-id', str(args.grasp_id),
            '--current-object-orientation',
            *[f"{v:.6f}" for v in args.current_object_orientation],
            '--fz', '-9.0',
            '--step-back', 'auto',
            # The gate serves two purposes: clearing the operator, and letting
            # the arm come to rest before zero_ftsensor samples its baseline.
            # Only the first is redundant here. The ZERO phase retracts ~5.4mm,
            # and zeroing while that motion is still settling captures the
            # transient instead of a static baseline: measured 0.0s settle ->
            # post_zero_bias Fz = 15.6N, which force_mode then relieved by
            # driving the TCP 116mm UP before the no-contact timeout fired.
            # A 5.36s settle on the same arm/payload gave Fz = 0.11N.
            # Anything between 0.0s and 5.36s is uncharacterized; the
            # post-zero drift check in the wrapper is the actual guard.
            '--auto-step-back-seconds', '3.0',
            '--no-prompt-notes',
            '--override-fz-cap',
            # 2026-05-07: skip the PRE F/T smoke test (~5-6s) — saves time on
            # back-to-back inserts within one session. The wrapper itself
            # documents this as the right use case ("repeated rapid attempts").
            '--skip-smoke',
            '--autonomous-search',
            # 2026-05-07: bumped F_press 5→9, Fmax 5→8 after observing u_brown
            # SEARCH stalls when peg lands 1-2mm BELOW the rim (partially engaged
            # in slot edge). 5N press wasn't enough to drive the peg through the
            # chamfer, and 5N lateral wasn't enough to overcome partial-engagement
            # friction. With 9N press + 8N lateral, the same insert seated within
            # 1.7s of SEARCH. Real crashes are still well-bounded by --override-fz-cap.
            '--search-F-press-N', '9.0',
            '--search-Fmax-N', '8.0',
            '--search-v-s-mm-s', '5.0',
            '--search-pitch-mm', '2.0',
            '--search-R-max-mm', '8.0',
            '--search-max-duration-s', '60.0',
            '--timeout', '180',
            # Leave the EE at the seated pose: caller (run_assembly_step /
            # replay_real_assembly / etc.) is responsible for the post-insert
            # release + retract. Without this the wrapper's internal
            # safe_height move pulls the still-gripped peg straight out of
            # the slot before the caller's gripper-open fires.
            '--no-post-insert-move',
        ]
        if args.use_default_base_position:
            wrapper_argv.append('--use-default-base-position')
        if args.final_base_pos is not None:
            wrapper_argv += ['--final-base-pos', *[f"{v:.6f}" for v in args.final_base_pos]]
        if args.final_base_orientation is not None:
            wrapper_argv += ['--final-base-orientation',
                             *[f"{v:.6f}" for v in args.final_base_orientation]]
        # Echo a base-world-pose to the wrapper so its CAD prediction matches
        # whatever offset/override the caller injected via --final-base-pos.
        if args.final_base_pos is not None and args.final_base_orientation is not None:
            wrapper_argv += ['--base-world-pose',
                             *[f"{v:.6f}" for v in args.final_base_pos],
                             *[f"{v:.6f}" for v in args.final_base_orientation]]

        log.info(f"Compliant insert: delegating to compliant_insertion_studio.wrapper.compliant_insert")
        cmd = [sys.executable] + wrapper_argv
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=_make_env(),
        )
        lines = []
        t = threading.Thread(target=_stream, args=(proc.stdout, lines), daemon=True)
        t.start()
        try:
            rc = proc.wait(timeout=300)
            t.join(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            t.join(timeout=1.0)
            _output({"result": "failure", "mode": args.mode, "movement_type": "insert",
                     "error": "compliant_insert wrapper timed out (300s)"})
            sys.exit(1)
        out_text = '\n'.join(lines)
        rj = _extract_json(out_text)
        if rj is None:
            rj = {"result": "success" if rc == 0 else "failure",
                  "mode": args.mode, "movement_type": "insert"}
            if rc != 0:
                rj["error"] = "compliant_insert wrapper exited non-zero with no JSON output"
        else:
            rj.setdefault("mode", args.mode)
            rj["movement_type"] = "insert"
            rj.setdefault("object_name", args.object_name)
            rj.setdefault("base_name", args.base_name)
        _output(rj)
        sys.exit(0 if rc == 0 and rj.get("result") == "success" else 1)


if __name__ == '__main__':
    _subprocess_fast_path()
    # If we reach here, it's a ROS-node path — continue to heavy imports below.


import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from primitives.shared.ik import (
    forward_kinematics, dh_params, compute_cartesian_waypoints_ik,
    IKSolverConfig, IKSolver,
)
from primitives.shared.velocity_profiles import s_curve_profile, single_point, compute_duration
from primitives.shared.config import (
    TABLE_HEIGHT, GRIPPER_CENTER_TOOL_OFFSET,
    DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION,
)
from primitives.shared.collision import (
    check_collision_with_table, check_self_collision,
    check_ee_below_base, check_compact_configuration,
)
from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
from primitives.rotate_object import ExtendedCardinalOrientations
from utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir, get_symmetry_dir, find_assembly_json_by_base_name

# Configuration (auto-discovered)
ASSEMBLY_DATA_DIR = str(get_assembly_data_dir())
BASE_TOPIC = "/objects_poses_sim"
OBJECT_TOPIC = "/objects_poses_sim"
EE_TOPIC = "/tcp_pose_broadcaster/pose"
HOVER_HEIGHT = 0.15  # Height to hover above base before descending
ORIENTATION_TOLERANCE_DEG = 5.0  # Max orientation error before allowing insertion

# Set up Python logging for non-ROS contexts (subprocess helpers)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('translate_object')


def _json_dumps_decimal(obj, **kwargs):
    """json.dumps that outputs decimal notation instead of scientific notation.
    Prevents LLM tokenization issues where e.g. 1e-6 gets corrupted to 16."""
    s = json.dumps(obj, **kwargs)
    parts = re.split(r'("(?:[^"\\]|\\.)*")', s)
    for i, part in enumerate(parts):
        if not part.startswith('"'):
            parts[i] = re.sub(
                r'-?\d+\.?\d*[eE][+-]?\d+',
                lambda m: f'{float(m.group()):.10f}'.rstrip('0').rstrip('.'),
                part
            )
    return ''.join(parts)

def output_result(result):
    """Output JSON result with markers for MCP server parsing"""
    print("__RESULT_JSON__")
    print(_json_dumps_decimal(result))
    print("__END_RESULT_JSON__")


def load_grasp_point_position(object_name, grasp_id, logger=None):
    """Load grasp point position from grasp points JSON file."""
    data_dir = get_aruco_data_dir() / "grasp_points"
    json_path = data_dir / f"{object_name}_grasp_points.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for gp in data.get('grasp_points', []):
                if gp['id'] == grasp_id:
                    pos = gp['position']
                    if logger:
                        logger.info(f"Loaded grasp point {grasp_id} for {object_name}: [{pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f}]")
                    return np.array([pos['x'], pos['y'], pos['z']])
            if logger:
                logger.warn(f"Grasp ID {grasp_id} not found in {json_path.name}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            if logger:
                logger.error(f"Error reading {json_path}: {e}")
    if logger:
        logger.error(f"No grasp points file found for '{object_name}'")
    return None


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class TranslateObject(Node):
    def __init__(self, mode):
        super().__init__('translate_object')

        if mode not in ('sim', 'real'):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'sim' or 'real'.")
        self.mode = mode

        # Error tracking for JSON output
        self.error_message = None
        self.object_name = None
        self.base_name = None

        # Assembly config (lazy-loaded per base)
        self.assembly_config = {}
        self.assembly_json_file = None
        self.loaded_base_name = None

        # Pose subscribers — topics only needed in sim mode
        if self.mode == 'sim':
            self.base_sub = self.create_subscription(TFMessage, BASE_TOPIC, self._base_cb, 10)
            self.object_sub = self.create_subscription(TFMessage, OBJECT_TOPIC, self._object_cb, 10)
        else:
            self.base_sub = None
            self.object_sub = None

        ee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.ee_sub = self.create_subscription(PoseStamped, EE_TOPIC, self._ee_cb, ee_qos)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        self.current_poses = {}
        self.current_ee_pose = None
        self.current_joint_angles = None
        self.joint_angles_received = False

        # Action client
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
        )

        self.get_logger().info(f"Using {self.mode.upper()} mode")

    # --- Callbacks ---

    def _base_cb(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform

    def _object_cb(self, msg):
        for transform in msg.transforms:
            self.current_poses[transform.child_frame_id] = transform

    def _ee_cb(self, msg):
        self.current_ee_pose = msg

    def _joint_state_cb(self, msg: JointState):
        if len(msg.name) == 6 and len(msg.position) == 6:
            joint_dict = dict(zip(msg.name, msg.position))
            ordered = [joint_dict[n] for n in self.joint_names if n in joint_dict]
            if len(ordered) == 6:
                self.current_joint_angles = np.array(ordered)
                self.joint_angles_received = True

    # --- Helpers ---

    def transform_to_matrix(self, transform):
        """Convert ROS Transform to 4x4 matrix."""
        t = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        q = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def pose_to_matrix(self, pose):
        """Convert ROS Pose to 4x4 matrix."""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def matrix_to_rpy(self, T):
        """Convert 4x4 matrix to (position, rpy_degrees)."""
        position = T[:3, 3]
        rpy_deg = np.degrees(R.from_matrix(T[:3, :3]).as_euler('xyz'))
        # Canonicalize
        roll, pitch, yaw = rpy_deg
        if abs(pitch) < 5 and abs(abs(roll) - 180) < 5:
            rpy_deg = np.array([0.0, 180.0, (yaw % 360) - 180])
        return position, rpy_deg

    def load_assembly_config(self, base_name):
        """Load assembly JSON for the given base."""
        json_file = find_assembly_json_by_base_name(base_name, ASSEMBLY_DATA_DIR, self.get_logger())
        if json_file:
            self.assembly_json_file = json_file
            self.loaded_base_name = base_name
        else:
            self.get_logger().error(f"Could not find assembly JSON for base '{base_name}'")
            return {}
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.get_logger().error(f"Error loading assembly JSON: {e}")
            return {}

    def get_object_target_position(self, object_name):
        """Get target position for object from assembly config (relative to base)."""
        for comp in self.assembly_config.get('components', []):
            if comp.get('name') == object_name:
                p = comp.get('position', {})
                return np.array([p.get('x', 0), p.get('y', 0), p.get('z', 0)])
        return None

    def get_object_target_orientation(self, object_name):
        """Get target orientation quaternion for object from assembly config (relative to base)."""
        for comp in self.assembly_config.get('components', []):
            if comp.get('name') == object_name:
                q = comp.get('rotation', {}).get('quaternion', {})
                return np.array([q.get('x', 0.0), q.get('y', 0.0), q.get('z', 0.0), q.get('w', 1.0)])
        return None

    def read_current_joint_angles(self):
        """Block until joint angles arrive (10s timeout)."""
        self.joint_angles_received = False
        timeout_count = 0
        while rclpy.ok() and not self.joint_angles_received and timeout_count < 100:
            rclpy.spin_once(self, timeout_sec=0.1)
            timeout_count += 1
        if not self.joint_angles_received or self.current_joint_angles is None:
            self.get_logger().error("Timeout waiting for joint angles")
            return None
        return self.current_joint_angles.copy()

    def compute_ik_with_current_seed(self, target_position, target_quat, max_tries=5, dx=0.001):
        """Compute IK using current joint angles as seed."""
        target_rotation = R.from_quat(target_quat)
        target_pose = np.eye(4)
        target_pose[:3, 3] = target_position
        target_pose[:3, :3] = target_rotation.as_matrix()

        if self.current_joint_angles is None:
            self.get_logger().error("Current joint angles not available!")
            return None

        def collision_checker(joint_angles):
            if self.mode != 'sim':
                return False
            return (check_collision_with_table(joint_angles)
                    or check_self_collision(joint_angles)
                    or check_ee_below_base(joint_angles)
                    or check_compact_configuration(joint_angles))

        joint_bounds = [
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-2 * np.pi, 2 * np.pi),
        ]
        solver = IKSolver(IKSolverConfig(joint_bounds=joint_bounds))
        result = solver.solve(
            seeds=[self.current_joint_angles.copy()],
            target_pose=target_pose,
            collision_checker=collision_checker,
            perturbations=max_tries,
            dx=dx,
        )
        if result is not None:
            return result
        self.get_logger().error("Motion planning failed: no reachable joint configuration exists for the target position")
        return None

    def execute_trajectory(self, trajectory):
        """Execute trajectory via FollowJointTrajectory action (blocking)."""
        try:
            if 'traj1' not in trajectory or not trajectory['traj1']:
                return False

            points = trajectory['traj1']
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            for point in points:
                tp = JointTrajectoryPoint()
                tp.positions = point['positions']
                if 'velocities' in point:
                    tp.velocities = point['velocities']
                tp.time_from_start = point['time_from_start']
                traj_msg.points.append(tp)

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj_msg
            goal.goal_time_tolerance = Duration(sec=1)

            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                self.error_message = "External control program stopped or robot in protective stop"
                self.get_logger().error(self.error_message)
                return False

            self.get_logger().info(f"Trajectory with {len(points)} waypoints sent and accepted")
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result()

            if result.status == 4:  # SUCCEEDED
                self.get_logger().info("Movement completed successfully")
                return True
            else:
                result_msg = result.result
                if result_msg.error_code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                    self.error_message = "Velocity or acceleration limits exceeded. Enable robot in URcap to fix this."
                else:
                    self.error_message = f"Trajectory failed with status code {result.status}"
                self.get_logger().error(self.error_message)
                return False
        except Exception as e:
            self.error_message = f"Trajectory execution error: {e}"
            self.get_logger().error(self.error_message)
            return False

    # ------------------------------------------------------------------
    # Sim mode: translate to target (hover or final position)
    # ------------------------------------------------------------------

    def translate_for_target_sim(self, object_name, base_name, hover=True, object_orientation=None):
        """
        Sim mode: Compute and execute EE translation.

        Args:
            object_name: Name of the held object
            base_name: Name of the base object
            hover: If True, target is HOVER_HEIGHT above base (move-to-base).
                   If False, target is final object position from JSON (perform-insert sim).
            object_orientation: Optional quaternion [x,y,z,w]. When provided, used for
                orientation verification instead of reading from sim topic. Enables
                ablation mode where the agent must carry forward orientation values.
        """
        self.object_name = object_name
        self.base_name = base_name

        # Load assembly config
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        # Wait for pose data
        if not self.current_poses or self.current_ee_pose is None:
            self.error_message = "No pose data available"
            self.get_logger().error(self.error_message)
            return False

        if object_name not in self.current_poses:
            self.error_message = f"Object {object_name} not found"
            self.get_logger().error(self.error_message)
            return False

        if base_name not in self.current_poses:
            self.error_message = f"Base {base_name} not found"
            self.get_logger().error(self.error_message)
            return False

        # Convert poses to matrices
        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        T_object_current = self.transform_to_matrix(self.current_poses[object_name].transform)
        T_base_current = self.transform_to_matrix(self.current_poses[base_name].transform)

        # Verify orientation before moving to base or inserting
        target_quat = self.get_object_target_orientation(object_name)
        if target_quat is None:
            self.error_message = f"No target orientation found for {object_name} in assembly config"
            self.get_logger().error(self.error_message)
            return False
        # Use agent-provided orientation when available (ablation mode),
        # otherwise fall back to sim topic (default behavior)
        if object_orientation is not None:
            quat_array = np.array(object_orientation, dtype=float)
            if np.any(np.abs(quat_array) > 1.0):
                self.error_message = f"current_object_orientation has component(s) outside [-1, 1]: {quat_array.tolist()}"
                self.get_logger().error(self.error_message)
                return False
            quat_norm_sq = float(np.sum(quat_array ** 2))
            if abs(quat_norm_sq - 1.0) > 0.02:
                self.error_message = f"current_object_orientation norm² = {quat_norm_sq:.4f} (expected ~1.0): {quat_array.tolist()}"
                self.get_logger().error(self.error_message)
                return False
            R_object_current = R.from_quat(object_orientation)
            self.get_logger().info(f"Using agent-provided object orientation: {object_orientation}")
        else:
            R_object_current = R.from_matrix(T_object_current[:3, :3])
        R_base = R.from_matrix(T_base_current[:3, :3])
        R_relative = R.from_matrix(R_base.as_matrix().T @ R_object_current.as_matrix())
        R_target_relative = R.from_quat(target_quat)
        symmetry_dir = str(get_symmetry_dir())
        fold_data = load_symmetry_data(object_name, symmetry_dir)
        equivalents = equivalent_orientations(R_target_relative.as_matrix(), fold_data)
        best_R_eq = equivalents[0]
        min_error_rad = float('inf')
        for R_eq in equivalents:
            err = (R_relative.inv() * R.from_matrix(R_eq)).magnitude()
            if err < min_error_rad:
                min_error_rad = err
                best_R_eq = R_eq
        min_error_deg = np.degrees(min_error_rad)
        if min_error_deg > ORIENTATION_TOLERANCE_DEG:
            self.error_message = (
                f"Object orientation error is {min_error_deg:.1f}° (tolerance: {ORIENTATION_TOLERANCE_DEG}°). "
                f"Call rotate_object before insert."
            )
            self.get_logger().error(self.error_message)
            return False
        self.get_logger().info(f"Orientation verified: {min_error_deg:.1f}° error (tolerance: {ORIENTATION_TOLERANCE_DEG}°)")

        # Snap to closest fold-equivalent orientation (world frame) so the
        # target position accounts for symmetry like real mode does.
        R_object_snapped = R_base.as_matrix() @ best_R_eq

        # Calculate grasp transformation
        T_grasp = np.linalg.inv(T_EE_current) @ T_object_current

        ee_current_position, _ = self.matrix_to_rpy(T_EE_current)
        base_current_position, _ = self.matrix_to_rpy(T_base_current)

        # Get target object position from JSON (relative to base)
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False

        # Transform to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative

        # Use snapped orientation for target (not raw current) to eliminate
        # position drift from orientation error propagating through T_grasp
        T_object_target = np.eye(4)
        T_object_target[:3, :3] = R_object_snapped
        T_object_target[:3, 3] = target_object_position_abs

        # Required EE position to place object at target
        T_EE_target = T_object_target @ np.linalg.inv(T_grasp)

        ee_target_position = T_EE_target[:3, 3]
        ee_target_rot_matrix = T_EE_target[:3, :3]

        if hover:
            # Hover: same XY as target, Z = base + HOVER_HEIGHT
            hover_gripper_center = ee_target_position.copy()
            hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT
            tool_offset_world = ee_target_rot_matrix @ GRIPPER_CENTER_TOOL_OFFSET
            target_flange = hover_gripper_center - tool_offset_world
            self.get_logger().info(
                f"Hover gripper center Z: {hover_gripper_center[2]:.4f}, "
                f"hover flange Z: {target_flange[2]:.4f} (offset: {tool_offset_world[2]:.4f})"
            )
        else:
            # Final position: use EE target directly
            target_flange = ee_target_position
            self.get_logger().info(f"Target flange position: {target_flange}")

        self.get_logger().info(
            f"Final object position: [{target_object_position_abs[0]:.4f}, "
            f"{target_object_position_abs[1]:.4f}, {target_object_position_abs[2]:.4f}]"
        )

        # Read current joint angles
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        if hover:
            # hover: single-point joint-space move — let UR controller
            # handle interpolation to avoid protective stops from Jacobian IK
            # velocity spikes on long lateral moves.
            target_quat = R.from_matrix(ee_target_rot_matrix).as_quat()
            ik_result = self.compute_ik_with_current_seed(target_flange, target_quat)
            if ik_result is None:
                self.error_message = "Motion planning failed: no collision-free path to the target position could be computed"
                self.get_logger().error(self.error_message)
                return False

            joint_dist = float(np.max(np.abs(np.array(ik_result) - np.array(self.current_joint_angles))))
            total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
            self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

            profile = single_point(ik_result, total_duration)
            trajectory_points = []
            for positions, velocities, t_i in profile:
                trajectory_points.append({
                    "positions": positions,
                    "velocities": velocities,
                    "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                })
        else:
            # insert: Cartesian Jacobian IK for precise straight-line descent
            num_waypoints = 60
            self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
            waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles,
                target_z=target_flange[2],
                num_waypoints=num_waypoints,
                target_pos=target_flange.tolist() if hasattr(target_flange, 'tolist') else list(target_flange),
                target_orientation=ee_target_rot_matrix,
            )
            if waypoints is None:
                self.error_message = "Motion planning failed: no collision-free path to the target position could be computed"
                self.get_logger().error(self.error_message)
                return False

            # Post-hoc collision check
            for i, wp_joints in enumerate(waypoints):
                if (check_collision_with_table(wp_joints)
                        or check_self_collision(wp_joints)
                        or check_ee_below_base(wp_joints)
                        or check_compact_configuration(wp_joints)):
                    self.get_logger().error(f"Collision detected at waypoint {i + 1}/{num_waypoints}")
                    self.error_message = "Motion planning failed: couldn't find a collision-free path to the target position"
                    return False

            all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

            joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
            total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
            self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

            profile = s_curve_profile(all_joint_angles, total_duration)
            trajectory_points = []
            for positions, velocities, t_i in profile:
                trajectory_points.append({
                    "positions": positions,
                    "velocities": velocities,
                    "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                })

            self.get_logger().info(
                f"Generated {len(trajectory_points)} Cartesian waypoints with s-curve velocity profile"
            )

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            return False

        if not hover:
            # For insert: just wait and return
            time.sleep(0.5)
            self.get_logger().info("Insert movement completed successfully")

        return success

    # ------------------------------------------------------------------
    # Real mode: translate to hover with grasp offsets + fold symmetry
    # ------------------------------------------------------------------

    def translate_for_target_real(self, object_name, base_name,
                                  final_base_pos=None, final_base_orientation=None,
                                  use_default_base=False, grasp_id=None,
                                  object_orientation=None):
        """
        Real mode: Calculate and execute EE translation to hover position.
        Uses provided base position/orientation (no sim topics).
        """
        self.object_name = object_name
        self.base_name = base_name

        if grasp_id is None:
            self.error_message = "grasp_id is required for real mode"
            self.get_logger().error(self.error_message)
            return False
        if object_orientation is None:
            self.error_message = "object_orientation is required for real mode"
            self.get_logger().error(self.error_message)
            return False

        # Load assembly config
        if self.loaded_base_name != base_name:
            self.assembly_config = self.load_assembly_config(base_name=base_name)
            if not self.assembly_config:
                self.error_message = f"Failed to load assembly config for base '{base_name}'"
                self.get_logger().error(self.error_message)
                return False

        if self.current_ee_pose is None:
            self.error_message = "End-effector pose not available"
            self.get_logger().error(self.error_message)
            return False

        # Resolve base position/orientation
        if final_base_pos is None:
            if use_default_base:
                final_base_pos = DEFAULT_BASE_POSITION
                self.get_logger().info(f"Using default base position: {final_base_pos}")
            else:
                self.error_message = "Base position not provided. Use --final-base-pos or --use-default-base-position"
                self.get_logger().error(self.error_message)
                return False

        if final_base_orientation is None:
            if use_default_base:
                final_base_orientation = DEFAULT_BASE_ORIENTATION
                self.get_logger().info(f"Using default base orientation: {final_base_orientation}")
            else:
                final_base_orientation = [0.0, 0.0, 0.0, 1.0]
                self.get_logger().info("Using identity base orientation (not provided)")

        # Create base pose matrix from args
        base_pose = PoseStamped()
        base_pose.pose.position.x = final_base_pos[0]
        base_pose.pose.position.y = final_base_pos[1]
        base_pose.pose.position.z = final_base_pos[2]
        base_pose.pose.orientation.x = final_base_orientation[0]
        base_pose.pose.orientation.y = final_base_orientation[1]
        base_pose.pose.orientation.z = final_base_orientation[2]
        base_pose.pose.orientation.w = final_base_orientation[3]
        T_base_current = self.pose_to_matrix(base_pose.pose)
        self.get_logger().info(f"Using base position: {final_base_pos}, orientation: {final_base_orientation}")

        T_EE_current = self.pose_to_matrix(self.current_ee_pose.pose)
        ee_current_position, _ = self.matrix_to_rpy(T_EE_current)
        base_current_position, _ = self.matrix_to_rpy(T_base_current)

        # Target object position from JSON
        target_position_relative = self.get_object_target_position(object_name)
        if target_position_relative is None:
            self.error_message = f"No target position found for {object_name} in JSON"
            self.get_logger().error(self.error_message)
            return False

        target_orientation_relative = self.get_object_target_orientation(object_name)
        if target_orientation_relative is None:
            self.get_logger().warn(f"No target orientation found for {object_name}, using identity")
            target_orientation_relative = np.array([0.0, 0.0, 0.0, 1.0])

        # Transform to world frame
        R_base_current = T_base_current[:3, :3]
        target_object_position_abs = base_current_position + R_base_current @ target_position_relative
        R_target_relative = R.from_quat(target_orientation_relative).as_matrix()
        R_target_abs = R_base_current @ R_target_relative
        target_orientation_abs = R.from_matrix(R_target_abs).as_quat()

        self.get_logger().info(f"Target object position (world): {target_object_position_abs}")
        self.get_logger().info(f"Target object orientation (world): {target_orientation_abs}")
        self.get_logger().info("Keeping current EE orientation unchanged (from reorient step)")

        # Load grasp point offset
        grasp_offset = load_grasp_point_position(object_name, grasp_id, logger=self.get_logger())
        if grasp_offset is None:
            self.error_message = f"Could not load grasp point {grasp_id} for '{object_name}'"
            self.get_logger().error(self.error_message)
            return False

        # Validate quaternion
        quat_array = np.array(object_orientation, dtype=float)
        if np.any(np.abs(quat_array) > 1.0):
            self.get_logger().error(f"Invalid quaternion: component(s) outside [-1, 1]: {quat_array.tolist()}")
            self.error_message = f"current_object_orientation has component(s) outside [-1, 1]: {quat_array.tolist()}"
            return False
        quat_norm_sq = float(np.sum(quat_array ** 2))
        if abs(quat_norm_sq - 1.0) > 0.02:
            self.get_logger().error(f"Invalid quaternion: norm² = {quat_norm_sq:.4f} (expected ~1.0): {quat_array.tolist()}")
            self.error_message = f"current_object_orientation norm² = {quat_norm_sq:.4f} (expected ~1.0): {quat_array.tolist()}"
            return False

        # Fold symmetry: snap to closest equivalent orientation
        R_object_current = R.from_quat(object_orientation).as_matrix()
        symmetry_dir = str(get_symmetry_dir())
        fold_data = load_symmetry_data(object_name, symmetry_dir)

        if fold_data is not None:
            equivalents = equivalent_orientations(R_target_abs, fold_data)
            best_pos_error = float('inf')
            best_orientation_error = float('inf')
            R_grasp_rotation = R_target_abs
            for R_eq in equivalents:
                orientation_error = ExtendedCardinalOrientations.rotation_matrix_distance(R_object_current, R_eq)
                grasp_world_offset_candidate = R_eq @ grasp_offset
                pos_error = np.linalg.norm(grasp_world_offset_candidate - (R_object_current @ grasp_offset))
                if pos_error < best_pos_error or (pos_error == best_pos_error and orientation_error < best_orientation_error):
                    best_pos_error = pos_error
                    best_orientation_error = orientation_error
                    R_grasp_rotation = R_eq
            self.get_logger().info(
                f"Snapped orientation to closest equivalent "
                f"(angle error: {np.degrees(best_orientation_error):.1f}°, "
                f"position error: {best_pos_error * 1000:.2f}mm)"
            )
        else:
            self.get_logger().info("No symmetry data, using current object orientation")
            R_grasp_rotation = R_object_current

        grasp_world_offset = R_grasp_rotation @ grasp_offset
        self.get_logger().info(f"Grasp point {grasp_id} offset (CAD frame): {grasp_offset}")
        self.get_logger().info(f"Grasp point offset (world frame): {grasp_world_offset}")

        # Target gripper center
        target_gripper_center = target_object_position_abs + grasp_world_offset
        hover_gripper_center = target_gripper_center.copy()
        hover_gripper_center[2] = base_current_position[2] + HOVER_HEIGHT

        self.get_logger().info(f"Target gripper center: {target_gripper_center}")
        self.get_logger().info(f"Hover gripper center (with {HOVER_HEIGHT}m offset): {hover_gripper_center}")

        # Read joints
        if self.current_joint_angles is None:
            if self.read_current_joint_angles() is None:
                self.error_message = "Could not read current joint angles"
                self.get_logger().error(self.error_message)
                return False

        # Convert gripper center to flange using FK-derived rotation
        T_fk = forward_kinematics(dh_params, self.current_joint_angles)
        R_fk = T_fk[:3, :3]
        tool_offset_world = R_fk @ GRIPPER_CENTER_TOOL_OFFSET
        hover_flange = hover_gripper_center - tool_offset_world
        self.get_logger().info(f"Hover flange position (FK-derived): {hover_flange}")

        # Jacobian IK
        num_waypoints = 60
        self.get_logger().info("Computing dense IK waypoints (Jacobian)...")
        waypoints = compute_cartesian_waypoints_ik(
            self.current_joint_angles,
            target_z=hover_flange[2],
            num_waypoints=num_waypoints,
            target_pos=hover_flange.tolist(),
        )
        if waypoints is None:
            self.error_message = "Motion planning failed: no collision-free path to the target position could be computed"
            self.get_logger().error(self.error_message)
            return False

        all_joint_angles = [self.current_joint_angles.copy()] + list(waypoints)

        joint_dist = float(np.max(np.abs(np.array(waypoints[-1]) - np.array(self.current_joint_angles))))
        total_duration = compute_duration(joint_distance=joint_dist, profile='s_curve')
        self.get_logger().info(f"Duration: {total_duration:.2f}s (joint={joint_dist:.2f}rad)")

        profile = s_curve_profile(all_joint_angles, total_duration)
        trajectory_points = []
        for positions, velocities, t_i in profile:
            trajectory_points.append({
                "positions": positions,
                "velocities": velocities,
                "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
            })

        self.get_logger().info(f"Generated {len(trajectory_points)} Cartesian waypoints with s-curve velocity profile")

        success = self.execute_trajectory({"traj1": trajectory_points})
        if not success:
            self.get_logger().error("Failed to reach target position")
            return False

        # Closed-loop correction
        CORRECTION_THRESHOLD = 0.00025  # 0.25mm
        MAX_CORRECTIONS = 3

        for correction_iter in range(MAX_CORRECTIONS):
            self.joint_angles_received = False
            self.current_ee_pose = None
            timeout = 0
            while rclpy.ok() and (not self.joint_angles_received or self.current_ee_pose is None) and timeout < 50:
                rclpy.spin_once(self, timeout_sec=0.1)
                timeout += 1

            if not self.joint_angles_received or self.current_ee_pose is None:
                self.get_logger().warn("Could not read pose data for correction")
                break

            ee_pos_topic = np.array([self.current_ee_pose.pose.position.x,
                                     self.current_ee_pose.pose.position.y,
                                     self.current_ee_pose.pose.position.z])
            ee_quat_topic = np.array([self.current_ee_pose.pose.orientation.x,
                                      self.current_ee_pose.pose.orientation.y,
                                      self.current_ee_pose.pose.orientation.z,
                                      self.current_ee_pose.pose.orientation.w])
            R_ee_topic = R.from_quat(ee_quat_topic).as_matrix()
            actual_gripper_center = ee_pos_topic + R_ee_topic @ GRIPPER_CENTER_TOOL_OFFSET
            gripper_center_error = hover_gripper_center - actual_gripper_center
            pos_error = np.linalg.norm(gripper_center_error)

            self.get_logger().info(
                f"Correction check {correction_iter + 1}: gripper center error = {pos_error * 1000:.2f}mm "
                f"(actual: [{actual_gripper_center[0] * 1000:.1f}, {actual_gripper_center[1] * 1000:.1f}, {actual_gripper_center[2] * 1000:.1f}]mm)"
            )

            if pos_error <= CORRECTION_THRESHOLD:
                self.get_logger().info(f"Position accuracy OK ({pos_error * 1000:.2f}mm <= {CORRECTION_THRESHOLD * 1000:.2f}mm)")
                break

            # Apply correction in FK space
            T_fk_current = forward_kinematics(dh_params, self.current_joint_angles)
            current_flange_fk = T_fk_current[:3, 3]
            corrected_flange = current_flange_fk + gripper_center_error

            self.get_logger().info(f"Applying correction move (error: {pos_error * 1000:.2f}mm)...")
            correction_waypoints = compute_cartesian_waypoints_ik(
                self.current_joint_angles,
                target_z=corrected_flange[2],
                num_waypoints=20,
                target_pos=corrected_flange.tolist(),
            )
            if correction_waypoints is None:
                self.get_logger().warn("Motion planning failed for post-placement correction, skipping")
                break

            corr_all = [self.current_joint_angles.copy()] + list(correction_waypoints)
            corr_n = len(corr_all)
            corr_duration = 1.0
            corr_points = []
            for i in range(corr_n):
                t_i = corr_duration * i / (corr_n - 1)
                if i == 0 or i == corr_n - 1:
                    vels = [0.0] * 6
                else:
                    delta = corr_all[min(i + 1, corr_n - 1)] - corr_all[max(i - 1, 0)]
                    dn = np.linalg.norm(delta)
                    if dn > 1e-8:
                        vels = [float(delta[j] / dn * pos_error / corr_duration) for j in range(6)]
                    else:
                        vels = [0.0] * 6
                corr_points.append({
                    "positions": [float(x) for x in corr_all[i]],
                    "velocities": vels,
                    "time_from_start": Duration(sec=int(t_i), nanosec=int((t_i - int(t_i)) * 1e9))
                })

            if not self.execute_trajectory({"traj1": corr_points}):
                self.get_logger().warn("Correction trajectory failed")
                break

        return True


# ---------------------------------------------------------------------------
# Subprocess helpers (for actions that remain as subprocesses)
# ---------------------------------------------------------------------------

def extract_json_from_output(output_text):
    """Extract JSON result from subprocess output."""
    if "__RESULT_JSON__" in output_text and "__END_RESULT_JSON__" in output_text:
        start = output_text.find("__RESULT_JSON__") + len("__RESULT_JSON__")
        end = output_text.find("__END_RESULT_JSON__")
        json_str = output_text[start:end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def stream_output(pipe, output_lines, prefix=""):
    """Stream subprocess output line by line."""
    for line in iter(pipe.readline, ''):
        if line:
            line = line.rstrip()
            if line:
                output_lines.append(line)
                logger.info(f"{prefix}{line}")
    pipe.close()


def _make_env():
    """Create environment with PYTHONPATH for subprocess."""
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    return env


def run_subprocess(script_path, cmd_args=None, timeout=None):
    """Run a subprocess script and return (success, output_text)."""
    cmd = [sys.executable, script_path] + (cmd_args or [])
    env = _make_env()
    output_lines = []

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    output_thread = threading.Thread(
        target=stream_output, args=(process.stdout, output_lines), daemon=True,
    )
    output_thread.start()

    try:
        returncode = process.wait(timeout=timeout)
        output_thread.join(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        output_thread.join(timeout=1.0)
        return False, '\n'.join(output_lines)

    output_text = '\n'.join(output_lines)
    return returncode == 0, output_text


def run_perform_insert_real(args):
    """Run real-mode insertion subprocess."""
    insertion_type = getattr(args, 'insertion_type', 'prismatic')

    if insertion_type == 'prismatic':
        script_path = os.path.join(os.path.dirname(__file__), 'prismatic_peg_insertion.py')
        logger.info("Using prismatic peg insertion")
    elif insertion_type == 'legacy':
        script_path = os.path.join(os.path.dirname(__file__), '_real_mode_stash', 'legacy', 'peg_in_hole_insert.py')
        logger.info("Using legacy peg_in_hole_insert")
    else:
        return False, f"Unknown insertion type: {insertion_type}"

    cmd_args = []
    if args.object_name:
        cmd_args.extend(['--object-name', args.object_name])
    if args.base_name:
        cmd_args.extend(['--base-name', args.base_name])
    if args.grasp_id is not None:
        cmd_args.extend(['--grasp-id', str(args.grasp_id)])
    if args.final_base_pos:
        cmd_args.extend(['--final-base-pos'] + [str(x) for x in args.final_base_pos])
    if args.final_base_orientation:
        cmd_args.extend(['--final-base-orientation'] + [str(x) for x in args.final_base_orientation])
    if args.use_default_base_position:
        cmd_args.append('--use-default-base-position')
    if args.current_object_orientation is not None:
        cmd_args.extend(['--current-object-orientation'] + [str(x) for x in args.current_object_orientation])

    logger.info("Moving down with passive compliance")
    return run_subprocess(script_path, cmd_args)


def run_move_to_clear_area(object_name=None, mode=None):
    """Run move_to_clear_area subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), 'core', 'move_to_clear_area.py')
    logger.info("Moving object to clear area")
    cmd_args = ['--move']
    if object_name:
        cmd_args += ['--object-name', object_name]
    if mode:
        cmd_args += ['--mode', mode]
    return run_subprocess(script_path, cmd_args)


def run_move_down(mode=None):
    """Run move_down subprocess."""
    script_path = os.path.join(os.path.dirname(__file__), 'core', 'move_down.py')
    logger.info("Lowering object onto table")
    cmd_args = ['--mode', mode] if mode else []
    return run_subprocess(script_path, cmd_args, timeout=310)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Translate Object - Move held object to assembly position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--mode', type=str, required=True, choices=['sim', 'real'])
    parser.add_argument('--object-name', type=str)
    parser.add_argument('--base-name', type=str)

    # Movement flags (mutually exclusive)
    parser.add_argument('--insert', action='store_true')
    parser.add_argument('--place-down', action='store_true', dest='place_down')

    # Real mode arguments
    parser.add_argument('--final-base-pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--final-base-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--use-default-base-position', action='store_true', dest='use_default_base_position')
    parser.add_argument('--grasp-id', type=int, default=None)
    parser.add_argument('--current-object-orientation', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--insertion-type', type=str, default='compliant',
                        choices=['compliant', 'prismatic', 'legacy'],
                        help="Real-mode insert backend. 'compliant' (default) routes to "
                             "the compliant_insertion_studio autonomous-search wrapper "
                             "via the fast path. 'prismatic'/'legacy' fall through to "
                             "the original real-mode hover+subprocess path.")

    args = parser.parse_args()

    # Validate flags
    flags_set = sum([args.insert, args.place_down])
    if flags_set == 0:
        parser.error("Specify one of --insert, --place-down")
    if flags_set > 1:
        parser.error("Cannot use multiple movement flags together")

    # Validate sim mode requirements
    if args.mode == 'sim' and not args.place_down:
        if args.object_name is None:
            parser.error("--object-name is required in sim mode")
        if args.base_name is None:
            parser.error("--base-name is required in sim mode")

    # Validate real mode requirements
    if args.mode == 'real' and args.insert:
        if args.base_name is None:
            parser.error("--base-name is required for --insert in real mode")
        if not args.use_default_base_position and args.final_base_pos is None:
            parser.error("--final-base-pos or --use-default-base-position required in real mode")

    # --- Validate quaternion if provided (all actions) ---

    if args.current_object_orientation is not None:
        quat_array = np.array(args.current_object_orientation, dtype=float)
        if np.any(np.abs(quat_array) > 1.0):
            output_result({"result": "failure", "mode": args.mode,
                           "error": f"current_object_orientation has component(s) outside [-1, 1]: {quat_array.tolist()}"})
            sys.exit(1)
        quat_norm_sq = float(np.sum(quat_array ** 2))
        if abs(quat_norm_sq - 1.0) > 0.02:
            output_result({"result": "failure", "mode": args.mode,
                           "error": f"current_object_orientation norm² = {quat_norm_sq:.4f} (expected ~1.0): {quat_array.tolist()}"})
            sys.exit(1)

    # --- Subprocess-only paths (no ROS node needed) ---

    if args.place_down:
        success, output_text = run_move_to_clear_area(object_name=args.object_name, mode=args.mode)
        if not success:
            subprocess_json = extract_json_from_output(output_text)
            if subprocess_json:
                subprocess_json["movement_type"] = "place_down"
                subprocess_json["mode"] = args.mode
                output_result(subprocess_json)
            else:
                output_result({
                    "result": "failure", "mode": args.mode,
                    "movement_type": "place_down",
                    "error": "move_to_clear_area failed",
                })
            sys.exit(1)
        # Step 2: lower onto table
        success, output_text = run_move_down(mode=args.mode)
        subprocess_json = extract_json_from_output(output_text)
        if subprocess_json:
            subprocess_json["movement_type"] = "place_down"
            subprocess_json["mode"] = args.mode
            output_result(subprocess_json)
        else:
            output_result({
                "result": "success" if success else "failure",
                "mode": args.mode,
                "movement_type": "place_down",
                **({"error": "move_down failed"} if not success else {}),
            })
        sys.exit(0 if success else 1)

    # --- ROS node paths (sim insert, real insert) ---

    # Verify grasp before insert
    if args.mode == 'sim' and args.object_name:
        movement_type = "insert"
        logger.info(f"Verifying grasp on {args.object_name} before {movement_type}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vg_script = os.path.join(os.path.dirname(script_dir), 'queries', 'verify_grasp.py')
        vg_result = subprocess.run(
            [sys.executable, vg_script, '--object-name', args.object_name, '--mode', args.mode],
            capture_output=True, text=True, timeout=15
        )
        vg_out = (vg_result.stdout or '') + (vg_result.stderr or '')
        if '__RESULT_JSON__' in vg_out:
            json_str = vg_out[vg_out.rfind('__RESULT_JSON__') + len('__RESULT_JSON__'):vg_out.rfind('__END_RESULT_JSON__')].strip()
            try:
                vj = json.loads(json_str)
            except json.JSONDecodeError:
                vj = None
        else:
            vj = None
        if vg_result.returncode != 0 or (vj and vj.get('result') == 'failure'):
            err = vj.get('error', 'grasp check failed') if vj else 'grasp check failed'
            output_result({"result": "failure", "mode": args.mode, "movement_type": movement_type,
                     "error": f"Grasp check failed before {movement_type}: {err}"})
            sys.exit(1)

    rclpy.init()
    node = None
    success = False
    error = None

    try:
        node = TranslateObject(mode=args.mode)
        node.action_client.wait_for_server()

        # Wait for EE pose (always needed)
        while node.current_ee_pose is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        # In sim mode, wait for object and base poses
        if args.mode == 'sim':
            while not node.current_poses:
                rclpy.spin_once(node, timeout_sec=0.1)

        if args.insert:
            if args.mode == 'sim':
                success = node.translate_for_target_sim(args.object_name, args.base_name, hover=True, object_orientation=args.current_object_orientation)
                if success:
                    # Let the robot settle at hover before reading fresh poses.
                    # The old two-subprocess approach had ~3-5s of LLM round-trip
                    # between hover and insert; this replicates that settling window.
                    time.sleep(1.0)
                    node.current_ee_pose = None
                    while node.current_ee_pose is None:
                        rclpy.spin_once(node, timeout_sec=0.1)
                    node.current_joint_angles = None
                    node.read_current_joint_angles()
                    success = node.translate_for_target_sim(args.object_name, args.base_name, hover=False, object_orientation=args.current_object_orientation)
            else:
                # Real mode: hover above base, then insertion subprocess
                success = node.translate_for_target_real(
                    args.object_name, args.base_name,
                    final_base_pos=args.final_base_pos,
                    final_base_orientation=args.final_base_orientation,
                    use_default_base=args.use_default_base_position,
                    grasp_id=args.grasp_id,
                    object_orientation=args.current_object_orientation,
                )
                if success:
                    insert_ok, insert_out = run_perform_insert_real(args)
                    if not insert_ok:
                        success = False
                        insert_json = extract_json_from_output(insert_out)
                        node.error_message = (insert_json or {}).get('error', 'insert failed')

        if success:
            node.get_logger().info("Operation completed successfully!")
        else:
            error = node.error_message

    except KeyboardInterrupt:
        error = "Interrupted by user"
    except Exception as e:
        error = str(e)
    finally:
        # Determine movement type for JSON output
        if args.insert:
            movement_type = "insert"
        elif args.place_down:
            movement_type = "place_down"
        else:
            movement_type = "unknown"

        result = {
            "result": "success" if success else "failure",
            "mode": args.mode,
            "movement_type": movement_type,
        }
        if args.object_name:
            result["object_name"] = args.object_name
        if args.base_name:
            result["base_name"] = args.base_name
        if not success:
            result["error"] = error or (node.error_message if node else "Unknown error")

        output_result(result)

        try:
            if node:
                node.action_client.destroy()
                node.destroy_node()
        except:
            pass
        try:
            rclpy.shutdown()
        except:
            pass

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
