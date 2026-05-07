#!/usr/bin/env python3
"""
Compliant Insert Episode Wrapper — Phase 2 deliverable.

Owns the full lifecycle:  PRE -> HOVER -> ZERO -> ACTIVE -> DONE/ABORT

Writes telemetry per the locked v1 schema (see ../docs/SCHEMA.md):
  - logs/insert_<object>_<YYYYMMDD_HHMMSS>.csv      (per-sample)
  - logs/insert_<object>_<YYYYMMDD_HHMMSS>.meta.json (per-episode)

Standalone entry point (dev path):
  python3 -m compliant_insertion_studio.wrapper.compliant_insert \
      --object-name u_brown --base-name fmb1_base --grasp-id 0 \
      --current-object-orientation 0 0 0 1 \
      --use-default-base-position --fz 3.0

Dispatcher entry point (production, after Phase 6):
  Dispatcher already drove HOVER; pass --skip-hover.

Signal interface (WRAP-10):
  SIGUSR1 -> increment event_marker counter (operator's "I'm pushing"/"I let go")
  SIGUSR2 -> mid-episode re-zero F/T (logged as zero_event row + meta entry)
  SIGTERM -> end as success (operator-terminated)
  SIGABRT -> end as abort

Force-mode constraint: commanded |Fz| stays <= 5 N by default per CONVENTIONS.

Reference: existing primitives/_real_mode_stash/{prismatic_peg_insertion,perform_insert}.py
informed the force_mode RPC patterns and zero/settle ordering.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from lifecycle_msgs.msg import TransitionEvent
from ur_msgs.srv import SetForceMode
from scipy.spatial.transform import Rotation as _SciRot
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from . import schema_v1 as s
from .telemetry import (
    CSVWriter, MetaJSONBuilder, compute_per_axis_errors,
    iso_local_now, filename_timestamp,
)


# ---------------------------------------------------------------------------
# Module paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent                     # compliant_insertion_studio/
_REPO_ROOT = _PKG_ROOT.parent                # ros-mcp-server/
LOG_DIR = _PKG_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Defaults / limits (from CONVENTIONS)
# ---------------------------------------------------------------------------

MAX_DEFAULT_FZ_N = 5.0           # CONVENTIONS: force-mode wrench <= 5 N default
DEFAULT_FZ_N = 3.0
DEFAULT_GAIN_SCALING = 0.5       # research-validated low-force default
DEFAULT_DAMPING_FACTOR = 0.7     # ur_robot_driver 2.13.0 defaults
DEFAULT_LIN_SPEED_M_S = 0.02
DEFAULT_ANG_SPEED_R_S = 0.20
DEFAULT_RATE_HZ = 100.0          # TELE-04: subsample 500 Hz wrench every 5th
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_AUTO_STEP_BACK_S = 5.0
DEFAULT_BIAS_WARN_N = 2.0
DEFAULT_DRIFT_WINDOW_S = 0.0     # 2026-05-07: was 1.0s. The drift check just
                                 # samples bias for meta-JSON reporting; it
                                 # doesn't gate operation. Skipping the wait
                                 # saves ~1s per insert. If we later need to
                                 # diagnose F/T drift, restore to 1.0.

# Controller names
POS_CTRL = "scaled_joint_trajectory_controller"
FORCE_CTRL = "force_mode_controller"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compliant insert episode wrapper (Phase 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Identity
    p.add_argument("--object-name", required=True, help="object id (used in CSV/meta filenames)")
    p.add_argument("--base-name", default=None, help="assembly base id (required unless --skip-hover)")
    p.add_argument("--grasp-id", type=int, default=None, help="grasp index (required unless --skip-hover)")
    p.add_argument("--current-object-orientation", nargs=4, type=float, default=None,
                   help="quat x y z w of held object (required unless --skip-hover)")
    p.add_argument("--final-base-pos", nargs=3, type=float, default=None,
                   help="xyz of base in world frame (or --use-default-base-position)")
    p.add_argument("--final-base-orientation", nargs=4, type=float, default=None,
                   help="quat xyzw of base in world frame")
    p.add_argument("--use-default-base-position", action="store_true",
                   help="use the host's DEFAULT_BASE_POSITION/ORIENTATION constants")
    # Phase skips
    p.add_argument("--skip-hover", action="store_true",
                   help="dispatcher path: HOVER already done by translate_object before this call")
    p.add_argument("--skip-smoke", action="store_true",
                   help="skip the PRE-phase F/T smoke test (use only for repeated rapid attempts in a session)")
    # STEP-BACK gate
    p.add_argument("--step-back", choices=["prompt", "signal", "auto"], default="prompt",
                   help="how operator confirms hands-off before ZERO: prompt=stdin Y/N (default), "
                        "signal=wait for SIGUSR1, auto=after --auto-step-back-seconds")
    p.add_argument("--auto-step-back-seconds", type=float, default=DEFAULT_AUTO_STEP_BACK_S,
                   help="seconds to wait when --step-back=auto")
    # Force-mode params
    p.add_argument("--fz", type=float, default=DEFAULT_FZ_N,
                   help=f"downward force in N (gentle default {DEFAULT_FZ_N}; hard cap {MAX_DEFAULT_FZ_N} unless --override-fz-cap)")
    p.add_argument("--override-fz-cap", action="store_true",
                   help="acknowledge >|5 N| force-mode wrench (CONVENTIONS)")
    p.add_argument("--gain", type=float, default=DEFAULT_GAIN_SCALING)
    p.add_argument("--damping", type=float, default=DEFAULT_DAMPING_FACTOR)
    p.add_argument("--lin-speed", type=float, default=DEFAULT_LIN_SPEED_M_S)
    p.add_argument("--ang-speed", type=float, default=DEFAULT_ANG_SPEED_R_S)
    p.add_argument("--selection",
                   default="1,1,1,1,1,1",
                   help="6-DOF compliance selection vector as comma-separated 1/0 (x,y,z,rx,ry,rz)")
    # Loop / output
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="ACTIVE-phase max duration before clean exit as 'timeout'")
    p.add_argument("--bias-warn-n", type=float, default=DEFAULT_BIAS_WARN_N,
                   help="warn (not abort) if any axis post-zero bias exceeds this")
    # Phase 5: per-shape termination config
    p.add_argument("--config", default=None,
                   help="path to a per-shape YAML in compliant_insertion_studio/configs/. "
                        "If unset, auto-resolves to configs/<object>.yaml; if that file "
                        "is also missing, falls back to legacy --timeout-only ACTIVE exit.")
    # Phase 5 v1: CAD-derived target via base world pose (camera, captured by
    # iterate_insert.py before grasp). Wrapper composes with assembly_index +
    # grasp_points to predict TCP at seat. Universal across all FMB1 parts.
    p.add_argument("--base-world-pose", nargs=7, type=float, default=None,
                   metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                   help="Base's pose in world (base_link) frame as captured from "
                        "/objects_poses_real BEFORE the gripper occluded the camera. "
                        "Enables CAD-derived predicted_tcp_at_seat in the termination predicate.")
    p.add_argument("--hole-xy-prior", nargs=2, type=float, default=None,
                   metavar=("X", "Y"),
                   help="Observed hole xy from a prior attempt (when the spiral "
                        "detected a z-descent spike). Replaces CAD-derived target_xy "
                        "for Mode B's TOWARD-target direction. Use this when the "
                        "physical slot is offset from CAD prediction (per-grasp variance).")
    # First-contact marker validation
    p.add_argument("--abort-on-first-contact", action="store_true",
                   help="Stop the wrapper IMMEDIATELY after APPROACH→FIND_HOLE transition "
                        "fires, before any FIND_HOLE wrench applies. Cleanup runs normally "
                        "(stop force mode + safe-height retract). Used to validate the "
                        "contact-detection marker (regime-decoding analysis uses this exact "
                        "instant as t=0).")
    # GUIDED mode (operator-drag data collection)
    p.add_argument("--guided-mode", action="store_true",
                   help="Route APPROACH-contact to GUIDED state instead of FIND_HOLE. In "
                        "GUIDED, force_mode is configured for operator-drag manipulation: XY "
                        "compliant, Z LOCKED, rotation LOCKED, zero commanded wrench, loose "
                        "gain+damping. Operator drags peg laterally on rim to reach the slot, "
                        "then sends SIGUSR1 to mark the hole and transition to INSERT_DESCENT "
                        "(pure Z descent at the operator-marked xy). Used for GOLD data "
                        "collection where each demo captures contact_xy + hole_xy + seat_xy "
                        "with full F/T trajectory across all phases.")
    # v4 Found Hole predicate (analysis/CONTROL_LAW.md)
    p.add_argument("--v4-autofire", action="store_true",
                   help="If set (stage 3b), v4 Found Hole predicate firing in GUIDED state "
                        "ALSO triggers GUIDED → INSERT_DESCENT (replacing operator's manual "
                        "SIGUSR1). If not set (stage 3a / default), v4 fire is logged to meta "
                        "as hole_observed_v4_predicate but only operator's SIGUSR1 triggers "
                        "the transition. Requires --guided-mode.")
    # Phase G: autonomous SEARCH director (analysis/SEARCH_CONTROL_LAW.md)
    p.add_argument("--autonomous-search", action="store_true",
                   help="Route APPROACH-contact to SEARCH state (autonomous F/T-driven "
                        "director that replaces operator-drag in GUIDED). Commands lateral "
                        "wrench in -r_cop direction at K=3N until v4 predicate fires the "
                        "rim-cross transition to INSERT_DESCENT. No operator hands required. "
                        "Mutually exclusive with --guided-mode.")
    p.add_argument("--search-K-N", type=float, default=3.0,
                   help="(LEGACY: ignored by spiral director).")
    p.add_argument("--search-F-press-N", type=float, default=9.0,
                   help="SEARCH director downward press force. Default 9.0N. Lower (7N) "
                        "reduces friction if stall observed.")
    p.add_argument("--search-max-duration-s", type=float, default=15.0,
                   help="SEARCH timeout. Default 15s.")
    p.add_argument("--search-Fmax-N", type=float, default=3.0,
                   help="Saturated lateral force magnitude in spiral PD director. Default 3.0N.")
    p.add_argument("--search-v-s-mm-s", type=float, default=5.0,
                   help="Tangential path speed in spiral. Default 5 mm/s.")
    p.add_argument("--search-pitch-mm", type=float, default=2.0,
                   help="Spiral pitch (radial growth per turn). Default 2 mm.")
    p.add_argument("--search-R-max-mm", type=float, default=8.0,
                   help="Spiral max radius (abort threshold). Default 8 mm.")
    # Calibration provenance (passed through to meta JSON)
    p.add_argument("--cal-yaml", default=None,
                   help="path to foundational calibration YAML (recorded in meta JSON)")
    # Smoke test integration
    p.add_argument("--smoke-script", default=str(_PKG_ROOT / "shared" / "ft_smoke_test.py"),
                   help="path to ft_smoke_test.py (default: ../shared/ft_smoke_test.py)")
    # Misc
    p.add_argument("--no-prompt-notes", action="store_true",
                   help="do not prompt for user_notes at end (useful for unattended dispatcher runs)")
    p.add_argument("--skip-home-on-done", action="store_true",
                   help="skip the final move_home in cleanup (saves ~5s during tuning; "
                        "still does stop force mode + switch + safe_height)")
    p.add_argument("--no-post-insert-move", action="store_true",
                   help="On successful insert, leave EE AT the inserted seat pose. "
                        "Skips BOTH move_to_safe_height and move_home (still stops "
                        "force_mode + switches controllers). Caller is responsible "
                        "for releasing the gripper and retracting. Use when the "
                        "wrapper is invoked from a higher-level orchestrator that "
                        "sequences the post-insert release/retract itself "
                        "(e.g. translate_object --insertion-type compliant).")
    p.add_argument("--wrapper-version", default=None,
                   help="version tag for meta JSON (default: 'compliant_insert.py@<git-sha>')")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, cwd=str(_REPO_ROOT),
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _switch_controllers(activate, deactivate, logger=None) -> bool:
    """Shell out to `ros2 control switch_controllers` (matches existing primitive style)."""
    cmd = ["ros2", "control", "switch_controllers"]
    if deactivate:
        cmd += ["--deactivate"] + deactivate
    if activate:
        cmd += ["--activate"] + activate
    msg = f"switch_controllers: -{deactivate} +{activate}"
    if logger:
        logger.info(msg)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0 and logger:
        logger.error(f"switch_controllers failed: {result.stderr.strip()}")
    return result.returncode == 0


_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')


def _list_active_controllers() -> set:
    """Poll `ros2 control list_controllers` and return set of active controller names.

    WRAP-07: switch_controller RPC success != actual controller transition complete.
    Strips ANSI color escape codes — `ros2 control list_controllers` emits color
    even when stdout is a pipe, which would otherwise leave `parts[-1]` as
    `'\\x1b[0m'` and silently break the active-set detection.
    """
    try:
        result = subprocess.run(
            ["ros2", "control", "list_controllers"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    active = set()
    for raw_line in result.stdout.splitlines():
        line = _ANSI_ESCAPE_RE.sub('', raw_line).strip()
        # Format after stripping: "<name>  <type>  <state>" — state is
        # "active" or "inactive". Trailing whitespace is preserved by the
        # CLI so we strip per-token below.
        parts = [p for p in line.split() if p]
        if len(parts) >= 3 and parts[-1] == "active":
            active.add(parts[0])
    return active


def _await_controller_active(name: str, *, timeout_s: float = 5.0,
                             logger=None) -> bool:
    """WRAP-07: poll list_controllers until `name` shows as active or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if name in _list_active_controllers():
            return True
        time.sleep(0.1)
    if logger:
        logger.error(f"Controller {name!r} did not become active within {timeout_s}s")
    return False


def _parse_selection(text: str) -> list[bool]:
    raw = [v.strip() for v in text.split(",") if v.strip()]
    if len(raw) != 6:
        raise ValueError(f"--selection expects 6 comma-separated 1/0 values, got {len(raw)}: {raw!r}")
    out = []
    for v in raw:
        if v not in ("0", "1"):
            raise ValueError(f"--selection values must be 0 or 1, got {v!r}")
        out.append(v == "1")
    return out


# ---------------------------------------------------------------------------
# Episode FSM
# ---------------------------------------------------------------------------

class CompliantInsertEpisode(Node):
    def __init__(self, args):
        super().__init__("compliant_insert_episode")
        self.args = args

        # Phase / FSM state
        self.phase: str = s.PHASE_PRE
        self.start_t: float | None = None
        self.episode_start_iso: str | None = None
        self.episode_end_iso: str | None = None
        self.outcome_signal: str | None = None   # set by signal handler

        # Telemetry state
        self.tcp: PoseStamped | None = None
        self.wrench: WrenchStamped | None = None
        self.gripper_width_v: float = float("nan")
        self.event_marker_counter: int = 0
        self.hands_off: int = 0
        self.zero_event_pending: int = 0
        self._zero_request: bool = False

        # Episode artifacts
        self.csv_writer: CSVWriter | None = None
        self.csv_path: str | None = None
        self.meta_path: str | None = None
        self.meta = MetaJSONBuilder()

        # Computed once per episode
        self.target_xyz: tuple[float, float, float] | None = None
        self.target_quat: tuple[float, float, float, float] | None = None
        self.commanded_fz: float = 0.0
        # Schema bump v1.2 (G002): track full 6-axis commanded wrench in base_link frame.
        self.commanded_wrench_baselink: tuple[float, float, float, float, float, float] = (0.0,) * 6
        # Schema bump v1.2 (G001/G003/G004): per-topic raw sidecar file handles.
        self._joints_raw_fh = None
        self._wrench_raw_fh = None
        self._cmd_wrench_raw_fh = None
        self._fm_events_fh = None
        self.in_force_mode: bool = False

        # v1.1: object-pose tracking — set after HOVER lands. The wrapper carries
        # the operator-provided `current_object_orientation` as ground truth at
        # HOVER end; per-row obj_q* in CSV is `R_tcp_now × R_tcp_to_object`.
        self.tcp_to_object_quat: tuple[float, float, float, float] | None = None

        # ROS QoS — match force_torque_sensor_broadcaster (RELIABLE, KEEP_LAST(1), VOLATILE)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose", self._tcp_cb, sensor_qos)
        self.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._wrench_cb, sensor_qos)
        self.create_subscription(Float32, "/gripper_width", self._gripper_cb, 10)

        # Schema bump v1.2 — raw-stream subscriptions feeding sidecar CSVs.
        # G001: joint_states (native ~125 Hz on UR Humble) → joints_raw.csv
        self.create_subscription(JointState, "/joint_states", self._joints_raw_cb, sensor_qos)
        # G003: wrench at native rate (500 Hz) — separate callback so existing 100 Hz CSV path is untouched
        self.create_subscription(WrenchStamped, "/force_torque_sensor_broadcaster/wrench", self._wrench_raw_cb, sensor_qos)
        # G004: force_mode_controller transition events (rare) → fm_events.csv
        self.create_subscription(TransitionEvent, "/force_mode_controller/transition_event", self._fm_event_cb, 10)

        self.start_fm = self.create_client(SetForceMode, "/force_mode_controller/start_force_mode")
        self.stop_fm = self.create_client(Trigger, "/force_mode_controller/stop_force_mode")
        # 2026-05-07: native client for zero_ftsensor. Replaces a
        # subprocess.run("ros2 service call ...") that paid ~1.5-2s of CLI
        # Python-startup overhead every insert.
        self.zero_ft_cli = self.create_client(Trigger, "/io_and_status_controller/zero_ftsensor")

        # TF for wrench frame transform — broadcaster publishes in tool0_controller
        # (post-driver-fix #1652), but SCHEMA.md commits to base_link for all logged
        # vectors. Transform per sample.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._wrench_tf_miss_count = 0   # diagnostic — non-fatal if unavailable

    # ------------------- callbacks ----------------------------------------
    def _tcp_cb(self, msg): self.tcp = msg
    def _wrench_cb(self, msg): self.wrench = msg
    def _gripper_cb(self, msg):
        try:
            self.gripper_width_v = float(msg.data)
        except Exception:
            pass

    # ----- Schema v1.2 sidecar callbacks ----------------------------------
    def _joints_raw_cb(self, msg: JointState):
        """G001: joint_states at native rate → joints_raw.csv.
        UR publishes 6 joints; pad with NaN if fewer/empty for any reason."""
        if self._joints_raw_fh is None:
            return
        try:
            n = 6
            pos = list(msg.position) + [float("nan")] * max(0, n - len(msg.position))
            vel = list(msg.velocity) + [float("nan")] * max(0, n - len(msg.velocity))
            eff = list(msg.effort)   + [float("nan")] * max(0, n - len(msg.effort))
            row = [str(msg.header.stamp.sec), str(msg.header.stamp.nanosec)]
            row += [f"{v:.6f}" for v in pos[:n]]
            row += [f"{v:.6f}" for v in vel[:n]]
            row += [f"{v:.4f}" for v in eff[:n]]
            self._joints_raw_fh.write(",".join(row) + "\n")
        except Exception:
            pass

    def _wrench_raw_cb(self, msg: WrenchStamped):
        """G003: wrench at native 500 Hz → wrench_raw.csv (unfiltered)."""
        if self._wrench_raw_fh is None:
            return
        try:
            w = msg.wrench
            self._wrench_raw_fh.write(
                f"{msg.header.stamp.sec},{msg.header.stamp.nanosec},"
                f"{w.force.x:.4f},{w.force.y:.4f},{w.force.z:.4f},"
                f"{w.torque.x:.4f},{w.torque.y:.4f},{w.torque.z:.4f},"
                f"{msg.header.frame_id}\n"
            )
        except Exception:
            pass

    def _fm_event_cb(self, msg: TransitionEvent):
        """G004: force_mode_controller lifecycle transitions → fm_events.csv."""
        if self._fm_events_fh is None:
            return
        try:
            wall = time.time()
            self._fm_events_fh.write(
                f"{wall:.4f},{msg.start_state.label},{msg.goal_state.label},"
                f"{msg.transition.id},{msg.transition.label}\n"
            )
        except Exception:
            pass

    def _log_cmd_wrench_event(self, intent_baselink, gain_eff, damping_eff, sel_vec, source: str):
        """G002: log every SetForceMode command (event-based, not periodic).
        intent_baselink is the 6-tuple commanded wrench in base_link frame."""
        if self._cmd_wrench_raw_fh is None:
            return
        try:
            t_rel = (time.time() - self.start_t) if getattr(self, "start_t", None) is not None else float("nan")
            ib = list(intent_baselink) + [0.0] * max(0, 6 - len(intent_baselink))
            sv = list(sel_vec) + [True] * max(0, 6 - len(sel_vec))
            self._cmd_wrench_raw_fh.write(
                f"{t_rel:.4f},{ib[0]:.4f},{ib[1]:.4f},{ib[2]:.4f},"
                f"{ib[3]:.4f},{ib[4]:.4f},{ib[5]:.4f},"
                f"{int(sv[0])},{int(sv[1])},{int(sv[2])},{int(sv[3])},{int(sv[4])},{int(sv[5])},"
                f"{gain_eff:.3f},{damping_eff:.3f},{source}\n"
            )
        except Exception:
            pass

    # ----- Schema v1.2 sidecar lifecycle ----------------------------------
    def _open_raw_sidecars(self, csv_path: str):
        """Open the 4 sidecar CSVs alongside the main CSV. Headers per file."""
        base = csv_path[:-4] if csv_path.endswith(".csv") else csv_path
        self._joints_raw_fh = open(f"{base}.joints_raw.csv", "w", buffering=1)
        self._joints_raw_fh.write(
            "stamp_sec,stamp_nsec,j0_pos,j1_pos,j2_pos,j3_pos,j4_pos,j5_pos,"
            "j0_vel,j1_vel,j2_vel,j3_vel,j4_vel,j5_vel,"
            "j0_eff,j1_eff,j2_eff,j3_eff,j4_eff,j5_eff\n"
        )
        self._wrench_raw_fh = open(f"{base}.wrench_raw.csv", "w", buffering=1)
        self._wrench_raw_fh.write("stamp_sec,stamp_nsec,fx,fy,fz,tx,ty,tz,frame_id\n")
        self._cmd_wrench_raw_fh = open(f"{base}.cmd_wrench_raw.csv", "w", buffering=1)
        self._cmd_wrench_raw_fh.write(
            "t_s,cmd_fx,cmd_fy,cmd_fz,cmd_tx,cmd_ty,cmd_tz,"
            "sel_x,sel_y,sel_z,sel_rx,sel_ry,sel_rz,gain,damping,source\n"
        )
        self._fm_events_fh = open(f"{base}.fm_events.csv", "w", buffering=1)
        self._fm_events_fh.write("wall_t_s,start_state,goal_state,transition_id,transition_label\n")

    def _close_raw_sidecars(self):
        for attr in ("_joints_raw_fh", "_wrench_raw_fh", "_cmd_wrench_raw_fh", "_fm_events_fh"):
            fh = getattr(self, attr, None)
            if fh is not None:
                try: fh.close()
                except Exception: pass
                setattr(self, attr, None)

    # ------------------- topic-ready gate ---------------------------------
    def wait_for_topics(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tcp is not None and self.wrench is not None:
                return True
        return False

    # ------------------- wrench frame transform ---------------------------
    def _wrench_in_base(self, wrench: WrenchStamped) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Transform wrench from sensor frame (tool0_controller) to base_link.

        Returns (force_xyz_base, torque_xyz_base). Falls back to raw values + warns
        if TF lookup fails — so a missing TF doesn't crash an episode mid-collection.
        """
        f_raw = (wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z)
        t_raw = (wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z)
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link",
                wrench.header.frame_id or "tool0_controller",
                rclpy.time.Time(),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            self._wrench_tf_miss_count += 1
            if self._wrench_tf_miss_count == 1 or self._wrench_tf_miss_count % 100 == 0:
                self.get_logger().warn(
                    f"Wrench TF lookup failed (count={self._wrench_tf_miss_count}); "
                    f"logging RAW sensor-frame values for affected samples"
                )
            return f_raw, t_raw
        q = tf.transform.rotation
        R = _SciRot.from_quat([q.x, q.y, q.z, q.w])
        f_base = R.apply(np.asarray(f_raw))
        t_base = R.apply(np.asarray(t_raw))
        return tuple(float(v) for v in f_base), tuple(float(v) for v in t_base)

    # ------------------- sample logging -----------------------------------
    def _log_sample(self) -> None:
        """Build one row from current state and write it.

        Called at the wrapper's tick rate (default 100 Hz) once CSV is open.
        """
        if self.csv_writer is None or self.tcp is None or self.wrench is None:
            return
        if self.target_xyz is None or self.target_quat is None:
            # No target yet (e.g., we're in PRE before HOVER computed). Use NaN target — so
            # error columns become NaN. Acceptable; pre-target rows are rare.
            tx_, ty_, tz_ = float("nan"), float("nan"), float("nan")
            tqx_, tqy_, tqz_, tqw_ = float("nan"), float("nan"), float("nan"), float("nan")
            dx_, dy_, dz_, droll_, dpitch_, dyaw_ = (float("nan"),) * 6
        else:
            tx_, ty_, tz_ = self.target_xyz
            tqx_, tqy_, tqz_, tqw_ = self.target_quat
            tcp_xyz = (self.tcp.pose.position.x, self.tcp.pose.position.y, self.tcp.pose.position.z)
            tcp_quat = (self.tcp.pose.orientation.x, self.tcp.pose.orientation.y,
                        self.tcp.pose.orientation.z, self.tcp.pose.orientation.w)
            dx_, dy_, dz_, droll_, dpitch_, dyaw_ = compute_per_axis_errors(
                tcp_xyz, tcp_quat, self.target_xyz, self.target_quat
            )

        t = (time.time() - self.start_t) if self.start_t is not None else 0.0
        # Transform wrench tool0_controller -> base_link so logged force/torque
        # share the frame with target pose, TCP pose, and per-axis errors.
        (f_x, f_y, f_z), (t_x, t_y, t_z) = self._wrench_in_base(self.wrench)
        p = self.tcp.pose.position
        q = self.tcp.pose.orientation

        # v1.1: object-pose estimate. Position copies TCP (Phase 5 can apply
        # gripper-to-object-center offset from per-object models if needed).
        # Orientation = R_tcp_now × R_tcp_to_object (constant transform set
        # at HOVER end from operator-provided current_object_orientation).
        if self.tcp_to_object_quat is not None:
            R_tcp = _SciRot.from_quat([q.x, q.y, q.z, q.w])
            R_t2o = _SciRot.from_quat(list(self.tcp_to_object_quat))
            R_obj = R_tcp * R_t2o
            obj_qx, obj_qy, obj_qz, obj_qw = R_obj.as_quat().tolist()
        else:
            obj_qx = obj_qy = obj_qz = obj_qw = float("nan")

        row = {
            "t_s": t, "phase": self.phase,
            "event_marker": self.event_marker_counter,
            "hands_off": self.hands_off,
            "zero_event": self.zero_event_pending,
            "tcp_x": p.x, "tcp_y": p.y, "tcp_z": p.z,
            "tcp_qx": q.x, "tcp_qy": q.y, "tcp_qz": q.z, "tcp_qw": q.w,
            "target_x": tx_, "target_y": ty_, "target_z": tz_,
            "target_qx": tqx_, "target_qy": tqy_, "target_qz": tqz_, "target_qw": tqw_,
            "dx": dx_, "dy": dy_, "dz": dz_,
            "droll": droll_, "dpitch": dpitch_, "dyaw": dyaw_,
            "fx": f_x, "fy": f_y, "fz": f_z,
            "tx": t_x, "ty": t_y, "tz": t_z,
            "gripper_width": self.gripper_width_v,
            "commanded_fz": self.commanded_fz,
            # --- v1.1 columns ---
            "wrench_frame_id": (self.wrench.header.frame_id or ""),
            "obj_x": p.x, "obj_y": p.y, "obj_z": p.z,
            "obj_qx": obj_qx, "obj_qy": obj_qy, "obj_qz": obj_qz, "obj_qw": obj_qw,
        }
        self.csv_writer.write(row)
        self.zero_event_pending = 0   # one-shot flag

    # ------------------- zero F/T (native rclpy service client) -----------
    def _zero_ftsensor_call(self) -> bool:
        """2026-05-07: was subprocess.run('ros2 service call ...') which paid
        ~1.5-2s for the ros2 CLI Python startup every insert. Native client
        completes in ~50-200ms."""
        if not self.zero_ft_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("zero_ftsensor service unavailable")
            return False
        future = self.zero_ft_cli.call_async(Trigger.Request())
        deadline = time.time() + 5.0
        while time.time() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            self.get_logger().error("zero_ftsensor call timed out (5s)")
            return False
        result = future.result()
        ok = bool(getattr(result, "success", False))
        if not ok:
            self.get_logger().error(
                f"zero_ftsensor failed: {getattr(result, 'message', 'no response')}"
            )
        return ok

    def _sample_bias(self, settle_s: float = 0.5) -> dict | None:
        """Drop cached wrench, settle, return the next fresh sample as a bias dict.

        Returns wrench in base_link frame (post-transform) so post_zero_bias values
        in meta JSON share the frame convention with logged CSV columns.
        """
        self.wrench = None
        deadline = time.time() + settle_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.wrench is None:
            self.get_logger().warn("No /wrench sample arrived during settle window")
            return None
        (f_x, f_y, f_z), (t_x, t_y, t_z) = self._wrench_in_base(self.wrench)
        return {"Fx": f_x, "Fy": f_y, "Fz": f_z, "Tx": t_x, "Ty": t_y, "Tz": t_z}

    # ------------------- start/stop force mode ----------------------------
    def _start_force_mode(self, sel_vec: list[bool],
                           override_wrench_baselink: tuple | None = None,
                           gain_override: float | None = None,
                           damping_override: float | None = None,
                           lin_speed_override: float | None = None,
                           ang_speed_override: float | None = None,
                           quiet: bool = False) -> bool:
        """Call SetForceMode. Default wrench = (0, 0, -fz, 0, 0, 0) in base_link
        (operator's intent: push down). When `override_wrench_baselink` is
        provided as (Fx, Fy, Fz, Tx, Ty, Tz), use those values directly —
        used by Phase 5 Mode B to apply correction wrench deltas during ACTIVE.

        Phase 5 Mode B v2 (spiral search) uses gain_override/damping_override
        to drop force-mode gain/damping during search bursts (per GPT/Chhatpar/
        FANUC research: search wants gain ~0.5 + damping ~0.2 vs nominal 1.0/0.7).
        Set quiet=True to skip the per-call log (used inside spiral re-call loop
        to avoid log spam at 20 Hz).
        """
        if not self.start_fm.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("start_force_mode service unavailable")
            return False

        fz = float(self.args.fz)
        if not self.args.override_fz_cap and abs(fz) > MAX_DEFAULT_FZ_N:
            self.get_logger().error(
                f"|fz|={abs(fz)} exceeds {MAX_DEFAULT_FZ_N} N cap (CONVENTIONS); "
                f"pass --override-fz-cap to acknowledge."
            )
            return False

        req = SetForceMode.Request()
        # Send task_frame in `base` directly so there's no auto-transform surprise.
        # The force_mode_controller.cpp transforms task_frame to `<tf_prefix>base`
        # internally before URScript sees it; if we sent frame_id="base_link" it would
        # be silently rotated 180° about Z (base ↔ base_link), and our wrench would
        # be interpreted in that rotated frame — flipping X and Y signs.
        # We instead apply the base_link → base sign flip explicitly when constructing
        # the wrench below, so operator + planning code can keep thinking in base_link.
        # Verified empirically 2026-05-03 by verify_baselink_motion.py.
        req.task_frame.header.frame_id = "base"
        req.task_frame.pose.orientation.w = 1.0   # identity in base

        req.selection_vector_x = sel_vec[0]
        req.selection_vector_y = sel_vec[1]
        req.selection_vector_z = sel_vec[2]
        req.selection_vector_rx = sel_vec[3]
        req.selection_vector_ry = sel_vec[4]
        req.selection_vector_rz = sel_vec[5]

        # Operator's intent (in base_link): default = push down with Fz N along
        # base_link -Z axis, no lateral / torque. Phase 5 Mode B can override via
        # `override_wrench_baselink` to apply correction deltas (lateral push +
        # counter-torque) for stuck-state escape.
        # Convert base_link wrench → base wrench: X and Y signs flipped, Z preserved
        # (verified empirically 2026-05-03 by verify_baselink_motion.py).
        if override_wrench_baselink is not None:
            intent_baselink = tuple(float(v) for v in override_wrench_baselink)
        else:
            # Default: push down only
            intent_baselink = (0.0, 0.0, -fz, 0.0, 0.0, 0.0)
        wrench_in_base = (
            -intent_baselink[0], -intent_baselink[1], intent_baselink[2],
            -intent_baselink[3], -intent_baselink[4], intent_baselink[5],
        )

        req.wrench.force.x  = wrench_in_base[0]
        req.wrench.force.y  = wrench_in_base[1]
        req.wrench.force.z  = wrench_in_base[2]
        req.wrench.torque.x = wrench_in_base[3]
        req.wrench.torque.y = wrench_in_base[4]
        req.wrench.torque.z = wrench_in_base[5]

        req.type = SetForceMode.Request.NO_TRANSFORM   # = 2

        lin_eff = float(lin_speed_override) if lin_speed_override is not None else float(self.args.lin_speed)
        ang_eff = float(ang_speed_override) if ang_speed_override is not None else float(self.args.ang_speed)
        req.speed_limits.linear.x = lin_eff
        req.speed_limits.linear.y = lin_eff
        req.speed_limits.linear.z = lin_eff
        req.speed_limits.angular.x = ang_eff
        req.speed_limits.angular.y = ang_eff
        req.speed_limits.angular.z = ang_eff

        gain_eff    = float(gain_override)    if gain_override    is not None else float(self.args.gain)
        damping_eff = float(damping_override) if damping_override is not None else float(self.args.damping)
        req.gain_scaling = gain_eff
        req.damping_factor = damping_eff

        # Up to 3 retry attempts with backoff. The force_mode_controller's
        # start_force_mode RPC can return success=False for ~0.5-1.5s after
        # the controller becomes "active" (controller-manager reports it
        # active, but the controller's internal state isn't ready to accept
        # commands yet — verified empirically 2026-05-06 session 5). Retrying
        # with backoff handles this transient.
        # 2026-05-07: bumped 3 → 6 attempts and 1.0s → 0.8s backoff. The
        # transient sometimes lasts longer than 3s (especially after a prior
        # force_mode session), and 6 × 0.8s = ~5s budget covers it.
        result = None
        last_err = None
        N_ATTEMPTS = 6
        for attempt in range(N_ATTEMPTS):
            future = self.start_fm.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            result = future.result()
            if result is not None and getattr(result, 'success', False):
                if attempt > 0 and not quiet:
                    self.get_logger().info(f"start_force_mode succeeded on attempt {attempt+1}")
                break
            last_err = (
                "service call timed out" if result is None else
                f"service returned success=False (attempt {attempt+1}/{N_ATTEMPTS})"
            )
            if attempt < N_ATTEMPTS - 1:
                if not quiet:
                    self.get_logger().warn(
                        f"start_force_mode {last_err} — retrying in 0.8s"
                    )
                time.sleep(0.8)
        if result is None or not getattr(result, 'success', False):
            self.get_logger().error(f"start_force_mode call failed after {N_ATTEMPTS} attempts: {last_err}")
            return False
        self.in_force_mode = True
        self.commanded_fz = intent_baselink[2]
        # G002: track the full 6-axis intent for sidecar logging.
        self.commanded_wrench_baselink = tuple(float(v) for v in intent_baselink)
        self._log_cmd_wrench_event(intent_baselink, gain_eff, damping_eff, sel_vec, "start_force_mode")
        if not quiet:
            self.get_logger().info(
                f"Force mode active: wrench={intent_baselink}, sel={sel_vec}, "
                f"gain={gain_eff} damp={damping_eff}"
            )
        return True

    def _stop_force_mode(self) -> None:
        if not self.in_force_mode:
            return
        # Send a zero-wrench first to settle the controller before stop. After
        # long force-mode runs (170s+) the controller can be busy/unresponsive
        # to the immediate stop call, leaving URCap in a stopped state. The
        # zero-wrench gives controller a brief idle frame to acknowledge.
        try:
            self._start_force_mode([False, False, False, False, False, False],
                                   override_wrench_baselink=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            time.sleep(0.5)
        except Exception as e:
            self.get_logger().debug(f"zero-wrench prep failed (non-fatal): {e}")

        # Up to 3 retries with backoff
        stopped_cleanly = False
        for attempt in range(3):
            if self.stop_fm.wait_for_service(timeout_sec=2.0):
                future = self.stop_fm.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
                r = future.result()
                if r is not None and r.success:
                    self.get_logger().info(f"Force mode stopped (attempt {attempt+1})")
                    stopped_cleanly = True
                    break
                self.get_logger().warn(
                    f"stop_force_mode attempt {attempt+1} reported: "
                    f"{getattr(r, 'message', 'no response')}"
                )
            time.sleep(0.5)
        if not stopped_cleanly:
            self.get_logger().warn("stop_force_mode never confirmed; proceeding")
        self.in_force_mode = False
        self.commanded_fz = 0.0
        self.commanded_wrench_baselink = (0.0,) * 6
        self._log_cmd_wrench_event((0.0,) * 6, 0.0, 0.0, (False,) * 6, "stop_force_mode")

    # ------------------- step-back gate -----------------------------------
    def _await_step_back(self, mode: str, auto_seconds: float) -> tuple[str, bool]:
        """Wait for the operator to acknowledge hands-off. Returns (trigger_str, success)."""
        if mode == "prompt":
            self.get_logger().info("STEP BACK from the robot. Press Y + Enter to confirm zero, anything else to abort.")
            try:
                ans = input("STEP-BACK confirmed? [y/N]: ").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                return "operator_step_back_confirmed", True
            return "operator_aborted_at_step_back", False
        elif mode == "signal":
            self.get_logger().info(f"STEP BACK from the robot. Send SIGUSR1 to this PID ({os.getpid()}) to confirm.")
            self.event_marker_counter_at_gate = self.event_marker_counter
            deadline = time.time() + 60.0   # generous; operator may need a moment
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
                if self.event_marker_counter > self.event_marker_counter_at_gate:
                    return "operator_signal_confirmed", True
                if self.outcome_signal in ("abort", "success"):
                    return "operator_aborted_at_step_back", False
            return "step_back_signal_timeout", False
        elif mode == "auto":
            self.get_logger().warn(f"AUTO step-back: assuming hands-off after {auto_seconds:.1f}s. Operator MUST step back NOW.")
            t_end = time.time() + float(auto_seconds)
            while time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.1)
            return "auto_after_seconds", True
        else:
            return f"unknown_step_back_mode:{mode}", False

    # ------------------- HOVER subprocess driver --------------------------
    def _drive_hover(self) -> tuple[bool, dict]:
        """Spawn _run_hover.py to navigate to HOVER. Returns (success, result_dict)."""
        a = self.args
        for missing in ("base_name", "grasp_id", "current_object_orientation"):
            if getattr(a, missing) is None:
                return False, {"error": f"--{missing.replace('_', '-')} required when not --skip-hover"}

        cmd = [
            sys.executable, "-m", "compliant_insertion_studio.wrapper._run_hover",
            "--object-name", a.object_name,
            "--base-name", a.base_name,
            "--grasp-id", str(a.grasp_id),
            "--current-object-orientation", *(str(v) for v in a.current_object_orientation),
        ]
        if a.final_base_pos:
            cmd += ["--final-base-pos", *(str(v) for v in a.final_base_pos)]
        if a.final_base_orientation:
            cmd += ["--final-base-orientation", *(str(v) for v in a.final_base_orientation)]
        if a.use_default_base_position:
            cmd += ["--use-default-base-position"]

        self.get_logger().info(f"HOVER subprocess: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                  cwd=str(_REPO_ROOT))
        except subprocess.TimeoutExpired:
            return False, {"error": "HOVER subprocess timeout (120 s)"}

        # Parse __RESULT_JSON__ block
        out = proc.stdout
        if "__RESULT_JSON__" in out and "__END_RESULT_JSON__" in out:
            blk = out.split("__RESULT_JSON__", 1)[1].split("__END_RESULT_JSON__", 1)[0].strip()
            try:
                result = json.loads(blk)
            except json.JSONDecodeError:
                result = {"error": f"HOVER subprocess result JSON malformed: {blk!r}"}
        else:
            result = {"error": f"HOVER subprocess produced no result JSON. stdout tail: {out[-500:]!r}"}

        success = (proc.returncode == 0 and result.get("result") == "success")
        if not success:
            self.get_logger().error(
                f"HOVER subprocess failed: rc={proc.returncode} result={result} stderr_tail={proc.stderr[-300:]!r}"
            )
        return success, result

    # ------------------- IK joint-limit pre-check (WRAP-11) ---------------
    def _hover_pose_passes_ik_check(self, hover_pose: dict) -> tuple[bool, dict]:
        """WRAP-11: reject HOVER if IK lands at a joint-limit edge.

        Lightweight version: read /joint_states once and compute distance-to-limit
        for each joint. UR5e joint limits are ±2π for most joints; we flag if any
        joint is within 0.1 rad of a hard limit.
        """
        # Best-effort: read joint state via subprocess to avoid extra subscriptions
        try:
            r = subprocess.run(
                ["ros2", "topic", "echo", "--once", "/joint_states", "sensor_msgs/msg/JointState"],
                capture_output=True, text=True, timeout=3,
            )
        except Exception as e:
            self.get_logger().warn(f"IK pre-check could not read /joint_states: {e}")
            return True, {"passed": True, "note": "joint_states unavailable, pre-check skipped"}

        # Parse YAML output (best-effort, non-fatal)
        positions: list[float] = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("position:"):
                # next non-empty lines until 'velocity:' are floats
                pass
            elif line.startswith("- ") and len(positions) < 6:
                try:
                    positions.append(float(line[2:].strip()))
                except ValueError:
                    pass
        if len(positions) < 6:
            return True, {"passed": True, "note": "could not parse 6 joint positions, pre-check skipped"}

        # UR5e joint limits: ±2π for joints 0,3,5; more conservative for others. Use ±2π universally.
        JOINT_HARD_LIMIT = 2 * 3.14159265358979
        margin_to_limit = min(JOINT_HARD_LIMIT - abs(p) for p in positions)
        if margin_to_limit < 0.1:
            return False, {"passed": False, "min_joint_margin_rad": round(margin_to_limit, 4)}
        return True, {"passed": True, "min_joint_margin_rad": round(margin_to_limit, 4)}

    # ------------------- smoke test (PRE) ---------------------------------
    def _run_smoke_test(self) -> dict:
        """Subprocess ft_smoke_test.py and parse its result. Returns smoke-test dict for meta JSON."""
        if self.args.skip_smoke:
            return {"result": "skipped"}
        if not Path(self.args.smoke_script).exists():
            self.get_logger().warn(f"Smoke test script not found at {self.args.smoke_script}; skipping")
            return {"result": "skipped", "reason": "script_not_found"}
        try:
            proc = subprocess.run(
                [sys.executable, self.args.smoke_script],
                capture_output=True, text=True, timeout=30,
                cwd=str(_REPO_ROOT),
            )
        except Exception as e:
            return {"result": "fail", "reason": f"smoke subprocess error: {e}"}
        # ft_smoke_test.py prints __RESULT_JSON__ block (matches host primitive convention)
        out = proc.stdout
        if "__RESULT_JSON__" in out and "__END_RESULT_JSON__" in out:
            blk = out.split("__RESULT_JSON__", 1)[1].split("__END_RESULT_JSON__", 1)[0].strip()
            try:
                d = json.loads(blk)
                return d
            except json.JSONDecodeError:
                pass
        # Fallback: assume pass if rc=0
        return {"result": "pass" if proc.returncode == 0 else "fail",
                "stdout_tail": out[-200:], "stderr_tail": proc.stderr[-200:]}


# ---------------------------------------------------------------------------
# Phase runners (free functions to keep the class manageable)
# ---------------------------------------------------------------------------

def run_pre(ep: CompliantInsertEpisode) -> str:
    """PRE phase (WRAP-02). Returns next phase string."""
    ep.phase = s.PHASE_PRE
    ep.get_logger().info("=== PRE: preconditions + smoke test ===")

    if not ep.wait_for_topics(timeout=5.0):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "topics_not_live")
        return s.PHASE_ABORT

    smoke = ep._run_smoke_test()
    ep.meta.set_smoke_test(smoke)
    if smoke.get("result") == "fail":
        ep.meta.set_outcome(s.OUTCOME_ABORT, f"smoke_test_fail:{smoke.get('reason', 'unknown')}")
        return s.PHASE_ABORT

    ep._log_sample()   # one PRE row before transitioning
    return s.PHASE_HOVER if not ep.args.skip_hover else s.PHASE_ZERO


def run_hover(ep: CompliantInsertEpisode) -> str:
    """HOVER phase (WRAP-03 + WRAP-11). Returns next phase string."""
    ep.phase = s.PHASE_HOVER
    ep.get_logger().info("=== HOVER: navigate to per-object hover pose ===")

    success, result = ep._drive_hover()
    if not success:
        ep.meta.set_outcome(s.OUTCOME_ABORT, f"hover_failed:{result.get('error', 'unknown')}")
        return s.PHASE_ABORT

    # Compute target = per-object hole position (from assembly config) — but
    # the HOVER subprocess only returns post-HOVER EE pose, not the assembly
    # target. For v1 we treat the post-HOVER EE pose AS the assembly target's
    # XY (the hole is directly below); the wrapper user supplies target_z via
    # --final-base-pos[2] (base z) plus per-object hole offset (Phase 5 config).
    # For Phase 2 standalone runs, target = HOVER pose itself. Phase 5 will
    # parameterize target_z properly via per-object YAML.
    pose = result.get("ee_pose_at_hover", {})
    xyz = pose.get("xyz_m", [float("nan")] * 3)
    quat = pose.get("quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    ep.target_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ep.target_quat = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    ep.meta.set_assembly_target(list(ep.target_xyz), list(ep.target_quat))
    # v1.1: hover_pose_world is now required (was optional in v1.0).
    ep.meta.set_optional("hover_pose_world", {"xyz_m": list(ep.target_xyz), "quat_xyzw": list(ep.target_quat)})

    # v1.1: derive TCP-to-object rotation transform. Operator-provided
    # current_object_orientation is treated as ground-truth at HOVER end (the
    # camera can't see the held part reliably, so this is the chain anchor).
    # Per-row obj_q* in CSV = R_tcp_now × R_tcp_to_object.
    try:
        cobj = ep.args.current_object_orientation
        if cobj is not None and len(cobj) == 4:
            R_tcp_hover = _SciRot.from_quat(list(ep.target_quat))
            R_obj_input = _SciRot.from_quat([float(c) for c in cobj])
            R_tcp_to_obj = R_tcp_hover.inv() * R_obj_input
            ep.tcp_to_object_quat = tuple(R_tcp_to_obj.as_quat().tolist())
            ep.meta.set_optional("current_object_orientation_input", [float(c) for c in cobj])
            ep.meta.set_optional("tcp_to_object_transform", {
                "tcp_xyz_at_hover": list(ep.target_xyz),
                "tcp_quat_at_hover": list(ep.target_quat),
                "tcp_to_object_quat_xyzw": list(ep.tcp_to_object_quat),
                "convention": "R_object_now = R_tcp_now * R_tcp_to_object",
                "position_note": "obj_xyz in CSV copies TCP xyz; gripper-to-object-center offset is per-object (apply in Phase 5)",
            })
    except Exception as _e:
        ep.get_logger().warn(f"tcp_to_object_quat derivation failed: {_e}")

    # Loud warning: target = HOVER pose for v1 — not the actual hole. dx/dy/dz
    # in the CSV will look like noise around zero (descent) and any in-plane
    # drift, NOT real per-axis error vs the hole. Phase 5 wires per-object
    # hole offsets from configs/<object>.yaml — DO NOT interpret today's
    # dx/dy/dz columns as termination-criterion-quality data.
    ep.get_logger().warn(
        "TARGET-POSE LIMITATION: dx/dy/dz columns reference the HOVER pose, "
        "NOT the actual hole. Use today's CSVs for FSM verification + force/pose "
        "shape inspection only — per-axis-error analysis is invalid until Phase 5."
    )

    # WRAP-11: IK pre-check after HOVER lands
    ok, ik_info = ep._hover_pose_passes_ik_check(pose)
    ep.meta.set_optional("ik_pre_check", ik_info)
    if not ok:
        ep.meta.set_outcome(s.OUTCOME_ABORT, f"ik_joint_limit_pre_check_failed:margin={ik_info.get('min_joint_margin_rad')}")
        return s.PHASE_ABORT

    ep._log_sample()
    return s.PHASE_ZERO


def run_zero(ep: CompliantInsertEpisode) -> str:
    """ZERO phase (WRAP-04 + WRAP-05). Returns next phase string."""
    ep.phase = s.PHASE_ZERO
    ep.get_logger().info("=== ZERO: switch to force_mode_controller, STEP-BACK gate, zero F/T ===")

    # Switch to force_mode_controller (WRAP-06's transition begins here)
    if not _switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL], logger=ep.get_logger()):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "switch_to_force_controller_failed")
        return s.PHASE_ABORT

    # WRAP-07: verify the switch actually took effect, not just RPC success
    if not _await_controller_active(FORCE_CTRL, timeout_s=15.0, logger=ep.get_logger()):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "force_mode_controller_did_not_activate")
        return s.PHASE_ABORT

    # 2026-05-07: bumped 0.3s → 2.5s. Root cause of "start_force_mode_failed"
    # transient: _await_controller_active returns the moment controller-manager
    # reports "active", but the controller's internal state takes 0.5-1.5s
    # (sometimes longer after a previous force_mode session) to actually
    # accept service RPCs. Per CLAUDE.md: "controller-manager reports it
    # 'active', but the controller's internal state isn't ready to accept
    # commands yet". 2.5s covers the warmup; the start_force_mode retry loop
    # (now 6 attempts) is the safety net for outliers.
    t_settle_end = time.time() + 2.5
    while time.time() < t_settle_end:
        rclpy.spin_once(ep, timeout_sec=0.05)
        ep._log_sample()

    # WRAP-05: STEP-BACK hands-off gate
    hands_off_start_iso = iso_local_now()
    trigger, ok = ep._await_step_back(ep.args.step_back, ep.args.auto_step_back_seconds)
    if not ok:
        ep.meta.set_outcome(s.OUTCOME_ABORT, f"step_back_gate:{trigger}")
        return s.PHASE_ABORT
    ep.hands_off = 1   # all subsequent rows are hands-off

    # Zero call
    if not ep._zero_ftsensor_call():
        ep.meta.set_outcome(s.OUTCOME_ABORT, "zero_ftsensor_failed")
        return s.PHASE_ABORT

    # 2026-05-07: was 0.5s. The post-zero F/T values stabilize within ~50ms;
    # 0.2s gives ample margin. Saves ~0.3s per insert.
    bias = ep._sample_bias(settle_s=0.2)
    if bias is None:
        ep.meta.set_outcome(s.OUTCOME_ABORT, "no_post_zero_wrench_sample")
        return s.PHASE_ABORT
    ep.meta.set_post_zero_bias(bias)
    # Stash the post-zero bias on ep so downstream contact detection can subtract
    # it before threshold check — without this, residual sensor bias (e.g., 2.67N
    # from session 4 of 2026-05-06 collection) gets misread as contact at the
    # very first APPROACH tick before any descent has happened.
    ep.post_zero_bias_baselink = (
        float(bias["Fx"]), float(bias["Fy"]), float(bias["Fz"]),
        float(bias["Tx"]), float(bias["Ty"]), float(bias["Tz"]),
    )

    max_axis_f = max(abs(bias["Fx"]), abs(bias["Fy"]), abs(bias["Fz"]))
    if max_axis_f > ep.args.bias_warn_n:
        ep.get_logger().warn(
            f"Post-zero residual force {max_axis_f:.2f} N > {ep.args.bias_warn_n} N "
            f"— will subtract from fz before contact-detection threshold check."
        )

    # WRAP-04: +1 s post-zero drift sample
    drift_t_end = time.time() + DEFAULT_DRIFT_WINDOW_S
    while time.time() < drift_t_end:
        rclpy.spin_once(ep, timeout_sec=0.05)
        ep._log_sample()
    ep.zero_event_pending = 1   # mark next row as the drift sample
    drift_bias = ep._sample_bias(settle_s=0.0)
    if drift_bias is not None:
        max_drift = max(
            abs(drift_bias["Fx"] - bias["Fx"]),
            abs(drift_bias["Fy"] - bias["Fy"]),
            abs(drift_bias["Fz"] - bias["Fz"]),
        )
        ep.meta.set_post_zero_drift_check({
            "delta_t_s": DEFAULT_DRIFT_WINDOW_S,
            "Fx": drift_bias["Fx"], "Fy": drift_bias["Fy"], "Fz": drift_bias["Fz"],
            "max_axis_drift_n": round(max_drift, 4),
        })
    ep._log_sample()   # explicit zero_event=1 row

    # Stamp hands-off window start (end is set in run_done/run_abort)
    ep._hands_off_start_iso = hands_off_start_iso
    ep._hands_off_trigger = trigger

    return s.PHASE_ACTIVE


def run_active(ep: CompliantInsertEpisode) -> str:
    """ACTIVE phase (WRAP-06 + WRAP-07).

    Phase 5: if a per-shape config (defaults.yaml + configs/<object>.yaml) is
    available, the loop also evaluates a termination predicate each tick and
    exits as success when it fires sustainedly. Falls back to SIGTERM /
    SIGABRT / --timeout if no config is present.
    """
    ep.phase = s.PHASE_ACTIVE
    ep.get_logger().info("=== ACTIVE: enter force mode, log telemetry until exit ===")

    # Resolve per-shape config (Phase 5). None = legacy timeout-only behaviour.
    term_eval = None
    correction_eval = None
    fsm = None       # ContactSearchFSM (Phase 5 v3, replaces term_eval+correction_eval)
    cfg_path = None
    cfg = None
    predicted_tcp_z_at_seat = None
    try:
        from compliant_insertion_studio.wrapper.insert_config import (
            load_config, resolve_config_for_object,
            TerminationEvaluator, CorrectionEvaluator,
        )
        from compliant_insertion_studio.wrapper.contact_search_fsm import ContactSearchFSM
        if ep.args.config:
            cfg_path = Path(ep.args.config)
        else:
            cfg_path = resolve_config_for_object(ep.args.object_name)
        if cfg_path is not None:
            cfg = load_config(cfg_path)
            hover_z = ep.target_xyz[2] if ep.target_xyz else None

            # Phase 5 v1: CAD-derived universal target. Computes the predicted
            # TCP-at-seat xy/z from the CAD chain. Uses --base-world-pose if
            # provided, else falls back to DEFAULT_BASE_POSITION + DEFAULT_BASE_ORIENTATION
            # (so autonomous SEARCH always has a bootstrap target).
            base_world_pose_for_cad = None
            if ep.args.base_world_pose is not None:
                base_world_pose_for_cad = list(ep.args.base_world_pose)
            else:
                # Fallback: use DEFAULT_BASE_POSITION (same as primitives use).
                try:
                    from primitives.shared.config import (
                        DEFAULT_BASE_POSITION,
                        DEFAULT_BASE_ORIENTATION,
                    )
                    base_world_pose_for_cad = [
                        float(DEFAULT_BASE_POSITION[0]),
                        float(DEFAULT_BASE_POSITION[1]),
                        float(DEFAULT_BASE_POSITION[2]),
                        float(DEFAULT_BASE_ORIENTATION[0]),
                        float(DEFAULT_BASE_ORIENTATION[1]),
                        float(DEFAULT_BASE_ORIENTATION[2]),
                        float(DEFAULT_BASE_ORIENTATION[3]),
                    ]
                    ep.get_logger().info(
                        f"CAD prior using DEFAULT_BASE_POSITION fallback: "
                        f"xyz={base_world_pose_for_cad[:3]}"
                    )
                except Exception as _e:
                    ep.get_logger().warn(f"Could not load DEFAULT_BASE_POSITION: {_e}")
            if base_world_pose_for_cad is not None:
                try:
                    from compliant_insertion_studio.wrapper.cad_lookup import (
                        predict_tcp_at_seat,
                    )
                    bp = base_world_pose_for_cad
                    # Phase 5 v2 (2026-05-04): cad_lookup now needs held_quat
                    # (object orientation in world) and EE orientation in world
                    # so it can apply fold-symmetry equivalents and project the
                    # flange offset along the gripper's tool-Z (not the object's
                    # rotated Z, which was the v1 bug — see cad_lookup docstring).
                    held_quat = ep.args.current_object_orientation
                    if held_quat is None:
                        raise ValueError(
                            "predict_tcp_at_seat requires --current-object-orientation"
                        )
                    # EE orientation at PRE — usually face-down already (set by
                    # the rotate_object primitive during setup). The wrapper
                    # samples ep.tcp from /tcp_pose_broadcaster.
                    if ep.tcp is None:
                        # Spin a few times to populate
                        import time as _time
                        for _ in range(50):
                            rclpy.spin_once(ep, timeout_sec=0.05)
                            if ep.tcp is not None:
                                break
                            _time.sleep(0.02)
                    if ep.tcp is None:
                        raise ValueError(
                            "EE pose not available from /tcp_pose_broadcaster"
                        )
                    ee_q = ep.tcp.pose.orientation
                    ee_quat = [float(ee_q.x), float(ee_q.y), float(ee_q.z), float(ee_q.w)]
                    pred = predict_tcp_at_seat(
                        base_name=ep.args.base_name,
                        object_name=ep.args.object_name,
                        grasp_id=int(ep.args.grasp_id),
                        base_world_xyz=bp[:3],
                        base_world_quat_xyzw=bp[3:],
                        held_quat_xyzw=list(held_quat),
                        ee_orientation_xyzw=ee_quat,
                    )
                    predicted_tcp_z_at_seat = pred["predicted_tcp_at_seat"]["xyz_m"][2]
                    fs = pred.get("fold_symmetry_used", {})
                    ep.get_logger().info(
                        f"Phase 5 v2 CAD-derived predicted_tcp_z_at_seat = "
                        f"{predicted_tcp_z_at_seat:.4f} m  "
                        f"(hover_z={hover_z:.4f}, expected descent="
                        f"{(hover_z - predicted_tcp_z_at_seat) * 1000:.1f} mm; "
                        f"fold_sym pos_err={fs.get('pos_error_mm', 0):.2f}mm "
                        f"ang_err={fs.get('angle_error_deg', 0):.1f}°)"
                    )
                    ep.meta.set_optional("cad_prediction", pred)
                except Exception as e:
                    ep.get_logger().error(
                        f"cad_lookup failed ({e}) — predicate falls back to v0 hover-relative"
                    )

            # Phase 5 v3 (2026-05-04 PM): ContactSearchFSM replaces the
            # term_eval + correction_eval architecture. The FSM follows
            # the canonical ConnTact pattern (APPROACH → FIND_HOLE → INSERT)
            # with measured (not assumed) state transitions. This honors
            # operator's "no hard z" requirement and the per-part agnostic
            # design — surface_z is measured at APPROACH exit, hole-drop is
            # detected as a relative z descent below surface_z, INSERT exits
            # on motion-stopped post-hole-entry.
            fsm_cfg = (cfg.get("fsm") or {}) if cfg else {}
            # 2026-05-07: per-object SEAT tolerance override. Tighter for
            # objects (line_green) where the autonomous can plausibly stop a
            # few mm above the true seat, so we want the global seat detector
            # to NOT fire prematurely.
            try:
                from primitives.shared.config import (
                    get_object_seat_tolerance_m, is_free_yaw_insert,
                )
                seat_tol = get_object_seat_tolerance_m(ep.args.object_name)
                if seat_tol != 0.005:  # only override if non-default
                    fsm_cfg = dict(fsm_cfg)
                    fsm_cfg['at_target_z_tol_m'] = seat_tol
                    ep.get_logger().info(
                        f"Per-object SEAT tolerance for {ep.args.object_name}: "
                        f"{seat_tol*1000:.1f}mm (default 5.0mm)"
                    )
                if is_free_yaw_insert(ep.args.object_name):
                    fsm_cfg = dict(fsm_cfg)
                    fsm_cfg['free_yaw_insert'] = True
                    # Free-yaw INSERT_DESCENT EXPECTS slot walls to push laterally
                    # to rotate the peg into alignment, so the default 30N/100ms
                    # F_lat safety abort fires on the very physics we want.
                    fsm_cfg['abort_F_lat_N']        = 60.0
                    fsm_cfg['abort_F_lat_window_s'] = 0.30
                    ep.get_logger().info(
                        f"Per-object free-yaw INSERT_DESCENT enabled for "
                        f"{ep.args.object_name}: peg can rotate about world Z "
                        f"under slot-wall torque feedback "
                        f"(abort_F_lat 30→60N, window 100→300ms)"
                    )
                # Per-object insert_fz_N override (forces during INSERT_DESCENT)
                from primitives.shared.config import (
                    get_object_insert_forces, get_object_search_mode,
                )
                obj_forces = get_object_insert_forces(ep.args.object_name)
                if 'insert_fz' in obj_forces:
                    fsm_cfg = dict(fsm_cfg)
                    fsm_cfg['insert_fz_N'] = float(obj_forces['insert_fz'])
                    ep.get_logger().info(
                        f"Per-object INSERT_DESCENT Fz for {ep.args.object_name}: "
                        f"{fsm_cfg['insert_fz_N']}N (default 8.0N)"
                    )
                # Per-object yaw bias (active Tz commanded during free-yaw)
                from primitives.shared.config import get_object_yaw_bias_nm
                yaw_bias = get_object_yaw_bias_nm(ep.args.object_name)
                if yaw_bias != 0.0:
                    fsm_cfg = dict(fsm_cfg)
                    fsm_cfg['yaw_bias_nm'] = float(yaw_bias)
                    ep.get_logger().info(
                        f"Per-object yaw bias for {ep.args.object_name}: "
                        f"Tz={yaw_bias:+.2f}Nm (active rotation bias under free-yaw)"
                    )
                # Per-object SEARCH mode (spiral vs compliant_descent)
                search_mode = get_object_search_mode(ep.args.object_name)
                if search_mode != 'spiral':
                    fsm_cfg = dict(fsm_cfg)
                    fsm_cfg['search_mode'] = search_mode
                    # In compliant_descent, sensed F_lat is the very signal
                    # we want the controller to RESPOND to (slot rim pushing
                    # peg into channel). Loosen abort thresholds further.
                    fsm_cfg['abort_F_lat_N']        = 80.0
                    fsm_cfg['abort_F_lat_window_s'] = 0.50
                    ep.get_logger().info(
                        f"Per-object SEARCH mode for {ep.args.object_name}: "
                        f"{search_mode} (no spiral; passive XYZ compliance + escalation; "
                        f"abort_F_lat 30→80N, window 100→500ms)"
                    )
            except Exception:
                pass
            # Inject --guided-mode into the FSM config so APPROACH→contact routes
            # to GUIDED state instead of FIND_HOLE.
            if getattr(ep.args, 'guided_mode', False):
                fsm_cfg = dict(fsm_cfg)  # don't mutate the loaded YAML dict
                fsm_cfg['guided_mode'] = True
                # Stage 3a/3b: v4 autofire flag passed through to FSM. False (3a)
                # = log only; True (3b) = v4 fire also triggers GUIDED→INSERT_DESCENT.
                fsm_cfg['v4_autofire'] = bool(getattr(ep.args, 'v4_autofire', False))
            if getattr(ep.args, 'autonomous_search', False):
                fsm_cfg = dict(fsm_cfg)
                fsm_cfg['autonomous_search'] = True
                fsm_cfg['search_F_press_N'] = float(getattr(ep.args, 'search_F_press_N', 9.0))
                fsm_cfg['search_max_duration_s'] = float(getattr(ep.args, 'search_max_duration_s', 15.0))
                fsm_cfg['search_Fmax_N'] = float(getattr(ep.args, 'search_Fmax_N', 3.0))
                fsm_cfg['search_v_s_m_s'] = float(getattr(ep.args, 'search_v_s_mm_s', 5.0)) / 1000.0
                fsm_cfg['search_pitch_m'] = float(getattr(ep.args, 'search_pitch_mm', 2.0)) / 1000.0
                fsm_cfg['search_R_max_m'] = float(getattr(ep.args, 'search_R_max_mm', 8.0)) / 1000.0
            # Pass hole_xy_prior as spiral origin override (cross-attempt learning).
            # If a previous run found the hole xy via z-drop detection, anchor
            # this spiral search there rather than peg's random contact xy.
            spiral_origin_override = None
            if ep.args.hole_xy_prior is not None:
                spiral_origin_override = (float(ep.args.hole_xy_prior[0]),
                                          float(ep.args.hole_xy_prior[1]))
            # Predicted TCP xy: target the FSM uses for recovery leash and
            # engagement distance gate. Priority: observed truth > CAD prediction.
            # When --hole-xy-prior is supplied (cross-attempt observed seat OR
            # operator-injected truth), use it as the leash target — observed
            # data is more reliable than CAD which can have 15-20mm error per
            # u_orange empirics. Falls back to CAD chain otherwise.
            predicted_tcp_xy = None
            if ep.args.hole_xy_prior is not None:
                predicted_tcp_xy = (float(ep.args.hole_xy_prior[0]),
                                    float(ep.args.hole_xy_prior[1]))
            else:
                try:
                    cad = ep.meta._meta.get("cad_prediction") if hasattr(ep.meta, "_meta") else None
                    if isinstance(cad, dict):
                        pt = cad.get("predicted_tcp_at_seat", {}).get("xyz_m")
                        if pt and len(pt) >= 2:
                            predicted_tcp_xy = (float(pt[0]), float(pt[1]))
                except Exception:
                    pass

            # Vector from TCP to part_center (object origin) in WORLD frame
            # at canonical seat. Used by FSM to compensate Fz→Ty lever-arm
            # moment that creates spurious tilt around part center even with
            # no commanded torques. Operator's geometric insight: grasp is
            # offset ~28mm from part center; pure Fz at TCP becomes
            # F + (r × F) torque at part center, biasing tilt.
            r_grasp_to_partcenter_world = (0.0, 0.0, 0.0)
            try:
                cad = ep.meta._meta.get("cad_prediction") if hasattr(ep.meta, "_meta") else None
                if isinstance(cad, dict):
                    tcp = cad.get("predicted_tcp_at_seat", {}).get("xyz_m")
                    obj = cad.get("T_world_object_seat", {}).get("xyz_m")
                    if tcp and obj and len(tcp) >= 2 and len(obj) >= 2:
                        r_grasp_to_partcenter_world = (
                            float(obj[0]) - float(tcp[0]),
                            float(obj[1]) - float(tcp[1]),
                            float(obj[2]) - float(tcp[2]) if len(tcp) >= 3 and len(obj) >= 3 else 0.0,
                        )
                        ep.get_logger().info(
                            f"r_grasp_to_partcenter_world (CAD-derived): "
                            f"({r_grasp_to_partcenter_world[0]*1000:+.1f}, "
                            f"{r_grasp_to_partcenter_world[1]*1000:+.1f}, "
                            f"{r_grasp_to_partcenter_world[2]*1000:+.1f}) mm — "
                            f"used for Fz→T lever-arm compensation"
                        )
            except Exception as e:
                ep.get_logger().warn(f"Failed to compute r_grasp_to_partcenter: {e}")
            try:
                # Compute object_origin in EE frame from CAD r_world. CAD
                # gives r_world at canonical face-down EE [0,-1,0,0] (rotation
                # matrix = diag(-1, 1, -1)). To get object_origin_in_EE,
                # apply inverse rotation (= self for diagonal ±1): just
                # negate x and z components of r_world.
                object_origin_in_EE = (
                    -r_grasp_to_partcenter_world[0],
                    +r_grasp_to_partcenter_world[1],
                    -r_grasp_to_partcenter_world[2],
                )
                # CAD-derived contact candidates: {SEAT, BASE_RIM, OBJ:<name>...}
                # Used by FSM at contact moment to classify and short-circuit
                # the seat-on-touchdown case to DONE. Inert if any input missing.
                contact_candidates = None
                try:
                    if (predicted_tcp_z_at_seat is not None
                            and pred is not None and bp is not None):
                        from compliant_insertion_studio.wrapper.cad_geometry import (
                            build_contact_candidates, get_seated_so_far,
                        )
                        R_grasp = pred.get("fold_symmetry_used", {}).get(
                            "R_eq_quat_xyzw")
                        if R_grasp is not None:
                            seated = get_seated_so_far(ep.args.base_name,
                                                        ep.args.object_name)
                            contact_candidates = build_contact_candidates(
                                target_object=ep.args.object_name,
                                target_grasp_id=int(ep.args.grasp_id),
                                base_name=ep.args.base_name,
                                base_world_z=float(bp[2]),
                                R_grasp_quat_xyzw=R_grasp,
                                predicted_tcp_z_at_seat=predicted_tcp_z_at_seat,
                                seated_objects=seated,
                            )
                            ep.get_logger().info(
                                "Contact candidates (CAD-derived): "
                                + ", ".join(f"{k}={v*1000:.2f}mm"
                                            for k, v in contact_candidates.items())
                                + f" — seated_so_far={seated}"
                            )
                except Exception as e:
                    ep.get_logger().warn(
                        f"build_contact_candidates failed ({e}) — FSM falls "
                        f"back to today's APPROACH→SEARCH/GUIDED dispatch")

                fsm = ContactSearchFSM(fsm_cfg,
                                       spiral_origin_override=spiral_origin_override,
                                       predicted_tcp_xy=predicted_tcp_xy,
                                       predicted_tcp_z=predicted_tcp_z_at_seat,
                                       r_grasp_to_partcenter_world=r_grasp_to_partcenter_world,
                                       object_origin_in_EE=object_origin_in_EE,
                                       contact_candidates=contact_candidates)
                # Bind FSM to episode object so signal handlers (SIGUSR1 hole-mark)
                # can reach it via ep.fsm. Without this, the SIGUSR1 handler's
                # getattr(ep, 'fsm', None) returns None and mark_hole() never fires.
                ep.fsm = fsm
                _orig_str = (f"override=({spiral_origin_override[0]:+.4f},"
                             f"{spiral_origin_override[1]:+.4f})"
                             if spiral_origin_override is not None else "= contact_xy")
                _pred_str = (f"predicted_xy=({predicted_tcp_xy[0]:+.4f},{predicted_tcp_xy[1]:+.4f})"
                             if predicted_tcp_xy is not None else "predicted_xy=None")
                ep.get_logger().info(
                    f"Phase 5 v3 FSM active — APPROACH → FIND_HOLE → ENTRY_SETTLE → INSERT "
                    f"(spiral v={fsm.spiral_v_m_s*1000:.1f}mm/s p={fsm.spiral_pitch_m*1000:.1f}mm "
                    f"max_radius={fsm.find_hole_max_radius_m*1000:.0f}mm origin {_orig_str}, {_pred_str}, "
                    f"recovery r_free={fsm.recovery_r_free_m*1000:.0f}mm r_full={fsm.recovery_r_full_m*1000:.0f}mm "
                    f"F_max={fsm.recovery_F_max_N:.1f}N, engagement_dist_gate={fsm.engagement_dist_thresh_m*1000:.0f}mm)"
                )
                ep.meta.set_optional("fsm_config", fsm_cfg)
            except Exception as _e:
                ep.get_logger().error(f"FSM init failed ({_e}) — falling back to --timeout")
                fsm = None
        else:
            ep.get_logger().info(
                f"No config found for object={ep.args.object_name!r} — "
                "running legacy --timeout-only ACTIVE loop."
            )
    except Exception as e:
        ep.get_logger().warn(f"Config load failed ({e}) — legacy mode.")

    # v1.1: snapshot TCP world pose at the moment ACTIVE begins. Phase 5
    # uses this as the time-zero anchor for all per-row alignment / signature
    # extraction (e.g., descent depth = tcp_z(t) - tcp_z(active_start)).
    if ep.tcp is not None:
        _p = ep.tcp.pose.position
        _q = ep.tcp.pose.orientation
        ep.meta.set_optional("tcp_pose_at_active_start", {
            "xyz_m": [_p.x, _p.y, _p.z],
            "quat_xyzw": [_q.x, _q.y, _q.z, _q.w],
        })

    sel_vec = _parse_selection(ep.args.selection)
    if not ep._start_force_mode(sel_vec):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "start_force_mode_failed")
        return s.PHASE_ABORT

    # Stamp the params used (for meta JSON)
    ep.meta.set_force_mode_params({
        "task_frame": "base_link", "type": 2,
        "selection_vector": sel_vec,
        "wrench": {"fx": 0.0, "fy": 0.0, "fz": -float(ep.args.fz),
                   "tx": 0.0, "ty": 0.0, "tz": 0.0},
        "speed_limits": {
            "linear_xyz_m_s":  [float(ep.args.lin_speed)] * 3,
            "angular_xyz_r_s": [float(ep.args.ang_speed)] * 3,
        },
        "gain_scaling": float(ep.args.gain),
        "damping_factor": float(ep.args.damping),
    })

    period = 1.0 / max(float(ep.args.rate_hz), 1.0)
    next_tick = time.time()
    active_deadline = time.time() + float(ep.args.timeout)

    while True:
        rclpy.spin_once(ep, timeout_sec=0.005)

        # Signal-driven outcome
        if ep.outcome_signal == "success":
            ep.meta.set_outcome(s.OUTCOME_SUCCESS, "operator_sigterm")
            # Persist GUIDED-mode hole_observed even on SIGTERM exit (data
            # collection demos typically end via Ctrl+C — without this, the
            # operator-marked hole xy is lost from meta despite being set).
            _fsm = getattr(ep, 'fsm', None)
            if _fsm is not None and getattr(_fsm, 'hole_observed_xy', None) is not None:
                ep.meta.set_optional("hole_observed_operator", {
                    "xy_m":   list(_fsm.hole_observed_xy),
                    "z_m":    float(_fsm.hole_observed_z) if _fsm.hole_observed_z is not None else 0.0,
                    "t_s":    float(_fsm.hole_observed_t) if _fsm.hole_observed_t is not None else 0.0,
                    "source": (
                        "fsm_guided_v4_autofire"
                        if (getattr(_fsm, 'v4_predicate_fire', None) is not None
                            and _fsm.v4_predicate_fire.get('autofired'))
                        else "fsm_guided_sigusr1"
                    ),
                })
            if _fsm is not None and getattr(_fsm, 'v4_predicate_fire', None) is not None:
                ep.meta.set_optional("hole_observed_v4_predicate", _fsm.v4_predicate_fire)
            if _fsm is not None and getattr(_fsm, 'contact_classification', None) is not None:
                ep.meta.set_optional("contact_classification", _fsm.contact_classification)
            return s.PHASE_DONE
        if ep.outcome_signal == "abort":
            ep.meta.set_outcome(s.OUTCOME_ABORT, "operator_sigabrt")
            _fsm = getattr(ep, 'fsm', None)
            if _fsm is not None and getattr(_fsm, 'hole_observed_xy', None) is not None:
                ep.meta.set_optional("hole_observed_operator", {
                    "xy_m":   list(_fsm.hole_observed_xy),
                    "z_m":    float(_fsm.hole_observed_z) if _fsm.hole_observed_z is not None else 0.0,
                    "t_s":    float(_fsm.hole_observed_t) if _fsm.hole_observed_t is not None else 0.0,
                    "source": (
                        "fsm_guided_v4_autofire"
                        if (getattr(_fsm, 'v4_predicate_fire', None) is not None
                            and _fsm.v4_predicate_fire.get('autofired'))
                        else "fsm_guided_sigusr1"
                    ),
                })
            if _fsm is not None and getattr(_fsm, 'v4_predicate_fire', None) is not None:
                ep.meta.set_optional("hole_observed_v4_predicate", _fsm.v4_predicate_fire)
            if _fsm is not None and getattr(_fsm, 'contact_classification', None) is not None:
                ep.meta.set_optional("contact_classification", _fsm.contact_classification)
            return s.PHASE_ABORT

        # Mid-episode re-zero (SIGUSR2)
        if ep._zero_request:
            ep._zero_request = False
            t_now_s = time.time() - ep.start_t
            ep.get_logger().warn(f"SIGUSR2 — mid-episode re-zero at t={t_now_s:.2f}s")
            ep._stop_force_mode()
            time.sleep(0.3)
            if ep._zero_ftsensor_call():
                bias = ep._sample_bias(settle_s=0.5)
                if bias is not None:
                    ep.meta.add_mid_episode_zero(t_now_s, bias)
                ep.zero_event_pending = 1
                ep._log_sample()
                # Restart force mode with same params
                if not ep._start_force_mode(sel_vec):
                    ep.meta.set_outcome(s.OUTCOME_ABORT, "restart_force_mode_after_rezero_failed")
                    return s.PHASE_ABORT

        # ====================================================================
        # Phase 5 v3 — ContactSearchFSM dispatch (replaces term_eval + Mode B)
        # ====================================================================
        if fsm is not None and ep.tcp is not None and ep.wrench is not None:
            t_now_fsm = time.time()
            _p = ep.tcp.pose.position
            _q = ep.tcp.pose.orientation
            (Fx_b, Fy_b, Fz_b), (Tx_b, Ty_b, Tz_b) = ep._wrench_in_base(ep.wrench)
            # Subtract post-zero residual bias before passing to FSM. Without
            # this, residual F/T zero contamination (e.g. 2.67N session-4 bias)
            # crosses contact_threshold_N at the very first APPROACH tick →
            # phantom contact at hover. Bias is set in PRE/ZERO phase.
            _bias = getattr(ep, 'post_zero_bias_baselink', None)
            if _bias is not None:
                Fx_b -= _bias[0]; Fy_b -= _bias[1]; Fz_b -= _bias[2]
                Tx_b -= _bias[3]; Ty_b -= _bias[4]; Tz_b -= _bias[5]
            fsm_action = fsm.update(
                t_now_fsm, _p.x, _p.y, _p.z, Fz_b,
                F_lat_baselink=(Fx_b, Fy_b),
                T_lat_baselink=(Tx_b, Ty_b),
                tcp_quat_xyzw=(float(_q.x), float(_q.y), float(_q.z), float(_q.w)),
            )

            # Throttled diagnostic
            if not hasattr(ep, '_last_fsm_diag_t'):
                ep._last_fsm_diag_t = 0.0
            if (t_now_fsm - ep._last_fsm_diag_t) >= 1.0:
                ep._last_fsm_diag_t = t_now_fsm
                if fsm_action.msg:
                    ep.get_logger().info(fsm_action.msg)

            # State-transition log (always)
            if fsm_action.transitioned:
                ep.get_logger().warn(
                    f"=== FSM → {fsm_action.new_state}: {fsm_action.transition_msg}"
                )

                # --abort-on-first-contact: stop right after APPROACH→FIND_HOLE
                # transition fires, before any FIND_HOLE wrench is applied. Used
                # to validate the contact-detection marker (regime-decoding analysis
                # uses this exact moment as t=0). Routes through the exit_abort
                # cleanup path so force mode stops + safe-height retract run.
                if (getattr(ep.args, 'abort_on_first_contact', False)
                        and fsm_action.new_state == fsm.FIND_HOLE):
                    ep.get_logger().warn(
                        f"=== ABORT-ON-FIRST-CONTACT (CLI flag): exiting at first "
                        f"contact moment for marker validation. surface_z={fsm.surface_z:.4f}m"
                    )
                    fsm_action.kind = "exit_abort"
                    fsm_action.abort_reason = "abort_on_first_contact"

            # Terminal lifecycles
            if fsm_action.kind == "exit_done":
                ep.get_logger().info(
                    f"FSM SEATED: surface_z={fsm.surface_z}, hole_xy={fsm.hole_xy}, "
                    f"hole_z={fsm.hole_z}, final_z={_p.z:.4f}, "
                    f"descent_post_hole_mm="
                    f"{((fsm.hole_z - _p.z)*1000) if fsm.hole_z is not None else None}"
                )
                ep.meta.set_outcome(s.OUTCOME_SUCCESS, "fsm_seated")
                ep.meta.set_optional("fsm_result", {
                    "surface_z_m":            fsm.surface_z,
                    "hole_xy_m":              list(fsm.hole_xy) if fsm.hole_xy else None,
                    "hole_z_m":               fsm.hole_z,
                    "final_tcp_z_m":          float(_p.z),
                    "descent_post_hole_m":    (fsm.hole_z - _p.z) if fsm.hole_z is not None else None,
                    "transition_msg":         fsm_action.transition_msg,
                })
                # Persist hole_xy for cross-attempt learning (iterate_insert reads this)
                if fsm.hole_xy is not None:
                    ep.meta.set_optional("hole_observed", {
                        "xy_m":   list(fsm.hole_xy),
                        "z_m":    float(fsm.hole_z) if fsm.hole_z is not None else 0.0,
                        "source": "fsm_find_hole",
                    })
                # GUIDED-mode: persist operator-marked hole position too. This is the
                # ground-truth hole_xy from the operator's hand at SIGUSR1 — the
                # primary data product of GOLD data collection.
                if getattr(fsm, 'hole_observed_xy', None) is not None:
                    ep.meta.set_optional("hole_observed_operator", {
                        "xy_m":  list(fsm.hole_observed_xy),
                        "z_m":   float(fsm.hole_observed_z) if fsm.hole_observed_z is not None else 0.0,
                        "t_s":   float(fsm.hole_observed_t) if fsm.hole_observed_t is not None else 0.0,
                        "source": (
                            "fsm_guided_v4_autofire"
                            if (getattr(fsm, 'v4_predicate_fire', None) is not None
                                and fsm.v4_predicate_fire.get('autofired'))
                            else "fsm_guided_sigusr1"
                        ),
                    })
                if getattr(fsm, 'v4_predicate_fire', None) is not None:
                    ep.meta.set_optional("hole_observed_v4_predicate", fsm.v4_predicate_fire)
                if getattr(fsm, 'contact_classification', None) is not None:
                    ep.meta.set_optional("contact_classification", fsm.contact_classification)
                return s.PHASE_DONE

            if fsm_action.kind == "exit_abort":
                ep.get_logger().error(
                    f"FSM ABORT in {fsm.state}: {fsm_action.abort_reason}"
                )
                ep.meta.set_outcome(s.OUTCOME_ABORT, f"fsm_abort:{fsm_action.abort_reason[:80]}")
                ep.meta.set_optional("fsm_result", {
                    "abort_state":   fsm.state,
                    "abort_reason":  fsm_action.abort_reason,
                    "surface_z_m":   fsm.surface_z,
                    "hole_xy_m":     list(fsm.hole_xy) if fsm.hole_xy else None,
                    "hole_z_m":      fsm.hole_z,
                    "final_tcp_z_m": float(_p.z),
                })
                # Same as DONE: persist GUIDED-mode operator-marked hole if present
                if getattr(fsm, 'hole_observed_xy', None) is not None:
                    ep.meta.set_optional("hole_observed_operator", {
                        "xy_m":  list(fsm.hole_observed_xy),
                        "z_m":   float(fsm.hole_observed_z) if fsm.hole_observed_z is not None else 0.0,
                        "t_s":   float(fsm.hole_observed_t) if fsm.hole_observed_t is not None else 0.0,
                        "source": (
                            "fsm_guided_v4_autofire"
                            if (getattr(fsm, 'v4_predicate_fire', None) is not None
                                and fsm.v4_predicate_fire.get('autofired'))
                            else "fsm_guided_sigusr1"
                        ),
                    })
                if getattr(fsm, 'v4_predicate_fire', None) is not None:
                    ep.meta.set_optional("hole_observed_v4_predicate", fsm.v4_predicate_fire)
                if getattr(fsm, 'contact_classification', None) is not None:
                    ep.meta.set_optional("contact_classification", fsm.contact_classification)
                return s.PHASE_ABORT

            # Wrench update if needed (PD-spiral re-issues every tick during FIND_HOLE)
            if fsm_action.new_wrench:
                ep._start_force_mode(
                    list(fsm_action.selection_vector),
                    override_wrench_baselink=fsm_action.wrench_baselink,
                    gain_override=fsm_action.gain,
                    damping_override=fsm_action.damping,
                    lin_speed_override=fsm_action.lin_speed,
                    ang_speed_override=fsm_action.ang_speed,
                    quiet=True,
                )

            # Skip the legacy term_eval / Mode B path below
            term_eval = None
            correction_eval = None

        # === Legacy paths (only run if FSM is None, e.g. config-less mode) ===
        # Phase 5 — termination predicate (autonomous exit)
        if term_eval is not None and ep.tcp is not None:
            _p = ep.tcp.pose.position
            # Pass current fz so the evaluator can auto-detect contact.
            # IMPORTANT: contact threshold expects BASE-frame Fz (peg pushed up = +Z).
            # Raw ep.wrench is in tool0_controller frame and has the OPPOSITE sign for
            # face-down EE — using raw fz would silently disable contact detection,
            # which in turn gates off Mode B and the diag prints. Use _wrench_in_base.
            fz_now = None
            if ep.wrench is not None:
                (_fx, _fy, _fz), _ = ep._wrench_in_base(ep.wrench)
                fz_now = _fz
            fired, dbg = term_eval.eval(_p.x, _p.y, _p.z, time.time(),
                                         fz_smoothed=fz_now)
            # Notify correction evaluator on first contact (so its warmup timer starts)
            if correction_eval is not None and term_eval.contact_z is not None:
                correction_eval.note_contact(time.time())
            # Predicate-state diagnostic — print every 1 s so we can see WHY
            # the predicate hasn't fired (which sub-condition is failing).
            _t_now_diag = time.time()
            if not hasattr(ep, '_last_pred_diag_t'):
                ep._last_pred_diag_t = 0.0
            if (_t_now_diag - ep._last_pred_diag_t) >= 1.0:
                ep._last_pred_diag_t = _t_now_diag
                _r = dbg.get('results', {})
                _v_lat = dbg.get('v_lat', 0.0) * 1000
                _v_z = dbg.get('v_z', 0.0) * 1000
                _dpc = dbg.get('descended_post_contact')
                _dpc_mm = (_dpc * 1000) if _dpc is not None else None
                _sustained = dbg.get('sustained_s', 0.0)
                ep.get_logger().info(
                    f"Predicate: motion={_r.get('motion_stopped','?')} "
                    f"at_seat={_r.get('tcp_z_reached_predicted','?')} "
                    f"dpc={_r.get('descended_post_contact','?')} | "
                    f"v_lat={_v_lat:.2f}mm/s v_z={_v_z:+.2f}mm/s "
                    f"tcp_z={_p.z:.4f} dpc_mm={_dpc_mm} "
                    f"sustained={_sustained:.2f}/{term_eval.sustain_s:.1f}s"
                )
            if fired:
                _desc_hover = dbg.get('descended_from_hover', 0.0) or 0.0
                _desc_pc = dbg.get('descended_post_contact')
                _desc_pc_str = f"{_desc_pc:.4f}" if _desc_pc is not None else "n/a"
                ep.get_logger().info(
                    f"Termination predicate fired: descended_from_hover={_desc_hover:.4f} m, "
                    f"descended_post_contact={_desc_pc_str} m, "
                    f"v_lat={dbg.get('v_lat', 0.0):.4f} m/s, "
                    f"sustained={dbg.get('sustained_s', 0.0):.2f} s, "
                    f"results={dbg.get('results', {})}"
                )
                ep.meta.set_outcome(s.OUTCOME_SUCCESS, "predicate_met")
                ep.meta.set_optional("termination_fire_debug", {
                    "v_lat_at_fire_m_s": float(dbg.get("v_lat", 0.0)),
                    "descended_from_hover_at_fire_m": float(_desc_hover),
                    "descended_post_contact_at_fire_m": (
                        float(_desc_pc) if _desc_pc is not None else None
                    ),
                    "sustained_s": float(dbg.get("sustained_s", 0.0)),
                    "predicate_results": dbg.get("results", {}),
                })
                return s.PHASE_DONE

        # Phase 5 Mode B — active correction (back-off-on-stuck + exploration)
        if correction_eval is not None and ep.tcp is not None and ep.wrench is not None:
            t_now = time.time()
            tcp_z_now = ep.tcp.pose.position.z
            # Sample current state in BASE_LINK frame
            (Fx_b, Fy_b, Fz_b), (Tx_b, Ty_b, Tz_b) = ep._wrench_in_base(ep.wrench)
            # Diagnostic — once per second so operator can SEE detection state
            diag = correction_eval.get_diag(t_now, Fz_b, tcp_z_now, (Fx_b, Fy_b), (Tx_b, Ty_b))
            if diag:
                ep.get_logger().info(diag)
            # Pass target xy if CAD prediction is available — Mode B uses this
            # for TOWARD-TARGET directional correction (much better than
            # counter-residual; analysis showed counter-residual pointed AWAY
            # from target 58% of corrections in iter3).
            tcp_xy_now = (ep.tcp.pose.position.x, ep.tcp.pose.position.y)
            target_xy = None
            # Highest-priority: --hole-xy-prior override (from a prior attempt
            # where the spiral detected the actual hole). This is the most
            # accurate target_xy because it accounts for per-grasp variance.
            if ep.args.hole_xy_prior is not None:
                target_xy = (float(ep.args.hole_xy_prior[0]), float(ep.args.hole_xy_prior[1]))
            # Otherwise fall back to CAD-derived prediction
            if target_xy is None and predicted_tcp_z_at_seat is not None:
                cad = (ep.meta._meta.get("cad_prediction") if hasattr(ep.meta, "_meta") else None)
                if isinstance(cad, dict):
                    pt = cad.get("predicted_tcp_at_seat", {}).get("xyz_m")
                    if pt and len(pt) >= 2:
                        target_xy = (float(pt[0]), float(pt[1]))
            # Final fallback: ep.target_xyz (hover xy ≈ assembly center xy)
            if target_xy is None and ep.target_xyz is not None:
                target_xy = (float(ep.target_xyz[0]), float(ep.target_xyz[1]))

            # at_seat: if predicted seat z is known and tcp_z is within 5mm,
            # suppress new Mode B triggers (predicate will fire as peg settles).
            at_seat = False
            if predicted_tcp_z_at_seat is not None:
                at_seat = abs(tcp_z_now - predicted_tcp_z_at_seat) <= 0.005
            action, payload = correction_eval.update(
                t_now, Fz_b, tcp_z_now, (Fx_b, Fy_b), (Tx_b, Ty_b),
                tcp_xy=tcp_xy_now, target_xy=target_xy, at_seat=at_seat,
            )
            if action == "apply":
                # Poke mode (v1 legacy): one-shot delta wrench
                dfx, dfy, dtx, dty = payload["delta"]
                fz_default = -float(ep.args.fz)  # base_link frame: push down = -|fz|
                ep.get_logger().warn(
                    f"=== Mode B #{payload['n']} ({payload['mode']}): "
                    f"residual F={payload['residual_F_N']:.2f}N T={payload['residual_T_Nm']:.3f}Nm "
                    f"net_descent={payload['net_descent_rate_mm_s']:.3f}mm/s → "
                    f"delta F=({dfx:+.2f},{dfy:+.2f})N T=({dtx:+.3f},{dty:+.3f})Nm for "
                    f"{correction_cfg.get('action',{}).get('duration_s', 0.4):.2f}s"
                )
                ep._start_force_mode(
                    sel_vec,
                    override_wrench_baselink=(dfx, dfy, fz_default, dtx, dty, 0.0),
                )
            elif action == "apply_spiral":
                # Spiral mode (v2): full base-link wrench setpoint with lower
                # gain/damping. Re-issued at spiral_command_period_s during burst.
                wrench = payload["wrench_baselink"]
                # Log only the FIRST setpoint of each burst (when 'n' is in payload)
                # OR when phase changes from retract→search. All other re-issues
                # are quiet to avoid log spam at 20 Hz.
                first_in_burst = "n" in payload
                if first_in_burst:
                    ep.get_logger().warn(
                        f"=== Mode B #{payload['n']} ({payload['mode']}): "
                        f"residual F={payload['residual_F_N']:.2f}N T={payload['residual_T_Nm']:.3f}Nm "
                        f"net_descent={payload['net_descent_rate_mm_s']:.3f}mm/s → "
                        f"wrench={wrench} gain={payload['gain']} damp={payload['damping']} "
                        f"for {correction_cfg.get('action',{}).get('duration_s', 1.5):.2f}s"
                    )
                ep._start_force_mode(
                    sel_vec,
                    override_wrench_baselink=wrench,
                    gain_override=payload["gain"],
                    damping_override=payload["damping"],
                    quiet=not first_in_burst,
                )
            elif action == "revert":
                # Two flavors of revert:
                # (a) normal end-of-burst (payload=None): cooldown after spiral completes
                # (b) early-exit hole-detected (payload={hole_xy, hole_z, rate, ...}):
                #     CorrectionEvaluator detected z-descent spike → peg found
                #     chamfer/slot. Stop the spiral, command default wrench (-Fz),
                #     let peg drop into hole. Termination predicate then fires
                #     when settled.
                if isinstance(payload, dict) and payload.get("reason") == "hole_detected":
                    hxy = payload["hole_xy"]
                    ep.get_logger().warn(
                        f"=== HOLE DETECTED at correction #{payload['correction']}: "
                        f"descent rate spiked to {payload['rate_mm_s']:.1f} mm/s; "
                        f"hole xy = ({hxy[0]:+.4f}, {hxy[1]:+.4f}) m, "
                        f"z = {payload['hole_z']:.4f} m. "
                        f"Reverting to default wrench so peg can settle."
                    )
                    # Compute predicted xy for traceability (delta = grasp variance)
                    predicted_xy = [None, None]
                    cad_pred = (ep.meta._meta.get("cad_prediction")
                                if hasattr(ep.meta, "_meta") else None)
                    if isinstance(cad_pred, dict):
                        pt = cad_pred.get("predicted_tcp_at_seat", {}).get("xyz_m")
                        if pt and len(pt) >= 2:
                            predicted_xy = [float(pt[0]), float(pt[1])]
                    # Persist for cross-attempt learning + analysis
                    ep.meta.set_optional("hole_observed", {
                        "xy_m":           list(hxy),
                        "z_m":            float(payload["hole_z"]),
                        "rate_mm_s":      float(payload["rate_mm_s"]),
                        "correction":     int(payload["correction"]),
                        "predicted_xy_m": predicted_xy,
                    })
                else:
                    ep.get_logger().info(
                        f"=== Mode B revert — back to default wrench (fz={ep.args.fz}, lat=0, T=0) "
                        f"+ nominal gain/damping ({ep.args.gain}/{ep.args.damping})"
                    )
                # Default wrench AND default gain/damping (nominal)
                ep._start_force_mode(sel_vec)
            elif action == "abort":
                ep.get_logger().error(f"Mode B abort: {payload}")
                ep.meta.set_outcome(s.OUTCOME_ABORT, f"correction_failed:{payload}")
                ep.meta.set_optional("termination_fire_debug", {
                    "correction_count": correction_eval.correction_count,
                    "abort_reason": str(payload),
                })
                return s.PHASE_ABORT

        # Timeout (legacy fallback or hard ceiling)
        if time.time() >= active_deadline:
            ep.get_logger().info(f"Timeout {ep.args.timeout}s reached — exiting as 'timeout'")
            ep.meta.set_outcome(s.OUTCOME_TIMEOUT, "timeout_reached")
            return s.PHASE_DONE

        # Tick logging
        if time.time() >= next_tick:
            ep._log_sample()
            next_tick += period


def run_done(ep: CompliantInsertEpisode) -> None:
    """DONE/ABORT exit (WRAP-08 + WRAP-09): stop force mode, switch back, safe-height first, then home.

    Idempotent: each step is wrapped in try/except so partial failure still proceeds.
    """
    ep.phase = s.PHASE_DONE if ep.outcome_signal != "abort" else s.PHASE_ABORT
    ep.get_logger().info(f"=== {ep.phase}: exit path (stop force mode -> switch -> safe height -> home) ===")
    ep._log_sample()

    # 1. Stop force mode (idempotent)
    try:
        ep._stop_force_mode()
    except Exception as e:
        ep.get_logger().error(f"stop_force_mode error: {e}")

    # 2. Switch back to position controller (idempotent — even if already switched).
    # CRITICAL: verify the switch actually took effect via _await_controller_active.
    # Previously this return value was ignored, leaving force_mode_controller active
    # while the wrapper plowed ahead to issue position-controlled trajectories that
    # were silently rejected — robot stayed where it was, operator got no warning.
    controller_ok = False
    try:
        switched = _switch_controllers(
            activate=[POS_CTRL], deactivate=[FORCE_CTRL], logger=ep.get_logger())
        if switched:
            controller_ok = _await_controller_active(
                POS_CTRL, timeout_s=5.0, logger=ep.get_logger())
        if not controller_ok:
            ep.get_logger().error(
                f"=== {ep.phase}: switch to {POS_CTRL} did NOT take effect — "
                f"SKIPPING safe-height/home moves. Robot is at last commanded pose; "
                f"{FORCE_CTRL} may still be active. Manually clear with:\n"
                f"  ros2 service call /force_mode_controller/stop_force_mode std_srvs/srv/Trigger\n"
                f"  ros2 control switch_controllers --activate {POS_CTRL} --deactivate {FORCE_CTRL}"
            )
    except Exception as e:
        ep.get_logger().error(f"switch back to position controller error: {e}")

    # 3. move_to_safe_height FIRST (avoids straight-line through inserted base).
    # Gated on controller_ok — useless otherwise, trajectory would be silently rejected.
    # Use python -m module mode (NOT script-path) so the primitive's
    # `from primitives.shared.config import ...` import resolves.
    # Also gated on --no-post-insert-move: when the caller sequences the
    # post-insert release+retract itself (e.g. translate_object --insertion-type
    # compliant), running safe_height here would yank the still-gripped peg
    # straight back out of the slot before the caller's release fires.
    no_post_move = getattr(ep.args, "no_post_insert_move", False)
    if no_post_move:
        ep.get_logger().info(
            "Subprocess: move_to_safe_height SKIPPED (--no-post-insert-move). "
            "EE left at seat pose; caller is responsible for release + retract."
        )
    elif controller_ok:
        try:
            ep.get_logger().info("Subprocess: move_to_safe_height")
            res = subprocess.run(
                [sys.executable, "-m", "primitives.move_to_safe_height", "--mode", "real"],
                capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
            )
            if res.returncode != 0:
                ep.get_logger().error(
                    f"move_to_safe_height rc={res.returncode}; "
                    f"stderr tail: {res.stderr.splitlines()[-3:] if res.stderr else '(empty)'}"
                )
        except Exception as e:
            ep.get_logger().error(f"move_to_safe_height subprocess error: {e}")

    # 4. move_home (optional — skipped during tuning to save ~5s).
    # Gated on controller_ok like safe_height; same module-mode fix.
    if no_post_move:
        ep.get_logger().info("Subprocess: move_home SKIPPED (--no-post-insert-move)")
    elif getattr(ep.args, "skip_home_on_done", False):
        ep.get_logger().info("Subprocess: move_home SKIPPED (--skip-home-on-done)")
    elif controller_ok:
        try:
            ep.get_logger().info("Subprocess: move_home")
            res = subprocess.run(
                [sys.executable, "-m", "primitives.move_home", "--joint-space",
                 "--mode", "real"],
                capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
            )
            if res.returncode != 0:
                ep.get_logger().error(
                    f"move_home rc={res.returncode}; "
                    f"stderr tail: {res.stderr.splitlines()[-3:] if res.stderr else '(empty)'}"
                )
        except Exception as e:
            ep.get_logger().error(f"move_home subprocess error: {e}")

    ep.hands_off = 0


def run_abort(ep: CompliantInsertEpisode) -> None:
    """ABORT exit — same idempotent cleanup as DONE."""
    ep.phase = s.PHASE_ABORT
    ep.get_logger().warn("=== ABORT: same cleanup as DONE ===")
    if ep.csv_writer:
        ep._log_sample()
    # Reuse DONE's cleanup (it handles already-stopped force mode + already-switched controller)
    ep.outcome_signal = ep.outcome_signal or "abort"
    run_done(ep)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = _parse_args(argv)

    rclpy.init()
    ep = CompliantInsertEpisode(args)

    # ---- Signal handlers (WRAP-10) ---------------------------------------
    def _sigusr1(signum, frame):
        ep.event_marker_counter += 1
        ep.get_logger().info(f"SIGUSR1: event_marker -> {ep.event_marker_counter}")
        # Guided-mode: SIGUSR1 also marks the hole when FSM is in GUIDED state.
        # FSM.mark_hole() is a no-op outside GUIDED, so safe to always call.
        fsm = getattr(ep, 'fsm', None)
        if fsm is not None and hasattr(fsm, 'mark_hole'):
            fsm.mark_hole()

    def _sigusr2(signum, frame):
        ep.get_logger().info("SIGUSR2: mid-episode re-zero requested")
        ep._zero_request = True

    def _sigterm(signum, frame):
        ep.get_logger().warn("SIGTERM: ending as success")
        ep.outcome_signal = "success"

    def _sigabrt(signum, frame):
        ep.get_logger().warn("SIGABRT: ending as abort")
        ep.outcome_signal = "abort"

    signal.signal(signal.SIGUSR1, _sigusr1)
    signal.signal(signal.SIGUSR2, _sigusr2)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)   # Ctrl-C also = success exit
    signal.signal(signal.SIGABRT, _sigabrt)

    # ---- Episode setup ---------------------------------------------------
    ep.start_t = time.time()
    ep.episode_start_iso = iso_local_now()
    ts = filename_timestamp()
    ep.csv_path = f"{LOG_DIR}/insert_{args.object_name}_{ts}.csv"
    ep.meta_path = ep.csv_path[:-4] + ".meta.json"
    ep.csv_writer = CSVWriter(ep.csv_path)
    # Schema bump v1.2 — open raw sidecar files (joints/wrench/cmd_wrench/fm_events).
    try:
        ep._open_raw_sidecars(ep.csv_path)
        ep.get_logger().info(f"Schema v1.2 sidecars open: {ep.csv_path[:-4]}.{{joints,wrench,cmd_wrench,fm_events}}_raw.csv")
    except Exception as e:
        ep.get_logger().warn(f"Could not open v1.2 sidecars: {e} — continuing with main CSV only")

    # Identity into meta
    wrapper_version = args.wrapper_version or f"compliant_insert.py@{_git_sha_short()}"
    ep.meta.set_identity(
        object_name=args.object_name, base=args.base_name, grasp_id=args.grasp_id,
        wrapper_version=wrapper_version,
    )
    ep.meta.set_start(ep.episode_start_iso)

    # Calibration provenance (best-effort: if user passed --cal-yaml, record path)
    if args.cal_yaml:
        ep.meta.set_foundational_calibration({"yaml_path": args.cal_yaml,
                                              "note": "loaded from --cal-yaml; mass/cog/age not parsed by wrapper"})
    else:
        ep.meta.set_foundational_calibration({"note": "no --cal-yaml provided; foundational calibration provenance unknown"})

    # ---- FSM loop --------------------------------------------------------
    final_phase = s.PHASE_ABORT
    try:
        next_phase = run_pre(ep)
        if next_phase == s.PHASE_HOVER:
            next_phase = run_hover(ep)
        if next_phase == s.PHASE_ZERO:
            next_phase = run_zero(ep)
        if next_phase == s.PHASE_ACTIVE:
            next_phase = run_active(ep)
        final_phase = next_phase
    except KeyboardInterrupt:
        ep.outcome_signal = "success"
        final_phase = s.PHASE_DONE
    except Exception as e:
        ep.get_logger().error(f"Wrapper crashed: {type(e).__name__}: {e}")
        if ep.meta.to_dict().get("outcome") is None:
            ep.meta.set_outcome(s.OUTCOME_ABORT, f"wrapper_exception:{type(e).__name__}:{str(e)[:120]}")
        final_phase = s.PHASE_ABORT

    # ---- Cleanup ---------------------------------------------------------
    try:
        if final_phase == s.PHASE_DONE:
            run_done(ep)
        elif final_phase == s.PHASE_ABORT:
            run_abort(ep)
    except Exception as e:
        ep.get_logger().error(f"Cleanup phase crashed: {e}")

    # ---- Hands-off window finalize ---------------------------------------
    if hasattr(ep, "_hands_off_start_iso"):
        end_iso = iso_local_now()
        try:
            from datetime import datetime as _dt
            dur = (_dt.fromisoformat(end_iso) - _dt.fromisoformat(ep._hands_off_start_iso)).total_seconds()
        except Exception:
            dur = 0.0
        ep.meta.set_hands_off_window(
            start_iso=ep._hands_off_start_iso, end_iso=end_iso,
            duration_s=dur, trigger=getattr(ep, "_hands_off_trigger", "unknown"),
        )

    # ---- End-of-episode timing + user notes ------------------------------
    end_iso = iso_local_now()
    ep.meta.set_end(end_iso, time.time() - ep.start_t)
    if not args.no_prompt_notes:
        try:
            notes = input("user_notes (free text, Enter to skip): ").strip()
        except EOFError:
            notes = ""
        ep.meta.set_user_notes(notes)

    # If outcome was never set (e.g., FSM completed PRE but bailed before any run_*), set abort
    if ep.meta.to_dict().get("outcome") is None:
        ep.meta.set_outcome(s.OUTCOME_ABORT, "outcome_never_set")

    # ---- Write artifacts -------------------------------------------------
    try:
        ep.csv_writer.close()
    except Exception:
        pass
    try:
        ep._close_raw_sidecars()
    except Exception:
        pass
    try:
        ep.meta.write(ep.meta_path)
    except Exception as e:
        print(f"[WARN] meta JSON write failed: {e}", file=sys.stderr)

    # ---- ROS shutdown ----------------------------------------------------
    try:
        ep.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    # ---- Output JSON for parent (matches host primitive convention) ------
    result = {
        "result": "success" if ep.meta.to_dict().get("outcome") == "success" else "failure",
        "outcome": ep.meta.to_dict().get("outcome"),
        "outcome_reason": ep.meta.to_dict().get("outcome_reason"),
        "csv_path": ep.csv_path,
        "meta_path": ep.meta_path,
        "samples_logged": ep.csv_writer.row_count if ep.csv_writer else 0,
    }
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")
    sys.exit(0 if result["result"] == "success" else 1)


if __name__ == "__main__":
    main()
