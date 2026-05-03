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
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from ur_msgs.srv import SetForceMode

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
DEFAULT_DRIFT_WINDOW_S = 1.0     # WRAP-04: +1 s post-zero drift sample

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
    # Calibration provenance (passed through to meta JSON)
    p.add_argument("--cal-yaml", default=None,
                   help="path to foundational calibration YAML (recorded in meta JSON)")
    # Smoke test integration
    p.add_argument("--smoke-script", default=str(_PKG_ROOT / "shared" / "ft_smoke_test.py"),
                   help="path to ft_smoke_test.py (default: ../shared/ft_smoke_test.py)")
    # Misc
    p.add_argument("--no-prompt-notes", action="store_true",
                   help="do not prompt for user_notes at end (useful for unattended dispatcher runs)")
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


def _list_active_controllers() -> set:
    """Poll `ros2 control list_controllers` and return set of active controller names.

    WRAP-07: switch_controller RPC success != actual controller transition complete.
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
    for line in result.stdout.splitlines():
        # Format: "<name> <type> <state>" — state is "active" or "inactive"
        parts = line.split()
        if len(parts) >= 3 and parts[-1] == "active":
            active.add(parts[0])
    return active


def _await_controller_active(name: str, *, timeout_s: float = 2.0,
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
        self.in_force_mode: bool = False

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

        self.start_fm = self.create_client(SetForceMode, "/force_mode_controller/start_force_mode")
        self.stop_fm = self.create_client(Trigger, "/force_mode_controller/stop_force_mode")

    # ------------------- callbacks ----------------------------------------
    def _tcp_cb(self, msg): self.tcp = msg
    def _wrench_cb(self, msg): self.wrench = msg
    def _gripper_cb(self, msg):
        try:
            self.gripper_width_v = float(msg.data)
        except Exception:
            pass

    # ------------------- topic-ready gate ---------------------------------
    def wait_for_topics(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.tcp is not None and self.wrench is not None:
                return True
        return False

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
        f = self.wrench.wrench.force
        tq = self.wrench.wrench.torque
        p = self.tcp.pose.position
        q = self.tcp.pose.orientation

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
            "fx": f.x, "fy": f.y, "fz": f.z,
            "tx": tq.x, "ty": tq.y, "tz": tq.z,
            "gripper_width": self.gripper_width_v,
            "commanded_fz": self.commanded_fz,
        }
        self.csv_writer.write(row)
        self.zero_event_pending = 0   # one-shot flag

    # ------------------- zero F/T (CLI service call) ----------------------
    def _zero_ftsensor_call(self) -> bool:
        result = subprocess.run(
            ["ros2", "service", "call", "/io_and_status_controller/zero_ftsensor",
             "std_srvs/srv/Trigger"],
            capture_output=True, text=True, timeout=10,
        )
        ok = result.returncode == 0 and "success=True" in result.stdout
        if not ok:
            self.get_logger().error(
                f"zero_ftsensor failed rc={result.returncode} out={result.stdout.strip()}"
            )
        return ok

    def _sample_bias(self, settle_s: float = 0.5) -> dict | None:
        """Drop cached wrench, settle, return the next fresh sample as a bias dict."""
        self.wrench = None
        deadline = time.time() + settle_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.wrench is None:
            self.get_logger().warn("No /wrench sample arrived during settle window")
            return None
        f = self.wrench.wrench.force
        t = self.wrench.wrench.torque
        return {"Fx": f.x, "Fy": f.y, "Fz": f.z, "Tx": t.x, "Ty": t.y, "Tz": t.z}

    # ------------------- start/stop force mode ----------------------------
    def _start_force_mode(self, sel_vec: list[bool]) -> bool:
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
        req.task_frame.header.frame_id = "base_link"
        req.task_frame.pose.orientation.w = 1.0   # identity

        req.selection_vector_x = sel_vec[0]
        req.selection_vector_y = sel_vec[1]
        req.selection_vector_z = sel_vec[2]
        req.selection_vector_rx = sel_vec[3]
        req.selection_vector_ry = sel_vec[4]
        req.selection_vector_rz = sel_vec[5]

        req.wrench.force.x = 0.0
        req.wrench.force.y = 0.0
        req.wrench.force.z = -fz   # negative = pushing down
        req.wrench.torque.x = 0.0
        req.wrench.torque.y = 0.0
        req.wrench.torque.z = 0.0

        req.type = 2   # NO_TRANSFORM (task_frame = base_link as-is)

        req.speed_limits.linear.x = float(self.args.lin_speed)
        req.speed_limits.linear.y = float(self.args.lin_speed)
        req.speed_limits.linear.z = float(self.args.lin_speed)
        req.speed_limits.angular.x = float(self.args.ang_speed)
        req.speed_limits.angular.y = float(self.args.ang_speed)
        req.speed_limits.angular.z = float(self.args.ang_speed)

        req.gain_scaling = float(self.args.gain)
        req.damping_factor = float(self.args.damping)

        future = self.start_fm.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error("start_force_mode call failed")
            return False
        self.in_force_mode = True
        self.commanded_fz = -fz
        self.get_logger().info(
            f"Force mode active: Fz={-fz} N, sel={sel_vec}, gain={self.args.gain} damp={self.args.damping}"
        )
        return True

    def _stop_force_mode(self) -> None:
        if not self.in_force_mode:
            return
        if self.stop_fm.wait_for_service(timeout_sec=2.0):
            future = self.stop_fm.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            r = future.result()
            if r is None or not r.success:
                self.get_logger().warn(f"stop_force_mode reported: {getattr(r, 'message', 'no response')}")
            else:
                self.get_logger().info("Force mode stopped")
        self.in_force_mode = False
        self.commanded_fz = 0.0

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
    ep.meta.set_optional("hover_pose_world", {"xyz_m": list(ep.target_xyz), "quat_xyzw": list(ep.target_quat)})

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
    if not _await_controller_active(FORCE_CTRL, timeout_s=2.0, logger=ep.get_logger()):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "force_mode_controller_did_not_activate")
        return s.PHASE_ABORT

    # 1.0 s settle after controller switch (force spikes contaminate zero otherwise)
    t_settle_end = time.time() + 1.0
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

    # 0.5 s settle, sample bias
    bias = ep._sample_bias(settle_s=0.5)
    if bias is None:
        ep.meta.set_outcome(s.OUTCOME_ABORT, "no_post_zero_wrench_sample")
        return s.PHASE_ABORT
    ep.meta.set_post_zero_bias(bias)

    max_axis_f = max(abs(bias["Fx"]), abs(bias["Fy"]), abs(bias["Fz"]))
    if max_axis_f > ep.args.bias_warn_n:
        ep.get_logger().warn(
            f"Post-zero residual force {max_axis_f:.2f} N > {ep.args.bias_warn_n} N (warning, not abort)"
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
    """ACTIVE phase (WRAP-06 + WRAP-07). Loop until SIGTERM / SIGABRT / timeout."""
    ep.phase = s.PHASE_ACTIVE
    ep.get_logger().info("=== ACTIVE: enter force mode, log telemetry until exit ===")

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
            return s.PHASE_DONE
        if ep.outcome_signal == "abort":
            ep.meta.set_outcome(s.OUTCOME_ABORT, "operator_sigabrt")
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

        # Timeout
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

    # 2. Switch back to position controller (idempotent — even if already switched)
    try:
        _switch_controllers(activate=[POS_CTRL], deactivate=[FORCE_CTRL], logger=ep.get_logger())
        _await_controller_active(POS_CTRL, timeout_s=2.0, logger=ep.get_logger())
    except Exception as e:
        ep.get_logger().error(f"switch back to position controller error: {e}")

    # 3. move_to_safe_height FIRST (avoids straight-line through inserted base)
    try:
        ep.get_logger().info("Subprocess: move_to_safe_height")
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / "primitives" / "move_to_safe_height.py")],
            capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
        )
    except Exception as e:
        ep.get_logger().error(f"move_to_safe_height subprocess error: {e}")

    # 4. move_home
    try:
        ep.get_logger().info("Subprocess: move_home")
        subprocess.run(
            [sys.executable, str(_REPO_ROOT / "primitives" / "move_home.py")],
            capture_output=True, text=True, timeout=60, cwd=str(_REPO_ROOT),
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
