#!/usr/bin/env python3
# Reference: motion-isolation diagnostic — test whether the spiral xy command pattern
# induces TCP yaw drift (which → peg arc motion through grasp lever) WITHOUT actually
# trying to insert. Peg held in air at hover height; commanded only lateral force.
#
# Per user's question (2026-05-05): "spiral search isn't wrong but works differently
# for prismatic peg-in-hole where TCP orientation while moving in xy matters. I want
# a test script for this return without the insert."
#
# What it does:
#   1. Move to hover above target (re-uses _run_hover for canonical setup)
#   2. Switch to force_mode_controller
#   3. Command lateral force pattern (configurable: spiral / fixed-direction / circle)
#      with cmd_fz = 0 — no axial push, no contact, no insertion attempt
#   4. Log TCP pose + sensed wrench at 100 Hz, schema v1.2 sidecars active
#   5. After max-duration-s, switch back + return to safe height
#
# Output:
#   - Episode files in compliant_insertion_studio/logs/motion_test_*.{csv, joints_raw.csv, ...}
#   - meta.json tagged with assist_level=motion_isolation_<pattern>
#   - Use compare.html to overlay against a real insert to see what motion alone produces
#
# IMPORTANT: peg should be HELD IN AIR (no contact with anything). Operator places it
# above the workspace, then runs this. Robot moves in xy at hover height; logs TCP yaw
# drift purely from the cmd_F_lat pattern + admittance.

import argparse
import json
import math
import os
import signal
import sys
import time

import rclpy
from std_srvs.srv import Trigger

from compliant_insertion_studio.wrapper.compliant_insert import (
    CompliantInsertEpisode, LOG_DIR, POS_CTRL, FORCE_CTRL,
    _switch_controllers, _await_controller_active, _git_sha_short,
)
from compliant_insertion_studio.wrapper import schema_v1 as s
from compliant_insertion_studio.wrapper.telemetry import (
    iso_local_now, filename_timestamp, CSVWriter,
)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--object-name", required=True)
    p.add_argument("--base-name", required=True)
    p.add_argument("--grasp-id", type=int, required=True)
    p.add_argument("--current-object-orientation", nargs=4, type=float, required=True)
    p.add_argument("--use-default-base-position", action="store_true")
    p.add_argument("--base-world-pose", nargs=7, type=float, default=None)
    p.add_argument("--pattern", choices=["spiral", "circle", "fixed_x", "fixed_y", "static"],
                   default="spiral", help="motion pattern to apply (no Fz push)")
    p.add_argument("--max-duration-s", type=float, default=15.0)
    p.add_argument("--F-lat-N", type=float, default=3.0,
                   help="lateral force magnitude (≤6N per hard rules)")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.7)
    p.add_argument("--lin-speed", type=float, default=0.54)
    p.add_argument("--ang-speed", type=float, default=0.5)
    p.add_argument("--mode", choices=["sim", "real"], default="real")
    # Args needed by CompliantInsertEpisode (not all used by this test)
    for arg, default in [("config", None), ("wrapper_version", None), ("smoke_script", None),
                         ("cal_yaml", None), ("step_back", "skip"), ("auto_step_back_seconds", 5.0),
                         ("rate_hz", 100.0), ("bias_warn_n", 2.0), ("selection", "1,1,1,1,1,1"),
                         ("fz", 0.0), ("hole_xy_prior", None)]:
        p.add_argument(f"--{arg.replace('_', '-')}", default=default,
                       type=type(default) if default is not None and not isinstance(default, bool) else str)
    p.add_argument("--override-fz-cap", action="store_true", default=True)
    p.add_argument("--no-prompt-notes", action="store_true", default=True)
    p.add_argument("--skip-smoke", action="store_true", default=True)
    p.add_argument("--skip-hover", action="store_true", default=False)
    p.add_argument("--user-notes", default="")
    p.add_argument("--timeout", type=float, default=60.0)
    return p


def compute_lateral_cmd(t_in_phase: float, pattern: str, F: float):
    """Returns (Fx, Fy) lateral force command in base_link frame."""
    if pattern == "static":
        return (0.0, 0.0)
    if pattern == "fixed_x":
        return (F, 0.0)
    if pattern == "fixed_y":
        return (0.0, F)
    if pattern == "circle":
        # 1 Hz circle of magnitude F
        omega = 2 * math.pi * 0.5  # 0.5 Hz to be slow
        return (F * math.cos(omega * t_in_phase), F * math.sin(omega * t_in_phase))
    if pattern == "spiral":
        # Archimedean — radius grows with t, theta grows linearly. Mimics _find_hole_wrench's spiral.
        v = 0.0015  # m/s (matches default spiral_v_m_s in defaults.yaml)
        pitch = 0.0006  # m
        kp = 1500.0  # N/m
        theta = (2 * math.pi / max(pitch, 1e-6)) * v * t_in_phase
        radius = (pitch / (2 * math.pi)) * theta
        target_x = radius * math.cos(theta)
        target_y = radius * math.sin(theta)
        # Spiral PD — push toward setpoint relative to t=0 origin
        Fx = -kp * (-target_x)  # err = target - pos; assume pos=0 (we're not moving against rim)
        Fy = -kp * (-target_y)
        # Cap magnitude
        mag = math.hypot(Fx, Fy)
        if mag > F:
            Fx *= F / mag; Fy *= F / mag
        return (Fx, Fy)
    return (0.0, 0.0)


def main():
    args = build_parser().parse_args()

    rclpy.init()
    try:
        ep = CompliantInsertEpisode(args)
        ep._stop_requested = False

        def _on_term(signum, frame):
            ep._stop_requested = True
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        # Episode setup
        ep.start_t = time.time()
        ep.episode_start_iso = iso_local_now()
        ts = filename_timestamp()
        ep.csv_path = f"{LOG_DIR}/motion_test_{args.object_name}_{args.pattern}_{ts}.csv"
        ep.meta_path = ep.csv_path[:-4] + ".meta.json"
        ep.csv_writer = CSVWriter(ep.csv_path)
        try:
            ep._open_raw_sidecars(ep.csv_path)
        except Exception as e:
            ep.get_logger().warn(f"sidecars open failed: {e}")

        ep.meta.set_identity(object_name=args.object_name, base=args.base_name,
                              grasp_id=args.grasp_id,
                              wrapper_version=f"motion_isolation_test.py@{_git_sha_short()}")
        ep.meta.set_start(ep.episode_start_iso)
        ep.meta.set_optional("assist_level", f"motion_isolation_{args.pattern}")
        ep.meta.set_user_notes(f"motion-isolation diagnostic: pattern={args.pattern} "
                                f"F_lat={args.F_lat_N}N max_dur={args.max_duration_s}s "
                                f"NO Fz push, NO insertion attempt. {args.user_notes}")

        # HOVER (subprocess, same as wrapper)
        ep.get_logger().info(f"=== HOVER: navigate to per-object hover pose ===")
        import subprocess
        hover_cmd = [sys.executable, "-m", "compliant_insertion_studio.wrapper._run_hover",
                     "--object-name", args.object_name, "--base-name", args.base_name,
                     "--grasp-id", str(args.grasp_id),
                     "--current-object-orientation",
                     *[str(v) for v in args.current_object_orientation]]
        if args.use_default_base_position:
            hover_cmd.append("--use-default-base-position")
        elif args.base_world_pose:
            hover_cmd += ["--final-base-pos", *[str(v) for v in args.base_world_pose[:3]]]
            hover_cmd += ["--final-base-orientation", *[str(v) for v in args.base_world_pose[3:]]]
        rc = subprocess.run(hover_cmd, timeout=30).returncode
        if rc != 0:
            ep.get_logger().error(f"HOVER failed rc={rc}")
            ep.meta.set_outcome(s.OUTCOME_ABORT, "hover_failed")
            sys.exit(_finalize(ep, 1))

        # Switch to force_mode_controller
        if not _switch_controllers(activate=[FORCE_CTRL], deactivate=[POS_CTRL], logger=ep.get_logger()):
            ep.meta.set_outcome(s.OUTCOME_ABORT, "switch_to_force_mode_failed"); sys.exit(_finalize(ep, 1))
        if not _await_controller_active(FORCE_CTRL, timeout_s=5.0, logger=ep.get_logger()):
            ep.meta.set_outcome(s.OUTCOME_ABORT, "force_mode_did_not_activate"); sys.exit(_finalize(ep, 1))

        # Zero F/T (gentle)
        try:
            zero_cli = ep.create_client(Trigger, "/force_torque_sensor_broadcaster/zero_ftsensor")
            if zero_cli.wait_for_service(timeout_sec=2.0):
                fut = zero_cli.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(ep, fut, timeout_sec=2.0)
            ep.destroy_client(zero_cli)
        except Exception as e:
            ep.get_logger().warn(f"F/T zero failed: {e}")

        # Initial start_force_mode call with zero wrench
        ep.phase = s.PHASE_ACTIVE
        if not ep._start_force_mode([True, True, True, True, True, True],
                                    override_wrench_baselink=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
            ep.meta.set_outcome(s.OUTCOME_ABORT, "start_force_mode_failed"); sys.exit(_finalize(ep, 1))

        ep.get_logger().info(f">>> MOTION TEST: pattern={args.pattern} F_lat={args.F_lat_N}N — peg should be in air; do NOT touch")
        phase_start = time.time()
        deadline = phase_start + args.max_duration_s
        last_cmd = (None, None)
        last_log = phase_start

        while time.time() < deadline and not ep._stop_requested:
            rclpy.spin_once(ep, timeout_sec=0.005)
            t_now = time.time()
            t_in_phase = t_now - phase_start

            # Log telemetry tick
            ep._log_sample()

            # Compute and issue lateral command
            Fx, Fy = compute_lateral_cmd(t_in_phase, args.pattern, args.F_lat_N)
            # Re-issue every 50ms; skip if barely changed
            if (last_cmd[0] is None
                or abs(Fx - last_cmd[0]) > 0.05 or abs(Fy - last_cmd[1]) > 0.05
                or (t_now - last_log) > 0.5):
                ep._start_force_mode([True, True, True, True, True, True],
                                     override_wrench_baselink=[Fx, Fy, 0.0, 0.0, 0.0, 0.0],
                                     gain_override=args.gain, damping_override=args.damping,
                                     quiet=True)
                last_cmd = (Fx, Fy)
                last_log = t_now

            time.sleep(0.01)

        # Cleanup
        ep._stop_force_mode()
        if not _switch_controllers(activate=[POS_CTRL], deactivate=[FORCE_CTRL], logger=ep.get_logger()):
            ep.get_logger().warn("could not switch back to POS_CTRL")
        else:
            _await_controller_active(POS_CTRL, timeout_s=5.0, logger=ep.get_logger())

        # Return to safe height
        try:
            subprocess.run([sys.executable, "-m", "primitives.move_to_safe_height",
                            "--mode", args.mode], timeout=15)
        except Exception as e:
            ep.get_logger().warn(f"safe height failed: {e}")

        ep.meta.set_outcome(s.OUTCOME_SUCCESS,
                            "operator_sigterm" if ep._stop_requested else "max_duration_reached")
        sys.exit(_finalize(ep, 0))
    finally:
        try: rclpy.shutdown()
        except Exception: pass


def _finalize(ep, rc):
    try:
        end_iso = iso_local_now()
        duration = (time.time() - ep.start_t) if getattr(ep, "start_t", None) else 0.0
        ep.meta.set_end(end_iso, duration_s=duration)
    except Exception: pass
    if ep.meta.to_dict().get("outcome") is None:
        ep.meta.set_outcome(s.OUTCOME_ABORT, "outcome_never_set")
    try: ep.csv_writer.close()
    except Exception: pass
    try: ep._close_raw_sidecars()
    except Exception: pass
    try:
        with open(ep.meta_path, "w") as fh:
            json.dump(ep.meta.to_dict(), fh, indent=2)
    except Exception as e:
        print(f"meta write failed: {e}", file=sys.stderr)
    md = ep.meta.to_dict()
    print("__RESULT_JSON__")
    print(json.dumps({
        "result": "success" if md.get("outcome") == s.OUTCOME_SUCCESS else "failure",
        "outcome": md.get("outcome"), "outcome_reason": md.get("outcome_reason"),
        "csv_path": ep.csv_path, "meta_path": ep.meta_path,
        "samples_logged": ep.csv_writer.row_count if ep.csv_writer else 0,
        "assist_level": md.get("assist_level"),
    }))
    print("__END_RESULT_JSON__")
    return rc


if __name__ == "__main__":
    main()
