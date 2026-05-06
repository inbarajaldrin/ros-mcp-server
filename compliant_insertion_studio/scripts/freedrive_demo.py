#!/usr/bin/env python3
# Reference: kinesthetic-teaching capture for the FMB1 insert primitive.
#
# WHY: the existing wrapper uses force_mode_controller with selection_vector all-True
# and cmd_lateral=0. That mode actively *resists* operator hand pushes (force regulation
# around 0). All "operator-assisted" data prior to this date was contaminated by that
# resistance. For clean kinesthetic data we use freedrive_mode_controller — robot
# becomes fully passive, operator drives the gripper freely.
#
# Flow (reuses CompliantInsertEpisode for subscriptions / CSV / v1.2 sidecars):
#     PRE -> HOVER -> ZERO -> KINESTHETIC -> DONE
# Differences from compliant_insert.py:
#   - KINESTHETIC replaces ACTIVE: switch to freedrive_mode_controller + enable_freedrive,
#     log telemetry until operator SIGTERMs, disable, switch back.
#   - No FSM, no termination predicate. Operator decides when seated and SIGTERMs.
#   - Meta JSON outcome=success / outcome_reason=operator_sigterm by default.
#   - assist_level=freedrive_kinesthetic so analysis can distinguish from prior demos.
#
# Usage:
#   python3 -m compliant_insertion_studio.scripts.freedrive_demo \
#     --object-name u_orange --base-name base1 --grasp-id 1 \
#     --current-object-orientation 0.7081 0.002 0.001 -0.7061 \
#     --use-default-base-position
#
# Press Ctrl-C / SIGTERM when peg is seated → cleanup runs → CSV + 4 sidecars persisted.

import argparse
import os
import signal
import sys
import time

import rclpy
from std_msgs.msg import Bool

from compliant_insertion_studio.wrapper.compliant_insert import (
    CompliantInsertEpisode,
    LOG_DIR, POS_CTRL,
    _switch_controllers, _await_controller_active, _git_sha_short,
)
from compliant_insertion_studio.wrapper import schema_v1 as s
from compliant_insertion_studio.wrapper.telemetry import iso_local_now, filename_timestamp

FREEDRIVE_CTRL = "freedrive_mode_controller"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kinesthetic freedrive demo for FMB1 insert.")
    # Identity
    p.add_argument("--object-name", required=True)
    p.add_argument("--base-name", required=True)
    p.add_argument("--grasp-id", type=int, required=True)
    p.add_argument("--mode", choices=["sim", "real"], default="real")
    # Pose chain inputs (same as wrapper)
    p.add_argument("--current-object-orientation", nargs=4, type=float, metavar=("QX","QY","QZ","QW"),
                   required=True, help="Held-object orientation in world (xyzw)")
    p.add_argument("--use-default-base-position", action="store_true",
                   help="Use DEFAULT_BASE_POSITION for hover target")
    p.add_argument("--base-world-pose", nargs=7, type=float, default=None,
                   metavar=("X","Y","Z","QX","QY","QZ","QW"))
    p.add_argument("--hole-xy-prior", nargs=2, type=float, default=None,
                   metavar=("X","Y"), help="Override target xy from prior demo")
    # Hover knobs (forwarded to _run_hover via wrapper)
    p.add_argument("--hover-z-offset-m", type=float, default=0.030,
                   help="Hover this height above predicted seat z")
    # Demo knobs
    p.add_argument("--max-duration-s", type=float, default=120.0,
                   help="Hard cap on KINESTHETIC phase before auto-cleanup")
    p.add_argument("--no-prompt-notes", action="store_true")
    p.add_argument("--user-notes", default=None,
                   help="Pre-fill user_notes (e.g. 'demo 3 — hand-driven')")
    # Echo args expected by wrapper subprocesses
    p.add_argument("--lin-speed", type=float, default=0.54)
    p.add_argument("--ang-speed", type=float, default=0.5)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.7)
    p.add_argument("--fz", type=float, default=9.0)
    p.add_argument("--override-fz-cap", action="store_true", default=True)
    p.add_argument("--step-back", default="auto")
    p.add_argument("--auto-step-back-seconds", type=float, default=5.0)
    p.add_argument("--skip-home-on-done", action="store_true", default=True)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--config", default=None)
    p.add_argument("--wrapper-version", default=None)
    # Args the wrapper's run_pre() / run_hover() reach into via self.args.* — we
    # don't expose them on the demo CLI, but we need them present in the Namespace.
    p.add_argument("--skip-smoke", action="store_true", default=False)
    p.add_argument("--smoke-script", default=None)
    p.add_argument("--skip-hover", action="store_true", default=False)
    p.add_argument("--selection", default="1,1,1,1,1,1")
    p.add_argument("--rate-hz", type=float, default=100.0)
    p.add_argument("--bias-warn-n", type=float, default=2.0)
    p.add_argument("--cal-yaml", default=None)
    return p


# ---- KINESTHETIC phase ------------------------------------------------------
def run_kinesthetic(ep: CompliantInsertEpisode, max_duration_s: float) -> str:
    """Kinesthetic capture: operator drives the gripper, we log everything.

    Switch from POS_CTRL → FREEDRIVE_CTRL, publish enable_freedrive_mode=True,
    log @ 100 Hz until SIGTERM/SIGABRT/timeout, then disable + switch back.
    """
    # Use ACTIVE phase enum (schema-valid); the assist_level=freedrive_kinesthetic
    # in meta JSON distinguishes this from force-mode ACTIVE.
    ep.phase = s.PHASE_ACTIVE
    ep.get_logger().info("=== KINESTHETIC (phase=ACTIVE, assist=freedrive): switching to freedrive controller, enabling freedrive, logging until operator SIGTERM ===")

    # Switch controllers
    if not _switch_controllers(activate=[FREEDRIVE_CTRL], deactivate=[POS_CTRL],
                               logger=ep.get_logger()):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "switch_to_freedrive_controller_failed")
        return s.PHASE_ABORT
    if not _await_controller_active(FREEDRIVE_CTRL, timeout_s=5.0, logger=ep.get_logger()):
        ep.meta.set_outcome(s.OUTCOME_ABORT, "freedrive_controller_did_not_activate")
        return s.PHASE_ABORT

    # Publisher for enable_freedrive_mode
    enable_pub = ep.create_publisher(Bool, "/freedrive_mode_controller/enable_freedrive_mode", 10)
    # Wait briefly for subscribers to attach (URCap side)
    for _ in range(20):
        if enable_pub.get_subscription_count() > 0:
            break
        rclpy.spin_once(ep, timeout_sec=0.05)
        time.sleep(0.05)

    # Track freedrive window start (set_hands_off_window requires both start+end)
    freedrive_start_iso = iso_local_now()
    freedrive_t0 = time.time()

    # Enable freedrive
    enable_pub.publish(Bool(data=True))
    ep.get_logger().info(">>> FREEDRIVE ENABLED — operator can now move the robot freely. SIGTERM (or Ctrl-C) when peg is seated.")

    # Log loop
    t_start = time.time()
    deadline = t_start + max_duration_s
    log_period = 0.01  # 100 Hz tick (same as wrapper)
    next_tick = time.time()
    try:
        while time.time() < deadline:
            rclpy.spin_once(ep, timeout_sec=0.005)
            now = time.time()
            if now >= next_tick:
                ep._log_sample()
                next_tick += log_period
                if next_tick < now:
                    next_tick = now + log_period
            # SIGTERM/SIGABRT handlers below set ep._stop_requested
            if getattr(ep, "_stop_requested", False):
                ep.get_logger().info(">>> Operator stop signal — exiting KINESTHETIC")
                break
    finally:
        # Disable freedrive (idempotent — publish False even if subscriber gone)
        try:
            enable_pub.publish(Bool(data=False))
            time.sleep(0.5)  # give controller time to ack
        except Exception:
            pass
        try:
            ep.destroy_publisher(enable_pub)
        except Exception:
            pass

    # Switch back to position controller
    if not _switch_controllers(activate=[POS_CTRL], deactivate=[FREEDRIVE_CTRL],
                               logger=ep.get_logger()):
        ep.get_logger().warn("Could not switch back to POS_CTRL — manual fix:")
        ep.get_logger().warn(f"  ros2 control switch_controllers --activate {POS_CTRL} --deactivate {FREEDRIVE_CTRL}")
    else:
        _await_controller_active(POS_CTRL, timeout_s=5.0, logger=ep.get_logger())

    # Outcome marking — operator SIGTERM is success; timeout is timeout.
    if getattr(ep, "_stop_requested", False):
        ep.meta.set_outcome(s.OUTCOME_SUCCESS, "operator_sigterm")
    else:
        ep.meta.set_outcome(s.OUTCOME_TIMEOUT, "kinesthetic_timeout_reached")
    ep.meta.set_hands_off_window(
        start_iso=freedrive_start_iso,
        end_iso=iso_local_now(),
        duration_s=time.time() - freedrive_t0,
        trigger="freedrive_kinesthetic",
    )
    return s.PHASE_DONE


# ---- main -------------------------------------------------------------------
def main():
    parser = _build_parser()
    args = parser.parse_args()

    # mock attrs the wrapper expects
    args.config = getattr(args, "config", None)
    args.no_prompt_notes = bool(args.no_prompt_notes)

    rclpy.init()
    try:
        ep = CompliantInsertEpisode(args)
        ep._stop_requested = False

        def _on_term(signum, frame):
            ep._stop_requested = True
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)
        signal.signal(signal.SIGABRT, _on_term)

        # ---- Episode setup (mirror compliant_insert.py main) ----------------
        ep.start_t = time.time()
        ep.episode_start_iso = iso_local_now()
        ts = filename_timestamp()
        ep.csv_path = f"{LOG_DIR}/insert_{args.object_name}_{ts}.csv"
        ep.meta_path = ep.csv_path[:-4] + ".meta.json"
        from compliant_insertion_studio.wrapper.telemetry import CSVWriter
        ep.csv_writer = CSVWriter(ep.csv_path)
        try:
            ep._open_raw_sidecars(ep.csv_path)
            ep.get_logger().info(
                f"Schema v1.2 sidecars open: {ep.csv_path[:-4]}.{{joints,wrench,cmd_wrench,fm_events}}_raw.csv"
            )
        except Exception as e:
            ep.get_logger().warn(f"Could not open v1.2 sidecars: {e}")

        wrapper_version = args.wrapper_version or f"freedrive_demo.py@{_git_sha_short()}"
        ep.meta.set_identity(
            object_name=args.object_name, base=args.base_name, grasp_id=args.grasp_id,
            wrapper_version=wrapper_version,
        )
        ep.meta.set_start(ep.episode_start_iso)
        ep.meta.set_optional("assist_level", "freedrive_kinesthetic")
        if args.user_notes:
            ep.meta.set_user_notes(args.user_notes)

        # ---- HOVER (via _run_hover subprocess — same as wrapper does) ------
        ep.get_logger().info("=== HOVER: navigate to per-object hover pose (subprocess) ===")
        import subprocess
        hover_cmd = [
            sys.executable, "-m", "compliant_insertion_studio.wrapper._run_hover",
            "--object-name", args.object_name,
            "--base-name", args.base_name,
            "--grasp-id", str(args.grasp_id),
            "--current-object-orientation",
            *[str(v) for v in args.current_object_orientation],
        ]
        if args.use_default_base_position:
            hover_cmd.append("--use-default-base-position")
        elif args.base_world_pose:
            # _run_hover takes pose as separate --final-base-pos / --final-base-orientation
            hover_cmd += ["--final-base-pos", *[str(v) for v in args.base_world_pose[:3]]]
            hover_cmd += ["--final-base-orientation", *[str(v) for v in args.base_world_pose[3:]]]
        rc = subprocess.run(hover_cmd, timeout=30).returncode
        if rc != 0:
            ep.get_logger().error(f"HOVER subprocess failed rc={rc}")
            ep.meta.set_outcome(s.OUTCOME_ABORT, "hover_subprocess_failed")
            raise SystemExit(_finalize(ep, rc=1))

        # ---- ZERO (skipped — freedrive doesn't need force-mode + zero) -----
        # We zero the F/T sensor anyway for clean wrench logging.
        try:
            from std_srvs.srv import Trigger
            zero_cli = ep.create_client(Trigger, "/force_torque_sensor_broadcaster/zero_ftsensor")
            if zero_cli.wait_for_service(timeout_sec=2.0):
                fut = zero_cli.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(ep, fut, timeout_sec=2.0)
                ep.get_logger().info("F/T sensor zeroed")
            ep.destroy_client(zero_cli)
        except Exception as e:
            ep.get_logger().warn(f"Could not zero F/T: {e}")

        # ---- KINESTHETIC --------------------------------------------------
        next_phase = run_kinesthetic(ep, args.max_duration_s)

        # ---- DONE / cleanup ----------------------------------------------
        # Move to safe height (skip move_home — leave robot at hover-ish for next attempt)
        from primitives import move_to_safe_height as _mts  # noqa: F401 (verify import)
        import subprocess
        try:
            subprocess.run(
                [sys.executable, "-m", "primitives.move_to_safe_height", "--mode", args.mode],
                check=False, timeout=20,
            )
        except Exception as e:
            ep.get_logger().warn(f"move_to_safe_height failed: {e}")

        rc = _finalize(ep, rc=0)
        sys.exit(rc)
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _finalize(ep: CompliantInsertEpisode, rc: int) -> int:
    """Close CSV + sidecars, write meta JSON, print __RESULT_JSON__ marker."""
    ep.episode_end_iso = iso_local_now()
    duration_s = (time.time() - ep.start_t) if getattr(ep, "start_t", None) else 0.0
    ep.meta.set_end(ep.episode_end_iso, duration_s=duration_s)
    if ep.meta.to_dict().get("outcome") is None:
        ep.meta.set_outcome(s.OUTCOME_ABORT, "outcome_never_set")
    try: ep.csv_writer.close()
    except Exception: pass
    try: ep._close_raw_sidecars()
    except Exception: pass
    try:
        with open(ep.meta_path, "w") as fh:
            import json
            json.dump(ep.meta.to_dict(), fh, indent=2)
    except Exception as e:
        print(f"meta write failed: {e}", file=sys.stderr)

    import json as _json
    md = ep.meta.to_dict()
    print("__RESULT_JSON__")
    print(_json.dumps({
        "result": "success" if md.get("outcome") == s.OUTCOME_SUCCESS else "failure",
        "outcome": md.get("outcome"),
        "outcome_reason": md.get("outcome_reason"),
        "csv_path": ep.csv_path,
        "meta_path": ep.meta_path,
        "samples_logged": ep.csv_writer.row_count if ep.csv_writer else 0,
        "assist_level": md.get("assist_level"),
    }))
    print("__END_RESULT_JSON__")
    return rc


if __name__ == "__main__":
    main()
