"""
Phase 5 iteration helper — delegates the canonical pick→rotate→place→regrasp→rotate
sequence to `run_assembly_step.py --setup-only`, then invokes the
`compliant_insert` wrapper directly with Phase 5's CAD-derived target pose.

Why two scripts: that's the canonical 2-script pattern verified in the
2026-05-03 Phase 3 collection (60 demos):
  Script A: run_assembly_step --setup-only      → prints HELD_QUAT=...
  Script B: compliant_insertion_studio.wrapper.compliant_insert
            --current-object-orientation HELD_QUAT
            --base-world-pose <camera-derived>   ← NEW for Phase 5 v1

The setup script handles the IK-singular-avoidance via place-and-regrasp; the
wrapper handles the lifecycle (PRE/HOVER/ZERO/ACTIVE/DONE) and now exits
autonomously on the universal CAD-derived termination predicate.

Usage:
  # First call (gripper empty, part placed in clear area):
  python3 -m compliant_insertion_studio.scripts.iterate_insert \\
    --object-name u_orange --base-name base1 --grasp-id 1

  # Subsequent calls — part is still held from the previous attempt.
  # `--already-held` makes run_assembly_step skip pick/place/regrasp and
  # just call `rotate_object` (step 12), which re-snaps the EE orientation
  # to canonical face-down BEFORE each wrapper attempt. Critical: the
  # wrapper's cleanup retracts to safe-height with whatever EE orientation
  # the prior insert left (typically tilted by a few degrees due to peg-in-
  # slot tolerance during force-mode wedging). Without re-rotating, the
  # next attempt starts tilted and wastes 20+ corrections fighting the
  # angle. With re-rotate, the descent starts canonical-flat.
  python3 -m compliant_insertion_studio.scripts.iterate_insert \\
    --object-name u_orange --base-name base1 --grasp-id 1 \\
    --already-held --held-quat QX QY QZ QW
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Signal forwarding — make Ctrl+C / SIGTERM trigger the wrapper's clean-stop
# path (stop_force_mode → switch back → safe_height → home) instead of leaving
# the robot in force mode. Critical for at-robot safety.
# ---------------------------------------------------------------------------

_ACTIVE_CHILD: subprocess.Popen | None = None
_CLEANUP_TIMEOUT_S = 30.0


def _install_signal_forwarding():
    def _on_signal(signum, frame):
        global _ACTIVE_CHILD
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n>>> iterate_insert received {sig_name}; forwarding to active subprocess "
              f"+ giving wrapper up to {_CLEANUP_TIMEOUT_S:.0f}s to run its cleanup "
              f"(stop_force_mode → switch back → safe_height).", flush=True)
        child = _ACTIVE_CHILD
        if child is None or child.poll() is not None:
            sys.exit(130)
        try:
            # Kill the entire process group — catches the python wrapper AND any
            # subprocess it spawned (translate_object subcommands etc.).
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError) as e:
            print(f"    (could not signal child group: {e})", flush=True)
        # Wait for wrapper's cleanup path to complete
        try:
            child.wait(timeout=_CLEANUP_TIMEOUT_S)
            print(f">>> child exited cleanly (rc={child.returncode})", flush=True)
        except subprocess.TimeoutExpired:
            print(f">>> child cleanup exceeded {_CLEANUP_TIMEOUT_S}s — sending SIGKILL. "
                  f"⚠ Robot may need MANUAL recovery (see wrapper docs).", flush=True)
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        sys.exit(130)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def _spawn_in_pgroup(cmd: list[str], cwd: str | None = None) -> subprocess.Popen:
    """Spawn a subprocess in its own process group so SIGTERM forwarding can
    target the whole subtree (wrapper + any nested subprocesses).
    """
    return subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        preexec_fn=os.setsid,           # new session/group
    )


# ---------------------------------------------------------------------------
# Base pose helper — pulled from primitives/shared/config.py
#
# IMPORTANT (operator decision 2026-05-04): the base is PHYSICALLY FIXED at
# the assembly fixture, always at the same location. The camera detector on
# /objects_poses_real has noise (mm-scale jitter) which propagates into the
# wrapper's CAD-derived predicted_tcp_at_seat and misaligns the spiral aim
# point. Iter 4 + iter 5 used camera-derived base poses and the spiral
# aimed slightly off-target; the first attempt today used config-derived
# default and the operator observed visibly closer-to-insertion behavior.
# Conclusion: always use the config value. Don't read the camera for base1.
# ---------------------------------------------------------------------------

def _config_base_pose() -> dict:
    """Return {xyz, quat} from primitives.shared.config — authoritative."""
    from primitives.shared.config import (
        DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION,
    )
    return {
        "xyz":  [float(v) for v in DEFAULT_BASE_POSITION],
        "quat": [float(v) for v in DEFAULT_BASE_ORIENTATION],
    }


def _read_prior_hole_xy(object_name: str) -> tuple[float, float] | None:
    """Search recent meta JSONs for a `hole_observed.xy_m` and return the most
    recent. None if no prior detection exists. Cross-attempt learning helper.
    """
    logs_dir = REPO_ROOT / "compliant_insertion_studio" / "logs"
    candidates = sorted(logs_dir.glob(f"insert_{object_name}_*.meta.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates[:20]:   # don't scan forever; recent 20 is plenty
        try:
            d = json.load(open(p))
        except Exception:
            continue
        ho = d.get("hole_observed")
        if isinstance(ho, dict):
            xy = ho.get("xy_m")
            if xy and len(xy) >= 2:
                return (float(xy[0]), float(xy[1]))
    return None


# ---------------------------------------------------------------------------
# Setup-only delegation
# ---------------------------------------------------------------------------

_HELD_QUAT_RE = re.compile(r"HELD_QUAT=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")


def _run_setup(args) -> list[float]:
    """Call run_assembly_step.py --setup-only and parse the printed HELD_QUAT.

    Returns the post-second-rotate quaternion (xyzw) of the held object —
    this is the orientation the wrapper consumes via --current-object-orientation.
    """
    cmd = [
        "python3", "-m", "compliant_insertion_studio.scripts.run_assembly_step",
        "--object-name", args.object_name,
        "--base-name",   args.base_name,
        "--grasp-id",    str(args.grasp_id),
        "--grasp-width", str(args.grasp_width),
        "--mode",        "real",
        "--setup-only",
    ]
    if args.already_held:
        if args.held_quat is None:
            raise SystemExit("--already-held requires --held-quat QX QY QZ QW")
        cmd += ["--already-held", "--current-object-orientation",
                f"{args.held_quat[0]:.6f}", f"{args.held_quat[1]:.6f}",
                f"{args.held_quat[2]:.6f}", f"{args.held_quat[3]:.6f}"]

    global _ACTIVE_CHILD
    print("\n>>> SETUP — run_assembly_step.py --setup-only")
    print(f"    cmd: {' '.join(cmd)}")
    # Stream live — operator watches the rotate/place/regrasp/rotate sequence
    proc = _spawn_in_pgroup(cmd, cwd=str(REPO_ROOT))
    _ACTIVE_CHILD = proc
    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    rc = proc.wait()
    _ACTIVE_CHILD = None
    out = "".join(captured)

    if rc != 0:
        raise SystemExit(f"setup phase failed (rc={rc}). See output above.")

    # Take the LAST HELD_QUAT line (post-second-rotate).
    matches = _HELD_QUAT_RE.findall(out)
    if not matches:
        raise SystemExit("could not parse HELD_QUAT from setup output — "
                         "did run_assembly_step actually emit it?")
    held = [float(x) for x in matches[-1]]
    print(f"\n    PARSED HELD_QUAT (post-second-rotate): {held}")
    return held


# ---------------------------------------------------------------------------
# Wrapper invocation
# ---------------------------------------------------------------------------

def _run_wrapper(args, held_quat: list[float], base_world_pose: dict | None) -> int:
    cmd = [
        "python3", "-m", "compliant_insertion_studio.wrapper.compliant_insert",
        "--object-name", args.object_name,
        "--base-name",   args.base_name,
        "--grasp-id",    str(args.grasp_id),
        "--current-object-orientation",
        f"{held_quat[0]:.6f}", f"{held_quat[1]:.6f}",
        f"{held_quat[2]:.6f}", f"{held_quat[3]:.6f}",
        "--use-default-base-position",
        "--fz", str(args.fz), "--override-fz-cap",
        "--lin-speed", str(args.lin_speed),
        "--gain",      str(args.gain),
        # damping uses wrapper default (0.7) — restored after 0.95 over-tuned
        # and blocked post-contact lateral motion needed by the spiral
        "--step-back", "auto",
        "--auto-step-back-seconds", str(args.auto_step_back_seconds),
        "--no-prompt-notes",
        "--timeout", str(args.timeout),
    ]
    if args.skip_home_on_done:
        cmd.append("--skip-home-on-done")
    if base_world_pose is not None:
        # Phase 5 v1: enable CAD-derived predicted_tcp_at_seat in the predicate.
        cmd += ["--base-world-pose"] + [
            f"{base_world_pose['xyz'][0]:.6f}",
            f"{base_world_pose['xyz'][1]:.6f}",
            f"{base_world_pose['xyz'][2]:.6f}",
            f"{base_world_pose['quat'][0]:.6f}",
            f"{base_world_pose['quat'][1]:.6f}",
            f"{base_world_pose['quat'][2]:.6f}",
            f"{base_world_pose['quat'][3]:.6f}",
        ]
    # Cross-attempt learning: if a prior attempt detected the hole via
    # descent-spike, pass it as --hole-xy-prior so this attempt aims directly
    # at the observed slot location instead of CAD-derived (which has per-
    # grasp variance baked in).
    if args.hole_xy_prior is not None:
        prior_hole_xy = (args.hole_xy_prior[0], args.hole_xy_prior[1])
        print(f"\n>>> HOLE_XY_PRIOR (manual override via CLI): "
              f"({prior_hole_xy[0]:+.4f}, {prior_hole_xy[1]:+.4f})")
        cmd += ["--hole-xy-prior", f"{prior_hole_xy[0]:.6f}", f"{prior_hole_xy[1]:.6f}"]
    else:
        prior_hole_xy = _read_prior_hole_xy(args.object_name)
        if prior_hole_xy is not None:
            print(f"\n>>> HOLE_XY_PRIOR (from prior attempt's meta): "
                  f"({prior_hole_xy[0]:+.4f}, {prior_hole_xy[1]:+.4f})")
            cmd += ["--hole-xy-prior", f"{prior_hole_xy[0]:.6f}", f"{prior_hole_xy[1]:.6f}"]

    global _ACTIVE_CHILD
    print("\n>>> WRAPPER — compliant_insertion_studio.wrapper.compliant_insert")
    print(f"    cmd: {' '.join(cmd)}")
    proc = _spawn_in_pgroup(cmd, cwd=str(REPO_ROOT))
    _ACTIVE_CHILD = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    _ACTIVE_CHILD = None
    return rc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--object-name", required=True)
    p.add_argument("--base-name",   required=True)
    p.add_argument("--grasp-id",    type=int, required=True)
    p.add_argument("--grasp-width", type=int, default=35,
                   help="Pre-grasp gripper width (mm). Ground truth from "
                        "<obj>_grasp_points.json:grasp_validity.x_axis_gripper_width_mm.")
    p.add_argument("--already-held", action="store_true",
                   help="Pass --already-held to run_assembly_step. Skips "
                        "pick/place/regrasp; jumps to rotate_object (step 12) "
                        "which re-snaps EE orientation to canonical face-down. "
                        "Critical for iter loops: gripper stays closed across "
                        "attempts, but each attempt re-rotates so a tilted "
                        "post-cleanup orientation doesn't propagate forward.")
    p.add_argument("--held-quat", nargs=4, type=float, default=None,
                   metavar=("QX", "QY", "QZ", "QW"),
                   help="Required with --already-held: the current held-part "
                        "quaternion (post-prior-rotate orientation). "
                        "rotate_object will use this as input and emit a new "
                        "(re-snapped) HELD_QUAT, which is then passed to the "
                        "wrapper.")
    # Locked Phase 3 force-mode tuning
    p.add_argument("--fz",                       type=float, default=9.0)
    p.add_argument("--lin-speed",                type=float, default=0.54)
    p.add_argument("--gain",                     type=float, default=1.0)
    p.add_argument("--auto-step-back-seconds",   type=float, default=5.0)
    p.add_argument("--timeout",                  type=float, default=600.0,
                   help="ACTIVE-phase safety ceiling. Predicate auto-exit fires far before this.")
    p.add_argument("--hole-xy-prior", nargs=2, type=float, default=None,
                   metavar=("X", "Y"),
                   help="Override hole_xy_prior (m, base_link). When set, takes precedence over the "
                        "auto-extracted prior from previous attempts. Use for sanity-check runs only — "
                        "in production the prior must come from feedback, not from ground truth.")
    p.add_argument("--skip-home-on-done", action="store_true", default=True,
                   help="Stay at safe-height after DONE (faster iteration).")
    args = p.parse_args()

    # Install signal forwarding NOW so any Ctrl+C / SIGTERM from this point
    # forward triggers the wrapper's clean-stop path.
    _install_signal_forwarding()

    # 1. Authoritative base pose from primitives/shared/config.py.
    #    The base is physically fixed; camera /objects_poses_real has mm-scale
    #    noise that misaligns the wrapper's CAD-derived target_xy.
    base_world_pose = _config_base_pose()
    print(f"\n>>> BASE_WORLD_POSE ({args.base_name}, from primitives.shared.config)")
    print(f"    xyz:  {base_world_pose['xyz']}")
    print(f"    quat: {base_world_pose['quat']}")

    # 2. SETUP — full canonical sequence on first attempt, or rotate-only on
    # subsequent attempts (--already-held). Rotate-only re-snaps EE
    # orientation to canonical before each wrapper attempt.
    held_quat = _run_setup(args)

    # 3. WRAPPER — autonomous insert with Phase 5 universal predicate.
    rc = _run_wrapper(args, held_quat, base_world_pose)
    print(f"\n>>> wrapper exit rc={rc}")

    # 4. Print copy/paste resume command for next iteration (held by gripper now).
    #    --already-held skips pick/place/regrasp; rotate_object re-snaps to
    #    canonical orientation each attempt.
    quat_str = " ".join(f"{v:.6f}" for v in held_quat)
    print("\nNext iteration command (copy/paste, no regrasp, re-rotate to canonical):")
    print(f"  python3 -m compliant_insertion_studio.scripts.iterate_insert \\")
    print(f"    --object-name {args.object_name} --base-name {args.base_name} "
          f"--grasp-id {args.grasp_id} \\")
    print(f"    --already-held --held-quat {quat_str}")


if __name__ == "__main__":
    main()
