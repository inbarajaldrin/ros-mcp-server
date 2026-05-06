#!/usr/bin/env python3
"""
Re-grasp a currently-held object: release at clear area, vision-find pose,
re-pick at the canonical grasp.

This is the inverse companion of run_assembly_step's setup phase: that
script assumes the object is on the table; this script assumes the
object is already in the gripper (e.g., after a prior insert + retract)
and you want a FRESH grasp before another insert run.

Steps (each is a separate primitive subprocess):
  1. (Caller responsibility) camera launched + /objects_poses_real publishing.
     This script verifies the topic is alive before proceeding.
  2. move_to_clear_area    — head to (-0.320, -0.5, SAFE_HEIGHT) where
                              dropping the part is safe.
  3. control_gripper <w>   — open to the same width that was used at the
                              original pick (default 35 mm). Part falls
                              from safe height onto the table.
  4. (settle wait)         — let the part come to rest + camera publish a
                              fresh pose for it.
  5. move_to_safe_height   — clear of the dropped part.
  6. move_to_grasp         — vision-driven plan to grasp pose for given
                              object/grasp_id; descends to grasp.
  7. control_gripper close — close on the part.
  8. move_to_safe_height   — ready for downstream wrapper.

Output (last line stdout, JSON):
  __RESULT_JSON__
  { "result": "success" | "failure",
    "post_regrasp_quat_xyzw": [...],   # held-object orientation after step 6
    "error": "..."                     # only on failure
  }
  __END_RESULT_JSON__

Usage:
  python3 -m compliant_insertion_studio.scripts.regrasp_held_object \
    --object-name u_orange --grasp-id 1 [--grasp-width 35] [--mode real] \
    [--clear-xy X Y] [--settle-s 1.5]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _green(s):  return f"\033[92m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _run(label: str, argv: list[str], timeout_s: int = 60) -> dict:
    """Run a primitive subprocess. Parse __RESULT_JSON__ block from stdout.
    Raises RuntimeError on non-success result."""
    print(_bold(f"\n>>> {label}"))
    print(f"    cmd: {' '.join(argv)}")
    try:
        proc = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True,
                               text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label}: timeout after {timeout_s}s")
    out = proc.stdout
    err = proc.stderr
    # Tee stdout
    for line in out.splitlines():
        print(f"    {line}")
    if err.strip():
        for line in err.splitlines():
            print(f"    [stderr] {line}")
    # Find __RESULT_JSON__ block
    m = re.search(
        r"__RESULT_JSON__\s*\n(.*?)\n\s*__END_RESULT_JSON__",
        _strip_ansi(out), re.DOTALL,
    )
    if not m:
        raise RuntimeError(f"{label}: no __RESULT_JSON__ block in stdout")
    payload = json.loads(m.group(1))
    if payload.get("result") != "success":
        raise RuntimeError(f"{label}: {payload.get('error', 'unknown failure')}")
    print(_green(f"    OK"))
    return payload


def _check_camera_topic_alive(timeout_s: float = 5.0) -> bool:
    """Verify /objects_poses_real is publishing. Returns True if a message
    arrives within timeout, False otherwise."""
    print(_bold("\n>>> verify camera topic alive: /objects_poses_real"))
    proc = subprocess.run(
        ["ros2", "topic", "echo", "--once", "--qos-reliability", "best_effort",
         "/objects_poses_real"],
        capture_output=True, text=True, timeout=int(timeout_s),
    )
    if proc.returncode != 0:
        # Try once more without best_effort flag (older ROS may not support it)
        proc = subprocess.run(
            ["ros2", "topic", "echo", "--once", "/objects_poses_real"],
            capture_output=True, text=True, timeout=int(timeout_s),
        )
    if proc.returncode == 0 and proc.stdout.strip():
        print(_green("    OK — at least one message received"))
        return True
    print(_red("    NO MESSAGE — camera topic is silent"))
    print(_yellow("    Run launch_camera.sh first:"))
    print(_yellow("      bash compliant_insertion_studio/scripts/launch_camera.sh --background"))
    return False


def _move_to_clear_area(target_xy: tuple[float, float] | None, mode: str) -> dict:
    argv = ["python3", "-m", "primitives.core.move_to_clear_area",
            "--mode", mode]
    if target_xy is not None:
        argv += ["--target-xy", str(target_xy[0]), str(target_xy[1])]
    return _run("move_to_clear_area", argv, timeout_s=60)


def _control_gripper(width_or_cmd: str, mode: str) -> dict:
    return _run(
        f"control_gripper {width_or_cmd}",
        ["python3", "-m", "primitives.control_gripper",
         width_or_cmd, "--mode", mode],
        timeout_s=20,
    )


def _move_to_safe_height(mode: str) -> dict:
    return _run(
        "move_to_safe_height",
        ["python3", "-m", "primitives.move_to_safe_height", "--mode", mode],
        timeout_s=30,
    )


def _move_to_grasp(object_name: str, grasp_id: int, mode: str) -> dict:
    return _run(
        f"move_to_grasp {object_name} grasp_id={grasp_id}",
        ["python3", "-m", "primitives.move_to_grasp",
         "--object-name", object_name,
         "--grasp-id", str(grasp_id),
         "--mode", mode],
        timeout_s=120,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object-name", required=True)
    p.add_argument("--grasp-id", type=int, required=True)
    p.add_argument("--grasp-width", type=int, default=None,
                   help="Gripper width (mm) for the open command at release. "
                        "If not given, auto-resolved per (object, grasp_id) from "
                        "ablations/eval_resources/fmb1_assembly.json; falls back to 35.")
    p.add_argument("--mode", default="real", choices=["real", "sim"])
    p.add_argument("--clear-xy", type=float, nargs=2, default=None,
                   help="(X Y) in robot base frame for clear-area drop. Defaults "
                        "to move_to_clear_area's built-in safe drop xy.")
    p.add_argument("--settle-s", type=float, default=1.5,
                   help="Pause after release for part + camera to settle. Default 1.5s.")
    p.add_argument("--skip-camera-check", action="store_true",
                   help="Skip the /objects_poses_real liveness check (e.g. for "
                        "scripted use where camera is known to be up).")
    args = p.parse_args()

    if args.grasp_width is None:
        from primitives.shared.config import get_gripper_width_mm
        w = get_gripper_width_mm(args.object_name, args.grasp_id, default_mm=35)
        args.grasp_width = int(round(float(w)))
        print(_green(f"Auto-resolved --grasp-width {args.grasp_width} mm "
                     f"for ({args.object_name}, grasp_id={args.grasp_id})"))

    print(_bold(f"\n=== Re-grasp held object: {args.object_name} (grasp_id={args.grasp_id}) ==="))
    print(f"  mode={args.mode} grasp_width={args.grasp_width}mm settle={args.settle_s}s")

    try:
        if not args.skip_camera_check:
            if not _check_camera_topic_alive():
                raise RuntimeError(
                    "camera not publishing /objects_poses_real — launch camera first"
                )

        # Step 2: move to clear area
        _move_to_clear_area(tuple(args.clear_xy) if args.clear_xy else None, args.mode)

        # Step 3: release (open to grasp_width). Part drops onto table.
        _control_gripper(str(args.grasp_width), args.mode)

        # Step 4: settle so the part comes to rest and camera publishes a fresh pose.
        print(_bold(f"\n>>> settle ({args.settle_s}s)"))
        time.sleep(float(args.settle_s))
        print(_green("    OK"))

        # Step 5: move to safe height (clear of the dropped part).
        _move_to_safe_height(args.mode)

        # Step 6: vision-driven move_to_grasp.
        grasp_result = _move_to_grasp(args.object_name, args.grasp_id, args.mode)

        # Step 7: close gripper on part.
        _control_gripper("close", args.mode)

        # Step 8: move to safe height — ready for the wrapper.
        _move_to_safe_height(args.mode)

        post_quat = (grasp_result.get("current_object_orientation")
                     or grasp_result.get("final_object_orientation")
                     or {}).get("quat")
        if post_quat is not None:
            post_quat = [float(post_quat["x"]), float(post_quat["y"]),
                         float(post_quat["z"]), float(post_quat["w"])]

        print(_bold("\n=== Re-grasp complete ==="))
        print(f"  post_regrasp_quat (xyzw): {post_quat}")
        print()
        print("__RESULT_JSON__")
        print(json.dumps({
            "result": "success",
            "object_name": args.object_name,
            "grasp_id": args.grasp_id,
            "grasp_width_mm": args.grasp_width,
            "post_regrasp_quat_xyzw": post_quat,
        }))
        print("__END_RESULT_JSON__")
        return 0

    except Exception as e:
        print(_red(f"\n=== Re-grasp FAILED: {e} ==="))
        print()
        print("__RESULT_JSON__")
        print(json.dumps({"result": "failure", "error": str(e)}))
        print("__END_RESULT_JSON__")
        return 1


if __name__ == "__main__":
    sys.exit(main())
