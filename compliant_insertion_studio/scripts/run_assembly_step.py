#!/usr/bin/env python3
"""
End-to-end orchestrator for one assembly step (pick + rotate + insert).

Implements the canonical sequence from
``ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json``:

    1.  control_gripper to grasp-width
    2.  move_to_grasp(object, grasp_id)               -> chain orientation_1
    3.  control_gripper close
    4.  move_to_safe_height
    5.  rotate_object(orientation_1)                  -> chain orientation_2
    6.  translate_object --place-down
    7.  control_gripper to grasp-width
    8.  move_to_safe_height
    9.  move_to_grasp(object, grasp_id)               -> chain orientation_3
    10. control_gripper close
    11. move_to_safe_height
    12. rotate_object(orientation_3)                  -> chain orientation_4
    13. compliant_insert wrapper(orientation_4)       -> FSM PRE/HOVER/ZERO/ACTIVE/DONE
        (wrapper handles its own cleanup: stop force mode -> safe_height -> home)

Step 13 replaces the legacy ``translate_object --insert`` call from the
ground-truth JSON with the new compliant-insert wrapper.

Usage
-----
Full sequence (object on the table):

    python3 -m compliant_insertion_studio.scripts.run_assembly_step \\
        --object-name u_brown --base-name base1 --grasp-id 1

Already-held mode (skip steps 1-11; rotate + insert with currently held part):

    python3 -m compliant_insertion_studio.scripts.run_assembly_step \\
        --object-name u_brown --base-name base1 --grasp-id 1 \\
        --already-held \\
        --current-object-orientation -0.005 -0.7058 0.7083 -0.0045

Per-object reusability
----------------------
The same invocation works for every FMB1 part — only ``--object-name`` and
``--grasp-id`` change per the ground-truth JSON. ``--grasp-width`` defaults to
35mm which matches the canonical command for u_brown / u_orange / line_green
/ inverted_u_yellow.
"""

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Match the JSON envelope the primitives print:
#     __RESULT_JSON__\n{...}\n__END_RESULT_JSON__
_JSON_BLOCK_RE = re.compile(
    r'__RESULT_JSON__\s*(\{.*?\})\s*__END_RESULT_JSON__', re.DOTALL)


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _run(label: str, argv: list, *, timeout_s: int = 120,
         allow_failure: bool = False) -> dict:
    """Run a primitive subprocess. Stream output live; return parsed result JSON.

    2026-05-06: switched from blocking capture to live streaming via Popen+readline.
    The previous capture_output=True buffered all wrapper output until exit, which
    broke collect_regime_data.py's interactive flow (it couldn't see "FSM → GUIDED"
    in real time and so couldn't time the operator prompt correctly).

    Raises SystemExit(1) if the primitive returned ``result == "failure"`` or
    no result envelope was emitted, UNLESS ``allow_failure=True``.
    """
    print(_bold(f"\n>>> {label}"))
    print(f"    cmd: {' '.join(argv)}")
    out_lines = []
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(REPO_ROOT),
        )
    except FileNotFoundError as e:
        print(_red(f"    EXEC FAILED: {e}"))
        sys.exit(1)

    # Forward operator signals (SIGUSR1, SIGUSR2) to the subprocess so they
    # reach the wrapper. Without this, SIGUSR1 sent by collect_regime_data
    # lands here (run_assembly_step) instead of the wrapper, and the wrapper's
    # mark_hole() never fires. Bug caught 2026-05-06 after operator's GUIDED-mode
    # demo timed out with hole_observed_operator=None despite SIGUSR1 being sent.
    _orig_sigusr1 = signal.getsignal(signal.SIGUSR1)
    _orig_sigusr2 = signal.getsignal(signal.SIGUSR2)
    def _forward_sigusr1(signum, frame):
        if proc.poll() is None:
            try: proc.send_signal(signal.SIGUSR1)
            except Exception: pass
    def _forward_sigusr2(signum, frame):
        if proc.poll() is None:
            try: proc.send_signal(signal.SIGUSR2)
            except Exception: pass
    signal.signal(signal.SIGUSR1, _forward_sigusr1)
    signal.signal(signal.SIGUSR2, _forward_sigusr2)

    # Stream stdout line-by-line: print each prefixed AND keep for JSON parsing.
    deadline = time.time() + timeout_s
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            sys.stdout.write(f"    {line}")
            sys.stdout.flush()
            out_lines.append(line)
            if time.time() > deadline:
                print(_red(f"    TIMEOUT after {timeout_s}s"))
                proc.kill()
                proc.wait(timeout=5)
                sys.exit(1)
        proc.wait(timeout=5)
    except KeyboardInterrupt:
        # Forward SIGINT to the child for clean cleanup
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            pass
        # Surface remaining output
        if proc.stdout is not None:
            try:
                tail = proc.stdout.read()
                if tail:
                    sys.stdout.write(tail)
                    out_lines.append(tail)
            except Exception:
                pass
        raise

    out = "".join(out_lines)
    matches = _JSON_BLOCK_RE.findall(out)
    if not matches:
        print(_red(f"    NO RESULT_JSON in output (rc={proc.returncode})"))
        sys.exit(1)

    # Take the LAST result envelope (subprocesses can echo intermediate ones)
    try:
        result = json.loads(matches[-1])
    except json.JSONDecodeError as e:
        print(_red(f"    Could not parse result JSON: {e}"))
        sys.exit(1)

    if result.get("result") == "failure":
        if allow_failure:
            print(_red(f"    soft failure: {result.get('outcome_reason', result.get('error', '?'))}"))
            return result
        print(_red(f"    FAILED: {result.get('error', result)}"))
        sys.exit(1)

    print(_green("    OK"))
    return result


def _quat_from(result: dict, key: str) -> Tuple[float, float, float, float]:
    """Pull a (qx, qy, qz, qw) tuple out of a result envelope.

    ``key`` is one of ``current_object_orientation`` or
    ``final_object_orientation``. The envelope shape varies slightly across
    primitives — handle both ``{key: {quat: {x,y,z,w}}}`` and
    ``{key: {x,y,z,w}}``.
    """
    block = result.get(key)
    if block is None:
        raise KeyError(f"{key!r} missing from result: {result}")
    if "quat" in block:
        block = block["quat"]
    return (
        float(block["x"]), float(block["y"]),
        float(block["z"]), float(block["w"]),
    )


def _quat_argv(quat: Tuple[float, float, float, float]) -> list:
    """Format quaternion for `--current-object-orientation` CLI arg."""
    return ["--current-object-orientation"] + [f"{v:.6f}" for v in quat]


# NOTE 2026-05-06: A `_bootstrap_seed_from_current_tcp` helper existed here briefly
# during regime-decoding work. It was removed because it discarded R_grasp's
# rotational signature (which encodes the FOLD info from initial perception at
# pick time) and substituted a synthetic canonical seed. The fold then became
# whatever IK happened to pick at rotate_object time — random across runs.
# Empirical test 2026-05-06: with a 3.66° pendant offset, the chained-held_quat
# path correctly recovered both canonical EE orientation AND the original fold,
# while the bootstrap flipped the fold to the mirror-image. The chained path is
# the design — don't add bootstrap-style overrides.


def _control_gripper(command: str, *, mode: str = "real") -> dict:
    return _run(
        f"control_gripper {command}",
        ["python3", "-m", "primitives.control_gripper", command, "--mode", mode],
        timeout_s=20,
    )


def _move_to_safe_height(*, mode: str = "real") -> dict:
    return _run(
        "move_to_safe_height",
        ["python3", "-m", "primitives.move_to_safe_height", "--mode", mode],
        timeout_s=30,
    )


def _move_home(*, mode: str = "real") -> dict:
    return _run(
        "move_home --joint-space",
        ["python3", "-m", "primitives.move_home", "--joint-space", "--mode", mode],
        timeout_s=30,
    )


def _move_to_grasp(object_name: str, grasp_id: int, *, mode: str = "real") -> dict:
    return _run(
        f"move_to_grasp {object_name} grasp_id={grasp_id}",
        ["python3", "-m", "primitives.move_to_grasp",
         "--object-name", object_name,
         "--grasp-id", str(grasp_id),
         "--mode", mode],
        timeout_s=120,
    )


def _rotate_object(object_name: str, base_name: str,
                   quat: Tuple[float, float, float, float],
                   *, mode: str = "real") -> dict:
    return _run(
        f"rotate_object {object_name} base={base_name}",
        ["python3", "-m", "primitives.rotate_object",
         "--mode", mode,
         "--object-name", object_name,
         "--base-name", base_name,
         *_quat_argv(quat)],
        timeout_s=90,
    )


def _place_down(object_name: str, *, mode: str = "real") -> dict:
    return _run(
        f"translate_object --place-down {object_name}",
        ["python3", "-m", "primitives.translate_object",
         "--mode", mode, "--place-down",
         "--object-name", object_name],
        timeout_s=180,
    )


def _wrapper(object_name: str, base_name: str, grasp_id: int,
             quat: Tuple[float, float, float, float],
             fz: float, step_back: str, step_back_seconds: float,
             use_default_base_position: bool,
             base_offset_xy: Tuple[float, float] | None = None,
             override_fz_cap: bool = False,
             abort_on_first_contact: bool = False,
             guided_mode: bool = False,
             v4_autofire: bool = False,
             autonomous_search: bool = False,
             search_F_press_N: float = 9.0,
             search_max_duration_s: float = 15.0,
             search_Fmax_N: float = 3.0,
             search_v_s_mm_s: float = 5.0,
             search_pitch_mm: float = 2.0,
             search_R_max_mm: float = 8.0) -> dict:
    argv = ["python3", "-u", "-m", "compliant_insertion_studio.wrapper.compliant_insert",
            "--object-name", object_name,
            "--base-name", base_name,
            "--grasp-id", str(grasp_id),
            *_quat_argv(quat),
            "--fz", str(fz),
            "--step-back", step_back,
            "--auto-step-back-seconds", str(step_back_seconds),
            "--no-prompt-notes",
            # 2026-05-07: skip the PRE-phase F/T smoke test on every run.
            # run_assembly_step is the orchestrated assembly entry point and
            # is always called within an active session — exactly the
            # "repeated rapid attempts" case the wrapper's --skip-smoke flag
            # is documented for. Saves ~5-6s per insert.
            "--skip-smoke"]
    if override_fz_cap:
        argv.append("--override-fz-cap")
    if abort_on_first_contact:
        argv.append("--abort-on-first-contact")
    if guided_mode:
        argv.append("--guided-mode")
        # Operator needs time to drag the peg in GUIDED state. Wrapper default
        # is 120s which times out before SIGUSR1 arrives. 600s gives plenty.
        argv += ["--timeout", "600"]
        if v4_autofire:
            argv.append("--v4-autofire")
    if autonomous_search:
        argv.append("--autonomous-search")
        argv += ["--search-F-press-N", str(search_F_press_N),
                 "--search-max-duration-s", str(search_max_duration_s),
                 "--search-Fmax-N", str(search_Fmax_N),
                 "--search-v-s-mm-s", str(search_v_s_mm_s),
                 "--search-pitch-mm", str(search_pitch_mm),
                 "--search-R-max-mm", str(search_R_max_mm)]
        # Generous timeout: full insert pipeline
        argv += ["--timeout", "120"]
    # 2026-05-07: combine the per-object PER_OBJECT_BASE_OFFSET_M with any
    # --base-offset-xy passed by the caller. Per-object offset is calibrated
    # for objects whose CAD seat doesn't match the u_brown-derived
    # DEFAULT_BASE_POSITION (e.g. inverted_u_yellow). Both offsets stack so
    # data-collection runs (which use --base-offset-xy as a perturbation)
    # still benefit from the per-object baseline calibration.
    from primitives.shared.config import (
        DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION, get_object_base_offset_m,
    )
    obj_off = get_object_base_offset_m(object_name)
    cli_off_xy = base_offset_xy if base_offset_xy is not None else (0.0, 0.0)
    total_dx = obj_off[0] + float(cli_off_xy[0])
    total_dy = obj_off[1] + float(cli_off_xy[1])
    total_dz = obj_off[2]   # CLI offset is xy-only by design
    has_offset = (total_dx != 0.0 or total_dy != 0.0 or total_dz != 0.0)
    if has_offset:
        bx = float(DEFAULT_BASE_POSITION[0]) + total_dx
        by = float(DEFAULT_BASE_POSITION[1]) + total_dy
        bz = float(DEFAULT_BASE_POSITION[2]) + total_dz
        bq = [float(v) for v in DEFAULT_BASE_ORIENTATION]
        argv += ["--base-world-pose", f"{bx:.6f}", f"{by:.6f}", f"{bz:.6f}",
                 f"{bq[0]:.6f}", f"{bq[1]:.6f}", f"{bq[2]:.6f}", f"{bq[3]:.6f}"]
        argv += ["--final-base-pos", f"{bx:.6f}", f"{by:.6f}", f"{bz:.6f}"]
        argv += ["--final-base-orientation",
                 f"{bq[0]:.6f}", f"{bq[1]:.6f}", f"{bq[2]:.6f}", f"{bq[3]:.6f}"]
    elif use_default_base_position:
        argv.append("--use-default-base-position")
    # Outer subprocess timeout: 300s default, but GUIDED mode lets operator drag
    # the peg manually so 700s gives ample headroom over the wrapper's --timeout 600.
    outer_timeout = 700 if guided_mode else 300
    return _run(
        f"compliant_insert wrapper ({object_name} -> {base_name})",
        argv, timeout_s=outer_timeout, allow_failure=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end pick + rotate + insert orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--object-name", required=True, help="e.g. u_brown")
    p.add_argument("--base-name", required=True, help="e.g. base1")
    p.add_argument("--grasp-id", type=int, required=True,
                   help="Grasp index (1-based, see <object>_grasp_points.json)")
    p.add_argument("--grasp-width", type=int, default=None,
                   help="Gripper width before grasping (mm). If not given, auto-"
                        "resolved per (object, grasp_id) from "
                        "ablations/eval_resources/fmb1_assembly.json; falls back to 35 "
                        "if not in table.")
    p.add_argument("--fz", type=float, default=3.0,
                   help="Wrapper downward force (N)")
    p.add_argument("--step-back", choices=["prompt", "auto", "signal"],
                   default="auto",
                   help="Wrapper STEP-BACK confirmation mode")
    p.add_argument("--step-back-seconds", type=float, default=5.0,
                   help="Seconds to wait when --step-back=auto")
    p.add_argument("--already-held", action="store_true",
                   help="Skip pick/place/regrasp (steps 1-11). Requires "
                        "--current-object-orientation.")
    p.add_argument("--current-object-orientation", nargs=4, type=float,
                   metavar=("QX", "QY", "QZ", "QW"),
                   help="Held-part orientation (required with --already-held)")
    p.add_argument("--use-default-base-position", action="store_true",
                   default=True,
                   help="Use the host's DEFAULT_BASE_POSITION/ORIENTATION")
    p.add_argument("--base-offset-xy", nargs=2, type=float, default=None,
                   metavar=("DX", "DY"),
                   help="Inject xy offset (m) into the base_world_pose forwarded to "
                        "the wrapper and hover. With non-zero offset, peg's first-contact "
                        "xy is shifted by (DX, DY) from the actual slot xy — used to "
                        "collect operator demos with controlled starting positions for "
                        "regime-decoding (see analysis/REGIME_DECODING.md). Both axes are "
                        "in robot base_link frame (m). Identity if not set.")
    p.add_argument("--override-fz-cap", action="store_true",
                   help="Forwarded to wrapper. Required when --fz exceeds the 5N safety cap.")
    p.add_argument("--abort-on-first-contact", action="store_true",
                   help="Forwarded to wrapper. Stops at first contact for marker validation "
                        "and EE-drift analysis (no force-mode insertion / no operator guide).")
    p.add_argument("--guided-mode", action="store_true",
                   help="Forwarded to wrapper. Routes APPROACH-contact to GUIDED state "
                        "(operator-drag) instead of FIND_HOLE. Operator drags peg to slot, "
                        "sends SIGUSR1 to mark hole, then INSERT_DESCENT runs autonomously.")
    p.add_argument("--v4-autofire", action="store_true",
                   help="Forwarded to wrapper (requires --guided-mode). v4 Found Hole "
                        "predicate firing in GUIDED state ALSO triggers GUIDED→INSERT_DESCENT "
                        "(stage 3b). Without this flag (stage 3a), v4 fire is logged but only "
                        "operator's SIGUSR1 triggers the transition. See analysis/CONTROL_LAW.md.")
    p.add_argument("--autonomous-search", action="store_true",
                   help="Forwarded to wrapper. Routes APPROACH-contact to SEARCH state "
                        "(autonomous F/T-driven search director). Mutually exclusive with "
                        "--guided-mode. See analysis/SEARCH_CONTROL_LAW.md.")
    p.add_argument("--search-F-press-N", type=float, default=9.0,
                   help="SEARCH spiral director downward press force. Default 9N.")
    p.add_argument("--search-max-duration-s", type=float, default=15.0,
                   help="SEARCH timeout. Default 15s.")
    p.add_argument("--search-Fmax-N", type=float, default=3.0,
                   help="Spiral PD lateral force saturation. Default 3N.")
    p.add_argument("--search-v-s-mm-s", type=float, default=5.0,
                   help="Spiral tangential speed mm/s. Default 5.")
    p.add_argument("--search-pitch-mm", type=float, default=2.0,
                   help="Spiral pitch (radial growth per turn) mm. Default 2.")
    p.add_argument("--search-R-max-mm", type=float, default=8.0,
                   help="Spiral max radius mm. Default 8.")
    p.add_argument("--mode", default="real", choices=["real", "sim"])
    p.add_argument("--setup-only", action="store_true",
                   help="Run canonical pick→rotate→place→regrasp→rotate sequence "
                        "and EXIT before invoking the compliant_insert wrapper. "
                        "Prints the held-part quaternion as the final line so the "
                        "caller (e.g., agent doing Phase 3 collection) can launch "
                        "the wrapper separately for SIGTERM control during ACTIVE.")
    args = p.parse_args()

    if args.already_held and args.current_object_orientation is None:
        p.error("--already-held requires --current-object-orientation")

    # Auto-resolve grasp_width from fmb1_assembly.json if not given.
    if args.grasp_width is None:
        from primitives.shared.config import get_gripper_width_mm
        w = get_gripper_width_mm(args.object_name, args.grasp_id, default_mm=35)
        args.grasp_width = int(round(float(w)))
        print(_green(f"Auto-resolved --grasp-width {args.grasp_width} mm "
                     f"for ({args.object_name}, grasp_id={args.grasp_id})"))

    print(_bold(f"\n=== Assembly step: insert {args.object_name} into "
                f"{args.base_name} (grasp_id={args.grasp_id}) ==="))
    print(f"=== Mode: {args.mode}  fz={args.fz}N  grasp_width={args.grasp_width}mm  "
          f"already_held={args.already_held}")
    if args.base_offset_xy and (args.base_offset_xy[0] != 0.0 or args.base_offset_xy[1] != 0.0):
        dxmm = args.base_offset_xy[0] * 1000
        dymm = args.base_offset_xy[1] * 1000
        print(_bold(f"=== BASE OFFSET ACTIVE: peg first-contact will be shifted "
                    f"({dxmm:+.1f}, {dymm:+.1f})mm from actual slot ==="))

    if args.already_held:
        held_quat = tuple(args.current_object_orientation)
        print(_bold("\n--- ALREADY-HELD path: starting from rotate (step 12) ---"))
        # Ensure robot is at safe height BEFORE rotating. Without this, a
        # caller (e.g. iter loop) that runs --already-held from an arbitrary
        # post-prior-attempt pose can have rotate_object's trajectory dip
        # below safe height and bump the held part out of the gripper.
        # Bug observed 2026-05-04 iter 8: part dropped during rotate when
        # robot started at clear-area pose (z=0.32, exactly at safe height).
        _move_to_safe_height(mode=args.mode)
    else:
        # ---- Steps 1-4: open, pick, close, lift ----
        _control_gripper(str(args.grasp_width), mode=args.mode)
        r = _move_to_grasp(args.object_name, args.grasp_id, mode=args.mode)
        post_pick_quat = _quat_from(r, "current_object_orientation")
        _control_gripper("close", mode=args.mode)
        _move_to_safe_height(mode=args.mode)

        # ---- Step 5: first rotate ----
        r = _rotate_object(args.object_name, args.base_name,
                           post_pick_quat, mode=args.mode)
        # post-rotation quat unused — next step is place_down which doesn't
        # consume orientation

        # ---- Step 6: place down ----
        _place_down(args.object_name, mode=args.mode)

        # ---- Steps 7-11: open, lift, regrasp, close, lift ----
        _control_gripper(str(args.grasp_width), mode=args.mode)
        _move_to_safe_height(mode=args.mode)
        r = _move_to_grasp(args.object_name, args.grasp_id, mode=args.mode)
        post_regrasp_quat = _quat_from(r, "current_object_orientation")
        _control_gripper("close", mode=args.mode)
        _move_to_safe_height(mode=args.mode)

        held_quat = post_regrasp_quat

    # ---- Step 12: second rotate (or first rotate in already-held mode) ----
    r = _rotate_object(args.object_name, args.base_name,
                       held_quat, mode=args.mode)
    insert_quat = _quat_from(r, "final_object_orientation")

    # --- setup-only exit point (no wrapper invocation) ---
    if args.setup_only:
        qx, qy, qz, qw = insert_quat
        print(_bold("\n=== Setup complete (--setup-only) — wrapper NOT invoked ==="))
        print(f"Held-part orientation (after second rotate):")
        # Machine-readable last line: caller greps for HELD_QUAT=
        print(f"HELD_QUAT={qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
        return 0

    # ---- Step 13: WRAPPER (replaces translate_object --insert) ----
    # The wrapper internally walks PRE -> HOVER -> ZERO -> ACTIVE -> DONE.
    # On any abort or on DONE it cleans up: stop force mode, switch back to
    # position controller, move_to_safe_height, move_home. So no separate
    # safe-height + home calls are needed after this returns.

    # 2026-05-07: per-object force overrides. Some objects need much lower
    # downforce throughout the pipeline (e.g. line_green where the gripper
    # collides with the base). Apply on top of CLI args.
    from primitives.shared.config import get_object_insert_forces
    obj_forces = get_object_insert_forces(args.object_name)
    fz_use            = obj_forces.get("fz_approach",    args.fz)
    f_press_use       = obj_forces.get("search_F_press", args.search_F_press_N)
    fmax_use          = obj_forces.get("search_Fmax",    args.search_Fmax_N)
    if obj_forces:
        # Use signed Fz consistent with caller convention (negative = down)
        if fz_use > 0:
            fz_use = -abs(fz_use)
        print(_bold(
            f"Per-object force overrides for {args.object_name}: "
            f"fz={fz_use}N  search_F_press={f_press_use}N  search_Fmax={fmax_use}N"
        ))

    wrapper_result = _wrapper(
        args.object_name, args.base_name, args.grasp_id,
        insert_quat, fz_use, args.step_back, args.step_back_seconds,
        args.use_default_base_position,
        base_offset_xy=tuple(args.base_offset_xy) if args.base_offset_xy else None,
        override_fz_cap=args.override_fz_cap,
        abort_on_first_contact=args.abort_on_first_contact,
        guided_mode=args.guided_mode,
        v4_autofire=args.v4_autofire,
        autonomous_search=args.autonomous_search,
        search_F_press_N=f_press_use,
        search_max_duration_s=args.search_max_duration_s,
        search_Fmax_N=fmax_use,
        search_v_s_mm_s=args.search_v_s_mm_s,
        search_pitch_mm=args.search_pitch_mm,
        search_R_max_mm=args.search_R_max_mm,
    )

    print(_bold("\n=== Assembly step complete ==="))
    print(f"Wrapper outcome: {wrapper_result.get('outcome', '?')}  "
          f"reason: {wrapper_result.get('outcome_reason', '?')}")
    print(f"CSV : {wrapper_result.get('csv_path', '?')}")
    print(f"Meta: {wrapper_result.get('meta_path', '?')}")
    print(f"Samples logged: {wrapper_result.get('samples_logged', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
