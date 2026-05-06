#!/usr/bin/env python3
"""
Interactive data-collection harness for regime-decoding GOLD demos.

Sequences 15 operator-guided insertion demos at varied --base-offset-xy values
(5 directions × 3 reps). For each demo:

  1. Prints which variation is up next.
  2. Waits for [Enter] to launch.
  3. Runs `run_assembly_step --already-held` synchronously, attached to the
     terminal. The wrapper enters force mode after STEP-BACK timer; operator
     guides the peg into the slot manually.
  4. Operator presses Ctrl+C when the insert is complete; SIGINT propagates
     through the terminal to the subprocess, the wrapper's cleanup runs (stop
     force mode, switch controllers, retract to safe height).
  5. Script verifies the CSV + sidecars exist, logs the outcome to a per-session
     manifest, and moves to the next demo.

The held_quat is constant across all demos: rotate_object recovers EE to
canonical from any starting pose (we verified this empirically on 2026-05-06
with a 3.66° pendant offset, which recovered to 0.30°). R_grasp's fold info
is what the held_quat carries, and that doesn't drift.

Usage:
  python3 -m compliant_insertion_studio.scripts.collect_regime_data \\
    --object u_orange --base base1 --grasp-id 1

Outputs:
  - 15× CSV + sidecars in compliant_insertion_studio/logs/
  - Manifest at compliant_insertion_studio/logs/regime_collection_<ts>.json
    mapping each demo basename → variation label + offset + outcome
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "compliant_insertion_studio" / "logs"

# 5 variations × 3 reps = 15 demos. Same fold across all (rotate_object handles).
# Magnitudes ~10mm to give the operator a clear lateral slide to perform.
VARIATIONS = [
    ("A_pos_x_10mm",     +0.010,  0.000, "peg lands +X side of slot, 10mm offset"),
    ("B_neg_x_10mm",     -0.010,  0.000, "peg lands -X side, 10mm offset"),
    ("C_pos_y_10mm",      0.000, +0.010, "peg lands +Y side, 10mm offset"),
    ("D_neg_y_10mm",      0.000, -0.010, "peg lands -Y side, 10mm offset"),
    ("E_diag_pxpy_7mm",  +0.007, +0.007, "peg lands +X+Y diagonal, ~10mm"),
]
REPS_PER_VARIATION = 3

# A known-good held_quat with the correct -X fold for this base setup. Captured
# 2026-05-05 from a successful chain. Won't drift across demos because rotate_object
# converges EE to canonical regardless of input pose.
DEFAULT_HELD_QUAT = (-0.0296, 0.7065, -0.7065, 0.0296)


def _bold(s): return f"\033[1m{s}\033[0m"
def _green(s): return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _red(s): return f"\033[91m{s}\033[0m"


def find_latest_csv(since_ts):
    """Return the basename of the most-recent CSV created since since_ts.
    Returns None if not found."""
    candidates = []
    for path in LOG_DIR.glob("insert_*.csv"):
        if "raw.csv" in path.name or "fm_events" in path.name:
            continue  # sidecar
        if path.stat().st_mtime >= since_ts:
            candidates.append(path)
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.stem


def run_one_demo(args, label, dx, dy, description, demo_idx, total) -> dict:
    """Run one assembly step. Block until subprocess exits. Return dict with
    basename + outcome + offset, or {'error': ...} on failure."""
    print()
    print("=" * 72)
    print(_bold(f"  Demo {demo_idx}/{total}: {label}"))
    print(f"  {description}")
    print(f"  --base-offset-xy {dx:+.4f} {dy:+.4f}  (DX={dx*1000:+.1f}mm DY={dy*1000:+.1f}mm)")
    print("=" * 72)
    print()
    print(_yellow("  Press [Enter] when ready to launch this demo, or 'skip' to skip, 'quit' to stop."))
    user_in = input("  > ").strip().lower()
    if user_in == "quit":
        return {"action": "quit"}
    if user_in == "skip":
        return {"action": "skip", "label": label}

    cmd = [
        "python3", "-u", "-m", "compliant_insertion_studio.scripts.run_assembly_step",
        "--object-name", args.object,
        "--base-name", args.base,
        "--grasp-id", str(args.grasp_id),
        "--grasp-width", str(args.grasp_width),
        "--mode", "real",
    ]
    if args.fresh_grasp:
        # Full pick path: camera → grasp → rotate → place → regrasp → rotate → approach.
        # Held quat is derived internally from camera observation; operator passes nothing.
        pass
    else:
        cmd += [
            "--already-held",
            "--current-object-orientation",
                f"{args.held_quat[0]:.6f}", f"{args.held_quat[1]:.6f}",
                f"{args.held_quat[2]:.6f}", f"{args.held_quat[3]:.6f}",
        ]
    cmd += [
        "--base-offset-xy", f"{dx:.6f}", f"{dy:.6f}",
        "--fz", str(args.fz), "--override-fz-cap",
        "--step-back", "auto",
        "--step-back-seconds", str(args.step_back_seconds),
        "--guided-mode",
    ]
    print(_green("  Launching subprocess. Wrapper does setup → APPROACH descent → on Contact"))
    print(_green("  enters GUIDED state. The script will WAIT for the GUIDED transition log,"))
    print(_green("  then prompt you to drag the peg to the slot."))
    print()

    t_before = time.time()
    # Pipe stdout so we can detect the GUIDED transition before prompting.
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    rc = -1

    # Background reader: tee subprocess output to terminal AND watch for the
    # "FSM → GUIDED" transition (and the post-SIGUSR1 INSERT_DESCENT log) so
    # the script can sequence prompts correctly.
    import threading
    guided_active = threading.Event()
    insert_descent_seen = threading.Event()

    def _stdout_reader():
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                if "FSM → GUIDED" in line or "GUIDED active" in line:
                    guided_active.set()
                if "INSERT_DESCENT" in line and "→" in line:
                    insert_descent_seen.set()
        except Exception as e:
            sys.stdout.write(f"[stdout-reader error: {e}]\n")

    reader = threading.Thread(target=_stdout_reader, daemon=True)
    reader.start()

    try:
        # Wait for GUIDED transition (= post-Contact). Don't prompt before this.
        print(_yellow("  Waiting for wrapper to reach GUIDED state (post-Contact)..."))
        while not guided_active.is_set():
            if proc.poll() is not None:
                # Subprocess exited before GUIDED — something failed earlier
                print(_red("  Subprocess exited before reaching GUIDED state. Aborting demo."))
                break
            time.sleep(0.2)

        if guided_active.is_set():
            print()
            print(_green("=" * 72))
            print(_green("  GUIDED state ACTIVE — drag the peg laterally to above the slot."))
            print(_green("=" * 72))
            try:
                input(_yellow("  [Press Enter when peg is above the hole to send SIGUSR1] > "))
                try:
                    proc.send_signal(signal.SIGUSR1)
                    print(_green("    SIGUSR1 sent. Waiting for INSERT_DESCENT..."))
                except Exception as e:
                    print(_red(f"    SIGUSR1 send failed: {e}"))
            except (EOFError, KeyboardInterrupt):
                print(_yellow("    Cancelled hole-mark; subprocess will need Ctrl+C to clean up."))

        # After SIGUSR1: subprocess does INSERT_DESCENT, eventually exits via
        # seat-detected DONE or operator Ctrl+C. We just wait.
        proc.wait()
        rc = proc.returncode
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=30)
        except Exception:
            pass
        rc = proc.returncode if proc.returncode is not None else -1
    finally:
        reader.join(timeout=2)
    t_after = time.time()

    basename = find_latest_csv(t_before)
    print()
    if basename is None:
        print(_red(f"  ✗ No CSV produced this run. rc={rc}"))
        return {"action": "fail", "label": label, "rc": rc, "reason": "no_csv"}

    csv_path = LOG_DIR / f"{basename}.csv"
    meta_path = LOG_DIR / f"{basename}.meta.json"
    sidecars = {
        "joints":    LOG_DIR / f"{basename}.joints_raw.csv",
        "wrench":    LOG_DIR / f"{basename}.wrench_raw.csv",
        "cmd_wrench":LOG_DIR / f"{basename}.cmd_wrench_raw.csv",
        "fm_events": LOG_DIR / f"{basename}.fm_events.csv",
    }
    missing = [k for k, p in sidecars.items() if not p.exists()]
    outcome = "?"
    reason = "?"
    if meta_path.exists():
        try:
            m = json.load(open(meta_path))
            outcome = m.get("outcome", "?")
            reason = m.get("outcome_reason", "?")
        except Exception:
            pass

    # === Physical-seat verification from raw CSV ===
    # FSM `outcome=success / reason=fsm_seated` is its OWN claim; we verify
    # against raw TCP-z trajectory. Anti-pattern caught 2026-05-06: F/T zero
    # contamination caused phantom contact at force-mode entry, FSM declared
    # seated, but peg actually descended through air 22mm off the slot.
    # The skill's hard rule: don't trust FSM outcome labels.
    physical_check = _verify_physical_seat(csv_path)
    physical_status = physical_check["status"]
    if physical_status == "SEATED":
        msg_color = _green
        msg = f"✓ basename={basename}  outcome={outcome}  duration={t_after-t_before:.1f}s  PHYSICAL: SEATED ({physical_check['descent_mm']:.1f}mm)"
    elif physical_status == "FALSE_SEAT_CLAIM":
        msg_color = _red
        msg = (f"✗ basename={basename}  FSM said {outcome} but PHYSICAL VERIFICATION FAILED. "
               f"DISCARD this demo. Reason: {physical_check['reason']}")
    elif physical_status == "PARTIAL":
        msg_color = _yellow
        msg = f"⚠ basename={basename}  PARTIAL seat ({physical_check['descent_mm']:.1f}mm) — review before keeping"
    else:
        msg_color = _yellow
        msg = f"⚠ basename={basename}  outcome={outcome}  could not verify physically: {physical_check.get('reason','?')}"
    print(msg_color(f"  {msg}"))
    if missing:
        print(_yellow(f"    missing sidecars: {missing}"))
    if physical_status != "SEATED":
        print(_yellow(f"    Press Enter to continue, or 'redo' to re-run this variation."))
        ans = input("    > ").strip().lower()
        if ans == "redo":
            return {"action": "redo_requested", "label": label, "basename": basename,
                    "physical_check": physical_check}

    return {
        "action": "ran",
        "label": label,
        "basename": basename,
        "dx_m": dx,
        "dy_m": dy,
        "outcome": outcome,
        "outcome_reason": reason,
        "duration_s": t_after - t_before,
        "missing_sidecars": missing,
        "rc": rc,
        "physical_check": physical_check,
        "use_for_analysis": physical_status == "SEATED",
    }


def _verify_physical_seat(csv_path) -> dict:
    """Verify a demo physically completed an insertion using raw TCP-z trajectory.

    Hard rules (derived empirically from GOLD demos + 2026-05-06 false-seat case):
      - Contact moment must be > 1s into ACTIVE (early contact = F/T-zero contamination)
      - Z descent post-contact must be 25-45mm (real seats are ~31mm; >50mm = peg fell through)
      - For GUIDED demos: final TCP xy must be within 10mm of operator-marked hole
        (peg seats AT the hole, not at contact xy — operator legitimately drags peg
        from contact_xy to hole_xy in GUIDED state, so contact→end drift is large by design)
      - For autonomous demos (no hole_observed_operator): final-xy-drift-from-contact
        must be < 15mm (original heuristic)
    """
    import csv as _csv
    import json as _json
    import os as _os
    try:
        rows = list(_csv.DictReader(open(csv_path)))
    except Exception as e:
        return {"status": "UNKNOWN", "reason": f"csv read failed: {e}"}
    # Try to read the operator-marked hole xy from meta JSON (GUIDED demos only)
    hole_obs_xy = None
    meta_path = csv_path.with_suffix('.meta.json') if hasattr(csv_path, 'with_suffix') else None
    if meta_path is None:
        meta_path = str(csv_path).replace('.csv', '.meta.json')
    try:
        if _os.path.exists(meta_path):
            meta = _json.load(open(meta_path))
            ho = meta.get("hole_observed_operator")
            if ho and ho.get("xy_m") and len(ho["xy_m"]) >= 2:
                hole_obs_xy = (float(ho["xy_m"][0]), float(ho["xy_m"][1]))
    except Exception:
        pass

    def f(r, k):
        try: return float(r[k])
        except: return float('nan')
    active = [r for r in rows if r.get('phase', '') == 'ACTIVE']
    if len(active) < 100:
        return {"status": "UNKNOWN", "reason": f"too few ACTIVE rows ({len(active)})"}
    fz = [abs(f(r, 'fz')) for r in active]
    t  = [f(r, 't_s') for r in active]
    pz = [f(r, 'tcp_z') for r in active]
    px = [f(r, 'tcp_x') for r in active]
    py = [f(r, 'tcp_y') for r in active]
    # First sample where smoothed fz > 3N for 5 consecutive samples
    contact_idx = None
    for i in range(len(fz) - 5):
        if all(fz[j] > 3.0 for j in range(i, i + 5)):
            contact_idx = i
            break
    if contact_idx is None:
        return {"status": "UNKNOWN", "reason": "no contact event detected (fz never sustained >3N)"}
    contact_t = t[contact_idx] - t[0]
    if contact_t < 1.0:
        return {"status": "FALSE_SEAT_CLAIM",
                "reason": f"phantom contact at t+{contact_t:.2f}s (< 1s = F/T zero contamination)",
                "contact_t_s": contact_t}
    descent_mm = (pz[contact_idx] - pz[-1]) * 1000
    final_xy_drift_from_contact_mm = ((px[-1] - px[contact_idx])**2 + (py[-1] - py[contact_idx])**2)**0.5 * 1000
    final_xy_drift_from_hole_mm = (
        ((px[-1] - hole_obs_xy[0])**2 + (py[-1] - hole_obs_xy[1])**2)**0.5 * 1000
        if hole_obs_xy is not None else None
    )
    info = {
        "contact_t_s": contact_t,
        "descent_mm": descent_mm,
        "final_xy_drift_from_contact_mm": final_xy_drift_from_contact_mm,
        "final_xy_drift_from_hole_mm": final_xy_drift_from_hole_mm,
        "final_xy": (px[-1], py[-1]),
        "hole_observed_xy": list(hole_obs_xy) if hole_obs_xy else None,
    }
    # Choose the appropriate xy-drift criterion based on demo type
    is_guided = hole_obs_xy is not None
    if is_guided:
        # Peg should seat AT the operator-marked hole xy; allow 10mm slack
        # (peg pushes a few mm into chamfer; lever-arm yaw causes mm-scale drift)
        xy_drift_mm = final_xy_drift_from_hole_mm
        xy_drift_threshold = 10.0
        xy_drift_label = "from hole_observed"
    else:
        xy_drift_mm = final_xy_drift_from_contact_mm
        xy_drift_threshold = 15.0
        xy_drift_label = "from contact"
    if 25 <= descent_mm <= 45 and xy_drift_mm < xy_drift_threshold:
        info["status"] = "SEATED"
    elif descent_mm > 45:
        info["status"] = "FALSE_SEAT_CLAIM"
        info["reason"] = f"excessive descent {descent_mm:.1f}mm (peg fell through, not seated)"
    elif descent_mm < 5:
        info["status"] = "FALSE_SEAT_CLAIM"
        info["reason"] = f"no real descent ({descent_mm:.1f}mm) — peg never engaged"
    elif xy_drift_mm > xy_drift_threshold:
        info["status"] = "FALSE_SEAT_CLAIM"
        info["reason"] = (f"large xy drift {xy_drift_label} ({xy_drift_mm:.1f}mm > "
                          f"{xy_drift_threshold:.0f}mm) — peg not seated at expected location")
    else:
        info["status"] = "PARTIAL"
        info["reason"] = f"descent {descent_mm:.1f}mm in chamfer range — review"
    return info


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    p.add_argument("--object", default="u_orange")
    p.add_argument("--base",   default="base1")
    p.add_argument("--grasp-id", type=int, default=None,
                   help="Auto-resolved from ablations/eval_resources/fmb1_assembly.json "
                        "via primitives.shared.config.get_grasp_id_for_assembly() if omitted.")
    p.add_argument("--grasp-width", type=int, default=None,
                   help="Auto-resolved from fmb1_assembly.json via "
                        "primitives.shared.config.get_gripper_width_mm() if omitted.")
    p.add_argument("--fz", type=float, default=9.0,
                   help="Wrapper Fz (N). Need --override-fz-cap (auto-passed) for >5N.")
    p.add_argument("--step-back-seconds", type=float, default=5.0)
    p.add_argument("--held-quat", nargs=4, type=float, default=list(DEFAULT_HELD_QUAT),
                   metavar=("QX","QY","QZ","QW"),
                   help="Held-part quaternion. Default is the empirically-validated -X fold "
                        "value. rotate_object converges EE to canonical from any input. "
                        "Ignored when --fresh-grasp is set (quat derived from camera).")
    p.add_argument("--fresh-grasp", action="store_true",
                   help="Run the full pick path before each demo (camera-grasp → rotate → "
                        "place → regrasp → rotate → approach). Held quat is derived from "
                        "camera observation; operator passes nothing. Between demos the "
                        "operator must place the part back on the table.")
    p.add_argument("--reps", type=int, default=REPS_PER_VARIATION,
                   help="Repetitions per variation (default 3, total = 5×reps).")
    p.add_argument("--variations", default=None,
                   help="Comma-separated subset of variation labels to run (default: all 5).")
    args = p.parse_args()
    args.held_quat = tuple(args.held_quat)

    # Auto-resolve grasp_id and grasp_width from fmb1_assembly.json (mirrors
    # loop_autonomous_insert.sh / run_assembly_step / regrasp_held_object).
    if args.grasp_id is None:
        from primitives.shared.config import get_grasp_id_for_assembly
        resolved = get_grasp_id_for_assembly(args.object)
        if resolved is None:
            sys.exit(f"Could not auto-resolve --grasp-id for object={args.object!r}. "
                     f"Pass --grasp-id explicitly or add an entry to fmb1_assembly.json.")
        args.grasp_id = resolved
        print(_yellow(f"  Auto-resolved grasp_id={args.grasp_id} for object={args.object} "
                      f"(from fmb1_assembly.json)"))
    if args.grasp_width is None:
        from primitives.shared.config import get_gripper_width_mm
        resolved = get_gripper_width_mm(args.object, args.grasp_id)
        if resolved is None:
            sys.exit(f"Could not auto-resolve --grasp-width for object={args.object!r} "
                     f"grasp_id={args.grasp_id}. Pass --grasp-width explicitly.")
        args.grasp_width = int(resolved)
        print(_yellow(f"  Auto-resolved grasp_width={args.grasp_width}mm "
                      f"(from fmb1_assembly.json)"))

    variations = VARIATIONS
    if args.variations:
        wanted = set(args.variations.split(","))
        variations = [v for v in VARIATIONS if v[0] in wanted]
        if not variations:
            sys.exit(f"No variations matched {args.variations}. Available: {[v[0] for v in VARIATIONS]}")

    plan = []
    for label, dx, dy, desc in variations:
        for _ in range(args.reps):
            plan.append((label, dx, dy, desc))
    total = len(plan)

    print(_bold("\n" + "=" * 72))
    print(_bold(f"  Regime-decoding data collection: {total} demos planned"))
    print(_bold("=" * 72))
    print(f"  Object: {args.object}  Base: {args.base}  GraspID: {args.grasp_id}")
    if args.fresh_grasp:
        print(f"  Fresh grasp: ON (full pick path each demo; quat from camera)")
    else:
        print(f"  Held quat: {args.held_quat}")
    print(f"  Variations × reps: {len(variations)} × {args.reps}")
    for label, dx, dy, desc in variations:
        print(f"    - {label}: ({dx*1000:+.1f}, {dy*1000:+.1f})mm — {desc}")
    print()
    print(_yellow("  Per-demo flow:"))
    print(_yellow("    1. Press [Enter] when ready (or 'skip' / 'quit')."))
    print(_yellow("    2. Wrapper enters force mode after STEP-BACK gate (~5s). Stay clear during gate."))
    print(_yellow("    3. Guide the peg into the slot manually."))
    print(_yellow("    4. Press Ctrl+C when seated to trigger cleanup."))
    print()
    input("  Press [Enter] to begin the session, or Ctrl+C to abort. > ")

    ts = int(time.time())
    manifest_path = LOG_DIR / f"regime_collection_{ts}.json"
    manifest = {
        "session_start_ts": ts,
        "object": args.object, "base": args.base, "grasp_id": args.grasp_id,
        "held_quat": list(args.held_quat),
        "fz_N": args.fz,
        "demos": [],
    }

    for i, (label, dx, dy, desc) in enumerate(plan, start=1):
        result = run_one_demo(args, label, dx, dy, desc, i, total)
        if result.get("action") == "quit":
            print(_yellow("\n  Quit requested. Saving manifest and exiting."))
            break
        manifest["demos"].append(result)
        # Persist manifest after every demo (in case operator stops mid-session)
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        # Fresh-grasp mode: peg is seated/released after each demo. Operator must
        # reset the part on the table before the next pick.
        if args.fresh_grasp and i < total:
            print(_yellow("\n  Fresh-grasp mode: place the part back on the table"
                          " (visible to camera), then press Enter for next demo."))
            try:
                input("  > ")
            except (EOFError, KeyboardInterrupt):
                print(_yellow("  Aborted between demos."))
                break

    # Final summary
    ran = [d for d in manifest["demos"] if d.get("action") == "ran"]
    skipped = [d for d in manifest["demos"] if d.get("action") == "skip"]
    failed = [d for d in manifest["demos"] if d.get("action") == "fail"]
    print()
    print(_bold("=" * 72))
    print(_bold("  Session complete"))
    print(_bold("=" * 72))
    print(f"  Demos run: {len(ran)}  skipped: {len(skipped)}  failed: {len(failed)}")
    print(f"  Manifest: {manifest_path}")
    if ran:
        print()
        print("  Per-variation count:")
        from collections import Counter
        cnt = Counter(d["label"] for d in ran)
        for label, dx, dy, _ in variations:
            print(f"    {label}: {cnt.get(label, 0)}")


if __name__ == "__main__":
    main()
