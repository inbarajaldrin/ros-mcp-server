"""
Phase 5 validation-loop runner — runs `iterate_insert.py` repeatedly for one
object, parses each attempt's `outcome` from the meta JSON, counts consecutive
successes, and stops when the target is reached or the operator interrupts.

This is the harness around `iterate_insert.py` for tuning the insert primitive
on a single object before generalizing to the next. Per the operator's plan:

  1. Pick an object → run loop_iterate.py with that --object-name
  2. Each attempt: gripper performs setup + insert. Wrapper exits autonomously
     (predicate met = success) OR by safety timeout / abort = failure.
  3. The runner prints a per-attempt verdict from the meta JSON:
       outcome=success / outcome_reason=predicate_met → ✓
       outcome=timeout / outcome_reason=timeout_reached → ✗ (stuck or wedge)
       outcome=abort  / outcome_reason=...            → ✗ (precondition fail)
  4. Tracks consecutive successes — stops at --target-success-count (default 5).
  5. Between attempts: optional sleep + visual confirmation by operator.

Iteration loop is INTENTIONALLY simple: just runs iterate_insert.py and reads
the resulting meta JSON. Stop the runner with Ctrl+C — it forwards SIGTERM
into iterate_insert which forwards into the wrapper which runs its full
cleanup path (force-mode off, controllers reset, retract to safe-height).

Usage:
  # One object, default target (5 consecutive successes)
  python3 -m compliant_insertion_studio.scripts.loop_iterate \\
    --object-name u_orange --base-name base1 --grasp-id 1

  # Tighter target / specific held quat for the FIRST attempt
  python3 -m compliant_insertion_studio.scripts.loop_iterate \\
    --object-name u_orange --base-name base1 --grasp-id 1 \\
    --target-success-count 3 --first-held-quat 0.0062 -0.6494 0.7604 0.0055

  # Pause between attempts so you can inspect / tweak between runs
  python3 -m compliant_insertion_studio.scripts.loop_iterate \\
    --object-name u_orange --base-name base1 --grasp-id 1 \\
    --inter-attempt-sleep-s 5
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
LOGS_DIR  = REPO_ROOT / "compliant_insertion_studio" / "logs"

_HELD_QUAT_RE = re.compile(r"PARSED HELD_QUAT.*?\[([-\d.eE+\s,]+)\]")


# ---------------------------------------------------------------------------
# Cancel-safety: forward SIGINT/SIGTERM into the running iterate_insert child
# ---------------------------------------------------------------------------

_CHILD: subprocess.Popen | None = None


def _on_signal(signum, frame):
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n>>> loop_iterate received {sig_name}; forwarding to iterate_insert child + waiting up to 60s for cleanup",
          flush=True)
    if _CHILD and _CHILD.poll() is None:
        try:
            os.killpg(os.getpgid(_CHILD.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            _CHILD.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_CHILD.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    sys.exit(130)


# ---------------------------------------------------------------------------
# Per-attempt: run iterate_insert.py, return outcome + path to meta JSON
# ---------------------------------------------------------------------------

def _run_one_attempt(object_name: str, base_name: str, grasp_id: int,
                      held_quat: list[float] | None,
                      attempt_idx: int, log_dir: Path,
                      hole_xy_prior: tuple[float, float] | None = None) -> dict:
    """Run iterate_insert.py once, return parsed result dict.

    Attempt 1 (held_quat is None): full setup — run_assembly_step does
    pick→rotate→place→regrasp→rotate. Captures the post-setup held_quat.

    Attempts 2+ (held_quat provided): --already-held. Skips pick/place/regrasp
    (gripper stays closed across all attempts) BUT calls rotate_object to
    re-snap EE orientation to canonical face-down before each wrapper attempt.
    This is critical because the wrapper's cleanup retracts holding whatever
    orientation the prior insert left — typically tilted by a few degrees due
    to peg-in-slot tolerance during force-mode wedging. Without re-rotating,
    the next attempt starts tilted and burns 20+ corrections fighting the angle.
    """
    global _CHILD
    cmd = [
        "python3", "-m", "compliant_insertion_studio.scripts.iterate_insert",
        "--object-name", object_name,
        "--base-name",   base_name,
        "--grasp-id",    str(grasp_id),
    ]
    if held_quat is not None:
        cmd += ["--already-held", "--held-quat",
                f"{held_quat[0]:.6f}", f"{held_quat[1]:.6f}",
                f"{held_quat[2]:.6f}", f"{held_quat[3]:.6f}"]
    if hole_xy_prior is not None:
        cmd += ["--hole-xy-prior",
                f"{hole_xy_prior[0]:.6f}", f"{hole_xy_prior[1]:.6f}"]
    log_path = log_dir / f"loop_attempt_{attempt_idx:03d}.log"
    print(f"\n{'='*72}\n>>> ATTEMPT {attempt_idx}\n{'='*72}")
    print(f"    cmd: {' '.join(cmd)}")
    print(f"    log: {log_path}")
    t0 = time.time()
    with open(log_path, "w") as logfile:
        _CHILD = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1, preexec_fn=os.setsid)
        # Stream live to both stdout and the log file
        assert _CHILD.stdout is not None
        last_held_quat = None
        for line in _CHILD.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            logfile.write(line)
            m = _HELD_QUAT_RE.search(line)
            if m:
                vals = [float(v) for v in m.group(1).replace(",", " ").split()]
                if len(vals) == 4:
                    last_held_quat = vals
        rc = _CHILD.wait()
    _CHILD = None
    duration = time.time() - t0

    # Parse the most-recent insert meta JSON for outcome
    meta_path = _find_latest_meta(object_name, since_t=t0)
    outcome, outcome_reason = ("unknown", f"no meta JSON written (rc={rc})")
    if meta_path:
        try:
            m = json.load(open(meta_path))
            outcome = m.get("outcome", "unknown")
            outcome_reason = m.get("outcome_reason", "?")
        except Exception as e:
            outcome_reason = f"meta parse error: {e}"

    return {
        "attempt":        attempt_idx,
        "duration_s":     duration,
        "rc":             rc,
        "outcome":        outcome,
        "outcome_reason": outcome_reason,
        "meta_path":      str(meta_path) if meta_path else None,
        "log_path":       str(log_path),
        "held_quat_for_next": last_held_quat,
    }


def _find_latest_meta(object_name: str, since_t: float) -> Path | None:
    """Return the most recent .meta.json for this object created after since_t."""
    candidates = sorted(LOGS_DIR.glob(f"insert_{object_name}_*.meta.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_mtime >= since_t - 5.0:  # 5 s grace
            return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--object-name",          required=True)
    p.add_argument("--base-name",            required=True)
    p.add_argument("--grasp-id",             type=int, required=True)
    p.add_argument("--target-success-count", type=int, default=5,
                   help="Stop after N consecutive successes (default 5 per ROADMAP VAL-01).")
    p.add_argument("--max-attempts",         type=int, default=20,
                   help="Hard ceiling on total attempts before giving up.")
    p.add_argument("--first-held-quat",      type=float, nargs=4, default=None,
                   metavar=("QX", "QY", "QZ", "QW"),
                   help="If set: pass --already-held --held-quat to attempt 1. "
                        "Subsequent attempts chain the post-rotate quat from the "
                        "previous attempt's setup automatically.")
    p.add_argument("--inter-attempt-sleep-s", type=float, default=2.0,
                   help="Sleep between attempts so robot finishes settling.")
    p.add_argument("--hole-xy-prior", nargs=2, type=float, default=None,
                   metavar=("X", "Y"),
                   help="Manual override of hole_xy_prior (m, base_link), forwarded to every "
                        "attempt. Use for sanity checks against a known-good seat XY (e.g. extracted "
                        "from a GOLD operator demo). In production the prior must come from feedback, "
                        "not from ground truth.")
    args = p.parse_args()

    # === Single-instance guard (added 2026-05-04 PM) ===
    # Prevent two loop_iterate instances running simultaneously — they would
    # fight over the robot. Bug observed when an operator/agent SIGTERM'd a
    # running loop and relaunched before the cleanup chain (wrapper run_done,
    # ~30-60s) finished. The PID file is removed at clean exit; a stale file
    # whose PID is no longer alive is treated as recoverable and overwritten.
    pidfile = Path("/tmp/loop_iterate.pid")
    if pidfile.exists():
        try:
            stale_pid = int(pidfile.read_text().strip())
        except Exception:
            stale_pid = None
        if stale_pid and stale_pid != os.getpid():
            try:
                os.kill(stale_pid, 0)   # raises if process is gone
                print(f">>> ERROR: another loop_iterate (pid={stale_pid}) is "
                      f"already running.\n"
                      f"    Stop it first with: kill -TERM {stale_pid}\n"
                      f"    Or wait for it to exit, then relaunch.\n"
                      f"    If you're SURE nothing's running, delete {pidfile}.")
                sys.exit(2)
            except ProcessLookupError:
                pass  # stale, fall through to overwrite
    pidfile.write_text(str(os.getpid()))
    import atexit
    def _cleanup_pidfile():
        try:
            if pidfile.exists() and pidfile.read_text().strip() == str(os.getpid()):
                pidfile.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_pidfile)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log_dir = REPO_ROOT / ".planning" / "phases" / "05-algorithm-derivation" / f"loop_{args.object_name}_{int(time.time())}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> Loop log dir: {log_dir}")

    summary: list[dict] = []
    consecutive_successes = 0
    held_quat = args.first_held_quat

    for attempt_idx in range(1, args.max_attempts + 1):
        result = _run_one_attempt(args.object_name, args.base_name, args.grasp_id,
                                    held_quat, attempt_idx, log_dir,
                                    hole_xy_prior=tuple(args.hole_xy_prior) if args.hole_xy_prior else None)
        summary.append(result)

        # Verdict
        if result["outcome"] == "success" and result["outcome_reason"] == "predicate_met":
            consecutive_successes += 1
            verdict = f"✓ SUCCESS  ({consecutive_successes}/{args.target_success_count} consecutive)"
        elif result["outcome"] == "success":
            consecutive_successes += 1
            verdict = f"✓ SUCCESS  ({consecutive_successes}/{args.target_success_count}) [reason={result['outcome_reason']}]"
        else:
            consecutive_successes = 0
            verdict = f"✗ {result['outcome'].upper():<8} (streak reset)  reason={result['outcome_reason']}"

        print(f"\n>>> ATTEMPT {attempt_idx} VERDICT: {verdict}  duration={result['duration_s']:.1f}s")
        print(f"    meta: {result['meta_path']}")

        # Persist running summary so we can inspect during the loop
        summary_path = log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "object_name":            args.object_name,
                "base_name":              args.base_name,
                "grasp_id":               args.grasp_id,
                "target_success_count":   args.target_success_count,
                "consecutive_successes":  consecutive_successes,
                "attempts_done":          attempt_idx,
                "results":                summary,
            }, f, indent=2)

        if consecutive_successes >= args.target_success_count:
            print(f"\n>>> 🎯 TARGET REACHED — {consecutive_successes} consecutive successes for "
                  f"{args.object_name}. Phase 5 validation gate met for this object.")
            break

        # Stop on first failure so the operator can analyze logs and tune.
        # The 5/5 target requires deterministic success per attempt; if any
        # attempt aborts/timeouts/etc the cause must be addressed before
        # continuing — auto-retrying just burns time and accumulates noise.
        if result["outcome"] != "success":
            print(f"\n>>> 🛑 STOPPING ON FIRST FAILURE — analyze {result['meta_path']} "
                  f"(reason={result['outcome_reason']}) before relaunching.")
            break

        # Chain the held-quat from this attempt's setup output to next attempt
        if result["held_quat_for_next"]:
            held_quat = result["held_quat_for_next"]

        if attempt_idx < args.max_attempts:
            print(f"\n>>> sleep {args.inter_attempt_sleep_s}s before attempt {attempt_idx+1} ...")
            time.sleep(args.inter_attempt_sleep_s)
    else:
        print(f"\n>>> Hit max-attempts ({args.max_attempts}) without "
              f"{args.target_success_count} consecutive successes. Inspect logs in {log_dir}.")

    # Final summary
    print(f"\n{'='*72}\nFINAL SUMMARY\n{'='*72}")
    print(f"  object: {args.object_name}/{args.base_name}/grasp_id={args.grasp_id}")
    print(f"  attempts: {len(summary)}")
    print(f"  consecutive successes (final streak): {consecutive_successes}")
    print(f"  outcomes: {[(r['outcome'], r['outcome_reason']) for r in summary]}")
    print(f"  detail JSON: {log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
