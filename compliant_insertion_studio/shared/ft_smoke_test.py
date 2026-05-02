#!/usr/bin/env python3
"""
F/T Smoke Test
==============

Confidence check that the UR5e's wrist F/T sensor is trustworthy this session.
Not a re-calibration — a 10-second "is the sensor still healthy?" probe.

Procedure:
  1. Optional: call /io_and_status_controller/zero_ftsensor (skip with --no-zero)
  2. Settle 0.5 s for the broadcaster to publish fresh post-zero readings
  3. Hold N seconds (default 5), sampling /force_torque_sensor_broadcaster/wrench at R Hz
  4. Compute per-axis residual bias (mean) and drift rate (linear-regression slope)
  5. Compare against thresholds from docs/ft_calibration_sop.md and PASS/FAIL

Preconditions:
  - Robot is in a steady pose (do NOT call this while motion is happening)
  - Operator's hands are off the robot, the gripper, AND any payload
  - F/T sensor has been powered for ≥ 10–30 min (warm-up window)
  - The gripper is empty (no part) for the canonical baseline check;
    payload-attached checks are valid but use --bias-fmax / --bias-tmax overrides

Usage:
  python3 ft_smoke_test.py                       # standard 5s, default thresholds
  python3 ft_smoke_test.py --hold-s 10            # longer hold
  python3 ft_smoke_test.py --no-zero              # diagnose: skip the zero call
  python3 ft_smoke_test.py --bias-fmax 5.0        # relax force threshold (e.g. with payload)

Exit codes:
  0 → PASS (sensor healthy, all thresholds met)
  1 → FAIL (one or more thresholds exceeded)
  2 → infrastructure error (service / topic missing)

Result is also emitted as __RESULT_JSON__ … __END_RESULT_JSON__ for orchestrators.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped


WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
ZERO_SERVICE = "/io_and_status_controller/zero_ftsensor"


class FTSmokeNode(Node):
    def __init__(self):
        super().__init__("ft_smoke_test")
        self.samples = []  # list of (t_s, fx, fy, fz, tx, ty, tz)
        self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._cb, 10)

    def _cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        t = msg.wrench.torque
        self.samples.append((time.time(), f.x, f.y, f.z, t.x, t.y, t.z))

    def wait_for_topic(self, timeout_s: float = 3.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.samples:
                return True
        return False

    def hold_and_sample(self, hold_s: float, rate_hz: float):
        """Spin and accumulate /wrench samples for hold_s seconds.

        We don't actively pace — the broadcaster publishes at ~500 Hz and we
        record every callback. Decimation to rate_hz happens in analysis.
        """
        self.samples.clear()
        t_end = time.time() + hold_s
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.01)


def call_zero_ftsensor(logger) -> bool:
    """Invoke the zero service via CLI (matches stash + working primitive pattern)."""
    logger("Calling zero_ftsensor …")
    try:
        result = subprocess.run(
            ["ros2", "service", "call", ZERO_SERVICE, "std_srvs/srv/Trigger"],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger("ERROR: zero_ftsensor timed out")
        return False
    if result.returncode != 0 or "success=True" not in result.stdout:
        logger(f"ERROR: zero_ftsensor failed: rc={result.returncode} out={result.stdout.strip()}")
        return False
    logger("zero_ftsensor: success")
    return True


def linear_drift(t, v):
    """Slope of the best-fit line v(t) — units of v per unit t.

    Uses plain numpy least-squares to keep dependencies minimal.
    """
    import numpy as np
    if len(t) < 2:
        return 0.0
    t = np.asarray(t)
    v = np.asarray(v)
    t = t - t[0]  # shift so intercept doesn't dominate
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(slope)


def main():
    parser = argparse.ArgumentParser(description="F/T sensor smoke test (validity check)")
    parser.add_argument("--hold-s", type=float, default=5.0,
                        help="Hold-and-sample window in seconds (default 5.0)")
    parser.add_argument("--rate-hz", type=float, default=100.0,
                        help="Reported / decimated rate (default 100; broadcaster publishes ~500 Hz)")
    parser.add_argument("--bias-fmax", type=float, default=2.0,
                        help="Per-axis residual force bias threshold in N (default 2.0)")
    parser.add_argument("--bias-tmax", type=float, default=0.3,
                        help="Per-axis residual torque bias threshold in Nm (default 0.3)")
    parser.add_argument("--drift-max", type=float, default=0.5,
                        help="Per-axis drift threshold in N/s (default 0.5; applied to F axes only)")
    parser.add_argument("--no-zero", action="store_true",
                        help="Skip the zero_ftsensor call (diagnose: read raw bias only)")
    args = parser.parse_args()

    rclpy.init()
    node = FTSmokeNode()
    log = lambda msg: node.get_logger().info(msg)

    result = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hold_s": args.hold_s,
        "thresholds": {
            "bias_fmax_N": args.bias_fmax,
            "bias_tmax_Nm": args.bias_tmax,
            "drift_max_N_per_s": args.drift_max,
        },
        "zero_called": not args.no_zero,
    }
    exit_code = 0

    try:
        if not node.wait_for_topic(timeout_s=3.0):
            log(f"ERROR: no /wrench message within 3 s — topic alive? broadcaster active?")
            result["result"] = "failure"
            result["error"] = f"{WRENCH_TOPIC} silent"
            exit_code = 2
            return

        if not args.no_zero:
            if not call_zero_ftsensor(log):
                result["result"] = "failure"
                result["error"] = "zero_ftsensor call failed"
                exit_code = 2
                return
            time.sleep(0.5)  # post-zero settle (matches stash convention)

        log(f"Holding {args.hold_s} s, sampling /wrench …")
        node.hold_and_sample(args.hold_s, args.rate_hz)
        n = len(node.samples)
        log(f"Captured {n} samples (~{n / args.hold_s:.0f} Hz)")
        if n < 10:
            result["result"] = "failure"
            result["error"] = f"Too few samples ({n}) — broadcaster slow or stalled"
            exit_code = 2
            return

        # Decompose
        ts = [s[0] for s in node.samples]
        fx = [s[1] for s in node.samples]
        fy = [s[2] for s in node.samples]
        fz = [s[3] for s in node.samples]
        tx = [s[4] for s in node.samples]
        ty = [s[5] for s in node.samples]
        tz = [s[6] for s in node.samples]

        bias = {
            "Fx": sum(fx) / n, "Fy": sum(fy) / n, "Fz": sum(fz) / n,
            "Tx": sum(tx) / n, "Ty": sum(ty) / n, "Tz": sum(tz) / n,
        }
        drift = {
            "Fx_per_s": linear_drift(ts, fx),
            "Fy_per_s": linear_drift(ts, fy),
            "Fz_per_s": linear_drift(ts, fz),
        }

        # Pass/fail per axis
        bias_fail = []
        for axis, val in bias.items():
            limit = args.bias_fmax if axis.startswith("F") else args.bias_tmax
            if abs(val) > limit:
                bias_fail.append({"axis": axis, "value": val, "limit": limit})
        drift_fail = []
        for axis, slope in drift.items():
            if abs(slope) > args.drift_max:
                drift_fail.append({"axis": axis, "slope": slope, "limit": args.drift_max})

        passed = not bias_fail and not drift_fail
        result["bias"] = {k: round(v, 4) for k, v in bias.items()}
        result["drift_per_s"] = {k: round(v, 4) for k, v in drift.items()}
        result["bias_failures"] = bias_fail
        result["drift_failures"] = drift_fail
        result["result"] = "success" if passed else "failure"
        if not passed:
            exit_code = 1

        # Human-readable summary
        log(f"Bias  F=({bias['Fx']:+.2f},{bias['Fy']:+.2f},{bias['Fz']:+.2f})N  "
            f"T=({bias['Tx']:+.3f},{bias['Ty']:+.3f},{bias['Tz']:+.3f})Nm")
        log(f"Drift Fx={drift['Fx_per_s']:+.3f} Fy={drift['Fy_per_s']:+.3f} Fz={drift['Fz_per_s']:+.3f}  N/s")
        if passed:
            log("PASS — sensor healthy")
        else:
            log(f"FAIL — bias_failures={len(bias_fail)} drift_failures={len(drift_fail)}")
            if bias_fail:
                log(f"  bias > limit: {bias_fail}")
            if drift_fail:
                log(f"  drift > limit: {drift_fail}")

    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
