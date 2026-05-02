#!/usr/bin/env python3
"""
F/T Payload Calibration — Foundational (Per-Mount, One-Time)
=============================================================

Recovers the gripper+jig payload mass, center-of-mass (CoG) in the F/T sensor
frame, and 6-axis F/T sensor bias by stationary-pose least-squares fitting,
following Kubus, Kröger, Wahl (IROS 2007) — same algorithm used by KTH-RPL's
`force_torque_tools` (ROS1) and the URCap "Measure" wizard.

This is the FOUNDATIONAL calibration (one of three CAL layers — see
.planning/codebase/CONVENTIONS.md §"When to use which calibration layer"
and _references/articles/ft_payload_calibration_math.md for the math).

Procedure:
  1. Move robot through N pre-defined calibration poses (joint-space, safe)
  2. At each pose: settle 1.5 s, sample /wrench for 1 s (averaged), read TCP pose
  3. Compute gravity-in-FT-frame: g_in_ft = R_ee.T @ [0, 0, -9.81]
  4. Build 6×10 measurement matrix per pose per Kubus 2007 stationary form
  5. Stack and solve least-squares: theta = pinv(H_stacked) @ z_stacked
  6. Recover mass, CoG, bias from theta = [m, m·cx, m·cy, m·cz, FBx..z, TBx..z]
  7. Sanity check (residuals, mass vs expected, conditioning) and write YAML

Output:
  compliant_insertion_studio/configs/ft_calibration_<gripper_id>_<YYYYMMDD>.yaml
  with mass, cog, bias, residuals, pose count, pose joint configs, and a
  ready-to-paste `set_target_payload(mass, cog)` line.

Preconditions (operator's responsibility):
  - F/T sensor warm-up complete (≥ 10–30 min powered on)
  - Robot bringup live (scaled_joint_trajectory_controller active)
  - Workspace clear (calibration moves the wrist through varied orientations)
  - Hands off the robot for the entire duration (~3 min for 8 poses)
  - Gripper in target configuration (with whatever jig, no part — for
    canonical "empty gripper" calibration; or with part if you want the
    calibration to include it)

Usage:
  python3 ft_calibration.py --gripper-id robotiq_2f85_with_camera
  python3 ft_calibration.py --gripper-id robotiq_2f85 --expected-mass-kg 1.05 \
                             --num-poses 10 --output-yaml configs/my_cal.yaml

Reference:
  Kubus, D., Kröger, T., Wahl, F. M. (2007). On-line rigid object recognition
  and pose estimation based on inertial parameters. IEEE/RSJ IROS, 1402–1408.
  Algorithm derived from `_references/repos/force_torque_tools/`
  (BSD-3, KTH-RPL, F. Viña 2012).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "compliant_insertion_studio" / "configs"
MOVE_JOINTS_SCRIPT = REPO_ROOT / "primitives" / "core" / "move_joints.py"

WRENCH_TOPIC = "/force_torque_sensor_broadcaster/wrench"
TCP_TOPIC = "/tcp_pose_broadcaster/pose"

GRAVITY_M_S2 = 9.81  # standard gravity, robot base assumed upright

# 8 calibration poses (joint-space, radians).
# Strategy: keep shoulder_pan / shoulder_lift / elbow steady at a known-safe
# config, vary wrist_1 + wrist_2 to point the tool-Z axis in different
# world-frame directions (so gravity-in-FT-frame spans 3D for LSQ conditioning).
# wrist_3 stays at 0 — spinning about tool Z does not change gravity-in-FT.
#
# All poses share base config: shoulder_pan=0, shoulder_lift=-π/2, elbow=π/2.
# Operator should visually verify the first pose looks safe before the run.
CALIBRATION_POSES_RAD = [
    # name,                    [pan,  lift,    elbow,   w1,        w2,        w3]
    ("face_down_canonical",    [0.0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]),
    ("face_right_w2_zero",     [0.0, -math.pi/2, math.pi/2, -math.pi/2,  0.0,       0.0]),
    ("face_up_w1_flipped",     [0.0, -math.pi/2, math.pi/2,  math.pi/2, -math.pi/2, 0.0]),
    ("face_left_w2_pi",        [0.0, -math.pi/2, math.pi/2, -math.pi/2,  math.pi,   0.0]),
    ("tilted_w1_neg_pi4",      [0.0, -math.pi/2, math.pi/2, -math.pi/4, -math.pi/2, 0.0]),
    ("tilted_w1_neg_3pi4",     [0.0, -math.pi/2, math.pi/2, -3*math.pi/4, -math.pi/2, 0.0]),
    ("oblique_w2_neg_pi4",     [0.0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/4, 0.0]),
    ("oblique_w1w2_pi4",       [0.0, -math.pi/2, math.pi/2, -math.pi/4, -math.pi/4, 0.0]),
]


# ---------------------------------------------------------------------------
# ROS node — collects /wrench and /tcp_pose
# ---------------------------------------------------------------------------

class FTCalibrationNode(Node):
    def __init__(self):
        super().__init__("ft_calibration")
        self.tcp = None
        self.wrench_samples = []  # list of (Fx, Fy, Fz, Tx, Ty, Tz)
        self._sampling = False
        self.create_subscription(PoseStamped, TCP_TOPIC, self._tcp_cb, 10)
        self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._wrench_cb, 10)

    def _tcp_cb(self, msg: PoseStamped):
        self.tcp = msg

    def _wrench_cb(self, msg: WrenchStamped):
        if self._sampling:
            f = msg.wrench.force
            t = msg.wrench.torque
            self.wrench_samples.append((f.x, f.y, f.z, t.x, t.y, t.z))

    def wait_for_topics(self, timeout_s: float = 5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.tcp is not None:
                return True
        return False

    def sample_pose(self, hold_s: float = 1.0) -> tuple:
        """Sample wrench for hold_s, return (mean_wrench, ee_quat).

        EE orientation is read AT END of sampling so gravity vector matches
        the wrench data (robot is stationary so this is fine).
        """
        self.wrench_samples = []
        self._sampling = True
        t_end = time.time() + hold_s
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.005)
        self._sampling = False

        if len(self.wrench_samples) < 10:
            raise RuntimeError(
                f"Too few wrench samples ({len(self.wrench_samples)}) — broadcaster slow?"
            )

        arr = np.array(self.wrench_samples)
        mean_wrench = arr.mean(axis=0)  # [Fx, Fy, Fz, Tx, Ty, Tz]

        if self.tcp is None:
            raise RuntimeError("No TCP pose available")
        q = self.tcp.pose.orientation
        ee_quat = np.array([q.x, q.y, q.z, q.w])

        return mean_wrench, ee_quat, len(self.wrench_samples)


# ---------------------------------------------------------------------------
# Calibration math (Kubus 2007 stationary form)
# ---------------------------------------------------------------------------

def gravity_in_ft_frame(ee_quat: np.ndarray) -> np.ndarray:
    """Compute gravity vector in the F/T sensor frame.

    Assumes the F/T sensor frame is colinear with the EE/tool0 frame
    (true for UR5e built-in sensor). If a separate F/T mount rotation
    exists, apply it here.

    Args:
        ee_quat: [x, y, z, w] quaternion of the EE in robot base frame.
    Returns:
        g_in_ft: 3-vector, gravity in F/T frame (units: m/s²).
    """
    R_base_ee = R.from_quat(ee_quat).as_matrix()  # base ← ee
    g_world = np.array([0.0, 0.0, -GRAVITY_M_S2])
    # gravity expressed in EE/FT frame: rotate world vector into EE frame
    g_in_ft = R_base_ee.T @ g_world
    return g_in_ft


def build_measurement_matrix(g: np.ndarray) -> np.ndarray:
    """Build the 6×10 H matrix for one stationary pose.

    Parameter vector: theta = [m, m·cx, m·cy, m·cz, FBx, FBy, FBz, TBx, TBy, TBz]^T
    Measurement: z = [Fx, Fy, Fz, Tx, Ty, Tz]^T (raw F/T reading)

    Stationary specialization (omega = alpha = a = 0) of Kubus 2007:
        F = -m·g + F_bias
        T = -m·(c × g) + T_bias

    Sign convention matches `force_torque_tools/src/ft_calib.cpp` (with our
    a=0 simplification). See _references/articles/ft_payload_calibration_math.md.
    """
    H = np.zeros((6, 10))

    # F = -m·g + F_bias
    H[0, 0] = -g[0]
    H[1, 0] = -g[1]
    H[2, 0] = -g[2]
    H[0, 4] = 1.0  # FBx
    H[1, 5] = 1.0  # FBy
    H[2, 6] = 1.0  # FBz

    # T = -m·(c × g) + T_bias
    # c × g = [c_y·g_z - c_z·g_y, c_z·g_x - c_x·g_z, c_x·g_y - c_y·g_x]
    # so -m·(c × g) = [-m·c_y·g_z + m·c_z·g_y, ...]
    H[3, 2] = -g[2]   # coefficient on m·cy for Tx
    H[3, 3] =  g[1]   # coefficient on m·cz for Tx
    H[4, 1] =  g[2]   # m·cx for Ty
    H[4, 3] = -g[0]   # m·cz for Ty
    H[5, 1] = -g[1]   # m·cx for Tz
    H[5, 2] =  g[0]   # m·cy for Tz
    H[3, 7] = 1.0     # TBx
    H[4, 8] = 1.0     # TBy
    H[5, 9] = 1.0     # TBz

    return H


def solve_calibration(measurements: list) -> dict:
    """Stack measurements and solve LSQ.

    Args:
        measurements: list of (g_in_ft, mean_wrench) tuples
    Returns:
        dict with mass, cog (3-vec), bias_force (3-vec), bias_torque (3-vec),
        residual_per_axis, condition_number, theta
    """
    n = len(measurements)
    H_stacked = np.zeros((6 * n, 10))
    z_stacked = np.zeros(6 * n)

    for i, (g, w) in enumerate(measurements):
        H_stacked[6*i:6*i+6, :] = build_measurement_matrix(g)
        z_stacked[6*i:6*i+6] = w

    # SVD-backed least squares (matches the C++ JacobiSVD)
    theta, residuals_sum, rank, sv = np.linalg.lstsq(H_stacked, z_stacked, rcond=None)

    # Recover physical parameters
    mass = float(theta[0])
    if abs(mass) < 1e-6:
        raise ValueError(f"Recovered mass {mass} is ~0 — pose set is rank-deficient or wrench is zero")
    cog = np.array([theta[1] / mass, theta[2] / mass, theta[3] / mass])
    bias_force = theta[4:7]
    bias_torque = theta[7:10]

    # Per-axis residuals (RMS over all poses)
    z_pred = H_stacked @ theta
    z_residual = z_stacked - z_pred
    # Per-axis: reshape to (n, 6) and RMS over poses
    res_per_axis = np.sqrt((z_residual.reshape(n, 6) ** 2).mean(axis=0))

    return {
        "mass_kg": mass,
        "cog_xyz_m": cog.tolist(),
        "bias_force_N": bias_force.tolist(),
        "bias_torque_Nm": bias_torque.tolist(),
        "residual_per_axis": {
            "Fx_N": float(res_per_axis[0]),
            "Fy_N": float(res_per_axis[1]),
            "Fz_N": float(res_per_axis[2]),
            "Tx_Nm": float(res_per_axis[3]),
            "Ty_Nm": float(res_per_axis[4]),
            "Tz_Nm": float(res_per_axis[5]),
        },
        "matrix_rank": int(rank),
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "theta": theta.tolist(),
    }


# ---------------------------------------------------------------------------
# Pose execution (subprocess to existing primitive)
# ---------------------------------------------------------------------------

def move_to_joint_pose(positions: list, duration_s: float = 6.0, log=None) -> bool:
    """Move robot to target joint positions via primitives/core/move_joints.py.

    Returns True on success.
    """
    cmd = [
        sys.executable,
        str(MOVE_JOINTS_SCRIPT),
        "send",
        "--positions",
    ] + [str(x) for x in positions] + [
        "--duration", str(duration_s),
    ]
    if log:
        log(f"  → move_joints: {' '.join(f'{x:.3f}' for x in positions)} (dur {duration_s:.1f}s)")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_s + 30)
    if proc.returncode != 0:
        if log:
            log(f"  ← move_joints FAILED: {proc.stderr.strip()[:300]}")
        return False
    return True


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def write_yaml(result: dict, args, output_path: Path):
    set_target_payload_line = (
        f"set_target_payload({result['mass_kg']:.4f}, "
        f"[{result['cog_xyz_m'][0]:.4f}, "
        f"{result['cog_xyz_m'][1]:.4f}, "
        f"{result['cog_xyz_m'][2]:.4f}])"
    )
    yaml_doc = {
        "schema_version": 1,
        "calibration_type": "ft_payload_kubus_2007_stationary",
        "gripper_id": args.gripper_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_mass_kg": args.expected_mass_kg,
        "num_poses_used": result["num_poses_used"],
        "pose_names": result["pose_names"],
        "pose_joint_configs_rad": result["pose_joint_configs"],
        "result": {
            "mass_kg": round(result["mass_kg"], 4),
            "cog_xyz_m": [round(x, 4) for x in result["cog_xyz_m"]],
            "bias_force_N": [round(x, 4) for x in result["bias_force_N"]],
            "bias_torque_Nm": [round(x, 4) for x in result["bias_torque_Nm"]],
        },
        "diagnostics": {
            "residual_per_axis": result["residual_per_axis"],
            "matrix_rank": result["matrix_rank"],
            "condition_number": round(result["condition_number"], 2),
        },
        "set_target_payload_line_for_bringup": set_target_payload_line,
        "notes": [
            "Paste set_target_payload(...) into the bringup launch file at robot startup.",
            "Restart bringup once after pasting; payload persists across sessions until gripper changes.",
            "Re-run this calibration whenever the gripper, jig, or sensor mount changes.",
            "See compliant_insertion_studio/docs/ft_calibration_sop.md for the full workflow.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(yaml_doc, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="F/T payload calibration (foundational, per-mount)")
    parser.add_argument("--gripper-id", type=str, required=True,
                        help="Identifier for this gripper config (e.g. 'robotiq_2f85')")
    parser.add_argument("--expected-mass-kg", type=float, default=None,
                        help="Approximate expected mass for sanity check (warns if recovered differs by > 20%%)")
    parser.add_argument("--num-poses", type=int, default=8,
                        help="Number of calibration poses (≥4 minimum, default 8 from builtin set)")
    parser.add_argument("--settle-s", type=float, default=1.5,
                        help="Settle time after each pose move (default 1.5)")
    parser.add_argument("--sample-s", type=float, default=1.0,
                        help="Wrench sampling window per pose (default 1.0)")
    parser.add_argument("--move-duration-s", type=float, default=6.0,
                        help="Trajectory duration per pose (default 6.0 — slow for safety)")
    parser.add_argument("--output-yaml", type=str, default=None,
                        help="Override output path (default auto: configs/ft_calibration_<id>_<date>.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pose plan and exit without moving the robot")
    parser.add_argument("--no-return-home", action="store_true",
                        help="Skip the move-back-to-pose-0 at the end (default returns to pose 0)")
    args = parser.parse_args()

    # Pose set selection
    if args.num_poses > len(CALIBRATION_POSES_RAD):
        parser.error(f"--num-poses {args.num_poses} exceeds builtin set ({len(CALIBRATION_POSES_RAD)}). Edit CALIBRATION_POSES_RAD to add more.")
    if args.num_poses < 4:
        parser.error("--num-poses must be ≥4 for the 10-parameter LSQ to be solvable.")
    poses = CALIBRATION_POSES_RAD[:args.num_poses]

    # Output path
    if args.output_yaml:
        output_path = Path(args.output_yaml)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = CONFIGS_DIR / f"ft_calibration_{args.gripper_id}_{date_str}.yaml"

    # Dry-run: print pose plan and exit
    if args.dry_run:
        print(f"DRY RUN — would execute {len(poses)} calibration poses for gripper '{args.gripper_id}':")
        for i, (name, joints) in enumerate(poses):
            joint_str = " ".join(f"{x:+.3f}" for x in joints)
            print(f"  Pose {i+1}/{len(poses)}: {name:32s}  [{joint_str}]")
        print(f"Output would be written to: {output_path}")
        print(f"Estimated total time: {len(poses) * (args.move_duration_s + args.settle_s + args.sample_s) + 5:.0f} s")
        sys.exit(0)

    # Real run
    rclpy.init()
    node = FTCalibrationNode()
    log = lambda msg: node.get_logger().info(msg)

    if not node.wait_for_topics(timeout_s=5.0):
        log(f"ERROR: timed out waiting for {TCP_TOPIC} — bringup live?")
        rclpy.shutdown()
        sys.exit(2)

    log(f"=== F/T Payload Calibration ===")
    log(f"Gripper ID: {args.gripper_id}")
    log(f"Poses: {len(poses)} (settle {args.settle_s}s, sample {args.sample_s}s, move duration {args.move_duration_s}s)")
    log(f"Expected total runtime: ~{len(poses) * (args.move_duration_s + args.settle_s + args.sample_s) + 5:.0f}s")
    if args.expected_mass_kg is not None:
        log(f"Expected mass for sanity check: {args.expected_mass_kg} kg")
    log(f"Output → {output_path}")
    log("")
    log("Operator: HANDS OFF the robot for the entire duration. Visually verify each pose is safe.")
    log("Starting in 3 seconds…")
    time.sleep(3)

    measurements = []  # list of (g_in_ft, mean_wrench)
    pose_names = []
    pose_joint_configs = []

    try:
        for i, (name, joints) in enumerate(poses):
            log(f"")
            log(f"--- Pose {i+1}/{len(poses)}: {name} ---")
            if not move_to_joint_pose(joints, duration_s=args.move_duration_s, log=log):
                log(f"  Aborting calibration: pose {name} unreachable or motion failed")
                sys.exit(2)

            log(f"  Settling {args.settle_s}s …")
            t_end = time.time() + args.settle_s
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)

            log(f"  Sampling /wrench for {args.sample_s}s …")
            mean_wrench, ee_quat, n_samples = node.sample_pose(hold_s=args.sample_s)
            g_in_ft = gravity_in_ft_frame(ee_quat)
            log(f"  Samples={n_samples}, EE_quat=[{ee_quat[0]:+.3f},{ee_quat[1]:+.3f},{ee_quat[2]:+.3f},{ee_quat[3]:+.3f}]")
            log(f"  g_in_ft=[{g_in_ft[0]:+.3f}, {g_in_ft[1]:+.3f}, {g_in_ft[2]:+.3f}] m/s²")
            log(f"  mean wrench: F=({mean_wrench[0]:+.2f},{mean_wrench[1]:+.2f},{mean_wrench[2]:+.2f})N "
                f"T=({mean_wrench[3]:+.3f},{mean_wrench[4]:+.3f},{mean_wrench[5]:+.3f})Nm")

            measurements.append((g_in_ft, mean_wrench))
            pose_names.append(name)
            pose_joint_configs.append([float(x) for x in joints])

        # Optional return to pose 0
        if not args.no_return_home and len(poses) > 1:
            log("")
            log(f"Returning to pose 0 ({poses[0][0]}) for clean exit …")
            move_to_joint_pose(poses[0][1], duration_s=args.move_duration_s, log=log)

        # Solve
        log("")
        log("=== Solving least-squares ===")
        result = solve_calibration(measurements)
        result["num_poses_used"] = len(measurements)
        result["pose_names"] = pose_names
        result["pose_joint_configs"] = pose_joint_configs

        # Sanity checks
        warnings = []
        if args.expected_mass_kg is not None:
            mass_err_pct = abs(result["mass_kg"] - args.expected_mass_kg) / args.expected_mass_kg * 100
            if mass_err_pct > 20:
                warnings.append(
                    f"Recovered mass {result['mass_kg']:.3f} kg differs from expected "
                    f"{args.expected_mass_kg} kg by {mass_err_pct:.1f}% (> 20% threshold)"
                )
        max_force_residual = max(
            result["residual_per_axis"]["Fx_N"],
            result["residual_per_axis"]["Fy_N"],
            result["residual_per_axis"]["Fz_N"],
        )
        max_torque_residual = max(
            result["residual_per_axis"]["Tx_Nm"],
            result["residual_per_axis"]["Ty_Nm"],
            result["residual_per_axis"]["Tz_Nm"],
        )
        if max_force_residual > 0.5:
            warnings.append(f"Max force residual {max_force_residual:.3f} N > 0.5 N (poor fit)")
        if max_torque_residual > 0.05:
            warnings.append(f"Max torque residual {max_torque_residual:.3f} Nm > 0.05 Nm (poor fit)")
        if result["matrix_rank"] < 10:
            warnings.append(f"H matrix rank {result['matrix_rank']} < 10 — poses too similar; recovered values unreliable")
        if result["condition_number"] > 1000:
            warnings.append(f"Condition number {result['condition_number']:.0f} > 1000 — poses poorly distributed")

        # Report
        log("")
        log("=== Calibration result ===")
        log(f"Mass:        {result['mass_kg']:.4f} kg")
        log(f"CoG (m):     [{result['cog_xyz_m'][0]:+.4f}, {result['cog_xyz_m'][1]:+.4f}, {result['cog_xyz_m'][2]:+.4f}]")
        log(f"Force bias:  [{result['bias_force_N'][0]:+.3f}, {result['bias_force_N'][1]:+.3f}, {result['bias_force_N'][2]:+.3f}] N")
        log(f"Torque bias: [{result['bias_torque_Nm'][0]:+.4f}, {result['bias_torque_Nm'][1]:+.4f}, {result['bias_torque_Nm'][2]:+.4f}] Nm")
        log(f"Residuals:   F max {max_force_residual:.3f} N, T max {max_torque_residual:.4f} Nm")
        log(f"Rank: {result['matrix_rank']}/10, condition number: {result['condition_number']:.1f}")
        for w in warnings:
            log(f"  ⚠ WARNING: {w}")
        log("")

        # Write YAML
        write_yaml(result, args, output_path)
        log(f"Wrote calibration to: {output_path}")

        # Print the bringup line
        set_payload_line = (
            f"set_target_payload({result['mass_kg']:.4f}, "
            f"[{result['cog_xyz_m'][0]:.4f}, {result['cog_xyz_m'][1]:.4f}, {result['cog_xyz_m'][2]:.4f}])"
        )
        log("")
        log("=== NEXT STEP — paste into your UR bringup launch ===")
        log(f"  {set_payload_line}")
        log("Then restart the bringup once. See docs/ft_calibration_sop.md.")

        # Final result JSON for orchestrators
        final = {
            "result": "success" if not warnings else "success_with_warnings",
            "mass_kg": result["mass_kg"],
            "cog_xyz_m": result["cog_xyz_m"],
            "yaml_path": str(output_path),
            "set_target_payload_line": set_payload_line,
            "warnings": warnings,
        }

    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        final = {"result": "failure", "error": f"{type(e).__name__}: {e}"}
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print("__RESULT_JSON__")
    print(json.dumps(final))
    print("__END_RESULT_JSON__")
    sys.exit(0 if final["result"] != "failure" else 1)


if __name__ == "__main__":
    main()
