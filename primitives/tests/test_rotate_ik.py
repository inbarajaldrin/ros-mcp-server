#!/usr/bin/env python3
"""
Test script for rotate_object IK seed analysis.
Queries current/target object poses and joint angles, then simulates
which IK seeds would win in rotate_object.

Usage:
    python3 primitives/tests/test_rotate_ik.py --object fork_orange --base base3
    python3 primitives/tests/test_rotate_ik.py --object fork_yellow --base base3
"""

import argparse
import json
import subprocess
import sys
import os
import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from primitives.shared.ik import dh_params, forward_kinematics

# Joint name order from /joint_states topic → standard order mapping
JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def run_query(script, args):
    """Run a query script and parse JSON output."""
    cmd = f"python3 {PROJECT_ROOT}/{script} {args}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running: {cmd}")
        print(result.stderr)
        sys.exit(1)

    stdout = result.stdout
    if "__RESULT_JSON__" not in stdout:
        print(f"ERROR: No JSON result from {script}")
        print(stdout)
        sys.exit(1)

    start = stdout.rfind("__RESULT_JSON__") + len("__RESULT_JSON__")
    end = stdout.rfind("__END_RESULT_JSON__")
    return json.loads(stdout[start:end].strip())


def get_joint_angles():
    """Get current joint angles from /joint_states topic."""
    cmd = "ros2 topic echo /joint_states sensor_msgs/msg/JointState --once"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("ERROR: Could not read /joint_states")
        print(result.stderr)
        sys.exit(1)

    # Parse YAML-ish output
    lines = result.stdout.strip().split('\n')
    names = []
    positions = []
    in_name = False
    in_position = False

    for line in lines:
        line = line.strip()
        if line.startswith('name:'):
            in_name = True
            in_position = False
            continue
        elif line.startswith('position:'):
            in_name = False
            in_position = True
            continue
        elif line.startswith('velocity:') or line.startswith('effort:'):
            in_name = False
            in_position = False
            continue

        if in_name and line.startswith('- '):
            names.append(line[2:].strip().strip("'\""))
        elif in_position and line.startswith('- '):
            positions.append(float(line[2:].strip()))

    # Reorder to standard order
    joint_map = dict(zip(names, positions))
    ordered = np.array([joint_map[name] for name in JOINT_ORDER])
    return ordered


def ik_objective_quaternion(joint_angles, target_pose, params=None):
    if params is None:
        params = dh_params
    T = forward_kinematics(params, joint_angles)
    pos_err = T[:3, 3] - target_pose[:3, 3]
    q_current = R.from_matrix(T[:3, :3]).as_quat()
    q_target = R.from_matrix(target_pose[:3, :3]).as_quat()
    dot = np.abs(np.dot(q_current, q_target))
    dot = min(dot, 1.0)
    orient_err = 1.0 - dot
    return np.sum(pos_err**2) * 1000 + orient_err * 10


def run_ik_test(q_current, obj_current_quat, obj_target_quat, object_name=None, base_name=None):
    """Run the full rotate_object IK simulation. Returns a log entry dict."""

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "object": object_name,
        "base": base_name,
        "current_joints_deg": np.degrees(q_current).tolist(),
        "obj_current_quat": obj_current_quat.tolist(),
        "obj_target_quat": obj_target_quat.tolist(),
    }

    joint_bounds = [
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-np.pi, np.pi),
        (-2*np.pi, 2*np.pi)
    ]
    acceptable_cost = 0.1
    max_tries = 5
    dx = 0.001

    # Compute target EE pose
    T_EE = forward_kinematics(dh_params, q_current)
    R_EE = T_EE[:3, :3]
    ee_pos = T_EE[:3, 3]

    R_obj_current = R.from_quat(obj_current_quat).as_matrix()
    R_obj_target = R.from_quat(obj_target_quat).as_matrix()
    R_rotation = R_obj_target @ R_obj_current.T
    R_EE_target = R_rotation @ R_EE

    target_pose = np.eye(4)
    target_pose[:3, 3] = ee_pos
    target_pose[:3, :3] = R_EE_target

    target_quat = R.from_matrix(R_EE_target).as_quat()

    # Info
    R_diff = R_EE_target @ R_EE.T
    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))

    print(f"\nCurrent joints (deg): [{', '.join(f'{np.degrees(v):.1f}' for v in q_current)}]")
    print(f"Current EE pos: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
    print(f"Current EE RPY: {R.from_matrix(R_EE).as_euler('xyz', degrees=True).round(1)}")
    print(f"Target  EE RPY: {R.from_matrix(R_EE_target).as_euler('xyz', degrees=True).round(1)}")
    print(f"Angle difference: {np.degrees(angle_diff):.1f}°")

    wrist_2_near_singular = abs(abs(q_current[4]) - np.pi/2) < np.radians(15)
    target_roll_deg = np.degrees(R.from_quat(target_quat).as_euler('xyz')[0])
    use_yaw_flip = 60 < abs(target_roll_deg) < 120

    yaw_rad = np.arctan2(2.0*(target_quat[3]*target_quat[2] + target_quat[0]*target_quat[1]),
                          1.0 - 2.0*(target_quat[1]**2 + target_quat[2]**2))
    yaw_deg = np.degrees(yaw_rad)
    curr_deg = np.degrees(q_current)

    print(f"wrist_2={curr_deg[4]:.1f}° (near ±90°: {wrist_2_near_singular})")
    print(f"Target roll={target_roll_deg:.1f}° (yaw_flip: {use_yaw_flip})")
    print(f"Target yaw for seeds: {yaw_deg:.1f}°")

    log_entry.update({
        "ee_position": ee_pos.tolist(),
        "ee_rpy_current": R.from_matrix(R_EE).as_euler('xyz', degrees=True).tolist(),
        "ee_rpy_target": R.from_matrix(R_EE_target).as_euler('xyz', degrees=True).tolist(),
        "angle_diff_deg": float(np.degrees(angle_diff)),
        "wrist_2_near_singular": bool(wrist_2_near_singular),
        "use_yaw_flip": bool(use_yaw_flip),
        "target_yaw_deg": float(yaw_deg),
    })

    all_results = []  # (phase, name, joints, cost)

    # === PHASE 0: Yaw-adjusted ===
    print(f"\n{'='*70}")
    print("PHASE 0: Yaw-adjusted seed")
    phase0_triggers = wrist_2_near_singular and angle_diff < np.radians(90)
    if phase0_triggers:
        current_yaw = np.arctan2(R_EE[1,0], R_EE[0,0])
        target_yaw_ee = np.arctan2(R_EE_target[1,0], R_EE_target[0,0])
        yaw_diff = target_yaw_ee - current_yaw
        while yaw_diff > np.pi: yaw_diff -= 2*np.pi
        while yaw_diff < -np.pi: yaw_diff += 2*np.pi

        yaw_adjusted_seed = q_current.copy()
        if q_current[4] < 0:
            yaw_adjusted_seed[5] -= yaw_diff
        else:
            yaw_adjusted_seed[5] += yaw_diff

        raw_cost = ik_objective_quaternion(yaw_adjusted_seed, target_pose)
        print(f"  Yaw diff: {np.degrees(yaw_diff):.1f}°, raw cost: {raw_cost:.6f}")
        if raw_cost < 0.1:
            all_results.append(("Phase0", "raw-yaw-adjust", yaw_adjusted_seed.copy(), raw_cost))

        res = minimize(ik_objective_quaternion, yaw_adjusted_seed, args=(target_pose,),
                       method='L-BFGS-B', bounds=joint_bounds,
                       options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 200})
        status = "✓" if res.fun < acceptable_cost else "✗"
        print(f"  Optimized: cost={res.fun:.6f} {status}")
        if res.fun < acceptable_cost:
            all_results.append(("Phase0", "yaw-adjusted-opt", res.x, res.fun))
    else:
        reasons = []
        if not wrist_2_near_singular: reasons.append(f"w2 not near ±90°")
        if angle_diff >= np.radians(90): reasons.append(f"angle_diff={np.degrees(angle_diff):.0f}° >= 90°")
        print(f"  SKIPPED ({', '.join(reasons)})")

    # === PHASE 1: Current joints ===
    print(f"\n{'='*70}")
    print("PHASE 1: Current joints")
    for p in range(max_tries):
        perturbed = target_pose.copy()
        perturbed[0, 3] += p * dx
        res = minimize(ik_objective_quaternion, q_current, args=(perturbed,),
                       method='L-BFGS-B', bounds=joint_bounds,
                       options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 200})
        status = "✓" if res.fun < acceptable_cost else "✗"
        print(f"  Pert {p}: cost={res.fun:.6f} {status}")
        if res.fun < acceptable_cost:
            all_results.append(("Phase1", f"q_guess/pert{p}", res.x, res.fun))

    # === PHASE 2a: Fallback seeds ===
    print(f"\n{'='*70}")
    print("PHASE 2a: Fallback seeds (1 perturbation)")

    seeds_info = [
        ("wrist-only w1=-30 w2=90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, 90, yaw_deg])),
        ("wrist-only w1=0 w2=90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], 0, 90, yaw_deg])),
        ("wrist-only w1=30 w2=90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], 30, 90, yaw_deg])),
        ("wrist-only w1=-30 w2=-90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, -90, yaw_deg])),
        ("wrist-only w1=0 w2=-90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], 0, -90, yaw_deg])),
        ("wrist-only w1=30 w2=-90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], 30, -90, yaw_deg])),
    ]

    if use_yaw_flip:
        yaw_flip = yaw_deg + 180.0 if yaw_deg < 0 else yaw_deg - 180.0
        seeds_info.extend([
            ("yaw-flip w1=-30 w2=90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, 90, yaw_flip])),
            ("yaw-flip w1=0 w2=90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], 0, 90, yaw_flip])),
            ("yaw-flip w1=-30 w2=-90", np.radians([curr_deg[0], curr_deg[1], curr_deg[2], -30, -90, yaw_flip])),
        ])

    seeds_info.extend([
        ("85/-80/90/-90/-90", np.radians([85, -80, 90, -90, -90, yaw_deg])),
        ("90/-90/90/-90/-90", np.radians([90, -90, 90, -90, -90, yaw_deg])),
        ("0/-90/90/-90/-90", np.radians([0, -90, 90, -90, -90, yaw_deg])),
        ("180/-90/90/-90/-90", np.radians([180, -90, 90, -90, -90, yaw_deg])),
        ("85/-80/90/-90/90", np.radians([85, -80, 90, -90, 90, yaw_deg])),
        ("90/-90/90/-90/90", np.radians([90, -90, 90, -90, 90, yaw_deg])),
        ("85/-108/106/2/76", np.radians([85, -108, 106, 2, 76, yaw_deg])),
        ("85/-80/90/90/-90", np.radians([85, -80, 90, 90, -90, yaw_deg])),
    ])

    for name, seed in seeds_info:
        res = minimize(ik_objective_quaternion, seed, args=(target_pose,),
                       method='L-BFGS-B', bounds=joint_bounds,
                       options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 200})
        status = "✓" if res.fun < acceptable_cost else "✗"
        print(f"  {name}: cost={res.fun:.6f} {status}")
        if res.fun < acceptable_cost:
            all_results.append(("Phase2a", name, res.x, res.fun))

    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(all_results)} valid solutions (OLD strategy, all seeds)")
    print(f"{'='*70}")

    if all_results:
        all_results.sort(key=lambda x: x[3])
        print(f"\nTop 5 by cost:")
        for i, (phase, name, joints, cost) in enumerate(all_results[:5]):
            max_change = np.max(np.abs(joints - q_current))
            w2 = np.degrees(joints[4])
            print(f"  {i+1}. [{phase}] {name}: cost={cost:.6f}, max_change={np.degrees(max_change):.1f}°, w2={w2:.1f}°")
            print(f"     joints(deg): [{', '.join(f'{np.degrees(v):.1f}' for v in joints)}]")

    # IK solve counts (old)
    n_phase0 = 2 if phase0_triggers else 0
    n_phase1 = max_tries
    n_phase2_old = len(seeds_info)

    # === NEW strategy: Phase 0 + Phase 1 + two wrist-only seeds (w2=±90) ===
    new_phase2a_seeds = {"wrist-only w1=-30 w2=90", "wrist-only w1=-30 w2=-90"}
    new_results = []  # only from new strategy seeds

    # Phase 0 results carry over
    for r in all_results:
        if r[0] == "Phase0":
            new_results.append(r)
    # Phase 1 results carry over
    for r in all_results:
        if r[0] == "Phase1":
            new_results.append(r)
    # Only the two wrist-only seeds from Phase 2a
    for r in all_results:
        if r[0] == "Phase2a" and r[1] in new_phase2a_seeds:
            new_results.append(r)

    n_phase2_new = len(new_phase2a_seeds)

    # First valid by solve order (old strategy)
    old_first = None
    if phase0_triggers:
        for r in all_results:
            if r[0] == "Phase0":
                old_first = r
                break
    if not old_first:
        for r in all_results:
            if r[0] == "Phase1":
                old_first = r
                break
    if not old_first:
        for name, _ in seeds_info:
            for r in all_results:
                if r[0] == "Phase2a" and r[1] == name:
                    old_first = r
                    break
            if old_first:
                break

    # First valid by solve order (new strategy)
    new_first = None
    if phase0_triggers:
        for r in new_results:
            if r[0] == "Phase0":
                new_first = r
                break
    if not new_first:
        for r in new_results:
            if r[0] == "Phase1":
                new_first = r
                break
    if not new_first:
        for r in new_results:
            if r[0] == "Phase2a":
                new_first = r
                break

    # Comparison
    print(f"\n{'='*70}")
    print("OLD vs NEW STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"  OLD: {len(all_results)} valid solutions, {n_phase0+n_phase1+n_phase2_old} IK solves")
    print(f"  NEW: {len(new_results)} valid solutions, {n_phase0+n_phase1+n_phase2_new} IK solves")

    if old_first:
        print(f"  OLD first: [{old_first[0]}] {old_first[1]}, cost={old_first[3]:.6f}")
    else:
        print(f"  OLD first: NONE")
    if new_first:
        print(f"  NEW first: [{new_first[0]}] {new_first[1]}, cost={new_first[3]:.6f}")
    else:
        print(f"  NEW first: NONE")

    # Flag regressions
    old_found = old_first is not None
    new_found = new_first is not None
    if old_found and not new_found:
        print(f"\n  *** REGRESSION: Old strategy found solution but NEW strategy FAILED ***")
        print(f"  Old winner was: [{old_first[0]}] {old_first[1]}")
        regression = True
    elif old_found and new_found:
        print(f"\n  OK: Both strategies find solutions")
        regression = False
    else:
        print(f"\n  Both strategies failed (no valid solution)")
        regression = False

    # Build log entry
    log_entry["phase0_triggered"] = bool(phase0_triggers)
    log_entry["phase1_any_valid"] = bool(any(r[0] == "Phase1" for r in all_results))
    log_entry["total_valid_solutions"] = len(all_results)
    log_entry["ik_solves"] = {"phase0": n_phase0, "phase1": n_phase1, "phase2a": n_phase2_old, "total": n_phase0 + n_phase1 + n_phase2_old}

    if old_first:
        log_entry["first_valid"] = {"phase": old_first[0], "seed": old_first[1], "cost": float(old_first[3])}
    else:
        log_entry["first_valid"] = None

    # New strategy fields
    log_entry["new_strategy"] = {
        "valid_count": len(new_results),
        "ik_solves": n_phase0 + n_phase1 + n_phase2_new,
        "first_valid": {"phase": new_first[0], "seed": new_first[1], "cost": float(new_first[3])} if new_first else None,
        "regression": regression,
    }

    if all_results:
        all_results.sort(key=lambda x: x[3])
        phase, name, joints, cost = all_results[0]
        log_entry["best_solution"] = {
            "phase": phase,
            "seed": name,
            "cost": float(cost),
            "joints_deg": np.degrees(joints).tolist(),
            "max_joint_change_deg": float(np.degrees(np.max(np.abs(joints - q_current)))),
        }
    else:
        log_entry["best_solution"] = None

    # Log all phase results summary
    log_entry["phase_results"] = {}
    for phase_name in ["Phase0", "Phase1", "Phase2a"]:
        phase_solns = [(name, float(cost)) for p, name, j, cost in all_results if p == phase_name]
        log_entry["phase_results"][phase_name] = {
            "valid_count": len(phase_solns),
            "solutions": phase_solns,
        }

    return log_entry


def main():
    parser = argparse.ArgumentParser(description='Test rotate_object IK seeds')
    parser.add_argument('--object', required=True, help='Object name (e.g. fork_orange)')
    parser.add_argument('--base', default='base3', help='Base name (default: base3)')
    parser.add_argument('--mode', default='sim', help='Mode (default: sim)')
    args = parser.parse_args()

    print(f"Testing rotate_object IK for {args.object} → {args.base}")
    print(f"{'='*70}")

    # Get data
    print("Querying current object pose...")
    obj_data = run_query("queries/get_current_object_pose.py", f"--mode {args.mode} --object-name {args.object}")
    obj_quat = obj_data['object_pose']['orientation']['quat']
    obj_current_quat = np.array([obj_quat['x'], obj_quat['y'], obj_quat['z'], obj_quat['w']])
    obj_pos = obj_data['object_pose']['position']
    print(f"  Position: [{obj_pos['x']:.4f}, {obj_pos['y']:.4f}, {obj_pos['z']:.4f}]")
    print(f"  Quat: {obj_current_quat}")

    print("Querying target object pose...")
    target_data = run_query("queries/get_target_object_pose.py", f"--mode {args.mode} --object-name {args.object} --base-name {args.base}")
    tgt_quat = target_data['target_object_pose']['orientation']['quat']
    obj_target_quat = np.array([tgt_quat['x'], tgt_quat['y'], tgt_quat['z'], tgt_quat['w']])
    tgt_pos = target_data['target_object_pose']['position']
    print(f"  Position: [{tgt_pos['x']:.4f}, {tgt_pos['y']:.4f}, {tgt_pos['z']:.4f}]")
    print(f"  Quat: {obj_target_quat}")

    print("Reading joint angles...")
    q_current = get_joint_angles()
    print(f"  Joints (deg): [{', '.join(f'{np.degrees(v):.1f}' for v in q_current)}]")

    # Run IK test
    log_entry = run_ik_test(q_current, obj_current_quat, obj_target_quat,
                            object_name=args.object, base_name=args.base)

    # Append to log file
    log_file = os.path.join(PROJECT_ROOT, "tests", "rotate_ik_log.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []

    log_data.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"\nLogged to {log_file} (run #{len(log_data)})")


if __name__ == '__main__':
    main()
