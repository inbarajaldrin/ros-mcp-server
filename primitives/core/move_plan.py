#!/usr/bin/env python3
"""Execute one planned move of a seqplan plan: exact joint waypoints, no IK, no wrapping (R29, lead 2026-11-19).

The plan's joint values are sent exactly as planned.  move_joints wraps each target to the shortest path, which
sends wrist_3 the other way round (into the cable window or the -360 deg limit), so this never wraps: the
planner's absolute values are the ones its cable window and travel budget were checked on.

- Start-stance check: the measured joints must equal the plan's first waypoint within START_TOL_RAD, compared as
  unwrapped values; otherwise the move is refused, typed, and nothing is sent (the plan is stale: re-plan).
- A route (a few stops: direct, split, detour, via home) is sent as one goal with a stop at every waypoint: zero
  velocity at each point, so every segment is the joint-linear motion the planner checked.
- A dense leg (the Cartesian legs: hover -> grasp, lift, release, seat) is sent with move_down's s-curve profile over
  its waypoints (the checked samples).
- goal_guard cancels and confirms the goal on any exit.

Usage: python move_plan.py <plan move json> [--mode sim|real]
  The JSON: {"label": str, "waypoints": [[6 joint values, rad], ...], "dense": bool}
Real mode refuses unless ROS_MCP_SEQPLAN_REAL=1 (the planner's real-path use waits for review).
"""
import argparse
import json
import math
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np  # noqa: E402

START_TOL_RAD = 0.01           # ~0.6 deg: the arm settles well inside this at the end of a planned move
DENSE_MIN_POINTS = 8           # a move with this many waypoints or more is a dense (Cartesian) leg


def plan_trajectory(waypoints, dense, time_scale=1.0):
    """[(positions, velocities, t)] for the waypoints, exact values (never wrapped).  time_scale > 1 stretches
    the timing (a diagnostic: 2 = half the joint speed, a quarter of the acceleration)."""
    from primitives.shared.velocity_profiles import compute_duration, s_curve_profile
    W = [np.asarray(w, float) for w in waypoints]
    if dense:
        joint_dist = float(np.max(np.abs(W[-1] - W[0])))
        path = float(sum(np.max(np.abs(b - a)) for a, b in zip(W[:-1], W[1:])))
        total = compute_duration(joint_distance=max(joint_dist, path), profile="s_curve")
        return [(list(map(float, p)), list(map(float, np.asarray(v) / time_scale)), float(t) * time_scale)
                for p, v, t in s_curve_profile(W, total)]
    out, t = [], 0.0
    for a, b in zip(W[:-1], W[1:]):
        t += compute_duration(joint_distance=float(np.max(np.abs(b - a))), profile="s_curve") * time_scale
        out.append((list(map(float, b)), [0.0] * 6, t))
    return out


def start_refusal(measured, planned_start, tol=START_TOL_RAD):
    """None, or a typed reason: the measured joints are not the plan's start (unwrapped comparison)."""
    d = np.abs(np.asarray(measured, float) - np.asarray(planned_start, float))
    if float(d.max()) > tol:
        j = int(np.argmax(d))
        return (f"PLAN_START_MISMATCH: joint {j} measured {math.degrees(measured[j]):+.2f} deg, plan starts at "
                f"{math.degrees(planned_start[j]):+.2f} deg ({math.degrees(d[j]):.2f} deg apart, unwrapped; tolerance "
                f"{math.degrees(tol):.2f}); not sent, re-plan")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("move_json")
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--time-scale", type=float, default=1.0, help="sim diagnostic: stretch the timing (>= 1)")
    args = ap.parse_args()
    move = json.load(open(args.move_json))
    W = move["waypoints"]
    result = {"primitive": "move_plan", "label": move.get("label"), "success": False, "error": None}
    if args.mode == "real" and os.environ.get("ROS_MCP_SEQPLAN_REAL") != "1":
        result["error"] = "SEQPLAN_REAL_NOT_APPROVED: the planner's real-path use waits for review"
        _emit(result)
        return 1
    if len(W) < 2:
        result["error"] = "PLAN_MOVE_EMPTY"
        _emit(result)
        return 1

    import rclpy
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from primitives.shared.goal_guard import init_ros
    from primitives.shared.robot_contract import (ARM_ACTION, DROIDSTACK_SIM, JOINT_STATES_TOPIC, arm_positions,
                                                  public_arm_joints)

    init_ros()
    node = Node("move_plan")
    state = {"q": None, "done": False, "ok": False, "err": None}

    def on_joints(msg):
        p = arm_positions(msg)
        if p is not None:
            state["q"] = np.array(p, float)
    node.create_subscription(JointState, JOINT_STATES_TOPIC, on_joints, 10)
    client = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("UR robot driver isn't running - action server not available")
        for _ in range(100):
            if state["q"] is not None:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        if state["q"] is None:
            raise RuntimeError("Failed to read current joint angles")
        why = start_refusal(state["q"], W[0])
        if why:
            raise RuntimeError(why)
        dense = bool(move.get("dense", len(W) >= DENSE_MIN_POINTS))
        traj = JointTrajectory(joint_names=public_arm_joints())
        if args.time_scale < 1.0 or (args.time_scale != 1.0 and args.mode != "sim"):
            raise RuntimeError("TIME_SCALE_REFUSED: a sim diagnostic, >= 1 only")
        for pos, vel, t in plan_trajectory(W, dense, args.time_scale):
            traj.points.append(JointTrajectoryPoint(positions=pos, velocities=vel,
                                                    time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
        goal = FollowJointTrajectory.Goal(trajectory=traj)
        if not DROIDSTACK_SIM:
            goal.goal_time_tolerance = Duration(sec=1)
        result["duration_s"] = round(traj.points[-1].time_from_start.sec + traj.points[-1].time_from_start.nanosec * 1e-9, 2)
        result["points"] = len(traj.points)

        def on_result(fut):
            r = fut.result()
            state["ok"] = r.status == 4
            state["err"] = None if state["ok"] else f"trajectory ended with status {r.status}, error {r.result.error_code}"
            state["done"] = True

        def on_goal(fut):
            h = fut.result()
            if not h.accepted:
                state["err"] = "Goal rejected"
                state["done"] = True
                return
            h.get_result_async().add_done_callback(on_result)
        client.send_goal_async(goal).add_done_callback(on_goal)
        while rclpy.ok() and not state["done"]:
            rclpy.spin_once(node, timeout_sec=0.1)
        result["success"], result["error"] = state["ok"], state["err"]
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        if state["q"] is not None:
            result["end_error_deg"] = round(float(np.degrees(np.max(np.abs(state["q"] - np.asarray(W[-1]))))), 3)
    except KeyboardInterrupt:
        result["error"] = "Interrupted"
    except Exception as e:                                       # noqa: BLE001 - reported, typed by its text
        result["error"] = str(e)
    finally:
        try:
            node.destroy_node()
        except Exception:                                        # noqa: BLE001
            pass
        try:
            rclpy.shutdown()
        except Exception:                                        # noqa: BLE001
            pass
    _emit(result)
    return 0 if result["success"] else 1


def _emit(result):
    """The result envelope the replays parse (result: success | failure), plus the plain JSON line."""
    result = dict(result, result="success" if result.get("success") else "failure")
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__", flush=True)


if __name__ == "__main__":
    sys.exit(main())
