#!/usr/bin/env python3
"""Execute one planned move of a seqplan plan: exact joint waypoints, no IK, no wrapping (R29, lead 2026-11-19).

The plan's joint values are sent exactly as planned.  move_joints wraps each target to the shortest path, which
sends wrist_3 the other way round (into the cable window or the -360 deg limit), so this never wraps: the
planner's absolute values are the ones its cable window and travel budget were checked on.

- Gate (review r1 P1-1): the goal goes to the real arm's controller unless ROS_MCP_DROIDSTACK_SIM=1
  (primitives.shared.robot_contract); a real-arm target is refused unless ROS_MCP_SEQPLAN_REAL=1, whatever --mode
  says, and --mode must name the target it goes to.  URSim uses the real controller names: it needs the variable too.
- The move JSON states what it is: {"label": str, "waypoints": [[6 joint values, rad], ...], "dense": bool,
  "contact": bool, "ends_in_grip": bool [, "opening_m": raw m, "close_width_m": raw m]}.  The three flags are
  required (review r1 P2-2; ends_in_grip: lead 2026-11-20, r4 P2-1): a move that ends where a Grip closes has its
  final stance judged at the close width too (gripper_rules.grasp_stance_refusal; close_width_m None = fully closed).
- No contact stop here (review r1 P1-5): a move that declares contact, or that ends lower than it starts with the
  fingertips under CONTACT_CLEAR_M above the table, is refused.  Fly a descent toward a part or the table with
  move_down (its force stop), as move_to_grasp and the place-down passes do.
- Checks before anything is sent (review r1 P2-3, P3-2): every waypoint finite and inside the hard limits (joint
  limits, |wrist_3|, the cable window; seqplan.rules.waypoint_refusal); a dense leg's steps at most DENSE_STEP_MAX_RAD;
  the final stance against the camera rule (camera_up.camera_stance_refusal) and the gripper vs the table
  (gripper_rules.gripper_table_refusal).  The route itself is the caller's to check with seqplan.rules.route_refusal.
- Start-stance check (review r1 P2-1): the measured joints within START_TOL_RAD of the first waypoint on every joint
  and within START_TOL_M at the gripper centre (unwrapped); a dense leg then starts from the measured joints.
- A route (a few stops: direct, split, detour, via home) is sent as one goal with a stop at every waypoint: zero
  velocity at each point, so every segment is the joint-linear motion the planner checked.
- A dense leg (the Cartesian legs: lift, retreat, the approach to a hover) is sent with move_down's s-curve profile
  over its waypoints, timed on the gripper centre's path at CART_VEL_LIMIT (move_down's and move_to_safe_height's
  Cartesian rate) as well as on joint distance, whichever is slower (review r1 P1-5); no 10 s cap (P3-4).
- The result wait has a budget (duration + max(RESULT_MARGIN_S, half the duration)); past it the move fails and
  goal_guard cancels and confirms the goal, as on any exit.

Usage: python move_plan.py <plan move json> [--mode sim|real]
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

START_TOL_RAD = 0.003          # per joint (review r1 P2-1: 0.01 rad was 15-20 mm at the gripper centre)
START_TOL_M = 0.003            # at the gripper centre
DENSE_STEP_MAX_RAD = 0.02      # a dense leg's samples: at most this far apart on any joint
CONTACT_CLEAR_M = 0.050        # a descent whose fingertips end closer than this to the table needs a contact stop
RESULT_MARGIN_S = 10.0


def target_refusal(mode, droidstack_sim, env):
    """None, or why move_plan must not send: the gate keys on where the goal goes (review r1 P1-1)."""
    if mode not in ("sim", "real"):
        return f"MODE_INVALID: {mode!r}"
    if droidstack_sim:
        if mode != "sim":
            return "MODE_TARGET_MISMATCH: --mode real, but ROS_MCP_DROIDSTACK_SIM=1 sends the goal to the simulator"
        return None
    if mode != "real":
        return ("MODE_TARGET_MISMATCH: --mode sim, but without ROS_MCP_DROIDSTACK_SIM=1 the goal goes to the real "
                "arm's /scaled_joint_trajectory_controller")
    if env.get("ROS_MCP_SEQPLAN_REAL") != "1":
        return "SEQPLAN_REAL_NOT_APPROVED: the planner's real-path use waits for review (ROS_MCP_SEQPLAN_REAL=1)"
    return None


def _gc(q):
    from primitives.shared.config import GRIPPER_CENTER_TOOL_OFFSET
    from primitives.shared.ik import dh_params, forward_kinematics
    T = forward_kinematics(dh_params, np.asarray(q, float))
    return T[:3, 3] + T[:3, :3] @ np.asarray(GRIPPER_CENTER_TOOL_OFFSET, float)


def move_refusal(move):
    """None, or why this move JSON must not be sent (shape, flags, limits, the final stance, contact)."""
    from primitives.seqplan.rules import waypoint_refusal
    for key in ("dense", "contact", "ends_in_grip"):
        if not isinstance(move.get(key), bool):
            return f"MOVE_FLAG_MISSING: '{key}' must be true or false"
    W = move.get("waypoints")
    if not isinstance(W, list) or len(W) < 2:
        return "PLAN_MOVE_EMPTY"
    for w in W:
        why = waypoint_refusal(w)
        if why:
            return why
    W = [np.asarray(w, float) for w in W]
    if move["dense"]:
        step = max(float(np.max(np.abs(b - a))) for a, b in zip(W[:-1], W[1:]))
        if step > DENSE_STEP_MAX_RAD + 1e-9:
            return f"DENSE_STEP: {step:.3f} rad between samples (max {DENSE_STEP_MAX_RAD})"
    if move["contact"]:
        return "CONTACT_LEG: move_plan has no contact stop; fly this leg with move_down"
    from primitives.shared.camera_up import camera_stance_refusal
    from primitives.shared.config import TABLE_HEIGHT
    from primitives.shared.gripper_rules import gripper_points_tool, gripper_table_refusal
    from primitives.shared.ik import dh_params, forward_kinematics
    T_end = forward_kinematics(dh_params, W[-1])
    why = camera_stance_refusal(T_end) or gripper_table_refusal(T_end, move.get("opening_m"))
    if not why and move["ends_in_grip"]:
        from primitives.shared.gripper_rules import grasp_stance_refusal     # review r3 P1-1: judged at the close
        why = grasp_stance_refusal(T_end, move.get("close_width_m"))
    if why:
        return f"{why} (final stance)"
    z_tips = float((gripper_points_tool(move.get("opening_m")) @ T_end[2, :3] + T_end[2, 3]).min()) - TABLE_HEIGHT
    if _gc(W[-1])[2] < _gc(W[0])[2] - 0.005 and z_tips < CONTACT_CLEAR_M:
        return (f"DESCENT_NEEDS_CONTACT_STOP: the move descends to fingertips {1000 * z_tips:.0f} mm above the table "
                f"(< {1000 * CONTACT_CLEAR_M:.0f}); fly it with move_down")
    return None


def plan_trajectory(waypoints, dense, time_scale=1.0):
    """[(positions, velocities, t)] for the waypoints, exact values (never wrapped).  time_scale > 1 stretches
    the timing (a diagnostic: 2 = half the joint speed, a quarter of the acceleration)."""
    from primitives.shared.config import CART_VEL_LIMIT
    from primitives.shared.velocity_profiles import compute_duration, s_curve_profile
    W = [np.asarray(w, float) for w in waypoints]
    if dense:
        joint_dist = float(np.max(np.abs(W[-1] - W[0])))
        path = float(sum(np.max(np.abs(b - a)) for a, b in zip(W[:-1], W[1:])))
        P = [_gc(w) for w in W]
        cart = float(sum(np.linalg.norm(b - a) for a, b in zip(P[:-1], P[1:])))
        total = compute_duration(joint_distance=max(joint_dist, path), cartesian_distance=cart, profile="s_curve",
                                 cart_vel=CART_VEL_LIMIT, max_duration=math.inf)
        return [(list(map(float, p)), list(map(float, np.asarray(v) / time_scale)), float(t) * time_scale)
                for p, v, t in s_curve_profile(W, total)]
    out, t = [], 0.0
    for a, b in zip(W[:-1], W[1:]):
        t += compute_duration(joint_distance=float(np.max(np.abs(b - a))), profile="s_curve",
                              max_duration=math.inf) * time_scale
        out.append((list(map(float, b)), [0.0] * 6, t))
    return out


def start_refusal(measured, planned_start, tol=START_TOL_RAD, tol_m=START_TOL_M):
    """None, or a typed reason: the measured joints are not the plan's start (unwrapped comparison, per joint and at
    the gripper centre)."""
    m, p = np.asarray(measured, float), np.asarray(planned_start, float)
    if m.shape != (6,) or p.shape != (6,) or not (np.all(np.isfinite(m)) and np.all(np.isfinite(p))):
        return f"PLAN_START_INVALID: measured {m.tolist()}, planned {p.tolist()}"
    d = np.abs(m - p)
    if float(d.max()) > tol:
        j = int(np.argmax(d))
        return (f"PLAN_START_MISMATCH: joint {j} measured {math.degrees(m[j]):+.2f} deg, plan starts at "
                f"{math.degrees(p[j]):+.2f} deg ({math.degrees(d[j]):.2f} deg apart, unwrapped; tolerance "
                f"{math.degrees(tol):.2f}); not sent, re-plan")
    off = float(np.linalg.norm(_gc(m) - _gc(p)))
    if off > tol_m:
        return (f"PLAN_START_MISMATCH: the gripper centre is {1000 * off:.1f} mm from the plan's start (tolerance "
                f"{1000 * tol_m:.0f}); not sent, re-plan")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("move_json")
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--time-scale", type=float, default=1.0, help="sim diagnostic: stretch the timing (>= 1)")
    args = ap.parse_args()
    move = json.load(open(args.move_json))
    result = {"primitive": "move_plan", "label": move.get("label"), "success": False, "error": None}
    from primitives.shared.robot_contract import DROIDSTACK_SIM as _SIM_TARGET
    why = target_refusal(args.mode, _SIM_TARGET, os.environ) or move_refusal(move)
    if why:
        result["error"] = why
        _emit(result)
        return 1
    W = [np.asarray(w, float) for w in move["waypoints"]]

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
        dense = move["dense"]
        if dense:
            W = [state["q"].copy()] + W[1:]                   # start the dense leg where the arm is (review r1 P2-1)
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
        import time as _time
        budget = result["duration_s"] + max(RESULT_MARGIN_S, 0.5 * result["duration_s"])
        t_sent = _time.monotonic()
        client.send_goal_async(goal).add_done_callback(on_goal)
        while rclpy.ok() and not state["done"]:
            rclpy.spin_once(node, timeout_sec=0.1)
            if _time.monotonic() - t_sent > budget:
                raise RuntimeError(f"RESULT_TIMEOUT: no result {budget:.1f} s after sending (the goal is cancelled)")
        if not state["done"]:
            raise RuntimeError("ROS_SHUTDOWN: rclpy stopped before the trajectory reported a result")
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
