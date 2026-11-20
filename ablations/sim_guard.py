"""Refuse to run a sim entry point anywhere but the DroidStack simulator's own ROS graph.

The sim replay and the sim seat scorer run the cell's primitives in mode sim.  Outside a DroidStack
simulator the robot contract falls back to the real driver's names, so the same command on the cell PC
would drive the REAL arm, with sim-only code paths: an insert would descend with no force guard, since
wrench_sim is absent there.  Each entry point therefore proves where it is before the first command:

1. environment: ROS_MCP_DROIDSTACK_SIM=1, ROS_LOCALHOST_ONLY=1, and ROS_DOMAIN_ID equal to the declared
   sim domain ROS_MCP_SIM_DOMAIN_ID.  The declared domain must be a non-default integer (the real stack
   runs on the default domain 0);
2. graph: the DroidStack arm action /<instance>/arm_controller/follow_joint_trajectory and the
   /<instance>/joint_states topic are present, and the real driver's
   /scaled_joint_trajectory_controller is absent.

The graph is read only after the environment passes, so a misconfigured shell never joins a domain.
"""
from __future__ import annotations

import os
import time

REAL_ARM_CONTROLLER = "/scaled_joint_trajectory_controller"


def sim_environment_refusal(env=None) -> str | None:
    env = os.environ if env is None else env
    if env.get("ROS_MCP_DROIDSTACK_SIM") != "1":
        return "NOT_SIM: ROS_MCP_DROIDSTACK_SIM is not 1 (the robot contract would use the real driver's names)"
    if env.get("ROS_LOCALHOST_ONLY") != "1":
        return "NOT_SIM: ROS_LOCALHOST_ONLY is not 1 (the ROS graph would reach the network)"
    declared = env.get("ROS_MCP_SIM_DOMAIN_ID", "")
    domain = env.get("ROS_DOMAIN_ID", "")
    if not declared.isdigit() or int(declared) == 0:
        return "NOT_SIM: ROS_MCP_SIM_DOMAIN_ID must declare a non-default ROS domain for the simulator"
    if domain != declared:
        return f"NOT_SIM: ROS_DOMAIN_ID {domain or '(unset)'} is not the declared sim domain {declared}"
    return None


def sim_graph_refusal(names, instance: str) -> str | None:
    """`names`: the topic names on the graph (an action shows as <action>/_action/status)."""
    names = set(names)
    if any(n.startswith(REAL_ARM_CONTROLLER + "/") for n in names):
        return f"NOT_SIM: the real driver's {REAL_ARM_CONTROLLER} is on this ROS graph"
    action = f"/{instance}/arm_controller/follow_joint_trajectory/_action/status"
    if action not in names:
        return f"NOT_SIM: the DroidStack arm action /{instance}/arm_controller/follow_joint_trajectory is absent"
    if f"/{instance}/joint_states" not in names:
        return f"NOT_SIM: the DroidStack /{instance}/joint_states topic is absent"
    return None


def ros_graph_topic_names(wait_s: float = 10.0, instance: str = "ur5e_0") -> set[str]:
    """Topic names seen on the current domain, polled until the sim action AND its joint states appear or `wait_s`
    passes (discovery is not atomic: stopping at the action alone refused a twin run whose /<instance>/joint_states
    arrived a moment later, w8 2026-11-20).  The real driver is judged on EVERY snapshot (review r1 P2): any of its
    topics seen while polling is kept in the returned set (so sim_graph_refusal refuses) and the probe stops there;
    the sim topics must be in the last snapshot."""
    import rclpy
    from rclpy.context import Context

    context = Context()
    rclpy.init(context=context)
    try:
        node = rclpy.create_node("sim_guard_probe", context=context)
        want = {f"/{instance}/arm_controller/follow_joint_trajectory/_action/status", f"/{instance}/joint_states"}
        deadline = time.monotonic() + wait_s
        names: set[str] = set()
        real_seen: set[str] = set()
        while True:
            names = {n for n, _types in node.get_topic_names_and_types()}
            real_seen |= {n for n in names if n.startswith(REAL_ARM_CONTROLLER + "/")}
            if real_seen or want <= names or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        node.destroy_node()
        return names | real_seen
    finally:
        rclpy.shutdown(context=context)


def require_sim_container(env=None, graph_names=None) -> str | None:
    """None when this process is proven to be in the simulator; else the typed refusal.
    `graph_names` (a callable returning topic names) is injectable for tests."""
    env = os.environ if env is None else env
    why = sim_environment_refusal(env)
    if why:
        return why
    instance = env.get("ROS_MCP_ROBOT_INSTANCE", "ur5e_0")
    names = (graph_names or (lambda: ros_graph_topic_names(instance=instance)))()
    return sim_graph_refusal(names, instance)
