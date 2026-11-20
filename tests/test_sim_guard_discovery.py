"""ros_graph_topic_names waits for BOTH the sim arm action and /<instance>/joint_states (discovery is not atomic: a
twin run, w8 2026-11-20, was refused NOT_SIM on joint_states right after the same guard had passed), and still
returns within its deadline when one never appears (the refusal stays the caller's)."""
import sys
import types

from ablations import sim_guard

ACTION = "/ur5e_0/arm_controller/follow_joint_trajectory/_action/status"
JOINTS = "/ur5e_0/joint_states"


def _fake_rclpy(monkeypatch, discovered):
    """rclpy whose node's graph shows discovered[i] on the i-th read (the last one from then on)."""
    reads = []

    class _Node:
        def get_topic_names_and_types(self):
            reads.append(1)
            return [(n, ["t"]) for n in discovered[min(len(reads), len(discovered)) - 1]]

        def destroy_node(self):
            pass

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda context=None: None
    rclpy.shutdown = lambda context=None: None
    rclpy.create_node = lambda name, context=None: _Node()
    ctx = types.ModuleType("rclpy.context")
    ctx.Context = object
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.context", ctx)
    monkeypatch.setattr(sim_guard.time, "sleep", lambda s: None)
    return reads


def test_the_probe_waits_for_joint_states_after_the_action(monkeypatch):
    reads = _fake_rclpy(monkeypatch, [[ACTION], [ACTION], [ACTION, JOINTS]])
    names = sim_guard.ros_graph_topic_names(wait_s=10.0)
    assert {ACTION, JOINTS} <= names and len(reads) == 3
    assert sim_guard.sim_graph_refusal(names, "ur5e_0") is None


def test_the_probe_gives_up_at_its_deadline_and_the_refusal_names_the_missing_topic(monkeypatch):
    _fake_rclpy(monkeypatch, [[ACTION]])
    names = sim_guard.ros_graph_topic_names(wait_s=0.0)
    assert sim_guard.sim_graph_refusal(names, "ur5e_0").startswith("NOT_SIM: the DroidStack /ur5e_0/joint_states")
