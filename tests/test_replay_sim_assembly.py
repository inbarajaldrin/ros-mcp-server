from pathlib import Path

import json
import os
import subprocess
import sys

import pytest

from ablations import replay_sim_assembly, sim_guard
from ablations.replay_sim_assembly import _command_for_step, _result_from_output

ROOT = Path(__file__).resolve().parents[1]
SIM_ENV = {"ROS_MCP_DROIDSTACK_SIM": "1", "ROS_LOCALHOST_ONLY": "1", "ROS_DOMAIN_ID": "231",
           "ROS_MCP_SIM_DOMAIN_ID": "231"}
SIM_GRAPH = {"/ur5e_0/arm_controller/follow_joint_trajectory/_action/status", "/ur5e_0/joint_states",
             "/ur5e_0/rg2/grip/_action/status", "/clock"}


def test_commands_preserve_recorded_sim_calls():
    cmd, timeout = _command_for_step(
        "ros-mcp-server__translate_object(action='insert', object_name='u_brown', "
        "base_name='base1', mode='sim')"
    )
    assert cmd[-7:] == ["--mode", "sim", "--object-name", "u_brown",
                        "--base-name", "base1", "--insert"]
    assert timeout == 360


def test_non_sim_step_is_refused():
    try:
        _command_for_step("ros-mcp-server__move_home(mode='real')")
    except ValueError as error:
        assert "refuses non-sim" in str(error)
    else:
        raise AssertionError("real step was not refused")


def test_move_home_uses_sim_primitive():
    command, timeout = _command_for_step("ros-mcp-server__move_home(mode='sim')")
    assert command[-4:] == ["primitives.move_home", "--mode", "sim", "--joint-space"]
    assert timeout == 120


def test_last_result_envelope_wins():
    text = "__RESULT_JSON__\n{\"result\":\"failure\"}\n__END_RESULT_JSON__\n" \
           "__RESULT_JSON__\n{\"result\":\"success\"}\n__END_RESULT_JSON__\n"
    assert _result_from_output(text) == {"result": "success"}


@pytest.mark.parametrize("drop,expect", [
    ("ROS_MCP_DROIDSTACK_SIM", "ROS_MCP_DROIDSTACK_SIM"), ("ROS_LOCALHOST_ONLY", "ROS_LOCALHOST_ONLY"),
    ("ROS_DOMAIN_ID", "not the declared sim domain"), ("ROS_MCP_SIM_DOMAIN_ID", "non-default")])
def test_the_sim_environment_is_required(drop, expect):
    env = {k: v for k, v in SIM_ENV.items() if k != drop}
    assert expect in sim_guard.sim_environment_refusal(env)
    assert sim_guard.sim_environment_refusal(SIM_ENV) is None
    assert "non-default" in sim_guard.sim_environment_refusal(dict(SIM_ENV, ROS_DOMAIN_ID="0", ROS_MCP_SIM_DOMAIN_ID="0"))
    assert "not the declared" in sim_guard.sim_environment_refusal(dict(SIM_ENV, ROS_DOMAIN_ID="0"))


def test_the_sim_graph_is_required():
    assert sim_guard.sim_graph_refusal(SIM_GRAPH, "ur5e_0") is None
    real = SIM_GRAPH | {"/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status"}
    assert "real driver" in sim_guard.sim_graph_refusal(real, "ur5e_0")
    assert "arm action" in sim_guard.sim_graph_refusal({"/joint_states", "/ur5e_0/joint_states"}, "ur5e_0")
    no_states = SIM_GRAPH - {"/ur5e_0/joint_states"}
    assert "joint_states" in sim_guard.sim_graph_refusal(no_states, "ur5e_0")


def test_the_graph_is_not_read_until_the_environment_passes():
    def probe():
        raise AssertionError("joined a ROS domain from an unproven shell")
    assert sim_guard.require_sim_container({}, graph_names=probe).startswith("NOT_SIM")
    assert sim_guard.require_sim_container(SIM_ENV, graph_names=lambda: SIM_GRAPH) is None


@pytest.mark.parametrize("env", [{}, {"ROS_MCP_MODE": "sim"}, dict(SIM_ENV, ROS_MCP_DROIDSTACK_SIM="0")])
def test_the_replay_refuses_before_any_command_outside_the_simulator(env, monkeypatch, capsys, tmp_path):
    for key in list(os.environ):
        if key.startswith("ROS_"):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(replay_sim_assembly, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a primitive")))
    monkeypatch.setattr(sim_guard, "ros_graph_topic_names",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("read the graph")))
    rc = replay_sim_assembly.main(["--assembly-json", str(tmp_path / "unused.json"),
                                   "--log-dir", str(tmp_path / "logs")])
    assert rc == 3
    assert json.loads(capsys.readouterr().out)["replay"] == "refused"
    assert not (tmp_path / "logs").exists()


def test_the_replay_refuses_on_the_real_graph(monkeypatch, capsys, tmp_path):
    for key, value in SIM_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sim_guard, "ros_graph_topic_names", lambda *a, **k: SIM_GRAPH | {
        "/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status"})
    monkeypatch.setattr(replay_sim_assembly, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a primitive")))
    assert replay_sim_assembly.main(["--assembly-json", str(tmp_path / "unused.json")]) == 3
    assert "real driver" in capsys.readouterr().out


def test_the_scorer_refuses_outside_the_simulator(monkeypatch, capsys):
    from ablations import sim_seat_scorer
    monkeypatch.setenv("ROS_MCP_MODE", "sim")
    monkeypatch.delenv("ROS_MCP_DROIDSTACK_SIM", raising=False)
    assert sim_seat_scorer.main(["--observe", "u_brown"]) == 2
    assert "NOT_SIM" in capsys.readouterr().out


def test_the_replay_has_no_real_mode_command():
    assert "--mode\", \"real" not in (ROOT / "ablations/replay_sim_assembly.py").read_text()
    assert "--fov-gated" not in (ROOT / "ablations/replay_sim_assembly.py").read_text()
    for step in ("ros-mcp-server__move_home(mode='sim')", "ros-mcp-server__control_gripper(command='open', mode='sim')",
                 "ros-mcp-server__move_to_safe_height(object_name='u_brown', mode='sim')"):
        cmd, _ = _command_for_step(step, "u_brown", 1)
        assert cmd[cmd.index("--mode") + 1] == "sim"


@pytest.mark.parametrize("flag,arm,states", [
    (None, "/scaled_joint_trajectory_controller/follow_joint_trajectory", "/joint_states"),
    ("0", "/scaled_joint_trajectory_controller/follow_joint_trajectory", "/joint_states"),
    ("1", "/ur5e_0/arm_controller/follow_joint_trajectory", "/ur5e_0/joint_states")])
def test_robot_contract_names_follow_the_sim_flag(flag, arm, states):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROS_MCP_")}
    if flag is not None:
        env["ROS_MCP_DROIDSTACK_SIM"] = flag
    out = subprocess.run([sys.executable, "-c", "from primitives.shared import robot_contract as c; "
                          "print(c.ARM_ACTION); print(c.JOINT_STATES_TOPIC)"],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout.split()
    assert out == [arm, states]


def test_regular_sim_insert_descends_to_the_physical_contact_floor():
    source = (Path(__file__).resolve().parents[1] / "primitives/translate_object.py").read_text()
    assert "run_move_down(mode='sim', to_table=True, require_contact=True)" in source
