import ast
from pathlib import Path

import pytest

from primitives.core.force_filter import median_wrench_window


MOVE_DOWN = Path(__file__).resolve().parents[1] / "primitives/core/move_down.py"


def test_sim_endpoint_without_contact_is_a_failure():
    """The sim must not call an unforced full descent a successful landing."""
    tree = ast.parse(MOVE_DOWN.read_text())
    goal_result = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "goal_result"
    )
    source = ast.get_source_segment(MOVE_DOWN.read_text(), goal_result)
    assert "if self.mode == 'sim'" in source
    assert "without a measured contact" in source
    assert "self.error_message" in source
    execute = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_move_down"
    )
    execute_source = ast.get_source_segment(MOVE_DOWN.read_text(), execute)
    assert "already at its endpoint without a measured contact" in execute_source


def test_sim_contact_stop_is_the_real_one():
    """Lead 2026-11-19: the sim contact-axis stop is the real one - threshold, sign, baseline and median spans."""
    _n, md = _move_down_node()
    assert md.SIM_Z_FORCE_THRESHOLD == md.Z_FORCE_THRESHOLD < 0
    assert md.SIM_CONTACT_MEDIAN_WINDOW_S == md.CONTACT_MEDIAN_N / md.REAL_WRENCH_HZ
    assert md.SIM_BASELINE_WINDOW_S == md.BASELINE_MEDIAN_N / md.REAL_WRENCH_HZ


@pytest.mark.parametrize("resistance_n", [2.0, 4.9, 5.0, 6.0, 20.0])
def test_sim_and_real_make_the_same_contact_decision(resistance_n):
    """The same upward resistance on the contact axis (the tool pointing down: Fz), held for longer than the median
    span, stops sim and real alike: at the real 5 N, not at 20 N."""
    import time
    verdicts = {}
    for mode in ("real", "sim"):
        n, md = _move_down_node(mode=mode)
        n.moving, n.movement_start_time, n.force_check_grace_period = True, time.time() - 10.0, 0.0
        base = (0.0, 0.0, -26.0, 0.0, 0.0, 0.0)
        (n.baseline_force_x, n.baseline_force_y, n.baseline_force_z,
         n.baseline_torque_x, n.baseline_torque_y, n.baseline_torque_z) = base
        loaded = (0.0, 0.0, -26.0 - resistance_n, 0.0, 0.0, 0.0)      # resistance: Fz delta negative
        (n.current_force_x, n.current_force_y, n.current_force_z,
         n.current_torque_x, n.current_torque_y, n.current_torque_z) = loaded
        n.contact_axis, n.contact_direction, n.contact_axis_name = 2, -1.0, "Fz"
        from collections import deque
        n._wbuf = deque([loaded] * 200, maxlen=200)
        n._sim_wbuf = deque([(i * 0.002,) + loaded for i in range(200)], maxlen=2000)
        n.stopped, n._sim_raw_transient_peak_n, n._last_wrench_stamp_s = [], 0.0, 0.4
        n.contact_detected_stop = lambda n=n: n.stopped.append(True)
        n.check_force_during_execution()
        verdicts[mode] = bool(n.stopped)
    assert verdicts["sim"] == verdicts["real"] == (resistance_n >= 5.0)


def test_contact_stop_admits_confirmed_cancel_but_never_aborted():
    """A controller/path abort cannot be promoted by a coincident force sample."""

    source = MOVE_DOWN.read_text()
    tree = ast.parse(source)
    goal_result = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "goal_result"
    )
    terminal_check = next(
        node for node in ast.walk(goal_result)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "status"
        and any(isinstance(op, ast.In) for op in node.ops)
    )
    admitted = {
        ast.unparse(element)
        for comparator in terminal_check.comparators
        if isinstance(comparator, (ast.Tuple, ast.List, ast.Set))
        for element in comparator.elts
    }
    assert admitted == {
        "GoalStatus.STATUS_SUCCEEDED",
        "GoalStatus.STATUS_CANCELED",
    }
    body = ast.get_source_segment(source, goal_result)
    assert "if self.force_threshold_reached" in body
    assert "Trajectory failed with status" in body
    assert "if result.status == 1" not in body


def test_sim_timestamp_window_rejects_a_short_60n_impulse_but_admits_sustained_contact():
    samples = []
    for index in range(21):
        stamp = index * 0.01
        force = 60.0 if 12 <= index <= 14 else 0.0
        samples.append((stamp, 0.0, 0.0, force, 0.0, 0.0, 0.0))
    assert median_wrench_window(samples, 0.100)[2] == 0.0

    for index in range(15, 21):
        stamp = index * 0.01
        samples[index] = (stamp, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0)
    assert median_wrench_window(samples, 0.100)[2] == 25.0


def test_real_force_baseline_and_contact_median_contract_is_unchanged():
    """The real arm's stop: 5 N upward resistance, 20 N off-axis, a 0.35 s settle, a 150-sample baseline median and a
    51-sample contact median."""
    _n, md = _move_down_node()
    assert (md.Z_FORCE_THRESHOLD, md.FORCE_THRESHOLD) == (-5.0, 20.0)
    assert (md.BASELINE_SETTLE_S, md.BASELINE_MEDIAN_N, md.CONTACT_MEDIAN_N) == (0.35, 150, 51)
    source = MOVE_DOWN.read_text()
    assert "m = self._median_wrench(BASELINE_MEDIAN_N)" in source
    assert "m = self._median_wrench(CONTACT_MEDIAN_N)" in source


def test_only_the_insert_requires_contact_and_the_place_down_drop_is_bounded():
    """Lead approval 2026-11-19: a sim place-down accepts its floor endpoint like the real arm; the
    insert keeps the no-contact refusal; the GT release height is refused above the real floor bound."""
    source = MOVE_DOWN.read_text()
    assert source.count("if self.mode == 'sim' and self.require_contact") == 2   # both endpoint refusals
    assert source.count("self.mode == 'sim' and self.require_contact and not self.force_threshold_reached") == 1
    assert "'--require-contact'" in source
    root = MOVE_DOWN.parents[2]
    translate = (root / "primitives/translate_object.py").read_text()
    assert "run_move_down(mode='sim', to_table=True, require_contact=True)" in translate
    assert "['--drop', args.object_name]" in translate
    assert "'--held-quat', *[f\"{float(v):.6f}\" for v in args.current_object_orientation]" in translate
    scorer = (root / "ablations/sim_seat_scorer.py").read_text()
    assert "drop <= TABLE_COLLISION_MARGIN_SIDEWAYS" in scorer
    assert "SIM_PLACE_DOWN_DROP_EXCEEDS_FLOOR" in scorer


def _move_down_node(mode="real", require_contact=False, force_reached=False, at_endpoint=False):
    import types
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives.core import move_down as md
    n = object.__new__(md.MoveDown)
    n.mode, n.require_contact, n.error_message = mode, require_contact, None
    n.force_threshold_reached, n.moving, n._current_goal_handle = force_reached, True, object()
    n.target_height, n.ee_position = 0.10, ([0.0, 0.0, 0.10] if at_endpoint else [0.0, 0.0, 0.30])
    n.stop_force_monitoring = lambda: None
    n.scheduled = []
    n.schedule_next_movement = lambda: n.scheduled.append(True)
    log = types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
    n.get_logger = lambda: log
    return n, md


def _result(status, error_code=0):
    import types
    msg = types.SimpleNamespace(error_code=error_code, error_string="")
    return types.SimpleNamespace(result=lambda: types.SimpleNamespace(status=status, result=msg))



@pytest.mark.parametrize("status,ok", [(4, True), (5, True), (6, False), (2, False), (0, False), (3, False)])
def test_goal_result_admits_only_succeeded_or_canceled_with_a_contact(status, ok):
    """Executed with GoalStatus values: SUCCEEDED/CANCELED plus a force stop are a landing; ABORTED,
    EXECUTING, UNKNOWN and CANCELING are failures even with a coincident force sample."""
    n, _md = _move_down_node(force_reached=True)
    n.goal_result(_result(status))
    assert (n.error_message is None) == ok
    if not ok:
        assert f"status: {status}" in n.error_message


def test_goal_result_names_a_path_tolerance_abort():
    n, md = _move_down_node(force_reached=True)
    n.goal_result(_result(6, md.FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED))
    assert "Velocity or acceleration limits exceeded" in n.error_message


def test_goal_result_sim_endpoint_without_contact_fails_and_real_succeeds():
    n, _md = _move_down_node(mode="sim", require_contact=True, at_endpoint=True)
    n.goal_result(_result(4))
    assert "without a measured contact" in n.error_message
    n, _md = _move_down_node(mode="real", at_endpoint=True)
    n.goal_result(_result(4))
    assert n.error_message is None and not n.scheduled


def test_goal_result_short_of_the_endpoint_continues_the_descent():
    n, _md = _move_down_node(mode="sim", require_contact=True, at_endpoint=False)
    n.goal_result(_result(4))
    assert n.error_message is None and n.scheduled == [True]


def test_the_to_table_floor_follows_the_given_opening():
    """--opening-mm: the face-down floor puts the fingertips AT that opening 1 mm above the table (rg2_geometry); an
    open RG2 reaches lower than the closed-tip floor, which stands without the option (every existing caller)."""
    import numpy as np
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives.core import move_down as md
    from primitives.shared import rg2_geometry as g
    from primitives.shared.config import RG2_TIP_CLOSED_M, TABLE_HEIGHT
    down = np.diag([1.0, -1.0, -1.0])
    closed = md.to_table_floor_z(down)
    assert abs(closed - (TABLE_HEIGHT + md.PLACE_TIP_CLEARANCE_M + RG2_TIP_CLOSED_M)) < 1e-12
    open95 = md.to_table_floor_z(down, 95.2)
    assert abs(open95 - (TABLE_HEIGHT + md.PLACE_TIP_CLEARANCE_M + g.tip_z_tool(0.0952))) < 1e-12
    assert open95 < closed - 0.010                                    # the open fingers sit > 10 mm higher
    assert md.to_table_floor_z(down, 0.0) > closed                    # closed by the mesh: 234.2 mm, deeper


def test_the_to_table_floor_falls_back_to_the_closed_tips_without_the_model(monkeypatch):
    import numpy as np
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives.core import move_down as md
    from primitives.shared import rg2_geometry as g
    monkeypatch.setattr(g, "tip_z_tool", lambda opening_m=None: None)
    down = np.diag([1.0, -1.0, -1.0])
    assert md.to_table_floor_z(down, 95.2) == md.to_table_floor_z(down)
