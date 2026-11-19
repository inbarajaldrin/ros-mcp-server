import ast
from pathlib import Path

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


def test_sim_wrist_reaction_stop_matches_real_20n_delta():
    source = MOVE_DOWN.read_text()
    assert "SIM_Z_FORCE_THRESHOLD = 20.0" in source
    assert "FORCE_THRESHOLD = 20.0" in source


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
    source = MOVE_DOWN.read_text()
    assert "end = time.time() + 0.35" in source
    assert "m = self._median_wrench(150)" in source
    assert "CONTACT_MEDIAN_N = 51" in source
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
