import ast
from pathlib import Path


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


def test_sim_wrist_reaction_stop_matches_real_20n_delta():
    source = MOVE_DOWN.read_text()
    assert "SIM_Z_FORCE_THRESHOLD = 20.0" in source
    assert "FORCE_THRESHOLD = 20.0" in source
