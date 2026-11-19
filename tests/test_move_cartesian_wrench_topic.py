import ast
from pathlib import Path

MOVE_CARTESIAN = Path(__file__).resolve().parents[1] / "primitives/move_cartesian.py"


def test_real_force_guard_reads_the_real_wrench_topic():
    """Only a DroidStack sim run may read the simulator's wrench_sim topic."""
    source = MOVE_CARTESIAN.read_text()
    tree = ast.parse(source)
    choice = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and ast.unparse(node.test) == "DROIDSTACK_SIM"
        and ast.unparse(node.body) == "WRENCH_TOPIC"
    )
    assert ast.literal_eval(choice.orelse) == "/force_torque_sensor_broadcaster/wrench"
    assert "create_subscription(WrenchStamped, wrench_topic," in source
