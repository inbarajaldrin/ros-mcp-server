import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "ablations/ground_truth_resources/robot_config_v31.yaml"


def _config(mode):
    env = {k: v for k, v in os.environ.items() if k not in ("ROS_MCP_MODE", "FMB_LOCALIZER_CONFIG")}
    env.update(PYTHONPATH=str(ROOT), FMB_LOCALIZER_CONFIG=str(PINNED))
    if mode is not None:
        env["ROS_MCP_MODE"] = mode
    return subprocess.run([sys.executable, "-c", "from primitives.shared.config import LOCALIZER_CONFIG; "
                           "print(LOCALIZER_CONFIG)"], cwd=ROOT, env=env, text=True, capture_output=True)


def test_localizer_config_can_be_pinned_for_isolated_sim():
    env = dict(os.environ, PYTHONPATH=str(ROOT), FMB_LOCALIZER_CONFIG=str(PINNED), ROS_MCP_MODE="sim")
    proc = subprocess.run(
        [sys.executable, "-c",
         "from primitives.shared.config import LOCALIZER_CONFIG; "
         "from primitives.shared.camera_up import CAMERA_SIDE_TOOL; "
         "print(LOCALIZER_CONFIG, CAMERA_SIDE_TOOL)"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert str(PINNED) in proc.stdout
    assert "0.999" in proc.stdout
    assert "IGNORED" not in proc.stderr


@pytest.mark.parametrize("mode", [None, "real"])
def test_a_pinned_localizer_config_is_ignored_outside_sim(mode):
    """Review P2-4: a pin leaked into a real-run shell must not replace the live calibration."""
    proc = _config(mode)
    assert proc.returncode == 0, proc.stderr
    assert str(PINNED) not in proc.stdout
    assert proc.stdout.strip().endswith("aruco_camera_localizer/config/robot_config.yaml")
    assert "FMB_LOCALIZER_CONFIG" in proc.stderr and "IGNORED" in proc.stderr
