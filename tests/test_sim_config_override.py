import os
from pathlib import Path
import subprocess
import sys


def test_localizer_config_can_be_pinned_for_isolated_sim():
    root = Path(__file__).resolve().parents[1]
    config = root / "ablations/ground_truth_resources/robot_config_v31.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["FMB_LOCALIZER_CONFIG"] = str(config)
    proc = subprocess.run(
        [sys.executable, "-c",
         "from primitives.shared.config import LOCALIZER_CONFIG; "
         "from primitives.shared.camera_up import CAMERA_SIDE_TOOL; "
         "print(LOCALIZER_CONFIG, CAMERA_SIDE_TOOL)"],
        cwd=root, env=env, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert str(config) in proc.stdout
    assert "0.999" in proc.stdout
