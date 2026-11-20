"""The no-width --to-table floor from per-opening finger boxes swept across the stroke (review w8-floor-notilt r1 P2),
checked against the EXACT posed finger meshes, and the floor module is pure (P3).

1210 face-down orientations, as the reviewer checked them: 1200 random tilts up to 60 deg about any horizontal axis
(seeded) plus 0/10/20/40/60 deg in the jaw plane (about tool y; the fingers open along tool x) and across it (about
tool x).  The exact lowest RG2 point: the posed link meshes (rg2_geometry.posed) at 60 jaw angles over the whole
stroke, plus the housing and quick-changer corners.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]


def _orientations():
    down = R.from_euler("x", 180, degrees=True)
    out = []
    for axis in ("y", "x"):
        for tilt in (0, 10, 20, 40, 60):
            out.append((R.from_euler(axis, tilt, degrees=True) * down).as_matrix())
    rng = np.random.default_rng(20261120)
    while len(out) < 1210:
        phi = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(0, 60))
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        out.append((R.from_rotvec(axis * tilt) * down).as_matrix())
    return out


@pytest.fixture(scope="module")
def exact_points():
    """Hull vertices of the posed finger meshes at 60 jaw angles (the lowest point of a linear height is a hull
    vertex) plus the housing / quick-changer corners, in the flange frame."""
    from primitives.shared import rg2_geometry as g
    from primitives.shared.gripper_rules import _housing_points
    pts = []
    for a in np.linspace(g.ALPHA_MIN, g.ALPHA_MAX, 60):
        posed = g.posed(a)
        assert posed is not None, g.why_unavailable()
        V = np.concatenate([np.asarray(t, float).reshape(-1, 3) for t in posed.values()])
        pts.append(V[ConvexHull(V).vertices])
    pts.append(_housing_points())
    return np.concatenate(pts)


def _merged_floor(Rm):
    """The floor as merged (dev 4368e19): one box over the whole stroke."""
    from primitives.shared.config import GRASP_TIP_CLEARANCE_M, RG2_TIP_CLOSED_M, TABLE_HEIGHT
    from primitives.shared.gripper_rules import gripper_points_tool
    closed = TABLE_HEIGHT + 0.001 - float((Rm @ np.array([0.0, 0.0, RG2_TIP_CLOSED_M]))[2])
    return max(closed, TABLE_HEIGHT + GRASP_TIP_CLEARANCE_M - float((gripper_points_tool(None) @ Rm[2, :]).min()))


def test_the_swept_floor_keeps_the_exact_rg2_2_mm_clear_and_never_exceeds_the_merged_floor(exact_points):
    from primitives.shared.config import TABLE_HEIGHT
    from primitives.shared.table_floor import to_table_floor_z
    worst, rises = np.inf, []
    for i, Rm in enumerate(_orientations()):
        floor = to_table_floor_z(Rm)
        lowest = float((exact_points @ Rm[2, :]).min()) + floor - TABLE_HEIGHT
        worst = min(worst, lowest)
        merged = _merged_floor(Rm)
        assert floor <= merged + 1e-9, i                      # never above the merged floor
        rises.append(merged - floor)
        if i in (0, 5):                                       # vertical (both families start at 0 deg): equal
            assert abs(floor - merged) < 1e-9
    assert worst >= 0.002 - 1e-9, worst
    assert max(rises) > 0.010                                  # the artifact is gone somewhere (> 10 mm at a tilt)


def test_the_swept_floor_at_small_jaw_plane_tilts():
    """The reviewer's numbers: the merged floor rose 10.9 / 17.3 / 29.9 mm over dev's old floor at 5 / 10 / 20 deg in
    the jaw plane; the sweep is strictly lower there while staying exact (each opening's boxes span [w - 1, w + 10.2]
    mm of opening, which bounds how much of the artifact a box sweep can remove)."""
    from primitives.shared.config import RG2_TIP_CLOSED_M, TABLE_HEIGHT
    from primitives.shared.table_floor import to_table_floor_z
    down = R.from_euler("x", 180, degrees=True)
    for tilt in (5, 10, 20):
        Rm = (R.from_euler("y", tilt, degrees=True) * down).as_matrix()
        assert to_table_floor_z(Rm) < _merged_floor(Rm) - 0.001, tilt


def test_the_floor_module_is_pure():
    """P3: the replay's mirror imports it with no ROS (no rclpy) in the process."""
    code = ("import sys; sys.path.insert(0, %r); import primitives.shared.table_floor as t; "
            "import numpy as np; t.to_table_floor_z(np.diag([1.0, -1.0, -1.0])); "
            "print('rclpy' in sys.modules)" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"
    src = (ROOT / "ablations/replay_real_assembly.py").read_text()
    assert "from primitives.shared.table_floor import to_table_floor_z" in src
    md = (ROOT / "primitives/core/move_down.py").read_text()
    assert "def to_table_floor_z" not in md and "from primitives.shared.table_floor import" in md
