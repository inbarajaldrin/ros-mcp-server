"""Every declared grasp span matches the part's CAD along that grasp's jaw axis (all four FMB1 parts).

A grasp point's `x_axis_gripper_width_mm` / `y_axis_gripper_width_mm` is the opening for jaws closing along the
part's x / y axis: the material the jaws close on plus the file's `grasp_clearance_mm`.  cad_grip_width_mm() takes
the span as width - clearance, and the R31 grasp closes to span - 2 mm and judges the hold against it (band
-1.42 / +0.50 mm), so a span wider than the material means the jaws stop before they touch it (r49 attempt 7:
inverted_u_yellow grasp 1 declared 63.1 -> 49.1 mm over a 42.0 mm leg; the close never stalled).

The CAD width is measured on the part's mesh (ablations/grasp_depth.load_mesh, the grasp file's cad_center frame):
a line along the jaw axis through the grasp point at mid-thickness and at +-25 % of the thickness, the material it
crosses inside the declared opening window, outermost entry to outermost exit (a grasp across both legs of a U spans
both; a grasp on one leg spans that leg).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ablations"))
from grasp_depth import load_mesh  # noqa: E402

from primitives.shared import config as C  # noqa: E402

PARTS = ("u_brown", "u_orange", "line_green", "inverted_u_yellow")
TOL_MM = 0.5


def _line_hits(V, F, origin, direction):
    """Parameters t of every intersection of origin + t * direction with the triangle mesh (Moller-Trumbore)."""
    A, B, Cc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = B - A, Cc - A
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - A
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    return np.unique(np.round(t[hit], 7))


def _cad_span_mm(V, F, point, axis, window_mm, z):
    direction = np.eye(3)[axis]
    origin = np.array([point[0], point[1], z])
    t = _line_hits(V, F, origin, direction)
    half = window_mm / 2000.0
    t = t[(t >= -half) & (t <= half)]
    if len(t) < 2:
        return None
    return float((t.max() - t.min()) * 1000.0)


def _grasps():
    for part in PARTS:
        d = json.load(open(f"{C._GRASP_POINTS_DIR}/{part}_grasp_points.json"))
        clearance = float(d.get("filter_params", {}).get("grasp_clearance_mm", 14.0))
        thickness = float(d["object_thickness"])
        for g in d["grasp_points"]:
            for axis, key in ((0, "x_axis_gripper_width_mm"), (1, "y_axis_gripper_width_mm")):
                w = g["grasp_validity"].get(key)
                if w is not None:
                    yield pytest.param(part, g["id"], axis, float(w), clearance, thickness, g["position"],
                                       id=f"{part}-g{g['id']}-{'xy'[axis]}")


@pytest.mark.parametrize("part,gid,axis,width,clearance,thickness,position", list(_grasps()))
def test_declared_span_is_the_cad_width_along_the_jaw_axis(part, gid, axis, width, clearance, thickness, position):
    V, F = load_mesh(part)
    point = (position["x"], position["y"])
    declared = width - clearance
    for z in (position.get("z", 0.0), position.get("z", 0.0) + 0.25 * thickness, position.get("z", 0.0) - 0.25 * thickness):
        cad = _cad_span_mm(V, F, point, axis, width, z)
        assert cad is not None, f"{part} grasp {gid}: no material along {'xy'[axis]} inside the {width} mm opening"
        assert abs(cad - declared) <= TOL_MM, (f"{part} grasp {gid} along {'xy'[axis]}: declared span {declared:.1f} mm "
                                              f"({width} - {clearance}), CAD {cad:.1f} mm at z {1000 * z:+.1f} mm")


def test_the_span_the_cell_uses_is_the_cad_one_for_yellow():
    """cad_grip_width_mm (the R31 close and its band) reads the corrected figures."""
    assert [C.cad_grip_width_mm("inverted_u_yellow", g) for g in (1, 2, 3)] == [42.0, 42.7, 42.0]
