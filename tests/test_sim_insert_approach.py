"""The sim insert's approach (lead 2026-11-20, ruling A): normal speed to 3 mm above the first possible contact, then
to contact at 5 mm/s, with the speed cap sim-only and the real code byte for byte the reviewed one.

The twin's one full-speed descent met the part at 80-100 mm/s and pressed 372-459 N before move_down's 102 ms median
stopped it; the arm then rebounded ~1 mm while holding (w5b (d), lab mujoco/evidence/w5b-contact-overshoot).
"""
import ast
import hashlib
import io
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRANSLATE = ROOT / "primitives/translate_object.py"
MOVE_DOWN = ROOT / "primitives/core/move_down.py"

# sha256 of the reviewed real insert source (dev fa40ee6 = d19916d for these segments; the text w8 pinned at 76d4e07)
PINNED_REAL_SOURCE_SHA256 = {
    "run_perform_insert_real": "407fe25c04261bcfa5a3a76cb6e294dcba4bbf844122dc50e7dfe93ab7797cf2",
    "TranslateObject.translate_for_target_real": "4f8d94a5075dc497880c756abc1913960c8b0edbf06ca6ec346bff94aa85895f",
    "main: the insert's real-mode branch": "927e1229c3588147733b53590029fcee98c9557838206ee7e28b1436a022f3fa",
}
# sha256 of primitives/core/move_down.py at dev 34a4f2f (the reviewed move_down after the floor merges: the --to-table
# floor in primitives.shared.table_floor); this file minus its marked sim-only blocks must be it
PINNED_MOVE_DOWN_SHA256 = "d130a9a9353f24009a2ecb16b15a6bb4a72ac9dd5c20a2fbaa317a90fcbc5a2a"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _real_segments(source):
    tree = ast.parse(source)
    seg = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_perform_insert_real":
            seg["run_perform_insert_real"] = ast.get_source_segment(source, node)
        if isinstance(node, ast.ClassDef) and node.name == "TranslateObject":
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == "translate_for_target_real":
                    seg["TranslateObject.translate_for_target_real"] = ast.get_source_segment(source, m)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and ast.unparse(node.test) == "args.mode == 'sim'"
                and any("translate_for_target_real" in ast.unparse(s) for s in node.orelse)):
            seg["main: the insert's real-mode branch"] = "\n".join(
                lines[node.orelse[0].lineno - 1:node.orelse[-1].end_lineno])
    return seg


def test_the_real_insert_source_is_byte_for_byte_the_reviewed_one():
    seg = _real_segments(TRANSLATE.read_text())
    assert set(seg) == set(PINNED_REAL_SOURCE_SHA256)
    for name, digest in PINNED_REAL_SOURCE_SHA256.items():
        assert _sha(seg[name]) == digest, f"{name} changed"


def test_move_down_minus_its_sim_only_blocks_is_the_reviewed_file():
    out, skip, blocks = [], False, 0
    for line in MOVE_DOWN.read_text().split("\n"):
        if line.strip().startswith("# >>> sim-only"):
            assert not skip
            skip, blocks = True, blocks + 1
            continue
        if line.strip().startswith("# <<< sim-only"):
            assert skip
            skip = False
            continue
        if not skip:
            out.append(line)
    assert not skip and blocks == 5
    assert _sha("\n".join(out)) == PINNED_MOVE_DOWN_SHA256


def _md():
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives.core import move_down as md
    return md


def test_the_speed_cap_is_refused_in_real_mode(monkeypatch):
    md = _md()
    monkeypatch.setattr(md, "_SIM_MAX_SPEED_M_S", None)
    with pytest.raises(ValueError):
        md.set_sim_speed_cap("real", 5.0)
    assert md._SIM_MAX_SPEED_M_S is None
    md.set_sim_speed_cap("sim", 5.0)
    assert md._SIM_MAX_SPEED_M_S == 0.005


def test_the_cap_is_in_place_when_the_constructor_descends(monkeypatch):
    """MoveDown's constructor already runs the descent (move_down() at its end): the cap must be set before it.
    (Twin run fmb-w5b-simA-u_brown, 2026-11-20: a cap set after construction left the descent at 28 mm/s, 201 N.)"""
    md = _md()
    monkeypatch.setattr(md, "_SIM_MAX_SPEED_M_S", None)
    seen = {}

    class Fake:
        def __init__(self, *a, **k):
            seen["cap_at_construction"] = md._SIM_MAX_SPEED_M_S
            self.finish_requested, self.error_message, self.trajectory_sent = True, None, False
            self.action_client = types.SimpleNamespace(destroy=lambda: None)

        def output_result_json(self):
            pass

        def destroy_node(self):
            pass

    monkeypatch.setattr(md, "MoveDown", Fake)
    monkeypatch.setattr(md, "init_ros", lambda **k: None)
    monkeypatch.setattr(md.rclpy, "ok", lambda: True)
    monkeypatch.setattr(md.rclpy, "shutdown", lambda: None)
    monkeypatch.setattr(sys, "argv", ["move_down.py", "--mode", "sim", "--to-table", "--require-contact",
                                      "--max-speed-mm-s", "5"])
    with pytest.raises(SystemExit) as e:
        md.main()
    assert e.value.code == 0 and seen["cap_at_construction"] == 0.005


def test_main_refuses_the_cap_in_real_mode_before_ros(monkeypatch):
    md = _md()
    monkeypatch.setattr(md, "init_ros", lambda **k: (_ for _ in ()).throw(AssertionError("ROS started")))
    monkeypatch.setattr(sys, "argv", ["move_down.py", "--mode", "real", "--to-table", "--max-speed-mm-s", "5"])
    buf = io.StringIO()
    with redirect_stdout(buf), pytest.raises(SystemExit) as e:
        md.main()
    assert e.value.code == 1
    result = json.loads(buf.getvalue().split("__RESULT_JSON__")[1].split("__END_RESULT_JSON__")[0])
    assert result["result"] == "failure" and "sim-only" in result["error"]


def test_the_capped_s_curve_peaks_at_the_cap():
    """capped_duration: the s-curve (20 % raised-cosine ramps) over a straight path peaks at <= the cap, and an
    uncapped duration is returned unchanged."""
    md = _md()
    from primitives.shared.velocity_profiles import s_curve_profile
    d = 0.034                                                  # rim + 3 mm above a 31 mm-deep seat
    T = md.capped_duration(1.0, d, 0.005)
    assert md.capped_duration(1.0, d, None) == 1.0 and T > 1.0
    q0, q1 = np.zeros(6), np.full(6, 0.1)                     # a straight joint line: arc length ~ Cartesian distance
    path = [q0 + (q1 - q0) * k / 60 for k in range(61)]
    prof = s_curve_profile(path, T)
    t = np.array([p[2] for p in prof]); s = np.array([np.linalg.norm(np.asarray(p[0]) - q0) for p in prof])
    v = np.diff(s) / np.diff(t) * d / np.linalg.norm(q1 - q0)
    assert v.max() <= 0.005 * 1.01 and v.max() >= 0.005 * 0.95


def test_the_contact_flag_is_reported_in_sim_only():
    md = _md()
    for mode, has in (("real", False), ("sim", True)):
        n = object.__new__(md.MoveDown)
        n.mode, n.error_message, n.force_threshold_reached = mode, None, True
        buf = io.StringIO()
        with redirect_stdout(buf):
            n.output_result_json()
        r = json.loads(buf.getvalue().split("__RESULT_JSON__")[1].split("__END_RESULT_JSON__")[0])
        assert ("contact" in r) == has
        assert r == ({"result": "success", "mode": mode, "movement_type": "move_down", **({"contact": True} if has else {})})


def test_run_move_down_default_command_is_unchanged_and_the_phases_are_sim(monkeypatch):
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    import primitives.translate_object as T
    seen = []
    monkeypatch.setattr(T, "run_subprocess", lambda path, args, timeout=None: seen.append(list(args)) or (True, ""))
    T.run_move_down(mode="real")
    T.run_move_down(mode="sim", to_table=True, require_contact=True)
    T.run_move_down(mode="sim", to_table=True, height=0.22191)
    T.run_move_down(mode="sim", to_table=True, require_contact=True, max_speed_mm_s=T.SIM_INSERT_CONTACT_SPEED_MM_S)
    assert seen[0] == ["--mode", "real"]
    assert seen[1] == ["--mode", "sim", "--to-table", "--require-contact"]
    assert seen[2] == ["--mode", "sim", "--height", "0.22191", "--to-table"]
    assert seen[3] == ["--mode", "sim", "--to-table", "--require-contact", "--max-speed-mm-s", "5"]


def _pose(z, xy=(0.0, 0.0)):
    M = np.eye(4)
    M[:3, 3] = [xy[0], xy[1], z]
    return M


def _box(sx, sy, sz):
    return np.array([[x, y, zz] for x in (-sx, sx) for y in (-sy, sy) for zz in (0.0, sz)])


def test_the_approach_stops_three_mm_above_the_highest_surface_under_the_part():
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives.translate_object import SIM_INSERT_APPROACH_ABOVE_M, sim_insert_approach_flange_z
    assert SIM_INSERT_APPROACH_ABOVE_M == 0.003
    part = _box(0.01, 0.01, 0.02)                              # 20 mm tall, its origin at its bottom
    T_part, T_tcp = _pose(0.30), _pose(0.45)                    # part bottom 0.30, flange 0.45 (150 mm above it)
    base = (_pose(0.0), _box(0.05, 0.05, 0.04))               # base top 0.04
    seat = _pose(0.15 + 0.01)                                   # flange at the seat: bottom 10 mm below the base top
    z = sim_insert_approach_flange_z(T_tcp, T_part, part, [base], seat)
    assert abs(z - (0.45 - (0.30 - 0.04) + 0.003)) < 1e-12   # part bottom 3 mm above the base top
    tall = (_pose(0.0, (0.012, 0.0)), _box(0.005, 0.005, 0.07))  # a seated part under the footprint, top 0.07
    far = (_pose(0.0, (0.30, 0.0)), _box(0.005, 0.005, 0.20))   # a tall body elsewhere: ignored
    z2 = sim_insert_approach_flange_z(T_tcp, T_part, part, [base, tall, far], seat)
    assert abs(z2 - (0.45 - (0.30 - 0.07) + 0.003)) < 1e-12
    low_seat = _pose(0.40)                                      # never below the seat
    assert abs(sim_insert_approach_flange_z(T_tcp, T_part, part, [base], low_seat) - 0.403) < 1e-12
    assert abs(sim_insert_approach_flange_z(T_tcp, T_part, part, [], seat) - (0.16 + 0.003)) < 1e-12


def test_the_sim_insert_runs_both_phases_and_reports_them():
    source = TRANSLATE.read_text()
    assert "run_move_down(mode='sim', to_table=True, height=approach_z)" in source
    assert ("run_move_down(\n                                    mode='sim', to_table=True, require_contact=True,\n"
            "                                    max_speed_mm_s=SIM_INSERT_CONTACT_SPEED_MM_S)") in source
    assert "run_move_down(mode='sim', to_table=True, require_contact=True)\n" not in source
    assert 'result["sim_insert_descent"] = node._sim_insert_descent' in source
