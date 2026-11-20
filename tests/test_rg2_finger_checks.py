"""The RG2's finger links in the physical self checks: tier 1 boxes per opening, tier 2 on the exact links, the
opening setting (lead ruling 2026-11-20, w5b-tier2)."""
import math
import types

import numpy as np
import pytest

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives.shared import arm_surface as A  # noqa: E402
from primitives.shared import collision as C  # noqa: E402
from primitives.shared import rg2_geometry as F  # noqa: E402

pytestmark = pytest.mark.skipif(F.model() is None, reason="the RG2 finger meshes are not the pinned ones")


@pytest.fixture(autouse=True)
def _reset_openings():
    prev = C.set_gripper_openings(None)
    yield
    C.set_gripper_openings(prev)


def test_missing_finger_meshes_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "MESH_DIR", tmp_path)
    monkeypatch.setattr(F, "_STATE", {})
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    assert C.physical_self_collision_refusal(home).startswith("PHYSICAL_SELF_COLLISION")


def test_finger_tier2_refuses_a_real_near_collision_and_admits_clearance():
    C.set_gripper_openings((56.7,))
    links, alpha0, _ = F.model()
    a = F.alpha_samples((56.7,))[0]
    T = F._pose_points({"l_finger": links["l_finger"]}, a, alpha0)["l_finger"]
    k = int(np.argmax(T.mean(axis=1)[:, 0]))                       # a piece on the finger's outer side
    n = np.cross(T[k, 1] - T[k, 0], T[k, 2] - T[k, 0]); n /= np.linalg.norm(n)
    if n[0] < 0:
        n = -n
    centre = T[k].mean(axis=0)

    class _Body:                                                    # a UR "body" of one surface sample, no gap
        gap, frame = 0.0, 6
    import types
    for off_mm, refused in ((3.0, True), (40.0, False)):
        body = types.SimpleNamespace(gap=0.0, frame=6, samples=(centre + off_mm / 1000.0 * n)[None, :],
                                     pieces=np.zeros((0, 3, 3)))
        frames = [np.eye(4)] * 7
        orig_body, orig_world = A.body, A._world
        try:
            A.body = lambda label: body
            A._world = lambda b, fr: (b.samples, b.pieces)
            d = A._finger_lower_bound("forearm 1/8", "RG2 finger L tip", frames, 0.010)
        finally:
            A.body, A._world = orig_body, orig_world
        assert (d < 0.010) == refused, (off_mm, d)


def test_finger_tier2_does_not_admit_with_an_unknown_opening():
    C.set_gripper_openings(None)
    frames = [np.eye(4)] * 7
    assert A._finger_lower_bound("forearm 1/8", "RG2 finger L tip", frames, 0.010) is None



def test_finger_tier2_matches_brute_force(monkeypatch):
    """The batched bound (box prefilter, rigid map into the link's modelled pose, one tree) equals the brute force:
    exact point-triangle distances to the link posed at every sampled opening, same allowances and padding."""
    import types
    monkeypatch.setattr(A, "EARLY_EXIT", False)
    openings = (21.3,)
    C.set_gripper_openings(openings)
    links, alpha0, _ = F.model()
    rng = np.random.default_rng(11)
    for name in ("l_finger", "ri_knuckle"):
        label = F.LABELS[name]
        V = F._pose_points({name: links[name]}, F.alpha_samples(openings)[3], alpha0)[name].reshape(-1, 3)
        P = V[rng.choice(len(V), 40)] + rng.normal(0, 0.006, (40, 3))           # points around the posed link
        P = np.vstack([P, V.mean(axis=0) + [0.0, 0.2, 0.0]])                    # and one far away (box path)
        body = types.SimpleNamespace(gap=0.0005, frame=6, samples=P, pieces=np.zeros((0, 3, 3)))
        monkeypatch.setattr(A, "body", lambda label, b=body: b)
        monkeypatch.setattr(A, "_world", lambda b, fr: (b.samples, b.pieces))
        got = A._finger_lower_bound("forearm 1/8", label, [np.eye(4)] * 7, 0.010)
        brute = np.inf
        for a in F.alpha_samples(openings):
            T = F._pose_points({name: links[name]}, a, alpha0)[name]
            for p in P:
                brute = min(brute, float(A.point_triangle_distance(np.repeat(p[None], len(T), 0), T).min()))
        want = brute - 0.0005 - A.ALLOWANCE_PER_BODY_M - A.RG2_ALLOWANCE_M - F.padding(name)
        assert abs(got - want) < 1e-9, (name, got, want)


def test_the_replay_passes_the_measured_width_and_forgets_it_at_a_new_grasp():
    """The finger checks model the width the gripper MEASURED (lead 2026-11-20: held-part routes know the closed
    width); a new grasp forgets it until its close reports, and no width -> no flag (the primitive takes the stroke)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
    import replay_real_assembly as rra
    st = rra.AssemblyState()
    assert st.grip_argv() == []
    st.update_from_result("control_gripper", {"result": "success", "command": "close", "final_width_mm": 0.6})
    assert st.grip_argv() == ["--gripper-width-mm", "0.6"]
    argv, _ = rra.build_argv("rotate_object", {"object_name": "u_brown", "base_name": "base1"}, st)
    assert argv[argv.index("--gripper-width-mm") + 1] == "0.6"
    st.update_from_result("move_to_grasp", {"result": "success"})
    assert st.grip_argv() == []
    st.update_from_result("control_gripper", {"result": "failure", "final_width_mm": 30.0})
    assert st.grip_argv() == []


def test_rotate_object_without_a_measured_width_checks_the_whole_stroke(monkeypatch):
    from primitives import rotate_object as R
    seen = []
    monkeypatch.setattr(C, "set_gripper_openings", lambda w, held=False: seen.append(w))

    src = R.ReorientForAssembly.reorient_for_target
    for width, want in ((None, None), (21.3, (21.3,))):
        node = types.SimpleNamespace(gripper_width_mm=width)
        try:
            src(node, "u_brown", "base1", None)                 # stops right after the opening is set (node is bare)
        except Exception:
            pass
        assert seen[-1] == want


@pytest.mark.parametrize("measured, pre_grasp, want", [(None, 39.0, None), (100.8, 39.0, (100.8,)), (37.1, None, (37.1,))])
def test_fix_orient_models_only_the_measured_width(monkeypatch, measured, pre_grasp, want):
    """Review w5b-tier2 r1 P1-1: the pre-grasp width is a command applied after the fix-orient swing - the fingers are
    still at the last release's width (often 100.8 mm, fully open).  Only the measured width is modelled; none ->
    the whole stroke."""
    import types
    import tests.ros_stubs as ros_stubs
    ros_stubs.install()
    from primitives import move_to_grasp as MG
    seen = []
    monkeypatch.setattr(C, "set_gripper_openings", lambda w, held=False: seen.append((w, held)))
    log = types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None, error=lambda *a: None)
    node = types.SimpleNamespace(current_gripper_width=measured, pre_grasp_width=pre_grasp, current_joint_angles=None,
                                 current_ee_pose=None)
    assert MG.fix_orientation_inprocess(node, (0.3, -0.4), log) is False      # stops at the missing joints
    assert seen == [(want, False)]


def test_the_box_at_a_measured_open_width_encloses_the_open_fingers_and_a_pre_grasp_box_does_not():
    """The real case behind P1-1: fingers open at 100.8 mm (rubber gap; raw 110) while a pre-grasp of 39 was
    commanded.  The 100.8 boxes enclose them; the 39 boxes miss them by tens of mm (why a command is never used)."""
    links, alpha0, _ = F.model()
    a = F.alpha_of_raw(0.1008 + 0.0092)
    P = F._pose_points(links, a, alpha0)
    for openings, inside in (((100.8,), True), ((39.0,), False)):
        B = F.boxes(openings)
        ok = all(np.all(np.abs(T.reshape(-1, 3) - B[F.LABELS[n]][0]) <= B[F.LABELS[n]][1] + 1e-9) for n, T in P.items())
        assert ok == inside, openings


@pytest.mark.parametrize("width", [21.0, 43.4, 69.9])
def test_a_held_grip_covers_fully_closed(width):
    """Review r1 P2-1 (lead ruling): a held grip may close to fully closed (a slip) under the grip command: held(w)
    boxes enclose the links fully closed and at w (raw w + 9.2), and never cover more open than w + 10.2 raw."""
    links, alpha0, _ = F.model()
    B = F.boxes((F.held(width),))
    for raw in (0.0092, (width + 9.2) / 1000.0):
        for n, T in F._pose_points(links, F.alpha_of_raw(raw), alpha0).items():
            c, h = B[F.LABELS[n]]
            assert np.all(np.abs(T.reshape(-1, 3) - c) <= h + 1e-9), (width, raw, n)
    lo, hi = F.alpha_ranges((F.held(width),))[0]
    assert lo == F.ALPHA_MIN and abs(hi - F.alpha_of_raw(width / 1000.0 + F.KNOWN_WIDTH_ABOVE_M)) < 1e-12


def test_held_routes_set_held_openings(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "set_gripper_openings", lambda w, held=False: seen.append((w, held)))
    import types
    from primitives import rotate_object as R
    try:
        R.ReorientForAssembly.reorient_for_target(types.SimpleNamespace(gripper_width_mm=21.3), "u_brown", "base1", None)
    except Exception:
        pass
    assert seen[-1] == ((21.3,), True)


def test_set_gripper_openings_held_items():
    prev = C.set_gripper_openings((21.3,), held=True)
    try:
        assert C.gripper_openings() == (F.held(21.3),)
        C.set_gripper_openings((37.1,))
        assert C.gripper_openings() == (37.1,)
    finally:
        C.set_gripper_openings(prev)


def test_the_fingers_are_checked_against_the_base():
    """Review r1 P2-2: every finger link has a clearance to the UR5e base cylinder (the tips reach ~230 mm, past the
    RG2 body box)."""
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    for openings in (None, (F.held(21.3),)):
        C.set_gripper_openings(openings)
        base_pairs = {c[2] for c in C.physical_self_clearances(np.asarray(home)) if c[1] == "base"}
        assert set(F.LABELS.values()) <= base_pairs
    C.set_gripper_openings(None)
