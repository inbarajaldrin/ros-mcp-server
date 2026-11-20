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
    monkeypatch.setattr(C, "set_gripper_openings", lambda w: seen.append(w))

    src = R.ReorientForAssembly.reorient_for_target
    for width, want in ((None, None), (21.3, (21.3,))):
        node = types.SimpleNamespace(gripper_width_mm=width)
        try:
            src(node, "u_brown", "base1", None)                 # stops right after the opening is set (node is bare)
        except Exception:
            pass
        assert seen[-1] == want
