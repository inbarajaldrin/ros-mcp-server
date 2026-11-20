"""Held-part route admission for clear-area carries (lead approval 2026-11-19)."""
import ast
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from primitives.shared import held_part_collision as hpc
from primitives.shared.config import HOME_JOINTS
from primitives.shared.ik import WRIST3_LIMIT_RAD, dh_params, forward_kinematics

ROOT = Path(__file__).resolve().parents[1]

# MuJoCo r48 (2026-11-19), sim t=86.742 s: joints before the u_orange clear-area carry, the part's
# ground-truth pose in base_link, and the single IK candidate the planner flew (it hit upper_arm_link).
R48_START = np.array([1.9577617813281503, -1.2703284340015883, 1.6573525711332493,
                      -0.3780259770846624, -4.313644836486242, -3.1380991410469274])
R48_PART_P = [0.27933877235959836, -0.7251055430516932, 0.30106234894740497]
R48_PART_WXYZ = [-0.7105999719225897, 0.70357146657895, -0.003777956154058515, 0.004538542371395741]
R48_TARGET = np.array([-0.640851, -1.780175, 2.188892, -0.448530, -0.633096, -3.091593])

# Real cell 2026-10-24 replay_20261024_101941, u_orange#6 place_down: flown successfully (476.6 deg total
# travel, the closest real carry to the upper arm).
REAL_START = np.array([2.052016019821167, -0.9760609430125733, 1.2198908964740198,
                       -0.2443710130504151, -4.622496430073873, 3.142984628677368])
REAL_TARGET = np.array([0.118188, -1.598051, 2.037364, -0.389651, 0.125315, 3.091593])
REAL_TCP_P = [0.4426137393553686, -0.5786515358876899, 0.32005022263872646]
REAL_TCP_Q = [-0.13823774086747, 0.693420934563899, -0.6936794693267009, 0.13735548166350683]


def _r48_held():
    T = np.eye(4)
    w, x, y, z = R48_PART_WXYZ
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = R48_PART_P
    return hpc.measured_held_part("u_orange", np.linalg.inv(forward_kinematics(dh_params, R48_START)) @ T)


def test_r48_route_into_the_upper_arm_is_refused_and_via_home_is_admitted():
    held = _r48_held()
    reason = hpc.held_part_route_refusal(R48_START, R48_TARGET, held)
    assert reason is not None and reason.startswith("HELD_PART_ARM_COLLISION") and "link 1" in reason
    home = np.array(HOME_JOINTS, dtype=float)
    assert hpc.held_part_route_refusal(R48_START, home, held) is None
    assert hpc.held_part_route_refusal(home, R48_TARGET, held) is None


def test_the_flown_real_near_miss_is_rerouted_not_flown_into_the_upper_arm_margin():
    """With the conservative link radii the direct line passes 7.8 mm inside the upper-arm margin; the
    planner's split route (wrist_1 first, then the rest) clears it under the arm and held-part rules."""
    from primitives.shared.collision import check_trajectory_collision
    from primitives.shared.config import TABLE_COLLISION_MARGIN_SIDEWAYS, TABLE_HEIGHT
    from primitives.shared.route_split import admissible_route
    held = hpc.chained_held_part("u_orange", 1, [0.0006, 1.0, -0.0, 0.0], REAL_TCP_P, REAL_TCP_Q, 0.22038)
    reason = hpc.held_part_route_refusal(REAL_START, REAL_TARGET, held)
    assert reason is not None and reason.startswith("HELD_PART_ARM_COLLISION") and "link 1" in reason
    margins = hpc.held_part_min_clearances(REAL_START, REAL_TARGET, held)
    assert -0.010 < margins["links"][1] < 0.0

    def seg(a, b, last):
        if check_trajectory_collision(a, b, z_threshold=TABLE_HEIGHT - TABLE_COLLISION_MARGIN_SIDEWAYS, num_samples=20):
            return "ARM"
        return hpc.held_part_route_refusal(a, b, held)

    route, _ = admissible_route(REAL_START, REAL_TARGET, seg, vias=(np.array(HOME_JOINTS, float),))
    assert route is not None and len(route) == 3 and np.allclose(route[-1], REAL_TARGET)


def test_empty_gripper_needs_no_part_check_and_the_planner_builds_no_model():
    # within the cable caps an empty-gripper route passes (the caps themselves: test_an_empty_gripper_route_gets_...)
    assert hpc.held_part_route_refusal(R48_START, R48_TARGET, None) is None
    # The planner's model builder, compiled from source so the test needs no rclpy.
    tree = ast.parse((ROOT / "primitives/core/move_to_clear_area.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_held_part_model")
    namespace = {"np": np, "R": R, "forward_kinematics": forward_kinematics, "dh_params": dh_params}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "move_to_clear_area", "exec"), namespace)
    empty = types.SimpleNamespace(object_name=None)
    assert namespace["_held_part_model"](empty, R48_START) is None
    real = types.SimpleNamespace(object_name="u_orange", robot_mode="real", object_poses={},
                                 grasp_id=1, held_quat=[0.0006, 1.0, -0.0, 0.0])
    assert namespace["_held_part_model"](real, REAL_START).source == "measured"
    real.held_quat = None
    assert namespace["_held_part_model"](real, REAL_START).source == "envelope"


def test_travel_and_wrist3_bounds_fail_closed():
    held = _r48_held()
    wound = R48_START.copy(); wound[0] += 2 * np.pi + 0.1
    assert hpc.held_part_route_refusal(R48_START, wound, held).startswith("HELD_ROUTE_JOINT_TRAVEL")
    cable = R48_START.copy(); cable[5] = -(WRIST3_LIMIT_RAD + 0.05)
    assert hpc.held_part_route_refusal(R48_START, cable, held).startswith("HELD_ROUTE_WRIST3_LIMIT")


def test_envelope_contains_every_orientation_of_the_part():
    env = hpc.envelope_held_part("u_orange", 1, 0.22)
    held = hpc.chained_held_part("u_orange", 1, R.random(random_state=3).as_quat(), [0, 0, 0], [0, 0, 0, 1], 0.22)
    assert np.max(np.linalg.norm(held.points_flange - env.points_flange[0], axis=1)) <= env.radius_m + 1e-9


def test_planner_and_real_replay_pass_the_held_state():
    clear = (ROOT / "primitives/core/move_to_clear_area.py").read_text()
    assert "held_part_route_refusal(start_joints, target_joints, held_part)" in clear
    assert "held_part_route_refusal(start_joints, home, held_part)" in clear
    assert "HELD_PART_ROUTE_REFUSED" in clear
    # an empty gripper's refusal is reported as its own (review r1 P3-3), not as a held-part refusal
    assert 'elif held_refusals:' in clear and '"ROUTE_REFUSED: every route to the clear area was refused for the camera "' in clear
    replay = (ROOT / "ablations/replay_real_assembly.py").read_text()
    assert "'--held-quat', *[f\"{float(v):.6f}\" for v in state.held_quat]" in replay
    translate = ast.unparse(ast.parse((ROOT / "primitives/translate_object.py").read_text()))
    assert "held_quat=args.current_object_orientation" in translate


def test_the_physical_upper_arm_catches_what_the_dh_capsule_misses():
    """Review P2-2: at alpha 0.75 of r48's carry the held u_orange is 45 mm inside the physical upper-arm tube
    (the twin's capsules: -45.3 mm), while the DH upper-arm capsule, 0.138 m off the tube, reports it clear."""
    from primitives.shared.collision import UR5E_LINK_RADII, compute_all_joint_positions
    held = _r48_held()
    q = R48_START + 0.75 * (R48_TARGET - R48_START)
    T_f = forward_kinematics(dh_params, q)
    P = held.points_flange @ T_f[:3, :3].T + T_f[:3, 3]
    J = [np.asarray(p) for p in compute_all_joint_positions(q)]
    dh_upper = float(np.min(hpc._point_segment_distance(P, J[1], J[2]))) - UR5E_LINK_RADII[1]
    assert dh_upper > 0.05                                       # the DH model alone: clear by more than 50 mm
    reason = hpc.held_part_route_refusal(q, q, held)
    assert reason.startswith("HELD_PART_ARM_COLLISION") and "physical upper arm" in reason
    assert hpc.held_part_min_clearances(q, q, held)["physical"]["upper arm"] < -0.050


def test_an_empty_gripper_route_gets_the_joint_travel_and_wrist3_caps():
    """Review r5 P2-3 (w5-realpath-gaps): the cables wind the same with an empty gripper."""
    wound = np.array(R48_START, float)
    wound[0] += np.radians(370.0)
    reason = hpc.held_part_route_refusal(R48_START, wound, None)
    assert reason is not None and reason.startswith("ROUTE_JOINT_TRAVEL")
    cable = np.array(R48_START, float)
    cable[5] = -(hpc.WRIST3_LIMIT_RAD + 0.1)   # R48_START's wrist_3 is -3.14: 0.96 rad of travel, past the limit
    reason = hpc.held_part_route_refusal(R48_START, cable, None)
    assert reason is not None and reason.startswith("ROUTE_WRIST3_LIMIT")


def test_an_empty_gripper_route_within_the_caps_still_passes_with_no_part_check():
    assert hpc.held_part_route_refusal(R48_START, R48_TARGET, None) is None
