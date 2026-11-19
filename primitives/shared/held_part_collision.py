"""Route admission for a joint-space move that CARRIES a part (lead approval 2026-11-19).

check_trajectory_collision models the arm only (DH joint origins as capsules of
UR5E_LINK_RADII) against the table and itself.  A held part reaches up to
~70 mm beyond the pads and was never in the check: in the MuJoCo cell (r48,
2026-11-19) the clear-area carry's shortest-travel route swung a held u_orange
into the upper arm.  This module adds the held part to the same route samples.

Held part model, best available first:
  * "measured": the part's CAD surface (<= 5 mm samples) at a measured
    flange->part transform (sim ground truth, or the cell's chained held
    orientation placed by the CAD grasp point at the executed grasp depth -
    the geometry of ablations/held_clear_read.held_in_flange);
  * "envelope": no orientation known -> a sphere about the grasp point whose
    radius is the part's largest CAD distance from that grasp point (any
    orientation fits inside it).

Refusal (fail closed, typed reason) at any of num_samples+1 joint-linear
samples of start -> target:
  * a held-part point within (link radius + HELD_LINK_MARGIN_M) of an arm-link
    capsule axis, links 0..ARM_LINKS_CHECKED-1;
  * a held-part point below the table plus HELD_TABLE_MARGIN_M;
and before sampling, joint travel: any joint moving more than
MAX_JOINT_TRAVEL_RAD, or wrist_3 outside +-WRIST3_LIMIT_RAD (camera cable).
An empty gripper (held=None) keeps today's behaviour: nothing is added.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as R

from primitives.shared.collision import UR5E_LINK_RADII, compute_all_joint_positions, physical_link_capsules
from primitives.shared.config import CAD_DATA_DIR, TABLE_HEIGHT
from primitives.shared.ik import WRIST3_LIMIT_RAD, dh_params, forward_kinematics

HELD_LINK_MARGIN_M = 0.010        # the same 1 cm safety margin check_self_collision uses
HELD_TABLE_MARGIN_M = 0.0         # the carry's own table margin is applied to the arm; the part must not go below the table
ARM_LINKS_CHECKED = 4             # base, upper arm, forearm, wrist_1; wrist_2/3 carry the gripper the part hangs from
MAX_JOINT_TRAVEL_RAD = 2 * np.pi  # no joint winds through more than one full turn in one carry.  Half a turn would
                                  # refuse two carries the real cell flew safely (10-22 yellow wrist_3 187 deg, 10-24
                                  # u_orange wrist_2 272 deg); the held-part geometry, not a travel cap, is the guard.
SURFACE_STEP_M = 0.005


@dataclass(frozen=True)
class HeldPart:
    name: str
    source: str                   # "measured" or "envelope"
    points_flange: np.ndarray     # (N, 3) part points in the flange frame (envelope: the sphere centre)
    radius_m: float = 0.0         # envelope sphere radius (0 for measured points)


@lru_cache(maxsize=None)
def _cad_mesh(name: str):
    vertices, faces = [], []
    for line in open(CAD_DATA_DIR / f"models/{name}.obj"):
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.array(vertices) * 0.01, np.array(faces)     # the CAD OBJs are in cm (ablations/grasp_depth.load_mesh)


@lru_cache(maxsize=None)
def part_surface_points(name: str, step: float = SURFACE_STEP_M) -> np.ndarray:
    """Points on every CAD triangle no farther than ~step apart, plus every vertex (part frame, m)."""
    V, F = _cad_mesh(name)
    points = [V]
    for a, b, c in V[F]:
        n = max(1, int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / step)))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                points.append((a + (b - a) * i / n + (c - a) * j / n)[None])
    return np.unique(np.round(np.vstack(points), 6), axis=0)


def _T(p, q_xyzw):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    M[:3, 3] = p
    return M


def measured_held_part(name: str, T_flange_part: np.ndarray) -> HeldPart:
    P = part_surface_points(name)
    return HeldPart(name, "measured", P @ T_flange_part[:3, :3].T + T_flange_part[:3, 3])


def chained_held_part(name: str, grasp_id: int, held_quat_xyzw, tcp_p, tcp_q_xyzw, tool_offset_m: float) -> HeldPart:
    """The cell's chained held pose: grasp point at the gripper centre (flange + R_f [0, 0, L])."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_grasp_offset_in_object
    gp = np.asarray(load_grasp_offset_in_object(name, int(grasp_id)), dtype=float)
    gc = np.asarray(tcp_p) + R.from_quat(tcp_q_xyzw).apply([0.0, 0.0, tool_offset_m])
    po = gc - R.from_quat(held_quat_xyzw).apply(gp)
    return measured_held_part(name, np.linalg.inv(_T(tcp_p, tcp_q_xyzw)) @ _T(po, held_quat_xyzw))


def envelope_held_part(name: str, grasp_id: int, tool_offset_m: float) -> HeldPart:
    """No orientation known: every orientation of the part about its grasp point fits in this sphere."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_grasp_offset_in_object
    gp = np.asarray(load_grasp_offset_in_object(name, int(grasp_id)), dtype=float)
    radius = float(np.max(np.linalg.norm(part_surface_points(name) - gp, axis=1)))
    return HeldPart(name, "envelope", np.array([[0.0, 0.0, tool_offset_m]]), radius)


def _point_segment_distance(P, a, b):
    d = b - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * d), axis=1)


def joint_travel_refusal(start, target):
    start, target = np.asarray(start, float), np.asarray(target, float)
    delta = np.abs(target - start)
    if float(np.max(delta)) > MAX_JOINT_TRAVEL_RAD:
        j = int(np.argmax(delta))
        return f"HELD_ROUTE_JOINT_TRAVEL: joint {j} would move {np.degrees(delta[j]):.1f} deg (> 360)"
    if abs(float(target[5])) > WRIST3_LIMIT_RAD or abs(float(start[5])) > WRIST3_LIMIT_RAD:
        return f"HELD_ROUTE_WRIST3_LIMIT: wrist_3 outside +-{WRIST3_LIMIT_RAD} rad"
    return None


def held_part_route_refusal(start, target, held: HeldPart | None, num_samples: int = 40):
    """None when the carry is admissible, else a typed reason.  held=None: empty gripper, no added check."""
    if held is None:
        return None
    start, target = np.asarray(start, float), np.asarray(target, float)
    travel = joint_travel_refusal(start, target)
    if travel:
        return travel
    for i in range(num_samples + 1):
        alpha = i / num_samples
        q = start + alpha * (target - start)
        T_f = forward_kinematics(dh_params, q)
        P = held.points_flange @ T_f[:3, :3].T + T_f[:3, 3]
        lowest = float(np.min(P[:, 2])) - held.radius_m
        if lowest < TABLE_HEIGHT + HELD_TABLE_MARGIN_M:
            return (f"HELD_PART_TABLE: {held.name} ({held.source}) {1000 * (TABLE_HEIGHT - lowest):.1f} mm "
                    f"below the table at alpha={alpha:.3f}")
        joints = [np.asarray(p) for p in compute_all_joint_positions(q)]
        for link in range(ARM_LINKS_CHECKED):
            clearance = float(np.min(_point_segment_distance(P, joints[link], joints[link + 1]))) - held.radius_m
            need = UR5E_LINK_RADII[link] + HELD_LINK_MARGIN_M
            if clearance < need:
                return (f"HELD_PART_ARM_COLLISION: {held.name} ({held.source}) {1000 * clearance:.1f} mm from link "
                        f"{link} axis (needs {1000 * need:.1f}) at alpha={alpha:.3f}")
        for a, b, r, label in physical_link_capsules(q):
            clearance = float(np.min(_point_segment_distance(P, a, b))) - held.radius_m - r
            if clearance < HELD_LINK_MARGIN_M:
                return (f"HELD_PART_ARM_COLLISION: {held.name} ({held.source}) {1000 * clearance:.1f} mm from the "
                        f"physical {label} (needs {1000 * HELD_LINK_MARGIN_M:.0f}) at alpha={alpha:.3f}")
    return None


def held_part_min_clearances(start, target, held: HeldPart, num_samples: int = 40):
    """Diagnostics: per-link minimum (clearance - needed) along the route, metres (negative = would refuse): the DH
    capsules ("links") and the physical capsules ("physical", by label)."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    worst = {link: np.inf for link in range(ARM_LINKS_CHECKED)}
    physical = {}
    table = np.inf
    for i in range(num_samples + 1):
        q = start + (i / num_samples) * (target - start)
        T_f = forward_kinematics(dh_params, q)
        P = held.points_flange @ T_f[:3, :3].T + T_f[:3, 3]
        table = min(table, float(np.min(P[:, 2])) - held.radius_m - TABLE_HEIGHT - HELD_TABLE_MARGIN_M)
        joints = [np.asarray(p) for p in compute_all_joint_positions(q)]
        for link in range(ARM_LINKS_CHECKED):
            c = float(np.min(_point_segment_distance(P, joints[link], joints[link + 1]))) - held.radius_m
            worst[link] = min(worst[link], c - UR5E_LINK_RADII[link] - HELD_LINK_MARGIN_M)
        for a, b, r, label in physical_link_capsules(q):
            c = float(np.min(_point_segment_distance(P, a, b))) - held.radius_m - r - HELD_LINK_MARGIN_M
            physical[label] = min(physical.get(label, np.inf), c)
    return {"links": worst, "physical": physical, "table": table}


def held_part_model(name, T_flange, T_part=None, held_quat=None, grasp_id=None):
    """The held part for a route check, best source first: a measured pose (sim ground truth), the cell's
    chained orientation + CAD grasp point at the executed grasp depth, else the orientation-free envelope."""
    from primitives.shared.config import default_grasp_id, held_tool_offset_for
    T_flange = np.asarray(T_flange, float)
    if T_part is not None:
        return measured_held_part(name, np.linalg.inv(T_flange) @ np.asarray(T_part, float))
    grasp_id = default_grasp_id(name) if grasp_id is None else grasp_id
    offset = float(held_tool_offset_for(name)[2])
    if held_quat is not None:
        return chained_held_part(name, grasp_id, held_quat, T_flange[:3, 3],
                                 R.from_matrix(T_flange[:3, :3]).as_quat(), offset)
    return envelope_held_part(name, grasp_id, offset)
