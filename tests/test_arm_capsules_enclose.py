"""The physical arm capsules ENCLOSE the UR5e's real surface (lead ruling 2026-11-19, w5-enclose).

The surface is ur_description's visual meshes (UR's own), placed by ur_description's ur5e chain (declared below,
as xacro expands it) and expressed in each link's DH frame.  That transform is checked to be the same at every
configuration, so the proof in the link frame holds for all configurations.  The surface is cut into pieces (every
edge <= 20 mm); each piece must lie wholly inside ONE capsule of its link (a capsule is convex; a piece is the convex
hull of its corners), which proves the link's surface lies inside the union of its capsules.  The base: inside the
base cylinder.  The meshes come from the ROS install; where it is absent the test says so and skips.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from primitives.shared import collision as C
from primitives.shared.config import DH_PARAMS_NOMINAL
from primitives.shared.ik import dh_transform

VISUAL = Path("/opt/ros/humble/share/ur_description/meshes/ur5e/visual")
MAX_EDGE_M = 0.02
NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


# ur_description (ur5e) as xacro expands it: child link -> (parent, joint origin, joint index or None)
CHAIN = {"base_link_inertia": ("base_link", _T([0, 0, 0], [0, 0, np.pi]), None),
         "shoulder_link": ("base_link_inertia", _T([0, 0, 0.1625], [0, 0, 0]), 0),
         "upper_arm_link": ("shoulder_link", _T([0, 0, 0], [np.pi / 2, 0, 0]), 1),
         "forearm_link": ("upper_arm_link", _T([-0.425, 0, 0], [0, 0, 0]), 2),
         "wrist_1_link": ("forearm_link", _T([-0.3922, 0, 0.1333], [0, 0, 0]), 3),
         "wrist_2_link": ("wrist_1_link", _T([0, -0.0997, 0], [np.pi / 2, 0, 0]), 4),
         "wrist_3_link": ("wrist_2_link", _T([0, 0.0996, 0], [np.pi / 2, np.pi, np.pi]), 5)}
VISUAL_ORIGIN = {"base_link_inertia": (_T([0, 0, 0], [0, 0, np.pi]), "base.dae"),
                 "shoulder_link": (_T([0, 0, 0], [0, 0, np.pi]), "shoulder.dae"),
                 "upper_arm_link": (_T([0, 0, 0.138], [np.pi / 2, 0, -np.pi / 2]), "upperarm.dae"),
                 "forearm_link": (_T([0, 0, 0.007], [np.pi / 2, 0, -np.pi / 2]), "forearm.dae"),
                 "wrist_1_link": (_T([0, 0, -0.127], [np.pi / 2, 0, 0]), "wrist1.dae"),
                 "wrist_2_link": (_T([0, 0, -0.0997], [0, 0, 0]), "wrist2.dae"),
                 "wrist_3_link": (_T([0, -0.0005, -0.0989], [np.pi / 2, 0, 0]), "wrist3.dae")}
DH_FRAME = {"shoulder_link": 1, "upper_arm_link": 2, "forearm_link": 3, "wrist_1_link": 4, "wrist_2_link": 5,
            "wrist_3_link": 6}
RZ = np.diag([-1.0, -1.0, 1.0, 1.0])          # the DH base frame is URDF base_link turned pi about z


def _link_T(link, q):
    chain = []
    while link in CHAIN:
        chain.append(link)
        link = CHAIN[link][0]
    T = np.eye(4)
    for l in reversed(chain):
        _parent, To, j = CHAIN[l]
        T = T @ To
        if j is not None:
            T = T @ _T([0, 0, 0], [0, 0, q[j]])
    return T


def _dh_frames(q):
    T = np.eye(4)
    out = [T.copy()]
    for i, (th, d, a, al) in enumerate(DH_PARAMS_NOMINAL):
        T = T @ dh_transform(q[i] + th, d, a, al)
        out.append(T.copy())
    return out


def _dae_triangles(path):
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.parse(path).getroot()
    geoms = {}
    for g in root.iterfind(".//c:library_geometries/c:geometry", ns):
        mesh = g.find("c:mesh", ns)
        srcs = {s.get("id"): np.array(s.find("c:float_array", ns).text.split(), float) for s in mesh.iterfind("c:source", ns)}
        verts = {v.get("id"): v.find("c:input[@semantic='POSITION']", ns).get("source")[1:] for v in mesh.iterfind("c:vertices", ns)}
        tris = []
        for pl in list(mesh.iterfind("c:polylist", ns)) + list(mesh.iterfind("c:triangles", ns)):
            inputs = pl.findall("c:input", ns)
            stride = max(int(i.get("offset")) for i in inputs) + 1
            vin = next(i for i in inputs if i.get("semantic") == "VERTEX")
            P = srcs[verts[vin.get("source")[1:]]].reshape(-1, 3)
            idx = np.array(pl.find("c:p", ns).text.split(), int).reshape(-1, stride)[:, int(vin.get("offset"))]
            vc = pl.find("c:vcount", ns)
            counts = np.array(vc.text.split(), int) if vc is not None else np.full(len(idx) // 3, 3)
            k = 0
            for n in counts:
                poly = idx[k:k + n]
                k += n
                for t in range(1, n - 1):
                    tris.append(P[[poly[0], poly[t], poly[t + 1]]])
        geoms[g.get("id")] = np.array(tris)
    out = []
    for node in root.iterfind(".//c:library_visual_scenes//c:node", ns):
        m = node.find("c:matrix", ns)
        M = np.array(m.text.split(), float).reshape(4, 4) if m is not None else np.eye(4)
        for ig in node.iterfind("c:instance_geometry", ns):
            out.append(geoms[ig.get("url")[1:]] @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(out)


def _subdivide(T):
    done = []
    while len(T):
        e = np.max(np.stack([np.linalg.norm(T[:, 0] - T[:, 1], axis=1), np.linalg.norm(T[:, 1] - T[:, 2], axis=1),
                             np.linalg.norm(T[:, 2] - T[:, 0], axis=1)]), axis=0)
        done.append(T[e <= MAX_EDGE_M])
        T = T[e > MAX_EDGE_M]
        if not len(T):
            break
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                            np.stack([ab, bc, ca], 1)])
    return np.concatenate(done)


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _capsules_of(frame):
    return [(np.asarray(a, float), np.asarray(b, float), r, label)
            for k, a, b, r, label in tuple(C.PHYSICAL_LINK_CAPSULES) + tuple(C.PHYSICAL_WRIST_CAPSULES)
            if k == frame and label != "quick changer"]


needs_meshes = pytest.mark.skipif(not VISUAL.is_dir(), reason=f"UR's visual meshes not installed ({VISUAL})")


@needs_meshes
@pytest.mark.parametrize("link", list(DH_FRAME))
def test_every_piece_of_the_link_surface_lies_inside_one_capsule(link):
    k = DH_FRAME[link]
    rng = np.random.default_rng(k)
    Ms = [np.linalg.inv(_dh_frames(q)[k]) @ RZ @ _link_T(link, q) @ VISUAL_ORIGIN[link][0]
          for q in rng.uniform(-3, 3, (8, 6))]
    assert max(float(np.abs(M - Ms[0]).max()) for M in Ms) < 1e-6          # the link frame relation is constant
    T = _dae_triangles(VISUAL / VISUAL_ORIGIN[link][1])
    P = _subdivide(T @ Ms[0][:3, :3].T + Ms[0][:3, 3])
    caps = _capsules_of(k)
    assert caps, f"no capsules declared for {link}"
    outside = np.min([_seg_dist(P.reshape(-1, 3), a, b).reshape(-1, 3).max(axis=1) - r for a, b, r, _ in caps], axis=0)
    assert float(outside.max()) <= 0.0, f"{link}: a surface piece lies {1000 * outside.max():.2f} mm outside"


@needs_meshes
def test_the_base_surface_lies_inside_the_base_cylinder():
    T = _dae_triangles(VISUAL / VISUAL_ORIGIN["base_link_inertia"][1])
    M = RZ @ _link_T("base_link_inertia", np.zeros(6)) @ VISUAL_ORIGIN["base_link_inertia"][0]
    P = T.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]                       # the cylinder is convex: corners suffice
    assert float(np.max(C.base_cylinder_distance(P))) <= 1e-6          # float32 noise at the mounting plane z = 0
