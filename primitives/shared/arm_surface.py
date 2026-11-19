"""The UR5e's real surface (ur_description's visual meshes) for the second tier of the physical checks.

Tier 1 is collision.py's enclosing capsules (PHYSICAL_LINK_CAPSULES / PHYSICAL_WRIST_CAPSULES, proven to contain these
meshes by tests/test_arm_capsules_enclose.py).  Tier 2 runs only for a body pair (or a body and the table) that tier 1
refused, on the same pairs and the same margin, and can only admit that pair: it replaces the capsules by a provable
lower bound on the distance between the real surfaces.

- A body is one capsule bin of a link (or the base): the mesh pieces that lie wholly inside that capsule (the base
  cylinder for the base).  Every piece of a link lies in at least one bin, checked at load; a piece in several bins
  belongs to each.  The meshes are placed in the DH link frames by the transform the enclosure test validates.
- Distances: one side is sampled (the vertices of its pieces subdivided to SAMPLE_EDGE_M); every point of a piece is
  within its longest edge / sqrt(3) of one of its vertices (acute: the circumradius; obtuse: at most half the longest
  edge), so that covering radius is subtracted.  The other side is exact: point-to-triangle distances to its pieces
  (subdivided to EXACT_EDGE_M) where a cheap both-sides-sampled bound does not already clear the margin.  The RG2
  body box and the quick changer capsule are not UR's: their tier-1 shapes are kept, exactly.
- The table: the lowest point of a bin's surface is a vertex of its pieces (a linear function's minimum over a
  triangle is at a vertex): exact.
- Allowance per UR body (ALLOWANCE_PER_BODY_M): calibration and casting.  The cell places the bodies with the
  calibrated DH; the meshes are fixed to their frames by the nominal chain.  The largest displacement of any mesh
  point between the nominal and calibrated joint transforms is 0.75 mm (forearm; tests check it stays <= 1.0 mm);
  the casting may differ from UR's CAD by an assumed 0.5 mm.
- Fail closed: if the meshes are missing, differ from the pinned hashes or do not parse, tier 2 is unavailable and
  every tier-1 refusal stands (available() is False, why_unavailable() says why).  Links load on first need.
"""
import hashlib
import os
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

VISUAL_DIR = Path(os.environ.get("UR5E_VISUAL_MESH_DIR", "/opt/ros/humble/share/ur_description/meshes/ur5e/visual"))
# ros-humble-ur-description 2.13.0 (the cell PC's install, 2026-11-19)
MESH_SHA256 = {
    "base.dae": "c30e4d7228ef633dca2a02c26df2316741f6642cb6333e8289dba6e6d0b22819",
    "shoulder.dae": "c581bfa55b86061d9b99d315ed53817b6b693e6a500643bbf9bbb783322d585d",
    "upperarm.dae": "c040a05bb224f14cfab6aa6704d70cbb773ada940eca9a3e34e6ce486c4c4c29",
    "forearm.dae": "04694bf57dac46e9b8bc7c30f2eec75a1bc762e1f1a879ea30c944f24cacbad8",
    "wrist1.dae": "a07a3d96d2540f9be12e4a07bcb8577d9f1e36005920c396a407804343a3b588",
    "wrist2.dae": "982fa42933c6f81629882ad7975e9ba5dac5d474a6b56694e233708376cf840e",
    "wrist3.dae": "7a7664ac4adcd53b0870fe6a2b8b975693d846dd4d07a5f44194c7d16f4890ba",
}
SAMPLE_EDGE_M = 0.004
EXACT_EDGE_M = 0.008
CALIBRATION_ALLOWANCE_M = 0.001        # >= the 0.75 mm measured (tests)
CASTING_ALLOWANCE_M = 0.0005           # declared: UR's casting vs its CAD
ALLOWANCE_PER_BODY_M = CALIBRATION_ALLOWANCE_M + CASTING_ALLOWANCE_M


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


# ur_description (ur5e) as xacro expands it (ur_macro.xacro, config/ur5e/default_kinematics.yaml and
# visual_parameters.yaml): child link -> (parent, joint origin, joint index or None) and each link's visual origin.
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
DH_FRAME = {"base_link_inertia": 0, "shoulder_link": 1, "upper_arm_link": 2, "forearm_link": 3, "wrist_1_link": 4,
            "wrist_2_link": 5, "wrist_3_link": 6}
RZ = np.diag([-1.0, -1.0, 1.0, 1.0])          # the DH base frame is URDF base_link turned pi about z


def urdf_link_T(link, q):
    """A link's pose in URDF base_link at joints q."""
    chain = []
    while link in CHAIN:
        chain.append(link)
        link = CHAIN[link][0]
    T = np.eye(4)
    for name in reversed(chain):
        _parent, To, j = CHAIN[name]
        T = T @ To
        if j is not None:
            T = T @ _T([0, 0, 0], [0, 0, q[j]])
    return T


def nominal_dh_frames(q):
    from primitives.shared.config import DH_PARAMS_NOMINAL
    from primitives.shared.ik import dh_transform
    T = np.eye(4)
    out = [T.copy()]
    for i, (th, d, a, al) in enumerate(DH_PARAMS_NOMINAL):
        T = T @ dh_transform(q[i] + th, d, a, al)
        out.append(T.copy())
    return out


def mesh_to_dh_frame(link, q=np.zeros(6)):
    """The (constant) transform from a link's mesh coordinates to its DH frame (nominal chain)."""
    q = np.asarray(q, float)
    return np.linalg.inv(nominal_dh_frames(q)[DH_FRAME[link]]) @ RZ @ urdf_link_T(link, q) @ VISUAL_ORIGIN[link][0]


def dae_triangles(path):
    """Every triangle of a Collada file (polylists fan-triangulated), each node's matrix applied, metres."""
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.parse(path).getroot()
    geoms = {}
    for g in root.iterfind(".//c:library_geometries/c:geometry", ns):
        mesh = g.find("c:mesh", ns)
        srcs = {s.get("id"): np.array(s.find("c:float_array", ns).text.split(), float)
                for s in mesh.iterfind("c:source", ns)}
        verts = {v.get("id"): v.find("c:input[@semantic='POSITION']", ns).get("source")[1:]
                 for v in mesh.iterfind("c:vertices", ns)}
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


def subdivide(T, max_edge):
    """Midpoint subdivision until every edge <= max_edge (each piece lies inside the triangle it came from)."""
    done = []
    while len(T):
        e = np.max(np.stack([np.linalg.norm(T[:, 0] - T[:, 1], axis=1), np.linalg.norm(T[:, 1] - T[:, 2], axis=1),
                             np.linalg.norm(T[:, 2] - T[:, 0], axis=1)]), axis=0)
        done.append(T[e <= max_edge])
        T = T[e > max_edge]
        if not len(T):
            break
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1), np.stack([ca, bc, c], 1),
                            np.stack([ab, bc, ca], 1)])
    return np.concatenate(done)


def _longest_edge(T):
    return float(np.max([np.linalg.norm(T[:, i] - T[:, (i + 1) % 3], axis=1).max() for i in range(3)]))


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def point_triangle_distance(P, T):
    """Exact distances from points P[i] to triangles T[i] (n, 3, 3): the plane distance when the projection falls
    inside, else the nearest edge."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.einsum("ij,ij->i", n, n)
    ok = nn > 1e-24
    h = np.where(ok, np.einsum("ij,ij->i", P - a, n) / np.where(ok, nn, 1.0), 0.0)
    X = P - h[:, None] * n
    inside = ok & (np.einsum("ij,ij->i", np.cross(b - a, X - a), n) >= 0) \
        & (np.einsum("ij,ij->i", np.cross(c - b, X - b), n) >= 0) & (np.einsum("ij,ij->i", np.cross(a - c, X - c), n) >= 0)

    def seg(p, q):
        pq = q - p
        t = np.clip(np.einsum("ij,ij->i", P - p, pq) / np.maximum(np.einsum("ij,ij->i", pq, pq), 1e-24), 0.0, 1.0)
        return np.linalg.norm(P - (p + t[:, None] * pq), axis=1)
    edge = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside, np.abs(h) * np.sqrt(nn), edge)


class _Body:
    """One bin's surface in its DH frame: sample vertices (+ covering radius) and exact pieces (+ centre, radius)."""

    def __init__(self, frame, samples, gap, pieces):
        self.frame = frame
        self.samples = np.unique(samples.reshape(-1, 3), axis=0)
        self.gap = gap
        self.pieces = pieces
        self.centres = pieces.mean(axis=1)
        self.rho = float(np.linalg.norm(pieces - self.centres[:, None, :], axis=2).max()) if len(pieces) else 0.0


PROOF_EDGE_M = 0.02                    # the enclosure proof's pieces: each lies wholly inside a capsule of its link
_LOCK = threading.Lock()
_STATE = {"links": {}}


def _check_hashes():
    for name, digest in MESH_SHA256.items():
        h = hashlib.sha256((VISUAL_DIR / name).read_bytes()).hexdigest()
        if h != digest:
            raise ValueError(f"{VISUAL_DIR / name}: sha256 {h[:12]} is not the pinned {digest[:12]}")


def _link_of(label):
    from primitives.shared import collision as C
    if label == "base":
        return "base_link_inertia"
    for k, _a, _b, _r, lab in tuple(C.PHYSICAL_LINK_CAPSULES) + tuple(C.PHYSICAL_WRIST_CAPSULES):
        if lab == label and lab != "quick changer":
            return next(link for link, f in DH_FRAME.items() if f == k)
    return None


def _load_link(link):
    """{label: _Body} for the bins of one link (raises on any failure: the caller fails closed).  Pieces are assigned
    to bins at the proof's 20 mm level (a piece in several capsules goes to each) and then subdivided; a sub-piece
    lies inside its parent, so inside the same capsule."""
    from primitives.shared import collision as C
    k = DH_FRAME[link]
    M = mesh_to_dh_frame(link) if k else RZ @ urdf_link_T(link, np.zeros(6)) @ VISUAL_ORIGIN[link][0]
    T = dae_triangles(VISUAL_DIR / VISUAL_ORIGIN[link][1])
    P = subdivide(T @ M[:3, :3].T + M[:3, 3], PROOF_EDGE_M)
    if k == 0:
        bins = [("base", np.ones(len(P), bool))]
    else:
        caps = [(np.asarray(a, float), np.asarray(b, float), float(r), lab)
                for f, a, b, r, lab in tuple(C.PHYSICAL_LINK_CAPSULES) + tuple(C.PHYSICAL_WRIST_CAPSULES)
                if f == k and lab != "quick changer"]
        V = P.reshape(-1, 3)
        bins = [(lab, (_seg_dist(V, a, b) - r).reshape(-1, 3).max(axis=1) <= 1e-9) for a, b, r, lab in caps]
        if not np.any([m for _, m in bins], axis=0).all():
            raise ValueError(f"{link}: a surface piece lies in no capsule")
    out = {}
    for lab, m in bins:
        fine = subdivide(P[m], SAMPLE_EDGE_M)
        out[lab] = _Body(k, fine, _longest_edge(fine) / np.sqrt(3.0), subdivide(P[m], EXACT_EDGE_M))
    return out


def body(label):
    """A body's surface, or None when tier 2 is unavailable for it (fail closed)."""
    with _LOCK:
        if "hashes" not in _STATE:
            try:
                _check_hashes()
                _STATE["hashes"] = True
            except Exception as e:                                   # noqa: BLE001 - any failure fails closed
                _STATE["hashes"] = False
                _STATE["error"] = f"{type(e).__name__}: {e}"
        if not _STATE["hashes"]:
            return None
        link = _link_of(label)
        if link is None:
            return None
        if link not in _STATE["links"]:
            try:
                _STATE["links"][link] = _load_link(link)
            except Exception as e:                                   # noqa: BLE001
                _STATE["links"][link] = None
                _STATE["error"] = f"{link}: {type(e).__name__}: {e}"
        bodies = _STATE["links"][link]
        return None if bodies is None else bodies.get(label)


def available():
    """True when the pinned meshes are present (a link that then fails to load fails closed on its own)."""
    body("base")
    return bool(_STATE.get("hashes")) and _STATE["links"].get("base_link_inertia") is not None


def why_unavailable():
    available()
    return _STATE.get("error")


def _world(body, frames):
    F = frames[body.frame]
    return body.samples @ F[:3, :3].T + F[:3, 3], body.pieces @ F[:3, :3].T + F[:3, 3]


def pair_lower_bound(la, lb, frames, capsule_of, margin):
    """A lower bound (m) on the distance between the real surfaces of bodies la and lb, allowances included, or None
    when tier 2 is unavailable.  A bound >= margin admits the pair; below it, it is exact for the samples.
    capsule_of: label -> (a, b, r) of the tier-1 shapes in the base frame (for the cropping and the non-UR bodies)."""
    from primitives.shared import collision as C
    non_ur = ("RG2 body", "quick changer")
    if la in non_ur and lb in non_ur:
        return None
    if la in non_ur or lb in non_ur:                          # a UR body against the RG2 box or the quick changer
        ur, other = (lb, la) if la in non_ur else (la, lb)
        surf = body(ur)
        if surf is None:
            return None
        S, _ = _world(surf, frames)
        if other == "RG2 body":
            R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
            L = (S - p6) @ R6
            d = np.linalg.norm(np.maximum(np.abs(L - np.asarray(C.RG2_BODY_BOX[0])) - np.asarray(C.RG2_BODY_BOX[1]), 0.0),
                               axis=1)
        elif other in capsule_of:
            a, b, r = capsule_of[other]
            d = _seg_dist(S, a, b) - r
        else:
            return None
        return float(d.min()) - surf.gap - ALLOWANCE_PER_BODY_M
    A, Bb = body(la), body(lb)
    if A is None or Bb is None:
        return None
    allow = 2 * ALLOWANCE_PER_BODY_M
    reach = margin + A.gap + Bb.gap + allow                    # corners farther apart than this cannot matter
    SA, _ = _world(A, frames)
    SB, PB = _world(Bb, frames)

    def near(P, label, extra=0.0):
        """Points within reach (+ extra) of body label's tier-1 shell (its surface lies inside that shell)."""
        if label == "base":
            return C.base_cylinder_distance(P) < reach + extra
        a, b, r = capsule_of[label]
        return _seg_dist(P, a, b) - r < reach + extra
    # cropping keeps the verdict exact: a dropped sample is >= reach from the other body (module note)
    SA = SA[near(SA, lb)]
    SB = SB[near(SB, la)]
    if not len(SA) or not len(SB):
        return margin                                          # every sample pair is >= reach apart
    dv, _ = cKDTree(SB).query(SA, k=1)
    cheap = float(dv.min()) - A.gap - Bb.gap - allow           # both sides sampled: both covering radii
    if cheap >= margin:
        return cheap
    # exact on lb's side for the samples of la that the cheap bound cannot clear
    need = dv - Bb.gap < margin + A.gap + allow
    P = SA[need]
    # a sample needing the exact pass is inside la's shell and its nearest surface point of lb is within dv < reach,
    # so that point's piece has its centre within reach + rho of la's shell
    PBk = PB[near(PB.mean(axis=1), la, extra=Bb.rho)]
    centres = PBk.mean(axis=1)
    tree = cKDTree(centres)
    best = np.inf
    lists = tree.query_ball_point(P, r=dv[need] + Bb.rho)
    ii = np.repeat(np.arange(len(P)), [len(x) for x in lists])
    jj = np.concatenate([np.asarray(x, int) for x in lists]) if len(ii) else np.zeros(0, int)
    for s in range(0, len(ii), 200000):
        d = point_triangle_distance(P[ii[s:s + 200000]], PBk[jj[s:s + 200000]])
        best = min(best, float(d.min()))
    rest = dv[~need]
    exact_min = min(best, float(rest.min() - Bb.gap) if len(rest) else np.inf)
    return exact_min - A.gap - allow


def table_lower_bound(label, frames, table_z):
    """The lowest point of body label's real surface above table_z (m), allowance included, or None (fail closed)."""
    surf = body(label)
    if surf is None:
        return None
    F = frames[surf.frame]
    z = surf.samples @ F[2, :3] + F[2, 3]
    return float(z.min()) - table_z - ALLOWANCE_PER_BODY_M
