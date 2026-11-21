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


CACHE_DIR = Path(os.environ.get("ARM_SURFACE_CACHE", Path.home() / ".cache" / "fmb-arm-surface"))


def _cache_key(link):
    """The disk cache's key for one link's bodies: the link's pinned mesh and every source file the bodies are derived
    from (this module, collision's capsules, ik's DH, config's nominal DH).  Any change gives a new key: a stale cache
    cannot be read."""
    h = hashlib.sha256()
    h.update(link.encode())
    h.update(MESH_SHA256[VISUAL_ORIGIN[link][1]].encode())
    here = Path(__file__).resolve().parent
    for f in ("arm_surface.py", "collision.py", "ik.py", "config.py"):
        h.update((here / f).read_bytes())
    return h.hexdigest()[:32]


def _load_link_cached(link):
    """_load_link through a disk cache (N-1: the cold load is ~9 s per process on a4500, ~20 s on the cell PC).  Any
    cache problem - missing, unreadable, inconsistent - falls back to the exact computation (and rewrites it)."""
    import json as _json
    path = CACHE_DIR / f"{link}-{_cache_key(link)}.npz"
    try:
        with np.load(path, allow_pickle=False) as z:
            labels = _json.loads(str(z["labels"]))
            out = {}
            for i, lab in enumerate(labels):
                out[lab] = _Body(int(z[f"f{i}"]), z[f"s{i}"], float(z[f"g{i}"]), z[f"p{i}"])
            if out:
                return out
    except Exception:                                            # noqa: BLE001 - recompute
        pass
    out = _load_link(link)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        arrays = {"labels": np.array(_json.dumps(list(out)))}
        for i, (lab, bd) in enumerate(out.items()):
            arrays.update({f"f{i}": np.array(bd.frame), f"s{i}": bd.samples, f"g{i}": np.array(bd.gap),
                           f"p{i}": bd.pieces})
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        # prune: this link's files under other keys are stale (~45 MB per source edit otherwise, review r1 P3-2)
        for old in CACHE_DIR.glob(f"{link}-*.npz"):
            if old != path and ".tmp." not in old.name:
                try:
                    old.unlink()
                except OSError:
                    pass
    except Exception:                                            # noqa: BLE001 - a cache write never fails a check
        pass
    return out


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
                _STATE["links"][link] = _load_link_cached(link)
            except Exception as e:                                   # noqa: BLE001
                _STATE["links"][link] = None
                _STATE["error"] = f"{link}: {type(e).__name__}: {e}"
        bodies = _STATE["links"][link]
        return None if bodies is None else bodies.get(label)


def state_token():
    """What a memoized verdict depends on besides the joints (N-1 caches in collision / camera_up): the load state,
    the mesh locations and the early-exit switch.  A test that resets or redirects any of them gets fresh answers."""
    from primitives.shared import collision as C
    from primitives.shared import rg2_geometry as RF
    from primitives.shared import camera_geometry as CG
    return (id(_STATE), str(VISUAL_DIR), id(_RG2), str(RG2_MESH), EARLY_EXIT, C.gripper_openings(), id(RF._STATE),
            str(RF.MESH_DIR), id(_MOUNT), str(CG.MOUNT_MESH_DIR), tuple(map(repr, MOUNT_EXTRA_SOLIDS_TOOL)))


def preload_async():
    """Load every pinned link's bodies in a daemon thread (N-1: the cold load is ~9 s per process on the cell PC),
    overlapped with the caller's startup (grasp verify, ROS init, the first poses).  body() takes the same lock, so a
    check that needs a link first waits for the load instead of loading it twice.  Returns the thread."""
    import threading as _t

    def load():
        try:
            available()                                          # hashes + the base
            rg2_housing()                                        # the RG2 mesh (~14 MB OBJ)
            for link in DH_FRAME:
                with _LOCK:
                    done = link in _STATE["links"]
                if not done and _STATE.get("hashes"):
                    with _LOCK:
                        if link not in _STATE["links"]:
                            try:
                                _STATE["links"][link] = _load_link_cached(link)
                            except Exception as e:               # noqa: BLE001 - fail closed, as body() does
                                _STATE["links"][link] = None
                                _STATE["error"] = f"{link}: {type(e).__name__}: {e}"
        except Exception:                                        # noqa: BLE001 - a preload never raises
            pass
    th = _t.Thread(target=load, name="arm_surface_preload", daemon=True)
    th.start()
    return th


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


EARLY_EXIT = True          # N-1; tests turn it off to check the early verdict against the exact path


class EarlyRefusal(float):
    """A refusal decided before the exact pass (review w5-enclose r2 N-1): the value is min(sample-to-sample distance)
    less the allowances, an UPPER estimate of the bound, already below the margin - so the exact bound is too.
    Callers compare it like a float; its message must not call it a lower bound."""
    early = True


# The RG2's real housing (lead 2026-11-20, w5b-tier2): the cell's OnRobot mesh, pinned; tier 2 for UR vs "RG2 body".
# Mounted as the cell mounts it: tool0 -> rg2_base_link rpy (0, 0, pi), xyz 0 (ur5e_with_rg2.launch.py defaults).
# The housing is the mesh minus its six finger links (rg2_geometry: they reproduce the mesh's fingers to < 1e-6 m at
# its modelled opening), at every height, up to its top at 148.05 mm; the fingers are their own bodies, posed at the
# measured opening (lead ruling 2026-11-20: tier 2 only - tier 1 keeps RG2_BODY_BOX, which with the quick-changer
# capsule and the finger boxes encloses the whole mesh, tests/test_tier2_mount_and_early_exit.py).
RG2_MESH = Path(__file__).resolve().parents[2] / "compliant_insertion_studio" / "urdf" / "rg2" / "rg2.obj"
RG2_MESH_SHA256 = "d420ca2022260a0584b608da60fa460e12c59a2b7eca9368928a9c85ddcf02cb"
RG2_FINGER_TRIANGLES = 60000      # the six pinned finger links' triangles inside the mesh (10000 each)
RG2_ALLOWANCE_M = 0.0015          # declared, as a UR body's: OnRobot's mesh vs the real housing, and its mounting
_RG2 = {}


def _exact_min(P, tris, centres, radius):
    """Exact min distance from each point P[i] to the triangles whose centre lies within radius[i] of it (the caller
    proves the nearest triangle is among them), in chunks of points so memory stays bounded."""
    tree = cKDTree(centres)
    out = np.full(len(P), np.inf)
    for s0 in range(0, len(P), 500):
        Pc, rc = P[s0:s0 + 500], radius[s0:s0 + 500]
        lists = tree.query_ball_point(Pc, r=rc)
        ii = np.repeat(np.arange(len(Pc)), [len(x) for x in lists])
        if not len(ii):
            continue
        jj = np.concatenate([np.asarray(x, int) for x in lists])
        best = np.full(len(Pc), np.inf)
        for k in range(0, len(ii), 100000):
            d = point_triangle_distance(Pc[ii[k:k + 100000]], tris[jj[k:k + 100000]])
            np.minimum.at(best, ii[k:k + 100000], d)
        out[s0:s0 + 500] = best
    return out


def rg2_housing():
    """(triangles (n, 3, 3) in the flange frame, centres, covering radius) of the RG2 housing: the pinned mesh without
    its finger links, every height.  None when the mesh or the finger links are missing or differ (fail closed: the
    box stays)."""
    with _LOCK:
        if "tri" in _RG2:
            return _RG2["tri"]
    try:
        from primitives.shared import rg2_geometry as RF
        if hashlib.sha256(RG2_MESH.read_bytes()).hexdigest() != RG2_MESH_SHA256:
            raise ValueError(f"{RG2_MESH}: sha256 is not the pinned {RG2_MESH_SHA256[:12]}")
        if RF.model() is None:
            raise ValueError(f"the finger links are unavailable ({RF.why_unavailable()}): the housing cannot be separated")
        V, F = [], []
        for line in open(RG2_MESH):
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        V = np.asarray(V) @ np.diag([-1.0, -1.0, 1.0])                  # rpy (0, 0, pi)
        F = np.asarray(F)
        # the OBJ is the CAD: its fingertip links match the posed links with the measured correction undone
        # (rg2_geometry moves l_finger / r_finger by tip_correction_m; review rc5b P2-1)
        corr = np.array([0.0, 0.0, RF.tip_correction_m() if hasattr(RF, "tip_correction_m") else 0.0])
        fingers = np.vstack([(t - corr if ("finger" in n and "knuckle" not in n) else t).reshape(-1, 3)
                             for n, t in RF.posed(RF.model()[1]).items()])
        d, _ = cKDTree(fingers).query(V)
        finger_tri = (d[F] < 1e-6).all(axis=1)
        if int(finger_tri.sum()) != RG2_FINGER_TRIANGLES:
            raise ValueError(f"{int(finger_tri.sum())} mesh triangles match the finger links, not {RG2_FINGER_TRIANGLES}")
        T = subdivide(V[F[~finger_tri]], EXACT_EDGE_M)                   # the OBJ has triangles ~100 mm across
        c = T.mean(axis=1)
        out = (T, c, float(np.linalg.norm(T - c[:, None, :], axis=2).max()))
    except Exception as e:                                               # noqa: BLE001 - fail closed
        out = None
        with _LOCK:
            _RG2["error"] = f"RG2 mesh: {type(e).__name__}: {e}"
    with _LOCK:
        _RG2["tri"] = out
    return out


def _rg2_lower_bound(L, gap, margin):
    """Lower bound (m) on the distance from UR surface samples L (flange frame, covering radius gap) to the RG2
    housing (the real mesh without its fingers, every height); allowances in.  None: mesh unavailable."""
    hz = rg2_housing()
    if hz is None:
        return None
    T, cent, rho = hz
    allow = ALLOWANCE_PER_BODY_M + RG2_ALLOWANCE_M
    dc, _ = cKDTree(cent).query(L, k=1)                                       # d(sample, tri) >= dc - rho
    lower_h = dc - rho
    # a centroid lies on its triangle, so d(sample, housing) <= dc: the bound returned below is <= upper
    upper = float(dc.min()) - gap - allow
    if EARLY_EXIT and upper < margin:
        return EarlyRefusal(upper)
    need = lower_h - gap - allow < margin
    exact = np.full(len(L), np.inf)
    if need.any():
        # the nearest triangle's centre is within d + rho <= dc + rho of the sample (d <= dc: a centre is on its triangle)
        exact[need] = _exact_min(L[need], T, cent, dc[need] + rho)
    return float(np.where(need, exact, lower_h).min()) - gap - allow


_FINGER = {}


def _finger_link(name):
    """(triangles subdivided to EXACT_EDGE_M, centres, covering radius, centre tree) of finger link `name` in its
    modelled pose, or None when the meshes are unavailable."""
    from primitives.shared import rg2_geometry as RF
    key = (name, id(RF._STATE), str(RF.MESH_DIR))
    with _LOCK:
        hit = _FINGER.get(key)
    if hit is None:
        m = RF.model()
        if m is None:
            return None
        T = subdivide(m[0][name], EXACT_EDGE_M)                     # the STLs have triangles up to 48 mm across
        c = T.mean(axis=1)
        hit = (T, c, float(np.linalg.norm(T - c[:, None, :], axis=2).max()), cKDTree(c))
        with _LOCK:
            _FINGER[key] = hit
    return hit


def _finger_lower_bound(ur, finger, frames, margin):
    """Tier 2 for a UR body vs an RG2 finger link (lead ruling 2026-11-20): the UR mesh's samples against the finger
    link's exact triangles at every sampled opening of the current opening set (rg2_geometry), less the sampling
    travel; allowances UR 1.5 + RG2 1.5 mm.  None (fail closed) when a mesh is missing or the opening is unknown
    (the whole stroke is not re-checked exactly: the tier-1 sweep stands).

    Samples farther than the margin from the link's swept box (it encloses the link over the whole opening set,
    padded) keep the box distance; the rest are mapped into the link's modelled pose at each sampled opening
    (rg2_geometry.to_link_frame, rigid) and measured against its fixed triangles in one batch."""
    from primitives.shared import collision as C
    from primitives.shared import rg2_geometry as RF
    openings = C.gripper_openings()
    surf = body(ur)
    if openings is None or surf is None or RF.model() is None:
        return None
    name = RF.link_of_label(finger)
    fl = _finger_link(name)
    box = (RF.boxes(openings) or {}).get(finger)
    if fl is None or box is None:
        return None
    T, cent, rho, tree = fl
    S, _ = _world(surf, frames)
    R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
    L = (S - p6) @ R6
    slack = surf.gap + ALLOWANCE_PER_BODY_M + RG2_ALLOWANCE_M
    pad = RF.padding(name)
    bc, bh = np.asarray(box[0], float), np.asarray(box[1], float)
    d_box = np.linalg.norm(np.maximum(np.abs(L - bc) - bh, 0.0), axis=1)
    near = d_box - slack < margin
    far = float(d_box[~near].min()) - slack if (~near).any() else np.inf
    if not near.any():
        return far
    alpha0 = RF.model()[1]
    angles = RF.alpha_samples(openings)
    Q = np.stack([RF.to_link_frame(name, L[near], a, alpha0) for a in angles])          # (openings, samples, 3)
    dc, _ = tree.query(Q.reshape(-1, 3), k=1)
    dc = dc.reshape(len(angles), -1)
    # a centre lies on its triangle and every sampled angle is a real opening of the set: an upper estimate
    upper = float(dc.min()) - slack
    if EARLY_EXIT and upper < margin:
        return EarlyRefusal(min(upper, far))
    lower = dc - rho
    need = lower - slack - pad < margin
    d = lower.copy()
    if need.any():
        # the nearest triangle's centre is within d + rho <= dc + rho of the point
        d[need] = _exact_min(Q[need], T, cent, dc[need] + rho)
    return min(float(d.min()) - slack - pad, far)


def pair_lower_bound(la, lb, frames, capsule_of, margin):
    """A lower bound (m) on the distance between the real surfaces of bodies la and lb, allowances included, or None
    when tier 2 is unavailable.  A bound >= margin admits the pair; below it, it is exact for the samples.
    capsule_of: label -> (a, b, r) of the tier-1 shapes in the base frame (for the cropping and the non-UR bodies)."""
    from primitives.shared import collision as C
    non_ur = ("RG2 body", "quick changer")
    if la in non_ur and lb in non_ur:
        return None
    if la.startswith("RG2 finger") or lb.startswith("RG2 finger"):
        return _finger_lower_bound(lb if la.startswith("RG2 finger") else la,
                                   la if la.startswith("RG2 finger") else lb, frames, margin)
    if la in non_ur or lb in non_ur:                          # a UR body against the RG2 box or the quick changer
        ur, other = (lb, la) if la in non_ur else (la, lb)
        surf = body(ur)
        if surf is None:
            return None
        S, _ = _world(surf, frames)
        if other == "RG2 body":
            R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
            L = (S - p6) @ R6
            real = _rg2_lower_bound(L, surf.gap, margin)              # the real housing (w5b-tier2)
            if real is not None:
                return real
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
    # early exit (N-1): lb's samples lie on its surface, so every exact distance below is <= dv and the bound this
    # function would return is <= min(dv) - A.gap - allow.  Below the margin, the refusal is certain: skip the exact
    # pass (it cost up to 4 s per call on a deep pair).
    upper = float(dv.min()) - A.gap - allow
    if EARLY_EXIT and upper < margin:
        return EarlyRefusal(upper)
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



# Extra solids the mount side of tier 2 always includes, as (centre, half) axis-aligned boxes in tool0 (m): parts of
# the camera assembly the pinned mount/D415 meshes do not contain.  TODO (lead 2026-11-20): the D415's USB plug and
# strain relief, which w8 adds to tier 1 as a conservative box (P3-6) - add the same box here when it lands, or tier 2
# would admit through the plug.
MOUNT_EXTRA_SOLIDS_TOOL = []
_MOUNT = {}


def mount_mesh():
    """(triangles subdivided to EXACT_EDGE_M, centres, covering radius, centre tree) of the pinned v3.1 mount and D415
    meshes in tool0, or None when one is missing or not the pinned one (fail closed: the padded-box refusal stands)."""
    from primitives.shared import camera_geometry as CG
    key = (str(CG.MOUNT_MESH_DIR), tuple(sorted(CG.MOUNT_MESH_SHA256)))
    with _LOCK:
        if key in _MOUNT:
            return _MOUNT[key]
    try:
        T = np.vstack([CG.mount_mesh_triangles_tool(n) for n in CG.MOUNT_MESH_SHA256])
        T = subdivide(T, EXACT_EDGE_M)
        c = T.mean(axis=1)
        out = (T, c, float(np.linalg.norm(T - c[:, None, :], axis=2).max()), cKDTree(c))
    except Exception as e:                                               # noqa: BLE001 - fail closed
        out = None
        with _LOCK:
            _MOUNT["error"] = f"mount meshes: {type(e).__name__}: {e}"
    with _LOCK:
        _MOUNT[key] = out
    return out


def _mount_boxes():
    """(lo (B, 3), hi (B, 3)) of the mount's padded boxes in tool0 (camera_geometry: their union encloses the CAD)."""
    from primitives.shared import camera_geometry as CG
    with _LOCK:
        hit = _MOUNT.get("boxes")
    if hit is None:
        boxes = CG._all_boxes(CG.CAMERA_MOUNT_BOX_PAD_M)
        hit = (np.array([b[0] for b in boxes], float), np.array([b[1] for b in boxes], float))
        with _LOCK:
            _MOUNT["boxes"] = hit
    return hit


def mount_unavailable():
    """Why the pinned mount meshes are unavailable, or None."""
    mount_mesh()
    return _MOUNT.get("error")


def mount_lower_bound(label, frames, margin):
    """Tier 2 of the camera-mount rule (lead 2026-11-20, w5b-tier2): a lower bound (m) on the distance between the
    v3.1 mount / D415 and the UR5e's real surface inside the enclosing capsule `label`, allowances included; None when
    tier 2 is unavailable (fail closed); an EarlyRefusal when the refusal is certain before the exact pass.
    The finger method: the UR side is the pinned mesh's samples (their covering gap subtracted), the mount side the
    EXACT pinned mount/D415 triangles (camera_geometry.mount_mesh_triangles_tool; tier 1 keeps the padded boxes, which
    enclose them), plus MOUNT_EXTRA_SOLIDS_TOOL as sampled boxes; allowances 2 x ALLOWANCE_PER_BODY_M.
    First the padded boxes (camera_geometry, CAMERA_MOUNT_BOX_PAD_M: their union encloses the meshes, so a sample's
    exact distance to the nearest box - 0 inside one - is a lower bound on its distance to the mount): samples that
    clear the margin that way keep that bound, and only the rest are measured against the 111k-triangle meshes (the
    seed-48 yellow reorient's route search, review r2 timing).  With MOUNT_EXTRA_SOLIDS_TOOL set, every sample is.""" 
    from primitives.shared.collision import _box_surface
    surf = body(label)
    mesh = mount_mesh()
    if surf is None or mesh is None:
        return None
    T, cent, rho, tree = mesh
    allow = 2 * ALLOWANCE_PER_BODY_M
    S, _ = _world(surf, frames)
    if not len(S):
        return None
    R6, p6 = frames[6][:3, :3], frames[6][:3, 3]
    L = (S - p6) @ R6                                         # the UR samples in tool0
    lo, hi = _mount_boxes()
    dbox = np.min(np.linalg.norm(np.maximum(np.maximum(lo[None] - L[:, None], L[:, None] - hi[None]), 0.0), axis=2),
                  axis=1)
    box_lower = dbox - surf.gap - allow
    near = np.ones(len(L), bool) if MOUNT_EXTRA_SOLIDS_TOOL else box_lower < margin
    far = float(box_lower[~near].min()) if (~near).any() else np.inf
    if not near.any():
        return far
    L = L[near]
    dc, _ = tree.query(L, k=1)
    # a centre lies on its triangle and the samples on the UR surface: the exact distance is <= this
    upper = float(dc.min()) - allow
    extra = np.inf
    for c, h in MOUNT_EXTRA_SOLIDS_TOOL:
        pts, bgap = _box_surface(c, h)
        de, _ = cKDTree(L).query(pts, k=1)
        extra = min(extra, float(de.min()) - surf.gap - bgap - allow)
    if EARLY_EXIT and upper < margin:
        return EarlyRefusal(min(upper, extra, far))
    lower = dc - rho
    need = lower - surf.gap - allow < margin
    d = lower.copy()
    if need.any():
        # the nearest triangle's centre is within d + rho <= dc + rho of the sample
        d[need] = _exact_min(L[need], T, cent, dc[need] + rho)
    return min(float(d.min()) - surf.gap - allow, extra, far)


def table_lower_bound(label, frames, table_z):
    """The lowest point of body label's real surface above table_z (m), allowance included, or None (fail closed)."""
    surf = body(label)
    if surf is None:
        return None
    F = frames[surf.frame]
    z = surf.samples @ F[2, :3] + F[2, 3]
    return float(z.min()) - table_z - ALLOWANCE_PER_BODY_M
