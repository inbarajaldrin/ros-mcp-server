"""The RG2's finger links at any opening (lead ruling 2026-11-20, w5b-tier2): tier 1 must enclose the fingers too.

The cell's pinned rg2.obj is the RG2 at one opening; its six finger links are exactly the per-link visual STLs of
compliant_insertion_studio/urdf/rg2/meshes posed by rg2.urdf's origins (checked: 0.00 mm, tests).  They are moved by the
RG2's parallelogram 4-bar, the same arc model as onrobot_rg2_control.rg2_arc.Rg2Arc (the cell's width readback and
the RG2 driver; its parameters are copied below and checked against it when installed):
  raw width  = 2 (R sin(alpha) + c) + 2 offset,   alpha = alpha_zero + theta,   R = 55 mm,
  the outer knuckle turns about its proximal pivot A by d(alpha), the inner knuckle about its pivot B by the same angle,
  and the finger link translates with the outer knuckle's distal pivot P = A + R (s sin(alpha), 0, cos(alpha)).
Frames: rg2_base_link (the URDF's), then the flange (tool0 -> rg2_base_link rpy 0 0 pi, the cell's launch default).

Openings are a set of raw widths (mm) or None (unknown: the whole stroke, fail safe).  A known width w is taken as the
range [w - 1, w + 2 x 4.6 + 1] mm: the width a primitive knows may be the raw or the rubber gap (they differ by
2 x 4.6 mm, CLAUDE.md), and the real width is within ~1 mm of the command.  Every range is sampled uniformly in alpha
(<= ALPHA_STEP_RAD apart) and each link's travel between samples is added as padding, so the boxes and bounds hold
between samples too.  held(w): a held grip's measured width, [fully closed, w + 2 x 4.6 + 1] (a slip closes the jaws).

Two width conventions use this module side by side, deliberately (review w5b-tier2 r2, R2-3) - do not "fix" one into
the other:
- collision.set_gripper_openings takes the MEASURED rubber gap in mm (control_gripper's final_width_mm =
  /gripper_width_offset), set once per process for its physical self checks; a process that never sets it checks the
  fingers over the whole stroke.
- the *_tool API (finger_boxes_tool, tip_z_tool, ...; gripper_rules, move_down) takes RAW widths in metres, per call.
Both are conservative for either reading: [w - 1, w + 10.2] mm covers raw = gap + 9.2.
"""
from __future__ import annotations

import hashlib
import math
import struct
import threading
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[2] / "compliant_insertion_studio" / "urdf" / "rg2" / "meshes"
RG2_OBJ = MESH_DIR.parent / "rg2.obj"         # the cell's whole-RG2 mesh (its finger parts = the six STLs below)
RG2_OBJ_SHA256 = "d420ca2022260a0584b608da60fa460e12c59a2b7eca9368928a9c85ddcf02cb"
# sha256 of the six per-link visual STLs (the pinned rg2.obj's finger parts)
LINK_SHA256 = {
    "lo_knuckle": "6385dae70b7db2becaad23348d67f49a6dac04a8bb9c2307f7e93d7049c5c4b7",
    "li_knuckle": "3af3d55b8f0de04e0f1664a88752ae169272fcdf658b3c99d06acf7fbe7e7650",
    "l_finger": "70234910499ef9123364f7553ddd6f13e0176297dcb7da8d292594cddc651593",
    "ro_knuckle": "b40da2a42bb6ab40468c501922c5683fafd5be772c7f9928a8a44f4d98f3c6b8",
    "ri_knuckle": "0db96df9d5c2830569612f9e9a50a9816b094578b8bc604eae3496fe7aeae455",
    "r_finger": "9b31a156c994d4573bc4285e207363383e8c5d46e0b0498945077cb9018186e3",
}
# rg2.urdf collision/visual origins (rg2_base_link): xyz, yaw
LINK_ORIGIN = {
    "lo_knuckle": ((-0.017180, 0.0, 0.125800), -1.570797), "li_knuckle": ((-0.007680, 0.0, 0.142300), -1.570797),
    "l_finger": ((-0.056770, 0.0, 0.163970), -1.570797), "ro_knuckle": ((0.017180, 0.0, 0.125800), 1.570797),
    "ri_knuckle": ((0.007680, 0.0, 0.142300), 1.570796), "r_finger": ((0.056770, 0.0, 0.163970), 1.570797),
}
LABELS = {"lo_knuckle": "RG2 finger L outer", "li_knuckle": "RG2 finger L inner", "l_finger": "RG2 finger L tip",
          "ro_knuckle": "RG2 finger R outer", "ri_knuckle": "RG2 finger R inner", "r_finger": "RG2 finger R tip"}

# onrobot_rg2_control.rg2_arc.Rg2ArcParams (the cell's)
ARC_R_M = 0.055
ARC_C_M = -0.00198
ARC_ALPHA_ZERO_RAD = math.radians(27.95)
ARC_OFFSET_M = 0.0046
ARC_WIDTH_MAX_M = 0.110
ARC_THETA_CLOSED_RAD = -0.452

ALPHA_STEP_RAD = 0.005             # opening samples: <= 0.3 mm of travel for any finger point between two samples
KNOWN_WIDTH_BELOW_M = 0.001        # a known width w -> [w - 1 mm, w + 2 offset + 1 mm]
KNOWN_WIDTH_ABOVE_M = 2 * ARC_OFFSET_M + 0.001
SPARE_M = 0.0005

_LOCK = threading.Lock()
_STATE = {}


def _stl(path):
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    if 84 + n * 50 != len(b):
        raise ValueError(f"{path}: not a binary STL")
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].astype(float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def theta_at_raw_max():
    s = (ARC_WIDTH_MAX_M - 2 * ARC_OFFSET_M - 2 * ARC_C_M) / (2 * ARC_R_M)
    return math.asin(max(-1.0, min(1.0, s))) - ARC_ALPHA_ZERO_RAD


def alpha_of_raw(raw_m):
    """Rg2Arc.theta_of_raw_width + alpha_zero (raw width in m), clamped to the RG2's stroke."""
    raw = max(0.0, min(ARC_WIDTH_MAX_M, raw_m))
    s = (raw - 2 * ARC_OFFSET_M - 2 * ARC_C_M) / (2 * ARC_R_M)
    theta = math.asin(max(-1.0, min(1.0, s))) - ARC_ALPHA_ZERO_RAD
    return ARC_ALPHA_ZERO_RAD + max(ARC_THETA_CLOSED_RAD, min(theta_at_raw_max(), theta))


def raw_of_alpha(alpha):
    return 2 * (ARC_R_M * math.sin(alpha) + ARC_C_M) + 2 * ARC_OFFSET_M


ALPHA_MIN = ARC_ALPHA_ZERO_RAD + ARC_THETA_CLOSED_RAD
ALPHA_MAX = ARC_ALPHA_ZERO_RAD + theta_at_raw_max()


def model():
    """{link: (vertices (n, 3) rg2_base_link at the modelled opening, triangles (m, 3, 3), pivot, side)}, the
    modelled alpha, and the per-link radius about its moving pivot; None when a link mesh is missing or not pinned."""
    with _LOCK:
        if "model" not in _STATE:
            try:
                links = {}
                for name, (xyz, yaw) in LINK_ORIGIN.items():
                    path = MESH_DIR / f"{name}_visual.stl"
                    if hashlib.sha256(path.read_bytes()).hexdigest() != LINK_SHA256[name]:
                        raise ValueError(f"{path}: not the pinned mesh")
                    T = _stl(path) @ _rz(yaw).T + np.asarray(xyz)
                    links[name] = T
                A_l = np.asarray(LINK_ORIGIN["lo_knuckle"][0]); P_l = np.asarray(LINK_ORIGIN["l_finger"][0])
                # the URDF's distal pivot is 54.994 mm from A (Rg2Arc: 55.0): the finger moves on the URDF's own arc
                # so the knuckle and the finger stay rigidly coupled; the arc model's 55 mm sets the width only
                _STATE["link_r"] = float(np.hypot(P_l[0] - A_l[0], P_l[2] - A_l[2]))
                alpha0 = math.atan2(abs(P_l[0] - A_l[0]), P_l[2] - A_l[2])
                radius = {}
                for name, T in links.items():
                    V = T.reshape(-1, 3)
                    pivot = np.asarray(LINK_ORIGIN[name][0]) if "knuckle" in name else None
                    # a knuckle point moves on a circle about its pivot; the finger translates with P (radius R)
                    radius[name] = float(np.linalg.norm((V - pivot)[:, [0, 2]], axis=1).max()) if pivot is not None \
                        else ARC_R_M
                _STATE["model"] = (links, alpha0, radius)
            except Exception as e:                                   # noqa: BLE001 - fail closed (callers refuse)
                _STATE["model"] = None
                _STATE["error"] = f"RG2 finger meshes: {type(e).__name__}: {e}"
        return _STATE["model"]


def why_unavailable():
    model()
    return _STATE.get("error")


def _rot_xz(V, pivot, phi):
    """Rotate points V about the y axis through pivot by phi (rg2_base_link)."""
    c, s = math.cos(phi), math.sin(phi)
    D = V - pivot
    out = D.copy()
    out[..., 0] = D[..., 0] * c + D[..., 2] * s
    out[..., 2] = -D[..., 0] * s + D[..., 2] * c
    return out + pivot


def _pose_points(arrays, alpha, alpha0):
    """{link: points (any shape (..., 3)) in the FLANGE frame} at moment-arm angle alpha."""
    d = alpha - alpha0
    out = {}
    for name, T in arrays.items():
        side = -1.0 if name.startswith("l") else 1.0
        if "knuckle" in name:
            # the distal pivot moves on the arc A + R (s sin a, 0, cos a): about the y axis that is a turn of -d on the
            # left (s = -1) and +d on the right (s = +1) - checked against the finger's translation (tests)
            P = _rot_xz(T, np.asarray(LINK_ORIGIN[name][0]), -d if side < 0 else d)
        else:
            A = np.asarray(LINK_ORIGIN["lo_knuckle" if side < 0 else "ro_knuckle"][0])
            r = _STATE.get("link_r", ARC_R_M)
            P = T + r * np.array([side * (math.sin(alpha) - math.sin(alpha0)), 0.0, math.cos(alpha) - math.cos(alpha0)])
        out[name] = P @ np.diag([-1.0, -1.0, 1.0])                 # rg2_base_link -> flange (rpy 0 0 pi)
    return out


def to_link_frame(name, P, alpha, alpha0):
    """Flange-frame points P (..., 3) -> link `name`'s modelled pose (rg2_base_link at alpha0), the inverse of
    _pose_points at alpha.  A rigid motion, so a distance to the link posed at alpha is the distance to its modelled
    triangles from the mapped points: one fixed search tree per link serves every opening."""
    B = np.asarray(P, float) @ np.diag([-1.0, -1.0, 1.0])
    d = alpha - alpha0
    if "knuckle" in name:
        return _rot_xz(B, np.asarray(LINK_ORIGIN[name][0]), d if name.startswith("l") else -d)
    side = -1.0 if name.startswith("l") else 1.0
    r = _STATE.get("link_r", ARC_R_M)
    return B - r * np.array([side * (math.sin(alpha) - math.sin(alpha0)), 0.0, math.cos(alpha) - math.cos(alpha0)])


def posed(alpha):
    """{link: triangles (m, 3, 3) in the FLANGE frame} at moment-arm angle alpha."""
    m = model()
    if m is None:
        return None
    links, alpha0, _ = m
    return _pose_points(links, alpha, alpha0)


HELD = "held"


def held(width_mm):
    """An opening item for a held grip measured at width_mm: the fingers may close from there to fully closed (a slip
    closes the jaws under the grip command), never open past it - [closed, w + 2 offset + 1] (lead ruling 2026-11-20,
    review w5b-tier2 r1 P2-1)."""
    return (HELD, float(width_mm))


def alpha_ranges(openings):
    """The alpha intervals an opening set stands for: None -> the whole stroke; a known width w (mm) ->
    [w - 1, w + 2 offset + 1]; held(w) -> [closed, w + 2 offset + 1] (see the module note)."""
    if openings is None:
        return [(ALPHA_MIN, ALPHA_MAX)]
    out = []
    for item in openings:
        if isinstance(item, tuple) and item and item[0] == HELD:
            out.append((ALPHA_MIN, alpha_of_raw(float(item[1]) / 1000.0 + KNOWN_WIDTH_ABOVE_M)))
            continue
        lo = alpha_of_raw(float(item) / 1000.0 - KNOWN_WIDTH_BELOW_M)
        hi = alpha_of_raw(float(item) / 1000.0 + KNOWN_WIDTH_ABOVE_M)
        out.append((min(lo, hi), max(lo, hi)))
    return out


def _openings_key(openings):
    if openings is None:
        return None
    return tuple(sorted((it[0], round(float(it[1]), 3)) if isinstance(it, tuple) else ("w", round(float(it), 3))
                        for it in openings))


def alpha_samples(openings):
    """Sample angles covering the ranges, <= ALPHA_STEP_RAD apart (ends included)."""
    out = []
    for lo, hi in alpha_ranges(openings):
        n = max(1, int(math.ceil((hi - lo) / ALPHA_STEP_RAD)))
        out.extend(np.linspace(lo, hi, n + 1).tolist())
    return out


def padding(name):
    """Travel of link `name`'s points between two adjacent samples, halved, plus the spare (m)."""
    m = model()
    return 0.5 * m[2][name] * ALPHA_STEP_RAD + SPARE_M


def boxes(openings):
    """{label: (centre, half)} axis-aligned boxes (flange frame) enclosing each finger link over the opening set,
    padded for the sampling; None when the meshes are unavailable (callers fail closed)."""
    key = ("boxes", _openings_key(openings))
    with _LOCK:
        if key in _STATE:
            return _STATE[key]
    if model() is None:
        return None
    lo = {n: np.full(3, np.inf) for n in LINK_ORIGIN}
    hi = {n: np.full(3, -np.inf) for n in LINK_ORIGIN}
    links, alpha0, _ = model()
    verts = {n: np.unique(T.reshape(-1, 3).round(7), axis=0) for n, T in links.items()}
    for a in alpha_samples(openings):
        for name, V in _pose_points(verts, a, alpha0).items():
            lo[name] = np.minimum(lo[name], V.min(axis=0))
            hi[name] = np.maximum(hi[name], V.max(axis=0))
    out = {}
    for name in LINK_ORIGIN:
        p = padding(name)
        out[LABELS[name]] = ((lo[name] + hi[name]) / 2.0, (hi[name] - lo[name]) / 2.0 + p)
    with _LOCK:
        if len(_STATE) > 200:
            for k in [k for k in _STATE if isinstance(k, tuple)]:
                del _STATE[k]
        _STATE[key] = out
    return out


def link_of_label(label):
    return next(n for n, l in LABELS.items() if l == label)


# ---- the shared API (one RG2 finger model for the physical self checks and the gripper-vs-table rule; agreed with
#      w8, lead 2026-11-20).  Openings are RG2 RAW widths (getWidth, /gripper_width) in metres; None = unknown, i.e.
#      the whole stroke.  Frames: tool0 (= the DH flange).
OPENING_RANGE_M = (0.0, ARC_WIDTH_MAX_M)


def _openings_mm(opening_m):
    return None if opening_m is None else (1000.0 * float(opening_m),)


def finger_link_boxes_tool(opening_m=None):
    """{label: (lo, hi)} per finger link, enclosing it over the opening's range (see the module note)."""
    b = boxes(_openings_mm(opening_m))
    return None if b is None else {k: (c - h, c + h) for k, (c, h) in b.items()}


def finger_boxes_tool(opening_m=None):
    """[(lo, hi) left finger, (lo, hi) right finger]: each the union of that side's three link boxes."""
    b = finger_link_boxes_tool(opening_m)
    if b is None:
        return None
    out = []
    for side in ("L", "R"):
        los = [lo for k, (lo, hi) in b.items() if f"finger {side} " in k]
        his = [hi for k, (lo, hi) in b.items() if f"finger {side} " in k]
        out.append((np.min(los, axis=0), np.max(his, axis=0)))
    return out


def worst_finger_points_tool(opening_m=None):
    """(16, 3): the corners of the two finger boxes at the opening (None: the whole stroke)."""
    fb = finger_boxes_tool(opening_m)
    if fb is None:
        return None
    pts = []
    for lo, hi in fb:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    pts.append((x, y, z))
    return np.array(pts)


def tip_z_tool(opening_m=None):
    """The deepest finger point along tool0 +z over the opening's range, padded (the fingertip depth)."""
    fb = finger_boxes_tool(opening_m)
    return None if fb is None else float(max(hi[2] for _lo, hi in fb))
