# Recovered 2026-11-18 from session ea57148e (L7215, written 2026-10-14) - the 10-13 ChArUco hand-eye that reached ~0.9 mm held-out.
"""Two-height hand-eye: ChArUco boards on the table (z known) + base1's top-face markers (z known).

Unknowns: X = tcp->camera (6), focal scale f (fx,fy *= f; optional), each board's pose
(roll, pitch, yaw, x, y; plane height PINNED), base1's pose (roll, pitch, yaw, x, y; top face
PINNED), base1 marker edge length (free: annotation's 21 mm may or may not include a border).
Two planes 35 mm apart at known heights separate "focal length off" from "camera sits lower".

usage: final_solve.py LOC:file1,file2 [LOC:...]
env:   HE_BOARD_INIT  board pose json ({"T_world_board": 4x4}) for every LOC (default: identity -> weak start)
       HE_BASE1_XY    "x,y" (m) starting base1 centre (default 0.0513,-0.4855, the 10-13 spot)
       HE_OUT         directory for final_nof.json / final_f.json (default: the first survey's folder)
The starting mount X0 is the localizer config's (primitives.shared.config.LOCALIZER_CONFIG).
"""
import json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

K0 = np.array([[910.13, 0, 619.57], [0, 911.16, 349.15], [0, 0, 1]])
D = np.array([0.130074, -0.439479, -0.001453, -0.000549, 0.369498])
Z_TABLE = -0.0805 + 0.0001          # CAD table top (robot frame) + paper
Z_B1_TOP = -0.0805 + 0.035          # base1 top face (35 mm block on the table)
board = cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))
cdet = cv2.aruco.CharucoDetector(board)
mdet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), cv2.aruco.DetectorParameters())
ann = json.load(open(__import__("pathlib").Path(__file__).resolve().parents[2] / "assets/cad_data/aruco/base1_aruco.json"))
B1 = {m["aruco_id"]: np.array([m["T_object_to_marker"]["position"][k] for k in "xy"]) for m in ann["markers"]}

def T(Rm, t):
    M = np.eye(4); M[:3, :3] = Rm; M[:3, 3] = np.ravel(t); return M

def b1_corners(mid, size, rot90):
    """base1-frame corners (top face, z=0 here; plane pinned separately) in OpenCV order TL,TR,BR,BL."""
    h = size / 2
    c = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])
    c = np.roll(c, rot90, axis=0)
    return np.c_[c + B1[mid], np.zeros(4)]

views = []
for g in sys.argv[1:]:
    loc, files = g.split(":", 1)
    for path in files.split(","):
        for p in json.load(open(path))["poses"]:
            for f in p.get("frames", []):
                fp = Path(f["file"])
                if not fp.exists():                       # survey copied to another machine: frames/ beside survey.json
                    fp = Path(path).parent / "frames" / fp.name
                img = cv2.imread(str(fp), 0)
                cc, ci, _, _ = cdet.detectBoard(img)
                if ci is None or len(ci) < 12:
                    continue
                op, ip = board.matchImagePoints(cc, ci)
                mc, mi, _ = mdet.detectMarkers(img)
                b1 = [(int(i), c.reshape(4, 2)) for c, i in zip(mc, mi.ravel())] if mi is not None else []
                b1 = [(i, c) for i, c in b1 if i in B1]
                tcp = np.array(f["tcp"])
                views.append(dict(loc=loc, op=op.reshape(-1, 3), ip=ip.reshape(-1, 2), b1=b1,
                                  Tee=T(R.from_quat(tcp[3:]).as_matrix(), tcp[:3])))
locs = sorted({v["loc"] for v in views})
print(f"views {len(views)} ({', '.join(f'{l}:{sum(v[chr(108)+chr(111)+chr(99)] == l for v in views)}' for l in locs)}), "
      f"with base1 markers: {sum(bool(v['b1']) for v in views)}, base1 marker detections: {sum(len(v['b1']) for v in views)}")

import yaml
LOCALIZER_CONFIG = os.environ.get("HE_LOCALIZER_CONFIG")       # off the cell PC: the robot_config.yaml copied with the survey
if not LOCALIZER_CONFIG:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from primitives.shared.config import LOCALIZER_CONFIG
_c = yaml.safe_load(open(LOCALIZER_CONFIG))
X0 = T(R.from_quat(_c["cam_offset_quat"]).as_matrix(), _c["cam_offset_position"])
_bi = os.environ.get("HE_BOARD_INIT")
# HE_BOARD_INIT: one file for every LOC, or "LOC=file,LOC=file" (one board spot per survey)
if _bi and "=" in _bi:
    prev = {"boards": {k: json.load(open(v))["T_world_board"] for k, v in (kv.split("=", 1) for kv in _bi.split(","))}}
else:
    prev = {"boards": {l: json.load(open(_bi))["T_world_board"] for l in locs} if _bi else {}}
# HE_FIX_X=<final_*.json>: held-out check - the camera transform fixed at that solve's X, only the targets' poses fitted
_fx = os.environ.get("HE_FIX_X")
if _fx:
    X0 = np.array(json.load(open(_fx))["X"])

def bounds(n, fit_f):
    """X frozen (+-1e-9) under HE_FIX_X; the focal scale kept within +-5 % (unbounded it ran off to 0.0009)."""
    lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    if _fx:
        x0 = np.r_[R.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3]]
        lo[:6], hi[:6] = x0 - 1e-9, x0 + 1e-9
    lo[6], hi[6] = (0.95, 1.05) if fit_f else (1.0 - 1e-9, 1.0 + 1e-9)
    return lo, hi
B1_XY0 = [float(v) for v in os.environ.get("HE_BASE1_XY", "0.0513,-0.4855").split(",")]
OUT_DIR = Path(os.environ.get("HE_OUT") or Path(sys.argv[1].split(":", 1)[1].split(",")[0]).parent)

def init_pose(points_fn, zpin):
    """rough target pose from the X0 chain (mean of per-view PnP), plane at zpin"""
    return None

def plane_pose(x5, zc):
    roll, pitch, yaw, px, py = x5
    return T(R.from_euler("xyz", [roll, pitch, yaw]).as_matrix(), [px, py, zc])

def build(fit_f, rot90):
    nb = len(locs)
    def unpack(x):
        X = T(R.from_rotvec(x[:3]).as_matrix(), x[3:6])
        f = x[6] if fit_f else 1.0
        bo = {l: plane_pose(x[7 + 5 * i:12 + 5 * i], Z_TABLE) for i, l in enumerate(locs)}
        b1 = plane_pose(x[7 + 5 * nb:12 + 5 * nb], Z_B1_TOP)
        msz = x[12 + 5 * nb]
        return X, f, bo, b1, msz
    def res(x):
        X, f, bo, b1, msz = unpack(x)
        K = K0.copy(); K[0, 0] *= f; K[1, 1] *= f
        r = []
        for v in views:
            Tcb = np.linalg.inv(v["Tee"] @ X)
            M = Tcb @ bo[v["loc"]]
            pr, _ = cv2.projectPoints(v["op"], cv2.Rodrigues(M[:3, :3])[0], M[:3, 3], K, D)
            r.append((pr.reshape(-1, 2) - v["ip"]).ravel())
            if v["b1"]:
                M = Tcb @ b1
                for mid, c in v["b1"]:
                    pr, _ = cv2.projectPoints(b1_corners(mid, msz, rot90), cv2.Rodrigues(M[:3, :3])[0], M[:3, 3], K, D)
                    r.append((pr.reshape(-1, 2) - c).ravel())
        return np.concatenate(r)
    return unpack, res

def initial(prev):
    x = [R.from_matrix(X0[:3, :3]).as_rotvec(), X0[:3, 3], [1.0]]
    for l in locs:
        Tb = np.array(prev["boards"][l]) if l in prev["boards"] else None
        if Tb is None:
            Tb = np.eye(4)
        rpy = R.from_matrix(Tb[:3, :3]).as_euler("xyz")
        x.append([rpy[0], rpy[1], rpy[2], Tb[0, 3], Tb[1, 3]])
    x.append([0.0, 0.0, 0.0, B1_XY0[0], B1_XY0[1]])     # base1 near where the localizer sees it
    x.append([0.021])
    return np.concatenate(x)

# base1 yaw: try a coarse grid, the solver refines
def _coarse(args):
    rot90, yaw0 = args
    _, res = build(False, rot90)
    x0 = initial(prev)
    x0[7 + 5 * len(locs) + 2] = yaw0
    s = least_squares(res, x0, loss="soft_l1", f_scale=1.0, max_nfev=60, bounds=bounds(len(x0), False))
    return s.cost, rot90, s.x

# the 16 starts are independent: one process each (fork, so the views loaded above are shared)
starts = [(r, y) for r in range(4) for y in np.radians([0, 90, 180, 270])]
if not any(v["b1"] for v in views):           # no base1 in any frame: its corner order and yaw don't matter
    best = _coarse((0, 0.0))
else:
    with ProcessPoolExecutor(max_workers=min(len(starts), os.cpu_count() or 1), mp_context=mp.get_context("fork")) as ex:
        best = min(ex.map(_coarse, starts), key=lambda t: t[0])
print(f"base1 corner order rot90={best[1]}, yaw {np.degrees(best[2][7 + 5 * len(locs) + 2]) % 360:.1f} deg")
for fit_f in (False, True):
    unpack, res = build(fit_f, best[1])
    sol = least_squares(res, best[2], loss="soft_l1", f_scale=1.0, bounds=bounds(len(best[2]), fit_f))
    X, f, bo, b1, msz = unpack(sol.x)
    e = np.linalg.norm(res(sol.x).reshape(-1, 2), axis=1)
    nb_res = sum(len(v["op"]) for v in views)
    print(f"\n{'focal scale FREE ' if fit_f else 'focal scale fixed'}: RMS {np.sqrt(np.mean(e ** 2)):.3f} px "
          f"(board {np.sqrt(np.mean(e[:0] ** 2)) if False else np.sqrt(np.mean(np.concatenate([e]) ** 2)):.3f}), p95 {np.percentile(e, 95):.2f} px, "
          f"f={f:.4f} -> fx {K0[0,0]*f:.1f} fy {K0[1,1]*f:.1f}")
    print(f"   X t = {np.round(1000 * X[:3, 3], 2).tolist()} mm  quat = {np.round(R.from_matrix(X[:3, :3]).as_quat(), 5).tolist()}"
          f"  rotvec deg = {np.round(np.degrees(R.from_matrix(X[:3, :3]).as_rotvec()), 2).tolist()}")
    print(f"   base1: centre xy {np.round(1000 * b1[:2, 3], 1).tolist()} mm, yaw {np.degrees(R.from_matrix(b1[:3, :3]).as_euler('xyz')[2]):.2f} deg, "
          f"marker edge {1000 * msz:.2f} mm")
    for l in locs:
        print(f"   board {l}: xy {np.round(1000 * bo[l][:2, 3], 1).tolist()} mm, tilt {np.degrees(np.arccos(abs(bo[l][2, 2]))):.2f} deg")
    json.dump({"X": X.tolist(), "f": f, "base1": b1.tolist(), "marker_edge": msz},
              open(OUT_DIR / f"final_{'f' if fit_f else 'nof'}.json", "w"))
