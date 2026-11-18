# Recovered 2026-11-18 from session ea57148e (L7382, written 2026-10-14) - the 10-13 ChArUco hand-eye that reached ~0.9 mm held-out.
"""Detect ChArUco + base1 markers in every survey frame once; cache flat arrays for the GPU solver.
usage: detect_cache.py out.npz LOC:file1,file2 [...]
"""
import json, sys, multiprocessing as mp
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R

board = cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50))
B1_IDS = {0, 1, 2, 3}

def detect(job):
    path, tcp, loc = job
    cdet = cv2.aruco.CharucoDetector(board)
    mdet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50), cv2.aruco.DetectorParameters())
    img = cv2.imread(path, 0)
    cc, ci, _, _ = cdet.detectBoard(img)
    if ci is None or len(ci) < 12:
        return None
    op, ip = board.matchImagePoints(cc, ci)
    mc, mi, _ = mdet.detectMarkers(img)
    b1 = [(int(i), c.reshape(4, 2)) for c, i in zip(mc, mi.ravel())] if mi is not None else []
    return loc, tcp, op.reshape(-1, 3), ip.reshape(-1, 2), [(i, c) for i, c in b1 if i in B1_IDS]

jobs, locs = [], []
for g in sys.argv[2:]:
    loc, files = g.split(":", 1); locs.append(loc)
    for path in files.split(","):
        for p in json.load(open(path))["poses"]:
            for f in p.get("frames", []):
                jobs.append((f["file"], f["tcp"], loc))
with mp.Pool(24) as pool:
    out = [r for r in pool.map(detect, jobs) if r is not None]
Tee, bv, bt, bp, bo, mv, mid, mo = [], [], [], [], [], [], [], []
for v, (loc, tcp, op, ip, b1) in enumerate(out):
    t = np.array(tcp); M = np.eye(4); M[:3, :3] = R.from_quat(t[3:]).as_matrix(); M[:3, 3] = t[:3]; Tee.append(M)
    bv += [v] * len(op); bt += [locs.index(loc)] * len(op); bp.append(op); bo.append(ip)
    for i, c in b1:
        mv += [v] * 4; mid += [i] * 4; mo.append(c)
np.savez(sys.argv[1], Tee=np.array(Tee), locs=np.array(locs), view_loc=np.array([locs.index(o[0]) for o in out]),
         b_view=np.array(bv), b_tgt=np.array(bt), b_pt=np.concatenate(bp), b_obs=np.concatenate(bo),
         m_view=np.array(mv), m_id=np.array(mid), m_obs=np.concatenate(mo) if mo else np.zeros((0, 2)))
print(f"cached {len(out)}/{len(jobs)} views, {len(bv)} board corners, {len(mv)} base1 marker corners")
