#!/usr/bin/env python3
"""Held part's yaw against its seat, straight from the marker detections of one still frame (2026-10-21).

    python3 -u -m compliant_insertion_studio.wrapper.marker_yaw --object-name u_brown --base-name base1 [--align]
        [--tol-deg 0.5] [--max-iter 4] [--max-turn-deg 25]

The operator, watching u_brown go in: "the insertion yaw is wrong ... the compliant mechanism can handle minor errors,
not when the yaw itself is so off" (the image showed the U 15-20 deg off the base edge). align_at_hover measures
this from the localizer's published poses, but the localizer drops a HELD U part (its pose filters expect a part at
rest: 0 samples in 12 s with the part plainly in the image). Here: one still colour frame, ArUco detections
(the localizer's dictionary, sizes and corner convention), each object's pose in the CAMERA frame by solvePnP over
all its visible markers (CAD layouts from aruco-grasp-annotator/data/aruco), then the part in the base frame against
the CAD seat orientation (fold-aware). Base and part come from the same frame, so the hand-eye cancels.
--align turns the flange about world z by the error (the part turns with it) and re-measures until < tol.
Prints __RESULT_JSON__ {result, iterations, final_err_deg}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
ARUCO_DIR = Path.home() / 'Documents' / 'aruco-grasp-annotator' / 'data' / 'aruco'
STILL_WINDOW_S, STILL_TOL_M = 0.8, 0.0005


def _layout(name):
    d = json.load(open(ARUCO_DIR / f'{name}_aruco.json'))
    total = d['size'] - 2 * d['size'] * d['border_width']        # detectable black square (localizer convention)
    out = {}
    for m in d['markers']:
        T = m['T_object_to_marker']
        T = T if isinstance(T, dict) else json.loads(T.replace("'", '"'))
        q = T['rotation']['quaternion']
        out[int(m['aruco_id'])] = (np.array([T['position'][k] for k in 'xyz']),
                                   R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix(), m.get('edge_m', total))
    return d['aruco_dictionary'], out


def still_frame(timeout=12.0, n_avg=6):
    """(bgr image, tcp xyz): the mean of n_avg colour frames published >= 0.25 s after the TCP stopped moving (the
    mean cuts sensor noise - in the dark, at gain 128, a held part's marker was not detected from one frame)."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    rclpy.init(); n = rclpy.create_node('marker_yaw'); tcp = []; got = {}

    def on_img(m):
        t = time.time()
        win = [c for c in tcp if t - STILL_WINDOW_S - 0.25 <= c[0] <= t - 0.25]
        if len(win) >= 10 and max(np.linalg.norm(np.subtract(c[1:], win[-1][1:])) for c in win) <= STILL_TOL_M:
            got.setdefault('imgs', []).append(m); got['tcp'] = win[-1][1:]
            if len(got['imgs']) >= n_avg:
                got['img'] = True
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose',
                          lambda m: tcp.append((time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z)), 100)
    n.create_subscription(Image, '/camera/camera/color/image_raw', on_img, 2)
    end = time.time() + timeout
    while time.time() < end and 'img' not in got:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    if not got.get('imgs'):
        return None, None
    import cv2
    frames = [np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1).astype(np.float32) for m in got['imgs']]
    a = np.clip(np.mean(frames, 0), 0, 255).astype(np.uint8)
    if got['imgs'][0].encoding == 'rgb8':
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(a), got['tcp']


def object_pose_cam(name, corners, ids, K, D):
    """Rotation of `name` in the camera frame from all its visible markers (solvePnP), or (None, n_markers)."""
    import cv2
    _, lay = _layout(name)
    obj, img, n = [], [], 0
    for c, i in zip(corners, ids):
        i = int(i)
        if i not in lay:
            continue
        t, Rm, edge = lay[i]; h = edge / 2.0
        for lc in ([-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]):
            obj.append(Rm @ np.array(lc) + t)
        img.extend(c.reshape(-1, 2)); n += 1
    if n == 0:
        return None, 0
    ok, rvec, tvec = cv2.solvePnP(np.array(obj, np.float32), np.array(img, np.float32), K, D,
                                  flags=cv2.SOLVEPNP_IPPE if n == 1 else cv2.SOLVEPNP_ITERATIVE)
    return (R.from_rotvec(rvec.ravel()) if ok else None), n


def _detect(img, dict_name):
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)),
                                  cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) < 3:                                 # dim: equalise contrast and try again
        c2, i2, _ = det.detectMarkers(cv2.createCLAHE(3.0, (8, 8)).apply(gray))
        if i2 is not None and (ids is None or len(i2) > len(ids)):
            corners, ids = c2, i2
    return corners, (None if ids is None else ids.ravel())


def _intrinsics():
    from aruco_camera_localizer.robot_config import RobotConfig
    rc = RobotConfig()                                              # the localizer's calibrated intrinsics
    fx, fy, cx, cy = rc.camera_matrix
    return (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64),
            np.array(rc.distortion_coeffs, np.float64).reshape(-1, 1))


def base_pose_world(base, img=None, tcp_pose=None):
    """((x, y, z), quat_xyzw) of `base` in the robot frame from its markers in one still frame (solvePnP over all
    visible base markers, then flange pose x hand-eye), or (None, info). Works with a part in hand: the held part
    covers the image bottom, the base corners stay in view."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    if img is None:
        img, _ = still_frame()
        tcp_pose = _tcp_pose_now()
    if img is None or tcp_pose is None:
        return None, {'error': 'no still frame'}
    dict_name, lay = _layout(base)
    corners, ids = _detect(img, dict_name)
    if ids is None:
        return None, {'error': 'no markers'}
    K, D = _intrinsics()
    obj, imgp, used = [], [], []
    for c, i in zip(corners, ids):
        if int(i) not in lay:
            continue
        t, Rm, edge = lay[int(i)]; h = edge / 2.0
        for lc in ([-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]):
            obj.append(Rm @ np.array(lc) + t)
        imgp.extend(c.reshape(-1, 2)); used.append(int(i))
    if len(used) < 2:
        return None, {'error': f'{len(used)} base markers seen', 'markers': used}
    ok, rvec, tvec = cv2.solvePnP(np.array(obj, np.float32), np.array(imgp, np.float32), K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, {'error': 'solvePnP failed'}
    proj, _ = cv2.projectPoints(np.array(obj, np.float32), rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - np.array(imgp)) ** 2, 1))))
    Tcb = np.eye(4); Tcb[:3, :3] = R.from_rotvec(rvec.ravel()).as_matrix(); Tcb[:3, 3] = tvec.ravel()
    Tw = _T(tcp_pose[0], tcp_pose[1]) @ X_CAM @ Tcb
    q = R.from_matrix(Tw[:3, :3]).as_quat()
    return (tuple(float(v) for v in Tw[:3, 3]), tuple(float(v) for v in q)), {'markers': used, 'reproj_px': round(rms, 2)}


def _tcp_pose_now():
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init(); n = rclpy.create_node('marker_yaw_tcp'); s = {}
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: s.__setitem__('p', m.pose), 5)
    t = time.time()
    while 'p' not in s and time.time() - t < 3:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node(); rclpy.shutdown()
    if 'p' not in s:
        return None
    p = s['p']
    return ((p.position.x, p.position.y, p.position.z), (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))


DARK_R, RED_R = 110, 130            # colour image red channel: part top <= 110, base top >= 130 (lit or long exposure)


def held_offset(obj, base, held_q, grasp_id=1, img=None, tcp_pose=None, debug_path=None):
    """In-hand error of a held part, from ONE still frame of it over its base (operator 10-21: "do you see how much
    offset there is" - the part ~5-10 mm off its socket while the chained pose put it over it).

    Base pose from its markers in this frame (solvePnP); the part where the held-pose chain says it is (seat pose
    moved with the TCP: predict_tcp_at_seat); its CAD top-face outline projected; along each edge, the real edge is
    the dark (part) -> red (base top) transition searched along the edge normal, kept only where red lies outside;
    the 2-D image shift that best fits all edge offsets, scaled at the top face's depth, is the part's error in the
    camera frame -> world. Returns (dxyz_world_m [real - chained], info) or (None, info)."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset, held_tool_offset_for
    if img is None:
        img, _ = still_frame(n_avg=3); tcp_pose = _tcp_pose_now()
    if img is None or tcp_pose is None:
        return None, {'error': 'no still frame'}
    bpose, binfo = base_pose_world(base, img=img, tcp_pose=tcp_pose)
    if bpose is None:
        return None, dict(binfo, error='base not read')
    bp, bq = np.array(bpose[0]), bpose[1]; Rb = R.from_quat(bq)
    p_now, q_now = np.asarray(tcp_pose[0]), tcp_pose[1]
    K, D = _intrinsics()
    Tcw = np.linalg.inv(_T(p_now, q_now) @ X_CAM)
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=bp,
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=list(q_now),
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); dR = R.from_quat(q_now) * R.from_quat(pred['predicted_tcp_at_seat']['quat_xyzw']).inv()
    sp, sq = load_seat_pose_in_base(base, obj)
    V, F = load_mesh(obj); Pb = R.from_quat(sq).apply(surface_points(V, F, 0.002)) + sp
    top = Pb[:, 2].max(); x0, x1, y0, y1 = Pb[:, 0].min(), Pb[:, 0].max(), Pb[:, 1].min(), Pb[:, 1].max()
    corners_b = np.array([[x0, y0, top], [x1, y0, top], [x1, y1, top], [x0, y1, top]])
    Wc = dR.apply(Rb.apply(corners_b) + bp - st) + p_now            # chained top-face corners, world
    Pc = (Tcw[:3, :3] @ Wc.T).T + Tcw[:3, 3]
    uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
    Rch = img[:, :, 2].astype(np.float32)
    H, Wd = Rch.shape; cen = uv.mean(0)
    A, b = [], []
    for k in range(4):
        p, q = uv[k], uv[(k + 1) % 4]; e = q - p; L = np.linalg.norm(e)
        if L < 20:
            continue
        n = np.array([-e[1], e[0]]) / L
        if np.dot(n, (p + q) / 2 - cen) < 0:
            n = -n                                                  # outward normal
        for s_ in np.linspace(0.2, 0.8, 25):
            c = p + s_ * e
            ts = np.arange(-60, 61)
            pts = np.round(c[None] + ts[:, None] * n[None]).astype(int)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < Wd) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
            if not ok.all():
                continue
            r = Rch[pts[:, 1], pts[:, 0]]
            # the part -> base edge: a dark sample (part top <= 90) whose next 3 outward average >= 115 (a 1-2 px shadow
            # band at ~110 sits between them), outermost such crossing that still has red within 10 px beyond
            r = np.convolve(r, np.ones(3) / 3, mode='same')
            hit = None
            for i in range(1, len(ts) - 11):
                if r[i] <= 90 and r[i + 1:i + 4].mean() >= 115 and r[i + 1:i + 11].max() >= RED_R:
                    hit = i
            if hit is not None:
                A.append(n); b.append(ts[hit] + 0.5)
    if len(A) < 8:
        return None, dict(binfo, error=f'edges not found ({len(A)} samples)')
    A, b = np.array(A), np.array(b)
    shift, *_ = np.linalg.lstsq(A, b, rcond=None)                   # image shift real - chained (px)
    resid = float(np.sqrt(np.mean((A @ shift - b) ** 2)))
    Z = float(Pc[:, 2].mean())
    d_cam = np.array([shift[0] * Z / K[0, 0], shift[1] * Z / K[1, 1], 0.0])
    d_world = (_T(p_now, q_now) @ X_CAM)[:3, :3] @ d_cam
    info = dict(binfo, shift_px=[round(float(v), 1) for v in shift], resid_px=round(resid, 1), n_edge=len(A),
                depth_m=round(Z, 3), dxy_mm=[round(1000 * float(v), 1) for v in d_world[:2]])
    if debug_path:
        v = img.copy()
        cv2.polylines(v, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
        cv2.polylines(v, [np.round(uv + shift).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        cv2.imwrite(debug_path, v)
    return d_world, info


def socket_offset(obj, base, lift_m, img=None, tcp_pose=None, debug_path=None):
    """How far the held part's top face is from where it must be to sit over its socket, measured in ONE still frame
    with no hand-eye and no held-pose chain in the comparison (operator 10-21: "the yaw fix seemed to work but not the
    offset"). Target = the part's CAD top-face outline at its seat, raised by `lift_m` (how high the part is above its
    seat now), placed by the base pose solved from the base's own markers in THIS frame (camera frame). Real = the
    part's top-face edges (dark part -> red base). The image shift between them at the face's depth -> camera frame ->
    world (rotation only). Returns (d_world_xy real - target, info) or (None, info)."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    if img is None:
        img, _ = still_frame(n_avg=3); tcp_pose = _tcp_pose_now()
    if img is None or tcp_pose is None:
        return None, {'error': 'no still frame'}
    dict_name, lay = _layout(base)
    corners, ids = _detect(img, dict_name)
    if ids is None:
        return None, {'error': 'no markers'}
    K, D = _intrinsics()
    obj_pts, imgp, used = [], [], []
    for c, i in zip(corners, ids):
        if int(i) not in lay:
            continue
        t, Rm, edge = lay[int(i)]; h = edge / 2.0
        for lc in ([-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]):
            obj_pts.append(Rm @ np.array(lc) + t)
        imgp.extend(c.reshape(-1, 2)); used.append(int(i))
    if len(used) < 2:
        return None, {'error': f'{len(used)} base markers'}
    ok, rvec, tvec = cv2.solvePnP(np.array(obj_pts, np.float32), np.array(imgp, np.float32), K, D)
    Rcb = R.from_rotvec(rvec.ravel()); tcb = tvec.ravel()
    sp, sq = load_seat_pose_in_base(base, obj)
    V, F = load_mesh(obj); Pb = R.from_quat(sq).apply(surface_points(V, F, 0.002)) + sp
    top = Pb[:, 2].max(); x0, x1, y0, y1 = Pb[:, 0].min(), Pb[:, 0].max(), Pb[:, 1].min(), Pb[:, 1].max()
    corners_b = np.array([[x0, y0, top + lift_m], [x1, y0, top + lift_m], [x1, y1, top + lift_m], [x0, y1, top + lift_m]])
    Pc = Rcb.apply(corners_b) + tcb                                   # target top face, camera frame
    uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
    shift, A, resid = _edge_shift(img, uv)
    if shift is None:
        return None, {'error': f'edges not found ({A} samples)', 'markers': used}
    Z = float(Pc[:, 2].mean())
    d_cam = np.array([shift[0] * Z / K[0, 0], shift[1] * Z / K[1, 1], 0.0])
    d_world = (_T(tcp_pose[0], tcp_pose[1]) @ X_CAM)[:3, :3] @ d_cam
    info = {'markers': used, 'shift_px': [round(float(v), 1) for v in shift], 'resid_px': round(resid, 1), 'n_edge': A,
            'depth_m': round(Z, 3), 'dxy_mm': [round(1000 * float(v), 1) for v in d_world[:2]]}
    if debug_path:
        v = img.copy()
        cv2.polylines(v, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        cv2.polylines(v, [np.round(uv + shift).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        cv2.imwrite(debug_path, v)
    return d_world[:2], info


def base_pose_same_frame(base, img, tcp_pose, bq_chain, yaw_search_deg=6.0, step_deg=0.05):
    """Base pose in the robot frame from its markers in THIS frame, for a target projected into the SAME frame: the
    tilt comes from the chain's base quat, the in-plane yaw is searched +-yaw_search_deg and the translation solved
    linearly from the marker corners (2026-10-21 23:xx: 2 visible markers, 1.6 px reprojection, repeats to 0.3 mm /
    0.1 deg; a free 2-marker PnP cannot pin depth and rotation together: 8.8 px). Projecting the socket from THIS
    pose cancels the hand-eye for the comparison; a pose read at another height carries the hand-eye's rotation error
    x the height difference (the hover servo's bias: 0.1 mm reported with the part ~5 mm off).
    Returns (((x, y, z), quat_xyzw), info) or (None, info)."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    dict_name, lay = _layout(base)
    corners, ids = _detect(img, dict_name)
    det = {int(i): c.reshape(-1, 2) for c, i in zip(corners, ids)} if ids is not None else {}
    bid = [i for i in det if i in lay]
    if len(bid) < 2:
        return None, {'error': f'{len(bid)} base markers'}
    K, D = _intrinsics(); Twc = _T(tcp_pose[0], tcp_pose[1]) @ X_CAM; Tcw = np.linalg.inv(Twc)
    pts = np.concatenate([_marker_corners(lay, i) for i in bid]); pix = np.concatenate([det[i] for i in bid])
    nrm = cv2.undistortPoints(np.array(pix, np.float32).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    e = R.from_quat(bq_chain).as_euler('xyz'); tilt = R.from_euler('xyz', [e[0], e[1], 0.0])

    def solve(Rc):
        A, b = [], []
        for X, (u, v) in zip(pts, nrm):
            Xc = Rc @ X; A += [[1, 0, -u], [0, 1, -v]]; b += [u * Xc[2] - Xc[0], v * Xc[2] - Xc[1]]
        t = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
        uv, _ = cv2.projectPoints(((Rc @ pts.T).T + t).astype(np.float32), np.zeros(3), np.zeros(3), K, D)
        return t, float(np.linalg.norm(uv.reshape(-1, 2) - pix, axis=1).mean())
    best = None
    for dz in np.arange(-yaw_search_deg, yaw_search_deg + 1e-9, step_deg):
        yaw = float(e[2] + np.radians(dz)); Rwb = R.from_euler('z', yaw) * tilt
        t, rp = solve(Tcw[:3, :3] @ Rwb.as_matrix())
        if best is None or rp < best[1]:
            best = (t, rp, yaw, Rwb)
    t, rp, yaw, Rwb = best
    bw = Twc[:3, :3] @ t + Twc[:3, 3]
    return ((float(bw[0]), float(bw[1]), float(bw[2])), tuple(float(v) for v in Rwb.as_quat())), \
        {'markers': bid, 'reproj_px': round(rp, 2), 'yaw_deg': round(float(np.degrees(yaw)), 2),
         'yaw_minus_chain_deg': round(float(np.degrees(yaw - e[2])), 2), 'xy_mm': [round(1000 * float(v), 1) for v in bw[:2]]}


def _marker_corners(lay, i):
    t, Rm, edge = lay[i]; h = edge / 2.0
    return np.array([Rm @ np.array(lc) + t for lc in ([-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0])])


def held_outline_px(img, x_far=None, y_left=None):
    """Held part's far (top) and left edges in the image, over BARE TABLE (coloured part vs dark table). The camera is
    rigid with the gripper, so these pixels are the part's place in every frame of this grasp. The part is found as its
    own blob - the largest coloured region reaching the bottom of the image between the fingers - not in fixed windows
    (10-22 03:15: the old windows were u_brown's place in the image; held u_orange reaches x 280, every left scan began
    inside it at x 520 and the fit spread 25 px). Returns {'far': [(x, y)...], 'left': [(x, y)...]} or None."""
    import cv2
    Rc, Gc, Bc = [img[:, :, k].astype(np.float32) for k in (2, 1, 0)]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((Rc > Gc + 6) & (Rc > 14)) | ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 35))   # dark-table noise is saturated at V < 20
    m8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # the held part's own colour: the median of the coloured pixels between the fingers. Pixels unlike it (base1 beside
    # the table view - bright orange-red against maroon u_brown, overlapping it in the image 10-22 07:00) are dropped
    # before the blob is chosen (erosion alone did not cut the wide overlap)
    H0, W0 = mask.shape
    bs = m8[int(0.64 * H0):H0, int(0.48 * W0):int(0.66 * W0)] > 0
    if bs.sum() >= 300:
        hb = hsv[int(0.64 * H0):H0, int(0.48 * W0):int(0.66 * W0)][bs].astype(np.float32)
        hb = hb[hb[:, 2] >= max(40.0, float(np.percentile(hb[:, 2], 50)))]   # the part, not the dark finger pixels
        hs, vs = float(np.median(hb[:, 0])), float(np.median(hb[:, 2]))
        dh = np.abs(hsv[:, :, 0].astype(np.float32) - hs); dh = np.minimum(dh, 180 - dh)
        like = (dh <= 12) & (np.abs(hsv[:, :, 2].astype(np.float32) - vs) <= max(35.0, 0.65 * vs))   # side faces are darker
        m8 = (m8 > 0) & like
        m8 = cv2.morphologyEx(m8.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # a part touching another coloured thing in the image (the base beside the table view) merges with it: cut thin
    # bridges by eroding before choosing, then grow the chosen blob back inside the mask
    K = np.ones((15, 15), np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.erode(m8, K))
    H, W = mask.shape
    # the held part is the blob with the most pixels in the band between the fingers at the image bottom
    # (rows from 0.64 H: the held part's front face sits just above the fingers - u_brown at rows 480-580 of 720 with the
    # lights on, 10-22 04:22; the bottom band alone was fingers and the tool marker)
    # (columns 0.48-0.66 W: the gripper's centre, where only the held part can be - base1 beside the table view reached
    # x 862 of the old 0.44-0.70 band and outvoted u_brown's front face, 10-22 04:30)
    band = lab[int(0.64 * H):H, int(0.48 * W):int(0.66 * W)]
    ids, cnt = np.unique(band[band > 0], return_counts=True)
    if not len(ids) or cnt.max() < 300:
        return None
    best = int(ids[np.argmax(cnt)])
    blob = (cv2.dilate((lab == best).astype(np.uint8), K) > 0) & (m8 > 0)
    cols = np.nonzero(blob.any(0))[0]
    far, left = [], []
    for x in range(int(cols.min()) + 12, int(cols.max()) - 12, 8):
        ys = np.nonzero(blob[:, x])[0]
        if len(ys):
            far.append((x, int(ys.min())))
    if len(far) >= 5:
        # the far edge is the MOST COMMON top row (the side face's slant and the marker strip are fewer columns)
        ys_ = np.array([y for _, y in far]); h_, e_ = np.histogram(ys_, bins=np.arange(ys_.min(), ys_.max() + 4, 3))
        mode = e_[np.argmax(h_)] + 1.5
        far = [(x, y) for x, y in far if abs(y - mode) <= 6]
    if len(far) < 5:
        return None
    # the left SILHOUETTE: from inside the part (just right of the far edge's left end) walk left along the row through
    # the part, gaps <= 3 px, to its last pixel - the side face the camera sees counts (10-22 03:25: a window around the
    # far edge's end found u_orange's top-face corner, 96 px inside its silhouette)
    fx0, top = int(far[0][0]), int(np.median([y for _, y in far]))
    for y in range(top + 8, min(H - 1, top + 8 + 120), 6):
        row = m8[y] > 0
        x = min(W - 1, fx0 + 6)
        if not row[x]:
            continue
        gap = 0; last = x
        while x > 0 and gap <= 3:
            x -= 1
            if row[x]:
                last = x; gap = 0
            else:
                gap += 1
        left.append((last, y))
    if len(left) >= 5:
        # the left silhouette may slant (a visible side face): keep the points within 4 px of a robust line x = a*y + b
        # (rows that cross a marker stop at it and fall off the line)
        L = np.array(left, float); best_in = None
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if L[j, 1] == L[i, 1]:
                    continue
                a_ = (L[j, 0] - L[i, 0]) / (L[j, 1] - L[i, 1]); b_ = L[i, 0] - a_ * L[i, 1]
                inl = np.abs(L[:, 0] - (a_ * L[:, 1] + b_)) <= 4
                if best_in is None or inl.sum() > best_in.sum():
                    best_in = inl
        left = [tuple(int(v) for v in L[k]) for k in range(len(L)) if best_in is not None and best_in[k]]
    if len(far) < 10 or len(left) < 6:
        return None
    return {'far': far, 'left': left}


def socket_shift(obj, base, lift_m, outline, img=None, tcp_pose=None, debug_path=None):
    """Image shift (real - target) between the held part's recorded outline (held_outline_px) and its CAD top face at
    the seat raised by lift_m, placed by the base's own markers in THIS frame (no hand-eye, no held-pose chain),
    converted to world xy at the face's depth. (d_world_xy, info) or (None, info)."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    if img is None:
        img, _ = still_frame(n_avg=3); tcp_pose = _tcp_pose_now()
    dict_name, lay = _layout(base)
    corners, ids = _detect(img, dict_name)
    if ids is None:
        return None, {'error': 'no markers'}
    K, D = _intrinsics()
    op, ip, used = [], [], []
    for c, i in zip(corners, ids):
        if int(i) not in lay:
            continue
        t, Rm, edge = lay[int(i)]; h = edge / 2.0
        for lc in ([-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]):
            op.append(Rm @ np.array(lc) + t)
        ip.extend(c.reshape(-1, 2)); used.append(int(i))
    if len(used) < 2:
        return None, {'error': f'{len(used)} base markers'}
    ok, rvec, tvec = cv2.solvePnP(np.array(op, np.float32), np.array(ip, np.float32), K, D)
    Rcb = R.from_rotvec(rvec.ravel()); tcb = tvec.ravel()
    sp, sq = load_seat_pose_in_base(base, obj)
    V, F = load_mesh(obj); Pb = R.from_quat(sq).apply(surface_points(V, F, 0.002)) + sp
    top = Pb[:, 2].max(); x0, x1, y0, y1 = Pb[:, 0].min(), Pb[:, 0].max(), Pb[:, 1].min(), Pb[:, 1].max()
    cb = np.array([[x0, y0, top + lift_m], [x1, y0, top + lift_m], [x1, y1, top + lift_m], [x0, y1, top + lift_m]])
    Pc = Rcb.apply(cb) + tcb
    uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
    edges = [(uv[k], uv[(k + 1) % 4]) for k in range(4)]
    top_e = min(edges, key=lambda e: (e[0][1] + e[1][1]) / 2)        # the far edge (smallest image y)
    left_e = min(edges, key=lambda e: (e[0][0] + e[1][0]) / 2)       # the left edge (smallest image x)

    def y_at(e, x):
        (xa, ya), (xb, yb) = e
        return ya + (yb - ya) * (x - xa) / (xb - xa) if abs(xb - xa) > 1e-6 else None

    def x_at(e, y):
        (xa, ya), (xb, yb) = e
        return xa + (xb - xa) * (y - ya) / (yb - ya) if abs(yb - ya) > 1e-6 else None
    dy = [y - y_at(top_e, x) for x, y in outline['far'] if y_at(top_e, x) is not None]
    dx = [x - x_at(left_e, y) for x, y in outline['left'] if x_at(left_e, y) is not None]
    if len(dy) < 5 or len(dx) < 3:
        return None, {'error': 'target edges out of range'}
    shift = np.array([np.median(dx), np.median(dy)])
    Z = float(Pc[:, 2].mean())
    d_cam = np.array([shift[0] * Z / K[0, 0], shift[1] * Z / K[1, 1], 0.0])
    d_world = (_T(tcp_pose[0], tcp_pose[1]) @ X_CAM)[:3, :3] @ d_cam
    info = {'markers': used, 'shift_px': [round(float(v), 1) for v in shift],
            'spread_px': [round(float(np.std(dx)), 1), round(float(np.std(dy)), 1)], 'depth_m': round(Z, 3),
            'dxy_mm': [round(1000 * float(v), 1) for v in d_world[:2]]}
    if debug_path:
        v = img.copy()
        cv2.polylines(v, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for x, y in outline['far'] + outline['left']:
            cv2.circle(v, (int(x), int(y)), 3, (255, 0, 255), -1)
        cv2.imwrite(debug_path, v)
    return d_world[:2], info


def socket_shift_world(obj, base, base_pose_world, lift_m, outline, img=None, tcp_pose=None, debug_path=None):
    """socket_shift with the base from a WORLD pose (the 6-dof CAD fit read with the wrist in the SAME orientation as
    now) projected into this frame through the hand-eye: with equal wrist orientations the mount's translation error
    enters the read and the projection identically and cancels (only the arm's exact relative motion remains). From the
    hover the held part hides two of the base's four markers, and a base pose solved from the remaining two jittered the
    projected socket by tens of px (servo oscillated +-5 mm, 10-21)."""
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    if img is None:
        img, _ = still_frame(n_avg=3); tcp_pose = _tcp_pose_now()
    K, D = _intrinsics()
    Tcw = np.linalg.inv(_T(tcp_pose[0], tcp_pose[1]) @ X_CAM)
    bp, bq = np.asarray(base_pose_world[0], float), base_pose_world[1]; Rb = R.from_quat(bq)
    sp, sq = load_seat_pose_in_base(base, obj)
    V, F = load_mesh(obj); Pb = R.from_quat(sq).apply(surface_points(V, F, 0.0015)) + sp
    Pb = Pb + np.array([0.0, 0.0, lift_m])                           # the whole part at its seat, raised to now
    # SILHOUETTE, not the top face: the camera looks at the part from up-left of it and sees its far and left SIDE
    # faces, so the outermost dark edges are the part's bottom edges (10-21: top-face target -> servo converged off)
    Pw = Rb.apply(Pb) + bp
    Pc = (Tcw[:3, :3] @ Pw.T).T + Tcw[:3, 3]
    uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
    dy, dx, zs = [], [], []
    for x, y in outline['far']:
        m = np.abs(uv[:, 0] - x) < 2.5
        if m.sum() >= 3:
            i = np.argmin(np.where(m, uv[:, 1], np.inf)); dy.append(y - uv[i, 1]); zs.append(Pc[i, 2])
    for x, y in outline['left']:
        m = np.abs(uv[:, 1] - y) < 2.5
        if m.sum() >= 3:
            i = np.argmin(np.where(m, uv[:, 0], np.inf)); dx.append(x - uv[i, 0]); zs.append(Pc[i, 2])
    if len(dy) < 5 or len(dx) < 3:
        return None, {'error': 'silhouette edges out of range'}
    shift = np.array([np.median(dx), np.median(dy)]); Z = float(np.median(zs))
    # yaw: the far edge's slope difference (real - target) across its x span = the in-plane rotation between them
    fx = [x for x, y in outline['far'] if (np.abs(uv[:, 0] - x) < 2.5).sum() >= 3]
    yaw_img = None
    if len(fx) >= 8 and len(fx) == len(dy):
        b = np.polyfit(np.asarray(fx, float), np.asarray(dy, float), 1)[0]   # d(dy)/dx in px/px
        yaw_img = float(np.degrees(np.arctan(b)))
    d_cam = np.array([shift[0] * Z / K[0, 0], shift[1] * Z / K[1, 1], 0.0])
    Rwc = (_T(tcp_pose[0], tcp_pose[1]) @ X_CAM)[:3, :3]
    d_world = Rwc @ d_cam
    d_far_only = Rwc @ np.array([0.0, shift[1] * Z / K[1, 1], 0.0])    # the far edge's (image-y) component alone
    def _mad(v):
        v = np.asarray(v, float); return float(1.4826 * np.median(np.abs(v - np.median(v))))
    info = {'d_far_only_mm': [round(1000 * float(v), 1) for v in d_far_only[:2]],
            'shift_px': [round(float(v), 1) for v in shift],
            'spread_px': [round(float(np.std(dx)), 1), round(float(np.std(dy)), 1)], 'depth_m': round(Z, 3),
            # robust spreads and the far edge's residuals per column (10-23: line_green's far edge read std 101.7 px with
            # a clean-looking edge - which columns disagree, and is the median sound?)
            'spread_mad_px': [round(_mad(dx), 1), round(_mad(dy), 1)],
            'far_dy_px': [round(float(v), 1) for v in dy], 'far_x_px': [round(float(v), 1) for v in fx],
            'dxy_mm': [round(1000 * float(v), 1) for v in d_world[:2]],
            'yaw_img_deg': None if yaw_img is None else round(yaw_img, 2)}
    if debug_path:
        v = img.copy()
        for u_, w_ in uv[::4]:
            if 0 <= u_ < v.shape[1] and 0 <= w_ < v.shape[0]:
                v[int(w_), int(u_)] = (0, 255, 0)
        for x, y in outline['far'] + outline['left']:
            cv2.circle(v, (int(x), int(y)), 3, (255, 0, 255), -1)
        cv2.imwrite(debug_path, v)
    return d_world[:2], info


def _outline_vs_target(img, uv, Pc, outline, tcp_pose, K, debug_path=None):
    import cv2
    from seat_check import X_CAM
    from handeye_icp_survey import T as _T
    edges = [(uv[k], uv[(k + 1) % 4]) for k in range(4)]
    top_e = min(edges, key=lambda e: (e[0][1] + e[1][1]) / 2)
    left_e = min(edges, key=lambda e: (e[0][0] + e[1][0]) / 2)

    def y_at(e, x):
        (xa, ya), (xb, yb) = e
        return ya + (yb - ya) * (x - xa) / (xb - xa) if abs(xb - xa) > 1e-6 else None

    def x_at(e, y):
        (xa, ya), (xb, yb) = e
        return xa + (xb - xa) * (y - ya) / (yb - ya) if abs(yb - ya) > 1e-6 else None
    dy = [y - y_at(top_e, x) for x, y in outline['far'] if y_at(top_e, x) is not None]
    dx = [x - x_at(left_e, y) for x, y in outline['left'] if x_at(left_e, y) is not None]
    if len(dy) < 5 or len(dx) < 3:
        return None, {'error': 'target edges out of range'}
    shift = np.array([np.median(dx), np.median(dy)])
    Z = float(Pc[:, 2].mean())
    d_cam = np.array([shift[0] * Z / K[0, 0], shift[1] * Z / K[1, 1], 0.0])
    d_world = (_T(tcp_pose[0], tcp_pose[1]) @ X_CAM)[:3, :3] @ d_cam
    info = {'shift_px': [round(float(v), 1) for v in shift],
            'spread_px': [round(float(np.std(dx)), 1), round(float(np.std(dy)), 1)], 'depth_m': round(Z, 3),
            'dxy_mm': [round(1000 * float(v), 1) for v in d_world[:2]]}
    if debug_path:
        v = img.copy()
        cv2.polylines(v, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for x, y in outline['far'] + outline['left']:
            cv2.circle(v, (int(x), int(y)), 3, (255, 0, 255), -1)
        cv2.imwrite(debug_path, v)
    return d_world[:2], info


def _edge_shift(img, uv):
    """Image shift (px) that moves the outline `uv` onto the real dark-part -> red-base edges, (shift, n, resid)."""
    Rch = img[:, :, 2].astype(np.float32); H, Wd = Rch.shape; cen = uv.mean(0)
    A, b = [], []
    for k in range(4):
        p, q = uv[k], uv[(k + 1) % 4]; e = q - p; L = np.linalg.norm(e)
        if L < 20:
            continue
        n = np.array([-e[1], e[0]]) / L
        if np.dot(n, (p + q) / 2 - cen) < 0:
            n = -n
        for s_ in np.linspace(0.15, 0.85, 30):
            c = p + s_ * e; ts = np.arange(-90, 91)
            pts = np.round(c[None] + ts[:, None] * n[None]).astype(int)
            if not ((pts[:, 0] >= 0) & (pts[:, 0] < Wd) & (pts[:, 1] >= 0) & (pts[:, 1] < H)).all():
                continue
            r = np.convolve(Rch[pts[:, 1], pts[:, 0]], np.ones(3) / 3, mode='same'); hit = None
            for i in range(1, len(ts) - 11):
                if r[i] <= 90 and r[i + 1:i + 4].mean() >= 115 and r[i + 1:i + 11].max() >= RED_R:
                    hit = i
            if hit is not None:
                A.append(n); b.append(ts[hit] + 0.5)
    if len(A) < 8:
        return None, len(A), 0.0
    A, b = np.array(A), np.array(b)
    shift, *_ = np.linalg.lstsq(A, b, rcond=None)
    return shift, len(A), float(np.sqrt(np.mean((A @ shift - b) ** 2)))


def measure(obj, base):
    """(yaw error deg about the base z: turn the part by this to seat it, tilt deg, info) or (None, None, info)."""
    import cv2
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
    from utils.data_path_finder import get_symmetry_dir
    from aruco_camera_localizer.robot_config import RobotConfig
    rc = RobotConfig()                                            # the localizer's calibrated intrinsics
    img, tcp = still_frame()
    if img is None:
        return None, None, {'error': 'no still frame'}
    dict_name, _ = _layout(base)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)),
                                  cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) < 3:                                 # dim: equalise contrast and try again
        c2, i2, _ = det.detectMarkers(cv2.createCLAHE(3.0, (8, 8)).apply(gray))
        if i2 is not None and (ids is None or len(i2) > len(ids)):
            corners, ids = c2, i2
    if ids is None:
        return None, None, {'error': 'no markers'}
    ids = ids.ravel()
    fx, fy, cx, cy = rc.camera_matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.array(rc.distortion_coeffs, np.float64).reshape(-1, 1)
    Rb, nb = object_pose_cam(base, corners, ids, K, D)
    Ro, no = object_pose_cam(obj, corners, ids, K, D)
    info = {'markers': [int(i) for i in ids], 'n_base': nb, 'n_part': no}
    if Rb is None or Ro is None:
        return None, None, dict(info, error='base or part not seen')
    R_rel = Rb.inv() * Ro                                           # part in the base frame (hand-eye cancels)
    _, sq = load_seat_pose_in_base(base, obj)
    seat = R.from_quat(sq)
    try:
        cands = [R.from_matrix(m) for m in equivalent_orientations(seat.as_matrix(),
                                                                    load_symmetry_data(obj, str(get_symmetry_dir())))]
    except Exception:
        cands = [seat]
    best = min(cands, key=lambda c: (c * R_rel.inv()).magnitude())
    d = (best * R_rel.inv()).as_rotvec()                            # base-frame rotation: measured -> seat
    return float(np.degrees(d[2])), float(np.degrees(np.hypot(d[0], d[1]))), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object-name', required=True); ap.add_argument('--base-name', required=True)
    ap.add_argument('--align', action='store_true')
    ap.add_argument('--tol-deg', type=float, default=0.5); ap.add_argument('--max-iter', type=int, default=4)
    ap.add_argument('--max-turn-deg', type=float, default=25.0)
    a = ap.parse_args()
    its, final, result = [], None, 'unmeasured'
    for k in range(a.max_iter + 1):
        err, tilt, info = measure(a.object_name, a.base_name)
        it = dict(info, rel_err_deg=None if err is None else round(err, 2), tilt_deg=None if tilt is None else round(tilt, 1))
        print(f"[marker-yaw] {a.object_name}: " + (f"{err:+.2f} deg about z (tilt {tilt:.1f}) " if err is not None else "")
              + json.dumps(info), flush=True)
        if err is None:
            its.append(it); break
        final = err
        if abs(err) <= a.tol_deg or not a.align:
            result = 'aligned' if abs(err) <= a.tol_deg else 'measured'; its.append(it); break
        if abs(err) > a.max_turn_deg or k == a.max_iter:
            result = 'outside' if abs(err) > a.max_turn_deg else 'not_converged'; its.append(it); break
        from compliant_insertion_studio.wrapper.align_at_hover import turn_wrist
        it['turn_deg'] = round(err, 2); its.append(it)
        if not turn_wrist(err):
            result = 'turn_failed'; break
        time.sleep(0.5)
    print('__RESULT_JSON__'); print(json.dumps({'result': result, 'object_name': a.object_name, 'iterations': its,
                                                'final_err_deg': None if final is None else round(final, 2)}))
    print('__END_RESULT_JSON__', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
