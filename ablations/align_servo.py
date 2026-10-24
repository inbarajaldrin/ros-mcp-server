"""Bringing a held part over its seat before the insert (AR3: the step drive-mate's align makes) - the in-hand base read,
the marker servo, the depth-frame servo, the close base read, the held-part measurement, the servo prior, the projector
toggles they use. Moved out of the replay for AR3 (10-23): seams/drive_align imports this, not the planner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R_

import cell_io
from cell_io import REPO_ROOT, run_subprocess, snapshot_robot_state, make_env, _TIMING  # noqa: F401
if str(REPO_ROOT) not in sys.path:      # run as `python3 ablations/replay_real_assembly.py`: only ablations/ is on the path yet
    sys.path.insert(0, str(REPO_ROOT))  # (gate run 71: ModuleNotFoundError 'seams' at import)
from seams.servo_law import (SERVO_ITERS, SERVO_YAW_TOL_DEG, SERVO_YAW_MIN_TURN_DEG, SERVO_MAX_SPREAD_PX,  # noqa: E402,F401
                             SERVO_MAX_YAW_DEG, SERVO_GAIN_CLEAR, SERVO_YAW_GAIN, SERVO_BIAS_M, SERVO_BIAS_CLEAR_M,
                             SERVO_TOL_M, SERVO_MAX_M)

TRACK_VIEW_Z = 0.56            # flange height over the base for the in-hand read: the startup base view's height, so the


IN_HAND_N_MIN = 2500     # matched points for the base fit with a part in the hand (the tilt / coverage / height checks stay)


def track_base_in_hand(base, observed_base, say, log_fh=None, held_obj=None):
    """Free base (2026-10-21 19:31-19:35): two u_brown rim searches turned base1 from 0.19 to ~8.4 deg (8 N lateral
    search force on an untaped base) and the next attempt aimed at the startup read. With the part in hand, fly the
    camera over the base, read it from its corner markers by solvePnP (marker_yaw.base_pose_world: 2.4 px, repeats
    to 0.1 mm / 0.03 deg; the localizer drops frames with a held part in them) and return the base pose to aim at.
    Rule 1: re-read only after something pushed it - every insert attempt touches the base. Returns the new
    ((x, y, z), quat) or the old one when not read."""
    if observed_base is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_world
    from seat_check import X_CAM
    snap = snapshot_robot_state(); tcp = snap.get('tcp') or {}
    if not tcp.get('quat_xyzw'):
        return observed_base
    (bx, by, _), _ = observed_base
    Rt = R_.from_quat(tcp['quat_xyzw']).as_matrix()
    cam = Rt @ np.asarray(X_CAM)[:3, 3]                            # camera offset from the flange, world frame
    tx, ty = bx - cam[0], by - cam[1]                               # camera straight over the base centre
    r = _move_unless_there((tx, ty, TRACK_VIEW_Z), list(tcp['quat_xyzw']), 'base track view (part in hand)', log_fh)
    if r.get('result') == 'failure':
        say(f"  [base-track] view not reached ({r.get('error', '')}) - aiming at the previous read"); return observed_base
    # The pose comes from the 6-dof CAD fit to the depth cloud (operator 10-21: depth + CAD carry far more than the
    # markers; the localizer/markers only say which object is roughly where = the seed). Markers seed it.
    seed, info = base_pose_world(base)
    if seed is None:
        seed, info = observed_base, dict(info, seed='previous read')
    try:
        from base_icp import read_icp
        # the parts seated before the held one are part of the rigid body being read (10-22 06:48: line_green aims at
        # the U openings, which are where the U parts physically seated; the bare-base read is several mm off them -
        # three attempts rode the U tops at +61 mm). They add tall, distinct geometry to the fit too.
        seated_ = ()
        if held_obj is not None:
            try:
                from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
                seated_ = tuple(get_seated_so_far(base, held_obj))
            except Exception:
                seated_ = ()
        # the held part hides some of the base: a flat, covered fit at the table height with fewer matches is the base
        # (10-22..24: 9 in-hand fits refused only for 2861-3977 < 4000 matches, all flat, coverage >= 0.85, z -62.2..-63.2;
        # run 90c then aimed line_green at the marker read 9 mm off in y and 3 x 143 probes rode the U tops)
        pose, icp = read_icp(base, seed, seated=seated_, still_already=True, n_min=IN_HAND_N_MIN)   # base_pose_world just read here (>= 1.4 s still)
        if seated_:
            icp = dict(icp, seated=list(seated_))
    except Exception as e:
        pose, icp = None, {'error': str(e)}
    info = dict(info, icp=icp)
    if pose is None:
        say(f"  [base-track] {base}: CAD fit failed in hand ({icp}) - aiming at the previous read"); return observed_base
    (ox, oy, oz), oq = observed_base
    dyaw = float(np.degrees((R_.from_quat(pose[1]) * R_.from_quat(oq).inv()).as_rotvec()[2]))
    dxy = 1000 * np.hypot(pose[0][0] - ox, pose[0][1] - oy)
    say(f"  [base-track] {base} from the CAD fit to depth (part in hand): moved {dxy:.1f} mm, turned {dyaw:+.2f} deg "
        f"since the aim {json.dumps(info, default=str)}")
    # keep the measured z of the CAD fit (markers sit on the top face; their z carries the depth of the top plane)
    yaw_only = R_.from_euler('z', float(R_.from_quat(pose[1]).as_euler('xyz')[2]))
    tilt = R_.from_quat(oq) * R_.from_euler('z', float(R_.from_quat(oq).as_euler('xyz')[2])).inv()
    return ((pose[0][0], pose[0][1], oz), tuple(float(v) for v in (tilt * yaw_only).as_quat()))


HOVER_ALIGN_ABOVE_M = 0.10      # TCP height over its predicted seat for the in-image part-vs-socket alignment


HOVER_ALIGN_ITERS = 3           # reads (median)


HOVER_ALIGN_MAX_M = 0.012       # a larger 'offset' is a misread, not a pick error


def hover_align(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Operator 10-21 ("do you see how much offset there is ... it is not even above the socket"): the held part was
    ~5-10 mm off its socket while the chained pose put it there - the in-hand error of the pick. Over the predicted
    seat, the camera sees the part's top face against the red base top: marker_yaw.held_offset measures the part's
    real position against the chain in one still frame (edges vs the projected CAD outline, base from the same frame),
    the arm moves by minus that, and measures again (<= 3 rounds). Returns the base pose to aim the insert at:
    shifted by minus the total correction, so the seat target moves with the measured part."""
    if observed_base is None or held_q is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import held_offset
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset, held_tool_offset_for
    snap = snapshot_robot_state(); tcp = snap.get('tcp') or {}
    if not tcp.get('quat_xyzw'):
        return observed_base
    (bx, by, bz), bq = observed_base
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); sq = pred['predicted_tcp_at_seat']['quat_xyzw']
    # The measured offset is the part's error IN THE HAND (real - chained, relative to the flange): moving the arm does
    # not change it (10-21: four rounds read (1.0,-4.8) (1.2,-6.0) (1.2,-4.5) (1.4,-6.3) mm). Measure it a few times
    # over the seat, take the median, and move the aim by minus it once.
    tgt = [st[0], st[1], st[2] + HOVER_ALIGN_ABOVE_M]
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        *[f"{v:.5f}" for v in tgt], '--quat', *[f"{v:.6f}" for v in sq], '--min-z', '0.15'],
                       label='hover align view', timeout_s=60, log_fh=log_fh)
    if r.get('result') == 'failure':
        say(f"  [hover-align] {obj}: hover not reached ({r.get('error', '')})"); return observed_base
    reads = []
    _emitter(False)
    try:
        for k in range(HOVER_ALIGN_ITERS):
            d, info = held_offset(obj, base, held_q, grasp_id)
            if d is None:
                say(f"  [hover-align] {obj}: read {k} not measured ({info.get('error')})"); continue
            reads.append(np.asarray(d[:2]))
            say(f"  [hover-align] {obj} read {k}: part {info['dxy_mm']} mm off its chained pose "
                f"({info['n_edge']} edge samples, resid {info['resid_px']} px)")
    finally:
        _emitter(True)
    if len(reads) < 2:
        return observed_base
    total = np.median(np.array(reads), axis=0)
    if np.hypot(*total) > HOVER_ALIGN_MAX_M:
        say(f"  [hover-align] {obj}: {1000*np.hypot(*total):.1f} mm is more than a pick can be off - not applied")
        return observed_base
    say(f"  [hover-align] {obj}: aim corrected by ({-1000*total[0]:+.1f}, {-1000*total[1]:+.1f}) mm (median of {len(reads)})")
    return ((bx - total[0], by - total[1], bz), bq)


def _footprints_overlap(base, a, b, margin_m=0.005):
    """Do the seated CAD footprints (base-frame xy boxes) of parts a and b overlap, within margin_m?"""
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    box = []
    for o in (a, b):
        sp_, sq_ = load_seat_pose_in_base(base, o); V_, F_ = load_mesh(o)
        P_ = R_.from_quat(sq_).apply(surface_points(V_, F_, 0.003)) + sp_
        box.append((P_[:, 0].min(), P_[:, 0].max(), P_[:, 1].min(), P_[:, 1].max()))
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = box
    return ax0 < bx1 + margin_m and bx0 < ax1 + margin_m and ay0 < by1 + margin_m and by0 < ay1 + margin_m


def seated_under(base, obj):
    """Parts seated before obj whose CAD footprint overlaps obj's: it threads over them first (a corridor stage - yellow
    over line_green) rather than meeting the base rim (cad_geometry.seated_under, shared with touch_insert)."""
    from compliant_insertion_studio.wrapper.cad_geometry import seated_under as _su
    return _su(base, obj)


DSERVO_SPREAD_YAW_PX = 20.0     # a yaw slopes the far edge and spreads the fit: up to this the yaw is turned out first


DSERVO_ITERS = 6


HELD_AXIAL = os.environ.get('HELD_AXIAL', '1') == '1'


HELD_AXIAL_MAX_M = 0.006         # a larger 'offset' is a misread (the play along line_green's pocket is +-1.4 mm)


HELD_AXIAL_MAD_PX = 3.0


def held_axial_correction(obj, grasp_id, held_q, outline, tcp_pose, say):
    """World xy (m) to move the aim so the held part's CAD grasp point lands on the seat: minus the part's offset along
    image-y (the far edge vs the CAD part held at its grasp, projected from the TCP through the hand-eye). None when the
    far edge does not fit (MAD) or the offset is implausible."""
    try:
        import cv2
        from compliant_insertion_studio.wrapper.marker_yaw import _intrinsics
        from compliant_insertion_studio.wrapper.cad_lookup import load_grasp_offset_in_object
        from seat_check import X_CAM
        from handeye_icp_survey import T as _T
        from grasp_depth import load_mesh, surface_points
        from primitives.shared.config import held_tool_offset_for
        tp, tq = np.asarray(tcp_pose[0], float), np.asarray(tcp_pose[1], float)        # the TCP the outline was taken at
        Rh = R_.from_quat(held_q)
        centre = tp + R_.from_quat(tq).apply([0, 0, float(held_tool_offset_for(obj)[2])])
        P = Rh.apply(surface_points(*load_mesh(obj), 0.0015)) + centre - Rh.apply(load_grasp_offset_in_object(obj, int(grasp_id)))
        K, D = _intrinsics(); Twc = _T(tp, tq) @ X_CAM; Tcw = np.linalg.inv(Twc)
        Pc = (Tcw[:3, :3] @ P.T).T + Tcw[:3, 3]
        uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
        dy, zs = [], []
        for x, y in outline['far']:
            m = np.abs(uv[:, 0] - x) < 2.5
            if m.sum() >= 3:
                i = np.argmin(np.where(m, uv[:, 1], np.inf)); dy.append(y - uv[i, 1]); zs.append(Pc[i, 2])
        if len(dy) < 8:
            say(f"  [held-axial] {obj}: far edge out of the projection ({len(dy)} columns)"); return None
        dy = np.asarray(dy); med = float(np.median(dy)); mad = float(1.4826 * np.median(np.abs(dy - med)))
        Z = float(np.median(zs))
        # the left edge (image-x: across a bar) the same way: runs 37/38b/39 read -2.1 / -1.9 / -2.2 mm (MAD 0.1-0.2 px)
        # where the probe lattice then found the pocket 2.0-2.5 mm across (50 probes in run 39)
        dx = []
        for x, y in outline.get('left') or []:
            m = np.abs(uv[:, 1] - y) < 2.5
            if m.sum() >= 3:
                i = np.argmin(np.where(m, uv[:, 0], np.inf)); dx.append(x - uv[i, 0])
        dx = np.asarray(dx); medx = float(np.median(dx)) if len(dx) >= 5 else 0.0
        madx = float(1.4826 * np.median(np.abs(dx - medx))) if len(dx) >= 5 else float('inf')
        if madx > HELD_AXIAL_MAD_PX:
            medx = 0.0                                           # the left edge does not fit: along the far edge only
        d = Twc[:3, :3] @ np.array([medx * Z / K[0, 0], med * Z / K[1, 1], 0.0])     # part - CAD grasp, world
        say(f"  [held-axial] {obj}: far edge {med:+.1f} px (MAD {mad:.1f}, {len(dy)} columns), left edge {medx:+.1f} px "
            f"(MAD {madx:.1f}, {len(dx)} rows) from the CAD grasp = ({1000*d[0]:+.1f}, {1000*d[1]:+.1f}) mm")
        if mad > HELD_AXIAL_MAD_PX or np.hypot(d[0], d[1]) > HELD_AXIAL_MAX_M:
            say(f"  [held-axial] {obj}: not used (MAD > {HELD_AXIAL_MAD_PX} px or > {1000*HELD_AXIAL_MAX_M:.0f} mm)"); return None
        return np.array([-d[0], -d[1]])
    except Exception as e:
        say(f"  [held-axial] {obj}: not measured ({e})"); return None


DSERVO_ROBUST = os.environ.get('DSERVO_ROBUST', '1') == '1'   # far edge by its median when only its std is spread (38b: converged, seated)


def depth_servo(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Observe -> move -> observe with the target in the SAME frame as the image (10-22 11:16, the yellow seated by it):
    (1) wrist to the base yaw, (2) the held part's outline over bare table, (3) base1 fitted to depth with the camera
    centred over it (track_base_in_hand, seated parts in the fit), (4) the TCP translated straight over the predicted
    seat at that height - the hand-eye error cancels but for its rotation x the ~5 cm move (~0.4 mm) - then rounds:
    socket_shift_world; a yaw over 0.4 deg is turned out first (it alone spread the yellow's fit to 12 px: 2 px after),
    then xy steps by -d until < 1 mm. Returns (base pose shifted so its seat lies under the aim, held quat) or None when
    it cannot measure (outline, base read or fit) - the caller keeps its own aim."""
    if observed_base is None or held_q is None:
        return None
    from compliant_insertion_studio.wrapper.marker_yaw import socket_shift_world, held_outline_px, still_frame, _tcp_pose_now
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    err, _t = rel_yaw_error(obj, base, R_.from_quat(bq), R_.from_quat(held_q))
    if abs(err) > 0.3 and turn_wrist(err):
        held_q = list((R_.from_euler('z', err, degrees=True) * R_.from_quat(held_q)).as_quat())
        say(f"  [dservo] {obj}: wrist turned {err:+.2f} deg to the base yaw")
    q = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')
    r = _move_unless_there((bx + 0.17, by, 0.32), list(q), 'dservo: outline view over table', log_fh)   # in process
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=4)
    finally:
        _emitter(True)
    outline = held_outline_px(img_) if (img_ is not None and r.get('result') != 'failure') else None
    if outline is None:
        say(f"  [dservo] {obj}: held part not outlined over the table"); return None
    stem = REPO_ROOT / 'compliant_insertion_studio' / 'logs' / f"dservo_{obj}_{time.strftime('%Y%m%d_%H%M%S')}"
    tp_outline, hq_outline = _tcp_pose_now(), list(held_q)           # the outline's frame (the base read moves the arm)
    try:
        # the held part over bare table, every grasp: where it sits in the hand (camera rigid with the gripper) - the
        # data for measuring a bar's axial grasp error without the base (line_green: 2-4 mm toward -y, 16/19 inserts)
        import cv2 as _cv2
        json.dump({'outline': outline, 'tcp': [list(map(float, v)) for v in tp_outline], 'held_q': list(held_q)},
                  open(f"{stem}_outline.json", 'w')); _cv2.imwrite(f"{stem}_table.png", img_)
    except Exception:
        pass
    ob = track_base_in_hand(base, observed_base, say, log_fh, held_obj=obj)   # depth fit, camera over the base
    if ob is not None and ob is not observed_base:
        LAST_TRACK_READ['pose'] = ob
    _emitter_hold()                                                   # the rounds are colour-only
    if ob is None or ob is observed_base:
        say(f"  [dservo] {obj}: base not re-read in hand")
        # no base in this frame - the hand still is: the held part's far edge over bare table against the CAD grasp
        # (camera rigid with the gripper) gives where it sits along image-y (a bar's axis). 10-23: line_green's far end
        # read 18.3 px (MAD 1.1) past a centred grasp = 2.8 mm toward base -y, the overhang that wedged 16 of 19 inserts
        c = held_axial_correction(obj, grasp_id, held_q, outline, tp_outline, say) if HELD_AXIAL else None
        if c is None:
            return None
        (bx_, by_, bz_), bq_ = observed_base
        say(f"  [dservo] {obj}: aim shifted ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm by the held part's place in the hand")
        return ((bx_ + c[0], by_ + c[1], bz_), bq_), held_q
    (bx, by, bz), bq = ob
    off = float(held_tool_offset_for(obj)[2])

    def seat_tcp(q_ee, hq):
        pr = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                                 base_world_quat_xyzw=list(bq), held_quat_xyzw=list(hq), ee_orientation_xyzw=list(q_ee),
                                 flange_offset_m=off)
        return np.array(pr['predicted_tcp_at_seat']['xyz_m'])
    z = TRACK_VIEW_Z
    xy = seat_tcp(_tcp_pose_now()[1], held_q)[:2].copy()
    yaw_total, rounds, ok, misread = 0.0, [], False, False
    for k in range(DSERVO_ITERS):
        q_now = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')      # keep the orientation, in process
        rr_ = (_move_unless_there((xy[0], xy[1], z), q_now, f'dservo round {k}: over the aim', log_fh) if q_now else
               run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{xy[0]:.5f}",
                               f"{xy[1]:.5f}", f"{z:.5f}", '--keep-orientation', '--min-z', '0.15'],
                              label=f'dservo round {k}: over the aim', timeout_s=60, log_fh=log_fh))
        if rr_.get('result') == 'failure' or cell_io._SAFETY_STOP:
            break
        _emitter(False)
        try:
            img_, _ = still_frame(n_avg=4); tp = _tcp_pose_now(); tp = (np.asarray(tp[0]), np.asarray(tp[1]))
        finally:
            _emitter(True)
        st = seat_tcp(tp[1], held_q)
        d, info = socket_shift_world(obj, base, ob, float(tp[0][2] - st[2]), outline, img=img_, tcp_pose=tp,
                                     debug_path=f"{stem}_r{k}.png")
        rounds.append({'k': k, 'tcp': [tp[0].tolist(), tp[1].tolist()], 'seat_tcp': st.tolist(), 'info': info})
        say(f"  [dservo] {obj} round {k}: {info.get('dxy_mm')} mm, yaw {info.get('yaw_img_deg')} deg, spread "
            f"{info.get('spread_px')} px" + (f" ({info.get('error')})" if d is None else ""))
        if d is not None and np.hypot(*d) > SERVO_MAX_M:
            # more than the whole correction may be (the end check) - a misread too; run 52 stepped 73 mm on it, twice
            misread = True
            say(f"  [dservo] {obj}: reading {1000*np.hypot(*d):.1f} mm is past the servo's {1000*SERVO_MAX_M:.0f} mm "
                f"- a misread fit, its readings not used")
            break
        if d is not None and abs(info.get('yaw_img_deg') or 0.0) > SERVO_MAX_YAW_DEG:
            # a yaw past what the whole servo may turn, read right after the wrist was turned to the base yaw, is the
            # fit matching something else: line_green read 35.9 / 46.1 deg (runs 51 / 49), its far-edge median then aimed
            # 2.5 mm off the held part's measured place along x (clean fits agree with it to 0.1 mm, run 47) and run 51
            # wedged an end +6.7 mm. None of its readings are used.
            misread = True
            say(f"  [dservo] {obj}: image yaw {info.get('yaw_img_deg')} deg is past the servo's {SERVO_MAX_YAW_DEG:.0f} deg "
                f"- a misread fit, its readings not used")
            break
        far_bad = d is not None and info['spread_px'][1] > DSERVO_SPREAD_YAW_PX
        if (far_bad and DSERVO_ROBUST and (info.get('spread_mad_px') or [99, 99])[1] <= SERVO_MAX_SPREAD_PX):
            # the far edge's columns disagree in std but agree in MAD: its median holds, its slope (the yaw) does not -
            # step along the far edge's component only (10-23 line_green: std 101.7 px, median -14.4 px = 2.25 mm along
            # the bar toward -y, the overhang that wedged 16 of 19 inserts)
            d = np.asarray(info['d_far_only_mm'], float) / 1000.0
            say(f"  [dservo] {obj}: far edge std {info['spread_px'][1]} px but MAD {info['spread_mad_px'][1]} px - its "
                f"median only: {info['d_far_only_mm']} mm")
            if np.hypot(*d) <= SERVO_TOL_M:
                xy = xy - d; ok = True; break
            xy = xy - d
            continue
        if d is None or far_bad:
            break                                                    # the far edge decides; a spread left edge is handled below
        yi = info.get('yaw_img_deg') or 0.0
        if abs(yi) > SERVO_YAW_MIN_TURN_DEG:
            t = float(np.clip(yi / SERVO_YAW_GAIN, -5.0, 5.0))
            if abs(yaw_total + t) > SERVO_MAX_YAW_DEG:
                say(f"  [dservo] {obj}: yaw {yaw_total + t:+.1f} deg in all is more than the reads can be off - stopping"); break
            if turn_wrist(t):
                yaw_total += t
                held_q = list((R_.from_euler('z', t, degrees=True) * R_.from_quat(held_q)).as_quat())
                say(f"  [dservo] {obj}: wrist turned {t:+.2f} deg (image yaw)")
                continue
        if info['spread_px'][1] > SERVO_MAX_SPREAD_PX:
            say(f"  [dservo] {obj}: far edge spreads {info['spread_px']} px after the yaw - no xy step"); break
        if info['spread_px'][0] > SERVO_MAX_SPREAD_PX:
            # the far edge fits, the left one does not (13:13 yellow: left 45.5 px, far 1.2 px, far reading 7.8 then
            # 8.3 mm): step along the far edge's component only, as the marker servo does
            d = np.asarray(info['d_far_only_mm'], float) / 1000.0
            say(f"  [dservo] {obj}: left edge spreads {info['spread_px'][0]} px - far-edge component only "
                f"{info['d_far_only_mm']} mm")
        if np.hypot(*d) <= SERVO_TOL_M:
            xy = xy - np.asarray(d); ok = True; break
        xy = xy - np.asarray(d)
    try:
        json.dump({'rounds': rounds, 'held_q': list(held_q), 'base': [list(ob[0]), list(ob[1])], 'aim': list(xy), 'ok': ok},
                  open(f"{stem}.json", 'w'), indent=1)
    except Exception:
        pass
    if not ok and misread and HELD_AXIAL:
        # the held part's place in the hand, from its outline over bare table (taken above), against this frame's base read
        c = held_axial_correction(obj, grasp_id, hq_outline, outline, tp_outline, say)
        if c is not None and abs(yaw_total) > 0.05:
            # the rounds turned the wrist on image-yaw readings of a fit that then proved unreliable; the aim from the held
            # part assumes it square to the base as the base-yaw turn left it - turn those back (run 67: -0.79 and -3.26 deg
            # kept after a misread round, line_green went in 4 deg off its slot, rode the U tops at +38 mm, wedged +10 mm)
            if turn_wrist(-yaw_total):
                say(f"  [dservo] {obj}: the rounds' wrist turns ({yaw_total:+.2f} deg) turned back - the held-part aim is square")
                held_q = list((R_.from_euler('z', -yaw_total, degrees=True) * R_.from_quat(held_q)).as_quat())
                yaw_total = 0.0
            else:
                say(f"  [dservo] {obj}: could not turn the rounds' {yaw_total:+.2f} deg back - no aim from this servo"); return None
        if c is not None:
            say(f"  [dservo] {obj}: aim shifted ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm by the held part's place in the hand")
            return ((bx + c[0], by + c[1], bz), bq), held_q
    if not ok:
        say(f"  [dservo] {obj}: not converged - caller keeps its aim"); return None
    st = seat_tcp((snapshot_robot_state().get('tcp') or {}).get('quat_xyzw'), held_q)
    c = np.asarray(xy) - st[:2]
    if np.hypot(*c) > SERVO_MAX_M:
        say(f"  [dservo] {obj}: correction {1000*np.hypot(*c):.1f} mm is more than the reads can be off - not applied"); return None
    say(f"  [dservo] {obj}: converged, aim {1000*c[0]:+.1f} / {1000*c[1]:+.1f} mm from the predicted seat")
    return ((bx + c[0], by + c[1], bz), bq), held_q


def close_base_read(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Base pose from its markers in one frame over obj's seat at the clear height (tilt from the chain, yaw searched,
    translation solved: marker_yaw.base_pose_same_frame). Returns the new ((x, y, z), quat) (z kept) or the old one."""
    if observed_base is None or held_q is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_same_frame, still_frame
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    tcp = (snapshot_robot_state().get('tcp') or {})
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); q = tcp['quat_xyzw']
    z_view = float(st[2]) + HOVER_ALIGN_ABOVE_M
    cur = tcp.get('xyz')
    if cur and cur[2] < z_view - 0.002:
        run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{cur[0]:.5f}",
                        f"{cur[1]:.5f}", f"{z_view:.5f}", '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15'],
                       label='close base read: up', timeout_s=60, log_fh=log_fh)
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{st[0]:.5f}",
                        f"{st[1]:.5f}", f"{z_view:.5f}", '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15'],
                       label='close base read view', timeout_s=60, log_fh=log_fh)
    if r.get('result') == 'failure' or cell_io._SAFETY_STOP:
        return observed_base
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=6); tc = (snapshot_robot_state().get('tcp') or {})
        pose, info = base_pose_same_frame(base, img_, (np.asarray(tc['xyz']), np.asarray(tc['quat_xyzw'])), bq) \
            if img_ is not None else (None, {'error': 'no frame'})
    finally:
        _emitter(True)
    if pose is None or info.get('reproj_px', 99) > 3.0:
        say(f"  [close-base] {base}: not read over {obj}'s seat ({info}) - aiming at the base read")
        return observed_base
    d_mm = 1000 * np.hypot(pose[0][0] - bx, pose[0][1] - by)
    if d_mm > 15.0:
        say(f"  [close-base] {base}: close read {d_mm:.1f} mm from the base read - more than the read can be off, not used")
        return observed_base
    say(f"  [close-base] {base} over {obj}'s seat: {d_mm:.1f} mm from the 0.56 m read ({json.dumps(info)})")
    return ((pose[0][0], pose[0][1], bz), tuple(pose[1]))


SERVO_MEASURED: dict = {}         # obj -> did the last hover_servo take a reading


LAST_TRACK_READ = {}            # the hover servo's in-hand base read (with the turned wrist)


def align_for_insert(obj, base, measured_base, held_quat, grasp_id, say, log_fh=None):
    """Free base, one-node: bring the held part over its seat before the insert (AR3: the call drive-mate's align step
    makes). From the last MEASURED base: the in-hand base re-read, then by route - a prismatic part turns to the base
    yaw and re-reads the base close (it searches itself); a part threading over seated parts (CAD) runs the depth-frame
    servo (close read as fallback); any other the marker servo (depth-frame servo when it took no reading).
    Returns {aim_base, measured_base, held_quat, tcp_quat (the flange after a wrist turn, else None), route}."""
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    out = dict(aim_base=None, measured_base=None, held_quat=held_quat, tcp_quat=None, route='')
    try:
        from primitives.translate_object import PRISMATIC_ROUTE_OBJECTS as _PRISM
    except Exception:
        _PRISM = ()
    mode = os.environ.get('HOVER_ALIGN', 'servo')
    servo_route = mode == 'servo' and obj not in _PRISM
    if servo_route:
        # the marker servo and the depth servo each re-read the base in hand themselves (the read they project with): a
        # read here first only fed the wrist turn before it (run 34: ~15 s per U part). The depth servo's took the
        # yellow base view -> outline over the table (back by the pick) -> base view again: 11.5 s and a backtrack
        # (run 50b; the two reads agreed to 0.4 mm / 0.04 deg, the startup read's yaw to 0.18 deg)
        ob = measured_base
        LAST_TRACK_READ.pop('pose', None)
    else:
        ob = track_base_in_hand(base, measured_base, say, log_fh, held_obj=obj)
    out['measured_base'] = ob

    def _tcp_q():
        return (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')
    if mode == 'servo' and obj in _PRISM:
        # the prismatic route searches for itself; the servo aim (a +-2.5 mm dip grid needs it) only misleads it:
        # 10-22 05:17 line_green's outline fit broke (140 px), its 'best reading' moved the aim 6 mm and the bar stopped
        # 60 mm high, where it had seated from the read aim before (slot 6-8 mm away, found by the route itself)
        out['route'] = 'prismatic: yaw + close read'
        say(f"  [servo] {obj}: prismatic route - it searches itself, no xy servo")
        # but the wrist still turns to the base's yaw first: skipping the whole servo (9a19b76) skipped this turn and
        # line_green arrived +13.2 deg off its slot (base1 at ~16 deg; the U openings it threads need < 1.3 deg) -
        # jammed on the U tops at +61 mm, 10-22 06:27 (INSERT_GEOMETRY: the corridor stage is a yaw wedge)
        try:
            if ob is not None and held_quat is not None:
                err_, _t = rel_yaw_error(obj, base, R_.from_quat(ob[1]), R_.from_quat(held_quat))
                if abs(err_) > 0.3 and turn_wrist(err_):
                    held_quat = list((R_.from_euler('z', err_, degrees=True) * R_.from_quat(held_quat)).as_quat())
                    out['tcp_quat'] = _tcp_q()
                    say(f"  [servo] {obj}: wrist turned {err_:+.2f} deg to the base yaw")
        except Exception as e_:
            say(f"  [servo] {obj}: yaw to base not applied ({e_})")
        # and the base is re-read from its markers in a CLOSE frame over the seat (no xy servo): the 0.56 m CAD
        # fit carries the hand-eye rotation error x ~0.5 m of lever (several mm); line_green needs +-1.1 mm across
        # the U openings. Bare and seated-assembly fits agreed to 0.4 mm (10-22 07:00), so the U parts are not the
        # offset - the read height is. From the clear height (~0.17 m over the base top) the lever is ~1/3.
        try:
            ob = close_base_read(obj, base, ob, held_quat, grasp_id, say, log_fh)
            out['measured_base'] = ob                                # a measurement too (markers, one close frame)
            # the wrist turned to the base yaw BEFORE this read: turn again to the closer read's yaw if they differ
            if ob is not None and held_quat is not None:
                err2_, _t2 = rel_yaw_error(obj, base, R_.from_quat(ob[1]), R_.from_quat(held_quat))
                if abs(err2_) > 0.3 and turn_wrist(err2_):
                    held_quat = list((R_.from_euler('z', err2_, degrees=True) * R_.from_quat(held_quat)).as_quat())
                    out['tcp_quat'] = _tcp_q()
                    say(f"  [servo] {obj}: wrist turned {err2_:+.2f} deg to the close read's base yaw")
        except Exception as e_:
            say(f"  [servo] {obj}: close base read not applied ({e_})")
    elif mode == 'servo' and seated_under(base, obj):
        # threads over seated parts first (CAD): the depth-frame servo (10-22 11:16 seated the yellow), the close
        # marker read as the fallback when it cannot measure
        out['route'] = 'depth servo'
        try:
            ds = depth_servo(obj, base, ob, held_quat, grasp_id, say, log_fh)
        finally:
            _emitter_release()
        if LAST_TRACK_READ.get('pose') is not None:
            out['measured_base'] = LAST_TRACK_READ['pose']           # the depth servo's in-hand read
        if ds is not None:
            ob, held_quat = ds
        else:
            try:
                ob = close_base_read(obj, base, ob, held_quat, grasp_id, say, log_fh)
            except Exception as e_:
                say(f"  [dservo] {obj}: close base read not applied ({e_})")
        out['tcp_quat'] = _tcp_q()
    elif mode == 'servo':
        # the marker servo first (u_brown seated first try with it in attempts 16-18; the depth servo's clean,
        # converged reading put u_brown 6 mm off at 12:39 - not yet understood), the depth-frame servo when the
        # marker servo took no reading (12:05/12:22 u_orange: 0-1 base markers over its far-side seat; the depth
        # servo converged and it entered on the lattice)
        out['route'] = 'marker servo'
        ob0_, hq0_ = ob, held_quat
        try:
            ob, hq_ = hover_servo(obj, base, ob0_, hq0_, grasp_id, say, log_fh)
        finally:
            _emitter_release()
        if LAST_TRACK_READ.get('pose') is not None:
            out['measured_base'] = LAST_TRACK_READ['pose']           # the servo's in-hand read: drive-mate's measured base
        if not SERVO_MEASURED.get(obj) and not cell_io._SAFETY_STOP:
            out['route'] = 'marker servo -> depth servo'
            say(f"  [servo] {obj}: no marker reading - the depth-frame servo")
            try:
                ds = depth_servo(obj, base, ob0_, hq_ if hq_ is not None else hq0_, grasp_id, say, log_fh)
            finally:
                _emitter_release()
            if ds is not None:
                ob, hq_ = ds
        if hq_ is not None and held_quat is not None and hq_ is not held_quat:
            held_quat = hq_                                          # the wrist turned: the chained held pose turns with it
            out['tcp_quat'] = _tcp_q()
    elif mode == '1':
        out['route'] = 'hover align'
        ob = hover_align(obj, base, ob, held_quat, grasp_id, say, log_fh)
    out['aim_base'], out['held_quat'] = ob, held_quat
    return out


GRASP_DEPTH_MEAS_MIN_M = 0.003   # a servo touchdown this far over the CAD rim is the grasp depth, not noise ...


GRASP_DEPTH_MEAS_MAX_M = 0.012   # ... and up to this (more is a part on something else - the yellow met line_green's top +18 mm)


GRASP_DEPTH_FIX = {}             # obj -> depth correction (m) measured this grasp; drive_align reports it to drive-mate


SERVO_INPROC = os.environ.get('SERVO_INPROC', '1') == '1'   # servo moves in process (benched empty 10-23: 2.1 s a move vs 4.6)


SERVO_SHORT_M, SERVO_SHORT_SPEED_M_S, SERVO_SHORT_MIN_S = 0.020, 0.015, 1.0   # in process: <= 20 mm timed by distance


def _move_unless_there(xyz, q, label, log_fh=None, extra=()):
    """A servo move, skipped when the arm is already there (within 0.3 mm, same orientation): the 10-22 motion trace
    counted 15 servo moves to exactly the previous target (hover from the clear height = clear, up for a turn, done) at
    ~4 s each."""
    tcp = snapshot_robot_state().get('tcp') or {}
    if (tcp.get('xyz') and tcp.get('quat_xyzw') and np.linalg.norm(np.asarray(tcp['xyz']) - np.asarray(xyz, float)) < 0.0003
            and abs(float(np.dot(tcp['quat_xyzw'], q))) > 0.99999):
        return {'result': 'success', 'skipped': 'already there'}
    kw = {}
    ex = list(extra)
    while ex[:1] and ex[0] in ('--max-force', '--min-duration') and len(ex) >= 2:   # the guarded hover: same guard in process
        kw['max_force' if ex[0] == '--max-force' else 'min_duration'] = float(ex[1]); ex = ex[2:]
    if SERVO_INPROC and not ex:
        # the same plan_move the CLI runs, on one node, released before the next camera read (which starts its own
        # context): run 33's servo moves were ~4 s each, ~1.5 s of it a process start (u_brown: 8 moves / 99 s align)
        from compliant_insertion_studio.wrapper.touch_insert import move as _tmove, close_node as _close
        t0 = time.time()
        d_ = (float(np.linalg.norm(np.asarray(tcp['xyz']) - np.asarray(xyz, float)))
              if tcp.get('xyz') and abs(float(np.dot(tcp.get('quat_xyzw') or [0, 0, 0, 0], q))) > 0.9999 else 1.0)
        try:
            # short moves (the servo's 3-15 mm steps) timed by distance, as the touch probes' are: the planner's floor
            # held them at ~2.5 s (bench, nothing gripped: 3.5 s a move in process vs 4.6 s by subprocess)
            short = d_ <= SERVO_SHORT_M and 'min_duration' not in kw
            r = _tmove(list(map(float, xyz)), list(map(float, q)), min_z=0.15,
                       duration=max(SERVO_SHORT_MIN_S, d_ / SERVO_SHORT_SPEED_M_S) if short else None, **kw)
        finally:
            _close()
        fh = _TIMING.get('fh')
        if fh is not None:
            fh.write(json.dumps({'kind': 'move_inproc', 'label': label, 't0': t0, 't1': time.time(),
                                 'result': r.get('result')}) + '\n')
        return r
    return run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                           *[f"{float(v):.5f}" for v in xyz], '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15',
                           *extra], label=label, timeout_s=60, log_fh=log_fh)


SERVO_PRIOR = os.environ.get('SERVO_PRIOR', '1') == '1'


SERVO_PRIOR_LOG = REPO_ROOT / 'ablations' / 'logs' / 'servo_priors.jsonl'


SERVO_PRIOR_MAD_M, SERVO_PRIOR_MAD_DEG = 0.001, 1.0


def servo_prior(obj):
    """The marker servo's measured correction for obj, in the BASE frame (so it holds at any base yaw): the median of its
    last 6 converged servos (>= 3, agreeing within 1 mm / 1 deg MAD), or None. u_brown's servo moved its aim (-3.3..-4.5,
    -6.7..-8.2) mm and turned -1..-4.5 deg in every run of 10-22/23 - round 0 always found it ~9 mm / ~2 deg off, and
    each round is a lift, a turn, a step and a lowering (~13 s): the zigzag the operator saw. The servo still measures;
    the prior only starts it nearer, so a pose where the prior is wrong costs the round it costs today."""
    try:
        rows = [json.loads(l) for l in open(SERVO_PRIOR_LOG) if l.strip()]
    except Exception:
        return None
    rows = [r for r in rows if r.get('obj') == obj][-6:]
    if len(rows) < 3:
        return None
    d = np.array([r['dxy_base_m'] for r in rows], float); y = np.array([r['yaw_deg'] for r in rows], float)
    md, my = np.median(d, 0), float(np.median(y))
    if (np.median(np.abs(d - md), 0) > SERVO_PRIOR_MAD_M).any() or np.median(np.abs(y - my)) > SERVO_PRIOR_MAD_DEG:
        return None
    return md, my, len(rows)


def _record_servo(obj, c_world, yaw_deg, base_q, rounds):
    try:
        yb = float(R_.from_quat(base_q).as_euler('xyz')[2])
        cb = R_.from_euler('z', -yb).apply([c_world[0], c_world[1], 0.0])[:2]
        with open(SERVO_PRIOR_LOG, 'a') as f:
            f.write(json.dumps({'obj': obj, 't': time.time(), 'dxy_base_m': [float(cb[0]), float(cb[1])],
                                'yaw_deg': float(yaw_deg or 0.0), 'rounds': int(rounds)}) + '\n')
    except Exception:
        pass


def hover_servo(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Operator 10-21 ("the yaw fix seemed to work but not the offset"): align the held part to its SOCKET in the image,
    the way the operator does by eye. Yaw the wrist to the measured base (fold-aware, base x CAD seat vs held part),
    hover 0.10 m over the predicted seat, then marker_yaw.socket_offset: the part's real top-face edges vs its CAD top
    face at the seat raised to the current height, placed by the base's own markers in the same frame (no hand-eye,
    no held-pose chain); move by minus it, repeat (<= 4) until < 1 mm. Returns (base pose to aim at = shifted by the
    total move, held quat after the yaw turn). SERVO_MEASURED[obj] says whether a reading was taken (else only the bias)."""
    SERVO_MEASURED[obj] = False
    if observed_base is None or held_q is None:
        return observed_base, held_q
    from compliant_insertion_studio.wrapper.marker_yaw import socket_shift_world, held_outline_px, still_frame
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    err, _t = rel_yaw_error(obj, base, R_.from_quat(bq), R_.from_quat(held_q))
    prior = servo_prior(obj) if SERVO_PRIOR else None
    prior_yaw = prior[1] if prior is not None else 0.0
    if abs(err + prior_yaw) > 0.3 and turn_wrist(err + prior_yaw):
        held_q = list((R_.from_euler('z', err + prior_yaw, degrees=True) * R_.from_quat(held_q)).as_quat())
        say(f"  [servo] {obj}: wrist turned {err + prior_yaw:+.2f} deg to the base yaw"
            + (f" ({prior_yaw:+.2f} of it the servo prior)" if prior is not None else ""))
    elif prior is not None:
        prior_yaw = 0.0                                     # no turn taken: none of the prior's yaw applied
    # the base read again WITH the turned wrist: the servo projects it from the hover with this same orientation, so the
    # hand-eye's translation error cancels between the read and the projection
    observed_base = track_base_in_hand(base, observed_base, say, log_fh, held_obj=obj)
    LAST_TRACK_READ['pose'] = observed_base
    (bx, by, bz), bq = observed_base
    tcp = (snapshot_robot_state().get('tcp') or {})
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); q = tcp['quat_xyzw']
    # the held part's place in the image, once per grasp, over bare table beside the base (camera rigid with the gripper)
    _emitter_hold()                                     # colour-only from the outline on (the base was read before it)
    r = _move_unless_there((bx + 0.17, by, 0.32), list(q), 'servo: outline view over table', log_fh)   # in process
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=4)
    finally:
        _emitter(True)
    outline = held_outline_px(img_) if (img_ is not None and r.get('result') != 'failure') else None
    if outline is None:
        say(f"  [servo] {obj}: held part not outlined over the table - aiming as read"); return observed_base, held_q
    say(f"  [servo] {obj}: held part outlined ({len(outline['far'])} far / {len(outline['left'])} left edge points)")
    _emitter_hold()                                                   # colour-only from here: no depth read follows
    # evidence for every servo: the outline, the table frame and each round's overlay (target green, outline magenta)
    sv_stem = REPO_ROOT / 'compliant_insertion_studio' / 'logs' / f"servo_{obj}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        import cv2 as _cv2
        json.dump(outline, open(f"{sv_stem}_outline.json", 'w')); _cv2.imwrite(f"{sv_stem}_table.png", img_)
    except Exception:
        pass
    # measure LOW, 2.5 mm over the rim: the target (part at its seat raised by the lift) is projected from the base
    # pose read in the SAME frame, so what remains is the hand-eye's tilt error x the lift (at 0.10 m: 1 deg -> 1.7 mm;
    # at 0.035 m: 0.6 mm). Lateral moves and wrist turns happen 12 mm over the rim.
    try:
        # the FIRST surface the held part can meet: the base top or the top of any part seated before it (10-22 01:59:
        # inverted_u_yellow nests over the seated parts; measuring 2.5 mm over the base top hit them, 4 N guard)
        from compliant_insertion_studio.wrapper.cad_geometry import build_contact_candidates, get_seated_so_far
        # only parts seated UNDER the held part's footprint can meet it (10-22 02:36: u_orange's footprint does not
        # overlap u_brown's, yet u_brown's top set the measuring height 70 mm up and the base markers left the view)
        under = [o for o in get_seated_so_far(base, obj) if _footprints_overlap(base, obj, o)]
        cands = build_contact_candidates(target_object=obj, target_grasp_id=int(grasp_id), base_name=base,
                                         base_world_z=float(bz), R_grasp_quat_xyzw=held_q,
                                         predicted_tcp_z_at_seat=float(st[2]),
                                         seated_objects=under,
                                         flange_offset_world_z=-float(held_tool_offset_for(obj)[2]))
        rim_z = max(v for k, v in cands.items() if k != 'SEAT')
        say(f"  [servo] {obj}: first contact {1000*(rim_z - st[2]):+.1f} mm over the seat "
            f"({', '.join(f'{k} {1000*(v - st[2]):+.1f}' for k, v in cands.items() if k != 'SEAT')})")
    except Exception as e:
        rim_z = float(st[2]) + 0.031
        say(f"  [servo] {obj}: no CAD contact candidates ({e}) - first contact assumed 31 mm over the seat")
    z_meas = rim_z + SERVO_MEAS_OVER_RIM_M
    # lateral moves and wrist turns happen at the old hover height (100 mm over the seat): parts already seated stand up
    # to ~40 mm over the base top, and a held bar reaches far beyond the fingers (10-22 01:18: a diagonal move from the
    # table view down to 12 mm over the rim drove line_green into the seated U parts - protective stop). Only the last
    # leg to the measuring height is vertical, at 4 N guard.
    z_clear = max(float(st[2]) + HOVER_ALIGN_ABOVE_M, rim_z + SERVO_CLEAR_OVER_RIM_M)
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_same_frame
    from seams import servo_law as SL
    xy0 = None
    if prior is not None:
        pw = R_.from_euler('z', float(R_.from_quat(bq).as_euler('xyz')[2])).apply([prior[0][0], prior[0][1], 0.0])[:2]
        xy0 = st[:2] + pw
        say(f"  [servo] {obj}: starting at the servo prior ({1000*pw[0]:+.1f}, {1000*pw[1]:+.1f}) mm, {prior_yaw:+.2f} deg "
            f"(median of its last {prior[2]} servos)")
    law = SL.HoverServoLaw(st[:2], xy=xy0)      # the decisions (AR3 step 1); this function moves and reads
    if prior_yaw:
        law.turned(prior_yaw)                       # the prior's turn is part of the image yaw the insert's base takes
    xy = law.xy
    for k in range(SL.SERVO_ITERS + 1):
        cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
        if cur and cur[2] < z_clear - 0.002:                       # straight up first, never sideways at low height
            _move_unless_there((cur[0], cur[1], z_clear), q, f'servo up {k}', log_fh)
            if cell_io._SAFETY_STOP:
                return observed_base, held_q
        r = _move_unless_there((xy[0], xy[1], z_clear), q, f'servo clear {k}', log_fh)
        if r.get('result') != 'failure':
            r = _move_unless_there((xy[0], xy[1], z_meas), q, f'servo hover {k}', log_fh, extra=('--max-force', '4', '--min-duration', '2'))
        if r.get('result') == 'failure' and r.get('contact') and not cell_io._SAFETY_STOP:
            # touched down before the measuring height (10-22 12:05, u_orange twice at ~4.5 N: the grasp left it hanging
            # lower than the chain) - the image comparison projects the target at whatever height we are, so measure
            # from 2 mm over the contact instead of giving up (attempt 17 measured 4.5 mm and corrected it; 18 did not)
            cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
            if cur:
                z_meas = float(cur[2]) + 0.002
                r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                    f"{xy[0]:.5f}", f"{xy[1]:.5f}", f"{z_meas:.5f}", '--quat', *[f"{v:.6f}" for v in q],
                                    '--min-z', '0.15'], label=f'servo off the contact {k}', timeout_s=60, log_fh=log_fh)
                law.measuring_low()
                say(f"  [servo] {obj}: touched down {1000*(z_meas - 0.002 - rim_z):+.1f} mm over the CAD rim - measuring "
                    f"from 2 mm above the contact")
                # the touchdown MEASURES the grasp depth: the part hangs that much lower in the jaws than the chain says.
                # Carry it everywhere the depth matters - the seat and rim heights here (the image projects the target
                # at the part's real lift), and the insert (drive_align reports it; drive-mate inserts with it).
                # Attempt 26 (10-22 18:01): u_orange touched down +7.4 mm; the servo projected at the chain's lift and
                # oscillated (+10.6 -> +1.1 -> -3.0 -> +5.1 mm), the insert's 61 probes all met the rim +7.2 mm high
                # and its rim-measured depth fix (capped at 6 mm) ignored it.
                dz_ = (z_meas - 0.002) - rim_z
                if GRASP_DEPTH_MEAS_MIN_M < dz_ <= GRASP_DEPTH_MEAS_MAX_M and obj not in GRASP_DEPTH_FIX:
                    from primitives.shared.config import held_tool_offset_for
                    d0 = float(held_tool_offset_for(obj)[2])
                    os.environ['GRASP_TOOL_OFFSET_M'] = f"{d0 + dz_:.5f}"
                    GRASP_DEPTH_FIX[obj] = dz_
                    st = st + np.array([0.0, 0.0, dz_]); rim_z += dz_
                    z_clear = max(float(st[2]) + HOVER_ALIGN_ABOVE_M, rim_z + SERVO_CLEAR_OVER_RIM_M)
                    say(f"  [servo] {obj}: grasp depth {1000*d0:.1f} -> {1000*(d0 + dz_):.1f} mm (measured by the touchdown); "
                        f"seat and rim raised {1000*dz_:+.1f} mm")
        if r.get('result') == 'failure':
            say(f"  [servo] {obj}: hover not reached ({r.get('error', '')})")
            if cell_io._SAFETY_STOP:
                return observed_base, held_q                            # nothing more is sent on a stopped robot
            break
        _emitter(False)
        try:
            img_, _ = still_frame(n_avg=4); tcp_now = (snapshot_robot_state().get('tcp') or {})
            tp = (np.asarray(tcp_now['xyz']), np.asarray(tcp_now['quat_xyzw']))
            base_sf, sfinfo = base_pose_same_frame(base, img_, tp, bq) if img_ is not None else (None, {'error': 'no frame'})
            lift = float(tp[0][2] - st[2])
            d, info = socket_shift_world(obj, base, base_sf if base_sf is not None else observed_base, lift, outline, img=img_,
                                         tcp_pose=tp, debug_path=f"{sv_stem}_r{k}.png")
        finally:
            _emitter(True)
        if base_sf is None and z_meas < z_clear - 0.01:
            # the seat lies on the far side of the base for this view: from just over the rim the base fills the frame and
            # its corner markers leave it (10-22 02:52: u_orange, 0 markers); from the clear height 2-3 are in view.
            # Measure from there instead (the hand-eye tilt lever grows to ~1.7 mm/deg; the dip grid covers +-1.5 mm)
            say(f"  [servo] {obj} round {k}: {sfinfo.get('error')} over the rim - measuring from the clear height instead")
            z_meas = z_clear
            law.measuring_from_clear(float(R_.from_quat(bq).as_euler('xyz', degrees=True)[2]))   # the 2x2 response, at this base yaw
            r = _move_unless_there((xy[0], xy[1], z_clear), q, f'servo up to measure {k}', log_fh)
            if cell_io._SAFETY_STOP:
                return observed_base, held_q
            _emitter(False)
            try:
                img_, _ = still_frame(n_avg=4); tcp_now = (snapshot_robot_state().get('tcp') or {})
                tp = (np.asarray(tcp_now['xyz']), np.asarray(tcp_now['quat_xyzw']))
                base_sf, sfinfo = base_pose_same_frame(base, img_, tp, bq) if img_ is not None else (None, {'error': 'no frame'})
                lift = float(tp[0][2] - st[2])
                d, info = socket_shift_world(obj, base, base_sf if base_sf is not None else observed_base, lift, outline,
                                             img=img_, tcp_pose=tp, debug_path=f"{sv_stem}_r{k}_clear.png")
            finally:
                _emitter(True)
        if base_sf is None:
            # fail closed (10-22 02:25: with 0-1 base markers in view the fallback projected the base read - the biased
            # path - and its 27-70 px-spread fits turned u_orange's wrist 6 x -5 deg = -30 deg into the insert)
            say(f"  [servo] {obj} round {k}: base not read in the frame ({sfinfo.get('error')}) - no correction, aiming as read")
            break
        act = law.reading(d, info)
        for n_ in law.notes:
            say(f"  [servo] {obj} {n_}")
        law.notes.clear()
        if isinstance(act, SL.Stop) and act.why.startswith('not measured'):
            say(f"  [servo] {obj} round {k}: not measured ({info.get('error')})"); break
        if isinstance(act, SL.Stop) and act.why.startswith('fit spread'):
            say(f"  [servo] {obj} round {k}: fit spread {info.get('spread_px')} px > {SL.SERVO_MAX_SPREAD_PX} - the outline "
                f"does not match the target, no correction"); break
        say(f"  [servo] {obj} round {k}: part {info['dxy_mm']} mm from over its socket, yaw {info.get('yaw_img_deg')} deg in the image "
            f"(shift {info['shift_px']} px, spread {info['spread_px']} px; base in frame {json.dumps(sfinfo)})")
        if isinstance(act, SL.Converged):
            break
        if isinstance(act, SL.Stop):
            say(f"  [servo] {obj}: {act.why} - more than the reads can be off, stopping the servo"); break
        if isinstance(act, SL.Turn):
            _move_unless_there((xy[0], xy[1], z_clear), q, f'servo up for turn {k}', log_fh)   # vertical: xy is where we measured
            if cell_io._SAFETY_STOP:
                return observed_base, held_q
            if turn_wrist(act.deg):
                law.turned(act.deg)
                if act.xy is not None:
                    xy = np.array(act.xy)                           # the step taken with the turn
                held_q = list((R_.from_euler('z', act.deg, degrees=True) * R_.from_quat(held_q)).as_quat())
                q = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw') or q
                say(f"  [servo] {obj}: wrist turned {act.deg:+.2f} deg (image yaw)")
            else:
                law.xy = np.array(xy, float)                        # no turn, so no step either: re-measure here
            law.next_round()
            continue                                                # re-measure xy after the turn
        xy = np.array(act.xy)                                       # SL.Step: the next place to measure
        law.next_round()
    out = law.finish()
    xy = law.xy
    if isinstance(out, SL.Aim) and out.overshot:
        say(f"  [servo] {obj}: last step overshot - aiming from the best reading ({1000*out.overshot[0]:+.1f}, "
            f"{1000*out.overshot[1]:+.1f}) mm at gain {out.overshot[2]:.2f}")
    _move_unless_there((xy[0], xy[1], z_clear), q, 'servo done: clear', log_fh)
    SERVO_MEASURED[obj] = law.final_reading is not None
    if isinstance(out, SL.Stop):
        say(f"  [servo] {obj}: {out.why} - not applied")
        return observed_base, held_q
    c = np.array(out.dxy); yaw_total = out.yaw_deg
    if law.final_reading is not None:
        _record_servo(obj, c, yaw_total, bq, law.k)
    if yaw_total:
        # the image yaw becomes the base's yaw for the insert, turned about the SEAT point (the pocket stays put): the
        # insert's own yaw step (held part vs base x seat) then has nothing to undo
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        sp_, _ = load_seat_pose_in_base(base, obj)
        Rb0 = R_.from_quat(bq); sw = np.array([bx, by, bz]) + Rb0.apply(sp_)
        Rb1 = R_.from_euler('z', yaw_total, degrees=True) * Rb0
        bx, by, bz = sw - Rb1.apply(sp_); bq = tuple(Rb1.as_quat())
        say(f"  [servo] {obj}: base yaw for the insert turned {yaw_total:+.2f} deg about the seat (image)")
    say(f"  [servo] {obj}: aim moved by ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm")
    return ((bx + c[0], by + c[1], bz), bq), held_q


SERVO_MEAS_OVER_RIM_M = 0.004  # measuring height over the CAD rim contact (the real rim reads ~1.5 mm higher: ~2.5 mm clear)


SERVO_CLEAR_OVER_RIM_M = 0.012 # lateral moves and wrist turns happen here


_EMITTER = {'state': None, 'hold': False}


def _emitter(on):
    """The D415 projector's dots swamp the colour image in a dark room (10-21): off for colour-only reads. Each set is a
    ros2 CLI call + 0.8 s settle (~2.2 s): it is skipped when the projector is already so, and held off through a servo's
    colour-only rounds (_emitter_hold) - run 41's u_brown align toggled it 8 times (~18 s)."""
    if on and _EMITTER['hold']:
        return
    if _EMITTER['state'] is not None and _EMITTER['state'] == bool(on):
        return
    try:
        if not (EMITTER_INPROC and _set_emitter_inproc(on)):
            subprocess.run(['ros2', 'param', 'set', '/camera/camera', 'depth_module.emitter_enabled', '1' if on else '0'],
                           capture_output=True, timeout=10, env=make_env())
        time.sleep(0.8)
        _EMITTER['state'] = bool(on)
    except Exception:
        _EMITTER['state'] = None


EMITTER_INPROC = os.environ.get('EMITTER_INPROC', '1') == '1'


def _set_emitter_inproc(on):
    """The projector set as one SetParameters service call on a private context (~0.05 s) - the ros2 CLI took ~1.5 s
    to start each time (10-23). False when the service did not answer: the caller falls back to the CLI."""
    import rclpy
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.executors import SingleThreadedExecutor
    ctx = rclpy.context.Context()
    rclpy.init(context=ctx)
    try:
        n = rclpy.create_node('replay_emitter', context=ctx)
        ex = SingleThreadedExecutor(context=ctx); ex.add_node(n)
        cl = n.create_client(SetParameters, '/camera/camera/set_parameters')
        if not cl.wait_for_service(timeout_sec=2.0):
            return False
        req = SetParameters.Request()
        req.parameters = [Parameter(name='depth_module.emitter_enabled',
                                    value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1 if on else 0))]
        fut = cl.call_async(req)
        ex.spin_until_future_complete(fut, timeout_sec=3.0)
        res = fut.result()
        return bool(res and res.results and res.results[0].successful)
    finally:
        try:
            n.destroy_node()
        except Exception:
            pass
        rclpy.shutdown(context=ctx)


def _emitter_hold():
    """Projector off until _emitter_release: the frames from here on are colour-only (no depth read in between)."""
    _EMITTER['hold'] = False
    _emitter(False)
    _EMITTER['hold'] = True


def _emitter_release():
    _EMITTER['hold'] = False
    _emitter(True)
