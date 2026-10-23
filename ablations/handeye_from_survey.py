#!/usr/bin/env python3
"""Refine the wrist camera's mount (hand-eye) from a survey of one rigid, flat object (base1).

The localizer reports the object's pose in the robot frame through the mount X0 in its config. If X0 is
wrong, the same physical object is reported at different poses from different viewpoints. With the
object's RAW (pre-snap) poses from many views (scan_workspace --survey), find the correction D
(X = X0 * D) that makes all views agree and the object flat (roll = pitch = 0):

    C_i = (T_tcp_i X0)^-1 B_i          the object in the camera frame, view i (what the camera measured)
    B_i(D) = T_tcp_i X0 D C_i          the object in the robot frame with the corrected mount

minimise  sum_i |p_i - mean p|^2 / s_p^2 + |yaw_i - mean yaw|^2 / s_yaw^2 + (roll_i^2 + pitch_i^2) / s_tilt^2

Prints the spread before/after and the corrected cam_offset_position / cam_offset_quat for
aruco_camera_localizer/config/robot_config.yaml. Does not write the config.

  python3 ablations/handeye_from_survey.py ablations/logs/survey_base1_20261020.json [--target base1]
"""
import argparse
import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

CFG = '/home/aldrin/Desktop/ros2_ws/src/aruco_camera_localizer/config/robot_config.yaml'
S_P, S_YAW, S_TILT = 0.001, np.radians(0.2), np.radians(0.2)   # weights: 1 mm, 0.2 deg


def T(pos, quat):
    M = np.eye(4); M[:3, :3] = R.from_quat(quat).as_matrix(); M[:3, 3] = pos
    return M


def load_x0():
    import yaml
    c = yaml.safe_load(open(CFG))
    return T(c['cam_offset_position'], c['cam_offset_quat']), c['cam_offset_position'], c['cam_offset_quat']


def views(path, target):
    d = json.load(open(path))
    out = []
    for p in d['poses']:
        raw = p.get('objects_raw', {}).get(target) or p.get('objects', {}).get(target)
        tcp = p.get('tcp_mean')
        if not raw or not tcp or not p.get('still', True):
            continue
        A = np.array(raw)
        if A.ndim != 2 or A.shape[1] < 8:
            continue
        pos = np.median(A[:, 1:4], 0)
        Q = A[:, 4:8]; Q = Q * np.sign(Q @ Q[0])[:, None]; q = Q.mean(0); q /= np.linalg.norm(q)
        out.append((p['label'], T(tcp[:3], tcp[3:7]), T(pos, q)))
    return out


def corrected(params, X0, Ts, Bs):
    D = T(params[3:], R.from_rotvec(params[:3]).as_quat())
    res = []
    for Tt, B in zip(Ts, Bs):
        C = np.linalg.inv(Tt @ X0) @ B
        res.append(Tt @ X0 @ D @ C)
    return res


def summary(Bs):
    P = np.array([B[:3, 3] for B in Bs]); E = np.array([R.from_matrix(B[:3, :3]).as_euler('xyz', degrees=True) for B in Bs])
    yaw = np.degrees(np.unwrap(np.radians(E[:, 2])))
    return (1000 * np.linalg.norm(P[:, :2] - np.median(P[:, :2], 0), axis=1), yaw - np.median(yaw),
            np.hypot(E[:, 0] - np.round(E[:, 0] / 180) * 180, E[:, 1]), P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('survey')
    ap.add_argument('--target', default='base1')
    a = ap.parse_args()
    X0, p0, q0 = load_x0()
    V = views(a.survey, a.target)
    if len(V) < 6:
        sys.exit(f"only {len(V)} usable views")
    labels = [v[0] for v in V]; Ts = [v[1] for v in V]; Bs = [v[2] for v in V]

    def residual(x):
        Bc = corrected(x, X0, Ts, Bs)
        P = np.array([B[:3, 3] for B in Bc]); Rm = [R.from_matrix(B[:3, :3]) for B in Bc]
        yaw = np.unwrap(np.array([r.as_euler('xyz')[2] for r in Rm]))
        tilt = np.array([np.hypot(*r.as_rotvec()[:2]) if abs(r.as_euler('xyz')[0]) < np.pi / 2 else
                         np.hypot(*(r * R.from_euler('x', np.pi)).as_rotvec()[:2]) for r in Rm])
        return np.concatenate([((P - P.mean(0)) / S_P).ravel(), (yaw - yaw.mean()) / S_YAW, tilt / S_TILT])

    sol = least_squares(residual, np.zeros(6), x_scale=[0.01, 0.01, 0.01, 0.005, 0.005, 0.005])
    b_dxy, b_dyaw, b_tilt, _ = summary(Bs)
    Bc = corrected(sol.x, X0, Ts, Bs)
    a_dxy, a_dyaw, a_tilt, Pc = summary(Bc)
    print(f"{len(V)} views of {a.target}")
    print(f"  before: xy spread median {np.median(b_dxy):.1f} max {b_dxy.max():.1f} mm | yaw spread max {np.abs(b_dyaw).max():.2f} deg | tilt median {np.median(b_tilt):.2f} deg")
    print(f"  after : xy spread median {np.median(a_dxy):.1f} max {a_dxy.max():.1f} mm | yaw spread max {np.abs(a_dyaw).max():.2f} deg | tilt median {np.median(a_tilt):.2f} deg")
    print(f"  correction: rot {np.round(np.degrees(sol.x[:3]), 3).tolist()} deg (camera frame), trans {np.round(1000 * sol.x[3:], 2).tolist()} mm")
    X = X0 @ T(sol.x[3:], R.from_rotvec(sol.x[:3]).as_quat())
    print(f"  old cam_offset_position {p0}  cam_offset_quat {q0}")
    print(f"  new cam_offset_position {np.round(X[:3, 3], 5).tolist()}  cam_offset_quat {np.round(R.from_matrix(X[:3, :3]).as_quat(), 5).tolist()}")
    print(f"  {a.target} after correction: xy {np.round(1000 * np.median(Pc[:, :2], 0), 1).tolist()} mm")
    for lab, d1, d2, y1, y2 in zip(labels, b_dxy, a_dxy, b_dyaw, a_dyaw):
        print(f"    {lab:34s} dxy {d1:5.1f} -> {d2:4.1f} mm   dyaw {y1:+.2f} -> {y2:+.2f} deg")


if __name__ == '__main__':
    main()
