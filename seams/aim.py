#!/usr/bin/env python3
"""Aim (2026-10-22, AR1): the base pose an insert aims at, from the MEASURED base + the parts already seated.

    aim = aim_for_insert(obj, base_name, measured_base, load_facts())
    aim.pos, aim.quat          the base pose the insert aims at (seam terms: the per-part seat z NOT included)
    aim.argv_pos()             the same + the per-part seat z (translate_object's --final-base-pos convention)
    aim.corrections            what moved it off the measured base, by how much, from which record
    aim.notes                  why a correction was NOT applied (records missing, switched off)

Pure: every record (seated offsets, ledger, seat history, switches, clock) arrives in AimFacts, CAD through `cad`.
Before this module the aimed pose was composed inside replay_real_assembly.build_argv, sent as argv strings and parsed
back by insert_via_drive_mate (seat z added, removed, added again by the server). The rules are moved unchanged:

  follow    line_green -> the U parts' recorded seats (FOLLOWS_SEATED: slot x + yaw; source = the camera ledger
            `rel_err_mm` unless one-node, else the inserts' seated_offsets)          [replay 1567, 2026-10-15/19]
            any other part -> the mean measured seat error of the seated parts it overlaps (CAD footprints), across
            each elongated part's long axis only                                     [replay 1664, 2026-10-20]
  bias      u_brown / u_orange / inverted_u_yellow -> median of the last 5 recorded seat offsets (>= 3), <= 8 mm
  yaw-exp   INSERT_YAW_OFFSET_DEG: turn about the part's seat point (experiment hook, 2026-10-20)

Known debt kept visible: FOLLOWS_SEATED and AIM_BIAS_PARTS name parts; the two follow rules are one concept.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R_

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / 'ablations/logs'
SEATED_OFFSETS = LOGS / 'seated_offsets.json'       # {part: {seat_minus_pred_mm: [dx, dy, dz] world, t}}
SEAT_HISTORY = LOGS / 'seat_offset_history.json'    # {part: [[dx, dy] mm, ...]}
ASSEMBLY_STATE = LOGS / 'assembly_state.json'       # ledger {part: {verified, t, how, rel_err_mm?}}

FOLLOWS_SEATED = {'line_green': ('u_brown', 'u_orange')}
MAX_FOLLOW_MM = 8.0
MAX_FOLLOW_YAW_DEG = 4.0
MAX_FOLLOW_YAW_INSERT_DEG = 1.5
FOLLOW_GENERAL_MAX_MM = 4.0
FOLLOW_GENERAL_MAX_AGE_S = 7200.0
AIM_BIAS_PARTS = ('u_brown', 'u_orange', 'inverted_u_yellow')
AIM_BIAS_N = 5
AIM_BIAS_MIN = 3
AIM_BIAS_MAX_MM = 8.0

Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


@dataclass(frozen=True)
class Correction:
    name: str                      # follow | bias | yaw_exp
    dxy_m: Tuple[float, float]     # world shift of the aimed base origin
    dyaw_rad: float                # turn of the aimed base frame
    about: str                     # base_origin | seat
    detail: str                    # the log line (same text the replay printed before)


@dataclass(frozen=True)
class Aim:
    obj: str
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    seat_dz_m: float
    measured: Pose
    corrections: Tuple[Correction, ...] = ()
    notes: Tuple[str, ...] = ()

    def argv_pos(self) -> Tuple[float, float, float]:
        return (self.pos[0], self.pos[1], self.pos[2] + self.seat_dz_m)

    def shift(self, name: str) -> Tuple[float, float]:
        """World xy (m) of one named correction (0, 0 if not applied) - record_seat_offset adds the bias back."""
        c = next((c for c in self.corrections if c.name == name), None)
        return c.dxy_m if c else (0.0, 0.0)


@dataclass(frozen=True)
class AimFacts:
    seated_offsets: Mapping = field(default_factory=dict)
    ledger: Mapping = field(default_factory=dict)
    seat_history: Mapping = field(default_factory=dict)
    one_node: bool = False
    no_aim_bias: bool = False
    follow_yaw: bool = True
    yaw_exp_deg: float = 0.0
    now: float = 0.0


def _read(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def load_facts(env: Optional[Mapping] = None, now: Optional[float] = None) -> AimFacts:
    """The records as they are now (the only I/O in this module)."""
    env = os.environ if env is None else env
    return AimFacts(seated_offsets=_read(SEATED_OFFSETS) if SEATED_OFFSETS.exists() else {},
                    ledger=_read(ASSEMBLY_STATE), seat_history=_read(SEAT_HISTORY),
                    one_node=env.get('SEAMS_DRIVE') == '1', no_aim_bias=bool(env.get('NO_AIM_BIAS')),
                    follow_yaw=env.get('FOLLOW_YAW', '1') != '0',
                    yaw_exp_deg=float(env.get('INSERT_YAW_OFFSET_DEG', '0') or 0),
                    now=time.time() if now is None else now)


class Cad:
    """CAD questions the aim asks; a test passes its own."""

    def seat_pos(self, base: str, part: str):
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        return np.asarray(load_seat_pose_in_base(base, part)[0], float)

    def seat_dz(self, part: str) -> float:
        from primitives.shared.config import get_object_base_offset_m
        return float(get_object_base_offset_m(part)[2])

    def overlapping_seated(self, base: str, obj: str) -> list:
        """Parts seated before `obj` (assembly order) whose seated xy footprint overlaps obj's."""
        try:
            import sys
            sys.path.insert(0, str(REPO_ROOT / 'ablations'))
            from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
            from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
            from grasp_depth import load_mesh

            def box(p):
                V, _ = load_mesh(p); sp, sq = load_seat_pose_in_base(base, p)
                W = V @ R_.from_quat(sq).as_matrix().T + sp
                return W[:, :2].min(0), W[:, :2].max(0)
            lo, hi = box(obj)
            out = []
            for p in get_seated_so_far(base, obj):
                l2, h2 = box(p)
                if (l2 < hi).all() and (lo < h2).all():
                    out.append(p)
            return out
        except Exception:
            return []

    def top_face(self, base: str, part: str) -> dict:
        import sys
        sys.path.insert(0, str(REPO_ROOT / 'ablations'))
        from seat_check import TOP_FACE
        return TOP_FACE[(part, base)]


def _follow_general(obj, base, base_quat, f: AimFacts, cad: Cad):
    parts = cad.overlapping_seated(base, obj)
    if not parts:
        return None, None
    led = f.ledger
    errs = [np.array(led[p]['rel_err_mm'], dtype=float) for p in parts
            if isinstance((led.get(p) or {}).get('rel_err_mm'), list) and f.now - led[p].get('t', 0) < FOLLOW_GENERAL_MAX_AGE_S]
    if len(errs) < len(parts):
        return None, f"  [follow] {obj}: measured seat errors of {parts} not all recorded -> aim at the base"
    E = np.full((len(errs), 2), np.nan)
    for i, (p, e) in enumerate(zip(parts, errs)):
        tf = cad.top_face(base, p)
        ax = 0 if tf.get('long_axis', 'y') == 'y' else 1
        if tf['half_long_m'] >= 2 * tf['half_cross_m']:
            E[i, ax] = e[ax]                     # elongated: across only
        else:
            E[i] = e                             # compact: both
    m = np.array([np.nanmean(E[:, k]) if np.isfinite(E[:, k]).any() else 0.0 for k in (0, 1)])
    m = np.clip(m, -FOLLOW_GENERAL_MAX_MM, FOLLOW_GENERAL_MAX_MM)
    dw = R_.from_quat(base_quat).as_matrix()[:2, :2] @ m
    line = (f"  [follow] {obj}: interlocks with {parts} -> aim shifted ({m[0]:+.1f}, {m[1]:+.1f}) mm (base frame, mean of "
            f"their measured seat errors {[np.round(e, 1).tolist() for e in errs]})")
    return Correction('follow', (float(dw[0]) / 1000, float(dw[1]) / 1000), 0.0, 'base_origin', line), None


def _follow_named(obj, base, base_quat, parts, f: AimFacts, cad: Cad):
    try:
        last = {k: v['seat_minus_pred_mm'] for k, v in f.seated_offsets.items()}
        if not all(p in last for p in parts):
            return None, (f"  [follow] {obj}: seat offsets of {parts} not all recorded ({SEATED_OFFSETS.name}) "
                          f"-> aim at the base")
        Rb = R_.from_quat(base_quat).as_matrix()[:2, :2]
        (pa, pb) = parts
        cam = {p: (f.ledger.get(p) or {}).get('rel_err_mm') for p in parts}
        if all(isinstance(v, list) and len(v) == 2 for v in cam.values()) and not f.one_node:
            da, db = (np.array(cam[p], dtype=float) for p in parts)           # mm, base frame (camera)
            src, cap = 'camera', MAX_FOLLOW_YAW_DEG
        else:
            da, db = (Rb.T @ np.array(last[p][:2]) for p in parts)           # mm, base frame (inserts)
            src, cap = 'inserts', MAX_FOLLOW_YAW_INSERT_DEG
        ya = 1000 * cad.seat_pos(base, pa)[1]; yb = 1000 * cad.seat_pos(base, pb)[1]
        dx_b = float(np.clip((da[0] + db[0]) / 2, -MAX_FOLLOW_MM, MAX_FOLLOW_MM))
        dyaw = float(np.clip(-np.arctan2(db[0] - da[0], yb - ya), -np.radians(cap), np.radians(cap)))
        note = None
        if not f.follow_yaw:
            note = f"  [follow] {obj}: yaw {np.degrees(dyaw):+.2f} deg not applied (FOLLOW_YAW=0)"
            dyaw = 0.0
        dw = Rb @ np.array([dx_b, 0.0])
        line = (f"  [follow] {obj}: slot x {dx_b:+.1f} mm, yaw {np.degrees(dyaw):+.2f} deg (base frame, from the {src}"
                f", cap {cap:g} deg) — " + ", ".join(f"{p} x {d[0]:+.1f} mm" for p, d in zip(parts, (da, db))))
        return Correction('follow', (float(dw[0]) / 1000, float(dw[1]) / 1000), dyaw, 'base_origin', line), note
    except Exception as e:
        return None, f"  [follow] {obj}: {e} -> aim at the base"


def follow(obj, base, base_quat, f: AimFacts, cad: Cad):
    """(Correction or None, note or None) toward where the parts `obj` sits among actually seated."""
    parts = FOLLOWS_SEATED.get(obj)
    return _follow_named(obj, base, base_quat, parts, f, cad) if parts else _follow_general(obj, base, base_quat, f, cad)


def bias(obj, f: AimFacts):
    """Median of where `obj`'s last inserts seated vs the prediction (world mm -> m), or None."""
    if obj not in AIM_BIAS_PARTS or f.no_aim_bias:
        return None
    h = list((f.seat_history or {}).get(obj, []))
    if len(h) < AIM_BIAS_MIN:
        return None
    m = np.clip(np.median(np.array(h[-AIM_BIAS_N:]), axis=0), -AIM_BIAS_MAX_MM, AIM_BIAS_MAX_MM)
    bx, by = float(m[0]) / 1000, float(m[1]) / 1000
    if not (bx or by):
        return None
    return Correction('bias', (bx, by), 0.0, 'base_origin',
                      f"  [aim] {obj}: + seat-history bias ({1000*bx:+.1f}, {1000*by:+.1f}) mm")


def aim_for_insert(obj: str, base: str, measured: Pose, f: AimFacts, cad: Optional[Cad] = None) -> Aim:
    """The base pose `obj`'s insert aims at: the measured base moved by follow, bias and the yaw experiment."""
    cad = cad or Cad()
    (ox, oy, oz), oq = measured
    oq = tuple(float(v) for v in oq)
    corr, notes = [], []
    fc, note = follow(obj, base, oq, f, cad)
    if note:
        notes.append(note)
    if fc is not None:
        corr.append(fc)
        ox, oy = ox + fc.dxy_m[0], oy + fc.dxy_m[1]
    bc = bias(obj, f)
    if bc is not None:
        corr.append(bc)
        ox, oy = ox + bc.dxy_m[0], oy + bc.dxy_m[1]
    if fc is not None and fc.dyaw_rad:            # turn the base frame about its origin (line_green sits there)
        oq = tuple((R_.from_euler('z', fc.dyaw_rad) * R_.from_quat(oq)).as_quat())
    if f.yaw_exp_deg:
        sp = cad.seat_pos(base, obj)
        Rb_ = R_.from_quat(oq); o_ = np.array([ox, oy, oz])
        seat_w = o_ + Rb_.apply(sp)
        Rd = R_.from_euler('z', f.yaw_exp_deg, degrees=True)
        o_new = seat_w + Rd.apply(o_ - seat_w)
        corr.append(Correction('yaw_exp', (float(o_new[0]) - ox, float(o_new[1]) - oy), float(np.radians(f.yaw_exp_deg)),
                               'seat', f"  [yaw-exp] {obj}: insert yaw offset {f.yaw_exp_deg:+.2f} deg about the seat point"))
        ox, oy = float(o_new[0]), float(o_new[1])
        oq = tuple((Rd * Rb_).as_quat())
    return Aim(obj=obj, pos=(float(ox), float(oy), float(oz)), quat=tuple(float(v) for v in oq),
               seat_dz_m=cad.seat_dz(obj), measured=measured, corrections=tuple(corr), notes=tuple(notes))
