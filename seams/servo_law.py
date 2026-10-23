#!/usr/bin/env python3
"""The hover servo's control law (2026-10-22, AR3 step 1): what to do after each reading, with no motion and no camera.

    law = HoverServoLaw(seat_xy)                # the predicted seat TCP xy (m); the servo starts over it
    law.measuring_from_clear()                  # when the reading has to come from the clear height
    act = law.reading(d, info, base_seen)      # d: part - socket (m, world xy) or None; info: socket_shift_world's
      -> Turn(deg) | Step(xy) | Stop(why) | Converged
    law.turned(deg)                             # after the caller's wrist turn succeeded
    out = law.finish()                          # -> Aim(dxy, yaw_deg, measured) | Stop(why)

Moved unchanged from replay_real_assembly.hover_servo (10-21/22): the caller still moves the arm and reads the frame.
The logged rounds of every servo session are its test (seams/test_servo_law.py); a better law is tried against them
before it reaches the arm (u_brown: the first reading predicts the final aim to ~1 mm in 18/18 sessions; u_orange: not -
the wrist turn moves the reading through the grasp lever, and from the clear height the scale is ~2.4x).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

SERVO_ITERS = 6
SERVO_YAW_TOL_DEG = 0.3
SERVO_YAW_MIN_TURN_DEG = 0.4   # the image-yaw noise floor; smaller turns did not change the reading (10-22)
SERVO_MAX_SPREAD_PX = 12.0     # a fit whose edge residuals spread more than this does not match the part (good fits: 1-4 px)
SERVO_MAX_YAW_DEG = 4.0        # the whole servo may turn the wrist this much; base and grasp reads are within ~3 deg
SERVO_GAIN_CLEAR = 2.4         # reading change per mm of xy step when measured from the clear height (measured twice)
SERVO_YAW_GAIN = 1.0           # image-yaw change per +1 deg of wrist turn (measured 0.7-1.14, 10-21): turn 1:1 and iterate
# the pocket was entered at (-0.5, +0.9) mm from the converged aim in both seats so far (by hand 10-21 23:50, pipeline
# 10-22 00:13): the target's residual bias (hand-eye tilt x the 35 mm lift); the dip grid still covers +-1.5 mm around it.
# From the clear height (+0.2, -2.1) and (-1.1, -1.8) mm (u_orange, 10-22 04:47 and 03:55) - hand-eye tilt x the lever
SERVO_BIAS_M = np.array([float(v) for v in os.environ.get('SERVO_BIAS_MM', '-0.5,0.9').split(',')]) / 1000.0
SERVO_BIAS_CLEAR_M = np.array([float(v) for v in os.environ.get('SERVO_BIAS_CLEAR_MM', '-0.5,-2.0').split(',')]) / 1000.0
SERVO_TOL_M = 0.001
SERVO_MAX_M = 0.015


@dataclass(frozen=True)
class Turn:
    deg: float


@dataclass(frozen=True)
class Step:
    xy: Tuple[float, float]       # where to measure next (TCP xy, m)


@dataclass(frozen=True)
class Stop:
    why: str


@dataclass(frozen=True)
class Converged:
    pass


@dataclass(frozen=True)
class Aim:
    dxy: Tuple[float, float]      # aim shift (m, world) incl. the height's bias
    yaw_deg: float                # total wrist turn (the base yaw for the insert turns by it about the seat)
    measured: bool
    overshot: Optional[Tuple[float, float, float]] = None   # (best reading x, y in m, gain) when it aimed from the best


@dataclass
class HoverServoLaw:
    seat_xy: np.ndarray
    xy: np.ndarray = None
    gain: float = 1.0
    yaw_total: float = 0.0
    k: int = 0
    from_clear: bool = False
    last_move: Optional[np.ndarray] = None
    last_d: Optional[np.ndarray] = None
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    final_reading: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.seat_xy = np.asarray(self.seat_xy, float)
        self.xy = self.seat_xy.copy() if self.xy is None else np.asarray(self.xy, float)

    def measuring_from_clear(self):
        """The base's markers leave the frame over the rim: measure from the clear height, where a step moves the
        reading ~2.4x (u_orange 10-22 03:28 / 04:39: a 1:1 first step overshot 7.6 -> -10.3 mm)."""
        self.from_clear = True
        self.gain = SERVO_GAIN_CLEAR

    def measuring_low(self):
        """Touched down before the measuring height: measured from 2 mm over the contact (the rim bias applies)."""
        self.from_clear = False

    def reading(self, d, info: dict, base_seen: bool = True):
        """One measurement at round self.k (the caller advances k by calling next_round after a Turn or Step)."""
        if not base_seen:
            return Stop(f"base not read in the frame ({info.get('error')})")
        if d is None:
            return Stop(f"not measured ({info.get('error')})")
        sp_x, sp_y = (info.get('spread_px') or [0, 0])[:2]
        if sp_y > SERVO_MAX_SPREAD_PX:
            return Stop(f"fit spread {info.get('spread_px')} px > {SERVO_MAX_SPREAD_PX}")
        if sp_x > SERVO_MAX_SPREAD_PX:
            # the far edge fits, the left one does not (u_orange's slanted side face, 10-22): correct along the far
            # edge's image-y only
            d = np.asarray(info['d_far_only_mm'], float) / 1000.0
            self.notes.append(f"round {self.k}: left-edge spread {sp_x} px > {SERVO_MAX_SPREAD_PX} - far-edge component only "
                              f"{info['d_far_only_mm']} mm")
        d = np.asarray(d, float)
        yi = info.get('yaw_img_deg')
        self.final_reading = d.copy()
        if self.best is None or np.hypot(*d) < np.hypot(*self.best[1]):
            self.best = (self.xy.copy(), d.copy())
        yaw_ok = yi is None or abs(yi) <= SERVO_YAW_TOL_DEG or abs(yi / SERVO_YAW_GAIN) < SERVO_YAW_MIN_TURN_DEG
        if (np.hypot(*d) <= SERVO_TOL_M and yaw_ok) or self.k == SERVO_ITERS:
            return Converged()
        if not yaw_ok:
            t = float(np.clip(yi / SERVO_YAW_GAIN, -5.0, 5.0))
            if abs(self.yaw_total + t) > SERVO_MAX_YAW_DEG:
                return Stop(f"image yaw {yi} deg would turn the wrist {self.yaw_total + t:+.1f} deg in all "
                            f"(> {SERVO_MAX_YAW_DEG})")
            return Turn(t)
        if self.last_move is not None and self.last_d is not None:
            m_ = np.asarray(self.last_move); resp = self.last_d - d          # how much the reading moved for that step
            if np.dot(m_, m_) > 1e-8:
                self.gain = float(np.clip(np.dot(resp, m_) / np.dot(m_, m_), 0.4, 3.0))
        step = d / self.gain
        self.xy = self.xy - step
        self.last_move, self.last_d = step, d
        return Step((float(self.xy[0]), float(self.xy[1])))

    def turned(self, deg: float):
        """The wrist turn succeeded: it moves the reading too - re-learn the xy gain."""
        self.last_move, self.last_d = None, None
        self.yaw_total += deg

    def next_round(self):
        self.k += 1

    def finish(self):
        overshot = None
        if (self.best is not None and self.final_reading is not None
                and np.hypot(*self.final_reading) > np.hypot(*self.best[1]) + 0.0005):
            # the last step made the reading worse (an overshoot): aim from the best measurement with the learned gain
            self.xy = self.best[0] - self.best[1] / max(self.gain, 0.4)
            overshot = (float(self.best[1][0]), float(self.best[1][1]), self.gain)
        c = self.xy - self.seat_xy + (SERVO_BIAS_CLEAR_M if self.from_clear else SERVO_BIAS_M)
        if np.hypot(*c) > SERVO_MAX_M:
            return Stop(f"correction {1000*np.hypot(*c):.1f} mm is more than the reads can be off")
        return Aim((float(c[0]), float(c[1])), self.yaw_total, self.final_reading is not None, overshot)
